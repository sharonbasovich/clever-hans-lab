import * as tf from '@tensorflow/tfjs';
import { IMG_PIXELS, IMG_SIZE, type World } from './world.ts';

// Occlusion sensitivity: slide a grey patch over the image, measure how much
// the model's log-probability of its top class drops per patch. Log-prob (not
// raw prob) is used deliberately: a confident model saturates at p≈1, and
// probability drops then read as ~0 even for pixels the model depends on.
// Patches whose removal hurts most are what the model "looked at".
export function occlusionMap(
  model: tf.LayersModel,
  image: Float32Array,
  patch = 4,
): Float32Array {
  const grid = IMG_SIZE / patch; // 8x8 for 32px/4
  const heat = new Float32Array(IMG_PIXELS);
  const nPatches = grid * grid;
  const EPS = 1e-6;

  tf.tidy(() => {
    const xs = tf.tensor4d(image, [1, IMG_SIZE, IMG_SIZE, 3]);
    const base = (model.predict(xs) as tf.Tensor).dataSync();
    const cls = base.indexOf(Math.max(...base));
    const baseLogP = Math.log(Math.max(EPS, base[cls]));
    xs.dispose();

    // Batch all occluded variants through in one predict call.
    const variants = new Float32Array(nPatches * IMG_PIXELS * 3);
    for (let gy = 0; gy < grid; gy++) {
      for (let gx = 0; gx < grid; gx++) {
        const v = (gy * grid + gx) * IMG_PIXELS * 3;
        variants.set(image, v);
        for (let y = gy * patch; y < (gy + 1) * patch; y++) {
          for (let x = gx * patch; x < (gx + 1) * patch; x++) {
            const p = v + (y * IMG_SIZE + x) * 3;
            variants[p] = 0.5;
            variants[p + 1] = 0.5;
            variants[p + 2] = 0.5;
          }
        }
      }
    }
    const vs = tf.tensor4d(variants, [nPatches, IMG_SIZE, IMG_SIZE, 3]);
    const probs = (model.predict(vs) as tf.Tensor).dataSync();
    vs.dispose();

    for (let gy = 0; gy < grid; gy++) {
      for (let gx = 0; gx < grid; gx++) {
        const vIdx = gy * grid + gx;
        const drop = Math.max(0, baseLogP - Math.log(Math.max(EPS, probs[vIdx * 2 + cls])));
        for (let y = gy * patch; y < (gy + 1) * patch; y++) {
          for (let x = gx * patch; x < (gx + 1) * patch; x++) {
            heat[y * IMG_SIZE + x] = drop;
          }
        }
      }
    }
  });
  return heat;
}

// Fraction of total occlusion heat that sits on background pixels. In [0,1].
// Because we generated the image we know the true foreground mask — this is a
// measurement against ground truth, not a vibes-based claim.
export function bgMass(heatmap: Float32Array, fgMask: Float32Array): number {
  let bg = 0;
  let total = 0;
  for (let i = 0; i < IMG_PIXELS; i++) {
    const h = heatmap[i];
    total += h;
    if (fgMask[i] === 0) bg += h;
  }
  if (total <= 0) return 0;
  return Math.min(1, Math.max(0, bg / total));
}

// Mean bgMass over the first `k` images of a world.
export function meanBgMass(model: tf.LayersModel, world: World, k = 32, patch = 4): number {
  const n = Math.min(k, world.n);
  let sum = 0;
  for (let i = 0; i < n; i++) {
    const img = world.images.subarray(i * IMG_PIXELS * 3, (i + 1) * IMG_PIXELS * 3);
    const mask = world.fgMasks.subarray(i * IMG_PIXELS, (i + 1) * IMG_PIXELS);
    const heat = occlusionMap(model, img as Float32Array, patch);
    sum += bgMass(heat, mask as Float32Array);
  }
  return n === 0 ? 0 : sum / n;
}

export interface Ablation {
  n: number;
  acc: number;
  /** Accuracy with the background greyed out (shape only). */
  accNoBg: number;
  /** Accuracy with the shape erased into the background (cue only). */
  accNoFg: number;
  bgDrop: number;
  fgDrop: number;
  /** bgDrop / (bgDrop + fgDrop) in [0,1]: how much of the model's accuracy
   *  lives in the background rather than the shape. ~1 = shortcut; ~0 = shape.
   *  null when neither ablation cost the model anything — i.e. it learned
   *  nothing measurable, so there is no reliance to attribute. */
  bgReliance: number | null;
}

// Counterfactual ablation — the headline "did it cheat" measurement. Unlike
// patch occlusion, it does not under-measure a distributed cue: a single 4x4
// patch only removes 1/64 of the background colour, so patch-occlusion mass
// concentrates on the shape edge even for a model that provably uses the
// background. Blanking the WHOLE background (or the whole shape) measures the
// reliance directly. We can do this because we generated the pixels and know
// the true foreground mask.
export function ablationReliance(model: tf.LayersModel, world: World, k = 64): Ablation {
  const n = Math.min(k, world.n);
  const noBg = new Float32Array(n * IMG_PIXELS * 3);
  const noFg = new Float32Array(n * IMG_PIXELS * 3);
  for (let i = 0; i < n; i++) {
    const img = world.images.subarray(i * IMG_PIXELS * 3, (i + 1) * IMG_PIXELS * 3);
    const mask = world.fgMasks.subarray(i * IMG_PIXELS, (i + 1) * IMG_PIXELS);
    noBg.set(img, i * IMG_PIXELS * 3);
    noFg.set(img, i * IMG_PIXELS * 3);
    // Mean background colour for filling the erased shape.
    let r = 0, g = 0, b = 0, c = 0;
    for (let p = 0; p < IMG_PIXELS; p++) {
      if (mask[p] === 0) {
        r += img[p * 3];
        g += img[p * 3 + 1];
        b += img[p * 3 + 2];
        c++;
      }
    }
    if (c > 0) {
      r /= c;
      g /= c;
      b /= c;
    }
    const off = i * IMG_PIXELS * 3;
    for (let p = 0; p < IMG_PIXELS; p++) {
      if (mask[p] === 0) {
        noBg[off + p * 3] = 0.5;
        noBg[off + p * 3 + 1] = 0.5;
        noBg[off + p * 3 + 2] = 0.5;
      } else {
        noFg[off + p * 3] = r;
        noFg[off + p * 3 + 1] = g;
        noFg[off + p * 3 + 2] = b;
      }
    }
  }
  const acc = batchAcc(model, world.images, world.labels, n);
  const accNoBg = batchAcc(model, noBg, world.labels, n);
  const accNoFg = batchAcc(model, noFg, world.labels, n);
  const bgDrop = Math.max(0, acc - accNoBg);
  const fgDrop = Math.max(0, acc - accNoFg);
  const denom = bgDrop + fgDrop;
  return {
    n,
    acc,
    accNoBg,
    accNoFg,
    bgDrop,
    fgDrop,
    // A denominator near zero means the model shrugged at both ablations —
    // it learned nothing to measure. Report null, not a fake 0.5.
    bgReliance: denom > 0.02 ? bgDrop / denom : null,
  };
}

function batchAcc(model: tf.LayersModel, images: Float32Array, labels: Uint8Array, n: number): number {
  const out = tf.tidy(() => {
    const xs = tf.tensor4d(images.subarray(0, n * IMG_PIXELS * 3), [n, IMG_SIZE, IMG_SIZE, 3]);
    const pred = (model.predict(xs) as tf.Tensor).argMax(1);
    const t = tf.tensor1d(Array.from(labels.subarray(0, n)), 'float32');
    const eq = pred.equal(t).mean();
    const v = eq.dataSync()[0];
    return v;
  });
  return out;
}
