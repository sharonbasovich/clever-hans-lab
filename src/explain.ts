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
