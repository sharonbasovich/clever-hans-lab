import { mulberry32, randInt, type Rng } from './rng.ts';

// ---------------------------------------------------------------------------
// World generator.
//
// A "world" is a synthetic dataset of 32x32 RGB images. Every image contains
// exactly one shape (circle = class 0, triangle = class 1) on a coloured
// background. A cue (background hue, or an optional corner dot) is planted so
// that it agrees with the label with probability rho. Because we draw the
// pixels ourselves we know the ground-truth foreground mask and the planted
// cue of every image — that is what makes "the model cheated" measurable
// rather than speculative.
// ---------------------------------------------------------------------------

export const IMG_SIZE = 32;
export const IMG_PIXELS = IMG_SIZE * IMG_SIZE;
export const NUM_CLASSES = 2;

export type CueType = 'bg' | 'dot';

export interface WorldOptions {
  seed: number;
  n: number;
  /** Probability that the planted cue agrees with the label. */
  rho: number;
  cueType?: CueType;
  /**
   * 'matched'  — cue agrees with the label with probability rho.
   * 'flipped'  — the cue->label mapping is reversed (cue agrees with
   *              probability 1 - rho).
   * 'neutral'  — the cue is absent (grey background / no dot).
   */
  split?: 'matched' | 'flipped' | 'neutral';
  /** When true, cue colours also differ in brightness and carry a stripe
   *  pattern so the cue survives colour-blindness. */
  accessible?: boolean;
  /** When true, labels are shuffled after generation (null control). */
  shuffleLabels?: boolean;
}

export interface World {
  seed: number;
  n: number;
  rho: number;
  cueType: CueType;
  split: 'matched' | 'flipped' | 'neutral';
  accessible: boolean;
  /** n * 32 * 32 * 3, row-major RGB in [0, 1]. */
  images: Float32Array;
  /** class index per image: 0 = circle, 1 = triangle. */
  labels: Uint8Array;
  /** n * 32 * 32, 1 on shape pixels, 0 elsewhere. */
  fgMasks: Float32Array;
  /** planted cue value per image: 0 = cue-A colour, 1 = cue-B colour,
   *  -1 = no cue (neutral split). */
  cue: Int8Array;
}

// [cue=0 colour, cue=1 colour]
type CuePalette = [[number, number, number], [number, number, number]];

// Cue colours. Hue differs AND (in the accessible palette) lightness differs
// strongly, so the cue is visible without colour vision.
const BG_PALETTE: CuePalette = [
  [0.92, 0.30, 0.28], // warm red
  [0.25, 0.45, 0.92], // cool blue
];
const BG_PALETTE_ACCESSIBLE: CuePalette = [
  [0.55, 0.10, 0.10], // dark red + stripes
  [0.60, 0.78, 1.0], // light blue + opposite stripes
];
const DOT_PALETTE: CuePalette = [
  [1.0, 0.85, 0.15], // yellow dot
  [0.90, 0.35, 0.95], // magenta dot
];
const NEUTRAL_BG: [number, number, number] = [0.55, 0.55, 0.55];
const FG_COLOR: [number, number, number] = [0.96, 0.96, 0.94];

function jitter(p: [number, number, number], rng: Rng, amount: number): [number, number, number] {
  const j = () => (rng() - 0.5) * 2 * amount;
  return [
    Math.min(1, Math.max(0, p[0] + j())),
    Math.min(1, Math.max(0, p[1] + j())),
    Math.min(1, Math.max(0, p[2] + j())),
  ];
}

// Rasterize a filled shape into a 32x32 mask. Returns the mask.
function rasterizeShape(cls: number, cx: number, cy: number, size: number, rng: Rng): Float32Array {
  const mask = new Float32Array(IMG_PIXELS);
  if (cls === 0) {
    // circle: disk of radius `size`
    const r2 = size * size;
    for (let y = 0; y < IMG_SIZE; y++) {
      for (let x = 0; x < IMG_SIZE; x++) {
        const dx = x - cx;
        const dy = y - cy;
        if (dx * dx + dy * dy <= r2) mask[y * IMG_SIZE + x] = 1;
      }
    }
  } else {
    // triangle: apex up, with a random rotation chosen from 4 orientations so
    // orientation is not itself a free cue between runs.
    const orient = randInt(rng, 0, 3);
    for (let y = 0; y < IMG_SIZE; y++) {
      for (let x = 0; x < IMG_SIZE; x++) {
        let dx = x - cx;
        let dy = y - cy;
        if (orient === 1) { const t = dx; dx = dy; dy = -t; }
        else if (orient === 2) { dx = -dx; dy = -dy; }
        else if (orient === 3) { const t = dx; dx = -dy; dy = t; }
        // upright isoceles triangle: |dx| <= (dy + size) * 0.9, -size <= dy <= size
        if (dy >= -size && dy <= size && Math.abs(dx) <= (dy + size) * 0.9) {
          mask[y * IMG_SIZE + x] = 1;
        }
      }
    }
  }
  return mask;
}

function stripes(x: number, y: number, diagonal: boolean): boolean {
  const v = diagonal ? x + y : x - y;
  return Math.floor(v / 4) % 2 === 0;
}

export function generateWorld(opts: WorldOptions): World {
  const {
    seed,
    n,
    rho,
    cueType = 'bg',
    split = 'matched',
    accessible = false,
    shuffleLabels = false,
  } = opts;

  const rng = mulberry32(seed);
  const images = new Float32Array(n * IMG_PIXELS * 3);
  const fgMasks = new Float32Array(n * IMG_PIXELS);
  const labels = new Uint8Array(n);
  const cue = new Int8Array(n).fill(-1);

  const bgPalette = accessible ? BG_PALETTE_ACCESSIBLE : BG_PALETTE;
  const palette = cueType === 'dot' ? DOT_PALETTE : bgPalette;

  for (let i = 0; i < n; i++) {
    const cls = rng() < 0.5 ? 0 : 1;
    labels[i] = cls;

    // Decide the planted cue. cue==cls means "cue agrees with label".
    if (split === 'neutral') {
      cue[i] = -1;
    } else {
      const agree = rng() < rho;
      let cueVal = agree ? cls : 1 - cls;
      if (split === 'flipped') cueVal = 1 - cueVal;
      cue[i] = cueVal;
    }

    // Background.
    const base = split === 'neutral' ? NEUTRAL_BG : palette[cue[i] as 0 | 1];
    const imgOff = i * IMG_PIXELS * 3;
    const maskOff = i * IMG_PIXELS;

    for (let y = 0; y < IMG_SIZE; y++) {
      for (let x = 0; x < IMG_SIZE; x++) {
        let c = base;
        if (accessible && split !== 'neutral' && cueType === 'bg') {
          // In accessible mode the two cue colours carry opposite stripe
          // patterns, so the cue is a texture difference, not just a hue.
          if (cue[i] === 0 && stripes(x, y, true)) c = [c[0] * 0.8, c[1] * 0.8, c[2] * 0.8];
          if (cue[i] === 1 && stripes(x, y, false)) c = [c[0] * 0.8, c[1] * 0.8, c[2] * 0.8];
        }
        images[imgOff + (y * IMG_SIZE + x) * 3 + 0] = c[0];
        images[imgOff + (y * IMG_SIZE + x) * 3 + 1] = c[1];
        images[imgOff + (y * IMG_SIZE + x) * 3 + 2] = c[2];
      }
    }

    // Optional second cue: a corner dot whose colour carries the mapping.
    if (cueType === 'dot' && split !== 'neutral') {
      const dc = palette[cue[i] as 0 | 1];
      for (let dy = 0; dy < 4; dy++) {
        for (let dx = 0; dx < 4; dx++) {
          const p = ((2 + dy) * IMG_SIZE + (2 + dx)) * 3;
          images[imgOff + p + 0] = dc[0];
          images[imgOff + p + 1] = dc[1];
          images[imgOff + p + 2] = dc[2];
        }
      }
    }

    // Foreground shape, position jittered so position cannot carry the label.
    const size = randInt(rng, 8, 10);
    const cx = randInt(rng, 12, 19);
    const cy = randInt(rng, 12, 19);
    const mask = rasterizeShape(cls, cx, cy, size, rng);
    fgMasks.set(mask, maskOff);
    const fg = jitter(FG_COLOR, rng, 0.03);
    for (let p = 0; p < IMG_PIXELS; p++) {
      if (mask[p] === 1) {
        images[imgOff + p * 3 + 0] = fg[0];
        images[imgOff + p * 3 + 1] = fg[1];
        images[imgOff + p * 3 + 2] = fg[2];
      }
    }
  }

  if (shuffleLabels) {
    // Null control: permute labels so no input feature can predict them.
    // Uint32Array, not a typed-array .map of labels — Uint8 wraps at 256.
    const order = new Uint32Array(n).map((_, i) => i);
    const lr = mulberry32(seed ^ 0x9e3779b9);
    for (let i = order.length - 1; i > 0; i--) {
      const j = Math.floor(lr() * (i + 1));
      const t = order[i];
      order[i] = order[j];
      order[j] = t;
    }
    const shuffled = new Uint8Array(n);
    for (let i = 0; i < n; i++) shuffled[i] = labels[order[i]];
    labels.set(shuffled);
  }

  return {
    seed,
    n,
    rho,
    cueType,
    split,
    accessible,
    images,
    labels,
    fgMasks,
    cue,
  };
}

// Mix two worlds (e.g. correlated + neutral) into one training world. Used by
// Level 3 where the player designs the training distribution under a budget.
export function mixWorlds(parts: World[], seed: number): World {
  const total = parts.reduce((s, w) => s + w.n, 0);
  const out = generateWorld({ seed, n: 0, rho: parts[0]?.rho ?? 0 });
  const images = new Float32Array(total * IMG_PIXELS * 3);
  const fgMasks = new Float32Array(total * IMG_PIXELS);
  const labels = new Uint8Array(total);
  const cue = new Int8Array(total).fill(-1);
  let off = 0;
  for (const w of parts) {
    images.set(w.images, off * IMG_PIXELS * 3);
    fgMasks.set(w.fgMasks, off * IMG_PIXELS);
    labels.set(w.labels, off);
    cue.set(w.cue, off);
    off += w.n;
  }
  // Shuffle so batches see a mixture.
  const order = new Uint32Array(total).map((_, i) => i);
  const rng = mulberry32(seed ^ 0x51ed);
  for (let i = total - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    const t = order[i];
    order[i] = order[j];
    order[j] = t;
  }
  const sImages = new Float32Array(total * IMG_PIXELS * 3);
  const sMasks = new Float32Array(total * IMG_PIXELS);
  const sLabels = new Uint8Array(total);
  const sCue = new Int8Array(total);
  for (let i = 0; i < total; i++) {
    const s = order[i];
    sImages.set(images.subarray(s * IMG_PIXELS * 3, (s + 1) * IMG_PIXELS * 3), i * IMG_PIXELS * 3);
    sMasks.set(fgMasks.subarray(s * IMG_PIXELS, (s + 1) * IMG_PIXELS), i * IMG_PIXELS);
    sLabels[i] = labels[s];
    sCue[i] = cue[s];
  }
  return {
    ...out,
    n: total,
    images: sImages,
    labels: sLabels,
    fgMasks: sMasks,
    cue: sCue,
  };
}

// Empirical P(cue agrees with label) over a world's cue-bearing images.
export function empiricalCorrelation(world: World): number {
  let withCue = 0;
  let agree = 0;
  for (let i = 0; i < world.n; i++) {
    if (world.cue[i] < 0) continue;
    withCue++;
    if (world.cue[i] === world.labels[i]) agree++;
  }
  return withCue === 0 ? 0 : agree / withCue;
}
