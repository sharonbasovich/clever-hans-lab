import * as tf from '@tensorflow/tfjs';
import { beforeAll, describe, expect, it } from 'vitest';
import { bgMass, occlusionMap } from '../../src/explain.ts';
import { bootstrapCI, signTest } from '../../src/stats.ts';
import { IMG_PIXELS, IMG_SIZE } from '../../src/world.ts';

beforeAll(async () => {
  await tf.setBackend('cpu');
  await tf.ready();
});

describe('occlusionMap', () => {
  it('points to the pixels a known toy model actually reads', async () => {
    // Toy model: flatten -> dense whose class-0 logit is driven ONLY by a
    // 4x4 region in the top-left corner. Occluding that region must dominate
    // the heatmap.
    const model = tf.sequential();
    model.add(tf.layers.flatten({ inputShape: [IMG_SIZE, IMG_SIZE, 3] }));
    model.add(tf.layers.dense({ units: 2, activation: 'softmax', useBias: false }));
    const kernel = tf.tidy(() => tf.zeros([IMG_PIXELS * 3, 2]));
    const buf = await kernel.buffer();
      for (let y = 0; y < 4; y++) {
        for (let x = 0; x < 4; x++) {
          for (let c = 0; c < 3; c++) {
            // weight 0.1 keeps class-0 logit in the mid-range (~4.8) so the
            // softmax isn't saturated and the occlusion drop is measurable.
            buf.set(0.1, (y * IMG_SIZE + x) * 3 + c, 0); // class 0 reads this region
          }
        }
      }
    const kernelT = buf.toTensor();
    kernel.dispose();
    model.layers[1].setWeights([kernelT]);
    kernelT.dispose();

    const img = new Float32Array(IMG_PIXELS * 3).fill(0.5);
    for (let y = 0; y < 4; y++) {
      for (let x = 0; x < 4; x++) {
        for (let c = 0; c < 3; c++) img[(y * IMG_SIZE + x) * 3 + c] = 1;
      }
    }
    const heat = occlusionMap(model, img, 4);
    // Heat should be concentrated in the 4x4 top-left region.
    let targetHeat = 0;
    let totalHeat = 0;
    for (let i = 0; i < IMG_PIXELS; i++) {
      totalHeat += heat[i];
      const y = Math.floor(i / IMG_SIZE);
      const x = i % IMG_SIZE;
      if (x < 4 && y < 4) targetHeat += heat[i];
    }
    expect(totalHeat).toBeGreaterThan(0);
    expect(targetHeat / totalHeat).toBeGreaterThan(0.95);
    model.dispose();
  });

  it('bgMass is 1 when all heat is on background, 0 when all on foreground', () => {
    const fg = new Float32Array(IMG_PIXELS);
    fg[0] = 1; // only pixel 0 is foreground
    const heatBg = new Float32Array(IMG_PIXELS);
    heatBg[1] = 0.9;
    const heatFg = new Float32Array(IMG_PIXELS);
    heatFg[0] = 0.9;
    expect(bgMass(heatBg, fg)).toBe(1);
    expect(bgMass(heatFg, fg)).toBe(0);
    expect(bgMass(new Float32Array(IMG_PIXELS), fg)).toBe(0);
  });
});

describe('stats', () => {
  it('bootstrap CI brackets the sample mean', () => {
    const values = [0.8, 0.85, 0.9, 0.88, 0.92];
    const ci = bootstrapCI(values, 2000, 5);
    const mean = values.reduce((s, v) => s + v, 0) / values.length;
    expect(ci.mean).toBeCloseTo(mean, 10);
    expect(ci.lo).toBeLessThanOrEqual(mean);
    expect(ci.hi).toBeGreaterThanOrEqual(mean);
    expect(ci.lo).toBeGreaterThanOrEqual(Math.min(...values));
    expect(ci.hi).toBeLessThanOrEqual(Math.max(...values));
  });

  it('bootstrap is deterministic per seed', () => {
    const values = [0.8, 0.85, 0.9, 0.88, 0.92];
    expect(bootstrapCI(values, 500, 9)).toEqual(bootstrapCI(values, 500, 9));
  });

  it('sign test: all-positive diffs are significant, mixed diffs are not', () => {
    expect(signTest([1, 2, 3, 4, 5]).p).toBeLessThan(0.1);
    expect(signTest([1, -1, 2, -2, 0.5]).p).toBe(1);
    expect(signTest([0, 0, 0])).toEqual({ positives: 0, negatives: 0, p: 1 });
  });
});
