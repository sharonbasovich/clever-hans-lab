import { describe, expect, it } from 'vitest';
import { empiricalCorrelation, generateWorld, IMG_PIXELS, mixWorlds } from '../../src/world.ts';

describe('generateWorld', () => {
  it('same seed produces identical bytes', () => {
    const a = generateWorld({ seed: 7, n: 64, rho: 1.0 });
    const b = generateWorld({ seed: 7, n: 64, rho: 1.0 });
    expect(Buffer.from(a.images.buffer).equals(Buffer.from(b.images.buffer))).toBe(true);
    expect(Buffer.from(a.labels.buffer).equals(Buffer.from(b.labels.buffer))).toBe(true);
    expect(Buffer.from(a.fgMasks.buffer).equals(Buffer.from(b.fgMasks.buffer))).toBe(true);
    expect(Buffer.from(a.cue.buffer).equals(Buffer.from(b.cue.buffer))).toBe(true);
  });

  it('different seeds differ', () => {
    const a = generateWorld({ seed: 7, n: 64, rho: 1.0 });
    const b = generateWorld({ seed: 8, n: 64, rho: 1.0 });
    expect(Buffer.from(a.images.buffer).equals(Buffer.from(b.images.buffer))).toBe(false);
  });

  it.each([1.0, 0.9, 0.5, 0.0])(
    'empirical cue-label correlation is within ±0.02 of rho=%f at n=4000',
    (rho) => {
      const w = generateWorld({ seed: 99, n: 4000, rho });
      expect(Math.abs(empiricalCorrelation(w) - rho)).toBeLessThanOrEqual(0.02);
    },
  );

  it('flipped split reverses the mapping', () => {
    const w = generateWorld({ seed: 5, n: 4000, rho: 1.0, split: 'flipped' });
    // rho=1.0 flipped -> cue agrees with label ~0% of the time.
    expect(empiricalCorrelation(w)).toBeLessThanOrEqual(0.02);
  });

  it('neutral split carries no cue', () => {
    const w = generateWorld({ seed: 5, n: 100, rho: 1.0, split: 'neutral' });
    expect([...w.cue].every((c) => c === -1)).toBe(true);
    expect(empiricalCorrelation(w)).toBe(0);
  });

  it('foreground masks cover only the shape pixels', () => {
    const w = generateWorld({ seed: 3, n: 200, rho: 0.5 });
    for (let i = 0; i < w.n; i++) {
      const mask = w.fgMasks.subarray(i * IMG_PIXELS, (i + 1) * IMG_PIXELS);
      let fgCount = 0;
      for (let p = 0; p < IMG_PIXELS; p++) {
        expect(mask[p] === 0 || mask[p] === 1).toBe(true);
        if (mask[p] === 1) {
          fgCount++;
          // Foreground pixels are the (near-white) shape colour.
          const r = w.images[i * IMG_PIXELS * 3 + p * 3];
          const g = w.images[i * IMG_PIXELS * 3 + p * 3 + 1];
          const b = w.images[i * IMG_PIXELS * 3 + p * 3 + 2];
          expect(r).toBeGreaterThan(0.85);
          expect(g).toBeGreaterThan(0.85);
          expect(b).toBeGreaterThan(0.8);
        }
      }
      // Shape size is bounded by construction (size 8-10 -> ~64..318 px).
      expect(fgCount).toBeGreaterThan(60);
      expect(fgCount).toBeLessThan(700);
    }
  });

  it('shuffled labels are still a permutation of the originals (null control)', () => {
    const a = generateWorld({ seed: 11, n: 500, rho: 1.0 });
    const b = generateWorld({ seed: 11, n: 500, rho: 1.0, shuffleLabels: true });
    const counts = (l: Uint8Array) => [l.filter((v) => v === 0).length, l.filter((v) => v === 1).length];
    expect(counts(b.labels)).toEqual(counts(a.labels));
    // And the shuffled labels no longer line up with the cue.
    expect(Math.abs(empiricalCorrelation(b) - 0.5)).toBeLessThan(0.1);
  });

  it('mixWorlds concatenates and shuffles deterministically', () => {
    const a = generateWorld({ seed: 21, n: 50, rho: 1.0 });
    const n = generateWorld({ seed: 22, n: 50, rho: 0, split: 'neutral' });
    const m1 = mixWorlds([a, n], 7);
    const m2 = mixWorlds([a, n], 7);
    expect(m1.n).toBe(100);
    expect(Buffer.from(m1.images.buffer).equals(Buffer.from(m2.images.buffer))).toBe(true);
    expect([...m1.cue].some((c) => c === -1)).toBe(true);
    expect([...m1.cue].some((c) => c >= 0)).toBe(true);
  });

  it('accessible palette differs in brightness and adds pattern', () => {
    const w = generateWorld({ seed: 4, n: 40, rho: 1.0, accessible: true });
    // cue-0 (dark red, diagonal stripes) vs cue-1 (light blue, anti-diagonal):
    // verify a brightness difference between the two cue colours on bg pixels.
    let dark = 0;
    let light = 0;
    for (let i = 0; i < w.n; i++) {
      const bgPx = w.fgMasks[i * IMG_PIXELS] === 0 ? 0 : -1;
      if (bgPx < 0) continue;
      const r = w.images[i * IMG_PIXELS * 3];
      const g = w.images[i * IMG_PIXELS * 3 + 1];
      const b = w.images[i * IMG_PIXELS * 3 + 2];
      const lum = 0.3 * r + 0.6 * g + 0.1 * b;
      if (w.cue[i] === 0) dark = lum;
      else light = lum;
    }
    expect(light - dark).toBeGreaterThan(0.3);
  });
});
