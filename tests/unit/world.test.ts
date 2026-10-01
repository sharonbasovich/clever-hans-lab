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

  it('matched and flipped are pixel-paired: same shapes/labels, cue inverted', () => {
    // The paired cue swap relies on this invariant: same seed + 'flipped'
    // regenerates identical shapes/labels/positions with every cue reversed.
    for (const rho of [1.0, 0.7]) {
      const m = generateWorld({ seed: 42, n: 100, rho });
      const f = generateWorld({ seed: 42, n: 100, rho, split: 'flipped' });
      expect(Buffer.from(m.labels.buffer).equals(Buffer.from(f.labels.buffer))).toBe(true);
      expect(Buffer.from(m.fgMasks.buffer).equals(Buffer.from(f.fgMasks.buffer))).toBe(true);
      for (let i = 0; i < m.n; i++) expect(f.cue[i]).toBe(1 - m.cue[i]);
      // Shape pixels are byte-identical; only the background carries the swap.
      let fgIdentical = true;
      let bgDiffers = false;
      for (let i = 0; i < m.n && fgIdentical; i++) {
        const off = i * IMG_PIXELS;
        for (let p = 0; p < IMG_PIXELS; p++) {
          for (let c = 0; c < 3; c++) {
            const mv = m.images[(off + p) * 3 + c];
            const fv = f.images[(off + p) * 3 + c];
            if (m.fgMasks[off + p] === 1 && mv !== fv) fgIdentical = false;
            if (m.fgMasks[off + p] === 0 && mv !== fv) bgDiffers = true;
          }
        }
      }
      expect(fgIdentical).toBe(true);
      expect(bgDiffers).toBe(true);
    }
  });

  it('the fully cue-agreeing paired set shares the matched stream and inverts every cue', () => {
    // The cue-swap reliance metric evaluates on a rho=1.0 matched world and
    // its rho=1.0 flipped twin: identical shapes/labels/positions to the
    // same-seed matched world at ANY rho, with every cue agreeing on one and
    // every cue inverted on the other — a swap that can never help a cheat.
    for (const seed of [42, 31, 58313]) {
      const agree = generateWorld({ seed, n: 100, rho: 1.0, split: 'matched' });
      const swap = generateWorld({ seed, n: 100, rho: 1.0, split: 'flipped' });
      const mid = generateWorld({ seed, n: 100, rho: 0.7, split: 'matched' });
      // Every image's cue agrees in the agree set and disagrees in the swap.
      expect(empiricalCorrelation(agree)).toBe(1);
      expect(empiricalCorrelation(swap)).toBe(0);
      // Same image stream as the mixed-rho matched world: identical
      // labels and shape masks; only the cue colour differs.
      expect(Buffer.from(agree.labels.buffer).equals(Buffer.from(mid.labels.buffer))).toBe(true);
      expect(Buffer.from(agree.fgMasks.buffer).equals(Buffer.from(mid.fgMasks.buffer))).toBe(true);
      expect(Buffer.from(swap.labels.buffer).equals(Buffer.from(mid.labels.buffer))).toBe(true);
      for (let i = 0; i < agree.n; i++) expect(swap.cue[i]).toBe(1 - agree.cue[i]);
    }
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
