import { mulberry32 } from './rng.ts';

export interface CI {
  mean: number;
  lo: number;
  hi: number;
  n: number;
  B: number;
}

// Percentile bootstrap CI on the mean, with a seeded resampler.
export function bootstrapCI(values: number[], B = 2000, seed = 1): CI {
  const n = values.length;
  const mean = values.reduce((s, v) => s + v, 0) / n;
  if (n < 2) return { mean, lo: mean, hi: mean, n, B };
  const rng = mulberry32(seed);
  const means = new Float64Array(B);
  for (let b = 0; b < B; b++) {
    let s = 0;
    for (let i = 0; i < n; i++) s += values[Math.floor(rng() * n)];
    means[b] = s / n;
  }
  const sorted = Array.from(means).sort((a, b2) => a - b2);
  const q = (p: number) => sorted[Math.min(B - 1, Math.max(0, Math.floor(p * B)))];
  return { mean, lo: q(0.025), hi: q(0.975), n, B };
}

// Exact two-sided sign test on paired differences (no normal approximation —
// with ~5 seeds the exact binomial is both cheap and honest).
export function signTest(diffs: number[]): { positives: number; negatives: number; p: number } {
  let pos = 0;
  let neg = 0;
  for (const d of diffs) {
    if (d > 0) pos++;
    else if (d < 0) neg++;
  }
  const n = pos + neg;
  if (n === 0) return { positives: 0, negatives: 0, p: 1 };
  const k = Math.min(pos, neg);
  // p = 2 * P(X <= k), X ~ Binomial(n, 0.5)
  let tail = 0;
  for (let i = 0; i <= k; i++) tail += binom(n, i) / 2 ** n;
  return { positives: pos, negatives: neg, p: Math.min(1, 2 * tail) };
}

function binom(n: number, k: number): number {
  if (k < 0 || k > n) return 0;
  k = Math.min(k, n - k);
  let r = 1;
  for (let i = 0; i < k; i++) r = (r * (n - i)) / (i + 1);
  return r;
}
