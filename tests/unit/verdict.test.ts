import { describe, expect, it } from 'vitest';
import { classifyOutcome } from '../../src/game/verdict.ts';
import type { EvalSuite } from '../../src/evaluate.ts';
import type { Ablation } from '../../src/explain.ts';

const ev = (m: number, f: number, n: number): EvalSuite => ({
  matched: { split: 'matched', n: 1, correct: 0, acc: m },
  flipped: { split: 'flipped', n: 1, correct: 0, acc: f },
  neutral: { split: 'neutral', n: 1, correct: 0, acc: n },
  gap: m - f,
});

const ab = (over: Partial<Ablation>): Ablation => ({
  n: 1,
  acc: 1,
  accNoBg: 1,
  accNoFg: 1,
  accSwap: 1,
  bgDrop: 0,
  fgDrop: 0,
  swapDrop: 0,
  cueReliance: 0,
  fillReliance: 0,
  swapWorld: null,
  ...over,
});

describe('classifyOutcome — the one shared evidence classifier', () => {
  it('high cue-swap reliance + large reversal gap = shortcut', () => {
    expect(
      classifyOutcome(ev(1, 0.02, 0.55), ab({ accSwap: 0.02, cueReliance: 0.98, fillReliance: 1 })),
    ).toBe('shortcut');
  });

  it('D1 regression: clean neutral-fill cannot launder a cue-reversal collapse', () => {
    // QA's live repro: neutral-heavy training learns "neutral→shape,
    // coloured→cheat", so fillReliance reads 4.6% while reversed data scores
    // 10%. The paired cue swap must still convict it.
    expect(
      classifyOutcome(
        ev(1, 0.103, 0.983),
        ab({ accSwap: 0.103, cueReliance: 0.897, fillReliance: 0.046 }),
      ),
    ).toBe('shortcut');
  });

  it('D2 regression: 91% flipped + low reliance = shape, even below the 94.3% score bar', () => {
    // The score bar is a challenge target, not a cheat detector — evidence
    // decides the verdict.
    expect(
      classifyOutcome(
        ev(0.955, 0.913, 0.9),
        ab({ accSwap: 0.9, cueReliance: 0.09, fillReliance: 0.1 }),
      ),
    ).toBe('shape');
  });

  it('near-chance everywhere = undertrained, not a shortcut story', () => {
    expect(classifyOutcome(ev(0.51, 0.49, 0.5), ab({ acc: 0.5, cueReliance: 0, fillReliance: 0 }))).toBe(
      'undertrained',
    );
    // A collapsed model has no reliance to attribute (both null) and scored
    // little on the ablation window.
    expect(
      classifyOutcome(ev(0.7, 0.65, 0.7), ab({ acc: 0.6, cueReliance: null, fillReliance: null })),
    ).toBe('undertrained');
  });

  it('mixed evidence = inconclusive, never a forced story', () => {
    expect(
      classifyOutcome(
        ev(0.9, 0.7, 0.7),
        ab({ accSwap: 0.6, cueReliance: 0.4, fillReliance: 0.4 }),
      ),
    ).toBe('inconclusive');
  });
});
