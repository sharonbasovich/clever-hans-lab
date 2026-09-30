import type { EvalSuite } from '../evaluate.ts';
import type { Ablation } from '../explain.ts';

// The verdict is derived from measured evidence, never pre-written.
// Four honest outcomes:
//   shortcut      — accuracy collapses when the cue is reversed AND the
//                   ablation shows accuracy lives in the background.
//   shape         — survives the reversal AND relies on the shape.
//   undertrained  — the model learned almost nothing (constant/near-chance);
//                   there is no reliance to attribute.
//   inconclusive  — mixed evidence; show the numbers, force no story.
export type VerdictKind = 'shortcut' | 'shape' | 'undertrained' | 'inconclusive';

export function classifyOutcome(ev: EvalSuite, ab: Ablation): VerdictKind {
  const best = Math.max(ev.matched.acc, ev.flipped.acc, ev.neutral.acc);
  const learnedNothing = best < 0.62 || (ab.bgReliance === null && ab.acc < 0.65);
  if (learnedNothing) return 'undertrained';
  if (ab.bgReliance !== null && ab.bgReliance >= 0.5 && ev.gap >= 0.15) return 'shortcut';
  if (
    (ab.bgReliance === null || ab.bgReliance < 0.3) &&
    ev.flipped.acc >= 0.9 &&
    Math.abs(ev.gap) < 0.15
  )
    return 'shape';
  return 'inconclusive';
}
