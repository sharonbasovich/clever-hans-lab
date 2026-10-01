import type { EvalSuite } from '../evaluate.ts';
import type { Ablation } from '../explain.ts';

// The verdict is derived from measured evidence, never pre-written — and this
// one classifier is shared by the exam note, the quiz and the verdict screen
// so the game can never say two different things about the same run.
// cueReliance is measured on a FULLY cue-agreeing paired set: the identical
// image stream with every cue inverted. Because the base set has no
// disagreeing images, the swap is strictly harmful to a cue-user and cannot
// cancel harmful losses against helpful swaps (the round-3 mid-rho flaw).
// Five honest outcomes:
//   shortcut        — most of its wins die when the cue is reversed (reliance
//                     ≥ 0.5) AND the strict-reversal gap is large.
//   partial shortcut — real but not dominant reliance (≥ 0.25): it learned
//                     some shape AND still leans on the cue. Not "clean".
//   shape           — survives the reversal AND shows low cue reliance —
//                     clean shape learning.
//   undertrained    — the model learned almost nothing (constant/near-chance);
//                     there is no reliance to attribute.
//   inconclusive    — mixed evidence; show the numbers, force no story.
export type VerdictKind = 'shortcut' | 'partial' | 'shape' | 'undertrained' | 'inconclusive';

export function classifyOutcome(ev: EvalSuite, ab: Ablation): VerdictKind {
  const best = Math.max(ev.matched.acc, ev.flipped.acc, ev.neutral.acc);
  const learnedNothing =
    best < 0.62 ||
    (ab.cueReliance === null && ab.fillReliance === null && ab.acc < 0.65);
  if (learnedNothing) return 'undertrained';
  if (ab.cueReliance !== null && ab.cueReliance >= 0.5 && ev.gap >= 0.15)
    return 'shortcut';
  if (
    (ab.cueReliance === null || ab.cueReliance < 0.25) &&
    ev.flipped.acc >= 0.9 &&
    Math.abs(ev.gap) < 0.15
  )
    return 'shape';
  if (ab.cueReliance !== null && ab.cueReliance >= 0.25) return 'partial';
  return 'inconclusive';
}
