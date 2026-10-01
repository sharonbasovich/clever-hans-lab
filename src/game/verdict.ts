import type { EvalSuite } from '../evaluate.ts';
import type { Ablation } from '../explain.ts';

// The verdict is derived from measured evidence, never pre-written — and this
// one classifier is shared by the exam note, the quiz and the verdict screen
// so the game can never say two different things about the same run.
// Four honest outcomes:
//   shortcut      — reversing the cue colour destroys accuracy (cueReliance
//                   high) AND the strict-reversal gap is large. The paired
//                   cue swap — not neutral fill — is the reliance measure:
//                   fill-ablation can hide a cheat that learned
//                   "neutral → shape, coloured → cheat".
//   shape         — survives the reversal AND shows low cue reliance.
//   undertrained  — the model learned almost nothing (constant/near-chance);
//                   there is no reliance to attribute.
//   inconclusive  — mixed evidence; show the numbers, force no story.
export type VerdictKind = 'shortcut' | 'shape' | 'undertrained' | 'inconclusive';

export function classifyOutcome(ev: EvalSuite, ab: Ablation): VerdictKind {
  const best = Math.max(ev.matched.acc, ev.flipped.acc, ev.neutral.acc);
  const learnedNothing =
    best < 0.62 ||
    (ab.cueReliance === null && ab.fillReliance === null && ab.acc < 0.65);
  if (learnedNothing) return 'undertrained';
  if (ab.cueReliance !== null && ab.cueReliance >= 0.5 && ev.gap >= 0.15)
    return 'shortcut';
  if (
    (ab.cueReliance === null || ab.cueReliance < 0.3) &&
    ev.flipped.acc >= 0.9 &&
    Math.abs(ev.gap) < 0.15
  )
    return 'shape';
  return 'inconclusive';
}
