# Repair report — independent QA findings

Round 3 (below) covers the independent review at `9a76f5d`. Rounds 2 and
1 follow it. Every item is verifiable from the commit,
`results/results.{json,md}`, CI, or the live site. No seed, run, or
failure was dropped or hidden; the author's own QA recording is a smoke
check, not independent verification — the findings here come from the
independent reviewer.

---

# Round 3 — findings at 9a76f5d

**R1 — paired swap cancelled itself on mixed cue-agreement sets (high).**
The cue-swap metric from round 2 evaluated on the swapped twin of the
*matched* test set — but at rho<1.0 the matched set contains disagreeing
images, and on those the inversion HELPS a cheater, cancelling the drops
on agreeing images. QA: 14/39 runs scoring <50% on full reversal read
reliance 0.23–0.49 → "inconclusive" (e.g. rho0.7 seed31: matched 81%,
reversed 32.3%, reliance 0.314). Fix: the reliance pair is now a
**fully cue-agreeing paired evaluation set** — the identical image
stream (same seed, tested invariant) at rho=1.0, one copy with every cue
agreeing (`accAgree`) and one with every cue inverted (`accSwap`).
Every inversion is strictly harmful to a cue-user, so nothing cancels.
`cueReliance = (accAgree − accSwap) / accAgree`. Mid-rho runs that read
0.31 before now read 0.6–0.7. Matched and fully reversed accuracy are
retained alongside, and the classifier (`classifyOutcome`) now has five
precise outcomes — shortcut / **partial** / shape / undertrained /
inconclusive — so mid-reliance is named "partial shortcut reliance",
never laundered into "inconclusive" or "clean".

**R2 — quiz bypassed the shared classifier (high).**
The Level-2 quiz had its own binary `(cueReliance ?? 0) > 0.5` rule, so
all 28 inconclusive runs in QA's 90-run set got a false
"Correct — it really read the shape". Fix: the quiz is wired directly
to `classifyOutcome` with explicit handling of all five outcomes —
inconclusive answers report "no clean call — the numbers are genuinely
mixed" (never marked correct), undertrained keeps its "no call to make"
state, partial gets a third option ("a bit of both"). Rendered
browser tests assert the exam note, quiz response and verdict title for
each of the five outcomes on injected measured evidence — not just the
classifier unit tests.

**R3 — null gate per-seed bound not statistically justified (medium).**
Fresh QA seed 58313 read neutral accuracy 65.5%, above the [35,65]%
per-seed bound; QA's fresh mean was 53.8%, cue-consistency 105.1%, and
5/10 null runs collapsed (1 recovered). A memorizing null CAN
legitimately exceed a per-seed bound (a particular shuffle leaves
residual label-structure the model partially exploits), so the bound
was never justified. Redesign (predeclared): the gate is a one-sample
t-check on the MEAN neutral accuracy of non-degenerate nulls
(|t|≤3, ≥half of seeds non-degenerate) plus mean cue-consistency
∈[85,115]%. **Unrecovered collapses are excluded and disclosed
separately** — a constant predictor trivially sits at chance and must
not pass a chance gate for free. The old failure is retained verbatim:
this run's seed 58313 reads 65.5% as a disclosed "legacy bound
violation", and the round-2 gate's numbers (mean matched 39.3%,
mean|gap| 35.9%, max|gap| 93.7%) stay in `results.md`.

**R4 — coverage: intermediate-rho designs + fresh seeds (medium).**
The grid grew to 25 seeds × 8 designs: {rho, train-neutral} =
{1.0,0}, {0.9,0}, {0.85,0.6}, {0.7,0}, {0.7,0.5}, {0.6,0.5}, {0.5,0},
{0.5,0.6}; QA's disclosed seeds 31/47/58313 are pinned as regression
seeds and four fresh seeds (42, 1009, 31337, 777333) were predeclared
before measurement. New gate G6 requires clearly positive mean
reliance on the {0.7,0} design and strictly above the rho=0.5 baseline
— the cancellation bug would have read near-chance there. Measured:
{0.7,0}=0.435, {0.85,0.6}=0.455, {0.5,0}=0.006 — G6 PASS.

**R5 — stale copy (low).** Intro "exploit it in seconds" → "exploit it
on its own exam"; the unmeasured "WebGL is normally much faster" clause
removed (the honest measured-timings sentence stays); the screen-reader
ablation announcement now says "of its wins on cue-agreeing tests
depended on the cue", not "lived in the background"; the exam note's
"It never learned shapes" replaced with "On the tested shift, its wins
depended on the cue — not the shape" (a measured claim about cue
dependence, not about never learning).

---


# Round 2 — findings at 0e4fc50

**D1 — neutral-fill ablation laundered a cheat (high).**
QA showed models trained with neutral backgrounds learn
"neutral → shape, coloured → cheat": in two 60%-neutral designs, 17/20
runs scored <60% on fully reversed data while the fill-reliance
diagnostic read <50% or n/a (live repro: exam 100/10.3/98.3 with
fillReliance 4.6%). Fix: a **paired opposite-cue swap** is now the
headline reliance measure — `generateWorld(seed, 'flipped')` regenerates
byte-identical shapes/labels/positions with every cue inverted (the
invariant is unit-tested), and `cueReliance = swapDrop/matchedAcc` is the
fraction of wins that vanish. On the repro design it reads ~89.7%, not
4.6%. Neutral-fill stays as `fillReliance`, labelled a secondary
diagnostic that can under-report cheating. One shared classifier
(`src/game/verdict.ts#classifyOutcome`) drives the exam note, the
Level-2 quiz answer and the verdict screen.

**D2 — exam bar vs verdict bar contradiction (high).**
The 94.3% win target and the classifier's 90% evidence bar could disagree
(4/70 fresh runs). Now there is exactly one classifier (shortcut / shape
/ undertrained / inconclusive). The L3 threshold is labelled a
**challenge target — a score bar, not a cheat detector**: missing it with
clean evidence says "shape, below target", never implies cheating.

**D3 — cancellation race (high).**
A superseded run's unconditional `finally { trainingNow = false }`
cleared the flag for the *current* run, so navigate-away never cancelled
it; ~209 s later its stale completion evaluated and forced `#/exam`.
Also, the Accessible-cues toggle never cancelled an in-flight run.
Fix: the model trains in a local variable and is published only after an
ownership check (`gen === runGen && !cancelled`); a superseded model is
disposed locally; `finally` clears `trainingNow` only while still owner;
the a11y toggle calls `cancelTraining()`. A dedicated e2e test reproduces
the exact sequence (toggle mid-run → restart → navigate to Lab → wait
past the stale window → assert no hijack).

**D4 — G4 null gate was itself invalid (medium).**
QA measured mean matched 39.3% (gate asked 40–60%), mean |gap| 35.9%
(gate ≤15%), max |gap| 93.7% on fresh seeds — because a null at rho=1 can
fit shuffled labels via the cue with an *arbitrary sign*, so matched
accuracy and |gap| are not chance statistics. Redesigned (justification
in `results.md` and the lab page): the gate now requires **neutral
(shape-only) accuracy ≈ chance** on every seed (35–65%) and on the mean
(40–60%) — shape cannot predict shuffled labels — plus mean
cue-consistency (matched+flipped) ∈ [85,115]%. Per-seed sums are
disclosed, not bounded (the sum is 100% only for a pure cue responder —
seed 101 legitimately reads 128%). The old gate's failure numbers stay in
`results.md`.

**D5 — unsupported speed claims (medium).**
"Works on a phone" / "five minutes in any classroom browser" removed
everywhere. Published numbers are the measured ones: ~164 s on the
pure-JS CPU fallback and ~639 s under a 4× CPU slowdown (independent
QA's live measurements), ~1.4 s on the native harness backend.
Phones and WebGL/GPU speed are explicitly stated as unmeasured.

**D6 — demo captions covered content; demo mode undisclosed (medium).**
The caption bar now takes real layout space (the app scrolls in its own
region — a fixed overlay could cover content at ~11 s/24 s/44 s), and the
video shows a persistent on-screen disclosure: "Demo mode — Level 1 uses
reduced data (480 images, 5 epochs; the real level is 1,200/8)."

**D7 — seed-count inconsistency (low).**
Level-3 copy said "5-seed" while Lab listed 8; everything now reads the
real count: 18 seeds (original 8 + the 10 seeds QA probed, pinned as
disclosed regression seeds — never presented as held-out evidence).

## Fresh measurements (round-2 commit)

- rho=1.0: gap **100% in all 18 seeds**; cue-swap reliance 1.000.
- rho=0.9: 16/18 shortcut (86.8–94.5%), 2/18 partial (51–65%) — disclosed.
- rho=0.5: strict flipped 93.3–100%, gaps −1.8%…+1.5%, reliance ~0.002.
- Null: mean neutral 50.5% (per-seed 44–55%), matched free 3.5–65%,
  cue-consistency mean ≈100%, 6 collapse→retry recoveries.
- Gates G1–G5 PASS; determinism confirmed; L3 bar derived at 94.5%.
- No main-run collapse in 54 runs; unit tests 26; e2e incl. the D3
  regression test.

---

# Round 1 — findings at 32540b9 (repaired at b5790cd)

## High findings

**H1 — hard-coded verdict.**
`src/game/verdict.ts` now classifies from measured evidence into four honest
outcomes: `shortcut` (bgReliance ≥ 0.5 AND gap ≥ 15pp), `shape`
(bgReliance < 0.3 or null AND strict flipped ≥ 90%), `undertrained` (best
accuracy < 62% or ablation costs ~nothing), `inconclusive`. The verdict screen
and the Level-2 quiz render all four; `bgReliance` is `number | null` (null =
"learned nothing", displayed as such — M4).

**H2 — L3 training collapses.**
Root cause: Adam lr=0.01 overshoots into a constant predictor on ~1/5 of
inits (diagnosed on both builder and QA seed sets). Fix: lr=0.005 — zero
collapses across every measured (seed, rho) combination — PLUS a disclosed
detect-and-retry: `isConstantPredictor` probes the trained model; a collapse
triggers one deterministic reinit+retrain (seed+7919) flagged in the UI
("First attempt collapsed…") and in results (`collapsed`, `recovered`,
`collapses` block). Nothing is filtered or force-set. L3 evaluation is now a
strict full reversal at every rho, and L3 always uses the real 1600-image
budget — shrinking it in demo mode made the measured bar unreachable.

**H3 — run lifecycle.**
`Session.train()` is single-run and cancellable: a generation counter +
cancel flag; it returns `null` on cancel/supersede so stale completions can't
navigate or write UI. Routing cancels training when leaving `#/train`; result
screens (`exam/heatmap/verdict`) require a real `session.evaluation` — a
reload or direct link redirects to `#/build` with a notice; `#/train` without
a world rebuilds instead of dead-ending. Covered by a new e2e lifecycle test
(reload recovery, mid-run cancel, retry).

**H4 — gates/CI honesty.**
- 8 seeds now (original 5 + 606/707/808, declared before measurement); QA's
  held-out seeds are not reused.
- G2 back to |mean gap| < 5pp; G3 keeps the stronger ablation metric but at
  the original p<0.05 sign-test bar (measured p=0.0078, 8/8 positive).
- Flipped eval is a strict full reversal at every rho (H2/M3).
- Smoke vs full: smoke enforces G1+G5 and reports G2–G4 as advisory
  (meaningful only in aggregate — labelled, not hidden); the FULL 8-seed
  harness runs in CI, and `scripts/verify-results.ts` fails the build on any
  failed gate, missing seed, or a >5MB bundle.
- Brief's perf gate is disclosed as targets T1/T2: median run ~1.4s on the
  native backend; JS bundle 0.93MB; the UI states the real CPU expectation
  instead of faking speed.

## Mediums / lows

- **M1** null control: per-seed matched + |gap| disclosed (−25.8%…+26.3%),
  mean|gap|≤15pp gate, arbitrary-sign drift explained in results.md.
- **M2** exam copy: "never seen in training" replaced with an accurate
  small-space caveat; the strict flipped set carries the evidence.
- **M3** strict reversal everywhere (harness + Session.makeTestSets).
- **M5** demo re-recorded: 110s, caption bar no longer covers content
  (`body:has` padding), "confidence intervals" claim removed, and the L3
  run shows a real win on the real budget (94.4% ≥ 94.3% bar).
- **M6** backend label now read from `results.json` meta (tensorflow); lab
  table wrapped in overflow-x; L3 sliders have aria-labels (e2e-asserted).
- **M7** citations added: Ribeiro et al. 2016 (snow/husky), Zech et al.
  2018 PLoS Medicine (hospital confound) — in-app and teacher page.
- **L1** `engines: >=20 <23` (+ README Node 22 note); **L2** zero-padded
  final batch removed (real bug — trained toward the all-zero label);
  **L3** trivial.

## Fresh measurements (this commit)

- rho=1.0: gap **100% in all 8 seeds**, bgReliance 1.000.
- rho=0.9: 6/8 shortcut (88–92%), 2/8 partial (51–65%) — disclosed.
- rho=0.5: strict flipped 93.3–100%, gaps −1.2%…+0.7%.
- Null: mean matched 54.8%, mean|gap| 12.8%, 3 collapse→retry recoveries.
- Gates G1–G5 PASS; determinism confirmed; L3 bar derived at 94.3%.

## Residual honesty notes

- The 30s CPU target fails on a pure-JS browser CPU backend (measured ~140s
  earlier); reported, not hidden. Native backend ~1.4s.
- A single-seed null CAN drift far off chance (seen: |gap| 26.3%); that is
  why the control is powered at 8 seeds and per-seed values are published.
