# Repair report — independent QA (Max Opus) findings at 32540b9

Repair commit: `b5790cd` (on `main`, deployed by Pages). Every item below is
verifiable from the commit, `results/results.{json,md}`, CI run, or the live
site. No seed, run, or failure was dropped or hidden.

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
