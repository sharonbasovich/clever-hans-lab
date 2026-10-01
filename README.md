# Clever Hans Lab

A browser game that teaches neural-network **shortcut learning** by letting you
commit the crime yourself. You plant a cheat in a synthetic dataset (every
circle sits on red, every triangle on blue), train a *real* convolutional
network in your browser, watch it ace the test — then flip the colours and
watch it collapse. In the final level you redesign the training data so the
model is forced to learn shape.

No mocked training, no fabricated accuracy. Every number on screen comes from
live TensorFlow.js inference or from `results/results.json`, which the
deterministic headless harness regenerates.

**Live demo:** https://sharonbasovich.github.io/clever-hans-lab/

Built for the ML Empowerment Build Challenge 3.0 by Sharon Basovich
(University of Waterloo), solo.

## Quickstart

```bash
npm ci
npm run dev        # vite dev server
npm run build      # production build → dist/
npm run preview    # serve the production build
```

No accounts, no backend, no keys. Everything runs in the browser
(WebGL backend with automatic CPU fallback).

## Verify it yourself

```bash
npm run lint          # eslint
npm run typecheck     # tsc --noEmit
npm test              # vitest: unit tests (world/model/train/explain/stats)
npm run harness       # 18 seeds × rho {1.0, 0.9, 0.5} + null + determinism → results/
npx playwright test   # e2e golden path + lifecycle + keyboard/375px checks
npm run verify        # re-check committed results gates + bundle size
```

`npm run harness:smoke` is the fast CI plumbing check (1 seed): it enforces
G1/G5 and reports the other gates as advisory — the full harness enforces
every gate and also runs in CI.

## What the harness measured

Full table: [`results/results.md`](results/results.md)
(generated 2026-10-01, tfjs 4.22.0, tensorflow backend, ~112s wall).

Headline numbers, **18 seeds** (101–808 declared before measurement +
the 10 regression seeds independent QA probed, disclosed) ×
rho ∈ {1.0, 0.9, 0.5}, n=1200 train / 400 test per split. The flipped set
is a **strict full reversal** — the cue disagrees with the label on every
image, at every rho.

- **rho = 1.0** — the cheat always works: shortcut gap
  (matched − flipped accuracy) is **100% in all 18 seeds**. Paired
  cue-swap reliance = 1.000: invert only the cue colour on identical
  shapes and the model collapses to ~0%.
- **rho = 0.9** — seed-dependent, reported honestly: 16 of 18 seeds learn
  the shortcut (gaps 86.8–94.5%); 2 seeds partially collapse (gaps 51–65%).
  No seed was dropped.
- **rho = 0.5** — no cheat planted, no gap (−1.8% to +1.5%), shape learned
  (strict flipped accuracy 93.3–100%).
- **Null control** — labels shuffled across all 18 seeds: mean
  shape-only (neutral) accuracy 50.5%, every seed within 44–55%.
  Per-seed matched accuracy is free to sit anywhere (3.5% to 65%): with
  shuffled labels the model may still fit the noise via the cue with an
  arbitrary sign — cue-consistency (matched + flipped ≈ 100% for a pure
  cue responder) is disclosed per seed rather than gated as a chance
  statistic. The old matched/|gap| gate was itself invalid; its failure
  numbers are kept in `results/results.md`.
- **Training collapses** — an aggressive optimizer can collapse a small
  CNN to a constant predictor. lr=0.005 plus a deterministic
  detect-and-retry recovered every measured collapse (6 null-control runs);
  collapses are counted and disclosed, never hidden.
- **Determinism** — identical (seed, rho) reproduces identical metrics
  (bit-identical under Node 22 on the native backend).

### Gates (go/no-go, measured — not tuned)

| Gate | Claim | Result |
|---|---|---|
| G1 | Shortcut gap ≥ 15pp in every seed at rho=1.0 (CI-low ≥ 25pp) | PASS (mean 100%) |
| G2 | rho=0.5: mean \|gap\| < 5pp and shape learned ≥ 80% | PASS (0.3%, 97.6%) |
| G3 | paired cue-swap reliance rho=1.0 ≥ 0.8 vs rho=0.5 ≤ 0.2, sign test p<0.05 | PASS (1.000 vs ~0, p≈0) |
| G4 | null control: neutral acc ≈ chance per seed and on mean; mean cue-consistency ∈ [85%, 115%] | PASS (50.5%) |
| G5 | identical (seed, rho) → identical metrics | PASS |

Performance targets (brief: ≤30s CPU train, ≤5MB bundle) are disclosed in
`results.md` as T1/T2 rather than hidden: median run ≈1.4s on the native
backend, 0.95MB JS bundle. Browser wall time, measured on the live site
by independent QA: ~164s on the pure-JS CPU fallback, ~639s under a 4×
CPU throttle. Phones and WebGL/GPU speed were **not** measured — no claim
is made about them.

Disclosed changes vs the original brief: G3's metric is the paired
opposite-cue swap rather than patch-occlusion mass (the sign test bar is
the original p<0.05); G4's null control was redesigned after QA showed
the matched/|gap| version is not a chance test (its failure numbers are
retained in `results.md`); the perf gate is reported as targets T1/T2.
Everything else matches the brief.

## How it works

- `src/world.ts` — seeded generator: 32×32 RGB images, circle/triangle,
  background colour correlated with the label at rate rho. Foreground masks
  are stored so ablation can surgically erase either region. Same seed →
  identical bytes (tested).
- `src/model.ts` + `src/train.ts` — a small CNN (conv 8×3 → pool → conv 16×3
  → pool → dense 32 → softmax 2, ~12k params) trained with tf.js Adam. Seeded
  shuffling; the loss curve on screen is the measured curve.
- `src/evaluate.ts` — three held-out splits: **matched** (training rule
  holds), **flipped** (cue↔label mapping reversed), **neutral** (grey
  background). The gap is the confession.
- `src/explain.ts` — "where did it look" probes: the headline
  **paired cue-swap reliance** (regenerate the SAME images with only the
  cue colour inverted — `cueReliance` = fraction of matched wins lost),
  the labelled secondary **neutral-fill ablation** (blank the whole
  background vs. erase the whole shape — `fillReliance`), and
  patch-occlusion log-prob heatmaps as a supporting view. Fill-ablation
  alone can be dodged when training data includes neutral backgrounds
  (the model learns "neutral → shape, coloured → cheat"), which is why
  the cue swap — impossible to dodge — is the gate metric.
- `src/stats.ts` — bootstrap CIs and an exact sign test.
- `scripts/harness.ts` — the whole experiment, headless and deterministic,
  writing `results/results.json` (the single source of truth the UI imports
  for the Level-3 win threshold and the lab page).
- `scripts/record-demo.ts` — Playwright records the scripted `?demo=1` run:
  real training at reduced n, captions drawn by the app itself.

## The three levels

1. **Meet Hans** — plant rho=1.0, train, sit the three exams.
2. **Catch the cheat** — read the ablation + occlusion evidence and call it.
3. **Fix the data** — set rho and a neutral fraction under a sample budget;
   the win threshold (strict-reversal accuracy ≥ ~94.5%) is derived from
   the measured rho=0.5 distribution, never hard-coded — and it is a
   challenge target, not a cheat detector; evidence decides the verdict.

## Accessibility & failure states

Keyboard-playable end to end, skip-link, aria-live announcements of results,
canvas text alternatives, `prefers-reduced-motion`, a colour-independent cue
mode (dark red + diagonal stripes vs. light blue + anti-diagonal stripes —
the cues differ in brightness *and* pattern), and honest failure states (if
training fails in-browser you get an error, not a fake number).

## Data rights

The dataset is **synthetic and generated entirely in code** — no external
images, no scraped data, no third-party dataset. Each image is
deterministically reproducible from `(seed, index)`. The generator is MIT
licensed with this repo; the generated data carries no rights encumbrances.

## Dependencies

Runtime: `@tensorflow/tfjs` ^4.22. Dev: `vite`, `typescript`, `vitest`,
`playwright`, `tsx`, `eslint`, `@tensorflow/tfjs-node` (native backend for
the harness only). See `THIRD_PARTY.md` for licenses.

## Limitations

- This is a toy system by design: the shortcut is planted so it can be
  measured exactly. Measured shortcut gaps describe *this* synthetic task —
  they are evidence about shortcut learning as a phenomenon, **not** about
  any specific production model or real-world bias magnitude.
- rho=0.9 is bistable across seeds (shortcut vs. partial collapse) —
  reported, not smoothed over.
- Small synthetic space: some test shapes resemble training shapes. The
  flipped set shares zero geometry+cue combos with training, so the
  reversal test carries the evidence.
- Browser training uses the WebGL backend (CPU fallback); wall time varies
  by device. Independently measured on the live site: ~164s on the pure-JS
  CPU fallback, ~639s under a 4× CPU slowdown simulation. Phones and
  WebGL/GPU speed were not measured — the app states the honest
  expectation instead of claiming either. The demo mode (`?demo=1`)
  reduces Level-1 n (480 images / 5 epochs — disclosed on screen) to keep
  the video under two minutes; the harness numbers are the full-size runs.
- The model is intentionally small (~12k params) so training is watchable;
  it is not a benchmark of modern architectures.

## AI assistance disclosure

Built with substantial assistance from Devin (Cognition AI): implementation,
test design, evaluation harness, documentation drafts, and this README.
Conception, direction, review, and submission decisions are the author's.
The underlying science is cited on the teacher page (Pfungst 1907;
Lapuschkin et al. 2019; Geirhos et al. 2020).

## Build timeline

- Sep 29–30, 2026: scaffold, world generator + tests, CNN train/eval,
  explainability probes, deterministic harness (all gates green on first
  full run after fixing a label-buffer bug caught by the smoke run),
  3-level game UI, accessibility pass, demo recording, docs.
- Sep 30: independent QA repair round 1 — verdict derived from measured
  evidence (4 outcomes), strict full-reversal evaluation, collapse
  detect+retry at lr=0.005, cancellable run lifecycle + reload recovery,
  8-seed gates at the original p<0.05 bar enforced in CI, disclosed
  gate/target changes.
- Oct 1: independent QA repair round 2 — paired cue-swap reliance
  replaces neutral-fill as the headline ablation (fill kept as a labelled
  secondary diagnostic), one shared evidence classifier across
  exam/quiz/verdict, ownership guards on every training-lifecycle path
  (incl. the accessible-cues toggle), the null control redesigned around
  shape-only accuracy after the matched/|gap| gate was shown invalid
  (failure numbers retained), 18-seed regression suite, demo captions
  re-laid-out so they cannot cover content, demo-mode disclosure on
  screen.
- Commits are one per milestone; `results/` artifacts are committed, not
  regenerated on deploy.

## License

MIT — see `LICENSE`. `NOTICE` covers bundled third-party attribution.
