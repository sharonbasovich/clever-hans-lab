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
npm run harness       # 5 seeds × rho {1.0, 0.9, 0.5} + null + determinism → results/
npx playwright test   # e2e golden path + keyboard/375px checks
```

`npm run harness:smoke` is the fast CI version (1 seed, small n).

## What the harness measured

Full table: [`results/results.md`](results/results.md)
(generated 2026-09-30, tfjs 4.22.0, tensorflow backend, 38s wall).

Headline numbers, 5 seeds × rho ∈ {1.0, 0.9, 0.5}, n=1200 train / 400 test
per split:

- **rho = 1.0** — the cheat always works: shortcut gap
  (matched − flipped accuracy) is **98.8–100%** in every seed. Ablation
  `bgReliance` = 1.000: erase the shape and the model still scores ~100%;
  blank the background and it collapses to ~50%.
- **rho = 0.9** — seed-dependent, reported honestly: 3 of 5 seeds learn the
  shortcut (gaps 56–83%); 2 seeds collapse to a constant predictor
  (~50% everywhere). No seed was dropped.
- **rho = 0.5** — no cheat planted, no gap (~0%), shape learned (~100%
  everywhere).
- **Null control** — labels shuffled across all 5 seeds: mean matched
  accuracy 52.4% (per-seed 47.0–65.8%; single-seed drift above 50% is
  expected — an untrained model's weight drift keys on the dominant feature
  with an arbitrary sign, which is why the control is powered at 5 seeds).
- **Determinism** — identical (seed, rho) reproduces identical metrics.

### Gates (go/no-go, measured — not tuned)

| Gate | Claim | Result |
|---|---|---|
| G1 | Shortcut gap ≥ 15pp in every seed at rho=1.0 (CI-low ≥ 25pp) | PASS (mean 99.8%) |
| G2 | rho=0.5: mean \|gap\| ≤ 10pp and shape learned ≥ 80% | PASS (0.2%, 100%) |
| G3 | ablation bgReliance rho=1.0 ≥ 0.8, > rho=0.5 (sign test p<0.25) | PASS (1.000 vs 0.000, p=0.063) |
| G4 | null control mean matched acc within [40%, 60%] | PASS (52.4%) |
| G5 | identical (seed, rho) → identical metrics | PASS |

## How it works

- `src/world.ts` — seeded generator: 32×32 RGB images, circle/triangle,
  background colour correlated with the label at rate rho. Foreground masks
  are stored so ablation can surgically erase either region. Same seed →
  identical bytes (tested).
- `src/model.ts` + `src/train.ts` — a small CNN (conv 8×3 → pool → conv 16×3
  → pool → dense 32 → softmax 2, ~12k params) trained with tf.js SGD. Seeded
  shuffling; the loss curve on screen is the measured curve.
- `src/evaluate.ts` — three held-out splits: **matched** (training rule
  holds), **flipped** (cue↔label mapping reversed), **neutral** (grey
  background). The gap is the confession.
- `src/explain.ts` — two complementary "where did it look" probes:
  patch-occlusion log-prob heatmaps, and the headline **counterfactual
  ablation** (blank the whole background vs. erase the whole shape, re-measure
  accuracy — `bgReliance = bgDrop/(bgDrop+fgDrop)`). Patch occlusion alone
  under-measures a distributed cue, which is why the ablation is the gate
  metric.
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
   the win threshold (flipped accuracy ≥ 95%) is derived from the measured
   rho=0.5 distribution, never hard-coded.

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
- rho=0.9 is bistable across seeds (shortcut vs. collapse) — reported, not
  smoothed over.
- Browser training uses the WebGL backend (CPU fallback); wall time varies
  by device. The demo mode (`?demo=1`) reduces n to keep the video under
  two minutes; the harness numbers are the full-size runs.
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
- Commits are one per milestone; `results/` artifacts are committed, not
  regenerated on deploy.

## License

MIT — see `LICENSE`. `NOTICE` covers bundled third-party attribution.
