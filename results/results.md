# Clever Hans Lab — measured results

Generated 2026-10-01T09:11:08.573Z — tfjs 4.22.0, backend tensorflow, lr 0.005, wall 112s.
Config: seeds 101/202/303/404/505/606/707/808/11/23/577/3001/4999/12345/27182/77777/99991/271828, train n=1200, test n=400/split, 8 epochs, batch 64.
Flipped sets are a STRICT full reversal (cue disagrees on every image) for every rho.

cueReliance = fraction of matched wins lost under the paired opposite-cue swap (same images, cue inverted, shape untouched). fillReliance = legacy neutral-fill diagnostic — can under-report cheating when training data contains neutral backgrounds.

| seed | rho | matched | flipped | neutral | gap | bgMass | cue-swap acc | cueReliance | fillReliance | acc no-bg | acc no-fg | collapse |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 101 | 1.0 | 100.0% | 0.0% | 44.5% | 100.0% | 0.724 | 0.0% | 1.000 | 1.000 | 39.8% | 100.0% |  |
| 101 | 0.9 | 91.3% | 0.0% | 55.5% | 91.3% | 0.724 | 6.3% | 0.933 | 1.000 | 60.2% | 93.8% |  |
| 101 | 0.5 | 100.0% | 100.0% | 73.3% | 0.0% | 0.384 | 100.0% | 0.000 | 0.222 | 82.8% | 39.8% |  |
| 202 | 1.0 | 100.0% | 0.0% | 51.7% | 100.0% | 0.727 | 0.0% | 1.000 | 1.000 | 41.4% | 100.0% |  |
| 202 | 0.9 | 91.3% | 0.0% | 54.0% | 91.3% | 0.727 | 11.7% | 0.867 | 1.000 | 41.4% | 88.3% |  |
| 202 | 0.5 | 100.0% | 100.0% | 100.0% | 0.0% | 0.269 | 100.0% | 0.000 | 0.000 | 100.0% | 58.6% |  |
| 303 | 1.0 | 100.0% | 0.0% | 48.0% | 100.0% | 0.735 | 0.0% | 1.000 | 1.000 | 54.7% | 100.0% |  |
| 303 | 0.9 | 91.3% | 0.0% | 52.0% | 91.3% | 0.755 | 9.4% | 0.897 | 1.000 | 46.1% | 90.6% |  |
| 303 | 0.5 | 100.0% | 100.0% | 99.5% | 0.0% | 0.149 | 100.0% | 0.000 | 0.000 | 100.0% | 53.9% |  |
| 404 | 1.0 | 100.0% | 0.0% | 53.8% | 100.0% | 0.798 | 0.0% | 1.000 | 1.000 | 43.0% | 100.0% |  |
| 404 | 0.9 | 88.0% | 0.0% | 46.3% | 88.0% | 0.742 | 8.6% | 0.906 | 1.000 | 57.0% | 91.4% |  |
| 404 | 0.5 | 94.8% | 96.0% | 92.0% | -1.2% | 0.236 | 96.1% | 0.000 | 0.014 | 95.3% | 43.0% |  |
| 505 | 1.0 | 100.0% | 0.0% | 52.3% | 100.0% | 0.715 | 0.0% | 1.000 | 1.000 | 50.0% | 100.0% |  |
| 505 | 0.9 | 91.0% | 40.0% | 52.3% | 51.0% | 0.757 | 42.2% | 0.538 | 0.500 | 50.0% | 50.0% |  |
| 505 | 0.5 | 100.0% | 100.0% | 94.0% | 0.0% | 0.270 | 100.0% | 0.000 | 0.045 | 97.7% | 50.0% |  |
| 606 | 1.0 | 100.0% | 0.0% | 51.5% | 100.0% | 0.729 | 0.0% | 1.000 | 1.000 | 58.6% | 100.0% |  |
| 606 | 0.9 | 91.8% | 0.0% | 50.2% | 91.8% | 0.745 | 8.6% | 0.906 | 1.000 | 48.4% | 91.4% |  |
| 606 | 0.5 | 99.5% | 100.0% | 90.8% | -0.5% | 0.261 | 100.0% | 0.000 | 0.031 | 98.4% | 51.6% |  |
| 707 | 1.0 | 100.0% | 0.0% | 53.5% | 100.0% | 0.721 | 0.0% | 1.000 | 1.000 | 50.8% | 100.0% |  |
| 707 | 0.9 | 91.8% | 0.0% | 46.5% | 91.8% | 0.721 | 5.5% | 0.942 | 1.000 | 49.2% | 94.5% |  |
| 707 | 0.5 | 94.0% | 93.3% | 46.5% | 0.7% | 0.398 | 94.5% | 0.016 | 0.469 | 54.7% | 49.2% |  |
| 808 | 1.0 | 100.0% | 0.0% | 53.0% | 100.0% | 0.760 | 0.0% | 1.000 | 1.000 | 45.3% | 100.0% |  |
| 808 | 0.9 | 90.3% | 25.3% | 53.0% | 65.0% | 0.821 | 34.4% | 0.621 | 0.500 | 45.3% | 45.3% |  |
| 808 | 0.5 | 96.5% | 96.8% | 76.0% | -0.3% | 0.475 | 97.7% | 0.008 | 0.068 | 94.5% | 45.3% |  |
| 11 | 1.0 | 100.0% | 0.0% | 52.0% | 100.0% | 0.739 | 0.0% | 1.000 | 1.000 | 45.3% | 100.0% |  |
| 11 | 0.9 | 94.5% | 0.0% | 52.0% | 94.5% | 0.726 | 7.8% | 0.915 | 1.000 | 45.3% | 92.2% |  |
| 11 | 0.5 | 100.0% | 99.3% | 78.0% | 0.7% | 0.352 | 99.2% | 0.008 | 0.091 | 94.5% | 45.3% |  |
| 23 | 1.0 | 100.0% | 0.0% | 50.7% | 100.0% | 0.736 | 0.0% | 1.000 | 1.000 | 50.0% | 100.0% |  |
| 23 | 0.9 | 92.0% | 0.0% | 49.3% | 92.0% | 0.729 | 9.4% | 0.897 | 1.000 | 50.8% | 90.6% |  |
| 23 | 0.5 | 94.0% | 94.3% | 93.5% | -0.3% | 0.183 | 92.2% | 0.008 | 0.052 | 90.6% | 50.0% |  |
| 577 | 1.0 | 100.0% | 0.0% | 53.3% | 100.0% | 0.710 | 0.0% | 1.000 | 1.000 | 55.5% | 100.0% |  |
| 577 | 0.9 | 88.0% | 0.0% | 53.3% | 88.0% | 0.710 | 14.8% | 0.826 | 1.000 | 55.5% | 85.2% |  |
| 577 | 0.5 | 98.3% | 98.3% | 52.3% | 0.0% | 0.281 | 99.2% | 0.000 | 0.340 | 71.1% | 44.5% |  |
| 3001 | 1.0 | 100.0% | 0.0% | 55.3% | 100.0% | 0.726 | 0.0% | 1.000 | 1.000 | 38.3% | 100.0% |  |
| 3001 | 0.9 | 92.0% | 0.0% | 58.5% | 92.0% | 0.723 | 3.1% | 0.968 | 1.000 | 41.4% | 96.9% |  |
| 3001 | 0.5 | 98.3% | 99.3% | 99.0% | -1.0% | 0.323 | 99.2% | 0.000 | 0.000 | 98.4% | 61.7% |  |
| 4999 | 1.0 | 100.0% | 0.0% | 49.8% | 100.0% | 0.718 | 0.0% | 1.000 | 1.000 | 52.3% | 100.0% |  |
| 4999 | 0.9 | 91.0% | 0.0% | 49.8% | 91.0% | 0.728 | 8.6% | 0.906 | 1.000 | 52.3% | 91.4% |  |
| 4999 | 0.5 | 96.5% | 95.0% | 96.5% | 1.5% | 0.386 | 95.3% | 0.000 | 0.000 | 95.3% | 47.7% |  |
| 12345 | 1.0 | 100.0% | 0.0% | 49.3% | 100.0% | 0.741 | 0.0% | 1.000 | 1.000 | 54.7% | 100.0% |  |
| 12345 | 0.9 | 93.3% | 0.0% | 49.3% | 93.3% | 0.733 | 9.4% | 0.897 | 1.000 | 54.7% | 90.6% |  |
| 12345 | 0.5 | 95.5% | 97.0% | 49.3% | -1.5% | 0.483 | 95.3% | 0.000 | 0.500 | 54.7% | 54.7% |  |
| 27182 | 1.0 | 100.0% | 0.0% | 48.3% | 100.0% | 0.740 | 0.0% | 1.000 | 1.000 | 36.7% | 100.0% |  |
| 27182 | 0.9 | 86.8% | 0.0% | 63.5% | 86.8% | 0.741 | 13.3% | 0.847 | 1.000 | 71.1% | 86.7% |  |
| 27182 | 0.5 | 100.0% | 100.0% | 100.0% | 0.0% | 0.303 | 100.0% | 0.000 | 0.000 | 100.0% | 36.7% |  |
| 77777 | 1.0 | 100.0% | 0.0% | 50.5% | 100.0% | 0.721 | 0.0% | 1.000 | 1.000 | 51.6% | 100.0% |  |
| 77777 | 0.9 | 89.3% | 0.0% | 49.0% | 89.3% | 0.723 | 10.2% | 0.887 | 1.000 | 47.7% | 89.8% |  |
| 77777 | 0.5 | 95.5% | 97.3% | 85.3% | -1.8% | 0.269 | 96.1% | 0.000 | 0.088 | 91.4% | 47.7% |  |
| 99991 | 1.0 | 100.0% | 0.0% | 44.8% | 100.0% | 0.743 | 0.0% | 1.000 | 1.000 | 52.3% | 100.0% |  |
| 99991 | 0.9 | 90.8% | 0.0% | 55.3% | 90.8% | 0.751 | 9.4% | 0.897 | 1.000 | 47.7% | 90.6% |  |
| 99991 | 0.5 | 98.8% | 99.5% | 61.5% | -0.7% | 0.358 | 98.4% | 0.000 | 0.244 | 83.6% | 52.3% |  |
| 271828 | 1.0 | 100.0% | 0.0% | 56.0% | 100.0% | 0.735 | 0.0% | 1.000 | 1.000 | 49.2% | 100.0% |  |
| 271828 | 0.9 | 90.8% | 0.0% | 56.0% | 90.8% | 0.746 | 9.4% | 0.897 | 1.000 | 49.2% | 90.6% |  |
| 271828 | 0.5 | 95.3% | 96.3% | 96.3% | -1.0% | 0.228 | 95.3% | 0.000 | 0.000 | 95.3% | 49.2% |  |

Null control (labels shuffled, rho=1.0): seed 101 matched 51.2% flipped 77.0% neutral 55.0%; seed 202 matched 48.5% flipped 42.5% neutral 48.0%; seed 303 matched 63.0% flipped 36.8% neutral 44.8%; seed 404 matched 48.5% flipped 52.3% neutral 53.8%; seed 505 matched 63.2% flipped 46.0% neutral 55.3%; seed 606 matched 64.0% flipped 42.5% neutral 50.2%; seed 707 matched 51.7% flipped 53.5% neutral 46.5%; seed 808 matched 47.8% flipped 47.3% neutral 53.0%; seed 11 matched 65.0% flipped 35.5% neutral 52.0%; seed 23 matched 64.8% flipped 46.5% neutral 49.8%; seed 577 matched 62.5% flipped 47.5% neutral 55.0%; seed 3001 matched 55.5% flipped 31.0% neutral 44.0%; seed 4999 matched 17.5% flipped 70.5% neutral 49.8%; seed 12345 matched 3.5% flipped 97.3% neutral 49.3%; seed 27182 matched 13.3% flipped 76.3% neutral 51.7%; seed 77777 matched 35.5% flipped 53.0% neutral 51.0%; seed 99991 matched 48.5% flipped 50.5% neutral 55.3%; seed 271828 matched 27.0% flipped 70.0% neutral 44.0%. Mean neutral 50.5%; per-seed matched+flipped in [86.5%, 128.3%]. Diagnostics (not chance tests): mean matched 46.2%, mean |gap| 25.7%, max |gap| 93.8%. A null at rho=1 can still fit shuffled labels via the cue with an arbitrary sign, so matched≈0/≈100 is expected cue-consistency (matched+flipped≈100%); only neutral accuracy — shape alone — is a chance probe.

Collapses: 0 run(s) collapsed to a constant predictor, 0 recovered via deterministic retry; null-control collapses: 6.

## Gates
- **G1 PASS** — Hans cheats: at rho=1.0 the shortcut gap is large in every seed. per-seed gaps=100.0%, 100.0%, 100.0%, 100.0%, 100.0%, 100.0%, 100.0%, 100.0%, 100.0%, 100.0%, 100.0%, 100.0%, 100.0%, 100.0%, 100.0%, 100.0%, 100.0%, 100.0%; mean=100.0%
- **G2 PASS** — No cheat planted, no gap: at rho=0.5 matched=~flipped and shape is learned. mean|gap|=0.3%; mean matched=97.6%
- **G3 PASS** — Accuracy lives in the cue: paired cue-swap reliance at rho=1.0 vs rho=0.5, sign test p<0.05. mean cueReliance rho=1.0=1.000, rho=0.5=0.002; sign-test p=0.0000
- **G4 PASS** — Null control at chance on shape: shuffled-label NEUTRAL acc ≈ 50% per seed and on mean; mean cue-consistency ≈100% (matched/|gap| are cue-sign diagnostics, not chance tests — disclosed). per-seed neutral=55.0%, 48.0%, 44.8%, 53.8%, 55.3%, 50.2%, 46.5%, 53.0%, 52.0%, 49.8%, 55.0%, 44.0%, 49.8%, 49.3%, 51.7%, 51.0%, 55.3%, 44.0%; mean neutral=50.5%; per-seed matched+flipped=128.3%, 91.0%, 99.8%, 100.7%, 109.3%, 106.5%, 105.3%, 95.0%, 100.5%, 111.3%, 110.0%, 86.5%, 88.0%, 100.8%, 89.5%, 88.5%, 99.0%, 97.0% (cue-consistency, ≈100% for a pure cue responder); diagnostics: mean matched=46.2%, mean|gap|=25.7%, max|gap|=93.8%
- **G5 PASS** — Reproducible: identical (seed, rho) reproduces identical metrics. repeat matched=100.0% vs first=100.0%

## Performance targets (disclosed, not silently dropped)
- **T1 PASS** — Median per-run wall time ≤ 30 s on the harness backend. Target <=30000ms; measured 1399ms on tensorflow.
- **T2 n/a here** — JS bundle ≤ 5 MB uncompressed. Target <=5MB; measured measured by scripts/verify-results.ts against dist/.

### Gate changes vs the original brief (disclosed)
- G3 metric: brief asked for an occlusion-heatmap sign test; we report the stronger counterfactual cue-swap reliance sign test at the original p<0.05 bar. Round 2: the metric changed from neutral-fill reliance to the PAIRED opposite-cue swap after independent QA showed neutral-fill can be dodged (models learn "neutral→shape, coloured→cheat"; fillReliance read 4.6% on a run scoring 10.3% on reversed data). The swap keeps every pixel of the shape and inverts only the cue, so it is the direct reliance measure; neutral-fill is retained as a labelled secondary diagnostic.
- G4 metric: the old gate (null-control mean matched ∈[40,60]% and mean |gap| ≤15%) was itself invalid — QA measured mean matched 39.3%, mean |gap| 35.9%, max |gap| 93.7% on fresh seeds, because a null at rho=1 can learn an arbitrary cue sign. The gate now requires NEUTRAL (shape-only) accuracy ≈ chance on every seed and on mean, plus mean cue-consistency matched+flipped ∈[85,115]% — per-seed sums are disclosed but not bounded because the sum is 100% only for a pure cue responder. The old gate's failure numbers are retained here, not erased.
- Seed count: the full run grew from 8 to 18 seeds — QA round-2 probe seeds are pinned as regression seeds, disclosed, and never presented as held-out evidence.
- Brief's perf gate (≤30s CPU train, ≤5MB bundle) is reported as targets T1/T2: the harness backend number is measured here; the bundle size is measured in CI by scripts/verify-results.ts. The 30s target fails on a pure-JS CPU backend — the UI states the real expectation instead of faking speed.
- Smoke mode enforces G1 and G5 only; G2–G4 are measured and reported but advisory at one seed (the null per-seed bound and the rho=0.5 means are honest only in aggregate). The full 18-seed run enforces every gate and runs in CI.

Derived L3 win threshold (strict full reversal): 94.5%