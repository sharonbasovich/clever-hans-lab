# Clever Hans Lab — measured results

Generated 2026-09-30T03:12:59.658Z — tfjs 4.22.0, backend tensorflow, wall 38s.
Config: seeds 101/202/303/404/505, train n=1200, test n=400/split, 8 epochs, batch 64.

| seed | rho | matched | flipped | neutral | gap | bgMass | bgReliance | acc no-bg | acc no-fg |
|---|---|---|---|---|---|---|---|---|---|
| 101 | 1.0 | 100.0% | 0.0% | 58.5% | 100.0% | 0.805 | 1.000 | 51.6% | 100.0% |
| 101 | 0.9 | 51.5% | 52.3% | 44.5% | -0.7% | 0.724 | 0.500 | 39.8% | 39.8% |
| 101 | 0.5 | 100.0% | 100.0% | 98.8% | 0.0% | 0.277 | 0.000 | 100.0% | 39.8% |
| 202 | 1.0 | 100.0% | 0.0% | 41.8% | 100.0% | 0.727 | 1.000 | 43.8% | 100.0% |
| 202 | 0.9 | 91.3% | 8.0% | 48.3% | 83.3% | 0.727 | 1.000 | 58.6% | 88.3% |
| 202 | 0.5 | 99.8% | 98.8% | 99.3% | 1.0% | 0.259 | 0.000 | 99.2% | 58.6% |
| 303 | 1.0 | 100.0% | 1.3% | 48.0% | 98.8% | 0.735 | 1.000 | 53.9% | 100.0% |
| 303 | 0.9 | 91.3% | 8.8% | 52.0% | 82.5% | 0.735 | 1.000 | 46.1% | 90.6% |
| 303 | 0.5 | 100.0% | 100.0% | 99.8% | 0.0% | 0.255 | 0.000 | 100.0% | 53.9% |
| 404 | 1.0 | 100.0% | 0.0% | 53.8% | 100.0% | 0.729 | 1.000 | 43.0% | 100.0% |
| 404 | 0.9 | 48.5% | 52.3% | 53.8% | -3.7% | 0.708 | 0.500 | 43.0% | 43.0% |
| 404 | 0.5 | 100.0% | 100.0% | 100.0% | 0.0% | 0.097 | 0.000 | 100.0% | 43.0% |
| 505 | 1.0 | 100.0% | 0.0% | 56.0% | 100.0% | 0.715 | 1.000 | 57.8% | 100.0% |
| 505 | 0.9 | 90.0% | 33.8% | 52.3% | 56.3% | 0.666 | 0.500 | 50.0% | 50.0% |
| 505 | 0.5 | 100.0% | 100.0% | 95.8% | 0.0% | 0.269 | 0.000 | 100.0% | 50.0% |

Null control (labels shuffled, rho=1.0): seed 101 matched 65.8%; seed 202 matched 47.0%; seed 303 matched 51.0%; seed 404 matched 48.5%; seed 505 matched 49.5%. Mean matched 52.4%. Per-seed drift above 50% is expected: an untrained model's weight drift keys on the dominant feature (the large background) with an arbitrary sign.

## Gates
- **G1 PASS** — Hans cheats: at rho=1.0 the shortcut gap is large in every seed. per-seed gaps=100.0%, 100.0%, 98.8%, 100.0%, 100.0%; mean=99.8%
- **G2 PASS** — No cheat planted, no gap: at rho=0.5 matched=~flipped and shape is learned. mean|gap|=0.2%; mean matched=100.0%
- **G3 PASS** — Accuracy lives in the background: ablation bgReliance at rho=1.0 vs rho=0.5. mean bgReliance rho=1.0=1.000, rho=0.5=0.000; sign-test p=0.063
- **G4 PASS** — Null control at chance: mean shuffled-label matched acc ≈ 50% across seeds. per-seed matched=65.8%, 47.0%, 51.0%, 48.5%, 49.5%; mean=52.4%
- **G5 PASS** — Reproducible: identical (seed, rho) reproduces identical metrics. repeat matched=100.0% vs first=100.0%

Derived L3 win threshold (flipped acc): 95.0%