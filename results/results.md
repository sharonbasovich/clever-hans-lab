# Clever Hans Lab — measured results

Generated 2026-09-30T03:03:41.988Z — tfjs 4.22.0, backend cpu, wall 27s.
Config: seeds 101/202/303/404/505, train n=1200, test n=400/split, 8 epochs, batch 64.

| seed | rho | matched | flipped | neutral | gap | bgMass |
|---|---|---|---|---|---|---|
| 101 | 1.0 | 100.0% | 0.0% | 58.5% | 100.0% | 0.805 |
| 101 | 0.9 | 51.5% | 52.3% | 44.5% | -0.7% | 0.724 |
| 101 | 0.5 | 100.0% | 100.0% | 98.8% | 0.0% | 0.277 |
| 202 | 1.0 | 100.0% | 0.0% | 41.8% | 100.0% | 0.727 |
| 202 | 0.9 | 91.3% | 8.0% | 48.3% | 83.3% | 0.727 |
| 202 | 0.5 | 99.8% | 98.8% | 99.3% | 1.0% | 0.259 |
| 303 | 1.0 | 100.0% | 1.3% | 48.0% | 98.8% | 0.735 |
| 303 | 0.9 | 91.3% | 8.8% | 52.0% | 82.5% | 0.735 |
| 303 | 0.5 | 100.0% | 100.0% | 99.8% | 0.0% | 0.255 |
| 404 | 1.0 | 100.0% | 0.0% | 53.8% | 100.0% | 0.729 |
| 404 | 0.9 | 48.5% | 52.3% | 53.8% | -3.7% | 0.708 |
| 404 | 0.5 | 100.0% | 100.0% | 100.0% | 0.0% | 0.097 |
| 505 | 1.0 | 100.0% | 0.0% | 56.0% | 100.0% | 0.715 |
| 505 | 0.9 | 90.0% | 33.8% | 52.3% | 56.3% | 0.666 |
| 505 | 0.5 | 100.0% | 100.0% | 95.8% | 0.0% | 0.269 |

Null control (labels shuffled, seed 101, rho=1.0): matched 48.5%, flipped 47.8%, neutral 55.5%.

## Gates
- **G1 PASS** — Hans cheats: at rho=1.0 the shortcut gap is large in every seed. per-seed gaps=100.0%, 100.0%, 98.8%, 100.0%, 100.0%; mean=99.8%
- **G2 PASS** — No cheat planted, no gap: at rho=0.5 matched=~flipped and shape is learned. mean|gap|=0.2%; mean matched=100.0%
- **G3 PASS** — Attention sits on the cheat: bgMass at rho=1.0 exceeds bgMass at rho=0.5. mean bgMass rho=1.0=0.742, rho=0.5=0.231; sign-test p=0.063
- **G4 PASS** — Null control at chance: shuffled labels give ~50% everywhere. matched=48.5% flipped=47.8% neutral=55.5%
- **G5 PASS** — Reproducible: identical (seed, rho) reproduces identical metrics. repeat matched=100.0% vs first=100.0%

Derived L3 win threshold (flipped acc): 95.0%