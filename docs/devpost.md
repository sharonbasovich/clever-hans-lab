# Devpost draft — Clever Hans Lab

<!-- Draft text only. Sharon submits it; nothing here is auto-submitted. -->

## Title

Clever Hans Lab — let a neural network cheat, then catch it

## Opening hook / Inspiration

In 1907 a horse named Clever Hans wowed Europe by "doing maths" — tapping
out answers with his hoof. He wasn't counting. He was reading tiny, involuntary
cues in his trainer's face. Today's neural networks pull the same trick: a
"wolf" classifier that learned snow, a pneumonia model that learned a hospital
watermark. I wanted people to *feel* that failure mode, not just read about
it — so I built a game where **you** plant the cheat, watch a real neural
network exploit it, and then have to fix the data so it can't.

## Problem

"Is the model accurate?" is the wrong question — Hans scored 100% too. The
right question is *what did it learn?* Shortcut learning is well documented
in ML research (Geirhos et al. 2020; Lapuschkin et al. 2019) but invisible to
most students: you can't see the cue inside a model by staring at accuracy.
Teaching it usually means lecturing about someone else's failure instead of
reproducing one live.

## Solution

Clever Hans Lab is a zero-install browser game. You generate a synthetic
dataset of shapes where background colour perfectly predicts the label
(red↔circle, blue↔triangle). A real convolutional network trains in your
browser — live loss curve, real gradients, no mocks. It aces the matched
test. Then you flip the colours and it collapses to ~0%: it never learned
shape at all. A counterfactual ablation proves it: erase the shape and it
still scores ~100%; blank the background and it falls to chance. In the last
level you redesign the training data (cue strength, neutral fraction, sample
budget) and retrain until the network can't cheat.

## Key features

- Real TensorFlow.js training in-browser (conv → conv → dense, ~12k params)
  with the measured loss curve — nothing is faked or canned.
- Three held-out exams: matched / flipped / neutral — the gap is the
  confession.
- Counterfactual ablation (erase shape vs. hide background) plus a
  patch-occlusion heatmap as supporting evidence.
- A redesign level where the win threshold is **derived from measured
  results**, not hard-coded.
- Reproducible science page: a deterministic headless harness evaluates 5
  seeds × 3 cue strengths plus a shuffled-label null control, with bootstrap
  CIs and pass/fail gates — all committed to the repo.
- Accessible: full keyboard play, aria-live results, canvas text
  alternatives, reduced-motion, and a colour-independent cue mode
  (stripes + brightness, not hue).
- Free, no login, no backend — works on a phone.

## Technologies

Vite + TypeScript + TensorFlow.js (WebGL, CPU fallback). Fully static —
no accounts, no backend, no API keys. Deterministic evaluation via
`@tensorflow/tfjs-node`; Playwright for the e2e test and the captioned
demo video; GitHub Actions CI (lint, typecheck, unit tests, harness smoke,
build, e2e); deployed on GitHub Pages.

## Target users

Students meeting ML for the first time, teachers wanting a 20-minute
shortcut-learning lesson (a ready-made one is built in, with citations),
and anyone who evaluates models by test accuracy alone.

## Social impact

Shortcut learning is the mechanism behind many real AI harms — models that
are accurate on test data and wrong in the world. You can't fix what you
can't see. Clever Hans Lab makes the failure *visceral and reproducible* in
five minutes, for free, in any classroom browser.

## Limitations

Synthetic images by design — the cheat is planted so it can be measured
exactly. The demonstrated gap describes this toy system, not any production
model. At rho=0.9 the outcome is bistable across seeds (reported, not
smoothed). See README for the full list.

## Links

- Live demo: https://sharonbasovich.github.io/clever-hans-lab/
- Source + measured results: https://github.com/sharonbasovich/clever-hans-lab
- Demo video (108s, captioned): `demo/demo.mp4` in the repo

## Screenshots (5)

From `demo/shots/` —

1. `stage-build.png` — planting the cheat: world generation preview
2. `stage-exam.png` — three exams: matched 100%, flipped 0%, neutral ~54%
3. `stage-heatmap2.png` — the ablation confession: shape erased, still 100%
4. `stage-l3exam.png` — after the player's fix: flipped accuracy recovered
5. `stage-lab.png` — reproducibility page: all seeds, gates, null control

## Team

Sharon Basovich (University of Waterloo) — solo.

## AI assistance disclosure

Built with substantial assistance from Devin (Cognition AI): implementation,
test design, evaluation harness, and documentation drafts. Conception,
direction, review, and submission decisions are mine. The science is cited
in-app (Pfungst 1907; Lapuschkin et al. 2019; Geirhos et al. 2020).
