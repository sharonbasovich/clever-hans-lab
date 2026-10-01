# Devpost draft — Clever Hans Lab

<!-- Draft text only. Sharon submits it; nothing here is auto-submitted. -->

## Title

Clever Hans Lab — let a neural network cheat, then catch it

## Opening hook / Inspiration

In 1907 a horse named Clever Hans wowed Europe by "doing maths" — tapping
out answers with his hoof. He wasn't counting. He was reading tiny, involuntary
cues in his trainer's face. Today's neural networks pull the same trick: a
"wolf" classifier that learned snow, a pneumonia model that learned a hospital
watermark. Clever Hans Lab lets you *feel* that failure mode instead of
reading about it — a game where **you** plant the cheat, watch a real neural
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
test. Then you reverse every cue and measured accuracy collapses to ~0%
(all 25 seeds at rho=1.0): under that tested shift, its wins depended on
the cue, not the shape. A paired cue swap is the controlled intervention:
show the *same* images with only the cue colour inverted and its correct
answers drop.
In the last level you redesign the training data (cue strength, neutral
fraction, sample budget) and retrain until the network can't cheat.

## Key features

- Real TensorFlow.js training in-browser (conv → conv → dense, ~12k params)
  with the measured loss curve — nothing is faked or canned.
- Three separately generated exams: matched / flipped / neutral — the
  gap is the confession. (Exact images are deduplicated between train
  and test; the finite synthetic space means individual shape+label
  combinations do recur — a disclosed limitation, which is why the
  reversal and swap evidence carry the argument.)
- A paired opposite-cue swap on a fully cue-agreeing evaluation set —
  the identical image stream with only the cue inverted — as the
  headline reliance measure. It is a designed controlled intervention:
  on a set where every cue agrees, a harmful-helpful cancellation is
  impossible by construction (the mixed-base version cancelled itself —
  the bug independent QA caught), and measured reliance falls from 1.000
  at rho=1.0 to ~0.006 at rho=0.5. Neutral-fill ablation and a
  patch-occlusion heatmap remain as labelled secondary evidence.
- A redesign level where the win threshold is **derived from measured
  results**, not hard-coded.
- Reproducible science page: a deterministic headless harness evaluates
  25 seeds × 8 designs (intermediate cheat strengths and neutral-mixed
  training sets) against a strict full reversal, plus a shuffled-label
  null control with unrecovered collapses disclosed separately, with
  bootstrap CIs and pass/fail gates — all committed to the repo and
  re-verified in CI.
- The verdict is derived from measured evidence, not scripted:
  shortcut, partial reliance, honest, undertrained, and inconclusive
  outcomes each get their own report. The exam, the quiz and the verdict
  all read from one shared classifier — rendered browser tests verify
  each outcome stays consistent across all three surfaces. A collapsed
  training run is detected, retried transparently, and disclosed — never
  hidden.
- Accessible: full keyboard play, aria-live results, canvas text
  alternatives, reduced-motion, and a colour-independent cue mode
  (stripes + brightness, not hue).
- Free, no login, no backend — a fully static site. Browser wall time
  measured on the live deployment: ~164 s on the pure-JS CPU fallback
  (~639 s under a 4× CPU slowdown simulation). Phones and GPU/WebGL
  speed were not measured — no claim is made about them.

## Technologies

Vite + TypeScript + TensorFlow.js (WebGL, CPU fallback). Fully static —
no accounts, no backend, no API keys. Deterministic evaluation via
`@tensorflow/tfjs-node`; Playwright for the e2e tests and the captioned
demo video; GitHub Actions CI (lint, typecheck, unit tests, smoke + FULL
25-seed measured harness, build, results verification, e2e incl. lifecycle
and 375px); deployed on GitHub Pages.

## Target users

Students meeting ML for the first time, teachers wanting a 20-minute
shortcut-learning lesson (a ready-made one is built in, with citations),
and anyone who evaluates models by test accuracy alone.

## Social impact

Shortcut learning is the mechanism behind many real AI harms — models that
are accurate on test data and wrong in the world. You can't fix what you
can't see. Clever Hans Lab makes the failure *visceral and reproducible* —
free, zero-install, and honest about its own speed (the measured CPU
fallback takes a few minutes per training run).

## Limitations

Synthetic images by design — the cheat is planted so it can be measured
exactly. The demonstrated gap describes this toy system, not any production
model. At rho=0.9 the outcome is bistable across seeds (reported, not
smoothed). See README for the full list.

## Links

- Live demo: https://sharonbasovich.github.io/clever-hans-lab/
- Source + measured results: https://github.com/sharonbasovich/clever-hans-lab
- Demo video (~2min, captioned, real training — the on-screen disclosure
  states it runs on reduced data, 480 images / 5 epochs): `demo/demo.mp4`
  in the repo

## Screenshots (5)

From `demo/shots/` —

1. `stage-build.png` — planting the cheat: world generation preview
2. `stage-exam.png` — three exams: matched 100%, flipped 0%, neutral ~54%
3. `stage-heatmap2.png` — the ablation confession: shape erased, still 100%
4. `stage-l3exam.png` — after the player's fix: flipped accuracy recovered
5. `stage-lab.png` — reproducibility page: all seeds, gates, null control

## Team

Sharon Basovich (University of Waterloo). Team membership beyond Sharon
is pending final entrant/team confirmation.

## AI assistance disclosure

Built with AI-led assistance through dot and Devin (Cognition AI) under
Sharon's authorization: ideation, implementation, tests, evaluation
harness, and documentation were AI-produced and iterated through rounds
of independent review. Any submission decision is Sharon's. The science
is cited in-app (Pfungst 1907; Lapuschkin et al. 2019; Geirhos et al.
2020; Ribeiro et al. 2016 — the snow/husky example; Zech et al. 2018 —
hospital-confound shortcut).
