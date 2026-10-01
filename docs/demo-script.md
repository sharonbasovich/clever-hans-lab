# Demo video script — ~108 seconds, captioned, no voice

The video is **recorded, not animated**: `scripts/record-demo.ts` drives the
real app with Playwright (`?demo=1&fast=1` shrinks Level-1 n so real training
fits the runtime — disclosed **on screen** for the whole video in a persistent
line under the captions: "Demo mode — Level 1 uses reduced data (480 images,
5 epochs; the real level is 1,200 images, 8 epochs)"). The caption bar takes
real layout space below the scrollable app, so it can never cover the content
it narrates. Numbers shown are the actual run.

| Time | Stage | On screen | Caption (verbatim) |
|---|---|---|---|
| 0:00–0:09 | title | Intro + Hans | 1907: Clever Hans the horse "did maths". He was reading his trainer's face. |
| 0:09–0:16 | build | World preview, sample grid | Today's AI cheats the same way. Let's plant a cheat: every circle on red, every triangle on blue. |
| 0:16–~0:30 | train | Live loss/accuracy chart, progress bar | A real CNN trains in your browser. Watch the loss. |
| ~0:30–0:41 | exam | Three exam cards | Matched test: [M]. Now flip the colours: [M]. |
| 0:41–0:53 | heatmap | 4-panel counterfactual + occlusion map | Every cue agreeing, then every cue inverted → [M] correct. Hide the background → [M]. Its answers lived in the cue. |
| 0:53–1:00 | verdict → L3 | Verdict, L3 sliders | Your job: design the data so it can't cheat. |
| 1:00–1:28 | l3tune + retrain | Sliders set rho 0.5 + 60% neutral, retrain (real 1600-image budget) | Drop the cheat strength, fill the rest with neutral images — retrain. |
| 1:28–1:36 | l3exam | Exam cards, recovered flipped accuracy | Full reversal after your fix: [M] (bar: [M]). |
| 1:36–1:45 | lab | results table, gates, null control | Reproducible: 25 seeds, per-seed results and gates, code on GitHub. |
| 1:45–1:48 | teacher | Lesson page | Free, no login, lesson plan included. Clever Hans Lab: don't ask if AI is right — ask why. |

`[M]` = measured at runtime from the actual model, so the captions never make
a claim the video can't back.

Recording: `npx tsx scripts/record-demo.ts` (preview server on :4173).
Output: `demo/demo.mp4` (1280×720 H.264) + `demo/shots/stage-*.png`.
