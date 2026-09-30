# Demo video script — 108 seconds, captioned, no voice

The video is **recorded, not animated**: `scripts/record-demo.ts` drives the
real app with Playwright (`?demo=1&fast=1` shrinks n so real training fits the
runtime) while the app renders the captions in a fixed bar. Numbers shown are
the actual run.

| Time | Stage | On screen | Caption (verbatim) |
|---|---|---|---|
| 0:00–0:09 | title | Intro + Hans | 1907: Clever Hans the horse "did maths". He was reading his trainer's face. |
| 0:09–0:16 | build | World preview, sample grid | Today's AI cheats the same way. Let's plant a cheat: every circle on red, every triangle on blue. |
| 0:16–~0:45 | train | Live loss/accuracy chart, progress bar | A real CNN trains in your browser. Watch the loss. |
| ~0:45–0:56 | exam | Three exam cards | Matched test: [M]. Now flip the colours: [M]. |
| 0:56–1:08 | heatmap | 3-panel ablation + occlusion map | Erase the shape → still [M] correct. Hide the background → [M]. Its accuracy lived in the background. |
| 1:08–1:15 | verdict → L3 | Verdict, L3 sliders | Your job: design the data so it can't cheat. |
| 1:15–1:33 | l3tune + retrain | Sliders set rho 0.6 + 50% neutral, retrain | Lower the cheat strength, add neutral images, stay under budget — retrain. |
| 1:33–1:44 | l3exam | Exam cards, improved flipped | Flipped test after your fix: [M]. |
| 1:44–1:55 | lab | results table, gates, null control | Reproducible: 5 seeds, confidence intervals, code on GitHub. |
| 1:55–2:03 | teacher | Lesson page | Free, no login, 20-minute lesson included. Clever Hans Lab: don't ask if AI is right — ask why. |

`[M]` = measured at runtime from the actual model, so the captions never make
a claim the video can't back.

Recording: `npx tsx scripts/record-demo.ts` (preview server on :4173).
Output: `demo/demo.mp4` (1280×720 H.264) + `demo/shots/stage-*.png`.
