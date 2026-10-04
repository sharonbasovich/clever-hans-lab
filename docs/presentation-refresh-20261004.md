# Presentation consistency refresh — October 4, 2026

The revised demo is a real recording of the published application at commit `c2e01f16f7575e1f72bc238d07a61d9ba632bc49`. It replaces the universal “can't cheat” wording with “Redesign the data, then test which cues the model relies on.” No model, data-generation, diagnostic, or scoring logic changed.

**Corrected video:** https://youtu.be/GRavNWVOEC4

## Measured results in this recording

- Initial matched accuracy: 100.0%; reversed cue: 0.0%.
- Paired identical-shape cue swap: 100.0% to 0.0%.
- After redesign: 94.4% reversed-cue accuracy, below the 94.5% stretch target. The app explains that missing this score bar does not itself show shortcut reliance.
- Level 1 uses 480 images and 5 epochs; the persistent on-screen disclosure identifies the normal 1,200-image, 8-epoch configuration.

## Capture and edit provenance

GitHub Actions [capture run 37168495206](https://github.com/sharonbasovich/clever-hans-lab/actions/runs/37168495206) completed the genuine demo flow. Its packaging wrapper failed because the runner lacked ffmpeg/ffprobe. The preserved `demo.webm` is the authoritative capture; the stale tracked MP4 in that artifact was explicitly excluded.

Source WebM: 161.2 seconds, 1280×720, video-only.
SHA-256: `b0b8dce87cd181212f4c85d772674000c9235726ef7a9314e204dcec22fa63fb`.

The edited 93.16-second H.264 MP4 concatenates source 0–24s, 42–84s, 134–161.2s. Cuts remove only training waits. A four-second “Training wait shortened - real run, real results” label follows each cut. There is no synthetic progress, altered score, voice, or music.

Edited MP4 SHA-256: `0906865905f9008db117a38afd35284054250ae822cf4db4665ef9be728b4748`.

Independent source-vs-edit review passed: corrected wording legible, actual intervention and below-target redesign result retained, reduced-data notice and ending intact. Full video decode passed with 2,329 frames. No subjective audio review applies because no audio stream exists.

The original video remains preserved at https://www.youtube.com/watch?v=V3v4zA7CdAY and in repository history. No historical result is overwritten.
