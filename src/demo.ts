import type { Session } from './game/session.ts';
import { pct } from './ui/dom.ts';

// Scripted demo for ?demo=1 — drives the real UI so the recorded video shows
// the genuine training run (smaller n to fit 2 minutes). Captions render in
// the fixed caption bar; Playwright records the whole thing with no input.
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
// Primary action button only — quiz options are '.btn secondary'.
const click = (sel: string) =>
  (document.querySelector(`${sel}:not(.secondary)`) as HTMLElement | null)?.click();

function caption(text: string, stage: string) {
  document.getElementById('caption-text')!.textContent = text;
  document.body.dataset.demoStage = stage;
}

export async function runDemo(session: Session): Promise<void> {
  const D = 1; // timing scale
  try {
    caption('1907: Clever Hans the horse "did maths". He was reading his trainer\'s face.', 'title');
    await sleep(9000 * D);

    caption("Today's AI cheats the same way. Let's plant a cheat: every circle on red, every triangle on blue.", 'build');
    document.querySelector<HTMLButtonElement>('.btn')?.click();
    await sleep(7000 * D);

    caption('A real CNN trains in your browser. Watch the loss.', 'train');
    click('.btn'); // "Train the network →"
    await sleep(1000 * D);
    click('.btn'); // "Start training"
    await sleep(2000 * D);

    // Wait for training to complete (nav to exam happens automatically).
    await waitFor(() => location.hash === '#/exam', 60000);
    const ev = session.evaluation!;
    caption(
      `Matched test: ${pct(ev.matched.acc)}. Now flip the colours: ${pct(ev.flipped.acc)}.`,
      'exam',
    );
    await sleep(11000 * D);

    caption('Where did it look? Edit the image and ask again — a counterfactual test.', 'heatmap');
    click('.btn'); // "Where was it looking?"
    await sleep(1000 * D);
    const ab = session.ablation!;
    caption(
      `Same shapes, cue colour reversed → ${pct(ab.accSwap)} correct. Hide the background → ${pct(ab.accNoBg)}. Its answers lived in the cue.`,
      'heatmap2',
    );
    await sleep(11000 * D);

    caption('Your job: design the data so it can\'t cheat.', 'l3');
    click('.btn'); // heatmap -> verdict
    await sleep(2500 * D);
    click('.btn'); // verdict -> L3
    await sleep(4000 * D);

    // In L3, pull the cheat strength to the floor and max out neutral
    // samples — the design the harness shows wins on a strict reversal.
    const sliders = document.querySelectorAll<HTMLInputElement>('input[type=range]');
    if (sliders[0]) {
      sliders[0].value = '0.5';
      sliders[0].dispatchEvent(new Event('input'));
    }
    if (sliders[1]) {
      sliders[1].value = '0.6';
      sliders[1].dispatchEvent(new Event('input'));
    }
    caption('Drop the cheat strength, fill the rest with neutral images — retrain.', 'l3tune');
    await sleep(3000 * D);
    click('.btn'); // "Retrain on my data →"
    await sleep(1000 * D);
    click('.btn'); // "Start training" on the retrain screen
    // L3 trains the real 1600-image budget — allow real CPU time.
    await waitFor(() => location.hash === '#/exam', 240000);
    const ev3 = session.evaluation!;
    const thr = (await import('./game/levels.ts')).L3_FLIPPED_THRESHOLD;
    caption(`Full reversal after your fix: ${pct(ev3.flipped.acc)} (bar: ${pct(thr)}).`, 'l3exam');
    await sleep(8000 * D);

    location.hash = '#/lab';
    caption('Reproducible: 18 seeds, per-seed results and gates, code on GitHub.', 'lab');
    await sleep(9000 * D);

    location.hash = '#/teacher';
    caption('Free, no login, lesson plan included. Clever Hans Lab: don\'t ask if AI is right — ask why.', 'teacher');
    await sleep(7000 * D);

    document.body.dataset.demoStage = 'done';
    caption('Clever Hans Lab — github.com/sharonbasovich/clever-hans-lab', 'done');
  } catch (e) {
    console.error('demo failed', e);
    document.body.dataset.demoStage = 'failed';
    caption('Demo script hit an error — see console.', 'failed');
  }
}

async function waitFor(cond: () => boolean, timeoutMs: number): Promise<void> {
  const t0 = Date.now();
  while (!cond()) {
    if (Date.now() - t0 > timeoutMs) throw new Error('demo waitFor timeout');
    await sleep(200);
  }
}
