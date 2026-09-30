import type { Session } from './game/session.ts';
import { pct } from './ui/dom.ts';

// Scripted demo for ?demo=1 — drives the real UI so the recorded video shows
// the genuine training run (smaller n to fit 2 minutes). Captions render in
// the fixed caption bar; Playwright records the whole thing with no input.
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
const click = (sel: string) => (document.querySelector(sel) as HTMLElement | null)?.click();

function caption(text: string, stage: string) {
  const bar = document.getElementById('caption-bar')!;
  bar.textContent = text;
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
    click('.btn'); // start training
    await sleep(2000 * D);

    // Wait for training to complete (nav to exam happens automatically).
    await waitFor(() => location.hash === '#/exam', 60000);
    const ev = session.evaluation!;
    caption(
      `Matched test: ${pct(ev.matched.acc)}. Now flip the colours: ${pct(ev.flipped.acc)}.`,
      'exam',
    );
    await sleep(11000 * D);

    caption('Where did it look? Occluding pixels one patch at a time tells us.', 'heatmap');
    click('.btn'); // "Where was it looking?"
    await sleep(1000 * D);
    const bgm = pct(session.bgMass ?? 0);
    caption(`${bgm} of its attention sat on the background. We know because we drew the shapes.`, 'heatmap2');
    await sleep(11000 * D);

    caption('Your job: design the data so it can\'t cheat.', 'l3');
    click('.btn'); // verdict
    await sleep(2500 * D);
    click('.btn'); // -> L3
    await sleep(4000 * D);

    // In L3, set rho low + neutral samples, then retrain.
    const sliders = document.querySelectorAll<HTMLInputElement>('input[type=range]');
    if (sliders[0]) {
      sliders[0].value = '0.6';
      sliders[0].dispatchEvent(new Event('input'));
    }
    if (sliders[1]) {
      sliders[1].value = '0.5';
      sliders[1].dispatchEvent(new Event('input'));
    }
    caption('Lower the cheat strength, add neutral images, stay under budget — retrain.', 'l3tune');
    await sleep(3000 * D);
    click('.btn'); // retrain
    await waitFor(() => location.hash === '#/exam', 60000);
    const ev3 = session.evaluation!;
    caption(`Flipped test after your fix: ${pct(ev3.flipped.acc)}.`, 'l3exam');
    await sleep(11000 * D);

    location.hash = '#/lab';
    caption('Reproducible: 5 seeds, confidence intervals, code on GitHub.', 'lab');
    await sleep(11000 * D);

    location.hash = '#/teacher';
    caption('Free, no login, 20-minute lesson included. Clever Hans Lab: don\'t ask if AI is right — ask why.', 'teacher');
    await sleep(9000 * D);

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
