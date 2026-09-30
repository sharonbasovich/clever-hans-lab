import { occlusionMap } from '../explain.ts';
import { L3, L3_FLIPPED_THRESHOLD, RESULTS } from '../game/levels.ts';
import type { Session } from '../game/session.ts';
import { IMG_PIXELS } from '../world.ts';
import { announce, describeImage, drawHeatmap, drawImage, drawLossChart, el, pct } from './dom.ts';
import { hansSvg } from './hans.ts';

export type Nav = (screen: string) => void;

const reducedMotion = () => window.matchMedia('(prefers-reduced-motion: reduce)').matches;

function focusMain() {
  const app = document.getElementById('app');
  app?.focus({ preventScroll: true });
}

// ---------------------------------------------------------------- intro ---

export function renderIntro(session: Session, nav: Nav): void {
  const app = document.getElementById('app')!;
  app.innerHTML = '';
  const wrap = el('section', { class: 'hero' });
  wrap.innerHTML = `
    ${hansSvg()}
    <p class="kicker">A game about why AI cheats</p>
    <h1>Clever Hans Lab</h1>
    <p class="hook">1907: a horse "did maths" by reading his trainer's face.<br>
    Today: your neural network is about to pull the same trick — and you'll catch it.</p>
    <p class="sub">You will plant a cheat in a synthetic dataset, watch a real CNN exploit it
    in seconds, prove it with a heatmap, then redesign the data until it can't cheat.
    Everything trains in your browser. Nothing is faked.</p>
  `;
  wrap.append(
    el('div', { class: 'btn-row' },
      el('button', { class: 'btn', click: () => { session.level = 1; nav('build'); } }, 'Start: Level 1 — Meet Hans'),
      el('button', { class: 'btn secondary', click: () => nav('lab') }, 'See measured lab results'),
    ),
    el('p', { class: 'cite' },
      'Named for Clever Hans, the horse debunked by Oskar Pfungst in 1907. ',
      el('a', { href: '#/teacher', click: (e) => { e.preventDefault(); nav('teacher'); } }, 'Lesson plan + citations →'),
    ),
  );
  app.append(wrap);
  focusMain();
}

// ------------------------------------------------------------ build world ---

export function renderBuild(session: Session, nav: Nav): void {
  const app = document.getElementById('app')!;
  app.innerHTML = '';
  session.buildLevel1World();
  const world = session.trainWorld!;

  const panel = el('section', { class: 'panel' });
  panel.innerHTML = `
    <p class="kicker">Level 1 · Meet Hans</p>
    <h2>Plant the cheat</h2>
    <p class="lede">Here are ${world.n} code-generated training images. The task is trivial:
    <strong>circle or triangle?</strong> But we rigged the backgrounds —
    <strong style="color:var(--red)">red always means circle</strong>,
    <strong style="color:var(--blue)">blue always means triangle</strong>.
    A lazy model will notice that first.</p>
  `;

  const grid = el('div', { class: 'sample-grid', role: 'list', 'aria-label': 'Sample training images' });
  for (let i = 0; i < 16; i++) {
    const c = el('canvas', { role: 'img', 'aria-label': describeImage(world, i), title: describeImage(world, i) });
    drawImage(c, world, i, 3);
    grid.append(c);
  }
  panel.append(grid);
  panel.append(
    el('p', { class: 'note' },
      `These pixels are generated in this page — no dataset was downloaded and nothing is uploaded. ` +
      `Cheat strength rho = 1.0: the background colour predicts the label ${'100'}% of the time in training.`,
    ),
    el('div', { class: 'btn-row' },
      el('button', { class: 'btn', click: () => nav('train') }, 'Train the network →'),
      el('button', { class: 'btn secondary', click: () => { session.seed = Math.floor(Math.random() * 100000); renderBuild(session, nav); } }, 'Re-roll world'),
    ),
  );
  app.append(panel);
  announce('World built: training images with background colour perfectly correlated with the label.');
  focusMain();
}

// ------------------------------------------------------------------ train ---

export function renderTrain(session: Session, nav: Nav): void {
  const app = document.getElementById('app')!;
  app.innerHTML = '';
  const panel = el('section', { class: 'panel' });
  panel.innerHTML = `
    <p class="kicker">Level 1 · Meet Hans</p>
    <h2>Train a real CNN — right here</h2>
    <p class="lede">A small convolutional network (conv → conv → dense) trains on your world
    with TensorFlow.js. The loss curve below is measured, not animated.</p>
  `;
  const chartWrap = el('div', { class: 'chart-wrap' });
  const chart = el('canvas', { role: 'img', 'aria-label': 'Training loss and accuracy per epoch' });
  chartWrap.append(chart);
  const stats = el('div', { class: 'stat-row' },
    statBox('—', 'epoch'),
    statBox('—', 'loss'),
    statBox('—', 'train accuracy'),
  );
  const bar = el('div', { class: 'progress' });
  const fill = el('div');
  bar.append(fill);
  const startBtn = el('button', { class: 'btn' }, 'Start training');
  panel.append(chartWrap, stats, bar, el('div', { class: 'btn-row' }, startBtn));
  app.append(panel);
  focusMain();

  startBtn.addEventListener('click', async () => {
    startBtn.disabled = true;
    startBtn.textContent = 'Training…';
    const losses: number[] = [];
    const accs: number[] = [];
    try {
      await session.train({
        onBatch: (b, total) => {
          fill.style.width = `${(b / total) * 100}%`;
        },
        onEpoch: (e, loss, acc) => {
          losses.push(loss);
          accs.push(acc);
          drawLossChart(chart, losses, accs, reducedMotion());
          const boxes = stats.querySelectorAll('.num');
          boxes[0].textContent = String(e + 1);
          boxes[1].textContent = loss.toFixed(3);
          boxes[2].textContent = pct(acc);
        },
      });
      session.evaluate();
      session.explain();
      announce(`Training done. Matched accuracy ${pct(session.evaluation!.matched.acc)}, flipped ${pct(session.evaluation!.flipped.acc)}.`);
      nav('exam');
    } catch (err) {
      console.error(err);
      startBtn.disabled = false;
      startBtn.textContent = 'Retry training';
      panel.append(el('p', { class: 'note' }, 'Training failed in this browser. Try a different browser or reload — a failure state is better than a fake number.'));
    }
  });
}

function statBox(num: string, lbl: string, cls = ''): HTMLElement {
  const d = el('div', { class: `stat ${cls}` });
  d.append(el('div', { class: 'num' }, num), el('div', { class: 'lbl' }, lbl));
  return d;
}

// ------------------------------------------------------------------- exam ---

export function renderExam(session: Session, nav: Nav): void {
  const app = document.getElementById('app')!;
  app.innerHTML = '';
  const ev = session.evaluation!;
  const isL3 = session.level === 3;

  const panel = el('section', { class: 'panel' });
  panel.innerHTML = `
    <p class="kicker">${isL3 ? 'Level 3 · Fix the data' : 'Levels 1–2 · The exam'}</p>
    <h2>Three exams, one confession</h2>
    <p class="lede">Same task, three held-out test sets — each drawn fresh from code, never seen in training.</p>
  `;
  const grid = el('div', { class: 'exam-grid' });
  const cards = [
    {
      name: 'Matched',
      acc: ev.matched.acc,
      desc: 'Background colours follow the training rule (red = circle, blue = triangle).',
      cls: 'good',
    },
    {
      name: 'Flipped',
      acc: ev.flipped.acc,
      desc: 'The cheat is reversed: red now means triangle. If it learned shape, nothing changes.',
      cls: ev.flipped.acc < 0.6 ? 'bad' : 'good',
    },
    {
      name: 'Neutral',
      acc: ev.neutral.acc,
      desc: 'No cheat at all — plain grey backgrounds. Only the shape can help it now.',
      cls: ev.neutral.acc < 0.6 ? 'bad' : 'good',
    },
  ];
  for (const c of cards) {
    const card = el('div', { class: `exam-card ${c.cls}` });
    card.append(
      el('div', { class: 'big' }, pct(c.acc)),
      el('div', { class: 'name' }, c.name),
      el('div', { class: 'desc' }, c.desc),
    );
    grid.append(card);
  }
  panel.append(grid);

  if (isL3) {
    const thr = L3_FLIPPED_THRESHOLD;
    const win = ev.flipped.acc >= thr;
    panel.append(
      el('p', { class: 'note' },
        win
          ? `Flipped accuracy ${pct(ev.flipped.acc)} ≥ ${pct(thr)} (the measured debiased bar). Your data fixed the shortcut — Hans now has to read the shape.`
          : `Flipped accuracy ${pct(ev.flipped.acc)} < ${pct(thr)}. The cheat still works. Try lower rho or more neutral samples.`,
      ),
    );
  } else if (ev.flipped.acc < 0.6) {
    panel.append(
      el('p', { class: 'note' },
        `It aced the matched exam (${pct(ev.matched.acc)}) and collapsed when the colours flipped (${pct(ev.flipped.acc)}). ` +
        `It never learned shapes — it learned the cheat. Now prove it: where was it looking?`,
      ),
    );
  }

  panel.append(
    el('div', { class: 'btn-row' },
      isL3
        ? el('button', { class: 'btn', click: () => nav('l3') }, 'Adjust data →')
        : el('button', { class: 'btn', click: () => nav('heatmap') }, 'Where was it looking? →'),
      el('button', { class: 'btn secondary', click: () => nav('heatmap') }, isL3 ? 'See the heatmap' : 'Skip to heatmap'),
    ),
  );
  app.append(panel);
  announce(`Exam done. Matched ${pct(ev.matched.acc)}, flipped ${pct(ev.flipped.acc)}, neutral ${pct(ev.neutral.acc)}.`);
  focusMain();
}

// ---------------------------------------------------------------- heatmap ---

export function renderHeatmap(session: Session, nav: Nav): void {
  const app = document.getElementById('app')!;
  app.innerHTML = '';
  const world = session.testSets!.matched;
  const model = session.model!;
  // Use the highest-confidence image for a clear story.
  let bestIdx = 0;
  const heat = occlusionMap(model, world.images.subarray(0, IMG_PIXELS * 3) as Float32Array);

  const panel = el('section', { class: 'panel' });
  panel.innerHTML = `
    <p class="kicker">Level 2 · Catch the cheat</p>
    <h2>Where was it looking?</h2>
    <p class="lede">We slide a grey patch across the image and measure how much the model's
    confidence drops — an occlusion heatmap. Orange = pixels it relied on. The green
    outline is the true shape (we drew it, so we know).</p>
  `;

  const wrap = el('div', { class: 'heatmap-wrap' });
  const canvas = el('canvas', {
    role: 'img',
    'aria-label': `Occlusion heatmap: ${pct(session.bgMass ?? 0)} of the model's attention is on the background`,
  });
  drawHeatmap(canvas, world, bestIdx, heat, 8);
  wrap.append(canvas);
  const side = el('div', { style: 'flex:1;min-width:240px' });
  const bgPct = pct(session.bgMass ?? 0);
  side.innerHTML = `
    <div class="stat warn"><div class="num">${bgPct}</div><div class="lbl">of the model's attention is on the background</div></div>
    <p class="legend"><span class="sw" style="background:#f59e0b"></span>high-impact pixels (occlusion heat)<br>
    <span class="sw" style="background:#d4ff4f"></span>true shape outline (ground truth)</p>
  `;
  wrap.append(side);
  panel.append(wrap);

  const quiz = el('div', { class: 'quiz-options' });
  const q = el('h3', {}, session.level === 2 || session.level === 1 ? 'Level 2 — your call: where did it look?' : 'Where did it look?');
  panel.append(q, quiz);
  const optShape = el('button', { class: 'btn secondary' }, 'Mostly at the shape — it honestly classified');
  const optBg = el('button', { class: 'btn secondary' }, 'Mostly at the background — it took the shortcut');
  const verdict = el('p', { class: 'note', role: 'status' });
  const answer = (pickedBg: boolean) => {
    const correct = (session.bgMass ?? 0) > 0.5;
    const right = pickedBg === correct;
    verdict.textContent = right
      ? `Correct — ${bgPct} of its attention was background. You caught Hans.`
      : `Look again — ${bgPct} of its attention was background, not the shape.`;
    optShape.disabled = true;
    optBg.disabled = true;
    announce(verdict.textContent);
  };
  optShape.addEventListener('click', () => answer(false));
  optBg.addEventListener('click', () => answer(true));
  quiz.append(optShape, optBg, verdict);

  panel.append(
    el('div', { class: 'btn-row' },
      el('button', { class: 'btn', click: () => nav('verdict') }, 'Deliver the verdict →'),
    ),
  );
  app.append(panel);
  announce(`Heatmap ready. ${bgPct} of attention is on the background.`);
  focusMain();
}

// ---------------------------------------------------------------- verdict ---

export function renderVerdict(session: Session, nav: Nav): void {
  const app = document.getElementById('app')!;
  app.innerHTML = '';
  const ev = session.evaluation!;
  const panel = el('section', { class: 'panel' });
  panel.innerHTML = `
    <p class="kicker">The verdict</p>
    <h2>Guilty: shortcut learning</h2>
    <p class="lede">Your model scored <strong>${pct(ev.matched.acc)}</strong> on data that kept the cheat
    and <strong>${pct(ev.flipped.acc)}</strong> when the cheat was flipped. The heatmap puts
    <strong>${pct(session.bgMass ?? 0)}</strong> of its attention on the background.
    It didn't fail the exam — it answered a different question than the one you asked.</p>
    <p class="lede">This is <strong>shortcut learning</strong>: when a spurious cue predicts the label,
    gradient descent finds it first. Real versions: a "wolf" classifier keyed on snow backgrounds,
    a pneumonia model keyed on hospital watermarks.</p>
  `;
  panel.append(
    el('p', { class: 'note' },
      'This demo shows the mechanism on synthetic data — it does not prove anything about a specific real-world model. We planted the cheat, so we can measure it exactly.',
    ),
    el('div', { class: 'btn-row' },
      el('button', { class: 'btn', click: () => { session.level = 3; nav('l3'); } }, 'Level 3 — fix the data →'),
      el('button', { class: 'btn secondary', click: () => nav('teacher') }, 'How to teach this'),
    ),
  );
  app.append(panel);
  focusMain();
}

// -------------------------------------------------------------------- L3 ---

export function renderL3(session: Session, nav: Nav): void {
  const app = document.getElementById('app')!;
  app.innerHTML = '';
  const state = { rho: 0.9, neutral: 0.3, world: null as ReturnType<Session['buildLevel3World']> | null };

  const panel = el('section', { class: 'panel' });
  panel.innerHTML = `
    <p class="kicker">Level 3 · Fix the data</p>
    <h2>Design the training set so it can't cheat</h2>
    <p class="lede">You control two levers inside a budget of ${L3.budget} images.
    Goal: flipped-test accuracy ≥ <strong>${pct(L3_FLIPPED_THRESHOLD)}</strong> —
    the level a debiased model reaches in our 5-seed lab run (see Lab results).</p>
  `;

  const rhoField = sliderField('Cheat strength (rho)', '1.0 = background always matches the label', 0.5, 1.0, 0.05, state.rho, (v) => {
    state.rho = v;
    rhoOut.textContent = v.toFixed(2);
    refresh();
  });
  const rhoOut = el('span', {}, state.rho.toFixed(2));
  rhoField.querySelector('label')!.append(rhoOut);

  const neuField = sliderField('Neutral samples', 'fraction with plain grey backgrounds — no cheat at all', 0, L3.neutralMax, 0.05, state.neutral, (v) => {
    state.neutral = v;
    neuOut.textContent = pct(v, 0);
    refresh();
  });
  const neuOut = el('span', {}, pct(state.neutral, 0));
  neuField.querySelector('label')!.append(neuOut);

  const budgetLine = el('p', { class: 'budget-bar' });
  const grid = el('div', { class: 'sample-grid', role: 'list', 'aria-label': 'Preview of your training data' });
  const trainBtn = el('button', { class: 'btn' }, 'Retrain on my data →');
  trainBtn.addEventListener('click', () => nav('train'));

  function refresh() {
    const world = session.buildLevel3World(state.rho, state.neutral);
    budgetLine.textContent = `${world.n} images · ${pct(state.neutral, 0)} neutral · expected cue-label agreement ≈ ${pct(state.rho * (1 - state.neutral) + 0.5 * state.neutral, 0)}`;
    grid.innerHTML = '';
    for (let i = 0; i < 12; i++) {
      const c = el('canvas', { role: 'img', 'aria-label': describeImage(world, i), title: describeImage(world, i) });
      drawImage(c, world, i, 3);
      grid.append(c);
    }
  }

  panel.append(rhoField, neuField, budgetLine, grid, el('div', { class: 'btn-row' }, trainBtn));
  app.append(panel);
  refresh();
  focusMain();
}

function sliderField(
  name: string,
  hint: string,
  min: number,
  max: number,
  step: number,
  value: number,
  onInput: (v: number) => void,
): HTMLElement {
  const field = el('div', { class: 'slider-field' });
  const input = el('input', { type: 'range', min: String(min), max: String(max), step: String(step), value: String(value) }) as HTMLInputElement;
  input.addEventListener('input', () => onInput(parseFloat(input.value)));
  field.append(el('label', {}, `${name} `), input, el('p', { class: 'hint' }, hint));
  return field;
}

// ----------------------------------------------------------- lab results ---

export function renderLab(_session: Session, _nav: Nav): void {
  const app = document.getElementById('app')!;
  app.innerHTML = '';
  const R = RESULTS;
  const panel = el('section', { class: 'panel' });
  const date = new Date(R.generatedAt);
  panel.innerHTML = `
    <p class="kicker">Reproducibility</p>
    <h2>Lab results — measured, not staged</h2>
    <p class="lede">These numbers come from the headless harness
    (<code>npm run harness</code>), not from your browser run.
    Seeds ${R.config.seeds.join(', ')}, train n=${R.config.trainN}, test n=${R.config.testN} per split,
    ${R.config.epochs} epochs — generated ${date.toISOString().slice(0, 10)} on tfjs ${R.tfVersion} (cpu).</p>
  `;
  const table = el('table', { class: 'results' });
  table.innerHTML = `
    <thead><tr><th>seed</th><th>rho</th><th>matched</th><th>flipped</th><th>neutral</th><th>gap</th><th>bgMass</th></tr></thead>
    <tbody>
      ${R.runs
        .map(
          (r) =>
            `<tr><td>${r.seed}</td><td>${r.rho.toFixed(1)}</td><td>${pct(r.matchedAcc)}</td><td>${pct(r.flippedAcc)}</td><td>${pct(r.neutralAcc)}</td><td>${pct(r.gap)}</td><td>${r.bgMass.toFixed(3)}</td></tr>`,
        )
        .join('')}
    </tbody>`;
  panel.append(table);

  const gates = el('div');
  gates.innerHTML =
    '<h3>Gates</h3><ul>' +
    R.gates
      .map(
        (g) =>
          `<li><span class="${g.pass ? 'gate-pass' : 'gate-fail'}">${g.id} ${g.pass ? 'PASS' : 'FAIL'}</span> — ${g.name}. <em>${g.detail}</em></li>`,
      )
      .join('') +
    '</ul>' +
    `<p class="lede">Null control (shuffled labels, seed ${R.nullControl.seed}, rho=1.0): matched ${pct(
      R.nullControl.matchedAcc,
    )} — nothing to learn, nothing learned. Derived L3 win threshold: ${pct(R.derived.l3FlippedThreshold)} flipped accuracy.</p>`;
  panel.append(
    gates,
    el('p', { class: 'note' },
      'Synthetic images only. The planted cue makes the shortcut measurable by design; these results describe this toy system, not any production model.',
    ),
  );
  app.append(panel);
  focusMain();
}

// --------------------------------------------------------------- teacher ---

export function renderTeacher(session: Session, nav: Nav): void {
  const app = document.getElementById('app')!;
  app.innerHTML = '';
  const panel = el('section', { class: 'panel' });
  panel.innerHTML = `
    <p class="kicker">For teachers</p>
    <h2>A 20-minute lesson in shortcut learning</h2>
    <p class="lede">Goal: students leave able to explain why a model can be accurate and wrong at the
    same time — and how to check.</p>
    <h3>Run it (10 min)</h3>
    <ol>
      <li>Play Level 1: train, then compare the Matched vs Flipped exam.</li>
      <li>Level 2: before revealing, have each student call "shape or background" — the heatmap settles the bet.</li>
      <li>Level 3: in pairs, find the cheapest data design that clears the flipped bar.</li>
    </ol>
    <h3>Discuss (10 min)</h3>
    <ol>
      <li>The model scored ${'~99'}% on matched data. Was it "good"? What would you tell its designer?</li>
      <li>Name a real system that could learn a Hans-style cheat (hiring, medical imaging, content moderation). What is its "background colour"?</li>
      <li>We fixed the <em>data</em>, not the model. When would fixing the model or the test be better?</li>
    </ol>
    <h3>Limitations</h3>
    <p class="lede">Synthetic toy images; one planted cue; a tiny CNN. The lesson demonstrates the
    mechanism of shortcut learning — it does not prove that any particular real-world model cheats.
    Numbers here are from this page's own run or the labelled lab results.</p>
    <h3>About Clever Hans</h3>
    <p class="lede">Hans was a horse said to do arithmetic. Oskar Pfungst showed in 1907 that Hans
    read his trainer's involuntary cues — the "Clever Hans effect" is now a named pitfall in
    experimental design and machine learning.</p>
    <ul class="cite">
      <li>Pfungst, O. (1907/1911). <em>Clever Hans (The Horse of Mr. von Osten)</em>.</li>
      <li>Lapuschkin et al. (2019). Unmasking Clever Hans predictors… <em>Nature Communications</em>. <a href="https://www.nature.com/articles/s41467-019-08987-4">doi:10.1038/s41467-019-08987-4</a></li>
      <li>Geirhos et al. (2020). Shortcut Learning in Deep Neural Networks. <a href="https://arxiv.org/abs/2004.07780">arXiv:2004.07780</a></li>
    </ul>
  `;
  panel.append(el('div', { class: 'btn-row' }, el('button', { class: 'btn', click: () => { session.level = 1; nav('build'); } }, 'Run the lesson →')));
  app.append(panel);
  focusMain();
}
