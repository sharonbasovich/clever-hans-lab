import './style.css';
import { Session } from './game/session.ts';
import {
  renderBuild,
  renderExam,
  renderHeatmap,
  renderIntro,
  renderL3,
  renderLab,
  renderTeacher,
  renderTrain,
  renderVerdict,
} from './ui/screens.ts';
import { runDemo } from './demo.ts';

const params = new URLSearchParams(location.search);
const demo = params.has('demo');
// ?fast=1 shrinks the world for e2e runs — training is still real, n is small.
const fast = demo || params.has('fast');
const session = new Session(fast);
await session.ensureBackend();
// Exposed for e2e tests (inject measured-evidence objects, re-render screens).
(window as unknown as { __chlSession: Session }).__chlSession = session;

const routes: Record<string, () => void> = {
  intro: () => renderIntro(session, nav),
  build: () => renderBuild(session, nav),
  train: () => renderTrain(session, nav),
  exam: () => renderExam(session, nav),
  heatmap: () => renderHeatmap(session, nav),
  verdict: () => renderVerdict(session, nav),
  l3: () => renderL3(session, nav),
  lab: () => renderLab(session, nav),
  teacher: () => renderTeacher(session, nav),
};

function nav(screen: string) {
  location.hash = `#/${screen}`;
}

const NEEDS_EVAL = new Set(['exam', 'heatmap', 'verdict']);

function route() {
  let screen = location.hash.replace(/^#\//, '') || 'intro';

  // Leaving the train screen cancels the run — a stale completion can never
  // hijack navigation later.
  if (screen !== 'train' && session.trainingNow) session.cancelTraining();

  // Result screens need a real evaluation; a reload or direct link has none.
  if (NEEDS_EVAL.has(screen) && !session.evaluation) {
    session.notice = session.trainWorld
      ? 'No measured run exists (page was reloaded or the run was cancelled). Train first — results appear here.'
      : 'No world or training run exists yet. Start here.';
    screen = 'build';
  }

  // Train without a world (e.g. fresh reload straight to #/train): rebuild
  // the level-1 world rather than dead-ending on "build a world first".
  if (screen === 'train' && !session.trainWorld) {
    session.buildLevel1World();
    session.notice = 'Rebuilt a fresh Level 1 world — no saved run existed.';
  }

  (routes[screen] ?? routes.intro)();
}
window.addEventListener('hashchange', route);

document.getElementById('a11y-toggle')!.addEventListener('click', (e) => {
  const btn = e.currentTarget as HTMLButtonElement;
  session.accessible = !session.accessible;
  btn.setAttribute('aria-pressed', String(session.accessible));
  // Toggling mid-run invalidates the in-flight training: its generation is
  // superseded and it may never evaluate or navigate after completing.
  session.cancelTraining();
  route(); // re-render current screen with the new palette
});

route();

if (demo) {
  document.body.classList.add('demo-mode');
  document.getElementById('caption-bar')!.hidden = false;
  // Let the first paint settle, then drive the scripted demo.
  setTimeout(() => runDemo(session), 800);
}
