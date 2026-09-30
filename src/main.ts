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

function route() {
  const screen = location.hash.replace(/^#\//, '') || 'intro';
  (routes[screen] ?? routes.intro)();
}
window.addEventListener('hashchange', route);

document.getElementById('a11y-toggle')!.addEventListener('click', (e) => {
  const btn = e.currentTarget as HTMLButtonElement;
  session.accessible = !session.accessible;
  btn.setAttribute('aria-pressed', String(session.accessible));
  route(); // re-render current screen with the new palette
});

route();

if (demo) {
  document.getElementById('caption-bar')!.hidden = false;
  // Let the first paint settle, then drive the scripted demo.
  setTimeout(() => runDemo(session), 800);
}
