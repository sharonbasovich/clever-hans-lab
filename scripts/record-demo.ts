// Records the scripted demo (?demo=1&fast=1) with Playwright: 1280x720 video
// plus five stage screenshots. The demo drives the real UI — the training in
// the video is the genuine model run.
//
//   npm run preview -- --port 4173   (in another shell)
//   npx tsx scripts/record-demo.ts
//
// Outputs: demo/demo.webm (+ demo.mp4 when ffmpeg is available) and
// demo/shots/stage-*.png
import { chromium } from 'playwright';
import { mkdirSync, readdirSync, renameSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { execSync } from 'node:child_process';

const HERE = dirname(fileURLToPath(import.meta.url));
const OUT = join(HERE, '..', 'demo');
const SHOTS = join(OUT, 'shots');
const BASE = process.env.DEMO_URL ?? 'http://localhost:4173';

// Stages worth a still: keyed on document.body.dataset.demoStage.
const WANT_SHOTS = ['build', 'exam', 'heatmap2', 'l3exam', 'lab'];

async function main() {
  mkdirSync(SHOTS, { recursive: true });
  mkdirSync(join(OUT, 'video'), { recursive: true });

  const browser = await chromium.launch();
  const context = await browser.newContext({
    viewport: { width: 1280, height: 720 },
    recordVideo: { dir: join(OUT, 'video'), size: { width: 1280, height: 720 } },
  });
  const page = await context.newPage();
  await page.goto(`${BASE}/?demo=1&fast=1`);

  const taken = new Set<string>();
  const deadline = Date.now() + 240_000;
  while (Date.now() < deadline) {
    const stage = await page.evaluate(() => document.body.dataset.demoStage ?? '');
    if (WANT_SHOTS.includes(stage) && !taken.has(stage)) {
      taken.add(stage);
      await page.screenshot({ path: join(SHOTS, `stage-${stage}.png`) });
      console.log(`shot: ${stage}`);
    }
    if (stage === 'done' || stage === 'failed') break;
    await page.waitForTimeout(400);
  }
  const finalStage = await page.evaluate(() => document.body.dataset.demoStage ?? '');
  if (finalStage !== 'done') {
    console.error(`demo ended in stage '${finalStage}'`);
  }

  await context.close();
  await browser.close();

  const vids = readdirSync(join(OUT, 'video')).filter((f) => f.endsWith('.webm'));
  if (vids.length === 0) {
    console.error('no video produced');
    process.exit(1);
  }
  const webm = join(OUT, 'video', vids[0]);
  renameSync(webm, join(OUT, 'demo.webm'));
  try {
    execSync(`ffmpeg -y -i "${join(OUT, 'demo.webm')}" -c:v libx264 -pix_fmt yuv420p "${join(OUT, 'demo.mp4')}"`, {
      stdio: 'pipe',
    });
    console.log('demo/demo.mp4 written');
  } catch {
    console.log('ffmpeg unavailable — demo/demo.webm written');
  }
  console.log(`stage shots: ${[...taken].join(', ')}`);
}

await main();
