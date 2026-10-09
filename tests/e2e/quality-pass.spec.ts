import { expect, test } from '@playwright/test';
import type { Session } from '../../src/game/session.ts';

declare global {
  interface Window {
    __chlSession: Session;
    __testNow: number;
    __testAnnouncements: string[];
  }
}

for (const failFirst of [false, true]) {
  test(`training timer excludes pre-click dwell${failFirst ? ' and resets after failure' : ' and evaluation work'}`, async ({ page }) => {
    await page.goto('/?fast=1#/train');
    await expect(page.getByRole('button', { name: 'Start training' })).toBeEnabled();
    await page.evaluate((failFirst) => {
      window.__testNow = 0;
      window.__testAnnouncements = [];
      const live = document.getElementById('aria-live')!;
      new MutationObserver(() => {
        if (live.textContent?.startsWith('Training done')) window.__testAnnouncements.push(live.textContent);
      }).observe(live, { childList: true });
      performance.now = () => window.__testNow;
      let attempts = 0;
      const s = window.__chlSession;
      s.train = async () => {
        attempts++;
        window.__testNow += 2500;
        if (failFirst && attempts === 1) throw new Error('Expected test failure');
        return { losses: [0.1], accs: [1] };
      };
      s.evaluate = () => {
        window.__testNow += 5000; // evaluation is not training time either
        const m = (acc: number, split: 'matched' | 'flipped' | 'neutral') => ({ acc, split, n: 100, correct: acc * 100 });
        s.evaluation = { matched: m(1, 'matched'), flipped: m(0, 'flipped'), neutral: m(0.5, 'neutral'), gap: 1 };
        return s.evaluation!;
      };
      s.explain = () => 0;
      location.hash = '#/lab';
    }, failFirst);
    await expect(page.getByRole('heading', { name: /Lab results/ })).toBeVisible();
    await page.evaluate(() => { location.hash = '#/train'; });
    await expect(page.getByRole('button', { name: 'Start training' })).toBeEnabled();
    await page.evaluate(() => { window.__testNow += 60_000; });
    await page.getByRole('button', { name: 'Start training' }).click();
    if (failFirst) {
      await expect(page.getByRole('button', { name: 'Retry training' })).toBeEnabled();
      await expect(page.locator('.note').last()).toContainText('Training failed');
      await page.evaluate(() => { window.__testNow += 120_000; });
      await page.getByRole('button', { name: 'Retry training' }).click();
    }
    await expect(page.getByRole('heading', { name: 'Three exams, one confession' })).toBeVisible();
    await expect.poll(() => page.evaluate(() => window.__testAnnouncements)).toEqual([
      'Training done in 2.5s. Matched accuracy 100.0%, flipped 0.0%.',
    ]);
  });
}

test('cancelled completion never evaluates or announces a training duration', async ({ page }) => {
  await page.goto('/?fast=1#/train');
  await expect(page.getByRole('button', { name: 'Start training' })).toBeEnabled();
  await page.evaluate(() => {
    const s = window.__chlSession;
    s.train = async () => null;
    s.evaluate = () => { throw new Error('Cancelled run must not evaluate'); };
    s.explain = () => { throw new Error('Cancelled run must not explain'); };
  });
  await page.getByRole('button', { name: 'Start training' }).click();
  await expect(page.locator('#aria-live')).not.toContainText('Training done');
  await expect(page.getByRole('heading', { name: /Train a real CNN/ })).toBeVisible();
});

test('mixed-rho comparison, repeated keyboard toggle, reduced motion, mobile and forward-stage focus', async ({ page }, testInfo) => {
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(String(e)));
  await page.goto('/?fast=1#/train');
  await page.getByRole('button', { name: 'Start training' }).click();
  await expect(page.getByRole('heading', { name: 'Three exams, one confession' })).toBeVisible({ timeout: 180_000 });

  // Real trained weights, then real evaluation on a mixed-rho Level 3 world.
  const evidence = await page.evaluate(() => {
    const s = window.__chlSession;
    s.level = 3;
    s.buildLevel3World(0.5, 0.3);
    s.evaluate();
    s.explain(1);
    const ab = s.ablation!;
    const agree = ab.agreeWorld!;
    const swap = ab.swapWorld!;
    const mixed = s.testSets!.matched;
    let shapesIdentical = true;
    let oppositeCues = true;
    let mixedHasReversedCue = false;
    for (let i = 0; i < ab.n; i++) {
      if (agree.labels[i] !== swap.labels[i] || agree.cue[i] !== agree.labels[i] || swap.cue[i] !== 1 - agree.cue[i]) oppositeCues = false;
      if (mixed.cue[i] !== mixed.labels[i]) mixedHasReversedCue = true;
      for (let p = 0; p < 1024; p++) {
        const m = i * 1024 + p;
        if (agree.fgMasks[m] !== swap.fgMasks[m]) shapesIdentical = false;
        if (agree.fgMasks[m] === 1) {
          for (let c = 0; c < 3; c++) if (agree.images[m * 3 + c] !== swap.images[m * 3 + c]) shapesIdentical = false;
        }
      }
    }
    location.hash = '#/heatmap';
    return { acc: ab.acc, accAgree: ab.accAgree, accSwap: ab.accSwap, n: ab.n, shapesIdentical, oppositeCues, mixedHasReversedCue };
  });
  expect(evidence.shapesIdentical).toBe(true);
  expect(evidence.oppositeCues).toBe(true);
  expect(evidence.mixedHasReversedCue).toBe(true);
  expect(evidence.acc).not.toBe(evidence.accAgree);
  const pair = page.locator('.cue-pair');
  await expect(pair.locator('.diagnostic-card')).toHaveCount(2);
  await expect(pair.locator('h3').nth(0)).toHaveText('Cue agrees');
  await expect(pair.locator('h3').nth(1)).toHaveText('Cue reversed');
  await expect(pair.locator('.big').nth(0)).toHaveText(`${(evidence.accAgree * 100).toFixed(1)}%`);
  await expect(pair.locator('.big').nth(1)).toHaveText(`${(evidence.accSwap * 100).toFixed(1)}%`);
  await expect(pair.locator('figcaption').first()).toHaveText('Example image; no individual prediction shown');
  await expect(pair.locator('.desc').first()).toHaveText(`Aggregate accuracy on ${evidence.n} held-out images`);
  const pixelsBefore = await pair.locator('canvas').evaluateAll((nodes) => nodes.map((c) => (c as HTMLCanvasElement).toDataURL()));
  await expect(pair).toHaveCSS('grid-template-columns', /\d+px \d+px/);
  const toggle = page.locator('.cue-controls button');
  await page.keyboard.press('Tab');
  await expect(toggle).toBeFocused();
  await expect(toggle).toHaveCSS('outline-style', 'solid');
  for (let i = 0; i < 6; i++) {
    await page.keyboard.press(i % 2 === 0 ? 'Enter' : 'Space');
    await expect(pair).toHaveAttribute('data-emphasis', i % 2 === 0 ? 'swap' : 'agree');
    await expect(toggle).toBeFocused();
    await expect(toggle).toHaveAttribute('aria-pressed', String(i % 2 === 0));
  }
  expect(await pair.locator('canvas').evaluateAll((nodes) => nodes.map((c) => (c as HTMLCanvasElement).toDataURL()))).toEqual(pixelsBefore);
  await expect(pair.locator('canvas').first()).toHaveCSS('transition-duration', '0.18s');
  await page.screenshot({ path: testInfo.outputPath('paired-desktop.png'), fullPage: true });

  await page.keyboard.press('Tab');
  const summary = page.locator('.fill-diagnostics summary');
  await expect(summary).toBeFocused();
  await page.keyboard.press('Enter');
  await expect(page.locator('.fill-diagnostics')).toHaveAttribute('open', '');
  await expect(page.locator('.fill-diagnostics .big').first()).toHaveText(`${(evidence.acc * 100).toFixed(1)}%`);
  await page.keyboard.press('Space');
  await expect(page.locator('.fill-diagnostics')).not.toHaveAttribute('open', '');

  await page.emulateMedia({ reducedMotion: 'reduce' });
  await expect(pair.locator('canvas').first()).toHaveCSS('transition-duration', '0s');
  await toggle.focus();
  await page.keyboard.press('Enter');
  await expect(pair.locator('canvas').first()).toHaveCSS('opacity', '0.65');
  await expect(pair.locator('canvas').last()).toHaveCSS('opacity', '1');
  await page.setViewportSize({ width: 375, height: 700 });
  await expect(pair).toHaveCSS('grid-template-columns', /^\d+(\.\d+)?px$/);
  expect(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth)).toBe(false);
  await page.screenshot({ path: testInfo.outputPath('paired-mobile-reduced-motion.png'), fullPage: true });

  // Advance from the bottom using a real in-app button. Heading clears header.
  const deliver = page.getByRole('button', { name: /Deliver the verdict/ });
  await deliver.scrollIntoViewIfNeeded();
  const priorStageScroll = await page.evaluate(() => scrollY);
  expect(priorStageScroll).toBeGreaterThan(0);
  await deliver.click();
  await expect(page.locator('h2')).toContainText(/Guilty|Partly|Honest|Inconclusive/);
  await expect(page.locator('#app')).toBeFocused();
  expect(await page.evaluate(() => document.querySelector('h2')!.getBoundingClientRect().top >= document.querySelector('.site-header')!.getBoundingClientRect().bottom)).toBe(true);
  await page.screenshot({ path: testInfo.outputPath('verdict-mobile.png') });
  await page.goBack();
  await expect(pair).toBeVisible();
  await expect.poll(() => page.evaluate(() => scrollY)).toBe(priorStageScroll);
  // A state re-render does not jump to top.
  await page.evaluate(() => { window.scrollTo(0, 300); });
  const scrollBefore = await page.evaluate(() => scrollY);
  await page.evaluate(() => { document.getElementById('a11y-toggle')!.click(); });
  expect(await page.evaluate(() => scrollY)).toBe(scrollBefore);
  expect(errors).toEqual([]);
  await testInfo.attach('measured-pair-evidence', { body: JSON.stringify(evidence, null, 2), contentType: 'application/json' });
});
