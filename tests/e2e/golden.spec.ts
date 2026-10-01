import { expect, test } from '@playwright/test';

// Golden path: Intro → L1 build → train → exam (3 cards) → heatmap → verdict →
// L3 → lab results → teacher. ?fast=1 shrinks the world so training finishes
// in CI time; the training itself is still real.
test('golden path through all three levels', async ({ page }) => {
  const consoleErrors: string[] = [];
  page.on('pageerror', (e) => consoleErrors.push(String(e)));
  page.on('console', (m) => {
    if (m.type() === 'error') consoleErrors.push(m.text());
  });

  await page.goto('/?fast=1');

  await expect(page.getByRole('heading', { name: 'Clever Hans Lab' })).toBeVisible();
  await page.getByRole('button', { name: /Level 1 — Meet Hans/ }).click();

  await expect(page.getByRole('heading', { name: 'Plant the cheat' })).toBeVisible();
  await expect(page.locator('.sample-grid canvas').first()).toBeVisible();
  await page.getByRole('button', { name: 'Train the network →' }).click();

  await expect(page.getByRole('heading', { name: /Train a real CNN/ })).toBeVisible();
  await page.getByRole('button', { name: 'Start training' }).click();

  // Training runs for real; exam appears when it finishes.
  await expect(page.getByRole('heading', { name: 'Three exams, one confession' })).toBeVisible({
    timeout: 180_000,
  });
  const cards = page.locator('.exam-card');
  await expect(cards).toHaveCount(3);
  await expect(cards.nth(1)).toContainText('Flipped');

  await page.getByRole('button', { name: /Where was it looking/ }).click();
  await expect(page.getByRole('heading', { name: 'Where was it looking?' })).toBeVisible();
  await expect(page.locator('.heatmap-wrap canvas').first()).toBeVisible();
  const bgOption = page.getByRole('button', { name: /Mostly at the background/ });
  if (await bgOption.isVisible().catch(() => false)) await bgOption.click();
  await page.getByRole('button', { name: /Deliver the verdict/ }).click();

  // The verdict is derived, not hard-coded: assert the screen rendered a real
  // verdict (any of the four honest outcomes).
  await expect(page.locator('h2')).toContainText(/Guilty|Honest|Inconclusive/);
  await page.getByRole('button', { name: /Level 3 — fix the data/ }).click();

  await expect(page.getByRole('heading', { name: "Design the training set so it can't cheat" })).toBeVisible();
  await expect(page.locator('input[type=range]')).toHaveCount(2);
  await page.getByRole('button', { name: 'Retrain on my data →' }).click();
  await page.getByRole('button', { name: 'Start training' }).click();
  await expect(page.getByRole('heading', { name: 'Three exams, one confession' })).toBeVisible({
    timeout: 180_000,
  });

  await page.goto('/#/lab');
  await expect(page.getByRole('heading', { name: /Lab results/ })).toBeVisible();
  // 18 seeds x 3 rhos.
  await expect(page.locator('table.results tbody tr')).toHaveCount(54);
  await expect(page.locator('.gate-pass').first()).toBeVisible();

  await page.goto('/#/teacher');
  await expect(page.getByRole('heading', { name: /20-minute lesson/ })).toBeVisible();

  // Brief G5: no console errors on the golden path.
  expect(consoleErrors, `console errors: ${consoleErrors.join(' | ')}`).toEqual([]);
});

test('lifecycle: reload on result screens recovers, not blanks', async ({ page }) => {
  await page.goto('/?fast=1#/exam');
  // No measured run exists after reload → redirected to build with a notice.
  await expect(page.getByRole('heading', { name: 'Plant the cheat' })).toBeVisible();
  await expect(page.locator('.note').first()).toContainText(/No measured run|no world/i);

  // Straight to #/train on a fresh load must rebuild a world, not dead-end.
  await page.goto('/?fast=1#/train');
  await expect(page.getByRole('button', { name: 'Start training' })).toBeEnabled();
  await page.getByRole('button', { name: 'Start training' }).click();

  // Navigating away mid-run (hash change, same session) cancels the run; the
  // stale completion must not hijack the new route.
  await page.evaluate(() => { location.hash = '#/lab'; });
  await expect(page.getByRole('heading', { name: /Lab results/ })).toBeVisible();
  // Give any stale completion a window to wrongly re-navigate.
  await page.waitForTimeout(3000);
  expect(page.url()).toContain('#/lab');
  await page.evaluate(() => { location.hash = '#/train'; });
  const startBtn = page.getByRole('button', { name: /Start training|Retry training/ });
  await expect(startBtn).toBeEnabled();
  await startBtn.click();
  // After a cancelled run, stale completion must not hijack navigation —
  // wait past the old run's window and confirm we're still on train or exam.
  await expect(page.getByRole('heading', { name: 'Three exams, one confession' })).toBeVisible({
    timeout: 180_000,
  });
});

// D3 regression: the exact round-2 QA repro — toggle Accessible cues
// mid-run, restart, then navigate away. The superseded run's late
// completion must never evaluate, publish results, or hijack the route.
test('lifecycle: a11y toggle + restart + navigate-away cannot leak a stale run', async ({ page }) => {
  await page.goto('/?fast=1#/train');
  await page.getByRole('button', { name: 'Start training' }).click();

  // Toggle Accessible cues mid-run: must cancel the in-flight run and
  // rebuild the screen so Start is available again.
  await page.getByRole('button', { name: 'Accessible cues' }).click();
  const startBtn = page.getByRole('button', { name: /Start training|Retry training/ });
  await expect(startBtn).toBeEnabled({ timeout: 30_000 });
  await startBtn.click();

  // Navigate away mid-run. The run must be cancelled; its stale completion
  // must never evaluate or force #/exam with stale results (the D3 bug).
  await page.evaluate(() => {
    location.hash = '#/lab';
  });
  await expect(page.getByRole('heading', { name: /Lab results/ })).toBeVisible();

  // Wait past the window in which a leaked run would finish and hijack the
  // route (the bug surfaced ~3 min later at full size; fast mode is much
  // shorter — 90 s covers it with margin).
  await page.waitForTimeout(90_000);
  expect(page.url()).toContain('#/lab');
  await expect(page.getByRole('heading', { name: 'Three exams, one confession' })).toHaveCount(0);

  // And the session recovers coherently: train still starts a fresh run.
  await page.goto('/?fast=1#/train');
  await expect(page.getByRole('button', { name: /Start training|Retry training/ })).toBeEnabled();
});

test('keyboard access and 375px width', async ({ page }) => {
  await page.setViewportSize({ width: 375, height: 700 });
  await page.goto('/');
  await expect(page.getByRole('heading', { name: 'Clever Hans Lab' })).toBeVisible();
  // Tab to the primary CTA and activate with Enter.
  for (let i = 0; i < 20; i++) {
    const focused = await page.evaluate(() => {
      const a = document.activeElement as HTMLElement;
      return a?.tagName === 'BUTTON' ? (a.textContent ?? '') : '';
    });
    if (focused.includes('Level 1')) break;
    await page.keyboard.press('Tab');
  }
  await page.keyboard.press('Enter');
  await expect(page.getByRole('heading', { name: 'Plant the cheat' })).toBeVisible();

  // Lab table must not overflow the 375px viewport (scrolls internally).
  await page.goto('/#/lab');
  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth > document.documentElement.clientWidth,
  );
  expect(overflow).toBe(false);
  // L3 sliders have accessible names.
  await page.goto('/#/l3');
  await expect(page.locator('input[type=range][aria-label]')).toHaveCount(2);
});
