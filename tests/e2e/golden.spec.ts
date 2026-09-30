import { expect, test } from '@playwright/test';

// Golden path: Intro → L1 build → train → exam (3 cards) → heatmap → verdict →
// L3 → lab results → teacher. ?fast=1 shrinks the world so training finishes
// in CI time; the training itself is still real.
test('golden path through all three levels', async ({ page }) => {
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
  await page.getByRole('button', { name: /Mostly at the background/ }).click();
  await page.getByRole('button', { name: /Deliver the verdict/ }).click();

  await expect(page.getByRole('heading', { name: /Guilty: shortcut learning/ })).toBeVisible();
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
  await expect(page.locator('table.results tbody tr')).toHaveCount(15);
  await expect(page.locator('.gate-pass').first()).toBeVisible();

  await page.goto('/#/teacher');
  await expect(page.getByRole('heading', { name: /20-minute lesson/ })).toBeVisible();
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
});
