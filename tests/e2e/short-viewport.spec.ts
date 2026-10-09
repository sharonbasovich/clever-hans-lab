import { expect, test, type Locator, type Page } from '@playwright/test';

async function tabTo(page: Page, control: Locator) {
  for (let i = 0; i < 30; i++) {
    if (await control.evaluate(e => e === document.activeElement)) break;
    await page.keyboard.press('Tab');
  }
  await expect(control).toBeFocused();
  expect(await control.evaluate(e => {
    const r = e.getBoundingClientRect();
    const header = document.querySelector('.site-header')!;
    const h = header.getBoundingClientRect();
    const overlayBottom = getComputedStyle(header).position === 'sticky' && h.top <= 1 ? h.bottom : 0;
    return e.matches(':focus-visible') && r.left >= 0 && r.right <= innerWidth
      && Math.min(r.bottom, innerHeight) > Math.max(r.top, overlayBottom);
  })).toBe(true);
}

for (const reducedMotion of ['no-preference', 'reduce'] as const) {
  test(`short viewport keeps native Tab focus visible (${reducedMotion})`, async ({ page }) => {
    // CSS layout regression only; full native 400% default training evidence
    // is collected separately by the isolated Edge verification harness.
    await page.setViewportSize({ width: 320, height: 225 });
    await page.emulateMedia({ reducedMotion });
    await page.goto('/');
    await expect(page.locator('.site-header')).toHaveCSS('position', 'static');
    const start = page.getByRole('button', { name: /Meet Hans/ });
    await tabTo(page, start);
    await page.keyboard.press('Enter');
    const train = page.getByRole('button', { name: /Train the network/ });
    await tabTo(page, train);
    await page.keyboard.press('Enter');
    await tabTo(page, page.getByRole('button', { name: 'Start training', exact: true }));
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    await page.setViewportSize({ width: 640, height: 450 });
    await expect(page.locator('.site-header')).toHaveCSS('position', 'sticky');
  });
}
