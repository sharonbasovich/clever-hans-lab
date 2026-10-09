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
  const bgOption = page.getByRole('button', { name: /Mostly at the cue/ });
  if (await bgOption.isVisible().catch(() => false)) await bgOption.click();
  await page.getByRole('button', { name: /Deliver the verdict/ }).click();

  // The verdict is derived, not hard-coded: assert the screen rendered a real
  // verdict (any of the honest outcomes).
  await expect(page.locator('h2')).toContainText(/Guilty|Partly|Honest|Inconclusive/);
  await page.getByRole('button', { name: /Level 3 — fix the data/ }).click();

  await expect(page.getByRole('heading', { name: 'Redesign the data, then test which cues the model relies on.' })).toBeVisible();
  await expect(page.locator('input[type=range]')).toHaveCount(2);
  await page.getByRole('button', { name: 'Retrain on my data →' }).click();
  await page.getByRole('button', { name: 'Start training' }).click();
  await expect(page.getByRole('heading', { name: 'Three exams, one confession' })).toBeVisible({
    timeout: 180_000,
  });

  await page.goto('/#/lab');
  await expect(page.getByRole('heading', { name: /Lab results/ })).toBeVisible();
  // 25 seeds x 8 designs.
  await expect(page.locator('table.results tbody tr')).toHaveCount(200);
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

// Round-3 P1: the rendered quiz must agree with the shared classifier on
// every outcome — exam note, quiz response and verdict title are asserted on
// injected measured evidence for all five classifier outcomes, in a real
// browser, not only via classifyOutcome unit tests.
const OUTCOME_CASES = [
  {
    name: 'shortcut',
    ev: { matched: 1, flipped: 0.02, neutral: 0.55, gap: 0.98 },
    ab: { accAgree: 1, accSwap: 0.02, cueReliance: 0.98, fillReliance: 0.9 },
    examNote: /depended on the cue/,
    quizClick: /Mostly at the cue/,
    quizText: /Correct — /,
    verdictTitle: /Guilty: shortcut learning/,
  },
  {
    name: 'partial',
    ev: { matched: 0.9, flipped: 0.62, neutral: 0.8, gap: 0.28 },
    ab: { accAgree: 0.95, accSwap: 0.62, cueReliance: 0.35, fillReliance: 0.4 },
    examNote: /Partial shortcut reliance/,
    quizClick: /A bit of both/,
    quizText: /Correct — /,
    verdictTitle: /Partly guilty/,
  },
  {
    name: 'shape',
    ev: { matched: 0.955, flipped: 0.913, neutral: 0.9, gap: 0.042 },
    ab: { accAgree: 0.96, accSwap: 0.9, cueReliance: 0.09, fillReliance: 0.1 },
    examNote: /challenge bar|honest|shape/,
    quizClick: /Mostly at the shape/,
    quizText: /Correct — /,
    verdictTitle: /Honest: it learned the shape/,
  },
  {
    name: 'undertrained',
    ev: { matched: 0.51, flipped: 0.49, neutral: 0.5, gap: 0.02 },
    ab: { accAgree: 0.5, accSwap: 0.5, cueReliance: 0, fillReliance: 0 },
    examNote: /didn't learn|collapsed|retrain/i,
    verdictTitle: /learned almost nothing/,
  },
  {
    name: 'inconclusive',
    ev: { matched: 0.99, flipped: 0.89, neutral: 0.973, gap: 0.1 },
    ab: { accAgree: 0.99, accSwap: 0.9, cueReliance: 0.1, fillReliance: 0.2 },
    examNote: /mixed|inconclusive|no clean/i,
    quizClick: /Mostly at the shape/, // any click: must NOT be marked 'Correct'
    quizText: /no clean call|mixed/,
    verdictTitle: /Inconclusive: mixed evidence/,
  },
];

for (const c of OUTCOME_CASES) {
  test(`rendered outcome ${c.name}: exam note, quiz and verdict agree with the classifier`, async ({
    page,
  }) => {
    const consoleErrors: string[] = [];
    page.on('pageerror', (e) => consoleErrors.push(String(e)));
    await page.goto('/?fast=1');
    await page.getByRole('button', { name: /Level 1 — Meet Hans/ }).click();
    await page.getByRole('button', { name: 'Train the network →' }).click();
    await page.getByRole('button', { name: 'Start training' }).click();
    await expect(page.getByRole('heading', { name: 'Three exams, one confession' })).toBeVisible({
      timeout: 180_000,
    });

    // Inject the measured evidence for this outcome and re-render each screen.
    await page.evaluate((c) => {
      const s = (window as unknown as { __chlSession: Record<string, unknown> })
        .__chlSession as {
        evaluation: unknown;
        ablation: unknown;
        stillCollapsed: boolean;
        bgMass: number;
      };
      const m = (acc: number, split: string) => ({ split, n: 100, correct: acc, acc });
      s.evaluation = {
        matched: m(c.ev.matched, 'matched'),
        flipped: m(c.ev.flipped, 'flipped'),
        neutral: m(c.ev.neutral, 'neutral'),
        gap: c.ev.gap,
      };
      s.ablation = {
        n: 100,
        acc: c.ev.matched,
        accNoBg: 0.5,
        accNoFg: 0.5,
        accAgree: c.ab.accAgree,
        accSwap: c.ab.accSwap,
        bgDrop: 0.1,
        fgDrop: 0.1,
        swapDrop: c.ab.accAgree - c.ab.accSwap,
        cueReliance: c.ab.cueReliance,
        fillReliance: c.ab.fillReliance,
        agreeWorld: null,
        swapWorld: null,
      };
      s.stillCollapsed = false;
      s.bgMass = 0.5;
      // hash is already '#/exam' — navigate away first so hashchange re-renders.
      location.hash = '#/lab';
      location.hash = '#/exam';
    }, c);
    if (c.examNote) await expect(page.locator('.note').last()).toContainText(c.examNote);

    await page.evaluate(() => (location.hash = '#/heatmap'));
    await expect(page.getByRole('heading', { name: 'Where was it looking?' })).toBeVisible();
    if (c.name === 'undertrained') {
      await expect(page.locator('.quiz-options .note')).toContainText(/No call to make/);
    } else {
      await page.getByRole('button', { name: c.quizClick! }).click();
      const quizNote = page.locator('.quiz-options .note');
      await expect(quizNote).toContainText(c.quizText!);
      if (c.name === 'inconclusive') {
        await expect(quizNote).not.toContainText(/Correct/);
      }
    }

    await page.evaluate(() => (location.hash = '#/verdict'));
    await expect(page.locator('h2')).toContainText(c.verdictTitle);
    expect(consoleErrors, `console errors: ${consoleErrors.join(' | ')}`).toEqual([]);
  });
}

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

test('route context: direct #/l3 entry sets Level 3 context', async ({ page }) => {
  // Regression: opening #/l3 directly (e.g. after a cancellation recovery)
  // left session.level at 1, so later screens kept the 'Levels 1–2' context.
  await page.goto('/?fast=1#/l3');
  await expect(
    page.getByRole('heading', { name: 'Redesign the data, then test which cues the model relies on.' }),
  ).toBeVisible();
  await expect(page.locator('.lede').first()).toContainText(/Challenge target/);

  // The L3 flow's train screen must show Level 3 context, not 'Level 1'.
  await page.getByRole('button', { name: 'Retrain on my data' }).click();
  await expect(page.locator('.kicker')).toContainText('Level 3 · Fix the data');
  await expect(page.locator('.kicker')).not.toContainText('Level 1');

  // A measured run injected for the exam must also show Level 3 context.
  await page.evaluate(() => {
    const s = (window as unknown as { __chlSession: Record<string, unknown> })
      .__chlSession as {
      evaluation: unknown;
      ablation: unknown;
      stillCollapsed: boolean;
      bgMass: number;
    };
    const m = (acc: number, split: string) => ({ split, n: 100, correct: acc, acc });
    s.evaluation = {
      matched: m(0.97, 'matched'),
      flipped: m(0.95, 'flipped'),
      neutral: m(0.55, 'neutral'),
      gap: 0.02,
    };
    s.ablation = {
      n: 100, acc: 0.97, accNoBg: 0.9, accNoFg: 0.5, accAgree: 0.97,
      accSwap: 0.95, bgDrop: 0.1, fgDrop: 0.4, swapDrop: 0.02,
      cueReliance: 0.02, fillReliance: 0.1, agreeWorld: null, swapWorld: null,
    };
    s.stillCollapsed = false;
    s.bgMass = 0.2;
    location.hash = '#/exam';
  });
  await expect(page.locator('.kicker')).toContainText('Level 3 · Fix the data');
  await expect(page.locator('.kicker')).not.toContainText('Levels 1–2');

  // Normal navigation unaffected: back to the builder resets to Level 1.
  await page.evaluate(() => (location.hash = '#/build'));
  await expect(page.locator('.kicker')).toContainText('Level 1 · Meet Hans');
});
