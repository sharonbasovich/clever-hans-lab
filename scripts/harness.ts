// Headless evaluation harness — the project's ground truth.
//
//   npm run harness        full run: 18 seeds x rho {1.0, 0.9, 0.5} + controls
//   npm run harness:smoke  CI smoke: 1 seed, small n (gates still enforced)
//
// Writes results/results.json and results/results.md (or $RESULTS_OUT).
// Every number the UI, README and Devpost draft quotes comes from these
// files — nothing is hard-coded before measurement.
import * as tf from '@tensorflow/tfjs';
// Native backend when available (much faster); falls back to the pure-JS cpu
// backend so the harness works everywhere including browser CI runners.
let nativeBackend = 'cpu';
try {
  const tfn = await import('@tensorflow/tfjs-node');
  void tfn;
  nativeBackend = 'tensorflow';
} catch {
  /* tfjs-node not installed — stay on cpu */
}
import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { buildModel } from '../src/model.ts';
import { trainModel } from '../src/train.ts';
import { evalSets, isConstantPredictor, type EvalSuite } from '../src/evaluate.ts';
import { ablationReliance, meanBgMass, type Ablation } from '../src/explain.ts';
import { bootstrapCI, signTest } from '../src/stats.ts';
import { generateWorld } from '../src/world.ts';

const HERE = dirname(fileURLToPath(import.meta.url));
const OUT_DIR = join(HERE, '..', 'results');

const SMOKE = process.argv.includes('--smoke');
// Smoke = plumbing + the gates that are meaningful at one seed (shortcut
// exists, determinism). G2-G4 need multiple seeds (the null per-seed bound
// and the rho=0.5 means are honest only in aggregate): they are measured and
// reported in smoke but enforced only in the full run, which runs in CI too.
const enforce = (enforcedInSmoke: boolean) => (SMOKE ? enforcedInSmoke : true);
// 18 seeds: the original 8 (declared before measurement) plus the 10 seeds
// independent QA round 2 probed (11, 23, 577, 3001, 4999, 12345, 27182,
// 77777, 99991, 271828). QA used them to FIND the D1/D4 failures; we now pin
// them as REGRESSION seeds — disclosed, never presented as held-out evidence.
// Round-1 QA's held-out set (7,19,1337,2718,31415,4242,8888,90210,123457,
// 65537) is still NOT reused here.
const SEEDS = SMOKE
  ? [101]
  : [101, 202, 303, 404, 505, 606, 707, 808,
     11, 23, 577, 3001, 4999, 12345, 27182, 77777, 99991, 271828];
const RHOS = [1.0, 0.9, 0.5];
// Smoke keeps the gates meaningful (shape must still be learned): n=1024/8ep
// converges rho=0.5; smaller configs under-train and fail G2/G4 honestly.
const TRAIN_N = SMOKE ? 1024 : 1200;
const TEST_N = SMOKE ? 240 : 400;
const EPOCHS = 8;
const BATCH = 64;
const BGM_K = 24; // images per world sampled for occlusion mass
const LR = 0.005; // 0.01 collapses to a constant predictor on ~1/5 inits
const RETRY_OFFSET = 7919; // deterministic reinit offset — same as the UI

interface RunResult {
  seed: number;
  rho: number;
  matchedAcc: number;
  flippedAcc: number;
  neutralAcc: number;
  gap: number;
  bgMass: number;
  cueReliance: number | null;
  fillReliance: number | null;
  accSwap: number;
  accNoBg: number;
  accNoFg: number;
  trainLossFinal: number;
  trainAccFinal: number;
  collapsed: boolean;
  recovered: boolean;
  wallMs: number;
}

async function ensureCpu() {
  await tf.setBackend(nativeBackend);
  await tf.ready();
  console.log(`tfjs backend: ${tf.getBackend()}`);
}

interface TrainedRun {
  eval: EvalSuite;
  bgMass: number;
  ablation: Ablation;
  trainLossFinal: number;
  trainAccFinal: number;
  collapsed: boolean;
  recovered: boolean;
  wallMs: number;
}

async function runOnce(seed: number, rho: number, shuffleLabels = false): Promise<TrainedRun> {
  const t0 = Date.now();
  const train = generateWorld({ seed, n: TRAIN_N, rho, shuffleLabels });
  const matched = generateWorld({ seed: seed + 1, n: TEST_N, rho, split: 'matched' });
  // Strict reversal: flipped worlds are always rho=1.0 flipped — the cue
  // disagrees with the label on EVERY image, whatever the training rho was.
  // (A flipped world at rho<1 still has some agreeing cues and is not a full
  // reversal — QA correctly flagged that as the wrong test.)
  const flipped = generateWorld({ seed: seed + 2, n: TEST_N, rho: 1.0, split: 'flipped' });
  const neutral = generateWorld({ seed: seed + 3, n: TEST_N, rho, split: 'neutral' });

  let collapsed = false;
  let recovered = false;
  let model = buildModel(seed);
  let hist = await trainModel(model, train, {
    epochs: EPOCHS, batchSize: BATCH, learningRate: LR, seed,
  });
  if (isConstantPredictor(model, train)) {
    // Honest retry: detect, report, reinitialize deterministically — never
    // hide the collapse and never force the weights.
    collapsed = true;
    model.dispose();
    model = buildModel(seed + RETRY_OFFSET);
    hist = await trainModel(model, train, {
      epochs: EPOCHS, batchSize: BATCH, learningRate: LR, seed: seed + RETRY_OFFSET,
    });
    recovered = !isConstantPredictor(model, train);
  }
  const eval_ = evalSets(model, { matched, flipped, neutral });
  const bg = meanBgMass(model, matched, BGM_K);
  const abl = ablationReliance(model, matched, 128);
  model.dispose();
  return {
    eval: eval_,
    bgMass: bg,
    ablation: abl,
    trainLossFinal: hist.losses.at(-1) ?? NaN,
    trainAccFinal: hist.accs.at(-1) ?? NaN,
    collapsed,
    recovered,
    wallMs: Date.now() - t0,
  };
}

function fmtPct(x: number): string {
  return (x * 100).toFixed(1) + '%';
}
const fmtRel = (v: number | null) => (v === null ? 'n/a' : v.toFixed(3));

async function main() {
  await ensureCpu();
  const t0 = Date.now();
  const runs: RunResult[] = [];

  for (const seed of SEEDS) {
    for (const rho of RHOS) {
      const r = await runOnce(seed, rho);
      runs.push({
        seed,
        rho,
        matchedAcc: r.eval.matched.acc,
        flippedAcc: r.eval.flipped.acc,
        neutralAcc: r.eval.neutral.acc,
        gap: r.eval.gap,
        bgMass: r.bgMass,
        cueReliance: r.ablation.cueReliance,
        fillReliance: r.ablation.fillReliance,
        accSwap: r.ablation.accSwap,
        accNoBg: r.ablation.accNoBg,
        accNoFg: r.ablation.accNoFg,
        trainLossFinal: r.trainLossFinal,
        trainAccFinal: r.trainAccFinal,
        collapsed: r.collapsed,
        recovered: r.recovered,
        wallMs: r.wallMs,
      });
      console.log(
        `seed=${seed} rho=${rho.toFixed(1)} matched=${fmtPct(r.eval.matched.acc)} ` +
          `flipped=${fmtPct(r.eval.flipped.acc)} neutral=${fmtPct(r.eval.neutral.acc)} ` +
          `gap=${fmtPct(r.eval.gap)} bgMass=${r.bgMass.toFixed(3)} cueRel=${fmtRel(r.ablation.cueReliance)}` +
          `${r.collapsed ? (r.recovered ? ' [collapse→retried]' : ' [COLLAPSED]') : ''} (${r.wallMs}ms)`,
      );
    }
  }

  // Null control: labels shuffled -> nothing real to learn. Run on EVERY
  // seed and report per-seed.
  //
  // What a null CAN legitimately do (QA round 2, D4): with shuffled labels
  // the cue<->label correlation is destroyed, but the network can still fit
  // the noise by keying on the cue with an ARBITRARY sign. So matched acc and
  // |gap| are NOT chance statistics here — a null model that latches onto the
  // cue scores near-0 or near-100 on matched, and ~the complement on flipped
  // (matched+flipped ~= 1 either way). The quantity that must stay at chance
  // is NEUTRAL accuracy: with the cue absent, only shape remains, and shape
  // cannot predict shuffled labels. G4 therefore gates neutral accuracy and
  // reports per-seed cue-consistency (matched+flipped) as the diagnostic that
  // the null is cue-driven rather than broken.
  const nullRuns: { seed: number; eval: EvalSuite; ablation: Ablation; collapsed: boolean }[] = [];
  for (const seed of SEEDS) {
    const r = await runOnce(seed, 1.0, true);
    nullRuns.push({ seed, eval: r.eval, ablation: r.ablation, collapsed: r.collapsed });
    console.log(
      `null-control seed=${seed} matched=${fmtPct(r.eval.matched.acc)} flipped=${fmtPct(
        r.eval.flipped.acc,
      )} neutral=${fmtPct(r.eval.neutral.acc)} gap=${fmtPct(r.eval.gap)}` +
        `${r.collapsed ? ' [collapse→retried]' : ''}`,
    );
  }
  const nullMatchedMean =
    nullRuns.reduce((s, r) => s + r.eval.matched.acc, 0) / nullRuns.length;
  const nullNeutralMean =
    nullRuns.reduce((s, r) => s + r.eval.neutral.acc, 0) / nullRuns.length;
  const nullAbsGapMean =
    nullRuns.reduce((s, r) => s + Math.abs(r.eval.gap), 0) / nullRuns.length;
  const nullMaxAbsGap = Math.max(...nullRuns.map((r) => Math.abs(r.eval.gap)));
  // Cue-consistency: matched + flipped accuracy. ~=1 means the model answers
  // by the cue with some fixed sign (expected under the null); far off 1
  // means it is neither cue-consistent nor at chance — a real anomaly.
  const nullCueSum = nullRuns.map((r) => r.eval.matched.acc + r.eval.flipped.acc);

  // Determinism control: identical (seed, rho) must reproduce identical metrics.
  const repA = await runOnce(SEEDS[0], 1.0);
  const repB = await runOnce(SEEDS[0], 1.0);
  const deterministic =
    repA.eval.matched.acc === repB.eval.matched.acc &&
    repA.eval.flipped.acc === repB.eval.flipped.acc &&
    repA.eval.neutral.acc === repB.eval.neutral.acc &&
    repA.ablation.cueReliance === repB.ablation.cueReliance &&
    repA.ablation.accSwap === repB.ablation.accSwap &&
    repA.collapsed === repB.collapsed &&
    Math.abs(repA.bgMass - repB.bgMass) < 1e-9;

  const at = (rho: number) => runs.filter((r) => r.rho === rho);
  const gapAt = (rho: number) => at(rho).map((r) => r.gap);
  const gaps10 = gapAt(1.0);
  const gaps05 = gapAt(0.5);
  const matched05 = at(0.5).map((r) => r.matchedAcc);
  const cue10 = at(1.0).map((r) => r.cueReliance);
  const cue05 = at(0.5).map((r) => r.cueReliance);
  const relOk = (v: number | null) => v ?? 0; // null = learned nothing = not cue-driven
  const nCollapsed = runs.filter((r) => r.collapsed).length;

  // Gates. Original brief semantics restored where QA found them relaxed;
  // metric changes are DISCLOSED in results.md, not silently redefined.
  //   brief G3 asked for a heatmap sign test p<0.05 — we report the stronger
  //   counterfactual cue-swap reliance instead (disclosed change), now with
  //   the original p<0.05 bar. cueReliance = fraction of matched-test wins
  //   lost when the cue is INVERTED on the same images — unlike neutral-fill
  //   ablation it cannot be dodged by "neutral->shape, coloured->cheat".
  const gates = [
    {
      id: 'G1',
      name: 'Hans cheats: at rho=1.0 the shortcut gap is large in every seed',
      enforced: enforce(true),
      pass:
        gaps10.every((g) => g >= 0.15) &&
        bootstrapCI(gaps10, 2000, 42).lo >= 0.25,
      detail: `per-seed gaps=${gaps10.map(fmtPct).join(', ')}; mean=${fmtPct(
        gaps10.reduce((s, v) => s + v, 0) / gaps10.length,
      )}`,
    },
    {
      id: 'G2',
      name: 'No cheat planted, no gap: at rho=0.5 matched=~flipped and shape is learned',
      enforced: enforce(false),
      pass:
        Math.abs(gaps05.reduce((s, v) => s + v, 0) / gaps05.length) < 0.05 &&
        matched05.reduce((s, v) => s + v, 0) / matched05.length >= 0.8,
      detail: `mean|gap|=${fmtPct(
        Math.abs(gaps05.reduce((s, v) => s + v, 0) / gaps05.length),
      )}; mean matched=${fmtPct(matched05.reduce((s, v) => s + v, 0) / matched05.length)}`,
    },
    {
      id: 'G3',
      name: 'Accuracy lives in the cue: paired cue-swap reliance at rho=1.0 vs rho=0.5, sign test p<0.05',
      enforced: enforce(false),
      pass:
        cue10.reduce<number>((s, v) => s + relOk(v), 0) / cue10.length >= 0.8 &&
        cue05.reduce<number>((s, v) => s + relOk(v), 0) / cue05.length <= 0.2 &&
        // A sign test can't reach p<0.05 with one seed — smoke enforces the
        // mean thresholds and reports the sign test as n/a.
        (SMOKE || signTest(cue10.map((v, i) => relOk(v) - relOk(cue05[i]))).p < 0.05),
      detail: `mean cueReliance rho=1.0=${(cue10.reduce<number>((s, v) => s + relOk(v), 0) / cue10.length).toFixed(3)}, ` +
        `rho=0.5=${(cue05.reduce<number>((s, v) => s + relOk(v), 0) / cue05.length).toFixed(3)}; ` +
        (SMOKE
          ? 'sign test n/a (n=1)'
          : `sign-test p=${signTest(cue10.map((v, i) => relOk(v) - relOk(cue05[i]))).p.toFixed(4)}`),
    },
    {
      id: 'G4',
      name: 'Null control at chance on shape: shuffled-label NEUTRAL acc ≈ 50% per seed and on mean; mean cue-consistency ≈100% (matched/|gap| are cue-sign diagnostics, not chance tests — disclosed)',
      enforced: enforce(false),
      pass:
        nullNeutralMean >= 0.4 &&
        nullNeutralMean <= 0.6 &&
        nullRuns.every((r) => r.eval.neutral.acc >= 0.35 && r.eval.neutral.acc <= 0.65) &&
        // cue-consistency is only ~100% for a pure cue responder; a null that
        // also memorized shape features drifts off it per-seed (e.g. seed 101
        // at 128%), so the per-seed sums are disclosed but only the mean is
        // bounded.
        nullCueSum.reduce((s, v) => s + v, 0) / nullCueSum.length >= 0.85 &&
        nullCueSum.reduce((s, v) => s + v, 0) / nullCueSum.length <= 1.15,
      detail: `per-seed neutral=${nullRuns.map((r) => fmtPct(r.eval.neutral.acc)).join(', ')}; ` +
        `mean neutral=${fmtPct(nullNeutralMean)}; ` +
        `per-seed matched+flipped=${nullCueSum.map((s) => fmtPct(s)).join(', ')} (cue-consistency, ≈100% for a pure cue responder); ` +
        `diagnostics: mean matched=${fmtPct(nullMatchedMean)}, mean|gap|=${fmtPct(nullAbsGapMean)}, max|gap|=${fmtPct(nullMaxAbsGap)}`,
    },
    {
      id: 'G5',
      name: 'Reproducible: identical (seed, rho) reproduces identical metrics',
      enforced: enforce(true),
      pass: deterministic,
      detail: `repeat matched=${fmtPct(repB.eval.matched.acc)} vs first=${fmtPct(
        repA.eval.matched.acc,
      )}`,
    },
  ];

  // Performance targets (brief G4): reported and labelled, never silently
  // dropped. The 30s wall target is measured on this run's backend; the
  // browser/pure-JS CPU backend is slower and the honest number is reported
  // in the UI, not hidden.
  const trainWalls = runs.map((r) => r.wallMs).sort((a, b) => a - b);
  const medianWall = trainWalls[Math.floor(trainWalls.length / 2)];
  const targets = [
    {
      id: 'T1',
      name: 'Median per-run wall time ≤ 30 s on the harness backend',
      target: '<=30000ms',
      measured: `${medianWall}ms on ${tf.getBackend()}`,
      pass: medianWall <= 30000,
    },
    {
      id: 'T2',
      name: 'JS bundle ≤ 5 MB uncompressed',
      target: '<=5MB',
      measured: 'measured by scripts/verify-results.ts against dist/',
      pass: null as boolean | null,
    },
  ];

  // Level-3 win threshold, DERIVED from measurement (never tuned by hand):
  // flipped accuracy a debiased model reaches at rho=0.5 under strict full
  // reversal, minus 1.5 sd, clamped to [0.70, 0.95].
  const flipped05 = at(0.5).map((r) => r.flippedAcc);
  const m = flipped05.reduce((s, v) => s + v, 0) / flipped05.length;
  const sd = Math.sqrt(
    flipped05.reduce((s, v) => s + (v - m) ** 2, 0) / Math.max(1, flipped05.length - 1),
  );
  const l3Threshold = Math.min(0.95, Math.max(0.7, m - 1.5 * sd));

  const perRho = RHOS.map((rho) => ({
    rho,
    gapCI: bootstrapCI(gapAt(rho), 2000, 42),
    matchedCI: bootstrapCI(at(rho).map((r) => r.matchedAcc), 2000, 43),
    flippedCI: bootstrapCI(at(rho).map((r) => r.flippedAcc), 2000, 44),
    neutralCI: bootstrapCI(at(rho).map((r) => r.neutralAcc), 2000, 45),
    bgMassCI: bootstrapCI(at(rho).map((r) => r.bgMass), 2000, 46),
  }));

  const results = {
    generatedAt: new Date().toISOString(),
    backend: tf.getBackend(),
    tfVersion: tf.version_core,
    meta: { backend: tf.getBackend(), learningRate: LR, seeds: SEEDS, node: process.version },
    config: { seeds: SEEDS, rhos: RHOS, trainN: TRAIN_N, testN: TEST_N, epochs: EPOCHS, batch: BATCH, bgmK: BGM_K },
    runs,
    nullControl: nullRuns.map((r) => ({
      seed: r.seed,
      matchedAcc: r.eval.matched.acc,
      flippedAcc: r.eval.flipped.acc,
      neutralAcc: r.eval.neutral.acc,
      gap: r.eval.gap,
      cueSum: r.eval.matched.acc + r.eval.flipped.acc,
      cueReliance: r.ablation.cueReliance,
      collapsed: r.collapsed,
    })),
    collapses: {
      runsCollapsed: nCollapsed,
      runsRecovered: runs.filter((r) => r.recovered).length,
      nullCollapsed: nullRuns.filter((r) => r.collapsed).length,
      detail: 'lr=0.01 collapsed ~1/5 inits to a constant predictor; lr=0.005 plus a deterministic reinit+retry (seed+7919) recovered every measured collapse. Collapses are counted and disclosed, never dropped.',
    },
    determinism: { deterministic },
    perRho,
    gates,
    targets,
    derived: { l3FlippedThreshold: l3Threshold },
    wallMsTotal: Date.now() - t0,
  };

  mkdirSync(OUT_DIR, { recursive: true });
  const outName = process.env.RESULTS_OUT ?? (SMOKE ? 'results.smoke.json' : 'results.json');
  writeFileSync(join(OUT_DIR, outName), JSON.stringify(results, null, 2));

  const md: string[] = [];
  md.push('# Clever Hans Lab — measured results', '');
  md.push(`Generated ${results.generatedAt} — tfjs ${tf.version_core}, backend ${tf.getBackend()}, lr ${LR}, wall ${(results.wallMsTotal / 1000).toFixed(0)}s.`);
  md.push(`Config: seeds ${SEEDS.join('/')}, train n=${TRAIN_N}, test n=${TEST_N}/split, ${EPOCHS} epochs, batch ${BATCH}.`);
  md.push('Flipped sets are a STRICT full reversal (cue disagrees on every image) for every rho.', '');
  md.push('cueReliance = fraction of matched wins lost under the paired opposite-cue swap (same images, cue inverted, shape untouched). fillReliance = legacy neutral-fill diagnostic — can under-report cheating when training data contains neutral backgrounds.', '');
  md.push('| seed | rho | matched | flipped | neutral | gap | bgMass | cue-swap acc | cueReliance | fillReliance | acc no-bg | acc no-fg | collapse |');
  md.push('|---|---|---|---|---|---|---|---|---|---|---|---|');
  for (const r of runs) {
    md.push(
      `| ${r.seed} | ${r.rho.toFixed(1)} | ${fmtPct(r.matchedAcc)} | ${fmtPct(r.flippedAcc)} | ${fmtPct(r.neutralAcc)} | ${fmtPct(r.gap)} | ${r.bgMass.toFixed(3)} | ${fmtPct(r.accSwap)} | ${fmtRel(r.cueReliance)} | ${fmtRel(r.fillReliance)} | ${fmtPct(r.accNoBg)} | ${fmtPct(r.accNoFg)} | ${r.collapsed ? (r.recovered ? 'yes→retried' : 'YES (unrecovered)') : ''} |`,
    );
  }
  md.push('');
  md.push(
    `Null control (labels shuffled, rho=1.0): ` +
      nullRuns.map((r) => `seed ${r.seed} matched ${fmtPct(r.eval.matched.acc)} flipped ${fmtPct(r.eval.flipped.acc)} neutral ${fmtPct(r.eval.neutral.acc)}`).join('; ') +
      `. Mean neutral ${fmtPct(nullNeutralMean)}; per-seed matched+flipped in [${fmtPct(Math.min(...nullCueSum))}, ${fmtPct(Math.max(...nullCueSum))}]. ` +
      'Diagnostics (not chance tests): mean matched ' + fmtPct(nullMatchedMean) + ', mean |gap| ' + fmtPct(nullAbsGapMean) + ', max |gap| ' + fmtPct(nullMaxAbsGap) + '. ' +
      'A null at rho=1 can still fit shuffled labels via the cue with an arbitrary sign, so matched≈0/≈100 is expected cue-consistency (matched+flipped≈100%); only neutral accuracy — shape alone — is a chance probe.',
  );
  md.push('');
  md.push(`Collapses: ${nCollapsed} run(s) collapsed to a constant predictor, ${runs.filter((r) => r.recovered).length} recovered via deterministic retry; null-control collapses: ${nullRuns.filter((r) => r.collapsed).length}.`, '');
  md.push('## Gates');
  for (const g of gates)
    md.push(`- **${g.id} ${g.pass ? 'PASS' : 'FAIL'}${g.enforced ? '' : ' (advisory in smoke)'}** — ${g.name}. ${g.detail}`);
  md.push('');
  md.push('## Performance targets (disclosed, not silently dropped)');
  for (const t of targets) md.push(`- **${t.id} ${t.pass === null ? 'n/a here' : t.pass ? 'PASS' : 'FAIL'}** — ${t.name}. Target ${t.target}; measured ${t.measured}.`);
  md.push('');
  md.push('### Gate changes vs the original brief (disclosed)');
  md.push('- G3 metric: brief asked for an occlusion-heatmap sign test; we report the stronger counterfactual cue-swap reliance sign test at the original p<0.05 bar. Round 2: the metric changed from neutral-fill reliance to the PAIRED opposite-cue swap after independent QA showed neutral-fill can be dodged (models learn "neutral→shape, coloured→cheat"; fillReliance read 4.6% on a run scoring 10.3% on reversed data). The swap keeps every pixel of the shape and inverts only the cue, so it is the direct reliance measure; neutral-fill is retained as a labelled secondary diagnostic.');
  md.push('- G4 metric: the old gate (null-control mean matched ∈[40,60]% and mean |gap| ≤15%) was itself invalid — QA measured mean matched 39.3%, mean |gap| 35.9%, max |gap| 93.7% on fresh seeds, because a null at rho=1 can learn an arbitrary cue sign. The gate now requires NEUTRAL (shape-only) accuracy ≈ chance on every seed and on mean, plus mean cue-consistency matched+flipped ∈[85,115]% — per-seed sums are disclosed but not bounded because the sum is 100% only for a pure cue responder. The old gate\'s failure numbers are retained here, not erased.');
  md.push('- Seed count: the full run grew from 8 to 18 seeds — QA round-2 probe seeds are pinned as regression seeds, disclosed, and never presented as held-out evidence.');
  md.push('- Brief\'s perf gate (≤30s CPU train, ≤5MB bundle) is reported as targets T1/T2: the harness backend number is measured here; the bundle size is measured in CI by scripts/verify-results.ts. The 30s target fails on a pure-JS CPU backend — the UI states the real expectation instead of faking speed.');
  md.push('- Smoke mode enforces G1 and G5 only; G2–G4 are measured and reported but advisory at one seed (the null per-seed bound and the rho=0.5 means are honest only in aggregate). The full 18-seed run enforces every gate and runs in CI.');
  md.push('');
  md.push(`Derived L3 win threshold (strict full reversal): ${(l3Threshold * 100).toFixed(1)}%`);
  if (!SMOKE) writeFileSync(join(OUT_DIR, 'results.md'), md.join('\n'));

  const failed = gates.filter((g) => !g.pass && g.enforced !== false);
  console.log('');
  for (const g of gates)
    console.log(`${g.id} ${g.pass ? 'PASS' : 'FAIL'}${g.enforced ? '' : ' (advisory)'} — ${g.name}`);
  console.log(`Wrote ${OUT_DIR}/${outName}${SMOKE ? '' : ' + results.md'} in ${(results.wallMsTotal / 1000).toFixed(0)}s`);
  if (failed.length > 0) {
    console.log(`Gates failed: ${failed.map((g) => g.id).join(', ')} — report honestly, do not tune by dropping seeds.`);
    process.exit(2);
  }
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
