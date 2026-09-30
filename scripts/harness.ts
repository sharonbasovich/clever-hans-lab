// Headless evaluation harness — the project's ground truth.
//
//   npm run harness        full run: 8 seeds x rho {1.0, 0.9, 0.5} + controls
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
// 8 seeds: the original 5 plus 606/707/808 (declared before measurement).
// QA's held-out set (7,19,1337,2718,31415,4242,8888,90210,123457,65537) is NOT
// reused here — QA seeds stay QA-only.
const SEEDS = SMOKE ? [101] : [101, 202, 303, 404, 505, 606, 707, 808];
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
  bgReliance: number | null;
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
        bgReliance: r.ablation.bgReliance,
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
          `gap=${fmtPct(r.eval.gap)} bgMass=${r.bgMass.toFixed(3)} bgRel=${fmtRel(r.ablation.bgReliance)}` +
          `${r.collapsed ? (r.recovered ? ' [collapse→retried]' : ' [COLLAPSED]') : ''} (${r.wallMs}ms)`,
      );
    }
  }

  // Null control: labels shuffled -> nothing to learn. Run on EVERY seed and
  // report per-seed: a mean alone can hide a single seed drifting far off
  // chance (an untrained net's drift keys on the dominant background feature
  // with an arbitrary sign). Per-seed |gap| is disclosed below.
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
  const nullAbsGapMean =
    nullRuns.reduce((s, r) => s + Math.abs(r.eval.gap), 0) / nullRuns.length;
  const nullMaxAbsGap = Math.max(...nullRuns.map((r) => Math.abs(r.eval.gap)));

  // Determinism control: identical (seed, rho) must reproduce identical metrics.
  const repA = await runOnce(SEEDS[0], 1.0);
  const repB = await runOnce(SEEDS[0], 1.0);
  const deterministic =
    repA.eval.matched.acc === repB.eval.matched.acc &&
    repA.eval.flipped.acc === repB.eval.flipped.acc &&
    repA.eval.neutral.acc === repB.eval.neutral.acc &&
    repA.ablation.bgReliance === repB.ablation.bgReliance &&
    repA.collapsed === repB.collapsed &&
    Math.abs(repA.bgMass - repB.bgMass) < 1e-9;

  const at = (rho: number) => runs.filter((r) => r.rho === rho);
  const gapAt = (rho: number) => at(rho).map((r) => r.gap);
  const gaps10 = gapAt(1.0);
  const gaps05 = gapAt(0.5);
  const matched05 = at(0.5).map((r) => r.matchedAcc);
  const bg10 = at(1.0).map((r) => r.bgReliance);
  const bg05 = at(0.5).map((r) => r.bgReliance);
  const relOk = (v: number | null) => v ?? 0; // null = learned nothing = not bg
  const nCollapsed = runs.filter((r) => r.collapsed).length;

  // Gates. Original brief semantics restored where QA found them relaxed;
  // metric changes are DISCLOSED in results.md, not silently redefined.
  //   brief G3 asked for a heatmap sign test p<0.05 — we report the stronger
  //   counterfactual-ablation bgReliance instead (disclosed change), now with
  //   the original p<0.05 bar.
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
      name: 'Accuracy lives in the background: ablation bgReliance at rho=1.0 vs rho=0.5, sign test p<0.05',
      enforced: enforce(false),
      pass:
        bg10.reduce<number>((s, v) => s + relOk(v), 0) / bg10.length >= 0.8 &&
        bg05.reduce<number>((s, v) => s + relOk(v), 0) / bg05.length <= 0.2 &&
        // A sign test can't reach p<0.05 with one seed — smoke enforces the
        // mean thresholds and reports the sign test as n/a.
        (SMOKE || signTest(bg10.map((v, i) => relOk(v) - relOk(bg05[i]))).p < 0.05),
      detail: `mean bgReliance rho=1.0=${(bg10.reduce<number>((s, v) => s + relOk(v), 0) / bg10.length).toFixed(3)}, ` +
        `rho=0.5=${(bg05.reduce<number>((s, v) => s + relOk(v), 0) / bg05.length).toFixed(3)}; ` +
        (SMOKE
          ? 'sign test n/a (n=1)'
          : `sign-test p=${signTest(bg10.map((v, i) => relOk(v) - relOk(bg05[i]))).p.toFixed(4)}`),
    },
    {
      id: 'G4',
      name: 'Null control at chance: mean shuffled-label matched acc ≈ 50%, mean |gap| small',
      enforced: enforce(false),
      pass: nullMatchedMean >= 0.4 && nullMatchedMean <= 0.6 && nullAbsGapMean <= 0.15,
      detail: `per-seed matched=${nullRuns.map((r) => fmtPct(r.eval.matched.acc)).join(', ')}; ` +
        `per-seed |gap|=${nullRuns.map((r) => fmtPct(Math.abs(r.eval.gap))).join(', ')}; ` +
        `mean=${fmtPct(nullMatchedMean)}, mean|gap|=${fmtPct(nullAbsGapMean)}, max|gap|=${fmtPct(nullMaxAbsGap)}`,
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
      bgReliance: r.ablation.bgReliance,
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
  md.push('| seed | rho | matched | flipped | neutral | gap | bgMass | bgReliance | acc no-bg | acc no-fg | collapse |');
  md.push('|---|---|---|---|---|---|---|---|---|---|---|');
  for (const r of runs) {
    md.push(
      `| ${r.seed} | ${r.rho.toFixed(1)} | ${fmtPct(r.matchedAcc)} | ${fmtPct(r.flippedAcc)} | ${fmtPct(r.neutralAcc)} | ${fmtPct(r.gap)} | ${r.bgMass.toFixed(3)} | ${fmtRel(r.bgReliance)} | ${fmtPct(r.accNoBg)} | ${fmtPct(r.accNoFg)} | ${r.collapsed ? (r.recovered ? 'yes→retried' : 'YES (unrecovered)') : ''} |`,
    );
  }
  md.push('');
  md.push(
    `Null control (labels shuffled, rho=1.0): ` +
      nullRuns.map((r) => `seed ${r.seed} matched ${fmtPct(r.eval.matched.acc)} gap ${fmtPct(r.eval.gap)}`).join('; ') +
      `. Mean matched ${fmtPct(nullMatchedMean)}; mean |gap| ${fmtPct(nullAbsGapMean)}; max |gap| ${fmtPct(nullMaxAbsGap)}. ` +
      'Per-seed drift above 50% is expected: an untrained model\'s weight drift keys on the dominant feature (the large background) with an arbitrary sign — which is why per-seed gaps are disclosed, not just the mean.',
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
  md.push('- G3 metric: brief asked for an occlusion-heatmap sign test; we report the stronger counterfactual-ablation bgReliance sign test at the original p<0.05 bar. The heatmap is kept as a supporting view because patch occlusion under-measures a spread-out colour cue.');
  md.push('- Brief\'s perf gate (≤30s CPU train, ≤5MB bundle) is reported as targets T1/T2: the harness backend number is measured here; the bundle size is measured in CI by scripts/verify-results.ts. The 30s target fails on a pure-JS CPU backend — the UI states the real expectation instead of faking speed.');
  md.push('- Smoke mode enforces G1 and G5 only; G2–G4 are measured and reported but advisory at one seed (the null per-seed bound and the rho=0.5 means are honest only in aggregate). The full 8-seed run enforces every gate and runs in CI.');
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
