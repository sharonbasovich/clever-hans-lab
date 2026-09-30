// Headless evaluation harness — the project's ground truth.
//
//   npm run harness        full run: 5 seeds x rho {1.0, 0.9, 0.5} + controls
//   npm run harness:smoke  CI smoke: 1 seed, small n
//
// Writes results/results.json and results/results.md. Every number the UI,
// README and Devpost draft quotes comes from these files — nothing is
// hard-coded before measurement.
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
import { evalSets, type EvalSuite } from '../src/evaluate.ts';
import { ablationReliance, meanBgMass, type Ablation } from '../src/explain.ts';
import { bootstrapCI, signTest } from '../src/stats.ts';
import { generateWorld } from '../src/world.ts';

const HERE = dirname(fileURLToPath(import.meta.url));
const OUT_DIR = join(HERE, '..', 'results');

const SMOKE = process.argv.includes('--smoke');
const SEEDS = SMOKE ? [101] : [101, 202, 303, 404, 505];
const RHOS = [1.0, 0.9, 0.5];
const TRAIN_N = SMOKE ? 384 : 1200;
const TEST_N = SMOKE ? 160 : 400;
const EPOCHS = SMOKE ? 3 : 8;
const BATCH = 64;
const BGM_K = 24; // images per world sampled for occlusion mass

interface RunResult {
  seed: number;
  rho: number;
  matchedAcc: number;
  flippedAcc: number;
  neutralAcc: number;
  gap: number;
  bgMass: number;
  bgReliance: number;
  accNoBg: number;
  accNoFg: number;
  trainLossFinal: number;
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
  wallMs: number;
}

async function runOnce(seed: number, rho: number, shuffleLabels = false): Promise<TrainedRun> {
  const t0 = Date.now();
  const train = generateWorld({ seed, n: TRAIN_N, rho, shuffleLabels });
  const matched = generateWorld({ seed: seed + 1, n: TEST_N, rho, split: 'matched' });
  const flipped = generateWorld({ seed: seed + 2, n: TEST_N, rho, split: 'flipped' });
  const neutral = generateWorld({ seed: seed + 3, n: TEST_N, rho, split: 'neutral' });
  const model = buildModel(seed);
  const hist = await trainModel(model, train, {
    epochs: EPOCHS,
    batchSize: BATCH,
    learningRate: 0.01,
    seed,
  });
  const eval_ = evalSets(model, { matched, flipped, neutral });
  const bg = meanBgMass(model, matched, BGM_K);
  const abl = ablationReliance(model, matched, 128);
  model.dispose();
  return {
    eval: eval_,
    bgMass: bg,
    ablation: abl,
    trainLossFinal: hist.losses.at(-1) ?? NaN,
    wallMs: Date.now() - t0,
  };
}

function fmtPct(x: number): string {
  return (x * 100).toFixed(1) + '%';
}

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
        wallMs: r.wallMs,
      });
      console.log(
        `seed=${seed} rho=${rho.toFixed(1)} matched=${fmtPct(r.eval.matched.acc)} ` +
          `flipped=${fmtPct(r.eval.flipped.acc)} neutral=${fmtPct(r.eval.neutral.acc)} ` +
          `gap=${fmtPct(r.eval.gap)} bgMass=${r.bgMass.toFixed(3)} bgRel=${r.ablation.bgReliance.toFixed(3)} (${r.wallMs}ms)`,
      );
    }
  }

  // Null control: labels shuffled -> nothing to learn. Run on every seed:
  // a single-seed null can drift above 60% because an untrained model's
  // weight drift keys on the dominant feature (the big background) with an
  // arbitrary sign — the mean across seeds is the honest measurement, and
  // every seed is reported below.
  const nullRuns: { seed: number; eval: EvalSuite; ablation: Ablation }[] = [];
  for (const seed of SEEDS) {
    const r = await runOnce(seed, 1.0, true);
    nullRuns.push({ seed, eval: r.eval, ablation: r.ablation });
    console.log(
      `null-control seed=${seed} matched=${fmtPct(r.eval.matched.acc)} flipped=${fmtPct(
        r.eval.flipped.acc,
      )} neutral=${fmtPct(r.eval.neutral.acc)}`,
    );
  }
  const nullMatchedMean =
    nullRuns.reduce((s, r) => s + r.eval.matched.acc, 0) / nullRuns.length;

  // Determinism control: identical (seed, rho) must reproduce identical metrics.
  const repA = await runOnce(SEEDS[0], 1.0);
  const repB = await runOnce(SEEDS[0], 1.0);
  const deterministic =
    repA.eval.matched.acc === repB.eval.matched.acc &&
    repA.eval.flipped.acc === repB.eval.flipped.acc &&
    repA.eval.neutral.acc === repB.eval.neutral.acc &&
    repA.ablation.bgReliance === repB.ablation.bgReliance &&
    Math.abs(repA.bgMass - repB.bgMass) < 1e-9;

  const at = (rho: number) => runs.filter((r) => r.rho === rho);
  const gapAt = (rho: number) => at(rho).map((r) => r.gap);
  const gaps10 = gapAt(1.0);
  const gaps05 = gapAt(0.5);
  const matched05 = at(0.5).map((r) => r.matchedAcc);
  const bg10 = at(1.0).map((r) => r.bgReliance);
  const bg05 = at(0.5).map((r) => r.bgReliance);

  // Gates. These are the honest go/no-go measurements — no seed is dropped,
  // and a failed gate is reported rather than tuned around.
  const gates = [
    {
      id: 'G1',
      name: 'Hans cheats: at rho=1.0 the shortcut gap is large in every seed',
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
      pass:
        Math.abs(gaps05.reduce((s, v) => s + v, 0) / gaps05.length) <= 0.1 &&
        matched05.reduce((s, v) => s + v, 0) / matched05.length >= 0.8,
      detail: `mean|gap|=${fmtPct(
        Math.abs(gaps05.reduce((s, v) => s + v, 0) / gaps05.length),
      )}; mean matched=${fmtPct(matched05.reduce((s, v) => s + v, 0) / matched05.length)}`,
    },
    {
      id: 'G3',
      name: 'Accuracy lives in the background: ablation bgReliance at rho=1.0 vs rho=0.5',
      pass:
        bg10.reduce((s, v) => s + v, 0) / bg10.length >= 0.8 &&
        signTest(bg10.map((v, i) => v - bg05[i])).p < 0.25,
      detail: `mean bgReliance rho=1.0=${(bg10.reduce((s, v) => s + v, 0) / bg10.length).toFixed(3)}, ` +
        `rho=0.5=${(bg05.reduce((s, v) => s + v, 0) / bg05.length).toFixed(3)}; ` +
        `sign-test p=${signTest(bg10.map((v, i) => v - bg05[i])).p.toFixed(3)}`,
    },
    {
      id: 'G4',
      name: 'Null control at chance: mean shuffled-label matched acc ≈ 50% across seeds',
      pass: nullMatchedMean >= 0.4 && nullMatchedMean <= 0.6,
      detail: `per-seed matched=${nullRuns.map((r) => fmtPct(r.eval.matched.acc)).join(', ')}; ` +
        `mean=${fmtPct(nullMatchedMean)}`,
    },
    {
      id: 'G5',
      name: 'Reproducible: identical (seed, rho) reproduces identical metrics',
      pass: deterministic,
      detail: `repeat matched=${fmtPct(repB.eval.matched.acc)} vs first=${fmtPct(
        repA.eval.matched.acc,
      )}`,
    },
  ];

  // Level-3 win threshold, DERIVED from measurement (never tuned by hand):
  // flipped accuracy a debiased model reaches at rho=0.5, minus 1.5 sd,
  // clamped to [0.70, 0.95].
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
    bgRelianceCI: bootstrapCI(at(rho).map((r) => r.bgReliance), 2000, 47),
  }));

  const results = {
    generatedAt: new Date().toISOString(),
    backend: tf.getBackend(),
    tfVersion: tf.version_core,
    config: { seeds: SEEDS, rhos: RHOS, trainN: TRAIN_N, testN: TEST_N, epochs: EPOCHS, batch: BATCH, bgmK: BGM_K },
    runs,
    nullControl: nullRuns.map((r) => ({
      seed: r.seed,
      matchedAcc: r.eval.matched.acc,
      flippedAcc: r.eval.flipped.acc,
      neutralAcc: r.eval.neutral.acc,
      bgReliance: r.ablation.bgReliance,
    })),
    determinism: { deterministic },
    perRho,
    gates,
    derived: { l3FlippedThreshold: l3Threshold },
    wallMsTotal: Date.now() - t0,
  };

  mkdirSync(OUT_DIR, { recursive: true });
  // Smoke runs (CI plumbing check) write to a side file so they can never
  // clobber the committed full-run results.
  const outName = SMOKE ? 'results.smoke.json' : 'results.json';
  writeFileSync(join(OUT_DIR, outName), JSON.stringify(results, null, 2));

  const md: string[] = [];
  md.push('# Clever Hans Lab — measured results', '');
  md.push(`Generated ${results.generatedAt} — tfjs ${tf.version_core}, backend ${tf.getBackend()}, wall ${(results.wallMsTotal / 1000).toFixed(0)}s.`);
  md.push(`Config: seeds ${SEEDS.join('/')}, train n=${TRAIN_N}, test n=${TEST_N}/split, ${EPOCHS} epochs, batch ${BATCH}.`, '');
  md.push('| seed | rho | matched | flipped | neutral | gap | bgMass | bgReliance | acc no-bg | acc no-fg |');
  md.push('|---|---|---|---|---|---|---|---|---|---|');
  for (const r of runs) {
    md.push(
      `| ${r.seed} | ${r.rho.toFixed(1)} | ${fmtPct(r.matchedAcc)} | ${fmtPct(r.flippedAcc)} | ${fmtPct(r.neutralAcc)} | ${fmtPct(r.gap)} | ${r.bgMass.toFixed(3)} | ${r.bgReliance.toFixed(3)} | ${fmtPct(r.accNoBg)} | ${fmtPct(r.accNoFg)} |`,
    );
  }
  md.push('');
  md.push(
    `Null control (labels shuffled, rho=1.0): ` +
      nullRuns.map((r) => `seed ${r.seed} matched ${fmtPct(r.eval.matched.acc)}`).join('; ') +
      `. Mean matched ${fmtPct(nullMatchedMean)}. Per-seed drift above 50% is expected: an untrained model's weight drift keys on the dominant feature (the large background) with an arbitrary sign.`,
  );
  md.push('');
  md.push('## Gates');
  for (const g of gates) md.push(`- **${g.id} ${g.pass ? 'PASS' : 'FAIL'}** — ${g.name}. ${g.detail}`);
  md.push('');
  md.push(`Derived L3 win threshold (flipped acc): ${(l3Threshold * 100).toFixed(1)}%`);
  if (!SMOKE) writeFileSync(join(OUT_DIR, 'results.md'), md.join('\n'));

  const failed = gates.filter((g) => !g.pass);
  console.log('');
  for (const g of gates) console.log(`${g.id} ${g.pass ? 'PASS' : 'FAIL'} — ${g.name}`);
  console.log(`Wrote ${OUT_DIR}/${outName}${SMOKE ? '' : ' + results.md'} in ${(results.wallMsTotal / 1000).toFixed(0)}s`);
  if (failed.length > 0 && !SMOKE) {
    console.log(`Gates failed: ${failed.map((g) => g.id).join(', ')} — report honestly, do not tune by dropping seeds.`);
    process.exit(2);
  }
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
