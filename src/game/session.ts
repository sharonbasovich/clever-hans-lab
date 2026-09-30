import * as tf from '@tensorflow/tfjs';
import { buildModel } from '../model.ts';
import { trainModel, type TrainHistory, type TrainOptions } from '../train.ts';
import { evalSets, isConstantPredictor, type EvalSuite } from '../evaluate.ts';
import { ablationReliance, meanBgMass, type Ablation } from '../explain.ts';
import { generateWorld, mixWorlds, type World } from '../world.ts';
import { L1, L3 } from './levels.ts';

// Adam lr 0.01 overshoots into a constant predictor on ~1 in 5 inits
// (measured on both the builder and QA seed sets); 0.005 eliminated every
// observed collapse. A collapse detector + one deterministic retry below
// stays as defence-in-depth for browser/backend variance.
export const LEARNING_RATE = 0.005;

export interface TestSets {
  matched: World;
  flipped: World;
  neutral: World;
}

// The whole play session: one model, one training world the player designed,
// three held-out test sets that reveal the shortcut.
export class Session {
  accessible = false;
  seed = 1;
  level: 1 | 2 | 3 = 1;

  trainWorld: World | null = null;
  testSets: TestSets | null = null;
  model: tf.LayersModel | null = null;
  history: TrainHistory | null = null;
  evaluation: EvalSuite | null = null;
  bgMass: number | null = null;
  ablation: Ablation | null = null;
  l3Design: { rho: number; neutral: number } | null = null;
  /** Set when a route needs to explain a recovery (e.g. reload lost the run). */
  notice: string | null = null;
  /** True while a training run owns the model. */
  trainingNow = false;
  /** The last run's first attempt collapsed to a constant predictor and was
   *  transparently reinitialized and retrained (deterministic seed). */
  retriedAfterCollapse = false;
  /** Even the retry collapsed — report "didn't learn", not "cheat works". */
  stillCollapsed = false;

  private runGen = 0;
  private cancelFlag: { cancelled: boolean } = { cancelled: false };

  /** Demo mode shrinks the world so the recorded run stays under 2 minutes —
   *  the training is still real; only n is smaller. */
  constructor(public demo = false) {}

  /** Cancel any in-flight training and invalidate its completion. Route-away
   *  calls this so a stale run can't hijack navigation later. */
  cancelTraining(): void {
    this.cancelFlag.cancelled = true;
    this.runGen++;
    this.trainingNow = false;
  }

  trainN(): number {
    return this.demo ? 480 : L1.trainN;
  }
  testN(): number {
    return this.demo ? 160 : L1.testN;
  }
  epochs(): number {
    return this.demo ? 5 : L1.epochs;
  }

  async ensureBackend() {
    try {
      await tf.setBackend('webgl');
      await tf.ready();
    } catch {
      await tf.setBackend('cpu');
      await tf.ready();
    }
  }

  /** Level 1 world: rho=1.0 — every circle on red, every triangle on blue. */
  buildLevel1World(): void {
    const n = this.trainN();
    this.trainWorld = generateWorld({
      seed: this.seed,
      n,
      rho: L1.rho,
      accessible: this.accessible,
    });
    this.testSets = this.makeTestSets(L1.rho);
    this.evaluation = null;
    this.bgMass = null;
  }

  /** Level 3: player-designed world — rho plus a fraction of neutral samples
   *  inside a fixed budget. L3 uses the REAL budget even in demo mode: the
   *  win threshold was measured at this scale, so shrinking it would make the
   *  game unwinnable — demo mode shrinks Level 1 only. */
  buildLevel3World(rho: number, neutralFrac: number): World {
    const budget = L3.budget;
    const nNeutral = Math.round(budget * neutralFrac);
    const nCued = budget - nNeutral;
    const cued = generateWorld({
      seed: this.seed,
      n: nCued,
      rho,
      accessible: this.accessible,
    });
    const neutral = generateWorld({
      seed: this.seed + 77,
      n: nNeutral,
      rho: 0,
      split: 'neutral',
      accessible: this.accessible,
    });
    const mixed = mixWorlds([cued, neutral], this.seed + 5);
    this.l3Design = { rho, neutral: neutralFrac };
    this.trainWorld = mixed;
    // L3 is judged on a strict full reversal (rho=1 flipped): the cheat
    // disagrees with the label on every image, not just with prob 1-rho.
    this.testSets = this.makeTestSets(rho, true);
    this.evaluation = null;
    this.bgMass = null;
    return mixed;
  }

  private makeTestSets(rho: number, strictFlipped = false): TestSets {
    const n = this.testN();
    return {
      matched: generateWorld({ seed: this.seed + 1001, n, rho, split: 'matched', accessible: this.accessible }),
      flipped: generateWorld({
        seed: this.seed + 1002,
        n,
        rho: strictFlipped ? 1.0 : rho,
        split: 'flipped',
        accessible: this.accessible,
      }),
      neutral: generateWorld({ seed: this.seed + 1003, n, rho, split: 'neutral', accessible: this.accessible }),
    };
  }

  /** Train once, with collapse detection and at most one deterministic
   *  reinit-and-retry. Returns null when the run was cancelled or superseded
   *  — callers must NOT navigate or mutate UI on a null result. */
  async train(opts: Partial<TrainOptions> = {}): Promise<TrainHistory | null> {
    if (!this.trainWorld) throw new Error('build a world first');
    const gen = ++this.runGen;
    this.cancelFlag.cancelled = true;
    const flag = { cancelled: false };
    this.cancelFlag = flag;
    this.trainingNow = true;
    this.retriedAfterCollapse = false;
    this.stillCollapsed = false;
    try {
      for (let attempt = 0; attempt < 2; attempt++) {
        if (attempt > 0) this.retriedAfterCollapse = true;
        this.model?.dispose();
        this.model = buildModel(this.seed + attempt * 7919);
        this.history = await trainModel(this.model, this.trainWorld, {
          epochs: this.epochs(),
          batchSize: 64,
          learningRate: LEARNING_RATE,
          seed: this.seed + attempt * 7919,
          signal: flag,
          ...opts,
        });
        if (flag.cancelled || gen !== this.runGen) return null;
        if (!isConstantPredictor(this.model, this.trainWorld)) break;
        if (attempt === 1) this.stillCollapsed = true;
      }
      return this.history;
    } finally {
      this.trainingNow = false;
    }
  }

  evaluate(): EvalSuite {
    if (!this.model || !this.testSets) throw new Error('train first');
    this.evaluation = evalSets(this.model, this.testSets);
    return this.evaluation;
  }

  /** Mean occlusion mass + counterfactual ablation on matched tests. */
  explain(k = this.demo ? 12 : 24): number {
    if (!this.model || !this.testSets) throw new Error('train first');
    this.bgMass = meanBgMass(this.model, this.testSets.matched, k);
    this.ablation = ablationReliance(this.model, this.testSets.matched, this.demo ? 64 : 200);
    return this.bgMass;
  }
}
