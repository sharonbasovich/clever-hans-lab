import * as tf from '@tensorflow/tfjs';
import { buildModel } from '../model.ts';
import { trainModel, type TrainHistory, type TrainOptions } from '../train.ts';
import { evalSets, type EvalSuite } from '../evaluate.ts';
import { meanBgMass } from '../explain.ts';
import { generateWorld, mixWorlds, type World } from '../world.ts';
import { L1, L3 } from './levels.ts';

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
  l3Design: { rho: number; neutral: number } | null = null;

  /** Demo mode shrinks the world so the recorded run stays under 2 minutes —
   *  the training is still real; only n is smaller. */
  constructor(public demo = false) {}

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
   *  inside a fixed budget. */
  buildLevel3World(rho: number, neutralFrac: number): World {
    const budget = this.demo ? Math.round(L3.budget / 2.5) : L3.budget;
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
    this.testSets = this.makeTestSets(rho);
    this.evaluation = null;
    this.bgMass = null;
    return mixed;
  }

  private makeTestSets(rho: number): TestSets {
    const n = this.testN();
    return {
      matched: generateWorld({ seed: this.seed + 1001, n, rho, split: 'matched', accessible: this.accessible }),
      flipped: generateWorld({ seed: this.seed + 1002, n, rho, split: 'flipped', accessible: this.accessible }),
      neutral: generateWorld({ seed: this.seed + 1003, n, rho, split: 'neutral', accessible: this.accessible }),
    };
  }

  async train(opts: Partial<TrainOptions> = {}): Promise<TrainHistory> {
    if (!this.trainWorld) throw new Error('build a world first');
    if (this.model) this.model.dispose();
    this.model = buildModel(this.seed);
    this.history = await trainModel(this.model, this.trainWorld, {
      epochs: this.epochs(),
      batchSize: 64,
      learningRate: 0.01,
      seed: this.seed,
      ...opts,
    });
    return this.history;
  }

  evaluate(): EvalSuite {
    if (!this.model || !this.testSets) throw new Error('train first');
    this.evaluation = evalSets(this.model, this.testSets);
    return this.evaluation;
  }

  /** Mean occlusion mass on background pixels over a slice of matched tests. */
  explain(k = this.demo ? 12 : 24): number {
    if (!this.model || !this.testSets) throw new Error('train first');
    this.bgMass = meanBgMass(this.model, this.testSets.matched, k);
    return this.bgMass;
  }
}
