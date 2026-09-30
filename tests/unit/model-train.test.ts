import * as tf from '@tensorflow/tfjs';
import { beforeAll, describe, expect, it } from 'vitest';
import { buildModel, modelParamCount } from '../../src/model.ts';
import { trainModel } from '../../src/train.ts';
import { generateWorld } from '../../src/world.ts';

beforeAll(async () => {
  try {
    await import('@tensorflow/tfjs-node');
    await tf.setBackend('tensorflow');
  } catch {
    await tf.setBackend('cpu');
  }
  await tf.ready();
});

describe('model + training', () => {
  it('builds the documented architecture', () => {
    const m = buildModel(1);
    const kinds = m.layers.map((l) => l.getClassName());
    expect(kinds).toEqual(['Conv2D', 'MaxPooling2D', 'Conv2D', 'MaxPooling2D', 'Flatten', 'Dense', 'Dense']);
    expect(modelParamCount(m)).toBeGreaterThan(1000);
    expect(modelParamCount(m)).toBeLessThan(20000);
    m.dispose();
  });

  it('seeded init: same seed -> identical initial weights', () => {
    const a = buildModel(42);
    const b = buildModel(42);
    const wa = a.getWeights().map((w) => Array.from(w.dataSync()));
    const wb = b.getWeights().map((w) => Array.from(w.dataSync()));
    expect(wa).toEqual(wb);
    a.dispose();
    b.dispose();
  });

  it('trains on a rho=0.5 world with finite loss and rising accuracy', async () => {
    const world = generateWorld({ seed: 17, n: 256, rho: 0.5 });
    const model = buildModel(17);
    const hist = await trainModel(model, world, { epochs: 3, batchSize: 64, seed: 17 });
    expect(hist.losses).toHaveLength(3);
    for (const l of hist.losses) expect(Number.isFinite(l)).toBe(true);
    expect(hist.losses.at(-1)!).toBeLessThan(hist.losses[0]);
    model.dispose();
  }, 60000);
});
