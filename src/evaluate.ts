import * as tf from '@tensorflow/tfjs';
import { IMG_PIXELS, IMG_SIZE, type World } from './world.ts';

export interface SetEval {
  split: World['split'];
  n: number;
  correct: number;
  acc: number;
}

// Predict labels for a whole world, in batches.
export function predictLabels(model: tf.LayersModel, world: World, batchSize = 128): Uint8Array {
  const out = new Uint8Array(world.n);
  tf.tidy(() => {
    for (let b = 0; b < world.n; b += batchSize) {
      const bn = Math.min(batchSize, world.n - b);
      const xs = tf.tensor4d(
        world.images.subarray(b * IMG_PIXELS * 3, (b + bn) * IMG_PIXELS * 3),
        [bn, IMG_SIZE, IMG_SIZE, 3],
      );
      const pred = (model.predict(xs) as tf.Tensor).argMax(1);
      out.set(Array.from(pred.dataSync() as Float32Array).map((v) => Math.round(v)), b);
      pred.dispose();
      xs.dispose();
    }
  });
  return out;
}

/** A constant predictor learned nothing — check whether the model emits more
 *  than one class across the world. */
export function isConstantPredictor(model: tf.LayersModel, world: World): boolean {
  const preds = predictLabels(model, world);
  return new Set(preds).size < 2;
}

export function evalWorld(model: tf.LayersModel, world: World): SetEval {
  const pred = predictLabels(model, world);
  let correct = 0;
  for (let i = 0; i < world.n; i++) if (pred[i] === world.labels[i]) correct++;
  return { split: world.split, n: world.n, correct, acc: correct / world.n };
}

export interface EvalSuite {
  matched: SetEval;
  flipped: SetEval;
  neutral: SetEval;
  /** matched - flipped: the "shortcut gap" this lab exists to expose. */
  gap: number;
}

export function evalSets(
  model: tf.LayersModel,
  sets: { matched: World; flipped: World; neutral: World },
): EvalSuite {
  const matched = evalWorld(model, sets.matched);
  const flipped = evalWorld(model, sets.flipped);
  const neutral = evalWorld(model, sets.neutral);
  return { matched, flipped, neutral, gap: matched.acc - flipped.acc };
}
