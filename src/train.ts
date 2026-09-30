import * as tf from '@tensorflow/tfjs';
import { mulberry32 } from './rng.ts';
import { IMG_PIXELS, NUM_CLASSES, type World } from './world.ts';

export interface TrainOptions {
  epochs?: number;
  batchSize?: number;
  learningRate?: number;
  /** Seed controlling batch order. */
  seed: number;
  /** Called after each epoch with (epochIndex, meanLoss, meanAccuracy). */
  onEpoch?: (epoch: number, loss: number, acc: number) => void | Promise<void>;
  /** Called after each batch for progress UI. */
  onBatch?: (batch: number, totalBatches: number) => void;
  signal?: { cancelled: boolean };
}

export interface TrainHistory {
  losses: number[];
  accs: number[];
}

// Manual fit loop: batch shuffling is driven by our seeded RNG rather than
// tfjs's internal RNG so a (seed, rho, seed) triple fully determines the run.
export async function trainModel(
  model: tf.LayersModel,
  world: World,
  opts: TrainOptions,
): Promise<TrainHistory> {
  const { epochs = 8, batchSize = 64, learningRate = 0.01, seed, onEpoch, onBatch, signal } = opts;
  const optimizer = tf.train.adam(learningRate);
  const rng = mulberry32(seed ^ 0xba7c);
  const n = world.n;
  const history: TrainHistory = { losses: [], accs: [] };
  const order = new Uint32Array(n).map((_, i) => i);
  const totalBatches = Math.ceil(n / batchSize) * epochs;
  let batchIdx = 0;

  const xsBuf = new Float32Array(batchSize * IMG_PIXELS * 3);
  const ysBuf = new Float32Array(batchSize * NUM_CLASSES);

  for (let epoch = 0; epoch < epochs; epoch++) {
    // Fisher-Yates with our RNG.
    for (let i = n - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      const t = order[i];
      order[i] = order[j];
      order[j] = t;
    }
    let lossSum = 0;
    let accSum = 0;
    let seen = 0;
    for (let b = 0; b < n; b += batchSize) {
      if (signal?.cancelled) return history;
      const bn = Math.min(batchSize, n - b);
      for (let k = 0; k < bn; k++) {
        const src = order[b + k];
        xsBuf.set(
          world.images.subarray(src * IMG_PIXELS * 3, (src + 1) * IMG_PIXELS * 3),
          k * IMG_PIXELS * 3,
        );
        ysBuf.fill(0, k * NUM_CLASSES, (k + 1) * NUM_CLASSES);
        ysBuf[k * NUM_CLASSES + world.labels[src]] = 1;
      }
      if (bn < batchSize) {
        xsBuf.fill(0, bn * IMG_PIXELS * 3);
        ysBuf.fill(0, bn * NUM_CLASSES);
      }
      const xs = tf.tensor4d(xsBuf, [batchSize, 32, 32, 3]);
      const ys = tf.tensor2d(ysBuf, [batchSize, NUM_CLASSES]);
      // One forward pass: reuse the predictions computed inside the cost
      // function for the accuracy readout instead of predicting twice.
      const predsRef: { t: tf.Tensor | null } = { t: null };
      const loss = optimizer.minimize(() => {
        const p = model.apply(xs) as tf.Tensor;
        // keep(): optimizer.minimize runs the cost fn inside a tidy — without
        // this the predictions would be disposed before we read them.
        predsRef.t = tf.keep(p);
        return tf.losses.softmaxCrossEntropy(ys, p);
      }, true) as tf.Scalar;
      const preds = predsRef.t!;
      const acc = tf.tidy(() => preds.argMax(1).equal(ys.argMax(1)).mean()) as tf.Scalar;
      const [lossVal, accVal] = await Promise.all([loss.data(), acc.data()]).then(
        ([l, a]) => [l[0], a[0]],
      );
      xs.dispose();
      ys.dispose();
      loss.dispose();
      acc.dispose();
      preds.dispose();
      lossSum += lossVal * bn;
      accSum += accVal * bn;
      seen += bn;
      batchIdx++;
      onBatch?.(batchIdx, totalBatches);
      // Yield so the UI can repaint between batches in the browser.
      await new Promise((r) => setTimeout(r, 0));
    }
    history.losses.push(lossSum / seen);
    history.accs.push(accSum / seen);
    await onEpoch?.(epoch, lossSum / seen, accSum / seen);
  }
  return history;
}
