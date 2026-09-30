import * as tf from '@tensorflow/tfjs';
import { IMG_SIZE, NUM_CLASSES } from './world.ts';

// A deliberately small CNN: big enough to learn shape-or-shortcut, small
// enough to train in a browser tab in seconds.
//
//   conv(8, 3x3) -> maxpool -> conv(16, 3x3) -> maxpool -> dense(32) -> dense(2)
export function buildModel(seed: number): tf.LayersModel {
  const model = tf.sequential();
  model.add(
    tf.layers.conv2d({
      inputShape: [IMG_SIZE, IMG_SIZE, 3],
      filters: 8,
      kernelSize: 3,
      activation: 'relu',
      kernelInitializer: tf.initializers.glorotNormal({ seed: seed * 7 + 1 }),
      biasInitializer: 'zeros',
    }),
  );
  model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
  model.add(
    tf.layers.conv2d({
      filters: 16,
      kernelSize: 3,
      activation: 'relu',
      kernelInitializer: tf.initializers.glorotNormal({ seed: seed * 7 + 2 }),
      biasInitializer: 'zeros',
    }),
  );
  model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
  model.add(tf.layers.flatten());
  model.add(
    tf.layers.dense({
      units: 32,
      activation: 'relu',
      kernelInitializer: tf.initializers.glorotNormal({ seed: seed * 7 + 3 }),
      biasInitializer: 'zeros',
    }),
  );
  model.add(
    tf.layers.dense({
      units: NUM_CLASSES,
      activation: 'softmax',
      kernelInitializer: tf.initializers.glorotNormal({ seed: seed * 7 + 4 }),
      biasInitializer: 'zeros',
    }),
  );
  return model;
}

export function modelParamCount(model: tf.LayersModel): number {
  return model.getWeights().reduce((s, w) => s + w.size, 0);
}
