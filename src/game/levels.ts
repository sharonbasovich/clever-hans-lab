// Level definitions. Level 3's win threshold is imported from the measured
// harness output (src re-exports results/results.json) — never hard-coded.
import resultsData from '../../results/results.json';

export interface Level {
  id: 1 | 2 | 3;
  name: string;
  tagline: string;
}

export const LEVELS: Level[] = [
  { id: 1, name: 'Meet Hans', tagline: 'Plant a perfect cheat and watch it work.' },
  { id: 2, name: 'Catch the cheat', tagline: 'Read the heatmap: where did it look?' },
  { id: 3, name: 'Fix the data', tagline: 'Redesign the training set so it cannot cheat.' },
];

export const L1 = {
  rho: 1.0,
  trainN: 1200,
  testN: 300,
  epochs: 8,
  batch: 64,
};

// Level 3: the player gets a fixed sample budget and sets rho + neutral mix.
export const L3 = {
  budget: 1600,
  testN: 300,
  epochs: 8,
  batch: 64,
  rhoMin: 0.5,
  rhoMax: 1.0,
  rhoStep: 0.05,
  neutralMax: 0.6,
};

// Derived in scripts/harness.ts from measured rho=0.5 flipped accuracy.
export const L3_FLIPPED_THRESHOLD: number = resultsData.derived.l3FlippedThreshold;
export const RESULTS = resultsData;
