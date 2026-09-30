// Post-harness verification for CI: the committed results/results.json must
// have every gate passing, a consistent schema, and the built bundle must be
// under the 5 MB target. Any failure exits non-zero — CI cannot go green on
// failed gates.
import { readFileSync, readdirSync, statSync } from 'node:fs';
import { join } from 'node:path';

const resultsPath = 'results/results.json';
const R = JSON.parse(readFileSync(resultsPath, 'utf8'));

let failed = false;
const check = (ok: boolean, label: string) => {
  console.log(`${ok ? 'PASS' : 'FAIL'} — ${label}`);
  if (!ok) failed = true;
};

// 1. Every science gate must pass — a committed file with failed gates is
//    itself a CI failure.
check(Array.isArray(R.gates) && R.gates.length >= 5, 'gates present (>=5)');
for (const g of R.gates ?? []) check(g.pass === true, `gate ${g.id} passed: ${g.name}`);

// 2. Schema/scope sanity: all declared seeds and rhos actually measured, and
//    no seed silently dropped.
check(
  R.config.seeds.length >= 8 && R.runs.length === R.config.seeds.length * R.config.rhos.length,
  `full grid measured (${R.config.seeds.length} seeds x ${R.config.rhos.length} rhos = ${R.runs.length} runs)`,
);
check(R.nullControl?.length === R.config.seeds.length, 'null control ran on every seed');
check(
  R.runs.every((r: { collapsed?: boolean; recovered?: boolean }) => !r.collapsed || r.recovered !== undefined),
  'collapse fields present on every run',
);
check(typeof R.derived?.l3FlippedThreshold === 'number', 'L3 threshold derived and present');
check(R.determinism?.deterministic === true, 'determinism confirmed in-file');

// 3. Bundle target (brief: ≤5 MB). Measure the built dist.
try {
  const assetsDir = 'dist/assets';
  const files = readdirSync(assetsDir);
  const jsBytes = files
    .filter((f) => f.endsWith('.js'))
    .reduce((s, f) => s + statSync(join(assetsDir, f)).size, 0);
  const totalBytes = files.reduce((s, f) => s + statSync(join(assetsDir, f)).size, 0);
  console.log(`dist assets: JS ${(jsBytes / 1e6).toFixed(2)} MB, total ${(totalBytes / 1e6).toFixed(2)} MB`);
  check(jsBytes <= 5e6, `JS bundle <=5MB (measured ${(jsBytes / 1e6).toFixed(2)} MB)`);
} catch {
  console.log('dist/ not found — bundle target unchecked (run npm run build first)');
  check(false, 'dist/ present for bundle check');
}

if (failed) {
  console.log('verify-results: FAIL');
  process.exit(1);
}
console.log('verify-results: all checks pass');
