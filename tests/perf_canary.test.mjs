import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { mkdtempSync, rmSync, utimesSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';
import { compareMetric, mad, median } from '../scripts/lib/perf_canary_stats.mjs';
import { previousCanary } from '../scripts/perf_canary.mjs';

const repoRoot = fileURLToPath(new URL('..', import.meta.url));

test('median and MAD use the central sample, not the mean', () => {
  assert.equal(median([3, 1, 2]), 2);
  assert.equal(median([1, 2, 3, 4]), 2.5);
  assert.equal(mad([100, 101, 99]), 1);
});

test('noise-aware comparison ignores swings inside the floor and flags a tight regression', () => {
  const stable = compareMetric({ name: 'nps', previous: 100, samples: [100, 101, 99], higherIsBetter: true });
  assert.equal(stable.status, 'within-noise');

  const regressed = compareMetric({ name: 'nps', previous: 120, samples: [100, 101, 99], higherIsBetter: true });
  assert.equal(regressed.status, 'regressed');
  assert.ok(regressed.deltaRatio < -0.1);

  const improved = compareMetric({ name: 'time', previous: 120, samples: [100, 101, 99], higherIsBetter: false });
  assert.equal(improved.status, 'improved');
});

test('a spread above the floor is a noisy run, not a regression', () => {
  const noisy = compareMetric({ name: 'nps', previous: 100, samples: [80, 100, 130], higherIsBetter: true });
  assert.equal(noisy.status, 'noisy-run');
  assert.ok(noisy.madRatio > 0.1);
});

test('fewer than three repeats cannot clear the noise check', () => {
  const once = compareMetric({ name: 'nps', previous: 50, samples: [100], higherIsBetter: true });
  assert.equal(once.status, 'insufficient-repeats');
});

test('canary dry-run plans the CPU harness and skips the browser fixture unless asked', () => {
  const result = spawnSync(process.execPath, ['scripts/perf_canary.mjs', '--dry-run'], { cwd: repoRoot, encoding: 'utf8' });
  assert.equal(result.status, 0, result.stderr);
  const plan = JSON.parse(result.stdout);
  assert.equal(plan.repeats, 3);
  assert.ok(plan.cpu.includes('scripts/emscripten_uci_bench.mjs'));
  assert.ok(plan.cpu.includes('public/berserk/berserk-emscripten.js'));
  assert.equal(plan.browser, null);

  const browser = spawnSync(process.execPath, ['scripts/perf_canary.mjs', '--dry-run', '--browser', '--browser-port', '5188'], {
    cwd: repoRoot,
    encoding: 'utf8',
  });
  assert.equal(browser.status, 0, browser.stderr);
  const browserPlan = JSON.parse(browser.stdout);
  assert.ok(browserPlan.browser.includes('scripts/lc0_browser_hybrid_input_bench.mjs'));
  assert.ok(browserPlan.browser.includes('5188'));
  assert.ok(browserPlan.browser.includes('wasm'));
});

test('the previous canary is the latest stamped file, not latest.json or a newer mtime', () => {
  const dir = mkdtempSync(path.join(tmpdir(), 'perf-canary-'));
  try {
    writeFileSync(path.join(dir, 'latest.json'), '{"cpu":{"aggregate":{"npsMedian":{"median":1}}}}\n');
    writeFileSync(path.join(dir, 'canary-notes.json'), '{"createdAt":"notes"}\n');
    const earlier = path.join(dir, 'canary-2026-10-11T08-59-33-383Z.json');
    writeFileSync(earlier, '{"createdAt":"first"}\n');
    writeFileSync(path.join(dir, 'canary-2026-10-11T08-59-35-667Z.json'), '{"createdAt":"second"}\n');
    const future = new Date('2030-01-01T00:00:00Z');
    utimesSync(earlier, future, future);
    const picked = previousCanary(dir);
    assert.equal(picked.name, 'canary-2026-10-11T08-59-35-667Z.json');
    assert.equal(picked.report.createdAt, 'second');
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});
