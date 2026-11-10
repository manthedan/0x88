import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

const helper = new URL('./lib/required_assets.mjs', import.meta.url).href;

function runGuardedTest(t, paths, env) {
  const dir = mkdtempSync(join(tmpdir(), 'required-assets-'));
  t.after(() => rmSync(dir, { recursive: true, force: true }));
  const file = join(dir, 'guarded.test.mjs');
  writeFileSync(
    file,
    `import test from 'node:test';\nimport { skipUnlessAssets } from ${JSON.stringify(helper)};\ntest('needs assets', { skip: skipUnlessAssets(...${JSON.stringify(paths)}) }, () => {});\n`,
  );
  const { CI: _ci, NODE_TEST_CONTEXT: _context, ...base } = process.env;
  const result = spawnSync(process.execPath, ['--test', '--test-reporter=tap', file], { cwd: dir, encoding: 'utf8', env: { ...base, ...env } });
  const count = (name) => Number(result.stdout.match(new RegExp(`^# ${name} (\\d+)$`, 'm'))?.[1]);
  return { status: result.status, pass: count('pass'), fail: count('fail'), skipped: count('skipped'), stdout: result.stdout };
}

test('a missing asset skips locally', (t) => {
  const run = runGuardedTest(t, ['../models/lc0-bestnets/onnx/nope.onnx'], {});
  assert.deepEqual([run.status, run.fail, run.skipped], [0, 0, 1], run.stdout);
});

test('a missing asset fails under CI instead of passing as a skip', (t) => {
  const run = runGuardedTest(t, ['../models/lc0-bestnets/onnx/nope.onnx'], { CI: 'true' });
  assert.deepEqual([run.status, run.fail, run.skipped], [1, 1, 1], run.stdout);
  assert.match(run.stdout, /CI has the test assets: .*nope\.onnx/);
});

test('allowlisted assets CI cannot have still skip under CI', (t) => {
  const run = runGuardedTest(t, ['../../repos/lc0/src/neural/encoder.cc', '.venv-onnx/bin/python'], { CI: 'true' });
  assert.deepEqual([run.status, run.fail, run.skipped], [0, 0, 1], run.stdout);
});

test('present assets run the test', (t) => {
  const run = runGuardedTest(t, ['guarded.test.mjs'], { CI: 'true' });
  assert.deepEqual([run.status, run.pass, run.skipped], [0, 1, 0], run.stdout);
});
