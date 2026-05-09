import test from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';

test('PUCT core invariants with mock evaluator', () => {
  const output = execFileSync('node', ['--experimental-strip-types', 'eval/puct_core_tests.mjs'], { encoding: 'utf8' });
  assert.match(output, /METRIC puct_core_tests=9/);
});
