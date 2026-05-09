import test from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';

test('ChessOcean micro-environment invariants', () => {
  const output = execFileSync('node', ['--experimental-strip-types', 'eval/chess_ocean_tests.mjs'], { encoding: 'utf8' });
  assert.match(output, /METRIC chess_ocean_tests=6/);
});
