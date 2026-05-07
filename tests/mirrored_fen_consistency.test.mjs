import test from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';

test('mirrored FEN legal moves, action ids, and symmetric priors are consistent', () => {
  const output = execFileSync('node', ['--experimental-strip-types', 'eval/mirrored_fen_consistency_check.mjs'], { encoding: 'utf8' });
  assert.match(output, /METRIC mirrored_fen_roundtrips=6/);
  assert.match(output, /METRIC mirrored_fen_legal_sets=6/);
  assert.match(output, /METRIC mirrored_fen_prior_parity=6/);
});
