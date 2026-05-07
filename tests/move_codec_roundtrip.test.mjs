import test from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';

test('legal move UCI and action id roundtrip for castling/promotions/common positions', () => {
  const output = execFileSync('node', ['--experimental-strip-types', 'eval/move_codec_roundtrip_check.mjs'], { encoding: 'utf8' });
  assert.match(output, /METRIC move_codec_roundtrip_positions=5/);
});
