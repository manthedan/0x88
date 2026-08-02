import assert from 'node:assert/strict';
import test from 'node:test';
import { parseArgs } from '../scripts/lc0_tvmjs_webgpu_smoke.mjs';

test('TVMJS FEN fixture mode follows the final fixture flag', () => {
  assert.equal(parseArgs(['--no-fixtures', '--fens', 'suite.fen']).fixtures, true);
  assert.equal(parseArgs(['--no-fixtures', '--fens=suite.fen']).fixtures, true);
  const disabledSeparated = parseArgs(['--fens', 'suite.fen', '--no-fixtures']);
  assert.equal(disabledSeparated.fixtures, false);
  assert.equal(disabledSeparated.fensFile, '');
  const disabledInline = parseArgs(['--fens=suite.fen', '--no-fixtures']);
  assert.equal(disabledInline.fixtures, false);
  assert.equal(disabledInline.fensFile, '');
  assert.equal(parseArgs(['--no-fixtures', '--fens', 'suite.fen', '--fixtures']).fixtures, true);
});
