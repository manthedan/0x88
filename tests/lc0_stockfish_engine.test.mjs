import assert from 'node:assert/strict';
import { test } from 'node:test';
import { parseBestMove, stockfishGoCommand } from '../src/lc0/stockfishEngine.ts';

test('parseBestMove extracts the UCI move and handles (none)', () => {
  assert.equal(parseBestMove('bestmove e2e4 ponder e7e5'), 'e2e4');
  assert.equal(parseBestMove('bestmove a7a8q'), 'a7a8q');
  assert.equal(parseBestMove('bestmove (none)'), null);
  assert.equal(parseBestMove('info depth 12 score cp 31'), null);
});

test('stockfishGoCommand prefers movetime over depth and clamps depth', () => {
  assert.equal(stockfishGoCommand({ depth: 6 }), 'go depth 6');
  assert.equal(stockfishGoCommand({}), 'go depth 4');
  assert.equal(stockfishGoCommand({ depth: 0 }), 'go depth 1');
  assert.equal(stockfishGoCommand({ depth: 6, movetimeMs: 200 }), 'go movetime 200');
});
