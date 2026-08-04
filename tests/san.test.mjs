import assert from 'node:assert/strict';
import test from 'node:test';
import { parseFen, START_FEN } from '../src/chess/board.ts';
import { moveFromUci } from '../src/chess/moveCodec.ts';
import { makeMove } from '../src/chess/movegen.ts';
import { moveToSan, uciLineToSan, uciToSan } from '../src/chess/san.ts';

test('SAN formats ordinary moves and a legal line', () => {
  const board = parseFen(START_FEN);
  assert.equal(uciToSan(board, 'e2e4'), 'e4');
  assert.equal(uciLineToSan(board, ['e2e4', 'e7e5', 'g1f3']), 'e4 e5 Nf3');
});

test('SAN formats castling, disambiguation, and check', () => {
  assert.equal(uciToSan(parseFen('r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1'), 'e1g1'), 'O-O');
  assert.equal(uciToSan(parseFen('7k/8/8/8/8/2N1N3/8/7K w - - 0 1'), 'c3d5'), 'Ncd5');
  assert.equal(uciToSan(parseFen('7k/8/8/8/8/8/8/R6K w - - 0 1'), 'a1a8'), 'Ra8+');
});

test('SAN falls back to original notation for malformed or illegal input', () => {
  const board = parseFen(START_FEN);
  assert.equal(uciToSan(board, 'bad'), 'bad');
  assert.equal(moveToSan(board, moveFromUci('a3a4')), 'a3a4');
  assert.equal(uciLineToSan(board, ['e2e4', 'e7e5', 'bad', 'g1f3']), 'e4 e5 bad');
  assert.equal(makeMove(board, moveFromUci('e2e4')).turn, 'b');
});
