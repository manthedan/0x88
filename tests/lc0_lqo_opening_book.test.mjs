import assert from 'node:assert/strict';
import { test } from 'node:test';
import { parseFen } from '../src/chess/board.ts';
import { legalMoves, makeMove } from '../src/chess/movegen.ts';
import { moveToUci } from '../src/chess/moveCodec.ts';
import { lqoBlackBookMove, lqoBlackBookMoves, lqoWhiteBookMove, lqoWhiteBookMoves, lqoWhiteFirstPolicyBookMove } from '../src/lc0/lqoOpeningBook.ts';

const LQO_BLACK_START = 'rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';
const LQO_WHITE_START = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1';

function playUci(board, uci) {
  const move = legalMoves(board).find((candidate) => moveToUci(candidate) === uci);
  assert.ok(move, `${uci} should be legal`);
  return makeMove(board, move);
}

test('LQO black opening book covers normal first moves and returns legal replies', () => {
  const afterE4 = playUci(parseFen(LQO_BLACK_START), 'e2e4');
  const history = ['e2e4'];
  const moves = lqoBlackBookMoves(afterE4, 1, history);
  assert.ok(moves.length > 0);
  assert.ok(moves.every((move) => legalMoves(afterE4).some((legal) => moveToUci(legal) === move.uci)));
  assert.equal(lqoBlackBookMove(afterE4, 1, history, () => 0), moves[0].uci);
});

test('LQO black opening book falls back when a first move is uncovered', () => {
  const afterF3 = playUci(parseFen(LQO_BLACK_START), 'f2f3');
  assert.deepEqual(lqoBlackBookMoves(afterF3, 1, ['f2f3']), []);
  assert.equal(lqoBlackBookMove(afterF3, 1, ['f2f3'], () => 0), null);
});

test('LQO black opening book is limited to black early plies', () => {
  const start = parseFen(LQO_BLACK_START);
  assert.equal(lqoBlackBookMove(start, 0, [], () => 0), null);
  const afterE4 = playUci(start, 'e2e4');
  assert.equal(lqoBlackBookMove(afterE4, 6, ['e2e4'], () => 0), null);
});

test('LQO white first move samples from model policy moves that are also in book', () => {
  const start = parseFen(LQO_WHITE_START);
  assert.ok(lqoWhiteBookMoves(start, 0, []).some((move) => move.uci === 'g1f3'));
  const legalPriors = [
    { uci: 'f2f3', prior: 0.9 },
    { uci: 'g1f3', prior: 0.8 },
    { uci: 'c2c4', prior: 0.7 },
  ];
  assert.equal(lqoWhiteFirstPolicyBookMove(start, 0, [], legalPriors, () => 0, 1), null);
  assert.equal(lqoWhiteFirstPolicyBookMove(start, 0, [], legalPriors, () => 0, 3), 'g1f3');
  assert.equal(lqoWhiteFirstPolicyBookMove(start, 0, [], legalPriors, () => 0.999, 3), 'c2c4');
  assert.equal(lqoWhiteFirstPolicyBookMove(playUci(start, 'g1f3'), 1, ['g1f3'], legalPriors, () => 0), null);
});

test('LQO white opening book covers the second white move before search', () => {
  let board = parseFen(LQO_WHITE_START);
  board = playUci(board, 'g1f3');
  board = playUci(board, 'g8f6');
  const history = ['g1f3', 'g8f6'];
  const moves = lqoWhiteBookMoves(board, 2, history);
  assert.ok(moves.some((move) => move.uci === 'd2d4'));
  assert.equal(lqoWhiteBookMove(board, 2, history, () => 0), moves[0].uci);
});
