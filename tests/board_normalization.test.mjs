import assert from 'node:assert/strict';
import test from 'node:test';
import { boardToFen, parseFen } from '../src/chess/board.ts';
import {
  normalizedMoveToOriginal,
  normalizeFenForStmWhite,
  normalizeHistoryForStmWhite,
  normalizePositionForStmWhite,
  rankFlipColorSwapBoard,
  rankFlipMove,
  rankFlipSquare,
} from '../src/chess/boardNormalization.ts';
import { moveFromUci, moveToUci } from '../src/chess/moveCodec.ts';

const BLACK_TO_MOVE = '8/8/8/3pP3/8/8/8/4k2K b - e3 17 42';
const NORMALIZED = '4K2k/8/8/8/3Pp3/8/8/8 w - e6 17 42';

test('rank-flip square, board, and move transforms are involutions', () => {
  for (let square = 0; square < 64; square++) assert.equal(rankFlipSquare(rankFlipSquare(square)), square);
  const board = parseFen(BLACK_TO_MOVE);
  assert.equal(boardToFen(rankFlipColorSwapBoard(board)), NORMALIZED);
  assert.equal(boardToFen(rankFlipColorSwapBoard(rankFlipColorSwapBoard(board))), BLACK_TO_MOVE);
  const move = moveFromUci('d5d4');
  assert.equal(moveToUci(rankFlipMove(move)), 'd4d5');
  assert.equal(moveToUci(normalizedMoveToOriginal(rankFlipMove(move), true)), moveToUci(move));
});

test('side-to-move normalization transforms black history and legal moves together', () => {
  const board = parseFen(BLACK_TO_MOVE);
  const history = ['8/8/8/3p4/4P3/8/8/4k2K w - - 16 42'];
  const legal = [moveFromUci('d5d4')];
  const normalized = normalizePositionForStmWhite(board, history, legal);
  assert.equal(normalized.flipped, true);
  assert.equal(boardToFen(normalized.board), NORMALIZED);
  assert.deepEqual(normalized.historyFens, normalizeHistoryForStmWhite(history, true));
  assert.deepEqual(normalized.legalMoves?.map(moveToUci), ['d4d5']);
  assert.deepEqual(normalizeFenForStmWhite(BLACK_TO_MOVE), { fen: NORMALIZED, flipped: true });
});

test('side-to-move normalization rejects invalid coordinates and corrupt history', () => {
  assert.throws(() => rankFlipSquare(-1), /square out of range/);
  assert.throws(() => normalizeHistoryForStmWhite(['not-a-fen'], true), /Invalid FEN/);
});
