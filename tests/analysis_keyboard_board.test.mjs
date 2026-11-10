import assert from 'node:assert/strict';
import test from 'node:test';
import { parseFen } from '../src/chess/board.ts';
import { analysisBoardSquares } from '../src/lc0/analysisKeyboardBoard.ts';

test('keyboard board names all squares and pieces in visual order', () => {
  const board = parseFen();
  const white = analysisBoardSquares(board, 'white');
  assert.equal(white.length, 64);
  assert.equal(new Set(white.map((square) => square.key)).size, 64);
  assert.deepEqual(white[0], { key: 'a8', label: 'a8, Black rook' });
  assert.deepEqual(white[63], { key: 'h1', label: 'h1, White rook' });
  assert.equal(white.find((square) => square.key === 'e2').label, 'e2, White pawn');
  assert.equal(white.find((square) => square.key === 'e4').label, 'e4, empty');
  assert.deepEqual(analysisBoardSquares(board, 'black'), [...white].reverse());
});
