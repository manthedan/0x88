import assert from 'node:assert/strict';
import test from 'node:test';
import { parseFen } from '../src/chess/board.ts';
import { squareformerCompactInput, squareformerFloatInput } from '../src/nn/squareformerEvaluator.ts';

const START_FEN = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';
const board = parseFen(START_FEN);

test('SquareFormer inputs reject corrupt history FENs', () => {
  assert.throws(() => squareformerFloatInput(board, { history_plies: 1, input_dim: 34 }, ['not-a-fen']), /Invalid FEN/);
  assert.throws(() => squareformerCompactInput(board, { history_plies: 1, input_dim: 0, token_features: 10 }, ['not-a-fen'], 'int32'), /Invalid FEN/);
});
