import assert from 'node:assert/strict';
import { test } from 'node:test';
import { parseFen, START_FEN } from '../src/chess/board.ts';
import { onnxInputPlanes } from '../src/nn/onnxEvaluator.ts';

const afterE4 = 'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1';

test('ONNX input planes include newest-first history positions', () => {
  const board = parseFen(afterE4);
  const meta = { input_planes: 46, history_plies: 2 };
  const noHistory = onnxInputPlanes(board, meta);
  const withHistory = onnxInputPlanes(board, meta, [START_FEN]);
  const e2RankFile = (8 - 2) * 8 + (101 - 97);
  const whitePawnHistoryPlane = 12;
  assert.equal(noHistory[whitePawnHistoryPlane * 64 + e2RankFile], 0);
  assert.equal(withHistory[whitePawnHistoryPlane * 64 + e2RankFile], 1);
});
