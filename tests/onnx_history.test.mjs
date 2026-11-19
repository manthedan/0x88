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

test('ONNX input planes leave the two planes after the side-to-move flag zero, even in check', () => {
  // Black to move, in check from Bb5.
  const inCheck = parseFen('rnbqkbnr/ppp1pppp/8/1B1p4/4P3/8/PPPP1PPP/RNBQK1NR b KQkq - 1 2');
  const meta = { input_planes: 46, history_plies: 2 };
  const planes = onnxInputPlanes(inCheck, meta);
  const state0 = 12 * (meta.history_plies + 1);
  for (const plane of [state0 + 8, state0 + 9]) {
    assert.ok(
      planes.subarray(plane * 64, (plane + 1) * 64).every((v) => v === 0),
      `plane ${plane}`,
    );
  }
});
