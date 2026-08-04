import assert from 'node:assert/strict';
import { test } from 'node:test';
import { parseFen, START_FEN } from '../src/chess/board.ts';
import { POLICY_MAP, POLICY_MOVES, POLICY_SIZE } from '../src/chess/policyMap.ts';
import { OnnxEvaluator } from '../src/nn/onnxEvaluator.ts';

test('ONNX evaluator masks legal policy and returns normalized WDL', async () => {
  const meta = {
    kind: 'student_onnx',
    architecture: 'residual_tower',
    policy_map: POLICY_MAP,
    moves: POLICY_MOVES,
    channels: 1,
    blocks: 1,
    history_plies: 2,
    input_planes: 46,
  };
  const policyLogits = new Float32Array(POLICY_SIZE);
  policyLogits.fill(-4);
  const session = {
    run: async () => ({ policy_logits: { data: policyLogits }, wdl_logits: { data: new Float32Array([2, 1, 0]) } }),
  };
  const evaluator = new OnnxEvaluator(session, meta);
  const ev = await evaluator.evaluate(parseFen(START_FEN));
  assert.equal(ev.policy.size, 20);
  const mass = [...ev.policy.values()].reduce((a, b) => a + b, 0);
  assert.ok(Math.abs(mass - 1) < 1e-5, mass);
  assert.equal(ev.wdl.length, 3);
  assert.ok(ev.wdl[0] > ev.wdl[1] && ev.wdl[1] > ev.wdl[2]);
});

test('ONNX evaluator requires named policy and WDL outputs', async () => {
  const meta = {
    kind: 'student_onnx',
    architecture: 'residual_tower',
    policy_map: POLICY_MAP,
    moves: ['e2e4'],
    channels: 1,
    blocks: 1,
    history_plies: 2,
    input_planes: 46,
  };
  const session = { run: async () => ({ arbitrary_first_output: { data: new Float32Array(1968) }, wdl_logits: { data: new Float32Array([0, 0, 0]) } }) };
  const evaluator = new OnnxEvaluator(session, meta);
  await assert.rejects(() => evaluator.evaluate(parseFen(START_FEN)), /missing required tensor 'policy_logits'/);
});

test('move-token ONNX evaluator requires named legal policy and WDL outputs', async () => {
  const meta = {
    kind: 'student_onnx',
    architecture: 'cnn_move_token_transformer',
    policy_map: POLICY_MAP,
    moves: [],
    channels: 1,
    blocks: 1,
    history_plies: 2,
    input_planes: 46,
    onnx_fixed_legal_moves: 128,
    num_move_features: 20,
  };
  const session = { run: async () => ({ first: { data: new Float32Array(128) }, wdl_logits: { data: new Float32Array([0, 0, 0]) } }) };
  const evaluator = new OnnxEvaluator(session, meta);
  await assert.rejects(() => evaluator.evaluate(parseFen(START_FEN)), /missing required tensor 'policy_logits_legal'/);
});

test('move-token ONNX evaluator fails loudly when fixed legal width overflows', async () => {
  const meta = {
    kind: 'student_onnx',
    architecture: 'cnn_move_token_transformer',
    policy_map: POLICY_MAP,
    moves: [],
    channels: 1,
    blocks: 1,
    history_plies: 2,
    input_planes: 46,
    onnx_fixed_legal_moves: 1,
    num_move_features: 20,
  };
  const session = {
    run: async () => {
      throw new Error('session.run should not be reached on legal overflow');
    },
  };
  const evaluator = new OnnxEvaluator(session, meta);
  await assert.rejects(() => evaluator.evaluate(parseFen(START_FEN)), /legal move overflow: model accepts 1 legal moves but position has 20/);
});
