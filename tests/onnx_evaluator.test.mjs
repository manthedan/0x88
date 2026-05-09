import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { test } from 'node:test';
import { parseFen, START_FEN } from '../src/chess/board.ts';
import { POLICY_MAP } from '../src/chess/policyMap.ts';
import { OnnxEvaluator } from '../src/nn/onnxEvaluator.ts';

const model = '/tmp/residual_smoke.onnx';
const metaPath = '/tmp/residual_smoke.meta.json';

test('ONNX evaluator masks legal policy and returns WDL', { skip: !existsSync(model) || !existsSync(metaPath) }, async () => {
  const meta = JSON.parse(readFileSync(metaPath, 'utf8'));
  const evaluator = await OnnxEvaluator.create(model, meta);
  const ev = await evaluator.evaluate(parseFen(START_FEN));
  assert.equal(ev.policy.size, 20);
  const mass = [...ev.policy.values()].reduce((a, b) => a + b, 0);
  assert.ok(Math.abs(mass - 1) < 1e-5, mass);
  assert.equal(ev.wdl.length, 3);
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
  const session = { run: async () => { throw new Error('session.run should not be reached on legal overflow'); } };
  const evaluator = new OnnxEvaluator(session, meta);
  await assert.rejects(() => evaluator.evaluate(parseFen(START_FEN)), /legal move overflow: model accepts 1 legal moves but position has 20/);
});
