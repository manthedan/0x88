import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { test } from 'node:test';
import { parseFen, START_FEN } from '../src/chess/board.ts';
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
