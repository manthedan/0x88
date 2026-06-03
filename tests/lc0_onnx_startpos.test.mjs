import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { test } from 'node:test';
import { START_FEN } from '../src/chess/board.ts';
import { Lc0OnnxEvaluator } from '../src/lc0/onnxEvaluator.ts';

const MODEL = '../models/lc0-bestnets/onnx/t1-256x10-distilled-swa-2432500.batch1.f32.onnx';

test('LC0 f32 ONNX evaluator reproduces native startpos root priors', { skip: !existsSync(MODEL) && `missing ${MODEL}` }, async () => {
  const evaluator = await Lc0OnnxEvaluator.create(readFileSync(MODEL));
  const evaluation = await evaluator.evaluate(START_FEN);
  const top = Object.fromEntries(evaluation.legalPriors.slice(0, 5).map((entry) => [entry.uci, entry.prior]));

  assert.equal(evaluation.bestMove, 'd2d4');
  assert.deepEqual(evaluation.legalPriors.slice(0, 5).map((entry) => entry.uci), ['d2d4', 'g1f3', 'e2e4', 'c2c4', 'g2g3']);
  assert.ok(Math.abs(top.d2d4 - 0.1804) < 0.002, `d2d4 ${top.d2d4}`);
  assert.ok(Math.abs(top.g1f3 - 0.1568) < 0.005, `g1f3 ${top.g1f3}`);
  assert.ok(Math.abs(top.e2e4 - 0.0955) < 0.005, `e2e4 ${top.e2e4}`);
  assert.ok(Math.abs(top.c2c4 - 0.0906) < 0.005, `c2c4 ${top.c2c4}`);
  assert.ok(Math.abs(top.g2g3 - 0.0906) < 0.005, `g2g3 ${top.g2g3}`);
  assert.ok(Math.abs(evaluation.wdl.reduce((acc, value) => acc + value, 0) - 1) < 1e-5);
});
