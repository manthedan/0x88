import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { test } from 'node:test';
import { Lc0OnnxEvaluator } from '../src/lc0/onnxEvaluator.ts';

const MODEL = '../models/lc0-bestnets/onnx/t1-256x10-distilled-swa-2432500.batch1.f32.onnx';
const NATIVE_PRIORS = 'fixtures/lc0/native_fen_only_blas.jsonl';

function nativeCastlingToStandard(uci) {
  switch (uci) {
    case 'e1h1': return 'e1g1';
    case 'e1a1': return 'e1c1';
    case 'e8h8': return 'e8g8';
    case 'e8a8': return 'e8c8';
    default: return uci;
  }
}

test('LC0 f32 ONNX evaluator matches native BLAS/Eigen FEN-only fixture priors', { skip: (!existsSync(MODEL) || !existsSync(NATIVE_PRIORS)) && 'missing model or native prior artifact' }, async () => {
  const nativeRecords = readFileSync(NATIVE_PRIORS, 'utf8').trim().split('\n').map((line) => JSON.parse(line));
  const evaluator = await Lc0OnnxEvaluator.create(readFileSync(MODEL));
  for (const native of nativeRecords) {
    const evaluation = await evaluator.evaluate(native.fen);
    assert.equal(evaluation.bestMove, nativeCastlingToStandard(native.bestmove), `${native.id} bestmove`);
    for (const prior of native.topPriors) {
      const uci = nativeCastlingToStandard(prior.uci);
      const actual = evaluation.legalPriors.find((entry) => entry.uci === uci);
      assert.ok(actual, `${native.id} has ${uci}`);
      assert.equal(actual.index, prior.index, `${native.id} ${uci} index`);
      assert.ok(Math.abs(actual.prior - prior.prior) < 0.0025, `${native.id} ${uci} prior native=${prior.prior} onnx=${actual.prior}`);
    }
  }
});
