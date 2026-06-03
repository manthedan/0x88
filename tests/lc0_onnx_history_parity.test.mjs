import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import { Lc0OnnxEvaluator } from '../src/lc0/onnxEvaluator.ts';
import { buildBoardHistoryFromMoves } from '../src/lc0/history.ts';

const MODEL = new URL('../../models/lc0-bestnets/onnx/t1-256x10-distilled-swa-2432500.batch1.f32.onnx', import.meta.url);
const nativeRecords = readFileSync(new URL('../fixtures/lc0/native_history_blas.jsonl', import.meta.url), 'utf8')
  .trim()
  .split('\n')
  .map((line) => JSON.parse(line));

function normalizeNativeUci(uci) {
  return ({ e1h1: 'e1g1', e1a1: 'e1c1', e8h8: 'e8g8', e8a8: 'e8c8' })[uci] ?? uci;
}

test('LC0 f32 ONNX evaluator matches native BLAS explicit move-history fixture priors', async () => {
  const evaluator = await Lc0OnnxEvaluator.create(readFileSync(MODEL));
  for (const native of nativeRecords) {
    const positions = buildBoardHistoryFromMoves(native.moves, native.startFen);
    const evaluation = await evaluator.evaluate({ positions });
    assert.equal(evaluation.bestMove, normalizeNativeUci(native.bestmove), `${native.id} best move`);
    for (const expected of native.topPriors.slice(0, 5)) {
      const uci = normalizeNativeUci(expected.uci);
      const actual = evaluation.legalPriors.find((entry) => entry.uci === uci);
      assert.ok(actual, `${native.id} missing ${uci}`);
      assert.ok(Math.abs(actual.prior - expected.prior) < 0.003, `${native.id} ${uci}: ${actual.prior} != ${expected.prior}`);
    }
  }
});
