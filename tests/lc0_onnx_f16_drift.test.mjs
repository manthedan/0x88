import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import { Lc0OnnxEvaluator } from '../src/lc0/onnxEvaluator.ts';

const F32_MODEL = new URL('../../models/lc0-bestnets/onnx/t1-256x10-distilled-swa-2432500.batch1.f32.onnx', import.meta.url);
const F16_MODEL = new URL('../../models/lc0-bestnets/onnx/t1-256x10-distilled-swa-2432500.batch1.f16.onnx', import.meta.url);
const fixtures = JSON.parse(readFileSync(new URL('../fixtures/lc0/fen_only.json', import.meta.url), 'utf8'));

test('LC0 f16 ONNX evaluator stays close to f32 FEN-only priors under WASM', async () => {
  const f32 = await Lc0OnnxEvaluator.create(readFileSync(F32_MODEL));
  const f16 = await Lc0OnnxEvaluator.create(readFileSync(F16_MODEL));
  for (const fixture of fixtures) {
    const expected = await f32.evaluate(fixture.fen);
    const actual = await f16.evaluate(fixture.fen);
    assert.equal(actual.bestMove, expected.bestMove, `${fixture.id} best move`);
    assert.equal(actual.legalPriors.length, expected.legalPriors.length, `${fixture.id} legal count`);
    for (const expectedPrior of expected.legalPriors.slice(0, 8)) {
      const actualPrior = actual.legalPriors.find((entry) => entry.uci === expectedPrior.uci);
      assert.ok(actualPrior, `${fixture.id} missing ${expectedPrior.uci}`);
      assert.ok(Math.abs(actualPrior.prior - expectedPrior.prior) < 0.01, `${fixture.id} ${expectedPrior.uci}: ${actualPrior.prior} != ${expectedPrior.prior}`);
    }
    assert.ok(Math.abs(actual.q - expected.q) < 0.01, `${fixture.id} Q drift ${actual.q} != ${expected.q}`);
    assert.ok(Math.abs(actual.mlh - expected.mlh) < 1.0, `${fixture.id} MLH drift ${actual.mlh} != ${expected.mlh}`);
  }
});
