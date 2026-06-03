import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { test } from 'node:test';
import { START_FEN } from '../src/chess/board.ts';
import { Lc0OnnxEvaluator } from '../src/lc0/onnxEvaluator.ts';

const BATCH1 = '../models/lc0-bestnets/onnx/t1-256x10-distilled-swa-2432500.batch1.f16.onnx';
const BATCH4 = '../models/lc0-bestnets/onnx/t1-256x10-distilled-swa-2432500.batch4.f16.onnx';
const BATCH8 = '../models/lc0-bestnets/onnx/t1-256x10-distilled-swa-2432500.batch8.f16.onnx';

const FENS = [
  START_FEN,
  'rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq d6 0 2',
  'rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2',
  'rnbqkbnr/pppppppp/8/8/2P5/8/PP1PPPPP/RNBQKBNR b KQkq c3 0 1',
  'rnbqkbnr/ppp1pppp/8/3p4/2P5/8/PP1PPPPP/RNBQKBNR w KQkq d6 0 2',
];

function skipIfMissing(...paths) {
  const missing = paths.find((path) => !existsSync(path));
  return missing ? `missing ${missing}` : false;
}

async function compareBatchArtifact(path, size) {
  const baseline = await Lc0OnnxEvaluator.create(readFileSync(BATCH1));
  const batched = await Lc0OnnxEvaluator.create(readFileSync(path));
  const expected = [];
  for (const fen of FENS) expected.push(await baseline.evaluate(fen));
  const actual = await batched.evaluateBatch(FENS);
  assert.equal(actual.length, FENS.length);
  for (let i = 0; i < FENS.length; i++) {
    assert.equal(actual[i].fen, expected[i].fen);
    assert.equal(actual[i].bestMove, expected[i].bestMove, `batch${size} best move mismatch at ${i}`);
    assert.ok(Math.abs(actual[i].q - expected[i].q) < 1e-5, `batch${size} q mismatch at ${i}: ${actual[i].q} vs ${expected[i].q}`);
    assert.ok(Math.abs(actual[i].mlh - expected[i].mlh) < 1e-3, `batch${size} mlh mismatch at ${i}: ${actual[i].mlh} vs ${expected[i].mlh}`);
    for (let j = 0; j < 5; j++) {
      assert.equal(actual[i].legalPriors[j]?.uci, expected[i].legalPriors[j]?.uci, `batch${size} top-${j + 1} move mismatch at ${i}`);
      assert.ok(Math.abs(actual[i].legalPriors[j].prior - expected[i].legalPriors[j].prior) < 1e-5, `batch${size} top-${j + 1} prior mismatch at ${i}`);
    }
  }
  // Single eval should also work against fixed batch-N artifacts via padding.
  const single = await batched.evaluate(START_FEN);
  assert.equal(single.bestMove, expected[0].bestMove);
}

test('LC0 fixed batch-4 f16 ONNX matches batch-1 evaluator outputs', { skip: skipIfMissing(BATCH1, BATCH4) }, async () => {
  await compareBatchArtifact(BATCH4, 4);
});

test('LC0 fixed batch-8 f16 ONNX matches batch-1 evaluator outputs', { skip: skipIfMissing(BATCH1, BATCH8) }, async () => {
  await compareBatchArtifact(BATCH8, 8);
});
