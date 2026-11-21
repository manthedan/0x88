import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import { START_FEN } from '../src/chess/board.ts';
import { encodeLc0Classical112 } from '../src/lc0/encoder112.ts';
import { buildBoardHistoryFromMoves } from '../src/lc0/history.ts';
import { Lc0OnnxEvaluator, legalPolicyPriors, currentBoardAndFen } from '../src/lc0/onnxEvaluator.ts';
import { f16BitsToF32, f32ToF16Bits } from '../src/nn/float16.ts';
import { skipUnlessAssets } from './lib/required_assets.mjs';

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

test('f16 batches preserve exact input bits and legal outputs, lease distinct concurrent inputs, and reuse idle storage', async () => {
  const inputs = [
    { positions: buildBoardHistoryFromMoves(['g1f3', 'g8f6', 'f3g1', 'f6g8']) },
    'rnbqkbnr/pppppppp/8/8/2P5/8/PP1PPPPP/RNBQKBNR b KQkq c3 0 1',
    '4k3/8/8/8/8/8/8/4K2R w - - 4099 20',
  ];
  const release = Promise.withResolvers();
  const seen = [];
  const policy = Uint16Array.from({ length: 2 * 1858 }, (_, i) => [0x0001, 0x8000, 0xbc00, 0x3555, 0xfc00][i % 5]);
  const decoded = Array.from(policy, f16BitsToF32);
  policy[Symbol.iterator] = () => assert.fail('must not decode the entire f16 policy');
  const output = {
    '/output/policy': { type: 'float16', data: policy },
    '/output/wdl': { type: 'float16', data: Uint16Array.of(0x3800, 0x8000, 0x3800, 0x3400, 0x3800, 0x3400) },
    '/output/mlh': { type: 'float16', data: new Float16Array([36.5, 22.5]) },
  };
  const evaluator = new Lc0OnnxEvaluator({
    inputMetadata: [{ name: '/input/planes', type: 'float16', shape: [2, 112, 8, 8] }],
    async run(feeds) {
      const { data } = feeds['/input/planes'];
      seen.push(new Uint16Array(data.buffer, data.byteOffset, data.length));
      await release.promise;
      return output;
    },
  }, { policyTemperature: 1 });

  const first = evaluator.evaluateBatch(inputs.slice(0, 2));
  const second = evaluator.evaluate(inputs[2]);
  await new Promise(setImmediate);
  try {
    assert.equal(seen.length, 2);
    assert.notEqual(seen[0].buffer, seen[1].buffer);
    const expected = (input) => Uint16Array.from(encodeLc0Classical112(input).planes, f32ToF16Bits);
    for (const [actual, positions] of [[seen[0], inputs.slice(0, 2)], [seen[1], [inputs[2], inputs[2]]]]) {
      for (let slot = 0; slot < 2; slot++) assert.deepEqual(actual.subarray(slot * 7168, (slot + 1) * 7168), expected(positions[slot]));
    }
  } finally {
    release.resolve();
  }
  const results = (await first).concat(await second);
  const again = await evaluator.evaluate(START_FEN);
  assert.ok(seen.slice(0, 2).some(({ buffer }) => buffer === seen[2].buffer), 'next batch reuses an idle f16 allocation');
  const startBits = Uint16Array.from(encodeLc0Classical112(START_FEN).planes, f32ToF16Bits);
  assert.deepEqual(seen[2].subarray(0, 7168), startBits, 'reuse clears planes written only by the previous position');
  assert.deepEqual(seen[2].subarray(7168), startBits, 'the padded slot copies the new input');
  for (let i = 0; i < results.length; i++) {
    const slot = i % 2;
    assert.deepEqual(results[i].legalPriors, legalPolicyPriors(currentBoardAndFen(inputs[i]).board, decoded.slice(slot * 1858, (slot + 1) * 1858), 1));
    assert.deepEqual(results[i].wdl, slot === 0 ? [0.5, -0, 0.5] : [0.25, 0.5, 0.25]);
    assert.equal(results[i].mlh, slot === 0 ? 36.5 : 22.5);
  }
  assert.equal(again.fen, START_FEN);
  await evaluator.dispose();
});

test('uint16 ONNX outputs remain integers rather than being decoded as float16 bits', async () => {
  const board = currentBoardAndFen(START_FEN).board;
  const policy = new Uint16Array(1858).fill(36);
  policy[legalPolicyPriors(board, policy, 1)[0].index] = 37;
  const evaluator = new Lc0OnnxEvaluator({
    async run() {
      return {
        '/output/policy': { type: 'uint16', data: policy },
        '/output/wdl': { type: 'uint16', data: Uint16Array.of(1, 0, 0) },
        '/output/mlh': { type: 'uint16', data: Uint16Array.of(36) },
      };
    },
  }, { policyTemperature: 1 });
  const result = await evaluator.evaluate(START_FEN);
  assert.deepEqual(result.legalPriors, legalPolicyPriors(board, Float32Array.from(policy), 1));
  assert.deepEqual(result.wdl, [1, 0, 0]);
  assert.equal(result.mlh, 36);
  await evaluator.dispose();
});

const skipIfMissing = skipUnlessAssets;

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
