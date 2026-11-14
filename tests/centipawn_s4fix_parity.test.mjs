import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { boardToFen, parseFen, squareName } from '../src/chess/board.ts';
import { normalizePositionForStmWhite, normalizedMoveToOriginal } from '../src/chess/boardNormalization.ts';
import { moveToActionId, moveToUci } from '../src/chess/moveCodec.ts';
import { moveToSquareformerPolicyIndex } from '../src/chess/moveEncodings.ts';
import { legalMoves } from '../src/chess/movegen.ts';
import { softmax } from '../src/nn/numerics.ts';
import * as ort from '../src/nn/ortRuntime.ts';
import { S4FIX_V_EMA_MODEL_ID, squareformerModelSettings } from '../src/nn/runtimeRegistry.ts';
import { SquareFormerEvaluator, squareformerCompactInput, threatgraphSquareSummaryV1 } from '../src/nn/squareformerEvaluator.ts';
import { skipUnlessAssets } from './lib/required_assets.mjs';

const fixture = JSON.parse(readFileSync(new URL('./fixtures/centipawn_s4fix_parity_fixture.json', import.meta.url)));
const modelPath = new URL('../public/models/s4fix_v_ema.onnx', import.meta.url);
const meta = JSON.parse(readFileSync(new URL('../public/models/s4fix_v_ema.meta.json', import.meta.url)));
const settings = squareformerModelSettings(S4FIX_V_EMA_MODEL_ID);

function inputs(position) {
  const normalized = normalizePositionForStmWhite(parseFen(position.fen), position.history_fens);
  assert.equal(normalized.flipped, position.flipped);
  assert.equal(boardToFen(normalized.board), position.normalized_fen);
  return {
    normalized,
    tokens: squareformerCompactInput(normalized.board, meta, normalized.historyFens, 'int64', settings),
    attack: threatgraphSquareSummaryV1(normalized.board),
  };
}

function equalIntegers(actual, expected, width, label) {
  assert.equal(actual.length, expected.length);
  for (let i = 0; i < actual.length; i++) {
    assert.equal(Number(actual[i]), expected[i], `${label} square ${squareName(Math.floor(i / width))} plane ${i % width}`);
  }
}

test('s4fix encoder and legal policy indices equal native inputs at all 51 positions', () => {
  assert.equal(fixture.positions.length, 51);
  for (const position of fixture.positions) {
    const { normalized, tokens, attack } = inputs(position);
    equalIntegers(tokens, position.tokens, 24, `${position.label} tokens`);
    equalIntegers(attack, position.attack_summary, 28, `${position.label} attack_summary`);
    const moves = legalMoves(normalized.board);
    const indices = new Map(moves.map((move) => [moveToUci(normalizedMoveToOriginal(move, normalized.flipped)), moveToSquareformerPolicyIndex(move)]));
    assert.deepEqual([...indices.keys()].sort(), [...position.legal_uci].sort(), position.label);
    position.legal_uci.forEach((uci, i) => assert.equal(indices.get(uci), position.legal_policy_index[i], `${position.label} ${uci}`));
  }
});

test('s4fix ORT Web outputs and evaluator T=2 priors match the native fixture', { skip: skipUnlessAssets(modelPath) }, async (t) => {
  const bytes = readFileSync(modelPath);
  assert.equal(createHash('sha256').update(bytes).digest('hex'), fixture.model_sha256);
  const session = await ort.InferenceSession.create(bytes, ort.sessionOptions(['wasm']));
  const evaluator = new SquareFormerEvaluator(session, meta, settings);
  const baseline = new SquareFormerEvaluator(session, meta, { ...settings, policyTemperature: 1 });
  const max = { policy_logits: 0, wdl_logits: 0, wdl: 0, q: 0, wdl_sf18_logits: 0, prior: 0 };
  const check = (name, actual, expected, tolerance, label) => {
    assert.equal(actual.length, expected.length);
    for (let i = 0; i < actual.length; i++) {
      const delta = Math.abs(actual[i] - expected[i]);
      max[name] = Math.max(max[name], delta);
      assert.ok(Number.isFinite(delta) && delta <= tolerance, `${label} ${name}[${i}] delta ${delta}`);
    }
  };
  try {
    for (const position of fixture.positions) {
      const { tokens, attack } = inputs(position);
      const outputs = await session.run({
        tokens: new ort.Tensor('int64', tokens, [1, 64, 24]),
        attack_summary: new ort.Tensor('float32', attack, [1, 64, 28]),
      }, ['policy', 'wdl', 'q', 'wdl_sf18']);
      check('policy_logits', position.legal_policy_index.map((i) => outputs.policy.data[i]), position.expected.policy_logits, 2e-3, position.label);
      check('wdl_logits', outputs.wdl.data, position.expected.wdl_logits, 2e-3, position.label);
      check('wdl', softmax(outputs.wdl.data), position.expected.wdl, 2e-4, position.label);
      check('q', outputs.q.data, [position.expected.q], 2e-4, position.label);
      check('wdl_sf18_logits', outputs.wdl_sf18.data, position.expected.wdl_sf18_logits, 2e-3, position.label);
      const board = parseFen(position.fen);
      const context = { historyFens: position.history_fens };
      const evaluation = await evaluator.evaluate(board, context);
      const actionIds = new Map(legalMoves(board).map((move) => [moveToUci(move), moveToActionId(move)]));
      check('prior', position.legal_uci.map((uci) => evaluation.policy.get(actionIds.get(uci))), softmax(position.expected.policy_logits.map((x) => x / 2)), 2e-4, position.label);
      check('wdl', evaluation.wdl, position.expected.wdl, 2e-4, position.label);
      check('wdl', evaluation.auxiliaryWdls.wdl_sf18, softmax(position.expected.wdl_sf18_logits), 2e-4, position.label);
      if ([0, 6, 12, 50].includes(fixture.positions.indexOf(position))) {
        const plain = await baseline.evaluate(board, context);
        assert.deepEqual(evaluation.wdl, plain.wdl, 'policy temperature must not change WDL');
        assert.deepEqual(evaluation.auxiliaryWdls, plain.auxiliaryWdls, 'policy temperature must not change auxiliary WDL');
        check('prior', position.legal_uci.map((uci) => plain.policy.get(actionIds.get(uci))), softmax(position.expected.policy_logits), 2e-4, position.label);
        assert.notDeepEqual(evaluation.policy, plain.policy);
      }
    }
    t.diagnostic(`51 positions; integer encoder equality; max absolute differences ${JSON.stringify(max)}`);
  } finally {
    await session.release();
  }
});
