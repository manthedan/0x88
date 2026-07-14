import assert from 'node:assert/strict';
import { test } from 'node:test';
import { parseFen, START_FEN } from '../src/chess/board.ts';
import { legalMoves, makeMove } from '../src/chess/movegen.ts';
import { moveToActionId } from '../src/chess/moveCodec.ts';
import { encodeLc0Classical112 } from '../src/lc0/encoder112.ts';
import {
  currentBoardAndFen,
  legalPolicyPriors,
  prepareLc0EvaluatorInput,
} from '../src/lc0/onnxEvaluator.ts';
import { Lc0PuctSearcher, Lc0SearchEvaluator } from '../src/lc0/search.ts';

function arraysEqual(a, b) {
  assert.equal(a.length, b.length);
  for (let i = 0; i < a.length; i++) assert.equal(a[i], b[i], `array mismatch at ${i}`);
}

test('prepared LC0 input preserves planes and carries legal action/policy mappings', () => {
  const start = parseFen(START_FEN);
  const e2e4 = legalMoves(start).find((move) => move.from === 12 && move.to === 28);
  assert.ok(e2e4);
  const board = makeMove(start, e2e4);
  const moves = legalMoves(board);
  const prepared = prepareLc0EvaluatorInput(board, [start], moves, [START_FEN]);

  assert.equal(currentBoardAndFen(prepared).board, board, 'current board is reused by identity');
  assert.equal(prepared.positions[0], start, 'history board is not serialized and reparsed');
  assert.equal(prepared.positions[1], board);
  assert.equal(prepared.prepared.legalMoves.length, moves.length);

  const expectedPlanes = encodeLc0Classical112({ positions: [start, board] }).planes;
  const preparedPlanes = encodeLc0Classical112(prepared).planes;
  arraysEqual(preparedPlanes, expectedPlanes);

  const logits = new Float32Array(1858);
  const priors = legalPolicyPriors(board, logits, 1, prepared.prepared.legalMoves);
  assert.equal(priors.length, moves.length);
  assert.ok(priors.every((prior) => prior.actionId !== undefined));
  assert.deepEqual(
    new Set(priors.map((prior) => prior.actionId)),
    new Set(moves.map(moveToActionId)),
  );
});

test('LC0 search adapter passes prepared history and legal mappings through to the provider', async () => {
  const start = parseFen(START_FEN);
  const moves = legalMoves(start);
  let captured;
  const provider = {
    evaluate(input) {
      captured = input;
      const priors = legalPolicyPriors(currentBoardAndFen(input).board, new Float32Array(1858), 1);
      return { fen: START_FEN, wdl: [0.3, 0.4, 0.3], q: 0, mlh: 0, legalPriors: priors, bestMove: priors[0]?.uci };
    },
  };
  const adapter = new Lc0SearchEvaluator(provider);
  const evaluation = await adapter.evaluate(start, { historyFens: [], historyBoards: [], legalMoves: moves });

  assert.ok(captured && 'prepared' in captured);
  assert.equal(captured.prepared.board, start);
  assert.equal(captured.prepared.legalMoves.length, moves.length);
  assert.deepEqual(new Set(evaluation.policy.keys()), new Set(moves.map(moveToActionId)));
});

test('PUCT retains board-state history for every search-native evaluation', async () => {
  const start = parseFen(START_FEN);
  const seenInputs = [];
  const provider = {
    evaluate(input) {
      seenInputs.push(input);
      const { board, fen } = currentBoardAndFen(input);
      const priors = legalPolicyPriors(board, new Float32Array(1858), 1);
      return { fen, wdl: [0.3, 0.4, 0.3], q: 0, mlh: 0, legalPriors: priors, bestMove: priors[0]?.uci };
    },
    async evaluateBatch(inputs) { return Promise.all(inputs.map((input) => this.evaluate(input))); },
  };
  const searcher = new Lc0PuctSearcher(provider);
  await searcher.search({ positions: [start] }, { visits: 3, batchSize: 1 });

  assert.ok(seenInputs.length >= 2);
  assert.ok(seenInputs.every((input) => 'prepared' in input));
  const childInput = seenInputs.find((input) => input.positions.length > 1);
  assert.ok(childInput, 'a child evaluation received history');
  assert.ok(childInput.positions.every((position) => typeof position === 'object'));
});
