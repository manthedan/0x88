import assert from 'node:assert/strict';
import test from 'node:test';
import { boardToFen, parseFen, START_FEN } from '../src/chess/board.ts';
import { legalMoves } from '../src/chess/movegen.ts';
import { analysisResultCacheKey } from '../src/lc0/analysisCache.ts';
import { analysisRunBroker, captureAnalysisRun } from '../src/lc0/analysisRunSnapshot.ts';

function inputs() {
  const rows = [
    { family: 'lc0', variant: 'small', strength: 32 },
    { family: 'lc0', variant: 't3', strength: 64 },
    { family: 'sf', variant: 'lite', strength: 12 },
  ];
  return { fen: START_FEN, positions: [parseFen(START_FEN)], cpuPosition: { fen: START_FEN, moves: [] }, rows,
    multiPv: 3, runtime: 'onnx', backendGeneration: 7, servedModels: { small: '/small.onnx', t3: '/t3.splitk.onnx' },
    broker: { ...analysisRunBroker(rows, { dial: 'balanced', cpuBudget: 2 }) } };
}

test('snapshot keys preserve existing CPU, small and big-net cache identities', () => {
  const input = inputs(), run = captureAnalysisRun(input);
  const runtimes = ['onnx|/small.onnx', '/t3.splitk.onnx', ''];
  assert.deepEqual(run.rowKeys, input.rows.map((row, i) => analysisResultCacheKey(START_FEN, [START_FEN], row, {
    multiPv: 3, runtime: runtimes[i], cpu: i === 2 ? JSON.stringify(['balanced', 2, ['sf:lite']]) : '',
  })));
  assert.equal(run.selectionKey, run.rowKeys.join('\u0001'));
  assert.equal(run.backendGeneration, 7);
  const history = captureAnalysisRun({ ...input, positions: [...input.positions, parseFen(START_FEN)] });
  assert.notEqual(run.positionKey, history.positionKey, 'same FEN with different history is a different position');
  assert.notDeepEqual(run.rowKeys, history.rowKeys);
});

test('controls, maps and scratch-board mutations cannot change a captured run or its coverage', () => {
  const input = inputs(), run = captureAnalysisRun(input), original = JSON.stringify(run);
  input.rows[0].strength = 1000;
  input.rows.pop();
  input.multiPv = 1;
  input.runtime = 'hybrid-wgsl-heads';
  input.backendGeneration++;
  input.servedModels.t3 = '';
  input.broker.dial = 'eco';
  input.broker.cpuBudget = 1;
  input.cpuPosition.moves.push('e2e4');
  input.positions[0].squares[0] = null;
  assert.equal(JSON.stringify(run), original);
  assert.throws(() => { run.rows[0].strength++; }, TypeError);
  assert.throws(() => { run.positions[0].squares[0] = null; }, TypeError);
  assert.throws(() => { run.cpuPosition.moves.push('e2e4'); }, TypeError);
  const scratch = run.positions.map((board) => ({ ...board, squares: [...board.squares] }));
  assert.equal(legalMoves(scratch[0]).length, 20, 'search scratch boards still support mutable move generation');
  assert.equal(boardToFen(scratch[0]), START_FEN);
  assert.equal(JSON.stringify(run), original);
});
