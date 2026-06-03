import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { test } from 'node:test';
import { Lc0OnnxEvaluator } from '../src/lc0/onnxEvaluator.ts';
import { compareEvalSweeps, runEvalSweep, sweepFixtureInput } from '../src/lc0/driftSweep.ts';

const F32 = new URL('../../models/lc0-bestnets/onnx/t1-256x10-distilled-swa-2432500.batch1.f32.onnx', import.meta.url);
const F16 = new URL('../../models/lc0-bestnets/onnx/t1-256x10-distilled-swa-2432500.batch1.f16.onnx', import.meta.url);

const fenOnly = JSON.parse(readFileSync(new URL('../fixtures/lc0/fen_only.json', import.meta.url), 'utf8'));
const history = JSON.parse(readFileSync(new URL('../fixtures/lc0/history.json', import.meta.url), 'utf8'));
const suite = [...fenOnly, ...history];

test('sweepFixtureInput reconstructs fen-only and explicit-history inputs', () => {
  const fen = sweepFixtureInput(fenOnly[0]);
  assert.equal(typeof fen, 'string');
  const hist = sweepFixtureInput(history[0]);
  assert.ok(hist && typeof hist === 'object' && 'positions' in hist, 'history fixture yields positions input');
  assert.equal(hist.positions.length, history[0].moves.length + 1, 'history includes start + every played ply');
  assert.throws(() => sweepFixtureInput({ id: 'empty' }), /neither moves nor fen/);
});

test('f16/WASM stays within f32/WASM drift tolerances across both suites', { skip: (!existsSync(F32) || !existsSync(F16)) && 'missing ONNX models' }, async () => {
  const f32 = await Lc0OnnxEvaluator.create(readFileSync(F32));
  const f16 = await Lc0OnnxEvaluator.create(readFileSync(F16));

  const baseline = await runEvalSweep('f32/wasm', f32, suite);
  const candidate = await runEvalSweep('f16/wasm', f16, suite);
  assert.equal(baseline.records.length, suite.length);
  assert.ok(baseline.evalsPerSecond > 0 && candidate.evalsPerSecond > 0, 'records throughput');

  const cmp = compareEvalSweeps(baseline, candidate);
  // Tolerances calibrated from observed f16-vs-f32 drift on this net; best move
  // must be identical, the rest must stay tight.
  assert.equal(cmp.bestMoveMismatches, 0, `best-move mismatches: ${JSON.stringify(cmp.perFixture.filter((m) => !m.bestMoveMatch))}`);
  assert.ok(cmp.maxTopPriorDrift < 0.01, `maxTopPriorDrift ${cmp.maxTopPriorDrift}`);
  assert.ok(cmp.maxWdlDrift < 0.02, `maxWdlDrift ${cmp.maxWdlDrift}`);
  assert.ok(cmp.maxQDrift < 0.02, `maxQDrift ${cmp.maxQDrift}`);
  assert.ok(cmp.maxMlhDrift < 1.5, `maxMlhDrift ${cmp.maxMlhDrift}`);
});

test('compareEvalSweeps is a zero-drift identity against itself', async () => {
  // A tiny stub evaluator keeps this deterministic and model-free.
  const stub = {
    async evaluate() {
      return { wdl: [0.5, 0.3, 0.2], q: 0.3, mlh: 40, bestMove: 'e2e4', legalPriors: [{ uci: 'e2e4', prior: 0.6 }, { uci: 'd2d4', prior: 0.4 }] };
    },
  };
  const sweep = await runEvalSweep('stub', stub, fenOnly);
  const cmp = compareEvalSweeps(sweep, sweep);
  assert.equal(cmp.bestMoveMismatches, 0);
  assert.equal(cmp.maxWdlDrift, 0);
  assert.equal(cmp.maxTopPriorDrift, 0);
});
