import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { test } from 'node:test';
import { parseFen } from '../src/chess/board.ts';
import { moveToUci } from '../src/chess/moveCodec.ts';
import { legalMoves } from '../src/chess/movegen.ts';
import { Lc0OnnxEvaluator } from '../src/lc0/onnxEvaluator.ts';

const MODEL = '../models/lc0-bestnets/onnx/t1-256x10-distilled-swa-2432500.batch1.f32.onnx';
const fixtures = JSON.parse(readFileSync('fixtures/lc0/fen_only.json', 'utf8'));

const EXPECTED_BEST = new Map([
  ['startpos', 'd2d4'],
  ['black-to-move-after-e4', 'e7e5'],
  ['castling-rights', 'h1h8'],
  ['en-passant-available', 'd2d4'],
  ['promotion-near', 'a7a8q'],
  ['rule50-nonzero', 'h1h7'],
]);

test('LC0 f32 ONNX evaluator runs every FEN-only fixture with normalized legal priors', { skip: !existsSync(MODEL) && `missing ${MODEL}` }, async () => {
  const evaluator = await Lc0OnnxEvaluator.create(readFileSync(MODEL));
  for (const fixture of fixtures) {
    const evaluation = await evaluator.evaluate(fixture.fen);
    const legal = new Set(legalMoves(parseFen(fixture.fen)).map(moveToUci));
    const expectedBest = EXPECTED_BEST.get(fixture.id);
    if (expectedBest) assert.equal(evaluation.bestMove, expectedBest, fixture.id);
    assert.equal(legal.has(evaluation.bestMove), true, `${fixture.id} best move legal`);
    assert.equal(evaluation.legalPriors.length, legal.size, `${fixture.id} legal prior count`);
    assert.equal(evaluation.legalPriors.every((entry) => legal.has(entry.uci)), true, `${fixture.id} all priors legal`);
    assert.ok(Math.abs(evaluation.legalPriors.reduce((acc, entry) => acc + entry.prior, 0) - 1) < 1e-5, `${fixture.id} policy normalization`);
    assert.ok(Math.abs(evaluation.wdl.reduce((acc, value) => acc + value, 0) - 1) < 1e-5, `${fixture.id} WDL normalization`);
    assert.ok(Number.isFinite(evaluation.q), `${fixture.id} q finite`);
    assert.ok(Number.isFinite(evaluation.mlh), `${fixture.id} mlh finite`);
  }
});
