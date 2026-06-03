import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { test } from 'node:test';
import { parseFen } from '../src/chess/board.ts';
import { inCheck, legalMoves } from '../src/chess/movegen.ts';
import { moveToUci } from '../src/chess/moveCodec.ts';
import { Lc0OnnxEvaluator } from '../src/lc0/onnxEvaluator.ts';
import { Lc0PuctSearcher } from '../src/lc0/search.ts';

const MODEL = '../models/lc0-bestnets/onnx/t1-256x10-distilled-swa-2432500.batch1.f32.onnx';
const FIXTURES = 'fixtures/lc0/search_edge_cases.json';
const VISITS = 24;

const fixtures = JSON.parse(readFileSync(FIXTURES, 'utf8'));

function legalUciSet(fen) {
  return new Set(legalMoves(parseFen(fen)).map(moveToUci));
}

// Movegen-only checks run everywhere; the model-backed search runs only when the
// ONNX artifact is present so the suite degrades gracefully without it.
test('edge-case fixtures have the expected terminal/legal structure', () => {
  for (const fx of fixtures) {
    const board = parseFen(fx.fen);
    const legal = legalMoves(board);
    if (fx.kind === 'terminal-checkmate') {
      assert.equal(legal.length, 0, `${fx.id} has no legal moves`);
      assert.equal(inCheck(board), true, `${fx.id} is in check (checkmate)`);
    } else if (fx.kind === 'terminal-stalemate') {
      assert.equal(legal.length, 0, `${fx.id} has no legal moves`);
      assert.equal(inCheck(board), false, `${fx.id} is not in check (stalemate)`);
    } else {
      assert.ok(legal.length > 0, `${fx.id} has legal moves`);
    }
    for (const uci of fx.requiredLegal ?? []) {
      assert.ok(legalUciSet(fx.fen).has(uci), `${fx.id}: ${uci} is legal`);
    }
    if (fx.kind === 'promotion') {
      assert.ok([...legalUciSet(fx.fen)].some((uci) => uci.length === 5), `${fx.id} has a promotion move`);
    }
  }
});

test('LC0 search handles edge-case positions', { skip: !existsSync(MODEL) && 'missing ONNX model' }, async () => {
  const evaluator = await Lc0OnnxEvaluator.create(readFileSync(MODEL));
  const searcher = new Lc0PuctSearcher(evaluator);

  for (const fx of fixtures) {
    const board = parseFen(fx.fen);
    const legal = new Set(legalMoves(board).map(moveToUci));
    const result = await searcher.search(fx.fen, { visits: VISITS });

    if (fx.kind.startsWith('terminal')) {
      // A position with no legal moves yields no move and no expansions to run.
      assert.equal(result.move ?? null, null, `${fx.id}: terminal search returns no move`);
      assert.deepEqual(result.children, [], `${fx.id}: terminal search has no children`);
      continue;
    }

    assert.ok(result.move, `${fx.id}: search returns a move`);
    assert.ok(legal.has(result.move), `${fx.id}: chosen move ${result.move} is legal`);
    assert.ok(result.pv.every((uci) => typeof uci === 'string' && uci.length >= 4), `${fx.id}: pv is well-formed`);

    // The evaluator must produce finite, normalized priors for the special moves
    // (promotion/castling/en-passant) so the policy map covers them.
    const ev = await evaluator.evaluate(fx.fen);
    assert.ok(ev.wdl.every(Number.isFinite), `${fx.id}: finite WDL`);
    for (const uci of fx.requiredLegal ?? []) {
      const prior = ev.legalPriors.find((p) => p.uci === uci);
      assert.ok(prior && Number.isFinite(prior.prior), `${fx.id}: ${uci} has a finite prior`);
    }

    if (fx.kind === 'mate-in-1' && fx.expectBestMove) {
      assert.equal(result.move, fx.expectBestMove, `${fx.id}: finds the mate ${fx.expectBestMove}`);
    }
    if (fx.kind === 'promotion') {
      assert.equal(result.move.length, 5, `${fx.id}: best move ${result.move} is a promotion`);
    }
  }
});
