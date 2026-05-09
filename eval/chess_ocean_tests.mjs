#!/usr/bin/env node
import assert from 'node:assert/strict';
import { parseFen, boardToFen, START_FEN } from '../src/chess/board.ts';
import { legalMoves, makeMove, inCheck } from '../src/chess/movegen.ts';
import { moveFromUci, moveToActionId, moveToUci } from '../src/chess/moveCodec.ts';
import { searchRoot } from '../src/search/puct.ts';

function moveEq(a, b) { return a.from === b.from && a.to === b.to && (a.promotion ?? '') === (b.promotion ?? ''); }
function legalByUci(board, uci) {
  const want = moveFromUci(uci);
  const got = legalMoves(board).find((m) => moveEq(m, want));
  assert.ok(got, `${uci} should be legal in ${boardToFen(board)}`);
  return got;
}
function legalUcis(board) { return new Set(legalMoves(board).map(moveToUci)); }
function policyFor(board, entries) {
  const out = new Map();
  for (const [uci, p] of Object.entries(entries)) out.set(moveToActionId(legalByUci(board, uci)), p);
  return out;
}
function evaluator(fn) { return { async evaluate(board, opts = {}) { return fn(board, opts); } }; }

function testOpeningLegalCountAndRoundtrip() {
  const board = parseFen(START_FEN);
  const moves = legalMoves(board).map(moveToUci).sort();
  assert.equal(moves.length, 20);
  assert.ok(moves.includes('e2e4'));
  assert.equal(moveToUci(moveFromUci('e7e8q')), 'e7e8q');
}

function testCastlingRoundtrip() {
  const board = parseFen('r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1');
  const moves = legalUcis(board);
  assert.ok(moves.has('e1g1'));
  assert.ok(moves.has('e1c1'));
  const next = makeMove(board, legalByUci(board, 'e1g1'));
  assert.equal(next.squares[6], 'wk');
  assert.equal(next.squares[5], 'wr');
  assert.equal(next.castling.includes('K'), false);
}

function testEnPassantRoundtrip() {
  const board = parseFen('8/8/8/3pP3/8/8/8/4K2k w - d6 0 1');
  assert.ok(legalUcis(board).has('e5d6'));
  const next = makeMove(board, legalByUci(board, 'e5d6'));
  assert.equal(next.squares[43], 'wp'); // d6
  assert.equal(next.squares[35], null); // d5 captured en-passant
}

function testPromotionRoundtrip() {
  const board = parseFen('4k3/P7/8/8/8/8/8/4K3 w - - 0 1');
  const moves = legalUcis(board);
  for (const uci of ['a7a8q', 'a7a8r', 'a7a8b', 'a7a8n']) assert.ok(moves.has(uci), `${uci} promotion legal`);
  const next = makeMove(board, legalByUci(board, 'a7a8q'));
  assert.equal(next.squares[56], 'wq');
}

async function testTerminalAndValuePerspective() {
  const ev = evaluator((board) => {
    const ucis = legalMoves(board).map(moveToUci);
    const rootPolicy = ucis.includes('e2e4') ? { e2e4: 1.0 } : Object.fromEntries(ucis.slice(0, 1).map((uci) => [uci, 1.0]));
    return { policy: policyFor(board, rootPolicy), wdl: [0.9, 0.0, 0.1] };
  });
  const r = await searchRoot(parseFen(START_FEN), ev, { visits: 1, temperature: 0 });
  const edge = r.policy.find((x) => moveToUci(x.move) === 'e2e4');
  assert.ok(edge);
  assert.ok(Math.abs(edge.q + 0.8) < 1e-9, `child value should flip on backup, got q=${edge.q}`);

  const mate = parseFen('rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3');
  assert.equal(legalMoves(mate).length, 0);
  assert.equal(inCheck(mate), true);
  const stale = parseFen('7k/5Q2/6K1/8/8/8/8/8 b - - 0 1');
  assert.equal(legalMoves(stale).length, 0);
  assert.equal(inCheck(stale), false);
}

async function testRootPriorParityAndBatchSafety() {
  const calls = { batch: 0, boards: 0 };
  const ev = {
    async evaluate(board) { return { policy: policyFor(board, { g1f3: 0.7, e2e4: 0.3 }), wdl: [0.5, 0, 0.5] }; },
    async evaluateBatch(boards) { calls.batch++; calls.boards += boards.length; return boards.map((b) => {
      const ucis = legalMoves(b).map(moveToUci);
      const entries = Object.fromEntries(ucis.slice(0, Math.min(3, ucis.length)).map((uci, i) => [uci, 1 / (i + 1)]));
      return { policy: policyFor(b, entries), wdl: [0.5, 0, 0.5] };
    }); },
  };
  const r = await searchRoot(parseFen(START_FEN), ev, { visits: 8, batchSize: 4, temperature: 0 });
  assert.ok(calls.batch > 0);
  assert.equal(r.visits, 8);
  assert.ok(Math.abs(r.policy.reduce((s, e) => s + e.prior, 0) - 1) < 1e-9);
}

const tests = [
  testOpeningLegalCountAndRoundtrip,
  testCastlingRoundtrip,
  testEnPassantRoundtrip,
  testPromotionRoundtrip,
  testTerminalAndValuePerspective,
  testRootPriorParityAndBatchSafety,
];
for (const t of tests) {
  await t();
  console.log(`ok ${t.name}`);
}
console.log(`METRIC chess_ocean_tests=${tests.length}`);
