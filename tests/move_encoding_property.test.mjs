import test from 'node:test';
import assert from 'node:assert/strict';
import { parseFen, boardToFen } from '../src/chess/board.ts';
import { legalMoves, makeMove } from '../src/chess/movegen.ts';
import { moveFromUci, moveToActionId, moveToUci } from '../src/chess/moveCodec.ts';

function rng(seed) {
  let x = seed >>> 0;
  return () => {
    x = (1664525 * x + 1013904223) >>> 0;
    return x / 2 ** 32;
  };
}

const seedFens = [
  'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
  'r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1',
  '8/8/8/3pP3/8/8/8/4K2k w - d6 0 1',
  '4k3/P6P/8/8/8/8/p6p/4K3 w - - 0 1',
];

test('legal moves roundtrip through UCI and have unique action ids across deterministic random walks', () => {
  let checkedPositions = 0;
  let checkedMoves = 0;
  const seenFens = new Set();
  for (let seed = 1; seed <= 12; seed++) {
    const random = rng(seed);
    let board = parseFen(seedFens[seed % seedFens.length]);
    for (let ply = 0; ply < 24; ply++) {
      const fen = boardToFen(board);
      if (!seenFens.has(fen)) {
        seenFens.add(fen);
        checkedPositions++;
        const moves = legalMoves(board);
        const ids = new Set();
        for (const move of moves) {
          const uci = moveToUci(move);
          const decoded = moveFromUci(uci);
          assert.equal(decoded.from, move.from, `UCI from-square roundtrip failed for ${uci} in ${fen}`);
          assert.equal(decoded.to, move.to, `UCI to-square roundtrip failed for ${uci} in ${fen}`);
          assert.equal(decoded.promotion ?? '', move.promotion ?? '', `UCI promotion roundtrip failed for ${uci} in ${fen}`);
          assert.equal(moveToUci(decoded), uci, `UCI string roundtrip failed for ${uci} in ${fen}`);
          const actionId = moveToActionId(move);
          assert.ok(!ids.has(actionId), `duplicate action id ${actionId} for ${uci} in ${fen}`);
          ids.add(actionId);
          checkedMoves++;
        }
      }
      const moves = legalMoves(board);
      if (!moves.length) break;
      board = makeMove(board, moves[Math.floor(random() * moves.length)]);
    }
  }
  assert.ok(checkedPositions >= 50, `expected broad random-walk coverage, checked ${checkedPositions}`);
  assert.ok(checkedMoves >= 1000, `expected many legal moves, checked ${checkedMoves}`);
});
