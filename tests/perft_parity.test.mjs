import assert from 'node:assert/strict';
import { test } from 'node:test';
import { parseFen } from '../src/chess/board.ts';
import { legalMoves, makeMove } from '../src/chess/movegen.ts';

function perft(board, depth) {
  if (depth === 0) return 1;
  let nodes = 0;
  for (const move of legalMoves(board)) nodes += perft(makeMove(board, move), depth - 1);
  return nodes;
}

const cases = [
  {
    name: 'startpos d5',
    fen: 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
    depth: 5,
    expected: 4865609,
  },
  {
    name: 'kiwipete castling/check pins d4',
    fen: 'r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1',
    depth: 4,
    expected: 4085603,
  },
  {
    name: 'en-passant capture edge',
    fen: '8/8/8/3pP3/8/8/8/4K2k w - d6 0 1',
    depth: 3,
  },
  {
    name: 'promotion candidates',
    fen: '4k3/P6P/8/8/8/8/p6p/4K3 w - - 0 1',
    depth: 2,
  },
];

test('TypeScript perft matches known reference nodes', () => {
  for (const c of cases.filter((item) => item.expected !== undefined)) {
    assert.equal(perft(parseFen(c.fen), c.depth), c.expected, c.name);
  }
});