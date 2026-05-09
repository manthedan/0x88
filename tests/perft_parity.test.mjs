import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
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
    name: 'startpos',
    fen: 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
    depth: 3,
    expected: 8902,
  },
  {
    name: 'kiwipete castling/check pins',
    fen: 'r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1',
    depth: 2,
    expected: 2039,
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

test('Rust and TypeScript perft agree on representative FENs', { skip: process.env.TINY_LEELA_RUN_RUST_PERFT !== '1' }, () => {
  for (const c of cases) {
    const tsNodes = perft(parseFen(c.fen), c.depth);
    const stdout = execFileSync('cargo', [
      'run', '--quiet', '--manifest-path', 'rust/tiny_leela_core/Cargo.toml', '--bin', 'tiny-leela-rust-eval', '--',
      '--perft', String(c.depth), '--fen', c.fen,
    ], { encoding: 'utf8' });
    const match = /^nodes=(\d+)$/m.exec(stdout);
    assert.ok(match, `missing rust nodes output for ${c.name}: ${stdout}`);
    assert.equal(Number(match[1]), tsNodes, c.name);
  }
});
