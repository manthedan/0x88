import { execFileSync } from 'node:child_process';
import { performance } from 'node:perf_hooks';
import { parseFen } from '../src/chess/board.ts';
import { legalMoves, makeMove } from '../src/chess/movegen.ts';

const rustBin = 'rust/tiny_leela_core/target/release/tiny-leela-rust-eval';
execFileSync('cargo', [
  'build', '--release', '--quiet',
  '--manifest-path', 'rust/tiny_leela_core/Cargo.toml',
  '--bin', 'tiny-leela-rust-eval',
], { stdio: 'inherit' });

const cases = [
  { name: 'startpos_d4', fen: 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1', depth: 4 },
  { name: 'kiwipete_d3', fen: 'r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1', depth: 3 },
  { name: 'ep_edge_d4', fen: '8/8/8/3pP3/8/8/8/4K2k w - d6 0 1', depth: 4 },
  { name: 'promotions_d3', fen: '4k3/P6P/8/8/8/8/p6p/4K3 w - - 0 1', depth: 3 },
];

function perft(board, depth) {
  if (depth === 0) return 1;
  let nodes = 0;
  for (const mv of legalMoves(board)) nodes += perft(makeMove(board, mv), depth - 1);
  return nodes;
}

function rustPerft(fen, depth) {
  const out = execFileSync(rustBin, ['--perft', String(depth), '--fen', fen], { encoding: 'utf8' });
  const nodes = Number(out.match(/^nodes=(\d+)$/m)?.[1] ?? NaN);
  const seconds = Number(out.match(/^METRIC rust_perft_seconds=([0-9.]+)$/m)?.[1] ?? NaN);
  return { nodes, seconds };
}

let tsNodesTotal = 0;
let tsSecondsTotal = 0;
let rustNodesTotal = 0;
let rustSecondsTotal = 0;
for (const c of cases) {
  const board = parseFen(c.fen);
  const t0 = performance.now();
  const tsNodes = perft(board, c.depth);
  const tsSeconds = Math.max((performance.now() - t0) / 1000, 1e-9);
  const rust = rustPerft(c.fen, c.depth);
  const match = tsNodes === rust.nodes;
  tsNodesTotal += tsNodes;
  tsSecondsTotal += tsSeconds;
  rustNodesTotal += rust.nodes;
  rustSecondsTotal += rust.seconds;
  console.log(JSON.stringify({
    case: c.name,
    depth: c.depth,
    nodes: tsNodes,
    match,
    tsSeconds: Number(tsSeconds.toFixed(6)),
    rustSeconds: Number(rust.seconds.toFixed(6)),
    tsNodesPerSecond: Math.round(tsNodes / tsSeconds),
    rustNodesPerSecond: Math.round(rust.nodes / rust.seconds),
    rustSpeedup: Number((tsSeconds / rust.seconds).toFixed(3)),
  }));
  if (!match) process.exitCode = 1;
}
console.log(`METRIC rust_ts_perft_cases=${cases.length}`);
console.log(`METRIC ts_perft_nodes_per_second=${(tsNodesTotal / tsSecondsTotal).toFixed(3)}`);
console.log(`METRIC rust_perft_nodes_per_second=${(rustNodesTotal / rustSecondsTotal).toFixed(3)}`);
console.log(`METRIC rust_perft_speedup=${(tsSecondsTotal / rustSecondsTotal).toFixed(3)}`);
