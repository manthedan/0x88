import { performance } from 'node:perf_hooks';
import { parseFen, START_FEN } from '../src/chess/board.ts';
import { legalMoves } from '../src/chess/movegen.ts';

const iterations = Number(process.argv[2] ?? 100_000);
const board = parseFen(START_FEN);

// Warm up V8/JIT.
for (let i = 0; i < Math.min(10_000, iterations); i++) legalMoves(board);

const t0 = performance.now();
let nodes = 0;
for (let i = 0; i < iterations; i++) {
  nodes += legalMoves(board).length;
}
const elapsed = Math.max((performance.now() - t0) / 1000, 1e-9);
console.log(`METRIC ts_legal_movegen_iterations=${iterations}`);
console.log(`METRIC ts_legal_moves_total=${nodes}`);
console.log(`METRIC ts_legal_movegen_positions_per_second=${(iterations / elapsed).toFixed(6)}`);
