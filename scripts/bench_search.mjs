import fs from 'node:fs';
import { performance } from 'node:perf_hooks';
import { parseFen, START_FEN } from '../src/chess/board.ts';
import { StudentEvaluator } from '../src/nn/studentEvaluator.ts';
import { searchRoot } from '../src/search/puct.ts';
import { moveToUci } from '../src/chess/moveCodec.ts';

const artifactPath = process.argv[2] ?? 'artifacts/student_distill_benchmark.json';
const visits = Number(process.argv[3] ?? 64);
const repetitions = Number(process.argv[4] ?? 1);
const fen = process.argv[5] ?? START_FEN;

const artifact = JSON.parse(fs.readFileSync(artifactPath, 'utf8'));
const evaluator = new StudentEvaluator(artifact);
const board = parseFen(fen);

// Warm up V8 and populate evaluator feature cache for the root only; child positions are
// still generated/evaluated by each measured search.
await evaluator.evaluate(board);
await searchRoot(board, evaluator, { visits: Math.min(4, visits), temperature: 0 });

const t0 = performance.now();
let totalVisits = 0;
let bestMove = null;
for (let i = 0; i < repetitions; i++) {
  const result = await searchRoot(board, evaluator, { visits, temperature: 0 });
  totalVisits += result.visits;
  bestMove = result.move;
}
const seconds = Math.max((performance.now() - t0) / 1000, 1e-9);
console.log(`best_move=${bestMove ? moveToUci(bestMove) : 'none'}`);
console.log(`METRIC ts_student_search_repetitions=${repetitions}`);
console.log(`METRIC ts_student_search_visits=${totalVisits}`);
console.log(`METRIC ts_student_search_seconds=${seconds.toFixed(6)}`);
console.log(`METRIC ts_student_visits_per_second=${(totalVisits / seconds).toFixed(6)}`);
