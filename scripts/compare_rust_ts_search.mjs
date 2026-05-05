import fs from 'node:fs';
import { execFileSync } from 'node:child_process';
import { parseFen } from '../src/chess/board.ts';
import { StudentEvaluator } from '../src/nn/studentEvaluator.ts';
import { searchRoot } from '../src/search/puct.ts';
import { moveToUci } from '../src/chess/moveCodec.ts';

const artifactPath = process.argv[2] ?? 'artifacts/student_distill_benchmark.json';
const visitsList = (process.argv[3] ?? '1,2,4,8,16,64').split(',').map(Number).filter(Boolean);
const artifact = JSON.parse(fs.readFileSync(artifactPath, 'utf8'));
const evaluator = new StudentEvaluator(artifact);

const fens = [
  'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
  'rnbqkbnr/pppp1ppp/4p3/8/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 2',
  'r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1',
  '8/8/8/3pP3/8/8/8/4K2k w - d6 0 1',
  'k3r3/8/8/8/8/8/8/4K3 w - - 0 1',
  '7k/6Q1/6K1/8/8/8/8/8 b - - 0 1',
];

function runRust(fen, visits) {
  const out = execFileSync('cargo', [
    'run', '--release', '--quiet',
    '--manifest-path', 'rust/tiny_leela_core/Cargo.toml',
    '--bin', 'tiny-leela-rust-eval', '--', artifactPath, fen, String(visits),
  ], { encoding: 'utf8', maxBuffer: 1024 * 1024 * 16 });
  return {
    bestMove: out.match(/^best_move=(.*)$/m)?.[1]?.trim() ?? 'none',
    wdl: (out.match(/^wdl=([0-9eE+.,-]+)$/m)?.[1] ?? '').split(',').map(Number),
    legalCount: Number(out.match(/^policy_legal_count=(\d+)$/m)?.[1] ?? NaN),
    visitsPerSecond: Number(out.match(/^METRIC rust_student_visits_per_second=([0-9.]+)$/m)?.[1] ?? NaN),
  };
}

let cases = 0;
let wdlMaxAbsError = 0;
let legalCountMatches = 0;
let bestMoveMatches = 0;
let rustVpsSum = 0;
let tsVpsSum = 0;
const mismatches = [];

for (const fen of fens) {
  const board = parseFen(fen);
  const tsEval = await evaluator.evaluate(board);
  for (const visits of visitsList) {
    const t0 = performance.now();
    const tsSearch = await searchRoot(board, evaluator, { visits, temperature: 0 });
    const tsSeconds = Math.max((performance.now() - t0) / 1000, 1e-9);
    const rust = runRust(fen, visits);
    const tsBest = tsSearch.move ? moveToUci(tsSearch.move) : 'none';
    const legalMatch = rust.legalCount === tsEval.policy.size;
    const bestMatch = rust.bestMove === tsBest;
    const wdlErr = Math.max(...tsEval.wdl.map((v, i) => Math.abs(v - rust.wdl[i])));
    cases++;
    if (legalMatch) legalCountMatches++;
    if (bestMatch) bestMoveMatches++;
    wdlMaxAbsError = Math.max(wdlMaxAbsError, wdlErr);
    rustVpsSum += rust.visitsPerSecond;
    tsVpsSum += tsSearch.visits / tsSeconds;
    if (!bestMatch) mismatches.push({ fen, visits, tsBest, rustBest: rust.bestMove, wdlErr });
  }
}

const legalRate = legalCountMatches / cases;
const bestRate = bestMoveMatches / cases;
console.log(`METRIC parity_cases=${cases}`);
console.log(`METRIC parity_wdl_max_abs_error=${wdlMaxAbsError.toExponential(6)}`);
console.log(`METRIC parity_legal_count_match_rate=${legalRate.toFixed(6)}`);
console.log(`METRIC parity_best_move_match_rate=${bestRate.toFixed(6)}`);
console.log(`METRIC parity_ts_avg_visits_per_second=${(tsVpsSum / cases).toFixed(6)}`);
console.log(`METRIC parity_rust_avg_visits_per_second=${(rustVpsSum / cases).toFixed(6)}`);
if (mismatches.length) {
  console.log('MISMATCHES');
  for (const m of mismatches) console.log(JSON.stringify(m));
}
if (wdlMaxAbsError > 1e-5 || legalRate !== 1) process.exit(1);
