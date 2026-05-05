#!/usr/bin/env node
import { readFileSync } from 'node:fs';
import { performance } from 'node:perf_hooks';
import { parseFen, START_FEN } from '../src/chess/board.ts';
import { legalMoves, makeMove } from '../src/chess/movegen.ts';
import { moveToActionId, moveToUci } from '../src/chess/moveCodec.ts';
import { chooseMove } from '../src/search/puct.ts';
import { StudentEvaluator } from '../src/nn/studentEvaluator.ts';

const modelPath = process.env.TINY_LEELA_STUDENT_MODEL ?? 'artifacts/student_distill_benchmark.json';
const evaluator = StudentEvaluator.fromJson(readFileSync(modelPath, 'utf8'));
const fens = [
  START_FEN,
  'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1',
  'rnbqkb1r/pp1ppppp/5n2/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq c6 0 3',
  'rnbqkb1r/pp2pppp/2p2n2/3p4/2PP4/4PN2/PP3PPP/RNBQKB1R w KQkq - 0 5',
];

let legalSelections = 0;
let evaluatedPositions = 0;
let priorMass = 0;
let pliesCompleted = 0;
const t0 = performance.now();

for (const fen of fens) {
  const board = parseFen(fen);
  const legal = legalMoves(board);
  if (!legal.length) continue;
  const evaluation = evaluator.evaluate(board);
  priorMass += legal.reduce((sum, move) => sum + (evaluation.policy.get(moveToActionId(move)) ?? 0), 0);
  const result = await chooseMove(board, evaluator, { visits: 1 });
  if (result.move && legal.some((move) => moveToUci(move) === moveToUci(result.move))) legalSelections++;
  evaluatedPositions++;
}

let board = parseFen(START_FEN);
for (let i = 0; i < 12; i++) {
  const result = await chooseMove(board, evaluator, { visits: 1 });
  if (!result.move) break;
  board = makeMove(board, result.move);
  pliesCompleted++;
}
const elapsed = Math.max(performance.now() - t0, 1e-9);

console.log(`METRIC playable_positions=${evaluatedPositions}`);
console.log(`METRIC legal_move_selection_rate=${(legalSelections / evaluatedPositions).toFixed(6)}`);
console.log(`METRIC legal_policy_mass=${(priorMass / evaluatedPositions).toFixed(6)}`);
console.log(`METRIC selfplay_plies_completed=${pliesCompleted}`);
console.log(`METRIC evaluator_positions_per_second=${(((evaluatedPositions + pliesCompleted) * 1000) / elapsed).toFixed(6)}`);
console.log('METRIC playable_shell_ready=1');
