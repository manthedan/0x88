#!/usr/bin/env node
import { readFileSync } from 'node:fs';
import { parseFen, START_FEN } from '../src/chess/board.ts';
import { inCheck, legalMoves, makeMove } from '../src/chess/movegen.ts';
import { moveToUci } from '../src/chess/moveCodec.ts';
import { chooseMove } from '../src/search/puct.ts';
import { StudentEvaluator } from '../src/nn/studentEvaluator.ts';
import { rustChooseMove } from '../scripts/rust_engine.mjs';

function arg(name, fallback = undefined) {
  const prefix = `${name}=`;
  const inline = process.argv.find((value) => value.startsWith(prefix));
  if (inline) return inline.slice(prefix.length);
  const i = process.argv.indexOf(name);
  return i >= 0 ? process.argv[i + 1] : fallback;
}

function terminalWhiteScore(board) {
  const legal = legalMoves(board);
  if (legal.length) return null;
  if (!inCheck(board)) return 0.5;
  return board.turn === 'w' ? 0 : 1;
}

function candidateScore(whiteScore, candidateColor) {
  if (whiteScore === 0.5) return 0.5;
  return (whiteScore === 1 && candidateColor === 'w') || (whiteScore === 0 && candidateColor === 'b') ? 1 : 0;
}

function elo(scoreRate) {
  const s = Math.min(0.999, Math.max(0.001, scoreRate));
  return 400 * Math.log10(s / (1 - s));
}

async function adjudicatedWhiteScore(board) {
  const terminal = terminalWhiteScore(board);
  if (terminal !== null) return { score: terminal, adjudicated: false };
  if (adjudicate !== 'value') return { score: 0.5, adjudicated: false };
  const evaln = await baseline.evaluate(board);
  const sideValue = evaln.wdl[0] - evaln.wdl[2];
  if (Math.abs(sideValue) <= adjudicateThreshold) return { score: 0.5, adjudicated: true };
  const sideToMoveWins = sideValue > 0;
  const whiteWins = (board.turn === 'w' && sideToMoveWins) || (board.turn === 'b' && !sideToMoveWins);
  return { score: whiteWins ? 1 : 0, adjudicated: true };
}

const backend = arg('--backend', process.env.TINY_LEELA_BACKEND ?? 'rust');
const candidatePath = arg('--candidate', process.env.TINY_LEELA_CANDIDATE_MODEL ?? 'artifacts/student_distill_benchmark.json');
const baselinePath = arg('--baseline', process.env.TINY_LEELA_BASELINE_MODEL ?? 'artifacts/student_distill_benchmark.json');
const games = Number(arg('--games', '4'));
const visits = Number(arg('--visits', '2'));
const maxPlies = Number(arg('--max-plies', '40'));
const adjudicate = arg('--adjudicate', 'terminal');
const adjudicateThreshold = Number(arg('--adjudicate-threshold', '0.02'));
const progressEvery = Math.max(1, Number(arg('--progress-every', process.env.TINY_LEELA_PROGRESS_EVERY ?? '4')));
const candidate = StudentEvaluator.fromJson(readFileSync(candidatePath, 'utf8'));
const baseline = StudentEvaluator.fromJson(readFileSync(baselinePath, 'utf8'));
const openings = [
  START_FEN,
  'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1',
  'rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1',
  'rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R b KQkq - 1 1',
];

let wins = 0, draws = 0, losses = 0, illegalLosses = 0, pliesTotal = 0, adjudicatedGames = 0;
const startedAt = Date.now();
const logProgress = (message) => process.stderr.write(`[arena backend=${backend} visits=${visits}] ${message}\n`);
logProgress(`start games=${games} max_plies=${maxPlies} candidate=${candidatePath} baseline=${baselinePath}`);

for (let game = 0; game < games; game++) {
  logProgress(`game ${game + 1}/${games} start candidate_color=${game % 2 === 0 ? 'w' : 'b'}`);
  let board = parseFen(openings[game % openings.length]);
  const candidateColor = game % 2 === 0 ? 'w' : 'b';
  let whiteScore = terminalWhiteScore(board);
  let plies = 0;
  for (; whiteScore === null && plies < maxPlies; plies++) {
    const sideIsCandidate = board.turn === candidateColor;
    const evaluator = sideIsCandidate ? candidate : baseline;
    const model = sideIsCandidate ? candidatePath : baselinePath;
    const legal = legalMoves(board).map(moveToUci);
    const result = backend === 'ts' ? await chooseMove(board, evaluator, { visits }) : rustChooseMove(board, { model, visits });
    if (!result.move) { whiteScore = inCheck(board) ? (board.turn === 'w' ? 0 : 1) : 0.5; break; }
    const uci = moveToUci(result.move);
    if (!legal.includes(uci)) {
      illegalLosses += sideIsCandidate ? 1 : 0;
      whiteScore = sideIsCandidate ? (candidateColor === 'w' ? 0 : 1) : (candidateColor === 'w' ? 1 : 0);
      break;
    }
    board = makeMove(board, result.move);
    whiteScore = terminalWhiteScore(board);
    if ((plies + 1) % progressEvery === 0) logProgress(`game ${game + 1}/${games} ply=${plies + 1}/${maxPlies} move=${uci} elapsed_s=${((Date.now() - startedAt) / 1000).toFixed(1)}`);
  }
  if (whiteScore === null) {
    const adjudicated = await adjudicatedWhiteScore(board);
    whiteScore = adjudicated.score;
    if (adjudicated.adjudicated) adjudicatedGames++;
  }
  const score = candidateScore(whiteScore, candidateColor);
  if (score === 1) wins++;
  else if (score === 0) losses++;
  else draws++;
  pliesTotal += plies;
  logProgress(`game ${game + 1}/${games} done score=${score} wdl=${wins}/${draws}/${losses} illegal=${illegalLosses} elapsed_s=${((Date.now() - startedAt) / 1000).toFixed(1)}`);
}

const scoreRate = (wins + 0.5 * draws) / Math.max(1, games);
const promotionReady = scoreRate > 0.55 && illegalLosses === 0 ? 1 : 0;
console.log(`METRIC arena_backend_${backend}=1`);
console.log(`METRIC arena_score_rate=${scoreRate.toFixed(6)}`);
console.log(`METRIC arena_candidate_elo_estimate=${elo(scoreRate).toFixed(6)}`);
console.log(`METRIC arena_games=${games}`);
console.log(`METRIC arena_wins=${wins}`);
console.log(`METRIC arena_draws=${draws}`);
console.log(`METRIC arena_losses=${losses}`);
console.log(`METRIC arena_illegal_losses=${illegalLosses}`);
console.log(`METRIC arena_adjudicated_rate=${(adjudicatedGames / Math.max(1, games)).toFixed(6)}`);
console.log(`METRIC arena_avg_plies=${(pliesTotal / Math.max(1, games)).toFixed(6)}`);
console.log(`METRIC arena_promotion_ready=${promotionReady}`);
