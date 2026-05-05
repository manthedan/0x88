#!/usr/bin/env node
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname } from 'node:path';
import { parseFen, START_FEN, boardToFen } from '../src/chess/board.ts';
import { inCheck, legalMoves, makeMove } from '../src/chess/movegen.ts';
import { moveToUci } from '../src/chess/moveCodec.ts';
import { searchRoot } from '../src/search/puct.ts';
import { StudentEvaluator } from '../src/nn/studentEvaluator.ts';
import { rustPolicyForBoard } from './rust_engine.mjs';

function arg(name, fallback = undefined) {
  const prefix = `${name}=`;
  const inline = process.argv.find((value) => value.startsWith(prefix));
  if (inline) return inline.slice(prefix.length);
  const i = process.argv.indexOf(name);
  return i >= 0 ? process.argv[i + 1] : fallback;
}

function makeRng(seed) {
  let s = seed >>> 0;
  return () => {
    s = (1664525 * s + 1013904223) >>> 0;
    return s / 2 ** 32;
  };
}

function choose(policy, rng) {
  let r = rng();
  for (const entry of policy) {
    r -= entry.probability;
    if (r <= 0) return entry.move;
  }
  return policy.at(-1)?.move ?? null;
}

function terminalWhiteScore(board) {
  const legal = legalMoves(board);
  if (legal.length) return null;
  if (!inCheck(board)) return 0.5;
  return board.turn === 'w' ? 0 : 1;
}

function resultForTurn(whiteScore, turn) {
  if (whiteScore === 0.5) return [0, 1, 0];
  const sideWon = (whiteScore === 1 && turn === 'w') || (whiteScore === 0 && turn === 'b');
  return sideWon ? [1, 0, 0] : [0, 0, 1];
}

async function adjudicatedWhiteScore(board) {
  const terminal = terminalWhiteScore(board);
  if (terminal !== null) return { score: terminal, adjudicated: false };
  if (adjudicate !== 'value') return { score: 0.5, adjudicated: false };
  const evaln = await evaluator.evaluate(board);
  const sideValue = evaln.wdl[0] - evaln.wdl[2];
  if (Math.abs(sideValue) <= adjudicateThreshold) return { score: 0.5, adjudicated: true };
  const sideToMoveWins = sideValue > 0;
  const whiteWins = (board.turn === 'w' && sideToMoveWins) || (board.turn === 'b' && !sideToMoveWins);
  return { score: whiteWins ? 1 : 0, adjudicated: true };
}

const modelPath = arg('--model', 'artifacts/student_distill_benchmark.json');
const backend = arg('--backend', process.env.TINY_LEELA_BACKEND ?? 'rust');
const outPath = arg('--out', 'data/selfplay/bootstrap.jsonl');
const games = Number(arg('--games', '2'));
const visits = Number(arg('--visits', '4'));
const maxPlies = Number(arg('--max-plies', '40'));
const temperature = Number(arg('--temperature', '1'));
const seed = Number(arg('--seed', '1'));
const adjudicate = arg('--adjudicate', 'terminal');
const adjudicateThreshold = Number(arg('--adjudicate-threshold', '0.02'));
const evaluator = StudentEvaluator.fromJson(readFileSync(modelPath, 'utf8'));
const rng = makeRng(seed);
const rows = [];
let completedGames = 0;
let decisiveGames = 0;
let totalPlies = 0;
let policyMass = 0;
let adjudicatedGames = 0;

for (let game = 0; game < games; game++) {
  let board = parseFen(START_FEN);
  const pending = [];
  let whiteScore = null;
  for (let ply = 0; ply < maxPlies; ply++) {
    whiteScore = terminalWhiteScore(board);
    if (whiteScore !== null) break;
    const result = backend === 'ts'
      ? await searchRoot(board, evaluator, { visits, temperature })
      : rustPolicyForBoard(board, { model: modelPath, visits, temperature });
    if (!result.move || !result.policy.length) { whiteScore = 0.5; break; }
    const policy = Object.fromEntries(result.policy.filter((entry) => entry.probability > 0).map((entry) => [moveToUci(entry.move), Number(entry.probability.toFixed(8))]));
    pending.push({
      game_id: `g${String(game).padStart(6, '0')}`,
      ply,
      fen: boardToFen(board),
      turn: board.turn,
      visits: result.visits,
      policy,
      root_value: Number(((result.value ?? (result.wdl?.[0] - result.wdl?.[2]) ?? 0)).toFixed(8)),
    });
    policyMass += Object.values(policy).reduce((a, b) => a + b, 0);
    const move = choose(result.policy, rng);
    if (!move) { whiteScore = 0.5; break; }
    board = makeMove(board, move);
    totalPlies++;
  }
  if (whiteScore === null) {
    const adjudicated = await adjudicatedWhiteScore(board);
    whiteScore = adjudicated.score;
    if (adjudicated.adjudicated) adjudicatedGames++;
  }
  if (whiteScore !== 0.5) decisiveGames++;
  for (const row of pending) rows.push({ ...row, result: resultForTurn(whiteScore, row.turn), white_score: whiteScore });
  completedGames++;
}

mkdirSync(dirname(outPath), { recursive: true });
writeFileSync(outPath, rows.map((row) => JSON.stringify(row)).join('\n') + (rows.length ? '\n' : ''));

console.log(`METRIC selfplay_backend_${backend}=1`);
console.log(`METRIC selfplay_games=${completedGames}`);
console.log(`METRIC selfplay_positions=${rows.length}`);
console.log(`METRIC selfplay_avg_plies=${(totalPlies / Math.max(1, completedGames)).toFixed(6)}`);
console.log(`METRIC selfplay_decisive_rate=${(decisiveGames / Math.max(1, completedGames)).toFixed(6)}`);
console.log(`METRIC selfplay_adjudicated_rate=${(adjudicatedGames / Math.max(1, completedGames)).toFixed(6)}`);
console.log(`METRIC selfplay_policy_mass=${(policyMass / Math.max(1, rows.length)).toFixed(6)}`);
console.log(`METRIC selfplay_output_bytes=${Buffer.byteLength(rows.map((row) => JSON.stringify(row)).join('\n'))}`);
