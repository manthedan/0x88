import fs from 'node:fs';
import { execFileSync } from 'node:child_process';
import { parseFen, boardToFen, START_FEN } from '../src/chess/board.ts';
import { legalMoves, makeMove, inCheck } from '../src/chess/movegen.ts';
import { moveFromUci, moveToUci } from '../src/chess/moveCodec.ts';
import { StudentEvaluator } from '../src/nn/studentEvaluator.ts';
import { searchRoot } from '../src/search/puct.ts';

const artifactPath = process.argv[2] ?? 'artifacts/student_distill_benchmark.json';
const tsVisits = Number(process.argv[3] ?? 8);
const rustVisits = Number(process.argv[4] ?? process.argv[3] ?? 8);
const maxPlies = Number(process.argv[5] ?? 30);
const artifact = JSON.parse(fs.readFileSync(artifactPath, 'utf8'));
const tsEvaluator = new StudentEvaluator(artifact);

execFileSync('cargo', ['build', '--release', '--quiet', '--manifest-path', 'rust/tiny_leela_core/Cargo.toml', '--bin', 'tiny-leela-rust-eval'], { stdio: 'inherit' });
const rustBin = 'rust/tiny_leela_core/target/release/tiny-leela-rust-eval';

const starts = [
  START_FEN,
  'rnbqkbnr/pppp1ppp/4p3/8/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 2',
  'rnbqkbnr/ppp2ppp/3pp3/8/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3',
  'r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1',
];

function rustMove(board) {
  const fen = boardToFen(board);
  const out = execFileSync(rustBin, [artifactPath, fen, String(rustVisits)], { encoding: 'utf8', maxBuffer: 1024 * 1024 * 16 });
  return out.match(/^best_move=(.*)$/m)?.[1]?.trim() ?? 'none';
}

async function tsMove(board) {
  const result = await searchRoot(board, tsEvaluator, { visits: tsVisits, temperature: 0 });
  return result.move ? moveToUci(result.move) : 'none';
}

async function choose(engine, board) {
  return engine === 'rust' ? rustMove(board) : await tsMove(board);
}

function material(board) {
  const vals = { p: 1, n: 3, b: 3, r: 5, q: 9, k: 0 };
  let score = 0;
  for (const piece of board.squares) if (piece) score += (piece[0] === 'w' ? 1 : -1) * vals[piece[1]];
  return score;
}

async function play(startFen, whiteEngine, blackEngine) {
  let board = parseFen(startFen);
  let illegalLoss = null;
  for (let ply = 0; ply < maxPlies; ply++) {
    const legal = legalMoves(board);
    if (!legal.length) {
      if (!inCheck(board)) return { scoreRust: 0.5, result: 'stalemate', plies: ply, illegalLoss };
      const loser = board.turn === 'w' ? whiteEngine : blackEngine;
      return { scoreRust: loser === 'rust' ? 0 : 1, result: 'checkmate', plies: ply, illegalLoss };
    }
    const engine = board.turn === 'w' ? whiteEngine : blackEngine;
    const uci = await choose(engine, board);
    let move;
    try { move = moveFromUci(uci); } catch { illegalLoss = engine; return { scoreRust: engine === 'rust' ? 0 : 1, result: 'illegal_parse', plies: ply, illegalLoss }; }
    const legalUcis = new Set(legal.map(moveToUci));
    if (!legalUcis.has(uci)) return { scoreRust: engine === 'rust' ? 0 : 1, result: 'illegal_move', plies: ply, illegalLoss: engine };
    board = makeMove(board, move);
  }
  const mat = material(board);
  // Max-ply adjudication only to avoid all games being uninformative draws.
  const whiteScore = mat > 0.5 ? 1 : mat < -0.5 ? 0 : 0.5;
  const rustIsWhite = whiteEngine === 'rust';
  return { scoreRust: rustIsWhite ? whiteScore : 1 - whiteScore, result: 'material_adjudication', plies: maxPlies, illegalLoss };
}

let games = 0, rustScore = 0, rustIllegalLosses = 0, tsIllegalLosses = 0;
const rows = [];
for (const start of starts) {
  for (const [white, black] of [['rust', 'ts'], ['ts', 'rust']]) {
    const game = await play(start, white, black);
    games++;
    rustScore += game.scoreRust;
    if (game.illegalLoss === 'rust') rustIllegalLosses++;
    if (game.illegalLoss === 'ts') tsIllegalLosses++;
    rows.push({ start, white, black, ...game });
  }
}

for (const row of rows) console.log(JSON.stringify(row));
console.log(`METRIC rust_ts_arena_ts_visits=${tsVisits}`);
console.log(`METRIC rust_ts_arena_rust_visits=${rustVisits}`);
console.log(`METRIC rust_ts_arena_games=${games}`);
console.log(`METRIC rust_ts_arena_rust_score_rate=${(rustScore / games).toFixed(6)}`);
console.log(`METRIC rust_ts_arena_rust_illegal_losses=${rustIllegalLosses}`);
console.log(`METRIC rust_ts_arena_ts_illegal_losses=${tsIllegalLosses}`);
const rustScoreRate = rustScore / games;
if (rustIllegalLosses || tsIllegalLosses || Math.abs(rustScoreRate - 0.5) > 1e-9) process.exit(1);
