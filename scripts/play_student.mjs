#!/usr/bin/env node
import { readFileSync } from 'node:fs';
import { createInterface } from 'node:readline/promises';
import { stdin as input, stdout as output } from 'node:process';
import { parseFen, START_FEN, boardToFen } from '../src/chess/board.ts';
import { pseudoLegalMoves, makeMove } from '../src/chess/movegen.ts';
import { moveFromUci, moveToActionId, moveToUci } from '../src/chess/moveCodec.ts';
import { chooseMove } from '../src/search/puct.ts';
import { StudentEvaluator } from '../src/nn/studentEvaluator.ts';

function arg(name, fallback = undefined) {
  const i = process.argv.indexOf(name);
  return i >= 0 ? process.argv[i + 1] : fallback;
}

function has(name) {
  return process.argv.includes(name);
}

function render(board) {
  let out = '';
  for (let rank = 7; rank >= 0; rank--) {
    out += `${rank + 1} `;
    for (let file = 0; file < 8; file++) {
      const piece = board.squares[file + rank * 8];
      out += `${piece ? (piece[0] === 'w' ? piece[1].toUpperCase() : piece[1]) : '.'} `;
    }
    out += '\n';
  }
  return `${out}  a b c d e f g h\n${board.turn} to move`;
}

function legalUci(board) {
  return new Map(pseudoLegalMoves(board).map((move) => [moveToUci(move), move]));
}

async function engineMove(board, evaluator) {
  const result = await chooseMove(board, evaluator);
  if (!result.move) return { board, result, uci: null };
  return { board: makeMove(board, result.move), result, uci: moveToUci(result.move) };
}

const modelPath = arg('--model', 'artifacts/student_distill_benchmark.json');
const evaluator = StudentEvaluator.fromJson(readFileSync(modelPath, 'utf8'));
let board = parseFen(arg('--fen', START_FEN));

for (const move of process.argv.filter((x) => /^--move=/.test(x)).map((x) => x.slice('--move='.length))) {
  const legal = legalUci(board);
  if (!legal.has(move)) throw new Error(`Illegal move ${move} for ${boardToFen(board)}`);
  board = makeMove(board, legal.get(move));
}

if (has('--json')) {
  const before = boardToFen(board);
  const legal = legalUci(board);
  const evaluation = evaluator.evaluate(board);
  const ranked = [...legal.values()].map((move) => {
    const prior = evaluation.policy.get(moveToActionId(move)) ?? 0;
    return { move: moveToUci(move), prior };
  }).sort((a, b) => b.prior - a.prior);
  const played = await engineMove(board, evaluator);
  console.log(JSON.stringify({ fen: before, engineMove: played.uci, value: played.result.value, wdl: evaluation.wdl, legalMoves: ranked }, null, 2));
  process.exit(0);
}

if (has('--engine-once')) {
  const played = await engineMove(board, evaluator);
  console.log(played.uci ?? '(game over)');
  console.log(boardToFen(played.board));
  process.exit(0);
}

const rl = createInterface({ input, output });
console.log('Tiny Leela student shell. Enter UCI moves, "engine", "fen", "legal", or "quit".');
while (true) {
  console.log(render(board));
  const line = (await rl.question('> ')).trim();
  if (!line || line === 'quit' || line === 'exit') break;
  if (line === 'fen') { console.log(boardToFen(board)); continue; }
  if (line === 'legal') { console.log([...legalUci(board).keys()].join(' ')); continue; }
  if (line === 'engine') {
    const played = await engineMove(board, evaluator);
    console.log(`engine: ${played.uci ?? '(game over)'}`);
    board = played.board;
    continue;
  }
  try {
    moveFromUci(line);
    const legal = legalUci(board);
    if (!legal.has(line)) {
      console.log(`Illegal move. Legal: ${[...legal.keys()].join(' ')}`);
      continue;
    }
    board = makeMove(board, legal.get(line));
    const played = await engineMove(board, evaluator);
    console.log(`engine: ${played.uci ?? '(game over)'}`);
    board = played.board;
  } catch (err) {
    console.log(err.message);
  }
}
rl.close();
