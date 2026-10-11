#!/usr/bin/env node
// Run under the shared benchlock. A/B differs only in the adapter's old
// per-search Hash/Threads re-send; artifact, worker-equivalent instance,
// game reset, fixed positions and fixed depth stay the same.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { createRequire } from 'node:module';
import path from 'node:path';
import { boardToFen, parseFen, START_FEN } from '../src/chess/board.ts';
import { moveToUci } from '../src/chess/moveCodec.ts';
import { legalMoves, makeMove } from '../src/chess/movegen.ts';
import { parseInfo } from './emscripten_uci_protocol.mjs';

const out = process.argv[2];
const depth = Math.max(1, Number(process.argv[3] ?? 16));
const dir = path.resolve('.local-dev-artifacts/lifecycle-bench');
fs.mkdirSync(dir, { recursive: true });
const base = 'berserk-emscripten-relaxed-simd128';
fs.copyFileSync(`public/berserk/${base}.js`, `${dir}/${base}.cjs`);
fs.copyFileSync(`public/berserk/${base}.wasm`, `${dir}/${base}.wasm`);
fs.copyFileSync('public/berserk/berserk-emscripten.data', `${dir}/${base}.data`);
const factory = createRequire(import.meta.url)(`${dir}/${base}.cjs`);
const stdout = [];
const module = await factory({ locateFile: (file) => path.join(dir, file), print: (line) => stdout.push(String(line)), printErr: () => {} });
const command = (text) => module.ccall('command', null, ['string'], [text]);
command('uci');
command('setoption name Hash value 64');
command('setoption name Threads value 1');
command('isready');
function search(fen, depth, resend) {
  const start = stdout.length;
  const began = performance.now();
  if (resend) {
    command('setoption name Hash value 64');
    command('setoption name Threads value 1');
  }
  command('setoption name MultiPV value 1');
  command(`position fen ${fen}`);
  command(`go depth ${depth}`);
  const wallMs = performance.now() - began;
  const lines = stdout.slice(start);
  const bestmove = lines.find((line) => line.startsWith('bestmove '))?.split(' ')[1];
  assert.ok(bestmove, 'synchronous search must finish');
  const info = lines.filter((line) => line.startsWith('info ') && line.includes(' pv ')).at(-1);
  return { fen, bestmove, wallMs, ...parseInfo(info) };
}
function continuation(depth, resend, plies) {
  command('ucinewgame');
  let board = parseFen(START_FEN);
  const rows = [];
  for (let ply = 0; ply < plies; ply += 1) {
    const row = search(boardToFen(board), depth, resend);
    rows.push(row);
    const move = legalMoves(board).find((candidate) => moveToUci(candidate) === row.bestmove);
    assert.ok(move, `legal bestmove ${row.bestmove}`);
    board = makeMove(board, move);
  }
  return rows;
}
// Depth 1: no deeper inherited TT bound should decide a root move. Explicit
// newGame between variants, never between plies, models a short game.
const before = continuation(1, true, 8);
const after = continuation(1, false, 8);
assert.deepEqual(
  after.map((row) => row.bestmove),
  before.map((row) => row.bestmove),
);
const fens = continuation(8, true, 10).map((row) => row.fen);
// Warm both paths before timed ABAB rounds. Compilation/startup/reset excluded;
// every search includes the redundant-option overhead in A.
for (const resend of [true, false]) {
  command('ucinewgame');
  for (const fen of fens) search(fen, depth, resend);
}
const rounds = [];
for (let pair = 0; pair < 6; pair += 1) {
  for (const variant of ['before', 'after']) {
    command('ucinewgame');
    const rows = fens.map((fen) => search(fen, depth, variant === 'before'));
    rounds.push({ pair, variant, totalMs: rows.reduce((sum, row) => sum + row.wallMs, 0), rows });
  }
}
const median = (values) => {
  const sorted = [...values].sort((a, b) => a - b);
  const middle = sorted.length >> 1;
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
};
const summary = Object.fromEntries(
  ['before', 'after'].map((variant) => {
    const values = rounds.filter((round) => round.variant === variant).map((round) => round.totalMs);
    return [variant, { medianMs: median(values), minMs: Math.min(...values), maxMs: Math.max(...values) }];
  }),
);
const report = {
  artifact: base,
  hashMb: 64,
  depth,
  plies: fens.length,
  equivalence: { depth: 1, before, after },
  summary,
  wallSavingPercent: 100 * (1 - summary.after.medianMs / summary.before.medianMs),
  rounds,
};
if (out) fs.writeFileSync(out, `${JSON.stringify(report, null, 2)}\n`);
console.log(JSON.stringify({ summary, wallSavingPercent: report.wallSavingPercent, bestmoves: before.map((row) => row.bestmove) }, null, 2));
