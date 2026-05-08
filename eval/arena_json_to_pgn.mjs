#!/usr/bin/env node
import fs from 'node:fs';
import { parseFen, squareName } from '../src/chess/board.ts';
import { legalMoves, makeMove, inCheck } from '../src/chess/movegen.ts';
import { moveFromUci, moveToUci } from '../src/chess/moveCodec.ts';

function arg(name, fallback = null) {
  const i = process.argv.indexOf(name);
  return i >= 0 ? process.argv[i + 1] : fallback;
}
function has(name) { return process.argv.includes(name); }

const input = arg('--in');
const out = arg('--out');
const anchor = arg('--anchor');
const indexArg = arg('--index');
const maxGames = Number(arg('--limit', '0')) || Infinity;
if (!input || !out) {
  throw new Error('Usage: node --experimental-strip-types eval/arena_json_to_pgn.mjs --in arena.json --out games.pgn [--anchor maia1900] [--index 80] [--limit 20]');
}

const pieceLetter = { p: '', n: 'N', b: 'B', r: 'R', q: 'Q', k: 'K' };
function esc(s) { return String(s ?? '').replace(/\\/g, '\\\\').replace(/"/g, '\\"'); }
function tag(k, v) { return `[${k} "${esc(v)}"]`; }
function sameMove(a, b) { return a.from === b.from && a.to === b.to && (a.promotion ?? '') === (b.promotion ?? ''); }

function sanFor(board, move) {
  const piece = board.squares[move.from];
  if (!piece) throw new Error(`No piece on ${squareName(move.from)}`);
  const role = piece[1];
  const from = squareName(move.from);
  const to = squareName(move.to);
  const isCastle = role === 'k' && Math.abs(move.to - move.from) === 2;
  const captured = board.squares[move.to] || (role === 'p' && board.epSquare === move.to && from[0] !== to[0]);
  let san = '';
  if (isCastle) san = to[0] === 'g' ? 'O-O' : 'O-O-O';
  else {
    san += pieceLetter[role];
    if (role !== 'p') {
      const ambiguous = legalMoves(board).filter((m) => !sameMove(m, move) && m.to === move.to && board.squares[m.from]?.[1] === role);
      if (ambiguous.length) {
        const fileUnique = !ambiguous.some((m) => squareName(m.from)[0] === from[0]);
        const rankUnique = !ambiguous.some((m) => squareName(m.from)[1] === from[1]);
        san += fileUnique ? from[0] : rankUnique ? from[1] : from;
      }
    } else if (captured) san += from[0];
    if (captured) san += 'x';
    san += to;
    if (move.promotion) san += `=${pieceLetter[move.promotion]}`;
  }
  const next = makeMove(board, move);
  if (inCheck(next, next.turn)) san += legalMoves(next).length === 0 ? '#' : '+';
  return san;
}

function resultFor(g) {
  if (g.illegal) return '*';
  if (g.whiteScore === 1) return '1-0';
  if (g.whiteScore === 0) return '0-1';
  if (g.whiteScore === 0.5) return '1/2-1/2';
  return '*';
}

const data = JSON.parse(fs.readFileSync(input, 'utf8'));
let entries = data.games.map((g, i) => [i, g]);
if (indexArg !== null) entries = entries.filter(([i]) => i === Number(indexArg));
if (anchor) entries = entries.filter(([, g]) => g.anchor === anchor);
entries = entries.slice(0, maxGames);

const chunks = [];
for (const [i, g] of entries) {
  let board = parseFen(g.opening);
  const result = resultFor(g);
  const headers = [
    tag('Event', `tiny_leela arena ${data.candidate?.name ?? data.candidate ?? ''}`.trim()),
    tag('Site', 'local'),
    tag('Date', new Date().toISOString().slice(0, 10).replaceAll('-', '.')),
    tag('Round', String(i)),
    tag('White', g.white),
    tag('Black', g.black),
    tag('Result', result),
    tag('SetUp', '1'),
    tag('FEN', g.opening),
    tag('Anchor', g.anchor),
    tag('TinyScore', g.tinyScore),
    tag('PlyCount', g.plies),
  ];
  if (g.illegal) headers.push(tag('Termination', `illegal ${g.illegal}`));
  const tokens = [];
  for (const rec of g.moves) {
    const move = moveFromUci(rec.uci);
    const legal = legalMoves(board).find((m) => sameMove(m, move));
    if (!legal) { tokens.push(`{ illegal-or-unparseable ${rec.uci} }`); break; }
    if (board.turn === 'w') tokens.push(`${board.fullmove}.`);
    else if (tokens.length === 0) tokens.push(`${board.fullmove}...`);
    tokens.push(sanFor(board, legal));
    board = makeMove(board, legal);
  }
  tokens.push(result);
  chunks.push(`${headers.join('\n')}\n\n${tokens.join(' ')}\n`);
}
fs.writeFileSync(out, chunks.join('\n'));
console.log(`wrote ${entries.length} games to ${out}`);
