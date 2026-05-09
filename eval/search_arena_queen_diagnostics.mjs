#!/usr/bin/env node
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { dirname } from 'node:path';
import { parseFen, boardToFen, opposite } from '../src/chess/board.ts';
import { legalMoves, makeMove, isSquareAttacked } from '../src/chess/movegen.ts';
import { moveFromUci, moveToUci } from '../src/chess/moveCodec.ts';

function arg(name, fallback = '') {
  const p = `${name}=`;
  const x = process.argv.find(v => v.startsWith(p));
  if (x) return x.slice(p.length);
  const i = process.argv.indexOf(name);
  return i >= 0 ? process.argv[i + 1] : fallback;
}
function pieceAt(board, sq) { return board.squares[sq]; }
function queenSquares(board, color) {
  const q = `${color}q`, out = [];
  board.squares.forEach((p, i) => { if (p === q) out.push(i); });
  return out;
}
function material(board, color) {
  const v = { p:1,n:3,b:3,r:5,q:9,k:0 };
  return board.squares.reduce((s,p)=>s + (p?.[0] === color ? (v[p[1]] ?? 0) : 0), 0);
}
function moveEq(a,b) { return a.from === b.from && a.to === b.to && (a.promotion ?? '') === (b.promotion ?? ''); }
function legalByUci(board, uci) {
  const want = moveFromUci(uci);
  return legalMoves(board).find(m => moveEq(m, want));
}
function queenRiskAfterMove(board, move) {
  const us = board.turn;
  const beforeQueens = queenSquares(board, us);
  const movingQueen = pieceAt(board, move.from) === `${us}q`;
  const after = makeMove(board, move);
  const qs = queenSquares(after, us);
  const queenLostImmediately = beforeQueens.length > 0 && qs.length < beforeQueens.length;
  const queenSq = movingQueen ? move.to : qs[0];
  const queenEnPrise = queenSq !== undefined && isSquareAttacked(after, queenSq, after.turn);
  let captureReply = null, attackReply = null;
  for (const reply of legalMoves(after)) {
    const target = pieceAt(after, reply.to);
    if (target === `${us}q`) { captureReply = moveToUci(reply); break; }
    if (movingQueen && !attackReply) {
      const afterReply = makeMove(after, reply);
      const q2 = queenSquares(afterReply, us)[0];
      if (q2 !== undefined && isSquareAttacked(afterReply, q2, afterReply.turn)) attackReply = moveToUci(reply);
    }
  }
  const matBefore = material(board, us) - material(board, opposite(us));
  const matAfter = material(after, us) - material(after, opposite(us));
  return { movingQueen, queenLostImmediately, queenEnPrise, captureReply, attackReply, materialDeltaAfterMove: matAfter - matBefore, fenAfter: boardToFen(after) };
}
function emptyRow() { return { moves:0, queenMoves:0, risky:0, queenLostImmediately:0, queenEnPrise:0, captureReply:0, attackReply:0, actualReplyCapturedQueen:0, materialDropGe3:0 }; }
function add(row, risk, actualReplyCapturedQueen) {
  row.moves++;
  row.queenMoves += risk.movingQueen ? 1 : 0;
  row.risky += (risk.queenLostImmediately || risk.queenEnPrise || risk.captureReply || risk.attackReply || actualReplyCapturedQueen) ? 1 : 0;
  row.queenLostImmediately += risk.queenLostImmediately ? 1 : 0;
  row.queenEnPrise += risk.queenEnPrise ? 1 : 0;
  row.captureReply += risk.captureReply ? 1 : 0;
  row.attackReply += risk.attackReply ? 1 : 0;
  row.actualReplyCapturedQueen += actualReplyCapturedQueen ? 1 : 0;
  row.materialDropGe3 += risk.materialDeltaAfterMove <= -3 ? 1 : 0;
}
function finalize(row) {
  return { ...row, riskRate: row.risky / Math.max(1, row.moves), queenMoveRate: row.queenMoves / Math.max(1, row.moves), actualReplyCapturedQueenRate: row.actualReplyCapturedQueen / Math.max(1, row.moves) };
}

const input = arg('--input');
const out = arg('--out', 'artifacts/diagnostics/search_arena_queen_diagnostics.json');
if (!input) throw new Error('usage: node --experimental-strip-types eval/search_arena_queen_diagnostics.mjs --input arena.json --out diagnostics.json');
const data = JSON.parse(readFileSync(input, 'utf8'));
const playerMeta = Object.fromEntries((data.protocol?.players ?? []).map(p => [p.name, p]));
const byPlayer = {}, byPlayerColor = {}, incidents = [];
let gamesWithMoves = 0, gamesSkippedNoMoves = 0;
for (let gi = 0; gi < (data.games ?? []).length; gi++) {
  const g = data.games[gi];
  const moves = g.moves ?? [];
  if (!moves.length) { gamesSkippedNoMoves++; continue; }
  gamesWithMoves++;
  for (let mi = 0; mi < moves.length; mi++) {
    const rec = moves[mi];
    const board = parseFen(rec.fenBefore);
    const move = legalByUci(board, rec.uci);
    if (!move) continue;
    const player = rec.engine;
    const color = board.turn;
    const risk = queenRiskAfterMove(board, move);
    const next = moves[mi + 1];
    let actualReplyCapturedQueen = false;
    if (next) {
      const after = makeMove(board, move);
      const reply = legalByUci(after, next.uci);
      if (reply && pieceAt(after, reply.to) === `${color}q`) actualReplyCapturedQueen = true;
    }
    byPlayer[player] ??= emptyRow();
    byPlayerColor[`${player}:${color}`] ??= emptyRow();
    add(byPlayer[player], risk, actualReplyCapturedQueen);
    add(byPlayerColor[`${player}:${color}`], risk, actualReplyCapturedQueen);
    const isRisk = !!(risk.queenLostImmediately || risk.queenEnPrise || risk.captureReply || risk.attackReply || actualReplyCapturedQueen || risk.materialDeltaAfterMove <= -3);
    if (isRisk) incidents.push({
      game_index: gi,
      ply: rec.ply,
      player,
      player_color: color,
      tiny_color: color,
      opponent: rec.engine === g.white ? g.black : g.white,
      fen_before: rec.fenBefore,
      selected_move_uci: rec.uci,
      selected_prob: null,
      selected_topk_rank: null,
      risk,
      actual_reply_uci: next?.uci,
      actual_reply_engine: next?.engine,
      actual_reply_captured_queen: actualReplyCapturedQueen,
      game: { white:g.white, black:g.black, whiteScore:g.whiteScore, a:g.a, b:g.b, aScore:g.aScore, plies:g.plies, illegal:g.illegal },
    });
  }
}
incidents.sort((a,b) => (b.actual_reply_captured_queen - a.actual_reply_captured_queen) || ((a.risk.materialDeltaAfterMove) - (b.risk.materialDeltaAfterMove)));
const summary = {
  source: input,
  games: (data.games ?? []).length,
  gamesWithMoves,
  gamesSkippedNoMoves,
  players: Object.fromEntries(Object.entries(byPlayer).map(([k,v]) => [k, { ...finalize(v), config: playerMeta[k] ?? null }])),
  by_player_color: Object.fromEntries(Object.entries(byPlayerColor).map(([k,v]) => [k, finalize(v)])),
};
mkdirSync(dirname(out), { recursive: true });
writeFileSync(out, JSON.stringify({ summary, incidents }, null, 2));
for (const [name, row] of Object.entries(summary.players)) {
  console.log(`METRIC queen_${name}_moves=${row.moves}`);
  console.log(`METRIC queen_${name}_risk_rate=${row.riskRate.toFixed(6)}`);
  console.log(`METRIC queen_${name}_actual_reply_captured_queen=${row.actualReplyCapturedQueen}`);
  console.log(`METRIC queen_${name}_material_drop_ge3=${row.materialDropGe3}`);
}
console.log(`METRIC queen_games=${summary.games}`);
console.log(`METRIC queen_games_with_moves=${gamesWithMoves}`);
console.log(`METRIC queen_incidents=${incidents.length}`);
console.log(`wrote ${out}`);
