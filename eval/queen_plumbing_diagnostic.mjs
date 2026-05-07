#!/usr/bin/env node
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { dirname } from 'node:path';
import { parseFen, boardToFen, opposite } from '../src/chess/board.ts';
import { legalMoves, makeMove, isSquareAttacked } from '../src/chess/movegen.ts';
import { moveFromUci, moveToUci, moveToActionId } from '../src/chess/moveCodec.ts';
import { OnnxEvaluator } from '../src/nn/onnxEvaluator.ts';
import { SquareFormerEvaluator } from '../src/nn/squareformerEvaluator.ts';

function arg(name, fallback = '') { const p = `${name}=`; const x = process.argv.find(v => v.startsWith(p)); if (x) return x.slice(p.length); const i = process.argv.indexOf(name); return i >= 0 ? process.argv[i + 1] : fallback; }
function pieceAt(board, sq) { return board.squares[sq]; }
function queenSquares(board, color) { const q = `${color}q`; const out = []; board.squares.forEach((p, i) => { if (p === q) out.push(i); }); return out; }
function material(board, color) { const v = { p:1,n:3,b:3,r:5,q:9,k:0 }; return board.squares.reduce((s,p)=>s + (p?.[0] === color ? (v[p[1]] ?? 0) : 0), 0); }
function moveEq(a,b) { return a.from === b.from && a.to === b.to && (a.promotion ?? '') === (b.promotion ?? ''); }
function legalByUci(board, uci) { const want = moveFromUci(uci); return legalMoves(board).find(m => moveEq(m, want)); }

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

function topLegal(ev, board, k = 12) {
  return legalMoves(board).map(m => ({ move: m, uci: moveToUci(m), p: ev.policy.get(moveToActionId(m)) ?? 0, risk: queenRiskAfterMove(board, m) })).sort((a,b)=>b.p-a.p).slice(0,k);
}

function fenFlipColorBoard(fen) {
  // Rotate 180 degrees and swap colors. This creates a black-side analog of a white-side position.
  const b = parseFen(fen); const sq = Array(64).fill(null);
  for (let i = 0; i < 64; i++) {
    const p = b.squares[i]; if (!p) continue;
    const j = 63 - i; sq[j] = `${opposite(p[0])}${p[1]}`;
  }
  const cast = b.castling === '-' ? '-' : b.castling.split('').map(c => ({K:'k',Q:'q',k:'K',q:'Q'}[c])).join('') || '-';
  const ep = b.epSquare == null ? null : 63 - b.epSquare;
  return boardToFen({ squares: sq, turn: opposite(b.turn), castling: cast, epSquare: ep, halfmove: b.halfmove, fullmove: b.fullmove });
}

const BUILTIN = [
  { id:'white_queen_poisoned_h5', fen:'rnbqkbnr/pppppppp/8/8/7Q/8/PPPPPPPP/RNB1KBNR b KQkq - 0 1', unsafe:['g7g6'], note:'black can attack/capture an exposed white queen motif' },
  { id:'white_to_move_obvious_hang', fen:'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1', unsafe:['d1h5'], note:'early queen sortie: after ...g6 queen is attacked; useful model tendency probe' },
  { id:'black_to_move_obvious_hang', fen:fenFlipColorBoard('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'), unsafe:['d8h4'], note:'color-flipped analog for side confusion probe' },
  { id:'white_queen_directly_capturable', fen:'4k3/8/8/8/3q4/8/3Q4/4K3 w - - 0 1', unsafe:['d2d4'], note:'moving queen onto occupied/contested queen line sanity' },
];

async function loadEvaluator(model, metaPath) {
  const meta = JSON.parse(readFileSync(metaPath, 'utf8'));
  if (meta.kind === 'squareformer') return { meta, evaluator: await SquareFormerEvaluator.create(model, meta) };
  return { meta, evaluator: await OnnxEvaluator.create(model, meta) };
}

const gamesJson = arg('--games-json');
const positionsJson = arg('--positions-json');
let model = arg('--model');
let metaPath = arg('--meta');
const out = arg('--out', 'artifacts/diagnostics/queen_plumbing.json');
const topK = Number(arg('--top-k', '12'));
let gamesData = null, positionsData = null;
if (gamesJson) {
  gamesData = JSON.parse(readFileSync(gamesJson, 'utf8'));
  model ||= gamesData.candidate?.onnx;
  metaPath ||= gamesData.candidate?.meta;
}
if (positionsJson) positionsData = JSON.parse(readFileSync(positionsJson, 'utf8'));
if (!model || !metaPath) throw new Error('usage: node --experimental-strip-types eval/queen_plumbing_diagnostic.mjs --model model.onnx --meta model.meta.json [--games-json arena.json | --positions-json positions.json] [--out out.json]');
const { meta, evaluator } = await loadEvaluator(model, metaPath);

async function runBuiltin() {
const tests = [...BUILTIN];
const results = [];
let selectedUnsafe = 0, selectedQueenMove = 0, whiteUnsafe = 0, blackUnsafe = 0, whiteN = 0, blackN = 0;
for (const t of tests) {
  const board = parseFen(t.fen);
  const ev = await evaluator.evaluate(board);
  const top = topLegal(ev, board, topK);
  const selected = top[0];
  const unsafeTop = top.filter(x => x.risk.queenEnPrise || x.risk.captureReply || x.risk.attackReply || x.risk.queenLostImmediately);
  const unsafeExpected = (t.unsafe ?? []).map(u => {
    const m = legalByUci(board, u); return m ? { uci:u, p: ev.policy.get(moveToActionId(m)) ?? 0, risk: queenRiskAfterMove(board, m) } : { uci:u, error:'not legal' };
  });
  const selUnsafe = !!selected && (selected.risk.queenEnPrise || selected.risk.captureReply || selected.risk.attackReply || selected.risk.queenLostImmediately);
  selectedUnsafe += selUnsafe ? 1 : 0; selectedQueenMove += selected?.risk.movingQueen ? 1 : 0;
  if (board.turn === 'w') { whiteN++; whiteUnsafe += selUnsafe ? 1 : 0; } else { blackN++; blackUnsafe += selUnsafe ? 1 : 0; }
  results.push({ id:t.id, note:t.note, fen:t.fen, turn:board.turn, selected:selected && { uci:selected.uci, p:selected.p, risk:selected.risk }, unsafe_expected:unsafeExpected, unsafe_top:unsafeTop.map(x=>({uci:x.uci,p:x.p,risk:x.risk})), top:top.map(x=>({uci:x.uci,p:x.p,movingQueen:x.risk.movingQueen,queenEnPrise:x.risk.queenEnPrise,captureReply:x.risk.captureReply,attackReply:x.risk.attackReply})) });
}
const summary = { model, meta: metaPath, kind: meta.kind ?? meta.architecture, tests: tests.length, selectedUnsafe, selectedQueenMove, by_turn:{ white:{tests:whiteN, selectedUnsafe:whiteUnsafe}, black:{tests:blackN, selectedUnsafe:blackUnsafe} } };
const report = { summary, results };
mkdirSync(dirname(out), { recursive:true }); writeFileSync(out, JSON.stringify(report, null, 2));
console.log(`METRIC queen_tests=${tests.length}`);
console.log(`METRIC queen_selected_unsafe=${selectedUnsafe}`);
console.log(`METRIC queen_selected_queen_move=${selectedQueenMove}`);
console.log(`METRIC queen_white_selected_unsafe=${whiteUnsafe}`);
console.log(`METRIC queen_black_selected_unsafe=${blackUnsafe}`);
console.log(`wrote ${out}`);
for (const r of results) console.log(`${r.id} turn=${r.turn} selected=${r.selected?.uci} p=${r.selected?.p?.toFixed(4)} unsafe=${!!(r.selected?.risk?.queenEnPrise || r.selected?.risk?.captureReply || r.selected?.risk?.attackReply || r.selected?.risk?.queenLostImmediately)} topUnsafe=${r.unsafe_top.slice(0,3).map(x=>`${x.uci}:${x.p.toFixed(3)}`).join(',')}`);
}

async function runGames(data) {
  const cand = data.candidate?.name;
  const incidents = []; const by = {}; let tinyMoves = 0, queenMoves = 0, risky = 0, selectedTop1 = 0, whiteRisk = 0, blackRisk = 0, whiteMoves = 0, blackMoves = 0;
  for (let gi = 0; gi < (data.games ?? []).length; gi++) {
    const g = data.games[gi]; const tinyColor = g.white === cand ? 'w' : 'b';
    const moves = g.moves ?? [];
    for (let mi = 0; mi < moves.length; mi++) {
      const rec = moves[mi];
      if (rec.engine !== cand) continue;
      const board = parseFen(rec.fenBefore); const move = legalByUci(board, rec.uci); if (!move) continue;
      const ev = await evaluator.evaluate(board); const top = topLegal(ev, board, topK);
      const rank = Math.max(1, top.findIndex(x => x.uci === rec.uci) + 1 || 9999);
      const prob = ev.policy.get(moveToActionId(move)) ?? 0;
      const risk = queenRiskAfterMove(board, move);
      const next = moves[mi + 1];
      let actualReplyCapturedQueen = false;
      if (next) {
        const after = makeMove(board, move); const reply = legalByUci(after, next.uci);
        if (reply && pieceAt(after, reply.to) === `${tinyColor}q`) actualReplyCapturedQueen = true;
      }
      const isRisk = !!(risk.queenLostImmediately || risk.queenEnPrise || risk.captureReply || risk.attackReply || actualReplyCapturedQueen);
      tinyMoves++; selectedTop1 += rank === 1 ? 1 : 0; queenMoves += risk.movingQueen ? 1 : 0; risky += isRisk ? 1 : 0;
      if (board.turn === 'w') { whiteMoves++; whiteRisk += isRisk ? 1 : 0; } else { blackMoves++; blackRisk += isRisk ? 1 : 0; }
      const key = `${g.anchor}:${board.turn}`; by[key] ??= { moves:0, risky:0, queenMoves:0, actualReplyCapturedQueen:0 };
      by[key].moves++; by[key].risky += isRisk ? 1 : 0; by[key].queenMoves += risk.movingQueen ? 1 : 0; by[key].actualReplyCapturedQueen += actualReplyCapturedQueen ? 1 : 0;
      if (isRisk) incidents.push({ game_index:gi, ply:rec.ply, anchor:g.anchor, tiny_color:tinyColor, turn:board.turn, fen_before:rec.fenBefore, selected_move_uci:rec.uci, selected_prob:prob, selected_topk_rank:rank, risk, actual_reply_uci:next?.uci, actual_reply_engine:next?.engine, actual_reply_captured_queen:actualReplyCapturedQueen, top:top.map(x=>({uci:x.uci,p:x.p,risk:x.risk})) });
    }
  }
  incidents.sort((a,b)=>(b.actual_reply_captured_queen-a.actual_reply_captured_queen) || (b.selected_prob-a.selected_prob));
  const summary = { source: gamesJson, model, meta: metaPath, candidate:cand, games:(data.games??[]).length, tinyMoves, queenMoves, risky, selectedTop1, riskRate:risky/Math.max(1,tinyMoves), by_turn:{ white:{moves:whiteMoves,risky:whiteRisk,rate:whiteRisk/Math.max(1,whiteMoves)}, black:{moves:blackMoves,risky:blackRisk,rate:blackRisk/Math.max(1,blackMoves)} }, by_anchor_turn:by };
  const report = { summary, incidents };
  mkdirSync(dirname(out), { recursive:true }); writeFileSync(out, JSON.stringify(report, null, 2));
  console.log(`METRIC queen_game_tiny_moves=${tinyMoves}`);
  console.log(`METRIC queen_game_risky_moves=${risky}`);
  console.log(`METRIC queen_game_risk_rate=${(risky/Math.max(1,tinyMoves)).toFixed(6)}`);
  console.log(`METRIC queen_game_white_risk_rate=${summary.by_turn.white.rate.toFixed(6)}`);
  console.log(`METRIC queen_game_black_risk_rate=${summary.by_turn.black.rate.toFixed(6)}`);
  console.log(`METRIC queen_game_actual_reply_captured_queen=${incidents.filter(x=>x.actual_reply_captured_queen).length}`);
  console.log(`wrote ${out}`);
  for (const x of incidents.slice(0,12)) console.log(`incident game=${x.game_index} ply=${x.ply} ${x.anchor} turn=${x.turn} move=${x.selected_move_uci} p=${x.selected_prob.toFixed(4)} rank=${x.selected_topk_rank} reply=${x.actual_reply_uci ?? '-'} capturedQ=${x.actual_reply_captured_queen}`);
}

async function runPositions(data) {
  const positions = Array.isArray(data) ? data : (data.positions ?? []);
  const results = []; let selectedRisk = 0, queenMoves = 0, whiteN = 0, blackN = 0, whiteRisk = 0, blackRisk = 0;
  for (const p of positions) {
    const fen = p.fen ?? p.fen_before; if (!fen) continue;
    const board = parseFen(fen); const ev = await evaluator.evaluate(board); const top = topLegal(ev, board, topK); const selected = top[0];
    const isRisk = !!selected && (selected.risk.queenLostImmediately || selected.risk.queenEnPrise || selected.risk.captureReply || selected.risk.attackReply);
    selectedRisk += isRisk ? 1 : 0; queenMoves += selected?.risk.movingQueen ? 1 : 0;
    if (board.turn === 'w') { whiteN++; whiteRisk += isRisk ? 1 : 0; } else { blackN++; blackRisk += isRisk ? 1 : 0; }
    results.push({ id:p.id, source:p.source, fen, turn:board.turn, selected:selected && { uci:selected.uci, p:selected.p, risk:selected.risk }, selectedRisk:isRisk, original:p, top:top.map(x=>({uci:x.uci,p:x.p,risk:x.risk})) });
  }
  const summary = { source:positionsJson, model, meta:metaPath, positions:results.length, selectedRisk, queenMoves, riskRate:selectedRisk/Math.max(1,results.length), by_turn:{ white:{positions:whiteN, selectedRisk:whiteRisk, rate:whiteRisk/Math.max(1,whiteN)}, black:{positions:blackN, selectedRisk:blackRisk, rate:blackRisk/Math.max(1,blackN)} } };
  const report = { summary, results };
  mkdirSync(dirname(out), { recursive:true }); writeFileSync(out, JSON.stringify(report, null, 2));
  console.log(`METRIC queen_suite_positions=${results.length}`);
  console.log(`METRIC queen_suite_selected_risk=${selectedRisk}`);
  console.log(`METRIC queen_suite_risk_rate=${summary.riskRate.toFixed(6)}`);
  console.log(`METRIC queen_suite_queen_moves=${queenMoves}`);
  console.log(`METRIC queen_suite_white_risk_rate=${summary.by_turn.white.rate.toFixed(6)}`);
  console.log(`METRIC queen_suite_black_risk_rate=${summary.by_turn.black.rate.toFixed(6)}`);
  console.log(`wrote ${out}`);
  for (const r of results.filter(x=>x.selectedRisk).slice(0,12)) console.log(`risk ${r.id} turn=${r.turn} selected=${r.selected?.uci} p=${r.selected?.p.toFixed(4)} src=${r.source ?? ''}`);
}

if (positionsData) await runPositions(positionsData); else if (gamesData) await runGames(gamesData); else await runBuiltin();
