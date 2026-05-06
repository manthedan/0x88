import { Chessground } from 'chessground';
import type { Key } from 'chessground/types';
import { parseFen, boardToFen, squareName, START_FEN, type BoardState } from './chess/board.ts';
import { legalMoves, makeMove } from './chess/movegen.ts';
import { moveToActionId, moveToUci, type Move } from './chess/moveCodec.ts';
import { chooseMove } from './search/puct.ts';
import { OnnxEvaluator, type OnnxStudentMeta } from './nn/onnxEvaluator.ts';

let board: BoardState = parseFen(START_FEN);
let historyFens: string[] = [];
let evaluator: OnnxEvaluator | null = null;
let ground: ReturnType<typeof Chessground> | null = null;
let orientation: 'white' | 'black' = 'white';
let lastMove: string | null = null;
const visits = Number(new URLSearchParams(location.search).get('visits') ?? '256');

const $ = (id: string) => document.getElementById(id)!;
function legalDests() {
  const dests = new Map<Key, Key[]>();
  for (const m of legalMoves(board)) {
    const from = squareName(m.from) as Key, to = squareName(m.to) as Key;
    dests.set(from, [...(dests.get(from) ?? []), to]);
  }
  return dests;
}
function legalMoveByUci(uci: string) { return legalMoves(board).find((m) => moveToUci(m) === uci) ?? null; }
function renderWdl(wdl: [number, number, number]) {
  $('wdl').innerHTML = ['Win','Draw','Loss'].map((name,i)=>`<div class="bar"><span>${name}</span><div class="track"><div class="fill" style="width:${Math.round((wdl[i]??0)*100)}%"></div></div><span>${((wdl[i]??0)*100).toFixed(1)}%</span></div>`).join('');
}
async function render(message = '') {
  $('fen').textContent = boardToFen(board);
  $('status').textContent = evaluator ? `${visits} visits` : 'loading';
  if (message) $('message').textContent = message;
  if (!ground) {
    ground = Chessground($('ground'), { orientation, fen: boardToFen(board), movable: { free: false, color: board.turn === 'w' ? 'white' : 'black', dests: legalDests(), events: { after: onUserMove } } });
  } else {
    ground.set({ orientation, fen: boardToFen(board), lastMove: lastMove ? [lastMove.slice(0,2) as Key, lastMove.slice(2,4) as Key] : undefined, movable: { free: false, color: board.turn === 'w' ? 'white' : 'black', dests: legalDests(), events: { after: onUserMove } } });
  }
  if (!evaluator) return;
  const ev = await evaluator.evaluate(board, { historyFens });
  renderWdl(ev.wdl);
  const rows = legalMoves(board).map((m: Move) => ({ uci: moveToUci(m), prior: ev.policy.get(moveToActionId(m)) ?? 0 })).sort((a,b)=>b.prior-a.prior).slice(0,16);
  $('moves').innerHTML = rows.map((r)=>`<li><b>${r.uci}</b> ${(r.prior*100).toFixed(2)}%</li>`).join('');
}
async function playMove(move: Move, who: string) {
  const before = boardToFen(board);
  const uci = moveToUci(move);
  historyFens = [before, ...historyFens];
  board = makeMove(board, move);
  lastMove = uci;
  await render(`${who} played ${uci}.`);
}
async function onUserMove(from: string, to: string) {
  const move = legalMoveByUci(from + to) ?? legalMoves(board).find((m) => squareName(m.from) === from && squareName(m.to) === to);
  if (!move) { await render(`Illegal move ${from}${to}.`); return; }
  await playMove(move, 'You');
  await engineMove();
}
async function engineMove() {
  if (!evaluator) return;
  document.body.style.cursor = 'progress';
  try {
    const result = await chooseMove(board, evaluator, { visits, historyFens });
    if (result.move) await playMove(result.move, 'Engine');
    else await render('No legal engine move.');
  } finally { document.body.style.cursor = ''; }
}
$('engine').onclick = () => engineMove();
$('reset').onclick = async () => { board = parseFen(START_FEN); historyFens = []; lastMove = null; await render('Reset.'); };
$('flip').onclick = async () => { orientation = orientation === 'white' ? 'black' : 'white'; await render(); };
$('loadFen').onclick = async () => { board = parseFen(($('fenInput') as HTMLInputElement).value || START_FEN); historyFens = []; lastMove = null; await render('Loaded FEN.'); };

async function main() {
  await render();
  const meta = await fetch('/models/residual_32x4_history2.meta.json').then((r) => r.json()) as OnnxStudentMeta;
  evaluator = await OnnxEvaluator.create('/models/residual_32x4_history2.onnx', meta);
  await render(`Loaded ONNX model. Running ${visits} visits in-browser.`);
}
main().catch((e) => { console.error(e); $('message').textContent = `Failed: ${e.message}`; });
