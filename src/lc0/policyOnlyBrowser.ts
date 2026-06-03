import { Chessground } from 'chessground';
import type { Key } from 'chessground/types';
import { boardToFen, parseFen, squareName, START_FEN, type BoardState } from '../chess/board.ts';
import { legalMoves, makeMove } from '../chess/movegen.ts';
import { moveToUci, type Move } from '../chess/moveCodec.ts';
import { collectOrtRuntimeDiagnostics, describeOrtBackendConfig } from '../nn/ortRuntime.ts';
import { Lc0PolicyOnlyPlayer } from './policyOnlyPlayer.ts';

type Ground = ReturnType<typeof Chessground>;
type NativePrior = { uci: string; index: number; prior: number };
type NativeRecord = { id: string; backend?: string; fen: string; bestmove: string; topPriors: NativePrior[] };

const params = new URLSearchParams(location.search);
const DEFAULT_MODEL = '/models/lc0/t1-256x10-distilled-swa-2432500.batch1.f32.onnx';
const MODEL_URL = params.get('model') ?? DEFAULT_MODEL;
const PLAYER_SIDE = params.get('side') === 'black' ? 'black' : 'white';

let board: BoardState = parseFen(params.get('fen') ?? START_FEN);
let ground: Ground | null = null;
let player: Lc0PolicyOnlyPlayer | null = null;
let busy = false;
let lastMove: string | null = null;
let renderSeq = 0;
let orientation: 'white' | 'black' = PLAYER_SIDE;
const playedMoves: string[] = [];

function el(id: string): HTMLElement {
  const node = document.getElementById(id);
  if (!node) throw new Error(`Missing #${id}`);
  return node;
}

function htmlEscape(value: unknown): string {
  return String(value).replace(/[&<>"']/g, (ch) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[ch]!));
}

function boardFenOnly() {
  return boardToFen(board).split(' ')[0];
}

function sideToMoveName() {
  return board.turn === 'w' ? 'White' : 'Black';
}

function legalDests() {
  const dests = new Map<Key, Key[]>();
  for (const move of legalMoves(board)) {
    const from = squareName(move.from) as Key;
    const to = squareName(move.to) as Key;
    dests.set(from, [...(dests.get(from) ?? []), to]);
  }
  return dests;
}

function legalMoveFromUci(uci: string): Move | undefined {
  return legalMoves(board).find((move) => moveToUci(move) === uci);
}

function legalMoveFromDrag(from: Key, to: Key): Move | undefined {
  const base = `${from}${to}`;
  return legalMoveFromUci(base)
    ?? legalMoveFromUci(`${base}q`)
    ?? legalMoveFromUci(`${base}r`)
    ?? legalMoveFromUci(`${base}b`)
    ?? legalMoveFromUci(`${base}n`);
}

function applyMove(move: Move): string {
  const uci = moveToUci(move);
  board = makeMove(board, move);
  lastMove = uci;
  playedMoves.push(uci);
  return uci;
}

function setBusy(next: boolean, message?: string) {
  busy = next;
  if (message) el('message').textContent = message;
  el('engineMove').toggleAttribute('disabled', busy || !player);
  el('runParity').toggleAttribute('disabled', busy || !player);
}

function renderStatic() {
  el('fen').textContent = boardToFen(board);
  el('sideToMove').textContent = sideToMoveName();
  el('moveList').textContent = playedMoves.length ? playedMoves.join(' ') : '—';
  el('modelPath').textContent = MODEL_URL;
  el('backend').textContent = describeOrtBackendConfig();
  el('status').textContent = player ? 'ready' : 'loading';
  el('engineMove').toggleAttribute('disabled', busy || !player);
  el('runParity').toggleAttribute('disabled', busy || !player);
  const config = {
    orientation,
    fen: boardFenOnly(),
    turnColor: board.turn === 'w' ? 'white' as const : 'black' as const,
    coordinates: true,
    highlight: { lastMove: true, check: true },
    animation: { enabled: true, duration: 160 },
    movable: {
      free: false,
      color: busy ? undefined : board.turn === 'w' ? 'white' as const : 'black' as const,
      dests: busy ? new Map<Key, Key[]>() : legalDests(),
      showDests: !busy,
      events: { after: onUserMove },
    },
    lastMove: lastMove ? [lastMove.slice(0, 2) as Key, lastMove.slice(2, 4) as Key] : undefined,
  };
  if (!ground) ground = Chessground(el('ground'), config);
  else ground.set(config);
}

function renderEvaluation() {
  const seq = ++renderSeq;
  renderStatic();
  if (!player) return;
  const fen = boardToFen(board);
  player.chooseMove(fen).then((choice) => {
    if (seq !== renderSeq) return;
    const ev = choice.evaluation;
    const [win, draw, loss] = ev.wdl;
    el('bestMove').textContent = choice.move ?? '—';
    el('wdl').innerHTML = `<b>W</b> ${(win * 100).toFixed(2)}% · <b>D</b> ${(draw * 100).toFixed(2)}% · <b>L</b> ${(loss * 100).toFixed(2)}%`;
    el('qMlh').textContent = `Q ${ev.q.toFixed(5)} · MLH ${ev.mlh.toFixed(1)}`;
    const max = Math.max(1e-9, ...ev.legalPriors.slice(0, 10).map((entry) => entry.prior));
    el('priors').innerHTML = ev.legalPriors.slice(0, 10).map((entry, i) => {
      const width = Math.max(2, (entry.prior / max) * 100).toFixed(1);
      return `<li class="${i === 0 ? 'best' : ''}"><span>${i + 1}</span><b>${htmlEscape(entry.uci)}</b><meter min="0" max="100" value="${width}"></meter><code>${(entry.prior * 100).toFixed(2)}%</code></li>`;
    }).join('');
  }).catch((error) => {
    if (seq !== renderSeq) return;
    el('message').textContent = `Evaluation failed: ${(error as Error).message}`;
  });
}

async function onUserMove(from: Key, to: Key) {
  if (busy) return;
  const move = legalMoveFromDrag(from, to);
  if (!move) {
    renderStatic();
    return;
  }
  const uci = applyMove(move);
  el('message').textContent = `User played ${uci}`;
  renderEvaluation();
  if ((PLAYER_SIDE === 'white' && board.turn === 'b') || (PLAYER_SIDE === 'black' && board.turn === 'w')) {
    await engineMove();
  }
}

async function engineMove() {
  if (!player || busy) return;
  const legal = legalMoves(board);
  if (!legal.length) {
    el('message').textContent = 'No legal engine move.';
    return;
  }
  setBusy(true, 'LC0 policy-only engine thinking…');
  renderStatic();
  try {
    const choice = await player.chooseMove(boardToFen(board));
    const move = choice.move ? legalMoveFromUci(choice.move) : undefined;
    if (!move) throw new Error(`Evaluator chose illegal or missing move: ${choice.move ?? 'none'}`);
    const uci = applyMove(move);
    el('message').textContent = `Engine played ${uci} (argmax legal prior, no search)`;
  } catch (error) {
    el('message').textContent = `Engine move failed: ${(error as Error).message}`;
  } finally {
    setBusy(false);
    renderEvaluation();
  }
}

function nativeCastlingToStandard(uci: string) {
  switch (uci) {
    case 'e1h1': return 'e1g1';
    case 'e1a1': return 'e1c1';
    case 'e8h8': return 'e8g8';
    case 'e8a8': return 'e8c8';
    default: return uci;
  }
}

async function runParityFixtures() {
  if (!player || busy) return;
  setBusy(true, 'Running FEN-only fixture parity in browser…');
  el('parity').textContent = 'running…';
  try {
    const response = await fetch('/lc0/native_fen_only_blas.jsonl');
    if (!response.ok) throw new Error(`native fixture fetch failed: ${response.status}`);
    const records = (await response.text()).trim().split('\n').map((line) => JSON.parse(line) as NativeRecord);
    const failures: string[] = [];
    for (const native of records) {
      const choice = await player.chooseMove(native.fen);
      const expected = nativeCastlingToStandard(native.bestmove);
      if (choice.move !== expected) failures.push(`${native.id}: best ${choice.move} != ${expected}`);
      for (const prior of native.topPriors.slice(0, 5)) {
        const uci = nativeCastlingToStandard(prior.uci);
        const actual = choice.evaluation.legalPriors.find((entry) => entry.uci === uci);
        if (!actual || Math.abs(actual.prior - prior.prior) >= 0.003) failures.push(`${native.id}: ${uci} prior mismatch`);
      }
    }
    if (failures.length) {
      el('parity').textContent = `failed: ${failures.slice(0, 3).join('; ')}`;
      el('message').textContent = `Parity failed (${failures.length} issue(s)).`;
    } else {
      el('parity').textContent = `passed ${records.length}/${records.length} native BLAS fixtures`;
      el('message').textContent = 'Browser FEN-only fixture parity passed.';
    }
  } catch (error) {
    el('parity').textContent = `failed: ${(error as Error).message}`;
    el('message').textContent = `Parity failed: ${(error as Error).message}`;
  } finally {
    setBusy(false);
    renderEvaluation();
  }
}

function resetBoard() {
  board = parseFen(START_FEN);
  lastMove = null;
  playedMoves.length = 0;
  el('message').textContent = 'Reset to start position.';
  renderEvaluation();
}

async function init() {
  el('message').textContent = 'Loading LC0 f32 ONNX model…';
  renderStatic();
  try {
    player = await Lc0PolicyOnlyPlayer.create(MODEL_URL);
    const diagnostics = await collectOrtRuntimeDiagnostics();
    el('backend').textContent = diagnostics.describe;
    el('message').textContent = 'Ready. Drag a legal move or ask the engine to move.';
    renderEvaluation();
    if (params.get('parity') === '1' || params.get('fixtures') === '1') await runParityFixtures();
    if (params.get('engineMove') === '1') await engineMove();
  } catch (error) {
    el('message').textContent = `Model load failed: ${(error as Error).message}`;
    renderStatic();
  }
}

el('engineMove').addEventListener('click', () => { void engineMove(); });
el('runParity').addEventListener('click', () => { void runParityFixtures(); });
el('reset').addEventListener('click', resetBoard);
el('flip').addEventListener('click', () => { orientation = orientation === 'white' ? 'black' : 'white'; renderStatic(); });

void init();
