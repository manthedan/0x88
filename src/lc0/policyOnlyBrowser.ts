import { Chessground } from 'chessground';
import type { Key } from 'chessground/types';
import { boardToFen, parseFen, squareName, START_FEN, type BoardState } from '../chess/board.ts';
import { legalMoves, makeMove } from '../chess/movegen.ts';
import { moveToUci, type Move } from '../chess/moveCodec.ts';
import { collectOrtRuntimeDiagnostics, describeOrtBackendConfig, type OrtExecutionProviderPreference } from '../nn/ortRuntime.ts';
import { buildBoardHistoryFromMoves } from './history.ts';
import { Lc0OnnxEvaluator } from './onnxEvaluator.ts';
import { Lc0PolicyOnlyPlayer } from './policyOnlyPlayer.ts';
import { Lc0PuctSearcher, type Lc0SearchChild, type Lc0SearchResult } from './search.ts';

type Ground = ReturnType<typeof Chessground>;
type NativePrior = { uci: string; index: number; prior: number };
type NativeRecord = { id: string; backend?: string; fen: string; startFen?: string; moves?: string[]; bestmove: string; topPriors: NativePrior[] };
type RenderableSearchResult = Pick<Lc0SearchResult, 'fen' | 'move' | 'visits' | 'value'> & { children: Lc0SearchChild[]; elapsedMs?: number };
type WorkerResponse =
  | { type: 'ready'; id: number; backend: string }
  | { type: 'searchResult'; id: number; result: RenderableSearchResult }
  | { type: 'error'; id: number; error: string };

const params = new URLSearchParams(location.search);
const DEFAULT_MODEL = '/models/lc0/t1-256x10-distilled-swa-2432500.batch1.f32.onnx';
const MODEL_URL = params.get('model') ?? DEFAULT_MODEL;
const PLAYER_SIDE = params.get('side') === 'black' ? 'black' : 'white';
const SEARCH_VISITS = Math.max(1, Math.floor(Number(params.get('visits') ?? '32') || 32));
const SEARCH_WORKER_REQUESTED = params.get('worker') === '1' || params.get('searchWorker') === '1';

let board: BoardState = parseFen(params.get('fen') ?? START_FEN);
let historyBoards: BoardState[] = [board];
let ground: Ground | null = null;
let player: Lc0PolicyOnlyPlayer | null = null;
let searcher: Lc0PuctSearcher | null = null;
let searchWorker: Worker | null = null;
let useSearchWorker = SEARCH_WORKER_REQUESTED;
let searchWorkerReady = false;
let searchWorkerBackend = '—';
let workerRequestSeq = 0;
const workerPending = new Map<number, { resolve: (value: unknown) => void; reject: (error: Error) => void }>();
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

function requestedWorkerEp(): OrtExecutionProviderPreference {
  const raw = String(params.get('ortEp') ?? params.get('ep') ?? params.get('executionProviders') ?? '').toLowerCase();
  if (raw === 'webgpu' || raw === 'gpu') return 'webgpu';
  if (raw === 'webgpu,wasm' || raw === 'webgpu+wasm' || raw === 'gpu,wasm' || raw === 'gpu+wasm') return 'webgpu,wasm';
  if (raw === 'auto' || raw === '') return 'auto';
  return 'wasm';
}

function searchAvailable(): boolean {
  return useSearchWorker ? searchWorkerReady : !!searcher;
}

function searchModeLabel(): string {
  if (!useSearchWorker) return 'main thread';
  return searchWorkerReady ? `worker (${searchWorkerBackend})` : 'worker loading';
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

function currentEvaluationInput(): string | { positions: BoardState[] } {
  // A direct ?fen= load has no real prior boards. Evaluate it through the
  // evaluator's normal FEN-only path so non-start FENs get LC0-compatible
  // synthetic history. Once a move is played, preserve the actual browser move
  // history from the loaded root.
  return playedMoves.length === 0 ? boardToFen(board) : { positions: historyBoards };
}

function applyMove(move: Move): string {
  const uci = moveToUci(move);
  board = makeMove(board, move);
  historyBoards.push(board);
  lastMove = uci;
  playedMoves.push(uci);
  clearSearchResult();
  return uci;
}

function setBusy(next: boolean, message?: string) {
  busy = next;
  if (message) el('message').textContent = message;
  el('engineMove').toggleAttribute('disabled', busy || !player);
  el('searchMove').toggleAttribute('disabled', busy || !searchAvailable());
  el('runParity').toggleAttribute('disabled', busy || !player);
}

function renderStatic() {
  el('fen').textContent = boardToFen(board);
  el('sideToMove').textContent = sideToMoveName();
  el('moveList').textContent = playedMoves.length ? playedMoves.join(' ') : '—';
  el('modelPath').textContent = MODEL_URL;
  el('backend').textContent = describeOrtBackendConfig();
  el('status').textContent = player ? 'ready' : 'loading';
  el('searchMode').textContent = searchModeLabel();
  el('searchMove').textContent = `Search ${SEARCH_VISITS}`;
  el('engineMove').toggleAttribute('disabled', busy || !player);
  el('searchMove').toggleAttribute('disabled', busy || !searchAvailable());
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

function renderSearchResult(result: RenderableSearchResult) {
  el('searchSummary').textContent = `${result.move ?? '—'} · ${result.visits} visits · Q ${result.value.toFixed(5)}`;
  el('searchLatency').textContent = result.elapsedMs === undefined ? '—' : `${result.elapsedMs.toFixed(0)} ms`;
  const maxVisits = Math.max(1, ...result.children.slice(0, 10).map((entry) => entry.visits));
  el('searchChildren').innerHTML = result.children.slice(0, 10).map((entry, i) => {
    const width = Math.max(2, (entry.visits / maxVisits) * 100).toFixed(1);
    return `<li class="${i === 0 ? 'best' : ''}"><span>${i + 1}</span><b>${htmlEscape(entry.uci)}</b><meter min="0" max="100" value="${width}"></meter><code>${entry.visits} · ${(entry.prior * 100).toFixed(1)}%</code></li>`;
  }).join('');
}

function clearSearchResult() {
  el('searchSummary').textContent = 'not run';
  el('searchLatency').textContent = '—';
  el('searchChildren').innerHTML = '';
}

function renderEvaluation() {
  const seq = ++renderSeq;
  renderStatic();
  if (!player) return;
  player.chooseMove(currentEvaluationInput()).then((choice) => {
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

function postWorkerRequest<T>(message: Record<string, unknown>): Promise<T> {
  if (!searchWorker) return Promise.reject(new Error('LC0 search worker unavailable'));
  const id = ++workerRequestSeq;
  return new Promise<T>((resolve, reject) => {
    workerPending.set(id, { resolve: resolve as (value: unknown) => void, reject });
    searchWorker!.postMessage({ ...message, id });
  });
}

async function initSearchWorker(): Promise<void> {
  searchWorker = new Worker(new URL('./searchWorker.ts', import.meta.url), { type: 'module' });
  searchWorker.addEventListener('message', (event: MessageEvent<WorkerResponse>) => {
    const message = event.data;
    const pending = workerPending.get(message.id);
    if (!pending) return;
    workerPending.delete(message.id);
    if (message.type === 'error') pending.reject(new Error(message.error));
    else pending.resolve(message);
  });
  searchWorker.addEventListener('error', (event) => {
    for (const pending of workerPending.values()) pending.reject(new Error(event.message || 'LC0 search worker error'));
    workerPending.clear();
  });
  const ready = await postWorkerRequest<{ type: 'ready'; backend: string }>({ type: 'init', modelUrl: MODEL_URL, ep: requestedWorkerEp() });
  searchWorkerReady = true;
  searchWorkerBackend = ready.backend;
  renderStatic();
}

async function searchWithWorker(): Promise<RenderableSearchResult> {
  const response = await postWorkerRequest<{ type: 'searchResult'; result: RenderableSearchResult }>({
    type: 'search',
    input: currentEvaluationInput(),
    visits: SEARCH_VISITS,
  });
  return response.result;
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

async function searchRootPosition() {
  if (!searchAvailable() || busy) return;
  setBusy(true, `LC0 fixed-visit PUCT search running (${searchModeLabel()})…`);
  try {
    const started = performance.now();
    const result = useSearchWorker
      ? await searchWithWorker()
      : { ...await searcher!.search(currentEvaluationInput(), { visits: SEARCH_VISITS }), elapsedMs: performance.now() - started };
    renderSearchResult(result);
    el('message').textContent = `Search selected ${result.move ?? '—'} (${result.visits} visits, fixed PUCT via ${searchModeLabel()}).`;
  } catch (error) {
    el('message').textContent = `Search failed: ${(error as Error).message}`;
  } finally {
    setBusy(false);
    renderEvaluation();
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
    const choice = await player.chooseMove(currentEvaluationInput());
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

async function fetchNativeRecords(path: string): Promise<NativeRecord[]> {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`native fixture fetch failed for ${path}: ${response.status}`);
  return (await response.text()).trim().split('\n').filter(Boolean).map((line) => JSON.parse(line) as NativeRecord);
}

async function runParityFixtures() {
  if (!player || busy) return;
  setBusy(true, 'Running FEN-only and explicit-history fixture parity in browser…');
  el('parity').textContent = 'running…';
  try {
    const records = [
      ...await fetchNativeRecords('/lc0/native_fen_only_blas.jsonl'),
      ...await fetchNativeRecords('/lc0/native_history_blas.jsonl'),
    ];
    const failures: string[] = [];
    for (const native of records) {
      const input = native.moves ? { positions: buildBoardHistoryFromMoves(native.moves, native.startFen) } : native.fen;
      const choice = await player.chooseMove(input);
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
      el('message').textContent = 'Browser FEN-only and explicit-history fixture parity passed.';
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
  historyBoards = [board];
  lastMove = null;
  playedMoves.length = 0;
  clearSearchResult();
  el('message').textContent = 'Reset to start position.';
  renderEvaluation();
}

async function init() {
  el('message').textContent = 'Loading LC0 ONNX model…';
  renderStatic();
  try {
    const evaluator = await Lc0OnnxEvaluator.create(MODEL_URL);
    player = new Lc0PolicyOnlyPlayer(evaluator);
    searcher = new Lc0PuctSearcher(evaluator);
    const diagnostics = await collectOrtRuntimeDiagnostics();
    el('backend').textContent = diagnostics.describe;
    if (SEARCH_WORKER_REQUESTED) {
      el('message').textContent = 'Initializing LC0 search worker…';
      try {
        await initSearchWorker();
      } catch (error) {
        searchWorker?.terminate();
        searchWorker = null;
        searchWorkerReady = false;
        useSearchWorker = false;
        console.warn('LC0 search worker failed; falling back to main-thread search.', error);
      }
    }
    el('message').textContent = 'Ready. Drag a legal move or ask the engine to move.';
    renderEvaluation();
    if (params.get('parity') === '1' || params.get('fixtures') === '1') await runParityFixtures();
    if (params.get('search') === '1') await searchRootPosition();
    if (params.get('engineMove') === '1') await engineMove();
  } catch (error) {
    el('message').textContent = `Model load failed: ${(error as Error).message}`;
    renderStatic();
  }
}

el('engineMove').addEventListener('click', () => { void engineMove(); });
el('searchMove').addEventListener('click', () => { void searchRootPosition(); });
el('runParity').addEventListener('click', () => { void runParityFixtures(); });
el('reset').addEventListener('click', resetBoard);
el('flip').addEventListener('click', () => { orientation = orientation === 'white' ? 'black' : 'white'; renderStatic(); });

void init();
