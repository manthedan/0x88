import { Chessground } from 'chessground';
import type { Key } from 'chessground/types';
import { boardToFen, parseFen, squareName, START_FEN, type BoardState } from '../chess/board.ts';
import { legalMoves, makeMove } from '../chess/movegen.ts';
import { moveToUci, type Move } from '../chess/moveCodec.ts';
import { collectOrtRuntimeDiagnostics, describeOrtBackendConfig, type OrtExecutionProviderPreference, type OrtRuntimeDiagnostics } from '../nn/ortRuntime.ts';
import { buildBoardHistoryFromMoves } from './history.ts';
import { clearLc0ModelCache, describeLc0ModelLoad, loadLc0ModelForOrt } from './modelCache.ts';
import { Lc0OnnxEvaluator, type Lc0Evaluation, type Lc0EvaluatorInput } from './onnxEvaluator.ts';
import { Lc0PolicyOnlyPlayer } from './policyOnlyPlayer.ts';
import { Lc0PuctSearcher, type Lc0SearchChild, type Lc0SearchResult } from './search.ts';

type Ground = ReturnType<typeof Chessground>;
type NativePrior = { uci: string; index: number; prior: number };
type NativeRecord = { id: string; backend?: string; fen: string; startFen?: string; moves?: string[]; bestmove: string; topPriors: NativePrior[] };
type RenderableSearchResult = Pick<Lc0SearchResult, 'fen' | 'move' | 'visits' | 'value'> & { children: Lc0SearchChild[]; pv?: string[]; multiPv?: string[][]; elapsedMs?: number; cancelled?: boolean; stats?: Lc0SearchResult['search']['stats'] };
type WorkerResponse =
  | { type: 'ready'; id: number; backend: string; modelCache: string }
  | { type: 'evaluationResult'; id: number; result: Lc0Evaluation }
  | { type: 'searchResult'; id: number; result: RenderableSearchResult }
  | { type: 'error'; id: number; error: string };

type BrowserEvaluationChoice = { move?: string; evaluation: Lc0Evaluation };

type EngineReplyMode = 'policy' | 'search';

const params = new URLSearchParams(location.search);
const DEFAULT_MODEL = '/models/lc0/t1-256x10-distilled-swa-2432500.batch1.f32.onnx';
const MODEL_URL = params.get('model') ?? DEFAULT_MODEL;
const SEARCH_WORKER_REQUESTED = params.get('worker') === '1' || params.get('searchWorker') === '1';
const CACHE_MODEL = params.get('cache') === '1' || params.get('modelCache') === '1';
// Register the offline app-shell SW in production builds, or opt in with ?sw=1.
// Disabled in dev by default so it never serves stale HMR modules.
const SW_ENABLED = params.get('sw') === '1'
  || (params.get('sw') !== '0' && (import.meta as { env?: { PROD?: boolean } }).env?.PROD === true);

// Runtime-adjustable settings: seeded from query params, then driven by the UI.
let playerSide: 'white' | 'black' = params.get('side') === 'black' ? 'black' : 'white';
let searchVisits = Math.max(1, Math.floor(Number(params.get('visits') ?? '32') || 32));
let searchBatchSize = Math.max(1, Math.floor(Number(params.get('batch') ?? params.get('batchSize') ?? '1') || 1));
let searchMultiPv = Math.max(1, Math.floor(Number(params.get('multipv') ?? params.get('multiPv') ?? '1') || 1));
let engineReplyMode: EngineReplyMode = params.get('mode') === 'search' ? 'search' : 'policy';

let board: BoardState = parseFen(params.get('fen') ?? START_FEN);
let historyBoards: BoardState[] = [board];
let ground: Ground | null = null;
let player: Lc0PolicyOnlyPlayer | null = null;
let searcher: Lc0PuctSearcher | null = null;
let searchWorker: Worker | null = null;
let useSearchWorker = SEARCH_WORKER_REQUESTED;
let searchWorkerReady = false;
let searchWorkerBackend = '—';
let mainModelCacheStatus = CACHE_MODEL ? 'pending' : 'disabled';
let workerModelCacheStatus = '';
let workerRequestSeq = 0;
const workerPending = new Map<number, { resolve: (value: unknown) => void; reject: (error: Error) => void }>();
let busy = false;
let searching = false;
let mainSearchAbort: AbortController | null = null;
let activeWorkerSearchId: number | null = null;
let lastMove: string | null = null;
let renderSeq = 0;
let orientation: 'white' | 'black' = playerSide;
const playedMoves: string[] = [];

function el(id: string): HTMLElement {
  const node = document.getElementById(id);
  if (!node) throw new Error(`Missing #${id}`);
  return node;
}

function inputEl(id: string): HTMLInputElement {
  return el(id) as HTMLInputElement;
}

function selectEl(id: string): HTMLSelectElement {
  return el(id) as HTMLSelectElement;
}

function clampInt(raw: string, min: number, max: number, fallback: number): number {
  const value = Math.floor(Number(raw));
  if (!Number.isFinite(value)) return fallback;
  return Math.min(max, Math.max(min, value));
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
  el('analyze').toggleAttribute('disabled', busy || !searchAvailable());
  el('runParity').toggleAttribute('disabled', busy || !player);
  el('stopSearch').toggleAttribute('disabled', !searching);
}

function renderStatic() {
  el('fen').textContent = boardToFen(board);
  el('sideToMove').textContent = sideToMoveName();
  el('moveList').textContent = playedMoves.length ? playedMoves.join(' ') : '—';
  el('modelPath').textContent = MODEL_URL;
  el('modelCache').textContent = workerModelCacheStatus ? `main ${mainModelCacheStatus}; worker ${workerModelCacheStatus}` : mainModelCacheStatus;
  el('backend').textContent = describeOrtBackendConfig();
  el('status').textContent = player ? 'ready' : 'loading';
  el('searchMode').textContent = searchModeLabel();
  el('searchBatch').textContent = `${searchBatchSize}`;
  el('searchMove').textContent = `Search ${searchVisits}`;
  el('engineMove').toggleAttribute('disabled', busy || !player);
  el('searchMove').toggleAttribute('disabled', busy || !searchAvailable());
  el('analyze').toggleAttribute('disabled', busy || !searchAvailable());
  el('runParity').toggleAttribute('disabled', busy || !player);
  el('stopSearch').toggleAttribute('disabled', !searching);
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
  const visitsPerSecond = result.elapsedMs && result.elapsedMs > 0 ? result.visits / (result.elapsedMs / 1000) : undefined;
  const stats = result.stats;
  const batchStats = stats ? ` · eval batches ${stats.batchEvalCalls}/${stats.maxEvalBatch}` : '';
  el('searchLatency').textContent = result.elapsedMs === undefined ? '—' : `${result.elapsedMs.toFixed(0)} ms · ${visitsPerSecond?.toFixed(1) ?? '—'} visits/s${batchStats}`;
  if (result.multiPv && result.multiPv.length > 1) {
    el('searchPv').innerHTML = result.multiPv
      .map((line, i) => `<div><b>${i + 1}.</b> ${htmlEscape(line.join(' '))}</div>`)
      .join('');
  } else {
    el('searchPv').textContent = result.pv && result.pv.length ? result.pv.join(' ') : '—';
  }
  const maxVisits = Math.max(1, ...result.children.slice(0, 10).map((entry) => entry.visits));
  el('searchChildren').innerHTML = result.children.slice(0, 10).map((entry, i) => {
    const width = Math.max(2, (entry.visits / maxVisits) * 100).toFixed(1);
    return `<li class="${i === 0 ? 'best' : ''}"><span>${i + 1}</span><b>${htmlEscape(entry.uci)}</b><meter min="0" max="100" value="${width}"></meter><code>${entry.visits} · ${(entry.prior * 100).toFixed(1)}%</code></li>`;
  }).join('');
}

function clearSearchResult() {
  el('searchSummary').textContent = 'not run';
  el('searchLatency').textContent = '—';
  el('searchPv').textContent = '—';
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

function postWorkerRequest<T>(message: Record<string, unknown>, onId?: (id: number) => void): Promise<T> {
  if (!searchWorker) return Promise.reject(new Error('LC0 search worker unavailable'));
  const id = ++workerRequestSeq;
  onId?.(id);
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
  const ready = await postWorkerRequest<{ type: 'ready'; backend: string; modelCache: string }>({ type: 'init', modelUrl: MODEL_URL, ep: requestedWorkerEp(), cacheModel: CACHE_MODEL });
  searchWorkerReady = true;
  searchWorkerBackend = ready.backend;
  workerModelCacheStatus = ready.modelCache;
  renderStatic();
}

async function evaluateWithWorker(input: Lc0EvaluatorInput): Promise<BrowserEvaluationChoice> {
  const response = await postWorkerRequest<{ type: 'evaluationResult'; result: Lc0Evaluation }>({
    type: 'evaluate',
    input,
  });
  return { move: response.result.bestMove, evaluation: response.result };
}

async function searchWithWorker(): Promise<RenderableSearchResult> {
  const response = await postWorkerRequest<{ type: 'searchResult'; result: RenderableSearchResult }>({
    type: 'search',
    input: currentEvaluationInput(),
    visits: searchVisits,
    batchSize: searchBatchSize,
    multiPv: searchMultiPv,
  }, (id) => { activeWorkerSearchId = id; });
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
  const engineToMove = (playerSide === 'white' && board.turn === 'b') || (playerSide === 'black' && board.turn === 'w');
  if (engineToMove) {
    await engineMove();
  } else {
    renderEvaluation();
  }
}

function isAbortError(error: unknown): boolean {
  return error instanceof Error && error.name === 'AbortError';
}

function beginSearch() {
  searching = true;
  activeWorkerSearchId = null;
  // Worker searches are cancelled by message id; main-thread searches by signal.
  mainSearchAbort = useSearchWorker ? null : new AbortController();
}

function endSearch() {
  searching = false;
  mainSearchAbort = null;
  activeWorkerSearchId = null;
}

// Produce one search result for the current position. The caller owns the
// searching/abort lifecycle (beginSearch/endSearch) and the busy state.
async function executeSearchResult(): Promise<RenderableSearchResult> {
  if (useSearchWorker) return await searchWithWorker();
  const started = performance.now();
  // yieldEveryMs lets the main-thread search relinquish the event loop so the
  // Stop button stays responsive and the page never feels frozen.
  const search = await searcher!.search(currentEvaluationInput(), {
    visits: searchVisits,
    batchSize: searchBatchSize,
    multiPv: searchMultiPv,
    signal: mainSearchAbort!.signal,
    yieldEveryMs: 16,
  });
  return { ...search, stats: search.search.stats, elapsedMs: performance.now() - started };
}

async function searchRootPosition() {
  if (!searchAvailable() || busy) return;
  beginSearch();
  setBusy(true, `LC0 fixed-visit PUCT search running (${searchModeLabel()})… press Stop to cancel.`);
  try {
    const result = await executeSearchResult();
    if (result.cancelled) {
      clearSearchResult();
      el('message').textContent = `Search cancelled (${searchModeLabel()}).`;
    } else {
      renderSearchResult(result);
      el('message').textContent = `Search selected ${result.move ?? '—'} (${result.visits} visits, batch ${searchBatchSize}, fixed PUCT via ${searchModeLabel()}).`;
    }
  } catch (error) {
    if (isAbortError(error)) {
      clearSearchResult();
      el('message').textContent = `Search cancelled (${searchModeLabel()}).`;
    } else {
      el('message').textContent = `Search failed: ${(error as Error).message}`;
    }
  } finally {
    endSearch();
    setBusy(false);
    renderEvaluation();
  }
}

function stopSearch() {
  if (!searching) return;
  el('message').textContent = 'Cancelling search…';
  if (useSearchWorker) {
    if (activeWorkerSearchId !== null) searchWorker?.postMessage({ type: 'cancel', target: activeWorkerSearchId });
  } else {
    mainSearchAbort?.abort();
  }
}

async function engineMove() {
  if (!player || busy) return;
  const legal = legalMoves(board);
  if (!legal.length) {
    el('message').textContent = 'No legal engine move.';
    return;
  }
  const replyWithSearch = engineReplyMode === 'search' && searchAvailable();
  if (replyWithSearch) beginSearch();
  setBusy(true, replyWithSearch
    ? `LC0 engine replying with ${searchVisits}-visit search (${searchModeLabel()})… press Stop to cancel.`
    : 'LC0 policy-only engine thinking…');
  renderStatic();
  try {
    let uci: string | undefined;
    let note: string;
    if (replyWithSearch) {
      const result = await executeSearchResult();
      if (result.cancelled) {
        el('message').textContent = `Engine search reply cancelled (${searchModeLabel()}).`;
        return;
      }
      uci = result.move;
      note = `(${result.visits}-visit search via ${searchModeLabel()})`;
    } else {
      const choice = await player.chooseMove(currentEvaluationInput());
      uci = choice.move;
      note = '(argmax legal prior, no search)';
    }
    const move = uci ? legalMoveFromUci(uci) : undefined;
    if (!move) throw new Error(`Evaluator chose illegal or missing move: ${uci ?? 'none'}`);
    const played = applyMove(move);
    el('message').textContent = `Engine played ${played} ${note}`;
  } catch (error) {
    if (isAbortError(error)) {
      el('message').textContent = `Engine search reply cancelled (${searchModeLabel()}).`;
    } else {
      el('message').textContent = `Engine move failed: ${(error as Error).message}`;
    }
  } finally {
    if (replyWithSearch) endSearch();
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
  if ((!player && !searchWorkerReady) || busy) return;
  setBusy(true, 'Running FEN-only and explicit-history fixture parity in browser…');
  el('parity').textContent = 'running…';
  try {
    const records = [
      ...await fetchNativeRecords('/lc0/native_fen_only_blas.jsonl'),
      ...await fetchNativeRecords('/lc0/native_history_blas.jsonl'),
    ];
    const started = performance.now();
    let evaluated = 0;
    const failures: string[] = [];
    for (const native of records) {
      const input = native.moves ? { positions: buildBoardHistoryFromMoves(native.moves, native.startFen) } : native.fen;
      const choice = searchWorkerReady ? await evaluateWithWorker(input) : await player!.chooseMove(input);
      evaluated += 1;
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
      const elapsedMs = performance.now() - started;
      const evalsPerSecond = evaluated / Math.max(1e-9, elapsedMs / 1000);
      el('parity').textContent = `passed ${records.length}/${records.length} native BLAS fixtures · ${elapsedMs.toFixed(0)} ms · ${evalsPerSecond.toFixed(1)} eval/s`;
      el('message').textContent = `Browser FEN-only and explicit-history fixture parity passed (${evaluated} evals via ${describeOrtBackendConfig()}).`;
    }
  } catch (error) {
    el('parity').textContent = `failed: ${(error as Error).message}`;
    el('message').textContent = `Parity failed: ${(error as Error).message}`;
  } finally {
    setBusy(false);
    renderEvaluation();
  }
}

// Load a fresh root position with no real prior boards, matching a ?fen= load.
function loadPosition(next: BoardState) {
  board = next;
  historyBoards = [board];
  lastMove = null;
  playedMoves.length = 0;
  clearSearchResult();
}

function resetBoard() {
  loadPosition(parseFen(START_FEN));
  el('message').textContent = 'Reset to start position.';
  renderEvaluation();
}

function loadFenFromInput(): boolean {
  const raw = inputEl('fenInput').value.trim();
  if (!raw) {
    el('message').textContent = 'Enter a FEN to load.';
    return false;
  }
  let parsed: BoardState;
  try {
    parsed = parseFen(raw);
  } catch (error) {
    el('message').textContent = `Invalid FEN: ${(error as Error).message}`;
    return false;
  }
  loadPosition(parsed);
  el('message').textContent = `Loaded FEN. ${sideToMoveName()} to move.`;
  renderEvaluation();
  return true;
}

async function clearModelCache() {
  if (busy) return;
  try {
    const result = await clearLc0ModelCache();
    const summary = result.cleared
      ? `cleared ${result.removedEntries} entr${result.removedEntries === 1 ? 'y' : 'ies'}`
      : 'nothing to clear';
    mainModelCacheStatus = summary;
    workerModelCacheStatus = workerModelCacheStatus ? `stale (cleared); reload to refetch` : '';
    el('message').textContent = `Model cache ${summary}. Reload the page to refetch from the network.`;
  } catch (error) {
    el('message').textContent = `Clear model cache failed: ${(error as Error).message}`;
  }
  renderStatic();
}

function applySideChange(side: 'white' | 'black') {
  playerSide = side;
  orientation = side;
  renderStatic();
}

// Surface whether WebGPU is actually driving inference or silently fell back to
// WASM, so a degraded backend is visible instead of looking like success.
function renderGpuStatus(diag: OrtRuntimeDiagnostics) {
  const node = el('gpuStatus');
  const requestedGpu = diag.requestedEp !== 'wasm';
  const usingGpu = diag.resolvedExecutionProviders.includes('webgpu');
  const lastWebgpuError = [...diag.sessionAttempts].reverse().find((a) => a.providers.includes('webgpu') && a.error)?.error;
  let text: string;
  let warn = false;
  if (usingGpu) {
    text = 'active';
  } else if (!diag.webgpuAvailable) {
    text = 'unavailable — no navigator.gpu';
    warn = requestedGpu && diag.requestedEp !== 'auto';
  } else if (requestedGpu) {
    text = `requested → fell back to WASM${lastWebgpuError ? ` (${lastWebgpuError})` : ''}`;
    warn = true;
  } else {
    text = 'available — WASM selected';
  }
  node.textContent = text;
  node.classList.toggle('warn', warn);
}

async function init() {
  el('message').textContent = 'Loading LC0 ONNX model…';
  renderStatic();
  try {
    const modelLoad = await loadLc0ModelForOrt(MODEL_URL, { cache: CACHE_MODEL });
    mainModelCacheStatus = describeLc0ModelLoad(modelLoad);
    const evaluator = await Lc0OnnxEvaluator.create(modelLoad.model);
    player = new Lc0PolicyOnlyPlayer(evaluator);
    searcher = new Lc0PuctSearcher(evaluator);
    const diagnostics = await collectOrtRuntimeDiagnostics();
    el('backend').textContent = diagnostics.describe;
    renderGpuStatus(diagnostics);
    if (SEARCH_WORKER_REQUESTED) {
      el('message').textContent = 'Initializing LC0 search worker…';
      try {
        await initSearchWorker();
      } catch (error) {
        searchWorker?.terminate();
        searchWorker = null;
        searchWorkerReady = false;
        useSearchWorker = false;
        workerModelCacheStatus = 'worker unavailable';
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

function seedSettingsInputs() {
  inputEl('visitsInput').value = String(searchVisits);
  inputEl('batchInput').value = String(searchBatchSize);
  inputEl('multiPvInput').value = String(searchMultiPv);
  selectEl('sideSelect').value = playerSide;
  selectEl('modeSelect').value = engineReplyMode;
}

el('engineMove').addEventListener('click', () => { void engineMove(); });
el('searchMove').addEventListener('click', () => { void searchRootPosition(); });
el('stopSearch').addEventListener('click', stopSearch);
// "Analyze position" runs a search on the current board. Loading a different
// position is the explicit job of the FEN box + Load FEN.
el('analyze').addEventListener('click', () => { void searchRootPosition(); });
el('runParity').addEventListener('click', () => { void runParityFixtures(); });
el('reset').addEventListener('click', resetBoard);
el('flip').addEventListener('click', () => { orientation = orientation === 'white' ? 'black' : 'white'; renderStatic(); });
el('loadFen').addEventListener('click', () => { loadFenFromInput(); });
el('clearCache').addEventListener('click', () => { void clearModelCache(); });
inputEl('fenInput').addEventListener('keydown', (event) => { if ((event as KeyboardEvent).key === 'Enter') loadFenFromInput(); });
inputEl('visitsInput').addEventListener('change', () => {
  searchVisits = clampInt(inputEl('visitsInput').value, 1, 100000, searchVisits);
  inputEl('visitsInput').value = String(searchVisits);
  renderStatic();
});
inputEl('batchInput').addEventListener('change', () => {
  searchBatchSize = clampInt(inputEl('batchInput').value, 1, 512, searchBatchSize);
  inputEl('batchInput').value = String(searchBatchSize);
  renderStatic();
});
inputEl('multiPvInput').addEventListener('change', () => {
  searchMultiPv = clampInt(inputEl('multiPvInput').value, 1, 20, searchMultiPv);
  inputEl('multiPvInput').value = String(searchMultiPv);
  renderStatic();
});
selectEl('sideSelect').addEventListener('change', () => {
  applySideChange(selectEl('sideSelect').value === 'black' ? 'black' : 'white');
});
selectEl('modeSelect').addEventListener('change', () => {
  engineReplyMode = selectEl('modeSelect').value === 'search' ? 'search' : 'policy';
  el('message').textContent = `Engine reply mode: ${engineReplyMode === 'search' ? 'fixed-visit search' : 'policy-only'}.`;
});

function registerAppServiceWorker() {
  if (!SW_ENABLED || typeof navigator === 'undefined' || !('serviceWorker' in navigator)) return;
  window.addEventListener('load', () => {
    navigator.serviceWorker.register('/lc0-sw.js').then((registration) => {
      console.info('LC0 app shell service worker registered.', registration.scope);
    }).catch((error) => {
      console.warn('LC0 app shell service worker registration failed.', error);
    });
  });
}

seedSettingsInputs();
registerAppServiceWorker();
void init();
