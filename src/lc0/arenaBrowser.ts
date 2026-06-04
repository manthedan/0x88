import { Chessground } from 'chessground';
import type { DrawShape } from 'chessground/draw';
import type { Key } from 'chessground/types';
import { boardToFen, parseFen, START_FEN, type BoardState } from '../chess/board.ts';
import { legalMoves, makeMove } from '../chess/movegen.ts';
import { moveToUci, type Move } from '../chess/moveCodec.ts';
import { gameTreeToPgn } from '../chess/pgn.ts';
import { applyGameResult, gauntletPairings, initStandings, rankedStandings, roundRobinPairings, type ArenaPairing, type Standing } from './arena.ts';
import { BUILTIN_ARENA_OPENINGS, parseArenaOpenings, scheduleOpenings, type ArenaOpening } from './arenaOpenings.ts';
import { gameOutcome, type GameResultCode } from './engineBattle.ts';
import { GameTree } from './gameTree.ts';
import { loadLc0ModelForOrt } from './modelCache.ts';
import { collectOrtRuntimeDiagnostics } from '../nn/ortRuntime.ts';
import { CachedLc0Evaluator, Lc0OnnxEvaluator, type Lc0EvaluationCacheMetrics } from './onnxEvaluator.ts';
import { Lc0PolicyOnlyPlayer } from './policyOnlyPlayer.ts';
import { Lc0PuctSearcher, type Lc0SearchResult } from './search.ts';
import type { Node as PuctNode } from '../search/puct.ts';
import { DEFAULT_STOCKFISH_FLAVOR, StockfishEngine, normalizeStockfishFlavor, stockfishFlavorLabel, stockfishFlavorRequiresIsolation, stockfishFlavorUrl, type StockfishFlavor } from './stockfishEngine.ts';

type Ground = ReturnType<typeof Chessground>;
interface ArenaEngine {
  id: string;
  name: string;
  move(positions: BoardState[], signal: AbortSignal): Promise<string | null>;
  warmup?(signal: AbortSignal): Promise<void>;
}
interface GameRecord { pgn: string; }
interface ScheduledArenaGame extends ArenaPairing { opening: ArenaOpening; }
interface Lc0TreeTelemetry {
  engineName: string;
  searches: number;
  rootReused: number;
  completedVisits: number;
  reusedRootVisits: number;
  evalCalls: number;
  cacheHits: number;
  neuralEvalMisses: number;
  transpositionHits: number;
  replyChecks: number;
  replyParentsExpanded: number;
  replyVisited: number;
  replyTopPolicy5: number;
  replyTopVisits5: number;
}
interface PendingLc0ReplyProbe {
  engineId: string;
  engineName: string;
  afterMoveFen: string;
  child: PuctNode | null;
}

const params = new URLSearchParams(location.search);
const MODEL_URL = params.get('model') ?? '/models/lc0/t1-256x10-distilled-swa-2432500.batch1.f32.onnx';
const REQUESTED_STOCKFISH_FLAVOR = normalizeStockfishFlavor(params.get('sfFlavor') ?? params.get('stockfish'));

let ground: Ground | null = null;
let board: BoardState = parseFen(START_FEN);
let historyBoards: BoardState[] = [board];
let lastUci: string | null = null;
let boardWhiteName: string | null = null;
let boardBlackName: string | null = null;
let running = false;
let abort: AbortController | null = null;
let player: Lc0PolicyOnlyPlayer | null = null;
let searcher: Lc0PuctSearcher | null = null;
let lc0Cache: CachedLc0Evaluator | null = null;
let stockfish: StockfishEngine | null = null;
let runtimeIsolation = typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated;
let runtimeSharedArrayBuffer = typeof SharedArrayBuffer !== 'undefined';
const engines = new Map<string, ArenaEngine>();
const lc0Searchers = new Map<string, Lc0PuctSearcher>();
const lastLc0SearchResults = new Map<string, Lc0SearchResult>();
const pendingLc0ReplyProbes = new Map<string, PendingLc0ReplyProbe>();
const lc0TreeTelemetry = new Map<string, Lc0TreeTelemetry>();
const games: GameRecord[] = [];

function el(id: string): HTMLElement {
  const node = document.getElementById(id);
  if (!node) throw new Error(`Missing #${id}`);
  return node;
}
function inputEl(id: string): HTMLInputElement { return el(id) as HTMLInputElement; }
function selectEl(id: string): HTMLSelectElement { return el(id) as HTMLSelectElement; }
function htmlEscape(value: unknown): string {
  return String(value).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]!));
}

function renderSideLabels() {
  const update = (id: string, color: 'White' | 'Black', engineName: string | null, active: boolean) => {
    const node = el(id);
    node.classList.toggle('active', active);
    node.innerHTML = `<span><span class="color">${color}</span> <span class="engine">${htmlEscape(engineName ?? '—')}</span></span>${active ? '<span class="turn">to move</span>' : ''}`;
  };
  update('blackSideLabel', 'Black', boardBlackName, board.turn === 'b');
  update('whiteSideLabel', 'White', boardWhiteName, board.turn === 'w');
}

function setBoardSideEngines(whiteName: string | null, blackName: string | null): void {
  boardWhiteName = whiteName;
  boardBlackName = blackName;
  renderSideLabels();
}

function renderBoard() {
  const config = {
    orientation: 'white' as const,
    fen: boardToFen(board).split(' ')[0],
    coordinates: true,
    viewOnly: true,
    highlight: { lastMove: true, check: true },
    animation: { enabled: true, duration: 140 },
    lastMove: lastUci ? [lastUci.slice(0, 2) as Key, lastUci.slice(2, 4) as Key] : undefined,
  };
  if (!ground) ground = Chessground(el('ground'), config);
  else ground.set(config);
  const shapes: DrawShape[] = lastUci && lastUci.length >= 4
    ? [{ orig: lastUci.slice(0, 2) as Key, dest: lastUci.slice(2, 4) as Key, brush: 'green' }] : [];
  ground.setAutoShapes(shapes);
  renderSideLabels();
}

function selectedEngineIds(): string[] {
  return [...el('engines').querySelectorAll('input:checked')].map((node) => (node as HTMLInputElement).value);
}

function setEngineCheckboxLabel(engineId: string, labelText: string): void {
  const input = el('engines').querySelector(`input[value="${engineId}"]`) as HTMLInputElement | null;
  const label = input?.closest('label');
  if (!input || !label) return;
  const text = [...label.childNodes].find((node) => node.nodeType === Node.TEXT_NODE);
  if (text) text.textContent = ` ${labelText}`;
}

function arenaBudgetMode(): 'fixed' | 'movetime' {
  return selectEl('budgetModeSelect').value === 'movetime' ? 'movetime' : 'fixed';
}

function arenaMovetimeMs(): number {
  return Math.max(10, Math.min(60000, Math.floor(Number(inputEl('movetimeInput').value) || 500)));
}

function arenaCacheEntries(): number {
  return Math.max(0, Math.min(100000, Math.floor(Number(inputEl('cacheEntriesInput').value) || 0)));
}

function stockfishThreads(): number {
  const requested = Math.max(1, Math.min(32, Math.floor(Number(inputEl('stockfishThreadsInput').value) || 1)));
  return stockfishFlavorRequiresIsolation(selectedStockfishFlavor()) ? requested : 1;
}

function cacheMetricsText(metrics: Lc0EvaluationCacheMetrics | undefined): string {
  if (!metrics) return 'NN cache: unavailable';
  return `NN cache: ${metrics.entries}/${metrics.maxEntries} entries · ${metrics.hits} hit${metrics.hits === 1 ? '' : 's'} · ${metrics.misses} miss${metrics.misses === 1 ? '' : 'es'}`;
}

function renderCacheInfo(): void {
  lc0Cache?.setMaxEntries(arenaCacheEntries());
  el('cacheInfo').textContent = cacheMetricsText(lc0Cache?.metrics());
  renderSearchTelemetryInfo();
}

function emptyTreeTelemetry(engineName: string): Lc0TreeTelemetry {
  return {
    engineName,
    searches: 0,
    rootReused: 0,
    completedVisits: 0,
    reusedRootVisits: 0,
    evalCalls: 0,
    cacheHits: 0,
    neuralEvalMisses: 0,
    transpositionHits: 0,
    replyChecks: 0,
    replyParentsExpanded: 0,
    replyVisited: 0,
    replyTopPolicy5: 0,
    replyTopVisits5: 0,
  };
}

function telemetryFor(engineId: string, engineName: string): Lc0TreeTelemetry {
  const existing = lc0TreeTelemetry.get(engineId);
  if (existing) {
    existing.engineName = engineName;
    return existing;
  }
  const created = emptyTreeTelemetry(engineName);
  lc0TreeTelemetry.set(engineId, created);
  return created;
}

function ratioText(numerator: number, denominator: number): string {
  return denominator > 0 ? `${numerator}/${denominator}` : '0/0';
}

function searchTelemetryText(): string {
  const entries = [...lc0TreeTelemetry.values()].filter((t) => t.searches || t.replyChecks);
  if (!entries.length) return 'LC0 tree: waiting for searches…';
  return `LC0 tree: ${entries.map((t) => {
    const fresh = Math.max(0, t.searches - t.rootReused);
    return `${t.engineName}: reuse ${ratioText(t.rootReused, t.searches)} (${fresh} fresh) · reused visits ${t.reusedRootVisits} · reply parent ${ratioText(t.replyParentsExpanded, t.replyChecks)} · reply visited ${ratioText(t.replyVisited, t.replyChecks)} · opp reply top-policy≤5 ${ratioText(t.replyTopPolicy5, t.replyChecks)} · top-visits≤5 ${ratioText(t.replyTopVisits5, t.replyChecks)} · evals ${t.evalCalls} · cache hits ${t.cacheHits} · trans ${t.transpositionHits}`;
  }).join(' | ')}`;
}

function renderSearchTelemetryInfo(): void {
  el('searchTelemetryInfo').textContent = searchTelemetryText();
}

function isLc0SearchEngine(engineId: string): boolean {
  return engineId.startsWith('lc0-s');
}

function childForRootMove(result: Lc0SearchResult | undefined, uci: string): PuctNode | null {
  const root = result?.search.root;
  if (!root?.expanded) return null;
  return root.edges.find((edge) => moveToUci(edge.move) === uci)?.child ?? null;
}

function recordLc0SearchTelemetry(engineId: string, engineName: string, result: Lc0SearchResult): void {
  const stats = result.search.stats;
  const t = telemetryFor(engineId, engineName);
  const completed = stats?.completedVisits ?? 0;
  t.searches += 1;
  if (stats?.rootReused) t.rootReused += 1;
  t.completedVisits += completed;
  t.reusedRootVisits += Math.max(0, result.visits - completed);
  t.evalCalls += stats?.evalCalls ?? 0;
  t.cacheHits += stats?.cacheHits ?? 0;
  t.neuralEvalMisses += stats?.neuralEvalMisses ?? 0;
  t.transpositionHits += stats?.transpositionHits ?? 0;
  lastLc0SearchResults.set(engineId, result);
  renderSearchTelemetryInfo();
}

function noteLc0MoveForReplyProbe(engine: ArenaEngine, move: Move, nextBoard: BoardState): void {
  if (!isLc0SearchEngine(engine.id)) return;
  const result = lastLc0SearchResults.get(engine.id);
  const child = childForRootMove(result, moveToUci(move));
  pendingLc0ReplyProbes.set(engine.id, { engineId: engine.id, engineName: engine.name, afterMoveFen: boardToFen(nextBoard), child });
}

function recordPendingLc0ReplyProbes(replyEngine: ArenaEngine, replyBoard: BoardState, replyMove: Move): void {
  const replyFen = boardToFen(replyBoard);
  const replyUci = moveToUci(replyMove);
  for (const [engineId, pending] of [...pendingLc0ReplyProbes]) {
    if (engineId === replyEngine.id || pending.afterMoveFen !== replyFen) continue;
    const t = telemetryFor(pending.engineId, pending.engineName);
    t.replyChecks += 1;
    const child = pending.child;
    if (child?.expanded) {
      t.replyParentsExpanded += 1;
      const policyRank = child.edges.slice().sort((a, b) => b.prior - a.prior).findIndex((edge) => moveToUci(edge.move) === replyUci);
      const visitsRank = child.edges.slice().sort((a, b) => b.visits - a.visits || b.prior - a.prior).findIndex((edge) => moveToUci(edge.move) === replyUci);
      const replyEdge = child.edges.find((edge) => moveToUci(edge.move) === replyUci);
      if (replyEdge && replyEdge.visits > 0) t.replyVisited += 1;
      if (policyRank >= 0 && policyRank < 5) t.replyTopPolicy5 += 1;
      if (visitsRank >= 0 && visitsRank < 5) t.replyTopVisits5 += 1;
    }
    pendingLc0ReplyProbes.delete(engineId);
  }
  renderSearchTelemetryInfo();
}

function threadedStockfishAvailable(): boolean {
  return runtimeIsolation && runtimeSharedArrayBuffer;
}

function selectedStockfishFlavor(): StockfishFlavor {
  const selected = normalizeStockfishFlavor(selectEl('stockfishFlavorSelect').value);
  return stockfishFlavorRequiresIsolation(selected) && !threadedStockfishAvailable() ? DEFAULT_STOCKFISH_FLAVOR : selected;
}

function stockfishName(depth: number): string {
  return `${stockfishFlavorLabel(selectedStockfishFlavor())} d${depth}`;
}

function refreshStockfishFlavorAvailability(): void {
  const select = selectEl('stockfishFlavorSelect');
  for (const option of [...select.options]) {
    const flavor = normalizeStockfishFlavor(option.value);
    option.disabled = stockfishFlavorRequiresIsolation(flavor) && !threadedStockfishAvailable();
    if (option.disabled) option.textContent = option.textContent.replace(/ \(needs isolation\)$/, '') + ' (needs isolation)';
    else option.textContent = option.textContent.replace(/ \(needs isolation\)$/, '');
  }
  if (stockfishFlavorRequiresIsolation(normalizeStockfishFlavor(select.value)) && !threadedStockfishAvailable()) select.value = DEFAULT_STOCKFISH_FLAVOR;
  select.disabled = running;
  inputEl('stockfishThreadsInput').disabled = running || !stockfishFlavorRequiresIsolation(selectedStockfishFlavor());
  inputEl('stockfishThreadsInput').value = String(stockfishThreads());
}

async function renderRuntimeBadge(): Promise<void> {
  const badge = el('runtimeBadge');
  try {
    const diag = await collectOrtRuntimeDiagnostics({ probeAdapter: true });
    runtimeIsolation = diag.crossOriginIsolated === true;
    runtimeSharedArrayBuffer = diag.wasm.sharedArrayBuffer === true;
    refreshStockfishFlavorAvailability();
    const webgpu = diag.webgpuAvailable ? (diag.adapter?.ok === false ? 'WebGPU unavailable/blocked' : 'WebGPU available') : 'WebGPU unavailable';
    const isolated = runtimeIsolation ? 'isolated' : 'not isolated';
    const sab = runtimeSharedArrayBuffer ? 'SAB yes' : 'SAB no';
    const sfThreads = threadedStockfishAvailable() ? 'SF threaded yes' : 'SF threaded no';
    const ortThreads = `ORT wasm threads ${diag.wasm.numThreads ?? '?'}`;
    const ortSessions = `ORT sessions ${diag.sessions.active} active`;
    badge.textContent = `Runtime: ${isolated} · ${sab} · ${webgpu} · ${diag.describe} · ${ortThreads} · ${ortSessions} · ${sfThreads}`;
    badge.classList.toggle('ready', runtimeIsolation && (diag.webgpuAvailable || runtimeSharedArrayBuffer));
    badge.classList.toggle('warn', !runtimeIsolation || !diag.webgpuAvailable);
  } catch (error) {
    badge.textContent = `Runtime detection failed: ${(error as Error).message}`;
    badge.classList.add('warn');
  }
}

function lc0SearcherFor(engineId: string): Lc0PuctSearcher {
  const existing = lc0Searchers.get(engineId);
  if (existing) return existing;
  if (!lc0Cache) throw new Error('LC0 evaluator is not initialized');
  const created = new Lc0PuctSearcher(lc0Cache);
  lc0Searchers.set(engineId, created);
  return created;
}

function resetLc0SearchTrees(ids?: string[]): void {
  const targets = ids ? ids.map((id) => lc0Searchers.get(id)).filter((entry): entry is Lc0PuctSearcher => !!entry) : [...lc0Searchers.values()];
  for (const search of targets) search.resetTree();
}

function buildEngines() {
  engines.clear();
  const warmupPositions = [parseFen(START_FEN)];
  const lc0Search = (engineId: string, visits: number): ArenaEngine['move'] => async (positions, signal) => {
    const timed = arenaBudgetMode() === 'movetime';
    const result = await lc0SearcherFor(engineId).search({ positions }, {
      visits: timed ? undefined : visits,
      movetimeMs: timed ? arenaMovetimeMs() : undefined,
      signal,
      yieldEveryMs: 16,
      reuseTree: true,
    });
    recordLc0SearchTelemetry(engineId, engines.get(engineId)?.name ?? engineId, result);
    return result.move ?? null;
  };
  const sf = (depth: number): ArenaEngine['move'] => async (positions, signal) => {
    if (arenaBudgetMode() === 'movetime') stockfish!.setOptions({ depth: undefined, movetimeMs: arenaMovetimeMs(), threads: stockfishThreads() });
    else stockfish!.setOptions({ depth, movetimeMs: undefined, threads: stockfishThreads() });
    return stockfish!.bestMove(boardToFen(positions[positions.length - 1]), signal);
  };
  const lc0SearchWarmup = (engineId: string) => async (signal: AbortSignal) => {
    const search = lc0SearcherFor(engineId);
    await search.search({ positions: warmupPositions }, { visits: 1, signal, yieldEveryMs: 16 });
    search.resetTree();
    renderCacheInfo();
  };
  const stockfishWarmup = async (signal: AbortSignal) => {
    stockfish!.setOptions({ depth: 1, movetimeMs: undefined, threads: stockfishThreads() });
    await stockfish!.bestMove(START_FEN, signal);
  };
  engines.set('lc0-policy', {
    id: 'lc0-policy',
    name: 'LC0 policy',
    move: async (positions) => (await player!.chooseMove({ positions })).move ?? null,
    warmup: async () => { await player!.chooseMove({ positions: warmupPositions }); },
  });
  engines.set('lc0-s100', { id: 'lc0-s100', name: 'LC0 search 100', move: lc0Search('lc0-s100', 100), warmup: lc0SearchWarmup('lc0-s100') });
  engines.set('lc0-s400', { id: 'lc0-s400', name: 'LC0 search 400', move: lc0Search('lc0-s400', 400), warmup: lc0SearchWarmup('lc0-s400') });
  engines.set('sf-d4', { id: 'sf-d4', name: stockfishName(4), move: sf(4), warmup: stockfishWarmup });
  engines.set('sf-d8', { id: 'sf-d8', name: stockfishName(8), move: sf(8), warmup: stockfishWarmup });
  setEngineCheckboxLabel('sf-d4', stockfishName(4));
  setEngineCheckboxLabel('sf-d8', stockfishName(8));
}

function refreshChampionOptions() {
  const ids = selectedEngineIds();
  const select = selectEl('championSelect');
  const prev = select.value;
  select.innerHTML = ids.map((id) => `<option value="${id}">${htmlEscape(engines.get(id)?.name ?? id)}</option>`).join('');
  if (ids.includes(prev)) select.value = prev;
}

function selectedOpenings(): ArenaOpening[] {
  const mode = selectEl('startingPositionSelect').value;
  if (mode === 'built-in') return BUILTIN_ARENA_OPENINGS;
  if (mode === 'custom') {
    const parsed = parseArenaOpenings((el('openingText') as HTMLTextAreaElement).value);
    if (!parsed.length) throw new Error('Add at least one custom FEN, or choose Start position.');
    return parsed;
  }
  return [{ name: 'Start position', fen: START_FEN }];
}

function openingHistoryBoards(opening: ArenaOpening): BoardState[] {
  if (opening.positions?.length) return [...opening.positions];
  return [parseFen(opening.fen)];
}

function gameTreeFromOpening(opening: ArenaOpening): GameTree {
  const tree = new GameTree(opening.startFen ?? opening.fen);
  for (const uci of opening.moves ?? []) {
    if (!tree.addUci(uci)) throw new Error(`Opening ${opening.name} contains illegal move ${uci}`);
  }
  return tree;
}

function openingPgnSetupTags(opening: ArenaOpening): Record<string, string> {
  if (opening.moves?.length) {
    const startFen = opening.startFen ?? START_FEN;
    return startFen !== START_FEN ? { SetUp: '1', FEN: startFen } : {};
  }
  return opening.fen !== START_FEN ? { SetUp: '1', FEN: opening.fen } : {};
}

function setOpeningPreview(opening: ArenaOpening): void {
  historyBoards = openingHistoryBoards(opening);
  board = historyBoards[historyBoards.length - 1];
  lastUci = null;
  setBoardSideEngines(null, null);
  renderBoard();
}

function refreshOpeningPreview(): void {
  const select = selectEl('startingPositionSelect');
  const textarea = el('openingText') as HTMLTextAreaElement;
  const custom = select.value === 'custom';
  select.disabled = running;
  textarea.disabled = running || !custom;
  if (running) return;
  try {
    const openings = selectedOpenings();
    el('openingInfo').textContent = `${openings.length} starting position${openings.length === 1 ? '' : 's'} · each pair plays every selected position with the configured color schedule.`;
    setOpeningPreview(openings[0]);
  } catch (error) {
    el('openingInfo').textContent = `Opening setup error: ${(error as Error).message}`;
  }
}

function renderStandings(standings: Map<string, Standing>) {
  const ranked = rankedStandings(standings);
  el('standings').querySelector('tbody')!.innerHTML = ranked.map((s, i) =>
    `<tr class="${i === 0 && s.games > 0 ? 'leader' : ''}"><td>${i + 1}</td><td>${htmlEscape(s.name)}</td>`
    + `<td class="num">${s.score}</td><td class="num">${s.wins}</td><td class="num">${s.losses}</td><td class="num">${s.draws}</td><td class="num">${s.games}</td></tr>`).join('');
}

function appendLog(text: string) {
  const div = document.createElement('div');
  div.textContent = text;
  el('log').prepend(div);
}

function sleep(ms: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve) => {
    if (ms <= 0 || signal.aborted) return resolve();
    const timer = setTimeout(resolve, ms);
    signal.addEventListener('abort', () => { clearTimeout(timer); resolve(); }, { once: true });
  });
}

function isAbortError(error: unknown): boolean {
  return error instanceof Error && error.name === 'AbortError';
}

async function warmUpSelectedEngines(ids: string[], signal: AbortSignal): Promise<void> {
  const warmed = new Set<string>();
  for (const id of ids) {
    if (signal.aborted) return;
    const engine = engines.get(id);
    if (!engine?.warmup || warmed.has(id)) continue;
    el('message').textContent = `Warming up ${engine.name}…`;
    await engine.warmup(signal);
    warmed.add(id);
    renderCacheInfo();
  }
}

function legalFromUci(current: BoardState, uci: string | null): Move | undefined {
  return uci ? legalMoves(current).find((m) => moveToUci(m) === uci) : undefined;
}

async function playArenaGame(white: ArenaEngine, black: ArenaEngine, opening: ArenaOpening, signal: AbortSignal): Promise<{ result: GameResultCode; reason: string; tree: GameTree }> {
  pendingLc0ReplyProbes.clear();
  const tree = gameTreeFromOpening(opening);
  historyBoards = tree.historyBoards();
  board = historyBoards[historyBoards.length - 1];
  lastUci = null;
  setBoardSideEngines(white.name, black.name);
  renderBoard();
  const priorFens: string[] = historyBoards.slice(0, -1).map(boardToFen);
  const delay = Math.max(0, Math.floor(Number(inputEl('delayInput').value) || 0));
  for (let ply = 0; ply < 300; ply++) {
    if (signal.aborted) return { result: '1/2-1/2', reason: 'cancelled', tree };
    const outcome = gameOutcome(board, priorFens);
    if (outcome) return { ...outcome, tree };
    const engine = board.turn === 'w' ? white : black;
    let uci: string | null;
    try {
      uci = await engine.move(historyBoards, signal);
    } catch (error) {
      if (isAbortError(error)) return { result: '1/2-1/2', reason: 'cancelled', tree };
      throw error;
    }
    if (signal.aborted) return { result: '1/2-1/2', reason: 'cancelled', tree };
    const move = legalFromUci(board, uci);
    if (!move) return { result: board.turn === 'w' ? '0-1' : '1-0', reason: uci ? `illegal ${uci}` : 'resigned', tree };
    recordPendingLc0ReplyProbes(engine, board, move);
    priorFens.push(boardToFen(board));
    board = makeMove(board, move);
    noteLc0MoveForReplyProbe(engine, move, board);
    historyBoards.push(board);
    lastUci = moveToUci(move);
    tree.addUci(lastUci);
    renderBoard();
    await sleep(delay, signal);
  }
  return { result: '1/2-1/2', reason: 'max plies', tree };
}

async function startTournament() {
  if (running) return;
  const ids = selectedEngineIds();
  if (ids.length < 2) { el('message').textContent = 'Select at least two engines.'; return; }
  const participants = ids.map((id) => ({ id, name: engines.get(id)!.name }));
  const gamesPerPair = Math.max(1, Math.floor(Number(inputEl('gamesInput').value) || 2));
  const format = selectEl('formatSelect').value;
  let basePairings: ArenaPairing[];
  if (format === 'gauntlet') {
    const champion = selectEl('championSelect').value || ids[0];
    basePairings = gauntletPairings([champion], ids.filter((id) => id !== champion), gamesPerPair);
  } else {
    basePairings = roundRobinPairings(ids, gamesPerPair);
  }
  let pairings: ScheduledArenaGame[];
  try {
    pairings = scheduleOpenings(basePairings, selectedOpenings());
  } catch (error) {
    el('message').textContent = `Opening setup error: ${(error as Error).message}`;
    return;
  }
  if (!pairings.length) { el('message').textContent = 'No pairings to play.'; return; }

  running = true;
  abort = new AbortController();
  refreshOpeningPreview();
  refreshStockfishFlavorAvailability();
  games.length = 0;
  lastLc0SearchResults.clear();
  pendingLc0ReplyProbes.clear();
  lc0TreeTelemetry.clear();
  el('log').innerHTML = '';
  renderSearchTelemetryInfo();
  el('start').toggleAttribute('disabled', true);
  el('stop').toggleAttribute('disabled', false);
  const standings = initStandings(participants);
  renderStandings(standings);
  let played = 0;
  try {
    resetLc0SearchTrees(ids);
    await warmUpSelectedEngines(ids, abort.signal);
    if (abort.signal.aborted) return;
    for (let i = 0; i < pairings.length; i++) {
      if (abort.signal.aborted) break;
      const { white, black, opening } = pairings[i];
      const whiteEngine = engines.get(white)!;
      const blackEngine = engines.get(black)!;
      resetLc0SearchTrees([white, black]);
      setBoardSideEngines(whiteEngine.name, blackEngine.name);
      el('pairing').textContent = `Game ${i + 1}/${pairings.length}: ${whiteEngine.name} (W) vs ${blackEngine.name} (B) · ${opening.name}`;
      el('message').textContent = 'Playing…';
      const { result, reason, tree } = await playArenaGame(whiteEngine, blackEngine, opening, abort.signal);
      if (reason === 'cancelled') break;
      applyGameResult(standings, white, black, result);
      played += 1;
      const tags: Record<string, string> = { Event: 'LC0 arena', White: whiteEngine.name, Black: blackEngine.name, Opening: opening.name, ...openingPgnSetupTags(opening) };
      games.push({ pgn: gameTreeToPgn(tree, tags, result) });
      renderCacheInfo();
      appendLog(`${i + 1}. ${whiteEngine.name} vs ${blackEngine.name} [${opening.name}]: ${result} (${reason}) · ${cacheMetricsText(lc0Cache?.metrics())} · ${searchTelemetryText()}`);
      renderStandings(standings);
    }
    const leader = rankedStandings(standings)[0];
    el('message').textContent = abort.signal.aborted
      ? `Stopped after ${played} game(s). Leader: ${leader?.name ?? '—'}.`
      : `Tournament done (${played} games). Winner: ${leader?.name ?? '—'} with ${leader?.score ?? 0}.`;
    el('pairing').textContent = 'Tournament finished.';
  } catch (error) {
    if (isAbortError(error) || abort?.signal.aborted) {
      const leader = rankedStandings(standings)[0];
      el('message').textContent = `Stopped after ${played} game(s). Leader: ${leader?.name ?? '—'}.`;
      el('pairing').textContent = 'Tournament stopped.';
    } else {
      el('message').textContent = `Tournament failed: ${(error as Error).message}`;
    }
  } finally {
    running = false;
    abort = null;
    el('start').toggleAttribute('disabled', false);
    el('stop').toggleAttribute('disabled', true);
    refreshOpeningPreview();
    refreshStockfishFlavorAvailability();
  }
}

function exportPgn() {
  inputEl('pgnOut').value = games.map((g) => g.pgn).join('\n\n');
  el('message').textContent = games.length ? `Exported ${games.length} game(s) as PGN.` : 'No games to export yet.';
}

function disposeRuntimeResources(): void {
  abort?.abort();
  abort = null;
  void lc0Cache?.dispose();
  lc0Cache = null;
  player = null;
  searcher = null;
  lc0Searchers.clear();
  lastLc0SearchResults.clear();
  pendingLc0ReplyProbes.clear();
  stockfish?.dispose();
  stockfish = null;
}

function wireEvents() {
  el('start').addEventListener('click', () => { void startTournament(); });
  el('stop').addEventListener('click', () => { abort?.abort(); el('message').textContent = 'Stopping…'; });
  el('exportPgn').addEventListener('click', exportPgn);
  el('engines').addEventListener('change', refreshChampionOptions);
  el('startingPositionSelect').addEventListener('change', refreshOpeningPreview);
  el('openingText').addEventListener('input', refreshOpeningPreview);
  el('cacheEntriesInput').addEventListener('input', () => { renderCacheInfo(); resetLc0SearchTrees(); });
  el('budgetModeSelect').addEventListener('change', () => resetLc0SearchTrees());
  el('movetimeInput').addEventListener('input', () => resetLc0SearchTrees());
  el('stockfishFlavorSelect').addEventListener('change', () => {
    if (running) return;
    refreshStockfishFlavorAvailability();
    stockfish?.dispose();
    stockfish = new StockfishEngine({ depth: 4, threads: stockfishThreads() }, stockfishFlavorUrl(selectedStockfishFlavor()));
    buildEngines();
    refreshChampionOptions();
  });
  el('stockfishThreadsInput').addEventListener('input', () => {
    if (running) return;
    inputEl('stockfishThreadsInput').value = String(stockfishThreads());
    stockfish?.setOptions({ threads: stockfishThreads() });
  });
  window.addEventListener('pagehide', (event) => {
    if (!(event as PageTransitionEvent).persisted) disposeRuntimeResources();
  });
}

async function init() {
  renderBoard();
  selectEl('stockfishFlavorSelect').value = REQUESTED_STOCKFISH_FLAVOR;
  inputEl('stockfishThreadsInput').value = String(Math.max(1, Math.min(32, Math.floor(Number(params.get('sfThreads') ?? '1') || 1))));
  refreshStockfishFlavorAvailability();
  void renderRuntimeBadge();
  buildEngines();
  refreshChampionOptions();
  wireEvents();
  refreshOpeningPreview();
  try {
    const modelLoad = await loadLc0ModelForOrt(MODEL_URL, { cache: false });
    const evaluator = await Lc0OnnxEvaluator.create(modelLoad.model);
    lc0Cache = new CachedLc0Evaluator(evaluator, { maxEntries: arenaCacheEntries() });
    lc0Searchers.clear();
    player = new Lc0PolicyOnlyPlayer(lc0Cache);
    searcher = new Lc0PuctSearcher(lc0Cache);
    stockfish = new StockfishEngine({ depth: 4, threads: stockfishThreads() }, stockfishFlavorUrl(selectedStockfishFlavor()));
    renderCacheInfo();
    void renderRuntimeBadge();
    el('start').toggleAttribute('disabled', false);
    el('message').textContent = 'Ready. Pick engines and start a tournament.';
  } catch (error) {
    el('message').textContent = `Model load failed: ${(error as Error).message}`;
  }
}

void init();
