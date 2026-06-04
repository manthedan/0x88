import { Chessground } from 'chessground';
import type { DrawShape } from 'chessground/draw';
import type { Key } from 'chessground/types';
import { boardToFen, parseFen, squareName, START_FEN, type BoardState } from '../chess/board.ts';
import { legalMoves, makeMove } from '../chess/movegen.ts';
import { moveToUci, type Move } from '../chess/moveCodec.ts';
import { gameTreeToPgn, parsePgnGame, parsePgnGames } from '../chess/pgn.ts';
import { collectOrtRuntimeDiagnostics } from '../nn/ortRuntime.ts';
import { engineBrushes, evalBarWhitePercent, lc0AnalysisLines, stockfishAnalysisLines, type AnalysisLine } from './analysisFormat.ts';
import { GameTree, type GameNode } from './gameTree.ts';
import { fetchGameHistoryPgn, type ImportColor, type ImportSite } from './gameImport.ts';
import { openingStatsForPosition, openingSummary, type ImportedGame, type OpeningMoveStat } from './openingStats.ts';
import { loadLc0ModelForOrt } from './modelCache.ts';
import { Lc0OnnxEvaluator } from './onnxEvaluator.ts';
import { Lc0PuctSearcher } from './search.ts';
import { StockfishEngine } from './stockfishEngine.ts';
import { DEFAULT_RECKLESS_WASM_URL, RecklessEngine } from './recklessEngine.ts';

type Ground = ReturnType<typeof Chessground>;

const params = new URLSearchParams(location.search);
const MODEL_URL = params.get('model') ?? '/models/lc0/t1-256x10-distilled-swa-2432500.batch1.f32.onnx';
const RECKLESS_WASM_URL = params.get('recklessWasm') ?? DEFAULT_RECKLESS_WASM_URL;

let tree = new GameTree(params.get('fen') ?? START_FEN);
let searcher: Lc0PuctSearcher | null = null;
let mainEvaluator: Lc0OnnxEvaluator | null = null;
let stockfish: StockfishEngine | null = null;
let reckless: RecklessEngine | null = null;
let ground: Ground | null = null;
let orientation: 'white' | 'black' = 'white';
let analysisAbort: AbortController | null = null;
let analyzing = false;
const lineCache = new Map<string, AnalysisLine[]>();
const nodeIndex = new Map<number, GameNode>();
let importedGames: ImportedGame[] = [];
const bookCache = new Map<string, OpeningMoveStat[]>();
// Distinct brush for the opening-book most-played move (not LC0 green / SF blue).
const BOOK_BRUSH = 'yellow';
const BOOK_SWATCH = '#e68f00';

function currentBookStats(): OpeningMoveStat[] {
  if (!importedGames.length) return [];
  const fen = tree.current.fen;
  let stats = bookCache.get(fen);
  if (!stats) { stats = openingStatsForPosition(importedGames, fen); bookCache.set(fen, stats); }
  return stats;
}

// LC0 analysis runs in a dedicated search worker so navigation never blocks the
// UI; a new position cancels the in-flight worker search by id.
let searchWorker: Worker | null = null;
let workerReady = false;
let workerBackend = '';
let workerSeq = 0;
const workerPending = new Map<number, { resolve: (value: unknown) => void; reject: (error: Error) => void }>();
let activeWorkerSearchId: number | null = null;

interface WorkerSearchResult {
  value: number;
  visits: number;
  pv: string[];
  multiPv?: string[][];
  children: { uci: string; visits: number; q: number }[];
  cancelled?: boolean;
}

function requestedEp(): string {
  const raw = (params.get('ep') ?? 'auto').toLowerCase();
  if (raw === 'webgpu' || raw === 'gpu') return 'webgpu';
  if (raw === 'wasm') return 'wasm';
  if (raw === 'webgpu,wasm' || raw === 'gpu,wasm') return 'webgpu,wasm';
  return 'auto';
}

function postWorker<T>(message: Record<string, unknown>, onId?: (id: number) => void): Promise<T> {
  if (!searchWorker) return Promise.reject(new Error('LC0 worker unavailable'));
  const id = ++workerSeq;
  onId?.(id);
  return new Promise<T>((resolve, reject) => {
    workerPending.set(id, { resolve: resolve as (value: unknown) => void, reject });
    searchWorker!.postMessage({ ...message, id });
  });
}

async function initWorker(): Promise<string> {
  if (searchWorker && workerReady) return workerBackend;
  if (!searchWorker) searchWorker = new Worker(new URL('./searchWorker.ts', import.meta.url), { type: 'module' });
  searchWorker.addEventListener('message', (event: MessageEvent) => {
    const message = event.data as { id: number; type: string; error?: string };
    const pending = workerPending.get(message.id);
    if (!pending) return;
    workerPending.delete(message.id);
    if (message.type === 'error') pending.reject(new Error(message.error ?? 'worker error'));
    else pending.resolve(message);
  });
  searchWorker.addEventListener('error', (event) => {
    for (const pending of workerPending.values()) pending.reject(new Error(event.message || 'LC0 worker error'));
    workerPending.clear();
  });
  const ready = await postWorker<{ backend: string }>({ type: 'init', modelUrl: MODEL_URL, ep: requestedEp(), cacheModel: false });
  workerReady = true;
  workerBackend = ready.backend;
  return workerBackend;
}

async function workerLc0Lines(fen: string): Promise<AnalysisLine[]> {
  const response = await postWorker<{ result: WorkerSearchResult }>(
    { type: 'search', input: { positions: tree.historyBoards() }, visits: visits(), batchSize: 1, multiPv: multiPv() },
    (id) => { activeWorkerSearchId = id; },
  );
  return response.result.cancelled ? [] : lc0AnalysisLines(response.result, fen, 'LC0');
}

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
function setShapes(shapes: DrawShape[]) { ground?.setAutoShapes(shapes); }
function uciShape(uci: string, brush: string): DrawShape | null {
  return uci.length >= 4 ? { orig: uci.slice(0, 2) as Key, dest: uci.slice(2, 4) as Key, brush } : null;
}
function visits(): number { return Math.max(1, Math.floor(Number(inputEl('visitsInput').value) || 400)); }
function multiPv(): number { return Math.max(1, Math.floor(Number(inputEl('multiPvInput').value) || 3)); }
function sfDepth(): number { return Math.max(1, Math.floor(Number(inputEl('sfDepthInput').value) || 14)); }
function recklessDepth(): number { return Math.max(1, Math.floor(Number(inputEl('recklessDepthInput').value) || 4)); }
function useLc0(): boolean { return inputEl('useLc0').checked; }
function useStockfish(): boolean { return inputEl('useStockfish').checked; }
function useReckless(): boolean { return inputEl('useReckless').checked; }

function getStockfish(): StockfishEngine {
  if (!stockfish) stockfish = new StockfishEngine({ depth: sfDepth() });
  return stockfish;
}

function getReckless(): RecklessEngine {
  if (!reckless) reckless = new RecklessEngine({ depth: recklessDepth(), hashMb: 16 }, RECKLESS_WASM_URL);
  return reckless;
}

function legalDests(board: BoardState) {
  const dests = new Map<Key, Key[]>();
  for (const move of legalMoves(board)) {
    const from = squareName(move.from) as Key;
    dests.set(from, [...(dests.get(from) ?? []), squareName(move.to) as Key]);
  }
  return dests;
}
function legalMoveFromDrag(board: BoardState, from: Key, to: Key): Move | undefined {
  const base = `${from}${to}`;
  const all = legalMoves(board);
  return all.find((m) => moveToUci(m) === base)
    ?? all.find((m) => moveToUci(m) === `${base}q`)
    ?? all.find((m) => moveToUci(m) === `${base}r`)
    ?? all.find((m) => moveToUci(m) === `${base}b`)
    ?? all.find((m) => moveToUci(m) === `${base}n`);
}

// Board arrows, colored by source and de-duplicated by move so the board stays
// readable: each engine's best move (solid engine color) and the opening book's
// most-played move (yellow) take priority over engines' alternative MultiPV
// moves (pale). When two sources agree on a move, the higher-priority arrow wins.
function bestShapes(): DrawShape[] {
  const lines = lineCache.get(tree.current.fen) ?? [];
  const candidates: { uci: string; brush: string; prio: number }[] = [];
  for (const line of lines) {
    const uci = line.pvUci[0];
    if (!uci || uci.length < 4) continue;
    const brushes = engineBrushes(line.engine);
    candidates.push({ uci, brush: line.multipv === 1 ? brushes.primary : brushes.alt, prio: line.multipv === 1 ? 0 : 2 });
  }
  const book = currentBookStats()[0];
  if (book && book.uci.length >= 4) candidates.push({ uci: book.uci, brush: BOOK_BRUSH, prio: 1 });
  candidates.sort((a, b) => a.prio - b.prio);
  const shapes: DrawShape[] = [];
  const seen = new Set<string>();
  for (const candidate of candidates) {
    const key = candidate.uci.slice(0, 4);
    if (seen.has(key)) continue;
    seen.add(key);
    const shape = uciShape(candidate.uci, candidate.brush);
    if (shape) shapes.push(shape);
  }
  return shapes;
}

function renderBoard() {
  const board = tree.current.fen ? parseFen(tree.current.fen) : parseFen(START_FEN);
  const lastUci = tree.current.move ? moveToUci(tree.current.move) : null;
  const config = {
    orientation,
    fen: tree.current.fen.split(' ')[0],
    turnColor: board.turn === 'w' ? 'white' as const : 'black' as const,
    coordinates: true,
    highlight: { lastMove: true, check: true },
    animation: { enabled: true, duration: 160 },
    movable: {
      free: false,
      color: board.turn === 'w' ? 'white' as const : 'black' as const,
      dests: legalDests(board),
      events: { after: onUserMove },
    },
    lastMove: lastUci ? [lastUci.slice(0, 2) as Key, lastUci.slice(2, 4) as Key] : undefined,
  };
  if (!ground) ground = Chessground(el('ground'), config);
  else ground.set(config);
  el('sideToMove').textContent = board.turn === 'w' ? 'White to move' : 'Black to move';
  renderEvalBar();
  setShapes(bestShapes());
}

function renderEvalBar() {
  const line = (lineCache.get(tree.current.fen) ?? [])[0];
  const board = parseFen(tree.current.fen);
  const pct = line ? evalBarWhitePercent(line.scoreCp, line.mateIn, board.turn) : 50;
  (el('evalWhite') as HTMLElement).style.height = `${pct}%`;
  el('posEval').textContent = line ? `${line.scoreText} (${line.engine})` : '—';
}

function renderLegend(lines: AnalysisLine[]) {
  const keys = [...new Set(lines.map((line) => line.engine))].map((engine) => ({ label: engine, swatch: engineBrushes(engine).swatch }));
  if (currentBookStats().length) keys.push({ label: 'Book (most played)', swatch: BOOK_SWATCH });
  el('engineLegend').innerHTML = keys.map((key) =>
    `<span class="key"><span class="dot" style="background:${key.swatch}"></span>${htmlEscape(key.label)}</span>`).join('');
}

function renderLines() {
  const lines = lineCache.get(tree.current.fen) ?? [];
  renderLegend(lines);
  el('lines').innerHTML = lines.map((line) => {
    const cls = line.scoreCp === undefined ? '' : line.scoreCp > 0 ? 'pos' : line.scoreCp < 0 ? 'neg' : '';
    const swatch = engineBrushes(line.engine).swatch;
    return `<li data-uci="${htmlEscape(line.pvUci[0] ?? '')}" data-pv="${htmlEscape(line.pvUci.join(' '))}" data-engine="${htmlEscape(line.engine)}" style="border-left:3px solid ${swatch}">`
      + `<span class="score ${cls}">${htmlEscape(line.scoreText)}<br><span class="eng">${htmlEscape(line.engine)} · ${htmlEscape(line.detail)}</span></span>`
      + `<span class="pv">${htmlEscape(line.pvSan)}</span></li>`;
  }).join('') || '<li class="small">no analysis yet</li>';
}

function moveNumberPrefix(node: GameNode, force: boolean): string {
  const parent = parseFen(node.parent!.fen);
  if (parent.turn === 'w') return `${parent.fullmove}. `;
  return force ? `${parent.fullmove}… ` : '';
}

function renderVariation(node: GameNode): string {
  let html = '';
  let cursor: GameNode | undefined = node;
  let force = true;
  while (cursor) {
    html += moveToken(cursor, force);
    force = false;
    cursor = cursor.children[0];
  }
  return html;
}

function moveToken(node: GameNode, force: boolean): string {
  const current = node === tree.current ? ' current' : '';
  return `<span class="mv${current}" data-node="${node.id}">${moveNumberPrefix(node, force)}${htmlEscape(node.san ?? '')}</span> `;
}

function renderMoveList() {
  nodeIndex.clear();
  const collect = (n: GameNode) => { nodeIndex.set(n.id, n); n.children.forEach(collect); };
  collect(tree.root);
  let html = '';
  let node: GameNode | undefined = tree.root.children[0];
  let force = true;
  while (node) {
    html += moveToken(node, force);
    force = false;
    const variations = node.parent!.children.slice(1);
    for (const variation of variations) {
      html += `<span class="var">(${renderVariation(variation)})</span> `;
      force = true;
    }
    node = node.children[0];
  }
  el('movelist').innerHTML = html || '<span class="small">no moves — drag a piece or load a PGN</span>';
}

function renderOpening() {
  const body = el('opening').querySelector('tbody')!;
  if (!importedGames.length) { body.innerHTML = '<tr><td colspan="3" class="small">import games to see opening stats</td></tr>'; return; }
  const stats = currentBookStats();
  const summary = openingSummary(stats);
  el('importInfo').textContent = `${importedGames.length} games · ${summary.total} from here`;
  if (!stats.length) { body.innerHTML = '<tr><td colspan="3" class="small">no games reached this position</td></tr>'; return; }
  body.innerHTML = stats.map((stat) => {
    const pct = (n: number) => (stat.count ? (n / stat.count) * 100 : 0).toFixed(0);
    return `<tr class="mv" data-uci="${htmlEscape(stat.uci)}"><td class="san">${htmlEscape(stat.san)}</td>`
      + `<td class="num">${stat.count}</td>`
      + `<td><div class="wdlbar" title="W ${stat.whiteWins} / D ${stat.draws} / B ${stat.blackWins}">`
      + `<div class="w" style="width:${pct(stat.whiteWins)}%"></div><div class="d" style="width:${pct(stat.draws)}%"></div><div class="b" style="width:${pct(stat.blackWins)}%"></div></div></td></tr>`;
  }).join('');
}

function importGames() {
  const raw = inputEl('importGamesInput').value.trim();
  if (!raw) { el('importInfo').textContent = 'paste or fetch PGN first'; return; }
  try {
    importedGames = parsePgnGames(raw).map((game) => ({ tree: game.tree, result: game.result }));
    bookCache.clear();
    el('importInfo').textContent = `imported ${importedGames.length} games`;
    renderOpening();
    renderLines(); // refresh the legend so the Book key appears
    setShapes(bestShapes());
  } catch (error) {
    el('importInfo').textContent = `import failed: ${(error as Error).message}`;
  }
}

async function fetchGames() {
  const site = selectEl('importSite').value as ImportSite;
  const username = inputEl('importUser').value.trim();
  if (!username) { el('importInfo').textContent = 'enter a username'; return; }
  const opts = { max: Number(inputEl('importMax').value) || 40, color: selectEl('importColor').value as ImportColor };
  el('fetchGames').toggleAttribute('disabled', true);
  el('importInfo').textContent = `fetching ${username}'s games from ${site}…`;
  try {
    const pgn = await fetchGameHistoryPgn(site, username, opts, fetch);
    inputEl('importGamesInput').value = pgn;
    if (!pgn.trim()) { el('importInfo').textContent = 'no games found'; return; }
    importGames();
  } catch (error) {
    // A network/CORS failure surfaces as a TypeError with no status.
    const message = (error as Error).message || 'fetch failed';
    el('importInfo').textContent = `fetch failed: ${message}`;
  } finally {
    el('fetchGames').toggleAttribute('disabled', false);
  }
}

function downloadPgn() {
  const pgn = inputEl('importGamesInput').value;
  if (!pgn.trim()) { el('importInfo').textContent = 'nothing to download'; return; }
  const name = (inputEl('importUser').value.trim() || 'games').replace(/[^\w.-]+/g, '_');
  const blob = new Blob([pgn], { type: 'application/x-chess-pgn' });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = `${name}.pgn`;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
  el('importInfo').textContent = `downloaded ${name}.pgn`;
}

function renderAll() {
  renderBoard();
  renderLines();
  renderMoveList();
  renderOpening();
}

async function analyzeCurrent() {
  if (!useLc0() && !useStockfish() && !useReckless()) { el('message').textContent = 'Enable LC0, Stockfish, or Reckless to analyze.'; return; }
  // Interrupt any in-flight analysis: abort the Stockfish signal and cancel the
  // worker's LC0 search by id, so a new position takes over immediately.
  analysisAbort?.abort();
  if (activeWorkerSearchId !== null && searchWorker) searchWorker.postMessage({ type: 'cancel', target: activeWorkerSearchId });
  const controller = new AbortController();
  analysisAbort = controller;
  analyzing = true;
  el('stop').toggleAttribute('disabled', false);
  el('analyze').toggleAttribute('disabled', true);
  const fen = tree.current.fen;
  const board = parseFen(fen);
  if (legalMoves(board).length === 0) { analyzing = false; el('stop').toggleAttribute('disabled', true); el('analyze').toggleAttribute('disabled', false); return; }
  const selectedLabels = [useLc0() ? `LC0 ${visits()}v` : '', useStockfish() ? `SF d${sfDepth()}` : '', useReckless() ? `Reckless d${recklessDepth()}` : ''].filter(Boolean).join(' + ');
  el('message').textContent = `Analyzing (${selectedLabels}, ${multiPv()} lines)…`;
  try {
    const tasks: Promise<AnalysisLine[]>[] = [];
    if (useLc0()) {
      if (workerReady) tasks.push(workerLc0Lines(fen));
      else if (searcher) tasks.push(searcher.search({ positions: tree.historyBoards() }, { visits: visits(), multiPv: multiPv(), signal: controller.signal, yieldEveryMs: 16 })
        .then((result) => lc0AnalysisLines(result, fen, 'LC0')));
    }
    if (useStockfish()) {
      tasks.push(getStockfish().analyze(fen, { multipv: multiPv(), depth: sfDepth(), signal: controller.signal })
        .then((infos) => stockfishAnalysisLines(infos, fen, `SF d${sfDepth()}`)));
    }
    if (useReckless()) {
      tasks.push(getReckless().analyze(fen, { multipv: multiPv(), depth: recklessDepth(), signal: controller.signal })
        .then((infos) => stockfishAnalysisLines(infos, fen, `Reckless d${recklessDepth()}`)));
    }
    const grouped = await Promise.all(tasks);
    if (controller.signal.aborted) return;
    lineCache.set(fen, grouped.flat());
    if (tree.current.fen === fen) { renderLines(); renderEvalBar(); setShapes(bestShapes()); }
    el('message').textContent = `Analyzed: ${(lineCache.get(fen) ?? [])[0]?.scoreText ?? '—'}`;
  } catch (error) {
    if ((error as Error).name !== 'AbortError') el('message').textContent = `Analysis failed: ${(error as Error).message}`;
  } finally {
    if (analysisAbort === controller) {
      analyzing = false;
      analysisAbort = null;
      el('stop').toggleAttribute('disabled', true);
      el('analyze').toggleAttribute('disabled', false);
    }
  }
}

function afterNavigation() {
  renderAll();
  if (inputEl('autoAnalyze').checked && !lineCache.has(tree.current.fen)) void analyzeCurrent();
  else { renderEvalBar(); setShapes(bestShapes()); }
}

async function onUserMove(from: Key, to: Key) {
  const board = tree.current.fen ? parseFen(tree.current.fen) : parseFen(START_FEN);
  const move = legalMoveFromDrag(board, from, to);
  if (!move) { renderBoard(); return; }
  tree.addMove(move);
  afterNavigation();
}

function loadFen() {
  const raw = inputEl('fenInput').value.trim();
  if (!raw) return;
  try {
    parseFen(raw);
  } catch (error) {
    el('message').textContent = `Invalid FEN: ${(error as Error).message}`;
    return;
  }
  tree = new GameTree(raw);
  lineCache.clear();
  el('message').textContent = 'Loaded FEN.';
  afterNavigation();
}

function loadPgn() {
  const raw = inputEl('pgnInput').value.trim();
  if (!raw) return;
  try {
    const { tree: parsed, tags } = parsePgnGame(raw);
    tree = parsed;
    tree.toStart();
    lineCache.clear();
    el('message').textContent = `Loaded PGN${tags.White ? `: ${tags.White} – ${tags.Black}` : ''}.`;
    afterNavigation();
  } catch (error) {
    el('message').textContent = `PGN parse failed: ${(error as Error).message}`;
  }
}

function copyPgn() {
  const pgn = gameTreeToPgn(tree, {}, tree.root.children.length ? '*' : '*');
  inputEl('pgnInput').value = pgn;
  void navigator.clipboard?.writeText(pgn).catch(() => undefined);
  el('message').textContent = 'PGN copied to the box and clipboard.';
}

function hoverLine(pvUci: string[], engine: string) {
  // Only the line's first move: the rest of the PV is from future positions, so
  // drawing every ply on the current board is misleading clutter.
  const shape = pvUci[0] ? uciShape(pvUci[0], engineBrushes(engine).primary) : null;
  setShapes(shape ? [shape] : bestShapes());
}

function wireEvents() {
  el('navStart').addEventListener('click', () => { tree.toStart(); afterNavigation(); });
  el('navBack').addEventListener('click', () => { tree.back(); afterNavigation(); });
  el('navForward').addEventListener('click', () => { tree.forward(); afterNavigation(); });
  el('navEnd').addEventListener('click', () => { tree.toEnd(); afterNavigation(); });
  el('flip').addEventListener('click', () => { orientation = orientation === 'white' ? 'black' : 'white'; renderBoard(); });
  el('loadFen').addEventListener('click', loadFen);
  el('reset').addEventListener('click', () => { tree = new GameTree(); lineCache.clear(); el('message').textContent = 'Reset.'; afterNavigation(); });
  el('loadPgn').addEventListener('click', loadPgn);
  el('copyPgn').addEventListener('click', copyPgn);
  el('analyze').addEventListener('click', () => { void analyzeCurrent(); });
  el('stop').addEventListener('click', () => {
    analysisAbort?.abort();
    if (activeWorkerSearchId !== null && searchWorker) searchWorker.postMessage({ type: 'cancel', target: activeWorkerSearchId });
  });
  for (const id of ['useLc0', 'useStockfish', 'useReckless', 'sfDepthInput', 'recklessDepthInput', 'visitsInput', 'multiPvInput']) {
    el(id).addEventListener('change', () => { lineCache.delete(tree.current.fen); void analyzeCurrent(); });
  }
  el('movelist').addEventListener('click', (event) => {
    const target = (event.target as HTMLElement).closest('[data-node]');
    if (!target) return;
    const node = nodeIndex.get(Number(target.getAttribute('data-node')));
    if (node) { tree.goTo(node); afterNavigation(); }
  });
  el('lines').addEventListener('click', (event) => {
    const li = (event.target as HTMLElement).closest('li[data-uci]');
    const uci = li?.getAttribute('data-uci');
    if (uci) { tree.addUci(uci); afterNavigation(); }
  });
  el('lines').addEventListener('mouseover', (event) => {
    const li = (event.target as HTMLElement).closest('li[data-pv]');
    const pv = li?.getAttribute('data-pv');
    if (pv) hoverLine(pv.split(' ').filter(Boolean), li!.getAttribute('data-engine') ?? 'LC0');
  });
  el('lines').addEventListener('mouseout', () => setShapes(bestShapes()));
  el('importGames').addEventListener('click', importGames);
  el('fetchGames').addEventListener('click', () => { void fetchGames(); });
  el('downloadPgn').addEventListener('click', downloadPgn);
  inputEl('importUser').addEventListener('keydown', (event) => { if ((event as KeyboardEvent).key === 'Enter') void fetchGames(); });
  el('opening').addEventListener('click', (event) => {
    const row = (event.target as HTMLElement).closest('tr[data-uci]');
    const uci = row?.getAttribute('data-uci');
    if (uci && tree.addUci(uci)) afterNavigation();
  });
  document.addEventListener('keydown', (event) => {
    if (event.target instanceof HTMLInputElement || event.target instanceof HTMLTextAreaElement) return;
    if (event.key === 'ArrowLeft') { tree.back(); afterNavigation(); }
    else if (event.key === 'ArrowRight') { tree.forward(); afterNavigation(); }
    else if (event.key === 'ArrowUp') { tree.toStart(); afterNavigation(); }
    else if (event.key === 'ArrowDown') { tree.toEnd(); afterNavigation(); }
    else return;
    event.preventDefault();
  });
}

function disposeRuntimeResources(): void {
  analysisAbort?.abort();
  if (activeWorkerSearchId !== null) searchWorker?.postMessage({ type: 'cancel', target: activeWorkerSearchId });
  activeWorkerSearchId = null;
  searchWorker?.terminate();
  searchWorker = null;
  workerReady = false;
  workerBackend = '';
  for (const pending of workerPending.values()) pending.reject(new Error('LC0 worker disposed'));
  workerPending.clear();
  stockfish?.dispose();
  stockfish = null;
  reckless?.dispose();
  reckless = null;
  void mainEvaluator?.dispose();
  mainEvaluator = null;
  searcher = null;
}

async function init() {
  window.addEventListener('pagehide', (event) => {
    if (!(event as PageTransitionEvent).persisted) disposeRuntimeResources();
  });
  renderAll();
  wireEvents();
  el('message').textContent = 'Loading LC0 model in a worker…';
  try {
    el('backend').textContent = await initWorker();
    el('analyze').toggleAttribute('disabled', false);
    el('message').textContent = 'Ready. Drag a move, load a PGN/FEN, or Analyze. Navigation stays responsive.';
    if (inputEl('autoAnalyze').checked) void analyzeCurrent();
  } catch (workerError) {
    // Fall back to a main-thread evaluator (analysis will block the UI, but works).
    console.warn('LC0 worker init failed; falling back to the main thread.', workerError);
    try {
      const modelLoad = await loadLc0ModelForOrt(MODEL_URL, { cache: false });
      mainEvaluator = await Lc0OnnxEvaluator.create(modelLoad.model);
      searcher = new Lc0PuctSearcher(mainEvaluator);
      const diagnostics = await collectOrtRuntimeDiagnostics();
      el('backend').textContent = `${diagnostics.describe} (main thread)`;
      el('analyze').toggleAttribute('disabled', false);
      el('message').textContent = 'Ready (main-thread fallback — deep analysis may pause the UI).';
      if (inputEl('autoAnalyze').checked) void analyzeCurrent();
    } catch (error) {
      el('message').textContent = `Model load failed: ${(error as Error).message}`;
    }
  }
}

void init();
