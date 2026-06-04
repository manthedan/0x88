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
import { StockfishEngine, stockfishFlavorUrl } from './stockfishEngine.ts';
import { RecklessEngine } from './recklessEngine.ts';
import { RECKLESS_VARIANTS, checkRecklessVariantAsset, recklessVariantAssetStatus, recklessVariantByKey, recklessVariantFromParams, normalizeRecklessVariant, type RecklessVariant } from './recklessVariants.ts';
import { Bt4WorkerSearcher, bt4LoadWarning, bt4SupportedSync, probeBt4Support } from './bt4Engine.ts';

type Ground = ReturnType<typeof Chessground>;

const params = new URLSearchParams(location.search);
const MODEL_URL = params.get('model') ?? '/models/lc0/t1-256x10-distilled-swa-2432500.batch1.f32.onnx';
const REQUESTED_RECKLESS_VARIANT = recklessVariantFromParams(params);

let tree = new GameTree(params.get('fen') ?? START_FEN);
let searcher: Lc0PuctSearcher | null = null;
let mainEvaluator: Lc0OnnxEvaluator | null = null;
let stockfishLite: StockfishEngine | null = null;
let stockfishFull: StockfishEngine | null = null;
// Lc0 BT4 runs in its own worker (lazy, WebGPU-gated, disposable). See bt4Engine.ts.
const bt4 = new Bt4WorkerSearcher();
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

async function workerLc0Lines(fen: string, visits: number): Promise<AnalysisLine[]> {
  const response = await postWorker<{ result: WorkerSearchResult }>(
    { type: 'search', input: { positions: tree.historyBoards() }, visits, batchSize: 1, multiPv: multiPv() },
    (id) => { activeWorkerSearchId = id; },
  );
  return response.result.cancelled ? [] : lc0AnalysisLines(response.result, fen, 'Lc0');
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
function multiPv(): number { return Math.max(1, Math.floor(Number(inputEl('multiPvInput').value) || 3)); }
// Engines to analyze are chosen as an add/remove list of cascading selects:
// family (Lc0/Stockfish/Reckless) -> variant (Lc0: Small|BT4; SF: Lite|Full;
// Reckless: variant) -> strength (Lc0 visits, SF/Reckless depth), all per row.
type EngineFamily = 'lc0' | 'sf' | 'reckless';
interface EngineRow { family: EngineFamily; variant: string; strength: number; }
const DEFAULT_RECKLESS_VARIANT = REQUESTED_RECKLESS_VARIANT.key;

function strengthMeta(family: EngineFamily): { unit: string; min: number; max: number; def: number } {
  if (family === 'lc0') return { unit: 'visits', min: 1, max: 100000, def: 400 };
  if (family === 'sf') return { unit: 'depth', min: 1, max: 30, def: 14 };
  return { unit: 'depth', min: 1, max: 20, def: 4 };
}
function defaultStrength(family: EngineFamily): number { return strengthMeta(family).def; }

let engineRows: EngineRow[] = [{ family: 'lc0', variant: 'small', strength: 400 }];

function variantOptions(family: EngineFamily): { value: string; label: string; disabled?: boolean }[] {
  if (family === 'lc0') return [{ value: 'small', label: 'Small' }, { value: 'bt4', label: 'BT4', disabled: !bt4SupportedSync() }];
  if (family === 'sf') return [{ value: 'lite', label: 'Lite' }, { value: 'full', label: 'Full' }];
  return RECKLESS_VARIANTS.map((v) => ({ value: v.key, label: v.label }));
}

function defaultVariant(family: EngineFamily): string {
  return family === 'reckless' ? DEFAULT_RECKLESS_VARIANT : variantOptions(family)[0].value;
}

function rowLabel(row: EngineRow): string {
  if (row.family === 'lc0') return row.variant === 'bt4' ? 'Lc0 BT4' : 'Lc0';
  if (row.family === 'sf') return row.variant === 'lite' ? 'SF Lite' : 'SF';
  return recklessVariantByKey(normalizeRecklessVariant(row.variant)).label;
}

function activeEngineRows(): EngineRow[] {
  const seen = new Set<string>();
  return engineRows.filter((r) => { const k = `${r.family}:${r.variant}`; if (seen.has(k)) return false; seen.add(k); return true; });
}

function usesBt4Row(): boolean {
  return engineRows.some((r) => r.family === 'lc0' && r.variant === 'bt4');
}

function renderEngineList(): void {
  const families: [EngineFamily, string][] = [['lc0', 'Lc0'], ['sf', 'Stockfish'], ['reckless', 'Reckless']];
  el('engineList').innerHTML = engineRows.map((row, i) => {
    const famSel = families.map(([v, l]) => `<option value="${v}"${row.family === v ? ' selected' : ''}>${l}</option>`).join('');
    const varSel = variantOptions(row.family).map((o) => `<option value="${o.value}"${row.variant === o.value ? ' selected' : ''}${o.disabled ? ' disabled' : ''}>${htmlEscape(o.label)}</option>`).join('');
    const meta = strengthMeta(row.family);
    const remove = engineRows.length > 1 ? `<button class="row-rm" data-i="${i}" type="button" title="Remove engine">×</button>` : '';
    return `<div class="engine-row"><select class="row-fam" data-i="${i}">${famSel}</select><span class="arrow">→</span><select class="row-var" data-i="${i}">${varSel}</select><span class="arrow">→</span><input class="row-strength" data-i="${i}" type="number" min="${meta.min}" max="${meta.max}" step="1" value="${row.strength}" title="${meta.unit}"><span class="row-unit">${meta.unit}</span>${remove}</div>`;
  }).join('');
}

async function workerBt4Lines(fen: string, visits: number): Promise<AnalysisLine[]> {
  const result = await bt4.search({ positions: tree.historyBoards() }, { visits, multiPv: multiPv() });
  return result.cancelled ? [] : lc0AnalysisLines(result, fen, 'Lc0 BT4');
}

// Lc0 BT4 is WebGPU-only; its option is disabled in the list when WebGPU is unusable.
async function refreshBt4Availability(): Promise<void> {
  await probeBt4Support();
  if (!bt4SupportedSync()) {
    for (const row of engineRows) if (row.family === 'lc0' && row.variant === 'bt4') row.variant = 'small';
  }
  renderEngineList();
}

function renderRecklessRuntimeInfo(): void {
  const sab = typeof SharedArrayBuffer !== 'undefined' ? 'SAB yes' : 'SAB no';
  const mode = (typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated) ? 'persistent available' : 'one-shot fallback';
  el('recklessRuntimeInfo').textContent = `Reckless: ${mode} · ${sab}`;
}

function getStockfish(kind: 'lite' | 'full'): StockfishEngine {
  // Constructor depth is just a default; each analyze() call passes the row depth.
  if (kind === 'lite') {
    if (!stockfishLite) stockfishLite = new StockfishEngine({ depth: 14 }, stockfishFlavorUrl('lite-single'));
    return stockfishLite;
  }
  if (!stockfishFull) stockfishFull = new StockfishEngine({ depth: 14 }, stockfishFlavorUrl('single'));
  return stockfishFull;
}

const recklessByVariant = new Map<string, RecklessEngine>();
function getRecklessFor(variantKey: string): RecklessEngine {
  let engine = recklessByVariant.get(variantKey);
  if (!engine) {
    engine = new RecklessEngine({ depth: 4, hashMb: 16 }, recklessVariantByKey(normalizeRecklessVariant(variantKey)).wasmUrl);
    recklessByVariant.set(variantKey, engine);
  }
  return engine;
}

function disposeUnusedEngines(): void {
  if (!usesBt4Row()) bt4.dispose();
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
  const pct = line ? evalBarWhitePercent(line.scoreCp, line.mateIn) : 50;
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
  const rows = activeEngineRows();
  if (!rows.length) { el('message').textContent = 'Add an engine to analyze.'; return; }
  // Interrupt any in-flight analysis: abort the Stockfish signal and cancel the
  // worker LC0 / BT4 searches, so a new position takes over immediately.
  analysisAbort?.abort();
  if (activeWorkerSearchId !== null && searchWorker) searchWorker.postMessage({ type: 'cancel', target: activeWorkerSearchId });
  bt4.cancel();
  const controller = new AbortController();
  analysisAbort = controller;
  analyzing = true;
  el('stop').toggleAttribute('disabled', false);
  el('analyze').toggleAttribute('disabled', true);
  const fen = tree.current.fen;
  const board = parseFen(fen);
  if (legalMoves(board).length === 0) {
    analyzing = false;
    el('stop').toggleAttribute('disabled', true);
    el('analyze').toggleAttribute('disabled', false);
    return;
  }
  const selectedLabels = rows.map((row) => row.family === 'lc0' ? `${rowLabel(row)} ${row.strength}v` : `${rowLabel(row)} d${row.strength}`).join(' + ');
  el('message').textContent = `Analyzing (${selectedLabels}, ${multiPv()} lines)…`;
  try {
    const tasks: Promise<AnalysisLine[]>[] = [];
    for (const row of rows) {
      if (row.family === 'lc0' && row.variant === 'bt4') {
        tasks.push(workerBt4Lines(fen, row.strength));
      } else if (row.family === 'lc0') {
        if (workerReady) tasks.push(workerLc0Lines(fen, row.strength));
        else if (searcher) tasks.push(searcher.search({ positions: tree.historyBoards() }, { visits: row.strength, multiPv: multiPv(), signal: controller.signal, yieldEveryMs: 16 })
          .then((result) => lc0AnalysisLines(result, fen, 'Lc0')));
      } else if (row.family === 'sf') {
        const kind = row.variant === 'full' ? 'full' : 'lite';
        const label = kind === 'lite' ? `SF Lite d${row.strength}` : `SF d${row.strength}`;
        tasks.push(getStockfish(kind).analyze(fen, { multipv: multiPv(), depth: row.strength, signal: controller.signal })
          .then((infos) => stockfishAnalysisLines(infos, fen, label)));
      } else {
        const label = `${recklessVariantByKey(normalizeRecklessVariant(row.variant)).label} d${row.strength}`;
        tasks.push(getRecklessFor(row.variant).analyze(fen, { multipv: multiPv(), depth: row.strength, signal: controller.signal })
          .then((infos) => { renderRecklessRuntimeInfo(); return stockfishAnalysisLines(infos, fen, label); }));
      }
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
    bt4.cancel();
  });
  el('engineList').addEventListener('change', (event) => {
    const target = event.target as HTMLInputElement | HTMLSelectElement;
    const i = Number(target.dataset.i);
    if (Number.isNaN(i) || !engineRows[i]) return;
    if (target.classList.contains('row-fam')) {
      const family = target.value as EngineFamily;
      engineRows[i].family = family;
      engineRows[i].variant = defaultVariant(family);
      engineRows[i].strength = defaultStrength(family);
      renderEngineList();
    } else if (target.classList.contains('row-var')) {
      if (engineRows[i].family === 'lc0' && target.value === 'bt4') {
        // One-time gate before the ~353MB lazy load.
        if (!window.confirm(`${bt4LoadWarning()}\n\nUse Lc0 BT4?`)) { target.value = engineRows[i].variant; return; }
      }
      engineRows[i].variant = target.value;
    } else if (target.classList.contains('row-strength')) {
      const meta = strengthMeta(engineRows[i].family);
      engineRows[i].strength = Math.max(meta.min, Math.min(meta.max, Math.floor(Number(target.value) || meta.def)));
    }
    disposeUnusedEngines();
    lineCache.delete(tree.current.fen);
    void analyzeCurrent();
  });
  el('engineList').addEventListener('click', (event) => {
    const button = (event.target as HTMLElement).closest('.row-rm') as HTMLElement | null;
    if (!button) return;
    const i = Number(button.dataset.i);
    if (Number.isNaN(i) || engineRows.length <= 1) return;
    engineRows.splice(i, 1);
    renderEngineList();
    disposeUnusedEngines();
    lineCache.delete(tree.current.fen);
    void analyzeCurrent();
  });
  el('addEngine').addEventListener('click', () => {
    engineRows.push({ family: 'lc0', variant: 'small', strength: defaultStrength('lc0') });
    renderEngineList();
    lineCache.delete(tree.current.fen);
    void analyzeCurrent();
  });
  el('multiPvInput').addEventListener('change', () => { lineCache.delete(tree.current.fen); void analyzeCurrent(); });
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
  bt4.dispose();
  stockfishLite?.dispose();
  stockfishLite = null;
  stockfishFull?.dispose();
  stockfishFull = null;
  for (const engine of recklessByVariant.values()) engine.dispose();
  recklessByVariant.clear();
  void mainEvaluator?.dispose();
  mainEvaluator = null;
  searcher = null;
}

async function init() {
  window.addEventListener('pagehide', (event) => {
    if (!(event as PageTransitionEvent).persisted) disposeRuntimeResources();
  });
  renderAll();
  renderEngineList();
  renderRecklessRuntimeInfo();
  wireEvents();
  void refreshBt4Availability();
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
