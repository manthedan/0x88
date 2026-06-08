import { Chessground } from 'chessground';
import type { DrawShape } from 'chessground/draw';
import type { Key } from 'chessground/types';
import { boardToFen, parseFen, squareName, START_FEN, type BoardState } from '../chess/board.ts';
import { legalMoves, makeMove } from '../chess/movegen.ts';
import { moveToUci, type Move } from '../chess/moveCodec.ts';
import { gameTreeToPgn, parsePgnGame, parsePgnGames } from '../chess/pgn.ts';
import { collectOrtRuntimeDiagnostics } from '../nn/ortRuntime.ts';
import { CachedEvaluator, type Evaluator } from '../nn/evaluator.ts';
import { createBrowserSquareformerRuntimeEvaluator } from '../nn/browserRuntimeEvaluator.ts';
import { chooseMove, montyLitePuctPolicy } from '../search/puct.ts';
import { engineBrushes, evalBarWhitePercent, lc0AnalysisLines, stockfishAnalysisLines, tinyPuctAnalysisLines, type AnalysisLine } from './analysisFormat.ts';
import { GameTree, type GameNode } from './gameTree.ts';
import { fetchGameHistoryPgn, type ImportColor, type ImportSite } from './gameImport.ts';
import { openingStatsForPosition, openingSummary, type ImportedGame, type OpeningMoveStat } from './openingStats.ts';
import { loadLc0ModelForOrt } from './modelCache.ts';
import { Lc0OnnxEvaluator } from './onnxEvaluator.ts';
import { Lc0PuctSearcher } from './search.ts';
import { StockfishEngine, stockfishFlavorUrl } from './stockfishEngine.ts';
import { RecklessEngine, formatRecklessBrowserApiLoadStatus } from './recklessEngine.ts';
import { RECKLESS_VARIANTS, checkRecklessVariantAsset, hasExplicitRecklessVariant, recklessVariantAssetStatus, recklessVariantByKey, recklessVariantFromParams, normalizeRecklessVariant, resolveDefaultRecklessVariantAssetFallback, type RecklessVariant } from './recklessVariants.ts';
import { ViridithasEngine, canUsePersistentViridithasWasi } from './viridithasEngine.ts';
import { VIRIDITHAS_VARIANTS, checkViridithasVariantAsset, normalizeViridithasVariant, viridithasVariantAssetStatus, viridithasVariantByKey, viridithasVariantFromParams, type ViridithasVariant } from './viridithasVariants.ts';
import { BerserkEngine } from './berserkEngine.ts';
import { BERSERK_VARIANTS, berserkVariantAssetStatus, berserkVariantByKey, berserkVariantFromParams, checkBerserkVariantAsset, normalizeBerserkVariant, type BerserkVariant } from './berserkVariants.ts';
import { PlentyChessEngine } from './plentychessEngine.ts';
import { PLENTYCHESS_VARIANTS, checkPlentyChessVariantAsset, normalizePlentyChessVariant, plentyChessVariantAssetStatus, plentyChessVariantByKey, plentyChessVariantFromParams, type PlentyChessVariant } from './plentychessVariants.ts';
import { Bt4WorkerSearcher, bt4LoadWarning, bt4SupportedSync, probeBt4Support } from './bt4Engine.ts';
import { ENGINE_FAMILY_PRIORITY, defaultEngineStrength, defaultStaticEngineVariant, engineFamilyOptions, engineStrengthMeta, lc0EngineLabel, lc0VariantOptions, stockfishEngineLabel, stockfishVariantOptions, tinyEngineLabel, tinyVariantOptions, type EngineFamily, type EngineRow } from './engineCatalog.ts';

type Ground = ReturnType<typeof Chessground>;

const params = new URLSearchParams(location.search);
const DEFAULT_MODEL_URL = '/models/lc0/t1-256x10-distilled-swa-2432500.batch1.f32.onnx';
const MODEL_URL = params.get('model') ?? DEFAULT_MODEL_URL;
const DEFAULT_PACK_URL = '/models/lc0/t1-256x10-distilled-swa-2432500.batch8.f16.lc0web/model.lc0web.json';
const PACK_URL = params.get('pack') ?? params.get('modelPack') ?? DEFAULT_PACK_URL;
const DEFAULT_TINY_MODEL_URL = '/models/bt4_anneal_muon_best.onnx';
const DEFAULT_TINY_META_URL = '/models/bt4_anneal_muon_best.meta.json';
const DEFAULT_TINY_HYBRID_MANIFEST_URL = '/runtimes/squareformer-tvm-hybrid/bt4-anneal-muon-best/v1/manifest.json';
type Lc0AnalysisRuntime = 'onnx' | 'hybrid-ort-heads' | 'hybrid-wgsl-heads';
const REQUESTED_RECKLESS_EXPLICIT = hasExplicitRecklessVariant(params);
let REQUESTED_RECKLESS_VARIANT = recklessVariantFromParams(params);
const REQUESTED_VIRIDITHAS_VARIANT = viridithasVariantFromParams(params);
const REQUESTED_BERSERK_VARIANT = berserkVariantFromParams(params);
const REQUESTED_PLENTYCHESS_VARIANT = plentyChessVariantFromParams(params);

function sameOriginPathParam(names: string[], fallback: string, allowedPrefixes: string[]): string {
  for (const name of names) {
    const raw = params.get(name);
    if (!raw) continue;
    try {
      const url = new URL(raw, location.origin);
      if (url.origin === location.origin && allowedPrefixes.some((prefix) => url.pathname.startsWith(prefix))) return `${url.pathname}${url.search}${url.hash}`;
    } catch {
      // Ignore malformed experimental overrides and use the bundled default below.
    }
    console.warn(`[lc0-analysis] ignoring unsafe ${name} override`, raw);
  }
  return fallback;
}

const TINY_MODEL_URL = sameOriginPathParam(['tinyModel', 'tinyOnnx'], DEFAULT_TINY_MODEL_URL, ['/models/']);
const TINY_META_URL = sameOriginPathParam(['tinyMeta'], DEFAULT_TINY_META_URL, ['/models/']);
const TINY_HYBRID_MANIFEST_URL = sameOriginPathParam(['tinyManifest', 'manifest', 'manifestUrl'], DEFAULT_TINY_HYBRID_MANIFEST_URL, ['/runtimes/']);

let tree = new GameTree(params.get('fen') ?? START_FEN);
let searcher: Lc0PuctSearcher | null = null;
let mainEvaluator: Lc0OnnxEvaluator | null = null;
let stockfishLite: StockfishEngine | null = null;
let stockfishFull: StockfishEngine | null = null;
const tinyEvaluatorPromises = new Map<string, Promise<Evaluator>>();
const tinyEvaluators = new Set<CachedEvaluator>();
const tinyEvaluatorsByKey = new Map<string, CachedEvaluator>();
let tinyEvaluatorGeneration = 0;
let tinyHybridManifestStatus: 'unknown' | 'present' | 'missing' = 'unknown';
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

function normalizeLc0Runtime(value: string | null): Lc0AnalysisRuntime {
  const raw = (value ?? '').toLowerCase();
  if (raw === 'hybrid' || raw === 'lc0web' || raw === 'hybrid-ort-heads' || raw === 'wgsl-encoder') return 'hybrid-ort-heads';
  if (raw === 'hybrid-wgsl-heads' || raw === 'wgsl-heads' || raw === 'wgsl') return 'hybrid-wgsl-heads';
  return 'onnx';
}

function initialLc0Runtime(): Lc0AnalysisRuntime {
  if (params.get('headBackend') === 'wgsl' || params.get('hybridHeads') === 'wgsl') return 'hybrid-wgsl-heads';
  return normalizeLc0Runtime(params.get('lc0Runtime') ?? params.get('runtime'));
}

function selectedLc0Runtime(): Lc0AnalysisRuntime {
  return normalizeLc0Runtime(selectEl('lc0RuntimeSelect').value);
}

function lc0RuntimeLabel(runtime = selectedLc0Runtime()): string {
  if (runtime === 'hybrid-wgsl-heads') return 'WGSL encoder + WGSL heads';
  if (runtime === 'hybrid-ort-heads') return 'WGSL encoder + ORT heads';
  return 'ORT ONNX';
}

function lc0EncoderLayers(): number {
  return Math.min(32, Math.max(1, Math.floor(Number(params.get('encoderLayers') ?? params.get('layers') ?? '10') || 10)));
}

function lc0InitMessage(runtime = selectedLc0Runtime()): Record<string, unknown> {
  const common = { type: 'init', modelUrl: MODEL_URL, ep: requestedEp(), cacheModel: false };
  if (runtime === 'onnx') return common;
  return {
    ...common,
    runtime: 'hybrid',
    packUrl: PACK_URL,
    layers: lc0EncoderLayers(),
    verifyShards: params.get('packVerify') !== '0',
    headBackend: runtime === 'hybrid-wgsl-heads' ? 'wgsl' : 'ort',
    wgslBatchMode: 'physical',
    inputBackend: 'js',
    legalPriorsBackend: 'js',
  };
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
  const ready = await postWorker<{ backend: string }>(lc0InitMessage());
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

function tinyRuntimeForVariant(variant: string): 'auto' | 'ort' | 'custom-webgpu' {
  if (variant.endsWith('-ort')) return 'ort';
  if (variant.endsWith('-custom')) return 'custom-webgpu';
  return 'auto';
}

function tinyRuntimeFallbackForVariant(variant: string): boolean {
  return !variant.endsWith('-custom');
}

function tinyHybridManifestStatusText(): string {
  if (tinyHybridManifestStatus === 'present') return `Tiny hybrid bundle present (${TINY_HYBRID_MANIFEST_URL})`;
  if (tinyHybridManifestStatus === 'missing') return `Tiny hybrid bundle missing; auto uses ORT fallback (${TINY_HYBRID_MANIFEST_URL})`;
  return `Tiny hybrid bundle checking (${TINY_HYBRID_MANIFEST_URL})`;
}

async function refreshTinyHybridManifestStatus(): Promise<void> {
  try {
    const res = await fetch(TINY_HYBRID_MANIFEST_URL, { method: 'HEAD', cache: 'no-store' });
    tinyHybridManifestStatus = res.ok ? 'present' : 'missing';
  } catch {
    tinyHybridManifestStatus = 'missing';
  }
  if (tinyHybridManifestStatus === 'missing') {
    for (const row of engineRows) if (row.family === 'tiny' && row.variant === 'bt4-custom') row.variant = 'bt4-auto';
    renderEngineList();
  }
  renderRecklessRuntimeInfo();
}

function tinyEvaluatorCacheKey(variant: string): string {
  const runtime = tinyRuntimeForVariant(variant);
  const fallback = tinyRuntimeFallbackForVariant(variant);
  return `${runtime}:${fallback ? 'fallback' : 'strict'}:${TINY_MODEL_URL}:${TINY_META_URL}:${TINY_HYBRID_MANIFEST_URL}`;
}

function activeTinyEvaluatorKeys(): Set<string> {
  return new Set(activeEngineRows().filter((row) => row.family === 'tiny').map((row) => tinyEvaluatorCacheKey(row.variant)));
}

async function tinyEvaluator(variant: string): Promise<Evaluator> {
  const runtime = tinyRuntimeForVariant(variant);
  const fallback = tinyRuntimeFallbackForVariant(variant);
  const key = tinyEvaluatorCacheKey(variant);
  const existing = tinyEvaluatorPromises.get(key);
  if (existing) return existing;
  const generation = tinyEvaluatorGeneration;
  const created = (async () => {
    const loaded = await createBrowserSquareformerRuntimeEvaluator({
      id: 'bt4-anneal-muon-best',
      modelId: 'bt4-anneal-muon-best',
      label: tinyEngineLabel(variant),
      onnx: TINY_MODEL_URL,
      meta: TINY_META_URL,
      runtime,
      manifestUrl: TINY_HYBRID_MANIFEST_URL,
    }, { params, runtime, manifestUrl: TINY_HYBRID_MANIFEST_URL, fallback });
    console.info('[lc0-analysis] loaded Tiny Leela evaluator', {
      requestedRuntime: loaded.requestedRuntime,
      resolvedRuntime: loaded.resolvedRuntime,
      runtimeConfigId: loaded.runtimeConfigId,
      manifestUrl: loaded.manifestUrl,
      fallbackReason: loaded.fallbackReason,
    });
    const cached = new CachedEvaluator(loaded.evaluator, { maxEntries: 4096, includeHistory: true, includeLegalMoves: true, label: `tiny-leela-analysis:${runtime}` });
    if (tinyEvaluatorGeneration !== generation || !activeTinyEvaluatorKeys().has(key)) {
      destroyTinyEvaluator(cached);
      const error = new Error('Tiny Leela evaluator was disposed before it finished loading');
      error.name = 'AbortError';
      throw error;
    }
    tinyEvaluators.add(cached);
    tinyEvaluatorsByKey.set(key, cached);
    return cached;
  })();
  tinyEvaluatorPromises.set(key, created);
  try {
    return await created;
  } catch (error) {
    tinyEvaluatorPromises.delete(key);
    throw error;
  }
}

function tinyHistoryFens(positions: BoardState[]): string[] {
  return positions.slice(0, -1).map(boardToFen).reverse().slice(0, 16);
}

function destroyTinyEvaluator(evaluator: CachedEvaluator): void {
  evaluator.clear();
  const destroy = (evaluator.inner as Evaluator & { destroy?: () => void }).destroy;
  if (typeof destroy === 'function') destroy.call(evaluator.inner);
}

function disposeTinyEvaluators(): void {
  tinyEvaluatorGeneration++;
  for (const evaluator of tinyEvaluators) destroyTinyEvaluator(evaluator);
  tinyEvaluators.clear();
  tinyEvaluatorsByKey.clear();
  tinyEvaluatorPromises.clear();
}

function disposeUnusedTinyEvaluators(): void {
  const activeTinyKeys = activeTinyEvaluatorKeys();
  for (const key of [...tinyEvaluatorPromises.keys()]) if (!activeTinyKeys.has(key)) tinyEvaluatorPromises.delete(key);
  for (const [key, evaluator] of [...tinyEvaluatorsByKey]) {
    if (activeTinyKeys.has(key)) continue;
    destroyTinyEvaluator(evaluator);
    tinyEvaluators.delete(evaluator);
    tinyEvaluatorsByKey.delete(key);
    tinyEvaluatorPromises.delete(key);
  }
}

// Engines to analyze are chosen as an add/remove list of cascading selects:
// family (Lc0/Tiny Leela/Stockfish/Reckless/Viridithas/Berserk/PlentyChess)
// -> variant (Lc0: Small|BT4; Tiny: runtime config; SF: Lite|Full; UCI engines: variant)
// -> strength (neural visits, UCI depth), all per row.
function strengthMeta(family: EngineFamily) {
  return engineStrengthMeta(family, 'analysis');
}
function defaultStrength(family: EngineFamily): number { return defaultEngineStrength(family, 'analysis'); }

function availableRecklessVariants(): RecklessVariant[] {
  return REQUESTED_RECKLESS_VARIANT.key === 'custom' ? [...RECKLESS_VARIANTS, REQUESTED_RECKLESS_VARIANT] : [...RECKLESS_VARIANTS];
}

function recklessVariantForKey(variantKey: string): RecklessVariant {
  const key = normalizeRecklessVariant(variantKey);
  if (key === 'custom' && REQUESTED_RECKLESS_VARIANT.key === 'custom') return REQUESTED_RECKLESS_VARIANT;
  return recklessVariantByKey(key);
}

function recklessCacheKey(variant: RecklessVariant): string {
  return `${variant.key}:${variant.wasmUrl}:${variant.nnueUrl ?? ''}`;
}

function availableViridithasVariants(): ViridithasVariant[] {
  return REQUESTED_VIRIDITHAS_VARIANT.key === 'custom' ? [...VIRIDITHAS_VARIANTS, REQUESTED_VIRIDITHAS_VARIANT] : [...VIRIDITHAS_VARIANTS];
}

function viridithasVariantForKey(variantKey: string): ViridithasVariant {
  const key = normalizeViridithasVariant(variantKey);
  if (key === 'custom' && REQUESTED_VIRIDITHAS_VARIANT.key === 'custom') return REQUESTED_VIRIDITHAS_VARIANT;
  return viridithasVariantByKey(key);
}

function viridithasCacheKey(variant: ViridithasVariant): string {
  return `${variant.key}:${variant.wasmUrl}`;
}

function availableBerserkVariants(): BerserkVariant[] {
  const builtIns = BERSERK_VARIANTS.filter((variant) => !!variant.jsUrl);
  if (REQUESTED_BERSERK_VARIANT.key === 'custom' && REQUESTED_BERSERK_VARIANT.jsUrl) return [...builtIns, REQUESTED_BERSERK_VARIANT];
  return builtIns;
}

function berserkVariantForKey(variantKey: string): BerserkVariant {
  const key = normalizeBerserkVariant(variantKey);
  if (key === 'custom' && REQUESTED_BERSERK_VARIANT.key === 'custom' && REQUESTED_BERSERK_VARIANT.jsUrl) return REQUESTED_BERSERK_VARIANT;
  const variant = berserkVariantByKey(key);
  return variant.jsUrl ? variant : BERSERK_VARIANTS.find((entry) => entry.jsUrl)!;
}

function berserkCacheKey(variant: BerserkVariant): string {
  return `${variant.key}:${variant.jsUrl ?? ''}:${variant.wasmUrl}:${variant.dataUrl ?? ''}`;
}

function availablePlentyChessVariants(): PlentyChessVariant[] {
  if (REQUESTED_PLENTYCHESS_VARIANT.key === 'custom') return [...PLENTYCHESS_VARIANTS, REQUESTED_PLENTYCHESS_VARIANT];
  return [...PLENTYCHESS_VARIANTS];
}

function plentyChessVariantForKey(variantKey: string): PlentyChessVariant {
  const key = normalizePlentyChessVariant(variantKey);
  if (key === 'custom' && REQUESTED_PLENTYCHESS_VARIANT.key === 'custom') return REQUESTED_PLENTYCHESS_VARIANT;
  return plentyChessVariantByKey(key);
}

function plentyChessCacheKey(variant: PlentyChessVariant): string {
  return `${variant.key}:${variant.jsUrl}:${variant.wasmUrl}:${variant.dataUrl}`;
}

// "Add engine" fills the next missing family by priority
// (Lc0 → Tiny Leela → SF → Reckless → Viridithas → Berserk → PlentyChess),
// falling back to the top priority when all families are present.
function nextEngineFamily(): EngineFamily {
  const present = new Set(engineRows.map((row) => row.family));
  return ENGINE_FAMILY_PRIORITY.find((family) => !present.has(family)) ?? ENGINE_FAMILY_PRIORITY[0];
}

let engineRows: EngineRow[] = [{ family: 'lc0', variant: 'small', strength: 400 }];

function analysisEngineFamilyOptions(): { value: EngineFamily; label: string }[] {
  return engineFamilyOptions();
}

function variantOptions(family: EngineFamily): { value: string; label: string; disabled?: boolean }[] {
  if (family === 'tiny') return tinyVariantOptions().map((option) => option.value === 'bt4-custom' && tinyHybridManifestStatus === 'missing'
    ? { ...option, disabled: true, label: `${option.label} (bundle missing)` }
    : option);
  if (family === 'lc0') return lc0VariantOptions(bt4SupportedSync());
  if (family === 'sf') return stockfishVariantOptions();
  if (family === 'viridithas') return availableViridithasVariants().map((v) => ({ value: v.key, label: v.label }));
  if (family === 'berserk') return availableBerserkVariants().map((v) => ({ value: v.key, label: v.label }));
  if (family === 'plentychess') return availablePlentyChessVariants().map((v) => ({ value: v.key, label: v.label }));
  return availableRecklessVariants().map((v) => ({ value: v.key, label: v.label }));
}

function defaultVariant(family: EngineFamily): string {
  if (family === 'reckless') return REQUESTED_RECKLESS_VARIANT.key;
  if (family === 'viridithas') return REQUESTED_VIRIDITHAS_VARIANT.key;
  if (family === 'berserk') return REQUESTED_BERSERK_VARIANT.key;
  if (family === 'plentychess') return REQUESTED_PLENTYCHESS_VARIANT.key;
  return defaultStaticEngineVariant(family);
}

function rowLabel(row: EngineRow): string {
  if (row.family === 'tiny') return tinyEngineLabel(row.variant);
  if (row.family === 'lc0') return lc0EngineLabel(row.variant);
  if (row.family === 'sf') return stockfishEngineLabel(row.variant, 'analysis');
  if (row.family === 'viridithas') return viridithasVariantForKey(row.variant).label;
  if (row.family === 'berserk') return berserkVariantForKey(row.variant).label;
  if (row.family === 'plentychess') return plentyChessVariantForKey(row.variant).label;
  return recklessVariantForKey(row.variant).label;
}

function activeEngineRows(): EngineRow[] {
  const seen = new Set<string>();
  return engineRows.filter((r) => {
    const k = `${r.family}:${r.variant}`;
    if (seen.has(k)) return false;
    seen.add(k);
    return true;
  });
}

function usesBt4Row(): boolean {
  return engineRows.some((r) => r.family === 'lc0' && r.variant === 'bt4');
}

function renderEngineList(): void {
  const families = analysisEngineFamilyOptions();
  el('engineList').innerHTML = engineRows.map((row, i) => {
    const famSel = families.map(({ value, label }) => `<option value="${value}"${row.family === value ? ' selected' : ''}>${label}</option>`).join('');
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
  const fallbackMode = (typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated) ? 'persistent available' : 'one-shot fallback';
  const tinyRows = activeEngineRows().filter((row) => row.family === 'tiny');
  const tinyParts = tinyRows.length ? tinyRows.map((row) => `${tinyEngineLabel(row.variant)} · SquareFormer ${tinyRuntimeForVariant(row.variant)} · ${tinyHybridManifestStatusText()}`) : [];
  const recklessRows = activeEngineRows().filter((row) => row.family === 'reckless');
  const recklessVariants = recklessRows.length ? recklessRows.map((row) => recklessVariantForKey(row.variant)) : [REQUESTED_RECKLESS_VARIANT];
  const recklessParts = recklessVariants.map((variant) => {
    const engine = recklessByVariant.get(recklessCacheKey(variant));
    const status = engine?.runtimeStatus();
    const mode = engine?.runtimeLabel() ?? fallbackMode;
    const asset = recklessVariantAssetStatus(variant);
    if (asset === 'unknown') void checkRecklessVariantAsset(variant, renderRecklessRuntimeInfo);
    const assetText = asset === 'present' ? 'asset ok' : asset === 'missing' ? 'asset missing' : 'checking asset';
    const targetUrl = status?.wasmUrl ?? variant.wasmUrl;
    const assetUrlText = variant.nnueUrl ? `${targetUrl} + ${variant.nnueUrl}` : targetUrl;
    const loadText = formatRecklessBrowserApiLoadStatus(status?.browserApiLoad);
    return `${variant.label} · ${mode} · ${sab} · ${assetText} · ${assetUrlText}${loadText ? ` · ${loadText}` : ''}${status?.persistentDisabled ? ' · persistent disabled after fallback' : ''}${asset === 'missing' ? ' · build locally with npm run reckless:build-wasi, reckless:build-simd-wasi, reckless:build-browser-api-simd, reckless:build-browser-api-simd-external, or reckless:build-lite-wasi' : ''}`;
  });
  const viridithasRows = activeEngineRows().filter((row) => row.family === 'viridithas');
  const viridithasVariants = viridithasRows.length ? viridithasRows.map((row) => viridithasVariantForKey(row.variant)) : [REQUESTED_VIRIDITHAS_VARIANT];
  const viridithasParts = viridithasVariants.map((variant) => {
    const engine = viridithasByVariant.get(viridithasCacheKey(variant));
    const status = engine?.runtimeStatus();
    const mode = engine?.runtimeLabel() ?? (canUsePersistentViridithasWasi() ? 'persistent available' : 'one-shot fallback');
    const asset = viridithasVariantAssetStatus(variant);
    if (asset === 'unknown') void checkViridithasVariantAsset(variant, renderRecklessRuntimeInfo);
    const assetText = asset === 'ok' ? 'asset ok' : asset === 'missing' ? 'asset missing' : 'checking asset';
    return `${variant.label} · ${mode} · ${sab} · ${assetText} · ${status?.wasmUrl ?? variant.wasmUrl}${status?.persistentDisabled ? ' · persistent disabled after fallback' : ''}${asset === 'missing' ? ' · build locally with npm run viridithas:build-wasi or viridithas:build-simd-wasi' : ''}`;
  });
  const berserkRows = activeEngineRows().filter((row) => row.family === 'berserk');
  const berserkVariants = berserkRows.length ? berserkRows.map((row) => berserkVariantForKey(row.variant)) : [REQUESTED_BERSERK_VARIANT].filter((variant) => !!variant.jsUrl);
  const berserkParts = berserkVariants.map((variant) => {
    const engine = berserkByVariant.get(berserkCacheKey(variant));
    const asset = berserkVariantAssetStatus(variant);
    if (asset === 'unknown') void checkBerserkVariantAsset(variant, renderRecklessRuntimeInfo);
    const assetText = asset === 'present' ? 'asset ok' : asset === 'missing' ? 'asset missing' : 'checking asset';
    return `${variant.label} · ${engine?.runtimeLabel() ?? 'Emscripten worker idle'} · ${assetText} · ${variant.jsUrl ?? variant.wasmUrl}`;
  });
  const plentyRows = activeEngineRows().filter((row) => row.family === 'plentychess');
  const plentyVariants = plentyRows.length ? plentyRows.map((row) => plentyChessVariantForKey(row.variant)) : [REQUESTED_PLENTYCHESS_VARIANT];
  const plentyParts = plentyVariants.map((variant) => {
    const engine = plentyChessByVariant.get(plentyChessCacheKey(variant));
    const asset = plentyChessVariantAssetStatus(variant);
    if (asset === 'unknown') void checkPlentyChessVariantAsset(variant, renderRecklessRuntimeInfo);
    const assetText = asset === 'present' ? 'asset ok' : asset === 'missing' ? 'asset missing' : 'checking asset';
    return `${variant.label} · ${engine?.runtimeLabel() ?? 'Emscripten worker idle'} · ${assetText} · ${variant.jsUrl}`;
  });
  el('recklessRuntimeInfo').textContent = `Tiny: ${tinyParts.join(' | ') || 'not selected'} · Reckless: ${recklessParts.join(' | ')} · Viridithas: ${viridithasParts.join(' | ')} · Berserk: ${berserkParts.join(' | ') || 'not selected'} · PlentyChess: ${plentyParts.join(' | ') || 'not selected'}`;
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
const viridithasByVariant = new Map<string, ViridithasEngine>();
const berserkByVariant = new Map<string, BerserkEngine>();
const plentyChessByVariant = new Map<string, PlentyChessEngine>();
function getRecklessFor(variantKey: string): RecklessEngine {
  const variant = recklessVariantForKey(variantKey);
  const key = recklessCacheKey(variant);
  let engine = recklessByVariant.get(key);
  if (!engine) {
    engine = new RecklessEngine({ depth: 4, hashMb: 16 }, variant.wasmUrl, { backend: variant.backend ?? 'wasi', nnueUrl: variant.nnueUrl, onStatus: renderRecklessRuntimeInfo });
    recklessByVariant.set(key, engine);
    void engine.prewarm()
      .then(renderRecklessRuntimeInfo)
      .catch((error) => {
        if ((error as Error).name !== 'AbortError') console.warn('Reckless prewarm failed', error);
        renderRecklessRuntimeInfo();
      });
  }
  return engine;
}

function getViridithasFor(variantKey: string): ViridithasEngine {
  const variant = viridithasVariantForKey(variantKey);
  const key = viridithasCacheKey(variant);
  let engine = viridithasByVariant.get(key);
  if (!engine) {
    engine = new ViridithasEngine({ depth: 4, hashMb: 16 }, variant.wasmUrl);
    viridithasByVariant.set(key, engine);
    renderRecklessRuntimeInfo();
  }
  return engine;
}

function getBerserkFor(variantKey: string): BerserkEngine {
  const variant = berserkVariantForKey(variantKey);
  const key = berserkCacheKey(variant);
  let engine = berserkByVariant.get(key);
  if (!engine) {
    engine = new BerserkEngine({ depth: 4, hashMb: 16, threads: 1 }, variant.jsUrl, variant.wasmUrl, variant.dataUrl);
    berserkByVariant.set(key, engine);
    renderRecklessRuntimeInfo();
  }
  return engine;
}

function getPlentyChessFor(variantKey: string): PlentyChessEngine {
  const variant = plentyChessVariantForKey(variantKey);
  const key = plentyChessCacheKey(variant);
  let engine = plentyChessByVariant.get(key);
  if (!engine) {
    engine = new PlentyChessEngine({ depth: 4, hashMb: 16, threads: 1 }, variant.jsUrl, variant.wasmUrl, variant.dataUrl);
    plentyChessByVariant.set(key, engine);
    renderRecklessRuntimeInfo();
  }
  return engine;
}

function disposeUnusedEngines(): void {
  if (!usesBt4Row()) bt4.dispose();
  disposeUnusedTinyEvaluators();
  const activeRows = activeEngineRows();
  const activeRecklessKeys = new Set(activeRows.filter((row) => row.family === 'reckless').map((row) => recklessCacheKey(recklessVariantForKey(row.variant))));
  for (const [key, engine] of [...recklessByVariant]) {
    if (!activeRecklessKeys.has(key)) {
      engine.dispose();
      recklessByVariant.delete(key);
    }
  }
  const activeViridithasKeys = new Set(activeRows.filter((row) => row.family === 'viridithas').map((row) => viridithasCacheKey(viridithasVariantForKey(row.variant))));
  for (const [key, engine] of [...viridithasByVariant]) {
    if (!activeViridithasKeys.has(key)) {
      engine.dispose();
      viridithasByVariant.delete(key);
    }
  }
  const activeBerserkKeys = new Set(activeRows.filter((row) => row.family === 'berserk').map((row) => berserkCacheKey(berserkVariantForKey(row.variant))));
  for (const [key, engine] of [...berserkByVariant]) {
    if (!activeBerserkKeys.has(key)) {
      engine.dispose();
      berserkByVariant.delete(key);
    }
  }
  const activePlentyKeys = new Set(activeRows.filter((row) => row.family === 'plentychess').map((row) => plentyChessCacheKey(plentyChessVariantForKey(row.variant))));
  for (const [key, engine] of [...plentyChessByVariant]) {
    if (!activePlentyKeys.has(key)) {
      engine.dispose();
      plentyChessByVariant.delete(key);
    }
  }
  renderRecklessRuntimeInfo();
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

function firstSanMove(line: AnalysisLine): string {
  return line.pvSan.split(/\s+/).find(Boolean) ?? line.pvUci[0] ?? '—';
}

function signedCp(value: number): string {
  const sign = value > 0 ? '+' : '';
  return `${sign}${Math.round(value)}`;
}

function renderEngineComparison(lines: AnalysisLine[]): void {
  const bestByEngine = [...new Map(lines
    .filter((line) => line.multipv === 1 && line.pvUci[0])
    .map((line) => [line.engine, line])).values()];
  const body = el('engineCompare').querySelector('tbody')!;
  if (!bestByEngine.length) {
    el('engineConsensus').textContent = 'No analysis yet.';
    body.innerHTML = '<tr><td colspan="6" class="small">Run analysis to compare selected engines.</td></tr>';
    return;
  }
  const finiteScores = bestByEngine.map((line) => line.scoreCp).filter((score): score is number => score !== undefined);
  const evalSpread = finiteScores.length >= 2 ? Math.max(...finiteScores) - Math.min(...finiteScores) : null;
  const moveCounts = new Map<string, { count: number; san: string }>();
  for (const line of bestByEngine) {
    const uci = line.pvUci[0]!;
    const current = moveCounts.get(uci) ?? { count: 0, san: firstSanMove(line) };
    current.count += 1;
    moveCounts.set(uci, current);
  }
  const consensus = [...moveCounts.entries()].sort((a, b) => b[1].count - a[1].count)[0];
  const consensusUci = consensus?.[0] ?? '';
  const consensusText = consensus
    ? `${consensus[1].count}/${bestByEngine.length} engines prefer ${consensus[1].san} (${consensusUci})`
    : 'No first-move consensus.';
  const spreadText = evalSpread === null ? 'eval spread unavailable' : `eval spread ${signedCp(evalSpread)} cp`;
  el('engineConsensus').textContent = `${consensusText} · ${spreadText}`;
  const reference = finiteScores.length ? finiteScores[0] : undefined;
  body.innerHTML = bestByEngine.map((line) => {
    const swatch = engineBrushes(line.engine).swatch;
    const delta = reference === undefined || line.scoreCp === undefined ? '—' : signedCp(line.scoreCp - reference);
    const agreed = line.pvUci[0] === consensusUci && (consensus?.[1].count ?? 0) > 1;
    return `<tr style="border-left:3px solid ${swatch}">`
      + `<td>${htmlEscape(line.engine)}</td>`
      + `<td class="mono ${agreed ? 'agree' : ''}">${htmlEscape(firstSanMove(line))}<br><span class="small">${htmlEscape(line.pvUci[0] ?? '')}</span></td>`
      + `<td class="mono">${htmlEscape(line.scoreText)}</td>`
      + `<td class="mono">${htmlEscape(delta)}</td>`
      + `<td class="mono">${htmlEscape(line.detail)}</td>`
      + `<td class="pv">${htmlEscape(line.pvSan)}</td>`
      + '</tr>';
  }).join('');
}

function renderLines() {
  const lines = lineCache.get(tree.current.fen) ?? [];
  renderLegend(lines);
  renderEngineComparison(lines);
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
  const selectedLabels = rows.map((row) => (row.family === 'lc0' || row.family === 'tiny') ? `${rowLabel(row)} ${row.strength}v` : `${rowLabel(row)} d${row.strength}`).join(' + ');
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
      } else if (row.family === 'tiny') {
        const positions = tree.historyBoards();
        const current = positions[positions.length - 1];
        tasks.push(tinyEvaluator(row.variant)
          .then((evaluator) => chooseMove(current, evaluator, {
            visits: row.strength,
            batchSize: Math.max(1, Math.min(256, Math.floor(Number(params.get('tinyBatch') ?? '32') || 32))),
            signal: controller.signal,
            historyFens: tinyHistoryFens(positions),
            searchPolicy: montyLitePuctPolicy,
            includePv: true,
            multiPv: multiPv(),
            pvDepth: 12,
          }))
          .then((result) => tinyPuctAnalysisLines(result, fen, tinyEngineLabel(row.variant))));
      } else if (row.family === 'sf') {
        const kind = row.variant === 'full' ? 'full' : 'lite';
        const label = kind === 'lite' ? `SF Lite d${row.strength}` : `SF d${row.strength}`;
        tasks.push(getStockfish(kind).analyze(fen, { multipv: multiPv(), depth: row.strength, signal: controller.signal })
          .then((infos) => stockfishAnalysisLines(infos, fen, label)));
      } else if (row.family === 'viridithas') {
        const label = `${viridithasVariantForKey(row.variant).label} d${row.strength}`;
        const engine = getViridithasFor(row.variant);
        tasks.push(engine.newGame(controller.signal)
          .then(() => engine.analyze(fen, { multipv: multiPv(), depth: row.strength, signal: controller.signal }))
          .then((infos) => { renderRecklessRuntimeInfo(); return stockfishAnalysisLines(infos, fen, label); }));
      } else if (row.family === 'berserk') {
        const label = `${berserkVariantForKey(row.variant).label} d${row.strength}`;
        const engine = getBerserkFor(row.variant);
        tasks.push(engine.newGame(controller.signal)
          .then(() => engine.analyze(fen, { multipv: multiPv(), depth: row.strength, signal: controller.signal }))
          .then((infos) => { renderRecklessRuntimeInfo(); return stockfishAnalysisLines(infos, fen, label); }));
      } else if (row.family === 'plentychess') {
        const label = `${plentyChessVariantForKey(row.variant).label} d${row.strength}`;
        const engine = getPlentyChessFor(row.variant);
        tasks.push(engine.newGame(controller.signal)
          .then(() => engine.analyze(fen, { multipv: multiPv(), depth: row.strength, signal: controller.signal }))
          .then((infos) => { renderRecklessRuntimeInfo(); return stockfishAnalysisLines(infos, fen, label); }));
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
    const family = nextEngineFamily();
    engineRows.push({ family, variant: defaultVariant(family), strength: defaultStrength(family) });
    renderEngineList();
    lineCache.delete(tree.current.fen);
    void analyzeCurrent();
  });
  el('lc0RuntimeSelect').addEventListener('change', () => { void reloadLc0Backend(); });
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
  disposeTinyEvaluators();
  for (const engine of recklessByVariant.values()) engine.dispose();
  recklessByVariant.clear();
  for (const engine of viridithasByVariant.values()) engine.dispose();
  viridithasByVariant.clear();
  for (const engine of berserkByVariant.values()) engine.dispose();
  berserkByVariant.clear();
  for (const engine of plentyChessByVariant.values()) engine.dispose();
  plentyChessByVariant.clear();
  void mainEvaluator?.dispose();
  mainEvaluator = null;
  searcher = null;
}

async function loadLc0Backend(): Promise<void> {
  const runtime = selectedLc0Runtime();
  el('analyze').toggleAttribute('disabled', true);
  selectEl('lc0RuntimeSelect').disabled = true;
  el('backend').textContent = `loading ${lc0RuntimeLabel(runtime)}…`;
  el('message').textContent = `Loading LC0 ${lc0RuntimeLabel(runtime)} in a worker…`;
  try {
    el('backend').textContent = await initWorker();
    el('analyze').toggleAttribute('disabled', false);
    selectEl('lc0RuntimeSelect').disabled = false;
    el('message').textContent = 'Ready. Drag a move, load a PGN/FEN, or Analyze. Navigation stays responsive.';
    if (inputEl('autoAnalyze').checked) void analyzeCurrent();
  } catch (workerError) {
    if (runtime !== 'onnx') {
      selectEl('lc0RuntimeSelect').disabled = false;
      el('message').textContent = `LC0 ${lc0RuntimeLabel(runtime)} load failed: ${(workerError as Error).message}`;
      return;
    }
    // Fall back to a main-thread evaluator (analysis will block the UI, but works).
    console.warn('LC0 worker init failed; falling back to the main thread.', workerError);
    try {
      const modelLoad = await loadLc0ModelForOrt(MODEL_URL, { cache: false });
      mainEvaluator = await Lc0OnnxEvaluator.create(modelLoad.model);
      searcher = new Lc0PuctSearcher(mainEvaluator);
      const diagnostics = await collectOrtRuntimeDiagnostics();
      el('backend').textContent = `${diagnostics.describe} (main thread)`;
      el('analyze').toggleAttribute('disabled', false);
      selectEl('lc0RuntimeSelect').disabled = false;
      el('message').textContent = 'Ready (main-thread fallback — deep analysis may pause the UI).';
      if (inputEl('autoAnalyze').checked) void analyzeCurrent();
    } catch (error) {
      selectEl('lc0RuntimeSelect').disabled = false;
      el('message').textContent = `Model load failed: ${(error as Error).message}`;
    }
  }
}

async function reloadLc0Backend(): Promise<void> {
  lineCache.clear();
  disposeRuntimeResources();
  renderRecklessRuntimeInfo();
  await loadLc0Backend();
}

async function init() {
  REQUESTED_RECKLESS_VARIANT = await resolveDefaultRecklessVariantAssetFallback(REQUESTED_RECKLESS_VARIANT, REQUESTED_RECKLESS_EXPLICIT, renderRecklessRuntimeInfo);
  window.addEventListener('pagehide', (event) => {
    if (!(event as PageTransitionEvent).persisted) disposeRuntimeResources();
  });
  selectEl('lc0RuntimeSelect').value = initialLc0Runtime();
  renderAll();
  renderEngineList();
  renderRecklessRuntimeInfo();
  wireEvents();
  void refreshBt4Availability();
  void refreshTinyHybridManifestStatus();
  await loadLc0Backend();
}

void init();
