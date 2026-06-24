import { Chessground } from 'chessground';
import type { DrawShape } from 'chessground/draw';
import type { Key } from 'chessground/types';
import { boardToFen, parseFen, squareName, START_FEN, type BoardState } from '../chess/board.ts';
import { inCheck, legalMoves, makeMove } from '../chess/movegen.ts';
import { moveToUci, type Move } from '../chess/moveCodec.ts';
import { boardCheck, legalDests, matchUserMoves, showPromotionOverlay } from './boardUx.ts';
import { moveToSan } from '../chess/san.ts';
import { Maia3BrowserEvaluator, maia3WinProbability } from './maia3.ts';
import { gameTreeToPgn, parsePgnGame, parsePgnGames } from '../chess/pgn.ts';
import { collectOrtRuntimeDiagnostics } from '../nn/ortRuntime.ts';
import { CachedEvaluator, type Evaluator } from '../nn/evaluator.ts';
import { BROWSER_RUNTIME_AUDIT_EVENT, formatBrowserRuntimeAudit, publishBrowserRuntimeAudit, type BrowserRuntimeAuditDetail } from '../nn/runtimeAudit.ts';
import { createBrowserSquareformerRuntimeEvaluator } from '../nn/browserRuntimeEvaluator.ts';
import { chooseMove, montyLitePuctPolicy } from '../search/puct.ts';
import { ANALYSIS_DRAWABLE_BRUSHES, engineBrushes, evalBarWhitePercent, lc0AnalysisLines, stockfishAnalysisLines, tinyPuctAnalysisLines, type AnalysisLine } from './analysisFormat.ts';
import { annotatedPgn, reviewGame, type GameReview, type ReviewPosition, type ReviewedMove } from './gameReview.ts';
import { lineChartSvg } from './charts.ts';
import { GameTree, type GameNode } from './gameTree.ts';
import { fetchGameHistoryPgn, type ImportColor, type ImportSite } from './gameImport.ts';
import { buildOpeningPositionIndex, mergeOpeningMoveStats, openingStatsFromIndex, openingStatsForPosition, openingSummary, positionKey, type ImportedGame, type OpeningMoveStat, type OpeningPositionIndex } from './openingStats.ts';
import { defaultPgnCollectionName, deletePgnCollection, duplicatePgnCollection, exportPgnDatabaseBackup, formatPgnCollectionSummary, importPgnDatabaseBackup, listPgnCollections, loadPgnCollection, pgnDatabaseAvailable, pgnDatabaseBackupFilename, renamePgnCollection, savePgnCollection, searchPgnCollectionsByPosition, updatePgnCollectionPositionIndex, type PgnCollectionSource, type PgnCollectionSummary } from './pgnDatabase.ts';

type PgnDatabaseSearchResult = Awaited<ReturnType<typeof searchPgnCollectionsByPosition>>[number];
import { loadLc0ModelForOrt } from './modelCache.ts';
import { Lc0OnnxEvaluator, type Lc0EvaluationProvider } from './onnxEvaluator.ts';
import { Lc0PuctSearcher } from './search.ts';
import { Lc0WholeOnnxWebgpuEvaluator } from './wholeOnnxWebgpuEvaluator.ts';
import { StockfishEngine, stockfishFlavorUrl } from './stockfishEngine.ts';
import { RecklessEngine, formatRecklessBrowserApiLoadStatus } from './recklessEngine.ts';
import { RECKLESS_VARIANTS, checkRecklessVariantAsset, hasExplicitRecklessVariant, recklessVariantAssetStatus, recklessVariantByKey, recklessVariantFromParams, normalizeRecklessVariant, resolveDefaultRecklessVariantAssetFallback, supportsWasmRelaxedSimd, type RecklessVariant } from './recklessVariants.ts';
import { ViridithasEngine, canUsePersistentViridithasWasi } from './viridithasEngine.ts';
import { VIRIDITHAS_VARIANTS, checkViridithasVariantAsset, hasExplicitViridithasVariant, normalizeViridithasVariant, resolveDefaultViridithasVariantAssetFallback, viridithasVariantAssetStatus, viridithasVariantByKey, viridithasVariantFromParams, type ViridithasVariant } from './viridithasVariants.ts';
import { BerserkEngine } from './berserkEngine.ts';
import { BERSERK_VARIANTS, berserkVariantAssetStatus, berserkVariantByKey, berserkVariantFromParams, checkBerserkVariantAsset, hasExplicitBerserkVariant, normalizeBerserkVariant, resolveDefaultBerserkVariantAssetFallback, type BerserkVariant } from './berserkVariants.ts';
import { PlentyChessEngine } from './plentychessEngine.ts';
import { berserkCacheKey, createBerserkEngine, createPlentyChessEngine, createRecklessEngine, createViridithasEngine, plentyChessCacheKey, recklessCacheKey, viridithasCacheKey } from './engineProvision.ts';
import { PLENTYCHESS_VARIANTS, checkPlentyChessVariantAsset, hasExplicitPlentyChessVariant, normalizePlentyChessVariant, plentyChessVariantAssetStatus, plentyChessVariantByKey, plentyChessVariantFromParams, plentyChessVariantUnsupportedReason, resolveDefaultPlentyChessVariantAssetFallback, type PlentyChessVariant } from './plentychessVariants.ts';
import { BIG_NETS, bigNetAssetStatusSync, bigNetLoadWarning, bigNetOptionState, bt4SupportedSync, checkBigNetAsset, probeBt4Support, type BigNetConfig, type Bt4WorkerSearcher } from './bt4Engine.ts';
import { acquireBigNetSearcher, disposeBigNetSearcherNow, peekBigNetSearcher, releaseBigNetSearcher, type BigNetKey } from './bigNetSessionPool.ts';
import { ENGINE_FAMILY_PRIORITY, defaultEngineStrength, defaultStaticEngineVariant, engineFamilyOptions, engineResourceProfile, engineStrengthMeta, isEngineFamily, isLc0BigNetVariant, isV0DeployProfile, lc0EngineLabel, lc0VariantOptions, normalizeDeployEngineRow, stockfishEngineLabel, stockfishVariantOptions, tinyEngineLabel, tinyVariantOptions, type EngineFamily, type EngineRow } from './engineCatalog.ts';
import { engineLogoFamilyForEngineFamily, engineLogoHtml, engineLogoHtmlForName, probeEngineLogos } from './engineLogos.ts';
import { EngineResourceBroker, loadPerformanceDial, type PerformanceDial } from './resourceBroker.ts';
import { resolvePublicAssetUrl } from './assetUrls.ts';
import { hideLoadingProgress, renderLoadingProgress } from './loadingProgress.ts';

type Ground = ReturnType<typeof Chessground>;

const params = new URLSearchParams(location.search);
const DEFAULT_MODEL_URL = resolvePublicAssetUrl('/models/lc0/t1-256x10-distilled-swa-2432500.batch1.f16.qdq8.onnx');
const MODEL_URL = isV0DeployProfile() ? DEFAULT_MODEL_URL : resolvePublicAssetUrl(params.get('model') ?? DEFAULT_MODEL_URL);
const DEFAULT_PACK_URL = resolvePublicAssetUrl('/models/lc0/t1-256x10-distilled-swa-2432500.batch8.f16.lc0web/model.lc0web.json');
const PACK_URL = isV0DeployProfile() ? DEFAULT_PACK_URL : resolvePublicAssetUrl(params.get('pack') ?? params.get('modelPack') ?? DEFAULT_PACK_URL);
const DEFAULT_TINY_MODEL_URL = '/models/bt4_anneal_muon_best.onnx';
const DEFAULT_TINY_META_URL = '/models/bt4_anneal_muon_best.meta.json';
const DEFAULT_TINY_HYBRID_MANIFEST_URL = '/runtimes/squareformer-tvm-hybrid/bt4-anneal-muon-best/v1/manifest.json';
const DEFAULT_LC0_WHOLE_MODEL_MANIFEST_URL = '/runtimes/lc0-' + 'tvm' + 'js-webgpu/t1-256x10-distilled-swa-2432500/f16/v1/manifest.json';
const LC0_WHOLE_MODEL_WEBGPU_RUNTIME = 'whole-onnx-webgpu' as const;
type Lc0AnalysisRuntime = 'onnx' | 'hybrid-ort-heads' | 'hybrid-wgsl-heads' | typeof LC0_WHOLE_MODEL_WEBGPU_RUNTIME;
const REQUESTED_RECKLESS_EXPLICIT = hasExplicitRecklessVariant(params);
let REQUESTED_RECKLESS_VARIANT = recklessVariantFromParams(params);
const REQUESTED_VIRIDITHAS_EXPLICIT = hasExplicitViridithasVariant(params);
let REQUESTED_VIRIDITHAS_VARIANT = viridithasVariantFromParams(params);
const REQUESTED_BERSERK_EXPLICIT = hasExplicitBerserkVariant(params);
let REQUESTED_BERSERK_VARIANT = berserkVariantFromParams(params);
const REQUESTED_PLENTYCHESS_EXPLICIT = hasExplicitPlentyChessVariant(params);
let REQUESTED_PLENTYCHESS_VARIANT = plentyChessVariantFromParams(params);

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
const LC0_WHOLE_MODEL_MANIFEST_URL = sameOriginPathParam(['wholeModelManifest', 'wholeModelManifestUrl', 'tvm' + 'jsManifest'], DEFAULT_LC0_WHOLE_MODEL_MANIFEST_URL, ['/runtimes/lc0-' + 'tvm' + 'js-webgpu/']);

let tree = new GameTree(params.get('fen') ?? START_FEN);
let searcher: Lc0PuctSearcher | null = null;
let mainEvaluator: Lc0EvaluationProvider | null = null;
let stockfishLite: StockfishEngine | null = null;
let stockfishFull: StockfishEngine | null = null;
const tinyEvaluatorPromises = new Map<string, Promise<Evaluator>>();
const tinyEvaluators = new Set<CachedEvaluator>();
const tinyEvaluatorsByKey = new Map<string, CachedEvaluator>();
let tinyEvaluatorGeneration = 0;
let tinyHybridManifestStatus: 'unknown' | 'present' | 'missing' = 'unknown';

function initialPerformanceDial(): PerformanceDial {
  const raw = params.get('perfDial');
  if (raw === 'eco' || raw === 'balanced' || raw === 'max') return raw;
  try {
    return loadPerformanceDial(typeof localStorage !== 'undefined' ? localStorage : undefined);
  } catch {
    return 'balanced';
  }
}

// Multi-engine analysis runs all selected engines concurrently, so CPU budget
// is split with the shared policy: every selected CPU engine is registered as
// a participant before each run and threads divide deterministically (capped
// single-thread engines pass their surplus to engines that can use it).
const resourceBroker = new EngineResourceBroker({ policy: 'shared', dial: initialPerformanceDial() });
const registeredBrokerEngines = new Set<string>();

function syncBrokerParticipants(rows: EngineRow[]): void {
  const desired = new Map<string, EngineFamily>();
  for (const row of rows) {
    if (engineResourceProfile(row.family).resourceClass !== 'cpu') continue;
    desired.set(row.family === 'sf' ? `sf-${row.variant === 'full' ? 'full' : 'lite'}` : `${row.family}:${row.variant}`, row.family);
  }
  // Register the exact participant set for this run; stale entries skew shares.
  for (const known of registeredBrokerEngines) {
    if (!desired.has(known)) {
      resourceBroker.unregister(known);
      registeredBrokerEngines.delete(known);
    }
  }
  for (const [engineId, family] of desired) {
    resourceBroker.register(engineId, { ...engineResourceProfile(family) });
    registeredBrokerEngines.add(engineId);
  }
}
type AnalysisBigNetKey = Extract<BigNetKey, 'bt4' | 't3'>;
const ANALYSIS_BIG_NET_KEYS: readonly AnalysisBigNetKey[] = ['bt4', 't3'];
function bigNetFor(variant: string): { config: BigNetConfig; searcher: Bt4WorkerSearcher } {
  const key: BigNetKey = variant === 't3' ? 't3' : 'bt4';
  return { config: BIG_NETS[key], searcher: acquireBigNetSearcher(key) };
}
let ground: Ground | null = null;
let mountAbort = new AbortController();
function isStaleMount(signal: AbortSignal = mountAbort.signal): boolean {
  return signal.aborted || signal !== mountAbort.signal;
}
let analysisKeydownHandler: ((event: KeyboardEvent) => void) | null = null;
let analysisPagehideHandler: ((event: PageTransitionEvent) => void) | null = null;
let analysisAuditHandler: ((event: Event) => void) | null = null;
let orientation: 'white' | 'black' = 'white';
let analysisAbort: AbortController | null = null;
let reviewAbort: AbortController | null = null;
let lastReview: GameReview | null = null;
let lastReviewNodes: GameNode[] = [];
let lastReviewSignature = '';
let analyzing = false;
const lineCache = new Map<string, AnalysisLine[]>();
const engineLineCache = new Map<string, AnalysisLine[]>();
const completeAnalysisKeys = new Set<string>();
interface SearchProgressSnapshot {
  label: string;
  completed?: number;
  requested?: number;
  elapsedMs?: number;
  nps?: number;
  best?: string | null;
  value?: number;
  units: 'visits' | 'nodes' | 'search';
  indeterminate?: boolean;
}
const searchProgressByEngine = new Map<string, SearchProgressSnapshot>();
let activeAnalysisRunId = 0;
const nodeIndex = new Map<number, GameNode>();
const ENGINE_PROFILE_STORAGE_KEY = 'lc0-analysis-engine-profiles-v1';
const LAST_ENGINE_PROFILE_STORAGE_KEY = 'lc0-analysis-last-engine-profile-v1';
const ENGINE_PROFILE_BACKUP_KIND = 'lc0-analysis-engine-profile-backup';
const BUILT_IN_PROFILE_VALUE_PREFIX = 'builtin:';

interface EngineAnalysisProfile {
  name: string;
  rows: EngineRow[];
  multiPv: number;
  lc0Runtime: Lc0AnalysisRuntime;
}
interface BuiltInEngineAnalysisProfile extends EngineAnalysisProfile { id: string; note: string }
const BUILT_IN_ENGINE_PROFILES: BuiltInEngineAnalysisProfile[] = [
  { id: 'lc0-stockfish', name: 'Built-in · Lc0 + Stockfish', note: 'Small LC0 plus Stockfish Lite baseline for quick agreement checks.', rows: [{ family: 'lc0', variant: 'small', strength: 400 }, { family: 'sf', variant: 'lite', strength: 14 }], multiPv: 3, lc0Runtime: 'onnx' },
  { id: 'browser-native-survey', name: 'Built-in · Browser-native survey', note: 'Small LC0 plus staged browser-native UCI engines; missing assets stay visible in runtime status.', rows: [{ family: 'lc0', variant: 'small', strength: 300 }, { family: 'viridithas', variant: 'simd', strength: 8 }, { family: 'berserk', variant: 'emscripten', strength: 8 }, { family: 'plentychess', variant: 'emscripten', strength: 8 }], multiPv: 2, lc0Runtime: 'onnx' },
  { id: 'lc0-wgsl-heads', name: 'Built-in · LC0 WGSL heads probe', note: 'Opt-in LC0 hybrid WGSL-head analysis lane; stable defaults remain unchanged.', rows: [{ family: 'lc0', variant: 'small', strength: 400 }], multiPv: 3, lc0Runtime: 'hybrid-wgsl-heads' },
  { id: 'lc0-ladder', name: 'Built-in · Lc0 ladder (small/t3/BT4)', note: 'All three Lc0 nets side by side for strength/latency comparison; t3 and BT4 lazy-load in WebGPU workers with tree reuse.', rows: [{ family: 'lc0', variant: 'small', strength: 400 }, { family: 'lc0', variant: 't3', strength: 400 }, { family: 'lc0', variant: 'bt4', strength: 400 }], multiPv: 2, lc0Runtime: 'onnx' },
];
let engineProfiles: EngineAnalysisProfile[] = [];
let importedGames: ImportedGame[] = [];
let importedPositionIndex: OpeningPositionIndex | null = null;
let databasePositionStats: OpeningMoveStat[] = [];
let databasePositionKey = '';
let databasePositionCollectionCount = 0;
let pgnDatabaseSearchKey = '';
let pgnCollections: PgnCollectionSummary[] = [];
let activePgnCollectionId = '';
let lastImportSource: PgnCollectionSource = 'manual';
let lastImportUsername = '';
let lastImportColor = '';
const bookCache = new Map<string, OpeningMoveStat[]>();
// Distinct brush for the opening-book most-played move (not LC0 green / SF blue).
const BOOK_BRUSH = 'yellow';
const BOOK_SWATCH = '#e68f00';

function currentBookStats(): OpeningMoveStat[] {
  const fen = tree.current.fen;
  const key = positionKey(fen);
  if (databasePositionKey === key && databasePositionStats.length) return databasePositionStats;
  if (!importedGames.length) return [];
  let stats = bookCache.get(key);
  if (!stats) {
    stats = importedPositionIndex ? openingStatsFromIndex(importedPositionIndex, fen) : openingStatsForPosition(importedGames, fen);
    bookCache.set(key, stats);
  }
  return stats;
}

// LC0 analysis runs in a dedicated search worker so navigation never blocks the
// UI; a new position cancels the in-flight worker search by id.
let searchWorker: Worker | null = null;
let workerReady = false;
let workerBackend = '';
let workerSeq = 0;
const workerPending = new Map<number, { resolve: (value: unknown) => void; reject: (error: Error) => void; onProgress?: (progress: WorkerSearchProgress) => void }>();
let activeWorkerSearchId: number | null = null;

interface WorkerSearchResult {
  move?: string | null;
  value: number;
  visits: number;
  pv: string[];
  multiPv?: string[][];
  children: { uci: string; visits: number; q: number }[];
  elapsedMs?: number;
  cancelled?: boolean;
}

interface WorkerSearchProgress extends WorkerSearchResult {
  requestedVisits: number;
  completedVisits: number;
}

function requestedEp(): string {
  const raw = (params.get('ep') ?? 'auto').toLowerCase();
  if (raw === 'webgpu' || raw === 'gpu') return 'webgpu';
  if (raw === 'wasm') return 'wasm';
  if (raw === 'webgpu,wasm' || raw === 'gpu,wasm') return 'webgpu,wasm';
  return 'auto';
}

function normalizeLc0Runtime(value: string | null): Lc0AnalysisRuntime {
  if (isV0DeployProfile()) return 'onnx';
  const raw = (value ?? '').toLowerCase();
  if (raw === LC0_WHOLE_MODEL_WEBGPU_RUNTIME || raw === 'tvm' + 'js-webgpu' || raw === 'lc0-' + 'tvm' + 'js-webgpu') return LC0_WHOLE_MODEL_WEBGPU_RUNTIME;
  if (raw === 'hybrid' || raw === 'lc0web' || raw === 'hybrid-ort-heads' || raw === 'wgsl-encoder') return 'hybrid-ort-heads';
  if (raw === 'hybrid-wgsl-heads' || raw === 'wgsl-heads' || raw === 'wgsl') return 'hybrid-wgsl-heads';
  return 'onnx';
}

function initialLc0Runtime(): Lc0AnalysisRuntime {
  if (isV0DeployProfile()) return 'onnx';
  if (params.get('headBackend') === 'wgsl' || params.get('hybridHeads') === 'wgsl') return 'hybrid-wgsl-heads';
  return normalizeLc0Runtime(params.get('lc0Runtime') ?? params.get('runtime'));
}

function selectedLc0Runtime(): Lc0AnalysisRuntime {
  return normalizeLc0Runtime(selectEl('lc0RuntimeSelect').value);
}

function lc0WholeModelRuntimeRequested(): boolean {
  if (isV0DeployProfile()) return false;
  return normalizeLc0Runtime(params.get('lc0Runtime') ?? params.get('runtime')) === LC0_WHOLE_MODEL_WEBGPU_RUNTIME
    || params.get('enableWholeModelWebgpu') === '1'
    || params.get('enableTvm' + 'js') === '1';
}

function installExperimentalLc0RuntimeOption(): void {
  // Promoted 2026-06-10 (release-owner decision): the whole-model WebGPU
  // runtime is always listed. ORT remains the default and the fallback.
  const select = selectEl('lc0RuntimeSelect');
  if ([...select.options].some((option) => option.value === LC0_WHOLE_MODEL_WEBGPU_RUNTIME)) return;
  const option = document.createElement('option');
  option.value = LC0_WHOLE_MODEL_WEBGPU_RUNTIME;
  option.textContent = 'TVM whole-model WebGPU (fast, small net)';
  select.appendChild(option);
}

function lc0ResolvedRuntime(runtime: Lc0AnalysisRuntime): string {
  if (runtime === 'onnx') return 'ort-worker';
  if (runtime === LC0_WHOLE_MODEL_WEBGPU_RUNTIME) return 'whole-onnx-webgpu-worker-research';
  return `${runtime}-lazy`;
}

function installRuntimeAuditPanel(): void {
  if (analysisAuditHandler) window.removeEventListener(BROWSER_RUNTIME_AUDIT_EVENT, analysisAuditHandler);
  analysisAuditHandler = (event: Event) => {
    const detail = (event as CustomEvent<BrowserRuntimeAuditDetail>).detail;
    if (detail.family !== 'lc0') return;
    const target = document.getElementById('runtimeAudit');
    if (!target) return;
    target.textContent = formatBrowserRuntimeAudit(detail);
  };
  window.addEventListener(BROWSER_RUNTIME_AUDIT_EVENT, analysisAuditHandler);
}

function lc0RuntimeLabel(runtime = selectedLc0Runtime()): string {
  if (runtime === LC0_WHOLE_MODEL_WEBGPU_RUNTIME) return 'TVM whole-model WebGPU (research)';
  if (runtime === 'hybrid-wgsl-heads') return 'WGSL encoder + WGSL heads';
  if (runtime === 'hybrid-ort-heads') return 'WGSL encoder + ORT heads';
  return 'ORT ONNX';
}

function lc0EncoderLayers(): number {
  return Math.min(32, Math.max(1, Math.floor(Number(params.get('encoderLayers') ?? params.get('layers') ?? '10') || 10)));
}

function lc0WholeModelPhysicalBatch(): number {
  const parsed = Math.floor(Number(params.get('wholeModelBatch') ?? params.get('tvmBatch') ?? params.get('compiledBatch') ?? '8'));
  return Number.isFinite(parsed) ? Math.max(1, Math.min(64, parsed)) : 8;
}

function lc0WholeModelTensorCache(): boolean {
  return params.get('wholeModelTensorCache') === '1' || params.get('tensorCache') === '1';
}

function lc0InitMessage(runtime = selectedLc0Runtime()): Record<string, unknown> {
  const common = { type: 'init', modelUrl: MODEL_URL, ep: requestedEp(), cacheModel: false, reportDownloadProgress: MODEL_URL === DEFAULT_MODEL_URL };
  if (runtime === 'onnx') return common;
  if (runtime === LC0_WHOLE_MODEL_WEBGPU_RUNTIME) return {
    ...common,
    runtime: LC0_WHOLE_MODEL_WEBGPU_RUNTIME,
    wholeModelManifestUrl: LC0_WHOLE_MODEL_MANIFEST_URL,
    wholeModelBatch: lc0WholeModelPhysicalBatch(),
    wholeModelTensorCache: lc0WholeModelTensorCache(),
    evalCacheEntries: 0,
  };
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

function postWorker<T>(message: Record<string, unknown>, onId?: (id: number) => void, onProgress?: (progress: WorkerSearchProgress) => void): Promise<T> {
  if (!searchWorker) return Promise.reject(new Error('LC0 worker unavailable'));
  const id = ++workerSeq;
  onId?.(id);
  return new Promise<T>((resolve, reject) => {
    workerPending.set(id, { resolve: resolve as (value: unknown) => void, reject, onProgress });
    searchWorker!.postMessage({ ...message, id });
  });
}

async function initWorker(): Promise<string> {
  if (searchWorker && workerReady) return workerBackend;
  if (!searchWorker) searchWorker = new Worker(new URL('./searchWorker.ts', import.meta.url), { type: 'module' });
  searchWorker.addEventListener('message', (event: MessageEvent) => {
    const message = event.data as { id: number; type: string; error?: string; loadedBytes?: number; totalBytes?: number };
    if (message.type === 'downloadProgress') {
      showModelProgress('Lc0 small net', message.loadedBytes ?? 0, message.totalBytes, 'Downloading');
      return;
    }
    const pending = workerPending.get(message.id);
    if (!pending) return;
    if (message.type === 'searchProgress') {
      pending.onProgress?.((message as unknown as { progress: WorkerSearchProgress }).progress);
      return;
    }
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
  const runtime = selectedLc0Runtime();
  publishBrowserRuntimeAudit({
    source: 'lc0-analysis-worker',
    surface: 'analysis',
    family: 'lc0',
    engineLabel: 'LC0',
    modelId: 'lc0-default',
    modelUrl: runtime === LC0_WHOLE_MODEL_WEBGPU_RUNTIME ? LC0_WHOLE_MODEL_MANIFEST_URL : MODEL_URL,
    requestedRuntime: runtime,
    resolvedRuntime: lc0ResolvedRuntime(runtime),
    runtimeConfigId: runtime === 'onnx' ? undefined : runtime,
    manifestUrl: runtime === 'onnx' ? undefined : runtime === LC0_WHOLE_MODEL_WEBGPU_RUNTIME ? LC0_WHOLE_MODEL_MANIFEST_URL : PACK_URL,
    searchBudget: `multipv=${multiPv()}`,
    notes: runtime === 'onnx' ? [ready.backend] : runtime === LC0_WHOLE_MODEL_WEBGPU_RUNTIME ? [ready.backend, 'whole-model runtime is research-only and opt-in'] : [ready.backend, 'hybrid runtime is pack-lazy until first evaluation succeeds'],
  });
  return workerBackend;
}

function analysisProgressText(label: string, progress: { completedVisits?: number; requestedVisits?: number; visits: number; move?: string | null; value: number; elapsedMs?: number }): string {
  const completed = progress.completedVisits ?? progress.visits;
  const requested = progress.requestedVisits ?? progress.visits;
  const pct = requested > 0 ? ` ${(100 * completed / requested).toFixed(0)}%` : '';
  const speed = progress.elapsedMs && progress.elapsedMs > 0
    ? ` · ${(completed / Math.max(1e-9, progress.elapsedMs / 1000)).toFixed(1)} v/s`
    : '';
  return `${label}: ${completed}/${requested} visits${pct} · best ${progress.move ?? '—'} · Q ${progress.value.toFixed(3)}${speed}`;
}

function searchProgressText(progress: SearchProgressSnapshot): string {
  if (progress.indeterminate) return `${progress.label}: searching…`;
  const completed = progress.completed ?? 0;
  const requested = progress.requested ?? completed;
  const pct = requested > 0 ? ` ${(100 * completed / requested).toFixed(0)}%` : '';
  const elapsed = progress.elapsedMs && progress.elapsedMs > 0 ? ` · ${(progress.elapsedMs / 1000).toFixed(1)}s` : '';
  const nps = progress.nps && progress.nps > 0 ? ` · ${progress.nps.toFixed(1)} ${progress.units}/s` : '';
  const best = progress.best ? ` · best ${progress.best}` : '';
  const value = progress.value !== undefined ? ` · Q ${progress.value.toFixed(3)}` : '';
  return `${progress.label}: ${completed}/${requested} ${progress.units}${pct}${elapsed}${nps}${best}${value}`;
}

function searchProgressHtml(progress: SearchProgressSnapshot): string {
  const value = progress.indeterminate || progress.completed === undefined ? '' : ` value="${Math.max(0, Math.floor(progress.completed))}"`;
  const max = progress.indeterminate || progress.requested === undefined || progress.requested <= 0 ? '' : ` max="${Math.max(1, Math.floor(progress.requested))}"`;
  return `<div class="search-progress-row"><progress${value}${max}></progress><div class="search-progress-text">${htmlEscape(searchProgressText(progress))}</div></div>`;
}

function renderAnalysisSearchProgress(): void {
  const node = document.getElementById('analysisSearchProgress');
  if (!node) return;
  const items = [...searchProgressByEngine.values()];
  node.hidden = items.length === 0;
  node.innerHTML = items.map(searchProgressHtml).join('');
}

function clearAnalysisSearchProgress(): void {
  searchProgressByEngine.clear();
  renderAnalysisSearchProgress();
}

function progressKey(runId: number, label: string): string {
  return `${runId}\u0000${label}`;
}

function showAnalysisProgress(runId: number, fen: string, label: string, progress: WorkerSearchProgress): void {
  if (runId !== activeAnalysisRunId || mountAbort.signal.aborted || !analyzing || tree.current.fen !== fen) return;
  const completed = progress.completedVisits ?? progress.visits;
  const elapsedMs = progress.elapsedMs ?? 0;
  searchProgressByEngine.set(progressKey(runId, label), {
    label,
    completed,
    requested: progress.requestedVisits ?? progress.visits,
    elapsedMs,
    nps: elapsedMs > 0 ? completed / Math.max(1e-9, elapsedMs / 1000) : undefined,
    best: progress.move ?? null,
    value: progress.value,
    units: 'visits',
  });
  renderAnalysisSearchProgress();
  renderEngineComparison(lineCache.get(fen) ?? []);
  el('message').textContent = analysisProgressText(label, progress);
}

function showMoveSearchProgress(runId: number, fen: string, label: string, progress: { completedVisits?: number; requestedVisits?: number; visits: number; move?: Move | null; value: number; elapsedMs?: number }): void {
  showAnalysisProgress(runId, fen, label, {
    visits: progress.visits,
    requestedVisits: progress.requestedVisits ?? progress.visits,
    completedVisits: progress.completedVisits ?? progress.visits,
    value: progress.value,
    elapsedMs: progress.elapsedMs,
    move: progress.move ? moveToUci(progress.move) : null,
    children: [],
    pv: [],
  });
}

function showIndeterminateSearchProgress(runId: number, fen: string, label: string): void {
  if (runId !== activeAnalysisRunId || mountAbort.signal.aborted || !analyzing || tree.current.fen !== fen) return;
  searchProgressByEngine.set(progressKey(runId, label), { label, units: 'search', indeterminate: true });
  renderAnalysisSearchProgress();
  renderEngineComparison(lineCache.get(fen) ?? []);
}

function clearEngineSearchProgress(runId: number, label: string): void {
  searchProgressByEngine.delete(progressKey(runId, label));
  renderAnalysisSearchProgress();
}

function searchProgressForLabel(label: string): SearchProgressSnapshot | undefined {
  return searchProgressByEngine.get(progressKey(activeAnalysisRunId, label));
}

async function workerLc0Lines(runId: number, fen: string, visits: number, label = 'Lc0'): Promise<AnalysisLine[]> {
  const response = await postWorker<{ result: WorkerSearchResult }>(
    { type: 'search', input: { positions: tree.historyBoards() }, visits, batchSize: 1, multiPv: multiPv(), reportProgress: true },
    (id) => { activeWorkerSearchId = id; },
    (progress) => showAnalysisProgress(runId, fen, label, progress),
  );
  return response.result.cancelled ? [] : lc0AnalysisLines(response.result, fen, 'Lc0');
}

function el(id: string): HTMLElement {
  const node = document.getElementById(id);
  if (!node) throw new Error(`Missing #${id}`);
  return node;
}
function maybeEl(id: string): HTMLElement | null { return document.getElementById(id); }
function inputEl(id: string): HTMLInputElement { return el(id) as HTMLInputElement; }
function selectEl(id: string): HTMLSelectElement { return el(id) as HTMLSelectElement; }
function htmlEscape(value: unknown): string {
  return String(value).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]!));
}

function modelProgressEl(): HTMLElement {
  let node = document.getElementById('downloadProgress');
  if (node) return node;
  node = document.getElementById('modelLoadProgress');
  if (!node) {
    node = document.createElement('div');
    node.id = 'modelLoadProgress';
    node.className = 'model-load-progress';
    node.hidden = true;
    el('message').insertAdjacentElement('afterend', node);
  }
  return node;
}

function showModelProgress(label: string, loadedBytes?: number, totalBytes?: number, phase = 'Loading'): void {
  renderLoadingProgress(modelProgressEl(), { label, loadedBytes, totalBytes, phase });
}

function hideModelProgress(): void {
  hideLoadingProgress(modelProgressEl());
}
function storageGet(key: string): string | null {
  try { return localStorage.getItem(key); } catch { return null; }
}
function storageSet(key: string, value: string): void {
  try { localStorage.setItem(key, value); } catch { /* profiles are optional */ }
}
function storageRemove(key: string): void {
  try { localStorage.removeItem(key); } catch { /* profiles are optional */ }
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
    }, {
      runtime,
      manifestUrl: TINY_HYBRID_MANIFEST_URL,
      fallback,
      audit: { surface: 'analysis', searchBudget: `multipv=${multiPv()}` },
    });
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

function clampStrengthForRow(row: EngineRow): number {
  const meta = strengthMeta(row.family);
  return Math.max(meta.min, Math.min(meta.max, Math.floor(Number(row.strength) || meta.def)));
}
function sanitizeEngineRow(value: unknown, index = 0): EngineRow | null {
  if (!value || typeof value !== 'object') return null;
  const raw = value as Partial<EngineRow>;
  if (!isEngineFamily(String(raw.family))) return null;
  const family = raw.family as EngineFamily;
  if (raw.variant === 'custom') return null;
  const row = { family, variant: String(raw.variant || defaultVariant(family)), strength: Number(raw.strength) || defaultStrength(family) };
  return normalizeDeployEngineRow({ ...row, strength: clampStrengthForRow(row) }, 'analysis', index);
}
function currentEngineProfile(name: string): EngineAnalysisProfile {
  return {
    name,
    rows: activeEngineRows().map((row) => ({ ...row })),
    multiPv: multiPv(),
    lc0Runtime: selectedLc0Runtime(),
  };
}
function profileHasBt4(rows: EngineRow[]): boolean {
  return rows.some((row) => row.family === 'lc0' && isLc0BigNetVariant(row.variant));
}
function bigNetSelectableSync(config: BigNetConfig): boolean {
  // WebGPU no longer gates selection: 'auto' falls back to wasm-CPU (slow).
  return bigNetAssetStatusSync(config) === 'present';
}
function bigNetUnavailableText(config: BigNetConfig): string {
  if (bigNetAssetStatusSync(config) === 'missing') return `Lc0 ${config.name} model asset is missing at ${config.modelUrl}. Run node scripts/lc0_prepare_model_assets.mjs in this package.`;
  if (bigNetAssetStatusSync(config) === 'unknown') return `Lc0 ${config.name} model asset is still being checked.`;
  return '';
}
function profileRowsForUse(rows: EngineRow[], allowBt4Prompt: boolean): EngineRow[] {
  return rows.map((row, index) => {
    const next = normalizeDeployEngineRow({ ...row, strength: clampStrengthForRow(row) }, 'analysis', index);
    if (next.family === 'lc0' && isLc0BigNetVariant(next.variant)) {
      const { config } = bigNetFor(next.variant);
      const allowed = bigNetSelectableSync(config) && allowBt4Prompt && window.confirm(`${bigNetLoadWarning(config)}\n\nLoad the saved Lc0 ${config.name} profile row?`);
      if (!allowed) next.variant = 'small';
    }
    return next;
  });
}
function normalizeEngineProfiles(parsed: unknown): EngineAnalysisProfile[] {
  const entries = Array.isArray(parsed)
    ? parsed
    : (parsed && typeof parsed === 'object' && (parsed as { kind?: unknown }).kind === ENGINE_PROFILE_BACKUP_KIND && Array.isArray((parsed as { profiles?: unknown }).profiles) ? (parsed as { profiles: unknown[] }).profiles : []);
  return entries.flatMap((entry) => {
    if (!entry || typeof entry !== 'object') return [];
    const raw = entry as Partial<EngineAnalysisProfile>;
    const name = String(raw.name ?? '').trim();
    if (!name || name.startsWith(BUILT_IN_PROFILE_VALUE_PREFIX) || !Array.isArray(raw.rows)) return [];
    const rows = raw.rows.map((row, index) => sanitizeEngineRow(row, index)).filter((row): row is EngineRow => !!row);
    if (!rows.length) return [];
    return [{ name, rows, multiPv: Math.max(1, Math.min(10, Math.floor(Number(raw.multiPv) || 3))), lc0Runtime: normalizeLc0Runtime(raw.lc0Runtime ?? 'onnx') }];
  }).sort((a, b) => a.name.localeCompare(b.name));
}
function loadEngineProfiles(): EngineAnalysisProfile[] {
  try { return normalizeEngineProfiles(JSON.parse(storageGet(ENGINE_PROFILE_STORAGE_KEY) ?? '[]') as unknown); }
  catch { return []; }
}
function persistEngineProfiles(): void {
  storageSet(ENGINE_PROFILE_STORAGE_KEY, JSON.stringify(engineProfiles));
}
function profileSummary(profile: EngineAnalysisProfile, note = ''): string {
  const rows = profile.rows.map((row) => `${rowLabel(row)} ${clampStrengthForRow(row)} ${strengthMeta(row.family).unit}`).join(' · ');
  return `${note ? `${note} ` : ''}${rows || 'no engines'} · MultiPV ${profile.multiPv} · LC0 ${lc0RuntimeLabel(profile.lc0Runtime)}`;
}
function builtInProfileValue(id: string): string { return `${BUILT_IN_PROFILE_VALUE_PREFIX}${id}`; }
function renderEngineProfiles(selected = storageGet(LAST_ENGINE_PROFILE_STORAGE_KEY) ?? ''): void {
  const builtInOptions = BUILT_IN_ENGINE_PROFILES.map((profile) => `<option value="${htmlEscape(builtInProfileValue(profile.id))}"${builtInProfileValue(profile.id) === selected ? ' selected' : ''}>${htmlEscape(profile.name)}</option>`);
  const savedOptions = engineProfiles.map((profile) => `<option value="${htmlEscape(profile.name)}"${profile.name === selected ? ' selected' : ''}>${htmlEscape(profile.name)}</option>`);
  selectEl('engineProfileSelect').innerHTML = ['<option value="">manual / default</option>', '<optgroup label="Built-in profiles">', ...builtInOptions, '</optgroup>', '<optgroup label="Saved profiles">', ...savedOptions, '</optgroup>'].join('');
  const builtIn = BUILT_IN_ENGINE_PROFILES.find((profile) => builtInProfileValue(profile.id) === selected);
  const saved = engineProfiles.find((profile) => profile.name === selected);
  const profile = builtIn ?? saved;
  inputEl('engineProfileName').value = saved?.name ?? '';
  el('engineProfileSummary').textContent = profile ? profileSummary(profile, builtIn?.note) : 'Manual engine setup. Choose a built-in profile or save the current engine rows.';
}
function applyEngineProfile(profile: EngineAnalysisProfile, options: { selected?: string; persistLast?: boolean; note?: string } = {}): void {
  const runtimeChanged = selectedLc0Runtime() !== profile.lc0Runtime;
  engineRows = profileRowsForUse(profile.rows, true);
  inputEl('multiPvInput').value = String(profile.multiPv);
  selectEl('lc0RuntimeSelect').value = profile.lc0Runtime;
  if (options.persistLast !== false) storageSet(LAST_ENGINE_PROFILE_STORAGE_KEY, profile.name);
  renderEngineList();
  renderEngineProfiles(options.selected ?? profile.name);
  el('engineProfileSummary').textContent = profileSummary(profile, options.note);
  disposeUnusedEngines();
  lineCache.clear();
  completeAnalysisKeys.clear();
  if (runtimeChanged) void reloadLc0Backend(true);
  else void analyzeCurrent();
}
function saveCurrentEngineProfile(): void {
  const name = inputEl('engineProfileName').value.trim();
  if (!name) { el('message').textContent = 'Name the engine profile before saving.'; return; }
  if (name.startsWith(BUILT_IN_PROFILE_VALUE_PREFIX)) { el('message').textContent = `Saved profile names cannot start with “${BUILT_IN_PROFILE_VALUE_PREFIX}”.`; return; }
  if (activeEngineRows().some((row) => row.variant === 'custom')) {
    el('message').textContent = 'Custom URL variants are not saved in profiles yet; choose a built-in variant first.';
    return;
  }
  const profile = currentEngineProfile(name);
  engineProfiles = [...engineProfiles.filter((entry) => entry.name !== name), profile].sort((a, b) => a.name.localeCompare(b.name));
  persistEngineProfiles();
  storageSet(LAST_ENGINE_PROFILE_STORAGE_KEY, name);
  renderEngineProfiles(name);
  el('message').textContent = `Saved engine profile “${name}”.`;
}
function exportEngineProfiles(): void {
  const backup = { kind: ENGINE_PROFILE_BACKUP_KIND, version: 1, exportedAt: new Date().toISOString(), profiles: engineProfiles };
  downloadTextFile('lc0-analysis-engine-profiles.json', JSON.stringify(backup, null, 2), 'application/json');
  el('message').textContent = `Exported ${engineProfiles.length} saved engine profiles. Custom URL variants are excluded by design.`;
}
async function importEngineProfilesFile(file: File | undefined): Promise<void> {
  if (!file) return;
  try {
    const incoming = normalizeEngineProfiles(JSON.parse(await file.text()) as unknown);
    if (!incoming.length) throw new Error('No valid engine profiles found');
    engineProfiles = [...incoming, ...engineProfiles.filter((profile) => !incoming.some((entry) => entry.name === profile.name))].sort((a, b) => a.name.localeCompare(b.name));
    persistEngineProfiles();
    applyEngineProfile(incoming[0], { selected: incoming[0].name });
    el('message').textContent = `Imported ${incoming.length} engine profiles and applied “${incoming[0].name}”. Custom URL variants were ignored.`;
  } catch (error) {
    el('message').textContent = `Engine profile import failed: ${(error as Error).message}`;
  } finally {
    inputEl('importEngineProfilesFile').value = '';
  }
}
function deleteSelectedEngineProfile(): void {
  const selected = selectEl('engineProfileSelect').value;
  if (selected.startsWith(BUILT_IN_PROFILE_VALUE_PREFIX)) {
    el('message').textContent = 'Built-in engine profiles cannot be deleted.';
    return;
  }
  const name = selected || inputEl('engineProfileName').value.trim();
  if (!name) return;
  if (!engineProfiles.some((entry) => entry.name === name)) {
    el('message').textContent = `No saved engine profile named “${name}”.`;
    return;
  }
  engineProfiles = engineProfiles.filter((entry) => entry.name !== name);
  persistEngineProfiles();
  storageRemove(LAST_ENGINE_PROFILE_STORAGE_KEY);
  renderEngineProfiles('');
  el('message').textContent = `Deleted engine profile “${name}”.`;
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
  const variants = [...RECKLESS_VARIANTS];
  if (!variants.some((variant) => variant.key === REQUESTED_RECKLESS_VARIANT.key)) variants.push(REQUESTED_RECKLESS_VARIANT);
  return variants;
}

function recklessVariantForKey(variantKey: string): RecklessVariant {
  const key = normalizeRecklessVariant(variantKey);
  if (key === 'custom' && REQUESTED_RECKLESS_VARIANT.key === 'custom') return REQUESTED_RECKLESS_VARIANT;
  return recklessVariantByKey(key);
}

function availableViridithasVariants(): ViridithasVariant[] {
  return REQUESTED_VIRIDITHAS_VARIANT.key === 'custom' ? [...VIRIDITHAS_VARIANTS, REQUESTED_VIRIDITHAS_VARIANT] : [...VIRIDITHAS_VARIANTS];
}

function viridithasVariantForKey(variantKey: string): ViridithasVariant {
  const key = normalizeViridithasVariant(variantKey);
  if (key === 'custom' && REQUESTED_VIRIDITHAS_VARIANT.key === 'custom') return REQUESTED_VIRIDITHAS_VARIANT;
  return viridithasVariantByKey(key);
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


function availablePlentyChessVariants(): PlentyChessVariant[] {
  if (REQUESTED_PLENTYCHESS_VARIANT.key === 'custom') return [...PLENTYCHESS_VARIANTS, REQUESTED_PLENTYCHESS_VARIANT];
  return [...PLENTYCHESS_VARIANTS];
}

function plentyChessVariantForKey(variantKey: string): PlentyChessVariant {
  const key = normalizePlentyChessVariant(variantKey);
  if (key === 'custom' && REQUESTED_PLENTYCHESS_VARIANT.key === 'custom') return REQUESTED_PLENTYCHESS_VARIANT;
  return plentyChessVariantByKey(key);
}


// "Add engine" fills the next missing family by priority
// (Lc0 → SF → Reckless → Viridithas → Berserk → PlentyChess → Tiny Leela),
// falling back to the top priority when all families are present.
function nextEngineFamily(): EngineFamily {
  const present = new Set(engineRows.map((row) => row.family));
  const priority = analysisEngineFamilyOptions().map((option) => option.value);
  return priority.find((family) => !present.has(family)) ?? priority[0] ?? ENGINE_FAMILY_PRIORITY[0];
}

let engineRows: EngineRow[] = [{ family: 'lc0', variant: 'small', strength: 400 }];

function analysisEngineFamilyOptions(): { value: EngineFamily; label: string }[] {
  return engineFamilyOptions();
}

function variantOptions(family: EngineFamily): { value: string; label: string; disabled?: boolean }[] {
  if (isV0DeployProfile() && !['lc0', 'sf', 'reckless', 'berserk', 'viridithas', 'plentychess'].includes(family)) return [];
  if (family === 'tiny') return tinyVariantOptions().map((option) => option.value === 'bt4-custom' && tinyHybridManifestStatus === 'missing'
    ? { ...option, disabled: true, label: `${option.label} (bundle missing)` }
    : option);
  if (family === 'lc0') return lc0VariantOptions(true).map((option) => {
    if (!isLc0BigNetVariant(option.value)) return option;
    const config = BIG_NETS[option.value];
    if (bigNetAssetStatusSync(config) === 'unknown') void checkBigNetAsset(config, renderEngineList);
    const state = bigNetOptionState(config);
    return { ...option, label: `${option.label}${state.suffix}`, disabled: option.disabled || state.disabled };
  });
  if (family === 'sf') return stockfishVariantOptions();
  if (family === 'viridithas') return availableViridithasVariants().map((v) => {
    const status = viridithasVariantAssetStatus(v);
    const unsupported = v.key === 'relaxed-simd' && !supportsWasmRelaxedSimd();
    if (!unsupported && v.key === 'relaxed-simd' && status === 'unknown') void checkViridithasVariantAsset(v, renderEngineList);
    const disabled = unsupported || (v.key === 'relaxed-simd' && status !== 'ok') || status === 'missing';
    const suffix = unsupported ? ' (unsupported by this browser)' : status === 'missing' ? ' (asset missing)' : v.key === 'relaxed-simd' && status !== 'ok' ? ' (checking asset)' : '';
    return { value: v.key, label: `${v.label}${suffix}`, disabled };
  });
  if (family === 'berserk') return availableBerserkVariants().map((v) => {
    const status = berserkVariantAssetStatus(v);
    const unsupported = v.key === 'emscripten-relaxed' && !supportsWasmRelaxedSimd();
    if (!unsupported && status === 'unknown') void checkBerserkVariantAsset(v, renderEngineList);
    const needsGeneratedAsset = v.key === 'emscripten-simd' || v.key === 'emscripten-relaxed';
    const disabled = unsupported || (needsGeneratedAsset && status !== 'present') || status === 'missing';
    const suffix = unsupported ? ' (unsupported by this browser)' : status === 'missing' ? ' (asset missing)' : needsGeneratedAsset && status !== 'present' ? ' (checking asset)' : '';
    return { value: v.key, label: `${v.label}${suffix}`, disabled };
  });
  if (family === 'plentychess') return availablePlentyChessVariants().map((v) => {
    const status = plentyChessVariantAssetStatus(v);
    const unsupportedReason = plentyChessVariantUnsupportedReason(v);
    const needsGeneratedAsset = v.key === 'emscripten-sse41' || v.key === 'emscripten-relaxed';
    if (!unsupportedReason && needsGeneratedAsset && status === 'unknown') void checkPlentyChessVariantAsset(v, renderEngineList);
    const disabled = Boolean(unsupportedReason) || (needsGeneratedAsset && status !== 'present') || status === 'missing';
    const suffix = unsupportedReason ? ` (${unsupportedReason})` : status === 'missing' ? ' (asset missing)' : needsGeneratedAsset && status !== 'present' ? ' (checking asset)' : '';
    return { value: v.key, label: `${v.label}${suffix}`, disabled };
  });
  const recklessVariants = availableRecklessVariants().filter((v) => !isV0DeployProfile() || ['full', 'simd', 'relaxed-simd'].includes(v.key));
  return recklessVariants.map((v) => {
    const status = recklessVariantAssetStatus(v);
    const unsupported = v.key === 'relaxed-simd' && !supportsWasmRelaxedSimd();
    const suffix = unsupported ? ' (unsupported by this browser)' : status === 'missing' ? ' (asset missing)' : '';
    return { value: v.key, label: `${v.label}${suffix}`, disabled: unsupported || status === 'missing' };
  });
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

function engineAnalysisCacheKey(fen: string, row: EngineRow): string {
  const runtime = row.family === 'lc0' && !isLc0BigNetVariant(row.variant) ? selectedLc0Runtime() : '';
  const history = tree.historyBoards().map(boardToFen).join('|');
  return [fen, history, row.family, row.variant, row.strength, `multipv=${multiPv()}`, runtime].join('\u0000');
}

function analysisSelectionCacheKey(fen: string, rows = activeEngineRows()): string {
  return rows.map((row) => engineAnalysisCacheKey(fen, row)).join('\u0001');
}

function activeEngineRows(): EngineRow[] {
  const seen = new Set<string>();
  return engineRows.map((row, index) => normalizeDeployEngineRow(row, 'analysis', index)).filter((r) => {
    const k = `${r.family}:${r.variant}`;
    if (seen.has(k)) return false;
    seen.add(k);
    return true;
  });
}

function usesBigNetRow(variant: 'bt4' | 't3'): boolean {
  return engineRows.some((r) => r.family === 'lc0' && r.variant === variant);
}

function renderEngineList(): void {
  engineRows = engineRows.map((row, index) => normalizeDeployEngineRow(row, 'analysis', index));
  const families = analysisEngineFamilyOptions();
  el('engineList').innerHTML = engineRows.map((row, i) => {
    const famSel = families.map(({ value, label }) => `<option value="${value}"${row.family === value ? ' selected' : ''}>${label}</option>`).join('');
    const varSel = variantOptions(row.family).map((o) => `<option value="${o.value}"${row.variant === o.value ? ' selected' : ''}${o.disabled ? ' disabled' : ''}>${htmlEscape(o.label)}</option>`).join('');
    const meta = strengthMeta(row.family);
    const remove = engineRows.length > 1 ? `<button class="row-rm" data-i="${i}" type="button" title="Remove engine">×</button>` : '';
    return `<div class="engine-row">${engineLogoHtml(engineLogoFamilyForEngineFamily(row.family))}<select class="row-fam" data-i="${i}">${famSel}</select><span class="arrow">→</span><select class="row-var" data-i="${i}">${varSel}</select><span class="arrow">→</span><input class="row-strength" data-i="${i}" type="number" min="${meta.min}" max="${meta.max}" step="1" value="${row.strength}" title="${meta.unit}"><span class="row-unit">${meta.unit}</span>${remove}</div>`;
  }).join('');
}

async function workerBigNetLines(runId: number, variant: string, fen: string, visits: number): Promise<AnalysisLine[]> {
  const { config, searcher } = bigNetFor(variant);
  const label = `Lc0 ${config.name}`;
  searcher.onDownloadProgress = (loaded, total) => showModelProgress(label, loaded, total, 'Downloading');
  try {
    const result = await searcher.search({ positions: tree.historyBoards() }, {
      visits,
      multiPv: multiPv(),
      batchSize: config.recommendedBatchSize,
      batchPipelineDepth: config.recommendedPipelineDepth,
      onProgress: (progress) => showAnalysisProgress(runId, fen, label, progress),
    });
    return result.cancelled ? [] : lc0AnalysisLines(result, fen, `Lc0 ${config.name}`);
  } finally {
    hideModelProgress();
  }
}

// Lc0 big nets are WebGPU-only and require the large local ONNX assets.
async function refreshBt4Availability(): Promise<void> {
  if (isV0DeployProfile()) return;
  await Promise.all([probeBt4Support(), checkBigNetAsset(BIG_NETS.bt4, renderRecklessRuntimeInfo), checkBigNetAsset(BIG_NETS.t3, renderRecklessRuntimeInfo)]);
  for (const row of engineRows) {
    if (row.family === 'lc0' && isLc0BigNetVariant(row.variant) && !bigNetSelectableSync(bigNetFor(row.variant).config)) row.variant = 'small';
  }
  renderEngineList();
  renderRecklessRuntimeInfo();
}

function renderRecklessRuntimeInfo(): void {
  const sab = typeof SharedArrayBuffer !== 'undefined' ? 'SAB yes' : 'SAB no';
  const bigNetTexts = isV0DeployProfile() ? [] : (['bt4', 't3'] as const).map((key) => {
    const config = BIG_NETS[key];
    const asset = bigNetAssetStatusSync(config);
    if (asset === 'unknown') void checkBigNetAsset(config, renderRecklessRuntimeInfo);
    const assetText = asset === 'present' ? 'asset ok' : asset === 'missing' ? `asset missing · ${config.modelUrl} · run node scripts/lc0_prepare_model_assets.mjs` : 'checking asset';
    return `Lc0 ${config.name}: ${bt4SupportedSync() ? 'WebGPU ok' : 'WebGPU unavailable'} · batch ${config.recommendedBatchSize} · pipeline depth ${config.recommendedPipelineDepth} · ${assetText}`;
  });
  const bt4Text = bigNetTexts.join(' | ');
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
    if (asset === 'unknown') void checkRecklessVariantAsset(variant, () => { renderEngineList(); renderRecklessRuntimeInfo(); });
    const assetText = asset === 'present' ? 'asset ok' : asset === 'missing' ? 'asset missing' : 'checking asset';
    const targetUrl = status?.wasmUrl ?? variant.wasmUrl;
    const assetUrlText = variant.nnueUrl ? `${targetUrl} + ${variant.nnueUrl}` : targetUrl;
    const loadText = formatRecklessBrowserApiLoadStatus(status?.browserApiLoad);
    return `${variant.label} · ${mode} · ${sab} · ${assetText} · ${assetUrlText}${loadText ? ` · ${loadText}` : ''}${status?.persistentDisabled ? ' · persistent disabled after fallback' : ''}${asset === 'missing' ? ' · build locally with npm run reckless:build-wasi, reckless:build-simd-wasi, reckless:build-relaxed-simd-wasi, reckless:build-browser-api-simd, reckless:build-browser-api-simd-external, or reckless:build-lite-wasi' : ''}`;
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
    const unsupportedReason = plentyChessVariantUnsupportedReason(variant);
    if (!unsupportedReason && asset === 'unknown') void checkPlentyChessVariantAsset(variant, renderRecklessRuntimeInfo);
    const assetText = unsupportedReason ? unsupportedReason : asset === 'present' ? 'asset ok' : asset === 'missing' ? 'asset missing' : 'checking asset';
    return `${variant.label} · ${engine?.runtimeLabel() ?? 'Emscripten worker idle'} · ${assetText} · ${variant.jsUrl}`;
  });
  el('recklessRuntimeInfo').textContent = `${bt4Text} · Tiny: ${tinyParts.join(' | ') || 'not selected'} · Reckless: ${recklessParts.join(' | ')} · Viridithas: ${viridithasParts.join(' | ')} · Berserk: ${berserkParts.join(' | ') || 'not selected'} · PlentyChess: ${plentyParts.join(' | ') || 'not selected'}`;
}

function threadedStockfishAvailable(): boolean {
  return typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated === true && typeof SharedArrayBuffer !== 'undefined';
}

function getStockfish(kind: 'lite' | 'full'): StockfishEngine {
  // Constructor depth is just a default; each analyze() call passes the row
  // depth and a broker-leased thread count. Analysis defaults to the threaded
  // flavor whenever isolation allows it (flavor is fixed per page lifetime).
  const threaded = threadedStockfishAvailable();
  if (kind === 'lite') {
    if (!stockfishLite) stockfishLite = new StockfishEngine({ depth: 14 }, stockfishFlavorUrl(threaded ? 'lite-threaded' : 'lite-single'));
    return stockfishLite;
  }
  if (!stockfishFull) stockfishFull = new StockfishEngine({ depth: 14 }, stockfishFlavorUrl(threaded ? 'threaded' : 'single'));
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
    engine = createRecklessEngine(variant, renderRecklessRuntimeInfo);
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
    engine = createViridithasEngine(variant);
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
    engine = createBerserkEngine(variant);
    berserkByVariant.set(key, engine);
    renderRecklessRuntimeInfo();
  }
  return engine;
}

function getPlentyChessFor(variantKey: string): PlentyChessEngine {
  const variant = plentyChessVariantForKey(variantKey);
  const unsupportedReason = plentyChessVariantUnsupportedReason(variant);
  if (unsupportedReason) throw new Error(`${variant.label} ${unsupportedReason}.`);
  const key = plentyChessCacheKey(variant);
  let engine = plentyChessByVariant.get(key);
  if (!engine) {
    engine = createPlentyChessEngine(variant);
    plentyChessByVariant.set(key, engine);
    renderRecklessRuntimeInfo();
  }
  return engine;
}

function disposeUnusedEngines(): void {
  for (const key of ANALYSIS_BIG_NET_KEYS) {
    if (usesBigNetRow(key)) continue;
    peekBigNetSearcher(key)?.cancel();
    releaseBigNetSearcher(key);
  }
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
    // Allows synthetic pointer events so automated browser checks can move pieces.
    trustAllEvents: true,
    check: boardCheck(board),
    highlight: { lastMove: true, check: true },
    animation: { enabled: true, duration: 160 },
    movable: {
      free: false,
      color: board.turn === 'w' ? 'white' as const : 'black' as const,
      dests: legalDests(board),
      events: { after: onUserMove },
    },
    lastMove: lastUci ? [lastUci.slice(0, 2) as Key, lastUci.slice(2, 4) as Key] : undefined,
    drawable: { brushes: ANALYSIS_DRAWABLE_BRUSHES },
  };
  // Cast: chessground's DrawBrushes type exposes the built-in keys, but custom
  // brush keys are supported at runtime and are needed to distinguish 3+ engines.
  const cfg = config as unknown as NonNullable<Parameters<typeof Chessground>[1]>;
  if (!ground) ground = Chessground(el('ground'), cfg);
  else ground.set(cfg);
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
    `<span class="key"><span class="dot" style="background:${key.swatch}"></span>${engineLogoHtmlForName(key.label)}${htmlEscape(key.label)}</span>`).join('');
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
    const progressRows = [...searchProgressByEngine.values()].map((progress) => `<tr><td>${htmlEscape(progress.label)}</td><td colspan="3" class="small">searching</td><td colspan="2">${searchProgressHtml(progress)}</td></tr>`);
    body.innerHTML = progressRows.length ? progressRows.join('') : '<tr><td colspan="6" class="small">Run analysis to compare selected engines.</td></tr>';
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
    const progress = searchProgressForLabel(line.engine);
    const searchText = progress
      ? (progress.indeterminate ? '...' : `${progress.completed ?? 0}/${progress.requested ?? 0}`)
      : htmlEscape(line.detail);
    return `<tr style="border-left:3px solid ${swatch}">`
      + `<td><span class="engine-name-with-logo">${engineLogoHtmlForName(line.engine)}${htmlEscape(line.engine)}</span></td>`
      + `<td class="mono ${agreed ? 'agree' : ''}">${htmlEscape(firstSanMove(line))}</td>`
      + `<td class="mono">${htmlEscape(line.scoreText)}</td>`
      + `<td class="mono">${htmlEscape(delta)}</td>`
      + `<td class="mono">${searchText}</td>`
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
      + `<span class="score ${cls}">${htmlEscape(line.scoreText)}</span>`
      + `<span class="pv">${htmlEscape(line.pvSan)}</span>`
      + `<span class="eng">${engineLogoHtmlForName(line.engine)}${htmlEscape(line.engine)} · ${htmlEscape(line.detail)}</span></li>`;
  }).join('') || '<li class="small placeholder">No analysis yet — make a move or press Analyze.</li>';
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
  const hasDatabasePositionStats = databasePositionKey === positionKey(tree.current.fen) && databasePositionStats.length > 0;
  if (!importedGames.length && !hasDatabasePositionStats) { body.innerHTML = '<tr><td colspan="3" class="small">import games or search the local DB to see opening stats</td></tr>'; return; }
  const stats = currentBookStats();
  const summary = openingSummary(stats);
  if (hasDatabasePositionStats && !importedGames.length) el('importInfo').textContent = `local DB search · ${databasePositionCollectionCount} collections · ${summary.total} from here`;
  else el('importInfo').textContent = `${importedGames.length} games · ${summary.total} from here`;
  if (!stats.length) { body.innerHTML = '<tr><td colspan="3" class="small">no games reached this position</td></tr>'; return; }
  body.innerHTML = stats.map((stat) => {
    const pct = (n: number) => (stat.count ? (n / stat.count) * 100 : 0).toFixed(0);
    return `<tr class="mv" data-uci="${htmlEscape(stat.uci)}"><td class="san">${htmlEscape(stat.san)}</td>`
      + `<td class="num">${stat.count}</td>`
      + `<td><div class="wdlbar" title="W ${stat.whiteWins} / D ${stat.draws} / B ${stat.blackWins}">`
      + `<div class="w" style="width:${pct(stat.whiteWins)}%"></div><div class="d" style="width:${pct(stat.draws)}%"></div><div class="b" style="width:${pct(stat.blackWins)}%"></div></div></td></tr>`;
  }).join('');
}

function setImportedPgn(raw: string, messagePrefix = 'imported'): number {
  const games = parsePgnGames(raw).map((game) => ({ tree: game.tree, result: game.result }));
  importedGames = games;
  importedPositionIndex = buildOpeningPositionIndex(games);
  databasePositionStats = [];
  databasePositionKey = '';
  databasePositionCollectionCount = 0;
  pgnDatabaseSearchKey = '';
  bookCache.clear();
  el('importInfo').textContent = `${messagePrefix} ${games.length} games`;
  renderOpening();
  renderLines(); // refresh the legend so the Book key appears
  setShapes(bestShapes());
  return games.length;
}

function selectedPgnCollectionId(): string {
  return selectEl('pgnDbSelect').value;
}

function renderPgnDatabaseList(selected = activePgnCollectionId): void {
  const list = el('pgnDbList');
  if (!pgnDatabaseAvailable()) { list.innerHTML = '<div class="empty">IndexedDB unavailable; local PGN collections cannot be listed.</div>'; return; }
  if (!pgnCollections.length) {
    list.innerHTML = '<div class="empty">No saved collections yet. Import PGN, name it, then Save to DB.</div>';
    return;
  }
  list.innerHTML = pgnCollections.map((entry) => {
    const selectedClass = entry.id === selected ? ' selected' : '';
    const indexed = entry.indexedPositionCount ? `${entry.indexedPositionCount} indexed positions` : 'index rebuilds on load/import';
    return `<button type="button" class="pgn-db-card${selectedClass}" data-id="${htmlEscape(entry.id)}">`
      + `<span class="name">${htmlEscape(entry.name)}</span><span class="meta">${entry.gameCount} games</span>`
      + `<span class="meta">${htmlEscape(formatPgnCollectionSummary(entry))}</span><span class="meta">${indexed}</span></button>`;
  }).join('');
}

function clearPgnDatabaseSearchResults(message = 'Search position checks saved collections and shows matching collection hits here.'): void {
  pgnDatabaseSearchKey = '';
  el('pgnDbSearchResults').innerHTML = `<div class="empty">${htmlEscape(message)}</div>`;
}

function clearStalePgnDatabaseSearchResults(): void {
  if (pgnDatabaseSearchKey && pgnDatabaseSearchKey !== positionKey(tree.current.fen)) {
    clearPgnDatabaseSearchResults('Search results cleared after moving to another position.');
  }
}

function renderPgnDatabaseSearchResults(results: PgnDatabaseSearchResult[]): void {
  const container = el('pgnDbSearchResults');
  if (!results.length) {
    container.innerHTML = '<div class="empty">No saved collection reached this position.</div>';
    return;
  }
  container.innerHTML = results.map((result) => {
    const total = result.total;
    const moves = result.stats.slice(0, 5).map((stat) => `${stat.san} ${stat.count}`).join(' · ');
    const extra = result.stats.length > 5 ? ` · +${result.stats.length - 5} moves` : '';
    return `<div class="pgn-db-hit"><div class="name">${htmlEscape(result.summary.name)}</div>`
      + `<div class="meta">${total} games reached this position · ${htmlEscape(formatPgnCollectionSummary(result.summary))}</div>`
      + `<div class="moves">${htmlEscape(moves || 'terminal/no next move')}${extra}</div></div>`;
  }).join('');
}

function renderPgnDatabaseCollections(selected = activePgnCollectionId): void {
  const select = selectEl('pgnDbSelect');
  if (!pgnDatabaseAvailable()) {
    select.innerHTML = '<option value="">IndexedDB unavailable</option>';
    select.disabled = true;
    el('savePgnDb').toggleAttribute('disabled', true);
    el('loadPgnDb').toggleAttribute('disabled', true);
    el('deletePgnDb').toggleAttribute('disabled', true);
    el('renamePgnDb').toggleAttribute('disabled', true);
    el('duplicatePgnDb').toggleAttribute('disabled', true);
    el('exportPgnDbCollection').toggleAttribute('disabled', true);
    el('exportPgnDb').toggleAttribute('disabled', true);
    el('importPgnDb').toggleAttribute('disabled', true);
    el('searchPgnDbPosition').toggleAttribute('disabled', true);
    el('pgnDbInfo').textContent = 'Local PGN database unavailable in this browser context.';
    renderPgnDatabaseList('');
    clearPgnDatabaseSearchResults('IndexedDB unavailable; position search is disabled.');
    return;
  }
  select.disabled = false;
  el('savePgnDb').toggleAttribute('disabled', false);
  el('exportPgnDb').toggleAttribute('disabled', false);
  el('importPgnDb').toggleAttribute('disabled', false);
  el('searchPgnDbPosition').toggleAttribute('disabled', false);
  const current = pgnCollections.some((entry) => entry.id === selected) ? selected : '';
  select.innerHTML = ['<option value="">new collection</option>', ...pgnCollections.map((entry) => `<option value="${htmlEscape(entry.id)}"${entry.id === current ? ' selected' : ''}>${htmlEscape(entry.name)} (${entry.gameCount})</option>`)].join('');
  el('loadPgnDb').toggleAttribute('disabled', !current);
  el('deletePgnDb').toggleAttribute('disabled', !current);
  el('renamePgnDb').toggleAttribute('disabled', !current);
  el('duplicatePgnDb').toggleAttribute('disabled', !current);
  el('exportPgnDbCollection').toggleAttribute('disabled', !current);
  if (current) {
    const summary = pgnCollections.find((entry) => entry.id === current)!;
    inputEl('pgnDbName').value = summary.name;
    el('pgnDbInfo').textContent = formatPgnCollectionSummary(summary);
  } else {
    el('pgnDbInfo').textContent = pgnCollections.length ? `${pgnCollections.length} saved PGN collections` : 'No saved PGN collections yet.';
  }
  renderPgnDatabaseList(current);
  if (!databasePositionStats.length) clearPgnDatabaseSearchResults();
}

async function refreshPgnDatabaseCollections(selected = activePgnCollectionId): Promise<void> {
  if (!pgnDatabaseAvailable()) { renderPgnDatabaseCollections(''); return; }
  try {
    pgnCollections = await listPgnCollections();
    activePgnCollectionId = pgnCollections.some((entry) => entry.id === selected) ? selected : '';
    renderPgnDatabaseCollections(activePgnCollectionId);
  } catch (error) {
    el('pgnDbInfo').textContent = `Local PGN database failed: ${(error as Error).message}`;
  }
}

function suggestPgnDatabaseName(): void {
  if (inputEl('pgnDbName').value.trim()) return;
  inputEl('pgnDbName').value = defaultPgnCollectionName(lastImportSource, lastImportUsername);
}

function importGames() {
  const raw = inputEl('importGamesInput').value.trim();
  if (!raw) { el('importInfo').textContent = 'paste or fetch PGN first'; return; }
  try {
    lastImportSource = 'manual';
    lastImportUsername = '';
    lastImportColor = '';
    activePgnCollectionId = '';
    setImportedPgn(raw);
    suggestPgnDatabaseName();
    renderPgnDatabaseCollections('');
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
    lastImportSource = site;
    lastImportUsername = username;
    lastImportColor = opts.color;
    activePgnCollectionId = '';
    inputEl('pgnDbName').value = defaultPgnCollectionName(site, username);
    setImportedPgn(pgn, `fetched ${username}:`);
    renderPgnDatabaseCollections('');
  } catch (error) {
    // A network/CORS failure surfaces as a TypeError with no status.
    const message = (error as Error).message || 'fetch failed';
    el('importInfo').textContent = `fetch failed: ${message}`;
  } finally {
    el('fetchGames').toggleAttribute('disabled', false);
  }
}

function safeFilename(name: string, fallback = 'games'): string {
  return (name.trim() || fallback).replace(/[^\w.-]+/g, '_').slice(0, 80) || fallback;
}

function downloadTextFile(filename: string, content: string, type: string): void {
  const blob = new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function downloadPgn() {
  const pgn = inputEl('importGamesInput').value;
  if (!pgn.trim()) { el('importInfo').textContent = 'nothing to download'; return; }
  const name = safeFilename(inputEl('importUser').value.trim() || inputEl('pgnDbName').value.trim() || 'games');
  downloadTextFile(`${name}.pgn`, pgn, 'application/x-chess-pgn');
  el('importInfo').textContent = `downloaded ${name}.pgn`;
}

async function saveCurrentPgnCollection(): Promise<void> {
  const raw = inputEl('importGamesInput').value.trim();
  if (!raw) { el('pgnDbInfo').textContent = 'Paste, fetch, or load PGN before saving to the local database.'; return; }
  let gameCount = 0;
  try {
    // Always parse the textarea at save time: users can edit PGN after an
    // import/load, and the persisted metadata must describe the saved text.
    gameCount = setImportedPgn(raw);
  } catch (error) {
    el('pgnDbInfo').textContent = `Cannot save invalid PGN: ${(error as Error).message}`;
    return;
  }
  el('savePgnDb').toggleAttribute('disabled', true);
  try {
    const record = await savePgnCollection({
      id: selectedPgnCollectionId() || activePgnCollectionId || undefined,
      name: inputEl('pgnDbName').value || defaultPgnCollectionName(lastImportSource, lastImportUsername),
      pgn: raw,
      gameCount,
      source: lastImportSource,
      username: lastImportUsername,
      color: lastImportColor,
      positionIndex: importedPositionIndex ?? undefined,
      indexedPositionCount: importedPositionIndex ? Object.keys(importedPositionIndex).length : 0,
    });
    activePgnCollectionId = record.id;
    inputEl('pgnDbName').value = record.name;
    await refreshPgnDatabaseCollections(record.id);
    el('pgnDbInfo').textContent = `Saved “${record.name}” (${record.gameCount} games) to local IndexedDB.`;
  } catch (error) {
    el('pgnDbInfo').textContent = `Save failed: ${(error as Error).message}`;
  } finally {
    el('savePgnDb').toggleAttribute('disabled', false);
  }
}

async function loadSelectedPgnCollection(): Promise<void> {
  const id = selectedPgnCollectionId();
  if (!id) { el('pgnDbInfo').textContent = 'Choose a saved PGN collection to load.'; return; }
  try {
    const record = await loadPgnCollection(id);
    if (!record) { el('pgnDbInfo').textContent = 'Saved PGN collection not found.'; await refreshPgnDatabaseCollections(''); return; }
    activePgnCollectionId = record.id;
    lastImportSource = record.source;
    lastImportUsername = record.username ?? '';
    lastImportColor = record.color ?? '';
    inputEl('pgnDbName').value = record.name;
    inputEl('importGamesInput').value = record.pgn;
    setImportedPgn(record.pgn, `loaded “${record.name}”:`);
    if (!record.positionIndex && importedPositionIndex) {
      await updatePgnCollectionPositionIndex(record.id, importedPositionIndex);
      await refreshPgnDatabaseCollections(record.id);
      el('pgnDbInfo').textContent = `Loaded and indexed “${record.name}” (${record.gameCount} games).`;
    } else {
      renderPgnDatabaseCollections(record.id);
    }
  } catch (error) {
    el('pgnDbInfo').textContent = `Load failed: ${(error as Error).message}`;
  }
}

async function renameSelectedPgnCollection(): Promise<void> {
  const id = selectedPgnCollectionId();
  if (!id) { el('pgnDbInfo').textContent = 'Choose a saved PGN collection to rename.'; return; }
  try {
    const record = await renamePgnCollection(id, inputEl('pgnDbName').value);
    inputEl('pgnDbName').value = record.name;
    await refreshPgnDatabaseCollections(record.id);
    el('pgnDbInfo').textContent = `Renamed collection to “${record.name}”.`;
  } catch (error) {
    el('pgnDbInfo').textContent = `Rename failed: ${(error as Error).message}`;
  }
}

async function duplicateSelectedPgnCollection(): Promise<void> {
  const id = selectedPgnCollectionId();
  if (!id) { el('pgnDbInfo').textContent = 'Choose a saved PGN collection to duplicate.'; return; }
  const summary = pgnCollections.find((entry) => entry.id === id);
  const requestedName = inputEl('pgnDbName').value.trim();
  const duplicateName = requestedName && requestedName !== summary?.name ? requestedName : `${summary?.name ?? 'PGN collection'} copy`;
  try {
    const record = await duplicatePgnCollection(id, duplicateName);
    activePgnCollectionId = record.id;
    inputEl('pgnDbName').value = record.name;
    await refreshPgnDatabaseCollections(record.id);
    el('pgnDbInfo').textContent = `Duplicated “${summary?.name ?? 'collection'}” as “${record.name}”.`;
  } catch (error) {
    el('pgnDbInfo').textContent = `Duplicate failed: ${(error as Error).message}`;
  }
}

async function exportSelectedPgnCollection(): Promise<void> {
  const id = selectedPgnCollectionId();
  if (!id) { el('pgnDbInfo').textContent = 'Choose a saved PGN collection to export.'; return; }
  try {
    const record = await loadPgnCollection(id);
    if (!record) { el('pgnDbInfo').textContent = 'Saved PGN collection not found.'; await refreshPgnDatabaseCollections(''); return; }
    const filename = `${safeFilename(record.name, 'collection')}.pgn`;
    downloadTextFile(filename, record.pgn, 'application/x-chess-pgn');
    el('pgnDbInfo').textContent = `Exported “${record.name}” as ${filename}.`;
  } catch (error) {
    el('pgnDbInfo').textContent = `Export failed: ${(error as Error).message}`;
  }
}

async function exportPgnDatabase(): Promise<void> {
  if (!pgnDatabaseAvailable()) { el('pgnDbInfo').textContent = 'Local PGN database unavailable in this browser context.'; return; }
  try {
    const backup = await exportPgnDatabaseBackup();
    const filename = pgnDatabaseBackupFilename(new Date(backup.exportedAt));
    downloadTextFile(filename, JSON.stringify(backup, null, 2), 'application/json');
    el('pgnDbInfo').textContent = `Exported ${backup.collections.length} PGN collections as ${filename}. Raw PGN is included; position indexes are rebuildable.`;
  } catch (error) {
    el('pgnDbInfo').textContent = `Database export failed: ${(error as Error).message}`;
  }
}

async function importPgnDatabaseFile(file: File | undefined): Promise<void> {
  if (!file) return;
  try {
    const backup = JSON.parse(await file.text()) as unknown;
    const count = await importPgnDatabaseBackup(backup);
    await refreshPgnDatabaseCollections(activePgnCollectionId);
    el('pgnDbInfo').textContent = `Imported ${count} PGN collections from ${file.name}; position indexes rebuilt from raw PGN.`;
  } catch (error) {
    el('pgnDbInfo').textContent = `Database import failed: ${(error as Error).message}`;
  } finally {
    inputEl('importPgnDbFile').value = '';
  }
}

async function searchCurrentPositionInPgnDatabase(): Promise<void> {
  if (!pgnDatabaseAvailable()) { el('pgnDbInfo').textContent = 'Local PGN database unavailable in this browser context.'; return; }
  el('searchPgnDbPosition').toggleAttribute('disabled', true);
  try {
    const searchFen = tree.current.fen;
    const searchKey = positionKey(searchFen);
    const results = await searchPgnCollectionsByPosition(searchFen);
    if (positionKey(tree.current.fen) !== searchKey) {
      clearPgnDatabaseSearchResults('Search results cleared after moving to another position.');
      return;
    }
    pgnDatabaseSearchKey = searchKey;
    databasePositionStats = mergeOpeningMoveStats(results.map((result) => result.stats));
    databasePositionKey = results.length ? searchKey : '';
    databasePositionCollectionCount = results.length;
    if (results.length) {
      importedGames = [];
      importedPositionIndex = null;
      bookCache.clear();
      const summary = openingSummary(databasePositionStats);
      const names = results.slice(0, 3).map((result) => result.summary.name).join(', ');
      const extra = results.length > 3 ? `, +${results.length - 3} more` : '';
      el('pgnDbInfo').textContent = `Found ${summary.total} games from this position in ${results.length} local collections: ${names}${extra}.`;
      renderPgnDatabaseSearchResults(results);
    } else {
      databasePositionStats = [];
      databasePositionKey = '';
      databasePositionCollectionCount = 0;
      el('pgnDbInfo').textContent = 'No indexed local PGN collections reached this position.';
      renderPgnDatabaseSearchResults([]);
    }
    renderOpening();
    renderLines();
    setShapes(bestShapes());
  } catch (error) {
    el('pgnDbInfo').textContent = `Position search failed: ${(error as Error).message}`;
    clearPgnDatabaseSearchResults(`Position search failed: ${(error as Error).message}`);
  } finally {
    el('searchPgnDbPosition').toggleAttribute('disabled', false);
  }
}

async function deleteSelectedPgnCollection(): Promise<void> {
  const id = selectedPgnCollectionId();
  if (!id) return;
  const summary = pgnCollections.find((entry) => entry.id === id);
  if (summary && !window.confirm(`Delete local PGN collection “${summary.name}”?`)) return;
  try {
    await deletePgnCollection(id);
    if (activePgnCollectionId === id) activePgnCollectionId = '';
    await refreshPgnDatabaseCollections('');
    el('pgnDbInfo').textContent = summary ? `Deleted “${summary.name}”.` : 'Deleted PGN collection.';
  } catch (error) {
    el('pgnDbInfo').textContent = `Delete failed: ${(error as Error).message}`;
  }
}

function renderAll() {
  renderBoard();
  renderLines();
  renderMoveList();
  renderOpening();
}

function mainlineNodes(): GameNode[] {
  const nodes: GameNode[] = [tree.root];
  let node: GameNode | undefined = tree.root.children[0];
  while (node) { nodes.push(node); node = node.children[0]; }
  return nodes;
}

function reviewSignature(nodes = mainlineNodes()): string {
  return nodes.map((node) => `${node.fen}\u0000${node.move ? moveToUci(node.move) : ''}`).join('\u0001');
}

function clearReviewState(message?: string): void {
  lastReview = null;
  lastReviewNodes = [];
  lastReviewSignature = '';
  el('reviewSummary').hidden = true;
  el('reviewSummary').innerHTML = '';
  el('reviewChart').hidden = true;
  el('reviewChart').innerHTML = '';
  el('reviewCritical').hidden = true;
  el('reviewCritical').innerHTML = '';
  el('reviewCopyPgn').hidden = true;
  if (message) el('reviewStatus').textContent = message;
}

function clearReviewIfMainlineChanged(): void {
  if (lastReview && reviewSignature() !== lastReviewSignature) clearReviewState('Review cleared after game changed.');
}

interface ReviewEngineChoice { engine: { analyze(fen: string, opts?: { multipv?: number; depth?: number; signal?: AbortSignal }): Promise<{ scoreCp?: number; mateIn?: number; pvUci: string[] }[]> }; label: string; depth: number; }

/** First selected UCI-family engine reviews the game; SF Lite d12 is the fallback. */
function reviewEngineChoice(): ReviewEngineChoice {
  for (const row of activeEngineRows()) {
    if (row.family === 'sf') return { engine: getStockfish(row.variant === 'full' ? 'full' : 'lite'), label: row.variant === 'full' ? 'SF' : 'SF Lite', depth: row.strength };
    if (row.family === 'reckless') return { engine: getRecklessFor(row.variant), label: recklessVariantByKey(normalizeRecklessVariant(row.variant)).label, depth: row.strength };
    if (row.family === 'viridithas') return { engine: getViridithasFor(row.variant), label: viridithasVariantForKey(row.variant).label, depth: row.strength };
    if (row.family === 'berserk') return { engine: getBerserkFor(row.variant), label: berserkVariantForKey(row.variant).label, depth: row.strength };
    if (row.family === 'plentychess') return { engine: getPlentyChessFor(row.variant), label: plentyChessVariantForKey(row.variant).label, depth: row.strength };
  }
  return { engine: getStockfish('lite'), label: 'SF Lite', depth: 12 };
}

function winWhiteFromInfo(fen: string, info: { scoreCp?: number; mateIn?: number } | undefined): number {
  if (!info) return 0.5;
  const turn = parseFen(fen).turn;
  const whiteCp = info.scoreCp === undefined ? undefined : turn === 'w' ? info.scoreCp : -info.scoreCp;
  const whiteMate = info.mateIn === undefined ? undefined : turn === 'w' ? info.mateIn : -info.mateIn;
  return evalBarWhitePercent(whiteCp, whiteMate) / 100;
}

const REVIEW_CLASS_LABEL: Record<ReviewedMove['class'], string> = {
  best: 'Best', good: 'Good', inaccuracy: 'Inaccuracy', mistake: 'Mistake', blunder: 'Blunder', forced: 'Forced',
};

function reviewSummaryHtml(review: GameReview): string {
  const side = (label: string, accuracy: number, counts: GameReview['counts']['white']) =>
    `<div><div class="small">${label}</div><div class="acc">${accuracy.toFixed(1)}%</div>`
    + `<div class="small">${counts.blunder}×<span class="review-badge blunder">??</span> ${counts.mistake}×<span class="review-badge mistake">?</span> ${counts.inaccuracy}×<span class="review-badge inaccuracy">?!</span></div></div>`;
  return `<div class="review-summary">${side('White accuracy', review.accuracy.white, review.counts.white)}${side('Black accuracy', review.accuracy.black, review.counts.black)}</div>`;
}

function renderReview(review: GameReview, nodes: GameNode[]): void {
  el('reviewSummary').hidden = false;
  el('reviewSummary').innerHTML = reviewSummaryHtml(review);
  el('reviewChart').hidden = false;
  el('reviewChart').innerHTML = lineChartSvg([{
    label: 'White win %',
    color: '#4a7a2a',
    points: review.moves.map((move) => ({ x: move.ply, y: move.winAfter })),
  }], { yMin: 0, yMax: 1, midline: 0.5, formatY: (v) => `${Math.round(v * 100)}%`, height: 110 });
  const critical = el('reviewCritical');
  critical.hidden = review.criticalMoves.length === 0;
  critical.innerHTML = review.criticalMoves.map((move) => {
    const node = nodes[move.ply];
    const moveNo = `${Math.ceil(move.ply / 2)}${move.side === 'w' ? '.' : '…'}`;
    return `<li data-node="${node?.id ?? ''}"><span class="review-badge ${move.class}">${REVIEW_CLASS_LABEL[move.class]}</span> `
      + `<span class="mono">${moveNo} ${htmlEscape(move.san)}</span> · win ${Math.round(move.winBefore * 100)}%→${Math.round(move.winAfter * 100)}%`
      + `${move.bestUci ? ` · best <span class="mono">${htmlEscape(move.bestUci)}</span>` : ''}</li>`;
  }).join('');
}

async function runGameReview(): Promise<void> {
  if (reviewAbort) return;
  const nodes = mainlineNodes();
  if (nodes.length < 2) { el('reviewStatus').textContent = 'Load a PGN or play some moves first.'; return; }
  const { engine, label, depth } = reviewEngineChoice();
  const controller = new AbortController();
  reviewAbort = controller;
  el('reviewGame').toggleAttribute('disabled', true);
  el('reviewStop').toggleAttribute('disabled', false);
  el('reviewCopyPgn').hidden = true;
  try {
    const positions: ReviewPosition[] = [];
    for (let i = 0; i < nodes.length; i++) {
      if (controller.signal.aborted) break;
      el('reviewStatus').textContent = `Reviewing with ${label}: position ${i + 1}/${nodes.length}…`;
      const fen = nodes[i].fen;
      const board = parseFen(fen);
      const legal = legalMoves(board).length;
      if (legal === 0) {
        // Terminal: engines return no lines here. Checkmate scores the side to
        // move as lost; stalemate is a draw.
        positions.push({ winWhite: inCheck(board) ? (board.turn === 'w' ? 0 : 1) : 0.5, bestUci: null, legalMoves: 0 });
        continue;
      }
      const lines = await engine.analyze(fen, { multipv: 1, depth, signal: controller.signal });
      positions.push({
        winWhite: winWhiteFromInfo(fen, lines[0]),
        bestUci: lines[0]?.pvUci?.[0] ?? null,
        legalMoves: legal,
      });
    }
    if (controller.signal.aborted || positions.length !== nodes.length) {
      el('reviewStatus').textContent = 'Review stopped.';
      return;
    }
    const moves = nodes.slice(1).map((node) => ({ san: node.san ?? '?', uci: node.move ? moveToUci(node.move) : '' }));
    lastReview = reviewGame(positions, moves, parseFen(nodes[0].fen).turn);
    lastReviewNodes = nodes;
    lastReviewSignature = reviewSignature(nodes);
    renderReview(lastReview, nodes);
    el('reviewCopyPgn').hidden = false;
    el('reviewStatus').textContent = `Reviewed ${moves.length} moves with ${label}.`;
  } catch (error) {
    el('reviewStatus').textContent = (error as Error).name === 'AbortError' ? 'Review stopped.' : `Review failed: ${(error as Error).message}`;
  } finally {
    reviewAbort = null;
    el('reviewGame').toggleAttribute('disabled', false);
    el('reviewStop').toggleAttribute('disabled', true);
  }
}

async function copyReviewPgn(): Promise<void> {
  if (!lastReview || !lastReviewNodes.length) return;
  if (reviewSignature() !== lastReviewSignature) {
    clearReviewState('Review cleared after game changed; run review again before copying annotated PGN.');
    return;
  }
  const start = parseFen(lastReviewNodes[0].fen);
  const pgn = annotatedPgn(lastReview, {
    tags: { Event: 'LC0 analysis review', ...(lastReviewNodes[0].fen === START_FEN ? {} : { SetUp: '1', FEN: lastReviewNodes[0].fen }) },
    startFullmove: start.fullmove,
    startTurn: start.turn,
  });
  try {
    await navigator.clipboard.writeText(pgn);
    el('reviewStatus').textContent = 'Annotated PGN copied to clipboard.';
  } catch {
    el('reviewStatus').textContent = 'Clipboard unavailable — annotated PGN logged to console.';
    console.info(pgn);
  }
}

async function analyzeCurrent(options: { force?: boolean } = {}) {
  if (mountAbort.signal.aborted) return;
  const rows = activeEngineRows();
  if (!rows.length) { el('message').textContent = 'Add an engine to analyze.'; return; }
  const runId = ++activeAnalysisRunId;
  // Interrupt any in-flight analysis: abort the Stockfish signal and cancel the
  // worker LC0 / big-net searches, so a new position takes over immediately.
  analysisAbort?.abort();
  if (activeWorkerSearchId !== null && searchWorker) searchWorker.postMessage({ type: 'cancel', target: activeWorkerSearchId });
  for (const key of ANALYSIS_BIG_NET_KEYS) peekBigNetSearcher(key)?.cancel();
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
  const selectedLabels = rows.map((row) => rowLabel(row)).join(' + ');
  const selectionCacheKey = analysisSelectionCacheKey(fen, rows);
  el('message').textContent = `Analyzing (${selectedLabels}, ${multiPv()} lines)…`;
  clearAnalysisSearchProgress();
  syncBrokerParticipants(rows);
  try {
    const lineGroups: AnalysisLine[][] = rows.map((row) => (!options.force ? (engineLineCache.get(engineAnalysisCacheKey(fen, row)) ?? []) : []));
    const publishLines = () => {
      lineCache.set(fen, lineGroups.flat());
      if (tree.current.fen === fen) { renderLines(); renderEvalBar(); setShapes(bestShapes()); }
    };
    if (lineGroups.some((lines) => lines.length > 0)) publishLines();
    const tasks: Promise<void>[] = [];
    const pushTask = (index: number, cacheKey: string, label: string, task: Promise<AnalysisLine[]>) => {
      showIndeterminateSearchProgress(runId, fen, label);
      tasks.push(task.then((lines) => {
        if (runId !== activeAnalysisRunId || analysisAbort !== controller || !analyzing || controller.signal.aborted || mountAbort.signal.aborted) return;
        if (lines.length) engineLineCache.set(cacheKey, lines);
        else engineLineCache.delete(cacheKey);
        lineGroups[index] = lines;
        publishLines();
      }).finally(() => clearEngineSearchProgress(runId, label)));
    };
    for (let index = 0; index < rows.length; index++) {
      const row = rows[index];
      const cacheKey = engineAnalysisCacheKey(fen, row);
      if (!options.force && engineLineCache.has(cacheKey)) continue;
      if (row.family === 'lc0' && isLc0BigNetVariant(row.variant)) {
        pushTask(index, cacheKey, `Lc0 ${BIG_NETS[row.variant as AnalysisBigNetKey].name}`, workerBigNetLines(runId, row.variant, fen, row.strength));
      } else if (row.family === 'lc0') {
        const label = 'Lc0';
        if (workerReady) pushTask(index, cacheKey, label, workerLc0Lines(runId, fen, row.strength, label));
        else if (searcher) pushTask(index, cacheKey, label, searcher.search({ positions: tree.historyBoards() }, {
          visits: row.strength,
          multiPv: multiPv(),
          signal: controller.signal,
          yieldEveryMs: 16,
          onProgress: (progress) => showAnalysisProgress(runId, fen, label, progress),
        })
          .then((result) => lc0AnalysisLines(result, fen, 'Lc0')));
      } else if (row.family === 'tiny') {
        const positions = tree.historyBoards();
        const current = positions[positions.length - 1];
        const label = tinyEngineLabel(row.variant);
        pushTask(index, cacheKey, label, tinyEvaluator(row.variant)
          .then((evaluator) => chooseMove(current, evaluator, {
            visits: row.strength,
            batchSize: Math.max(1, Math.min(256, Math.floor(Number(params.get('tinyBatch') ?? '32') || 32))),
            signal: controller.signal,
            historyFens: tinyHistoryFens(positions),
            searchPolicy: montyLitePuctPolicy,
            includePv: true,
            multiPv: multiPv(),
            pvDepth: 12,
            onProgress: (progress) => showMoveSearchProgress(runId, fen, label, progress),
          }))
          .then((result) => tinyPuctAnalysisLines(result, fen, label)));
      } else if (row.family === 'sf') {
        const kind = row.variant === 'full' ? 'full' : 'lite';
        const label = kind === 'lite' ? 'SF Lite' : 'SF';
        pushTask(index, cacheKey, label, resourceBroker.acquire({ engineId: `sf-${kind}`, signal: controller.signal }).then(async (lease) => {
          try {
            const engine = getStockfish(kind);
            engine.setOptions({ threads: lease.threads });
            const infos = await engine.analyze(fen, { multipv: multiPv(), depth: row.strength, signal: controller.signal });
            return stockfishAnalysisLines(infos, fen, label);
          } finally {
            lease.release();
          }
        }));
      } else if (row.family === 'viridithas') {
        const label = `${viridithasVariantForKey(row.variant).label}`;
        const engine = getViridithasFor(row.variant);
        pushTask(index, cacheKey, label, engine.newGame(controller.signal)
          .then(() => engine.analyze(fen, { multipv: multiPv(), depth: row.strength, signal: controller.signal }))
          .then((infos) => { renderRecklessRuntimeInfo(); return stockfishAnalysisLines(infos, fen, label); }));
      } else if (row.family === 'berserk') {
        const label = `${berserkVariantForKey(row.variant).label}`;
        const engine = getBerserkFor(row.variant);
        pushTask(index, cacheKey, label, engine.newGame(controller.signal)
          .then(() => engine.analyze(fen, { multipv: multiPv(), depth: row.strength, signal: controller.signal }))
          .then((infos) => { renderRecklessRuntimeInfo(); return stockfishAnalysisLines(infos, fen, label); }));
      } else if (row.family === 'plentychess') {
        const label = `${plentyChessVariantForKey(row.variant).label}`;
        const engine = getPlentyChessFor(row.variant);
        pushTask(index, cacheKey, label, engine.newGame(controller.signal)
          .then(() => engine.analyze(fen, { multipv: multiPv(), depth: row.strength, signal: controller.signal }))
          .then((infos) => { renderRecklessRuntimeInfo(); return stockfishAnalysisLines(infos, fen, label); }));
      } else {
        const label = `${recklessVariantByKey(normalizeRecklessVariant(row.variant)).label}`;
        pushTask(index, cacheKey, label, getRecklessFor(row.variant).analyze(fen, { multipv: multiPv(), depth: row.strength, signal: controller.signal })
          .then((infos) => { renderRecklessRuntimeInfo(); return stockfishAnalysisLines(infos, fen, label); }));
      }
    }
    await Promise.all(tasks);
    if (controller.signal.aborted || mountAbort.signal.aborted) return;
    publishLines();
    completeAnalysisKeys.add(selectionCacheKey);
    const message = document.getElementById('message');
    if (message) message.textContent = `Analyzed: ${(lineCache.get(fen) ?? [])[0]?.scoreText ?? '—'}`;
  } catch (error) {
    controller.abort();
    const message = document.getElementById('message');
    if ((error as Error).name !== 'AbortError' && !mountAbort.signal.aborted && message) message.textContent = `Analysis failed: ${(error as Error).message}`;
  } finally {
    if (analysisAbort === controller && runId === activeAnalysisRunId && !mountAbort.signal.aborted) {
      analyzing = false;
      analysisAbort = null;
      clearAnalysisSearchProgress();
      document.getElementById('stop')?.toggleAttribute('disabled', true);
      document.getElementById('analyze')?.toggleAttribute('disabled', false);
    }
  }
}

function afterNavigation() {
  clearStalePgnDatabaseSearchResults();
  clearReviewIfMainlineChanged();
  renderAll();
  scheduleMaia3Panel();
  if (inputEl('autoAnalyze').checked && !completeAnalysisKeys.has(analysisSelectionCacheKey(tree.current.fen))) void analyzeCurrent();
  else { renderEvalBar(); setShapes(bestShapes()); }
}

// ---------------------------------------------------------------------------
// Human moves · Maia3: rating-conditioned move predictions for the current
// position. One batched evaluateConditions run per navigation (≈50ms on
// WebGPU for the whole grid), so it can follow the board live.
// ---------------------------------------------------------------------------
const MAIA3_PANEL_RATINGS = [1100, 1300, 1500, 1700, 1900, 2200];
let maia3PanelEvaluator: Maia3BrowserEvaluator | null = null;
let maia3PanelLoading = false;
let maia3PanelSeq = 0;
let maia3PanelTimer: ReturnType<typeof setTimeout> | null = null;

function maia3PanelStatus(text: string): void {
  el('maia3PanelStatus').textContent = text;
}

async function enableMaia3Panel(): Promise<void> {
  if (maia3PanelEvaluator || maia3PanelLoading) return;
  maia3PanelLoading = true;
  (el('maia3Enable') as HTMLButtonElement).disabled = true;
  try {
    maia3PanelStatus('Loading Maia3…');
    showModelProgress('Maia3', undefined, undefined, 'Preparing');
    maia3PanelEvaluator = await Maia3BrowserEvaluator.create({
      onProgress: (loaded, total) => {
        maia3PanelStatus(`Downloading Maia3 ${(loaded / 1e6).toFixed(0)}/${total ? (total / 1e6).toFixed(0) : '?'}MB…`);
        showModelProgress('Maia3', loaded, total, 'Downloading');
      },
    });
    hideModelProgress();
    el('maia3Enable').hidden = true;
    el('maia3Grid').hidden = false;
    el('maia3Caption').hidden = false;
    maia3PanelStatus('');
    scheduleMaia3Panel();
  } catch (error) {
    maia3PanelStatus(`Maia3 load failed: ${(error as Error).message}`);
    (el('maia3Enable') as HTMLButtonElement).disabled = false;
  } finally {
    hideModelProgress();
    maia3PanelLoading = false;
  }
}

function scheduleMaia3Panel(): void {
  if (!maia3PanelEvaluator) return;
  if (maia3PanelTimer) clearTimeout(maia3PanelTimer);
  maia3PanelTimer = setTimeout(() => { void renderMaia3Panel(); }, 120);
}

/** Best engine move for the current position, for the agreement marks. */
function bestEngineUci(fen: string): string | null {
  const lines = lineCache.get(fen) ?? [];
  const first = lines.find((line) => line.multipv === 1) ?? lines[0];
  return first?.pvUci[0] ?? null;
}

async function renderMaia3Panel(): Promise<void> {
  const evaluator = maia3PanelEvaluator;
  if (!evaluator) return;
  const seq = ++maia3PanelSeq;
  const fen = tree.current.fen || START_FEN;
  const board = parseFen(fen);
  if (!legalMoves(board).length) { el('maia3Grid').innerHTML = '<div class="small">Game over — no moves to predict.</div>'; return; }
  let evaluations;
  try {
    evaluations = await evaluator.evaluateConditions(board, MAIA3_PANEL_RATINGS.map((elo) => ({ selfElo: elo, oppoElo: elo })));
  } catch (error) {
    maia3PanelStatus(`Maia3 evaluation failed: ${(error as Error).message}`);
    return;
  }
  if (seq !== maia3PanelSeq) return;
  const engineUci = bestEngineUci(fen);
  const sanByUci = new Map(legalMoves(board).map((move) => [moveToUci(move), moveToSan(board, move)]));
  const rows = evaluations.map((evaluation, i) => {
    const score = maia3WinProbability(evaluation);
    const top = evaluation.legalPriors.slice(0, 3).map((entry) => {
      const san = sanByUci.get(entry.uci) ?? entry.uci;
      const agree = engineUci && entry.uci === engineUci ? ' ✓' : '';
      const width = Math.max(3, Math.round(entry.prior * 70));
      return `<span class="maia3-move">${san}${agree} <span class="maia3-bar" style="width:${width}px"></span> ${(entry.prior * 100).toFixed(0)}%</span>`;
    }).join('');
    return `<div class="maia3-row"><span class="maia3-elo">${MAIA3_PANEL_RATINGS[i]}</span><span class="maia3-score" title="Expected points for White in a human game at this rating">${score.toFixed(2)}</span>${top}</div>`;
  });
  el('maia3Grid').innerHTML = rows.join('');
}

async function onUserMove(from: Key, to: Key) {
  const board = tree.current.fen ? parseFen(tree.current.fen) : parseFen(START_FEN);
  const matching = matchUserMoves(board, from, to);
  if (!matching.length) { renderBoard(); return; }
  if (matching.length > 1) {
    // Promotion: let the user pick instead of silently auto-queening.
    showPromotionOverlay({
      boardContainer: el('ground'),
      orientation,
      color: board.turn,
      choices: matching,
      onPick: (move) => { tree.addMove(move); afterNavigation(); },
      onCancel: () => renderBoard(),
    });
    return;
  }
  tree.addMove(matching[0]);
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
  completeAnalysisKeys.clear();
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
    completeAnalysisKeys.clear();
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
  el('maia3Enable').addEventListener('click', () => { void enableMaia3Panel(); });
  el('navStart').addEventListener('click', () => { tree.toStart(); afterNavigation(); });
  el('navBack').addEventListener('click', () => { tree.back(); afterNavigation(); });
  el('navForward').addEventListener('click', () => { tree.forward(); afterNavigation(); });
  el('navEnd').addEventListener('click', () => { tree.toEnd(); afterNavigation(); });
  el('flip').addEventListener('click', () => { orientation = orientation === 'white' ? 'black' : 'white'; renderBoard(); });
  el('loadFen').addEventListener('click', loadFen);
  inputEl('fenInput').addEventListener('keydown', (event) => { if ((event as KeyboardEvent).key === 'Enter') loadFen(); });
  el('reset').addEventListener('click', () => { tree = new GameTree(); lineCache.clear(); completeAnalysisKeys.clear(); el('message').textContent = 'Reset.'; afterNavigation(); });
  el('loadPgn').addEventListener('click', loadPgn);
  el('copyPgn').addEventListener('click', copyPgn);
  el('analyze').addEventListener('click', () => { void analyzeCurrent({ force: true }); });
  el('stop').addEventListener('click', () => {
    analysisAbort?.abort();
    if (activeWorkerSearchId !== null && searchWorker) searchWorker.postMessage({ type: 'cancel', target: activeWorkerSearchId });
    for (const key of ANALYSIS_BIG_NET_KEYS) peekBigNetSearcher(key)?.cancel();
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
      if (engineRows[i].family === 'lc0' && isLc0BigNetVariant(target.value)) {
        // One-time gate before the large lazy load.
        const { config } = bigNetFor(target.value);
        if (!bigNetSelectableSync(config)) { el('message').textContent = bigNetUnavailableText(config); target.value = engineRows[i].variant; return; }
        if (!window.confirm(`${bigNetLoadWarning(config)}\n\nUse Lc0 ${config.name}?`)) { target.value = engineRows[i].variant; return; }
      }
      engineRows[i].variant = target.value;
    } else if (target.classList.contains('row-strength')) {
      const meta = strengthMeta(engineRows[i].family);
      engineRows[i].strength = Math.max(meta.min, Math.min(meta.max, Math.floor(Number(target.value) || meta.def)));
    }
    engineRows[i] = normalizeDeployEngineRow(engineRows[i], 'analysis', i);
    disposeUnusedEngines();
    lineCache.delete(tree.current.fen);
    void analyzeCurrent();
  });
  el('engineProfileSelect').addEventListener('change', () => {
    const value = selectEl('engineProfileSelect').value;
    const builtIn = BUILT_IN_ENGINE_PROFILES.find((entry) => builtInProfileValue(entry.id) === value);
    if (builtIn) { applyEngineProfile(builtIn, { selected: value, persistLast: false, note: builtIn.note }); return; }
    const profile = engineProfiles.find((entry) => entry.name === value);
    if (profile) applyEngineProfile(profile);
    else { storageRemove(LAST_ENGINE_PROFILE_STORAGE_KEY); inputEl('engineProfileName').value = ''; renderEngineProfiles(''); }
  });
  el('saveEngineProfile').addEventListener('click', saveCurrentEngineProfile);
  el('deleteEngineProfile').addEventListener('click', deleteSelectedEngineProfile);
  el('exportEngineProfiles').addEventListener('click', exportEngineProfiles);
  el('importEngineProfiles').addEventListener('click', () => inputEl('importEngineProfilesFile').click());
  el('importEngineProfilesFile').addEventListener('change', () => { void importEngineProfilesFile(inputEl('importEngineProfilesFile').files?.[0]); });
  inputEl('engineProfileName').addEventListener('keydown', (event) => { if ((event as KeyboardEvent).key === 'Enter') saveCurrentEngineProfile(); });
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
  el('reviewGame').addEventListener('click', () => { void runGameReview(); });
  el('reviewStop').addEventListener('click', () => reviewAbort?.abort());
  el('reviewCopyPgn').addEventListener('click', () => { void copyReviewPgn(); });
  el('reviewCritical').addEventListener('click', (event) => {
    const target = (event.target as HTMLElement).closest('[data-node]');
    if (!target) return;
    const node = nodeIndex.get(Number(target.getAttribute('data-node')));
    if (node) { tree.goTo(node); afterNavigation(); }
  });
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
  el('pgnDbSelect').addEventListener('change', () => { activePgnCollectionId = selectedPgnCollectionId(); renderPgnDatabaseCollections(activePgnCollectionId); });
  el('pgnDbList').addEventListener('click', (event) => {
    const button = (event.target as HTMLElement).closest<HTMLButtonElement>('.pgn-db-card');
    const id = button?.dataset.id;
    if (!id) return;
    activePgnCollectionId = id;
    selectEl('pgnDbSelect').value = id;
    renderPgnDatabaseCollections(id);
  });
  el('savePgnDb').addEventListener('click', () => { void saveCurrentPgnCollection(); });
  el('loadPgnDb').addEventListener('click', () => { void loadSelectedPgnCollection(); });
  el('deletePgnDb').addEventListener('click', () => { void deleteSelectedPgnCollection(); });
  el('renamePgnDb').addEventListener('click', () => { void renameSelectedPgnCollection(); });
  el('duplicatePgnDb').addEventListener('click', () => { void duplicateSelectedPgnCollection(); });
  el('exportPgnDbCollection').addEventListener('click', () => { void exportSelectedPgnCollection(); });
  el('exportPgnDb').addEventListener('click', () => { void exportPgnDatabase(); });
  el('importPgnDb').addEventListener('click', () => inputEl('importPgnDbFile').click());
  el('importPgnDbFile').addEventListener('change', () => { void importPgnDatabaseFile(inputEl('importPgnDbFile').files?.[0]); });
  el('searchPgnDbPosition').addEventListener('click', () => { void searchCurrentPositionInPgnDatabase(); });
  inputEl('pgnDbName').addEventListener('keydown', (event) => { if ((event as KeyboardEvent).key === 'Enter') void saveCurrentPgnCollection(); });
  inputEl('importGamesInput').addEventListener('input', () => {
    activePgnCollectionId = '';
    lastImportSource = 'manual';
    lastImportUsername = '';
    lastImportColor = '';
    databasePositionStats = [];
    databasePositionKey = '';
    databasePositionCollectionCount = 0;
    renderPgnDatabaseCollections('');
  });
  inputEl('importUser').addEventListener('keydown', (event) => { if ((event as KeyboardEvent).key === 'Enter') void fetchGames(); });
  el('opening').addEventListener('click', (event) => {
    const row = (event.target as HTMLElement).closest('tr[data-uci]');
    const uci = row?.getAttribute('data-uci');
    if (uci && tree.addUci(uci)) afterNavigation();
  });
  if (analysisKeydownHandler) document.removeEventListener('keydown', analysisKeydownHandler);
  analysisKeydownHandler = (event: KeyboardEvent) => {
    if (event.target instanceof HTMLInputElement || event.target instanceof HTMLTextAreaElement) return;
    if (event.key === 'ArrowLeft') { tree.back(); afterNavigation(); }
    else if (event.key === 'ArrowRight') { tree.forward(); afterNavigation(); }
    else if (event.key === 'ArrowUp') { tree.toStart(); afterNavigation(); }
    else if (event.key === 'ArrowDown') { tree.toEnd(); afterNavigation(); }
    else return;
    event.preventDefault();
  };
  document.addEventListener('keydown', analysisKeydownHandler);
}

function disposeRuntimeResources(): void {
  analysisAbort?.abort();
  analysisAbort = null;
  if (activeWorkerSearchId !== null) searchWorker?.postMessage({ type: 'cancel', target: activeWorkerSearchId });
  activeWorkerSearchId = null;
  searchWorker?.terminate();
  searchWorker = null;
  workerReady = false;
  workerBackend = '';
  for (const pending of workerPending.values()) pending.reject(new Error('LC0 worker disposed'));
  workerPending.clear();
  for (const key of ANALYSIS_BIG_NET_KEYS) {
    peekBigNetSearcher(key)?.cancel();
    releaseBigNetSearcher(key);
  }
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
  void mainEvaluator?.dispose?.();
  mainEvaluator = null;
  searcher = null;
}

function disposePageResources(): void {
  disposeRuntimeResources();
  if (maia3PanelTimer) clearTimeout(maia3PanelTimer);
  maia3PanelTimer = null;
  if (analysisKeydownHandler) document.removeEventListener('keydown', analysisKeydownHandler);
  analysisKeydownHandler = null;
  if (analysisPagehideHandler) window.removeEventListener('pagehide', analysisPagehideHandler);
  analysisPagehideHandler = null;
  if (analysisAuditHandler) window.removeEventListener(BROWSER_RUNTIME_AUDIT_EVENT, analysisAuditHandler);
  analysisAuditHandler = null;
  (ground as { destroy?: () => void } | null)?.destroy?.();
  ground = null;
}

async function loadLc0Backend(runAutoAnalyze = true, mountSignal: AbortSignal = mountAbort.signal): Promise<boolean> {
  if (isStaleMount(mountSignal)) return false;
  const runtime = selectedLc0Runtime();
  el('analyze').toggleAttribute('disabled', true);
  selectEl('lc0RuntimeSelect').disabled = true;
  el('backend').textContent = `loading ${lc0RuntimeLabel(runtime)}…`;
  el('message').textContent = `Loading LC0 ${lc0RuntimeLabel(runtime)} in a worker…`;
  showModelProgress(`LC0 ${lc0RuntimeLabel(runtime)}`, undefined, undefined, 'Preparing');
  try {
    el('backend').textContent = await initWorker();
    if (isStaleMount(mountSignal)) return false;
    hideModelProgress();
    el('analyze').toggleAttribute('disabled', false);
    selectEl('lc0RuntimeSelect').disabled = false;
    el('message').textContent = 'Ready. Drag a move, load a PGN/FEN, or Analyze. Navigation stays responsive.';
    if (runAutoAnalyze && inputEl('autoAnalyze').checked) void analyzeCurrent();
    return true;
  } catch (workerError) {
    if (isStaleMount(mountSignal)) return false;
    if (runtime !== 'onnx' && runtime !== LC0_WHOLE_MODEL_WEBGPU_RUNTIME) {
      selectEl('lc0RuntimeSelect').disabled = false;
      el('message').textContent = `LC0 ${lc0RuntimeLabel(runtime)} load failed: ${(workerError as Error).message}`;
      hideModelProgress();
      return false;
    }
    // Fall back to a main-thread evaluator (analysis will block the UI, but works).
    console.warn('LC0 worker init failed; falling back to the main thread.', workerError);
    try {
      let nextEvaluator: Lc0EvaluationProvider | null = null;
      if (runtime === LC0_WHOLE_MODEL_WEBGPU_RUNTIME) {
        nextEvaluator = await Lc0WholeOnnxWebgpuEvaluator.create({
          manifestUrl: LC0_WHOLE_MODEL_MANIFEST_URL,
          batch: lc0WholeModelPhysicalBatch(),
          fetchTensorCache: lc0WholeModelTensorCache(),
          logger: (line) => console.info('[lc0 whole-model analysis]', line),
        });
      } else {
        const modelLoad = await loadLc0ModelForOrt(MODEL_URL, {
          cache: false,
          onProgress: (loaded, total) => showModelProgress('Lc0 small net', loaded, total, 'Downloading'),
        });
        nextEvaluator = await Lc0OnnxEvaluator.create(modelLoad.model);
      }
      const nextSearcher = new Lc0PuctSearcher(nextEvaluator);
      const diagnostics = runtime === 'onnx' ? await collectOrtRuntimeDiagnostics() : undefined;
      if (isStaleMount(mountSignal)) {
        void nextEvaluator.dispose?.();
        return false;
      }
      mainEvaluator = nextEvaluator;
      searcher = nextSearcher;
      el('backend').textContent = `${diagnostics?.describe ?? 'whole-onnx-webgpu'} (main thread)`;
      publishBrowserRuntimeAudit({
        source: 'lc0-analysis-main-thread-fallback',
        surface: 'analysis',
        family: 'lc0',
        engineLabel: 'LC0',
        modelId: 'lc0-default',
        modelUrl: runtime === LC0_WHOLE_MODEL_WEBGPU_RUNTIME ? LC0_WHOLE_MODEL_MANIFEST_URL : MODEL_URL,
        requestedRuntime: runtime,
        resolvedRuntime: runtime === LC0_WHOLE_MODEL_WEBGPU_RUNTIME ? 'whole-onnx-webgpu-main-thread-fallback' : 'ort-main-thread-fallback',
        manifestUrl: runtime === LC0_WHOLE_MODEL_WEBGPU_RUNTIME ? LC0_WHOLE_MODEL_MANIFEST_URL : undefined,
        fallbackReason: (workerError as Error).message,
        searchBudget: `multipv=${multiPv()}`,
        notes: [diagnostics?.describe ?? 'whole-model runtime is research-only and opt-in'],
      });
      el('analyze').toggleAttribute('disabled', false);
      selectEl('lc0RuntimeSelect').disabled = false;
      el('message').textContent = 'Ready (main-thread fallback — deep analysis may pause the UI).';
      hideModelProgress();
      if (!isStaleMount(mountSignal) && runAutoAnalyze && inputEl('autoAnalyze').checked) void analyzeCurrent();
      return true;
    } catch (error) {
      if (isStaleMount(mountSignal)) return false;
      selectEl('lc0RuntimeSelect').disabled = false;
      el('message').textContent = `Model load failed: ${(error as Error).message}`;
      hideModelProgress();
      return false;
    }
  }
}

async function reloadLc0Backend(forceAnalyzeAfterLoad = false): Promise<void> {
  lineCache.clear();
  completeAnalysisKeys.clear();
  disposeRuntimeResources();
  renderRecklessRuntimeInfo();
  const loaded = await loadLc0Backend(!forceAnalyzeAfterLoad);
  if (loaded && forceAnalyzeAfterLoad) void analyzeCurrent();
}

async function init(mountSignal: AbortSignal) {
  if (!isV0DeployProfile()) {
    REQUESTED_RECKLESS_VARIANT = await resolveDefaultRecklessVariantAssetFallback(REQUESTED_RECKLESS_VARIANT, REQUESTED_RECKLESS_EXPLICIT, renderRecklessRuntimeInfo);
  }
  REQUESTED_VIRIDITHAS_VARIANT = await resolveDefaultViridithasVariantAssetFallback(REQUESTED_VIRIDITHAS_VARIANT, REQUESTED_VIRIDITHAS_EXPLICIT, renderRecklessRuntimeInfo);
  REQUESTED_BERSERK_VARIANT = await resolveDefaultBerserkVariantAssetFallback(REQUESTED_BERSERK_VARIANT, REQUESTED_BERSERK_EXPLICIT, renderRecklessRuntimeInfo);
  REQUESTED_PLENTYCHESS_VARIANT = await resolveDefaultPlentyChessVariantAssetFallback(REQUESTED_PLENTYCHESS_VARIANT, REQUESTED_PLENTYCHESS_EXPLICIT, renderRecklessRuntimeInfo);
  if (isStaleMount(mountSignal)) return;
  if (analysisPagehideHandler) window.removeEventListener('pagehide', analysisPagehideHandler);
  analysisPagehideHandler = (event: PageTransitionEvent) => {
    if (!event.persisted) {
      for (const key of ANALYSIS_BIG_NET_KEYS) disposeBigNetSearcherNow(key);
      disposeRuntimeResources();
    }
  };
  window.addEventListener('pagehide', analysisPagehideHandler);
  engineProfiles = loadEngineProfiles();
  installExperimentalLc0RuntimeOption();
  selectEl('lc0RuntimeSelect').value = initialLc0Runtime();
  const storedLastProfile = engineProfiles.find((profile) => profile.name === storageGet(LAST_ENGINE_PROFILE_STORAGE_KEY));
  // Do not silently restore saved BT4 rows on page load: selecting that profile
  // later goes through the explicit support check and large-download prompt.
  const lastProfile = storedLastProfile && !profileHasBt4(storedLastProfile.rows) ? storedLastProfile : undefined;
  if (lastProfile) {
    engineRows = profileRowsForUse(lastProfile.rows, false);
    inputEl('multiPvInput').value = String(lastProfile.multiPv);
    selectEl('lc0RuntimeSelect').value = lastProfile.lc0Runtime;
  }
  renderEngineProfiles(lastProfile?.name ?? '');
  installRuntimeAuditPanel();
  renderAll();
  renderEngineList();
  renderRecklessRuntimeInfo();
  if (!isV0DeployProfile()) void probeEngineLogos(() => { renderEngineList(); renderAll(); });
  wireEvents();
  void refreshPgnDatabaseCollections();
  if (!isV0DeployProfile()) {
    void refreshBt4Availability();
    void refreshTinyHybridManifestStatus();
  }
  await loadLc0Backend(true, mountSignal);
}

export function mountAnalysisBrowser(): () => void {
  const controller = new AbortController();
  mountAbort = controller;
  // Test hook for automated browser checks: synthetic chessground drags are
  // unreliable, so smokes call this to route through the real user-move path.
  const hook = (from: string, to: string) => { void onUserMove(from as Key, to as Key); };
  (globalThis as unknown as { __analysisUserMove?: (from: string, to: string) => void }).__analysisUserMove = hook;
  void init(controller.signal);
  return () => {
    controller.abort();
    if (mountAbort === controller) disposePageResources();
    // Only clear the test hook if it is still ours (a newer mount may have
    // installed its own).
    const g = globalThis as unknown as { __analysisUserMove?: (from: string, to: string) => void };
    if (g.__analysisUserMove === hook) delete g.__analysisUserMove;
  };
}
