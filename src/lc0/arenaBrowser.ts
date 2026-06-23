import { Chessground } from 'chessground';
import type { DrawShape } from 'chessground/draw';
import type { Key } from 'chessground/types';
import { boardToFen, parseFen, START_FEN, type BoardState } from '../chess/board.ts';
import { boardCheck } from './boardUx.ts';
import { legalMoves, makeMove } from '../chess/movegen.ts';
import { moveToUci, type Move } from '../chess/moveCodec.ts';
import { gameTreeToPgn } from '../chess/pgn.ts';
import { BUILTIN_ARENA_OPENINGS, parseArenaOpenings, type ArenaOpening } from './arenaOpenings.ts';
import { gameOutcome, type GameResultCode } from './engineBattle.ts';
import { GameTree } from './gameTree.ts';
import { loadLc0ModelForOrt } from './modelCache.ts';
import { collectOrtRuntimeDiagnostics } from '../nn/ortRuntime.ts';
import { CachedEvaluator, type Evaluator } from '../nn/evaluator.ts';
import { createBrowserSquareformerRuntimeEvaluator } from '../nn/browserRuntimeEvaluator.ts';
import { BROWSER_RUNTIME_AUDIT_EVENT, formatBrowserRuntimeAudit, publishBrowserRuntimeAudit, type BrowserRuntimeAuditDetail } from '../nn/runtimeAudit.ts';
import { chooseMove, montyLitePuctPolicy, type SearchResult as TinySearchResult } from '../search/puct.ts';
import { CachedLc0Evaluator, Lc0OnnxEvaluator, type Lc0Evaluation, type Lc0EvaluationCacheFootprint, type Lc0EvaluationCacheMetrics } from './onnxEvaluator.ts';
import { Lc0PolicyOnlyPlayer } from './policyOnlyPlayer.ts';
import { Lc0PuctSearcher, type Lc0SearchResult } from './search.ts';
import { Lc0WebHybridEvaluator, type Lc0WebEncoderKernelVariant, type Lc0WebExecutionFootprint } from './wgslMatmulAddProbe.ts';
import { Lc0WholeOnnxWebgpuEvaluator } from './wholeOnnxWebgpuEvaluator.ts';
import type { Node as PuctNode } from '../search/puct.ts';
import { StockfishEngine, stockfishFlavorLabel, stockfishFlavorUrl, type StockfishFlavor, type StockfishInfoLine } from './stockfishEngine.ts';
import { RecklessEngine, formatRecklessBrowserApiLoadStatus } from './recklessEngine.ts';
import { RECKLESS_VARIANTS, checkRecklessVariantAsset, hasExplicitRecklessVariant, recklessVariantAssetStatus, recklessVariantByKey, recklessVariantFromParams, normalizeRecklessVariant, resolveDefaultRecklessVariantAssetFallback, supportsWasmRelaxedSimd, type RecklessVariant } from './recklessVariants.ts';
import { ViridithasEngine, canUsePersistentViridithasWasi } from './viridithasEngine.ts';
import { VIRIDITHAS_VARIANTS, checkViridithasVariantAsset, hasExplicitViridithasVariant, normalizeViridithasVariant, resolveDefaultViridithasVariantAssetFallback, viridithasVariantAssetStatus, viridithasVariantByKey, viridithasVariantFromParams, type ViridithasVariant } from './viridithasVariants.ts';
import { BerserkEngine } from './berserkEngine.ts';
import { BERSERK_VARIANTS, berserkVariantAssetStatus, berserkVariantByKey, berserkVariantFromParams, checkBerserkVariantAsset, hasExplicitBerserkVariant, normalizeBerserkVariant, resolveDefaultBerserkVariantAssetFallback, type BerserkVariant } from './berserkVariants.ts';
import { PlentyChessEngine } from './plentychessEngine.ts';
import { berserkCacheKey, createBerserkEngine, createPlentyChessEngine, createRecklessEngine, createViridithasEngine, plentyChessCacheKey, recklessCacheKey, viridithasCacheKey } from './engineProvision.ts';
import { PLENTYCHESS_VARIANTS, checkPlentyChessVariantAsset, hasExplicitPlentyChessVariant, normalizePlentyChessVariant, plentyChessVariantAssetStatus, plentyChessVariantByKey, plentyChessVariantFromParams, plentyChessVariantUnsupportedReason, resolveDefaultPlentyChessVariantAssetFallback, type PlentyChessVariant } from './plentychessVariants.ts';
import { BIG_NETS, Bt4WorkerSearcher, T3_NET, bigNetAssetStatusSync, bigNetLoadWarning, bt4SupportedSync, checkBigNetAsset, probeBt4Support, type BigNetConfig, type Bt4SearchResult } from './bt4Engine.ts';
import { TournamentStandings, buildSchedule, tournamentPairings, type ScheduledGame, type TournamentMode } from './tournament.ts';
import { hBarChartSvg, lineChartSvg, type ChartSeries } from './charts.ts';
import { defaultStaticEngineVariant, engineFamilyOptions, engineResourceProfile, engineStrengthMeta, isEngineFamily, isLc0BigNetVariant, isV0DeployProfile, lc0EngineLabel, lc0VariantOptions, normalizeDeployEngineRow, stockfishEngineLabel, stockfishVariantOptions, tinyEngineLabel, tinyVariantOptions, type EngineFamily, type EngineRow } from './engineCatalog.ts';
import { EngineResourceBroker, loadPerformanceDial, type PerformanceDial } from './resourceBroker.ts';
import { resolvePublicAssetUrl } from './assetUrls.ts';

type Ground = ReturnType<typeof Chessground>;
// Seats are array indices into seatRows; tournament pids are String(index).
interface ArenaEngine {
  id: string;
  name: string;
  move(positions: BoardState[], signal: AbortSignal): Promise<string | null>;
  warmup?(signal: AbortSignal): Promise<void>;
}
interface GameRecord { pgn: string; }
interface MatchScore { a: number; b: number; aWins: number; bWins: number; draws: number; games: number; }
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
  totalElapsedMs?: number;
  lastElapsedMs?: number;
  evalBackendTimingSamples?: number;
  evalBackendTimingPositions?: number;
  evalBackendTimingTotals?: Record<string, number>;
  evalBackendTimingMeans?: Record<string, number>;
  evalBackendTimingPerPositionMeans?: Record<string, number>;
  lastBackendTimingMeans?: Record<string, number>;
  lastBackendTimingPerPositionMeans?: Record<string, number>;
  lastBatchSize?: number;
  lastBatchPipelineDepth?: number;
  maxEvalBatch?: number;
  evalBatchSizeHistogram?: Record<string, number>;
}
interface UciEngineTelemetry {
  engineName: string;
  searches: number;
  totalNodes: number;
  lastDepth?: number;
  lastNodes?: number;
  lastNps?: number;
  lastPvMoves?: number;
  lastMultiPv?: number;
  totalElapsedMs?: number;
  lastElapsedMs?: number;
}
interface Bt4Telemetry {
  engineName: string;
  searches: number;
  completedVisits: number;
  evalCalls: number;
  cacheHits: number;
  totalElapsedMs?: number;
  lastElapsedMs?: number;
}
interface PendingLc0ReplyProbe {
  engineId: string;
  engineName: string;
  afterMoveFen: string;
  child: PuctNode | null;
}
interface EngineEvalBar {
  /** White expected score, with 0.5 as equal. Displayed board-oriented: black at top, white at bottom. */
  whiteScore: number;
  label: string;
}
interface EngineOutputSnapshot {
  engineId: string;
  engineName: string;
  kind: 'lc0' | 'uci';
  fen: string;
  move?: string;
  summary: string;
  shortEval: string;
  evalBar?: EngineEvalBar;
  detail?: string;
  pv?: string[];
  /** UCI score converted to White's perspective, in centipawns. Positive is good for White. */
  whiteCp?: number;
  /** UCI mate score converted to White's perspective. Positive means White mates. */
  mateInWhitePov?: number;
  depth?: number;
  nodes?: number;
  nps?: number;
  elapsedMs?: number;
}

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
const TINY_MODEL_URL = params.get('tinyModel') ?? params.get('tinyOnnx') ?? DEFAULT_TINY_MODEL_URL;
const TINY_META_URL = params.get('tinyMeta') ?? DEFAULT_TINY_META_URL;
const TINY_HYBRID_MANIFEST_URL = params.get('tinyManifest') ?? params.get('manifest') ?? params.get('manifestUrl') ?? DEFAULT_TINY_HYBRID_MANIFEST_URL;
const LC0_WHOLE_MODEL_MANIFEST_URL = params.get('wholeModelManifest') ?? params.get('wholeModelManifestUrl') ?? params.get('tvm' + 'jsManifest') ?? DEFAULT_LC0_WHOLE_MODEL_MANIFEST_URL;
type Lc0ArenaRuntime = 'onnx' | 'hybrid-ort-heads' | 'hybrid-wgsl-heads' | typeof LC0_WHOLE_MODEL_WEBGPU_RUNTIME;
type Lc0ArenaPreset = 'stable' | 'benchmarked-small' | 'custom';
const REQUESTED_RECKLESS_EXPLICIT = hasExplicitRecklessVariant(params);
let REQUESTED_RECKLESS_VARIANT = recklessVariantFromParams(params);
const REQUESTED_VIRIDITHAS_EXPLICIT = hasExplicitViridithasVariant(params);
let REQUESTED_VIRIDITHAS_VARIANT = viridithasVariantFromParams(params);
const REQUESTED_BERSERK_EXPLICIT = hasExplicitBerserkVariant(params);
let REQUESTED_BERSERK_VARIANT = berserkVariantFromParams(params);
const REQUESTED_PLENTYCHESS_EXPLICIT = hasExplicitPlentyChessVariant(params);
let REQUESTED_PLENTYCHESS_VARIANT = plentyChessVariantFromParams(params);

let ground: Ground | null = null;
let arenaKeydownHandler: ((event: KeyboardEvent) => void) | null = null;
let board: BoardState = parseFen(START_FEN);
let historyBoards: BoardState[] = [board];
let lastUci: string | null = null;

// Game-history review: clicking a chart, the move strip, or a finished game in
// the log flips the board into a read-only replay; "Live" returns to the
// running game (which keeps playing in the background meanwhile).
interface TrailEntry { fen: string; uci: string | null; san: string | null }
interface GameTrail { label: string; entries: TrailEntry[]; openingPlies: number }
let liveTrail: GameTrail | null = null;
const finishedTrails: GameTrail[] = [];
let reviewTrail: GameTrail | null = null;
let reviewIndex = 0;
let reviewing = false;
/** Candidate-move arrows shown instead of the last-move arrow while reviewing a search. */
let reviewShapes: DrawShape[] | null = null;
/** What the root-visits chart currently shows, for click-to-board. */
let rootChartContext: { fen: string; top: { uci: string; visits: number }[] } | null = null;
let boardWhiteId: string | null = null;
let boardBlackId: string | null = null;
let boardWhiteName: string | null = null;
let boardBlackName: string | null = null;
let loadingLc0 = false;
let running = false;
let startPending = false;
let abort: AbortController | null = null;
let player: Lc0PolicyOnlyPlayer | null = null;
let searcher: Lc0PuctSearcher | null = null;
let lc0Cache: CachedLc0Evaluator | null = null;
let stockfishLite: StockfishEngine | null = null;
let stockfishFull: StockfishEngine | null = null;
const recklessByVariant = new Map<string, RecklessEngine>();
const viridithasByVariant = new Map<string, ViridithasEngine>();
const berserkByVariant = new Map<string, BerserkEngine>();
const plentyChessByVariant = new Map<string, PlentyChessEngine>();
const tinyEvaluatorPromises = new Map<string, Promise<Evaluator>>();
let tinyHybridManifestStatus: 'unknown' | 'present' | 'missing' = 'unknown';
// Lc0 BT4 runs in its own worker (lazy, WebGPU-gated, disposable). See bt4Engine.ts.
const bt4 = new Bt4WorkerSearcher();
const t3BigNet = new Bt4WorkerSearcher(T3_NET);
const bigNetSearchers: Record<'bt4' | 't3', Bt4WorkerSearcher> = { bt4, t3: t3BigNet };
function bigNetFor(variant: string): { config: BigNetConfig; searcher: Bt4WorkerSearcher } {
  const key = variant === 't3' ? 't3' : 'bt4';
  return { config: BIG_NETS[key], searcher: bigNetSearchers[key] };
}
let runtimeIsolation = typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated;
let runtimeSharedArrayBuffer = typeof SharedArrayBuffer !== 'undefined';

function initialPerformanceDial(): PerformanceDial {
  const raw = params.get('perfDial');
  if (raw === 'eco' || raw === 'balanced' || raw === 'max') return raw;
  try {
    return loadPerformanceDial(typeof localStorage !== 'undefined' ? localStorage : undefined);
  } catch {
    return 'balanced';
  }
}

// Arena turn-taking: the engine on move leases the full CPU budget; the idle
// side holds zero threads. Threaded SF play stays opt-in via the threads cap
// input (0 = auto/full broker grant, default 1 preserves rating-lane parity).
const resourceBroker = new EngineResourceBroker({ policy: 'exclusive', dial: initialPerformanceDial() });
resourceBroker.register('sf-lite', { ...engineResourceProfile('sf') });
resourceBroker.register('sf-full', { ...engineResourceProfile('sf') });
const engines = new Map<string, ArenaEngine>();
const lc0Searchers = new Map<string, Lc0PuctSearcher>();
const lastLc0SearchResults = new Map<string, Lc0SearchResult>();
interface GameChartSample { ply: number; engineId: string; whiteScore?: number; moveMs: number; nps?: number; }
let gameChartSamples: GameChartSample[] = [];
const CHART_COLORS = ['#4a7a2a', '#a5461b', '#1c5f8a', '#7a4a9a', '#8a7a1c', '#5a5a5a'];
const pendingLc0ReplyProbes = new Map<string, PendingLc0ReplyProbe>();
const lc0TreeTelemetry = new Map<string, Lc0TreeTelemetry>();
const uciTelemetry = new Map<string, UciEngineTelemetry>();
const bt4Telemetry = new Map<string, Bt4Telemetry>();
const engineOutputs = new Map<string, EngineOutputSnapshot>();
const engineOutputHistory: EngineOutputSnapshot[] = [];
let engineOutputTotalCount = 0;
const MAX_ENGINE_OUTPUT_HISTORY = 1000;
const thinkingEngineIds = new Set<string>();
let activeEngineIds: string[] = [];
const games: GameRecord[] = [];
const seatRows: EngineRow[] = [
  { family: 'lc0', variant: 'small', strength: 100 },
  { family: 'sf', variant: 'lite', strength: 8 },
];

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
    for (const row of activeSeatRows()) {
      if (row.family === 'tiny' && row.variant === 'bt4-custom') row.variant = 'bt4-auto';
    }
  }
  void renderRuntimeBadge();
}

async function tinyEvaluator(variant: string): Promise<Evaluator> {
  const runtime = tinyRuntimeForVariant(variant);
  const fallback = tinyRuntimeFallbackForVariant(variant);
  const key = `${runtime}:${fallback ? 'fallback' : 'strict'}:${TINY_MODEL_URL}:${TINY_META_URL}`;
  const existing = tinyEvaluatorPromises.get(key);
  if (existing) return existing;
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
      params,
      runtime,
      manifestUrl: TINY_HYBRID_MANIFEST_URL,
      fallback,
      audit: { surface: 'arena', searchBudget: 'visits=seat strength' },
    });
    console.info('[lc0-arena] loaded Tiny Leela evaluator', {
      requestedRuntime: loaded.requestedRuntime,
      resolvedRuntime: loaded.resolvedRuntime,
      runtimeConfigId: loaded.runtimeConfigId,
      manifestUrl: loaded.manifestUrl,
      fallbackReason: loaded.fallbackReason,
    });
    return new CachedEvaluator(loaded.evaluator, { maxEntries: arenaCacheEntries(), includeHistory: true, includeLegalMoves: true, label: `tiny-leela-arena:${runtime}` });
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

function normalizeLc0Runtime(value: string | null): Lc0ArenaRuntime {
  if (isV0DeployProfile()) return 'onnx';
  const raw = (value ?? '').toLowerCase();
  if (raw === LC0_WHOLE_MODEL_WEBGPU_RUNTIME || raw === 'tvm' + 'js-webgpu' || raw === 'lc0-' + 'tvm' + 'js-webgpu') return LC0_WHOLE_MODEL_WEBGPU_RUNTIME;
  if (raw === 'hybrid' || raw === 'lc0web' || raw === 'hybrid-ort-heads' || raw === 'wgsl-encoder') return 'hybrid-ort-heads';
  if (raw === 'hybrid-wgsl-heads' || raw === 'wgsl-heads' || raw === 'wgsl') return 'hybrid-wgsl-heads';
  return 'onnx';
}

function initialLc0Runtime(): Lc0ArenaRuntime {
  if (isV0DeployProfile()) return 'onnx';
  if (params.get('headBackend') === 'wgsl' || params.get('hybridHeads') === 'wgsl') return 'hybrid-wgsl-heads';
  return normalizeLc0Runtime(params.get('lc0Runtime') ?? params.get('runtime'));
}

function selectedLc0Runtime(): Lc0ArenaRuntime {
  return normalizeLc0Runtime(selectEl('lc0RuntimeSelect').value);
}

function lc0WholeModelRuntimeRequested(): boolean {
  if (isV0DeployProfile()) return false;
  return normalizeLc0Runtime(params.get('lc0Runtime') ?? params.get('runtime')) === LC0_WHOLE_MODEL_WEBGPU_RUNTIME
    || params.get('enableWholeModelWebgpu') === '1'
    || params.get('enableTvm' + 'js') === '1';
}

function installExperimentalLc0RuntimeOption(_force = false): void {
  // Promoted 2026-06-10 (release-owner decision): the whole-model WebGPU
  // runtime is always listed. ORT remains the default and the fallback.
  const select = selectEl('lc0RuntimeSelect');
  if ([...select.options].some((option) => option.value === LC0_WHOLE_MODEL_WEBGPU_RUNTIME)) return;
  const option = document.createElement('option');
  option.value = LC0_WHOLE_MODEL_WEBGPU_RUNTIME;
  option.textContent = 'TVM whole-model WebGPU (fast, small net)';
  select.appendChild(option);
}

function normalizeLc0Preset(value: string | null): Lc0ArenaPreset {
  return value === 'benchmarked-small' || value === 'custom' ? value : 'stable';
}

function inferredLc0Preset(): Lc0ArenaPreset {
  if (isV0DeployProfile()) return 'stable';
  const explicit = params.get('lc0Preset') ?? params.get('preset');
  if (explicit) return normalizeLc0Preset(explicit);
  const requestedRuntime = normalizeLc0Runtime(params.get('lc0Runtime') ?? params.get('runtime'));
  if (requestedRuntime === LC0_WHOLE_MODEL_WEBGPU_RUNTIME || lc0WholeModelRuntimeRequested()) return 'benchmarked-small';
  if (requestedRuntime !== 'onnx' || params.has('inputBackend') || params.has('encoderKernel') || params.has('legalPriorsBackend') || params.has('lc0BatchSize') || params.has('batchPipelineDepth')) return 'custom';
  return 'stable';
}

function selectedLc0Preset(): Lc0ArenaPreset {
  return normalizeLc0Preset(selectEl('lc0PresetSelect').value);
}

function setLc0PresetNote(preset = selectedLc0Preset()): void {
  const note = el('lc0PresetNote');
  if (preset === 'benchmarked-small') {
    note.textContent = `Fast preset: LC0 Small via whole-model WebGPU research path, compiled batch ${lc0WholeModelPhysicalBatch()}, search batch ${lc0WholeModelPhysicalBatch()}, pipeline depth 1. Arena budget is set to equal movetime so eval speed can affect strength.`;
  } else if (preset === 'custom') {
    note.textContent = 'Custom mode: advanced runtime knobs are open. Only use these for experiments or reproducing old benchmark cells.';
  } else {
    note.textContent = 'Stable default: LC0 Small via ORT with WebGPU/WASM fallback. Use the fast preset to test benchmarked eval speed in fixed-time arena games.';
  }
}

function applyLc0Preset(preset: Lc0ArenaPreset, options: { reload?: boolean } = {}): void {
  selectEl('lc0PresetSelect').value = preset;
  const advanced = el('lc0AdvancedRuntime') as HTMLDetailsElement;
  if (preset === 'benchmarked-small') {
    installExperimentalLc0RuntimeOption(true);
    selectEl('lc0RuntimeSelect').value = LC0_WHOLE_MODEL_WEBGPU_RUNTIME;
    inputEl('lc0BatchSizeInput').value = String(lc0WholeModelPhysicalBatch());
    inputEl('lc0BatchPipelineDepthInput').value = '1';
    selectEl('lc0InputBackendSelect').value = 'js';
    selectEl('lc0EncoderKernelSelect').value = 'hand';
    selectEl('lc0LegalPriorsSelect').value = 'js';
    selectEl('budgetModeSelect').value = 'movetime';
    advanced.open = false;
  } else if (preset === 'stable') {
    selectEl('lc0RuntimeSelect').value = 'onnx';
    inputEl('lc0BatchSizeInput').value = '1';
    inputEl('lc0BatchPipelineDepthInput').value = '1';
    selectEl('lc0InputBackendSelect').value = 'js';
    selectEl('lc0EncoderKernelSelect').value = 'hand';
    selectEl('lc0LegalPriorsSelect').value = 'js';
    advanced.open = false;
  } else {
    advanced.open = true;
    const container = advanced.closest('details.advanced-settings') as HTMLDetailsElement | null;
    if (container) container.open = true;
  }
  setLc0PresetNote(preset);
  refreshBudgetControls();
  refreshSeatControls();
  renderCacheInfo();
  resetLc0SearchTrees();
  if (options.reload && !running) void reloadLc0Evaluator();
}

function markLc0PresetCustom(): void {
  if (selectedLc0Preset() === 'custom') return;
  selectEl('lc0PresetSelect').value = 'custom';
  const advanced = el('lc0AdvancedRuntime') as HTMLDetailsElement;
  advanced.open = true;
  const container = advanced.closest('details.advanced-settings') as HTMLDetailsElement | null;
  if (container) container.open = true;
  setLc0PresetNote('custom');
}

function lc0ResolvedRuntime(runtime: Lc0ArenaRuntime): string {
  if (runtime === 'onnx') return 'ort-main-thread';
  if (runtime === LC0_WHOLE_MODEL_WEBGPU_RUNTIME) return 'whole-onnx-webgpu-research';
  return `${runtime}-lazy`;
}

function installRuntimeAuditPanel(): void {
  window.addEventListener(BROWSER_RUNTIME_AUDIT_EVENT, (event) => {
    const detail = (event as CustomEvent<BrowserRuntimeAuditDetail>).detail;
    if (detail.family !== 'lc0') return;
    const target = document.getElementById('runtimeAuditInfo');
    if (!target) return;
    target.innerHTML = diagBlockHtml('LC0 audit', [htmlEscape(formatBrowserRuntimeAudit(detail))]);
  });
}

function lc0RuntimeLabel(runtime = selectedLc0Runtime()): string {
  if (runtime === LC0_WHOLE_MODEL_WEBGPU_RUNTIME) return 'TVM whole-model WebGPU (research)';
  if (runtime === 'hybrid-wgsl-heads') return 'WGSL encoder + WGSL heads';
  if (runtime === 'hybrid-ort-heads') return 'WGSL encoder + ORT heads';
  return 'ORT ONNX';
}

function normalizeLc0InputBackend(value: string | null): 'js' | 'wgsl' | 'wasm' {
  return value === 'wgsl' || value === 'wasm' ? value : 'js';
}

function lc0HybridInputBackend(): 'js' | 'wgsl' | 'wasm' {
  const node = document.getElementById('lc0InputBackendSelect') as HTMLSelectElement | null;
  return normalizeLc0InputBackend(node?.value ?? params.get('inputBackend') ?? params.get('lc0InputBackend'));
}

function normalizeLc0LegalPriorsBackend(value: string | null): 'js' | 'wasm' | 'gpu' {
  return value === 'wasm' || value === 'gpu' ? value : 'js';
}

function lc0HybridLegalPriorsBackend(): 'js' | 'wasm' | 'gpu' {
  const node = document.getElementById('lc0LegalPriorsSelect') as HTMLSelectElement | null;
  return normalizeLc0LegalPriorsBackend(node?.value ?? params.get('legalPriorsBackend') ?? params.get('lc0LegalPriorsBackend') ?? params.get('hybridLegalPriors'));
}

function lc0EncoderLayers(): number {
  return Math.min(32, Math.max(1, Math.floor(Number(params.get('encoderLayers') ?? params.get('layers') ?? '10') || 10)));
}

function normalizeLc0EncoderKernelVariant(value: string | null): Lc0WebEncoderKernelVariant {
  return value === 'tvm-packed-f16' || value === 'mixed-tvm-ffn' || value === 'mixed-tvm-ffn-outproj' || value === 'mixed-tvm-ffn-smolgen-project' ? value : 'hand';
}

function lc0EncoderKernelVariant(): Lc0WebEncoderKernelVariant {
  const node = document.getElementById('lc0EncoderKernelSelect') as HTMLSelectElement | null;
  return normalizeLc0EncoderKernelVariant(node?.value ?? params.get('encoderKernel') ?? params.get('lc0EncoderKernel') ?? params.get('encoderKernelVariant'));
}

function lc0HybridConfigLabel(runtime = selectedLc0Runtime()): string {
  if (runtime === 'onnx') return '';
  if (runtime === LC0_WHOLE_MODEL_WEBGPU_RUNTIME) return `manifest ${LC0_WHOLE_MODEL_MANIFEST_URL} · physical batch ${lc0WholeModelPhysicalBatch()}${lc0WholeModelTensorCache() ? ' · tensor cache' : ''}`;
  const legal = lc0HybridLegalPriorsBackend();
  const effectiveLegal = legal === 'gpu' && runtime !== 'hybrid-wgsl-heads' ? 'js' : legal;
  return `input ${lc0HybridInputBackend()} · encoder ${lc0EncoderKernelVariant()} · legal ${effectiveLegal}`;
}

function boundedIntValue(value: unknown, fallback: number, min: number, max: number): number {
  const parsed = Math.floor(Number(value ?? fallback));
  return Number.isFinite(parsed) ? Math.max(min, Math.min(max, parsed)) : fallback;
}

function lc0BatchSize(): number {
  const node = document.getElementById('lc0BatchSizeInput') as HTMLInputElement | null;
  return boundedIntValue(node?.value ?? params.get('lc0BatchSize') ?? params.get('batchSize') ?? params.get('batch'), 1, 1, 64);
}

function lc0BatchPipelineDepth(): number {
  const node = document.getElementById('lc0BatchPipelineDepthInput') as HTMLInputElement | null;
  return boundedIntValue(node?.value ?? params.get('lc0BatchPipelineDepth') ?? params.get('batchPipelineDepth'), 1, 1, 16);
}

function lc0WholeModelPhysicalBatch(): number {
  return boundedIntValue(params.get('wholeModelBatch') ?? params.get('tvmBatch') ?? params.get('compiledBatch'), 8, 1, 64);
}

function lc0WholeModelTensorCache(): boolean {
  return params.get('wholeModelTensorCache') === '1' || params.get('tensorCache') === '1';
}

function intParam(name: string, fallback: number, min: number, max: number): number {
  const raw = params.get(name);
  if (raw == null) return fallback;
  const parsed = Math.floor(Number(raw));
  return Number.isFinite(parsed) ? Math.max(min, Math.min(max, parsed)) : fallback;
}

function parseSeatSpec(value: string | null): EngineRow | undefined {
  if (!value) return undefined;
  const [familyRaw, variantRaw, strengthRaw] = value.split(/[:;,]/).map((part) => part.trim());
  if (!isEngineFamily(familyRaw)) return undefined;
  const variant = variantRaw || defaultVariant(familyRaw);
  const strength = strengthRaw === undefined || strengthRaw === '' ? strengthMeta(familyRaw).def : Number(strengthRaw);
  const row: EngineRow = { family: familyRaw, variant, strength };
  clampStrength(row);
  return row;
}

function applyArenaQueryParams(): void {
  const requestedMode = params.get('arenaMode') ?? params.get('tournament');
  if (requestedMode === 'round-robin' || requestedMode === 'gauntlet' || requestedMode === 'match') {
    selectEl('tournamentModeSelect').value = requestedMode;
  }
  const seatA = parseSeatSpec(params.get('seatA') ?? params.get('engineA'));
  const seatB = parseSeatSpec(params.get('seatB') ?? params.get('engineB'));
  if (seatA) seatRows[0] = seatA;
  if (seatB) seatRows[1] = seatB;

  if (params.has('lc0Strength')) {
    seatRows[0].family = 'lc0';
    seatRows[0].variant = 'small';
    seatRows[0].strength = Number(params.get('lc0Strength'));
    clampStrength(seatRows[0]);
  }
  const opponentFamily = params.get('opponentFamily');
  if (opponentFamily && isEngineFamily(opponentFamily)) {
    seatRows[1].family = opponentFamily;
    seatRows[1].variant = params.get('opponentVariant') ?? defaultVariant(opponentFamily);
    seatRows[1].strength = Number(params.get('opponentStrength') ?? strengthMeta(opponentFamily).def);
    clampStrength(seatRows[1]);
  }
  normalizeSeatRowsForDeploy();

  inputEl('gamesInput').value = String(intParam('gamesPerOpening', intParam('games', Number(inputEl('gamesInput').value) || 2, 1, 20), 1, 20));
  inputEl('delayInput').value = String(intParam('delayMs', intParam('delay', Number(inputEl('delayInput').value) || 0, 0, 3000), 0, 3000));
  const budget = params.get('budgetMode') ?? params.get('budget');
  if (budget === 'movetime' || budget === 'fixed') selectEl('budgetModeSelect').value = budget;
  inputEl('movetimeInput').value = String(intParam('movetimeMs', intParam('movetime', Number(inputEl('movetimeInput').value) || 500, 10, 60000), 10, 60000));
  inputEl('cacheEntriesInput').value = String(intParam('cacheEntries', intParam('cache', Number(inputEl('cacheEntriesInput').value) || 2048, 0, 100000), 0, 100000));
  inputEl('stockfishThreadsInput').value = String(intParam('sfThreads', Number(inputEl('stockfishThreadsInput').value) || 1, 1, 32));
  inputEl('lc0BatchSizeInput').value = String(boundedIntValue(params.get('lc0BatchSize') ?? params.get('batchSize') ?? params.get('batch'), Number(inputEl('lc0BatchSizeInput').value) || 1, 1, 64));
  inputEl('lc0BatchPipelineDepthInput').value = String(boundedIntValue(params.get('lc0BatchPipelineDepth') ?? params.get('batchPipelineDepth'), Number(inputEl('lc0BatchPipelineDepthInput').value) || 1, 1, 16));
  selectEl('lc0InputBackendSelect').value = normalizeLc0InputBackend(params.get('inputBackend') ?? params.get('lc0InputBackend'));
  selectEl('lc0EncoderKernelSelect').value = normalizeLc0EncoderKernelVariant(params.get('encoderKernel') ?? params.get('lc0EncoderKernel') ?? params.get('encoderKernelVariant'));
  selectEl('lc0LegalPriorsSelect').value = normalizeLc0LegalPriorsBackend(params.get('legalPriorsBackend') ?? params.get('lc0LegalPriorsBackend') ?? params.get('hybridLegalPriors'));

  const suite = params.get('openingSuite') ?? params.get('openings') ?? params.get('startingPosition');
  if (suite === 'start' || suite === 'built-in' || suite === 'custom') selectEl('startingPositionSelect').value = suite;
  const openingText = params.get('openingText') ?? params.get('customOpenings');
  if (openingText != null) {
    selectEl('startingPositionSelect').value = 'custom';
    (el('openingText') as HTMLTextAreaElement).value = openingText;
  }
}

function percent(value: number): string {
  return `${(100 * value).toFixed(1)}%`;
}

function percent0(value: number): string {
  return `${Math.round(100 * value)}%`;
}

function signed(value: number, digits = 2): string {
  return `${value >= 0 ? '+' : ''}${value.toFixed(digits)}`;
}

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value));
}

function fenTurn(fen: string): 'w' | 'b' {
  return fen.split(/\s+/)[1] === 'b' ? 'b' : 'w';
}

function stmScoreToWhiteScore(fen: string, stmScore: number): number {
  const score = clamp01(stmScore);
  return fenTurn(fen) === 'w' ? score : 1 - score;
}

function pvText(pv: string[] | undefined, maxMoves = 8): string {
  if (!pv?.length) return 'PV —';
  const shown = pv.slice(0, maxMoves).join(' ');
  return `PV ${shown}${pv.length > maxMoves ? ' …' : ''}`;
}

// All engine scores are displayed from White's perspective (positive = good for
// White), matching the eval bar and standard chess GUIs, so the sign no longer
// flips with the side to move. Engines report from the side-to-move POV, so flip
// when Black is to move (WDL swaps win/loss; Q and cp negate).
function toWhiteWdl(wdl: [number, number, number], fen: string): [number, number, number] {
  return fenTurn(fen) === 'w' ? wdl : [wdl[2], wdl[1], wdl[0]];
}
function toWhiteQ(q: number, fen: string): number {
  return fenTurn(fen) === 'w' ? q : -q;
}

function stockfishScoreText(info: StockfishInfoLine | undefined, fen: string): string {
  if (!info) return 'score unavailable';
  const w = fenTurn(fen) === 'w' ? 1 : -1;
  if (info.mateIn !== undefined) return `mate ${signed(w * info.mateIn, 0)} · d${info.depth}`;
  if (info.scoreCp !== undefined) return `${signed(w * info.scoreCp / 100, 2)} · d${info.depth}`;
  return `score unavailable · d${info.depth}`;
}

function stockfishScoreCompact(info: StockfishInfoLine | undefined, fen: string): string {
  if (!info) return 'eval —';
  const w = fenTurn(fen) === 'w' ? 1 : -1;
  if (info.mateIn !== undefined) return `M${signed(w * info.mateIn, 0)} · d${info.depth}`;
  if (info.scoreCp !== undefined) return `${signed(w * info.scoreCp / 100, 2)} · d${info.depth}`;
  return `d${info.depth}`;
}

function stockfishWhiteCp(info: StockfishInfoLine | undefined, fen: string): number | undefined {
  return info?.scoreCp === undefined ? undefined : (fenTurn(fen) === 'w' ? info.scoreCp : -info.scoreCp);
}

function stockfishMateInWhitePov(info: StockfishInfoLine | undefined, fen: string): number | undefined {
  return info?.mateIn === undefined ? undefined : (fenTurn(fen) === 'w' ? info.mateIn : -info.mateIn);
}

function lc0WdlText(evaluation: Lc0Evaluation): string {
  const wdl = toWhiteWdl(evaluation.wdl, evaluation.fen);
  return `WDL ${percent(wdl[0])}/${percent(wdl[1])}/${percent(wdl[2])} · Q ${signed(toWhiteQ(evaluation.q, evaluation.fen), 3)} · MLH ${evaluation.mlh.toFixed(1)}`;
}

function lc0WdlCompact(wdl: [number, number, number], q: number, fen: string): string {
  const w = toWhiteWdl(wdl, fen);
  return `WDL ${percent0(w[0])}/${percent0(w[1])}/${percent0(w[2])} · Q ${signed(toWhiteQ(q, fen), 2)}`;
}

function lc0EvalBar(fen: string, wdl: [number, number, number]): EngineEvalBar {
  const stmExpectedScore = clamp01(wdl[0] + 0.5 * wdl[1]);
  const whiteScore = stmScoreToWhiteScore(fen, stmExpectedScore);
  return { whiteScore, label: percent0(whiteScore) };
}

function qEvalBar(fen: string, q: number): EngineEvalBar {
  const whiteScore = stmScoreToWhiteScore(fen, (clamp01((q + 1) / 2)));
  return { whiteScore, label: percent0(whiteScore) };
}

function stockfishEvalBar(fen: string, info: StockfishInfoLine | undefined): EngineEvalBar | undefined {
  if (!info) return undefined;
  const turn = fenTurn(fen);
  if (info.mateIn !== undefined) {
    const whiteMateSign = turn === 'w' ? info.mateIn : -info.mateIn;
    return { whiteScore: whiteMateSign > 0 ? 1 : 0, label: `M${Math.abs(whiteMateSign)}` };
  }
  if (info.scoreCp === undefined) return undefined;
  const whiteCp = turn === 'w' ? info.scoreCp : -info.scoreCp;
  const whiteScore = 1 / (1 + Math.exp(-whiteCp / 320));
  return { whiteScore: clamp01(whiteScore), label: signed(whiteCp / 100, 1) };
}

function searchWdlText(wdl: [number, number, number], q: number, fen: string): string {
  const w = toWhiteWdl(wdl, fen);
  return `WDL ${percent(w[0])}/${percent(w[1])}/${percent(w[2])} · search Q ${signed(toWhiteQ(q, fen), 3)}`;
}

function recordEngineOutput(snapshot: EngineOutputSnapshot): void {
  thinkingEngineIds.delete(snapshot.engineId);
  engineOutputs.set(snapshot.engineId, snapshot);
  engineOutputTotalCount += 1;
  engineOutputHistory.push(snapshot);
  if (engineOutputHistory.length > MAX_ENGINE_OUTPUT_HISTORY) engineOutputHistory.splice(0, engineOutputHistory.length - MAX_ENGINE_OUTPUT_HISTORY);
  renderSideLabels();
  renderEngineOutputs();
  renderEngineDiagnosticsInfo();
}

function shortEngineTag(name: string): string {
  const n = name.toLowerCase();
  if (n.includes('tiny leela')) return 'TL';
  if (n.includes('bt4')) return 'BT4';
  if (n.includes('lc0') || n.includes('leela')) return 'Lc0';
  if (n.includes('reckless')) return 'Reck';
  if (n.includes('viridithas')) return 'Viri';
  if (n.includes('stockfish') || /\bsf\b/.test(n)) return n.includes('lite') ? 'SF-L' : 'SF';
  return name.split(/[\s·|]+/)[0] || name;
}

// Optional favicon-sized engine logos (public/engine-logos/, not bundled by
// default). We probe once which files exist and only emit an <img> for those, so
// when a logo is absent the markup is unchanged — no per-render insert/remove jitter.
const availableEngineLogos = new Set<string>();

function engineLogoFamily(name: string): string {
  const n = name.toLowerCase();
  if (n.includes('tiny leela')) return '';
  if (n.includes('bt4') || n.includes('lc0') || n.includes('leela')) return 'lc0';
  if (n.includes('reckless')) return 'reckless';
  if (n.includes('viridithas')) return 'viridithas';
  if (n.includes('stockfish') || /\bsf\b/.test(n)) return 'stockfish';
  return '';
}

function engineLogoHtml(name: string): string {
  const family = engineLogoFamily(name);
  return family && availableEngineLogos.has(family) ? `<img class="engine-logo" src="/engine-logos/${family}.png" alt="">` : '';
}

async function probeEngineLogos(): Promise<void> {
  await Promise.all(['lc0', 'stockfish', 'reckless', 'viridithas'].map(async (family) => {
    try {
      const response = await fetch(`/engine-logos/${family}.png`, { method: 'HEAD', cache: 'no-store' });
      // Demand an image content-type: preview/SPA-fallback servers answer 200
      // text/html for missing files, which made every chip a broken <img>.
      if (response.ok && (response.headers.get('content-type') ?? '').startsWith('image/')) availableEngineLogos.add(family);
    } catch { /* absent */ }
  }));
  if (availableEngineLogos.size) renderSideLabels();
}

function renderEvalBars(): void {
  const render = (id: string, color: 'White' | 'Black', engineId: string | null, engineName: string | null) => {
    const node = el(id);
    const output = engineId ? engineOutputs.get(engineId) : undefined;
    const bar = output?.evalBar;
    const thinking = engineId ? thinkingEngineIds.has(engineId) : false;
    const whiteScore = clamp01(bar?.whiteScore ?? 0.5);
    const label = bar?.label ?? (thinking ? 'thinking…' : 'eval —');
    node.classList.toggle('empty', !bar);
    node.classList.toggle('thinking', thinking);
    node.title = `${color} engine: ${engineName ?? '—'}${bar ? ` · ${bar.label}` : ''}`;
    node.innerHTML = `<div class="eval-fill" style="height:${(100 * whiteScore).toFixed(1)}%"></div><div class="eval-midline"></div><div class="eval-bar-caption">${color[0]}</div><div class="eval-bar-value">${htmlEscape(label)}</div>`;
    const chip = document.getElementById(id.replace('EvalBar', 'Chip'));
    if (chip) {
      chip.innerHTML = engineName ? `${engineLogoHtml(engineName)}<span>${htmlEscape(shortEngineTag(engineName))}</span>` : '';
      chip.title = engineName ? `${color} engine: ${engineName}` : '';
      chip.style.display = engineName ? '' : 'none';
    }
  };
  render('whiteEngineEvalBar', 'White', boardWhiteId, boardWhiteName);
  render('blackEngineEvalBar', 'Black', boardBlackId, boardBlackName);
}

function renderEngineOutputs(): void {
  const ids = activeEngineIds.length ? activeEngineIds : [...new Set([seatEngineId(0), seatEngineId(1)])].filter((id) => engines.has(id));
  const cards = ids.map((id) => {
    const snapshot = engineOutputs.get(id);
    const name = snapshot?.engineName ?? engines.get(id)?.name ?? id;
    const thinking = thinkingEngineIds.has(id);
    // No per-ply "active" highlight here: at fast movetimes it strobes. Whose turn
    // it is is shown calmly by the last-move arrow on the board instead.
    if (!snapshot) return `<div class="eval-card"><strong>${htmlEscape(name)}</strong>${thinking ? 'thinking on current position…' : 'waiting for output…'}</div>`;
    const status = thinking ? '<span class="eval-status">thinking… keeping last eval</span>' : '';
    const detail = snapshot.detail ? `<br>${htmlEscape(snapshot.detail)}` : '';
    const pv = snapshot.pv?.length ? `<br>${htmlEscape(pvText(snapshot.pv))}` : '';
    const move = snapshot.move ? ` · move ${snapshot.move}` : '';
    return `<div class="eval-card"><strong>${htmlEscape(name)}${status}</strong>${htmlEscape(snapshot.summary)}${htmlEscape(move)}${detail}${pv}</div>`;
  });
  el('engineEvalInfo').innerHTML = cards.length ? cards.join('') : '<div class="eval-card">Engine outputs: waiting for a move…</div>';
}

function renderSideLabels() {
  // Side labels are static identity rows (chip + logo + engine + eval). Whose turn
  // it is is conveyed by the board's last-move arrow, not a per-ply highlight that
  // would strobe at fast movetimes.
  const update = (id: string, color: 'White' | 'Black', engineId: string | null, engineName: string | null) => {
    const node = el(id);
    const output = engineId ? engineOutputs.get(engineId) : undefined;
    const thinking = engineId ? thinkingEngineIds.has(engineId) : false;
    const evalText = output?.shortEval ?? (thinking ? 'thinking…' : 'eval —');
    node.classList.remove('active');
    node.innerHTML = `<span class="side-main"><span class="color">${color}</span> ${engineName ? engineLogoHtml(engineName) : ''}<span class="engine">${htmlEscape(engineName ?? '—')}</span> <span class="side-eval">${htmlEscape(evalText)}</span></span>`;
  };
  update('blackSideLabel', 'Black', boardBlackId, boardBlackName);
  update('whiteSideLabel', 'White', boardWhiteId, boardWhiteName);
  renderEvalBars();
}

function setBoardSideEngines(whiteId: string | null, whiteName: string | null, blackId: string | null, blackName: string | null): void {
  boardWhiteId = whiteId;
  boardBlackId = blackId;
  boardWhiteName = whiteName;
  boardBlackName = blackName;
  renderSideLabels();
}

function shownPosition(): { fen: string; uci: string | null } {
  if (reviewing && reviewTrail) {
    const entry = reviewTrail.entries[Math.max(0, Math.min(reviewIndex, reviewTrail.entries.length - 1))];
    if (entry) return { fen: entry.fen, uci: entry.uci };
  }
  return { fen: boardToFen(board), uci: lastUci };
}

function renderBoard() {
  const shown = shownPosition();
  const shownUci = shown.uci;
  const shownBoard = parseFen(shown.fen);
  const config = {
    orientation: 'white' as const,
    fen: shown.fen.split(' ')[0],
    // turnColor matters even in viewOnly mode: `check: true` highlights the
    // king of turnColor, which must be the side to move in the shown position.
    turnColor: shownBoard.turn === 'w' ? 'white' as const : 'black' as const,
    check: boardCheck(shownBoard),
    coordinates: true,
    viewOnly: true,
    highlight: { lastMove: true, check: true },
    animation: { enabled: true, duration: 140 },
    lastMove: shownUci ? [shownUci.slice(0, 2) as Key, shownUci.slice(2, 4) as Key] : undefined,
    // Custom brushes in the two side identity colors so the last-move arrow also
    // shows which side just moved — a calm alternative to per-ply "to move" flashing.
    drawable: { enabled: false, brushes: {
      moveWhite: { key: 'moveWhite', color: '#2f6e7d', opacity: 0.9, lineWidth: 14 },
      moveBlack: { key: 'moveBlack', color: '#b15c2b', opacity: 0.9, lineWidth: 14 },
      candidate: { key: 'candidate', color: '#5a6e2a', opacity: 0.95, lineWidth: 14 },
      candidateDim: { key: 'candidateDim', color: '#8a8474', opacity: 0.5, lineWidth: 9 },
    } },
  };
  // Cast: chessground's DrawBrushes type has fixed keys, but custom brush keys
  // merge fine at runtime.
  const cfg = config as unknown as NonNullable<Parameters<typeof Chessground>[1]>;
  if (!ground) ground = Chessground(el('ground'), cfg);
  else ground.set(cfg);
  // The mover is the side NOT to move now; tint the arrow with their identity hue.
  const moverBrush = fenTurn(shown.fen) === 'w' ? 'moveBlack' : 'moveWhite';
  const shapes: DrawShape[] = reviewing && reviewShapes
    ? reviewShapes
    : shownUci && shownUci.length >= 4
      ? [{ orig: shownUci.slice(0, 2) as Key, dest: shownUci.slice(2, 4) as Key, brush: moverBrush }] : [];
  ground.setAutoShapes(shapes);
  renderSideLabels();
  renderEngineOutputs();
  renderReviewBar();
  renderMoveStrip();
}

// ---------------------------------------------------------------------------
// Game-history review
// ---------------------------------------------------------------------------
function trailFromTree(tree: GameTree, label: string, openingPlies: number): GameTrail {
  const entries: TrailEntry[] = [{ fen: tree.root.fen, uci: null, san: null }];
  for (const node of tree.mainlineFrom(tree.root)) {
    entries.push({ fen: node.fen, uci: node.move ? moveToUci(node.move) : null, san: node.san });
  }
  return { label, entries, openingPlies };
}

function enterReview(trail: GameTrail, index: number, shapes: DrawShape[] | null = null): void {
  reviewTrail = trail;
  reviewIndex = Math.max(0, Math.min(index, trail.entries.length - 1));
  reviewing = true;
  reviewShapes = shapes;
  renderBoard();
}

function stepReview(delta: number | 'start' | 'end'): void {
  if (!reviewing || !reviewTrail) return;
  if (delta === 'start') reviewIndex = 0;
  else if (delta === 'end') reviewIndex = reviewTrail.entries.length - 1;
  else reviewIndex = Math.max(0, Math.min(reviewIndex + delta, reviewTrail.entries.length - 1));
  reviewShapes = null;
  renderBoard();
}

function exitReview(): void {
  reviewing = false;
  reviewTrail = null;
  reviewShapes = null;
  renderBoard();
}

function renderReviewBar(): void {
  const bar = el('reviewBar');
  bar.hidden = !reviewing || !reviewTrail;
  if (bar.hidden || !reviewTrail) return;
  el('reviewLabel').textContent = `${reviewTrail.label} · move ${reviewIndex}/${reviewTrail.entries.length - 1}`;
  (el('revPrev') as HTMLButtonElement).disabled = reviewIndex <= 0;
  (el('revStart') as HTMLButtonElement).disabled = reviewIndex <= 0;
  (el('revNext') as HTMLButtonElement).disabled = reviewIndex >= reviewTrail.entries.length - 1;
  (el('revEnd') as HTMLButtonElement).disabled = reviewIndex >= reviewTrail.entries.length - 1;
}

function renderMoveStrip(): void {
  const strip = el('gameMoves');
  const trail = reviewing && reviewTrail ? reviewTrail : liveTrail;
  if (!trail || trail.entries.length <= 1) { strip.hidden = true; strip.innerHTML = ''; return; }
  strip.hidden = false;
  const shownIndex = reviewing && reviewTrail === trail ? reviewIndex : trail.entries.length - 1;
  const parts: string[] = [];
  for (let i = 1; i < trail.entries.length; i++) {
    const entry = trail.entries[i];
    if (i % 2 === 1) parts.push(`<span class="num">${(i + 1) >> 1}.</span>`);
    parts.push(`<span class="mv${i === shownIndex ? ' current' : ''}" data-idx="${i}">${htmlEscape(entry.san ?? entry.uci ?? '?')}</span>`);
  }
  strip.innerHTML = parts.join(' ');
  const current = strip.querySelector('.mv.current');
  if (current) (current as HTMLElement).scrollIntoView({ block: 'nearest', inline: 'nearest' });
}

/**
 * Click on the root-visits bar chart: jump to the position that search was
 * made from and draw the candidate moves — the clicked bar bold, the rest dim.
 */
function reviewRootChartClick(event: MouseEvent): void {
  if (!rootChartContext || !liveTrail) return;
  const svg = el('rootChart').querySelector('svg');
  if (!svg) return;
  // Find the searched position in the current game's trail (latest match wins;
  // exact FEN strings, both produced by boardToFen of the same board).
  let index = -1;
  for (let i = liveTrail.entries.length - 1; i >= 0; i--) {
    if (liveTrail.entries[i].fen === rootChartContext.fen) { index = i; break; }
  }
  if (index < 0) return;
  // Mirror hBarChartSvg's layout: each bar row is 14 viewBox units high.
  const rect = svg.getBoundingClientRect();
  const viewH = rootChartContext.top.length * 14 + 2;
  const row = Math.floor((((event.clientY - rect.top) / rect.height) * viewH - 1) / 14);
  const clicked = rootChartContext.top[Math.max(0, Math.min(row, rootChartContext.top.length - 1))];
  const shapes: DrawShape[] = rootChartContext.top.slice(0, 5).map((child) => ({
    orig: child.uci.slice(0, 2) as Key,
    dest: child.uci.slice(2, 4) as Key,
    brush: child.uci === clicked.uci ? 'candidate' : 'candidateDim',
  }));
  if (!shapes.some((shape) => shape.brush === 'candidate')) {
    shapes.push({ orig: clicked.uci.slice(0, 2) as Key, dest: clicked.uci.slice(2, 4) as Key, brush: 'candidate' });
  }
  enterReview(liveTrail, index, shapes);
}

function chartDrawnPlies(chartId: 'evalChart' | 'timeChart' | 'npsChart'): number[] {
  return gameChartSamples.flatMap((sample) => {
    const value = chartId === 'evalChart' ? sample.whiteScore : chartId === 'timeChart' ? sample.moveMs : sample.nps;
    return value === undefined || !Number.isFinite(value) ? [] : [sample.ply];
  });
}

/** Map a click on a per-ply line chart back to the chart's ply index. */
function chartPlyFromClick(chartId: 'evalChart' | 'timeChart' | 'npsChart', target: HTMLElement, event: MouseEvent): number | null {
  const svg = target.closest('.chart-card')?.querySelector('svg');
  const plies = chartDrawnPlies(chartId);
  if (!svg || !plies.length) return null;
  const rect = svg.getBoundingClientRect();
  if (!rect.width) return null;
  // Mirror lineChartSvg's layout: viewBox width 360, plot area between
  // pad.left=36 and width-pad.right=352, x spanning the actually drawn points.
  const viewX = ((event.clientX - rect.left) / rect.width) * 360;
  const xMin = Math.min(...plies);
  const xMax = Math.max(...plies);
  const frac = Math.max(0, Math.min(1, (viewX - 36) / (352 - 36)));
  return Math.round(xMin + frac * (xMax - xMin));
}

// The arena is a two-seat head-to-head. Each seat uses the same staged selector
// pattern as the analysis page: family → variant → strength.
function strengthMeta(family: EngineFamily) {
  return engineStrengthMeta(family, 'arena');
}

function defaultVariant(family: EngineFamily): string {
  if (family === 'reckless') return REQUESTED_RECKLESS_VARIANT.key;
  if (family === 'viridithas') return REQUESTED_VIRIDITHAS_VARIANT.key;
  if (family === 'berserk') return REQUESTED_BERSERK_VARIANT.key;
  if (family === 'plentychess') return REQUESTED_PLENTYCHESS_VARIANT.key;
  return defaultStaticEngineVariant(family);
}

function bigNetSelectableSync(config: BigNetConfig): boolean {
  return bt4SupportedSync() && bigNetAssetStatusSync(config) === 'present';
}

function bigNetUnavailableText(config: BigNetConfig): string {
  if (!bt4SupportedSync()) return `Lc0 ${config.name} needs WebGPU support.`;
  if (bigNetAssetStatusSync(config) === 'missing') return `Lc0 ${config.name} model asset is missing at ${config.modelUrl}. Run node scripts/lc0_prepare_model_assets.mjs in this package.`;
  if (bigNetAssetStatusSync(config) === 'unknown') return `Lc0 ${config.name} model asset is still being checked.`;
  return '';
}

function variantOptions(family: EngineFamily): { value: string; label: string; disabled?: boolean }[] {
  if (isV0DeployProfile() && !['lc0', 'sf', 'reckless', 'berserk', 'viridithas', 'plentychess'].includes(family)) return [];
  if (family === 'lc0') return lc0VariantOptions(bt4SupportedSync()).map((option) => {
    if (!isLc0BigNetVariant(option.value)) return option;
    const config = BIG_NETS[option.value];
    const asset = bigNetAssetStatusSync(config);
    if (bt4SupportedSync() && asset === 'unknown') void checkBigNetAsset(config, () => { renderSeatSelectors(); refreshSeatControls(); });
    const suffix = !bt4SupportedSync() ? ' (WebGPU unavailable)' : asset === 'missing' ? ' (asset missing)' : asset === 'unknown' ? ' (checking asset)' : '';
    return { ...option, label: `${option.label}${suffix}`, disabled: option.disabled || asset !== 'present' };
  });
  if (family === 'tiny') return tinyVariantOptions().map((option) => option.value === 'bt4-custom' && tinyHybridManifestStatus === 'missing'
    ? { ...option, label: `${option.label} (bundle missing)`, disabled: true }
    : option);
  if (family === 'sf') return stockfishVariantOptions();
  if (family === 'viridithas') return availableViridithasVariants().map((v) => {
    const status = viridithasVariantAssetStatus(v);
    const unsupported = v.key === 'relaxed-simd' && !supportsWasmRelaxedSimd();
    if (!unsupported && v.key === 'relaxed-simd' && status === 'unknown') void checkViridithasVariantAsset(v, populateSeats);
    const disabled = unsupported || (v.key === 'relaxed-simd' && status !== 'ok') || status === 'missing';
    const suffix = unsupported ? ' (unsupported by this browser)' : status === 'missing' ? ' (asset missing)' : v.key === 'relaxed-simd' && status !== 'ok' ? ' (checking asset)' : '';
    return { value: v.key, label: `${v.label}${suffix}`, disabled };
  });
  if (family === 'berserk') return availableBerserkVariants().map((v) => {
    const status = berserkVariantAssetStatus(v);
    const unsupported = v.key === 'emscripten-relaxed' && !supportsWasmRelaxedSimd();
    if (!unsupported && status === 'unknown') void checkBerserkVariantAsset(v, populateSeats);
    const needsGeneratedAsset = v.key === 'emscripten-simd' || v.key === 'emscripten-relaxed';
    const disabled = unsupported || (needsGeneratedAsset && status !== 'present') || status === 'missing';
    const suffix = unsupported ? ' (unsupported by this browser)' : status === 'missing' ? ' (asset missing)' : needsGeneratedAsset && status !== 'present' ? ' (checking asset)' : '';
    return { value: v.key, label: `${v.label}${suffix}`, disabled };
  });
  if (family === 'plentychess') return availablePlentyChessVariants().map((v) => {
    const status = plentyChessVariantAssetStatus(v);
    const unsupportedReason = plentyChessVariantUnsupportedReason(v);
    const needsGeneratedAsset = v.key === 'emscripten-sse41' || v.key === 'emscripten-relaxed';
    if (!unsupportedReason && needsGeneratedAsset && status === 'unknown') void checkPlentyChessVariantAsset(v, populateSeats);
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

function clampStrength(row: EngineRow): void {
  const meta = strengthMeta(row.family);
  row.strength = Math.max(meta.min, Math.min(meta.max, Math.floor(Number(row.strength) || meta.def)));
}

function normalizeSeatRowForDeploy(index: number): void {
  seatRows[index] = normalizeDeployEngineRow(seatRows[index], 'arena', index);
}

function normalizeSeatRowsForDeploy(): void {
  for (let index = 0; index < seatRows.length; index++) normalizeSeatRowForDeploy(index);
}

function rowLabel(row: EngineRow): string {
  if (row.family === 'lc0') return lc0EngineLabel(row.variant);
  if (row.family === 'tiny') return tinyEngineLabel(row.variant);
  if (row.family === 'sf') return stockfishEngineLabel(row.variant, 'arena');
  if (row.family === 'viridithas') return viridithasVariantForKey(row.variant).label;
  if (row.family === 'berserk') return berserkVariantForKey(row.variant).label;
  if (row.family === 'plentychess') return plentyChessVariantForKey(row.variant).label;
  return recklessVariantForKey(row.variant).label;
}

function engineIdForRow(row: EngineRow): string {
  return `${row.family}:${row.variant}:${row.strength}`;
}

function arenaTournamentMode(): TournamentMode {
  const value = selectEl('tournamentModeSelect').value;
  return value === 'round-robin' || value === 'gauntlet' ? value : 'match';
}

/** Seats participating under the current mode (match uses only the first two). */
function activeSeatRows(): EngineRow[] {
  return arenaTournamentMode() === 'match' ? seatRows.slice(0, 2) : [...seatRows];
}

function seatEngineId(index: number): string {
  return engineIdForRow(seatRows[index]);
}

function renderSeatSelectors(): void {
  const families = engineFamilyOptions();
  const matchMode = arenaTournamentMode() === 'match';
  el('arenaSeatList').innerHTML = seatRows.map((row, index) => {
    const meta = strengthMeta(row.family);
    const famSel = families.map(({ value, label }) => `<option value="${value}"${row.family === value ? ' selected' : ''}>${label}</option>`).join('');
    const varSel = variantOptions(row.family).map((option) => `<option value="${option.value}"${row.variant === option.value ? ' selected' : ''}${option.disabled ? ' disabled' : ''}>${htmlEscape(option.label)}</option>`).join('');
    const label = `Engine ${index + 1}`;
    const inactive = matchMode && index >= 2 ? ' seat-inactive' : '';
    const remove = seatRows.length > 2 ? `<button type="button" class="seat-remove" data-seat="${index}" title="Remove ${label}" aria-label="Remove ${label}">×</button>` : '';
    return `<div class="engine-row seat-row${inactive}" data-seat="${index}"><span class="seat-name">${label}</span><select class="seat-fam" data-seat="${index}" aria-label="${label} family">${famSel}</select><span class="arrow">→</span><select class="seat-var" data-seat="${index}" aria-label="${label} variant">${varSel}</select><span class="arrow">→</span><input class="seat-strength row-strength" data-seat="${index}" aria-label="${label} strength" type="number" min="${meta.min}" max="${meta.max}" step="1" value="${row.strength}" title="${meta.unit}"><span class="row-unit">${meta.unit}</span>${remove}</div>`;
  }).join('');
}

function syncSeatRowsFromDom(): void {
  const host = el('arenaSeatList');
  for (let index = 0; index < seatRows.length; index++) {
    const family = host.querySelector<HTMLSelectElement>(`.seat-fam[data-seat="${index}"]`)?.value;
    if (family && isEngineFamily(family)) seatRows[index].family = family;
    const variant = host.querySelector<HTMLSelectElement>(`.seat-var[data-seat="${index}"]`)?.value;
    if (variant) seatRows[index].variant = variant;
    const strength = host.querySelector<HTMLInputElement>(`.seat-strength[data-seat="${index}"]`)?.value;
    if (strength != null) seatRows[index].strength = Number(strength);
    clampStrength(seatRows[index]);
    normalizeSeatRowForDeploy(index);
  }
}

function populateSeats(): void {
  const options = [...engines.values()].map((engine) => `<option value="${htmlEscape(engine.id)}">${htmlEscape(engine.name)}</option>`).join('');
  selectEl('seatA').innerHTML = options;
  selectEl('seatB').innerHTML = options;
  selectEl('seatA').value = seatEngineId(0);
  selectEl('seatB').value = seatEngineId(1);
  renderSeatSelectors();
}

function refreshSeatControls(): void {
  selectEl('seatA').disabled = running;
  selectEl('seatB').disabled = running;
  selectEl('lc0PresetSelect').disabled = running || loadingLc0;
  selectEl('lc0RuntimeSelect').disabled = running || loadingLc0;
  const runtime = selectedLc0Runtime();
  const hybridControlsDisabled = running || loadingLc0 || (runtime !== 'hybrid-ort-heads' && runtime !== 'hybrid-wgsl-heads');
  selectEl('lc0InputBackendSelect').disabled = hybridControlsDisabled;
  selectEl('lc0EncoderKernelSelect').disabled = hybridControlsDisabled;
  selectEl('lc0LegalPriorsSelect').disabled = hybridControlsDisabled;
  inputEl('lc0BatchSizeInput').disabled = running;
  inputEl('lc0BatchPipelineDepthInput').disabled = running;
  for (const selector of ['.seat-fam', '.seat-var', '.seat-strength', '.seat-remove']) {
    for (const node of el('arenaSeatList').querySelectorAll<HTMLInputElement | HTMLSelectElement>(selector)) node.disabled = running;
  }
  el('addSeat').toggleAttribute('disabled', running);
  selectEl('tournamentModeSelect').disabled = running;
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

/** Manual SF threads cap from the UI; 0 means auto (resource broker decides). */
function stockfishThreadsCap(): number {
  const requested = Math.max(0, Math.min(32, Math.floor(Number(inputEl('stockfishThreadsInput').value) || 0)));
  return threadedStockfishAvailable() ? requested : 1;
}

function formatCount(value: number | undefined): string {
  if (value === undefined || !Number.isFinite(value)) return '—';
  const abs = Math.abs(value);
  if (abs >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`;
  if (abs >= 1_000) return `${(value / 1_000).toFixed(1)}k`;
  return String(Math.round(value));
}

function formatMs(value: number | undefined): string {
  if (value === undefined || !Number.isFinite(value)) return '—';
  return value >= 1000 ? `${(value / 1000).toFixed(2)}s` : `${Math.round(value)}ms`;
}

function averageMs(total: number | undefined, count: number): string {
  return total !== undefined && count > 0 ? formatMs(total / count) : '—';
}

function timingMeansText(means: Record<string, number> | undefined): string {
  if (!means) return '';
  const entries = Object.entries(means).filter(([, value]) => Number.isFinite(value));
  if (!entries.length) return '';
  return entries.slice(0, 4).map(([key, value]) => `${key} ${formatMs(value)}`).join(' · ');
}

/** Thread count to plan worker flavor/warmup around before any lease exists. */
function stockfishThreadsPlanned(): number {
  const cap = stockfishThreadsCap();
  if (cap > 0) return cap;
  return Math.max(1, Math.min(32, resourceBroker.cpuBudget()));
}

/** Per-search threads: the broker grant, narrowed by the manual cap when set. */
function stockfishThreadsGranted(leaseThreads: number): number {
  const cap = stockfishThreadsCap();
  return Math.max(1, cap > 0 ? Math.min(cap, leaseThreads) : leaseThreads);
}

/**
 * One labeled diagnostics block: a muted label column plus mono/tabular-nums
 * value lines, so live figures update in place instead of reflowing a long
 * prose line. Lines are pre-escaped by callers via htmlEscape.
 */
function diagBlockHtml(label: string, valueLines: string[]): string {
  const lines = valueLines.length ? valueLines : ['—'];
  return lines.map((line, i) => `<span class="diag-label">${i === 0 ? htmlEscape(label) : ''}</span><span class="diag-value">${line}</span>`).join('');
}

function cacheMetricsText(metrics: Lc0EvaluationCacheMetrics | undefined): string {
  if (!metrics) return 'NN cache unavailable';
  return `NN cache ${metrics.entries}/${metrics.maxEntries} entries · ${metrics.hits} hit${metrics.hits === 1 ? '' : 's'} · ${metrics.misses} miss${metrics.misses === 1 ? '' : 'es'}`;
}

function lc0ExecutionFootprint(): Lc0WebExecutionFootprint | undefined {
  return (lc0Cache?.inner as { executionFootprint?: () => Lc0WebExecutionFootprint | undefined } | undefined)?.executionFootprint?.();
}

function lc0CacheFootprint(): Lc0EvaluationCacheFootprint | undefined {
  return lc0Cache?.cacheFootprint();
}

function diagnosticEngineIds(): string[] {
  const ids = activeEngineIds.length ? activeEngineIds : [seatEngineId(0), seatEngineId(1)];
  return [...new Set(ids)].filter((id) => engines.has(id));
}

function rowForEngineId(engineId: string): EngineRow | undefined {
  return activeSeatRows().find((row) => engineIdForRow(row) === engineId);
}

function budgetText(row: EngineRow | undefined): string {
  if (arenaBudgetMode() === 'movetime') return `${arenaMovetimeMs()}ms/move`;
  if (!row) return 'budget —';
  const meta = strengthMeta(row.family);
  return `${meta.unit} ${row.strength}`;
}

function engineRuntimeDiagnosticsText(): string {
  const ids = diagnosticEngineIds();
  if (!ids.length) return 'Engine diagnostics: choose engines.';
  const parts = ids.map((id) => {
    const row = rowForEngineId(id);
    const name = engines.get(id)?.name ?? id;
    if (row?.family === 'lc0' && !isLc0BigNetVariant(row.variant)) {
      const hybrid = lc0HybridConfigLabel();
      return `${name}: ${budgetText(row)} · ${lc0RuntimeLabel()}${hybrid ? ` · ${hybrid}` : ''} · batch ${lc0BatchSize()} · pipeline depth ${lc0BatchPipelineDepth()} · ${cacheMetricsText(lc0Cache?.metrics())}`;
    }
    if (row?.family === 'lc0' && isLc0BigNetVariant(row.variant)) {
      const { config, searcher } = bigNetFor(row.variant);
      return `${name}: ${budgetText(row)} · ${searcher.loaded ? `loaded ${searcher.backend || 'WebGPU'}` : 'lazy WebGPU worker'} · ${config.name} · batch ${config.recommendedBatchSize} · pipeline depth ${config.recommendedPipelineDepth} · eval cache ${arenaCacheEntries()} · ~${config.approxMb}MB net · ${config.modelUrl}`;
    }
    if (row?.family === 'tiny') return `${name}: ${budgetText(row)} · SquareFormer ${tinyRuntimeForVariant(row.variant)} · ${tinyHybridManifestStatusText()}`;
    if (row?.family === 'sf') {
      const kind = row.variant === 'full' ? 'full' : 'lite';
      const threads = stockfishThreadsPlanned();
      return `${name}: ${budgetText(row)} · ${stockfishFlavorLabel(stockfishFlavorFor(kind))} · ${threads} thread${threads === 1 ? '' : 's'}`;
    }
    if (row?.family === 'reckless') {
      const variant = recklessVariantForKey(row.variant);
      const engine = recklessByVariant.get(recklessCacheKey(variant));
      return `${name}: ${budgetText(row)} · ${variant.label} · ${engine?.runtimeLabel() ?? 'not loaded'} · hash 16MB`;
    }
    if (row?.family === 'viridithas') {
      const variant = viridithasVariantForKey(row.variant);
      const engine = viridithasByVariant.get(viridithasCacheKey(variant));
      return `${name}: ${budgetText(row)} · ${variant.label} · ${engine?.runtimeLabel() ?? 'not loaded'} · hash 16MB`;
    }
    if (row?.family === 'berserk') {
      const variant = berserkVariantForKey(row.variant);
      const engine = berserkByVariant.get(berserkCacheKey(variant));
      return `${name}: ${budgetText(row)} · ${variant.label} · ${engine?.runtimeLabel() ?? 'not loaded'} · hash 16MB`;
    }
    if (row?.family === 'plentychess') {
      const variant = plentyChessVariantForKey(row.variant);
      const engine = plentyChessByVariant.get(plentyChessCacheKey(variant));
      return `${name}: ${budgetText(row)} · ${variant.label} · ${engine?.runtimeLabel() ?? 'not loaded'} · hash 16MB`;
    }
    return `${name}: diagnostics unavailable`;
  });
  return `Engine diagnostics: ${parts.join(' | ')}`;
}

function renderCacheInfo(): void {
  lc0Cache?.setMaxEntries(arenaCacheEntries());
  renderEngineDiagnosticsInfo();
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

function lc0TreeTelemetrySummary(t: Lc0TreeTelemetry): string {
  const fresh = Math.max(0, t.searches - t.rootReused);
  const backend = timingMeansText(t.evalBackendTimingPerPositionMeans ?? t.evalBackendTimingMeans ?? t.lastBackendTimingPerPositionMeans ?? t.lastBackendTimingMeans);
  const batch = t.lastBatchSize ? ` · batch ${t.lastBatchSize} · maxEvalBatch ${t.maxEvalBatch ?? t.lastBatchSize}${t.lastBatchPipelineDepth && t.lastBatchPipelineDepth > 1 ? ` · pipeline ${t.lastBatchPipelineDepth}` : ''}` : '';
  return `${t.engineName}: searches ${t.searches} · avg ${averageMs(t.totalElapsedMs, t.searches)} · last ${formatMs(t.lastElapsedMs)}${backend ? ` · backend avg ${backend}` : ''}${batch} · tree reuse ${ratioText(t.rootReused, t.searches)} (${fresh} fresh) · reused visits ${t.reusedRootVisits} · reply parent ${ratioText(t.replyParentsExpanded, t.replyChecks)} · reply visited ${ratioText(t.replyVisited, t.replyChecks)} · opp reply top-policy≤5 ${ratioText(t.replyTopPolicy5, t.replyChecks)} · top-visits≤5 ${ratioText(t.replyTopVisits5, t.replyChecks)} · evals ${t.evalCalls} · cache hits ${t.cacheHits} · trans ${t.transpositionHits}`;
}

function uciTelemetrySummary(t: UciEngineTelemetry): string {
  const last = t.lastDepth !== undefined
    ? `last d${t.lastDepth} · nodes ${formatCount(t.lastNodes)} · nps ${formatCount(t.lastNps)} · PV ${t.lastPvMoves ?? 0}`
    : 'waiting for first PV';
  return `${t.engineName}: searches ${t.searches} · avg ${averageMs(t.totalElapsedMs, t.searches)} · last ${formatMs(t.lastElapsedMs)} · total nodes ${formatCount(t.totalNodes)} · ${last}${t.lastMultiPv && t.lastMultiPv > 1 ? ` · MultiPV ${t.lastMultiPv}` : ''}`;
}

function bt4TelemetrySummary(t: Bt4Telemetry, searcher: Bt4WorkerSearcher): string {
  return `${t.engineName}: searches ${t.searches} · avg ${averageMs(t.totalElapsedMs, t.searches)} · visits ${t.completedVisits} · evals ${t.evalCalls} · cache hits ${t.cacheHits} · last ${formatMs(t.lastElapsedMs)} · ${searcher.loaded ? `backend ${searcher.backend || 'WebGPU'}` : 'worker not loaded'}`;
}

function engineSearchDiagnosticsText(): string {
  const ids = diagnosticEngineIds();
  if (!ids.length) return 'Search diagnostics: choose engines.';
  const parts = ids.map((id) => {
    const row = rowForEngineId(id);
    const name = engines.get(id)?.name ?? id;
    if (row?.family === 'lc0' && !isLc0BigNetVariant(row.variant)) {
      const t = lc0TreeTelemetry.get(id);
      return t && (t.searches || t.replyChecks) ? lc0TreeTelemetrySummary(t) : `${name}: tree waiting for searches · ${cacheMetricsText(lc0Cache?.metrics())}`;
    }
    if (row?.family === 'lc0' && isLc0BigNetVariant(row.variant)) {
      const t = bt4Telemetry.get(id);
      const { config, searcher } = bigNetFor(row.variant);
      return t?.searches ? bt4TelemetrySummary(t, searcher) : `${name}: ${config.name} search waiting${searcher.loaded ? ` · backend ${searcher.backend || 'WebGPU'}` : ''}`;
    }
    if (row?.family === 'tiny') return engineOutputs.has(id) ? `${name}: Tiny SquareFormer search output ready` : `${name}: Tiny SquareFormer search waiting`;
    const uci = uciTelemetry.get(id);
    return uci?.searches ? uciTelemetrySummary(uci) : `${name}: UCI search waiting for info`;
  });
  return `Search diagnostics: ${parts.join(' | ')}`;
}

function diagnosticsHtml(label: string, text: string): string {
  const value = text.replace(/^[^:]+:\s*/, '');
  return diagBlockHtml(label, value.split(' | ').map(htmlEscape));
}

function renderEngineDiagnosticsInfo(): void {
  el('cacheInfo').innerHTML = diagnosticsHtml('Engines', engineRuntimeDiagnosticsText());
  el('searchTelemetryInfo').innerHTML = diagnosticsHtml('Search', engineSearchDiagnosticsText());
}

function searchTelemetryText(): string {
  return engineSearchDiagnosticsText();
}

function renderSearchTelemetryInfo(): void {
  renderEngineDiagnosticsInfo();
}

function isLc0SearchEngine(engineId: string): boolean {
  return engineId.startsWith('lc0:small:');
}

function childForRootMove(result: Lc0SearchResult | undefined, uci: string): PuctNode | null {
  const root = result?.search.root;
  if (!root?.expanded) return null;
  return root.edges.find((edge) => moveToUci(edge.move) === uci)?.child ?? null;
}

function addNumericTotals(target: Record<string, number>, source: Record<string, number> | undefined): void {
  if (!source) return;
  for (const [key, value] of Object.entries(source)) {
    if (Number.isFinite(value)) target[key] = (target[key] ?? 0) + value;
  }
}

function meanRecord(totals: Record<string, number> | undefined, denominator: number | undefined): Record<string, number> | undefined {
  if (!totals || !denominator || denominator <= 0) return undefined;
  return Object.fromEntries(Object.entries(totals).map(([key, value]) => [key, Number((value / denominator).toFixed(6))]));
}

function mergeHistogram(target: Record<string, number> | undefined, source: Record<string, number> | undefined): Record<string, number> | undefined {
  if (!source) return target;
  const out = { ...(target ?? {}) };
  for (const [key, value] of Object.entries(source)) out[key] = (out[key] ?? 0) + value;
  return out;
}

function recordLc0SearchTelemetry(engineId: string, engineName: string, result: Lc0SearchResult, elapsedMs?: number): void {
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
  t.evalBackendTimingSamples = (t.evalBackendTimingSamples ?? 0) + (stats?.evalBackendTimingSamples ?? 0);
  t.evalBackendTimingPositions = (t.evalBackendTimingPositions ?? 0) + (stats?.evalBackendTimingPositions ?? 0);
  t.evalBackendTimingTotals ??= {};
  addNumericTotals(t.evalBackendTimingTotals, stats?.evalBackendTimingTotals);
  t.evalBackendTimingMeans = meanRecord(t.evalBackendTimingTotals, t.evalBackendTimingSamples);
  t.evalBackendTimingPerPositionMeans = meanRecord(t.evalBackendTimingTotals, t.evalBackendTimingPositions);
  t.lastBatchSize = stats?.batchSize;
  t.lastBatchPipelineDepth = stats?.batchPipelineDepth;
  t.maxEvalBatch = Math.max(t.maxEvalBatch ?? 0, stats?.maxEvalBatch ?? 0);
  t.evalBatchSizeHistogram = mergeHistogram(t.evalBatchSizeHistogram, stats?.evalBatchSizeHistogram);
  if (elapsedMs !== undefined) {
    t.totalElapsedMs = (t.totalElapsedMs ?? 0) + elapsedMs;
    t.lastElapsedMs = elapsedMs;
  }
  t.lastBackendTimingMeans = stats?.evalBackendTimingMeans;
  t.lastBackendTimingPerPositionMeans = stats?.evalBackendTimingPerPositionMeans;
  lastLc0SearchResults.set(engineId, result);
  renderSearchTelemetryInfo();
}

function recordBt4Telemetry(engineId: string, engineName: string, result: Bt4SearchResult): void {
  const existing = bt4Telemetry.get(engineId) ?? { engineName, searches: 0, completedVisits: 0, evalCalls: 0, cacheHits: 0 };
  existing.engineName = engineName;
  existing.searches += 1;
  existing.completedVisits += result.visits ?? 0;
  existing.evalCalls += result.stats?.evalCalls ?? 0;
  existing.cacheHits += result.stats?.cacheHits ?? 0;
  existing.totalElapsedMs = (existing.totalElapsedMs ?? 0) + (result.elapsedMs ?? 0);
  existing.lastElapsedMs = result.elapsedMs;
  bt4Telemetry.set(engineId, existing);
  renderSearchTelemetryInfo();
}

function recordUciTelemetry(engineId: string, engineName: string, lines: StockfishInfoLine[], elapsedMs?: number): void {
  const best = lines[0];
  const existing = uciTelemetry.get(engineId) ?? { engineName, searches: 0, totalNodes: 0 };
  existing.engineName = engineName;
  existing.searches += 1;
  if (best?.nodes !== undefined) existing.totalNodes += best.nodes;
  existing.lastDepth = best?.depth;
  existing.lastNodes = best?.nodes;
  existing.lastNps = best?.nps;
  existing.lastPvMoves = best?.pvUci.length;
  existing.lastMultiPv = lines.length;
  if (elapsedMs !== undefined) {
    existing.totalElapsedMs = (existing.totalElapsedMs ?? 0) + elapsedMs;
    existing.lastElapsedMs = elapsedMs;
  }
  uciTelemetry.set(engineId, existing);
  renderSearchTelemetryInfo();
}

function recordLc0PolicyOutput(engineId: string, engineName: string, evaluation: Lc0Evaluation, move?: string): void {
  recordEngineOutput({
    engineId,
    engineName,
    kind: 'lc0',
    fen: evaluation.fen,
    move,
    summary: lc0WdlText(evaluation),
    shortEval: lc0WdlCompact(evaluation.wdl, evaluation.q, evaluation.fen),
    evalBar: lc0EvalBar(evaluation.fen, evaluation.wdl),
    detail: `top policy ${evaluation.legalPriors.slice(0, 5).map((p) => `${p.uci} ${(100 * p.prior).toFixed(1)}%`).join(', ') || '—'}`,
  });
}

function recordLc0SearchOutput(engineId: string, engineName: string, result: Lc0SearchResult): void {
  const stats = result.search.stats;
  const rootWdl = result.search.root?.evaluation?.wdl;
  recordEngineOutput({
    engineId,
    engineName,
    kind: 'lc0',
    fen: result.fen,
    move: result.move,
    summary: rootWdl ? searchWdlText(rootWdl, result.value, result.fen) : `search Q ${signed(toWhiteQ(result.value, result.fen), 3)}`,
    shortEval: rootWdl ? lc0WdlCompact(rootWdl, result.value, result.fen) : `Q ${signed(toWhiteQ(result.value, result.fen), 2)}`,
    evalBar: rootWdl ? lc0EvalBar(result.fen, rootWdl) : qEvalBar(result.fen, result.value),
    detail: `visits ${result.visits} · evals ${stats?.evalCalls ?? 0} · cache hits ${stats?.cacheHits ?? 0}`,
    pv: result.pv,
  });
}

function recordBt4SearchOutput(engineId: string, engineName: string, result: Bt4SearchResult): void {
  recordBt4Telemetry(engineId, engineName, result);
  recordEngineOutput({
    engineId,
    engineName,
    kind: 'lc0',
    fen: result.fen,
    move: result.move ?? undefined,
    summary: `search Q ${signed(toWhiteQ(result.value, result.fen), 3)}`,
    shortEval: `Q ${signed(toWhiteQ(result.value, result.fen), 2)}`,
    evalBar: qEvalBar(result.fen, result.value),
    detail: `visits ${result.visits} · evals ${result.stats?.evalCalls ?? 0} · cache hits ${result.stats?.cacheHits ?? 0}`,
    pv: result.pv,
  });
}

function recordTinySearchOutput(engineId: string, engineName: string, fen: string, result: TinySearchResult): void {
  const pv = result.principalVariation?.map((entry) => moveToUci(entry.move));
  recordEngineOutput({
    engineId,
    engineName,
    kind: 'lc0',
    fen,
    move: result.move ? moveToUci(result.move) : undefined,
    summary: `Tiny Q ${signed(toWhiteQ(result.value, fen), 3)}`,
    shortEval: `Q ${signed(toWhiteQ(result.value, fen), 2)}`,
    evalBar: qEvalBar(fen, result.value),
    detail: `visits ${result.visits} · evals ${result.stats?.evalCalls ?? 0} · cache hits ${result.stats?.cacheHits ?? 0}`,
    pv,
  });
}

function recordUciOutput(engineId: string, engineName: string, label: string, fen: string, move: string | null, lines: StockfishInfoLine[], elapsedMs?: number): void {
  recordUciTelemetry(engineId, engineName, lines, elapsedMs);
  const best = lines[0];
  recordEngineOutput({
    engineId,
    engineName,
    kind: 'uci',
    fen,
    move: move ?? undefined,
    summary: `${label} ${stockfishScoreText(best, fen)}`,
    shortEval: stockfishScoreCompact(best, fen),
    evalBar: stockfishEvalBar(fen, best),
    detail: lines.length > 1 ? `MultiPV ${(lines.slice(0, 3) as StockfishInfoLine[]).map((line) => `#${line.multipv ?? 1} ${stockfishScoreText(line, fen)}`).join(' · ')}` : undefined,
    pv: best?.pvUci,
    whiteCp: stockfishWhiteCp(best, fen),
    mateInWhitePov: stockfishMateInWhitePov(best, fen),
    depth: best?.depth,
    nodes: best?.nodes,
    nps: best?.nps,
    elapsedMs,
  });
}

function recordStockfishOutput(engineId: string, engineName: string, fen: string, move: string | null, lines: StockfishInfoLine[], elapsedMs?: number): void {
  recordUciOutput(engineId, engineName, 'SF', fen, move, lines, elapsedMs);
}

function recordRecklessOutput(engineId: string, engineName: string, fen: string, move: string | null, lines: StockfishInfoLine[], elapsedMs?: number): void {
  recordUciOutput(engineId, engineName, 'Reckless', fen, move, lines, elapsedMs);
}

function recordViridithasOutput(engineId: string, engineName: string, fen: string, move: string | null, lines: StockfishInfoLine[], elapsedMs?: number): void {
  recordUciOutput(engineId, engineName, 'Viridithas', fen, move, lines, elapsedMs);
}

function recordBerserkOutput(engineId: string, engineName: string, fen: string, move: string | null, lines: StockfishInfoLine[], elapsedMs?: number): void {
  recordUciOutput(engineId, engineName, 'Berserk', fen, move, lines, elapsedMs);
}

function recordPlentyChessOutput(engineId: string, engineName: string, fen: string, move: string | null, lines: StockfishInfoLine[], elapsedMs?: number): void {
  recordUciOutput(engineId, engineName, 'PlentyChess', fen, move, lines, elapsedMs);
}

function recordEngineThinking(engine: ArenaEngine): void {
  thinkingEngineIds.add(engine.id);
  renderSideLabels();
  renderEngineOutputs();
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

function stockfishFlavorFor(kind: 'lite' | 'full'): StockfishFlavor {
  const threaded = threadedStockfishAvailable() && stockfishThreadsPlanned() > 1;
  if (kind === 'lite') return threaded ? 'lite-threaded' : 'lite-single';
  return threaded ? 'threaded' : 'single';
}

// Stockfish Lite and full are independent engines (they may face each other), so
// each has its own lazily-created instance.
function stockfishEngineFor(kind: 'lite' | 'full'): StockfishEngine {
  if (kind === 'lite') {
    if (!stockfishLite) stockfishLite = new StockfishEngine({ depth: 4, threads: stockfishThreadsPlanned() }, stockfishFlavorUrl(stockfishFlavorFor('lite')));
    return stockfishLite;
  }
  if (!stockfishFull) stockfishFull = new StockfishEngine({ depth: 4, threads: stockfishThreadsPlanned() }, stockfishFlavorUrl(stockfishFlavorFor('full')));
  return stockfishFull;
}

function disposeStockfish(): void {
  stockfishLite?.dispose();
  stockfishLite = null;
  stockfishFull?.dispose();
  stockfishFull = null;
}

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

function getRecklessFor(variantKey: string): RecklessEngine {
  const variant = recklessVariantForKey(variantKey);
  const key = recklessCacheKey(variant);
  let engine = recklessByVariant.get(key);
  if (!engine) {
    engine = createRecklessEngine(variant, renderRecklessRuntimeInfo);
    recklessByVariant.set(key, engine);
  }
  return engine;
}

function prewarmReckless(engine: RecklessEngine): void {
  void engine.prewarm()
    .then(renderRecklessRuntimeInfo)
    .catch((error) => {
      if ((error as Error).name !== 'AbortError') console.warn('Reckless prewarm failed', error);
      renderRecklessRuntimeInfo();
    });
}

function recklessMissingAssetMessage(variants: RecklessVariant[]): string {
  const urls = variants.flatMap((variant) => [variant.wasmUrl, ...(variant.nnueUrl ? [variant.nnueUrl] : [])]);
  return `Reckless asset missing: ${urls.join(', ')}. Build/publish Reckless artifacts with npm run reckless:build-production, or choose Stockfish/Tiny/LC0 instead.`;
}

function renderRecklessRuntimeInfo(): void {
  const rows = activeSeatRows().filter((row) => row.family === 'reckless');
  el('recklessRuntimeInfo').hidden = !rows.length;
  if (!rows.length) { el('recklessRuntimeInfo').textContent = ''; return; }
  const parts = rows.map((row) => {
    const variant = recklessVariantForKey(row.variant);
    const engine = recklessByVariant.get(recklessCacheKey(variant));
    const status = engine?.runtimeStatus();
    const mode = engine?.runtimeLabel() ?? (typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated ? 'persistent available' : 'one-shot fallback');
    const asset = recklessVariantAssetStatus(variant);
    if (asset === 'unknown') void checkRecklessVariantAsset(variant, renderRecklessRuntimeInfo);
    const assetText = asset === 'present' ? 'asset ok' : asset === 'missing' ? 'asset missing' : 'checking asset';
    const loadText = formatRecklessBrowserApiLoadStatus(status?.browserApiLoad);
    return `${variant.label} d${row.strength} · ${mode} · ${assetText}${loadText ? ` · ${loadText}` : ''}${status?.persistentDisabled ? ' · persistent disabled after fallback' : ''}`;
  });
  el('recklessRuntimeInfo').innerHTML = diagBlockHtml('Reckless', [...new Set(parts)].map(htmlEscape));
}

function refreshRecklessVariantUi(): void {
  const select = selectEl('recklessVariantSelect');
  if (!select.options.length) {
    select.innerHTML = availableRecklessVariants().map((variant) => `<option value="${variant.key}">${htmlEscape(variant.label)}</option>`).join('');
  }
  select.disabled = running;
  renderRecklessRuntimeInfo();
}

function availableViridithasVariants(): ViridithasVariant[] {
  return REQUESTED_VIRIDITHAS_VARIANT.key === 'custom' ? [...VIRIDITHAS_VARIANTS, REQUESTED_VIRIDITHAS_VARIANT] : [...VIRIDITHAS_VARIANTS];
}

function viridithasVariantForKey(variantKey: string): ViridithasVariant {
  const key = normalizeViridithasVariant(variantKey);
  if (key === 'custom' && REQUESTED_VIRIDITHAS_VARIANT.key === 'custom') return REQUESTED_VIRIDITHAS_VARIANT;
  return viridithasVariantByKey(key);
}

function getViridithasFor(variantKey: string): ViridithasEngine {
  const variant = viridithasVariantForKey(variantKey);
  const key = viridithasCacheKey(variant);
  let engine = viridithasByVariant.get(key);
  if (!engine) {
    engine = createViridithasEngine(variant);
    viridithasByVariant.set(key, engine);
  }
  return engine;
}

function renderViridithasRuntimeInfo(): void {
  const rows = activeSeatRows().filter((row) => row.family === 'viridithas');
  el('viridithasRuntimeInfo').hidden = !rows.length;
  if (!rows.length) { el('viridithasRuntimeInfo').textContent = ''; return; }
  const parts = rows.map((row) => {
    const variant = viridithasVariantForKey(row.variant);
    const engine = viridithasByVariant.get(viridithasCacheKey(variant));
    const status = engine?.runtimeStatus();
    const mode = engine?.runtimeLabel() ?? (canUsePersistentViridithasWasi() ? 'persistent available' : 'one-shot fallback');
    const asset = viridithasVariantAssetStatus(variant);
    if (asset === 'unknown') void checkViridithasVariantAsset(variant, renderViridithasRuntimeInfo);
    const assetText = asset === 'ok' ? 'asset ok' : asset === 'missing' ? 'asset missing' : 'checking asset';
    return `${variant.label} d${row.strength} · ${mode} · ${assetText}${status?.persistentDisabled ? ' · persistent disabled after fallback' : ''}`;
  });
  el('viridithasRuntimeInfo').innerHTML = diagBlockHtml('Viridithas', [...new Set(parts)].map(htmlEscape));
}

function refreshViridithasVariantUi(): void {
  const select = selectEl('viridithasVariantSelect');
  if (!select.options.length) {
    select.innerHTML = availableViridithasVariants().map((variant) => `<option value="${variant.key}">${htmlEscape(variant.label)}</option>`).join('');
  }
  select.disabled = running;
  renderViridithasRuntimeInfo();
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

function getBerserkFor(variantKey: string): BerserkEngine {
  const variant = berserkVariantForKey(variantKey);
  const key = berserkCacheKey(variant);
  let engine = berserkByVariant.get(key);
  if (!engine) {
    engine = createBerserkEngine(variant);
    berserkByVariant.set(key, engine);
  }
  return engine;
}

function renderBerserkRuntimeInfo(): void {
  const rows = activeSeatRows().filter((row) => row.family === 'berserk');
  el('berserkRuntimeInfo').hidden = !rows.length;
  if (!rows.length) { el('berserkRuntimeInfo').textContent = ''; return; }
  const parts = rows.map((row) => {
    const variant = berserkVariantForKey(row.variant);
    const engine = berserkByVariant.get(berserkCacheKey(variant));
    const asset = berserkVariantAssetStatus(variant);
    if (asset === 'unknown') void checkBerserkVariantAsset(variant, renderBerserkRuntimeInfo);
    const assetText = asset === 'present' ? 'asset ok' : asset === 'missing' ? 'asset missing' : 'checking asset';
    return `${variant.label} d${row.strength} · ${engine?.runtimeLabel() ?? 'Emscripten worker idle'} · ${assetText}`;
  });
  el('berserkRuntimeInfo').innerHTML = diagBlockHtml('Berserk', [...new Set(parts)].map(htmlEscape));
}

function refreshBerserkVariantUi(): void {
  const select = selectEl('berserkVariantSelect');
  if (!select.options.length) {
    select.innerHTML = availableBerserkVariants().map((variant) => `<option value="${variant.key}">${htmlEscape(variant.label)}</option>`).join('');
  }
  select.disabled = running;
  renderBerserkRuntimeInfo();
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

function getPlentyChessFor(variantKey: string): PlentyChessEngine {
  const variant = plentyChessVariantForKey(variantKey);
  const key = plentyChessCacheKey(variant);
  let engine = plentyChessByVariant.get(key);
  if (!engine) {
    engine = createPlentyChessEngine(variant);
    plentyChessByVariant.set(key, engine);
  }
  return engine;
}

function renderPlentyChessRuntimeInfo(): void {
  const rows = activeSeatRows().filter((row) => row.family === 'plentychess');
  el('plentychessRuntimeInfo').hidden = !rows.length;
  if (!rows.length) { el('plentychessRuntimeInfo').textContent = ''; return; }
  const parts = rows.map((row) => {
    const variant = plentyChessVariantForKey(row.variant);
    const engine = plentyChessByVariant.get(plentyChessCacheKey(variant));
    const asset = plentyChessVariantAssetStatus(variant);
    const unsupportedReason = plentyChessVariantUnsupportedReason(variant);
    if (!unsupportedReason && asset === 'unknown') void checkPlentyChessVariantAsset(variant, renderPlentyChessRuntimeInfo);
    const assetText = unsupportedReason ? unsupportedReason : asset === 'present' ? 'asset ok' : asset === 'missing' ? 'asset missing' : 'checking asset';
    return `${variant.label} d${row.strength} · ${engine?.runtimeLabel() ?? 'Emscripten worker idle'} · ${assetText}`;
  });
  el('plentychessRuntimeInfo').innerHTML = diagBlockHtml('PlentyChess', [...new Set(parts)].map(htmlEscape));
}

function refreshPlentyChessVariantUi(): void {
  const select = selectEl('plentychessVariantSelect');
  const selected = select.value;
  select.innerHTML = availablePlentyChessVariants().map((variant) => {
    const status = plentyChessVariantAssetStatus(variant);
    const unsupportedReason = plentyChessVariantUnsupportedReason(variant);
    const needsGeneratedAsset = variant.key === 'emscripten-sse41' || variant.key === 'emscripten-relaxed';
    if (!unsupportedReason && needsGeneratedAsset && status === 'unknown') void checkPlentyChessVariantAsset(variant, refreshPlentyChessVariantUi);
    const disabled = Boolean(unsupportedReason) || (needsGeneratedAsset && status !== 'present') || status === 'missing';
    const suffix = unsupportedReason ? ` (${unsupportedReason})` : status === 'missing' ? ' (asset missing)' : needsGeneratedAsset && status !== 'present' ? ' (checking asset)' : '';
    return `<option value="${variant.key}"${disabled ? ' disabled' : ''}>${htmlEscape(`${variant.label}${suffix}`)}</option>`;
  }).join('');
  if (selected) select.value = selected;
  select.disabled = running;
  renderPlentyChessRuntimeInfo();
}

function refreshStockfishControls(): void {
  inputEl('stockfishThreadsInput').disabled = running || !threadedStockfishAvailable();
  inputEl('stockfishThreadsInput').value = String(stockfishThreadsCap());
  inputEl('stockfishThreadsInput').title = `0 = auto (broker grants ${resourceBroker.cpuBudget()} on this device, ${resourceBroker.getDial()} dial)`;
}

// Lc0 big nets are WebGPU-only and locally staged; disable/downgrade them
// when WebGPU is unusable or their ONNX asset is missing.
async function refreshBt4Availability(): Promise<void> {
  if (isV0DeployProfile()) return;
  await Promise.all([
    probeBt4Support(),
    checkBigNetAsset(BIG_NETS.bt4, renderSeatSelectors),
    checkBigNetAsset(BIG_NETS.t3, renderSeatSelectors),
  ]);
  for (const row of activeSeatRows()) {
    if (row.family === 'lc0' && isLc0BigNetVariant(row.variant) && !bigNetSelectableSync(bigNetFor(row.variant).config)) row.variant = 'small';
  }
  buildEngines();
  renderSeatSelectors();
  refreshSeatControls();
}

async function renderRuntimeBadge(): Promise<void> {
  const badge = el('runtimeBadge');
  if (isV0DeployProfile()) {
    badge.textContent = 'Runtime: v0 deploy · Lc0 small + Stockfish Lite';
    badge.classList.add('ready');
    badge.classList.remove('warn');
    return;
  }
  try {
    const diag = await collectOrtRuntimeDiagnostics({ probeAdapter: true });
    runtimeIsolation = diag.crossOriginIsolated === true;
    runtimeSharedArrayBuffer = diag.wasm.sharedArrayBuffer === true;
    refreshStockfishControls();
    const webgpu = diag.webgpuAvailable ? (diag.adapter?.ok === false ? 'WebGPU unavailable/blocked' : 'WebGPU available') : 'WebGPU unavailable';
    const isolated = runtimeIsolation ? 'isolated' : 'not isolated';
    const sab = runtimeSharedArrayBuffer ? 'SAB yes' : 'SAB no';
    const sfThreads = threadedStockfishAvailable() ? 'SF threaded yes' : 'SF threaded no';
    const ortThreads = `ORT wasm threads ${diag.wasm.numThreads ?? '?'}`;
    const ortSessions = `ORT sessions ${diag.sessions.active} active`;
    badge.textContent = `Runtime: ${isolated} · ${sab} · ${webgpu} · ${diag.describe} · ${ortThreads} · ${ortSessions} · ${sfThreads} · ${tinyHybridManifestStatusText()}`;
    badge.classList.toggle('ready', runtimeIsolation && (diag.webgpuAvailable || runtimeSharedArrayBuffer));
    badge.classList.toggle('warn', !runtimeIsolation || !diag.webgpuAvailable);
  } catch (error) {
    badge.textContent = `Runtime detection failed: ${(error as Error).message}`;
    badge.classList.add('warn');
  }
}

function selectedSeatsNeedLc0Evaluator(): boolean {
  return activeSeatRows().some((row) => row.family === 'lc0' && !isLc0BigNetVariant(row.variant));
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

function pruneUnusedLc0Searchers(activeIds: Set<string>): void {
  for (const id of lc0Searchers.keys()) {
    if (activeIds.has(id)) continue;
    lc0Searchers.delete(id);
    lastLc0SearchResults.delete(id);
    lc0TreeTelemetry.delete(id);
  }
}

function disposeUnusedUciEngines(): void {
  const activeReckless = new Set(activeSeatRows().filter((row) => row.family === 'reckless').map((row) => recklessCacheKey(recklessVariantForKey(row.variant))));
  for (const [key, engine] of recklessByVariant) {
    if (!activeReckless.has(key)) { engine.dispose(); recklessByVariant.delete(key); }
  }
  const activeViridithas = new Set(activeSeatRows().filter((row) => row.family === 'viridithas').map((row) => viridithasCacheKey(viridithasVariantForKey(row.variant))));
  for (const [key, engine] of viridithasByVariant) {
    if (!activeViridithas.has(key)) { engine.dispose(); viridithasByVariant.delete(key); }
  }
  const activeBerserk = new Set(activeSeatRows().filter((row) => row.family === 'berserk').map((row) => berserkCacheKey(berserkVariantForKey(row.variant))));
  for (const [key, engine] of berserkByVariant) {
    if (!activeBerserk.has(key)) { engine.dispose(); berserkByVariant.delete(key); }
  }
  const activePlenty = new Set(activeSeatRows().filter((row) => row.family === 'plentychess').map((row) => plentyChessCacheKey(plentyChessVariantForKey(row.variant))));
  for (const [key, engine] of plentyChessByVariant) {
    if (!activePlenty.has(key)) { engine.dispose(); plentyChessByVariant.delete(key); }
  }
}

function buildEngines() {
  for (const row of activeSeatRows()) clampStrength(row);
  const activeIds = new Set(activeSeatRows().map(engineIdForRow));
  pruneUnusedLc0Searchers(activeIds);
  engines.clear();
  disposeUnusedUciEngines();
  const warmupPositions = [parseFen(START_FEN)];
  const lc0Search = (engineId: string, row: EngineRow): ArenaEngine['move'] => async (positions, signal) => {
    const timed = arenaBudgetMode() === 'movetime';
    const started = performance.now();
    const result = await lc0SearcherFor(engineId).search({ positions }, {
      visits: timed ? undefined : row.strength,
      movetimeMs: timed ? arenaMovetimeMs() : undefined,
      signal,
      yieldEveryMs: 16,
      reuseTree: true,
      batchSize: lc0BatchSize(),
      batchPipelineDepth: lc0BatchPipelineDepth(),
    });
    const elapsedMs = performance.now() - started;
    const engineName = engines.get(engineId)?.name ?? engineId;
    recordLc0SearchTelemetry(engineId, engineName, result, elapsedMs);
    recordLc0SearchOutput(engineId, engineName, result);
    return result.move ?? null;
  };
  const lc0BigNetMove = (engineId: string, row: EngineRow): ArenaEngine['move'] => async (positions, signal) => {
    const { config, searcher } = bigNetFor(row.variant);
    const onAbort = () => searcher.cancel();
    signal.addEventListener('abort', onAbort, { once: true });
    try {
      const timed = arenaBudgetMode() === 'movetime';
      const result = await searcher.search({ positions }, {
        visits: timed ? undefined : row.strength,
        movetimeMs: timed ? arenaMovetimeMs() : undefined,
        reuseTree: true,
        batchSize: config.recommendedBatchSize,
        batchPipelineDepth: config.recommendedPipelineDepth,
        evalCacheEntries: arenaCacheEntries(),
      });
      if (result.cancelled) return null;
      recordBt4SearchOutput(engineId, engines.get(engineId)?.name ?? `Lc0 ${config.name}`, result);
      return result.move ?? null;
    } finally {
      signal.removeEventListener('abort', onAbort);
    }
  };
  const tinyMove = (engineId: string, row: EngineRow): ArenaEngine['move'] => async (positions, signal) => {
    const current = positions[positions.length - 1];
    const fen = boardToFen(current);
    const evaluator = await tinyEvaluator(row.variant);
    const timed = arenaBudgetMode() === 'movetime';
    const result = await chooseMove(current, evaluator, {
      visits: timed ? undefined : row.strength,
      movetimeMs: timed ? arenaMovetimeMs() : undefined,
      batchSize: Math.max(1, Math.min(256, Math.floor(Number(params.get('tinyBatch') ?? '32') || 32))),
      signal,
      historyFens: tinyHistoryFens(positions),
      searchPolicy: montyLitePuctPolicy,
    });
    recordTinySearchOutput(engineId, engines.get(engineId)?.name ?? tinyEngineLabel(row.variant), fen, result);
    return result.move ? moveToUci(result.move) : null;
  };
  const sf = (engineId: string, row: EngineRow, kind: 'lite' | 'full'): ArenaEngine['move'] => async (positions, signal) => {
    const engine = stockfishEngineFor(kind);
    const lease = await resourceBroker.acquire({ engineId: `sf-${kind}`, signal });
    try {
      const threads = stockfishThreadsGranted(lease.threads);
      if (arenaBudgetMode() === 'movetime') engine.setOptions({ depth: undefined, movetimeMs: arenaMovetimeMs(), threads });
      else engine.setOptions({ depth: row.strength, movetimeMs: undefined, threads });
      const fen = boardToFen(positions[positions.length - 1]);
      const started = performance.now();
      const move = await engine.bestMove(fen, signal);
      const elapsedMs = performance.now() - started;
      recordStockfishOutput(engineId, engines.get(engineId)?.name ?? (kind === 'lite' ? 'Stockfish Lite' : 'Stockfish'), fen, move, engine.lastInfo(), elapsedMs);
      return move;
    } finally {
      lease.release();
    }
  };
  const recklessMove = (engineId: string, row: EngineRow, engine: RecklessEngine): ArenaEngine['move'] => async (positions, signal) => {
    if (arenaBudgetMode() === 'movetime') engine.setOptions({ depth: undefined, movetimeMs: arenaMovetimeMs() });
    else engine.setOptions({ depth: row.strength, movetimeMs: undefined });
    const fen = boardToFen(positions[positions.length - 1]);
    const started = performance.now();
    const move = await engine.bestMove(fen, signal);
    const elapsedMs = performance.now() - started;
    recordRecklessOutput(engineId, engines.get(engineId)?.name ?? 'Reckless', fen, move, engine.lastInfo(), elapsedMs);
    renderRecklessRuntimeInfo();
    return move;
  };
  const viridithasMove = (engineId: string, row: EngineRow, engine: ViridithasEngine): ArenaEngine['move'] => async (positions, signal) => {
    if (arenaBudgetMode() === 'movetime') engine.setOptions({ depth: undefined, movetimeMs: arenaMovetimeMs() });
    else engine.setOptions({ depth: row.strength, movetimeMs: undefined });
    const fen = boardToFen(positions[positions.length - 1]);
    const started = performance.now();
    const move = await engine.bestMove(fen, signal);
    const elapsedMs = performance.now() - started;
    recordViridithasOutput(engineId, engines.get(engineId)?.name ?? 'Viridithas', fen, move, engine.lastInfo(), elapsedMs);
    renderViridithasRuntimeInfo();
    return move;
  };
  const berserkMove = (engineId: string, row: EngineRow, engine: BerserkEngine): ArenaEngine['move'] => async (positions, signal) => {
    if (arenaBudgetMode() === 'movetime') engine.setOptions({ depth: undefined, movetimeMs: arenaMovetimeMs(), threads: 1 });
    else engine.setOptions({ depth: row.strength, movetimeMs: undefined, threads: 1 });
    const fen = boardToFen(positions[positions.length - 1]);
    const started = performance.now();
    const move = await engine.bestMove(fen, signal);
    const elapsedMs = performance.now() - started;
    recordBerserkOutput(engineId, engines.get(engineId)?.name ?? 'Berserk', fen, move, engine.lastInfo(), elapsedMs);
    renderBerserkRuntimeInfo();
    return move;
  };
  const plentyChessMove = (engineId: string, row: EngineRow, engine: PlentyChessEngine): ArenaEngine['move'] => async (positions, signal) => {
    if (arenaBudgetMode() === 'movetime') engine.setOptions({ depth: undefined, movetimeMs: arenaMovetimeMs(), threads: 1 });
    else engine.setOptions({ depth: row.strength, movetimeMs: undefined, threads: 1 });
    const fen = boardToFen(positions[positions.length - 1]);
    const started = performance.now();
    const move = await engine.bestMove(fen, signal);
    const elapsedMs = performance.now() - started;
    recordPlentyChessOutput(engineId, engines.get(engineId)?.name ?? 'PlentyChess', fen, move, engine.lastInfo(), elapsedMs);
    renderPlentyChessRuntimeInfo();
    return move;
  };
  const lc0SearchWarmup = (engineId: string) => async (signal: AbortSignal) => {
    const search = lc0SearcherFor(engineId);
    await search.search({ positions: warmupPositions }, { visits: 1, signal, yieldEveryMs: 16 });
    search.resetTree();
    renderCacheInfo();
  };
  const lc0BigNetWarmup = (variant: string) => async (signal: AbortSignal) => {
    const { config, searcher } = bigNetFor(variant);
    const onAbort = () => searcher.cancel();
    signal.addEventListener('abort', onAbort, { once: true });
    try {
      await searcher.search({ positions: warmupPositions }, { visits: 1, batchSize: config.recommendedBatchSize, batchPipelineDepth: config.recommendedPipelineDepth, evalCacheEntries: arenaCacheEntries() });
      await searcher.resetTree();
    } finally {
      signal.removeEventListener('abort', onAbort);
    }
  };
  const tinyWarmup = (row: EngineRow) => async (signal: AbortSignal) => {
    const evaluator = await tinyEvaluator(row.variant);
    await chooseMove(warmupPositions[0], evaluator, {
      visits: 1,
      batchSize: 1,
      signal,
      searchPolicy: montyLitePuctPolicy,
    });
  };
  const stockfishWarmup = (kind: 'lite' | 'full') => async (signal: AbortSignal) => {
    const engine = stockfishEngineFor(kind);
    // Planned (lease-free) threads so the pthread pool spawns before move one.
    engine.setOptions({ depth: 1, movetimeMs: undefined, threads: stockfishThreadsPlanned() });
    await engine.bestMove(START_FEN, signal);
  };
  const recklessWarmup = (engine: RecklessEngine) => async (signal: AbortSignal) => {
    engine.setOptions({ depth: 1, movetimeMs: undefined });
    await engine.bestMove(START_FEN, signal);
    renderRecklessRuntimeInfo();
  };
  const viridithasWarmup = (engine: ViridithasEngine) => async (signal: AbortSignal) => {
    engine.setOptions({ depth: 1, movetimeMs: undefined });
    await engine.bestMove(START_FEN, signal);
    await engine.newGame(signal);
    renderViridithasRuntimeInfo();
  };
  const berserkWarmup = (engine: BerserkEngine) => async (signal: AbortSignal) => {
    engine.setOptions({ depth: 1, movetimeMs: undefined, threads: 1 });
    await engine.bestMove(START_FEN, signal);
    await engine.newGame(signal);
    renderBerserkRuntimeInfo();
  };
  const plentyChessWarmup = (engine: PlentyChessEngine) => async (signal: AbortSignal) => {
    engine.setOptions({ depth: 1, movetimeMs: undefined, threads: 1 });
    await engine.bestMove(START_FEN, signal);
    await engine.newGame(signal);
    renderPlentyChessRuntimeInfo();
  };
  for (const row of activeSeatRows()) {
    const id = engineIdForRow(row);
    if (engines.has(id)) continue;
    if (row.family === 'lc0') {
      if (isLc0BigNetVariant(row.variant)) engines.set(id, { id, name: `${rowLabel(row)} v${row.strength}`, move: lc0BigNetMove(id, row), warmup: lc0BigNetWarmup(row.variant) });
      else engines.set(id, { id, name: `${rowLabel(row)} v${row.strength}`, move: lc0Search(id, row), warmup: lc0SearchWarmup(id) });
    } else if (row.family === 'tiny') {
      engines.set(id, { id, name: `${rowLabel(row)} v${row.strength}`, move: tinyMove(id, row), warmup: tinyWarmup(row) });
    } else if (row.family === 'sf') {
      const kind = row.variant === 'full' ? 'full' : 'lite';
      engines.set(id, { id, name: `${rowLabel(row)} d${row.strength}`, move: sf(id, row, kind), warmup: stockfishWarmup(kind) });
    } else if (row.family === 'reckless') {
      const variant = recklessVariantForKey(row.variant);
      if (variant.key !== 'custom' && recklessVariantAssetStatus(variant) === 'missing') {
        engines.set(id, { id, name: `${rowLabel(row)} d${row.strength}`, move: async () => { throw new Error(recklessMissingAssetMessage([variant])); } });
        continue;
      }
      const engine = getRecklessFor(row.variant);
      prewarmReckless(engine);
      engines.set(id, { id, name: `${rowLabel(row)} d${row.strength}`, move: recklessMove(id, row, engine), warmup: recklessWarmup(engine) });
    } else if (row.family === 'viridithas') {
      const engine = getViridithasFor(row.variant);
      engines.set(id, { id, name: `${rowLabel(row)} d${row.strength}`, move: viridithasMove(id, row, engine), warmup: viridithasWarmup(engine) });
    } else if (row.family === 'berserk') {
      const engine = getBerserkFor(row.variant);
      engines.set(id, { id, name: `${rowLabel(row)} d${row.strength}`, move: berserkMove(id, row, engine), warmup: berserkWarmup(engine) });
    } else {
      const variant = plentyChessVariantForKey(row.variant);
      const unsupportedReason = plentyChessVariantUnsupportedReason(variant);
      if (unsupportedReason) {
        engines.set(id, { id, name: `${rowLabel(row)} d${row.strength}`, move: async () => { throw new Error(`${variant.label} ${unsupportedReason}.`); } });
        continue;
      }
      const engine = getPlentyChessFor(row.variant);
      engines.set(id, { id, name: `${rowLabel(row)} d${row.strength}`, move: plentyChessMove(id, row, engine), warmup: plentyChessWarmup(engine) });
    }
  }
  renderRecklessRuntimeInfo();
  renderViridithasRuntimeInfo();
  renderBerserkRuntimeInfo();
  renderPlentyChessRuntimeInfo();
  renderEngineDiagnosticsInfo();
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
  setBoardSideEngines(null, null, null, null);
  renderBoard();
}

function refreshBudgetControls(): void {
  const movetime = arenaBudgetMode() === 'movetime';
  el('movetimeField').hidden = !movetime;
  el('matchupNote').textContent = movetime
    ? 'Every engine gets the same time per move (strength fields are ignored); colors alternate each game.'
    : 'Each engine searches to its configured strength; colors alternate each game.';
}

function refreshOpeningPreview(): void {
  const select = selectEl('startingPositionSelect');
  const textarea = el('openingText') as HTMLTextAreaElement;
  const custom = select.value === 'custom';
  el('openingTextField').hidden = !custom;
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

function formatScoreHalf(value: number): string {
  return value % 1 ? `${Math.floor(value) || ''}½` : String(value);
}

function renderMatchScore(nameA: string, nameB: string, sameEngine: boolean, score: MatchScore): void {
  if (score.games === 0) { el('matchScore').textContent = 'No games played yet.'; return; }
  const games = `${score.games} game${score.games === 1 ? '' : 's'}`;
  el('matchScore').textContent = sameEngine
    ? `${nameA} mirror · ${score.aWins}–${score.bWins}–${score.draws} (W–L–D, Engine 1 seat) over ${games}`
    : `${nameA} ${formatScoreHalf(score.a)} – ${formatScoreHalf(score.b)} ${nameB} · ${score.aWins}W ${score.draws}D ${score.bWins}L over ${games}`;
}

function renderStandings(standings: TournamentStandings, scheduledGames: number): void {
  const rows = standings.table();
  const elo = (row: ReturnType<TournamentStandings['table']>[number]) => row.eloDiff === null
    ? '—'
    : `${row.eloDiff >= 0 ? '+' : ''}${Math.round(row.eloDiff)}${row.eloError !== null ? ` ±${Math.round(row.eloError)}` : ''}`;
  const body = rows.map((row, i) =>
    `<tr><td>${i + 1}</td><td>${htmlEscape(row.name)}</td><td>${elo(row)}</td><td>${row.wins}</td><td>${row.draws}</td><td>${row.losses}</td><td>${formatScoreHalf(row.points) || '0'}</td><td>${row.games}</td></tr>`).join('');
  el('matchScore').innerHTML = `<div class="small">${standings.totalGames()}/${scheduledGames} games · Elo vs pool, ±95% approx</div>`
    + `<table class="standings"><thead><tr><th>#</th><th>Engine</th><th>Elo</th><th>W</th><th>D</th><th>L</th><th>Pts</th><th>G</th></tr></thead><tbody>${body}</tbody></table>`;
}

function appendLog(text: string, gameIndex?: number) {
  const div = document.createElement('div');
  div.textContent = text;
  if (gameIndex !== undefined) {
    div.dataset.game = String(gameIndex);
    div.classList.add('replayable');
    div.title = 'Click to replay this game on the board';
  }
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

function chartColorFor(engineId: string): string {
  const index = Math.max(0, activeEngineIds.indexOf(engineId));
  return CHART_COLORS[index % CHART_COLORS.length];
}

function formatCompactNumber(value: number): string {
  if (value >= 1e6) return `${(value / 1e6).toFixed(1)}M`;
  if (value >= 1000) return `${Math.round(value / 1000)}k`;
  return String(Math.round(value));
}

function resetGameCharts(): void {
  gameChartSamples = [];
  rootChartContext = null;
  el('chartsPanel').hidden = false;
  for (const id of ['evalChart', 'timeChart', 'npsChart', 'rootChart']) el(id).innerHTML = '';
  el('rootChartTitle').textContent = 'LC0 root visits';
  el('chartLegend').innerHTML = '';
}

function renderGameCharts(): void {
  const ids = [...new Set(gameChartSamples.map((sample) => sample.engineId))];
  const seriesFor = (value: (sample: GameChartSample) => number | undefined): ChartSeries[] => ids.map((id) => ({
    label: engines.get(id)?.name ?? id,
    color: chartColorFor(id),
    points: gameChartSamples
      .filter((sample) => sample.engineId === id)
      .flatMap((sample) => {
        const y = value(sample);
        return y === undefined || !Number.isFinite(y) ? [] : [{ x: sample.ply, y }];
      }),
  }));
  el('evalChart').innerHTML = lineChartSvg(seriesFor((sample) => sample.whiteScore), { yMin: 0, yMax: 1, midline: 0.5, formatY: (v) => `${Math.round(v * 100)}%` });
  el('timeChart').innerHTML = lineChartSvg(seriesFor((sample) => sample.moveMs), { yMin: 0, formatY: formatCompactNumber });
  el('npsChart').innerHTML = lineChartSvg(seriesFor((sample) => sample.nps), { yMin: 0, formatY: formatCompactNumber });
  el('chartLegend').innerHTML = ids.map((id) => `<span><span class="swatch" style="background:${chartColorFor(id)}"></span>${htmlEscape(engines.get(id)?.name ?? id)}</span>`).join('');
}

/** Live PUCT-root distribution from the real search tree (Lc0SearchResult.children). */
function renderRootChart(engineId: string): void {
  const result = lastLc0SearchResults.get(engineId);
  if (!result?.children?.length) return;
  const top = [...result.children].sort((a, b) => b.visits - a.visits).slice(0, 8).filter((child) => child.visits > 0);
  if (!top.length) return;
  rootChartContext = { fen: result.fen, top: top.map((child) => ({ uci: child.uci, visits: child.visits })) };
  el('rootChartTitle').textContent = `${engines.get(engineId)?.name ?? engineId} root visits (Q from side to move)`;
  el('rootChart').innerHTML = hBarChartSvg(top.map((child) => ({
    label: child.uci,
    value: child.visits,
    detail: `Q ${signed(child.q, 2)} · P ${Math.round(child.prior * 100)}%`,
    color: chartColorFor(engineId),
  })));
}

function recordGameChartSample(ply: number, engine: ArenaEngine, moveMs: number): void {
  const snapshot = engineOutputs.get(engine.id);
  const lc0 = lastLc0SearchResults.get(engine.id);
  const nps = snapshot?.nps ?? (lc0 && moveMs > 0 ? Math.round((lc0.visits / moveMs) * 1000) : undefined);
  gameChartSamples.push({ ply, engineId: engine.id, whiteScore: snapshot?.evalBar?.whiteScore, moveMs: Math.round(moveMs), nps });
  renderGameCharts();
  renderRootChart(engine.id);
}

async function playArenaGame(white: ArenaEngine, black: ArenaEngine, opening: ArenaOpening, signal: AbortSignal): Promise<{ result: GameResultCode; reason: string; tree: GameTree }> {
  pendingLc0ReplyProbes.clear();
  const tree = gameTreeFromOpening(opening);
  historyBoards = tree.historyBoards();
  board = historyBoards[historyBoards.length - 1];
  lastUci = null;
  activeEngineIds = [white.id, black.id];
  engineOutputs.delete(white.id);
  engineOutputs.delete(black.id);
  thinkingEngineIds.delete(white.id);
  thinkingEngineIds.delete(black.id);
  setBoardSideEngines(white.id, white.name, black.id, black.name);
  liveTrail = trailFromTree(tree, `${white.name} vs ${black.name}`, historyBoards.length - 1);
  renderBoard();
  renderEngineOutputs();
  resetGameCharts();
  const priorFens: string[] = historyBoards.slice(0, -1).map(boardToFen);
  const delay = Math.max(0, Math.floor(Number(inputEl('delayInput').value) || 0));
  for (let ply = 0; ply < 300; ply++) {
    if (signal.aborted) return { result: '1/2-1/2', reason: 'cancelled', tree };
    const outcome = gameOutcome(board, priorFens);
    if (outcome) return { ...outcome, tree };
    const engine = board.turn === 'w' ? white : black;
    recordEngineThinking(engine);
    let uci: string | null;
    const moveStarted = performance.now();
    try {
      uci = await engine.move(historyBoards, signal);
    } catch (error) {
      if (isAbortError(error)) return { result: '1/2-1/2', reason: 'cancelled', tree };
      throw error;
    }
    if (signal.aborted) return { result: '1/2-1/2', reason: 'cancelled', tree };
    const move = legalFromUci(board, uci);
    if (!move) return { result: board.turn === 'w' ? '0-1' : '1-0', reason: uci ? `illegal ${uci}` : 'resigned', tree };
    recordGameChartSample(ply, engine, performance.now() - moveStarted);
    recordPendingLc0ReplyProbes(engine, board, move);
    priorFens.push(boardToFen(board));
    board = makeMove(board, move);
    noteLc0MoveForReplyProbe(engine, move, board);
    historyBoards.push(board);
    lastUci = moveToUci(move);
    const node = tree.addUci(lastUci);
    liveTrail?.entries.push({ fen: boardToFen(board), uci: lastUci, san: node?.san ?? null });
    renderBoard();
    await sleep(delay, signal);
  }
  return { result: '1/2-1/2', reason: 'max plies', tree };
}

async function ensureSelectedRecklessAssetsAvailable(): Promise<boolean> {
  const variants = [...new Map(activeSeatRows()
    .filter((row) => row.family === 'reckless')
    .map((row) => {
      const variant = recklessVariantForKey(row.variant);
      return [recklessCacheKey(variant), variant] as const;
    })).values()];
  if (!variants.length) return true;
  const missing: RecklessVariant[] = [];
  for (const variant of variants) {
    const status = await checkRecklessVariantAsset(variant, () => {
      renderRecklessRuntimeInfo();
      renderSeatSelectors();
      refreshSeatControls();
    });
    if (status === 'missing' && variant.key !== 'custom') missing.push(variant);
  }
  if (!missing.length) return true;
  renderRecklessRuntimeInfo();
  renderSeatSelectors();
  refreshSeatControls();
  el('message').textContent = recklessMissingAssetMessage(missing);
  return false;
}

function clearStartPending(): void {
  startPending = false;
  if (!running) el('start').toggleAttribute('disabled', false);
}

async function startMatch() {
  if (running || startPending) return;
  startPending = true;
  el('start').toggleAttribute('disabled', true);
  syncSeatRowsFromDom();
  if (!(await ensureSelectedRecklessAssetsAvailable())) { clearStartPending(); return; }
  if (running) { clearStartPending(); return; }
  buildEngines();
  populateSeats();
  const mode = arenaTournamentMode();
  const participantIdx = mode === 'match' ? [0, 1] : seatRows.map((_, index) => index);
  const idCounts = new Map<string, number>();
  for (const index of participantIdx) {
    const engineId = seatEngineId(index);
    idCounts.set(engineId, (idCounts.get(engineId) ?? 0) + 1);
  }
  const participants = participantIdx.map((index) => {
    const engineId = seatEngineId(index);
    const engine = engines.get(engineId);
    // Duplicate engine configs are legal (mirror/self-play pools); standings
    // key on seat pids and names get a seat suffix to stay distinguishable.
    const name = engine && (idCounts.get(engineId) ?? 0) > 1 && mode !== 'match' ? `${engine.name} (seat ${index + 1})` : engine?.name ?? engineId;
    return { pid: String(index), index, engineId, engine, name };
  });
  if (participants.length < 2 || participants.some((participant) => !participant.engine)) {
    el('message').textContent = 'Pick at least two engines.';
    clearStartPending();
    return;
  }
  const byPid = new Map(participants.map((participant) => [participant.pid, participant]));
  const bigNetRows = activeSeatRows().filter((row) => row.family === 'lc0' && isLc0BigNetVariant(row.variant));
  if (bigNetRows.length && !(await probeBt4Support())) {
    el('message').textContent = 'Lc0 big nets need WebGPU, which is unavailable in this browser.';
    clearStartPending();
    return;
  }
  const checkedBigNetVariants = new Set<string>();
  for (const row of bigNetRows) {
    if (!isLc0BigNetVariant(row.variant) || checkedBigNetVariants.has(row.variant)) continue;
    checkedBigNetVariants.add(row.variant);
    const { config } = bigNetFor(row.variant);
    const status = await checkBigNetAsset(config, renderSeatSelectors);
    if (status !== 'present') {
      el('message').textContent = bigNetUnavailableText(config);
      renderSeatSelectors();
      refreshSeatControls();
      clearStartPending();
      return;
    }
  }
  const engineA = participants[0].engine!;
  const engineB = participants[1].engine!;
  const sameEngine = mode === 'match' && participants[0].engineId === participants[1].engineId;
  const seatIds = [...new Set(participants.map((participant) => participant.engineId))];
  const gamesPerOpening = Math.max(1, Math.floor(Number(inputEl('gamesInput').value) || 2));
  const standings = new TournamentStandings(participants.map((participant) => ({ id: participant.pid, name: participant.name })));
  let schedule: ScheduledGame<ArenaOpening>[];
  try {
    // Colors alternate per game index within each pairing (see tournament.ts).
    schedule = buildSchedule(tournamentPairings(mode, participants.map((participant) => participant.pid)), selectedOpenings(), gamesPerOpening);
  } catch (error) {
    el('message').textContent = `Opening setup error: ${(error as Error).message}`;
    clearStartPending();
    return;
  }
  if (!schedule.length) { el('message').textContent = 'No games to play.'; clearStartPending(); return; }

  startPending = false;
  running = true;
  abort = new AbortController();
  refreshOpeningPreview();
  refreshStockfishControls();
  refreshRecklessVariantUi();
  refreshViridithasVariantUi();
  refreshBerserkVariantUi();
  refreshPlentyChessVariantUi();
  refreshSeatControls();
  games.length = 0;
  activeEngineIds = [];
  engineOutputs.clear();
  engineOutputHistory.length = 0;
  engineOutputTotalCount = 0;
  thinkingEngineIds.clear();
  lastLc0SearchResults.clear();
  pendingLc0ReplyProbes.clear();
  lc0TreeTelemetry.clear();
  uciTelemetry.clear();
  bt4Telemetry.clear();
  el('log').innerHTML = '';
  finishedTrails.length = 0;
  exitReview();
  renderEngineOutputs();
  renderSearchTelemetryInfo();
  el('start').toggleAttribute('disabled', true);
  el('stop').toggleAttribute('disabled', false);
  const score: MatchScore = { a: 0, b: 0, aWins: 0, bWins: 0, draws: 0, games: 0 };
  if (mode === 'match') renderMatchScore(engineA.name, engineB.name, sameEngine, score);
  else renderStandings(standings, schedule.length);
  try {
    resetLc0SearchTrees(seatIds);
    await warmUpSelectedEngines(seatIds, abort.signal);
    if (abort.signal.aborted) return;
    for (let i = 0; i < schedule.length; i++) {
      if (abort.signal.aborted) break;
      const { whiteId, blackId, opening } = schedule[i];
      const white = byPid.get(whiteId)!;
      const black = byPid.get(blackId)!;
      const whiteEngine = white.engine!;
      const blackEngine = black.engine!;
      // Reset trees per game (fresh game tree); within a game the shared tree is
      // reused across both sides' plies — i.e. self-play when both seats match.
      resetLc0SearchTrees(seatIds);
      for (const searcher of Object.values(bigNetSearchers)) if (searcher.loaded) await searcher.resetTree();
      for (const engine of viridithasByVariant.values()) await engine.newGame(abort.signal);
      for (const engine of berserkByVariant.values()) await engine.newGame(abort.signal);
      for (const engine of plentyChessByVariant.values()) await engine.newGame(abort.signal);
      setBoardSideEngines(whiteEngine.id, whiteEngine.name, blackEngine.id, blackEngine.name);
      el('pairing').textContent = `Game ${i + 1}/${schedule.length}: ${whiteEngine.name} (W) vs ${blackEngine.name} (B) · ${opening.name}`;
      el('message').textContent = 'Playing…';
      const { result, reason, tree } = await playArenaGame(whiteEngine, blackEngine, opening, abort.signal);
      if (reason === 'cancelled') break;
      score.games += 1;
      standings.record(whiteId, blackId, result);
      if (result === '1/2-1/2') { score.draws += 1; score.a += 0.5; score.b += 0.5; }
      else {
        const winnerIsA = (whiteId === participants[0].pid) === (result === '1-0');
        if (winnerIsA) { score.a += 1; score.aWins += 1; } else { score.b += 1; score.bWins += 1; }
      }
      const tags: Record<string, string> = { Event: 'LC0 arena', White: white.name, Black: black.name, Opening: opening.name, ...openingPgnSetupTags(opening) };
      games.push({ pgn: gameTreeToPgn(tree, tags, result) });
      if (liveTrail) {
        liveTrail.label = `Game ${i + 1}: ${whiteEngine.name} vs ${blackEngine.name} (${result})`;
        finishedTrails[i] = liveTrail;
      }
      renderCacheInfo();
      appendLog(`${i + 1}. ${whiteEngine.name} vs ${blackEngine.name} [${opening.name}]: ${result} (${reason}) · ${engineRuntimeDiagnosticsText()} · ${searchTelemetryText()}`, i);
      if (mode === 'match') renderMatchScore(engineA.name, engineB.name, sameEngine, score);
      else renderStandings(standings, schedule.length);
    }
    el('message').textContent = abort.signal.aborted ? `Stopped after ${score.games} game(s).` : `Match done (${score.games} game${score.games === 1 ? '' : 's'}).`;
    el('pairing').textContent = abort.signal.aborted ? 'Match stopped.' : 'Match finished.';
  } catch (error) {
    if (isAbortError(error) || abort?.signal.aborted) {
      el('message').textContent = `Stopped after ${score.games} game(s).`;
      el('pairing').textContent = 'Match stopped.';
    } else {
      el('message').textContent = `Match failed: ${(error as Error).message}`;
    }
  } finally {
    running = false;
    startPending = false;
    abort = null;
    activeEngineIds = [];
    engineOutputs.clear();
    thinkingEngineIds.clear();
    el('start').toggleAttribute('disabled', false);
    el('stop').toggleAttribute('disabled', true);
    refreshOpeningPreview();
    refreshStockfishControls();
    refreshRecklessVariantUi();
    refreshViridithasVariantUi();
    refreshBerserkVariantUi();
    refreshPlentyChessVariantUi();
    refreshSeatControls();
  }
}

function exportPgn() {
  inputEl('pgnOut').value = games.map((g) => g.pgn).join('\n\n');
  el('message').textContent = games.length ? `Exported ${games.length} game(s) as PGN.` : 'No games to export yet.';
}

function disposeLc0Resources(): void {
  void lc0Cache?.dispose();
  lc0Cache = null;
  player = null;
  searcher = null;
  lc0Searchers.clear();
  lastLc0SearchResults.clear();
  pendingLc0ReplyProbes.clear();
  lc0TreeTelemetry.clear();
  uciTelemetry.clear();
  bt4Telemetry.clear();
  engineOutputs.clear();
  engineOutputHistory.length = 0;
  engineOutputTotalCount = 0;
  thinkingEngineIds.clear();
  activeEngineIds = [];
}

function disposeRuntimeResources(): void {
  abort?.abort();
  abort = null;
  disposeLc0Resources();
  tinyEvaluatorPromises.clear();
  disposeStockfish();
  for (const engine of recklessByVariant.values()) engine.dispose();
  recklessByVariant.clear();
  for (const engine of viridithasByVariant.values()) engine.dispose();
  viridithasByVariant.clear();
  for (const engine of berserkByVariant.values()) engine.dispose();
  berserkByVariant.clear();
  for (const engine of plentyChessByVariant.values()) engine.dispose();
  plentyChessByVariant.clear();
  for (const searcher of Object.values(bigNetSearchers)) searcher.dispose();
}

async function createSelectedLc0Evaluator(): Promise<Lc0OnnxEvaluator | Lc0WebHybridEvaluator | Lc0WholeOnnxWebgpuEvaluator> {
  const runtime = selectedLc0Runtime();
  if (runtime === 'onnx') {
    const modelLoad = await loadLc0ModelForOrt(MODEL_URL, { cache: false });
    return Lc0OnnxEvaluator.create(modelLoad.model);
  }
  if (runtime === LC0_WHOLE_MODEL_WEBGPU_RUNTIME) {
    return Lc0WholeOnnxWebgpuEvaluator.create({
      manifestUrl: LC0_WHOLE_MODEL_MANIFEST_URL,
      batch: lc0WholeModelPhysicalBatch(),
      fetchTensorCache: lc0WholeModelTensorCache(),
      logger: (line) => console.info('[lc0 whole-model arena]', line),
    });
  }
  const headBackend = runtime === 'hybrid-wgsl-heads' ? 'wgsl' : 'ort';
  const legalPriorsBackend = lc0HybridLegalPriorsBackend();
  return new Lc0WebHybridEvaluator({
    packUrl: PACK_URL,
    layers: lc0EncoderLayers(),
    verifyShards: params.get('packVerify') !== '0',
    headBackend,
    wgslBatchMode: 'physical',
    inputBackend: lc0HybridInputBackend(),
    legalPriorsBackend: legalPriorsBackend === 'gpu' && headBackend !== 'wgsl' ? 'js' : legalPriorsBackend,
    encoderKernelVariant: lc0EncoderKernelVariant(),
  });
}

async function loadLc0Evaluator(): Promise<void> {
  const runtime = selectedLc0Runtime();
  if (!selectedSeatsNeedLc0Evaluator()) {
    el('start').toggleAttribute('disabled', false);
    refreshSeatControls();
    el('message').textContent = 'Ready (LC0 not loaded for current matchup). Pick engines and start a tournament.';
    return;
  }
  loadingLc0 = true;
  el('start').toggleAttribute('disabled', true);
  refreshSeatControls();
  const hybrid = lc0HybridConfigLabel(runtime);
  el('message').textContent = `Loading LC0 ${lc0RuntimeLabel(runtime)}${hybrid ? ` (${hybrid})` : ''}…`;
  try {
    const evaluator = await createSelectedLc0Evaluator();
    publishBrowserRuntimeAudit({
      source: 'lc0-arena-evaluator',
      surface: 'arena',
      family: 'lc0',
      engineLabel: 'LC0',
      modelId: 'lc0-default',
      modelUrl: runtime === 'onnx' ? MODEL_URL : runtime === LC0_WHOLE_MODEL_WEBGPU_RUNTIME ? LC0_WHOLE_MODEL_MANIFEST_URL : PACK_URL,
      requestedRuntime: runtime,
      resolvedRuntime: lc0ResolvedRuntime(runtime),
      runtimeConfigId: runtime === 'onnx' ? undefined : runtime,
      manifestUrl: runtime === 'onnx' ? undefined : runtime === LC0_WHOLE_MODEL_WEBGPU_RUNTIME ? LC0_WHOLE_MODEL_MANIFEST_URL : PACK_URL,
      searchBudget: activeSeatRows().filter((row) => row.family === 'lc0' && !isLc0BigNetVariant(row.variant)).map((row) => budgetText(row)).join(', '),
      notes: [lc0RuntimeLabel(runtime), hybrid, runtime === 'onnx' ? undefined : runtime === LC0_WHOLE_MODEL_WEBGPU_RUNTIME ? 'whole-model runtime is research-only and opt-in' : 'hybrid runtime is pack-lazy until first evaluation succeeds'].filter((part): part is string => !!part),
    });
    lc0Cache = new CachedLc0Evaluator(evaluator, { maxEntries: arenaCacheEntries() });
    lc0Searchers.clear();
    player = new Lc0PolicyOnlyPlayer(lc0Cache);
    searcher = new Lc0PuctSearcher(lc0Cache);
    renderCacheInfo();
    el('start').toggleAttribute('disabled', false);
    el('message').textContent = `Ready (${lc0RuntimeLabel(runtime)}${hybrid ? ` · ${hybrid}` : ''}). Pick engines and start a tournament.`;
  } catch (error) {
    if (!selectedSeatsNeedLc0Evaluator()) {
      el('start').toggleAttribute('disabled', false);
      el('message').textContent = `LC0 ${lc0RuntimeLabel(runtime)} load failed, but current non-LC0 matchup is ready: ${(error as Error).message}`;
    } else {
      el('message').textContent = `LC0 ${lc0RuntimeLabel(runtime)} load failed: ${(error as Error).message}`;
    }
  } finally {
    loadingLc0 = false;
    refreshSeatControls();
  }
}

async function reloadLc0Evaluator(): Promise<void> {
  abort?.abort();
  disposeLc0Resources();
  renderCacheInfo();
  await loadLc0Evaluator();
}

function setBenchResult(result: unknown): void {
  const node = el('benchResult');
  node.hidden = false;
  node.textContent = JSON.stringify(result, null, 2);
}

function mapValues<T>(map: Map<string, T>): Record<string, T> {
  return Object.fromEntries(map.entries());
}

async function safeOrtRuntimeDiagnostics(): Promise<unknown> {
  try { return await collectOrtRuntimeDiagnostics({ probeAdapter: false }); }
  catch (error) { return { error: (error as Error).message }; }
}

function fixedSuiteFensFromParams(): string[] {
  const raw = params.get('fixedSuiteFens') ?? params.get('fens') ?? '';
  return raw.split(/[|\n]/).map((fen) => fen.trim()).filter(Boolean);
}

function stockfishScoreMs(): number {
  return intParam('stockfishScoreMs', arenaMovetimeMs(), 1, 60_000);
}

function stockfishScoreDepth(): number | undefined {
  const raw = params.get('stockfishScoreDepth');
  if (raw == null || raw === '') return undefined;
  return intParam('stockfishScoreDepth', 8, 1, 245);
}

async function runFixedSuiteBenchAutorun(): Promise<void> {
  if (params.get('fixedSuiteBench') !== '1' && params.get('benchmark') !== 'fixed-suite') return;
  const fens = fixedSuiteFensFromParams();
  setBenchResult({ status: 'LC0_FIXED_SUITE_RUNNING', runtime: selectedLc0Runtime(), positions: fens.length, startedAt: new Date().toISOString() });
  const started = performance.now();
  const controller = new AbortController();
  try {
    if (!fens.length) throw new Error('fixedSuiteFens is empty');
    engineOutputs.clear();
    engineOutputHistory.length = 0;
    engineOutputTotalCount = 0;
    thinkingEngineIds.clear();
    lastLc0SearchResults.clear();
    pendingLc0ReplyProbes.clear();
    lc0TreeTelemetry.clear();
    uciTelemetry.clear();
    bt4Telemetry.clear();
    buildEngines();
    const lc0Id = seatEngineId(0);
    const lc0Row = seatRows[0];
    if (lc0Row.family !== 'lc0') throw new Error('fixed suite expects seat A to be LC0');
    const lc0Name = engines.get(lc0Id)?.name ?? rowLabel(lc0Row);
    const sfKind = seatRows[1].family === 'sf' && seatRows[1].variant === 'full' ? 'full' : 'lite';
    const sfId = seatRows[1].family === 'sf' ? seatEngineId(1) : 'sf:lite:8';
    const sfName = seatRows[1].family === 'sf' ? (engines.get(sfId)?.name ?? rowLabel(seatRows[1])) : 'Stockfish Lite d8';
    await warmUpSelectedEngines([lc0Id, sfId], controller.signal);
    const sfEngine = stockfishEngineFor(sfKind);
    const timed = arenaBudgetMode() === 'movetime';
    const scoreMs = stockfishScoreMs();
    const scoreDepth = stockfishScoreDepth();
    sfEngine.setOptions({ depth: scoreDepth ?? (timed ? undefined : seatRows[1].strength), movetimeMs: scoreDepth === undefined && timed ? scoreMs : undefined, threads: stockfishThreadsPlanned() });
    resetLc0SearchTrees([lc0Id]);
    const positions = [];
    for (let i = 0; i < fens.length; i++) {
      const boardAtMove = parseFen(fens[i]);
      const searchStarted = performance.now();
      const search = await lc0SearcherFor(lc0Id).search({ positions: [boardAtMove] }, {
        visits: timed ? undefined : lc0Row.strength,
        movetimeMs: timed ? arenaMovetimeMs() : undefined,
        signal: controller.signal,
        yieldEveryMs: 16,
        reuseTree: false,
        batchSize: lc0BatchSize(),
        batchPipelineDepth: lc0BatchPipelineDepth(),
      });
      const searchElapsedMs = performance.now() - searchStarted;
      recordLc0SearchTelemetry(lc0Id, lc0Name, search, searchElapsedMs);
      recordLc0SearchOutput(lc0Id, lc0Name, search);
      const legal = legalFromUci(boardAtMove, search.move ?? null);
      if (!legal || !search.move) {
        positions.push({ index: i + 1, fen: fens[i], lc0Move: search.move, error: 'LC0 returned no legal move' });
        continue;
      }
      const afterBoard = makeMove(boardAtMove, legal);
      const afterFen = boardToFen(afterBoard);
      const scoreStarted = performance.now();
      const sfMove = await sfEngine.bestMove(afterFen, controller.signal);
      const scoreElapsedMs = performance.now() - scoreStarted;
      const lines = sfEngine.lastInfo();
      recordStockfishOutput(sfId, sfName, afterFen, sfMove, lines, scoreElapsedMs);
      const best = lines[0];
      const whiteCp = stockfishWhiteCp(best, afterFen);
      const mateInWhitePov = stockfishMateInWhitePov(best, afterFen);
      positions.push({
        index: i + 1,
        fen: fens[i],
        sideToMove: boardAtMove.turn,
        lc0Move: search.move,
        lc0Pv: search.pv,
        lc0Search: {
          visits: search.visits,
          evals: search.search.stats?.evalCalls ?? 0,
          cacheHits: search.search.stats?.cacheHits ?? 0,
          elapsedMs: searchElapsedMs,
          batchSize: search.search.stats?.batchSize,
          batchPipelineDepth: search.search.stats?.batchPipelineDepth,
          maxEvalBatch: search.search.stats?.maxEvalBatch,
          evalBatchSizeHistogram: search.search.stats?.evalBatchSizeHistogram,
          evalBackendTimingMeans: search.search.stats?.evalBackendTimingMeans,
          evalBackendTimingPerPositionMeans: search.search.stats?.evalBackendTimingPerPositionMeans,
          qWhite: toWhiteQ(search.value, search.fen),
        },
        afterFen,
        stockfish: {
          scoreMovetimeMs: scoreDepth === undefined ? scoreMs : undefined,
          scoreDepth,
          reply: sfMove,
          whiteCp,
          mateInWhitePov,
          lc0PerspectiveCp: whiteCp === undefined ? undefined : (boardAtMove.turn === 'w' ? whiteCp : -whiteCp),
          lc0PerspectiveMate: mateInWhitePov === undefined ? undefined : (boardAtMove.turn === 'w' ? mateInWhitePov : -mateInWhitePov),
          depth: best?.depth,
          nodes: best?.nodes,
          nps: best?.nps,
          pv: best?.pvUci,
          elapsedMs: scoreElapsedMs,
        },
      });
    }
    const cpValues = positions.map((p) => p.stockfish?.lc0PerspectiveCp).filter((value): value is number => Number.isFinite(value));
    const ortRuntimeDiagnostics = await safeOrtRuntimeDiagnostics();
    const result = {
      status: 'LC0_FIXED_SUITE_DONE',
      runtime: selectedLc0Runtime(),
      runtimeLabel: lc0RuntimeLabel(),
      elapsedMs: Math.round(performance.now() - started),
      configuration: {
        seatA: { ...seatRows[0], id: lc0Id, label: rowLabel(seatRows[0]) },
        stockfish: { id: sfId, label: sfName, scoreMovetimeMs: scoreDepth === undefined ? scoreMs : undefined, scoreDepth, threads: stockfishThreadsPlanned() },
        budgetMode: arenaBudgetMode(),
        movetimeMs: arenaMovetimeMs(),
        cacheEntries: arenaCacheEntries(),
        lc0BatchSize: lc0BatchSize(),
        lc0BatchPipelineDepth: lc0BatchPipelineDepth(),
        positions: fens.length,
        ortRuntimeDiagnostics,
      },
      summary: {
        avgStockfishLc0PerspectiveCp: cpValues.length ? cpValues.reduce((sum, value) => sum + value, 0) / cpValues.length : null,
        evaluatedCpPositions: cpValues.length,
        runtimeDiagnostics: el('cacheInfo').textContent ?? '',
        searchDiagnostics: el('searchTelemetryInfo').textContent ?? '',
      },
      telemetry: {
        lc0Tree: mapValues(lc0TreeTelemetry),
        uci: mapValues(uciTelemetry),
        lc0Cache: lc0Cache?.metrics(),
        lc0CacheFootprint: lc0CacheFootprint(),
        lc0ExecutionFootprint: lc0ExecutionFootprint(),
      },
      positions,
      engineOutputCount: engineOutputTotalCount,
      engineOutputRetainedCount: engineOutputHistory.length,
      engineOutputsTruncated: engineOutputHistory.length < engineOutputTotalCount,
      engineOutputs: [...engineOutputHistory],
    };
    setBenchResult(result);
  } catch (error) {
    setBenchResult({ status: 'LC0_FIXED_SUITE_FAILED', runtime: selectedLc0Runtime(), elapsedMs: Math.round(performance.now() - started), error: (error as Error).message, stack: (error as Error).stack });
  }
}

async function runArenaBenchAutorun(): Promise<void> {
  if (params.get('arenaBench') !== '1' && params.get('benchmark') !== 'arena') return;
  setBenchResult({ status: 'ARENA_BENCH_RUNNING', runtime: selectedLc0Runtime(), startedAt: new Date().toISOString() });
  const started = performance.now();
  try {
    if ((el('start') as HTMLButtonElement).disabled) throw new Error(el('message').textContent || 'arena start is disabled');
    await startMatch();
    const matchMessage = el('message').textContent ?? '';
    if (!games.length || /^(Match failed|Opening setup error|No games|Select two|Stopped|Lc0 BT4 needs)/.test(matchMessage)) {
      throw new Error(matchMessage || 'arena benchmark did not complete any games');
    }
    const ortRuntimeDiagnostics = await safeOrtRuntimeDiagnostics();
    const result = {
      status: 'ARENA_BENCH_DONE',
      runtime: selectedLc0Runtime(),
      runtimeLabel: lc0RuntimeLabel(),
      elapsedMs: Math.round(performance.now() - started),
      configuration: {
        seatA: { ...seatRows[0], id: seatEngineId(0), label: rowLabel(seatRows[0]) },
        seatB: { ...seatRows[1], id: seatEngineId(1), label: rowLabel(seatRows[1]) },
        budgetMode: arenaBudgetMode(),
        movetimeMs: arenaMovetimeMs(),
        gamesPerOpening: Math.max(1, Math.floor(Number(inputEl('gamesInput').value) || 2)),
        openingSuite: selectEl('startingPositionSelect').value,
        openings: selectedOpenings().map((opening) => opening.name),
        cacheEntries: arenaCacheEntries(),
        lc0BatchSize: lc0BatchSize(),
        lc0BatchPipelineDepth: lc0BatchPipelineDepth(),
        stockfishThreads: stockfishThreadsPlanned(),
        ortRuntimeDiagnostics,
      },
      summary: {
        message: el('message').textContent ?? '',
        pairing: el('pairing').textContent ?? '',
        matchScore: el('matchScore').textContent ?? '',
        runtimeDiagnostics: el('cacheInfo').textContent ?? '',
        searchDiagnostics: el('searchTelemetryInfo').textContent ?? '',
        runtimeBadge: el('runtimeBadge').textContent ?? '',
      },
      telemetry: {
        lc0Tree: mapValues(lc0TreeTelemetry),
        uci: mapValues(uciTelemetry),
        bt4: mapValues(bt4Telemetry),
        lc0Cache: lc0Cache?.metrics(),
        lc0CacheFootprint: lc0CacheFootprint(),
        lc0ExecutionFootprint: lc0ExecutionFootprint(),
      },
      engineOutputCount: engineOutputTotalCount,
      engineOutputRetainedCount: engineOutputHistory.length,
      engineOutputsTruncated: engineOutputHistory.length < engineOutputTotalCount,
      engineOutputs: [...engineOutputHistory],
      log: [...el('log').children].reverse().map((node) => node.textContent ?? ''),
      pgn: games.map((game) => game.pgn).join('\n\n'),
    };
    setBenchResult(result);
  } catch (error) {
    setBenchResult({ status: 'ARENA_BENCH_FAILED', runtime: selectedLc0Runtime(), elapsedMs: Math.round(performance.now() - started), error: (error as Error).message, stack: (error as Error).stack });
  }
}

function wireEvents() {
  el('start').addEventListener('click', () => { void startMatch(); });
  el('stop').addEventListener('click', () => { abort?.abort(); el('message').textContent = 'Stopping…'; });
  el('exportPgn').addEventListener('click', exportPgn);
  // History review: charts, move strip, and finished-game log rows jump the
  // board to a past position; Live (or Escape) returns to the running game.
  el('revStart').addEventListener('click', () => stepReview('start'));
  el('revPrev').addEventListener('click', () => stepReview(-1));
  el('revNext').addEventListener('click', () => stepReview(1));
  el('revEnd').addEventListener('click', () => stepReview('end'));
  el('revLive').addEventListener('click', exitReview);
  for (const id of ['evalChart', 'timeChart', 'npsChart'] as const) {
    el(id).addEventListener('click', (event) => {
      if (!liveTrail) return;
      const ply = chartPlyFromClick(id, event.target as HTMLElement, event as MouseEvent);
      if (ply === null) return;
      enterReview(liveTrail, liveTrail.openingPlies + ply);
    });
  }
  el('rootChart').addEventListener('click', (event) => reviewRootChartClick(event as MouseEvent));
  el('gameMoves').addEventListener('click', (event) => {
    const span = (event.target as HTMLElement).closest('.mv') as HTMLElement | null;
    if (!span) return;
    const trail = reviewing && reviewTrail ? reviewTrail : liveTrail;
    if (trail) enterReview(trail, Number(span.dataset.idx));
  });
  el('log').addEventListener('click', (event) => {
    const row = (event.target as HTMLElement).closest('[data-game]') as HTMLElement | null;
    const trail = row ? finishedTrails[Number(row.dataset.game)] : undefined;
    if (trail) enterReview(trail, trail.entries.length - 1);
  });
  if (arenaKeydownHandler) document.removeEventListener('keydown', arenaKeydownHandler);
  arenaKeydownHandler = (event: KeyboardEvent) => {
    if (!reviewing) return;
    const target = event.target as HTMLElement | null;
    if (target?.closest?.('input,textarea,select')) return;
    if (event.key === 'ArrowLeft') { event.preventDefault(); stepReview(-1); }
    else if (event.key === 'ArrowRight') { event.preventDefault(); stepReview(1); }
    else if (event.key === 'Escape') exitReview();
  };
  document.addEventListener('keydown', arenaKeydownHandler);
  el('arenaSeatList').addEventListener('change', (event) => {
    if (running) return;
    const target = event.target as HTMLInputElement | HTMLSelectElement;
    const seat = Number(target.dataset.seat);
    if (!Number.isInteger(seat) || !seatRows[seat]) return;
    const row = seatRows[seat];
    if (target.classList.contains('seat-fam')) {
      row.family = target.value as EngineFamily;
      row.variant = defaultVariant(row.family);
      row.strength = strengthMeta(row.family).def;
    } else if (target.classList.contains('seat-var')) {
      if (row.family === 'lc0' && isLc0BigNetVariant(target.value) && !window.confirm(`${bigNetLoadWarning(BIG_NETS[target.value])}\n\nUse Lc0 ${BIG_NETS[target.value].name}?`)) { target.value = row.variant; return; }
      row.variant = target.value;
    } else if (target.classList.contains('seat-strength')) {
      row.strength = Number(target.value);
      clampStrength(row);
    }
    normalizeSeatRowForDeploy(seat);
    for (const key of ['bt4', 't3'] as const) if (!activeSeatRows().some((r) => r.family === 'lc0' && r.variant === key)) bigNetSearchers[key].dispose();
    buildEngines();
    populateSeats();
    if (selectedSeatsNeedLc0Evaluator() && !lc0Cache && !loadingLc0) void loadLc0Evaluator();
    else if (!loadingLc0) el('start').toggleAttribute('disabled', false);
  });
  el('arenaSeatList').addEventListener('input', (event) => {
    if (running) return;
    const target = event.target as HTMLInputElement;
    const seat = Number(target.dataset.seat);
    if (Number.isInteger(seat) && seatRows[seat] && target.classList.contains('seat-strength')) {
      seatRows[seat].strength = Number(target.value);
      clampStrength(seatRows[seat]);
      renderRecklessRuntimeInfo();
      renderViridithasRuntimeInfo();
      renderBerserkRuntimeInfo();
      renderPlentyChessRuntimeInfo();
    }
  });
  el('arenaSeatList').addEventListener('click', (event) => {
    if (running) return;
    const target = event.target as HTMLElement;
    if (!target.classList.contains('seat-remove')) return;
    const index = Number(target.dataset.seat);
    if (!Number.isInteger(index) || seatRows.length <= 2) return;
    syncSeatRowsFromDom();
    seatRows.splice(index, 1);
    buildEngines();
    populateSeats();
  });
  el('addSeat').addEventListener('click', () => {
    if (running) return;
    syncSeatRowsFromDom();
    seatRows.push({ family: 'sf', variant: 'lite', strength: 8 });
    buildEngines();
    populateSeats();
  });
  el('tournamentModeSelect').addEventListener('change', () => {
    if (running) return;
    syncSeatRowsFromDom();
    buildEngines();
    populateSeats();
    renderRecklessRuntimeInfo();
    renderViridithasRuntimeInfo();
    renderBerserkRuntimeInfo();
    renderPlentyChessRuntimeInfo();
  });
  el('startingPositionSelect').addEventListener('change', refreshOpeningPreview);
  el('openingText').addEventListener('input', refreshOpeningPreview);
  el('cacheEntriesInput').addEventListener('input', () => { renderCacheInfo(); resetLc0SearchTrees(); });
  el('lc0PresetSelect').addEventListener('change', () => applyLc0Preset(selectedLc0Preset(), { reload: true }));
  el('lc0BatchSizeInput').addEventListener('input', () => { markLc0PresetCustom(); inputEl('lc0BatchSizeInput').value = String(lc0BatchSize()); renderCacheInfo(); resetLc0SearchTrees(); });
  el('lc0BatchPipelineDepthInput').addEventListener('input', () => { markLc0PresetCustom(); inputEl('lc0BatchPipelineDepthInput').value = String(lc0BatchPipelineDepth()); renderCacheInfo(); resetLc0SearchTrees(); });
  el('lc0RuntimeSelect').addEventListener('change', () => { markLc0PresetCustom(); refreshSeatControls(); if (!running) void reloadLc0Evaluator(); });
  for (const id of ['lc0InputBackendSelect', 'lc0EncoderKernelSelect', 'lc0LegalPriorsSelect']) {
    el(id).addEventListener('change', () => { markLc0PresetCustom(); renderCacheInfo(); if (!running && selectedLc0Runtime() !== 'onnx') void reloadLc0Evaluator(); });
  }
  el('budgetModeSelect').addEventListener('change', () => {
    refreshBudgetControls();
    resetLc0SearchTrees();
  });
  el('movetimeInput').addEventListener('input', () => resetLc0SearchTrees());
  el('stockfishThreadsInput').addEventListener('input', () => {
    if (running) return;
    inputEl('stockfishThreadsInput').value = String(stockfishThreadsCap());
    // Threads flips single<->threaded flavor (different wasm); rebuild on next use.
    disposeStockfish();
  });
  window.addEventListener('pagehide', (event) => {
    if (!(event as PageTransitionEvent).persisted) disposeRuntimeResources();
  });
}

async function init() {
  if (!isV0DeployProfile()) {
    REQUESTED_RECKLESS_VARIANT = await resolveDefaultRecklessVariantAssetFallback(REQUESTED_RECKLESS_VARIANT, REQUESTED_RECKLESS_EXPLICIT, renderRecklessRuntimeInfo);
  }
  REQUESTED_VIRIDITHAS_VARIANT = await resolveDefaultViridithasVariantAssetFallback(REQUESTED_VIRIDITHAS_VARIANT, REQUESTED_VIRIDITHAS_EXPLICIT, renderRecklessRuntimeInfo);
  REQUESTED_BERSERK_VARIANT = await resolveDefaultBerserkVariantAssetFallback(REQUESTED_BERSERK_VARIANT, REQUESTED_BERSERK_EXPLICIT, renderRecklessRuntimeInfo);
  REQUESTED_PLENTYCHESS_VARIANT = await resolveDefaultPlentyChessVariantAssetFallback(REQUESTED_PLENTYCHESS_VARIANT, REQUESTED_PLENTYCHESS_EXPLICIT, renderRecklessRuntimeInfo);
  renderBoard();
  installRuntimeAuditPanel();
  installExperimentalLc0RuntimeOption();
  selectEl('lc0RuntimeSelect').value = initialLc0Runtime();
  if (!isV0DeployProfile()) refreshRecklessVariantUi();
  refreshViridithasVariantUi();
  refreshBerserkVariantUi();
  refreshPlentyChessVariantUi();
  selectEl('recklessVariantSelect').value = REQUESTED_RECKLESS_VARIANT.key;
  selectEl('viridithasVariantSelect').value = REQUESTED_VIRIDITHAS_VARIANT.key;
  selectEl('berserkVariantSelect').value = REQUESTED_BERSERK_VARIANT.key;
  selectEl('plentychessVariantSelect').value = REQUESTED_PLENTYCHESS_VARIANT.key;
  applyArenaQueryParams();
  applyLc0Preset(inferredLc0Preset());
  if (!isV0DeployProfile()) await ensureSelectedRecklessAssetsAvailable();
  renderRecklessRuntimeInfo();
  renderViridithasRuntimeInfo();
  renderBerserkRuntimeInfo();
  renderPlentyChessRuntimeInfo();
  // sfThreads=0 (or sfThreads=auto) hands thread selection to the resource broker.
  inputEl('stockfishThreadsInput').value = String(Math.max(0, Math.min(32, Math.floor(Number(params.get('sfThreads') ?? '1') || 0))));
  refreshStockfishControls();
  void renderRuntimeBadge();
  if (!isV0DeployProfile()) await refreshTinyHybridManifestStatus();
  buildEngines();
  populateSeats();
  if (!isV0DeployProfile()) void refreshBt4Availability();
  if (!isV0DeployProfile()) void probeEngineLogos();
  wireEvents();
  refreshBudgetControls();
  refreshOpeningPreview();
  await loadLc0Evaluator();
  void renderRuntimeBadge();
  void runFixedSuiteBenchAutorun();
  void runArenaBenchAutorun();
}

export function mountArenaBrowser(): () => void {
  void init();
  return () => {
    disposeRuntimeResources();
    if (arenaKeydownHandler) document.removeEventListener('keydown', arenaKeydownHandler);
    arenaKeydownHandler = null;
    (ground as { destroy?: () => void } | null)?.destroy?.();
    ground = null;
  };
}
