import { Chessground } from 'chessground';
import artifactChannel from '../../public/channels/stable.json';
import type { DrawShape } from 'chessground/draw';
import type { Key } from 'chessground/types';
import { type BoardState, boardToFen, parseFen, START_FEN } from '../chess/board.ts';
import { parseValidatedFen } from '../chess/fenValidation.ts';
import { type Move, moveToUci } from '../chess/moveCodec.ts';
import { legalMoves, makeMove } from '../chess/movegen.ts';
import { gameTreeToPgn } from '../chess/pgn.ts';
import { uciLineToSan, uciToSan } from '../chess/san.ts';
import { createBrowserSquareformerRuntimeEvaluator } from '../nn/browserRuntimeEvaluator.ts';
import { CachedEvaluator, type Evaluator } from '../nn/evaluator.ts';
import { collectOrtRuntimeDiagnostics } from '../nn/ortRuntime.ts';
import {
  BROWSER_RUNTIME_AUDIT_EVENT,
  type BrowserRuntimeAuditDetail,
  browserRuntimeAuditIdentity,
  formatBrowserRuntimeAudit,
  publishBrowserRuntimeAudit,
  reconcileRuntimeFallbackWarning,
  updateRuntimeFallbackWarning,
} from '../nn/runtimeAudit.ts';
import type { Node as PuctNode } from '../search/puct.ts';
import { type SearchResult as CentipawnSearchResult, chooseMove, montyLitePuctPolicy } from '../search/puct.ts';
import { engineBrushes, evalBarWhitePercent, formatScore, qToCentipawns } from './analysisFormat.ts';
import { analysisEngineName, engineMonogram } from './analysisPresentation.ts';
import {
  appendCpuArenaTrail,
  arenaChartEngineName,
  type CpuArenaPosition,
  type CpuArenaSeat,
  captureCpuArenaDiagnostics,
  cpuArenaRuntimeSummary,
  playCpuArenaGame,
  settleCpuArenaSearch,
} from './arenaCpuGame.ts';
import {
  ARENA_CONTROL_IDS, ArenaCheckpointConflictError, ArenaCheckpointInvalidError, type ArenaCheckpoint, type ArenaCheckpointToken, type ArenaChartSample,
  type ArenaCompletedGame, type ArenaCheckpointConfig, type ArenaRootChartSample, arenaCompletedTree, arenaRemainingIndices, claimArenaCheckpoint, commitArenaGame,
  createArenaCheckpoint, discardArenaCheckpoint, hydrateArenaSchedule, loadArenaCheckpoint, normalizeArenaControls, saveArenaCheckpoint,
} from './arenaCheckpoint.ts';
import { type ArenaOpening, BUILTIN_ARENA_OPENINGS, parseArenaOpenings } from './arenaOpenings.ts';
import { loadArenaResource } from './arenaResourceLoad.ts';
import { arenaChartColorIndex, arenaOpeningPairs, arenaUciChartSample, formatArenaPoints } from './arenaResults.ts';
import { arenaCpuOnlyPool, arenaGameConcurrency, runArenaSchedule } from './arenaScheduleRunner.ts';
import { resolvePublicAssetUrl } from './assetUrls.ts';
import { BerserkEngine } from './berserkEngine.ts';
import {
  BERSERK_VARIANTS,
  type BerserkVariant,
  berserkVariantAssetStatus,
  berserkVariantByKey,
  berserkVariantFromParams,
  checkBerserkVariantAsset,
  hasExplicitBerserkVariant,
  normalizeBerserkVariant,
  resolveDefaultBerserkVariantAssetFallback,
} from './berserkVariants.ts';
import { acquireBigNetSearcher, type BigNetKey, disposeBigNetSearcherNow, peekBigNetSearcher, releaseBigNetSearcher } from './bigNetSessionPool.ts';
import { boardCheck } from './boardUx.ts';
import { el, htmlEscape, inputEl, maybeEl, selectEl } from './browserDom.ts';
import {
  type BrowserLc0Runtime,
  boundedIntValue,
  lc0EncoderLayers as browserLc0EncoderLayers,
  lc0RuntimeLabel as browserLc0RuntimeLabel,
  lc0WholeModelPhysicalBatch as browserLc0WholeModelPhysicalBatch,
  lc0WholeModelRuntimeRequested as browserLc0WholeModelRuntimeRequested,
  lc0WholeModelTensorCache as browserLc0WholeModelTensorCache,
  initialLc0Runtime as initialBrowserLc0Runtime,
  installExperimentalLc0RuntimeOption as installBrowserLc0RuntimeOption,
  LC0_WHOLE_MODEL_WEBGPU_RUNTIME,
  normalizeLc0Runtime,
} from './browserLc0Runtime.ts';
import type { BrowserUciEngine } from './browserUciEngine.ts';
import { browserVariantOption, createBrowserVariantSelector } from './browserVariantOption.ts';
import {
  BIG_NETS,
  type BigNetConfig,
  type Bt4SearchResult,
  type Bt4WorkerSearcher,
  bigNetAssetStatusSync,
  bigNetLoadWarning,
  bt4SupportedSync,
  checkBigNetAsset,
  probeBt4Support,
} from './bt4Engine.ts';
import { type ChartSeries, hBarChartSvg, lineChartSvg } from './charts.ts';
import { DisposableVariantPool } from './disposableVariantPool.ts';
import { type GameResultCode, gameOutcome } from './engineBattle.ts';
import {
  canonicalEngineFamily,
  centipawnEngineLabel,
  centipawnVariantOptions,
  defaultStaticEngineVariant,
  type EngineFamily,
  type EngineRow,
  engineFamilyOptions,
  engineResourceProfile,
  engineStrengthMeta,
  isLc0BigNetVariant,
  isV0DeployProfile,
  lc0EngineLabel,
  lc0VariantOptions,
  normalizeDeployEngineRow,
  stockfishEngineLabel,
  stockfishVariantOptions,
} from './engineCatalog.ts';
import { probeEngineLogos } from './engineLogos.ts';
import {
  berserkCacheKey,
  cpuEngineHashMbForSurface,
  createBerserkEngine,
  createPlentyChessEngine,
  createRecklessEngine,
  createStormphraxEngine,
  createViridithasEngine,
  plentyChessCacheKey,
  recklessCacheKey,
  stormphraxCacheKey,
  viridithasCacheKey,
} from './engineProvision.ts';
import { engineVariantMenuLabel } from './engineVariantLabels.ts';
import { GameTree } from './gameTree.ts';
import { type Lc0WebEncoderKernelVariant, type Lc0WebExecutionFootprint, Lc0WebHybridEvaluator } from './hybridEvaluator.ts';
import { hideLoadingProgress, renderLoadingProgress } from './loadingProgress.ts';
import { loadLc0ModelForOrt } from './modelCache.ts';
import { CachedLc0Evaluator, type Lc0EvaluationCacheFootprint, type Lc0EvaluationCacheMetrics, Lc0OnnxEvaluator } from './onnxEvaluator.ts';
import { PlentyChessEngine } from './plentychessEngine.ts';
import {
  checkPlentyChessVariantAsset,
  hasExplicitPlentyChessVariant,
  normalizePlentyChessVariant,
  PLENTYCHESS_VARIANTS,
  type PlentyChessVariant,
  plentyChessVariantAssetStatus,
  plentyChessVariantByKey,
  plentyChessVariantFromParams,
  plentyChessVariantUnsupportedReason,
  resolveDefaultPlentyChessVariantAssetFallback,
} from './plentychessVariants.ts';
import { Lc0PolicyOnlyPlayer } from './policyOnlyPlayer.ts';
import { formatRecklessBrowserApiLoadStatus, RecklessEngine } from './recklessEngine.ts';
import {
  checkRecklessVariantAsset,
  hasExplicitRecklessVariant,
  normalizeRecklessVariant,
  RECKLESS_VARIANTS,
  type RecklessVariant,
  recklessVariantAssetStatus,
  recklessVariantByKey,
  recklessVariantFromParams,
  resolveDefaultRecklessVariantAssetFallback,
  supportsWasmRelaxedSimd,
} from './recklessVariants.ts';
import { EngineResourceBroker, loadPerformanceDial, type PerformanceDial } from './resourceBroker.ts';
import { Lc0PuctSearcher, type Lc0SearchProgress, type Lc0SearchResult } from './search.ts';
import { StockfishEngine, type StockfishFlavor, type StockfishInfoLine, stockfishFlavorLabel, stockfishFlavorUrl } from './stockfishEngine.ts';
import { StormphraxEngine } from './stormphraxEngine.ts';
import {
  checkStormphraxVariantAsset,
  hasExplicitStormphraxVariant,
  normalizeStormphraxVariant,
  resolveDefaultStormphraxVariantAssetFallback,
  STORMPHRAX_VARIANTS,
  type StormphraxVariant,
  stormphraxVariantAssetStatus,
  stormphraxVariantByKey,
  stormphraxVariantFromParams,
  stormphraxVariantUnsupportedReason,
} from './stormphraxVariants.ts';
import { buildSchedule, type ScheduledGame, type TournamentMode, TournamentStandings, tournamentPairings } from './tournament.ts';
import { canUsePersistentViridithasWasi, ViridithasEngine } from './viridithasEngine.ts';
import {
  checkViridithasVariantAsset,
  hasExplicitViridithasVariant,
  normalizeViridithasVariant,
  resolveDefaultViridithasVariantAssetFallback,
  VIRIDITHAS_VARIANTS,
  type ViridithasVariant,
  viridithasVariantAssetStatus,
  viridithasVariantByKey,
  viridithasVariantFromParams,
} from './viridithasVariants.ts';
import { Lc0WholeOnnxWebgpuEvaluator } from './wholeOnnxWebgpuEvaluator.ts';

type Ground = ReturnType<typeof Chessground>;
// Seats are array indices into seatRows; tournament pids are String(index).
interface ArenaWarmupStatus {
  phase: string;
  detail: string;
}

interface ArenaEngine {
  id: string;
  name: string;
  move(positions: BoardState[], signal: AbortSignal): Promise<string | null>;
  warmup?(signal: AbortSignal): Promise<void>;
  warmupStatus?: ArenaWarmupStatus;
}
interface GameRecord {
  pgn: string;
  scheduleIndex: number;
  result: GameResultCode;
}
interface MatchScore {
  a: number;
  b: number;
  aWins: number;
  bWins: number;
  draws: number;
  games: number;
}
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
  /** Display eval from White's perspective in pawns ("+0.16", "#3"); Lc0 Q is mapped with lc0's Q->cp curve. */
  evalText?: string;
  /** Lc0-family WDL from White's perspective. */
  wdlWhite?: [number, number, number];
  qWhite?: number;
  visits?: number;
}

const params = new URLSearchParams(location.search);
const DEFAULT_MODEL_URL = resolvePublicAssetUrl('/models/lc0/t1-256x10-distilled-swa-2432500.batch1.f16.qdq8.onnx');
const MODEL_URL = isV0DeployProfile() ? DEFAULT_MODEL_URL : resolvePublicAssetUrl(params.get('model') ?? DEFAULT_MODEL_URL);
const DEFAULT_PACK_URL = resolvePublicAssetUrl('/models/lc0/t1-256x10-distilled-swa-2432500.batch8.f16.lc0web/model.lc0web.json');
const PACK_URL = isV0DeployProfile() ? DEFAULT_PACK_URL : resolvePublicAssetUrl(params.get('pack') ?? params.get('modelPack') ?? DEFAULT_PACK_URL);
const DEFAULT_CENTIPAWN_MODEL_URL = '/models/bt4_soap_rem_c19000_final.onnx';
const DEFAULT_CENTIPAWN_META_URL = '/models/bt4_soap_rem_c19000_final.meta.json';
const LEGACY_CENTIPAWN_MODEL_URL = '/models/bt4_anneal_muon_best.onnx';
const LEGACY_CENTIPAWN_META_URL = '/models/bt4_anneal_muon_best.meta.json';
const DEFAULT_CENTIPAWN_TVMJS_MANIFEST_URL = '/runtimes/centipawn-tvmjs-webgpu/bt4-soap-rem-c19000-final/f32/v2-shape-k16/manifest.json';
const LEGACY_CENTIPAWN_HYBRID_MANIFEST_URL = '/runtimes/squareformer-tvm-hybrid/bt4-anneal-muon-best/v1/manifest.json';
const DEFAULT_LC0_WHOLE_MODEL_MANIFEST_URL = '/runtimes/lc0-' + 'tvm' + 'js-webgpu/t1-256x10-distilled-swa-2432500/f16/v1/manifest.json';
const CENTIPAWN_MODEL_URL = resolvePublicAssetUrl(
  isV0DeployProfile() ? DEFAULT_CENTIPAWN_MODEL_URL : (params.get('centipawnModel') ?? params.get('centipawnOnnx') ?? DEFAULT_CENTIPAWN_MODEL_URL),
);
const CENTIPAWN_META_URL = resolvePublicAssetUrl(
  isV0DeployProfile() ? DEFAULT_CENTIPAWN_META_URL : (params.get('centipawnMeta') ?? DEFAULT_CENTIPAWN_META_URL),
);
const LEGACY_CENTIPAWN_RESOLVED_MODEL_URL = resolvePublicAssetUrl(params.get('tinyModel') ?? params.get('tinyOnnx') ?? LEGACY_CENTIPAWN_MODEL_URL);
const LEGACY_CENTIPAWN_RESOLVED_META_URL = resolvePublicAssetUrl(params.get('tinyMeta') ?? LEGACY_CENTIPAWN_META_URL);
const LEGACY_CENTIPAWN_OVERRIDE =
  !isV0DeployProfile() && (params.has('tinyModel') || params.has('tinyOnnx') || params.has('tinyMeta') || params.has('tinyManifest'));
const CENTIPAWN_HYBRID_MANIFEST_URL =
  params.get('centipawnManifest') ??
  params.get('manifest') ??
  params.get('manifestUrl') ??
  params.get('tinyManifest') ??
  (LEGACY_CENTIPAWN_OVERRIDE ? LEGACY_CENTIPAWN_HYBRID_MANIFEST_URL : DEFAULT_CENTIPAWN_TVMJS_MANIFEST_URL);
const LC0_WHOLE_MODEL_MANIFEST_URL =
  params.get('wholeModelManifest') ?? params.get('wholeModelManifestUrl') ?? params.get('tvm' + 'jsManifest') ?? DEFAULT_LC0_WHOLE_MODEL_MANIFEST_URL;
type Lc0ArenaRuntime = BrowserLc0Runtime;
type Lc0ArenaPreset = 'stable' | 'benchmarked-small' | 'custom';
const REQUESTED_RECKLESS_EXPLICIT = hasExplicitRecklessVariant(params);
let REQUESTED_RECKLESS_VARIANT = recklessVariantFromParams(params);
const REQUESTED_VIRIDITHAS_EXPLICIT = hasExplicitViridithasVariant(params);
let REQUESTED_VIRIDITHAS_VARIANT = viridithasVariantFromParams(params);
const REQUESTED_BERSERK_EXPLICIT = hasExplicitBerserkVariant(params);
let REQUESTED_BERSERK_VARIANT = berserkVariantFromParams(params);
const REQUESTED_PLENTYCHESS_EXPLICIT = hasExplicitPlentyChessVariant(params);
let REQUESTED_PLENTYCHESS_VARIANT = plentyChessVariantFromParams(params);
const REQUESTED_STORMPHRAX_EXPLICIT = hasExplicitStormphraxVariant(params);
let REQUESTED_STORMPHRAX_VARIANT = stormphraxVariantFromParams(params);
const recklessBrowserVariants = createBrowserVariantSelector({
  builtIns: RECKLESS_VARIANTS,
  requested: () => REQUESTED_RECKLESS_VARIANT,
  normalize: normalizeRecklessVariant,
  byKey: recklessVariantByKey,
});
const viridithasBrowserVariants = createBrowserVariantSelector({
  builtIns: VIRIDITHAS_VARIANTS,
  requested: () => REQUESTED_VIRIDITHAS_VARIANT,
  normalize: normalizeViridithasVariant,
  byKey: viridithasVariantByKey,
});
const berserkBrowserVariants = createBrowserVariantSelector({
  builtIns: BERSERK_VARIANTS,
  requested: () => REQUESTED_BERSERK_VARIANT,
  normalize: normalizeBerserkVariant,
  byKey: berserkVariantByKey,
  usable: (variant) => Boolean(variant.jsUrl),
});
const plentyChessBrowserVariants = createBrowserVariantSelector({
  builtIns: PLENTYCHESS_VARIANTS,
  requested: () => REQUESTED_PLENTYCHESS_VARIANT,
  normalize: normalizePlentyChessVariant,
  byKey: plentyChessVariantByKey,
});
const stormphraxBrowserVariants = createBrowserVariantSelector({
  builtIns: STORMPHRAX_VARIANTS,
  requested: () => REQUESTED_STORMPHRAX_VARIANT,
  normalize: normalizeStormphraxVariant,
  byKey: stormphraxVariantByKey,
});

let ground: Ground | null = null;
let mountAbort = new AbortController();
let arenaMountedCallbackCache = new WeakMap<() => void, () => void>();
function isStaleMount(signal: AbortSignal = mountAbort.signal): boolean {
  return signal.aborted || signal !== mountAbort.signal;
}
function whileArenaMounted(callback: () => void): () => void {
  const existing = arenaMountedCallbackCache.get(callback);
  if (existing) return existing;
  const signal = mountAbort.signal;
  const guarded = () => {
    if (!isStaleMount(signal)) callback();
  };
  arenaMountedCallbackCache.set(callback, guarded);
  return guarded;
}
function assetProbePending(status: string): boolean {
  return status === 'unknown' || status === 'checking';
}
let arenaKeydownHandler: ((event: KeyboardEvent) => void) | null = null;
let arenaAuditHandler: ((event: Event) => void) | null = null;
let arenaPagehideHandler: ((event: PageTransitionEvent) => void) | null = null;
let board: BoardState = parseFen(START_FEN);
let historyBoards: BoardState[] = [board];
let lastUci: string | null = null;

// Game-history review: clicking a chart, the move strip, or a finished game in
// the log flips the board into a read-only replay; "Live" returns to the
// running game (which keeps playing in the background meanwhile).
interface TrailEntry {
  fen: string;
  uci: string | null;
  san: string | null;
}
interface GameTrail {
  label: string;
  entries: TrailEntry[];
  openingPlies: number;
  charts: ArenaChartSample[];
  rootChart?: ArenaRootChartSample;
  sides?: { whiteName: string; blackName: string };
}
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
let lc0LoadAbort: AbortController | null = null;
let running = false;
let startPending = false;
let abort: AbortController | null = null;
let player: Lc0PolicyOnlyPlayer | null = null;
let searcher: Lc0PuctSearcher | null = null;
let lc0Cache: CachedLc0Evaluator | null = null;
let stockfishLite: StockfishEngine | null = null;
let stockfishFull: StockfishEngine | null = null;
const ARENA_CPU_ENGINE_OPTIONS = { surface: 'arena' } as const;
const recklessEngines = new DisposableVariantPool(recklessCacheKey, (variant: RecklessVariant) =>
  createRecklessEngine(variant, renderRecklessRuntimeInfo, ARENA_CPU_ENGINE_OPTIONS),
);
const viridithasEngines = new DisposableVariantPool(viridithasCacheKey, (variant: ViridithasVariant) =>
  createViridithasEngine(variant, {}, ARENA_CPU_ENGINE_OPTIONS),
);
const berserkEngines = new DisposableVariantPool(berserkCacheKey, (variant: BerserkVariant) => createBerserkEngine(variant, ARENA_CPU_ENGINE_OPTIONS));
const plentyChessEngines = new DisposableVariantPool(plentyChessCacheKey, (variant: PlentyChessVariant) =>
  createPlentyChessEngine(variant, ARENA_CPU_ENGINE_OPTIONS),
);
const stormphraxEngines = new DisposableVariantPool(stormphraxCacheKey, (variant: StormphraxVariant) =>
  createStormphraxEngine(variant, ARENA_CPU_ENGINE_OPTIONS),
);
const centipawnEvaluatorPromises = new Map<string, Promise<Evaluator>>();
const centipawnEvaluatorLoadTokens = new Map<string, symbol>();
const centipawnEvaluators = new Set<CachedEvaluator>();
const centipawnEvaluatorsByKey = new Map<string, CachedEvaluator>();
let centipawnEvaluatorGeneration = 0;
let centipawnHybridManifestStatus: 'unknown' | 'present' | 'missing' = 'unknown';
const ARENA_BIG_NET_KEYS: readonly BigNetKey[] = ['bt4', 't3'];
function bigNetFor(variant: string): { config: BigNetConfig; searcher: Bt4WorkerSearcher } {
  const key: BigNetKey = variant === 't3' ? 't3' : 'bt4';
  return { config: BIG_NETS[key], searcher: acquireBigNetSearcher(key) };
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
const isolatedCpuEngines = new Set<BrowserUciEngine>();
const isolatedCpuSeats = new Map<string, { seat: CpuArenaSeat; row: EngineRow }>();
let cpuTournamentBroker: EngineResourceBroker | null = null;
const engines = new Map<string, ArenaEngine>();
const lc0Searchers = new Map<string, Lc0PuctSearcher>();
const lastLc0SearchResults = new Map<string, Lc0SearchResult>();
type GameChartSample = ArenaChartSample;
let gameChartSamples: GameChartSample[] = [];
const CHART_COLORS = [
  'var(--chart-1,#4a7a2a)',
  'var(--chart-2,#a5461b)',
  'var(--chart-3,#1c5f8a)',
  'var(--chart-4,#7a4a9a)',
  'var(--chart-5,#8a7a1c)',
  'var(--chart-6,#5a5a5a)',
];
const pendingLc0ReplyProbes = new Map<string, PendingLc0ReplyProbe>();
const lc0TreeTelemetry = new Map<string, Lc0TreeTelemetry>();
const uciTelemetry = new Map<string, UciEngineTelemetry>();
const bt4Telemetry = new Map<string, Bt4Telemetry>();
const engineOutputs = new Map<string, EngineOutputSnapshot>();
const engineOutputHistory: EngineOutputSnapshot[] = [];
let engineOutputTotalCount = 0;
const MAX_ENGINE_OUTPUT_HISTORY = 1000;
const thinkingEngineIds = new Set<string>();
/** When the engine on move started thinking (performance.now()), for the live strip clock. */
const thinkingSince = new Map<string, number>();
/** Search time used by each side in the game on the board. */
const sideTimeMs = { w: 0, b: 0 };
let arenaClockTimer: ReturnType<typeof setInterval> | null = null;
let arenaChartResizeObserver: ResizeObserver | null = null;
interface ArenaGameResultChip {
  index: number;
  result: GameResultCode;
  whiteName: string;
  blackName: string;
  /** Engine 1's points in a two-engine match (null in pooled tournaments). */
  aPoints: number | null;
}
const gameResultChips: ArenaGameResultChip[] = [];
let scheduledGameCount = 0;
interface ArenaSearchProgress {
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
const arenaSearchProgress = new Map<string, ArenaSearchProgress>();
let activeEngineIds: string[] = [];
const games: GameRecord[] = [];
let resultsSchedule: ScheduledGame<ArenaOpening>[] = [];
let resultsNames = new Map<string, string>();
let checkpoint: ArenaCheckpoint | null = null;
let checkpointToken: ArenaCheckpointToken | null = null;
let checkpointWrites: Promise<void> = Promise.resolve();
let checkpointProblem = '';
let checkpointUnreadable = false;
let checkpointSaving = false;
let pendingCheckpointSaves = 0;
const seatRows: EngineRow[] = [
  { family: 'lc0', variant: 'small', strength: 100 },
  { family: 'sf', variant: 'lite', strength: 8 },
];
const ARENA_SEAT_STORAGE_KEY = '0x88-arena-seats-v1';
let bt4AvailabilityResolved = false;

function persistedVariantOrDefault(family: EngineFamily, value: unknown): string {
  const variant = String(value ?? '');
  const options = variantOptions(family);
  const selected = options.find((option) => option.value === variant);
  const pendingBigNet = !bt4AvailabilityResolved && family === 'lc0' && isLc0BigNetVariant(variant);
  if (selected && (!selected.disabled || pendingBigNet || /\(checking\b/.test(selected.label))) return selected.value;
  return options.find((option) => !option.disabled)?.value ?? defaultVariant(family);
}

function restoreArenaSeatRows(): void {
  try {
    const parsed = JSON.parse(localStorage.getItem(ARENA_SEAT_STORAGE_KEY) ?? 'null');
    if (!Array.isArray(parsed) || parsed.length < 2) return;
    const restored = parsed.slice(0, 8).flatMap((value): EngineRow[] => {
      if (!value || typeof value !== 'object') return [];
      const raw = value as Record<string, unknown>;
      const family = canonicalEngineFamily(String(raw.family ?? ''));
      if (!family) return [];
      const row: EngineRow = { family, variant: persistedVariantOrDefault(family, raw.variant), strength: Number(raw.strength) };
      clampStrength(row);
      return [row];
    });
    if (restored.length >= 2) seatRows.splice(0, seatRows.length, ...restored);
  } catch {
    // Preferences are optional when storage is blocked or corrupt.
  }
}

function persistArenaSeatRows(): void {
  try {
    localStorage.setItem(ARENA_SEAT_STORAGE_KEY, JSON.stringify(seatRows));
  } catch {
    /* optional preference */
  }
}

function formatNps(nps: number): string {
  if (nps >= 1e6) return `${(nps / 1e6).toFixed(1)}M`;
  if (nps >= 1e3) return `${Math.round(nps / 1e3)}k`;
  return String(Math.round(nps));
}

function formatNodes(nodes: number): string {
  if (nodes >= 1e9) return `${(nodes / 1e9).toFixed(1)}B`;
  if (nodes >= 1e6) return `${(nodes / 1e6).toFixed(1)}M`;
  if (nodes >= 1e3) return `${(nodes / 1e3).toFixed(0)}k`;
  return String(nodes);
}

function downloadProgressEl(): HTMLElement | null {
  return document.getElementById('downloadProgress');
}

function showDownloadProgress(label: string, loadedBytes?: number, totalBytes?: number, phase = 'Loading'): void {
  const node = downloadProgressEl();
  if (!node) return;
  renderLoadingProgress(node, { label, loadedBytes, totalBytes, phase });
}

function hideDownloadProgress(): void {
  const node = downloadProgressEl();
  if (!node) return;
  hideLoadingProgress(node);
}

function hasBoundedArenaSearch(progress: ArenaSearchProgress): boolean {
  return (
    !progress.indeterminate &&
    progress.completed !== undefined &&
    progress.requested !== undefined &&
    progress.requested > 0 &&
    progress.requested < Number.MAX_SAFE_INTEGER
  );
}

/** Live activity never borrows evaluation/PV/metrics from the previous search. */
function arenaSearchProgressHtml(progress: ArenaSearchProgress): string {
  return hasBoundedArenaSearch(progress)
    ? `<progress class="eng-search-meter" aria-label="Current search progress" value="${Math.max(0, Math.floor(progress.completed!))}" max="${Math.max(1, Math.floor(progress.requested!))}"></progress>`
    : '';
}

function setArenaSearchProgress(engineId: string, label: string, progress: ArenaSearchProgress): void {
  arenaSearchProgress.set(engineId, { ...progress, label });
  renderEngineOutputs();
}

function setArenaVisitProgress(
  engineId: string,
  label: string,
  progress: { completedVisits?: number; requestedVisits?: number; visits: number; move?: string | null; value: number; elapsedMs?: number },
): void {
  const completed = progress.completedVisits ?? progress.visits;
  const elapsedMs = progress.elapsedMs ?? 0;
  setArenaSearchProgress(engineId, label, {
    label,
    completed,
    requested: progress.requestedVisits ?? progress.visits,
    elapsedMs,
    nps: elapsedMs > 0 ? completed / Math.max(1e-9, elapsedMs / 1000) : undefined,
    best: progress.move ?? null,
    value: progress.value,
    units: 'visits',
  });
}

function clearArenaSearchProgress(engineId: string): void {
  arenaSearchProgress.delete(engineId);
  renderEngineOutputs();
}

function centipawnRuntimeForVariant(variant: string): 'auto' | 'ort' | 'custom-webgpu' | 'tvmjs-webgpu' {
  if (variant.endsWith('-ort')) return 'ort';
  if (variant.endsWith('-custom')) return LEGACY_CENTIPAWN_OVERRIDE ? 'custom-webgpu' : 'tvmjs-webgpu';
  return 'auto';
}

function centipawnRuntimeFallbackForVariant(variant: string): boolean {
  return !variant.endsWith('-custom');
}

function centipawnHybridManifestStatusText(): string {
  if (centipawnHybridManifestStatus === 'present') return `Centipawn TVMJS bundle present (${CENTIPAWN_HYBRID_MANIFEST_URL})`;
  if (centipawnHybridManifestStatus === 'missing') return `Centipawn TVMJS bundle missing; auto uses ORT fallback (${CENTIPAWN_HYBRID_MANIFEST_URL})`;
  return `Centipawn TVMJS bundle checking (${CENTIPAWN_HYBRID_MANIFEST_URL})`;
}

async function refreshCentipawnHybridManifestStatus(mountSignal = mountAbort.signal): Promise<void> {
  let status: typeof centipawnHybridManifestStatus;
  try {
    const res = await fetch(resolvePublicAssetUrl(CENTIPAWN_HYBRID_MANIFEST_URL), { method: 'HEAD', cache: 'no-store' });
    status = res.ok ? 'present' : 'missing';
  } catch {
    status = 'missing';
  }
  if (isStaleMount(mountSignal)) return;
  centipawnHybridManifestStatus = status;
  if (centipawnHybridManifestStatus === 'missing') {
    for (const row of activeSeatRows()) {
      if (row.family === 'centipawn' && row.variant === 'bt4-custom') row.variant = 'bt4-auto';
    }
  }
  void renderRuntimeBadge();
}

function centipawnModelForVariant(variant: string): { modelId: string; onnx: string; meta: string } {
  return LEGACY_CENTIPAWN_OVERRIDE
    ? { modelId: 'bt4-anneal-muon-best', onnx: LEGACY_CENTIPAWN_RESOLVED_MODEL_URL, meta: LEGACY_CENTIPAWN_RESOLVED_META_URL }
    : { modelId: 'bt4-soap-rem-c19000-final', onnx: CENTIPAWN_MODEL_URL, meta: CENTIPAWN_META_URL };
}

function centipawnEvaluatorCacheKey(variant: string): string {
  const runtime = centipawnRuntimeForVariant(variant);
  const fallback = centipawnRuntimeFallbackForVariant(variant);
  const model = centipawnModelForVariant(variant);
  return `${runtime}:${fallback ? 'fallback' : 'strict'}:${model.onnx}:${model.meta}:${CENTIPAWN_HYBRID_MANIFEST_URL}`;
}

function activeCentipawnEvaluatorKeys(): Set<string> {
  return new Set(
    activeSeatRows()
      .filter((row) => row.family === 'centipawn')
      .map((row) => centipawnEvaluatorCacheKey(row.variant)),
  );
}

async function centipawnEvaluator(variant: string): Promise<Evaluator> {
  const runtime = centipawnRuntimeForVariant(variant);
  const fallback = centipawnRuntimeFallbackForVariant(variant);
  const model = centipawnModelForVariant(variant);
  const key = centipawnEvaluatorCacheKey(variant);
  const existing = centipawnEvaluatorPromises.get(key);
  if (existing) return existing;
  const loadToken = Symbol(key);
  const generation = centipawnEvaluatorGeneration;
  const created = (async () => {
    const loaded = await createBrowserSquareformerRuntimeEvaluator(
      {
        id: model.modelId,
        modelId: model.modelId,
        label: centipawnEngineLabel(variant),
        onnx: model.onnx,
        meta: model.meta,
        runtime,
        manifestUrl: CENTIPAWN_HYBRID_MANIFEST_URL,
      },
      {
        params,
        runtime,
        manifestUrl: CENTIPAWN_HYBRID_MANIFEST_URL,
        fallback,
        audit: { surface: 'arena', searchBudget: 'visits=seat strength' },
      },
    );
    console.info('[lc0-arena] loaded Centipawn evaluator', {
      requestedRuntime: loaded.requestedRuntime,
      resolvedRuntime: loaded.resolvedRuntime,
      runtimeConfigId: loaded.runtimeConfigId,
      manifestUrl: loaded.manifestUrl,
      fallbackReason: loaded.fallbackReason,
    });
    const cached = new CachedEvaluator(loaded.evaluator, {
      maxEntries: arenaCacheEntries(),
      includeHistory: true,
      includeLegalMoves: true,
      label: `centipawn-arena:${runtime}`,
    });
    if (centipawnEvaluatorGeneration !== generation || centipawnEvaluatorLoadTokens.get(key) !== loadToken) {
      destroyCentipawnEvaluator(cached);
      const error = new Error('Centipawn evaluator was disposed before it finished loading');
      error.name = 'AbortError';
      throw error;
    }
    centipawnEvaluators.add(cached);
    centipawnEvaluatorsByKey.set(key, cached);
    return cached;
  })();
  centipawnEvaluatorPromises.set(key, created);
  centipawnEvaluatorLoadTokens.set(key, loadToken);
  try {
    return await created;
  } catch (error) {
    if (centipawnEvaluatorPromises.get(key) === created) {
      centipawnEvaluatorPromises.delete(key);
      if (centipawnEvaluatorLoadTokens.get(key) === loadToken) centipawnEvaluatorLoadTokens.delete(key);
    }
    throw error;
  }
}

function destroyCentipawnEvaluator(evaluator: CachedEvaluator): void {
  evaluator.clear();
  const destroy = (evaluator.inner as Evaluator & { destroy?: () => void }).destroy;
  if (typeof destroy === 'function') destroy.call(evaluator.inner);
}

function disposeUnusedCentipawnEvaluators(): void {
  const activeKeys = activeCentipawnEvaluatorKeys();
  for (const key of [...centipawnEvaluatorPromises.keys()]) {
    if (activeKeys.has(key)) continue;
    centipawnEvaluatorPromises.delete(key);
    centipawnEvaluatorLoadTokens.delete(key);
  }
  for (const [key, evaluator] of [...centipawnEvaluatorsByKey]) {
    if (activeKeys.has(key)) continue;
    destroyCentipawnEvaluator(evaluator);
    centipawnEvaluators.delete(evaluator);
    centipawnEvaluatorsByKey.delete(key);
    centipawnEvaluatorPromises.delete(key);
    centipawnEvaluatorLoadTokens.delete(key);
  }
}

function centipawnHistoryFens(positions: BoardState[]): string[] {
  return positions.slice(0, -1).map(boardToFen).reverse().slice(0, 16);
}

function initialLc0Runtime(): Lc0ArenaRuntime {
  return initialBrowserLc0Runtime(params);
}

function selectedLc0Runtime(): Lc0ArenaRuntime {
  return normalizeLc0Runtime(selectEl('lc0RuntimeSelect').value);
}

function lc0WholeModelRuntimeRequested(): boolean {
  return browserLc0WholeModelRuntimeRequested(params);
}

function installExperimentalLc0RuntimeOption(_force = false): void {
  installBrowserLc0RuntimeOption(selectEl('lc0RuntimeSelect'));
}

function normalizeLc0Preset(value: string | null): Lc0ArenaPreset {
  if (isV0DeployProfile() && value === 'benchmarked-small') return 'stable';
  return value === 'benchmarked-small' || value === 'custom' ? value : 'stable';
}

function installLc0PresetOptions(): void {
  if (!isV0DeployProfile()) return;
  selectEl('lc0PresetSelect').querySelector('option[value="benchmarked-small"]')?.remove();
}

function inferredLc0Preset(): Lc0ArenaPreset {
  if (isV0DeployProfile()) return 'stable';
  const explicit = params.get('lc0Preset') ?? params.get('preset');
  if (explicit) return normalizeLc0Preset(explicit);
  const requestedRuntime = normalizeLc0Runtime(params.get('lc0Runtime') ?? params.get('runtime'));
  if (requestedRuntime === LC0_WHOLE_MODEL_WEBGPU_RUNTIME || lc0WholeModelRuntimeRequested()) return 'benchmarked-small';
  if (
    requestedRuntime !== 'onnx' ||
    params.has('inputBackend') ||
    params.has('encoderKernel') ||
    params.has('legalPriorsBackend') ||
    params.has('lc0BatchSize') ||
    params.has('batchPipelineDepth')
  )
    return 'custom';
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
    note.textContent = isV0DeployProfile()
      ? 'Stable deployment: LC0 Small via ORT with WebGPU/WASM fallback.'
      : 'Stable default: LC0 Small via ORT with WebGPU/WASM fallback. Use the fast preset to test benchmarked eval speed in fixed-time arena games.';
  }
}

function applyLc0Preset(preset: Lc0ArenaPreset, options: { reload?: boolean } = {}): void {
  preset = normalizeLc0Preset(preset);
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
  if (arenaAuditHandler) window.removeEventListener(BROWSER_RUNTIME_AUDIT_EVENT, arenaAuditHandler);
  arenaAuditHandler = (event: Event) => {
    const detail = (event as CustomEvent<BrowserRuntimeAuditDetail>).detail;
    if (detail.surface && detail.surface !== 'arena') return;
    const warningTarget = document.getElementById('runtimeWarning');
    if (warningTarget && activeRuntimeAuditIdentities().has(browserRuntimeAuditIdentity(detail))) updateRuntimeFallbackWarning(warningTarget, detail);
    if (detail.family !== 'lc0') return;
    const target = document.getElementById('runtimeAuditInfo');
    if (!target) return;
    target.innerHTML = diagBlockHtml('LC0 audit', [htmlEscape(formatBrowserRuntimeAudit(detail))]);
  };
  window.addEventListener(BROWSER_RUNTIME_AUDIT_EVENT, arenaAuditHandler);
}

function lc0RuntimeLabel(runtime = selectedLc0Runtime()): string {
  return browserLc0RuntimeLabel(runtime);
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
  return normalizeLc0LegalPriorsBackend(
    node?.value ?? params.get('legalPriorsBackend') ?? params.get('lc0LegalPriorsBackend') ?? params.get('hybridLegalPriors'),
  );
}

function lc0EncoderLayers(): number {
  return browserLc0EncoderLayers(params);
}

function normalizeLc0EncoderKernelVariant(value: string | null): Lc0WebEncoderKernelVariant {
  return value === 'tvm-packed-f16' || value === 'mixed-tvm-ffn' || value === 'mixed-tvm-ffn-outproj' || value === 'mixed-tvm-ffn-smolgen-project'
    ? value
    : 'hand';
}

function lc0EncoderKernelVariant(): Lc0WebEncoderKernelVariant {
  const node = document.getElementById('lc0EncoderKernelSelect') as HTMLSelectElement | null;
  return normalizeLc0EncoderKernelVariant(node?.value ?? params.get('encoderKernel') ?? params.get('lc0EncoderKernel') ?? params.get('encoderKernelVariant'));
}

function lc0HybridConfigLabel(runtime = selectedLc0Runtime()): string {
  if (runtime === 'onnx') return '';
  if (runtime === LC0_WHOLE_MODEL_WEBGPU_RUNTIME)
    return `manifest ${LC0_WHOLE_MODEL_MANIFEST_URL} · physical batch ${lc0WholeModelPhysicalBatch()}${lc0WholeModelTensorCache() ? ' · tensor cache' : ''}`;
  const legal = lc0HybridLegalPriorsBackend();
  const effectiveLegal = legal === 'gpu' && runtime !== 'hybrid-wgsl-heads' ? 'js' : legal;
  return `input ${lc0HybridInputBackend()} · encoder ${lc0EncoderKernelVariant()} · legal ${effectiveLegal}`;
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
  return browserLc0WholeModelPhysicalBatch(params);
}

function lc0WholeModelTensorCache(): boolean {
  return browserLc0WholeModelTensorCache(params);
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
  const family = canonicalEngineFamily(familyRaw);
  if (!family) return undefined;
  const variant = variantRaw || defaultVariant(family);
  const strength = strengthRaw === undefined || strengthRaw === '' ? strengthMeta(family).def : Number(strengthRaw);
  const row: EngineRow = { family, variant, strength };
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
  const opponentFamilyRaw = params.get('opponentFamily');
  const opponentFamily = opponentFamilyRaw ? canonicalEngineFamily(opponentFamilyRaw) : null;
  if (opponentFamily) {
    seatRows[1].family = opponentFamily;
    seatRows[1].variant = params.get('opponentVariant') ?? defaultVariant(opponentFamily);
    seatRows[1].strength = Number(params.get('opponentStrength') ?? strengthMeta(opponentFamily).def);
    clampStrength(seatRows[1]);
  }
  normalizeSeatRowsForDeploy();

  inputEl('gamesInput').value = String(intParam('gamesPerOpening', intParam('games', Number(inputEl('gamesInput').value) || 2, 1, 20), 1, 20));
  inputEl('delayInput').value = String(intParam('delayMs', intParam('delay', Number(inputEl('delayInput').value) || 0, 0, 3000), 0, 3000));
  const concurrency = params.get('arenaConcurrency');
  if (concurrency && ['auto', '1', '2', '3'].includes(concurrency)) selectEl('arenaConcurrencySelect').value = concurrency;
  const budget = params.get('budgetMode') ?? params.get('budget');
  if (budget === 'movetime' || budget === 'fixed') selectEl('budgetModeSelect').value = budget;
  inputEl('movetimeInput').value = String(intParam('movetimeMs', intParam('movetime', Number(inputEl('movetimeInput').value) || 500, 10, 60000), 10, 60000));
  inputEl('cacheEntriesInput').value = String(
    intParam('cacheEntries', intParam('cache', Number(inputEl('cacheEntriesInput').value) || 2048, 0, 100000), 0, 100000),
  );
  inputEl('stockfishThreadsInput').value = String(intParam('sfThreads', Number(inputEl('stockfishThreadsInput').value) || 1, 1, 32));
  inputEl('lc0BatchSizeInput').value = String(
    boundedIntValue(params.get('lc0BatchSize') ?? params.get('batchSize') ?? params.get('batch'), Number(inputEl('lc0BatchSizeInput').value) || 1, 1, 64),
  );
  inputEl('lc0BatchPipelineDepthInput').value = String(
    boundedIntValue(params.get('lc0BatchPipelineDepth') ?? params.get('batchPipelineDepth'), Number(inputEl('lc0BatchPipelineDepthInput').value) || 1, 1, 16),
  );
  selectEl('lc0InputBackendSelect').value = normalizeLc0InputBackend(params.get('inputBackend') ?? params.get('lc0InputBackend'));
  selectEl('lc0EncoderKernelSelect').value = normalizeLc0EncoderKernelVariant(
    params.get('encoderKernel') ?? params.get('lc0EncoderKernel') ?? params.get('encoderKernelVariant'),
  );
  selectEl('lc0LegalPriorsSelect').value = normalizeLc0LegalPriorsBackend(
    params.get('legalPriorsBackend') ?? params.get('lc0LegalPriorsBackend') ?? params.get('hybridLegalPriors'),
  );

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
  if (info.scoreCp !== undefined) return `${signed((w * info.scoreCp) / 100, 2)} · d${info.depth}`;
  return `score unavailable · d${info.depth}`;
}

function stockfishScoreCompact(info: StockfishInfoLine | undefined, fen: string): string {
  if (!info) return 'eval —';
  const w = fenTurn(fen) === 'w' ? 1 : -1;
  if (info.mateIn !== undefined) return `M${signed(w * info.mateIn, 0)} · d${info.depth}`;
  if (info.scoreCp !== undefined) return `${signed((w * info.scoreCp) / 100, 2)} · d${info.depth}`;
  return `d${info.depth}`;
}

function stockfishWhiteCp(info: StockfishInfoLine | undefined, fen: string): number | undefined {
  return info?.scoreCp === undefined ? undefined : fenTurn(fen) === 'w' ? info.scoreCp : -info.scoreCp;
}

function stockfishMateInWhitePov(info: StockfishInfoLine | undefined, fen: string): number | undefined {
  return info?.mateIn === undefined ? undefined : fenTurn(fen) === 'w' ? info.mateIn : -info.mateIn;
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
  const whiteScore = stmScoreToWhiteScore(fen, clamp01((q + 1) / 2));
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
  const whiteScore = evalBarWhitePercent(whiteCp, undefined) / 100;
  return { whiteScore: clamp01(whiteScore), label: signed(whiteCp / 100, 1) };
}

function searchWdlText(wdl: [number, number, number], q: number, fen: string): string {
  const w = toWhiteWdl(wdl, fen);
  return `WDL ${percent(w[0])}/${percent(w[1])}/${percent(w[2])} · search Q ${signed(toWhiteQ(q, fen), 3)}`;
}

/** White-view pawn eval for an Lc0-family Q, on lc0's standard Q->centipawn curve (as Analysis shows it). */
function lc0EvalFields(fen: string, q: number, wdl?: [number, number, number]): Pick<EngineOutputSnapshot, 'evalText' | 'qWhite' | 'wdlWhite'> {
  const qWhite = toWhiteQ(q, fen);
  return { evalText: formatScore(qToCentipawns(qWhite), undefined), qWhite, wdlWhite: wdl ? toWhiteWdl(wdl, fen) : undefined };
}

function recordEngineOutput(snapshot: EngineOutputSnapshot): void {
  thinkingEngineIds.delete(snapshot.engineId);
  thinkingSince.delete(snapshot.engineId);
  arenaSearchProgress.delete(snapshot.engineId);
  engineOutputs.set(snapshot.engineId, snapshot);
  lastOutputEngineId = snapshot.engineId;
  engineOutputTotalCount += 1;
  engineOutputHistory.push(snapshot);
  if (engineOutputHistory.length > MAX_ENGINE_OUTPUT_HISTORY) engineOutputHistory.splice(0, engineOutputHistory.length - MAX_ENGINE_OUTPUT_HISTORY);
  renderSideLabels();
  renderEngineOutputs();
  renderEngineDiagnosticsInfo();
}

function displayEngineName(name: string): string {
  return analysisEngineName(name);
}

/** Engine identity colour + monogram, shared with Analysis (Lc0 violet, Stockfish blue, …). */
function engineAvatarHtml(name: string, size: 'sm' | 'md' = 'md'): string {
  return `<span class="arena-avatar ${size}" style="--eng:${engineBrushes(name).swatch}" aria-hidden="true">${htmlEscape(engineMonogram(name))}</span>`;
}

function formatSeconds(ms: number): string {
  return `${(ms / 1000).toFixed(ms < 10_000 ? 2 : 1)} s`;
}

let lastOutputEngineId: string | null = null;

/** One shared bar: the eval of the engine that searched most recently (White's view), its value printed at the bar's end. */
function renderEvalBars(): void {
  const node = el('arenaEvalBar');
  const ids = [boardWhiteId, boardBlackId].filter((id): id is string => !!id);
  const sourceId = lastOutputEngineId && ids.includes(lastOutputEngineId) && engineOutputs.has(lastOutputEngineId) ? lastOutputEngineId : ids.find((id) => engineOutputs.has(id));
  const output = sourceId ? engineOutputs.get(sourceId) : undefined;
  const bar = output?.evalBar;
  const whiteScore = clamp01(bar?.whiteScore ?? 0.5);
  node.classList.toggle('empty', !bar);
  node.style.setProperty('--eng', output ? engineBrushes(output.engineName).swatch : 'var(--border-input)');
  const text = output?.evalText ?? bar?.label;
  const label = output && text ? `Evaluation ${text} from White's view, by ${displayEngineName(output.engineName)}` : 'Evaluation: none yet';
  node.title = label;
  node.setAttribute('aria-label', label);
  node.innerHTML = bar
    ? `<div class="eval-fill" style="height:${(100 * whiteScore).toFixed(1)}%"></div><div class="eval-midline"></div><span class="eval-bar-value ${whiteScore >= 0.5 ? 'at-white' : 'at-black'}">${htmlEscape(text ?? '—')}</span>`
    : '<span class="eval-none">—</span>';
  // The bar prints its own value; the caption repeats it only on phones, where the bar is too narrow to.
  el('evalSource').innerHTML = output && bar
    ? `<span class="eval-source-value"><strong>${htmlEscape(text ?? '—')}</strong> · </span>${htmlEscape(displayEngineName(output.engineName))} · White’s view · last completed search`
    : 'No evaluation yet';
}

/** Numbered SAN for a PV searched from `fen` ("16… Bb3 17. a5 Rfd1"), first move emphasised. */
function pvSanHtml(fen: string, pv: string[], maxMoves = 10): string {
  const start = parseFen(fen);
  const sans = uciLineToSan(start, pv, maxMoves).split(/\s+/).filter(Boolean);
  const startPly = (start.fullmove - 1) * 2 + (start.turn === 'b' ? 1 : 0);
  const moves = sans.map((san, index) => {
    const ply = startPly + index;
    const number = Math.floor(ply / 2) + 1;
    const label = ply % 2 === 0 ? `${number}.` : index === 0 ? `${number}…` : '';
    return `<span class="pv-move${index === 0 ? ' first' : ''}">${label ? `<span class="pv-num">${label}</span>` : ''}${htmlEscape(san)}</span>`;
  });
  return moves.join(' ') + (pv.length > maxMoves ? ' <span class="pv-num">…</span>' : '');
}

function boardSideFor(id: string): string {
  if (boardWhiteId === id && boardBlackId === id) return 'Both sides';
  if (boardWhiteId === id) return 'White';
  if (boardBlackId === id) return 'Black';
  return '';
}

function engineCardHtml(id: string): string {
  const snapshot = engineOutputs.get(id);
  const rawName = snapshot?.engineName ?? isolatedCpuSeats.get(id)?.seat.name ?? engines.get(id)?.name ?? id;
  const name = displayEngineName(rawName);
  const thinking = running && thinkingEngineIds.has(id);
  const side = boardSideFor(id);
  const progress = arenaSearchProgress.get(id) ?? { label: rawName, units: 'search' as const, indeterminate: true };
  const live = thinking ? arenaSearchProgressHtml(progress) : '';
  const searched = snapshot ? parseFen(snapshot.fen) : null;
  const resultLabel = searched ? `Previous move ${searched.fullmove}${searched.turn === 'b' ? '…' : '.'}` : 'No completed search';
  const activity = thinking ? `<span class="eng-thinking-label" title="Searching move ${board.fullmove}; results below are from the previous completed search">Thinking</span>` : '';
  const head = `<header class="eng-head">${engineAvatarHtml(rawName)}<span class="eng-title"><span class="eng-name">${htmlEscape(name)}${activity}</span><span class="eng-side">${side ? `${side} · ` : ''}${resultLabel}</span></span><span class="eng-eval" title="Previous move’s evaluation from White's view">${htmlEscape(snapshot?.evalText ?? '—')}</span></header>`;
  if (!snapshot) {
    return `<article class="eng-card${thinking ? ' thinking' : ''}" data-engine-id="${htmlEscape(id)}" style="--eng:${engineBrushes(rawName).swatch}">${head}${live}${thinking ? '' : '<p class="eng-empty">Waiting for its first move.</p>'}</article>`;
  }
  const wdl = snapshot.wdlWhite
    ? `<div class="eng-wdl"><span class="wdl-bar" aria-hidden="true"><span class="w" style="width:${(100 * snapshot.wdlWhite[0]).toFixed(1)}%"></span><span class="d" style="width:${(100 * snapshot.wdlWhite[1]).toFixed(1)}%"></span><span class="l" style="width:${(100 * snapshot.wdlWhite[2]).toFixed(1)}%"></span></span><span class="wdl-text">W ${percent0(snapshot.wdlWhite[0])} · D ${percent0(snapshot.wdlWhite[1])} · L ${percent0(snapshot.wdlWhite[2])}</span></div>`
    : '';
  const stats: [string, string][] = [];
  if (snapshot.kind === 'lc0') {
    if (snapshot.visits !== undefined) stats.push(['Visits', formatCount(snapshot.visits)]);
    if (snapshot.nps != null) stats.push(['Visits/s', formatNps(snapshot.nps)]);
  } else {
    if (snapshot.depth != null) stats.push(['Depth', String(snapshot.depth)]);
    if (snapshot.nps != null) stats.push(['Nodes/s', formatNps(snapshot.nps)]);
    if (snapshot.nodes != null) stats.push(['Nodes', formatNodes(snapshot.nodes)]);
  }
  if (snapshot.elapsedMs != null) stats.push(['Move time', formatSeconds(snapshot.elapsedMs)]);
  const statsHtml = stats.length
    ? `<dl class="eng-stats">${stats.map(([label, value]) => `<div${label === 'Move time' ? ' class="move-time"' : ''}><dt>${label}</dt><dd>${htmlEscape(value)}</dd></div>`).join('')}</dl>`
    : '';
  const line = snapshot.pv?.length
    ? pvSanHtml(snapshot.fen, snapshot.pv)
    : snapshot.move
      ? pvSanHtml(snapshot.fen, [snapshot.move])
      : '';
  const lineHtml = line ? `<p class="eng-line" tabindex="0" data-fen="${htmlEscape(snapshot.fen)}" aria-label="Previous move’s principal variation"><span class="eng-line-label">PV</span> ${line}</p>` : '';
  return `<article class="eng-card${thinking ? ' thinking' : ''}" data-engine-id="${htmlEscape(id)}" style="--eng:${engineBrushes(rawName).swatch}">${head}${live}${wdl}${statsHtml}${lineHtml}</article>`;
}

/** Raw engine output (UCI PV, policy, cache counters) for the "Search details" disclosure. */
function engineRawHtml(id: string): string {
  const snapshot = engineOutputs.get(id);
  if (!snapshot) return '';
  const parts = [snapshot.summary, snapshot.detail, snapshot.pv?.length ? `PV (UCI) ${snapshot.pv.slice(0, 12).join(' ')}` : ''].filter(Boolean);
  return `<div class="eng-raw-block"><strong>${htmlEscape(displayEngineName(snapshot.engineName))}</strong>${parts.map((part) => `<div>${htmlEscape(part!)}</div>`).join('')}</div>`;
}

function renderEngineOutputs(): void {
  const ids = [...new Set(activeEngineIds)];
  const cards = ids.filter((id) => running || engineOutputs.has(id)).map(engineCardHtml);
  const grid = el('engineEvalInfo');
  // Progress refreshes must not interrupt a spectator reading a long retained PV.
  const pvState = new Map([...grid.querySelectorAll<HTMLElement>('.eng-line')].map((line) => [
    line.closest<HTMLElement>('.eng-card')!.dataset.engineId,
    { fen: line.dataset.fen, scrollLeft: line.scrollLeft, focused: document.activeElement === line },
  ]));
  grid.innerHTML = cards.join('');
  for (const line of grid.querySelectorAll<HTMLElement>('.eng-line')) {
    const previous = pvState.get(line.closest<HTMLElement>('.eng-card')!.dataset.engineId);
    if (previous?.focused) line.focus({ preventScroll: true });
    // A newly completed result starts at its first move; ongoing searches retain the reader's place.
    if (previous && previous.fen === line.dataset.fen) line.scrollLeft = previous.scrollLeft;
  }
  const raw = document.getElementById('engineRaw');
  if (raw) raw.innerHTML = ids.map(engineRawHtml).join('');
  const section = document.getElementById('engineOutputsSection');
  if (section) section.hidden = !cards.length;
}

interface StripSide {
  id: string | null;
  name: string | null;
  seat: string | null;
}

/** Before any game, the strips preview the selected seats (Engine 1 plays White in game 1). */
function stripSides(): { w: StripSide; b: StripSide } {
  if (reviewing && reviewTrail?.sides) return {
    w: { id: null, name: reviewTrail.sides.whiteName, seat: null }, b: { id: null, name: reviewTrail.sides.blackName, seat: null },
  };
  if (running || liveTrail || boardWhiteName || boardBlackName) {
    return { w: { id: boardWhiteId, name: boardWhiteName, seat: null }, b: { id: boardBlackId, name: boardBlackName, seat: null } };
  }
  const seatName = (index: number): string | null => {
    const row = seatRows[index];
    if (!row) return null;
    try {
      return engines.get(engineIdForRow(row))?.name ?? rowLabel(row);
    } catch {
      return null;
    }
  };
  return { w: { id: null, name: seatName(0), seat: 'Engine 1' }, b: { id: null, name: seatName(1), seat: 'Engine 2' } };
}

function renderStrips(): void {
  const sides = stripSides();
  const turn = board.turn;
  const now = performance.now();
  const update = (nodeId: string, color: 'w' | 'b', side: StripSide) => {
    const node = el(nodeId);
    const colorName = color === 'w' ? 'White' : 'Black';
    const thinking = running && !reviewing && turn === color && !!side.id && thinkingEngineIds.has(side.id);
    const since = side.id ? thinkingSince.get(side.id) : undefined;
    const moveMs = thinking && since !== undefined ? Math.max(0, now - since) : 0;
    const usedMs = sideTimeMs[color] + moveMs;
    const meta = thinking
      ? `<span class="strip-thinking">${colorName} · thinking · move ${board.fullmove}</span>`
      : `${colorName}${side.seat ? ` · ${side.seat}` : ''}`;
    const name = side.name ? displayEngineName(side.name) : 'No engine';
    const showClock = !reviewing && (running || usedMs > 0);
    node.dataset.turn = String(thinking);
    node.innerHTML =
      `${side.name ? engineAvatarHtml(side.name) : ''}` +
      `<span class="strip-id"><strong>${htmlEscape(name)}</strong><small><span class="side-dot ${color}" aria-hidden="true"></span>${meta}</small></span>` +
      (showClock ? `<span class="strip-metric strip-clock" title="Cumulative search time used this game, not a countdown"><small>Total used</small>${formatSeconds(usedMs)}</span>` : '');
  };
  update('blackSideLabel', 'b', sides.b);
  update('whiteSideLabel', 'w', sides.w);
}

function renderSideLabels() {
  renderStrips();
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
    turnColor: shownBoard.turn === 'w' ? ('white' as const) : ('black' as const),
    check: boardCheck(shownBoard),
    coordinates: true,
    viewOnly: true,
    highlight: { lastMove: true, check: true },
    animation: { enabled: true, duration: 140 },
    lastMove: shownUci ? [shownUci.slice(0, 2) as Key, shownUci.slice(2, 4) as Key] : undefined,
    // Last moves use square highlights only. Arrows are reserved for root candidates.
    drawable: {
      enabled: false,
      brushes: {
        candidate: { key: 'candidate', color: '#5a6e2a', opacity: 0.95, lineWidth: 15 },
        candidateDim: { key: 'candidateDim', color: '#8a8474', opacity: 0.5, lineWidth: 15 },
      },
    },
  };
  // Cast: chessground's DrawBrushes type has fixed keys, but custom brush keys
  // merge fine at runtime.
  const cfg = config as unknown as NonNullable<Parameters<typeof Chessground>[1]>;
  if (!ground) ground = Chessground(el('ground'), cfg);
  else ground.set(cfg);
  const shapes: DrawShape[] = reviewing && reviewShapes ? reviewShapes : [];
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
  return { label, entries, openingPlies, charts: [] };
}

function enterReview(trail: GameTrail, index: number, shapes: DrawShape[] | null = null): void {
  reviewTrail = trail;
  reviewIndex = Math.max(0, Math.min(index, trail.entries.length - 1));
  reviewing = true;
  reviewShapes = shapes;
  gameChartSamples = trail.charts;
  el('chartsPanel').hidden = false;
  renderGameCharts();
  renderSavedRootChart(trail.rootChart);
  renderBoard();
}

/** Step through the shown game; from the live game this enters review, and stepping back onto the live end returns to live. */
function stepReview(delta: number | 'start' | 'end'): void {
  const trail = reviewing && reviewTrail ? reviewTrail : liveTrail;
  if (!trail) return;
  const last = trail.entries.length - 1;
  const current = reviewing && reviewTrail === trail ? reviewIndex : last;
  const next = delta === 'start' ? 0 : delta === 'end' ? last : Math.max(0, Math.min(current + delta, last));
  if (!reviewing && next === last) return;
  if (reviewing && trail === liveTrail && running && next === last) {
    exitReview();
    return;
  }
  if (!reviewing) {
    enterReview(trail, next);
    return;
  }
  reviewIndex = next;
  reviewShapes = null;
  renderBoard();
}

function exitReview(): void {
  reviewing = false;
  reviewTrail = null;
  reviewShapes = null;
  gameChartSamples = liveTrail?.charts ?? [];
  renderGameCharts();
  renderSavedRootChart(liveTrail?.rootChart);
  renderBoard();
}

function renderReviewBar(): void {
  const bar = el('reviewBar');
  bar.hidden = !reviewing || !reviewTrail;
  if (!bar.hidden && reviewTrail) el('reviewLabel').textContent = `Reviewing ${reviewTrail.label} · ply ${reviewIndex} of ${reviewTrail.entries.length - 1}`;
  const chartPly = inputEl('chartPly');
  const shownTrail = reviewing && reviewTrail ? reviewTrail : liveTrail;
  chartPly.max = String(Math.max(0, (shownTrail?.entries.length ?? 1) - 1));
  chartPly.value = String(reviewing ? reviewIndex : Number(chartPly.max));
  chartPly.disabled = !shownTrail || shownTrail.entries.length <= 1;
  chartPly.setAttribute('aria-valuetext', chartPly.value === '0' ? 'Opening position' : `Ply ${chartPly.value}`);
  const trail = reviewing && reviewTrail ? reviewTrail : liveTrail;
  const last = trail ? trail.entries.length - 1 : 0;
  const index = reviewing && reviewTrail ? reviewIndex : last;
  (el('revPrev') as HTMLButtonElement).disabled = !trail || index <= 0;
  (el('revStart') as HTMLButtonElement).disabled = !trail || index <= 0;
  (el('revNext') as HTMLButtonElement).disabled = !trail || index >= last;
  (el('revEnd') as HTMLButtonElement).disabled = !trail || index >= last;
  const liveButton = el('revLive') as HTMLButtonElement;
  liveButton.hidden = !reviewing;
  liveButton.textContent = running ? 'Back to live' : 'Exit review';
}

/** Following the live game, the list stays scrolled to its newest row (even as panels above resize it). */
function keepMovesAtLiveEnd(): void {
  if (reviewing) return;
  const strip = document.getElementById('gameMoves');
  if (strip) strip.scrollTop = strip.scrollHeight;
}

/** Scoresheet rows (number | White | Black), as on Play; opening-book plies are muted. */
function renderMoveStrip(): void {
  const strip = el('gameMoves');
  const block = document.getElementById('movesBlock');
  const trail = reviewing && reviewTrail ? reviewTrail : liveTrail;
  if (!trail || trail.entries.length <= 1) {
    if (block) block.hidden = true;
    strip.innerHTML = '';
    return;
  }
  if (block) block.hidden = false;
  const shownIndex = reviewing && reviewTrail === trail ? reviewIndex : trail.entries.length - 1;
  const start = parseFen(trail.entries[0].fen);
  const startPly = (start.fullmove - 1) * 2 + (start.turn === 'b' ? 1 : 0);
  const cell = (index: number) => {
    const entry = trail.entries[index];
    if (!entry) return '<span class="san empty"></span>';
    const classes = ['san', index === shownIndex ? 'current' : '', index <= trail.openingPlies ? 'book' : ''].filter(Boolean).join(' ');
    return `<button type="button" class="${classes}" data-idx="${index}"${index <= trail.openingPlies ? ' title="Opening move"' : ''}>${htmlEscape(entry.san ?? entry.uci ?? '?')}</button>`;
  };
  const rows: string[] = [];
  let index = 1;
  while (index < trail.entries.length) {
    const ply = startPly + index - 1;
    const number = Math.floor(ply / 2) + 1;
    if (ply % 2 === 1) {
      rows.push(`<div class="move-row"><span class="num">${number}</span><span class="san empty">…</span>${cell(index)}</div>`);
      index += 1;
    } else {
      rows.push(`<div class="move-row"><span class="num">${number}</span>${cell(index)}${cell(index + 1)}</div>`);
      index += 2;
    }
  }
  strip.innerHTML = rows.join('');
  if (!reviewing) {
    keepMovesAtLiveEnd();
    return;
  }
  const current = strip.querySelector<HTMLElement>('.san.current');
  if (current) {
    // Scroll only the list (never the page) so the current move stays in view.
    const top = current.offsetTop;
    if (top < strip.scrollTop) strip.scrollTop = top;
    else if (top + current.offsetHeight > strip.scrollTop + strip.clientHeight) strip.scrollTop = top - strip.clientHeight + current.offsetHeight;
  }
}

/**
 * Click on the root-visits bar chart: jump to the position that search was
 * made from and draw the candidate moves — the clicked bar bold, the rest dim.
 */
function reviewRootChartClick(event: MouseEvent): void {
  const trail = reviewing && reviewTrail ? reviewTrail : liveTrail;
  if (!rootChartContext || !trail) return;
  const svg = el('rootChart').querySelector('svg');
  if (!svg) return;
  // Find the searched position in the current game's trail (latest match wins;
  // exact FEN strings, both produced by boardToFen of the same board).
  let index = -1;
  for (let i = trail.entries.length - 1; i >= 0; i--) {
    if (trail.entries[i].fen === rootChartContext.fen) {
      index = i;
      break;
    }
  }
  if (index < 0) return;
  // Mirror hBarChartSvg's layout: each bar row is 14 viewBox units high.
  const rect = svg.getBoundingClientRect();
  const viewH = rootChartContext.top.length * 14 + 2;
  const row = Math.floor((((event.clientY - rect.top) / rect.height) * viewH - 1) / 14);
  previewRootCandidate(Math.max(0, Math.min(row, rootChartContext.top.length - 1)), index);
}

/** Keyboard and pointer previews share the same candidate/position mapping. */
function previewRootCandidate(row: number, knownIndex?: number): void {
  const trail = reviewing && reviewTrail ? reviewTrail : liveTrail;
  if (!rootChartContext || !trail) return;
  const index = knownIndex ?? trail.entries.findLastIndex((entry) => entry.fen === rootChartContext!.fen);
  const clicked = rootChartContext.top[row];
  if (index < 0 || !clicked) return;
  selectEl('rootCandidate').value = String(row);
  const shapes: DrawShape[] = [{ orig: clicked.uci.slice(0, 2) as Key, dest: clicked.uci.slice(2, 4) as Key, brush: 'candidate' }];
  enterReview(trail, index, shapes);
  selectEl('rootCandidate').value = String(row);
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
  // Mirror lineChartSvg's plot padding at the responsive SVG viewBox width.
  const width = svg.viewBox.baseVal.width;
  const viewX = ((event.clientX - rect.left) / rect.width) * width;
  const xMin = Math.min(...plies);
  const xMax = Math.max(...plies);
  const frac = Math.max(0, Math.min(1, (viewX - 36) / (width - 8 - 36)));
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
  if (family === 'stormphrax') return REQUESTED_STORMPHRAX_VARIANT.key;
  return defaultStaticEngineVariant(family);
}

function bigNetSelectableSync(config: BigNetConfig): boolean {
  return bt4SupportedSync() && bigNetAssetStatusSync(config) === 'present';
}

function bigNetUnavailableText(config: BigNetConfig): string {
  if (!bt4SupportedSync()) return `Lc0 ${config.name} needs WebGPU support.`;
  if (bigNetAssetStatusSync(config) === 'missing')
    return `Lc0 ${config.name} model asset is missing at ${config.modelUrl}. Run node scripts/lc0_prepare_model_assets.mjs in this package.`;
  if (bigNetAssetStatusSync(config) === 'unknown') return `Lc0 ${config.name} model asset is still being checked.`;
  return '';
}

function variantOptions(family: EngineFamily): { value: string; label: string; disabled?: boolean }[] {
  if (isV0DeployProfile() && !['lc0', 'centipawn', 'sf', 'reckless', 'berserk', 'viridithas', 'plentychess', 'stormphrax'].includes(family)) return [];
  if (family === 'lc0')
    return lc0VariantOptions(bt4SupportedSync()).map((option) => {
      if (!isLc0BigNetVariant(option.value)) return option;
      const config = BIG_NETS[option.value];
      const asset = bigNetAssetStatusSync(config);
      if (bt4SupportedSync() && asset === 'unknown')
        void checkBigNetAsset(
          config,
          whileArenaMounted(() => {
            renderSeatSelectors();
            refreshSeatControls();
          }),
        );
      const suffix = !bt4SupportedSync() ? ' (WebGPU unavailable)' : asset === 'missing' ? ' (asset missing)' : asset === 'unknown' ? ' (checking asset)' : '';
      return { ...option, label: `${option.label}${suffix}`, disabled: option.disabled || asset !== 'present' };
    });
  if (family === 'centipawn')
    return centipawnVariantOptions().map((option) =>
      option.value === 'bt4-custom' && centipawnHybridManifestStatus === 'missing'
        ? { ...option, label: `${option.label} (bundle missing)`, disabled: true }
        : option,
    );
  if (family === 'sf') return stockfishVariantOptions();
  if (family === 'viridithas')
    return availableViridithasVariants().map((v) => {
      const status = viridithasVariantAssetStatus(v);
      const unsupported = v.key === 'relaxed-simd' && !supportsWasmRelaxedSimd();
      if (!unsupported && v.key === 'relaxed-simd' && assetProbePending(status)) void checkViridithasVariantAsset(v, whileArenaMounted(populateSeats));
      return browserVariantOption(v.key, engineVariantMenuLabel(family, v.key, v.label), {
        assetStatus: status,
        unsupportedReason: unsupported ? 'unsupported by this browser' : null,
        requirePresent: v.key === 'relaxed-simd',
      });
    });
  if (family === 'berserk')
    return availableBerserkVariants().map((v) => {
      const status = berserkVariantAssetStatus(v);
      const unsupported = v.key === 'emscripten-relaxed' && !supportsWasmRelaxedSimd();
      if (!unsupported && assetProbePending(status)) void checkBerserkVariantAsset(v, whileArenaMounted(populateSeats));
      const needsGeneratedAsset = v.key === 'emscripten-simd' || v.key === 'emscripten-relaxed';
      return browserVariantOption(v.key, engineVariantMenuLabel(family, v.key, v.label), {
        assetStatus: status,
        unsupportedReason: unsupported ? 'unsupported by this browser' : null,
        requirePresent: needsGeneratedAsset,
      });
    });
  if (family === 'plentychess')
    return availablePlentyChessVariants().map((v) => {
      const status = plentyChessVariantAssetStatus(v);
      const unsupportedReason = plentyChessVariantUnsupportedReason(v);
      const needsGeneratedAsset = v.key === 'emscripten-sse41' || v.key === 'emscripten-relaxed';
      if (!unsupportedReason && needsGeneratedAsset && assetProbePending(status)) void checkPlentyChessVariantAsset(v, whileArenaMounted(populateSeats));
      return browserVariantOption(v.key, engineVariantMenuLabel(family, v.key, v.label), {
        assetStatus: status,
        unsupportedReason,
        requirePresent: needsGeneratedAsset,
      });
    });
  if (family === 'stormphrax')
    return availableStormphraxVariants().map((v) => {
      const status = stormphraxVariantAssetStatus(v);
      const unsupportedReason = stormphraxVariantUnsupportedReason(v);
      const needsGeneratedAsset = v.key === 'emscripten-relaxed';
      if (!unsupportedReason && needsGeneratedAsset && assetProbePending(status)) void checkStormphraxVariantAsset(v, whileArenaMounted(populateSeats));
      return browserVariantOption(v.key, engineVariantMenuLabel(family, v.key, v.label), {
        assetStatus: status,
        unsupportedReason,
        requirePresent: needsGeneratedAsset,
      });
    });
  const recklessVariants = availableRecklessVariants().filter((v) => !isV0DeployProfile() || ['full', 'simd', 'relaxed-simd'].includes(v.key));
  return recklessVariants.map((v) => {
    const status = recklessVariantAssetStatus(v);
    const unsupported = v.key === 'relaxed-simd' && !supportsWasmRelaxedSimd();
    return browserVariantOption(v.key, engineVariantMenuLabel(family, v.key, v.label), {
      assetStatus: status,
      unsupportedReason: unsupported ? 'unsupported by this browser' : null,
    });
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
  if (row.family === 'centipawn') return centipawnEngineLabel(row.variant);
  if (row.family === 'sf') return stockfishEngineLabel(row.variant, 'arena');
  if (row.family === 'viridithas') return viridithasVariantForKey(row.variant).label;
  if (row.family === 'berserk') return berserkVariantForKey(row.variant).label;
  if (row.family === 'plentychess') return plentyChessVariantForKey(row.variant).label;
  if (row.family === 'stormphrax') return stormphraxVariantForKey(row.variant).label;
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

function runtimeAuditIdentityForRow(row: EngineRow): string | null {
  if (row.family === 'centipawn') {
    const model = centipawnModelForVariant(row.variant);
    return browserRuntimeAuditIdentity({
      source: 'active-arena-engine',
      surface: 'arena',
      family: 'centipawn',
      engineLabel: centipawnEngineLabel(row.variant),
      modelId: model.modelId,
      requestedRuntime: centipawnRuntimeForVariant(row.variant),
    });
  }
  if (row.family === 'lc0' && !isLc0BigNetVariant(row.variant)) {
    return browserRuntimeAuditIdentity({
      source: 'active-arena-engine',
      surface: 'arena',
      family: 'lc0',
      engineLabel: 'LC0',
      modelId: 'lc0-default',
      requestedRuntime: selectedLc0Runtime(),
    });
  }
  return null;
}

function activeRuntimeAuditIdentities(): Set<string> {
  const identities = new Set<string>();
  for (const row of activeSeatRows()) {
    const identity = runtimeAuditIdentityForRow(row);
    if (identity) identities.add(identity);
  }
  return identities;
}

function reconcileActiveRuntimeFallbackWarning(): void {
  const warningTarget = document.getElementById('runtimeWarning');
  if (warningTarget) reconcileRuntimeFallbackWarning(warningTarget, activeRuntimeAuditIdentities());
}

function seatEngineId(index: number): string {
  return engineIdForRow(seatRows[index]);
}

function renderSeatSelectors(): void {
  for (const row of seatRows) {
    const options = variantOptions(row.family);
    const selected = options.find((option) => option.value === row.variant);
    const pendingBigNet = !bt4AvailabilityResolved && row.family === 'lc0' && isLc0BigNetVariant(row.variant);
    if (!selected || (selected.disabled && !pendingBigNet && !/\(checking\b/.test(selected.label))) {
      row.variant = options.find((option) => !option.disabled)?.value ?? defaultVariant(row.family);
    }
  }
  const families = engineFamilyOptions();
  const matchMode = arenaTournamentMode() === 'match';
  const head = `<div class="seat-head" aria-hidden="true"><span>Seat</span><span>Engine</span><span>Version</span></div>`;
  el('arenaSeatList').innerHTML =
    head +
    seatRows
      .map((row, index) => {
        const meta = strengthMeta(row.family);
        const famSel = families.map(({ value, label }) => `<option value="${value}"${row.family === value ? ' selected' : ''}>${label}</option>`).join('');
        const varSel = variantOptions(row.family)
          .map(
            (option) =>
              `<option value="${option.value}"${row.variant === option.value ? ' selected' : ''}${option.disabled ? ' disabled' : ''}>${htmlEscape(option.label)}</option>`,
          )
          .join('');
        const label = `Engine ${index + 1}`;
        const inactive = matchMode && index >= 2 ? ' seat-inactive' : '';
        const remove =
          seatRows.length > 2
            ? `<button type="button" class="seat-remove" data-seat="${index}" title="Remove ${label}" aria-label="Remove ${label}">×</button>`
            : '';
        let avatarName: string = row.family;
        try {
          avatarName = rowLabel(row);
        } catch {
          // Unknown variant: the family id still picks a stable colour.
        }
        return `<div class="engine-row seat-row${inactive}" data-seat="${index}"><span class="seat-name">${engineAvatarHtml(avatarName, 'sm')}<span>${label}</span></span><select class="seat-fam" data-seat="${index}" aria-label="${label} engine">${famSel}</select><select class="seat-var" data-seat="${index}" aria-label="${label} version">${varSel}</select><span class="seat-strength-wrap"><input class="seat-strength row-strength" data-seat="${index}" aria-label="${label} strength (${meta.unit})" type="number" min="${meta.min}" max="${meta.max}" step="1" value="${row.strength}" title="${meta.unit}"><span class="row-unit">${meta.unit}</span></span>${remove}</div>`;
      })
      .join('');
  el('addSeat').toggleAttribute('disabled', running || seatRows.length >= 8);
  reconcileActiveRuntimeFallbackWarning();
  if (!running) renderStrips();
  renderSetupSummary();
}

function syncSeatRowsFromDom(): void {
  const host = el('arenaSeatList');
  for (let index = 0; index < seatRows.length; index++) {
    const familyRaw = host.querySelector<HTMLSelectElement>(`.seat-fam[data-seat="${index}"]`)?.value;
    const family = familyRaw ? canonicalEngineFamily(familyRaw) : null;
    if (family) seatRows[index].family = family;
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

type ArenaMatchState = 'none' | 'live' | 'final' | 'stopped' | 'failed';
let arenaMatchState: ArenaMatchState = 'none';

function readyMessage(): string {
  return arenaTournamentMode() === 'match'
    ? 'Ready. Set up your match.'
    : 'Ready. Set up your tournament.';
}

function isReadyMessage(text: string | null): boolean {
  return !!text && text.startsWith('Ready. Set up ');
}

function arenaGamesPerOpening(): number {
  return Math.max(1, Math.min(20, Math.floor(Number(inputEl('gamesInput').value) || 2)));
}

/** One-line description of the locked setup, shown in place of the forms while a match runs. */
function renderSetupSummary(): void {
  const node = document.getElementById('setupSummaryText');
  if (!node) return;
  const mode = arenaTournamentMode();
  const names = activeSeatRows().map((row) => {
    try {
      return displayEngineName(engines.get(engineIdForRow(row))?.name ?? rowLabel(row));
    } catch {
      return row.family;
    }
  });
  const who = mode === 'match' ? names.join(' vs ') : `${mode === 'gauntlet' ? 'Gauntlet' : 'Round robin'} · ${names.length} engines`;
  const budget = arenaBudgetMode() === 'movetime' ? `${arenaMovetimeMs()} ms/move` : 'Fixed visits / depth';
  node.textContent = checkpoint ? checkpoint.config.controls.budgetModeSelect === 'movetime' ? `${checkpoint.config.controls.movetimeInput} ms/move` : 'Fixed visits / depth' : budget;
  let scheduleText: string;
  try {
    const openings = selectedOpenings();
    const pairs = tournamentPairings(mode, activeSeatRows().map((_, index) => String(index)));
    const count = openings.length * pairs.length * arenaGamesPerOpening();
    scheduleText = `${count} game${count === 1 ? '' : 's'} · ${budget} · ${selectEl('startingPositionSelect').selectedOptions[0]?.textContent ?? 'Custom openings'}`;
  } catch {
    scheduleText = `${budget} · Check custom openings`;
  }
  el('startSummary').innerHTML = `<strong>${htmlEscape(who)}</strong>${htmlEscape(scheduleText)}`;
}

/** Setup vs running sidebar states (CSS keys off these data attributes on <main>). */
function syncArenaPhase(): void {
  const main = document.getElementById('main');
  if (!main) return;
  const wasRunning = main.dataset.phase === 'running';
  main.dataset.phase = running ? 'running' : 'setup';
  if (running && !wasRunning && window.matchMedia('(max-width: 900px)').matches) main.scrollIntoView({ block: 'start' });
  main.dataset.match = arenaMatchState;
  (el('setupPane') as HTMLFieldSetElement).disabled = running || startPending;
  if (!running) main.dataset.config = 'closed';
  const editButton = document.getElementById('editSetup');
  if (editButton) editButton.setAttribute('aria-expanded', String(main.dataset.config === 'open'));
  el('start').textContent = checkpoint ? 'Start new tournament' : arenaTournamentMode() === 'match' ? 'Start match' : 'Start tournament';
  const title = document.getElementById('scoreboardTitle');
  const displayedMode = checkpoint?.config.mode ?? arenaTournamentMode();
  if (title) title.textContent = displayedMode === 'match' ? 'Match' : displayedMode === 'gauntlet' ? 'Gauntlet' : 'Round robin';
  renderSetupSummary();
  renderCheckpointStatus();
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
  const fixedBudget = arenaBudgetMode() === 'fixed';
  for (const node of el('arenaSeatList').querySelectorAll<HTMLElement>('.seat-strength-wrap')) {
    node.hidden = !fixedBudget;
  }
  el('arenaSeatList').dataset.fixed = String(fixedBudget);
  el('addSeat').toggleAttribute('hidden', arenaTournamentMode() === 'match');
  el('addSeat').toggleAttribute('disabled', running || seatRows.length >= 8);
  selectEl('tournamentModeSelect').disabled = running;
  selectEl('arenaConcurrencySelect').disabled = running;
  // Everything that shapes the schedule is locked for the whole match, so no edit silently applies mid-game.
  selectEl('budgetModeSelect').disabled = running;
  inputEl('movetimeInput').disabled = running;
  inputEl('gamesInput').disabled = running;
  refreshArenaConcurrencyNote();
  syncArenaPhase();
}

function isolatedCpuPool(): boolean {
  return (
    arenaCpuOnlyPool(activeSeatRows().map((row) => row.family)) && (!activeSeatRows().some((row) => row.family === 'sf') || stockfishThreadsPlanned() === 1)
  );
}

function selectedArenaConcurrency(): number {
  if (!isolatedCpuPool()) return 1;
  return arenaGameConcurrency(
    { hardwareConcurrency: navigator.hardwareConcurrency, deviceMemory: (navigator as Navigator & { deviceMemory?: number }).deviceMemory },
    selectEl('arenaConcurrencySelect').value,
  );
}

function refreshArenaConcurrencyNote(): void {
  el('arenaConcurrencyNote').textContent = isolatedCpuPool()
    ? `${selectedArenaConcurrency()} game(s) at once · private workers, Threads=1, Hash=64 MiB per engine. Board follows the earliest active game.`
    : 'One game at a time for neural pools or threaded Stockfish.';
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
  return entries
    .slice(0, 4)
    .map(([key, value]) => `${key} ${formatMs(value)}`)
    .join(' · ');
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
  return [...new Set(ids)].filter((id) => engines.has(id) || isolatedCpuSeats.has(id));
}

function rowForEngineId(engineId: string): EngineRow | undefined {
  return isolatedCpuSeats.get(engineId)?.row ?? activeSeatRows().find((row) => engineIdForRow(row) === engineId);
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
    const privateSeat = isolatedCpuSeats.get(id);
    if (privateSeat) return cpuArenaRuntimeSummary(privateSeat.seat, cpuEngineHashMbForSurface('arena'));
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
    if (row?.family === 'centipawn')
      return `${name}: ${budgetText(row)} · SquareFormer ${centipawnRuntimeForVariant(row.variant)} · ${centipawnHybridManifestStatusText()}`;
    if (row?.family === 'sf') {
      const kind = row.variant === 'full' ? 'full' : 'lite';
      const threads = stockfishThreadsPlanned();
      return `${name}: ${budgetText(row)} · ${stockfishFlavorLabel(stockfishFlavorFor(kind))} · ${threads} thread${threads === 1 ? '' : 's'}`;
    }
    if (row?.family === 'reckless') {
      const variant = recklessVariantForKey(row.variant);
      const engine = recklessEngines.peek(variant);
      return `${name}: ${budgetText(row)} · ${variant.label} · ${engine?.runtimeLabel() ?? 'not loaded'} · hash 16MB`;
    }
    if (row?.family === 'viridithas') {
      const variant = viridithasVariantForKey(row.variant);
      const engine = viridithasEngines.peek(variant);
      return `${name}: ${budgetText(row)} · ${variant.label} · ${engine?.runtimeLabel() ?? 'not loaded'} · hash 16MB`;
    }
    if (row?.family === 'berserk') {
      const variant = berserkVariantForKey(row.variant);
      const engine = berserkEngines.peek(variant);
      return `${name}: ${budgetText(row)} · ${variant.label} · ${engine?.runtimeLabel() ?? 'not loaded'} · hash 16MB`;
    }
    if (row?.family === 'plentychess') {
      const variant = plentyChessVariantForKey(row.variant);
      const engine = plentyChessEngines.peek(variant);
      return `${name}: ${budgetText(row)} · ${variant.label} · ${engine?.runtimeLabel() ?? 'not loaded'} · hash 16MB`;
    }
    if (row?.family === 'stormphrax') {
      const variant = stormphraxVariantForKey(row.variant);
      const engine = stormphraxEngines.peek(variant);
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
  const backend = timingMeansText(
    t.evalBackendTimingPerPositionMeans ?? t.evalBackendTimingMeans ?? t.lastBackendTimingPerPositionMeans ?? t.lastBackendTimingMeans,
  );
  const batch = t.lastBatchSize
    ? ` · batch ${t.lastBatchSize} · maxEvalBatch ${t.maxEvalBatch ?? t.lastBatchSize}${t.lastBatchPipelineDepth && t.lastBatchPipelineDepth > 1 ? ` · pipeline ${t.lastBatchPipelineDepth}` : ''}`
    : '';
  return `${t.engineName}: searches ${t.searches} · avg ${averageMs(t.totalElapsedMs, t.searches)} · last ${formatMs(t.lastElapsedMs)}${backend ? ` · backend avg ${backend}` : ''}${batch} · tree reuse ${ratioText(t.rootReused, t.searches)} (${fresh} fresh) · reused visits ${t.reusedRootVisits} · reply parent ${ratioText(t.replyParentsExpanded, t.replyChecks)} · reply visited ${ratioText(t.replyVisited, t.replyChecks)} · opp reply top-policy≤5 ${ratioText(t.replyTopPolicy5, t.replyChecks)} · top-visits≤5 ${ratioText(t.replyTopVisits5, t.replyChecks)} · evals ${t.evalCalls} · cache hits ${t.cacheHits} · trans ${t.transpositionHits}`;
}

function uciTelemetrySummary(t: UciEngineTelemetry): string {
  const last =
    t.lastDepth !== undefined
      ? `last d${t.lastDepth} · nodes ${formatCount(t.lastNodes)} · nps ${formatCount(t.lastNps)} · PV ${t.lastPvMoves ?? 0}`
      : 'waiting for first PV';
  return `${t.engineName}: searches ${t.searches} · avg ${averageMs(t.totalElapsedMs, t.searches)} · last ${formatMs(t.lastElapsedMs)} · total nodes ${formatCount(t.totalNodes)} · ${last}${t.lastMultiPv && t.lastMultiPv > 1 ? ` · MultiPV ${t.lastMultiPv}` : ''}`;
}

function bt4TelemetrySummary(t: Bt4Telemetry, searcher: Bt4WorkerSearcher): string {
  return `${t.engineName}: searches ${t.searches} · avg ${averageMs(t.totalElapsedMs, t.searches)} · visits ${t.completedVisits} · evals ${t.evalCalls} · cache hits ${t.cacheHits} · last ${formatMs(t.lastElapsedMs)} · ${searcher.loaded ? `backend ${searcher.backend || 'WebGPU'}` : 'worker not loaded'}`;
}

function engineSearchDiagnosticsText(ids = diagnosticEngineIds()): string {
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
      return t?.searches
        ? bt4TelemetrySummary(t, searcher)
        : `${name}: ${config.name} search waiting${searcher.loaded ? ` · backend ${searcher.backend || 'WebGPU'}` : ''}`;
    }
    if (row?.family === 'centipawn') return engineOutputs.has(id) ? `${name}: Centipawn search output ready` : `${name}: Centipawn search waiting`;
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

function searchTelemetryText(ids = diagnosticEngineIds()): string {
  return engineSearchDiagnosticsText(ids);
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

function recordLc0SearchOutput(engineId: string, engineName: string, result: Lc0SearchResult, elapsedMs?: number): void {
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
    ...lc0EvalFields(result.fen, result.value, rootWdl),
    visits: result.visits,
    elapsedMs,
    nps: elapsedMs && elapsedMs > 0 ? (result.visits * 1000) / elapsedMs : undefined,
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
    ...lc0EvalFields(result.fen, result.value),
    visits: result.visits,
  });
}

function recordCentipawnSearchOutput(engineId: string, engineName: string, fen: string, result: CentipawnSearchResult): void {
  const pv = result.principalVariation?.map((entry) => moveToUci(entry.move));
  recordEngineOutput({
    engineId,
    engineName,
    kind: 'lc0',
    fen,
    move: result.move ? moveToUci(result.move) : undefined,
    summary: `Centipawn Q ${signed(toWhiteQ(result.value, fen), 3)}`,
    shortEval: `Q ${signed(toWhiteQ(result.value, fen), 2)}`,
    evalBar: qEvalBar(fen, result.value),
    detail: `visits ${result.visits} · evals ${result.stats?.evalCalls ?? 0} · cache hits ${result.stats?.cacheHits ?? 0}`,
    pv,
    ...lc0EvalFields(fen, result.value),
    visits: result.visits,
  });
}

function recordUciOutput(
  engineId: string,
  engineName: string,
  label: string,
  fen: string,
  move: string | null,
  lines: StockfishInfoLine[],
  elapsedMs?: number,
  visible = true,
): void {
  settleCpuArenaSearch(engineId, thinkingEngineIds, arenaSearchProgress);
  recordUciTelemetry(engineId, engineName, lines, elapsedMs);
  if (!visible) return;
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
    detail:
      lines.length > 1
        ? `MultiPV ${(lines.slice(0, 3) as StockfishInfoLine[]).map((line) => `#${line.multipv ?? 1} ${stockfishScoreText(line, fen)}`).join(' · ')}`
        : undefined,
    pv: best?.pvUci,
    whiteCp: stockfishWhiteCp(best, fen),
    mateInWhitePov: stockfishMateInWhitePov(best, fen),
    evalText: best ? formatScore(stockfishWhiteCp(best, fen), stockfishMateInWhitePov(best, fen)) : undefined,
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

function recordStormphraxOutput(engineId: string, engineName: string, fen: string, move: string | null, lines: StockfishInfoLine[], elapsedMs?: number): void {
  recordUciOutput(engineId, engineName, 'Stormphrax', fen, move, lines, elapsedMs);
}

function recordEngineThinking(engine: Pick<ArenaEngine, 'id' | 'name'>): void {
  thinkingEngineIds.add(engine.id);
  thinkingSince.set(engine.id, performance.now());
  arenaSearchProgress.set(engine.id, { label: engine.name, units: 'search', indeterminate: true });
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
      const policyRank = child.edges
        .slice()
        .sort((a, b) => b.prior - a.prior)
        .findIndex((edge) => moveToUci(edge.move) === replyUci);
      const visitsRank = child.edges
        .slice()
        .sort((a, b) => b.visits - a.visits || b.prior - a.prior)
        .findIndex((edge) => moveToUci(edge.move) === replyUci);
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
    if (!stockfishLite)
      stockfishLite = new StockfishEngine(
        { depth: 4, threads: stockfishThreadsPlanned(), hashMb: cpuEngineHashMbForSurface('arena') },
        stockfishFlavorUrl(stockfishFlavorFor('lite')),
      );
    return stockfishLite;
  }
  if (!stockfishFull)
    stockfishFull = new StockfishEngine(
      { depth: 4, threads: stockfishThreadsPlanned(), hashMb: cpuEngineHashMbForSurface('arena') },
      stockfishFlavorUrl(stockfishFlavorFor('full')),
    );
  return stockfishFull;
}

function disposeStockfish(): void {
  stockfishLite?.dispose();
  stockfishLite = null;
  stockfishFull?.dispose();
  stockfishFull = null;
}

function availableRecklessVariants(): RecklessVariant[] {
  return recklessBrowserVariants.available();
}

function recklessVariantForKey(variantKey: string): RecklessVariant {
  return recklessBrowserVariants.resolve(variantKey);
}

function getRecklessFor(variantKey: string): RecklessEngine {
  return recklessEngines.getOrCreate(recklessVariantForKey(variantKey));
}

function recklessMissingAssetMessage(variants: RecklessVariant[]): string {
  const urls = variants.flatMap((variant) => [variant.wasmUrl, ...(variant.nnueUrl ? [variant.nnueUrl] : [])]);
  return `Reckless asset missing: ${urls.join(', ')}. Build/publish Reckless artifacts with npm run reckless:build-production, or choose Stockfish/Centipawn/LC0 instead.`;
}

function renderRecklessRuntimeInfo(): void {
  const rows = activeSeatRows().filter((row) => row.family === 'reckless');
  el('recklessRuntimeInfo').hidden = !rows.length;
  if (!rows.length) {
    el('recklessRuntimeInfo').textContent = '';
    return;
  }
  const parts = rows.map((row) => {
    const variant = recklessVariantForKey(row.variant);
    const engine = recklessEngines.peek(variant);
    const status = engine?.runtimeStatus();
    const mode = engine?.runtimeLabel() ?? (typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated ? 'persistent available' : 'one-shot fallback');
    const asset = recklessVariantAssetStatus(variant);
    if (assetProbePending(asset)) void checkRecklessVariantAsset(variant, whileArenaMounted(renderRecklessRuntimeInfo));
    const assetText = asset === 'present' ? 'asset ok' : asset === 'missing' ? 'asset missing' : 'checking asset';
    const loadText = formatRecklessBrowserApiLoadStatus(status?.browserApiLoad);
    return `${variant.label} · ${mode} · ${assetText}${loadText ? ` · ${loadText}` : ''}${status?.persistentDisabled ? ' · persistent disabled after fallback' : ''}`;
  });
  el('recklessRuntimeInfo').innerHTML = diagBlockHtml('Reckless', [...new Set(parts)].map(htmlEscape));
}

function refreshRecklessVariantUi(): void {
  const select = selectEl('recklessVariantSelect');
  if (!select.options.length) {
    select.innerHTML = availableRecklessVariants()
      .map((variant) => `<option value="${variant.key}">${htmlEscape(variant.label)}</option>`)
      .join('');
  }
  select.disabled = running;
  renderRecklessRuntimeInfo();
}

function availableViridithasVariants(): ViridithasVariant[] {
  return viridithasBrowserVariants.available();
}

function viridithasVariantForKey(variantKey: string): ViridithasVariant {
  return viridithasBrowserVariants.resolve(variantKey);
}

function getViridithasFor(variantKey: string) {
  return viridithasEngines.getOrCreate(viridithasVariantForKey(variantKey));
}

function renderViridithasRuntimeInfo(): void {
  const rows = activeSeatRows().filter((row) => row.family === 'viridithas');
  el('viridithasRuntimeInfo').hidden = !rows.length;
  if (!rows.length) {
    el('viridithasRuntimeInfo').textContent = '';
    return;
  }
  const parts = rows.map((row) => {
    const variant = viridithasVariantForKey(row.variant);
    const engine = viridithasEngines.peek(variant);
    const status = engine?.runtimeStatus();
    const mode = engine?.runtimeLabel() ?? (canUsePersistentViridithasWasi() ? 'persistent available' : 'one-shot fallback');
    const asset = viridithasVariantAssetStatus(variant);
    if (assetProbePending(asset)) void checkViridithasVariantAsset(variant, whileArenaMounted(renderViridithasRuntimeInfo));
    const assetText = asset === 'ok' ? 'asset ok' : asset === 'missing' ? 'asset missing' : 'checking asset';
    return `${variant.label} · ${mode} · ${assetText}${status?.persistentDisabled ? ' · persistent disabled after fallback' : ''}`;
  });
  el('viridithasRuntimeInfo').innerHTML = diagBlockHtml('Viridithas', [...new Set(parts)].map(htmlEscape));
}

function refreshViridithasVariantUi(): void {
  const select = selectEl('viridithasVariantSelect');
  if (!select.options.length) {
    select.innerHTML = availableViridithasVariants()
      .map((variant) => `<option value="${variant.key}">${htmlEscape(variant.label)}</option>`)
      .join('');
  }
  select.disabled = running;
  renderViridithasRuntimeInfo();
}

function availableBerserkVariants(): BerserkVariant[] {
  return berserkBrowserVariants.available();
}

function berserkVariantForKey(variantKey: string): BerserkVariant {
  return berserkBrowserVariants.resolve(variantKey);
}

function getBerserkFor(variantKey: string) {
  return berserkEngines.getOrCreate(berserkVariantForKey(variantKey));
}

function renderBerserkRuntimeInfo(): void {
  const rows = activeSeatRows().filter((row) => row.family === 'berserk');
  el('berserkRuntimeInfo').hidden = !rows.length;
  if (!rows.length) {
    el('berserkRuntimeInfo').textContent = '';
    return;
  }
  const parts = rows.map((row) => {
    const variant = berserkVariantForKey(row.variant);
    const engine = berserkEngines.peek(variant);
    const asset = berserkVariantAssetStatus(variant);
    if (assetProbePending(asset)) void checkBerserkVariantAsset(variant, whileArenaMounted(renderBerserkRuntimeInfo));
    const assetText = asset === 'present' ? 'asset ok' : asset === 'missing' ? 'asset missing' : 'checking asset';
    return `${variant.label} · ${engine?.runtimeLabel() ?? 'Emscripten worker idle'} · ${assetText}`;
  });
  el('berserkRuntimeInfo').innerHTML = diagBlockHtml('Berserk', [...new Set(parts)].map(htmlEscape));
}

function refreshBerserkVariantUi(): void {
  const select = selectEl('berserkVariantSelect');
  if (!select.options.length) {
    select.innerHTML = availableBerserkVariants()
      .map((variant) => `<option value="${variant.key}">${htmlEscape(variant.label)}</option>`)
      .join('');
  }
  select.disabled = running;
  renderBerserkRuntimeInfo();
}

function availablePlentyChessVariants(): PlentyChessVariant[] {
  return plentyChessBrowserVariants.available();
}

function plentyChessVariantForKey(variantKey: string): PlentyChessVariant {
  return plentyChessBrowserVariants.resolve(variantKey);
}

function getPlentyChessFor(variantKey: string) {
  return plentyChessEngines.getOrCreate(plentyChessVariantForKey(variantKey));
}

function renderPlentyChessRuntimeInfo(): void {
  const rows = activeSeatRows().filter((row) => row.family === 'plentychess');
  el('plentychessRuntimeInfo').hidden = !rows.length;
  if (!rows.length) {
    el('plentychessRuntimeInfo').textContent = '';
    return;
  }
  const parts = rows.map((row) => {
    const variant = plentyChessVariantForKey(row.variant);
    const engine = plentyChessEngines.peek(variant);
    const asset = plentyChessVariantAssetStatus(variant);
    const unsupportedReason = plentyChessVariantUnsupportedReason(variant);
    if (!unsupportedReason && assetProbePending(asset)) void checkPlentyChessVariantAsset(variant, whileArenaMounted(renderPlentyChessRuntimeInfo));
    const assetText = unsupportedReason ? unsupportedReason : asset === 'present' ? 'asset ok' : asset === 'missing' ? 'asset missing' : 'checking asset';
    return `${variant.label} · ${engine?.runtimeLabel() ?? 'Emscripten worker idle'} · ${assetText}`;
  });
  el('plentychessRuntimeInfo').innerHTML = diagBlockHtml('PlentyChess', [...new Set(parts)].map(htmlEscape));
}

function refreshPlentyChessVariantUi(): void {
  const select = selectEl('plentychessVariantSelect');
  const selected = select.value;
  select.innerHTML = availablePlentyChessVariants()
    .map((variant) => {
      const status = plentyChessVariantAssetStatus(variant);
      const unsupportedReason = plentyChessVariantUnsupportedReason(variant);
      const needsGeneratedAsset = variant.key === 'emscripten-sse41' || variant.key === 'emscripten-relaxed';
      if (!unsupportedReason && needsGeneratedAsset && assetProbePending(status))
        void checkPlentyChessVariantAsset(variant, whileArenaMounted(refreshPlentyChessVariantUi));
      const option = browserVariantOption(variant.key, variant.label, { assetStatus: status, unsupportedReason, requirePresent: needsGeneratedAsset });
      return `<option value="${option.value}"${option.disabled ? ' disabled' : ''}>${htmlEscape(option.label)}</option>`;
    })
    .join('');
  if (selected) select.value = selected;
  select.disabled = running;
  renderPlentyChessRuntimeInfo();
}

function availableStormphraxVariants(): StormphraxVariant[] {
  return stormphraxBrowserVariants.available();
}

function stormphraxVariantForKey(variantKey: string): StormphraxVariant {
  return stormphraxBrowserVariants.resolve(variantKey);
}

function getStormphraxFor(variantKey: string) {
  return stormphraxEngines.getOrCreate(stormphraxVariantForKey(variantKey));
}

function renderStormphraxRuntimeInfo(): void {
  const rows = activeSeatRows().filter((row) => row.family === 'stormphrax');
  el('stormphraxRuntimeInfo').hidden = !rows.length;
  if (!rows.length) {
    el('stormphraxRuntimeInfo').textContent = '';
    return;
  }
  const parts = rows.map((row) => {
    const variant = stormphraxVariantForKey(row.variant);
    const engine = stormphraxEngines.peek(variant);
    const asset = stormphraxVariantAssetStatus(variant);
    const unsupportedReason = stormphraxVariantUnsupportedReason(variant);
    if (!unsupportedReason && assetProbePending(asset)) void checkStormphraxVariantAsset(variant, whileArenaMounted(renderStormphraxRuntimeInfo));
    const assetText = unsupportedReason ?? (asset === 'present' ? 'asset ok' : asset === 'missing' ? 'asset missing' : 'checking asset');
    return `${variant.label} · ${engine?.runtimeLabel() ?? 'Emscripten worker idle'} · ${assetText}`;
  });
  el('stormphraxRuntimeInfo').innerHTML = diagBlockHtml('Stormphrax', [...new Set(parts)].map(htmlEscape));
}

function refreshStormphraxVariantUi(): void {
  const select = selectEl('stormphraxVariantSelect');
  const selected = select.value;
  select.innerHTML = availableStormphraxVariants()
    .map((variant) => `<option value="${variant.key}">${htmlEscape(variant.label)}</option>`)
    .join('');
  if (selected) select.value = selected;
  select.disabled = running;
  renderStormphraxRuntimeInfo();
}

function refreshStockfishControls(): void {
  inputEl('stockfishThreadsInput').disabled = running || !threadedStockfishAvailable();
  inputEl('stockfishThreadsInput').value = String(stockfishThreadsCap());
  inputEl('stockfishThreadsInput').title = `0 = auto (broker grants ${resourceBroker.cpuBudget()} on this device, ${resourceBroker.getDial()} dial)`;
}

// Lc0 big nets are WebGPU-only and locally staged; disable/downgrade them
// when WebGPU is unusable or their ONNX asset is missing.
async function refreshBt4Availability(mountSignal = mountAbort.signal): Promise<void> {
  await Promise.all([
    probeBt4Support(),
    checkBigNetAsset(BIG_NETS.bt4, whileArenaMounted(renderSeatSelectors)),
    checkBigNetAsset(BIG_NETS.t3, whileArenaMounted(renderSeatSelectors)),
  ]);
  bt4AvailabilityResolved = true;
  if (isStaleMount(mountSignal)) return;
  for (const row of activeSeatRows()) {
    if (row.family === 'lc0' && isLc0BigNetVariant(row.variant) && !bigNetSelectableSync(bigNetFor(row.variant).config)) row.variant = 'small';
  }
  buildEngines();
  renderSeatSelectors();
  refreshSeatControls();
}

async function renderRuntimeBadge(): Promise<void> {
  const badge = maybeEl('runtimeBadge');
  if (!badge) return;
  if (isV0DeployProfile()) {
    badge.textContent = 'Runtime: v0 deploy · Lc0 small + Stockfish Lite';
    badge.classList.add('ready');
    badge.classList.remove('warn');
    return;
  }
  try {
    const diag = await collectOrtRuntimeDiagnostics({ probeAdapter: true });
    if (!badge.isConnected) return;
    runtimeIsolation = diag.crossOriginIsolated === true;
    runtimeSharedArrayBuffer = diag.wasm.sharedArrayBuffer === true;
    refreshStockfishControls();
    const webgpu = diag.webgpuAvailable ? (diag.adapter?.ok === false ? 'WebGPU unavailable/blocked' : 'WebGPU available') : 'WebGPU unavailable';
    const isolated = runtimeIsolation ? 'isolated' : 'not isolated';
    const sab = runtimeSharedArrayBuffer ? 'SAB yes' : 'SAB no';
    const sfThreads = threadedStockfishAvailable() ? 'SF threaded yes' : 'SF threaded no';
    const ortThreads = `ORT wasm threads ${diag.wasm.numThreads ?? '?'}`;
    const ortSessions = `ORT sessions ${diag.sessions.active} active`;
    badge.textContent = `Runtime: ${isolated} · ${sab} · ${webgpu} · ${diag.describe} · ${ortThreads} · ${ortSessions} · ${sfThreads} · ${centipawnHybridManifestStatusText()}`;
    badge.classList.toggle('ready', runtimeIsolation && (diag.webgpuAvailable || runtimeSharedArrayBuffer));
    badge.classList.toggle('warn', !runtimeIsolation || !diag.webgpuAvailable);
  } catch (error) {
    if (!badge.isConnected) return;
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
  const rows = activeSeatRows();
  recklessEngines.retain(rows.filter((row) => row.family === 'reckless').map((row) => recklessVariantForKey(row.variant)));
  viridithasEngines.retain(rows.filter((row) => row.family === 'viridithas').map((row) => viridithasVariantForKey(row.variant)));
  berserkEngines.retain(rows.filter((row) => row.family === 'berserk').map((row) => berserkVariantForKey(row.variant)));
  plentyChessEngines.retain(rows.filter((row) => row.family === 'plentychess').map((row) => plentyChessVariantForKey(row.variant)));
  stormphraxEngines.retain(rows.filter((row) => row.family === 'stormphrax').map((row) => stormphraxVariantForKey(row.variant)));
}

function buildEngines() {
  for (const row of activeSeatRows()) clampStrength(row);
  const activeIds = new Set(activeSeatRows().map(engineIdForRow));
  pruneUnusedLc0Searchers(activeIds);
  engines.clear();
  disposeUnusedUciEngines();
  disposeUnusedCentipawnEvaluators();
  const warmupPositions = [parseFen(START_FEN)];
  const lc0Search =
    (engineId: string, row: EngineRow): ArenaEngine['move'] =>
    async (positions, signal) => {
      const timed = arenaBudgetMode() === 'movetime';
      const started = performance.now();
      const result = await lc0SearcherFor(engineId).search(
        { positions },
        {
          visits: timed ? undefined : row.strength,
          movetimeMs: timed ? arenaMovetimeMs() : undefined,
          signal,
          yieldEveryMs: 16,
          reuseTree: true,
          batchSize: lc0BatchSize(),
          batchPipelineDepth: lc0BatchPipelineDepth(),
          onProgress: (progress: Lc0SearchProgress) => setArenaVisitProgress(engineId, engines.get(engineId)?.name ?? engineId, progress),
        },
      );
      const elapsedMs = performance.now() - started;
      const engineName = engines.get(engineId)?.name ?? engineId;
      recordLc0SearchTelemetry(engineId, engineName, result, elapsedMs);
      recordLc0SearchOutput(engineId, engineName, result, elapsedMs);
      return result.move ?? null;
    };
  const lc0BigNetMove =
    (engineId: string, row: EngineRow): ArenaEngine['move'] =>
    async (positions, signal) => {
      const { config, searcher } = bigNetFor(row.variant);
      const onAbort = () => searcher.cancel();
      signal.addEventListener('abort', onAbort, { once: true });
      const label = engines.get(engineId)?.name ?? `Lc0 ${config.name}`;
      searcher.onDownloadProgress = (loaded, total) => showDownloadProgress(label, loaded, total, 'Downloading');
      try {
        const timed = arenaBudgetMode() === 'movetime';
        const result = await searcher.search(
          { positions },
          {
            visits: timed ? undefined : row.strength,
            movetimeMs: timed ? arenaMovetimeMs() : undefined,
            reuseTree: true,
            batchSize: config.recommendedBatchSize,
            batchPipelineDepth: config.recommendedPipelineDepth,
            evalCacheEntries: arenaCacheEntries(),
            signal,
            onProgress: (progress) => setArenaVisitProgress(engineId, label, progress),
          },
        );
        if (result.cancelled) return null;
        recordBt4SearchOutput(engineId, engines.get(engineId)?.name ?? `Lc0 ${config.name}`, result);
        return result.move ?? null;
      } finally {
        signal.removeEventListener('abort', onAbort);
        hideDownloadProgress();
        clearArenaSearchProgress(engineId);
      }
    };
  const centipawnMove =
    (engineId: string, row: EngineRow): ArenaEngine['move'] =>
    async (positions, signal) => {
      const current = positions[positions.length - 1];
      const fen = boardToFen(current);
      const evaluator = await centipawnEvaluator(row.variant);
      const timed = arenaBudgetMode() === 'movetime';
      const result = await chooseMove(current, evaluator, {
        visits: timed ? undefined : row.strength,
        movetimeMs: timed ? arenaMovetimeMs() : undefined,
        batchSize: Math.max(1, Math.min(256, Math.floor(Number(params.get('centipawnBatch') ?? params.get('tinyBatch') ?? '32') || 32))),
        signal,
        historyFens: centipawnHistoryFens(positions),
        searchPolicy: montyLitePuctPolicy,
        onProgress: (progress) => {
          const completed = progress.completedVisits ?? progress.visits;
          const elapsedMs = progress.elapsedMs ?? 0;
          setArenaSearchProgress(engineId, engines.get(engineId)?.name ?? centipawnEngineLabel(row.variant), {
            label: engines.get(engineId)?.name ?? centipawnEngineLabel(row.variant),
            completed,
            requested: progress.requestedVisits ?? progress.visits,
            elapsedMs,
            nps: elapsedMs > 0 ? completed / Math.max(1e-9, elapsedMs / 1000) : undefined,
            best: progress.move ? moveToUci(progress.move) : null,
            value: progress.value,
            units: 'visits',
          });
        },
      });
      recordCentipawnSearchOutput(engineId, engines.get(engineId)?.name ?? centipawnEngineLabel(row.variant), fen, result);
      return result.move ? moveToUci(result.move) : null;
    };
  const sf =
    (engineId: string, row: EngineRow, kind: 'lite' | 'full'): ArenaEngine['move'] =>
    async (positions, signal) => {
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
        recordStockfishOutput(
          engineId,
          engines.get(engineId)?.name ?? (kind === 'lite' ? 'Stockfish Lite' : 'Stockfish'),
          fen,
          move,
          engine.lastInfo(),
          elapsedMs,
        );
        return move;
      } finally {
        lease.release();
      }
    };
  const recklessMove =
    (engineId: string, row: EngineRow, engine: RecklessEngine): ArenaEngine['move'] =>
    async (positions, signal) => {
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
  const viridithasMove =
    (engineId: string, row: EngineRow, engine: ViridithasEngine): ArenaEngine['move'] =>
    async (positions, signal) => {
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
  const berserkMove =
    (engineId: string, row: EngineRow, engine: BerserkEngine): ArenaEngine['move'] =>
    async (positions, signal) => {
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
  const plentyChessMove =
    (engineId: string, row: EngineRow, engine: PlentyChessEngine): ArenaEngine['move'] =>
    async (positions, signal) => {
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
  const stormphraxMove =
    (engineId: string, row: EngineRow, engine: StormphraxEngine): ArenaEngine['move'] =>
    async (positions, signal) => {
      if (arenaBudgetMode() === 'movetime') engine.setOptions({ depth: undefined, movetimeMs: arenaMovetimeMs(), threads: 1 });
      else engine.setOptions({ depth: row.strength, movetimeMs: undefined, threads: 1 });
      const fen = boardToFen(positions[positions.length - 1]);
      const started = performance.now();
      const move = await engine.bestMove(fen, signal);
      const elapsedMs = performance.now() - started;
      recordStormphraxOutput(engineId, engines.get(engineId)?.name ?? 'Stormphrax', fen, move, engine.lastInfo(), elapsedMs);
      renderStormphraxRuntimeInfo();
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
    searcher.onDownloadProgress = (loaded, total) => showDownloadProgress(`Lc0 ${config.name}`, loaded, total, 'Downloading');
    try {
      await searcher.search(
        { positions: warmupPositions },
        { visits: 1, batchSize: config.recommendedBatchSize, batchPipelineDepth: config.recommendedPipelineDepth, evalCacheEntries: arenaCacheEntries() },
      );
      await searcher.resetTree();
    } finally {
      signal.removeEventListener('abort', onAbort);
      hideDownloadProgress();
    }
  };
  const centipawnWarmup = (row: EngineRow) => async (signal: AbortSignal) => {
    const evaluator = await centipawnEvaluator(row.variant);
    await chooseMove(warmupPositions[0], evaluator, {
      visits: 1,
      batchSize: 1,
      signal,
      searchPolicy: montyLitePuctPolicy,
    });
  };
  const stockfishWarmupStatus = (): ArenaWarmupStatus => {
    const threads = stockfishThreadsPlanned();
    const threadText = threads === 1 ? 'its engine worker' : `${threads} engine threads`;
    const hashMb = cpuEngineHashMbForSurface('arena');
    return {
      phase: 'Loading and configuring the local WebAssembly engine',
      detail: `Stockfish runs entirely in this browser. On first use, this loads its engine files, starts ${threadText}, allocates a ${hashMb} MB search cache, and completes its UCI readiness check. The match clock has not started; game 1 begins automatically.`,
    };
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
  const stormphraxWarmup = (engine: StormphraxEngine) => async (signal: AbortSignal) => {
    engine.setOptions({ depth: 1, movetimeMs: undefined, threads: 1 });
    await engine.bestMove(START_FEN, signal);
    await engine.newGame(signal);
    renderStormphraxRuntimeInfo();
  };
  for (const row of activeSeatRows()) {
    const id = engineIdForRow(row);
    if (engines.has(id)) continue;
    const meta = strengthMeta(row.family);
    const fixedBudget = arenaBudgetMode() === 'fixed';
    const suffix = fixedBudget ? ` ${row.strength}${meta.unit}` : '';
    const name = `${rowLabel(row)}${suffix}`;
    if (row.family === 'lc0') {
      if (isLc0BigNetVariant(row.variant)) engines.set(id, { id, name, move: lc0BigNetMove(id, row), warmup: lc0BigNetWarmup(row.variant) });
      else engines.set(id, { id, name, move: lc0Search(id, row), warmup: lc0SearchWarmup(id) });
    } else if (row.family === 'centipawn') {
      engines.set(id, { id, name, move: centipawnMove(id, row), warmup: centipawnWarmup(row) });
    } else if (row.family === 'sf') {
      const kind = row.variant === 'full' ? 'full' : 'lite';
      engines.set(id, { id, name, move: sf(id, row, kind), warmup: stockfishWarmup(kind), warmupStatus: stockfishWarmupStatus() });
    } else if (row.family === 'reckless') {
      const variant = recklessVariantForKey(row.variant);
      if (variant.key !== 'custom' && recklessVariantAssetStatus(variant) === 'missing') {
        engines.set(id, {
          id,
          name,
          move: async () => {
            throw new Error(recklessMissingAssetMessage([variant]));
          },
        });
        continue;
      }
      const engine = getRecklessFor(row.variant);
      engines.set(id, { id, name, move: recklessMove(id, row, engine), warmup: recklessWarmup(engine) });
    } else if (row.family === 'viridithas') {
      const engine = getViridithasFor(row.variant);
      engines.set(id, { id, name, move: viridithasMove(id, row, engine), warmup: viridithasWarmup(engine) });
    } else if (row.family === 'berserk') {
      const engine = getBerserkFor(row.variant);
      engines.set(id, { id, name, move: berserkMove(id, row, engine), warmup: berserkWarmup(engine) });
    } else if (row.family === 'plentychess') {
      const variant = plentyChessVariantForKey(row.variant);
      const unsupportedReason = plentyChessVariantUnsupportedReason(variant);
      if (unsupportedReason) {
        engines.set(id, {
          id,
          name,
          move: async () => {
            throw new Error(`${variant.label} ${unsupportedReason}.`);
          },
        });
        continue;
      }
      const engine = getPlentyChessFor(row.variant);
      engines.set(id, { id, name, move: plentyChessMove(id, row, engine), warmup: plentyChessWarmup(engine) });
    } else {
      const variant = stormphraxVariantForKey(row.variant);
      const unsupportedReason = stormphraxVariantUnsupportedReason(variant);
      if (unsupportedReason) {
        engines.set(id, {
          id,
          name,
          move: async () => {
            throw new Error(`${variant.label} ${unsupportedReason}.`);
          },
        });
        continue;
      }
      const engine = getStormphraxFor(row.variant);
      engines.set(id, { id, name, move: stormphraxMove(id, row, engine), warmup: stormphraxWarmup(engine) });
    }
  }
  renderRecklessRuntimeInfo();
  renderViridithasRuntimeInfo();
  renderBerserkRuntimeInfo();
  renderPlentyChessRuntimeInfo();
  renderStormphraxRuntimeInfo();
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
  el('matchupNote').textContent = movetime ? 'Equal time per move · colors alternate.' : 'Configured engine strength · colors alternate.';
}

function refreshOpeningPreview(): void {
  const select = selectEl('startingPositionSelect');
  const textarea = el('openingText') as HTMLTextAreaElement;
  const custom = select.value === 'custom';
  el('openingTextField').hidden = !custom;
  select.disabled = running;
  textarea.disabled = running || !custom;
  if (running) return;
  renderSetupSummary();
  try {
    const openings = selectedOpenings();
    el('openingInfo').textContent = `${openings.length} position${openings.length === 1 ? '' : 's'} in the schedule.`;
    if (!checkpoint) setOpeningPreview(openings[0]);
  } catch (error) {
    el('openingInfo').textContent = `Opening setup error: ${(error as Error).message}`;
  }
}

const formatScoreHalf = formatArenaPoints;

function renderMatchScore(nameA: string, nameB: string, sameEngine: boolean, score: MatchScore): void {
  const side = (name: string, seat: string) =>
    `<span class="sb-player">${engineAvatarHtml(name)}<span class="sb-name"><strong>${htmlEscape(displayEngineName(name))}</strong>${sameEngine ? `<small>${seat}</small>` : ''}</span></span>`;
  const points = (value: number) => formatScoreHalf(value) || '0';
  const record = score.games ? `<div class="sb-record">${htmlEscape(displayEngineName(nameA))} · W ${score.aWins} · D ${score.draws} · L ${score.bWins} · ${score.games}/${scheduledGameCount || score.games} finished</div>` : '';
  el('matchScore').innerHTML =
    `<div class="sb-match">${side(nameA, 'Engine 1')}<span class="sb-points" aria-label="Score ${points(score.a)} to ${points(score.b)}">${points(score.a)}<span class="sb-dash">–</span>${points(score.b)}</span>${side(nameB, 'Engine 2').replace('sb-player', 'sb-player right')}</div>` +
    record;
  renderGameResultChips();
}

function renderStandings(standings: TournamentStandings, scheduledGames: number): void {
  const rows = standings.table();
  const elo = (row: ReturnType<TournamentStandings['table']>[number]) =>
    row.eloDiff === null ? '—' : `${row.eloDiff >= 0 ? '+' : ''}${Math.round(row.eloDiff)}${row.eloError !== null ? ` ±${Math.round(row.eloError)}` : ''}`;
  const body = rows
    .map(
      (row, i) =>
        `<tr${i === 0 && row.games ? ' class="leader"' : ''}><td>${i + 1}</td><td><span class="sb-cell-name">${engineAvatarHtml(row.name, 'sm')}${htmlEscape(displayEngineName(row.name))}</span></td><td>${formatScoreHalf(row.points) || '0'}</td><td>${row.wins}</td><td>${row.draws}</td><td>${row.losses}</td><td>${elo(row)}</td></tr>`,
    )
    .join('');
  el('matchScore').innerHTML =
    `<div class="arena-table-scroll" tabindex="0" role="region" aria-label="Standings"><table class="standings"><thead><tr><th>#</th><th>Engine</th><th title="Points">Pts</th><th title="Wins">W</th><th title="Draws">D</th><th title="Losses">L</th><th title="Elo vs the pool, ±95% approx">Elo</th></tr></thead><tbody>${body}</tbody></table></div>` +
    `<div class="sb-record">${standings.totalGames()} of ${scheduledGames} games played · Elo vs the pool, ±95%</div>`;
  renderGameResultChips();
  renderCrosstable(standings);
}

function renderCrosstable(standings: TournamentStandings): void {
  const rows = standings.table();
  const host = el('crosstableSection');
  host.hidden = rows.length < 3;
  const name = (row: (typeof rows)[number]) => htmlEscape(displayEngineName(row.name));
  el('crosstable').innerHTML = `<table class="arena-detail-table"><caption>Row engine’s points / games against each opponent.</caption><thead><tr><th scope="col">Engine</th>${rows.map((row) => `<th scope="col">${name(row)}</th>`).join('')}</tr></thead><tbody>${rows.map((row) => `<tr><th scope="row">${name(row)}</th>${rows.map((opponent) => {
    const record = row.perOpponent.get(opponent.id);
    if (row.id === opponent.id) return '<td class="cross-self" aria-label="Same seat">—</td>';
    if (!record) return '<td aria-label="No games played">—</td>';
    const count = record.wins + record.draws + record.losses;
    return `<td title="${record.wins} wins, ${record.draws} draws, ${record.losses} losses">${formatArenaPoints(record.wins + record.draws / 2)} / ${count}</td>`;
  }).join('')}</tr>`).join('')}</tbody></table>`;
}

function renderOpeningPairs(): void {
  const section = el('openingPairsSection');
  section.hidden = !resultsSchedule.length;
  const selector = selectEl('pairMatchup');
  const previous = selector.value;
  const pairs = arenaOpeningPairs(resultsSchedule, games);
  const matchups = [...new Map(pairs.map((pair) => [pair.pairingKey, pair])).values()];
  const name = (id: string) => htmlEscape(displayEngineName(resultsNames.get(id) ?? id));
  selector.innerHTML = matchups.map((pair) => `<option value="${htmlEscape(pair.pairingKey)}">${name(pair.a)} vs ${name(pair.b)}</option>`).join('');
  if (matchups.some((pair) => pair.pairingKey === previous)) selector.value = previous;
  selector.hidden = matchups.length < 2;
  const selected = pairs.filter((pair) => pair.pairingKey === selector.value);
  const first = selected[0];
  if (!first) return;
  const cell = (pair: (typeof selected)[number], slot: number) => {
    const index = pair.indices[slot];
    const result = pair.results[slot];
    if (index === undefined) return '<td aria-label="Not scheduled">—</td>';
    if (!result) return `<td title="Game ${index + 1} is not finished">Pending</td>`;
    const points = result === '1/2-1/2' ? 0.5 : (resultsSchedule[index].whiteId === pair.a) === (result === '1-0') ? 1 : 0;
    return `<td><button type="button" data-game="${index}" title="Replay game ${index + 1}">${formatArenaPoints(points)}–${formatArenaPoints(1 - points)}</button></td>`;
  };
  el('openingPairs').innerHTML = `<table class="arena-detail-table"><caption>Scores in ${name(first.a)}–${name(first.b)} order. Select a game score to replay it.</caption><thead><tr><th scope="col">Opening</th><th scope="col">${name(first.a)}<br>White</th><th scope="col">${name(first.a)}<br>Black</th><th scope="col">Pair total</th></tr></thead><tbody>${selected.map((pair) => {
    const total = pair.played === 2 ? `${formatArenaPoints(pair.points[0])}–${formatArenaPoints(pair.points[1])}` : pair.indices.length < 2 ? 'Unpaired' : `${pair.played} of 2`;
    return `<tr><th scope="row">${htmlEscape(pair.opening.name)}${pair.round > 1 ? ` · pair ${pair.round}` : ''}</th>${cell(pair, 0)}${cell(pair, 1)}<td>${total}</td></tr>`;
  }).join('')}</tbody></table>`;
}

/** Finished games as small chips (Engine 1's result in a match); click one to replay it. */
function renderGameResultChips(): void {
  const host = el('gameChips');
  host.hidden = !gameResultChips.length;
  host.innerHTML = gameResultChips
    .map((chip) => {
      const outcome = chip.aPoints === null ? (chip.result === '1/2-1/2' ? 'draw' : 'decisive') : chip.aPoints === 1 ? 'win' : chip.aPoints === 0 ? 'loss' : 'draw';
      const text = chip.result === '1/2-1/2' ? '½–½' : chip.result.replace('-', '–');
      return `<button type="button" class="sb-chip ${outcome}" data-game="${chip.index}" title="Game ${chip.index + 1}: ${htmlEscape(displayEngineName(chip.whiteName))} (White) vs ${htmlEscape(displayEngineName(chip.blackName))} (Black), ${chip.result}. Click to replay."><span class="sb-chip-n">G${chip.index + 1}</span>${text}</button>`;
    })
    .join('');
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
    signal.addEventListener(
      'abort',
      () => {
        clearTimeout(timer);
        resolve();
      },
      { once: true },
    );
  });
}

function isAbortError(error: unknown): boolean {
  return error instanceof Error && error.name === 'AbortError';
}

function showEngineWarmupStatus(engine: ArenaEngine, index: number, total: number): void {
  const status = el('engineWarmupStatus');
  status.hidden = false;
  el('engineWarmupTitle').textContent = `Preparing ${engine.name} · engine ${index} of ${total}`;
  const fallback: ArenaWarmupStatus = {
    phase: 'Loading the engine and checking that it is ready',
    detail: 'The arena performs a tiny untimed readiness check before game 1 so startup work is not charged to the first move. The match begins automatically.',
  };
  const copy = engine.warmupStatus ?? fallback;
  el('engineWarmupPhase').textContent = copy.phase;
  el('engineWarmupDetail').textContent = copy.detail;
}

function hideEngineWarmupStatus(): void {
  const status = el('engineWarmupStatus');
  status.hidden = true;
  el('engineWarmupTitle').textContent = '';
  el('engineWarmupPhase').textContent = '';
  el('engineWarmupDetail').textContent = '';
}

async function warmUpSelectedEngines(ids: string[], signal: AbortSignal, minimumVisibleMs = 400): Promise<void> {
  const pending: ArenaEngine[] = [];
  const seen = new Set<string>();
  for (const id of ids) {
    const engine = engines.get(id);
    if (!engine?.warmup || seen.has(id)) continue;
    seen.add(id);
    pending.push(engine);
  }
  if (!pending.length) return;
  el('message').textContent = 'Preparing engines before game 1…';
  const visibleSince = performance.now();
  try {
    for (let index = 0; index < pending.length; index++) {
      if (signal.aborted) return;
      const engine = pending[index];
      showEngineWarmupStatus(engine, index + 1, pending.length);
      await engine.warmup!(signal);
      renderCacheInfo();
    }
  } finally {
    const remainingVisibleMs = minimumVisibleMs - (performance.now() - visibleSince);
    if (!signal.aborted && remainingVisibleMs > 0) await sleep(remainingVisibleMs, signal);
    hideEngineWarmupStatus();
  }
}

function legalFromUci(current: BoardState, uci: string | null): Move | undefined {
  return uci ? legalMoves(current).find((m) => moveToUci(m) === uci) : undefined;
}

function chartColorFor(engineId: string): string {
  return CHART_COLORS[arenaChartColorIndex(gameChartSamples, engineId) % CHART_COLORS.length];
}

function formatCompactNumber(value: number): string {
  if (value >= 1e6) return `${(value / 1e6).toFixed(1)}M`;
  if (value >= 1000) return `${Math.round(value / 1000)}k`;
  return String(Math.round(value));
}

function resetGameCharts(): void {
  gameChartSamples = [];
  sideTimeMs.w = 0;
  sideTimeMs.b = 0;
  rootChartContext = null;
  el('chartsPanel').hidden = false;
  selectChart('evalChart');
  el('rootChartTab').hidden = true;
  for (const id of ['evalChart', 'timeChart', 'npsChart', 'rootChart']) el(id).innerHTML = '';
  el('rootChartTitle').textContent = 'LC0 root visits';
  (el('rootChart').parentElement as HTMLElement).hidden = true;
  el('chartLegend').innerHTML = '';
}

function selectChart(id: string): void {
  el('chartsPanel').dataset.chart = id;
  for (const button of el('chartsPanel').querySelectorAll<HTMLButtonElement>('button[data-chart]')) {
    button.setAttribute('aria-pressed', String(button.dataset.chart === id));
  }
}

/** Label generated SVGs locally; the shared chart builders also serve other pages. */
function describeChart(id: string, summary: string): void {
  const svg = el(id).querySelector('svg');
  if (!svg) return;
  svg.setAttribute('aria-label', summary);
  const title = document.createElementNS('http://www.w3.org/2000/svg', 'title');
  title.textContent = summary;
  svg.prepend(title);
}

function renderGameCharts(): void {
  const chartSize = { width: Math.max(300, el('chartsPanel').clientWidth - 24), height: 64 };
  const ids = [...new Set(gameChartSamples.map((sample) => sample.engineId))];
  const seriesFor = (value: (sample: GameChartSample) => number | undefined): ChartSeries[] =>
    ids.map((id) => ({
      label: arenaChartEngineName(gameChartSamples, id),
      color: chartColorFor(id),
      points: gameChartSamples
        .filter((sample) => sample.engineId === id)
        .flatMap((sample) => {
          const y = value(sample);
          return y === undefined || !Number.isFinite(y) ? [] : [{ x: sample.ply, y }];
        }),
    }));
  el('evalChart').innerHTML = lineChartSvg(
    seriesFor((sample) => sample.whiteScore),
    { ...chartSize, yMin: 0, yMax: 1, midline: 0.5, formatY: (v) => `${Math.round(v * 100)}%` },
  );
  el('timeChart').innerHTML = lineChartSvg(
    seriesFor((sample) => sample.moveMs),
    { ...chartSize, yMin: 0, formatY: formatCompactNumber },
  );
  el('npsChart').innerHTML = lineChartSvg(
    seriesFor((sample) => sample.nps),
    { ...chartSize, yMin: 0, formatY: formatCompactNumber },
  );
  const summaries = (value: (sample: GameChartSample) => string) => ids.map((id) => {
    const sample = gameChartSamples.findLast((entry) => entry.engineId === id);
    return sample ? `${displayEngineName(sample.engineName)}: ${value(sample)}` : '';
  }).join('; ');
  describeChart('evalChart', `White expected score by ply. Latest: ${summaries((sample) => sample.whiteScore === undefined ? 'unavailable' : percent0(sample.whiteScore))}. Use the Review game position slider to explore.`);
  describeChart('timeChart', `Search time by ply. Latest: ${summaries((sample) => formatSeconds(sample.moveMs))}. Use the Review game position slider to explore.`);
  describeChart('npsChart', `Search speed by ply. Latest: ${summaries((sample) => sample.nps === undefined ? 'unavailable' : `${formatNps(sample.nps)} nodes or visits per second`)}. Use the Review game position slider to explore.`);
  el('chartLegend').innerHTML = ids
    .map((id) => `<span><span class="swatch" style="background:${chartColorFor(id)}"></span>${htmlEscape(arenaChartEngineName(gameChartSamples, id))}</span>`)
    .join('');
}

/** Live PUCT-root distribution from the real search tree (Lc0SearchResult.children). */
function renderRootChart(engineId: string, display = true): void {
  const result = lastLc0SearchResults.get(engineId);
  if (!result?.children?.length) return;
  const top = [...result.children]
    .sort((a, b) => b.visits - a.visits)
    .slice(0, 8)
    .filter((child) => child.visits > 0);
  if (!top.length) return;
  const data = { fen: result.fen, engineName: engines.get(engineId)?.name ?? engineId,
    children: top.map((child) => ({ uci: child.uci, visits: child.visits, q: child.q, prior: child.prior })) };
  if (liveTrail) liveTrail.rootChart = data;
  if (display) renderSavedRootChart(data);
}

function renderSavedRootChart(data?: ArenaRootChartSample): void {
  if (!data) {
    rootChartContext = null;
    el('rootChartTab').hidden = true;
    (el('rootChart').parentElement as HTMLElement).hidden = true;
    if (el('chartsPanel').dataset.chart === 'rootChart') selectChart('evalChart');
    return;
  }
  const top = data.children;
  rootChartContext = { fen: data.fen, top };
  el('rootChartTitle').textContent = `${data.engineName} root visits (Q from side to move)`;
  (el('rootChart').parentElement as HTMLElement).hidden = false;
  el('rootChartTab').hidden = false;
  selectEl('rootCandidate').innerHTML = '<option value="">Choose a move…</option>' + top.map((child, index) => `<option value="${index}">${htmlEscape(uciToSan(parseFen(data.fen), child.uci))} · ${formatCount(child.visits)} ${child.visits === 1 ? 'visit' : 'visits'} · Q ${signed(child.q, 2)}</option>`).join('');
  el('rootChart').innerHTML = hBarChartSvg(
    top.map((child) => ({
      label: child.uci,
      value: child.visits,
      detail: `Q ${signed(child.q, 2)} · P ${Math.round(child.prior * 100)}%`,
      color: CHART_COLORS[0],
    })),
  );
  describeChart('rootChart', `Root search candidates by visits. ${top.map((child) => `${child.uci}: ${child.visits} ${child.visits === 1 ? 'visit' : 'visits'}`).join('; ')}. Use Preview a root search candidate to show arrows on the board.`);
}

function recordGameChartSample(ply: number, engine: Pick<ArenaEngine, 'id' | 'name'>, moveMs: number, samples = liveTrail?.charts ?? gameChartSamples, display = true, captured?: ArenaChartSample): void {
  const snapshot = engineOutputs.get(engine.id);
  const lc0 = lastLc0SearchResults.get(engine.id);
  const nps = snapshot?.nps ?? (lc0 && moveMs > 0 ? Math.round((lc0.visits / moveMs) * 1000) : undefined);
  // `board` is still the position the engine searched, so its side to move is the mover.
  if (display) sideTimeMs[board.turn] += moveMs;
  if (snapshot && snapshot.elapsedMs == null) snapshot.elapsedMs = moveMs;
  if (snapshot && snapshot.nps == null && nps !== undefined) snapshot.nps = nps;
  samples.push(captured ?? { ply, engineId: engine.id, engineName: engine.name, whiteScore: snapshot?.evalBar?.whiteScore, moveMs: Math.round(moveMs), nps });
  if (display) renderRootChart(engine.id, !reviewing);
  if (display && !reviewing) {
    gameChartSamples = samples;
    renderGameCharts();
  }
}

async function playArenaGame(
  white: ArenaEngine,
  black: ArenaEngine,
  opening: ArenaOpening,
  signal: AbortSignal,
): Promise<{ result: GameResultCode; reason: string; tree: GameTree }> {
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
  if (!reviewing) {
    resetGameCharts();
    liveTrail.charts = gameChartSamples;
  }
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
      clearArenaSearchProgress(engine.id);
      if (isAbortError(error)) return { result: '1/2-1/2', reason: 'cancelled', tree };
      throw error;
    }
    clearArenaSearchProgress(engine.id);
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

async function ensureSelectedRecklessAssetsAvailable(mountSignal = mountAbort.signal): Promise<boolean> {
  const variants = [
    ...new Map(
      activeSeatRows()
        .filter((row) => row.family === 'reckless')
        .map((row) => {
          const variant = recklessVariantForKey(row.variant);
          return [recklessCacheKey(variant), variant] as const;
        }),
    ).values(),
  ];
  if (!variants.length) return true;
  const missing: RecklessVariant[] = [];
  for (const variant of variants) {
    const status = await checkRecklessVariantAsset(
      variant,
      whileArenaMounted(() => {
        renderRecklessRuntimeInfo();
        renderSeatSelectors();
        refreshSeatControls();
      }),
    );
    if (isStaleMount(mountSignal)) return false;
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
  syncArenaPhase();
}

function createIsolatedCpuEngine(row: EngineRow): BrowserUciEngine {
  if (row.family === 'sf')
    return new StockfishEngine(
      { threads: 1, hashMb: cpuEngineHashMbForSurface('arena'), skillLevel: 20 },
      stockfishFlavorUrl(row.variant === 'full' ? 'single' : 'lite-single'),
    );
  if (row.family === 'reckless') return createRecklessEngine(recklessVariantForKey(row.variant), undefined, ARENA_CPU_ENGINE_OPTIONS);
  if (row.family === 'viridithas') return createViridithasEngine(viridithasVariantForKey(row.variant), {}, ARENA_CPU_ENGINE_OPTIONS);
  if (row.family === 'berserk') return createBerserkEngine(berserkVariantForKey(row.variant), ARENA_CPU_ENGINE_OPTIONS);
  if (row.family === 'plentychess') return createPlentyChessEngine(plentyChessVariantForKey(row.variant), ARENA_CPU_ENGINE_OPTIONS);
  if (row.family === 'stormphrax') return createStormphraxEngine(stormphraxVariantForKey(row.variant), ARENA_CPU_ENGINE_OPTIONS);
  throw new Error(`Not a CPU arena seat: ${row.family}`);
}

async function runIsolatedCpuTournament(
  schedule: ScheduledGame<ArenaOpening>[],
  pendingIndices: number[],
  byPid: Map<string, { index: number; name: string }>,
  signal: AbortSignal,
  complete: (index: number, result: GameResultCode, reason: string, tree: GameTree, trail: GameTrail, diagnostics: { runtime: string; search: string }) => Promise<void>,
): Promise<void> {
  const concurrency = selectedArenaConcurrency();
  const broker = new EngineResourceBroker({ policy: 'bounded', maxConcurrentCpu: concurrency, dial: resourceBroker.getDial() });
  cpuTournamentBroker = broker;
  const rows = seatRows.map((row) => ({ ...row }));
  const timed = arenaBudgetMode() === 'movetime';
  const movetimeMs = arenaMovetimeMs();
  const delayMs = Math.max(0, Math.floor(Number(inputEl('delayInput').value) || 0));
  const live = new Map<number, { position: CpuArenaPosition; white: CpuArenaSeat; black: CpuArenaSeat; trail: GameTrail }>();
  let displayed = -1;
  const show = (index: number) => {
    if (signal.aborted || index !== Math.min(...live.keys())) return;
    const game = live.get(index)!;
    if (displayed !== index) {
      displayed = index;
      el('message').textContent = '';
      activeEngineIds = [game.white.id, game.black.id];
      setBoardSideEngines(game.white.id, game.white.name, game.black.id, game.black.name);
      if (!reviewing) resetGameCharts();
    }
    const position = game.position;
    historyBoards = position.positions;
    board = historyBoards[historyBoards.length - 1];
    lastUci = position.lastUci;
    liveTrail = game.trail;
    if (!reviewing) {
      gameChartSamples = game.trail.charts;
      renderGameCharts();
    }
    el('pairing').textContent =
      `Game ${index + 1} of ${schedule.length} · ${game.white.name === game.black.name ? game.white.name : `${displayEngineName(game.white.name)} vs ${displayEngineName(game.black.name)}`}${concurrency > 1 ? ` · ${live.size} running at once` : ''}`;
    renderBoard();
  };
  try {
    // Do not retain the old surface pools alongside the per-game workers.
    disposeStockfish();
    recklessEngines.disposeAll();
    viridithasEngines.disposeAll();
    berserkEngines.disposeAll();
    plentyChessEngines.disposeAll();
    stormphraxEngines.disposeAll();
    await runArenaSchedule({
      count: pendingIndices.length,
      concurrency,
      signal,
      play: async (pendingIndex, gameSignal) => {
        const index = pendingIndices[pendingIndex];
        const seats: CpuArenaSeat[] = [];
        const game = schedule[index];
        try {
          for (const [color, pid] of [
            ['w', game.whiteId],
            ['b', game.blackId],
          ] as const) {
            const participant = byPid.get(pid)!;
            const row = rows[participant.index];
            const engine = createIsolatedCpuEngine(row);
            isolatedCpuEngines.add(engine);
            const seat = { id: `arena:${index}:${color}:${pid}`, name: participant.name, engine };
            seats.push(seat);
            isolatedCpuSeats.set(seat.id, { seat, row });
            broker.register(seat.id, { resourceClass: 'cpu', maxThreads: 1 });
            engine.setOptions({ depth: timed ? undefined : row.strength, movetimeMs: timed ? movetimeMs : undefined, threads: 1 });
            engines.set(seat.id, { id: seat.id, name: seat.name, move: async () => null });
          }
          const [white, black] = seats;
          const tree = gameTreeFromOpening(game.opening);
          const trail = trailFromTree(tree, `${white.name} vs ${black.name}`, openingHistoryBoards(game.opening).length - 1);
          const outcome = await playCpuArenaGame({
            tree,
            white,
            black,
            broker,
            signal: gameSignal,
            beforeMove: (seat) => {
              if (!signal.aborted && displayed === index) recordEngineThinking(seat);
            },
            afterMove: (seat, fen, move, elapsedMs) => {
              if (signal.aborted) return;
              const lines = seat.engine.lastInfo();
              const ply = live.get(index)!.position.ply;
              recordUciOutput(seat.id, seat.name, seat.name, fen, move, lines, elapsedMs, displayed === index);
              recordGameChartSample(ply, seat, elapsedMs, trail.charts, displayed === index, arenaUciChartSample(ply, seat, fen, lines, elapsedMs));
            },
            position: (position) => {
              if (signal.aborted) return;
              appendCpuArenaTrail(trail, position);
              live.set(index, { position, white, black, trail });
              show(index);
            },
            delay: () => sleep(delayMs, gameSignal),
          });
          return { ...outcome, trail, diagnostics: captureCpuArenaDiagnostics(seats, cpuEngineHashMbForSurface('arena'), searchTelemetryText) };
        } finally {
          for (const seat of seats) {
            seat.engine.dispose();
            isolatedCpuEngines.delete(seat.engine);
            isolatedCpuSeats.delete(seat.id);
            broker.unregister(seat.id);
            engines.delete(seat.id);
            thinkingEngineIds.delete(seat.id);
            arenaSearchProgress.delete(seat.id);
          }
          live.delete(index);
          if (live.size) show(Math.min(...live.keys()));
        }
      },
      complete: (pendingIndex, outcome) => complete(pendingIndices[pendingIndex], outcome.result, outcome.reason, outcome.tree, outcome.trail, outcome.diagnostics),
    });
  } finally {
    if (cpuTournamentBroker === broker) cpuTournamentBroker = null;
  }
}

function renderCheckpointStatus(): void {
  const host = document.getElementById('checkpointSection');
  if (!host) return;
  // Routine storage status belongs with exports, not above the live scoresheet.
  host.hidden = running ? !checkpointProblem : !checkpoint && !checkpointProblem;
  const storage = el('checkpointStorageText');
  storage.hidden = !checkpoint || !!checkpointProblem;
  storage.textContent = checkpointSaving ? 'Saving completed games on this device…' : 'Completed games are saved on this device.';
  const pending = checkpoint ? arenaRemainingIndices(checkpoint).length : 0;
  el('checkpointText').textContent = checkpointProblem || (checkpointSaving ? 'Saving completed games on this device…' : running
    ? 'Completed games are saved on this device.'
    : checkpoint ? pending ? 'Saved on this device. Unfinished games restart from their opening.' : 'Finished results are saved on this device until you start a new run.' : '');
  const resume = el('resumeTournament') as HTMLButtonElement;
  resume.hidden = !checkpoint || !pending || running;
  resume.disabled = startPending || checkpointSaving || checkpointUnreadable;
  resume.textContent = checkpoint ? `Resume tournament (${checkpoint.completed.length} of ${checkpoint.schedule.length} games played)` : 'Resume tournament';
  el('discardTournament').hidden = running || (!checkpoint && !checkpointProblem);
  (el('discardTournament') as HTMLButtonElement).disabled = startPending || checkpointSaving;
  el('checkpointRoster').textContent = checkpoint && !running ? checkpoint.config.seats.map((seat) => displayEngineName(seat.name)).join(' vs ') : '';
}

async function checkpointCompatibility(): Promise<string> {
  // Metadata only; never fetch engine binaries here. The hashed controller URL also
  // prevents combining results from different app/search builds after an update.
  const channelUrl = resolvePublicAssetUrl('/models/arena-channel-probe').replace('/models/arena-channel-probe', '/channels/stable.json');
  const channel = await fetch(channelUrl, { cache: 'no-cache', signal: AbortSignal.timeout(5000) })
    .then((response) => response.ok ? response.json() : artifactChannel).catch(() => artifactChannel);
  return JSON.stringify({ bundle: import.meta.url, release: channel.releaseId, query: location.search,
    model: MODEL_URL, pack: PACK_URL, centipawn: CENTIPAWN_MODEL_URL, centipawnRuntime: CENTIPAWN_HYBRID_MANIFEST_URL,
    whole: LC0_WHOLE_MODEL_MANIFEST_URL, dial: resourceBroker.getDial() });
}

function captureCheckpointConfig(participants: { pid: string; index: number; name: string }[], compatibility: string): ArenaCheckpointConfig {
  const controls = normalizeArenaControls(Object.fromEntries(ARENA_CONTROL_IDS.map((id) => [id, (el(id) as HTMLInputElement).value])) as ArenaCheckpointConfig['controls']);
  return { mode: arenaTournamentMode(), seats: participants.map((participant) => ({
    id: participant.pid, name: participant.name, ...seatRows[participant.index],
  })), controls, compatibility };
}

function applyCheckpointConfig(saved: ArenaCheckpoint): void {
  for (const id of ARENA_CONTROL_IDS) {
    const control = el(id) as HTMLInputElement | HTMLSelectElement;
    control.value = saved.config.controls[id];
    if (control.value !== saved.config.controls[id]) throw new Error(`Saved setting ${id} is no longer supported. Export the completed PGNs and start a new run.`);
  }
  seatRows.splice(0, seatRows.length, ...saved.config.seats.map((seat) => ({ family: seat.family as EngineFamily, variant: seat.variant, strength: seat.strength })));
  renderSeatSelectors();
  if (seatRows.some((row, i) => row.variant !== saved.config.seats[i].variant)) throw new Error('A saved engine version is unavailable. Completed results are still viewable and exportable.');
  refreshBudgetControls();
  refreshSeatControls();
}

function showCheckpointResults(saved: ArenaCheckpoint): void {
  resultsSchedule = hydrateArenaSchedule(saved);
  resultsNames = new Map(saved.config.seats.map((seat) => [seat.id, seat.name]));
  scheduledGameCount = saved.schedule.length;
  games.length = 0;
  finishedTrails.length = 0;
  gameResultChips.length = 0;
  el('log').innerHTML = '';
  const standings = new TournamentStandings(saved.config.seats);
  for (const game of saved.completed) {
    const scheduled = saved.schedule[game.scheduleIndex];
    standings.record(scheduled.whiteId, scheduled.blackId, game.result);
    games.push({ pgn: game.pgn, scheduleIndex: game.scheduleIndex, result: game.result });
    const whiteName = resultsNames.get(scheduled.whiteId)!;
    const blackName = resultsNames.get(scheduled.blackId)!;
    const trail = trailFromTree(arenaCompletedTree(game), `Game ${game.scheduleIndex + 1}: ${whiteName} vs ${blackName} (${game.result})`, scheduled.opening.moves?.length ?? 0);
    trail.charts = game.charts;
    trail.rootChart = game.rootChart;
    trail.sides = { whiteName, blackName };
    finishedTrails[game.scheduleIndex] = trail;
    gameResultChips.push({ index: game.scheduleIndex, result: game.result, whiteName, blackName,
      aPoints: saved.config.mode === 'match' ? game.result === '1/2-1/2' ? 0.5 : (scheduled.whiteId === '0') === (game.result === '1-0') ? 1 : 0 : null });
    appendLog(`${game.scheduleIndex + 1}. ${whiteName} vs ${blackName} [${scheduled.opening.name}]: ${game.result} (${game.reason}) · ${game.diagnostics.runtime} · ${game.diagnostics.search}`, game.scheduleIndex);
  }
  if (saved.config.mode === 'match') {
    const a = standings.table().find((row) => row.id === '0')!;
    const b = standings.table().find((row) => row.id === '1')!;
    renderMatchScore(a.name, b.name, a.name === b.name, { a: a.points, b: b.points, aWins: a.wins, bWins: b.wins, draws: a.draws, games: a.games });
    el('crosstableSection').hidden = true;
  } else renderStandings(standings, saved.schedule.length);
  renderOpeningPairs();
  el('pairing').textContent = `${saved.completed.length} of ${saved.schedule.length} games played`;
}

function queueCheckpointSave(saved: ArenaCheckpoint, mountSignal: AbortSignal): void {
  if (checkpointProblem) return;
  pendingCheckpointSaves++;
  checkpointSaving = true;
  renderCheckpointStatus();
  checkpointWrites = checkpointWrites.then(async () => {
    if (checkpointProblem) return;
    await saveArenaCheckpoint(saved, { runId: saved.runId, revision: saved.revision - 1 });
    checkpointToken = { runId: saved.runId, revision: saved.revision };
  }).catch((error) => {
    checkpointProblem = `Not saved: ${error instanceof Error ? error.message : String(error)} Export PGN before leaving.`;
    if (error instanceof ArenaCheckpointConflictError) abort?.abort();
  }).finally(() => {
    pendingCheckpointSaves--;
    if (!isStaleMount(mountSignal)) {
      checkpointSaving = pendingCheckpointSaves > 0;
      renderCheckpointStatus();
    }
  });
}

async function startMatchUnchecked(minimumWarmupVisibleMs = 400, resume: ArenaCheckpoint | null = null): Promise<void> {
  if (running || startPending) return;
  const mountSignal = mountAbort.signal;
  startPending = true;
  el('start').toggleAttribute('disabled', true);
  syncArenaPhase();
  await checkpointWrites;
  if (isStaleMount(mountSignal)) return;
  if (checkpointUnreadable) throw new Error('Discard the unreadable saved tournament before starting a new run.');
  const compatibility = await checkpointCompatibility();
  if (isStaleMount(mountSignal)) return;
  if (resume) {
    if (resume.config.compatibility !== compatibility) throw new Error('The app build, engine assets or runtime overrides changed. Completed results are still viewable and exportable, but cannot be mixed with a new build.');
    applyCheckpointConfig(resume);
    // UI runtime changes while paused may have created a different evaluator.
    disposeLc0Resources();
  }
  const effectiveControls = normalizeArenaControls(Object.fromEntries(ARENA_CONTROL_IDS.map((id) => [id, (el(id) as HTMLInputElement).value])) as ArenaCheckpointConfig['controls']);
  for (const id of ARENA_CONTROL_IDS) (el(id) as HTMLInputElement).value = effectiveControls[id];
  syncSeatRowsFromDom();
  const selectedAssetsAvailable = await ensureSelectedRecklessAssetsAvailable();
  if (isStaleMount(mountSignal)) return;
  if (!selectedAssetsAvailable) {
    clearStartPending();
    return;
  }
  if (running) {
    clearStartPending();
    return;
  }
  buildEngines();
  populateSeats();
  if (resume && seatRows.some((row, i) => row.family !== resume.config.seats[i]?.family || row.variant !== resume.config.seats[i]?.variant || row.strength !== resume.config.seats[i]?.strength)) {
    throw new Error('The saved engine configuration is no longer supported. Export the completed PGNs before starting a new run.');
  }
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
    const name = engine && (idCounts.get(engineId) ?? 0) > 1 && mode !== 'match' ? `${engine.name} (seat ${index + 1})` : (engine?.name ?? engineId);
    return { pid: String(index), index, engineId, engine, name };
  });
  if (participants.length < 2 || participants.some((participant) => !participant.engine)) {
    el('message').textContent = 'Pick at least two engines.';
    clearStartPending();
    return;
  }
  const byPid = new Map(participants.map((participant) => [participant.pid, participant]));
  const bigNetRows = activeSeatRows().filter((row) => row.family === 'lc0' && isLc0BigNetVariant(row.variant));
  const bigNetRuntimeSupported = !bigNetRows.length || (await probeBt4Support());
  if (isStaleMount(mountSignal)) return;
  if (!bigNetRuntimeSupported) {
    el('message').textContent = 'Lc0 big nets need WebGPU, which is unavailable in this browser.';
    clearStartPending();
    return;
  }
  const checkedBigNetVariants = new Set<string>();
  for (const row of bigNetRows) {
    if (!isLc0BigNetVariant(row.variant) || checkedBigNetVariants.has(row.variant)) continue;
    checkedBigNetVariants.add(row.variant);
    const { config } = bigNetFor(row.variant);
    const status = await checkBigNetAsset(config, whileArenaMounted(renderSeatSelectors));
    if (isStaleMount(mountSignal)) return;
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
  const gamesPerOpening = arenaGamesPerOpening();
  const standings = new TournamentStandings(participants.map((participant) => ({ id: participant.pid, name: participant.name })));
  let schedule: ScheduledGame<ArenaOpening>[];
  try {
    // Colors alternate per game index within each pairing (see tournament.ts).
    schedule = resume ? hydrateArenaSchedule(resume) : buildSchedule(
      tournamentPairings(
        mode,
        participants.map((participant) => participant.pid),
      ),
      selectedOpenings(),
      gamesPerOpening,
    );
  } catch (error) {
    el('message').textContent = `Opening setup error: ${(error as Error).message}`;
    clearStartPending();
    return;
  }
  if (!schedule.length) {
    el('message').textContent = 'No games to play.';
    clearStartPending();
    return;
  }

  const runCheckpoint = resume ? claimArenaCheckpoint(resume) : createArenaCheckpoint(captureCheckpointConfig(participants, compatibility), schedule);
  // Establish the slot before any games run. A stale tab must not overwrite it.
  try {
    await saveArenaCheckpoint(runCheckpoint, checkpointToken);
    checkpointToken = { runId: runCheckpoint.runId, revision: runCheckpoint.revision };
    checkpointProblem = '';
  } catch (error) {
    if (error instanceof ArenaCheckpointConflictError) throw error;
    checkpointProblem = `Not saved: ${error instanceof Error ? error.message : String(error)} Keep this tab open and export PGN before leaving.`;
  }
  if (isStaleMount(mountSignal)) return;
  checkpoint = runCheckpoint;
  const pendingIndices = arenaRemainingIndices(runCheckpoint);
  startPending = false;
  running = true;
  arenaMatchState = 'live';
  scheduledGameCount = schedule.length;
  resultsSchedule = schedule;
  resultsNames = new Map(participants.map((participant) => [participant.pid, participant.name]));
  el('crosstableSection').hidden = mode === 'match';
  gameResultChips.length = 0;
  el('pairing').textContent = `Game 1 of ${schedule.length}`;
  const matchAbort = new AbortController();
  abort = matchAbort;
  refreshOpeningPreview();
  refreshStockfishControls();
  refreshRecklessVariantUi();
  refreshViridithasVariantUi();
  refreshBerserkVariantUi();
  refreshPlentyChessVariantUi();
  refreshStormphraxVariantUi();
  refreshSeatControls();
  games.length = 0;
  inputEl('pgnOut').value = '';
  inputEl('pgnOut').hidden = true;
  (el('resultSection') as HTMLDetailsElement).open = false;
  liveTrail = null;
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
  if (resume) {
    showCheckpointResults(resume);
    for (const game of resume.completed) {
      const scheduled = schedule[game.scheduleIndex];
      standings.record(scheduled.whiteId, scheduled.blackId, game.result);
    }
    const first = standings.table().find((row) => row.id === participants[0].pid)!;
    const second = standings.table().find((row) => row.id === participants[1].pid)!;
    Object.assign(score, { a: first.points, b: second.points, aWins: first.wins, bWins: second.wins, draws: first.draws, games: resume.completed.length });
  } else {
    if (mode === 'match') renderMatchScore(engineA.name, engineB.name, sameEngine, score);
    else renderStandings(standings, schedule.length);
    renderOpeningPairs();
  }
  renderCheckpointStatus();
  const completeGame = async (
    i: number,
    result: GameResultCode,
    reason: string,
    tree: GameTree,
    trail: GameTrail | null,
    diagnostics?: { runtime: string; search: string },
  ) => {
    if (isStaleMount(mountSignal) || matchAbort.signal.aborted || checkpoint!.completed.some((game) => game.scheduleIndex === i)) return;
    const { whiteId, blackId, opening } = schedule[i];
    const white = byPid.get(whiteId)!;
    const black = byPid.get(blackId)!;
    score.games += 1;
    standings.record(whiteId, blackId, result);
    if (result === '1/2-1/2') {
      score.draws += 1;
      score.a += 0.5;
      score.b += 0.5;
    } else {
      const winnerIsA = (whiteId === participants[0].pid) === (result === '1-0');
      if (winnerIsA) {
        score.a += 1;
        score.aWins += 1;
      } else {
        score.b += 1;
        score.bWins += 1;
      }
    }
    const tags: Record<string, string> = { Event: 'LC0 arena', White: white.name, Black: black.name, Opening: opening.name, ...openingPgnSetupTags(opening) };
    const pgn = gameTreeToPgn(tree, tags, result);
    const completed: ArenaCompletedGame = {
      pgn, scheduleIndex: i, result, reason, rootFen: tree.root.fen,
      moves: tree.mainlineFrom(tree.root).map((node) => moveToUci(node.move!)),
      charts: (trail?.charts ?? []).map((sample) => ({ ...sample })),
      ...(trail?.rootChart ? { rootChart: structuredClone(trail.rootChart) } : {}),
      diagnostics: diagnostics ?? { runtime: engineRuntimeDiagnosticsText(), search: searchTelemetryText() },
    };
    checkpoint = commitArenaGame(checkpoint!, completed);
    queueCheckpointSave(checkpoint, mountSignal);
    // Do not show a committed score or advance this lane until its save settles.
    await checkpointWrites;
    if (isStaleMount(mountSignal)) return;
    games.push({ pgn, scheduleIndex: i, result });
    games.sort((a, b) => a.scheduleIndex - b.scheduleIndex);
    if (trail) {
      trail.label = `Game ${i + 1}: ${white.name} vs ${black.name} (${result})`;
      trail.sides = { whiteName: white.name, blackName: black.name };
      finishedTrails[i] = trail;
    }
    gameResultChips.push({
      index: i,
      result,
      whiteName: white.name,
      blackName: black.name,
      aPoints: mode === 'match' ? (result === '1/2-1/2' ? 0.5 : (whiteId === participants[0].pid) === (result === '1-0') ? 1 : 0) : null,
    });
    gameResultChips.sort((x, y) => x.index - y.index);
    renderCacheInfo();
    appendLog(
      `${i + 1}. ${white.name} vs ${black.name} [${opening.name}]: ${result} (${reason}) · ${diagnostics?.runtime ?? engineRuntimeDiagnosticsText()} · ${diagnostics?.search ?? searchTelemetryText()}`,
      i,
    );
    if (mode === 'match') renderMatchScore(engineA.name, engineB.name, sameEngine, score);
    else renderStandings(standings, schedule.length);
    renderOpeningPairs();
  };
  let enginesReady = false;
  try {
    if (isolatedCpuPool()) {
      enginesReady = true;
      el('message').textContent = 'Preparing private CPU workers…';
      await runIsolatedCpuTournament(schedule, pendingIndices, byPid, matchAbort.signal, completeGame);
    } else {
      if (selectedSeatsNeedLc0Evaluator() && !lc0Cache) {
        await loadLc0Evaluator(mountSignal, matchAbort.signal);
        if (!lc0Cache) throw new Error('The saved Lc0 network could not be loaded.');
      }
      resetLc0SearchTrees(seatIds);
      await warmUpSelectedEngines(seatIds, matchAbort.signal, minimumWarmupVisibleMs);
      enginesReady = true;
      if (isStaleMount(mountSignal) || matchAbort.signal.aborted) return;
      for (const i of pendingIndices) {
        if (matchAbort.signal.aborted) break;
        const { whiteId, blackId, opening } = schedule[i];
        const white = byPid.get(whiteId)!;
        const black = byPid.get(blackId)!;
        const whiteEngine = white.engine!;
        const blackEngine = black.engine!;
        // Reset trees per game (fresh game tree); within a game the shared tree is
        // reused across both sides' plies — i.e. self-play when both seats match.
        resetLc0SearchTrees(seatIds);
        for (const key of ARENA_BIG_NET_KEYS) {
          const searcher = peekBigNetSearcher(key);
          if (!searcher) continue;
          if (searcher.loaded) await searcher.resetTree();
        }
        // Reset only this game's CPU participants; mirror seats share a worker.
        const gameRows = [rowForEngineId(whiteEngine.id), rowForEngineId(blackEngine.id)];
        const gameCpuEngines = new Set<BrowserUciEngine>();
        for (const row of gameRows) {
          if (row?.family === 'viridithas') gameCpuEngines.add(getViridithasFor(row.variant));
          else if (row?.family === 'berserk') gameCpuEngines.add(getBerserkFor(row.variant));
          else if (row?.family === 'plentychess') gameCpuEngines.add(getPlentyChessFor(row.variant));
          else if (row?.family === 'stormphrax') gameCpuEngines.add(getStormphraxFor(row.variant));
        }
        for (const engine of gameCpuEngines) await engine.newGame(matchAbort.signal);
        if (isStaleMount(mountSignal) || matchAbort.signal.aborted) break;
        setBoardSideEngines(whiteEngine.id, whiteEngine.name, blackEngine.id, blackEngine.name);
        el('pairing').textContent = `Game ${i + 1} of ${schedule.length} · ${opening.name}`;
        el('message').textContent = '';
        const { result, reason, tree } = await playArenaGame(whiteEngine, blackEngine, opening, matchAbort.signal);
        if (reason === 'cancelled') break;
        await completeGame(i, result, reason, tree, liveTrail);
      }
    }
    if (!isStaleMount(mountSignal)) {
      const noun = mode === 'match' ? 'Match' : 'Tournament';
      el('message').textContent = matchAbort.signal.aborted ? `${noun} stopped.` : `${noun} finished.`;
      el('pairing').textContent = `${score.games} of ${schedule.length} game${schedule.length === 1 ? '' : 's'} played`;
      arenaMatchState = matchAbort.signal.aborted ? 'stopped' : 'final';
    }
  } catch (error) {
    if (!isStaleMount(mountSignal)) {
      if (isAbortError(error) || matchAbort.signal.aborted) {
        el('message').textContent = mode === 'match' ? 'Match stopped.' : 'Tournament stopped.';
        el('pairing').textContent = `${score.games} of ${schedule.length} game${schedule.length === 1 ? '' : 's'} played`;
        arenaMatchState = 'stopped';
      } else if (enginesReady) {
        el('message').textContent = `Match failed: ${error instanceof Error ? error.message : String(error)}`;
        arenaMatchState = 'failed';
      } else {
        arenaMatchState = 'failed';
        el('message').textContent = `Engine preparation failed: ${error instanceof Error ? error.message : String(error)}`;
        el('pairing').textContent = 'Match did not start.';
      }
    }
  } finally {
    await checkpointWrites;
    const ownsMatchState = abort === matchAbort;
    if (ownsMatchState) {
      running = false;
      startPending = false;
      abort = null;
      activeEngineIds = [];
      engineOutputs.clear();
      thinkingEngineIds.clear();
    }
    if (ownsMatchState && !isStaleMount(mountSignal)) {
      el('start').toggleAttribute('disabled', false);
      el('stop').toggleAttribute('disabled', true);
      refreshOpeningPreview();
      refreshStockfishControls();
      refreshRecklessVariantUi();
      refreshViridithasVariantUi();
      refreshBerserkVariantUi();
      refreshPlentyChessVariantUi();
      refreshStormphraxVariantUi();
      refreshSeatControls();
      renderEngineOutputs();
    }
  }
}
async function startMatch(minimumWarmupVisibleMs = 400, resume: ArenaCheckpoint | null = null): Promise<void> {
  const mountSignal = mountAbort.signal;
  try {
    await startMatchUnchecked(minimumWarmupVisibleMs, resume);
  } catch (error) {
    if (isStaleMount(mountSignal)) return;
    hideEngineWarmupStatus();
    clearStartPending();
    const detail = error instanceof Error ? error.message : String(error);
    el('message').textContent = `Match preparation failed: ${detail}`;
    if (!checkpoint) el('pairing').textContent = 'Match did not start.';
    renderCheckpointStatus();
  }
}

function exportPgn() {
  const output = inputEl('pgnOut');
  output.value = games.map((g) => g.pgn).join('\n\n');
  output.hidden = !games.length;
  el('message').textContent = games.length ? `Exported ${games.length} game(s) as PGN.` : 'No games to export yet.';
}

function disposeLc0Resources(): void {
  lc0LoadAbort?.abort();
  lc0LoadAbort = null;
  loadingLc0 = false;
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
  arenaSearchProgress.clear();
  hideDownloadProgress();
  disposeLc0Resources();
  centipawnEvaluatorGeneration++;
  for (const evaluator of centipawnEvaluators) destroyCentipawnEvaluator(evaluator);
  centipawnEvaluators.clear();
  centipawnEvaluatorsByKey.clear();
  centipawnEvaluatorPromises.clear();
  centipawnEvaluatorLoadTokens.clear();
  disposeStockfish();
  for (const engine of isolatedCpuEngines) engine.dispose();
  isolatedCpuEngines.clear();
  isolatedCpuSeats.clear();
  recklessEngines.disposeAll();
  viridithasEngines.disposeAll();
  berserkEngines.disposeAll();
  plentyChessEngines.disposeAll();
  stormphraxEngines.disposeAll();
  for (const key of ARENA_BIG_NET_KEYS) {
    peekBigNetSearcher(key)?.cancel();
    releaseBigNetSearcher(key);
  }
}

async function createSelectedLc0Evaluator(signal: AbortSignal): Promise<Lc0OnnxEvaluator | Lc0WebHybridEvaluator | Lc0WholeOnnxWebgpuEvaluator> {
  const runtime = selectedLc0Runtime();
  if (runtime === 'onnx') {
    const modelLoad = await loadLc0ModelForOrt(MODEL_URL, {
      cache: true,
      signal,
      onProgress: (loaded, total) => { if (!signal.aborted) showDownloadProgress('Lc0 small net', loaded, total, 'Downloading'); },
    });
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

async function loadLc0Evaluator(mountSignal: AbortSignal = mountAbort.signal, matchSignal?: AbortSignal): Promise<void> {
  if (isStaleMount(mountSignal) || matchSignal?.aborted) return;
  lc0LoadAbort?.abort();
  const loadAbort = new AbortController();
  lc0LoadAbort = loadAbort;
  const signal = AbortSignal.any([mountSignal, loadAbort.signal, ...(matchSignal ? [matchSignal] : [])]);
  const runtime = selectedLc0Runtime();
  if (!selectedSeatsNeedLc0Evaluator()) {
    lc0LoadAbort = null;
    loadingLc0 = false;
    el('start').toggleAttribute('disabled', false);
    refreshSeatControls();
    el('message').textContent = readyMessage();
    return;
  }
  loadingLc0 = true;
  el('start').toggleAttribute('disabled', true);
  refreshSeatControls();
  const hybrid = lc0HybridConfigLabel(runtime);
  el('message').textContent = 'Loading the Lc0 network…';
  showDownloadProgress(`LC0 ${lc0RuntimeLabel(runtime)}`, undefined, undefined, 'Preparing');
  try {
    const evaluator = await loadArenaResource(() => createSelectedLc0Evaluator(signal), signal);
    const ortDiagnostics = runtime === 'onnx' ? await collectOrtRuntimeDiagnostics({ probeAdapter: false }) : undefined;
    if (isStaleMount(mountSignal) || signal.aborted) {
      void evaluator.dispose?.();
      return;
    }
    publishBrowserRuntimeAudit({
      source: 'lc0-arena-evaluator',
      surface: 'arena',
      family: 'lc0',
      engineLabel: 'LC0',
      modelId: 'lc0-default',
      modelUrl: runtime === 'onnx' ? MODEL_URL : runtime === LC0_WHOLE_MODEL_WEBGPU_RUNTIME ? LC0_WHOLE_MODEL_MANIFEST_URL : PACK_URL,
      requestedRuntime: runtime,
      resolvedRuntime: ortDiagnostics?.fallback ? 'ort-main-thread-wasm-fallback' : lc0ResolvedRuntime(runtime),
      fallbackReason: ortDiagnostics?.fallback?.reason,
      runtimeConfigId: runtime === 'onnx' ? undefined : runtime,
      manifestUrl: runtime === 'onnx' ? undefined : runtime === LC0_WHOLE_MODEL_WEBGPU_RUNTIME ? LC0_WHOLE_MODEL_MANIFEST_URL : PACK_URL,
      searchBudget: activeSeatRows()
        .filter((row) => row.family === 'lc0' && !isLc0BigNetVariant(row.variant))
        .map((row) => budgetText(row))
        .join(', '),
      notes: [
        lc0RuntimeLabel(runtime),
        hybrid,
        runtime === 'onnx'
          ? undefined
          : runtime === LC0_WHOLE_MODEL_WEBGPU_RUNTIME
            ? 'whole-model runtime is research-only and opt-in'
            : 'hybrid runtime is pack-lazy until first evaluation succeeds',
      ].filter((part): part is string => !!part),
    });
    lc0Cache = new CachedLc0Evaluator(evaluator, { maxEntries: arenaCacheEntries() });
    lc0Searchers.clear();
    player = new Lc0PolicyOnlyPlayer(lc0Cache);
    searcher = new Lc0PuctSearcher(lc0Cache);
    renderCacheInfo();
    hideDownloadProgress();
    el('start').toggleAttribute('disabled', false);
    el('message').textContent = readyMessage();
  } catch (error) {
    if (isStaleMount(mountSignal) || signal.aborted) return;
    hideDownloadProgress();
    if (!selectedSeatsNeedLc0Evaluator()) {
      el('start').toggleAttribute('disabled', false);
      el('message').textContent = `LC0 ${lc0RuntimeLabel(runtime)} load failed, but current non-LC0 matchup is ready: ${(error as Error).message}`;
    } else {
      el('message').textContent = `LC0 ${lc0RuntimeLabel(runtime)} load failed: ${(error as Error).message}`;
    }
  } finally {
    if (!isStaleMount(mountSignal) && lc0LoadAbort === loadAbort) {
      lc0LoadAbort = null;
      loadingLc0 = false;
      hideDownloadProgress();
      refreshSeatControls();
    }
  }
}

async function reloadLc0Evaluator(): Promise<void> {
  abort?.abort();
  disposeLc0Resources();
  reconcileActiveRuntimeFallbackWarning();
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
  try {
    return await collectOrtRuntimeDiagnostics({ probeAdapter: false });
  } catch (error) {
    return { error: (error as Error).message };
  }
}

function fixedSuiteFensFromParams(): string[] {
  const raw = params.get('fixedSuiteFens') ?? params.get('fens') ?? '';
  return raw
    .split(/[|\n]/)
    .map((fen) => fen.trim())
    .filter(Boolean);
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
    for (const fen of fens) parseValidatedFen(fen);
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
    if (!lc0Cache) await loadLc0Evaluator(mountAbort.signal, controller.signal);
    if (!lc0Cache) throw new Error('LC0 evaluator could not be initialized for the fixed suite.');
    await warmUpSelectedEngines([lc0Id, sfId], controller.signal, 0);
    const sfEngine = stockfishEngineFor(sfKind);
    const timed = arenaBudgetMode() === 'movetime';
    const scoreMs = stockfishScoreMs();
    const scoreDepth = stockfishScoreDepth();
    sfEngine.setOptions({
      depth: scoreDepth ?? (timed ? undefined : seatRows[1].strength),
      movetimeMs: scoreDepth === undefined && timed ? scoreMs : undefined,
      threads: stockfishThreadsPlanned(),
    });
    resetLc0SearchTrees([lc0Id]);
    const positions = [];
    for (let i = 0; i < fens.length; i++) {
      const boardAtMove = parseFen(fens[i]);
      const searchStarted = performance.now();
      const search = await lc0SearcherFor(lc0Id).search(
        { positions: [boardAtMove] },
        {
          visits: timed ? undefined : lc0Row.strength,
          movetimeMs: timed ? arenaMovetimeMs() : undefined,
          signal: controller.signal,
          yieldEveryMs: 16,
          reuseTree: false,
          batchSize: lc0BatchSize(),
          batchPipelineDepth: lc0BatchPipelineDepth(),
        },
      );
      const searchElapsedMs = performance.now() - searchStarted;
      recordLc0SearchTelemetry(lc0Id, lc0Name, search, searchElapsedMs);
      recordLc0SearchOutput(lc0Id, lc0Name, search, searchElapsedMs);
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
          lc0PerspectiveCp: whiteCp === undefined ? undefined : boardAtMove.turn === 'w' ? whiteCp : -whiteCp,
          lc0PerspectiveMate: mateInWhitePov === undefined ? undefined : boardAtMove.turn === 'w' ? mateInWhitePov : -mateInWhitePov,
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
    setBenchResult({
      status: 'LC0_FIXED_SUITE_FAILED',
      runtime: selectedLc0Runtime(),
      elapsedMs: Math.round(performance.now() - started),
      error: (error as Error).message,
      stack: (error as Error).stack,
    });
  }
}

async function runArenaBenchAutorun(): Promise<void> {
  if (params.get('arenaBench') !== '1' && params.get('benchmark') !== 'arena') return;
  setBenchResult({ status: 'ARENA_BENCH_RUNNING', runtime: selectedLc0Runtime(), startedAt: new Date().toISOString() });
  const started = performance.now();
  try {
    if ((el('start') as HTMLButtonElement).disabled) throw new Error(el('message').textContent || 'arena start is disabled');
    await startMatch(0);
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
        arenaConcurrency: selectedArenaConcurrency(),
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
    setBenchResult({
      status: 'ARENA_BENCH_FAILED',
      runtime: selectedLc0Runtime(),
      elapsedMs: Math.round(performance.now() - started),
      error: (error as Error).message,
      stack: (error as Error).stack,
    });
  }
}

function wireEvents() {
  el('start').addEventListener('click', () => {
    void startMatch();
  });
  el('resumeTournament').addEventListener('click', () => {
    if (checkpoint) void startMatch(400, checkpoint);
  });
  el('discardTournament').addEventListener('click', () => {
    void (async () => {
      if (running || startPending) return;
      await checkpointWrites;
      try {
        await discardArenaCheckpoint(checkpointToken);
        checkpoint = null;
        checkpointToken = null;
        checkpointProblem = '';
        checkpointUnreadable = false;
        games.length = 0;
        finishedTrails.length = 0;
        gameResultChips.length = 0;
        resultsSchedule = [];
        arenaMatchState = 'none';
        liveTrail = null;
        exitReview();
        resetGameCharts();
        el('chartsPanel').hidden = true;
        el('crosstableSection').hidden = true;
        el('openingPairsSection').hidden = true;
        el('message').textContent = 'Saved tournament discarded. Set up a new run.';
        refreshOpeningPreview();
        refreshSeatControls();
        el('start').toggleAttribute('disabled', false);
      } catch (error) {
        checkpointProblem = `Could not discard: ${error instanceof Error ? error.message : String(error)}`;
        renderCheckpointStatus();
      }
    })();
  });
  el('stop').addEventListener('click', () => {
    abort?.abort();
    el('message').textContent = 'Stopping…';
  });
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
      const trail = reviewing && reviewTrail ? reviewTrail : liveTrail;
      if (!trail) return;
      const ply = chartPlyFromClick(id, event.target as HTMLElement, event as MouseEvent);
      if (ply === null) return;
      enterReview(trail, trail.openingPlies + ply);
    });
  }
  el('rootChart').addEventListener('click', (event) => reviewRootChartClick(event as MouseEvent));
  for (const button of el('chartsPanel').querySelectorAll<HTMLButtonElement>('button[data-chart]')) {
    button.addEventListener('click', () => selectChart(button.dataset.chart!));
  }
  el('rootCandidate').addEventListener('change', () => {
    const value = selectEl('rootCandidate').value;
    if (value !== '') previewRootCandidate(Number(value));
  });
  el('chartPly').addEventListener('input', () => {
    const trail = reviewing && reviewTrail ? reviewTrail : liveTrail;
    if (trail) enterReview(trail, Number(inputEl('chartPly').value));
  });
  el('chartLive').addEventListener('click', () => {
    if (running) exitReview();
    else if (liveTrail) enterReview(liveTrail, liveTrail.entries.length - 1);
  });
  el('gameMoves').addEventListener('click', (event) => {
    const span = (event.target as HTMLElement).closest('.san[data-idx]') as HTMLElement | null;
    if (!span) return;
    const trail = reviewing && reviewTrail ? reviewTrail : liveTrail;
    if (trail) enterReview(trail, Number(span.dataset.idx));
  });
  el('gameChips').addEventListener('click', (event) => {
    const chip = (event.target as HTMLElement).closest('[data-game]') as HTMLElement | null;
    const trail = chip ? finishedTrails[Number(chip.dataset.game)] : undefined;
    if (trail) enterReview(trail, trail.entries.length - 1);
  });
  el('pairMatchup').addEventListener('change', renderOpeningPairs);
  el('openingPairs').addEventListener('click', (event) => {
    const button = (event.target as HTMLElement).closest<HTMLElement>('[data-game]');
    const trail = button ? finishedTrails[Number(button.dataset.game)] : undefined;
    if (trail) enterReview(trail, trail.entries.length - 1);
  });
  el('editSetup').addEventListener('click', () => {
    const main = document.getElementById('main');
    if (!main) return;
    main.dataset.config = main.dataset.config === 'open' ? 'closed' : 'open';
    syncArenaPhase();
  });
  arenaChartResizeObserver?.disconnect();
  let previousChartWidth = 0;
  arenaChartResizeObserver = new ResizeObserver(([entry]) => {
    const width = Math.round(entry.contentRect.width);
    if (width <= 0 || width === previousChartWidth) return;
    previousChartWidth = width;
    if (gameChartSamples.length) renderGameCharts();
  });
  arenaChartResizeObserver.observe(el('chartsPanel'));
  if (arenaClockTimer) clearInterval(arenaClockTimer);
  arenaClockTimer = setInterval(() => {
    if (running && thinkingSince.size) renderStrips();
  }, 100);
  el('log').addEventListener('click', (event) => {
    const row = (event.target as HTMLElement).closest('[data-game]') as HTMLElement | null;
    const trail = row ? finishedTrails[Number(row.dataset.game)] : undefined;
    if (trail) enterReview(trail, trail.entries.length - 1);
  });
  if (arenaKeydownHandler) document.removeEventListener('keydown', arenaKeydownHandler);
  arenaKeydownHandler = (event: KeyboardEvent) => {
    if (!reviewing && !(event.key === 'ArrowLeft' && liveTrail && liveTrail.entries.length > 1)) return;
    const target = event.target as HTMLElement | null;
    if (target?.closest?.('input,textarea,select,.eng-line,.arena-table-scroll')) return;
    if (event.key === 'ArrowLeft') {
      event.preventDefault();
      stepReview(-1);
    } else if (event.key === 'ArrowRight') {
      event.preventDefault();
      stepReview(1);
    } else if (event.key === 'Escape') exitReview();
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
      if (
        row.family === 'lc0' &&
        isLc0BigNetVariant(target.value) &&
        !window.confirm(`${bigNetLoadWarning(BIG_NETS[target.value])}\n\nUse Lc0 ${BIG_NETS[target.value].name}?`)
      ) {
        target.value = row.variant;
        return;
      }
      row.variant = target.value;
    } else if (target.classList.contains('seat-strength')) {
      row.strength = Number(target.value);
      clampStrength(row);
    }
    normalizeSeatRowForDeploy(seat);
    reconcileActiveRuntimeFallbackWarning();
    persistArenaSeatRows();
    for (const key of ARENA_BIG_NET_KEYS) {
      if (activeSeatRows().some((r) => r.family === 'lc0' && r.variant === key)) continue;
      peekBigNetSearcher(key)?.cancel();
      releaseBigNetSearcher(key);
    }
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
      persistArenaSeatRows();
      renderRecklessRuntimeInfo();
      renderViridithasRuntimeInfo();
      renderBerserkRuntimeInfo();
      renderPlentyChessRuntimeInfo();
      renderStormphraxRuntimeInfo();
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
    persistArenaSeatRows();
    reconcileActiveRuntimeFallbackWarning();
    buildEngines();
    populateSeats();
  });
  el('addSeat').addEventListener('click', () => {
    if (running || seatRows.length >= 8) return;
    syncSeatRowsFromDom();
    seatRows.push({ family: 'sf', variant: 'lite', strength: 8 });
    persistArenaSeatRows();
    buildEngines();
    populateSeats();
  });
  el('tournamentModeSelect').addEventListener('change', () => {
    if (running) return;
    if (isReadyMessage(el('message').textContent)) el('message').textContent = readyMessage();
    syncSeatRowsFromDom();
    buildEngines();
    populateSeats();
    refreshSeatControls();
    renderRecklessRuntimeInfo();
    renderViridithasRuntimeInfo();
    renderBerserkRuntimeInfo();
    renderPlentyChessRuntimeInfo();
    renderStormphraxRuntimeInfo();
  });
  el('arenaConcurrencySelect').addEventListener('change', refreshArenaConcurrencyNote);
  el('startingPositionSelect').addEventListener('change', refreshOpeningPreview);
  el('openingText').addEventListener('input', refreshOpeningPreview);
  el('cacheEntriesInput').addEventListener('input', () => {
    renderCacheInfo();
    resetLc0SearchTrees();
  });
  el('lc0PresetSelect').addEventListener('change', () => applyLc0Preset(selectedLc0Preset(), { reload: true }));
  el('lc0BatchSizeInput').addEventListener('input', () => {
    markLc0PresetCustom();
    inputEl('lc0BatchSizeInput').value = String(lc0BatchSize());
    renderCacheInfo();
    resetLc0SearchTrees();
  });
  el('lc0BatchPipelineDepthInput').addEventListener('input', () => {
    markLc0PresetCustom();
    inputEl('lc0BatchPipelineDepthInput').value = String(lc0BatchPipelineDepth());
    renderCacheInfo();
    resetLc0SearchTrees();
  });
  el('lc0RuntimeSelect').addEventListener('change', () => {
    markLc0PresetCustom();
    refreshSeatControls();
    if (!running) void reloadLc0Evaluator();
  });
  for (const id of ['lc0InputBackendSelect', 'lc0EncoderKernelSelect', 'lc0LegalPriorsSelect']) {
    el(id).addEventListener('change', () => {
      markLc0PresetCustom();
      renderCacheInfo();
      if (!running && selectedLc0Runtime() !== 'onnx') void reloadLc0Evaluator();
    });
  }
  el('budgetModeSelect').addEventListener('change', () => {
    refreshBudgetControls();
    refreshSeatControls();
    resetLc0SearchTrees();
  });
  el('movetimeInput').addEventListener('input', () => resetLc0SearchTrees());
  for (const id of ['movetimeInput', 'gamesInput', 'budgetModeSelect']) el(id).addEventListener('input', renderSetupSummary);
  el('stockfishThreadsInput').addEventListener('input', () => {
    if (running) return;
    inputEl('stockfishThreadsInput').value = String(stockfishThreadsCap());
    // Threads flips single<->threaded flavor (different wasm); rebuild on next use.
    disposeStockfish();
    refreshArenaConcurrencyNote();
  });
  if (arenaPagehideHandler) window.removeEventListener('pagehide', arenaPagehideHandler);
  arenaPagehideHandler = (event: PageTransitionEvent) => {
    if (!event.persisted) {
      for (const key of ARENA_BIG_NET_KEYS) disposeBigNetSearcherNow(key);
      disposeRuntimeResources();
    }
  };
  window.addEventListener('pagehide', arenaPagehideHandler);
}

async function init(mountSignal: AbortSignal) {
  if (!isV0DeployProfile()) {
    REQUESTED_RECKLESS_VARIANT = await resolveDefaultRecklessVariantAssetFallback(
      REQUESTED_RECKLESS_VARIANT,
      REQUESTED_RECKLESS_EXPLICIT,
      whileArenaMounted(renderRecklessRuntimeInfo),
    );
  }
  if (isStaleMount(mountSignal)) return;
  REQUESTED_VIRIDITHAS_VARIANT = await resolveDefaultViridithasVariantAssetFallback(
    REQUESTED_VIRIDITHAS_VARIANT,
    REQUESTED_VIRIDITHAS_EXPLICIT,
    whileArenaMounted(renderRecklessRuntimeInfo),
  );
  if (isStaleMount(mountSignal)) return;
  REQUESTED_BERSERK_VARIANT = await resolveDefaultBerserkVariantAssetFallback(
    REQUESTED_BERSERK_VARIANT,
    REQUESTED_BERSERK_EXPLICIT,
    whileArenaMounted(renderRecklessRuntimeInfo),
  );
  if (isStaleMount(mountSignal)) return;
  REQUESTED_PLENTYCHESS_VARIANT = await resolveDefaultPlentyChessVariantAssetFallback(
    REQUESTED_PLENTYCHESS_VARIANT,
    REQUESTED_PLENTYCHESS_EXPLICIT,
    whileArenaMounted(renderRecklessRuntimeInfo),
  );
  if (isStaleMount(mountSignal)) return;
  REQUESTED_STORMPHRAX_VARIANT = await resolveDefaultStormphraxVariantAssetFallback(
    REQUESTED_STORMPHRAX_VARIANT,
    REQUESTED_STORMPHRAX_EXPLICIT,
    whileArenaMounted(renderStormphraxRuntimeInfo),
  );
  if (isStaleMount(mountSignal)) return;
  renderBoard();
  installRuntimeAuditPanel();
  installLc0PresetOptions();
  installExperimentalLc0RuntimeOption();
  selectEl('lc0RuntimeSelect').value = initialLc0Runtime();
  if (!isV0DeployProfile()) refreshRecklessVariantUi();
  refreshViridithasVariantUi();
  refreshBerserkVariantUi();
  refreshPlentyChessVariantUi();
  refreshStormphraxVariantUi();
  selectEl('recklessVariantSelect').value = REQUESTED_RECKLESS_VARIANT.key;
  selectEl('viridithasVariantSelect').value = REQUESTED_VIRIDITHAS_VARIANT.key;
  selectEl('berserkVariantSelect').value = REQUESTED_BERSERK_VARIANT.key;
  selectEl('plentychessVariantSelect').value = REQUESTED_PLENTYCHESS_VARIANT.key;
  selectEl('stormphraxVariantSelect').value = REQUESTED_STORMPHRAX_VARIANT.key;
  restoreArenaSeatRows();
  applyArenaQueryParams();
  applyLc0Preset(inferredLc0Preset());
  if (!isV0DeployProfile()) await ensureSelectedRecklessAssetsAvailable();
  if (isStaleMount(mountSignal)) return;
  renderRecklessRuntimeInfo();
  renderViridithasRuntimeInfo();
  renderBerserkRuntimeInfo();
  renderPlentyChessRuntimeInfo();
  renderStormphraxRuntimeInfo();
  // sfThreads=0 (or sfThreads=auto) hands thread selection to the resource broker.
  inputEl('stockfishThreadsInput').value = String(Math.max(0, Math.min(32, Math.floor(Number(params.get('sfThreads') ?? '1') || 0))));
  refreshStockfishControls();
  void renderRuntimeBadge();
  if (!isV0DeployProfile()) await refreshCentipawnHybridManifestStatus();
  if (isStaleMount(mountSignal)) return;
  buildEngines();
  populateSeats();
  void refreshBt4Availability();
  void probeEngineLogos(
    whileArenaMounted(() => {
      renderSeatSelectors();
      refreshSeatControls();
      renderSideLabels();
    }),
  );
  wireEvents();
  refreshBudgetControls();
  refreshOpeningPreview();
  await checkpointWrites;
  try {
    const saved = await loadArenaCheckpoint();
    if (isStaleMount(mountSignal)) return;
    checkpoint = saved;
    checkpointToken = saved ? { runId: saved.runId, revision: saved.revision } : null;
    if (saved) {
      showCheckpointResults(saved);
      arenaMatchState = arenaRemainingIndices(saved).length ? 'stopped' : 'final';
      el('message').textContent = arenaMatchState === 'final' ? 'Saved tournament results.' : 'A saved tournament is ready to resume.';
      // Inspect results without loading engines or changing the saved configuration.
      const last = saved.completed.at(-1);
      if (last) {
        const trail = finishedTrails[last.scheduleIndex];
        liveTrail = trail;
        historyBoards = trail.entries.map((entry) => parseFen(entry.fen));
        board = historyBoards[historyBoards.length - 1];
        lastUci = trail.entries.at(-1)?.uci ?? null;
        const scheduled = saved.schedule[last.scheduleIndex];
        setBoardSideEngines(null, resultsNames.get(scheduled.whiteId)!, null, resultsNames.get(scheduled.blackId)!);
        enterReview(trail, trail.entries.length - 1);
      }
      el('start').toggleAttribute('disabled', false);
      syncArenaPhase();
    }
  } catch (error) {
    if (isStaleMount(mountSignal)) return;
    checkpointUnreadable = error instanceof ArenaCheckpointInvalidError;
    checkpointProblem = checkpointUnreadable ? `${error instanceof Error ? error.message : String(error)}. Discard it to start a new run.` : `Local saving is unavailable: ${error instanceof Error ? error.message : String(error)}. Export PGN before leaving.`;
    renderCheckpointStatus();
  }
  if (!checkpoint) await loadLc0Evaluator(mountSignal);
  void renderRuntimeBadge();
  void runFixedSuiteBenchAutorun();
  void runArenaBenchAutorun();
}

let lastArenaMountHref: string | null = null;

function resetArenaPageState(): void {
  board = parseFen(START_FEN);
  historyBoards = [board];
  lastUci = null;
  liveTrail = null;
  finishedTrails.length = 0;
  reviewTrail = null;
  reviewIndex = 0;
  reviewing = false;
  reviewShapes = null;
  rootChartContext = null;
  boardWhiteId = null;
  boardBlackId = null;
  boardWhiteName = null;
  boardBlackName = null;
  loadingLc0 = false;
  running = false;
  startPending = false;
  gameChartSamples = [];
  games.length = 0;
  gameResultChips.length = 0;
  scheduledGameCount = 0;
  resultsSchedule = [];
  resultsNames.clear();
  arenaMatchState = 'none';
  checkpoint = null;
  checkpointToken = null;
  checkpointProblem = '';
  checkpointUnreadable = false;
  checkpointSaving = false;
  lastOutputEngineId = null;
  thinkingSince.clear();
  engines.clear();
  seatRows.splice(0, seatRows.length, { family: 'lc0', variant: 'small', strength: 100 }, { family: 'sf', variant: 'lite', strength: 8 });
}

export function mountArenaBrowser(): () => void {
  const href = location.href;
  if (lastArenaMountHref !== null && lastArenaMountHref !== href) {
    location.reload();
    return () => undefined;
  }
  lastArenaMountHref = href;
  resetArenaPageState();
  const controller = new AbortController();
  mountAbort = controller;
  arenaMountedCallbackCache = new WeakMap();
  void init(controller.signal).catch((error) => {
    if (controller.signal.aborted || mountAbort !== controller) return;
    hideEngineWarmupStatus();
    const detail = error instanceof Error ? error.message : String(error);
    el('message').textContent = `Arena failed to initialize: ${detail}`;
    el('pairing').textContent = 'Arena initialization failed.';
  });
  return () => {
    controller.abort();
    if (mountAbort !== controller) return;
    disposeRuntimeResources();
    if (arenaKeydownHandler) document.removeEventListener('keydown', arenaKeydownHandler);
    arenaKeydownHandler = null;
    if (arenaAuditHandler) window.removeEventListener(BROWSER_RUNTIME_AUDIT_EVENT, arenaAuditHandler);
    arenaAuditHandler = null;
    if (arenaPagehideHandler) window.removeEventListener('pagehide', arenaPagehideHandler);
    arenaPagehideHandler = null;
    if (arenaClockTimer) clearInterval(arenaClockTimer);
    arenaClockTimer = null;
    arenaChartResizeObserver?.disconnect();
    arenaChartResizeObserver = null;
    (ground as { destroy?: () => void } | null)?.destroy?.();
    ground = null;
  };
}
