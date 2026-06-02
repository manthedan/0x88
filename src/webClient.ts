import { Chessground } from 'chessground';
import type { Key } from 'chessground/types';
import { parseFen, boardToFen, squareName, START_FEN, type BoardState } from './chess/board.ts';
import { inCheck, legalMoves, makeMove } from './chess/movegen.ts';
import { moveFromUci, moveToActionId, moveToUci, type Move } from './chess/moveCodec.ts';
import { automaticDrawReason } from './chess/drawRules.ts';
import { moveToSan, uciLineToSan, uciToSan } from './chess/san.ts';
import { actionValuePuctPolicy, advanceSearchRoot, auxPuctPolicy, chooseMove, classicPuctPolicy, montyLitePuctPolicy, progressiveWideningPuctPolicy, type CpuctSchedule, type FpuStrategy, type Node as PuctNode, type PrincipalVariationEntry, type PrincipalVariationSelector, type SearchBudgetMode, type SearchOptions, type SearchPolicy, type SearchResult, type SearchStats } from './search/puct.ts';
import { chooseKingQueenVsKingMove } from './search/endgameOracle.ts';
import { OnnxEvaluator, type OnnxStudentMeta } from './nn/onnxEvaluator.ts';
import { SquareFormerEvaluator, type SquareFormerMeta } from './nn/squareformerEvaluator.ts';
import { CachedEvaluator, type Evaluator } from './nn/evaluator.ts';
import { browserWorkerEvaluatorEnabled, WorkerEvaluator, type WorkerChooseMoveOptions, type WorkerSearchPolicyName } from './nn/workerEvaluator.ts';
import { collectOrtRuntimeDiagnostics, describeOrtBackendConfig, resolvedOrtExecutionProviders, tinyLeelaDebugEnabled, tinyLeelaLogLatency, tinyLeelaNowMs, type OrtRuntimeDiagnostics } from './nn/ortRuntime.ts';
import { updateWebClientState, webClientStore, webQueryClient, type EvaluatorStatus } from './web/clientState.ts';

const OPENING_BOOKS: { name: string; moves: string[] }[] = [
  { name: 'Ruy Lopez', moves: ['e2e4', 'e7e5', 'g1f3', 'b8c6', 'f1b5', 'a7a6', 'b5a4', 'g8f6'] },
  { name: 'Italian Game', moves: ['e2e4', 'e7e5', 'g1f3', 'b8c6', 'f1c4', 'f8c5', 'c2c3', 'g8f6'] },
  { name: 'Scotch Game', moves: ['e2e4', 'e7e5', 'g1f3', 'b8c6', 'd2d4', 'e5d4', 'f3d4', 'g8f6'] },
  { name: 'Sicilian Najdorf/Dragon family', moves: ['e2e4', 'c7c5', 'g1f3', 'd7d6', 'd2d4', 'c5d4', 'f3d4', 'g8f6'] },
  { name: 'Sicilian Classical', moves: ['e2e4', 'c7c5', 'g1f3', 'b8c6', 'd2d4', 'c5d4', 'f3d4', 'g8f6'] },
  { name: 'Sicilian Kan/Taimanov family', moves: ['e2e4', 'c7c5', 'g1f3', 'e7e6', 'd2d4', 'c5d4', 'f3d4', 'g8f6'] },
  { name: 'Caro-Kann Advance', moves: ['e2e4', 'c7c6', 'd2d4', 'd7d5', 'e4e5', 'c8f5', 'g1f3', 'e7e6'] },
  { name: 'French Steinitz', moves: ['e2e4', 'e7e6', 'd2d4', 'd7d5', 'b1c3', 'g8f6', 'e4e5', 'f6d7'] },
  { name: 'Pirc/Austrian Attack', moves: ['e2e4', 'd7d6', 'd2d4', 'g8f6', 'b1c3', 'g7g6', 'f2f4', 'f8g7'] },
  { name: 'Alekhine Defense', moves: ['e2e4', 'g8f6', 'e4e5', 'f6d5', 'd2d4', 'd7d6', 'g1f3', 'c8g4'] },
  { name: "Queen's Gambit Declined", moves: ['d2d4', 'd7d5', 'c2c4', 'e7e6', 'b1c3', 'g8f6', 'c1g5', 'f8e7'] },
  { name: 'Slav Defense', moves: ['d2d4', 'd7d5', 'c2c4', 'c7c6', 'g1f3', 'g8f6', 'b1c3', 'd5c4'] },
  { name: 'Nimzo-Indian', moves: ['d2d4', 'g8f6', 'c2c4', 'e7e6', 'b1c3', 'f8b4', 'e2e3', 'e8g8'] },
  { name: "King's Indian", moves: ['d2d4', 'g8f6', 'c2c4', 'g7g6', 'b1c3', 'f8g7', 'e2e4', 'd7d6'] },
  { name: 'Dutch Defense', moves: ['d2d4', 'f7f5', 'c2c4', 'g8f6', 'g2g3', 'e7e6', 'f1g2', 'f8e7'] },
  { name: 'English Four Knights', moves: ['c2c4', 'e7e5', 'b1c3', 'g8f6', 'g2g3', 'd7d5', 'c4d5', 'f6d5'] },
  { name: 'Symmetrical English', moves: ['c2c4', 'c7c5', 'b1c3', 'b8c6', 'g2g3', 'g7g6', 'f1g2', 'f8g7'] },
  { name: 'Reti into QGD', moves: ['g1f3', 'd7d5', 'd2d4', 'g8f6', 'c2c4', 'e7e6', 'b1c3', 'f8e7'] },
  { name: 'Reti/KIA setup', moves: ['g1f3', 'g8f6', 'c2c4', 'g7g6', 'g2g3', 'f8g7', 'f1g2', 'e8g8'] },
  { name: "King's Indian Attack", moves: ['g2g3', 'd7d5', 'f1g2', 'e7e5', 'd2d3', 'g8f6', 'g1f3', 'b8c6'] },
  { name: 'Nimzo-Larsen', moves: ['b2b3', 'e7e5', 'c1b2', 'b8c6', 'e2e3', 'd7d5', 'f1b5', 'f8d6'] },
  { name: 'Bird Opening', moves: ['f2f4', 'd7d5', 'g1f3', 'g8f6', 'e2e3', 'e7e6', 'b2b3', 'f8e7'] },
  { name: 'Vienna/Van Geet', moves: ['b1c3', 'd7d5', 'e2e4', 'd5d4', 'c3e2', 'e7e5', 'g2g3', 'g8f6'] },
  { name: 'Colle System', moves: ['e2e3', 'd7d5', 'd2d4', 'g8f6', 'g1f3', 'e7e6', 'f1d3', 'c7c5'] },
  { name: 'London/Slav structure', moves: ['c2c3', 'd7d5', 'd2d4', 'g8f6', 'g1f3', 'e7e6', 'c1f4', 'f8d6'] },
];
const OPENING_BOOK_LINES = OPENING_BOOKS.map((book) => book.moves);

const params = new URLSearchParams(location.search);
type PlayerSide = 'white' | 'black';
type PlayStyle = 'normal' | 'nibbler-brain' | 'you-brain' | 'local-ai-brain' | 'local-ai-hand';
type PieceRole = 'p' | 'n' | 'b' | 'r' | 'q' | 'k';
const PIECE_NAMES: Record<PieceRole, string> = { p: 'pawn', n: 'knight', b: 'bishop', r: 'rook', q: 'queen', k: 'king' };
const PIECE_BUTTON_ORDER: PieceRole[] = ['p', 'n', 'b', 'r', 'q', 'k'];
function parsePlayStyle(value: string | null): PlayStyle {
  if (value === 'local-ai-brain' || value === 'local-brain') return 'local-ai-brain';
  if (value === 'local-ai-hand' || value === 'local-hand') return 'local-ai-hand';
  if (value === 'you-brain' || value === 'human-brain' || value === 'model-hand') return 'you-brain';
  if (value === 'nibbler-brain' || value === 'model-brain' || value === 'handbrain') return 'nibbler-brain';
  return params.get('handbrain') === '1' ? 'nibbler-brain' : 'normal';
}
function playStyleLabel(style = playStyle): string {
  if (style === 'nibbler-brain') return 'You are Hand';
  if (style === 'you-brain') return 'You are Brain';
  if (style === 'local-ai-brain') return 'Local · AI Brain';
  if (style === 'local-ai-hand') return 'Local · AI Hand';
  return 'normal';
}
function isLocalPlayStyle(style = playStyle) {
  return style === 'local-ai-brain' || style === 'local-ai-hand';
}
let playerSide: PlayerSide = params.get('side') === 'black' ? 'black' : 'white';
let playStyle: PlayStyle = parsePlayStyle(params.get('style'));
let board: BoardState = parseFen(START_FEN);
let historyFens: string[] = [];
let evaluator: Evaluator | null = null;
let ground: ReturnType<typeof Chessground> | null = null;
let orientation: 'white' | 'black' = playerSide;
let lastMove: string | null = null;
let playedMoves: string[] = [];
let playedMoveSans: string[] = [];
let positionFens: string[] = [boardToFen(board)];
let currentPly = 0;
type ClockSnapshot = { white: number; black: number };
let pendingPremove: { from: string; to: string } | null = null;
let brainPiece: PieceRole | null = null;
let gameStarted = false;
let gameOverMessage = '';
const clockParam = params.get('clock');
const parsedClockSeconds = Number(clockParam ?? '300');
const validClockSeconds = Number.isFinite(parsedClockSeconds) ? parsedClockSeconds : 300;
let timedGame = clockParam !== null && clockParam !== 'off' && clockParam !== '0' && validClockSeconds > 0;
let selectedClockMs = Math.max(10, validClockSeconds) * 1000;
let whiteClockMs = selectedClockMs;
let blackClockMs = selectedClockMs;
let clockSnapshots: ClockSnapshot[] = [{ white: whiteClockMs, black: blackClockMs }];
let lastClockTick = performance.now();
let puctSearchRoot: PuctNode | null = null;
const puctTranspositionTable = new Map<string, PuctNode>();
let clockRunning = false;
let stockfish: Worker | null = null;
let stockfishReady = false;
let stockfishThinking = false;
let stockfishBest = '';
let stockfishScore = '';
let stockfishPv = '';
let lastPuctPv = '';
let lastPuctPvMeta = '';
type GameStockfishRow = { ply: number; san: string; uci: string; fen: string; cpBefore: number | null; cpAfter: number | null; loss: number | null; bestSan: string; bestUci: string; label: 'ok' | 'inaccuracy' | 'mistake' | 'blunder' };
let gameStockfishWorker: Worker | null = null;
let gameStockfishRunning = false;
let gameStockfishCancelRequested = false;
let gameStockfishRows: GameStockfishRow[] = [];
let gameStockfishMessage = 'Analyze the current move list for eval graph, centipawn loss, mistakes and blunders.';
let stockfishSeq = 0;
let stockfishSearchTurn: 'w' | 'b' = board.turn;
let stockfishSearchFen = boardToFen(board);
const DEFAULT_MODEL_KEY = 'bt4-sampled1b-best';
const requestedModelKey = params.get('model') ?? DEFAULT_MODEL_KEY;
const puctBatchSize = Math.max(1, Number(params.get('batch') ?? '16'));
const temperature = Number(params.get('temperature') ?? '1');
const topK = Number(params.get('topk') ?? '0');
const topP = Number(params.get('topp') ?? '1');
const stockfishDepth = Number(params.get('sfdepth') ?? '10');
const openingMode = params.get('opening') ?? 'book';
let openingBookMaxPlies = Math.max(0, Number(params.get('bookPlies') ?? '8'));
const puctPvParam = params.get('pv') ?? params.get('showPv');
const puctPvEnabled = puctPvParam === null || !['0', 'false', 'no', 'off'].includes(puctPvParam.toLowerCase());
const puctPvDepth = Math.max(1, Math.min(32, Math.floor(Number(params.get('pvDepth') ?? '12')) || 12));
function parsePvSelector(value: string | null): PrincipalVariationSelector {
  return value === 'q' || value === 'puct' ? value : 'visits';
}
const puctPvSelector = parsePvSelector(params.get('pvSelector'));
function parseSearchBudgetMode(value: string | null): SearchBudgetMode {
  const normalized = String(value ?? '').toLowerCase().replace(/[ -]/g, '_');
  return normalized === 'neural' || normalized === 'neural_evals' || normalized === 'evals' ? 'neural' : 'visits';
}
const searchBudgetMode = parseSearchBudgetMode(params.get('budgetMode') ?? params.get('budget'));
const neuralMaxVisitsMultiplier = Math.max(1, Math.min(16, Number(params.get('maxVisitsMultiplier') ?? '4') || 4));
const adaptiveEarlyStop = searchBudgetMode === 'neural' && (params.get('earlyStop') ?? 'root-dominance').toLowerCase().replace(/[ -]/g, '_') === 'root_dominance' ? 'root-dominance' as const : 'none' as const;
const puctRootReuseEnabled = params.get('puctRootReuse') !== '0' && params.get('puctRootReuse') !== 'off';
const puctTranspositionsEnabled = params.get('puctTranspositions') === '1' || params.get('puctTranspositions') === 'on';
const puctPonderEnabled = params.get('ponder') === '1' || params.get('ponder') === 'on';
const puctPonderChunkMs = Math.max(4, Math.min(100, Math.floor(Number(params.get('ponderYieldMs') ?? '24')) || 24));
let busy = false;
let renderSeq = 0;
let engineRequestSeq = 0;
let searchAbortController: AbortController | null = null;
type PonderPrediction = { uci: string; fen: string; historyKey: string; root: PuctNode | null; visits: number };
let puctPonderPrediction: PonderPrediction | null = null;
let puctPonderAbortController: AbortController | null = null;
let puctPonderSeq = 0;
function cancelPonder() {
  puctPonderSeq += 1;
  puctPonderAbortController?.abort();
  puctPonderAbortController = null;
  puctPonderPrediction = null;
}
function cancelAsyncWork() {
  engineRequestSeq += 1;
  renderSeq += 1;
  searchAbortController?.abort();
  searchAbortController = null;
  cancelPonder();
}
function beginEngineRequest() {
  engineRequestSeq += 1;
  searchAbortController?.abort();
  cancelPonder();
  searchAbortController = new AbortController();
  return engineRequestSeq;
}
function currentSearchSignal() {
  return searchAbortController?.signal;
}
function isCurrentEngineRequest(requestId: number) {
  return requestId === engineRequestSeq;
}
function isAbortError(error: unknown) {
  return error instanceof Error && (error.name === 'AbortError' || /cancelled|aborted/i.test(error.message));
}
type SearchModeName = 'classic' | 'monty' | 'monty_lc0both' | 'widen' | 'widen_lc0both' | 'puct_lc0both';
type RuntimeEpHint = 'auto' | 'webgpu' | 'wasm';
type PlayableModelVariant = { onnx: string; meta?: string; label?: string; preferredEp?: RuntimeEpHint; requiredFeatures?: string[]; estimatedBytes?: number };
type PlayableModelRuntime = { preferredEp?: RuntimeEpHint; fallbackEp?: RuntimeEpHint; requiredFeatures?: string[]; estimatedBytes?: number; variants?: Record<string, PlayableModelVariant>; notes?: string };
type PlayableModel = { onnx: string; meta: string; label: string; forcedMode?: string; defaultMode?: string; defaultVisits?: number; defaultPuctPolicy?: string; defaultSearchMode?: SearchModeName; defaultAvWeight?: number; runtime?: PlayableModelRuntime };
type SelectedPlayableArtifact = { onnx: string; meta: string; label: string; variantKey: string; preferredEp?: RuntimeEpHint; requiredFeatures?: string[]; estimatedBytes?: number };
type PuctPolicyName = 'classic' | 'av' | 'aux';
type SearchTuning = {
  policy: PuctPolicyName;
  searchMode: SearchModeName;
  label: string;
  source?: string;
  cpuct?: number;
  fpu?: number;
  cpuctSchedule?: CpuctSchedule;
  fpuStrategy?: FpuStrategy;
  fpuReduction?: number;
  avWeight?: number;
  rankWeight?: number;
  regretWeight?: number;
  riskWeight?: number;
  uncertaintyWeight?: number;
};
const PLAYABLE_DEFAULT_SEARCH_MODE: SearchModeName = 'monty_lc0both';
const allModels: Record<string, PlayableModel> = {
  'cnn-32x4-100m-e3': { onnx: '/models/cnn_32x4_100m_e3.onnx', meta: '/models/cnn_32x4_100m_e3.meta.json', label: 'CNN 32x4 · 100M · e3', defaultMode: 'puct', defaultSearchMode: PLAYABLE_DEFAULT_SEARCH_MODE },
  'cnn-48x5-100m-e3': { onnx: '/models/cnn_48x5_100m_e3.onnx', meta: '/models/cnn_48x5_100m_e3.meta.json', label: 'CNN 48x5 · 100M · e3', defaultMode: 'puct', defaultSearchMode: PLAYABLE_DEFAULT_SEARCH_MODE },
  'cnn-64x6-100m-e3': { onnx: '/models/cnn_64x6_100m_e3.onnx', meta: '/models/cnn_64x6_100m_e3.meta.json', label: 'CNN 64x6 · 100M · e3', defaultMode: 'puct', defaultSearchMode: PLAYABLE_DEFAULT_SEARCH_MODE },
  'cnn-80x5-100m-e3': { onnx: '/models/cnn_80x5_100m_e3.onnx', meta: '/models/cnn_80x5_100m_e3.meta.json', label: 'CNN 80x5 · 100M · e3', defaultMode: 'puct', defaultSearchMode: PLAYABLE_DEFAULT_SEARCH_MODE },
  'cnn-96x8-100m-e8': { onnx: '/models/cnn96x8_100m_e8.onnx', meta: '/models/cnn96x8_100m_e8.meta.json', label: 'CNN 96x8 · 100M · e8', defaultMode: 'puct', defaultSearchMode: PLAYABLE_DEFAULT_SEARCH_MODE },
  'moveformer-80x5-100m-e8-k128': { onnx: '/models/moveformer_80x5_100m_e8_k128.onnx', meta: '/models/moveformer_80x5_100m_e8_k128.meta.json', label: 'MoveFormer 80x5 k128 · 100M · e8', defaultMode: 'puct', defaultVisits: 256, defaultPuctPolicy: 'auto', defaultSearchMode: PLAYABLE_DEFAULT_SEARCH_MODE, runtime: { preferredEp: 'auto', fallbackEp: 'wasm', estimatedBytes: 14_283_133 } },
  'bt4-sampled1b-best': { onnx: '/models/bt4_sampled1b_best.onnx', meta: '/models/bt4_sampled1b_best.meta.json', label: 'BT4 sampled-1B best · strongest baseline', defaultMode: 'puct', defaultVisits: 256, defaultPuctPolicy: 'auto', defaultSearchMode: PLAYABLE_DEFAULT_SEARCH_MODE, runtime: { preferredEp: 'auto', fallbackEp: 'wasm', estimatedBytes: 4_337_008 } },
  'bt4-1b-mix50-noav-best': { onnx: '/models/bt4_1b_mix50_noav_best.onnx', meta: '/models/bt4_1b_mix50_noav_best.meta.json', label: 'BT4 mix50 noAV 1B · best non-EMA', defaultMode: 'puct', defaultVisits: 256, defaultPuctPolicy: 'auto', defaultSearchMode: PLAYABLE_DEFAULT_SEARCH_MODE, runtime: { preferredEp: 'auto', fallbackEp: 'wasm', estimatedBytes: 4_464_375, notes: 'frozen mix50 LC0/SF18 noAV 1B baseline; best raw checkpoint export' } },
  'bt4-1b-mix50-noav-best-ema': { onnx: '/models/bt4_1b_mix50_noav_best_ema.onnx', meta: '/models/bt4_1b_mix50_noav_best_ema.meta.json', label: 'BT4 mix50 noAV 1B · best EMA', defaultMode: 'puct', defaultVisits: 256, defaultPuctPolicy: 'auto', defaultSearchMode: PLAYABLE_DEFAULT_SEARCH_MODE, runtime: { preferredEp: 'auto', fallbackEp: 'wasm', estimatedBytes: 4_464_375, notes: 'frozen mix50 LC0/SF18 noAV 1B baseline; best EMA checkpoint export' } },
  'bt4-tg-relbank-sampled1b-norm': { onnx: '/models/bt4_tg_relbank_sampled1b_norm.onnx', meta: '/models/bt4_tg_relbank_sampled1b_norm.meta.json', label: 'BT4 TG RelBank sampled-1B · norm h8', defaultMode: 'puct', defaultVisits: 256, defaultPuctPolicy: 'auto', defaultSearchMode: PLAYABLE_DEFAULT_SEARCH_MODE, runtime: { preferredEp: 'auto', fallbackEp: 'wasm', estimatedBytes: 4_745_999, notes: 'normalized h8 ThreatGraph input; WebGPU preferred when available, WASM remains semantic fallback' } },
};
const canonicalModelKeys = ['bt4-1b-mix50-noav-best-ema', 'bt4-1b-mix50-noav-best', 'bt4-tg-relbank-sampled1b-norm', 'bt4-sampled1b-best', 'moveformer-80x5-100m-e8-k128', 'cnn-96x8-100m-e8', 'cnn-80x5-100m-e3', 'cnn-64x6-100m-e3', 'cnn-48x5-100m-e3', 'cnn-32x4-100m-e3'];
const canonicalOnly = (import.meta as unknown as { env?: Record<string, string> }).env?.VITE_CANONICAL_MODELS === '1';
const models: Record<string, PlayableModel> = canonicalOnly
  ? Object.fromEntries(canonicalModelKeys.map((key) => [key, allModels[key]])) as Record<string, PlayableModel>
  : allModels;

function runtimeFeatureAvailable(feature: string, diagnostics: OrtRuntimeDiagnostics | undefined): boolean {
  const normalized = feature.toLowerCase();
  if (normalized === 'webgpu') return diagnostics?.webgpuAvailable === true;
  if (normalized === 'secure-context') return diagnostics?.secureContext !== false;
  return diagnostics?.adapter?.features?.some((item) => item.toLowerCase() === normalized) === true;
}

function missingRuntimeFeatures(features: string[] | undefined, diagnostics: OrtRuntimeDiagnostics | undefined): string[] {
  if (!features?.length) return [];
  return features.filter((feature) => !runtimeFeatureAvailable(feature, diagnostics));
}

function selectPlayableArtifact(model: PlayableModel, diagnostics: OrtRuntimeDiagnostics | undefined): SelectedPlayableArtifact {
  const providers = resolvedOrtExecutionProviders();
  const variants = Object.entries(model.runtime?.variants ?? {});
  for (const [variantKey, variant] of variants) {
    if (variant.preferredEp === 'webgpu' && !providers.includes('webgpu')) continue;
    if (variant.preferredEp === 'wasm' && !providers.includes('wasm')) continue;
    if (missingRuntimeFeatures(variant.requiredFeatures, diagnostics).length) continue;
    return {
      onnx: variant.onnx,
      meta: variant.meta ?? model.meta,
      label: variant.label ?? model.label,
      variantKey,
      preferredEp: variant.preferredEp,
      requiredFeatures: variant.requiredFeatures,
      estimatedBytes: variant.estimatedBytes,
    };
  }
  return {
    onnx: model.onnx,
    meta: model.meta,
    label: model.label,
    variantKey: 'base',
    preferredEp: model.runtime?.preferredEp,
    requiredFeatures: model.runtime?.requiredFeatures,
    estimatedBytes: model.runtime?.estimatedBytes,
  };
}

function logPlayableRuntimeSelection(model: PlayableModel, artifact: SelectedPlayableArtifact, diagnostics: OrtRuntimeDiagnostics | undefined): void {
  const missing = missingRuntimeFeatures(artifact.requiredFeatures, diagnostics);
  if (missing.length) {
    console.warn('Tiny Leela runtime: selected model artifact is missing requested browser/GPU features.', {
      label: model.label,
      artifact: artifact.onnx,
      requiredFeatures: artifact.requiredFeatures,
      missing,
      resolvedExecutionProviders: diagnostics?.resolvedExecutionProviders ?? resolvedOrtExecutionProviders(),
      webgpuAvailable: diagnostics?.webgpuAvailable,
      adapter: diagnostics?.adapter,
    });
    return;
  }
  if (!tinyLeelaDebugEnabled('runtime')) return;
  console.info('Tiny Leela runtime: selected model artifact.', {
    label: model.label,
    artifact: artifact.onnx,
    meta: artifact.meta,
    variantKey: artifact.variantKey,
    preferredEp: artifact.preferredEp,
    estimatedBytes: artifact.estimatedBytes,
    runtime: model.runtime,
    resolvedExecutionProviders: diagnostics?.resolvedExecutionProviders ?? resolvedOrtExecutionProviders(),
    webgpuAvailable: diagnostics?.webgpuAvailable,
    adapter: diagnostics?.adapter,
  });
}
const VISIT_OPTIONS = [16, 32, 64, 128, 256, 512, 1024];
const TUNED_PUCT_BY_MODEL_VISITS: Record<string, Record<number, Omit<SearchTuning, 'searchMode'>>> = {
  // Confirmed against classic PUCT in mac-mini cnn96 aux-PUCT tuning. Only
  // visits with a confirmed non-classic winner are enabled for auto mode.
  'cnn-96x8-100m-e8': {
    32: { policy: 'aux', label: 'tuned aux PUCT v32', source: 'mac-mini cnn96 puct tune 2026-05-09', cpuct: 1.5, fpu: 0, avWeight: 0, rankWeight: 0.0125, regretWeight: 0, riskWeight: 0, uncertaintyWeight: 0 },
    64: { policy: 'aux', label: 'tuned aux PUCT v64', source: 'mac-mini cnn96 puct tune 2026-05-09', cpuct: 1.2, fpu: -0.1, avWeight: 0.0005, rankWeight: 0.0195, regretWeight: 0.0015, riskWeight: 0, uncertaintyWeight: 0 },
  },
};
const selectedModelKey = models[requestedModelKey] ? requestedModelKey : (models[DEFAULT_MODEL_KEY] ? DEFAULT_MODEL_KEY : Object.keys(models)[0]);
const selectedModel = models[selectedModelKey];
const requestedPlayMode = params.get('mode') ?? selectedModel.defaultMode ?? 'puct';
const visits = Math.max(1, Math.floor(Number(params.get('visits') ?? selectedModel.defaultVisits ?? '128')) || (selectedModel.defaultVisits ?? 128));
const puctPonderVisits = Math.max(1, Math.min(4096, Math.floor(Number(params.get('ponderVisits') ?? Math.max(16, Math.floor(visits / 2)))) || Math.max(16, Math.floor(visits / 2))));
function parseSearchMode(value: string | null | undefined, fallback: SearchModeName): SearchModeName {
  const normalized = String(value ?? '').toLowerCase().replace(/[ -]/g, '_');
  if (normalized === 'classic' || normalized === 'puct') return 'classic';
  if (normalized === 'monty' || normalized === 'monty_lite') return 'monty';
  if (normalized === 'monty_lc0' || normalized === 'monty_lc0both') return 'monty_lc0both';
  if (normalized === 'widen' || normalized === 'progressive_widening') return 'widen';
  if (normalized === 'widen_lc0' || normalized === 'widen_lc0both') return 'widen_lc0both';
  if (normalized === 'puct_lc0' || normalized === 'puct_lc0both') return 'puct_lc0both';
  return fallback;
}
const requestedPuctPolicy = params.get('puctPolicy') ?? selectedModel.defaultPuctPolicy ?? 'auto';
const requestedSearchMode = parseSearchMode(params.get('searchMode') ?? params.get('search'), selectedModel.defaultSearchMode ?? PLAYABLE_DEFAULT_SEARCH_MODE);
const playMode = selectedModel.forcedMode ?? requestedPlayMode;

function numberParam(name: string, fallback: number | undefined): number | undefined {
  const raw = params.get(name);
  if (raw === null) return fallback;
  const parsed = Number(raw);
  return Number.isFinite(parsed) ? parsed : fallback;
}
function withSearchParamOverrides(tuning: SearchTuning): SearchTuning {
  return {
    ...tuning,
    cpuct: numberParam('cpuct', tuning.cpuct),
    fpu: numberParam('fpu', tuning.fpu),
    avWeight: numberParam('avWeight', tuning.avWeight),
    rankWeight: numberParam('rankWeight', tuning.rankWeight),
    regretWeight: numberParam('regretWeight', tuning.regretWeight),
    riskWeight: numberParam('riskWeight', tuning.riskWeight),
    uncertaintyWeight: numberParam('uncertaintyWeight', tuning.uncertaintyWeight),
  };
}
function tunedPuctFor(model: string, visitCount: number): Omit<SearchTuning, 'searchMode'> | null {
  return TUNED_PUCT_BY_MODEL_VISITS[model]?.[visitCount] ?? null;
}
function searchModeLabel(mode: SearchModeName): string {
  switch (mode) {
    case 'classic': return 'classic PUCT';
    case 'monty': return 'Monty-lite PUCT';
    case 'monty_lc0both': return 'Monty-lite + LC0 cpuct/FPU';
    case 'widen': return 'progressive widening PUCT';
    case 'widen_lc0both': return 'widen + LC0 cpuct/FPU';
    case 'puct_lc0both': return 'classic PUCT + LC0 cpuct/FPU';
  }
}
function lc0ScheduleFor(mode: SearchModeName): Pick<SearchTuning, 'cpuctSchedule' | 'fpuStrategy' | 'fpuReduction'> {
  return mode.endsWith('lc0both') ? { cpuctSchedule: 'lc0-log', fpuStrategy: 'lc0-reduction', fpuReduction: 0.330 } : { cpuctSchedule: 'constant', fpuStrategy: 'constant', fpuReduction: 0.330 };
}
function resolveSearchTuning(model: string, visitCount: number, requested: string, modelConfig: PlayableModel = models[model] ?? allModels[model] ?? selectedModel): SearchTuning {
  const mode = requestedSearchMode;
  const tuned = tunedPuctFor(model, visitCount);
  const modeBits = lc0ScheduleFor(mode);
  if (requested === 'classic') return withSearchParamOverrides({ policy: 'classic', searchMode: mode, label: searchModeLabel(mode), cpuct: 1.5, fpu: 0, ...modeBits });
  if (requested === 'av') return withSearchParamOverrides({ policy: 'av', searchMode: mode, label: `AV ${searchModeLabel(mode)}`, cpuct: tuned?.cpuct ?? 1.5, fpu: tuned?.fpu ?? 0, avWeight: tuned?.avWeight ?? modelConfig.defaultAvWeight ?? 0.25, ...modeBits });
  if (requested === 'aux') return withSearchParamOverrides({ ...(tuned ?? { policy: 'aux', label: `aux ${searchModeLabel(mode)}`, cpuct: 1.5, fpu: 0, avWeight: modelConfig.defaultAvWeight ?? 0.25, rankWeight: 0, regretWeight: 0, riskWeight: 0, uncertaintyWeight: 0 }), searchMode: mode, ...modeBits });
  return withSearchParamOverrides({ policy: 'classic', searchMode: mode, label: searchModeLabel(mode), cpuct: 1.5, fpu: 0, ...modeBits });
}
function searchPolicyFor(tuning: SearchTuning): SearchPolicy {
  if (tuning.policy === 'aux') return auxPuctPolicy;
  if (tuning.policy === 'av') return actionValuePuctPolicy;
  if (tuning.searchMode === 'monty' || tuning.searchMode === 'monty_lc0both') return montyLitePuctPolicy;
  if (tuning.searchMode === 'widen' || tuning.searchMode === 'widen_lc0both') return progressiveWideningPuctPolicy;
  return classicPuctPolicy;
}
function workerSearchPolicyNameFor(tuning: SearchTuning): WorkerSearchPolicyName {
  if (tuning.policy === 'aux') return 'aux';
  if (tuning.policy === 'av') return 'av';
  if (tuning.searchMode === 'monty' || tuning.searchMode === 'monty_lc0both') return 'monty';
  if (tuning.searchMode === 'widen' || tuning.searchMode === 'widen_lc0both') return 'widen';
  return 'classic';
}
function browserSearchWorkerEnabled(): boolean {
  const raw = params.get('searchWorker') ?? params.get('engineWorkerSearch') ?? params.get('engineWorker');
  return raw === 'search' || raw === '1-search' || raw === 'true-search' || raw === 'puct';
}
function workerBackedEvaluator(value: Evaluator): WorkerEvaluator | null {
  if (value instanceof WorkerEvaluator) return value;
  if (value instanceof CachedEvaluator && value.inner instanceof WorkerEvaluator) return value.inner;
  return null;
}
function abortError(): Error {
  const error = new Error('Search aborted');
  error.name = 'AbortError';
  return error;
}
function workerChooseMoveOptions(options: SearchOptions, tuning: SearchTuning): WorkerChooseMoveOptions {
  const { signal: _signal, root: _root, transpositionTable: _transpositionTable, searchPolicy: _searchPolicy, ...serializable } = options;
  return { ...serializable, searchPolicyName: workerSearchPolicyNameFor(tuning) };
}
async function chooseMoveMaybeWorker(board: BoardState, evaln: Evaluator, options: SearchOptions, tuning: SearchTuning): Promise<SearchResult> {
  const workerEval = browserSearchWorkerEnabled() ? workerBackedEvaluator(evaln) : null;
  if (!workerEval) return chooseMove(board, evaln, options);
  if (options.signal?.aborted) throw abortError();
  if (tinyLeelaDebugEnabled('runtime')) console.info('Tiny Leela runtime: PUCT search delegated to evaluator worker.', { visits: options.visits, batchSize: options.batchSize, policy: workerSearchPolicyNameFor(tuning), rootReuse: false });
  const result = await workerEval.chooseMove(board, workerChooseMoveOptions(options, tuning));
  if (options.signal?.aborted) throw abortError();
  return result;
}
function searchOptionOverrides(tuning: SearchTuning) {
  return {
    searchPolicy: searchPolicyFor(tuning),
    cpuct: tuning.cpuct,
    fpu: tuning.fpu,
    cpuctSchedule: tuning.cpuctSchedule,
    fpuStrategy: tuning.fpuStrategy,
    fpuReduction: tuning.fpuReduction,
    avWeight: tuning.avWeight,
    rankWeight: tuning.rankWeight,
    regretWeight: tuning.regretWeight,
    riskWeight: tuning.riskWeight,
    uncertaintyWeight: tuning.uncertaintyWeight,
    budgetMode: searchBudgetMode,
    maxVisitsMultiplier: neuralMaxVisitsMultiplier,
    earlyStop: adaptiveEarlyStop,
  };
}
const searchTuning = resolveSearchTuning(selectedModelKey, visits, requestedPuctPolicy);
function modelKeyOrDefault(value: string | null, fallback: string): string {
  return value && models[value] ? value : fallback;
}
let arenaWhiteKey = modelKeyOrDefault(params.get('arenaWhite'), 'bt4-sampled1b-best');
let arenaBlackKey = modelKeyOrDefault(params.get('arenaBlack'), 'moveformer-80x5-100m-e8-k128');
const arenaDelayMs = Math.max(0, Math.min(5000, Math.floor(Number(params.get('arenaDelay') ?? '350')) || 350));
const arenaMaxPlies = Math.max(1, Math.min(1000, Math.floor(Number(params.get('arenaMaxPlies') ?? '240')) || 240));
type ArenaOpeningMode = 'random-line' | 'mixed' | 'none' | `line-${number}`;
function parseArenaOpening(value: string | null): ArenaOpeningMode {
  if (value === 'none' || value === 'mixed' || value === 'random-line') return value;
  const line = value?.match(/^line-(\d+)$/);
  if (line && OPENING_BOOKS[Number(line[1])]) return value as ArenaOpeningMode;
  return 'random-line';
}
let arenaUseBook = params.get('arenaBook') !== '0';
let arenaOpeningMode = parseArenaOpening(params.get('arenaOpening'));
let arenaBookLineIndex: number | null = null;
let arenaGame = false;
let arenaRunning = false;
let arenaStopRequested = false;
const playableEvalCacheEnabled = params.get('evalCache') !== '0' && params.get('evalCache') !== 'off';
const playableEvalCacheEntries = Math.max(1, Math.min(65536, Math.floor(Number(params.get('evalCacheEntries') ?? '8192')) || 8192));
const arenaEvaluatorCache = new Map<string, Evaluator>();

const $ = (id: string) => document.getElementById(id)!;
function syncWebClientStore(patch: { evaluatorStatus?: EvaluatorStatus; message?: string; activeSideTab?: SideTab } = {}) {
  updateWebClientState({
    uiMode,
    activeSideTab,
    playerSide,
    playStyle,
    gameStarted,
    timedGame,
    selectedClockMs,
    selectedModelKey,
    selectedModelLabel: selectedModel.label,
    busy,
    evaluatorStatus: patch.evaluatorStatus ?? (evaluator ? 'ready' : webClientStore.state.evaluatorStatus),
    message: patch.message ?? webClientStore.state.message,
    ...patch,
  });
}
webClientStore.subscribe((state) => {
  document.body.dataset.uiMode = state.uiMode;
  document.body.dataset.activeTab = state.activeSideTab;
  document.body.dataset.evaluatorStatus = state.evaluatorStatus;
});
function escapeHtml(value: unknown): string {
  return String(value).replace(/[&<>"']/g, (ch) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[ch]!));
}
function percentWidth(value: number, min = 0): string {
  const pct = Math.max(min, Math.min(100, Number.isFinite(value) ? value : 0));
  return `${pct.toFixed(3)}%`;
}
function legalDests() {
  const dests = new Map<Key, Key[]>();
  for (const m of legalMovesForUser()) {
    const from = squareName(m.from) as Key, to = squareName(m.to) as Key;
    dests.set(from, [...(dests.get(from) ?? []), to]);
  }
  return dests;
}
function legalMoveByUci(uci: string) { return legalMoves(board).find((m) => moveToUci(m) === uci) ?? null; }
function boardFen() { return boardToFen(board).split(' ')[0]; }
function randomChoice<T>(values: T[]) { return values[Math.floor(Math.random() * values.length)]; }
function syncHistoryFens() {
  historyFens = positionFens.slice(0, currentPly).reverse();
}
function resetLineFromBoard() {
  positionFens = [boardToFen(board)];
  currentPly = 0;
  historyFens = [];
  lastMove = null;
  playedMoves = [];
  playedMoveSans = [];
  clockSnapshots = [{ white: whiteClockMs, black: blackClockMs }];
  gameStockfishRows = [];
  gameStockfishMessage = 'Analyze the current move list for eval graph, centipawn loss, mistakes and blunders.';
  puctSearchRoot = null;
  puctTranspositionTable.clear();
  cancelPonder();
}
function truncateLineForBranch() {
  if (currentPly >= playedMoves.length) return;
  playedMoves = playedMoves.slice(0, currentPly);
  playedMoveSans = playedMoveSans.slice(0, currentPly);
  positionFens = positionFens.slice(0, currentPly + 1);
  clockSnapshots = clockSnapshots.slice(0, currentPly + 1);
  puctSearchRoot = null;
  puctTranspositionTable.clear();
  cancelPonder();
}
function restoreClockSnapshot(ply: number) {
  const snapshot = clockSnapshots[ply];
  if (!snapshot) return;
  whiteClockMs = snapshot.white;
  blackClockMs = snapshot.black;
  lastClockTick = performance.now();
}
function historyKeyForCurrentPosition() {
  return historyFens.join('\n');
}
function recordMove(move: Move, san: string) {
  const ponderPrediction = puctPonderPrediction;
  truncateLineForBranch();
  gameStockfishRows = [];
  gameStockfishMessage = 'Game changed; rerun Stockfish review for updated cp-loss graph.';
  const uci = moveToUci(move);
  board = makeMove(board, move);
  currentPly += 1;
  lastMove = uci;
  playedMoves.push(uci);
  playedMoveSans.push(san);
  positionFens[currentPly] = boardToFen(board);
  clockSnapshots[currentPly] = { white: whiteClockMs, black: blackClockMs };
  syncHistoryFens();
  puctSearchRoot = puctRootReuseEnabled ? advanceSearchRoot(puctSearchRoot, move, board, historyFens) : null;
  if (puctRootReuseEnabled && ponderPrediction?.uci === uci && ponderPrediction.fen === boardToFen(board) && ponderPrediction.historyKey === historyKeyForCurrentPosition()) {
    puctSearchRoot = ponderPrediction.root;
    lastPuctPvMeta = `Ponder hit · reused ${ponderPrediction.visits} background visits after ${uci}`;
  }
  if (!puctSearchRoot && !puctTranspositionsEnabled) puctTranspositionTable.clear();
  cancelPonder();
}
function resetPositionState() {
  board = parseFen(START_FEN);
  resetClocks();
  resetLineFromBoard();
  pendingPremove = null;
  brainPiece = null;
  gameOverMessage = '';
  ground?.cancelPremove();
}
function applySetupMove(uci: string) {
  const move = legalMoves(board).find((m) => moveToUci(m) === uci) ?? moveFromUci(uci);
  recordMove(move, moveToSan(board, move));
}
function bookCandidatesForCurrentLine(lines = OPENING_BOOK_LINES, ignoreGlobalOpeningMode = false): string[] {
  if (uiMode === 'analysis' || (!ignoreGlobalOpeningMode && openingMode === 'start') || openingBookMaxPlies <= 0) return [];
  if (playedMoves.length >= openingBookMaxPlies) return [];
  const candidates = new Set<string>();
  for (const line of lines) {
    if (line.length <= playedMoves.length) continue;
    if (playedMoves.every((uci, idx) => line[idx] === uci)) candidates.add(line[playedMoves.length]);
  }
  return [...candidates];
}
function chooseBookMove(lines = OPENING_BOOK_LINES, ignoreGlobalOpeningMode = false): Move | null {
  const legalReplies = bookCandidatesForCurrentLine(lines, ignoreGlobalOpeningMode).map((uci) => legalMoveByUci(uci)).filter((move): move is Move => !!move);
  return legalReplies.length ? randomChoice(legalReplies) : null;
}
function chooseOpeningBookMove(): Move | null {
  if (uiMode === 'analysis' || openingMode === 'start' || !gameStarted || board.turn === playerColorToMove()) return null;
  return chooseBookMove();
}
function chooseArenaBookMove(): Move | null {
  if (!arenaUseBook || arenaOpeningMode === 'none') return null;
  if (arenaOpeningMode === 'mixed') return chooseBookMove(OPENING_BOOK_LINES, true);
  const idx = arenaOpeningMode.startsWith('line-') ? Number(arenaOpeningMode.slice(5)) : arenaBookLineIndex;
  const line = idx === null ? null : OPENING_BOOK_LINES[idx];
  return line ? chooseBookMove([line], true) : null;
}
function startPlayGame() {
  gameStarted = true;
  clockRunning = true;
  resetPositionState();
  orientation = playerSide;
  if (isLocalPlayStyle()) return `Local Hand & Brain. ${playStyleLabel()}: both humans take turns from the board. ${timedGame ? `${formatClock(selectedClockMs)} clocks.` : 'No clock.'}`;
  if (openingMode === 'start') return `Start position. You are playing ${playerSide}${playStyle !== 'normal' ? ` · ${playStyleLabel()}` : ''}${timedGame ? ` with ${formatClock(selectedClockMs)} clocks` : ' with no clock'}.`;
  if (playerSide === 'black') {
    const firstMove = chooseOpeningBookMove();
    if (firstMove) applySetupMove(moveToUci(firstMove));
    return `Book first move: ${playedMoveSans.at(-1) ?? 'start position'}. You to move as Black; book may continue through ply ${openingBookMaxPlies}${playStyle !== 'normal' ? ` · ${playStyleLabel()}` : ''}${timedGame ? ` · ${formatClock(selectedClockMs)} clocks` : ' · no clock'}.`;
  }
  return `Start position. Make White’s first move; Tiny Leela can answer from a varied book through ply ${openingBookMaxPlies}${playStyle !== 'normal' ? ` · ${playStyleLabel()}` : ''}${timedGame ? ` · ${formatClock(selectedClockMs)} clocks` : ' · no clock'}.`;
}
function showPlayIntro(message = 'Choose side and clock, then start the game.') {
  arenaGame = false;
  arenaRunning = false;
  arenaStopRequested = true;
  gameStarted = false;
  clockRunning = false;
  resetPositionState();
  orientation = playerSide;
  updateIntroControls();
  return message;
}
function startAnalysisBoard() {
  arenaGame = false;
  arenaRunning = false;
  arenaStopRequested = true;
  gameStarted = true;
  clockRunning = false;
  resetPositionState();
  return 'Free board: start position.';
}
function initModelSelect() {
  const select = document.getElementById('modelSelect') as HTMLSelectElement | null;
  if (!select) return;
  select.replaceChildren(...Object.entries(models).map(([key, model]) => {
    const option = document.createElement('option');
    option.value = key;
    option.selected = key === selectedModelKey;
    option.textContent = model.label;
    return option;
  }));
  select.onchange = () => {
    const url = new URL(location.href);
    url.searchParams.set('model', select.value);
    location.href = url.toString();
  };
}
function initVisitsSelect() {
  const select = document.getElementById('visitsSelect') as HTMLSelectElement | null;
  if (!select) return;
  const options = [...new Set([...VISIT_OPTIONS, visits])].sort((a, b) => a - b);
  select.replaceChildren(...options.map((value) => {
    const option = document.createElement('option');
    option.value = String(value);
    option.selected = value === visits;
    option.textContent = `${value} visits`;
    return option;
  }));
  select.onchange = () => {
    const url = new URL(location.href);
    url.searchParams.set('visits', select.value);
    location.href = url.toString();
  };
}
function initSearchModeSelect() {
  const select = document.getElementById('searchModeSelect') as HTMLSelectElement | null;
  if (!select) return;
  const entries: { value: SearchModeName; label: string }[] = [
    { value: 'monty_lc0both', label: 'Monty + LC0 cpuct/FPU — default' },
    { value: 'monty', label: 'Monty-lite' },
    { value: 'widen_lc0both', label: 'Widen + LC0 cpuct/FPU' },
    { value: 'widen', label: 'Progressive widening' },
    { value: 'puct_lc0both', label: 'Classic PUCT + LC0 cpuct/FPU' },
    { value: 'classic', label: 'Classic PUCT baseline' },
  ];
  select.replaceChildren(...entries.map((entry) => {
    const option = document.createElement('option');
    option.value = entry.value;
    option.selected = entry.value === requestedSearchMode;
    option.textContent = entry.label;
    return option;
  }));
  select.onchange = () => {
    const url = new URL(location.href);
    url.searchParams.set('searchMode', select.value);
    location.href = url.toString();
  };
}
function initBudgetModeSelect() {
  const select = document.getElementById('budgetModeSelect') as HTMLSelectElement | null;
  if (!select) return;
  const entries: { value: SearchBudgetMode; label: string }[] = [
    { value: 'visits', label: 'Fixed visits' },
    { value: 'neural', label: `Adaptive neural evals · max ${neuralMaxVisitsMultiplier}× visits` },
  ];
  select.replaceChildren(...entries.map((entry) => {
    const option = document.createElement('option');
    option.value = entry.value;
    option.selected = entry.value === searchBudgetMode;
    option.textContent = entry.label;
    return option;
  }));
  select.onchange = () => {
    const url = new URL(location.href);
    url.searchParams.set('budgetMode', select.value);
    location.href = url.toString();
  };
}
function arenaOpeningOptions(): { value: ArenaOpeningMode; label: string }[] {
  return [
    { value: 'random-line', label: 'Random named line each game' },
    { value: 'mixed', label: 'Mixed book candidates by position' },
    { value: 'none', label: 'No book / start position' },
    ...OPENING_BOOKS.map((book, idx) => ({ value: `line-${idx}` as ArenaOpeningMode, label: book.name })),
  ];
}
function arenaOpeningLabel(value = arenaOpeningMode) {
  if (value === 'random-line') return arenaBookLineIndex === null ? 'random named line each game' : `random: ${OPENING_BOOKS[arenaBookLineIndex]?.name ?? 'book line'}`;
  if (value === 'mixed') return 'mixed candidates';
  if (value === 'none') return 'none';
  const idx = Number(value.slice(5));
  return OPENING_BOOKS[idx]?.name ?? value;
}
function initArenaControls() {
  const white = document.getElementById('arenaWhiteSelect') as HTMLSelectElement | null;
  const black = document.getElementById('arenaBlackSelect') as HTMLSelectElement | null;
  const opening = document.getElementById('arenaOpeningSelect') as HTMLSelectElement | null;
  const plies = document.getElementById('arenaBookPliesSelect') as HTMLSelectElement | null;
  const optionFor = ([key, model]: [string, PlayableModel], selected: string) => {
    const option = document.createElement('option');
    option.value = key;
    option.selected = key === selected;
    option.textContent = model.label;
    return option;
  };
  white?.replaceChildren(...Object.entries(models).map((entry) => optionFor(entry, arenaWhiteKey)));
  black?.replaceChildren(...Object.entries(models).map((entry) => optionFor(entry, arenaBlackKey)));
  opening?.replaceChildren(...arenaOpeningOptions().map((entry) => {
    const option = document.createElement('option');
    option.value = entry.value;
    option.textContent = entry.label;
    option.selected = entry.value === arenaOpeningMode;
    return option;
  }));
  plies?.replaceChildren(...[0, 2, 4, 6, 8, 10, 12].map((n) => {
    const option = document.createElement('option');
    option.value = String(n);
    option.textContent = n === 0 ? '0 — disabled' : `${n} plies`;
    option.selected = n === openingBookMaxPlies;
    return option;
  }));
  if (white) white.onchange = () => { arenaWhiteKey = modelKeyOrDefault(white.value, arenaWhiteKey); updateArenaControls(); };
  if (black) black.onchange = () => { arenaBlackKey = modelKeyOrDefault(black.value, arenaBlackKey); updateArenaControls(); };
  if (opening) opening.onchange = () => { arenaOpeningMode = parseArenaOpening(opening.value); arenaBookLineIndex = null; updateArenaControls(); };
  if (plies) plies.onchange = () => { openingBookMaxPlies = Math.max(0, Number(plies.value) || 0); updateArenaControls(); };
  document.getElementById('arenaStart')?.addEventListener('click', () => { void startArena(); });
  document.getElementById('arenaStop')?.addEventListener('click', () => stopArena());
  updateArenaControls();
}
function arenaLabel(key: string) {
  return models[key]?.label ?? key;
}
function updateArenaControls(message?: string) {
  const white = document.getElementById('arenaWhiteSelect') as HTMLSelectElement | null;
  const black = document.getElementById('arenaBlackSelect') as HTMLSelectElement | null;
  const opening = document.getElementById('arenaOpeningSelect') as HTMLSelectElement | null;
  const plies = document.getElementById('arenaBookPliesSelect') as HTMLSelectElement | null;
  const start = document.getElementById('arenaStart') as HTMLButtonElement | null;
  const stop = document.getElementById('arenaStop') as HTMLButtonElement | null;
  const book = document.getElementById('arenaBookToggle') as HTMLInputElement | null;
  const status = document.getElementById('arenaStatus');
  if (white) { white.value = arenaWhiteKey; white.disabled = arenaRunning || busy; }
  if (black) { black.value = arenaBlackKey; black.disabled = arenaRunning || busy; }
  if (opening) { opening.value = arenaOpeningMode; opening.disabled = arenaRunning || busy || !arenaUseBook; }
  if (plies) { plies.value = String(openingBookMaxPlies); plies.disabled = arenaRunning || busy || !arenaUseBook; }
  if (book) { book.checked = arenaUseBook; book.disabled = arenaRunning || busy; book.onchange = () => { arenaUseBook = book.checked; updateArenaControls(); }; }
  if (start) {
    start.onclick = () => { void startArena(); };
    start.disabled = arenaRunning || busy || !evaluator;
  }
  if (stop) {
    stop.onclick = () => stopArena();
    stop.disabled = !arenaRunning;
  }
  const bookText = arenaUseBook && arenaOpeningMode !== 'none' && openingBookMaxPlies > 0 ? `${arenaOpeningLabel()} through ply ${openingBookMaxPlies}` : 'off';
  const budget = searchBudgetMode === 'neural' ? `${visits} neural evals · max ${neuralMaxVisitsMultiplier}× visits` : `${visits} visits`;
  if (status) status.innerHTML = message ?? `White: <code>${escapeHtml(arenaWhiteKey)}</code><br>Black: <code>${escapeHtml(arenaBlackKey)}</code><br>${escapeHtml(budget)} · ${escapeHtml(searchModeLabel(requestedSearchMode))} · batch ${puctBatchSize} · max ${arenaMaxPlies} plies · book ${escapeHtml(bookText)}`;
}
function searchSummary() {
  const budget = searchBudgetMode === 'neural' ? `${visits} neural evals · max ${neuralMaxVisitsMultiplier}× visits` : `${visits} visits`;
  const cache = playableEvalCacheEnabled ? ` · cache ${playableEvalCacheEntries}` : '';
  const backend = ` · ep ${describeOrtBackendConfig()}`;
  const tree = puctRootReuseEnabled ? ` · tree${puctTranspositionsEnabled ? '+tt' : ''}` : '';
  const ponder = puctPonderEnabled ? ` · ponder ${puctPonderVisits}` : '';
  if (playMode !== 'puct') return `policy ${playMode}${cache}${backend}`;
  const policy = searchTuning.label;
  return `${budget} · batch ${puctBatchSize} · ${policy}${cache}${tree}${ponder}${backend}`;
}
function searchDetails() {
  if (playMode !== 'puct') return `policy ${playMode}`;
  const bits = [searchSummary(), `cpuct ${searchTuning.cpuct ?? 1.5}`, `fpu ${searchTuning.fpu ?? 0}`];
  if (searchTuning.policy !== 'classic') {
    bits.push(`av ${searchTuning.avWeight ?? 0}`);
    bits.push(`rank ${searchTuning.rankWeight ?? 0}`);
    bits.push(`regret ${searchTuning.regretWeight ?? 0}`);
  }
  if (searchTuning.cpuctSchedule === 'lc0-log') bits.push('lc0 cpuct');
  if (searchTuning.fpuStrategy === 'lc0-reduction') bits.push(`lc0 fpu ${searchTuning.fpuReduction ?? 0.330}`);
  if (searchBudgetMode === 'neural') bits.push(`adaptive neural · root cap ${neuralMaxVisitsMultiplier}×${adaptiveEarlyStop === 'root-dominance' ? ' · dominance stop' : ''}`);
  if (requestedPuctPolicy === 'auto' && !searchTuning.source) bits.push('auto search default');
  return bits.join(' · ');
}
function initRunConfigChips() {
  syncWebClientStore();
  const visitsChip = document.getElementById('visitsChip');
  const batchChip = document.getElementById('batchChip');
  if (visitsChip) visitsChip.textContent = playMode === 'puct' ? `${searchBudgetMode === 'neural' ? 'neural' : 'visits'} ${visits}` : `policy ${playMode}`;
  if (batchChip) batchChip.textContent = searchTuning.searchMode === 'monty_lc0both' ? 'monty+lc0' : searchTuning.searchMode;
  updatePlayerSideControls();
  updateIntroControls();
  const modelInfo = document.getElementById('modelInfo');
  if (modelInfo) modelInfo.innerHTML = `<code>${escapeHtml(selectedModelKey)}</code> · ${escapeHtml(selectedModel.label)} · ${escapeHtml(searchDetails())} · ${escapeHtml(playerSide)} · ${escapeHtml(playStyleLabel())} · ${timedGame ? escapeHtml(formatClock(selectedClockMs)) : 'untimed'}`;
  updateBrainHint();
}
function updatePlayerSideControls() {
  syncWebClientStore();
  const state = webClientStore.state;
  document.getElementById('playWhite')?.classList.toggle('active', state.playerSide === 'white');
  document.getElementById('playBlack')?.classList.toggle('active', state.playerSide === 'black');
  document.getElementById('introWhite')?.classList.toggle('active', state.playerSide === 'white');
  document.getElementById('introBlack')?.classList.toggle('active', state.playerSide === 'black');
}
function updateIntroControls() {
  syncWebClientStore();
  const state = webClientStore.state;
  const intro = document.getElementById('playIntro') as HTMLElement | null;
  if (intro) intro.hidden = state.uiMode !== 'play' || state.gameStarted;
  updatePlayerSideControls();
  document.getElementById('modeNormal')?.classList.toggle('active', state.playStyle === 'normal');
  document.getElementById('modeHandBrain')?.classList.toggle('active', state.playStyle === 'nibbler-brain');
  document.getElementById('modeModelHand')?.classList.toggle('active', state.playStyle === 'you-brain');
  document.getElementById('modeLocalAiBrain')?.classList.toggle('active', state.playStyle === 'local-ai-brain');
  document.getElementById('modeLocalAiHand')?.classList.toggle('active', state.playStyle === 'local-ai-hand');
  document.getElementById('timeOff')?.classList.toggle('active', !state.timedGame);
  document.getElementById('time5')?.classList.toggle('active', state.timedGame && state.selectedClockMs === 300_000);
  document.getElementById('time10')?.classList.toggle('active', state.timedGame && state.selectedClockMs === 600_000);
}
function isUserTurn() {
  return uiMode === 'play' && gameStarted && !arenaGame && !gameOverMessage && (isLocalPlayStyle() || board.turn === playerColorToMove());
}
function updateBrainHint() {
  const hint = document.getElementById('brainHint');
  if (!hint) return;
  if (uiMode !== 'play' || !gameStarted || playStyle === 'normal') {
    hint.textContent = '';
  } else if (playStyle === 'nibbler-brain' || playStyle === 'local-ai-brain') {
    const side = board.turn === 'w' ? 'White' : 'Black';
    const prefix = playStyle === 'local-ai-brain' ? `${side} to move · AI Brain` : 'You are Hand';
    if (isUserTurn() && brainPiece) hint.innerHTML = `${escapeHtml(prefix)} says: <b>${escapeHtml(PIECE_NAMES[brainPiece])}</b>. Human chooses the move.`;
    else if (isUserTurn()) hint.textContent = `${prefix} is choosing a piece…`;
    else hint.textContent = 'You are Hand: wait for Tiny Leela, then move the named piece.';
  } else if (!isUserTurn()) {
    hint.textContent = 'You are Brain: wait for Tiny Leela’s reply, then name a piece for your move.';
  } else {
    const side = board.turn === 'w' ? 'White' : 'Black';
    const prefix = playStyle === 'local-ai-hand' ? `${side} to move · human Brain` : 'You are Brain';
    const legalByRole = legalMovesByPiece();
    const buttons = PIECE_BUTTON_ORDER.map((role) => {
      const count = legalByRole.get(role)?.length ?? 0;
      const active = brainPiece === role ? ' active' : '';
      return `<button class="brain-piece${active}" type="button" data-brain-piece="${role}" ${count && !busy ? '' : 'disabled'}>${escapeHtml(PIECE_NAMES[role])}<span class="kicker">${count}</span></button>`;
    }).join('');
    hint.innerHTML = `<div>${escapeHtml(prefix)}. Name a piece; Tiny Leela acts as Hand and chooses the move.</div><div class="brain-piece-row">${buttons}</div>`;
  }
}
function clickBrainPiece(role: PieceRole) {
  if (playStyle !== 'you-brain' && playStyle !== 'local-ai-hand') return;
  void playModelHandPiece(role);
}
function playerColorToMove() {
  return playerSide === 'white' ? 'w' : 'b';
}
function legalMovesByPiece() {
  const grouped = new Map<PieceRole, Move[]>();
  for (const move of legalMoves(board)) {
    const role = board.squares[move.from]?.[1] as PieceRole | undefined;
    if (!role) continue;
    const moves = grouped.get(role) ?? [];
    moves.push(move);
    grouped.set(role, moves);
  }
  return grouped;
}
function legalMovesForPiece(role: PieceRole) {
  return legalMoves(board).filter((move) => board.squares[move.from]?.[1] === role);
}
function legalMovesForUser() {
  if (uiMode === 'play' && gameOverMessage) return [];
  const moves = legalMoves(board);
  if (uiMode !== 'play' || !isUserTurn()) return moves;
  if (playStyle === 'you-brain' || playStyle === 'local-ai-hand') return [];
  if (playStyle !== 'nibbler-brain' && playStyle !== 'local-ai-brain') return moves;
  if (!brainPiece) return [];
  return moves.filter((move) => board.squares[move.from]?.[1] === brainPiece);
}
function movableColor() {
  if (uiMode === 'play') {
    if (arenaGame || !gameStarted || busy || gameOverMessage || playStyle === 'you-brain' || playStyle === 'local-ai-hand') return undefined;
    if (playStyle === 'local-ai-brain') return board.turn === 'w' ? 'white' : 'black';
    return playerSide;
  }
  return busy ? undefined : (board.turn === 'w' ? 'white' : 'black');
}
function setPlayerSide(side: PlayerSide) {
  if (busy || playerSide === side) return;
  playerSide = side;
  initRunConfigChips();
  if (gameStarted && uiMode === 'play') render(showPlayIntro('Choose settings for the next game.'));
}
function setClockOption(ms: number | null) {
  if (busy) return;
  timedGame = ms !== null;
  if (ms !== null) selectedClockMs = ms;
  resetClocks();
  initRunConfigChips();
}
function setPlayStyle(style: PlayStyle) {
  if (busy || playStyle === style) return;
  playStyle = style;
  brainPiece = null;
  initRunConfigChips();
  if (gameStarted && uiMode === 'play') render(showPlayIntro('Choose settings for the next game.'));
}
function beginConfiguredGame() {
  if (busy) return;
  arenaGame = false;
  arenaRunning = false;
  arenaStopRequested = true;
  updateIntroControls();
  render(startPlayGame());
}
function initPlayerSideControls() {
  document.getElementById('playWhite')?.addEventListener('click', () => setPlayerSide('white'));
  document.getElementById('playBlack')?.addEventListener('click', () => setPlayerSide('black'));
  document.getElementById('introWhite')?.addEventListener('click', () => setPlayerSide('white'));
  document.getElementById('introBlack')?.addEventListener('click', () => setPlayerSide('black'));
  document.getElementById('modeNormal')?.addEventListener('click', () => setPlayStyle('normal'));
  document.getElementById('modeHandBrain')?.addEventListener('click', () => setPlayStyle('nibbler-brain'));
  document.getElementById('modeModelHand')?.addEventListener('click', () => setPlayStyle('you-brain'));
  document.getElementById('modeLocalAiBrain')?.addEventListener('click', () => setPlayStyle('local-ai-brain'));
  document.getElementById('modeLocalAiHand')?.addEventListener('click', () => setPlayStyle('local-ai-hand'));
  document.getElementById('timeOff')?.addEventListener('click', () => setClockOption(null));
  document.getElementById('time5')?.addEventListener('click', () => setClockOption(300_000));
  document.getElementById('time10')?.addEventListener('click', () => setClockOption(600_000));
  document.getElementById('startGame')?.addEventListener('click', () => beginConfiguredGame());
  document.getElementById('brainHint')?.addEventListener('click', (event) => {
    const button = (event.target as HTMLElement | null)?.closest<HTMLButtonElement>('[data-brain-piece]');
    const role = button?.dataset.brainPiece;
    if (role === 'p' || role === 'n' || role === 'b' || role === 'r' || role === 'q' || role === 'k') clickBrainPiece(role);
  });
  updateIntroControls();
}
function resetClocks() {
  whiteClockMs = selectedClockMs;
  blackClockMs = selectedClockMs;
  lastClockTick = performance.now();
}
function formatClock(ms: number) {
  if (!timedGame) return '∞';
  const clamped = Math.max(0, Math.ceil(ms / 1000));
  const minutes = Math.floor(clamped / 60);
  const seconds = clamped % 60;
  return `${minutes}:${seconds.toString().padStart(2, '0')}`;
}
function updateClockDisplay() {
  const playerClock = document.getElementById('playerClock');
  const engineClock = document.getElementById('engineClock');
  if (!playerClock || !engineClock) return;
  playerClock.textContent = formatClock(whiteClockMs);
  engineClock.textContent = formatClock(blackClockMs);
  playerClock.classList.toggle('active', timedGame && gameStarted && board.turn === 'w');
  engineClock.classList.toggle('active', timedGame && gameStarted && board.turn === 'b');
  playerClock.classList.toggle('low', timedGame && whiteClockMs <= 30_000);
  engineClock.classList.toggle('low', timedGame && blackClockMs <= 30_000);
}
function settleClock() {
  const now = performance.now();
  const elapsed = now - lastClockTick;
  lastClockTick = now;
  if (clockRunning && timedGame && gameStarted && evaluator && uiMode !== 'analysis') {
    if (board.turn === 'w') whiteClockMs = Math.max(0, whiteClockMs - elapsed);
    else blackClockMs = Math.max(0, blackClockMs - elapsed);
  }
  updateClockDisplay();
}
function tickClocks() {
  settleClock();
}
type UiMode = 'play' | 'analysis';
let uiMode: UiMode = 'play';
function setUiMode(mode: UiMode) {
  cancelAsyncWork();
  uiMode = mode;
  busy = false;
  document.body.style.cursor = '';
  if (mode === 'analysis') {
    arenaRunning = false;
    arenaStopRequested = true;
    arenaGame = false;
    pendingPremove = null;
    brainPiece = null;
    ground?.cancelPremove();
  }
  syncWebClientStore({ message: webClientStore.state.message });
  updateIntroControls();
  document.body.classList.toggle('analysis-mode', mode === 'analysis');
  document.getElementById('playModeBtn')?.classList.toggle('active', mode === 'play');
  document.getElementById('analysisModeBtn')?.classList.toggle('active', mode === 'analysis');
}
async function toggleUiMode() {
  const next = uiMode === 'play' ? 'analysis' : 'play';
  setUiMode(next);
  await render(next === 'analysis' ? 'Free board: drag legal moves or use the history controls.' : gameStarted ? 'Returned to game.' : 'Choose settings to start a game.');
}
function initUiMode() {
  setUiMode(params.get('view') === 'analysis' ? 'analysis' : 'play');
  const pill = document.querySelector('.mode-pill');
  pill?.addEventListener('click', (event) => {
    event.preventDefault();
    void toggleUiMode();
  });
  pill?.addEventListener('keydown', (event) => {
    if (!(event instanceof KeyboardEvent) || (event.key !== 'Enter' && event.key !== ' ')) return;
    event.preventDefault();
    void toggleUiMode();
  });
}
type SideTab = 'game' | 'eval' | 'setup';
let activeSideTab: SideTab = 'game';
function isSideTab(value: string | null): value is SideTab {
  return value === 'game' || value === 'eval' || value === 'setup';
}
function setSideTab(tab: SideTab) {
  if (activeSideTab === tab) {
    syncWebClientStore({ activeSideTab: tab });
    return;
  }
  activeSideTab = tab;
  syncWebClientStore({ activeSideTab: tab });
  document.querySelectorAll<HTMLElement>('[data-tab-panel]').forEach((panel) => {
    const active = panel.dataset.tabPanel === tab;
    panel.classList.toggle('active', active);
    panel.hidden = !active;
  });
  document.querySelectorAll<HTMLButtonElement>('[data-tab-button]').forEach((button) => {
    const active = button.dataset.tabButton === tab;
    button.classList.toggle('active', active);
    button.setAttribute('aria-selected', active ? 'true' : 'false');
  });
}
function initSideTabs() {
  document.querySelectorAll<HTMLButtonElement>('[data-tab-button]').forEach((button) => {
    const tab = button.dataset.tabButton;
    if (tab && isSideTab(tab)) {
      const sideTab = tab;
      button.addEventListener('click', () => setSideTab(sideTab));
    }
  });
  const requestedTab = params.get('tab');
  setSideTab(isSideTab(requestedTab) ? requestedTab : 'game');
}
function initNavAndShortcuts() {
  document.getElementById('modelsNav')?.addEventListener('click', () => {
    setSideTab('setup');
    document.getElementById('modelSelect')?.focus();
  });
  document.getElementById('docsNav')?.addEventListener('click', () => {
    setSideTab('setup');
    const panel = document.getElementById('docsPanel') as HTMLElement | null;
    if (panel) panel.hidden = false;
  });
  document.addEventListener('keydown', (event) => {
    const target = document.activeElement;
    if (target instanceof HTMLInputElement || target instanceof HTMLSelectElement || target instanceof HTMLButtonElement) return;
    if (event.key === 'Escape') { event.preventDefault(); void stopCurrentSearch(); }
    else if (event.key === 'a') { event.preventDefault(); void toggleUiMode(); }
    else if (event.key === 'f') { event.preventDefault(); onFlip(); }
    else if (event.key === 'n') { event.preventDefault(); onReset(); }
    else if (event.key === 'u' || event.key === 'Backspace') { event.preventDefault(); void takeBack(); }
    else if (event.key === ' ') { event.preventDefault(); if (uiMode === 'play' && !gameStarted) beginConfiguredGame(); else engineMove(); }
    else if (event.key === 's') { event.preventDefault(); setSideTab('game'); startStockfish(); }
    else if (event.key === '1') { event.preventDefault(); setSideTab('game'); }
    else if (event.key === '2') { event.preventDefault(); setSideTab('eval'); }
    else if (event.key === '3' || event.key === '4') { event.preventDefault(); setSideTab('setup'); }
    else if (event.key === 'ArrowLeft') { event.preventDefault(); navigateHistory(currentPly - 1); }
    else if (event.key === 'ArrowRight') { event.preventDefault(); navigateHistory(currentPly + 1); }
  });
}
function wdlPerspectiveName() {
  if (uiMode === 'play' && gameStarted && arenaGame) return board.turn === 'w' ? `White (${models[arenaWhiteKey]?.label ?? arenaWhiteKey})` : `Black (${models[arenaBlackKey]?.label ?? arenaBlackKey})`;
  if (uiMode === 'play' && gameStarted) return board.turn === playerColorToMove() ? 'You' : 'Tiny Leela';
  return board.turn === 'w' ? 'White to move' : 'Black to move';
}
function terminalPositionMessage(): string | null {
  if (uiMode !== 'play' || !gameStarted) return null;
  const draw = automaticDrawReason(board, historyFens);
  if (draw === 'threefold') return 'Draw by threefold repetition.';
  if (draw === 'fiftyMove') return 'Draw by the 50-move rule.';
  if (draw === 'insufficientMaterial') return 'Draw by insufficient material.';
  const moves = legalMoves(board);
  if (moves.length) return null;
  if (!inCheck(board)) return 'Draw by stalemate.';
  if (arenaGame) {
    const winnerColor = board.turn === 'w' ? 'Black' : 'White';
    const winnerKey = board.turn === 'w' ? arenaBlackKey : arenaWhiteKey;
    return `Checkmate. ${winnerColor} (${models[winnerKey]?.label ?? winnerKey}) wins.`;
  }
  const winner = board.turn === playerColorToMove() ? 'Tiny Leela wins' : 'You win';
  return `Checkmate. ${winner}.`;
}
function updateGameOverState(): string | null {
  const message = terminalPositionMessage();
  if (!message) return null;
  gameOverMessage = message;
  clockRunning = false;
  pendingPremove = null;
  brainPiece = null;
  ground?.cancelPremove();
  return message;
}
function renderWdl(wdl: [number, number, number]) {
  const perspective = wdlPerspectiveName();
  const parts = [
    { name: `${perspective} win`, value: wdl[0] ?? 0, cls: 'wdl-win' },
    { name: 'draw', value: wdl[1] ?? 0, cls: 'wdl-draw' },
    { name: `${perspective} loss`, value: wdl[2] ?? 0, cls: 'wdl-loss' },
  ];
  $('wdl').innerHTML = `<div class="wdl-stack">${parts.map((p)=>`<div class="wdl-seg ${p.cls}" title="${escapeHtml(p.name)}" style="width:${percentWidth(p.value * 100)}">${escapeHtml((p.value * 100).toFixed(0))}%</div>`).join('')}</div><div class="wdl-labels"><span>${escapeHtml(parts[0].name)}</span><span>draw</span><span>${escapeHtml(parts[2].name)}</span></div><div class="wdl-perspective">WDL is from the current side-to-move perspective, not White/Black colors.</div>`;
}
async function takeBack() {
  if (busy || uiMode !== 'play' || !gameStarted || currentPly <= 0) return;
  cancelAsyncWork();
  gameOverMessage = '';
  settleClock();
  const plies = board.turn === playerColorToMove() && currentPly >= 2 ? 2 : 1;
  const targetPly = Math.max(0, currentPly - plies);
  playedMoves = playedMoves.slice(0, targetPly);
  playedMoveSans = playedMoveSans.slice(0, targetPly);
  positionFens = positionFens.slice(0, targetPly + 1);
  clockSnapshots = clockSnapshots.slice(0, targetPly + 1);
  const fen = positionFens[targetPly] ?? START_FEN;
  currentPly = targetPly;
  board = parseFen(fen);
  lastMove = currentPly > 0 ? playedMoves[currentPly - 1] : null;
  pendingPremove = null;
  brainPiece = null;
  ground?.cancelPremove();
  restoreClockSnapshot(currentPly);
  syncHistoryFens();
  await render(plies === 2 ? 'Took back the last move pair.' : 'Took back the last move.');
}
async function navigateHistory(ply: number) {
  if (busy || arenaRunning) {
    cancelAsyncWork();
    busy = false;
    arenaRunning = false;
    arenaStopRequested = true;
    document.body.style.cursor = '';
  }
  const nextPly = Math.max(0, Math.min(playedMoves.length, ply));
  const fen = positionFens[nextPly];
  if (!fen) return;
  currentPly = nextPly;
  board = parseFen(fen);
  lastMove = currentPly > 0 ? playedMoves[currentPly - 1] : null;
  pendingPremove = null;
  brainPiece = null;
  if (uiMode === 'play' && gameStarted) {
    arenaGame = false;
    clockRunning = currentPly === playedMoves.length;
  }
  ground?.cancelPremove();
  restoreClockSnapshot(currentPly);
  syncHistoryFens();
  await render(currentPly === playedMoves.length ? 'Returned to live position.' : 'Viewing earlier position. Make a move to branch from here.');
}
function initMoveHistoryControls() {
  document.getElementById('pgn')?.addEventListener('click', (event) => {
    const button = (event.target as HTMLElement | null)?.closest<HTMLButtonElement>('[data-ply]');
    const ply = Number(button?.dataset.ply ?? NaN);
    if (Number.isFinite(ply)) navigateHistory(ply);
  });
  document.getElementById('histStart')?.addEventListener('click', () => navigateHistory(0));
  document.getElementById('histPrev')?.addEventListener('click', () => navigateHistory(currentPly - 1));
  document.getElementById('histNext')?.addEventListener('click', () => navigateHistory(currentPly + 1));
  document.getElementById('histEnd')?.addEventListener('click', () => navigateHistory(playedMoves.length));
}
function updateTakebackControl() {
  const button = document.getElementById('takeback') as HTMLButtonElement | null;
  if (!button) return;
  button.disabled = busy || !evaluator || uiMode !== 'play' || !gameStarted || currentPly <= 0;
  button.title = currentPly <= 0 ? 'No moves to take back' : 'Undo your last turn, including Tiny Leela’s reply when available';
  for (const id of ['engine']) {
    const engineButton = document.getElementById(id) as HTMLButtonElement | null;
    if (engineButton && uiMode === 'play') engineButton.disabled = engineButton.disabled || !!gameOverMessage;
  }
}
function renderMoves() {
  const cells: string[] = [];
  for (let i = 0; i < playedMoves.length; i += 2) {
    const whiteSan = playedMoveSans[i] ?? playedMoves[i] ?? '';
    const blackSan = playedMoveSans[i + 1] ?? playedMoves[i + 1] ?? '';
    cells.push(`<span class="moveno">${Math.floor(i / 2) + 1}.</span><button class="moveuci ${currentPly === i + 1 ? 'active' : ''}" type="button" data-ply="${i + 1}" title="${escapeHtml(playedMoves[i] ?? '')}">${escapeHtml(whiteSan)}</button><button class="moveuci ${currentPly === i + 2 ? 'active' : ''}" type="button" data-ply="${i + 2}" title="${escapeHtml(playedMoves[i + 1] ?? '')}" ${blackSan ? '' : 'disabled'}>${escapeHtml(blackSan)}</button>`);
  }
  $('pgn').innerHTML = cells.join('') || '<span class="muted">No moves yet.</span>';
  const cursor = document.getElementById('moveCursor');
  if (cursor) cursor.textContent = currentPly === playedMoves.length ? 'live' : `ply ${currentPly}/${playedMoves.length}`;
  for (const [id, disabled] of Object.entries({ histStart: currentPly === 0, histPrev: currentPly === 0, histNext: currentPly >= playedMoves.length, histEnd: currentPly >= playedMoves.length })) {
    const button = document.getElementById(id) as HTMLButtonElement | null;
    if (button) button.disabled = disabled;
  }
}
function renderMaterial() {
  const values: Record<string, number> = { p: 1, n: 3, b: 3, r: 5, q: 9 };
  const start: Record<string, number> = { p: 8, n: 2, b: 2, r: 2, q: 1 };
  const counts: Record<string, number> = { wp:0,wn:0,wb:0,wr:0,wq:0,wk:0,bp:0,bn:0,bb:0,br:0,bq:0,bk:0 };
  for (const pc of board.squares) if (pc) counts[pc] = (counts[pc] ?? 0) + 1;
  let whiteMissing = '', blackMissing = '', whiteMat = 0, blackMat = 0;
  const glyph: Record<string, string> = { p:'♟', n:'♞', b:'♝', r:'♜', q:'♛' };
  for (const p of ['q','r','b','n','p']) {
    whiteMat += (counts[`w${p}`] ?? 0) * values[p]; blackMat += (counts[`b${p}`] ?? 0) * values[p];
    whiteMissing += glyph[p].repeat(Math.max(0, start[p] - (counts[`w${p}`] ?? 0)));
    blackMissing += glyph[p].repeat(Math.max(0, start[p] - (counts[`b${p}`] ?? 0)));
  }
  const diff = whiteMat - blackMat;
  $('material').innerHTML = `<div class="score">${diff === 0 ? 'Even' : diff > 0 ? `White +${diff}` : `Black +${-diff}`}</div><div class="pill">White missing <span class="captured">${escapeHtml(whiteMissing || '—')}</span></div><div class="pill">Black missing <span class="captured">${escapeHtml(blackMissing || '—')}</span></div>`;
}
function renderStockfish() {
  const status = !stockfish ? 'Off. Click “Start Stockfish” for browser-side depth analysis.' : stockfishThinking ? `Thinking to depth ${stockfishDepth}…` : stockfishReady ? 'Ready.' : 'Loading…';
  const btn = document.getElementById('stockfishBtn') as HTMLButtonElement | null;
  if (btn) btn.textContent = stockfish ? 'Stockfish running' : 'Start Stockfish';
  $('stockfish').innerHTML = `<div>${escapeHtml(status)}</div><div class="score">${escapeHtml(stockfishScore || '—')}</div><div>Best: <span class="mono">${escapeHtml(stockfishBest || '—')}</span></div><div class="pv">PV: <span class="mono">${escapeHtml(stockfishPv || '—')}</span></div><div class="eval-note">Scores and lines are shown in SAN. Score is normalized to White/Black advantage.</div>`;
}
function cpLabel(loss: number | null): GameStockfishRow['label'] {
  if (loss === null || loss < 50) return 'ok';
  if (loss >= 300) return 'blunder';
  if (loss >= 150) return 'mistake';
  return 'inaccuracy';
}
function formatCp(cp: number | null) {
  if (cp === null) return '—';
  const pawns = cp / 100;
  if (Math.abs(pawns) < 0.005) return '0.00';
  return `${pawns > 0 ? '+' : ''}${pawns.toFixed(2)}`;
}
function renderGameStockfish() {
  const start = document.getElementById('gameStockfishBtn') as HTMLButtonElement | null;
  const cancel = document.getElementById('gameStockfishCancel') as HTMLButtonElement | null;
  if (start) start.disabled = gameStockfishRunning || playedMoves.length === 0;
  if (cancel) cancel.disabled = !gameStockfishRunning;
  const target = document.getElementById('gameStockfish');
  if (!target) return;
  if (!gameStockfishRows.length) {
    target.innerHTML = `<div>${escapeHtml(gameStockfishMessage)}</div>`;
    return;
  }
  const cps = [gameStockfishRows[0]?.cpBefore, ...gameStockfishRows.map((row) => row.cpAfter)].map((cp) => cp ?? 0);
  const width = 320, height = 90, mid = height / 2, clamp = 800;
  const points = cps.map((cp, i) => {
    const x = cps.length <= 1 ? 0 : (i / (cps.length - 1)) * width;
    const y = mid - (Math.max(-clamp, Math.min(clamp, cp)) / clamp) * (height * 0.42);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(' ');
  const maxLoss = Math.max(50, ...gameStockfishRows.map((row) => row.loss ?? 0));
  const bars = gameStockfishRows.map((row) => `<span class="${row.label}" title="${escapeHtml(`${row.ply}. ${row.san}: ${row.loss ?? 0}cp ${row.label}`)}" style="height:${Math.max(2, Math.min(50, ((row.loss ?? 0) / maxLoss) * 50)).toFixed(0)}px"></span>`).join('');
  const flagged = gameStockfishRows.filter((row) => row.label !== 'ok').sort((a, b) => (b.loss ?? 0) - (a.loss ?? 0)).slice(0, 8);
  const avgLoss = gameStockfishRows.reduce((sum, row) => sum + (row.loss ?? 0), 0) / Math.max(1, gameStockfishRows.length);
  const counts = gameStockfishRows.reduce((acc, row) => { acc[row.label] = (acc[row.label] ?? 0) + 1; return acc; }, {} as Record<string, number>);
  target.innerHTML = `<div>${escapeHtml(gameStockfishMessage)}</div><svg class="sf-eval-graph" viewBox="0 0 ${width} ${height}" preserveAspectRatio="none"><line x1="0" x2="${width}" y1="${mid}" y2="${mid}" stroke="#d8cfbe"/><polyline fill="none" stroke="#2f5e4a" stroke-width="2" points="${points}"/></svg><div class="sf-loss-bars">${bars}</div><div class="eval-note">Avg loss ${avgLoss.toFixed(0)}cp · inaccuracies ${counts.inaccuracy ?? 0} · mistakes ${counts.mistake ?? 0} · blunders ${counts.blunder ?? 0}. Graph is White-centric; bars are mover cp loss.</div>${flagged.length ? `<ol class="sf-blunder-list">${flagged.map((row) => `<li><b>${row.label}</b> <code>${row.ply}. ${escapeHtml(row.san)}</code> loss ${row.loss}cp · eval ${escapeHtml(formatCp(row.cpBefore))} → ${escapeHtml(formatCp(row.cpAfter))} · SF: <code>${escapeHtml(row.bestSan || row.bestUci || '—')}</code></li>`).join('')}</ol>` : '<div class="eval-note">No ≥50cp losses found at this depth.</div>'}`;
}
function renderPuctPv() {
  const target = document.getElementById('puctPv');
  if (!target) return;
  if (!puctPvEnabled) {
    target.innerHTML = '<div class="muted">Disabled. Add <code>?pv=1</code> to surface the last PUCT search PV.</div>';
    return;
  }
  target.innerHTML = `<div class="pv">PV: <span class="mono">${escapeHtml(lastPuctPv || 'Run a PUCT engine move to populate PV.')}</span></div><div class="eval-note">${escapeHtml(lastPuctPvMeta || `Enabled · depth ${puctPvDepth} · selector ${puctPvSelector}`)}</div>`;
}
function formatPuctPv(fen: string, pv: PrincipalVariationEntry[] | undefined) {
  if (!pv?.length) return '';
  const ucis = pv.map((entry) => moveToUci(entry.move));
  return uciLineToSan(parseFen(fen), ucis, puctPvDepth);
}
function controlsEnabled(enabled: boolean) {
  for (const id of ['engine','engineAnalysis','takeback','reset','resetAnalysis','flip','flipAnalysis','loadFen','playWhite','playBlack','introWhite','introBlack','modeNormal','modeHandBrain','modeModelHand','modeLocalAiBrain','modeLocalAiHand','timeOff','time5','time10','startGame','modelSelect','visitsSelect','budgetModeSelect','searchModeSelect','arenaWhiteSelect','arenaBlackSelect','arenaOpeningSelect','arenaBookPliesSelect','arenaBookToggle','arenaStart','arenaStop','analyzeGame']) {
    const el = document.getElementById(id) as HTMLButtonElement | null;
    if (el) el.disabled = !enabled;
  }
  for (const id of ['stopSearch', 'stopSearchAnalysis']) {
    const button = document.getElementById(id) as HTMLButtonElement | null;
    if (button) button.disabled = !busy && !arenaRunning;
  }
  const analyzeButton = document.getElementById('analyzeGame') as HTMLButtonElement | null;
  if (analyzeButton) analyzeButton.disabled = !gameStarted && playedMoves.length === 0;
  const sfButton = document.getElementById('stockfishBtn') as HTMLButtonElement | null;
  if (sfButton) sfButton.disabled = !!stockfish;
  renderGameStockfish();
  updateArenaControls();
}
async function chooseBrainPieceForTurn(requestSeq = renderSeq) {
  if (!evaluator || uiMode !== 'play' || !gameStarted || (playStyle !== 'nibbler-brain' && playStyle !== 'local-ai-brain') || !isUserTurn()) {
    brainPiece = null;
    updateBrainHint();
    return;
  }
  if (brainPiece) { updateBrainHint(); return; }
  const moves = legalMoves(board);
  if (!moves.length) { updateBrainHint(); return; }
  const ev = await evaluator.evaluate(board, { historyFens });
  if (requestSeq !== renderSeq) return;
  const scores = new Map<PieceRole, number>();
  for (const move of moves) {
    const role = board.squares[move.from]?.[1] as PieceRole | undefined;
    if (!role) continue;
    scores.set(role, (scores.get(role) ?? 0) + Math.max(0, ev.policy.get(moveToActionId(move)) ?? 0));
  }
  const ranked = [...scores.entries()].sort((a, b) => b[1] - a[1]);
  brainPiece = ranked[0]?.[0] ?? (board.squares[moves[0].from]?.[1] as PieceRole | undefined) ?? null;
  updateBrainHint();
  ground?.set({ movable: { color: movableColor(), dests: legalDests() } });
}
async function render(message = '') {
  const seq = ++renderSeq;
  syncWebClientStore(message ? { message } : {});
  updateClockDisplay();
  $('fen').textContent = boardToFen(board);
  const statusText = evaluator ? `${selectedModel.label} · ${searchSummary()}${busy ? ' · thinking…' : ''}` : 'loading';
  $('status').textContent = statusText;
  const analysisStatus = document.getElementById('analysisStatus');
  if (analysisStatus) analysisStatus.textContent = statusText;
  controlsEnabled(!busy && !!evaluator && !arenaRunning);
  updateTakebackControl();
  if (gameOverMessage && uiMode === 'play') $('message').textContent = gameOverMessage;
  else if (message) $('message').textContent = message;
  renderMoves();
  renderMaterial();
  renderStockfish();
  renderGameStockfish();
  renderPuctPv();
  updateBrainHint();
  if (seq !== renderSeq) return;
  const moveColor = movableColor();
  updateIntroControls();
  const boardConfig = { orientation, fen: boardFen(), turnColor: board.turn === 'w' ? 'white' as const : 'black' as const, coordinates: true, highlight: { lastMove: true, check: true }, animation: { enabled: true, duration: 180 }, premovable: { enabled: uiMode === 'play' && gameStarted && !gameOverMessage && playStyle === 'normal', showDests: true, castle: true, events: { set: (from: Key, to: Key) => { pendingPremove = { from, to }; }, unset: () => { pendingPremove = null; } } }, predroppable: { enabled: false }, movable: { free: false, color: moveColor, dests: busy || (uiMode === 'play' && (!gameStarted || gameOverMessage)) ? new Map() : legalDests(), showDests: !busy && (uiMode !== 'play' || (gameStarted && !gameOverMessage)), events: { after: onUserMove } } };
  if (!ground) {
    ground = Chessground($('ground'), boardConfig);
  } else {
    ground.set({ ...boardConfig, lastMove: lastMove ? [lastMove.slice(0,2) as Key, lastMove.slice(2,4) as Key] : undefined });
  }
  void chooseBrainPieceForTurn(seq);
  if (!evaluator) return;
  const ev = await evaluator.evaluate(board, { historyFens });
  if (seq !== renderSeq) return;
  renderWdl(ev.wdl);
  const rows = legalMoves(board).map((m: Move) => ({ uci: moveToUci(m), san: moveToSan(board, m), prior: ev.policy.get(moveToActionId(m)) ?? 0 })).sort((a,b)=>b.prior-a.prior).slice(0,16);
  const maxPrior = Math.max(1e-9, ...rows.map((r) => r.prior));
  $('moves').innerHTML = rows.map((r, i)=>`<li class="policy-row ${i === 0 ? 'best' : ''}"><span class="rank">${i + 1}</span><b title="${escapeHtml(r.uci)}">${escapeHtml(r.san)}</b><span class="policy-meter"><span style="width:${percentWidth((r.prior / maxPrior) * 100, 2)}"></span></span><span class="pct">${escapeHtml((r.prior*100).toFixed(2))}%</span></li>`).join('');
  requestStockfishAnalysis();
}
function startStockfish() {
  if (stockfish) return;
  try {
    stockfish = new Worker('/stockfish/stockfish-18-lite-single.js');
    stockfish.onmessage = (ev) => handleStockfishLine(String(ev.data ?? ''));
    stockfish.onerror = () => { stockfishScore = 'Stockfish failed to load'; renderStockfish(); };
    sendStockfish('uci');
    renderStockfish();
  } catch (e) {
    stockfishScore = `Stockfish unavailable: ${(e as Error).message}`;
    renderStockfish();
  }
}
function sendStockfish(cmd: string) { stockfish?.postMessage(cmd); }
function formatStockfishScore(cp: RegExpMatchArray | null, mate: RegExpMatchArray | null, depth: RegExpMatchArray | null) {
  const whiteMultiplier = stockfishSearchTurn === 'w' ? 1 : -1;
  const depthSuffix = depth ? ` d${depth[1]}` : '';
  if (mate) {
    const whiteMate = whiteMultiplier * Number(mate[1]);
    return `${whiteMate >= 0 ? 'White' : 'Black'} M${Math.abs(whiteMate)}${depthSuffix}`;
  }
  if (cp) {
    const whitePawns = whiteMultiplier * Number(cp[1]) / 100;
    if (Math.abs(whitePawns) < 0.005) return `Equal${depthSuffix}`;
    return `${whitePawns > 0 ? 'White' : 'Black'} +${Math.abs(whitePawns).toFixed(2)}${depthSuffix}`;
  }
  return stockfishScore;
}
function handleStockfishLine(line: string) {
  if (line === 'uciok') { sendStockfish('isready'); return; }
  if (line === 'readyok') { stockfishReady = true; requestStockfishAnalysis(); renderStockfish(); return; }
  const best = line.match(/^bestmove\s+(\S+)/);
  if (best) { stockfishBest = uciToSan(parseFen(stockfishSearchFen), best[1]); stockfishThinking = false; renderStockfish(); return; }
  if (!line.startsWith('info ')) return;
  const pv = line.match(/\spv\s+(.+)$/);
  if (pv) stockfishPv = uciLineToSan(parseFen(stockfishSearchFen), pv[1].split(/\s+/), 12);
  const mate = line.match(/\sscore\s+mate\s+(-?\d+)/);
  const cp = line.match(/\sscore\s+cp\s+(-?\d+)/);
  const depth = line.match(/\sdepth\s+(\d+)/);
  stockfishScore = formatStockfishScore(cp, mate, depth);
  renderStockfish();
}
function requestStockfishAnalysis() {
  if (!stockfish || !stockfishReady || stockfishThinking) return;
  stockfishSeq += 1;
  stockfishThinking = true;
  stockfishSearchTurn = board.turn;
  stockfishSearchFen = boardToFen(board);
  stockfishBest = ''; stockfishPv = '';
  sendStockfish('stop');
  sendStockfish(`position fen ${stockfishSearchFen}`);
  sendStockfish(`go depth ${Math.max(1, stockfishDepth)}`);
  const seq = stockfishSeq;
  setTimeout(() => { if (seq === stockfishSeq && stockfishThinking) { sendStockfish('stop'); } }, 8000);
  renderStockfish();
}
function stockfishWhiteCpFromLine(line: string, fen: string): number | null {
  const turn = parseFen(fen).turn;
  const whiteMultiplier = turn === 'w' ? 1 : -1;
  const mate = line.match(/\sscore\s+mate\s+(-?\d+)/);
  if (mate) {
    const mateWhite = whiteMultiplier * Number(mate[1]);
    const magnitude = 100000 - Math.min(99, Math.abs(mateWhite)) * 100;
    return Math.sign(mateWhite || 1) * magnitude;
  }
  const cp = line.match(/\sscore\s+cp\s+(-?\d+)/);
  return cp ? whiteMultiplier * Number(cp[1]) : null;
}
function initStockfishWorker(worker: Worker): Promise<void> {
  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => { cleanup(); reject(new Error('Stockfish init timed out')); }, 10000);
    const cleanup = () => { clearTimeout(timeout); worker.removeEventListener('message', onMessage); worker.removeEventListener('error', onError); };
    const onError = () => { cleanup(); reject(new Error('Stockfish worker failed')); };
    const onMessage = (event: MessageEvent) => {
      const line = String(event.data ?? '');
      if (line === 'uciok') worker.postMessage('isready');
      if (line === 'readyok') { cleanup(); resolve(); }
    };
    worker.addEventListener('message', onMessage);
    worker.addEventListener('error', onError);
    worker.postMessage('uci');
  });
}
function evaluateFenWithWorker(worker: Worker, fen: string, depth: number): Promise<{ cpWhite: number | null; bestUci: string; bestSan: string }> {
  return new Promise((resolve) => {
    let cpWhite: number | null = null;
    let bestUci = '';
    let done = false;
    const finish = () => {
      if (done) return;
      done = true;
      cleanup();
      let bestSan = bestUci;
      try { if (bestUci) bestSan = uciToSan(parseFen(fen), bestUci); } catch {}
      resolve({ cpWhite, bestUci, bestSan });
    };
    const timeout = setTimeout(() => { worker.postMessage('stop'); setTimeout(finish, 1000); }, Math.max(4000, depth * 1000));
    const cancelPoll = setInterval(() => { if (gameStockfishCancelRequested) { worker.postMessage('stop'); finish(); } }, 100);
    const cleanup = () => { clearTimeout(timeout); clearInterval(cancelPoll); worker.removeEventListener('message', onMessage); };
    const onMessage = (event: MessageEvent) => {
      const line = String(event.data ?? '');
      if (line.startsWith('info ')) {
        const parsed = stockfishWhiteCpFromLine(line, fen);
        if (parsed !== null) cpWhite = parsed;
      }
      const best = line.match(/^bestmove\s+(\S+)/);
      if (best) { bestUci = best[1] === '(none)' ? '' : best[1]; finish(); }
    };
    worker.addEventListener('message', onMessage);
    worker.postMessage('stop');
    worker.postMessage(`position fen ${fen}`);
    worker.postMessage(`go depth ${Math.max(1, depth)}`);
  });
}
function lossForMove(cpBefore: number | null, cpAfter: number | null, fenBefore: string): number | null {
  if (cpBefore === null || cpAfter === null) return null;
  const turn = parseFen(fenBefore).turn;
  const raw = turn === 'w' ? cpBefore - cpAfter : cpAfter - cpBefore;
  return Math.max(0, Math.round(Math.min(2000, raw)));
}
async function startGameStockfishAnalysis() {
  if (gameStockfishRunning || playedMoves.length === 0) return;
  gameStockfishWorker?.terminate();
  gameStockfishWorker = new Worker('/stockfish/stockfish-18-lite-single.js');
  gameStockfishRunning = true;
  gameStockfishCancelRequested = false;
  gameStockfishRows = [];
  const fens = [...positionFens];
  const moves = [...playedMoves];
  const sans = [...playedMoveSans];
  try {
    gameStockfishMessage = `Starting Stockfish review for ${moves.length} plies…`;
    renderGameStockfish();
    await initStockfishWorker(gameStockfishWorker);
    const evals: { cpWhite: number | null; bestUci: string; bestSan: string }[] = [];
    for (let i = 0; i < fens.length; i += 1) {
      if (gameStockfishCancelRequested) throw new Error('cancelled');
      gameStockfishMessage = `Stockfish review: position ${i + 1}/${fens.length} at depth ${stockfishDepth}…`;
      renderGameStockfish();
      evals.push(await evaluateFenWithWorker(gameStockfishWorker, fens[i], Math.max(1, stockfishDepth)));
    }
    gameStockfishRows = moves.map((uci, idx) => {
      const cpBefore = evals[idx]?.cpWhite ?? null;
      const cpAfter = evals[idx + 1]?.cpWhite ?? null;
      const loss = lossForMove(cpBefore, cpAfter, fens[idx]);
      const label = cpLabel(loss);
      return { ply: idx + 1, san: sans[idx] ?? uci, uci, fen: fens[idx], cpBefore, cpAfter, loss, bestSan: evals[idx]?.bestSan ?? '', bestUci: evals[idx]?.bestUci ?? '', label };
    });
    const blunders = gameStockfishRows.filter((row) => row.label === 'blunder').length;
    const mistakes = gameStockfishRows.filter((row) => row.label === 'mistake').length;
    gameStockfishMessage = `Review complete: ${moves.length} plies, ${mistakes} mistakes, ${blunders} blunders at depth ${stockfishDepth}.`;
  } catch (e) {
    gameStockfishMessage = gameStockfishCancelRequested || isAbortError(e) || /cancelled/i.test((e as Error).message) ? 'Stockfish game review cancelled.' : `Stockfish review failed: ${(e as Error).message}`;
  } finally {
    gameStockfishRunning = false;
    gameStockfishCancelRequested = false;
    gameStockfishWorker?.terminate();
    gameStockfishWorker = null;
    renderGameStockfish();
  }
}
function cancelGameStockfishAnalysis() {
  gameStockfishCancelRequested = true;
  gameStockfishWorker?.postMessage('stop');
  gameStockfishMessage = 'Cancelling Stockfish game review…';
  renderGameStockfish();
}
async function playMove(move: Move, who: string): Promise<boolean> {
  settleClock();
  brainPiece = null;
  const san = moveToSan(board, move);
  recordMove(move, san);
  const terminal = updateGameOverState();
  await render(terminal ? `${who} played ${san}. ${terminal}` : `${who} played ${san}.`);
  return !!terminal;
}
function resolveUserMove(from: string, to: string) {
  const candidates = legalMovesForUser().filter((m) => squareName(m.from) === from && squareName(m.to) === to);
  return candidates.find((m) => moveToUci(m) === from + to) ?? candidates.find((m) => moveToUci(m).endsWith('q')) ?? candidates[0] ?? null;
}
async function applyPendingPremove() {
  if (!pendingPremove || busy || gameOverMessage || uiMode !== 'play' || board.turn !== playerColorToMove()) return false;
  const { from, to } = pendingPremove;
  pendingPremove = null;
  ground?.cancelPremove();
  const move = resolveUserMove(from, to);
  if (!move) { await render(`Premove ${from}${to} is no longer legal.`); return false; }
  await playMove(move, 'Premove');
  await engineMove();
  return true;
}
async function onUserMove(from: string, to: string) {
  if (busy || (uiMode === 'play' && (!gameStarted || gameOverMessage))) return;
  const move = resolveUserMove(from, to);
  if (!move) { await render((playStyle === 'nibbler-brain' || playStyle === 'local-ai-brain') && brainPiece ? `Brain said ${PIECE_NAMES[brainPiece]}; ${from}${to} is not available.` : `Illegal move ${from}${to}.`); return; }
  const ended = await playMove(move, uiMode === 'analysis' ? 'Analysis' : 'You');
  if (uiMode === 'play' && !ended && !isLocalPlayStyle()) await engineMove();
}
async function choosePolicyMove(rootMoves?: Move[]): Promise<Move | null> {
  if (!evaluator) return null;
  const moves = rootMoves ?? legalMoves(board);
  if (!moves.length) return null;
  const ev = await evaluator.evaluate(board, { historyFens, legalMoves: moves });
  let rows = moves.map((move) => ({ move, p: Math.max(0, ev.policy.get(moveToActionId(move)) ?? 0) })).sort((a,b)=>b.p-a.p);
  if (playMode === 'argmax' || temperature <= 0) return rows[0].move;
  if (topK > 0) rows = rows.slice(0, Math.max(1, topK));
  if (topP < 1) {
    const total = rows.reduce((s,r)=>s+r.p,0) || 1;
    let acc = 0, cut = rows.length;
    for (let i=0;i<rows.length;i++) { acc += rows[i].p / total; if (acc >= Math.max(0.01, topP)) { cut = i + 1; break; } }
    rows = rows.slice(0, Math.max(1, cut));
  }
  const weights = rows.map((r)=>Math.pow(Math.max(r.p, 1e-12), 1 / Math.max(1e-6, temperature)));
  const total = weights.reduce((a,b)=>a+b,0);
  let pick = Math.random() * total;
  for (let i=0;i<rows.length;i++) { pick -= weights[i]; if (pick <= 0) return rows[i].move; }
  return rows[0].move;
}
async function runtimeDiagnosticsForModel(model: PlayableModel): Promise<OrtRuntimeDiagnostics | undefined> {
  if (!model.runtime) return undefined;
  const providers = resolvedOrtExecutionProviders();
  return webQueryClient.fetchQuery({
    queryKey: ['ort-runtime-diagnostics', providers.join(','), providers.includes('webgpu')],
    queryFn: () => collectOrtRuntimeDiagnostics({ probeAdapter: providers.includes('webgpu') }),
    staleTime: Infinity,
    gcTime: Infinity,
  });
}
async function modelMetaForArtifact(artifact: SelectedPlayableArtifact): Promise<OnnxStudentMeta | SquareFormerMeta> {
  return webQueryClient.fetchQuery({
    queryKey: ['playable-model-meta', artifact.meta],
    queryFn: async () => {
      const response = await fetch(artifact.meta);
      if (!response.ok) throw new Error(`Failed to load ${artifact.meta}: ${response.status} ${response.statusText}`);
      return response.json() as Promise<OnnxStudentMeta | SquareFormerMeta>;
    },
    staleTime: Infinity,
    gcTime: Infinity,
  });
}
async function createEvaluatorForModelUncached(model: PlayableModel): Promise<Evaluator> {
  const t0 = tinyLeelaNowMs();
  const runtimeDiagnostics = await runtimeDiagnosticsForModel(model);
  const artifact = selectPlayableArtifact(model, runtimeDiagnostics);
  logPlayableRuntimeSelection(model, artifact, runtimeDiagnostics);
  const tRuntime = tinyLeelaNowMs();
  const meta = await modelMetaForArtifact(artifact);
  const tMeta = tinyLeelaNowMs();
  const workerEval = browserWorkerEvaluatorEnabled();
  const base = workerEval
    ? await WorkerEvaluator.create(artifact.onnx, meta)
    : (meta.kind === 'squareformer' || meta.kind === 'squareformer_v2')
      ? await SquareFormerEvaluator.create(artifact.onnx, meta as SquareFormerMeta)
      : await OnnxEvaluator.create(artifact.onnx, meta as OnnxStudentMeta);
  const tSession = tinyLeelaNowMs();
  const wrapped = playableEvalCacheEnabled ? new CachedEvaluator(base, {
    maxEntries: playableEvalCacheEntries,
    includeHistory: true,
    includeLegalMoves: true,
    label: artifact.label,
  }) : base;
  tinyLeelaLogLatency('webClient.createEvaluatorForModel', {
    label: model.label,
    artifactLabel: artifact.label,
    model: artifact.onnx,
    meta: artifact.meta,
    artifactVariant: artifact.variantKey,
    preferredEp: artifact.preferredEp,
    estimatedBytes: artifact.estimatedBytes,
    kind: meta.kind ?? 'onnx',
    workerEval,
    runtimeProbeMs: tRuntime - t0,
    metaFetchMs: tMeta - tRuntime,
    sessionCreateMs: tSession - tMeta,
    totalMs: tSession - t0,
    evalCache: playableEvalCacheEnabled,
  });
  return wrapped;
}
async function createEvaluatorForModel(model: PlayableModel, _modelKey = model.label): Promise<Evaluator> {
  // Query Core caches lightweight metadata/runtime probes above. Do not place
  // live ONNX sessions or worker-backed evaluators in the Query cache: those
  // objects own browser/GPU/worker resources and need explicit lifetime
  // management by the playable/arena evaluator owners.
  return createEvaluatorForModelUncached(model);
}
async function arenaEvaluatorFor(key: string): Promise<Evaluator> {
  if (key === selectedModelKey && evaluator) return evaluator;
  const cached = arenaEvaluatorCache.get(key);
  if (cached) return cached;
  const model = models[key];
  if (!model) throw new Error(`Unknown arena model: ${key}`);
  const loaded = await createEvaluatorForModel(model, key);
  arenaEvaluatorCache.set(key, loaded);
  return loaded;
}
function delay(ms: number) { return new Promise((resolve) => setTimeout(resolve, ms)); }
function puctBudgetStatsText(stats: SearchStats | undefined): string {
  if (!stats) return '';
  const bits: string[] = [];
  if (stats.budgetMode === 'neural') {
    bits.push(`neural misses ${stats.neuralEvalMisses ?? 0}/${stats.requestedNeuralEvals ?? visits}`);
    bits.push(`cache hits ${stats.cacheHits ?? 0}`);
    if (stats.maxRootVisits !== undefined) bits.push(`cap ${stats.maxRootVisits} root visits`);
  }
  if (stats.stopReason && stats.stopReason !== 'visit-budget') bits.push(stats.stopReason);
  return bits.length ? ` · ${bits.join(' · ')}` : '';
}
function puctSearchMeta(prefix: string, result: { visits: number; stats?: SearchStats }, suffix = ''): string {
  const budget = result.stats?.budgetMode === 'neural' ? `${result.visits} realized visits` : `${result.visits} visits`;
  return `${prefix} · ${budget}${puctBudgetStatsText(result.stats)}${suffix}`;
}
function kqkEndgameMeta(prefix: string, mateInPlies: number | null): string {
  return `${prefix} · KQK endgame oracle${mateInPlies === null ? '' : ` · forced mate in ${mateInPlies} ply`}`;
}
async function chooseArenaMove(modelEval: Evaluator, modelKey: string): Promise<Move | null> {
  const rootMoves = legalMoves(board);
  if (!rootMoves.length) return null;
  const endgameMove = chooseKingQueenVsKingMove(board);
  if (endgameMove) {
    lastPuctPv = moveToSan(board, endgameMove.move);
    lastPuctPvMeta = kqkEndgameMeta(arenaLabel(modelKey), endgameMove.mateInPlies);
    return endgameMove.move;
  }
  const tuning = resolveSearchTuning(modelKey, visits, requestedPuctPolicy, models[modelKey]);
  if (playMode === 'puct') {
    const searchFen = boardToFen(board);
    const result = await chooseMove(board, modelEval, {
      visits,
      batchSize: puctBatchSize,
      historyFens,
      ...searchOptionOverrides(tuning),
      includePv: puctPvEnabled,
      pvDepth: puctPvDepth,
      pvSelector: puctPvSelector,
      signal: currentSearchSignal(),
      yieldEveryMs: 24,
    });
    if (puctPvEnabled) {
      lastPuctPv = formatPuctPv(searchFen, result.principalVariation);
      lastPuctPvMeta = puctSearchMeta(arenaLabel(modelKey), result, ` · selector ${puctPvSelector}`);
    }
    return result.move;
  }
  const ev = await modelEval.evaluate(board, { historyFens, legalMoves: rootMoves });
  const rows = rootMoves.map((move) => ({ move, p: Math.max(0, ev.policy.get(moveToActionId(move)) ?? 0) })).sort((a, b) => b.p - a.p);
  return rows[0]?.move ?? null;
}
async function startArena() {
  if (arenaRunning || busy) return;
  cancelAsyncWork();
  setUiMode('play');
  setSideTab('game');
  arenaRunning = true;
  arenaStopRequested = false;
  arenaGame = true;
  gameStarted = true;
  gameOverMessage = '';
  clockRunning = false;
  pendingPremove = null;
  brainPiece = null;
  orientation = 'white';
  resetPositionState();
  arenaBookLineIndex = arenaOpeningMode === 'random-line' ? Math.floor(Math.random() * OPENING_BOOK_LINES.length) : null;
  try {
    updateArenaControls('Loading arena models…');
    await render(`Arena loading: White ${arenaLabel(arenaWhiteKey)} vs Black ${arenaLabel(arenaBlackKey)}.`);
    const whiteEval = await arenaEvaluatorFor(arenaWhiteKey);
    const blackEval = await arenaEvaluatorFor(arenaBlackKey);
    await render(`Arena started: White ${arenaLabel(arenaWhiteKey)} vs Black ${arenaLabel(arenaBlackKey)}.`);
    while (arenaRunning && !arenaStopRequested && !gameOverMessage && currentPly < arenaMaxPlies) {
      const turnIsWhite = board.turn === 'w';
      const key = turnIsWhite ? arenaWhiteKey : arenaBlackKey;
      const modelEval = turnIsWhite ? whiteEval : blackEval;
      const who = `${turnIsWhite ? 'White' : 'Black'} (${arenaLabel(key)})`;
      const requestId = beginEngineRequest();
      busy = true;
      document.body.style.cursor = 'progress';
      updateArenaControls(`${escapeHtml(who)} thinking…<br>${currentPly}/${arenaMaxPlies} plies`);
      await render(`${who} thinking…`);
      const bookMove = chooseArenaBookMove();
      const move = bookMove ?? await chooseArenaMove(modelEval, key);
      busy = false;
      document.body.style.cursor = '';
      if (!isCurrentEngineRequest(requestId) || arenaStopRequested) break;
      if (!move) { await render(`Arena stopped: ${who} has no legal move.`); break; }
      await playMove(move, who);
      if (arenaDelayMs) await delay(arenaDelayMs);
    }
    if (!gameOverMessage && currentPly >= arenaMaxPlies) {
      gameOverMessage = `Arena reached max ${arenaMaxPlies} plies; adjudicate manually.`;
      await render(gameOverMessage);
    } else if (arenaStopRequested) {
      await render('Arena stopped.');
    }
  } catch (e) {
    if (isAbortError(e)) await render('Arena stopped.');
    else {
      console.error(e);
      await render(`Arena failed: ${(e as Error).message}`);
    }
  } finally {
    arenaRunning = false;
    arenaStopRequested = false;
    busy = false;
    document.body.style.cursor = '';
    updateArenaControls();
    await render();
  }
}
function stopArena() {
  arenaStopRequested = true;
  arenaRunning = false;
  searchAbortController?.abort();
  updateArenaControls('Stopping search…');
}
async function stopCurrentSearch() {
  if (!busy && !arenaRunning) return;
  cancelAsyncWork();
  arenaStopRequested = true;
  arenaRunning = false;
  busy = false;
  document.body.style.cursor = '';
  updateArenaControls('Stopped.');
  await render('Search stopped.');
}
async function analyzeCurrentGame() {
  cancelAsyncWork();
  arenaStopRequested = true;
  arenaRunning = false;
  arenaGame = false;
  busy = false;
  clockRunning = false;
  document.body.style.cursor = '';
  setUiMode('analysis');
  setSideTab('game');
  await render('Free board: current game loaded. Move list, WDL, policy and PVs stay together while you step through history.');
}
async function playModelHandPiece(role: PieceRole) {
  if (!evaluator || busy || uiMode !== 'play' || !gameStarted || gameOverMessage || !isUserTurn()) return;
  const rootMoves = legalMovesForPiece(role);
  if (!rootMoves.length) { await render(`No legal ${PIECE_NAMES[role]} moves.`); return; }
  const requestId = beginEngineRequest();
  brainPiece = role;
  busy = true;
  document.body.style.cursor = 'progress';
  let moved = false;
  let ended = false;
  try {
    await render(`Tiny Leela Hand is choosing a ${PIECE_NAMES[role]} move…`);
    if (!isCurrentEngineRequest(requestId)) return;
    const searchFen = boardToFen(board);
    let move: Move | null;
    if (playMode === 'puct') {
      const result = await chooseMove(board, evaluator, {
        visits,
        batchSize: puctBatchSize,
        historyFens,
        ...searchOptionOverrides(searchTuning),
        includePv: puctPvEnabled,
        pvDepth: puctPvDepth,
        pvSelector: puctPvSelector,
        rootMoves,
        signal: currentSearchSignal(),
        yieldEveryMs: 24,
      });
      if (!isCurrentEngineRequest(requestId)) return;
      move = result.move;
      if (puctPvEnabled) {
        lastPuctPv = formatPuctPv(searchFen, result.principalVariation);
        lastPuctPvMeta = puctSearchMeta(`Tiny Leela Hand restricted to ${PIECE_NAMES[role]}`, result, ` · selector ${puctPvSelector}`);
      }
    } else {
      move = await choosePolicyMove(rootMoves);
      if (puctPvEnabled) {
        lastPuctPv = '';
        lastPuctPvMeta = `Tiny Leela Hand restricted to ${PIECE_NAMES[role]} · mode ${playMode}; no PUCT PV was run.`;
      }
    }
    if (!isCurrentEngineRequest(requestId)) return;
    if (move) {
      ended = await playMove(move, `Tiny Leela Hand (${PIECE_NAMES[role]})`);
      moved = true;
    } else {
      await render(`No legal ${PIECE_NAMES[role]} move.`);
    }
  } catch (e) {
    if (!isCurrentEngineRequest(requestId)) return;
    if (isAbortError(e)) await render('Search stopped.');
    else {
      console.error(e);
      await render(`Tiny Leela Hand failed: ${(e as Error).message}`);
    }
  } finally {
    busy = false;
    document.body.style.cursor = '';
    await render();
  }
  if (moved && !ended && playStyle === 'you-brain' && isCurrentEngineRequest(requestId)) await engineMove();
}
async function startPuctPonderIfUseful() {
  if (!puctPonderEnabled || !puctRootReuseEnabled || !evaluator || playMode !== 'puct' || gameOverMessage) return;
  if (uiMode === 'play' && (!gameStarted || board.turn !== playerColorToMove())) return;
  const seq = ++puctPonderSeq;
  puctPonderAbortController?.abort();
  puctPonderAbortController = new AbortController();
  const ponderBoard = parseFen(boardToFen(board));
  const ponderHistory = [...historyFens];
  const ponderRoot = puctSearchRoot;
  try {
    const result = await chooseMove(ponderBoard, evaluator, {
      visits: puctPonderVisits,
      batchSize: puctBatchSize,
      historyFens: ponderHistory,
      ...searchOptionOverrides(searchTuning),
      root: ponderRoot,
      signal: puctPonderAbortController.signal,
      yieldEveryMs: puctPonderChunkMs,
    });
    if (seq !== puctPonderSeq || !result.move) return;
    const predictedBoard = makeMove(ponderBoard, result.move);
    const predictedHistory = [boardToFen(ponderBoard), ...ponderHistory].slice(0, 8);
    const predictedRoot = advanceSearchRoot(result.root ?? null, result.move, predictedBoard, predictedHistory);
    puctPonderPrediction = {
      uci: moveToUci(result.move),
      fen: boardToFen(predictedBoard),
      historyKey: predictedHistory.join('\n'),
      root: predictedRoot,
      visits: result.visits,
    };
    lastPuctPvMeta = `${lastPuctPvMeta}${lastPuctPvMeta ? ' · ' : ''}ponder ${puctPonderPrediction.uci} @ ${result.visits}v${puctBudgetStatsText(result.stats)}`;
    await render();
  } catch (e) {
    if (!isAbortError(e)) console.warn('Ponder search failed', e);
  }
}

async function engineMove() {
  if (!evaluator || busy || (uiMode === 'play' && (!gameStarted || gameOverMessage))) return;
  const requestId = beginEngineRequest();
  busy = true;
  document.body.style.cursor = 'progress';
  await render('Engine thinking…');
  try {
    if (!isCurrentEngineRequest(requestId)) return;
    const endgameMove = chooseKingQueenVsKingMove(board);
    if (endgameMove) {
      lastPuctPv = moveToSan(board, endgameMove.move);
      lastPuctPvMeta = kqkEndgameMeta('Endgame override', endgameMove.mateInPlies);
      if (!isCurrentEngineRequest(requestId)) return;
      await playMove(endgameMove.move, 'KQK oracle');
      return;
    }
    const bookMove = chooseOpeningBookMove();
    let move: Move | null = bookMove;
    if (bookMove) {
      if (puctPvEnabled && playMode === 'puct') {
        const searchFen = boardToFen(board);
        const result = await chooseMove(board, evaluator, {
          visits,
          batchSize: puctBatchSize,
          historyFens,
          ...searchOptionOverrides(searchTuning),
          includePv: true,
          pvDepth: puctPvDepth,
          pvSelector: puctPvSelector,
          root: puctRootReuseEnabled ? puctSearchRoot : null,
          transpositionTable: puctTranspositionsEnabled ? puctTranspositionTable : undefined,
          signal: currentSearchSignal(),
          yieldEveryMs: 24,
        });
        puctSearchRoot = puctRootReuseEnabled ? result.root ?? null : null;
        if (!isCurrentEngineRequest(requestId)) return;
        lastPuctPv = formatPuctPv(searchFen, result.principalVariation);
        const puctMove = result.move ? moveToSan(parseFen(searchFen), result.move) : 'none';
        lastPuctPvMeta = puctSearchMeta(`Book move selected; diagnostic PUCT PV · PUCT would play ${puctMove}`, result, ` · selector ${puctPvSelector}`);
      } else if (puctPvEnabled) {
        lastPuctPv = '';
        lastPuctPvMeta = 'Book move selected; no PUCT search was run.';
      }
    } else if (playMode === 'puct') {
      const searchFen = boardToFen(board);
      const result = await chooseMoveMaybeWorker(board, evaluator, {
        visits,
        batchSize: puctBatchSize,
        historyFens,
        ...searchOptionOverrides(searchTuning),
        includePv: puctPvEnabled,
        pvDepth: puctPvDepth,
        pvSelector: puctPvSelector,
        root: puctRootReuseEnabled ? puctSearchRoot : null,
        transpositionTable: puctTranspositionsEnabled ? puctTranspositionTable : undefined,
        signal: currentSearchSignal(),
        yieldEveryMs: 24,
      }, searchTuning);
      puctSearchRoot = puctRootReuseEnabled ? result.root ?? null : null;
      if (!isCurrentEngineRequest(requestId)) return;
      move = result.move;
      if (puctPvEnabled) {
        lastPuctPv = formatPuctPv(searchFen, result.principalVariation);
        lastPuctPvMeta = puctSearchMeta('Last PUCT search', result, ` · depth ${puctPvDepth} · selector ${puctPvSelector}`);
      }
    } else {
      move = await choosePolicyMove();
      if (puctPvEnabled) {
        lastPuctPv = '';
        lastPuctPvMeta = `Mode ${playMode} selected a policy move; no PUCT PV was run.`;
      }
    }
    if (!isCurrentEngineRequest(requestId)) return;
    if (move) await playMove(move, bookMove ? 'Book' : 'Engine');
    else await render('No legal engine move.');
  } catch (e) {
    if (!isCurrentEngineRequest(requestId)) return;
    if (isAbortError(e)) await render('Search stopped.');
    else {
      console.error(e);
      await render(`Engine failed: ${(e as Error).message}`);
    }
  } finally {
    busy = false;
    document.body.style.cursor = '';
    await render();
    if (isCurrentEngineRequest(requestId) && !gameOverMessage && !pendingPremove) void startPuctPonderIfUseful();
    if (isCurrentEngineRequest(requestId) && !gameOverMessage) await applyPendingPremove();
  }
}
const onReset = async () => { if (busy) return; await render(uiMode === 'analysis' ? startAnalysisBoard() : showPlayIntro('Choose settings for a new game.')); };
const onResetAnalysis = async () => { if (busy) return; await render(startAnalysisBoard()); };
const onFlip = async () => { if (busy) return; orientation = orientation === 'white' ? 'black' : 'white'; await render(); };
$('engine').onclick = () => engineMove();
$('engineAnalysis').onclick = () => engineMove();
$('stopSearch').onclick = () => { void stopCurrentSearch(); };
$('stopSearchAnalysis').onclick = () => { void stopCurrentSearch(); };
$('stockfishBtn').onclick = () => { setSideTab('game'); startStockfish(); };
$('gameStockfishBtn').onclick = () => { setSideTab('eval'); void startGameStockfishAnalysis(); };
$('gameStockfishCancel').onclick = () => cancelGameStockfishAnalysis();
$('takeback').onclick = () => { void takeBack(); };
$('analyzeGame').onclick = () => { void analyzeCurrentGame(); };
$('reset').onclick = onReset;
$('resetAnalysis').onclick = onResetAnalysis;
$('flip').onclick = onFlip;
$('flipAnalysis').onclick = onFlip;
$('loadFen').onclick = async () => {
  if (busy) return;
  try {
    board = parseFen(($('fenInput') as HTMLInputElement).value || START_FEN);
    resetClocks();
    resetLineFromBoard();
    const terminal = updateGameOverState();
    await render(terminal ?? 'Loaded FEN.');
  } catch (err) {
    await render(`Invalid FEN: ${err instanceof Error ? err.message : String(err)}`);
  }
};

async function main() {
  initUiMode();
  initSideTabs();
  initNavAndShortcuts();
  initMoveHistoryControls();
  initPlayerSideControls();
  initModelSelect();
  initVisitsSelect();
  initSearchModeSelect();
  initBudgetModeSelect();
  initArenaControls();
  initRunConfigChips();
  const initialMessage = uiMode === 'analysis' ? startAnalysisBoard() : showPlayIntro();
  await render(initialMessage);
  syncWebClientStore({ evaluatorStatus: 'loading', message: 'Loading ONNX model…' });
  evaluator = await createEvaluatorForModel(selectedModel, selectedModelKey);
  arenaEvaluatorCache.set(selectedModelKey, evaluator);
  syncWebClientStore({ evaluatorStatus: 'ready' });
  setInterval(tickClocks, 250);
  await render(uiMode === 'analysis' ? `Loaded ${selectedModel.label}. Mode: ${searchDetails()}.` : `Loaded ${selectedModel.label}. Choose side and clock to start.`);
}
main().catch((e) => {
  console.error(e);
  const message = `Failed: ${e.message}`;
  syncWebClientStore({ evaluatorStatus: 'error', message });
  $('message').textContent = message;
});
