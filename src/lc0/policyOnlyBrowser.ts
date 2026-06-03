import { Chessground } from 'chessground';
import type { DrawShape } from 'chessground/draw';
import type { Key } from 'chessground/types';
import { boardToFen, parseFen, squareName, START_FEN, type BoardState } from '../chess/board.ts';
import { legalMoves, makeMove } from '../chess/movegen.ts';
import { moveToUci, type Move } from '../chess/moveCodec.ts';
import { bestMoveShapes, searchShapes } from './boardArrows.ts';
import { collectOrtRuntimeDiagnostics, describeOrtBackendConfig, type OrtExecutionProviderPreference, type OrtRuntimeDiagnostics } from '../nn/ortRuntime.ts';
import { gameOutcome, type GameResultCode } from './engineBattle.ts';
import { buildBoardHistoryFromMoves } from './history.ts';
import { clearLc0ModelCache, describeLc0ModelLoad, loadLc0ModelForOrt } from './modelCache.ts';
import { Lc0OnnxEvaluator, type Lc0Evaluation, type Lc0EvaluatorInput } from './onnxEvaluator.ts';
import { Lc0PolicyOnlyPlayer } from './policyOnlyPlayer.ts';
import { Lc0PuctSearcher, type Lc0SearchChild, type Lc0SearchResult } from './search.ts';
import { StockfishEngine } from './stockfishEngine.ts';

type Ground = ReturnType<typeof Chessground>;
type NativePrior = { uci: string; index: number; prior: number };
type NativeRecord = { id: string; backend?: string; fen: string; startFen?: string; moves?: string[]; bestmove: string; topPriors: NativePrior[] };
type RenderableSearchResult = Pick<Lc0SearchResult, 'fen' | 'move' | 'visits' | 'value'> & { children: Lc0SearchChild[]; pv?: string[]; multiPv?: string[][]; elapsedMs?: number; cancelled?: boolean; stats?: Lc0SearchResult['search']['stats'] };
type PackLoadResult = {
  packUrl: string;
  modelName: string;
  sourceSha256?: string;
  layout?: string;
  recommendedRuntime?: string;
  tensorCount: number;
  loadedTensorCount: number;
  loadedTensorBytes: number;
  shardCount: number;
  verifiedShardCount: number;
  shardBytes: number;
  elapsedMs: number;
};

type KernelVariant = 'scalar' | 'tiled16' | 'scalar-transposed';

type KernelProbeResult = {
  status: 'KERNEL_DONE';
  packUrl: string;
  modelName: string;
  adapterInfo?: Record<string, unknown>;
  weightTensor: string;
  biasTensor: string;
  variant: KernelVariant;
  k: number;
  n: number;
  warmup: number;
  iterations: number;
  packLoadMs: number;
  avgMs: number;
  minMs: number;
  maxMs: number;
  firstMs: number;
  timesMs?: number[];
  maxAbsError: number;
  rmsError: number;
  outputSample: number[];
};

type KernelBenchmarkResult = {
  status: 'KERNEL_BENCH_DONE';
  packUrl: string;
  modelName: string;
  adapterInfo?: Record<string, unknown>;
  weightTensor: string;
  biasTensor: string;
  variant: KernelVariant;
  k: number;
  n: number;
  warmup: number;
  iterations: number;
  packLoadMs: number;
  uploadSetupMs: number;
  dispatchLoopMs: number;
  dispatchLoopAvgMs: number;
  readbackSyncedMs: number;
  endToEndMs: number;
  maxAbsError: number;
  rmsError: number;
  outputSample: number[];
};

type QkvProbeResult = {
  status: 'QKV_DONE';
  packUrl: string;
  modelName: string;
  adapterInfo?: Record<string, unknown>;
  k: number;
  n: number;
  warmup: number;
  iterations: number;
  packLoadMs: number;
  avgMs: number;
  minMs: number;
  maxMs: number;
  firstMs: number;
  timesMs?: number[];
  maxAbsError: { q: number; k: number; v: number };
  rmsError: { q: number; k: number; v: number };
  outputSample: { q: number[]; k: number[]; v: number[] };
};

type QkvBenchmarkResult = {
  status: 'QKV_BENCH_DONE';
  packUrl: string;
  modelName: string;
  adapterInfo?: Record<string, unknown>;
  k: number;
  n: number;
  warmup: number;
  iterations: number;
  packLoadMs: number;
  uploadSetupMs: number;
  dispatchLoopMs: number;
  dispatchLoopAvgMs: number;
  readbackSyncedMs: number;
  endToEndMs: number;
  maxAbsError: { q: number; k: number; v: number };
  rmsError: { q: number; k: number; v: number };
  outputSample: { q: number[]; k: number[]; v: number[] };
};

type AttentionScoreBenchmarkResult = {
  status: 'ATTENTION_SCORE_BENCH_DONE';
  packUrl: string;
  modelName: string;
  adapterInfo?: Record<string, unknown>;
  tokens: number;
  channels: number;
  scale: number;
  smolgen?: { enabled: boolean; epsilon: number };
  warmup: number;
  iterations: number;
  packLoadMs: number;
  uploadSetupMs: number;
  dispatchLoopMs: number;
  dispatchLoopAvgMs: number;
  readbackSyncedMs: number;
  endToEndMs: number;
  maxAbsError: number;
  rmsError: number;
  outputSample: number[];
};

type AttentionScoreOrtBenchmarkResult = {
  status: 'ATTENTION_SCORE_ORT_BENCH_DONE';
  packUrl: string;
  modelName: string;
  tokens: number;
  channels: number;
  heads?: number;
  headDim?: number;
  scale: number;
  smolgen?: { enabled: boolean; epsilon: number };
  warmup: number;
  iterations: number;
  packLoadMs: number;
  modelBuildMs: number;
  sessionCreateMs: number;
  avgMs: number;
  minMs: number;
  maxMs: number;
  firstMs: number;
  timesMs?: number[];
  runsPerSecond: number;
  maxAbsError: number;
  rmsError: number;
  outputSample: number[];
};

type SoftmaxBenchmarkResult = {
  status: 'SOFTMAX_BENCH_DONE';
  packUrl: string;
  modelName: string;
  adapterInfo?: Record<string, unknown>;
  tokens: number;
  heads: number;
  rows: number;
  warmup: number;
  iterations: number;
  packLoadMs: number;
  uploadSetupMs: number;
  dispatchLoopMs: number;
  dispatchLoopAvgMs: number;
  readbackSyncedMs: number;
  endToEndMs: number;
  maxAbsError: number;
  rmsError: number;
  outputSample: number[];
};

type AttentionValueBenchmarkResult = {
  status: 'ATTENTION_VALUE_BENCH_DONE';
  packUrl: string;
  modelName: string;
  adapterInfo?: Record<string, unknown>;
  tokens: number;
  channels: number;
  heads: number;
  headDim: number;
  warmup: number;
  iterations: number;
  packLoadMs: number;
  uploadSetupMs: number;
  dispatchLoopMs: number;
  dispatchLoopAvgMs: number;
  readbackSyncedMs: number;
  endToEndMs: number;
  maxAbsError: number;
  rmsError: number;
  outputSample: number[];
};

type AttentionValueOrtBenchmarkResult = {
  status: 'ATTENTION_VALUE_ORT_BENCH_DONE';
  packUrl: string;
  modelName: string;
  tokens: number;
  channels: number;
  heads: number;
  headDim: number;
  warmup: number;
  iterations: number;
  packLoadMs: number;
  modelBuildMs: number;
  sessionCreateMs: number;
  avgMs: number;
  minMs: number;
  maxMs: number;
  firstMs: number;
  timesMs?: number[];
  runsPerSecond: number;
  maxAbsError: number;
  rmsError: number;
  outputSample: number[];
};

type AttentionBlockBenchmarkResult = {
  status: 'ATTENTION_BLOCK_BENCH_DONE';
  packUrl: string;
  modelName: string;
  adapterInfo?: Record<string, unknown>;
  tokens: number;
  channels: number;
  heads: number;
  headDim: number;
  warmup: number;
  iterations: number;
  packLoadMs: number;
  uploadSetupMs: number;
  dispatchLoopMs: number;
  dispatchLoopAvgMs: number;
  readbackSyncedMs: number;
  endToEndMs: number;
  maxAbsError: number;
  rmsError: number;
  outputSample: number[];
};

type AttentionOutputBenchmarkResult = {
  status: 'ATTENTION_OUTPUT_BENCH_DONE';
  packUrl: string;
  modelName: string;
  adapterInfo?: Record<string, unknown>;
  tokens: number;
  channels: number;
  heads: number;
  headDim: number;
  epsilon: number;
  alpha: number;
  warmup: number;
  iterations: number;
  packLoadMs: number;
  uploadSetupMs: number;
  dispatchLoopMs: number;
  dispatchLoopAvgMs: number;
  readbackSyncedMs: number;
  endToEndMs: number;
  maxAbsError: number;
  rmsError: number;
  outputSample: number[];
};

type AttentionOutputOrtBenchmarkResult = {
  status: 'ATTENTION_OUTPUT_ORT_BENCH_DONE';
  packUrl: string;
  modelName: string;
  tokens: number;
  channels: number;
  epsilon: number;
  alpha: number;
  warmup: number;
  iterations: number;
  packLoadMs: number;
  modelBuildMs: number;
  sessionCreateMs: number;
  avgMs: number;
  minMs: number;
  maxMs: number;
  firstMs: number;
  timesMs?: number[];
  runsPerSecond: number;
  maxAbsError: number;
  rmsError: number;
  outputSample: number[];
};

type Encoder0BlockStageTiming = {
  stage: string;
  label: string;
  iterations: number;
  totalMs: number;
  avgMs: number;
};

type Encoder0BlockBenchmarkResult = {
  status: 'ENCODER0_BLOCK_BENCH_DONE';
  packUrl: string;
  modelName: string;
  adapterInfo?: Record<string, unknown>;
  tokens: number;
  channels: number;
  heads: number;
  headDim: number;
  ffnHidden: number;
  lnEpsilon: number;
  attentionAlpha: number;
  ffnAlpha: number;
  smolgen?: { enabled: boolean; epsilon: number };
  warmup: number;
  iterations: number;
  packLoadMs: number;
  uploadSetupMs: number;
  dispatchLoopMs: number;
  dispatchLoopAvgMs: number;
  readbackSyncedMs: number;
  gpuTimestampSupported?: boolean;
  gpuTimestampMs?: number;
  stageTimings: Encoder0BlockStageTiming[];
  stageTimingTotalMs: number;
  endToEndMs: number;
  maxAbsError: number;
  rmsError: number;
  outputSample: number[];
};

type Encoder0BlockOrtBenchmarkResult = {
  status: 'ENCODER0_BLOCK_ORT_BENCH_DONE';
  packUrl: string;
  modelName: string;
  tokens: number;
  channels: number;
  heads: number;
  headDim: number;
  ffnHidden: number;
  lnEpsilon: number;
  attentionAlpha: number;
  ffnAlpha: number;
  smolgen?: { enabled: boolean; epsilon: number };
  warmup: number;
  iterations: number;
  packLoadMs: number;
  modelBuildMs: number;
  sessionCreateMs: number;
  avgMs: number;
  minMs: number;
  maxMs: number;
  firstMs: number;
  timesMs?: number[];
  runsPerSecond: number;
  maxAbsError: number;
  rmsError: number;
  outputSample: number[];
};

type EncoderStackHeadsResult = {
  mode: 'ort-policy-value';
  modelBuildMs: number;
  sessionCreateMs: number;
  runMs: number;
  policyMaxAbsError: number;
  policyRmsError: number;
  mappedPolicyMaxAbsError: number;
  mappedPolicyRmsError: number;
  wdlMaxAbsError: number;
  wdlRmsError: number;
  policySample: number[];
  mappedPolicySample: number[];
  wdl: number[];
};

type EncoderStackBenchmarkResult = {
  status: 'ENCODER_STACK_BENCH_DONE';
  packUrl: string;
  modelName: string;
  adapterInfo?: Record<string, unknown>;
  tokens: number;
  channels: number;
  heads: number;
  headDim: number;
  ffnHidden: number;
  lnEpsilon: number;
  warmup: number;
  layers: number;
  prefixes: string[];
  compareOrt: boolean;
  compareHeads: boolean;
  ortCoveredStages: string;
  packLoadMs: number;
  setupAndDispatchMs: number;
  dispatchSyncedMs: number;
  avgBlockDispatchSyncedMs: number;
  maxAbsError: number;
  rmsError: number;
  ortMaxAbsError?: number;
  outputSample: number[];
  policyValueHeads?: EncoderStackHeadsResult;
  blocks: Array<{
    layer: number;
    prefix: string;
    dispatchSyncedMs: number;
    maxAbsError: number;
    rmsError: number;
    ortMaxAbsError?: number;
    ortRmsError?: number;
    ortVsCpuMaxAbsError?: number;
    ortVsCpuRmsError?: number;
    outputSample: number[];
  }>;
};

type Encoder0FfnBenchmarkResult = {
  status: 'FFN_BENCH_DONE';
  packUrl: string;
  modelName: string;
  adapterInfo?: Record<string, unknown>;
  tokens: number;
  channels: number;
  hidden: number;
  epsilon: number;
  alpha: number;
  warmup: number;
  iterations: number;
  packLoadMs: number;
  uploadSetupMs: number;
  dispatchLoopMs: number;
  dispatchLoopAvgMs: number;
  readbackSyncedMs: number;
  endToEndMs: number;
  maxAbsError: number;
  rmsError: number;
  outputSample: number[];
};

type Encoder0FfnOrtBenchmarkResult = {
  status: 'FFN_ORT_BENCH_DONE';
  packUrl: string;
  modelName: string;
  tokens: number;
  channels: number;
  hidden: number;
  epsilon: number;
  alpha: number;
  warmup: number;
  iterations: number;
  packLoadMs: number;
  modelBuildMs: number;
  sessionCreateMs: number;
  avgMs: number;
  minMs: number;
  maxMs: number;
  firstMs: number;
  timesMs?: number[];
  runsPerSecond: number;
  maxAbsError: number;
  rmsError: number;
  outputSample: number[];
};

type OrtBenchmarkResult = {
  status: 'ORT_BENCH_DONE';
  packUrl: string;
  modelName: string;
  weightTensor: string;
  biasTensor: string;
  k: number;
  n: number;
  warmup: number;
  iterations: number;
  packLoadMs: number;
  modelBuildMs: number;
  sessionCreateMs: number;
  avgMs: number;
  minMs: number;
  maxMs: number;
  firstMs: number;
  timesMs?: number[];
  runsPerSecond: number;
  maxAbsError: number;
  rmsError: number;
  outputSample: number[];
};

type WorkerResponse =
  | { type: 'ready'; id: number; backend: string; modelCache: string }
  | { type: 'evaluationResult'; id: number; result: Lc0Evaluation }
  | { type: 'evaluationBatchResult'; id: number; result: Lc0Evaluation[] }
  | { type: 'packLoadResult'; id: number; result: PackLoadResult }
  | { type: 'kernelProbeResult'; id: number; result: KernelProbeResult }
  | { type: 'kernelBenchmarkResult'; id: number; result: KernelBenchmarkResult }
  | { type: 'ortBenchmarkResult'; id: number; result: OrtBenchmarkResult }
  | { type: 'qkvProbeResult'; id: number; result: QkvProbeResult }
  | { type: 'qkvBenchmarkResult'; id: number; result: QkvBenchmarkResult }
  | { type: 'attentionScoreBenchmarkResult'; id: number; result: AttentionScoreBenchmarkResult }
  | { type: 'attentionScoreOrtBenchmarkResult'; id: number; result: AttentionScoreOrtBenchmarkResult }
  | { type: 'softmaxBenchmarkResult'; id: number; result: SoftmaxBenchmarkResult }
  | { type: 'attentionValueBenchmarkResult'; id: number; result: AttentionValueBenchmarkResult }
  | { type: 'attentionValueOrtBenchmarkResult'; id: number; result: AttentionValueOrtBenchmarkResult }
  | { type: 'attentionBlockBenchmarkResult'; id: number; result: AttentionBlockBenchmarkResult }
  | { type: 'attentionOutputBenchmarkResult'; id: number; result: AttentionOutputBenchmarkResult }
  | { type: 'attentionOutputOrtBenchmarkResult'; id: number; result: AttentionOutputOrtBenchmarkResult }
  | { type: 'encoder0BlockBenchmarkResult'; id: number; result: Encoder0BlockBenchmarkResult }
  | { type: 'encoder0BlockOrtBenchmarkResult'; id: number; result: Encoder0BlockOrtBenchmarkResult }
  | { type: 'encoderStackBenchmarkResult'; id: number; result: EncoderStackBenchmarkResult }
  | { type: 'encoder0FfnBenchmarkResult'; id: number; result: Encoder0FfnBenchmarkResult }
  | { type: 'encoder0FfnOrtBenchmarkResult'; id: number; result: Encoder0FfnOrtBenchmarkResult }
  | { type: 'searchResult'; id: number; result: RenderableSearchResult }
  | { type: 'error'; id: number; error: string };

type BrowserEvaluationChoice = { move?: string; evaluation: Lc0Evaluation };
type EvalBenchResult = {
  status: 'BENCH_DONE';
  model: string;
  backend: string;
  workerOnly: boolean;
  warmup: number;
  iterations: number;
  avgMs: number;
  medianMs: number;
  minMs: number;
  maxMs: number;
  p90Ms: number;
  evalsPerSecond: number;
  workerInitMs?: number;
  timesMs: number[];
  bestMove?: string;
  q?: number;
  mlh?: number;
};

type EngineReplyMode = 'policy' | 'search';

const params = new URLSearchParams(location.search);
const DEFAULT_MODEL = '/models/lc0/t1-256x10-distilled-swa-2432500.batch1.f32.onnx';
const MODEL_URL = params.get('model') ?? DEFAULT_MODEL;
const DEFAULT_PACK_URL = '/models/lc0/t1-256x10-distilled-swa-2432500.batch8.f16.lc0web/model.lc0web.json';
const PACK_URL = params.get('pack') ?? params.get('modelPack') ?? DEFAULT_PACK_URL;
const ENCODER_PREFIX = params.get('encoderPrefix') ?? undefined;
const SOFTMAX_BENCH_REQUESTED = params.get('softmaxBench') === '1' || params.get('attentionSoftmaxBench') === '1';
const ATTENTION_VALUE_BENCH_REQUESTED = params.get('attentionValueBench') === '1' || params.get('valueBench') === '1';
const ATTENTION_VALUE_ORT_BENCH_REQUESTED = params.get('attentionValueOrtBench') === '1' || params.get('valueOrtBench') === '1';
const ATTENTION_BLOCK_BENCH_REQUESTED = params.get('attentionBlockBench') === '1' || params.get('attnBlockBench') === '1';
const ATTENTION_OUTPUT_BENCH_REQUESTED = params.get('attentionOutputBench') === '1' || params.get('attentionNormBench') === '1' || params.get('attnOutBench') === '1';
const ATTENTION_OUTPUT_ORT_BENCH_REQUESTED = params.get('attentionOutputOrtBench') === '1' || params.get('outputOrtBench') === '1' || params.get('attnOutOrtBench') === '1';
const ENCODER0_BLOCK_BENCH_REQUESTED = params.get('encoder0BlockBench') === '1' || params.get('fullEncoder0Bench') === '1';
const ENCODER0_BLOCK_ORT_BENCH_REQUESTED = params.get('encoder0BlockOrtBench') === '1' || params.get('fullEncoder0OrtBench') === '1';
const ENCODER_STACK_BENCH_REQUESTED = params.get('encoderStackBench') === '1' || params.get('encoderBlocksBench') === '1' || params.get('encoderStackHeadsBench') === '1';
const ENCODER0_FFN_BENCH_REQUESTED = params.get('encoder0FfnBench') === '1' || params.get('ffnBench') === '1';
const ENCODER0_FFN_ORT_BENCH_REQUESTED = params.get('encoder0FfnOrtBench') === '1' || params.get('ffnOrtBench') === '1';
const ATTENTION_SCORE_BENCH_REQUESTED = params.get('attentionScoreBench') === '1' || params.get('scoreBench') === '1';
const ATTENTION_SCORE_ORT_BENCH_REQUESTED = params.get('attentionScoreOrtBench') === '1' || params.get('scoreOrtBench') === '1';
const QKV_BENCH_REQUESTED = params.get('qkvBench') === '1' || params.get('qkvBenchmark') === '1';
const QKV_PROBE_REQUESTED = params.get('qkvProbe') === '1';
const ORT_OP_BENCH_REQUESTED = params.get('ortOpBench') === '1' || params.get('ortBench') === '1';
const KERNEL_BENCH_REQUESTED = params.get('kernelBench') === '1' || params.get('kernelBenchmark') === '1' || params.get('wgslBench') === '1';
const KERNEL_PROBE_REQUESTED = ENCODER_STACK_BENCH_REQUESTED || ENCODER0_BLOCK_ORT_BENCH_REQUESTED || ENCODER0_BLOCK_BENCH_REQUESTED || ENCODER0_FFN_ORT_BENCH_REQUESTED || ENCODER0_FFN_BENCH_REQUESTED || ATTENTION_OUTPUT_ORT_BENCH_REQUESTED || ATTENTION_OUTPUT_BENCH_REQUESTED || ATTENTION_BLOCK_BENCH_REQUESTED || ATTENTION_VALUE_ORT_BENCH_REQUESTED || ATTENTION_VALUE_BENCH_REQUESTED || SOFTMAX_BENCH_REQUESTED || ATTENTION_SCORE_BENCH_REQUESTED || ATTENTION_SCORE_ORT_BENCH_REQUESTED || QKV_BENCH_REQUESTED || QKV_PROBE_REQUESTED || ORT_OP_BENCH_REQUESTED || KERNEL_BENCH_REQUESTED || params.get('kernelProbe') === '1' || params.get('wgslProbe') === '1';
const PACK_PROBE_REQUESTED = KERNEL_PROBE_REQUESTED || params.get('packProbe') === '1' || params.get('pack') !== null || params.get('modelPack') !== null;
const BENCH_REQUESTED = params.get('bench') === '1' || params.get('timing') === '1';
const WORKER_ONLY_MODEL = PACK_PROBE_REQUESTED || BENCH_REQUESTED || params.get('workerOnly') === '1' || params.get('dedicatedWorker') === '1' || params.get('bigModel') === '1';
const SEARCH_WORKER_REQUESTED = WORKER_ONLY_MODEL || params.get('worker') === '1' || params.get('searchWorker') === '1';
const CACHE_MODEL = params.get('cache') === '1' || params.get('modelCache') === '1';
const BENCH_WARMUP = Math.min(100, Math.max(0, Math.floor(Number(params.get('benchWarmup') ?? '5') || 0)));
const BENCH_ITERS = Math.min(1000, Math.max(1, Math.floor(Number(params.get('benchIters') ?? params.get('iters') ?? '25') || 25)));
function requestedKernelVariant(): KernelVariant {
  const value = params.get('kernelVariant') ?? params.get('variant');
  return value === 'tiled16' || value === 'scalar-transposed' ? value : 'scalar';
}
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
let searchWorkerInitMs: number | undefined;
let workerRequestSeq = 0;
const workerPending = new Map<number, { resolve: (value: unknown) => void; reject: (error: Error) => void }>();
let busy = false;
let searching = false;
let mainSearchAbort: AbortController | null = null;
let activeWorkerSearchId: number | null = null;
let battleRunning = false;
let battleGames = Math.max(1, Math.floor(Number(params.get('battleGames') ?? '1') || 1));
// Delay between plies so the game is watchable on the board.
let battleDelayMs = Math.max(0, Math.floor(Number(params.get('battleDelay') ?? '350') || 350));
let battleAbort: AbortController | null = null;
type BattleOpponent = 'policy' | 'stockfish';
let battleOpponent: BattleOpponent = params.get('opponent') === 'stockfish' ? 'stockfish' : 'policy';
let stockfishDepth = Math.max(1, Math.floor(Number(params.get('sfDepth') ?? '4') || 4));
let stockfish: StockfishEngine | null = null;
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

function setBoardShapes(shapes: DrawShape[]) {
  ground?.setAutoShapes(shapes);
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

function evaluationAvailable(): boolean {
  return !!player || searchWorkerReady;
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
  el('engineMove').toggleAttribute('disabled', busy || !evaluationAvailable());
  el('searchMove').toggleAttribute('disabled', busy || !searchAvailable());
  el('analyze').toggleAttribute('disabled', busy || !searchAvailable());
  el('runParity').toggleAttribute('disabled', busy || !evaluationAvailable());
  el('stopSearch').toggleAttribute('disabled', !(searching || battleRunning));
  el('battleStart').toggleAttribute('disabled', busy || battleRunning || !evaluationAvailable());
}

function renderStatic() {
  el('fen').textContent = boardToFen(board);
  el('sideToMove').textContent = sideToMoveName();
  el('moveList').textContent = playedMoves.length ? playedMoves.join(' ') : '—';
  el('modelPath').textContent = PACK_PROBE_REQUESTED ? `${MODEL_URL} · pack ${PACK_URL}` : MODEL_URL;
  el('modelCache').textContent = workerModelCacheStatus ? `main ${mainModelCacheStatus}; worker ${workerModelCacheStatus}` : mainModelCacheStatus;
  el('backend').textContent = WORKER_ONLY_MODEL && searchWorkerReady ? searchWorkerBackend : describeOrtBackendConfig();
  el('status').textContent = PACK_PROBE_REQUESTED ? 'pack probe' : evaluationAvailable() ? 'ready' : 'loading';
  el('searchMode').textContent = searchModeLabel();
  el('searchBatch').textContent = `${searchBatchSize}`;
  el('searchMove').textContent = `Search ${searchVisits}`;
  el('engineMove').toggleAttribute('disabled', busy || !evaluationAvailable());
  el('searchMove').toggleAttribute('disabled', busy || !searchAvailable());
  el('analyze').toggleAttribute('disabled', busy || !searchAvailable());
  el('runParity').toggleAttribute('disabled', busy || !evaluationAvailable());
  el('stopSearch').toggleAttribute('disabled', !(searching || battleRunning));
  el('battleStart').toggleAttribute('disabled', busy || battleRunning || !evaluationAvailable());
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
  // Draw the chosen move (green) and other MultiPV candidates (blue) on the board.
  setBoardShapes(searchShapes(result.move, result.multiPv));
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
  if (!evaluationAvailable()) return;
  choosePolicyMove(currentEvaluationInput()).then((choice) => {
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
    // Reflect the policy pick on the board so analysis is visible there.
    setBoardShapes(bestMoveShapes(choice.move));
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

async function initSearchWorker(options: { initModel?: boolean } = {}): Promise<void> {
  const initModel = options.initModel ?? true;
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
  if (!initModel) return;
  const initStarted = performance.now();
  const ready = await postWorkerRequest<{ type: 'ready'; backend: string; modelCache: string }>({ type: 'init', modelUrl: MODEL_URL, ep: requestedWorkerEp(), cacheModel: CACHE_MODEL });
  searchWorkerInitMs = performance.now() - initStarted;
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

async function choosePolicyMove(input: Lc0EvaluatorInput): Promise<BrowserEvaluationChoice> {
  if (WORKER_ONLY_MODEL || !player) return evaluateWithWorker(input);
  return player.chooseMove(input);
}

function summarizeTimes(times: number[]): Pick<EvalBenchResult, 'avgMs' | 'medianMs' | 'minMs' | 'maxMs' | 'p90Ms' | 'evalsPerSecond'> {
  const sorted = [...times].sort((a, b) => a - b);
  const avg = times.reduce((sum, value) => sum + value, 0) / Math.max(1, times.length);
  const percentile = (p: number) => sorted[Math.min(sorted.length - 1, Math.max(0, Math.ceil(sorted.length * p) - 1))] ?? 0;
  return {
    avgMs: avg,
    medianMs: percentile(0.5),
    minMs: sorted[0] ?? 0,
    maxMs: sorted[sorted.length - 1] ?? 0,
    p90Ms: percentile(0.9),
    evalsPerSecond: 1000 / Math.max(1e-9, avg),
  };
}

type BenchmarkReportInput = {
  adapterInfo?: Record<string, unknown>;
  iterations?: number;
  readbackSyncedMs?: number;
  dispatchLoopAvgMs?: number;
  avgMs?: number;
  minMs?: number;
  maxMs?: number;
  timesMs?: number[];
};

function roundReportMs(value: number | undefined, digits = 4): number | undefined {
  return value === undefined || !Number.isFinite(value) ? undefined : Number(value.toFixed(digits));
}

function browserReportInfo(): Record<string, unknown> {
  return {
    userAgent: navigator.userAgent,
    platform: navigator.platform,
    hardwareConcurrency: navigator.hardwareConcurrency,
  };
}

function sampleTimingStats(samples: number[], source: string): Record<string, unknown> | undefined {
  const finite = samples.filter((value) => Number.isFinite(value)).sort((a, b) => a - b);
  if (!finite.length) return undefined;
  const percentile = (p: number) => finite[Math.min(finite.length - 1, Math.max(0, Math.ceil(finite.length * p) - 1))] ?? 0;
  const trim = finite.length >= 5 ? Math.floor(finite.length * 0.1) : 0;
  const trimmed = finite.slice(trim, finite.length - trim || finite.length);
  const mean = finite.reduce((sum, value) => sum + value, 0) / finite.length;
  const trimmedMean = trimmed.reduce((sum, value) => sum + value, 0) / trimmed.length;
  return {
    source,
    sampleCount: finite.length,
    meanMs: roundReportMs(mean),
    trimmedMeanMs: roundReportMs(trimmedMean),
    p50Ms: roundReportMs(percentile(0.5)),
    p95Ms: roundReportMs(percentile(0.95)),
    minMs: roundReportMs(finite[0]),
    maxMs: roundReportMs(finite[finite.length - 1]),
  };
}

function buildBenchmarkReport(result: BenchmarkReportInput): Record<string, unknown> {
  const perDispatchSyncedMs = result.readbackSyncedMs !== undefined && result.iterations ? result.readbackSyncedMs / result.iterations : undefined;
  const sampleStats = result.timesMs ? sampleTimingStats(result.timesMs, 'timed samples') : undefined;
  const aggregateStats = sampleStats ?? (result.avgMs !== undefined && result.minMs !== undefined && result.maxMs !== undefined ? {
    source: 'aggregate result fields (raw samples not returned)',
    sampleCount: result.iterations ?? 1,
    meanMs: roundReportMs(result.avgMs),
    trimmedMeanMs: roundReportMs(result.avgMs),
    minMs: roundReportMs(result.minMs),
    maxMs: roundReportMs(result.maxMs),
    percentileNote: 'p50/p95 unavailable because this result did not include raw timing samples',
  } : undefined);
  return {
    browserInfo: browserReportInfo(),
    gpuAdapterInfo: result.adapterInfo,
    packVerification: params.get('packVerify') === '0' ? 'packVerify=0; shard sha256 verification skipped for this benchmark run' : 'pack shard sha256 verification enabled',
    perDispatchSyncedMs: roundReportMs(perDispatchSyncedMs, 6),
    dispatchLoopAvgMs: roundReportMs(result.dispatchLoopAvgMs, 6),
    timingStats: aggregateStats ?? (perDispatchSyncedMs === undefined ? undefined : sampleTimingStats([perDispatchSyncedMs], 'single queued readback/iterations estimate')),
  };
}

function renderBenchmarkResult(result: EvalBenchResult) {
  const rounded: EvalBenchResult = {
    ...result,
    avgMs: Number(result.avgMs.toFixed(3)),
    medianMs: Number(result.medianMs.toFixed(3)),
    minMs: Number(result.minMs.toFixed(3)),
    maxMs: Number(result.maxMs.toFixed(3)),
    p90Ms: Number(result.p90Ms.toFixed(3)),
    evalsPerSecond: Number(result.evalsPerSecond.toFixed(3)),
    workerInitMs: result.workerInitMs === undefined ? undefined : Number(result.workerInitMs.toFixed(3)),
    timesMs: result.timesMs.map((time) => Number(time.toFixed(3))),
    q: result.q === undefined ? undefined : Number(result.q.toFixed(8)),
    mlh: result.mlh === undefined ? undefined : Number(result.mlh.toFixed(3)),
  };
  el('benchResult').textContent = JSON.stringify(rounded);
  el('message').textContent = `BENCH_DONE ${rounded.iterations} evals · avg ${rounded.avgMs.toFixed(1)} ms · ${rounded.evalsPerSecond.toFixed(2)} eval/s · ${rounded.backend}`;
}

async function runPackProbe(): Promise<void> {
  if (!searchWorker) throw new Error('pack probe requires LC0 worker');
  const tensorParam = params.get('packTensor') ?? params.get('tensor');
  const tensorNames = tensorParam ? tensorParam.split(',').map((name) => name.trim()).filter(Boolean) : undefined;
  const verifyShards = params.get('packVerify') !== '0';
  el('benchResult').textContent = 'PACK_RUNNING';
  setBusy(true, `Loading lc0web pack in dedicated worker${tensorNames ? ` (${tensorNames.length} tensor filter)` : ''}…`);
  try {
    const started = performance.now();
    const response = await postWorkerRequest<{ type: 'packLoadResult'; result: PackLoadResult }>({
      type: 'loadPack',
      packUrl: PACK_URL,
      loadWeights: params.get('packWeights') !== '0',
      verifyShards,
      tensorNames,
    });
    const result = {
      status: 'PACK_DONE',
      ...response.result,
      roundTripMs: Number((performance.now() - started).toFixed(3)),
      elapsedMs: Number(response.result.elapsedMs.toFixed(3)),
      shardMB: Number((response.result.shardBytes / 1_000_000).toFixed(3)),
      loadedTensorMB: Number((response.result.loadedTensorBytes / 1_000_000).toFixed(3)),
    };
    el('benchResult').textContent = JSON.stringify(result);
    el('message').textContent = `PACK_DONE ${result.modelName} · ${result.shardMB.toFixed(1)} MB shards · ${result.elapsedMs.toFixed(0)} ms worker load`;
  } catch (error) {
    el('benchResult').textContent = `PACK_FAILED ${(error as Error).message}`;
    el('message').textContent = `Pack probe failed: ${(error as Error).message}`;
    throw error;
  } finally {
    setBusy(false);
  }
}

async function runSoftmaxBenchmark(): Promise<void> {
  if (!searchWorker) throw new Error('softmax benchmark requires LC0 worker');
  const rawIters = Number(params.get('softmaxIters') ?? params.get('attentionSoftmaxIters') ?? params.get('kernelBenchIters') ?? '1000');
  const rawWarmup = Number(params.get('softmaxWarmup') ?? params.get('attentionSoftmaxWarmup') ?? params.get('kernelBenchWarmup') ?? '10');
  const iterations = Math.min(100_000, Math.max(1, Math.floor(Number.isFinite(rawIters) ? rawIters : 1000)));
  const warmup = Math.min(1000, Math.max(0, Math.floor(Number.isFinite(rawWarmup) ? rawWarmup : 10)));
  el('benchResult').textContent = 'SOFTMAX_BENCH_RUNNING';
  setBusy(true, `Benchmarking lc0web WGSL attention softmax: ${warmup} warmup + ${iterations} queued dispatches, one final readback…`);
  try {
    const response = await postWorkerRequest<{ type: 'softmaxBenchmarkResult'; result: SoftmaxBenchmarkResult }>({
      type: 'softmaxBenchmark',
      packUrl: PACK_URL,
      iterations,
      warmup,
      verifyShards: params.get('packVerify') !== '0',
    });
    const rounded = {
      ...response.result,
      benchmarkReport: buildBenchmarkReport(response.result),
      packLoadMs: Number(response.result.packLoadMs.toFixed(3)),
      uploadSetupMs: Number(response.result.uploadSetupMs.toFixed(3)),
      dispatchLoopMs: Number(response.result.dispatchLoopMs.toFixed(4)),
      dispatchLoopAvgMs: Number(response.result.dispatchLoopAvgMs.toExponential(6)),
      readbackSyncedMs: Number(response.result.readbackSyncedMs.toFixed(4)),
      endToEndMs: Number(response.result.endToEndMs.toFixed(3)),
      maxAbsError: Number(response.result.maxAbsError.toExponential(6)),
      rmsError: Number(response.result.rmsError.toExponential(6)),
      outputSample: response.result.outputSample.map((value) => Number(value.toFixed(8))),
    };
    el('benchResult').textContent = JSON.stringify(rounded);
    el('message').textContent = `SOFTMAX_BENCH_DONE ${rounded.rows}x${rounded.tokens} · ${rounded.iterations} queued dispatches · readback-sync ${rounded.readbackSyncedMs.toFixed(3)} ms · max |err| ${rounded.maxAbsError.toExponential(2)}`;
  } catch (error) {
    el('benchResult').textContent = `SOFTMAX_BENCH_FAILED ${(error as Error).message}`;
    el('message').textContent = `Softmax benchmark failed: ${(error as Error).message}`;
    throw error;
  } finally {
    setBusy(false);
  }
}

async function runAttentionOutputBenchmark(): Promise<void> {
  if (!searchWorker) throw new Error('attention output benchmark requires LC0 worker');
  const rawIters = Number(params.get('attentionOutputIters') ?? params.get('attentionNormIters') ?? params.get('attnOutIters') ?? '50');
  const rawWarmup = Number(params.get('attentionOutputWarmup') ?? params.get('attentionNormWarmup') ?? params.get('attnOutWarmup') ?? '3');
  const iterations = Math.min(10_000, Math.max(1, Math.floor(Number.isFinite(rawIters) ? rawIters : 50)));
  const warmup = Math.min(1000, Math.max(0, Math.floor(Number.isFinite(rawWarmup) ? rawWarmup : 3)));
  el('benchResult').textContent = 'ATTENTION_OUTPUT_BENCH_RUNNING';
  setBusy(true, `Benchmarking lc0web WGSL attention output projection/residual/norm: ${warmup} warmup + ${iterations} queued blocks, one final readback…`);
  try {
    const response = await postWorkerRequest<{ type: 'attentionOutputBenchmarkResult'; result: AttentionOutputBenchmarkResult }>({
      type: 'attentionOutputBenchmark',
      packUrl: PACK_URL,
      iterations,
      warmup,
      verifyShards: params.get('packVerify') !== '0',
      encoderPrefix: ENCODER_PREFIX,
    });
    const rounded = {
      ...response.result,
      benchmarkReport: buildBenchmarkReport(response.result),
      packLoadMs: Number(response.result.packLoadMs.toFixed(3)),
      uploadSetupMs: Number(response.result.uploadSetupMs.toFixed(3)),
      dispatchLoopMs: Number(response.result.dispatchLoopMs.toFixed(4)),
      dispatchLoopAvgMs: Number(response.result.dispatchLoopAvgMs.toExponential(6)),
      readbackSyncedMs: Number(response.result.readbackSyncedMs.toFixed(4)),
      endToEndMs: Number(response.result.endToEndMs.toFixed(3)),
      maxAbsError: Number(response.result.maxAbsError.toExponential(6)),
      rmsError: Number(response.result.rmsError.toExponential(6)),
      outputSample: response.result.outputSample.map((value) => Number(value.toFixed(8))),
    };
    el('benchResult').textContent = JSON.stringify(rounded);
    el('message').textContent = `ATTENTION_OUTPUT_BENCH_DONE ${rounded.tokens}x${rounded.channels} · ${rounded.iterations} queued blocks · readback-sync ${rounded.readbackSyncedMs.toFixed(3)} ms · max |err| ${rounded.maxAbsError.toExponential(2)}`;
  } catch (error) {
    el('benchResult').textContent = `ATTENTION_OUTPUT_BENCH_FAILED ${(error as Error).message}`;
    el('message').textContent = `Attention output benchmark failed: ${(error as Error).message}`;
    throw error;
  } finally {
    setBusy(false);
  }
}

async function runAttentionOutputOrtBenchmark(): Promise<void> {
  if (!searchWorker) throw new Error('attention-output ORT benchmark requires LC0 worker');
  const rawIters = Number(params.get('attentionOutputOrtIters') ?? params.get('outputOrtIters') ?? params.get('ortBenchIters') ?? '25');
  const rawWarmup = Number(params.get('attentionOutputOrtWarmup') ?? params.get('outputOrtWarmup') ?? params.get('ortBenchWarmup') ?? '5');
  const iterations = Math.min(1000, Math.max(1, Math.floor(Number.isFinite(rawIters) ? rawIters : 25)));
  const warmup = Math.min(100, Math.max(0, Math.floor(Number.isFinite(rawWarmup) ? rawWarmup : 5)));
  el('benchResult').textContent = 'ATTENTION_OUTPUT_ORT_BENCH_RUNNING';
  setBusy(true, `Benchmarking ORT tiny attention-output projection/residual/ln1: ${warmup} warmup + ${iterations} timed runs…`);
  try {
    const response = await postWorkerRequest<{ type: 'attentionOutputOrtBenchmarkResult'; result: AttentionOutputOrtBenchmarkResult }>({
      type: 'attentionOutputOrtBenchmark',
      packUrl: PACK_URL,
      ep: requestedWorkerEp(),
      iterations,
      warmup,
      verifyShards: params.get('packVerify') !== '0',
      encoderPrefix: ENCODER_PREFIX,
    });
    const rounded = {
      ...response.result,
      benchmarkReport: buildBenchmarkReport(response.result),
      packLoadMs: Number(response.result.packLoadMs.toFixed(3)),
      modelBuildMs: Number(response.result.modelBuildMs.toFixed(3)),
      sessionCreateMs: Number(response.result.sessionCreateMs.toFixed(3)),
      avgMs: Number(response.result.avgMs.toFixed(4)),
      minMs: Number(response.result.minMs.toFixed(4)),
      maxMs: Number(response.result.maxMs.toFixed(4)),
      firstMs: Number(response.result.firstMs.toFixed(4)),
      runsPerSecond: Number(response.result.runsPerSecond.toFixed(3)),
      maxAbsError: Number(response.result.maxAbsError.toExponential(6)),
      rmsError: Number(response.result.rmsError.toExponential(6)),
      outputSample: response.result.outputSample.map((value) => Number(value.toFixed(6))),
    };
    el('benchResult').textContent = JSON.stringify(rounded);
    el('message').textContent = `ATTENTION_OUTPUT_ORT_BENCH_DONE ${rounded.tokens}x${rounded.channels} · avg ${rounded.avgMs.toFixed(3)} ms · max |err| ${rounded.maxAbsError.toExponential(2)}`;
  } catch (error) {
    el('benchResult').textContent = `ATTENTION_OUTPUT_ORT_BENCH_FAILED ${(error as Error).message}`;
    el('message').textContent = `Attention-output ORT benchmark failed: ${(error as Error).message}`;
    throw error;
  } finally {
    setBusy(false);
  }
}

async function runEncoder0BlockOrtBenchmark(): Promise<void> {
  if (!searchWorker) throw new Error('encoder0 block ORT benchmark requires LC0 worker');
  const rawIters = Number(params.get('encoder0BlockOrtIters') ?? params.get('fullEncoder0OrtIters') ?? params.get('ortBenchIters') ?? '10');
  const rawWarmup = Number(params.get('encoder0BlockOrtWarmup') ?? params.get('fullEncoder0OrtWarmup') ?? params.get('ortBenchWarmup') ?? '3');
  const iterations = Math.min(1000, Math.max(1, Math.floor(Number.isFinite(rawIters) ? rawIters : 10)));
  const warmup = Math.min(100, Math.max(0, Math.floor(Number.isFinite(rawWarmup) ? rawWarmup : 3)));
  el('benchResult').textContent = 'ENCODER0_BLOCK_ORT_BENCH_RUNNING';
  setBusy(true, `Benchmarking ORT tiny encoder0 attention-output+FFN block: ${warmup} warmup + ${iterations} timed runs…`);
  try {
    const response = await postWorkerRequest<{ type: 'encoder0BlockOrtBenchmarkResult'; result: Encoder0BlockOrtBenchmarkResult }>({
      type: 'encoder0BlockOrtBenchmark',
      packUrl: PACK_URL,
      ep: requestedWorkerEp(),
      iterations,
      warmup,
      verifyShards: params.get('packVerify') !== '0',
      encoderPrefix: ENCODER_PREFIX,
    });
    const rounded = {
      ...response.result,
      benchmarkReport: buildBenchmarkReport(response.result),
      packLoadMs: Number(response.result.packLoadMs.toFixed(3)),
      modelBuildMs: Number(response.result.modelBuildMs.toFixed(3)),
      sessionCreateMs: Number(response.result.sessionCreateMs.toFixed(3)),
      avgMs: Number(response.result.avgMs.toFixed(4)),
      minMs: Number(response.result.minMs.toFixed(4)),
      maxMs: Number(response.result.maxMs.toFixed(4)),
      firstMs: Number(response.result.firstMs.toFixed(4)),
      runsPerSecond: Number(response.result.runsPerSecond.toFixed(3)),
      maxAbsError: Number(response.result.maxAbsError.toExponential(6)),
      rmsError: Number(response.result.rmsError.toExponential(6)),
      outputSample: response.result.outputSample.map((value) => Number(value.toFixed(6))),
    };
    el('benchResult').textContent = JSON.stringify(rounded);
    el('message').textContent = `ENCODER0_BLOCK_ORT_BENCH_DONE attention+FFN ${rounded.tokens}x${rounded.channels} · avg ${rounded.avgMs.toFixed(3)} ms · max |err| ${rounded.maxAbsError.toExponential(2)}`;
  } catch (error) {
    el('benchResult').textContent = `ENCODER0_BLOCK_ORT_BENCH_FAILED ${(error as Error).message}`;
    el('message').textContent = `Encoder0 block ORT benchmark failed: ${(error as Error).message}`;
    throw error;
  } finally {
    setBusy(false);
  }
}

async function runEncoder0BlockBenchmark(): Promise<void> {
  if (!searchWorker) throw new Error('encoder0 block benchmark requires LC0 worker');
  const rawIters = Number(params.get('encoder0BlockIters') ?? params.get('fullEncoder0Iters') ?? '5');
  const rawWarmup = Number(params.get('encoder0BlockWarmup') ?? params.get('fullEncoder0Warmup') ?? '1');
  const iterations = Math.min(10_000, Math.max(1, Math.floor(Number.isFinite(rawIters) ? rawIters : 5)));
  const warmup = Math.min(1000, Math.max(0, Math.floor(Number.isFinite(rawWarmup) ? rawWarmup : 1)));
  el('benchResult').textContent = 'ENCODER0_BLOCK_BENCH_RUNNING';
  setBusy(true, `Benchmarking lc0web WGSL full encoder0 attention+FFN block: ${warmup} warmup + ${iterations} queued blocks, one final readback…`);
  try {
    const response = await postWorkerRequest<{ type: 'encoder0BlockBenchmarkResult'; result: Encoder0BlockBenchmarkResult }>({
      type: 'encoder0BlockBenchmark',
      packUrl: PACK_URL,
      iterations,
      warmup,
      verifyShards: params.get('packVerify') !== '0',
      encoderPrefix: ENCODER_PREFIX,
    });
    const rounded = {
      ...response.result,
      benchmarkReport: buildBenchmarkReport(response.result),
      packLoadMs: Number(response.result.packLoadMs.toFixed(3)),
      uploadSetupMs: Number(response.result.uploadSetupMs.toFixed(3)),
      dispatchLoopMs: Number(response.result.dispatchLoopMs.toFixed(4)),
      dispatchLoopAvgMs: Number(response.result.dispatchLoopAvgMs.toExponential(6)),
      readbackSyncedMs: Number(response.result.readbackSyncedMs.toFixed(4)),
      gpuTimestampMs: response.result.gpuTimestampMs === undefined ? undefined : Number(response.result.gpuTimestampMs.toFixed(6)),
      stageTimings: response.result.stageTimings.map((timing) => ({
        ...timing,
        totalMs: Number(timing.totalMs.toFixed(4)),
        avgMs: Number(timing.avgMs.toFixed(6)),
      })),
      stageTimingTotalMs: Number(response.result.stageTimingTotalMs.toFixed(4)),
      endToEndMs: Number(response.result.endToEndMs.toFixed(3)),
      maxAbsError: Number(response.result.maxAbsError.toExponential(6)),
      rmsError: Number(response.result.rmsError.toExponential(6)),
      outputSample: response.result.outputSample.map((value) => Number(value.toFixed(8))),
    };
    el('benchResult').textContent = JSON.stringify(rounded);
    const slowestStage = rounded.stageTimings.reduce((best, timing) => timing.avgMs > best.avgMs ? timing : best, rounded.stageTimings[0]);
    const gpuTimeText = rounded.gpuTimestampMs === undefined ? '' : ` · gpu-timestamp ${rounded.gpuTimestampMs.toFixed(3)} ms`;
    el('message').textContent = `ENCODER0_BLOCK_BENCH_DONE attention+FFN ${rounded.tokens}x${rounded.channels} · ${rounded.iterations} queued blocks · readback-sync ${rounded.readbackSyncedMs.toFixed(3)} ms${gpuTimeText} · slowest stage ${slowestStage.label} ${slowestStage.avgMs.toFixed(3)} ms avg · max |err| ${rounded.maxAbsError.toExponential(2)}`;
  } catch (error) {
    el('benchResult').textContent = `ENCODER0_BLOCK_BENCH_FAILED ${(error as Error).message}`;
    el('message').textContent = `Encoder0 block benchmark failed: ${(error as Error).message}`;
    throw error;
  } finally {
    setBusy(false);
  }
}

async function runEncoderStackBenchmark(): Promise<void> {
  if (!searchWorker) throw new Error('encoder stack benchmark requires LC0 worker');
  const rawLayers = Number(params.get('encoderLayers') ?? params.get('layers') ?? '2');
  const rawWarmup = Number(params.get('encoderStackWarmup') ?? '0');
  const layers = Math.min(32, Math.max(1, Math.floor(Number.isFinite(rawLayers) ? rawLayers : 2)));
  const warmup = Math.min(10, Math.max(0, Math.floor(Number.isFinite(rawWarmup) ? rawWarmup : 0)));
  const compareOrt = params.get('encoderStackOrt') !== '0';
  const compareHeads = params.get('encoderStackHeads') === '1' || params.get('encoderStackHeadsBench') === '1';
  el('benchResult').textContent = 'ENCODER_STACK_BENCH_RUNNING';
  setBusy(true, `Running reusable WGSL encoder-block stack over ${layers} layer(s), with block-by-block ${compareOrt ? 'f32 ONNX/ORT ' : ''}parity${compareHeads ? ' and ORT policy/value heads' : ''}…`);
  try {
    const response = await postWorkerRequest<{ type: 'encoderStackBenchmarkResult'; result: EncoderStackBenchmarkResult }>({
      type: 'encoderStackBenchmark',
      packUrl: PACK_URL,
      ep: requestedWorkerEp(),
      layers,
      warmup,
      verifyShards: params.get('packVerify') !== '0',
      compareOrt,
      compareHeads,
    });
    const rounded = {
      ...response.result,
      benchmarkReport: buildBenchmarkReport(response.result),
      packLoadMs: Number(response.result.packLoadMs.toFixed(3)),
      setupAndDispatchMs: Number(response.result.setupAndDispatchMs.toFixed(3)),
      dispatchSyncedMs: Number(response.result.dispatchSyncedMs.toFixed(4)),
      avgBlockDispatchSyncedMs: Number(response.result.avgBlockDispatchSyncedMs.toFixed(4)),
      maxAbsError: Number(response.result.maxAbsError.toExponential(6)),
      rmsError: Number(response.result.rmsError.toExponential(6)),
      ortMaxAbsError: response.result.ortMaxAbsError === undefined ? undefined : Number(response.result.ortMaxAbsError.toExponential(6)),
      outputSample: response.result.outputSample.map((value) => Number(value.toFixed(8))),
      policyValueHeads: response.result.policyValueHeads ? {
        ...response.result.policyValueHeads,
        modelBuildMs: Number(response.result.policyValueHeads.modelBuildMs.toFixed(3)),
        sessionCreateMs: Number(response.result.policyValueHeads.sessionCreateMs.toFixed(3)),
        runMs: Number(response.result.policyValueHeads.runMs.toFixed(3)),
        policyMaxAbsError: Number(response.result.policyValueHeads.policyMaxAbsError.toExponential(6)),
        policyRmsError: Number(response.result.policyValueHeads.policyRmsError.toExponential(6)),
        mappedPolicyMaxAbsError: Number(response.result.policyValueHeads.mappedPolicyMaxAbsError.toExponential(6)),
        mappedPolicyRmsError: Number(response.result.policyValueHeads.mappedPolicyRmsError.toExponential(6)),
        wdlMaxAbsError: Number(response.result.policyValueHeads.wdlMaxAbsError.toExponential(6)),
        wdlRmsError: Number(response.result.policyValueHeads.wdlRmsError.toExponential(6)),
        policySample: response.result.policyValueHeads.policySample.map((value) => Number(value.toFixed(8))),
        mappedPolicySample: response.result.policyValueHeads.mappedPolicySample.map((value) => Number(value.toFixed(8))),
        wdl: response.result.policyValueHeads.wdl.map((value) => Number(value.toFixed(8))),
      } : undefined,
      blocks: response.result.blocks.map((block) => ({
        ...block,
        dispatchSyncedMs: Number(block.dispatchSyncedMs.toFixed(4)),
        maxAbsError: Number(block.maxAbsError.toExponential(6)),
        rmsError: Number(block.rmsError.toExponential(6)),
        ortMaxAbsError: block.ortMaxAbsError === undefined ? undefined : Number(block.ortMaxAbsError.toExponential(6)),
        ortRmsError: block.ortRmsError === undefined ? undefined : Number(block.ortRmsError.toExponential(6)),
        ortVsCpuMaxAbsError: block.ortVsCpuMaxAbsError === undefined ? undefined : Number(block.ortVsCpuMaxAbsError.toExponential(6)),
        ortVsCpuRmsError: block.ortVsCpuRmsError === undefined ? undefined : Number(block.ortVsCpuRmsError.toExponential(6)),
        outputSample: block.outputSample.map((value) => Number(value.toFixed(8))),
      })),
    };
    el('benchResult').textContent = JSON.stringify(rounded);
    const ortText = rounded.ortMaxAbsError === undefined ? '' : ` · ORT max |err| ${rounded.ortMaxAbsError.toExponential(2)}`;
    const headsText = rounded.policyValueHeads ? ` · heads policy |err| ${rounded.policyValueHeads.policyMaxAbsError.toExponential(2)} · mapped |err| ${rounded.policyValueHeads.mappedPolicyMaxAbsError.toExponential(2)} · WDL |err| ${rounded.policyValueHeads.wdlMaxAbsError.toExponential(2)}` : '';
    el('message').textContent = `ENCODER_STACK_BENCH_DONE ${rounded.layers} reusable WGSL block(s) · avg block ${rounded.avgBlockDispatchSyncedMs.toFixed(3)} ms · max |err| ${rounded.maxAbsError.toExponential(2)}${ortText}${headsText}`;
  } catch (error) {
    el('benchResult').textContent = `ENCODER_STACK_BENCH_FAILED ${(error as Error).message}`;
    el('message').textContent = `Encoder stack benchmark failed: ${(error as Error).message}`;
    throw error;
  } finally {
    setBusy(false);
  }
}

async function runEncoder0FfnBenchmark(): Promise<void> {
  if (!searchWorker) throw new Error('encoder0 FFN benchmark requires LC0 worker');
  const rawIters = Number(params.get('encoder0FfnIters') ?? params.get('ffnIters') ?? '10');
  const rawWarmup = Number(params.get('encoder0FfnWarmup') ?? params.get('ffnWarmup') ?? '2');
  const iterations = Math.min(10_000, Math.max(1, Math.floor(Number.isFinite(rawIters) ? rawIters : 10)));
  const warmup = Math.min(1000, Math.max(0, Math.floor(Number.isFinite(rawWarmup) ? rawWarmup : 2)));
  el('benchResult').textContent = 'FFN_BENCH_RUNNING';
  setBusy(true, `Benchmarking lc0web WGSL encoder0 FFN dense1/sqrrelu/dense2/residual/ln2: ${warmup} warmup + ${iterations} queued blocks, one final readback…`);
  try {
    const response = await postWorkerRequest<{ type: 'encoder0FfnBenchmarkResult'; result: Encoder0FfnBenchmarkResult }>({
      type: 'encoder0FfnBenchmark',
      packUrl: PACK_URL,
      iterations,
      warmup,
      verifyShards: params.get('packVerify') !== '0',
      encoderPrefix: ENCODER_PREFIX,
    });
    const rounded = {
      ...response.result,
      benchmarkReport: buildBenchmarkReport(response.result),
      packLoadMs: Number(response.result.packLoadMs.toFixed(3)),
      uploadSetupMs: Number(response.result.uploadSetupMs.toFixed(3)),
      dispatchLoopMs: Number(response.result.dispatchLoopMs.toFixed(4)),
      dispatchLoopAvgMs: Number(response.result.dispatchLoopAvgMs.toExponential(6)),
      readbackSyncedMs: Number(response.result.readbackSyncedMs.toFixed(4)),
      endToEndMs: Number(response.result.endToEndMs.toFixed(3)),
      maxAbsError: Number(response.result.maxAbsError.toExponential(6)),
      rmsError: Number(response.result.rmsError.toExponential(6)),
      outputSample: response.result.outputSample.map((value) => Number(value.toFixed(8))),
    };
    el('benchResult').textContent = JSON.stringify(rounded);
    el('message').textContent = `FFN_BENCH_DONE encoder0 ${rounded.tokens}x${rounded.channels}→${rounded.hidden}→${rounded.channels} · ${rounded.iterations} queued blocks · readback-sync ${rounded.readbackSyncedMs.toFixed(3)} ms · max |err| ${rounded.maxAbsError.toExponential(2)}`;
  } catch (error) {
    el('benchResult').textContent = `FFN_BENCH_FAILED ${(error as Error).message}`;
    el('message').textContent = `Encoder0 FFN benchmark failed: ${(error as Error).message}`;
    throw error;
  } finally {
    setBusy(false);
  }
}

async function runEncoder0FfnOrtBenchmark(): Promise<void> {
  if (!searchWorker) throw new Error('encoder0 FFN ORT benchmark requires LC0 worker');
  const rawIters = Number(params.get('encoder0FfnOrtIters') ?? params.get('ffnOrtIters') ?? params.get('ortBenchIters') ?? '25');
  const rawWarmup = Number(params.get('encoder0FfnOrtWarmup') ?? params.get('ffnOrtWarmup') ?? params.get('ortBenchWarmup') ?? '5');
  const iterations = Math.min(1000, Math.max(1, Math.floor(Number.isFinite(rawIters) ? rawIters : 25)));
  const warmup = Math.min(100, Math.max(0, Math.floor(Number.isFinite(rawWarmup) ? rawWarmup : 5)));
  el('benchResult').textContent = 'FFN_ORT_BENCH_RUNNING';
  setBusy(true, `Benchmarking ORT tiny encoder0 FFN dense1/sqrrelu/dense2/residual/ln2: ${warmup} warmup + ${iterations} timed runs…`);
  try {
    const response = await postWorkerRequest<{ type: 'encoder0FfnOrtBenchmarkResult'; result: Encoder0FfnOrtBenchmarkResult }>({
      type: 'encoder0FfnOrtBenchmark',
      packUrl: PACK_URL,
      ep: requestedWorkerEp(),
      iterations,
      warmup,
      verifyShards: params.get('packVerify') !== '0',
      encoderPrefix: ENCODER_PREFIX,
    });
    const rounded = {
      ...response.result,
      benchmarkReport: buildBenchmarkReport(response.result),
      packLoadMs: Number(response.result.packLoadMs.toFixed(3)),
      modelBuildMs: Number(response.result.modelBuildMs.toFixed(3)),
      sessionCreateMs: Number(response.result.sessionCreateMs.toFixed(3)),
      avgMs: Number(response.result.avgMs.toFixed(4)),
      minMs: Number(response.result.minMs.toFixed(4)),
      maxMs: Number(response.result.maxMs.toFixed(4)),
      firstMs: Number(response.result.firstMs.toFixed(4)),
      runsPerSecond: Number(response.result.runsPerSecond.toFixed(3)),
      maxAbsError: Number(response.result.maxAbsError.toExponential(6)),
      rmsError: Number(response.result.rmsError.toExponential(6)),
      outputSample: response.result.outputSample.map((value) => Number(value.toFixed(6))),
    };
    el('benchResult').textContent = JSON.stringify(rounded);
    el('message').textContent = `FFN_ORT_BENCH_DONE encoder0 ${rounded.tokens}x${rounded.channels}→${rounded.hidden}→${rounded.channels} · avg ${rounded.avgMs.toFixed(3)} ms · max |err| ${rounded.maxAbsError.toExponential(2)}`;
  } catch (error) {
    el('benchResult').textContent = `FFN_ORT_BENCH_FAILED ${(error as Error).message}`;
    el('message').textContent = `Encoder0 FFN ORT benchmark failed: ${(error as Error).message}`;
    throw error;
  } finally {
    setBusy(false);
  }
}

async function runAttentionBlockBenchmark(): Promise<void> {
  if (!searchWorker) throw new Error('attention block benchmark requires LC0 worker');
  const rawIters = Number(params.get('attentionBlockIters') ?? params.get('attnBlockIters') ?? '100');
  const rawWarmup = Number(params.get('attentionBlockWarmup') ?? params.get('attnBlockWarmup') ?? '5');
  const iterations = Math.min(10_000, Math.max(1, Math.floor(Number.isFinite(rawIters) ? rawIters : 100)));
  const warmup = Math.min(1000, Math.max(0, Math.floor(Number.isFinite(rawWarmup) ? rawWarmup : 5)));
  el('benchResult').textContent = 'ATTENTION_BLOCK_BENCH_RUNNING';
  setBusy(true, `Benchmarking lc0web WGSL attention block: ${warmup} warmup + ${iterations} queued QKV/QK/softmax/value blocks, one final readback…`);
  try {
    const response = await postWorkerRequest<{ type: 'attentionBlockBenchmarkResult'; result: AttentionBlockBenchmarkResult }>({
      type: 'attentionBlockBenchmark',
      packUrl: PACK_URL,
      iterations,
      warmup,
      verifyShards: params.get('packVerify') !== '0',
    });
    const rounded = {
      ...response.result,
      benchmarkReport: buildBenchmarkReport(response.result),
      packLoadMs: Number(response.result.packLoadMs.toFixed(3)),
      uploadSetupMs: Number(response.result.uploadSetupMs.toFixed(3)),
      dispatchLoopMs: Number(response.result.dispatchLoopMs.toFixed(4)),
      dispatchLoopAvgMs: Number(response.result.dispatchLoopAvgMs.toExponential(6)),
      readbackSyncedMs: Number(response.result.readbackSyncedMs.toFixed(4)),
      endToEndMs: Number(response.result.endToEndMs.toFixed(3)),
      maxAbsError: Number(response.result.maxAbsError.toExponential(6)),
      rmsError: Number(response.result.rmsError.toExponential(6)),
      outputSample: response.result.outputSample.map((value) => Number(value.toFixed(8))),
    };
    el('benchResult').textContent = JSON.stringify(rounded);
    el('message').textContent = `ATTENTION_BLOCK_BENCH_DONE ${rounded.tokens}x${rounded.channels} · ${rounded.iterations} queued blocks · readback-sync ${rounded.readbackSyncedMs.toFixed(3)} ms · max |err| ${rounded.maxAbsError.toExponential(2)}`;
  } catch (error) {
    el('benchResult').textContent = `ATTENTION_BLOCK_BENCH_FAILED ${(error as Error).message}`;
    el('message').textContent = `Attention block benchmark failed: ${(error as Error).message}`;
    throw error;
  } finally {
    setBusy(false);
  }
}

async function runAttentionValueBenchmark(): Promise<void> {
  if (!searchWorker) throw new Error('attention value benchmark requires LC0 worker');
  const rawIters = Number(params.get('attentionValueIters') ?? params.get('valueIters') ?? params.get('kernelBenchIters') ?? '1000');
  const rawWarmup = Number(params.get('attentionValueWarmup') ?? params.get('valueWarmup') ?? params.get('kernelBenchWarmup') ?? '10');
  const iterations = Math.min(100_000, Math.max(1, Math.floor(Number.isFinite(rawIters) ? rawIters : 1000)));
  const warmup = Math.min(1000, Math.max(0, Math.floor(Number.isFinite(rawWarmup) ? rawWarmup : 10)));
  el('benchResult').textContent = 'ATTENTION_VALUE_BENCH_RUNNING';
  setBusy(true, `Benchmarking lc0web WGSL attention value: ${warmup} warmup + ${iterations} queued dispatches, one final readback…`);
  try {
    const response = await postWorkerRequest<{ type: 'attentionValueBenchmarkResult'; result: AttentionValueBenchmarkResult }>({
      type: 'attentionValueBenchmark',
      packUrl: PACK_URL,
      iterations,
      warmup,
      verifyShards: params.get('packVerify') !== '0',
    });
    const rounded = {
      ...response.result,
      benchmarkReport: buildBenchmarkReport(response.result),
      packLoadMs: Number(response.result.packLoadMs.toFixed(3)),
      uploadSetupMs: Number(response.result.uploadSetupMs.toFixed(3)),
      dispatchLoopMs: Number(response.result.dispatchLoopMs.toFixed(4)),
      dispatchLoopAvgMs: Number(response.result.dispatchLoopAvgMs.toExponential(6)),
      readbackSyncedMs: Number(response.result.readbackSyncedMs.toFixed(4)),
      endToEndMs: Number(response.result.endToEndMs.toFixed(3)),
      maxAbsError: Number(response.result.maxAbsError.toExponential(6)),
      rmsError: Number(response.result.rmsError.toExponential(6)),
      outputSample: response.result.outputSample.map((value) => Number(value.toFixed(8))),
    };
    el('benchResult').textContent = JSON.stringify(rounded);
    el('message').textContent = `ATTENTION_VALUE_BENCH_DONE ${rounded.tokens}x${rounded.channels} · ${rounded.iterations} queued dispatches · readback-sync ${rounded.readbackSyncedMs.toFixed(3)} ms · max |err| ${rounded.maxAbsError.toExponential(2)}`;
  } catch (error) {
    el('benchResult').textContent = `ATTENTION_VALUE_BENCH_FAILED ${(error as Error).message}`;
    el('message').textContent = `Attention value benchmark failed: ${(error as Error).message}`;
    throw error;
  } finally {
    setBusy(false);
  }
}

async function runAttentionValueOrtBenchmark(): Promise<void> {
  if (!searchWorker) throw new Error('attention-value ORT benchmark requires LC0 worker');
  const rawIters = Number(params.get('attentionValueOrtIters') ?? params.get('valueOrtIters') ?? params.get('ortBenchIters') ?? '25');
  const rawWarmup = Number(params.get('attentionValueOrtWarmup') ?? params.get('valueOrtWarmup') ?? params.get('ortBenchWarmup') ?? '5');
  const iterations = Math.min(1000, Math.max(1, Math.floor(Number.isFinite(rawIters) ? rawIters : 25)));
  const warmup = Math.min(100, Math.max(0, Math.floor(Number.isFinite(rawWarmup) ? rawWarmup : 5)));
  el('benchResult').textContent = 'ATTENTION_VALUE_ORT_BENCH_RUNNING';
  setBusy(true, `Benchmarking ORT tiny attention-value batched MatMul: ${warmup} warmup + ${iterations} timed runs…`);
  try {
    const response = await postWorkerRequest<{ type: 'attentionValueOrtBenchmarkResult'; result: AttentionValueOrtBenchmarkResult }>({
      type: 'attentionValueOrtBenchmark',
      packUrl: PACK_URL,
      ep: requestedWorkerEp(),
      iterations,
      warmup,
      verifyShards: params.get('packVerify') !== '0',
    });
    const rounded = {
      ...response.result,
      benchmarkReport: buildBenchmarkReport(response.result),
      packLoadMs: Number(response.result.packLoadMs.toFixed(3)),
      modelBuildMs: Number(response.result.modelBuildMs.toFixed(3)),
      sessionCreateMs: Number(response.result.sessionCreateMs.toFixed(3)),
      avgMs: Number(response.result.avgMs.toFixed(4)),
      minMs: Number(response.result.minMs.toFixed(4)),
      maxMs: Number(response.result.maxMs.toFixed(4)),
      firstMs: Number(response.result.firstMs.toFixed(4)),
      runsPerSecond: Number(response.result.runsPerSecond.toFixed(3)),
      maxAbsError: Number(response.result.maxAbsError.toExponential(6)),
      rmsError: Number(response.result.rmsError.toExponential(6)),
      outputSample: response.result.outputSample.map((value) => Number(value.toFixed(6))),
    };
    el('benchResult').textContent = JSON.stringify(rounded);
    el('message').textContent = `ATTENTION_VALUE_ORT_BENCH_DONE ${rounded.tokens}x${rounded.channels} · avg ${rounded.avgMs.toFixed(3)} ms · max |err| ${rounded.maxAbsError.toExponential(2)}`;
  } catch (error) {
    el('benchResult').textContent = `ATTENTION_VALUE_ORT_BENCH_FAILED ${(error as Error).message}`;
    el('message').textContent = `Attention-value ORT benchmark failed: ${(error as Error).message}`;
    throw error;
  } finally {
    setBusy(false);
  }
}

async function runAttentionScoreBenchmark(): Promise<void> {
  if (!searchWorker) throw new Error('attention-score benchmark requires LC0 worker');
  const rawIters = Number(params.get('attentionScoreIters') ?? params.get('scoreIters') ?? params.get('kernelBenchIters') ?? '1000');
  const rawWarmup = Number(params.get('attentionScoreWarmup') ?? params.get('scoreWarmup') ?? params.get('kernelBenchWarmup') ?? '10');
  const iterations = Math.min(100_000, Math.max(1, Math.floor(Number.isFinite(rawIters) ? rawIters : 1000)));
  const warmup = Math.min(1000, Math.max(0, Math.floor(Number.isFinite(rawWarmup) ? rawWarmup : 10)));
  el('benchResult').textContent = 'ATTENTION_SCORE_BENCH_RUNNING';
  setBusy(true, `Benchmarking lc0web WGSL attention scores Q @ Kᵀ * scale: ${warmup} warmup + ${iterations} queued dispatches, one final readback…`);
  try {
    const response = await postWorkerRequest<{ type: 'attentionScoreBenchmarkResult'; result: AttentionScoreBenchmarkResult }>({
      type: 'attentionScoreBenchmark',
      packUrl: PACK_URL,
      iterations,
      warmup,
      verifyShards: params.get('packVerify') !== '0',
    });
    const rounded = {
      ...response.result,
      benchmarkReport: buildBenchmarkReport(response.result),
      scale: Number(response.result.scale.toExponential(6)),
      packLoadMs: Number(response.result.packLoadMs.toFixed(3)),
      uploadSetupMs: Number(response.result.uploadSetupMs.toFixed(3)),
      dispatchLoopMs: Number(response.result.dispatchLoopMs.toFixed(4)),
      dispatchLoopAvgMs: Number(response.result.dispatchLoopAvgMs.toExponential(6)),
      readbackSyncedMs: Number(response.result.readbackSyncedMs.toFixed(4)),
      endToEndMs: Number(response.result.endToEndMs.toFixed(3)),
      maxAbsError: Number(response.result.maxAbsError.toExponential(6)),
      rmsError: Number(response.result.rmsError.toExponential(6)),
      outputSample: response.result.outputSample.map((value) => Number(value.toFixed(6))),
    };
    el('benchResult').textContent = JSON.stringify(rounded);
    el('message').textContent = `ATTENTION_SCORE_BENCH_DONE ${rounded.tokens}x${rounded.tokens} · ${rounded.iterations} queued dispatches · readback-sync ${rounded.readbackSyncedMs.toFixed(3)} ms · max |err| ${rounded.maxAbsError.toExponential(2)}`;
  } catch (error) {
    el('benchResult').textContent = `ATTENTION_SCORE_BENCH_FAILED ${(error as Error).message}`;
    el('message').textContent = `Attention-score benchmark failed: ${(error as Error).message}`;
    throw error;
  } finally {
    setBusy(false);
  }
}

async function runAttentionScoreOrtBenchmark(): Promise<void> {
  if (!searchWorker) throw new Error('attention-score ORT benchmark requires LC0 worker');
  const rawIters = Number(params.get('attentionScoreOrtIters') ?? params.get('scoreOrtIters') ?? params.get('ortBenchIters') ?? '25');
  const rawWarmup = Number(params.get('attentionScoreOrtWarmup') ?? params.get('scoreOrtWarmup') ?? params.get('ortBenchWarmup') ?? '5');
  const iterations = Math.min(1000, Math.max(1, Math.floor(Number.isFinite(rawIters) ? rawIters : 25)));
  const warmup = Math.min(100, Math.max(0, Math.floor(Number.isFinite(rawWarmup) ? rawWarmup : 5)));
  el('benchResult').textContent = 'ATTENTION_SCORE_ORT_BENCH_RUNNING';
  setBusy(true, `Benchmarking ORT tiny attention-score op: ${warmup} warmup + ${iterations} timed runs…`);
  try {
    const response = await postWorkerRequest<{ type: 'attentionScoreOrtBenchmarkResult'; result: AttentionScoreOrtBenchmarkResult }>({
      type: 'attentionScoreOrtBenchmark',
      packUrl: PACK_URL,
      ep: requestedWorkerEp(),
      iterations,
      warmup,
      verifyShards: params.get('packVerify') !== '0',
    });
    const rounded = {
      ...response.result,
      benchmarkReport: buildBenchmarkReport(response.result),
      scale: Number(response.result.scale.toExponential(6)),
      packLoadMs: Number(response.result.packLoadMs.toFixed(3)),
      modelBuildMs: Number(response.result.modelBuildMs.toFixed(3)),
      sessionCreateMs: Number(response.result.sessionCreateMs.toFixed(3)),
      avgMs: Number(response.result.avgMs.toFixed(4)),
      minMs: Number(response.result.minMs.toFixed(4)),
      maxMs: Number(response.result.maxMs.toFixed(4)),
      firstMs: Number(response.result.firstMs.toFixed(4)),
      runsPerSecond: Number(response.result.runsPerSecond.toFixed(3)),
      maxAbsError: Number(response.result.maxAbsError.toExponential(6)),
      rmsError: Number(response.result.rmsError.toExponential(6)),
      outputSample: response.result.outputSample.map((value) => Number(value.toFixed(6))),
    };
    el('benchResult').textContent = JSON.stringify(rounded);
    el('message').textContent = `ATTENTION_SCORE_ORT_BENCH_DONE ${rounded.tokens}x${rounded.tokens} · avg ${rounded.avgMs.toFixed(3)} ms · max |err| ${rounded.maxAbsError.toExponential(2)}`;
  } catch (error) {
    el('benchResult').textContent = `ATTENTION_SCORE_ORT_BENCH_FAILED ${(error as Error).message}`;
    el('message').textContent = `Attention-score ORT benchmark failed: ${(error as Error).message}`;
    throw error;
  } finally {
    setBusy(false);
  }
}

async function runQkvBenchmark(): Promise<void> {
  if (!searchWorker) throw new Error('QKV projection benchmark requires LC0 worker');
  const rawIters = Number(params.get('qkvBenchIters') ?? params.get('qkvIters') ?? params.get('kernelBenchIters') ?? '1000');
  const rawWarmup = Number(params.get('qkvBenchWarmup') ?? params.get('qkvWarmup') ?? params.get('kernelBenchWarmup') ?? '10');
  const iterations = Math.min(100_000, Math.max(1, Math.floor(Number.isFinite(rawIters) ? rawIters : 1000)));
  const warmup = Math.min(1000, Math.max(0, Math.floor(Number.isFinite(rawWarmup) ? rawWarmup : 10)));
  el('benchResult').textContent = 'QKV_BENCH_RUNNING';
  setBusy(true, `Benchmarking lc0web WGSL Q/K/V projections: ${warmup} warmup + ${iterations} queued dispatches, one final readback…`);
  try {
    const response = await postWorkerRequest<{ type: 'qkvBenchmarkResult'; result: QkvBenchmarkResult }>({
      type: 'qkvBenchmark',
      packUrl: PACK_URL,
      iterations,
      warmup,
      verifyShards: params.get('packVerify') !== '0',
    });
    const rounded = {
      ...response.result,
      benchmarkReport: buildBenchmarkReport(response.result),
      packLoadMs: Number(response.result.packLoadMs.toFixed(3)),
      uploadSetupMs: Number(response.result.uploadSetupMs.toFixed(3)),
      dispatchLoopMs: Number(response.result.dispatchLoopMs.toFixed(4)),
      dispatchLoopAvgMs: Number(response.result.dispatchLoopAvgMs.toExponential(6)),
      readbackSyncedMs: Number(response.result.readbackSyncedMs.toFixed(4)),
      endToEndMs: Number(response.result.endToEndMs.toFixed(3)),
      maxAbsError: Object.fromEntries(Object.entries(response.result.maxAbsError).map(([key, value]) => [key, Number(value.toExponential(6))])),
      rmsError: Object.fromEntries(Object.entries(response.result.rmsError).map(([key, value]) => [key, Number(value.toExponential(6))])),
      outputSample: Object.fromEntries(Object.entries(response.result.outputSample).map(([key, values]) => [key, values.map((value) => Number(value.toFixed(6)))])),
    };
    const maxErr = Math.max(...Object.values(response.result.maxAbsError));
    el('benchResult').textContent = JSON.stringify(rounded);
    el('message').textContent = `QKV_BENCH_DONE encoder0 Q/K/V projections · ${rounded.iterations} queued dispatches · dispatch loop ${rounded.dispatchLoopMs.toFixed(3)} ms · readback-sync ${rounded.readbackSyncedMs.toFixed(3)} ms · max |err| ${maxErr.toExponential(2)}`;
  } catch (error) {
    el('benchResult').textContent = `QKV_BENCH_FAILED ${(error as Error).message}`;
    el('message').textContent = `QKV projection benchmark failed: ${(error as Error).message}`;
    throw error;
  } finally {
    setBusy(false);
  }
}

async function runQkvProbe(): Promise<void> {
  if (!searchWorker) throw new Error('QKV projection probe requires LC0 worker');
  const rawIters = Number(params.get('qkvIters') ?? params.get('kernelIters') ?? '10');
  const rawWarmup = Number(params.get('qkvWarmup') ?? params.get('kernelWarmup') ?? '2');
  const iterations = Math.min(1000, Math.max(1, Math.floor(Number.isFinite(rawIters) ? rawIters : 10)));
  const warmup = Math.min(50, Math.max(0, Math.floor(Number.isFinite(rawWarmup) ? rawWarmup : 2)));
  el('benchResult').textContent = 'QKV_RUNNING';
  setBusy(true, `Running lc0web WGSL Q/K/V projection probe: ${warmup} warmup + ${iterations} timed…`);
  try {
    const response = await postWorkerRequest<{ type: 'qkvProbeResult'; result: QkvProbeResult }>({
      type: 'qkvProbe',
      packUrl: PACK_URL,
      iterations,
      warmup,
      verifyShards: params.get('packVerify') !== '0',
    });
    const rounded = {
      ...response.result,
      benchmarkReport: buildBenchmarkReport(response.result),
      packLoadMs: Number(response.result.packLoadMs.toFixed(3)),
      avgMs: Number(response.result.avgMs.toFixed(4)),
      minMs: Number(response.result.minMs.toFixed(4)),
      maxMs: Number(response.result.maxMs.toFixed(4)),
      firstMs: Number(response.result.firstMs.toFixed(4)),
      maxAbsError: Object.fromEntries(Object.entries(response.result.maxAbsError).map(([key, value]) => [key, Number(value.toExponential(6))])),
      rmsError: Object.fromEntries(Object.entries(response.result.rmsError).map(([key, value]) => [key, Number(value.toExponential(6))])),
      outputSample: Object.fromEntries(Object.entries(response.result.outputSample).map(([key, values]) => [key, values.map((value) => Number(value.toFixed(6)))])),
    };
    const maxErr = Math.max(...Object.values(response.result.maxAbsError));
    el('benchResult').textContent = JSON.stringify(rounded);
    el('message').textContent = `QKV_DONE encoder0 Q/K/V projections · avg ${rounded.avgMs.toFixed(3)} ms · max |err| ${maxErr.toExponential(2)}`;
  } catch (error) {
    el('benchResult').textContent = `QKV_FAILED ${(error as Error).message}`;
    el('message').textContent = `QKV projection probe failed: ${(error as Error).message}`;
    throw error;
  } finally {
    setBusy(false);
  }
}

async function runOrtOpBenchmark(): Promise<void> {
  if (!searchWorker) throw new Error('ORT op benchmark requires LC0 worker');
  const rawIters = Number(params.get('ortBenchIters') ?? params.get('kernelBenchIters') ?? params.get('iters') ?? '25');
  const rawWarmup = Number(params.get('ortBenchWarmup') ?? params.get('kernelBenchWarmup') ?? '5');
  const iterations = Math.min(1000, Math.max(1, Math.floor(Number.isFinite(rawIters) ? rawIters : 25)));
  const warmup = Math.min(100, Math.max(0, Math.floor(Number.isFinite(rawWarmup) ? rawWarmup : 5)));
  el('benchResult').textContent = 'ORT_BENCH_RUNNING';
  setBusy(true, `Benchmarking ORT MatMul+Add tiny ONNX op: ${warmup} warmup + ${iterations} timed runs…`);
  try {
    const response = await postWorkerRequest<{ type: 'ortBenchmarkResult'; result: OrtBenchmarkResult }>({
      type: 'ortBenchmark',
      packUrl: PACK_URL,
      ep: requestedWorkerEp(),
      iterations,
      warmup,
      verifyShards: params.get('packVerify') !== '0',
      weightTensorName: params.get('weightTensor') ?? undefined,
      biasTensorName: params.get('biasTensor') ?? undefined,
    });
    const rounded = {
      ...response.result,
      benchmarkReport: buildBenchmarkReport(response.result),
      packLoadMs: Number(response.result.packLoadMs.toFixed(3)),
      modelBuildMs: Number(response.result.modelBuildMs.toFixed(3)),
      sessionCreateMs: Number(response.result.sessionCreateMs.toFixed(3)),
      avgMs: Number(response.result.avgMs.toFixed(4)),
      minMs: Number(response.result.minMs.toFixed(4)),
      maxMs: Number(response.result.maxMs.toFixed(4)),
      firstMs: Number(response.result.firstMs.toFixed(4)),
      runsPerSecond: Number(response.result.runsPerSecond.toFixed(3)),
      maxAbsError: Number(response.result.maxAbsError.toExponential(6)),
      rmsError: Number(response.result.rmsError.toExponential(6)),
      outputSample: response.result.outputSample.map((value) => Number(value.toFixed(6))),
    };
    el('benchResult').textContent = JSON.stringify(rounded);
    el('message').textContent = `ORT_BENCH_DONE tiny MatMul+Add · avg ${rounded.avgMs.toFixed(3)} ms · ${rounded.runsPerSecond.toFixed(1)} run/s · max |err| ${rounded.maxAbsError.toExponential(2)}`;
  } catch (error) {
    el('benchResult').textContent = `ORT_BENCH_FAILED ${(error as Error).message}`;
    el('message').textContent = `ORT op benchmark failed: ${(error as Error).message}`;
    throw error;
  } finally {
    setBusy(false);
  }
}

async function runKernelBenchmark(): Promise<void> {
  if (!searchWorker) throw new Error('kernel benchmark requires LC0 worker');
  const rawKernelIters = Number(params.get('kernelBenchIters') ?? params.get('kernelIters') ?? '1000');
  const rawKernelWarmup = Number(params.get('kernelBenchWarmup') ?? params.get('kernelWarmup') ?? '10');
  const iterations = Math.min(100_000, Math.max(1, Math.floor(Number.isFinite(rawKernelIters) ? rawKernelIters : 1000)));
  const warmup = Math.min(1000, Math.max(0, Math.floor(Number.isFinite(rawKernelWarmup) ? rawKernelWarmup : 10)));
  const variant = requestedKernelVariant();
  el('benchResult').textContent = 'KERNEL_BENCH_RUNNING';
  setBusy(true, `Benchmarking lc0web WGSL MatMul+Add (${variant}): ${warmup} warmup + ${iterations} queued dispatches, one final readback…`);
  try {
    const response = await postWorkerRequest<{ type: 'kernelBenchmarkResult'; result: KernelBenchmarkResult }>({
      type: 'kernelBenchmark',
      packUrl: PACK_URL,
      iterations,
      warmup,
      verifyShards: params.get('packVerify') !== '0',
      weightTensorName: params.get('weightTensor') ?? undefined,
      biasTensorName: params.get('biasTensor') ?? undefined,
      variant,
    });
    const rounded = {
      ...response.result,
      benchmarkReport: buildBenchmarkReport(response.result),
      packLoadMs: Number(response.result.packLoadMs.toFixed(3)),
      uploadSetupMs: Number(response.result.uploadSetupMs.toFixed(3)),
      dispatchLoopMs: Number(response.result.dispatchLoopMs.toFixed(4)),
      dispatchLoopAvgMs: Number(response.result.dispatchLoopAvgMs.toExponential(6)),
      readbackSyncedMs: Number(response.result.readbackSyncedMs.toFixed(4)),
      endToEndMs: Number(response.result.endToEndMs.toFixed(3)),
      maxAbsError: Number(response.result.maxAbsError.toExponential(6)),
      rmsError: Number(response.result.rmsError.toExponential(6)),
      outputSample: response.result.outputSample.map((value) => Number(value.toFixed(6))),
    };
    el('benchResult').textContent = JSON.stringify(rounded);
    el('message').textContent = `KERNEL_BENCH_DONE ${rounded.variant} · ${rounded.iterations} queued dispatches · dispatch loop ${rounded.dispatchLoopMs.toFixed(3)} ms · readback-sync ${rounded.readbackSyncedMs.toFixed(3)} ms · max |err| ${rounded.maxAbsError.toExponential(2)}`;
  } catch (error) {
    el('benchResult').textContent = `KERNEL_BENCH_FAILED ${(error as Error).message}`;
    el('message').textContent = `Kernel benchmark failed: ${(error as Error).message}`;
    throw error;
  } finally {
    setBusy(false);
  }
}

async function runKernelProbe(): Promise<void> {
  if (!searchWorker) throw new Error('kernel probe requires LC0 worker');
  const rawKernelIters = Number(params.get('kernelIters') ?? '25');
  const rawKernelWarmup = Number(params.get('kernelWarmup') ?? '3');
  const iterations = Math.min(1000, Math.max(1, Math.floor(Number.isFinite(rawKernelIters) ? rawKernelIters : 25)));
  const warmup = Math.min(50, Math.max(0, Math.floor(Number.isFinite(rawKernelWarmup) ? rawKernelWarmup : 3)));
  const variant = requestedKernelVariant();
  el('benchResult').textContent = 'KERNEL_RUNNING';
  setBusy(true, `Running lc0web WGSL MatMul+Add kernel probe (${variant}): ${warmup} warmup + ${iterations} timed…`);
  try {
    const response = await postWorkerRequest<{ type: 'kernelProbeResult'; result: KernelProbeResult }>({
      type: 'kernelProbe',
      packUrl: PACK_URL,
      iterations,
      warmup,
      verifyShards: params.get('packVerify') !== '0',
      weightTensorName: params.get('weightTensor') ?? undefined,
      biasTensorName: params.get('biasTensor') ?? undefined,
      variant,
    });
    const rounded = {
      ...response.result,
      benchmarkReport: buildBenchmarkReport(response.result),
      packLoadMs: Number(response.result.packLoadMs.toFixed(3)),
      avgMs: Number(response.result.avgMs.toFixed(4)),
      minMs: Number(response.result.minMs.toFixed(4)),
      maxMs: Number(response.result.maxMs.toFixed(4)),
      firstMs: Number(response.result.firstMs.toFixed(4)),
      maxAbsError: Number(response.result.maxAbsError.toExponential(6)),
      rmsError: Number(response.result.rmsError.toExponential(6)),
      outputSample: response.result.outputSample.map((value) => Number(value.toFixed(6))),
    };
    el('benchResult').textContent = JSON.stringify(rounded);
    el('message').textContent = `KERNEL_DONE ${rounded.variant} 256x256 MatMul+Add · avg ${rounded.avgMs.toFixed(3)} ms · max |err| ${rounded.maxAbsError.toExponential(2)}`;
  } catch (error) {
    el('benchResult').textContent = `KERNEL_FAILED ${(error as Error).message}`;
    el('message').textContent = `Kernel probe failed: ${(error as Error).message}`;
    throw error;
  } finally {
    setBusy(false);
  }
}

async function runWorkerEvalBenchmark(): Promise<void> {
  if (!searchWorkerReady) throw new Error('benchmark requires ready LC0 worker');
  const input = currentEvaluationInput();
  const times: number[] = [];
  let last: BrowserEvaluationChoice | undefined;
  setBusy(true, `Running LC0 worker eval benchmark: ${BENCH_WARMUP} warmup + ${BENCH_ITERS} timed evals…`);
  el('benchResult').textContent = 'BENCH_RUNNING';
  try {
    for (let i = 0; i < BENCH_WARMUP; i++) {
      last = await evaluateWithWorker(input);
      el('benchResult').textContent = `BENCH_WARMUP ${i + 1}/${BENCH_WARMUP}`;
      await new Promise<void>((resolve) => setTimeout(resolve, 0));
    }
    for (let i = 0; i < BENCH_ITERS; i++) {
      const started = performance.now();
      last = await evaluateWithWorker(input);
      times.push(performance.now() - started);
      el('benchResult').textContent = `BENCH_TIMED ${i + 1}/${BENCH_ITERS}`;
      await new Promise<void>((resolve) => setTimeout(resolve, 0));
    }
    const stats = summarizeTimes(times);
    renderBenchmarkResult({
      status: 'BENCH_DONE',
      model: MODEL_URL,
      backend: searchWorkerBackend,
      workerOnly: WORKER_ONLY_MODEL,
      warmup: BENCH_WARMUP,
      iterations: BENCH_ITERS,
      workerInitMs: searchWorkerInitMs,
      timesMs: times,
      bestMove: last?.move,
      q: last?.evaluation.q,
      mlh: last?.evaluation.mlh,
      ...stats,
    });
  } catch (error) {
    el('benchResult').textContent = `BENCH_FAILED ${(error as Error).message}`;
    el('message').textContent = `Benchmark failed: ${(error as Error).message}`;
    throw error;
  } finally {
    setBusy(false);
  }
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
  // Tracks whether a result is on screen so the finally does not re-run the
  // evaluator and overwrite the richer search arrows with the plain best move.
  let rendered = false;
  try {
    const result = await executeSearchResult();
    if (result.cancelled) {
      clearSearchResult();
      el('message').textContent = `Search cancelled (${searchModeLabel()}).`;
    } else {
      renderSearchResult(result);
      rendered = true;
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
    // Keep the search arrows when a result is shown; otherwise refresh the
    // evaluation (which restores the plain best-move arrow).
    if (rendered) renderStatic();
    else renderEvaluation();
  }
}

function stopSearch() {
  if (battleRunning) {
    el('message').textContent = 'Stopping game…';
    battleAbort?.abort();
    // Also abort the in-flight per-move worker search so it stops immediately.
    if (searchWorkerReady && activeWorkerSearchId !== null) searchWorker?.postMessage({ type: 'cancel', target: activeWorkerSearchId });
    return;
  }
  if (!searching) return;
  el('message').textContent = 'Cancelling search…';
  if (useSearchWorker) {
    if (activeWorkerSearchId !== null) searchWorker?.postMessage({ type: 'cancel', target: activeWorkerSearchId });
  } else {
    mainSearchAbort?.abort();
  }
}

async function engineMove() {
  if (!evaluationAvailable() || busy) return;
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
      const choice = await choosePolicyMove(currentEvaluationInput());
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
  if (!evaluationAvailable() || busy) return;
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
      const choice = await choosePolicyMove(input);
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
      el('message').textContent = `Browser FEN-only and explicit-history fixture parity passed (${evaluated} evals via ${WORKER_ONLY_MODEL ? searchWorkerBackend : describeOrtBackendConfig()}).`;
    }
  } catch (error) {
    el('parity').textContent = `failed: ${(error as Error).message}`;
    el('message').textContent = `Parity failed: ${(error as Error).message}`;
  } finally {
    setBusy(false);
    renderEvaluation();
  }
}

// Spin up the search worker lazily so battle search runs off the main thread,
// even when the page was not opened with ?worker=1, without changing the normal
// (main-thread) interactive search path.
async function ensureBattleWorker(): Promise<boolean> {
  if (searchWorkerReady) return true;
  if (!searchWorker) {
    try {
      await initSearchWorker();
    } catch (error) {
      // initSearchWorker may have created the worker before failing; cast past
      // the narrowing of the module-level binding to tear it down.
      (searchWorker as Worker | null)?.terminate();
      searchWorker = null;
      searchWorkerReady = false;
      console.warn('LC0 battle worker unavailable; using main-thread search.', error);
      return false;
    }
  }
  return searchWorkerReady;
}

function battleSleep(ms: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve) => {
    if (ms <= 0 || signal.aborted) return resolve();
    const timer = setTimeout(resolve, ms);
    signal.addEventListener('abort', () => { clearTimeout(timer); resolve(); }, { once: true });
  });
}

type MoveProvider = (positions: BoardState[]) => Promise<string | null>;

// LC0 fixed-visit search move, run in the worker when available so the board
// keeps animating; falls back to a cancellable main-thread search otherwise.
async function battleSearchMove(positions: BoardState[]): Promise<string | null> {
  if (searchWorkerReady) {
    const response = await postWorkerRequest<{ type: 'searchResult'; result: RenderableSearchResult }>(
      { type: 'search', input: { positions }, visits: searchVisits, batchSize: searchBatchSize },
      (id) => { activeWorkerSearchId = id; },
    );
    return response.result.cancelled ? null : (response.result.move ?? null);
  }
  const result = await searcher!.search({ positions }, { visits: searchVisits, batchSize: searchBatchSize, signal: battleAbort!.signal, yieldEveryMs: 16 });
  return result.move ?? null;
}

async function battlePolicyMove(positions: BoardState[]): Promise<string | null> {
  return (await choosePolicyMove({ positions })).move ?? null;
}

function getStockfish(): StockfishEngine {
  if (!stockfish) stockfish = new StockfishEngine({ depth: stockfishDepth });
  else stockfish.setOptions({ depth: stockfishDepth });
  return stockfish;
}

// Stockfish only needs the current FEN, not LC0 history.
async function battleStockfishMove(positions: BoardState[]): Promise<string | null> {
  const current = positions[positions.length - 1];
  return getStockfish().bestMove(boardToFen(current), battleAbort?.signal);
}

function opponentProvider(): { provider: MoveProvider; label: string } {
  if (battleOpponent === 'stockfish') return { provider: battleStockfishMove, label: `Stockfish d${stockfishDepth}` };
  return { provider: battlePolicyMove, label: 'LC0 policy' };
}

// Play one full game on the visible board, animating each ply, so the engines'
// moves are watchable. Reuses the page board/history/move-list state.
async function playGameOnBoard(white: MoveProvider, black: MoveProvider, signal: AbortSignal): Promise<{ result: GameResultCode; reason: string }> {
  loadPosition(parseFen(START_FEN));
  // Show the start position without kicking off an evaluation: that eval shares
  // the main ORT session with the policy/search move providers, and concurrent
  // session.run() on one session is unsafe. The eval panel refreshes when the
  // game ends.
  renderStatic();
  const priorFens: string[] = [];
  const maxPlies = 300;
  for (let ply = 0; ply < maxPlies; ply++) {
    if (signal.aborted) return { result: '1/2-1/2', reason: 'cancelled' };
    const outcome = gameOutcome(board, priorFens);
    if (outcome) return outcome;
    const provider = board.turn === 'w' ? white : black;
    let uci: string | null;
    try {
      uci = await provider(historyBoards);
    } catch (error) {
      if (isAbortError(error)) return { result: '1/2-1/2', reason: 'cancelled' };
      throw error;
    }
    // A null move from a cancelled search must read as cancelled, not a forfeit.
    if (signal.aborted) return { result: '1/2-1/2', reason: 'cancelled' };
    const move = uci ? legalMoveFromUci(uci) : undefined;
    if (!move) return { result: board.turn === 'w' ? '0-1' : '1-0', reason: uci ? `illegal ${uci}` : 'resigned' };
    priorFens.push(boardToFen(board));
    const played = applyMove(move);
    renderStatic();
    setBoardShapes(bestMoveShapes(played));
    await battleSleep(battleDelayMs, signal);
  }
  return { result: '1/2-1/2', reason: 'max plies' };
}

function appendBattleResultLine(text: string) {
  const li = document.createElement('li');
  li.textContent = text;
  el('battleResults').appendChild(li);
}

async function startBattle() {
  if (busy || battleRunning) return;
  await ensureBattleWorker();
  battleRunning = true;
  battleAbort = new AbortController();
  activeWorkerSearchId = null;
  const mode = searchWorkerReady ? 'worker' : 'main thread';
  const { provider: opponentMove, label: opponentLabel } = opponentProvider();
  const lc0Label = `LC0 search ${searchVisits}`;
  setBusy(true, `Watching ${lc0Label} vs ${opponentLabel} (${mode})… press Stop to end.`);
  el('battleResults').innerHTML = '';
  let aWins = 0, bWins = 0, draws = 0, played = 0, cancelled = false;
  try {
    for (let game = 0; game < battleGames; game++) {
      if (battleAbort.signal.aborted) { cancelled = true; break; }
      const aIsWhite = game % 2 === 0;
      el('battleSummary').textContent = `game ${game + 1}/${battleGames}: ${lc0Label} is ${aIsWhite ? 'White' : 'Black'} · playing…`;
      const outcome = await playGameOnBoard(
        aIsWhite ? battleSearchMove : opponentMove,
        aIsWhite ? opponentMove : battleSearchMove,
        battleAbort.signal,
      );
      if (outcome.reason === 'cancelled') { cancelled = true; break; }
      played += 1;
      if (outcome.result === '1/2-1/2') draws += 1;
      else if ((outcome.result === '1-0') === aIsWhite) aWins += 1;
      else bWins += 1;
      appendBattleResultLine(`game ${game + 1}: ${outcome.result} (${outcome.reason}) · LC0 ${aIsWhite ? 'White' : 'Black'}`);
      el('battleSummary').textContent = `${lc0Label} ${aWins}W ${bWins}L ${draws}D vs ${opponentLabel} · ${played}/${battleGames}`;
    }
    el('message').textContent = cancelled
      ? `Game stopped (LC0 ${aWins}W ${bWins}L ${draws}D vs ${opponentLabel} over ${played} game(s)).`
      : `Done: LC0 scored ${aWins + draws * 0.5}/${played} vs ${opponentLabel}.`;
  } catch (error) {
    el('battleSummary').textContent = `failed: ${(error as Error).message}`;
    el('message').textContent = `Battle failed: ${(error as Error).message}`;
  } finally {
    battleRunning = false;
    battleAbort = null;
    activeWorkerSearchId = null;
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

function renderWorkerGpuStatus(backend: string) {
  const node = el('gpuStatus');
  const requestedGpu = requestedWorkerEp() !== 'wasm';
  const usingGpu = backend.includes('webgpu->webgpu') || backend === 'webgpu';
  node.textContent = usingGpu
    ? 'active (worker-only)'
    : requestedGpu ? `worker ${backend} — GPU fallback` : `worker ${backend}`;
  node.classList.toggle('warn', requestedGpu && !usingGpu);
}

async function init() {
  el('message').textContent = PACK_PROBE_REQUESTED ? 'Preparing dedicated worker for lc0web pack probe…' : WORKER_ONLY_MODEL ? 'Loading LC0 model in dedicated worker…' : 'Loading LC0 ONNX model…';
  renderStatic();
  try {
    if (PACK_PROBE_REQUESTED) {
      mainModelCacheStatus = 'pack-probe worker-only (no ONNX session)';
      workerModelCacheStatus = 'pack shards worker-owned';
      useSearchWorker = true;
      await initSearchWorker({ initModel: false });
      searchWorkerBackend = ENCODER_STACK_BENCH_REQUESTED ? 'lc0web-wgsl-encoder-stack-bench' : ENCODER0_BLOCK_ORT_BENCH_REQUESTED ? 'ort-tiny-encoder0-block-bench' : ENCODER0_BLOCK_BENCH_REQUESTED ? 'lc0web-wgsl-encoder0-block-bench' : ENCODER0_FFN_ORT_BENCH_REQUESTED ? 'ort-tiny-encoder0-ffn-bench' : ENCODER0_FFN_BENCH_REQUESTED ? 'lc0web-wgsl-encoder0-ffn-bench' : ATTENTION_OUTPUT_ORT_BENCH_REQUESTED ? 'ort-tiny-attention-output-bench' : ATTENTION_OUTPUT_BENCH_REQUESTED ? 'lc0web-wgsl-attention-output-bench' : ATTENTION_BLOCK_BENCH_REQUESTED ? 'lc0web-wgsl-attention-block-bench' : ATTENTION_VALUE_ORT_BENCH_REQUESTED ? 'ort-tiny-attention-value-bench' : ATTENTION_VALUE_BENCH_REQUESTED ? 'lc0web-wgsl-attention-value-bench' : SOFTMAX_BENCH_REQUESTED ? 'lc0web-wgsl-softmax-bench' : ATTENTION_SCORE_ORT_BENCH_REQUESTED ? 'ort-tiny-attention-score-bench' : ATTENTION_SCORE_BENCH_REQUESTED ? 'lc0web-wgsl-attention-score-bench' : QKV_BENCH_REQUESTED ? 'lc0web-wgsl-qkv-bench' : QKV_PROBE_REQUESTED ? 'lc0web-wgsl-qkv-probe' : ORT_OP_BENCH_REQUESTED ? 'ort-tiny-matmul-add-bench' : KERNEL_BENCH_REQUESTED ? 'lc0web-wgsl-kernel-bench' : KERNEL_PROBE_REQUESTED ? 'lc0web-wgsl-kernel' : 'lc0web-pack-loader';
      renderStatic();
      if (ENCODER_STACK_BENCH_REQUESTED) await runEncoderStackBenchmark();
      else if (ENCODER0_BLOCK_ORT_BENCH_REQUESTED) await runEncoder0BlockOrtBenchmark();
      else if (ENCODER0_BLOCK_BENCH_REQUESTED) await runEncoder0BlockBenchmark();
      else if (ENCODER0_FFN_ORT_BENCH_REQUESTED) await runEncoder0FfnOrtBenchmark();
      else if (ENCODER0_FFN_BENCH_REQUESTED) await runEncoder0FfnBenchmark();
      else if (ATTENTION_OUTPUT_ORT_BENCH_REQUESTED) await runAttentionOutputOrtBenchmark();
      else if (ATTENTION_OUTPUT_BENCH_REQUESTED) await runAttentionOutputBenchmark();
      else if (ATTENTION_BLOCK_BENCH_REQUESTED) await runAttentionBlockBenchmark();
      else if (ATTENTION_VALUE_ORT_BENCH_REQUESTED) await runAttentionValueOrtBenchmark();
      else if (ATTENTION_VALUE_BENCH_REQUESTED) await runAttentionValueBenchmark();
      else if (SOFTMAX_BENCH_REQUESTED) await runSoftmaxBenchmark();
      else if (ATTENTION_SCORE_ORT_BENCH_REQUESTED) await runAttentionScoreOrtBenchmark();
      else if (ATTENTION_SCORE_BENCH_REQUESTED) await runAttentionScoreBenchmark();
      else if (QKV_BENCH_REQUESTED) await runQkvBenchmark();
      else if (QKV_PROBE_REQUESTED) await runQkvProbe();
      else if (ORT_OP_BENCH_REQUESTED) await runOrtOpBenchmark();
      else if (KERNEL_BENCH_REQUESTED) await runKernelBenchmark();
      else if (KERNEL_PROBE_REQUESTED) await runKernelProbe();
      else await runPackProbe();
      return;
    }
    if (WORKER_ONLY_MODEL) {
      mainModelCacheStatus = 'worker-only (not loaded on main thread)';
      useSearchWorker = true;
      await initSearchWorker();
      renderWorkerGpuStatus(searchWorkerBackend);
    } else {
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
    }
    el('message').textContent = WORKER_ONLY_MODEL
      ? 'Ready. LC0 model is loaded only in the dedicated worker.'
      : 'Ready. Drag a legal move or ask the engine to move.';
    if (BENCH_REQUESTED) await runWorkerEvalBenchmark();
    else renderEvaluation();
    if (!BENCH_REQUESTED && (params.get('parity') === '1' || params.get('fixtures') === '1')) await runParityFixtures();
    if (!BENCH_REQUESTED && params.get('search') === '1') await searchRootPosition();
    if (!BENCH_REQUESTED && params.get('engineMove') === '1') await engineMove();
  } catch (error) {
    el('message').textContent = `Model load failed: ${(error as Error).message}`;
    renderStatic();
  }
}

function seedSettingsInputs() {
  inputEl('visitsInput').value = String(searchVisits);
  inputEl('batchInput').value = String(searchBatchSize);
  inputEl('multiPvInput').value = String(searchMultiPv);
  inputEl('battleGamesInput').value = String(battleGames);
  inputEl('sfDepthInput').value = String(stockfishDepth);
  selectEl('opponentSelect').value = battleOpponent;
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
inputEl('battleGamesInput').addEventListener('change', () => {
  battleGames = clampInt(inputEl('battleGamesInput').value, 1, 100, battleGames);
  inputEl('battleGamesInput').value = String(battleGames);
});
inputEl('sfDepthInput').addEventListener('change', () => {
  stockfishDepth = clampInt(inputEl('sfDepthInput').value, 1, 20, stockfishDepth);
  inputEl('sfDepthInput').value = String(stockfishDepth);
});
selectEl('opponentSelect').addEventListener('change', () => {
  battleOpponent = selectEl('opponentSelect').value === 'stockfish' ? 'stockfish' : 'policy';
});
el('battleStart').addEventListener('click', () => { void startBattle(); });
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
