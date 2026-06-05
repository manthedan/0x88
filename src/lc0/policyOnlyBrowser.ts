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
import { Lc0PuctSearcher, type Lc0SearchChild, type Lc0SearchOptions, type Lc0SearchResult } from './search.ts';
import { StockfishEngine } from './stockfishEngine.ts';
import type { CpuctSchedule, FpuStrategy, SearchBatchCollisionMode, SearchEarlyStop } from '../search/puct.ts';

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

type KernelVariant = 'scalar' | 'tiled16' | 'scalar-transposed' | 'scalar-shader-f16-accum-f32';

type KernelProbeResult = {
  status: 'KERNEL_DONE';
  packUrl: string;
  modelName: string;
  adapterInfo?: Record<string, unknown>;
  shaderF16Supported?: boolean;
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

type WgslDeferredReadbackBenchmarkResult = {
  status: 'WGSL_DEFERRED_READBACK_BENCH_DONE';
  backend: string;
  stableBackend: string;
  legalPriorsBackend?: 'js' | 'wasm' | 'gpu';
  batchSize: number;
  iterations: number;
  warmup: number;
  inputCount: number;
  allBestMovesMatch: boolean;
  immediate: { wallMs: number; evalsPerSecond: number; timingMeans: Record<string, number>; bestMoves: Array<string | undefined> };
  deferred: { wallMs: number; evalsPerSecond: number; timingMeans: Record<string, number>; bestMoves: Array<string | undefined> };
};

type BrowserMemorySample = {
  usedJSHeapSize?: number;
  totalJSHeapSize?: number;
  jsHeapSizeLimit?: number;
  userAgentSpecificBytes?: number;
  unavailableReason?: string;
};

type KernelBenchmarkResult = {
  status: 'KERNEL_BENCH_DONE';
  packUrl: string;
  modelName: string;
  adapterInfo?: Record<string, unknown>;
  shaderF16Supported?: boolean;
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
  fusedScoreSoftmax?: boolean;
  qkvKernelVariant?: 'hand' | 'tvm-packed-f16';
  dispatchesPerIteration?: number;
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
  outProjKernelVariant?: 'hand' | 'tvm-packed-f16';
  dispatchesPerIteration?: number;
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

type MappedPolicyProbeResult = {
  status: 'MAPPED_POLICY_PROBE_DONE';
  adapterInfo?: Record<string, unknown>;
  outputs: number;
  normalOutputs: number;
  promotionOutputs: number;
  pipelineCompileMs: number;
  dispatchSyncedMs: number;
  readbackSyncedMs: number;
  maxAbsError: number;
  rmsError: number;
  normalMaxAbsError: number;
  promotionMaxAbsError: number;
  normalSample: number[];
  promotionSample: number[];
  outputSample: number[];
  nonzero: boolean;
  nonuniform: boolean;
};

type WgslHeadsProbeResult = {
  status: 'WGSL_HEADS_PROBE_DONE';
  packUrl: string;
  modelName: string;
  adapterInfo?: Record<string, unknown>;
  tokens: number;
  channels: number;
  valueEmbedChannels: number;
  packLoadMs: number;
  pipelineCompileMs: number;
  dispatchSyncedMs: number;
  readbackSyncedMs: number;
  policyDenseMaxAbsError: number;
  policyDenseRmsError: number;
  policyLogitsMaxAbsError: number;
  policyLogitsRmsError: number;
  mappedPolicyMaxAbsError: number;
  mappedPolicyRmsError: number;
  valueEmbedMaxAbsError: number;
  valueEmbedRmsError: number;
  wgslWdlMaxAbsError: number;
  wgslWdlRmsError: number;
  policyDenseSample: number[];
  policyLogitsSample: number[];
  mappedPolicySample: number[];
  valueEmbedSample: number[];
  wgslWdl: number[];
  nonzero: { policyDense: boolean; policyLogits: boolean; mappedPolicy: boolean; valueEmbed: boolean; wgslWdl: boolean };
  nonuniform: { policyDense: boolean; policyLogits: boolean; mappedPolicy: boolean; valueEmbed: boolean; wgslWdl: boolean };
  ortHeads: {
    mode: 'ort-policy-value';
    runMs: number;
    mappedPolicySample: number[];
    wdl: number[];
    wdlMaxAbsError: number;
  };
};

type WgslHeadsVsOrtFixturesResult = {
  status: 'WGSL_HEADS_VS_ORT_FIXTURES_DONE';
  backend: 'lc0web-wgsl-encoder-wgsl-heads-probe';
  stableBackend: 'lc0web-wgsl-encoder-ort-heads';
  packUrl: string;
  layers: number;
  fixtures: number;
  mappedPolicyTolerance: number;
  wdlTolerance: number;
  bestMoveMatches: number;
  maxMappedPolicyAbsDiff: number;
  maxWdlAbsDiff: number;
  evaluations: Array<{
    id: string;
    fen: string;
    encoderDispatchSyncedMs: number;
    wgslDispatchSyncedMs: number;
    wgslReadbackSyncedMs: number;
    ortRunMs: number;
    mappedPolicyMaxAbsDiff: number;
    mappedPolicyRmsDiff: number;
    wdlMaxAbsDiff: number;
    wdlRmsDiff: number;
    wgslBestMove?: string;
    ortBestMove?: string;
    bestMoveMatch: boolean;
    wgslWdl: number[];
    ortWdl: number[];
    wgslMappedPolicySample: number[];
    ortMappedPolicySample: number[];
  }>;
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
  shaderF16Supported?: boolean;
  tokens: number;
  channels: number;
  hidden: number;
  epsilon: number;
  alpha: number;
  ffnKernelVariant: 'hand' | 'tvm-packed-f16' | 'hand-shader-f16-accum-f32';
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

type HybridEncoderProfileMode = 'sync-staged' | 'gpu-timestamp';

type HybridEncoderProfileResult = {
  status: 'HYBRID_ENCODER_PROFILE_DONE';
  packUrl: string;
  layers: number;
  encoderKernelVariant: 'hand' | 'tvm-packed-f16' | 'mixed-tvm-ffn' | 'mixed-tvm-ffn-outproj';
  inputBackend: 'js' | 'wgsl' | 'wasm';
  warmup: number;
  iterations: number;
  packLoadMs: number;
  profileMode: HybridEncoderProfileMode;
  requestedProfileMode: HybridEncoderProfileMode;
  gpuTimestampSupported: boolean;
  profiledStageTotalMs: number;
  readbackSyncedMs: number;
  outputSample: number[];
  aggregateStageTimings: Array<{ stage: string; label: string; iterations: number; totalMs: number; avgMs: number; percentOfProfiledStageMs: number }>;
  layerTimings: Array<{ layer: number; totalMs: number; stages: Array<{ stage: string; label: string; iterations: number; totalMs: number; avgMs: number; percentOfProfiledStageMs: number }> }>;
  note: string;
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
  | { type: 'wgslHeadsVsOrtFixturesResult'; id: number; result: WgslHeadsVsOrtFixturesResult }
  | { type: 'mappedPolicyProbeResult'; id: number; result: MappedPolicyProbeResult }
  | { type: 'encoder0FfnBenchmarkResult'; id: number; result: Encoder0FfnBenchmarkResult }
  | { type: 'encoder0FfnOrtBenchmarkResult'; id: number; result: Encoder0FfnOrtBenchmarkResult }
  | { type: 'hybridEncoderProfileResult'; id: number; result: HybridEncoderProfileResult }
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
const MAPPED_POLICY_PROBE_REQUESTED = params.get('mappedPolicyProbe') === '1' || params.get('policyMappingProbe') === '1';
const WGSL_HEADS_PROBE_REQUESTED = params.get('wgslHeadsProbe') === '1' || params.get('policyValueHeadsProbe') === '1';
const WGSL_HEADS_VS_ORT_FIXTURES_REQUESTED = params.get('wgslHeadsVsOrt') === '1' || params.get('wgslHeadsFixtures') === '1';
const ENCODER0_FFN_BENCH_REQUESTED = params.get('encoder0FfnBench') === '1' || params.get('ffnBench') === '1';
const ENCODER0_FFN_ORT_BENCH_REQUESTED = params.get('encoder0FfnOrtBench') === '1' || params.get('ffnOrtBench') === '1';
const ATTENTION_SCORE_BENCH_REQUESTED = params.get('attentionScoreBench') === '1' || params.get('scoreBench') === '1';
const ATTENTION_SCORE_ORT_BENCH_REQUESTED = params.get('attentionScoreOrtBench') === '1' || params.get('scoreOrtBench') === '1';
const QKV_BENCH_REQUESTED = params.get('qkvBench') === '1' || params.get('qkvBenchmark') === '1';
const QKV_PROBE_REQUESTED = params.get('qkvProbe') === '1';
const ORT_OP_BENCH_REQUESTED = params.get('ortOpBench') === '1' || params.get('ortBench') === '1';
const KERNEL_BENCH_REQUESTED = params.get('kernelBench') === '1' || params.get('kernelBenchmark') === '1' || params.get('wgslBench') === '1';
const SHADER_F16_PROBE_REQUESTED = params.get('shaderF16Probe') === '1' || params.get('shader-f16-probe') === '1';
const KERNEL_PROBE_REQUESTED = MAPPED_POLICY_PROBE_REQUESTED || WGSL_HEADS_PROBE_REQUESTED || WGSL_HEADS_VS_ORT_FIXTURES_REQUESTED || ENCODER_STACK_BENCH_REQUESTED || ENCODER0_BLOCK_ORT_BENCH_REQUESTED || ENCODER0_BLOCK_BENCH_REQUESTED || ENCODER0_FFN_ORT_BENCH_REQUESTED || ENCODER0_FFN_BENCH_REQUESTED || ATTENTION_OUTPUT_ORT_BENCH_REQUESTED || ATTENTION_OUTPUT_BENCH_REQUESTED || ATTENTION_BLOCK_BENCH_REQUESTED || ATTENTION_VALUE_ORT_BENCH_REQUESTED || ATTENTION_VALUE_BENCH_REQUESTED || SOFTMAX_BENCH_REQUESTED || ATTENTION_SCORE_BENCH_REQUESTED || ATTENTION_SCORE_ORT_BENCH_REQUESTED || QKV_BENCH_REQUESTED || QKV_PROBE_REQUESTED || ORT_OP_BENCH_REQUESTED || KERNEL_BENCH_REQUESTED || params.get('kernelProbe') === '1' || params.get('wgslProbe') === '1';
const BENCH_REQUESTED = params.get('bench') === '1' || params.get('timing') === '1';
const HYBRID_DRIFT_REQUESTED = params.get('hybridDrift') === '1' || params.get('hybridFixtures') === '1';
const HYBRID_SEARCH_FIXTURE_PARITY_REQUESTED = params.get('hybridSearchFixtureParity') === '1' || params.get('searchFixtureParity') === '1';
const HYBRID_SEARCH_BENCH_REQUESTED = params.get('hybridSearchBench') === '1' || params.get('hybridSearchBenchmark') === '1';
const HYBRID_ENCODER_PROFILE_REQUESTED = params.get('hybridEncoderProfile') === '1' || params.get('encoderProfile') === '1' || params.get('hybridProfile') === '1';
const HYBRID_INPUT_BENCH_REQUESTED = params.get('hybridInputBench') === '1' || params.get('hybridInputBenchmark') === '1' || params.get('wasmInputBench') === '1';
const HYBRID_DEFERRED_READBACK_BENCH_REQUESTED = params.get('wgslDeferredReadbackBench') === '1' || params.get('deferredReadbackBench') === '1';
const HYBRID_DEFERRED_READBACK_LIFECYCLE_REQUESTED = params.get('wgslDeferredReadbackLifecycle') === '1' || params.get('deferredReadbackLifecycle') === '1' || params.get('wgslLifecycleSmoke') === '1';
const HYBRID_WGSL_HEADS_REQUESTED = params.get('headBackend') === 'wgsl' || params.get('hybridHeads') === 'wgsl' || params.get('runtime') === 'hybrid-wgsl-heads' || params.get('runtime') === 'wgsl-heads';
const HYBRID_WGSL_BATCH_MODE = params.get('wgslBatchMode') === 'serial' || params.get('wgslBatch') === 'serial' ? 'serial' : 'physical';
const HYBRID_INPUT_BACKEND_PARAM = params.get('inputBackend') ?? params.get('hybridInput');
const HYBRID_INPUT_BACKEND_REQUESTED = HYBRID_INPUT_BACKEND_PARAM === 'wgsl' || HYBRID_INPUT_BACKEND_PARAM === 'wasm';
const HYBRID_INPUT_BACKEND = HYBRID_INPUT_BACKEND_PARAM === 'wasm' ? 'wasm' : (HYBRID_INPUT_BACKEND_PARAM === 'wgsl' ? 'wgsl' : 'js');
const HYBRID_LEGAL_PRIORS_BACKEND_PARAM = params.get('legalPriorsBackend') ?? params.get('hybridLegalPriors');
const HYBRID_LEGAL_PRIORS_BACKEND_REQUESTED = HYBRID_LEGAL_PRIORS_BACKEND_PARAM === 'wasm' || HYBRID_LEGAL_PRIORS_BACKEND_PARAM === 'gpu';
const HYBRID_LEGAL_PRIORS_BACKEND = HYBRID_LEGAL_PRIORS_BACKEND_PARAM === 'gpu' ? 'gpu' : (HYBRID_LEGAL_PRIORS_BACKEND_PARAM === 'wasm' ? 'wasm' : 'js');
const HYBRID_ENCODER_KERNEL_PARAM = params.get('encoderKernel') ?? params.get('hybridEncoderKernel') ?? params.get('encoderKernelVariant');
const HYBRID_ENCODER_KERNEL_VARIANT = HYBRID_ENCODER_KERNEL_PARAM === 'tvm-packed-f16' || HYBRID_ENCODER_KERNEL_PARAM === 'mixed-tvm-ffn' || HYBRID_ENCODER_KERNEL_PARAM === 'mixed-tvm-ffn-outproj' ? HYBRID_ENCODER_KERNEL_PARAM : 'hand';
const HYBRID_EVALUATOR_REQUESTED = HYBRID_DRIFT_REQUESTED || HYBRID_SEARCH_FIXTURE_PARITY_REQUESTED || HYBRID_SEARCH_BENCH_REQUESTED || HYBRID_ENCODER_PROFILE_REQUESTED || HYBRID_INPUT_BENCH_REQUESTED || HYBRID_DEFERRED_READBACK_BENCH_REQUESTED || HYBRID_DEFERRED_READBACK_LIFECYCLE_REQUESTED || HYBRID_WGSL_HEADS_REQUESTED || HYBRID_INPUT_BACKEND_REQUESTED || HYBRID_LEGAL_PRIORS_BACKEND_REQUESTED || HYBRID_ENCODER_KERNEL_VARIANT !== 'hand' || params.get('runtime') === 'hybrid' || params.get('hybridEvaluator') === '1' || params.get('lc0webHybrid') === '1';
const PACK_PROBE_REQUESTED = !HYBRID_EVALUATOR_REQUESTED && (KERNEL_PROBE_REQUESTED || params.get('packProbe') === '1' || params.get('pack') !== null || params.get('modelPack') !== null);
const WORKER_ONLY_MODEL = HYBRID_EVALUATOR_REQUESTED || PACK_PROBE_REQUESTED || BENCH_REQUESTED || params.get('workerOnly') === '1' || params.get('dedicatedWorker') === '1' || params.get('bigModel') === '1';
const SEARCH_WORKER_REQUESTED = WORKER_ONLY_MODEL || params.get('worker') === '1' || params.get('searchWorker') === '1';
const CACHE_MODEL = params.get('cache') === '1' || params.get('modelCache') === '1';
const HYBRID_EVAL_CACHE_ENTRIES = clampInt(params.get('evalCacheEntries') ?? (params.get('evalCache') === '1' ? '2048' : '0'), 0, 100000, 0);
function paramTruthy(name: string): boolean {
  const value = params.get(name);
  return value !== null && !['0', 'false', 'no', 'off'].includes(value.toLowerCase());
}

function paramFalsey(name: string): boolean {
  const value = params.get(name);
  return value !== null && ['0', 'false', 'no', 'off'].includes(value.toLowerCase());
}

const ORT_READBACK_PROFILE_REQUESTED = paramTruthy('ortReadbackProfile') || paramTruthy('ortDiagnostics');
const ORT_WEBGPU_PROFILE_REQUESTED = !paramFalsey('ortWebGpuProfile') && !paramFalsey('ortKernelProfile') && (ORT_READBACK_PROFILE_REQUESTED || paramTruthy('ortWebGpuProfile') || paramTruthy('ortKernelProfile'));
const ORT_WEBGPU_API_TRACE_REQUESTED = !paramFalsey('ortMonkeyPatchWebGpu') && !paramFalsey('ortWebGpuApiTrace') && (ORT_READBACK_PROFILE_REQUESTED || paramTruthy('ortMonkeyPatchWebGpu') || paramTruthy('ortWebGpuApiTrace'));
const ORT_PREFERRED_OUTPUT_LOCATION = params.get('ortPreferredOutputLocation') === 'cpu' || params.get('ortPreferredOutputLocation') === 'cpu-pinned' || params.get('ortPreferredOutputLocation') === 'gpu-buffer'
  ? params.get('ortPreferredOutputLocation') as 'cpu' | 'cpu-pinned' | 'gpu-buffer'
  : (!paramFalsey('ortGpuOutputs') && (ORT_READBACK_PROFILE_REQUESTED || paramTruthy('ortGpuOutputs')) ? 'gpu-buffer' : undefined);
const BENCH_WARMUP = Math.min(100, Math.max(0, Math.floor(Number(params.get('benchWarmup') ?? '5') || 0)));
const BENCH_ITERS = Math.min(1000, Math.max(1, Math.floor(Number(params.get('benchIters') ?? params.get('iters') ?? '25') || 25)));
function requestedKernelVariant(): KernelVariant {
  const value = params.get('kernelVariant') ?? params.get('variant');
  return value === 'tiled16' || value === 'scalar-transposed' || value === 'scalar-shader-f16-accum-f32' ? value : 'scalar';
}
// Register the offline app-shell SW in production builds, or opt in with ?sw=1.
// Disabled in dev by default so it never serves stale HMR modules.
const SW_ENABLED = params.get('sw') === '1'
  || (params.get('sw') !== '0' && (import.meta as { env?: { PROD?: boolean } }).env?.PROD === true);

function parseEarlyStop(raw: string | null): SearchEarlyStop {
  const normalized = (raw ?? 'none').toLowerCase().replace(/[ _]/g, '-');
  if (normalized === 'root-dominance' || normalized === 'best-stable' || normalized === 'kld-stable') return normalized;
  return 'none';
}

function parseCpuctSchedule(raw: string | null): CpuctSchedule {
  return raw === 'constant' ? 'constant' : 'lc0-log';
}

function parseFpuStrategy(raw: string | null): FpuStrategy {
  return raw === 'constant' ? 'constant' : 'lc0-reduction';
}

function parseBatchCollisionMode(raw: string | null): SearchBatchCollisionMode {
  return raw === 'backup' ? 'backup' : 'retry';
}

function clampFloat(value: string | null, min: number, max: number, fallback: number): number {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return fallback;
  return Math.min(max, Math.max(min, parsed));
}

// Runtime-adjustable settings: seeded from query params, then driven by the UI.
let playerSide: 'white' | 'black' = params.get('side') === 'black' ? 'black' : 'white';
let searchVisits = clampInt(params.get('visits') ?? '32', 1, 100000, 32);
let searchBatchSize = clampInt(params.get('batch') ?? params.get('batchSize') ?? '1', 1, 512, 1);
let searchBatchPipelineDepth = clampInt(params.get('batchPipelineDepth') ?? params.get('pipelineDepth') ?? '1', 1, 16, 1);
let searchBatchCollisionMode: SearchBatchCollisionMode = parseBatchCollisionMode(params.get('collision') ?? params.get('batchCollisionMode'));
let searchMultiPv = clampInt(params.get('multipv') ?? params.get('multiPv') ?? '1', 1, 20, 1);
let searchEarlyStop: SearchEarlyStop = parseEarlyStop(params.get('earlyStop') ?? params.get('stop'));
let searchMovetimeMs = clampInt(params.get('movetime') ?? params.get('movetimeMs') ?? '0', 0, 600000, 0);
let searchCpuct = clampFloat(params.get('cpuct'), 0, 100, 1.5);
let searchCpuctSchedule: CpuctSchedule = parseCpuctSchedule(params.get('cpuctSchedule'));
let searchFpuStrategy: FpuStrategy = parseFpuStrategy(params.get('fpuStrategy'));
let searchFpuReduction = clampFloat(params.get('fpuReduction'), 0, 5, 0.330);
let searchTemperature = clampFloat(params.get('temperature'), 0, 10, 0);
let engineReplyMode: EngineReplyMode = params.get('mode') === 'search' ? 'search' : 'policy';

let board: BoardState = parseFen(params.get('fen') ?? START_FEN);
let historyBoards: BoardState[] = [board];
let ground: Ground | null = null;
let player: Lc0PolicyOnlyPlayer | null = null;
let searcher: Lc0PuctSearcher | null = null;
let mainEvaluator: Lc0OnnxEvaluator | null = null;
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

function requestedOrtDiagnosticsPayload() {
  if (!ORT_WEBGPU_PROFILE_REQUESTED && !ORT_WEBGPU_API_TRACE_REQUESTED && !ORT_PREFERRED_OUTPUT_LOCATION) return undefined;
  return {
    webgpuProfiling: ORT_WEBGPU_PROFILE_REQUESTED,
    webgpuApiInstrumentation: ORT_WEBGPU_API_TRACE_REQUESTED,
    ...(ORT_PREFERRED_OUTPUT_LOCATION ? { preferredOutputLocation: ORT_PREFERRED_OUTPUT_LOCATION } : {}),
  };
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

function currentSearchLimitLabel(): string {
  return searchMovetimeMs > 0 ? `${searchMovetimeMs}ms` : `${searchVisits}`;
}

function currentSearchOptions(extra: Partial<Lc0SearchOptions> = {}): Lc0SearchOptions {
  return {
    ...(searchMovetimeMs > 0 ? { movetimeMs: searchMovetimeMs } : { visits: searchVisits }),
    batchSize: searchBatchSize,
    batchPipelineDepth: searchBatchPipelineDepth,
    batchCollisionMode: searchBatchCollisionMode,
    multiPv: searchMultiPv,
    earlyStop: searchEarlyStop,
    cpuct: searchCpuct,
    cpuctSchedule: searchCpuctSchedule,
    fpuStrategy: searchFpuStrategy,
    fpuReduction: searchFpuReduction,
    temperature: searchTemperature,
    ...extra,
  };
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
  const pipelineText = searchBatchPipelineDepth > 1 ? ` · pipe${searchBatchPipelineDepth}` : '';
  el('searchBatch').textContent = searchEarlyStop === 'none' ? `${searchBatchSize}${pipelineText} · ${searchBatchCollisionMode} · ${searchCpuctSchedule}` : `${searchBatchSize}${pipelineText} · ${searchBatchCollisionMode} · ${searchCpuctSchedule} · ${searchEarlyStop}`;
  el('searchMove').textContent = `Search ${currentSearchLimitLabel()}`;
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
  const stop = result.stats?.stopReason ? ` · ${result.stats.stopReason}` : '';
  el('searchSummary').textContent = `${result.move ?? '—'} · ${result.visits} visits${stop} · Q ${result.value.toFixed(5)}`;
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
  if (!searchWorker) {
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
  }
  if (!initModel || searchWorkerReady) return;
  const initStarted = performance.now();
  const ready = await postWorkerRequest<{ type: 'ready'; backend: string; modelCache: string }>({
    type: 'init',
    modelUrl: MODEL_URL,
    ep: requestedWorkerEp(),
    cacheModel: CACHE_MODEL,
    ortDiagnostics: requestedOrtDiagnosticsPayload(),
    ...(HYBRID_EVALUATOR_REQUESTED ? {
      runtime: 'hybrid',
      packUrl: PACK_URL,
      layers: Math.min(32, Math.max(1, Math.floor(Number(params.get('encoderLayers') ?? params.get('layers') ?? '10') || 10))),
      verifyShards: params.get('packVerify') !== '0',
      headBackend: HYBRID_WGSL_HEADS_REQUESTED ? 'wgsl' : 'ort',
      wgslBatchMode: HYBRID_WGSL_BATCH_MODE,
      inputBackend: HYBRID_INPUT_BACKEND,
      legalPriorsBackend: HYBRID_LEGAL_PRIORS_BACKEND,
      encoderKernelVariant: HYBRID_ENCODER_KERNEL_VARIANT,
      evalCacheEntries: HYBRID_EVAL_CACHE_ENTRIES,
    } : {}),
  });
  searchWorkerInitMs = performance.now() - initStarted;
  searchWorkerReady = true;
  searchWorkerBackend = ready.backend;
  workerModelCacheStatus = ready.modelCache;
  renderStatic();
}

async function initHybridWorkerWithInputBackend(inputBackend: 'js' | 'wgsl' | 'wasm'): Promise<void> {
  if (!searchWorker) await initSearchWorker({ initModel: false });
  const initStarted = performance.now();
  const ready = await postWorkerRequest<{ type: 'ready'; backend: string; modelCache: string }>({
    type: 'init',
    modelUrl: MODEL_URL,
    ep: requestedWorkerEp(),
    cacheModel: CACHE_MODEL,
    ortDiagnostics: requestedOrtDiagnosticsPayload(),
    runtime: 'hybrid',
    packUrl: PACK_URL,
    layers: Math.min(32, Math.max(1, Math.floor(Number(params.get('encoderLayers') ?? params.get('layers') ?? '10') || 10))),
    verifyShards: params.get('packVerify') !== '0',
    headBackend: HYBRID_WGSL_HEADS_REQUESTED ? 'wgsl' : 'ort',
    wgslBatchMode: HYBRID_WGSL_BATCH_MODE,
    inputBackend,
    legalPriorsBackend: HYBRID_LEGAL_PRIORS_BACKEND,
    encoderKernelVariant: HYBRID_ENCODER_KERNEL_VARIANT,
    evalCacheEntries: HYBRID_EVAL_CACHE_ENTRIES,
  });
  searchWorkerInitMs = performance.now() - initStarted;
  searchWorkerReady = true;
  searchWorkerBackend = ready.backend;
  workerModelCacheStatus = ready.modelCache;
}

async function evaluateWithWorker(input: Lc0EvaluatorInput): Promise<BrowserEvaluationChoice> {
  const response = await postWorkerRequest<{ type: 'evaluationResult'; result: Lc0Evaluation }>({
    type: 'evaluate',
    input,
  });
  return { move: response.result.bestMove, evaluation: response.result };
}

async function evaluateBatchWithWorker(inputs: Lc0EvaluatorInput[]): Promise<BrowserEvaluationChoice[]> {
  const response = await postWorkerRequest<{ type: 'evaluationBatchResult'; result: Lc0Evaluation[] }>({
    type: 'evaluateBatch',
    inputs,
  });
  return response.result.map((evaluation) => ({ move: evaluation.bestMove, evaluation }));
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
  const p50 = percentile(0.5);
  const p95 = percentile(0.95);
  return {
    source,
    sampleCount: finite.length,
    meanMs: roundReportMs(mean),
    trimmedMeanMs: roundReportMs(trimmedMean),
    p50Ms: roundReportMs(p50),
    p95Ms: roundReportMs(p95),
    minMs: roundReportMs(finite[0]),
    maxMs: roundReportMs(finite[finite.length - 1]),
    outlierCount: finite.filter((value) => value > p95).length,
  };
}

function roundedNumericRecord(value: unknown, digits = 4): Record<string, number> | undefined {
  if (!value || typeof value !== 'object') return undefined;
  const entries = Object.entries(value as Record<string, unknown>)
    .filter((entry): entry is [string, number] => typeof entry[1] === 'number' && Number.isFinite(entry[1]))
    .map(([key, numberValue]) => [key, Number(numberValue.toFixed(digits))] as const);
  return entries.length ? Object.fromEntries(entries) : undefined;
}

function recordNumericTimingSamples(samples: Record<string, number[]>, value: unknown): void {
  const rounded = roundedNumericRecord(value, 8);
  if (!rounded) return;
  for (const [key, numberValue] of Object.entries(rounded)) {
    (samples[key] ??= []).push(numberValue);
  }
}

function aggregateBatchEvaluationTiming(evaluations: BrowserEvaluationChoice[] | undefined): Record<string, number> | undefined {
  const records = evaluations
    ?.map((entry) => roundedNumericRecord((entry.evaluation as { timing?: unknown }).timing, 8))
    .filter((record): record is Record<string, number> => record !== undefined) ?? [];
  if (!records.length) return undefined;
  if (records[0].physicalBatchSize === records.length) return records[0];
  const totals: Record<string, number> = {};
  for (const record of records) {
    for (const [key, value] of Object.entries(record)) {
      if (key === 'batchPosition' || key === 'physicalBatchSize') continue;
      totals[key] = (totals[key] ?? 0) + value;
    }
  }
  return Object.fromEntries(Object.entries(totals).map(([key, value]) => [key, Number(value.toFixed(8))]));
}

function summarizeNumericTimingSamples(samples: Record<string, number[]>, sourcePrefix: string): Record<string, unknown> | undefined {
  const entries = Object.entries(samples).map(([key, values]) => [key, sampleTimingStats(values, `${sourcePrefix}.${key}`)] as const).filter(([, stats]) => stats !== undefined);
  return entries.length ? Object.fromEntries(entries) : undefined;
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

async function runShaderF16Probe(): Promise<void> {
  el('benchResult').textContent = 'SHADER_F16_PROBE_RUNNING';
  setBusy(true, 'Probing WebGPU shader-f16 feature support…');
  const started = performance.now();
  const gpu = (navigator as Navigator & { gpu?: { requestAdapter: () => Promise<unknown> } }).gpu;
  const maxAbsError = (actual: Float32Array, expected: number[]) => Math.max(0, ...expected.map((value, index) => Math.abs((actual[index] ?? NaN) - value)));
  try {
    if (!gpu) {
      const result = { status: 'SHADER_F16_PROBE_DONE', shaderF16Supported: false, reason: 'navigator.gpu unavailable', maxAbsError: 0, elapsedMs: roundReportMs(performance.now() - started) };
      el('benchResult').textContent = JSON.stringify(result);
      el('message').textContent = 'SHADER_F16_PROBE_DONE unavailable: no navigator.gpu';
      return;
    }
    const adapter = await gpu.requestAdapter() as { features?: Iterable<string> & { has?: (feature: string) => boolean }; requestDevice: (descriptor?: Record<string, unknown>) => Promise<unknown>; info?: Record<string, unknown> } | null;
    if (!adapter) {
      const result = { status: 'SHADER_F16_PROBE_DONE', shaderF16Supported: false, reason: 'WebGPU adapter unavailable', maxAbsError: 0, elapsedMs: roundReportMs(performance.now() - started) };
      el('benchResult').textContent = JSON.stringify(result);
      el('message').textContent = 'SHADER_F16_PROBE_DONE unavailable: no WebGPU adapter';
      return;
    }
    const adapterFeatures = adapter.features ? Array.from(adapter.features).map(String).sort() : [];
    const shaderF16Supported = adapter.features?.has?.('shader-f16') ?? adapterFeatures.includes('shader-f16');
    if (!shaderF16Supported) {
      const result = { status: 'SHADER_F16_PROBE_DONE', shaderF16Supported: false, adapterFeatures, adapterInfo: adapter.info, reason: 'adapter does not advertise shader-f16', maxAbsError: 0, elapsedMs: roundReportMs(performance.now() - started) };
      el('benchResult').textContent = JSON.stringify(result);
      el('message').textContent = 'SHADER_F16_PROBE_DONE unsupported on this adapter';
      return;
    }
    const device = await adapter.requestDevice({ requiredFeatures: ['shader-f16'] }) as any;
    const bufferUsage = (globalThis as any).GPUBufferUsage;
    const mapMode = (globalThis as any).GPUMapMode;
    const outputBuffer = device.createBuffer({ size: 16, usage: bufferUsage.STORAGE | bufferUsage.COPY_SRC });
    const readbackBuffer = device.createBuffer({ size: 16, usage: bufferUsage.MAP_READ | bufferUsage.COPY_DST });
    try {
      const shader = `enable f16;
@group(0) @binding(0) var<storage, read_write> output: array<f32>;
@compute @workgroup_size(1)
fn main() {
  let a = vec4<f16>(f16(1.5), f16(-2.0), f16(0.25), f16(4.0));
  let b = vec4<f16>(f16(2.0), f16(-0.5), f16(8.0), f16(0.125));
  let c = a * b + vec4<f16>(f16(0.5), f16(-1.0), f16(0.25), f16(1.0));
  output[0] = f32(c.x);
  output[1] = f32(c.y);
  output[2] = f32(c.z);
  output[3] = f32(c.w);
}`;
      const pipeline = device.createComputePipeline({ layout: 'auto', compute: { module: device.createShaderModule({ label: 'lc0 shader-f16 feature probe', code: shader }), entryPoint: 'main' } });
      const bindGroup = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: outputBuffer } }] });
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(1);
      pass.end();
      encoder.copyBufferToBuffer(outputBuffer, 0, readbackBuffer, 0, 16);
      device.queue.submit([encoder.finish()]);
      await readbackBuffer.mapAsync(mapMode.READ);
      const actual = new Float32Array(readbackBuffer.getMappedRange().slice(0));
      readbackBuffer.unmap();
      const expected = [3.5, 0, 2.25, 1.5];
      const result = {
        status: 'SHADER_F16_PROBE_DONE',
        shaderF16Supported: true,
        adapterFeatures,
        adapterInfo: adapter.info,
        expected,
        actual: Array.from(actual).map((value) => Number(value.toFixed(6))),
        maxAbsError: maxAbsError(actual, expected),
        elapsedMs: roundReportMs(performance.now() - started),
      };
      el('benchResult').textContent = JSON.stringify(result);
      el('message').textContent = `SHADER_F16_PROBE_DONE supported · max error ${result.maxAbsError}`;
    } finally {
      outputBuffer.destroy();
      readbackBuffer.destroy();
      device.destroy?.();
    }
  } catch (error) {
    el('benchResult').textContent = `SHADER_F16_PROBE_FAILED ${(error as Error).message}`;
    el('message').textContent = `shader-f16 probe failed: ${(error as Error).message}`;
    throw error;
  } finally {
    setBusy(false);
  }
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
  const attentionOutProjKernelVariant = params.get('attentionOutProjKernel') === 'tvm-packed-f16' || params.get('attnOutProjKernel') === 'tvm-packed-f16' ? 'tvm-packed-f16' : 'hand';
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
      attentionOutProjKernelVariant,
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

async function runMappedPolicyProbe(): Promise<void> {
  if (!searchWorker) throw new Error('mapped-policy probe requires LC0 worker');
  el('benchResult').textContent = 'MAPPED_POLICY_PROBE_RUNNING';
  setBusy(true, 'Running tiny synthetic WGSL mapped-policy probe…');
  try {
    const response = await postWorkerRequest<{ type: 'mappedPolicyProbeResult'; result: MappedPolicyProbeResult }>({
      type: 'mappedPolicyProbe',
    });
    const rounded = {
      ...response.result,
      benchmarkReport: buildBenchmarkReport(response.result),
      pipelineCompileMs: Number(response.result.pipelineCompileMs.toFixed(3)),
      dispatchSyncedMs: Number(response.result.dispatchSyncedMs.toFixed(4)),
      readbackSyncedMs: Number(response.result.readbackSyncedMs.toFixed(4)),
      maxAbsError: Number(response.result.maxAbsError.toExponential(6)),
      rmsError: Number(response.result.rmsError.toExponential(6)),
      normalMaxAbsError: Number(response.result.normalMaxAbsError.toExponential(6)),
      promotionMaxAbsError: Number(response.result.promotionMaxAbsError.toExponential(6)),
      normalSample: response.result.normalSample.map((value) => Number(value.toFixed(8))),
      promotionSample: response.result.promotionSample.map((value) => Number(value.toFixed(8))),
      outputSample: response.result.outputSample.map((value) => Number(value.toFixed(8))),
    };
    el('benchResult').textContent = JSON.stringify(rounded);
    el('message').textContent = `MAPPED_POLICY_PROBE_DONE ${rounded.outputs} outputs · normal ${rounded.normalOutputs} max |err| ${rounded.normalMaxAbsError.toExponential(2)} · promotion ${rounded.promotionOutputs} max |err| ${rounded.promotionMaxAbsError.toExponential(2)} · nonzero/nonuniform ${rounded.nonzero && rounded.nonuniform ? 'yes' : 'no'}`;
  } catch (error) {
    el('benchResult').textContent = `MAPPED_POLICY_PROBE_FAILED ${(error as Error).message}`;
    el('message').textContent = `Mapped-policy probe failed: ${(error as Error).message}`;
    throw error;
  } finally {
    setBusy(false);
  }
}

async function runWgslHeadsProbe(): Promise<void> {
  if (!searchWorker) throw new Error('WGSL heads probe requires LC0 worker');
  el('benchResult').textContent = 'WGSL_HEADS_PROBE_RUNNING';
  setBusy(true, 'Running isolated WGSL policy/value head dense probes against a deterministic encoder-shaped input…');
  try {
    const response = await postWorkerRequest<{ type: 'wgslHeadsProbeResult'; result: WgslHeadsProbeResult }>({
      type: 'wgslHeadsProbe',
      packUrl: PACK_URL,
      ep: requestedWorkerEp(),
      verifyShards: params.get('packVerify') !== '0',
    });
    const rounded = {
      ...response.result,
      benchmarkReport: buildBenchmarkReport(response.result),
      packLoadMs: Number(response.result.packLoadMs.toFixed(3)),
      pipelineCompileMs: Number(response.result.pipelineCompileMs.toFixed(3)),
      dispatchSyncedMs: Number(response.result.dispatchSyncedMs.toFixed(4)),
      readbackSyncedMs: Number(response.result.readbackSyncedMs.toFixed(4)),
      policyDenseMaxAbsError: Number(response.result.policyDenseMaxAbsError.toExponential(6)),
      policyDenseRmsError: Number(response.result.policyDenseRmsError.toExponential(6)),
      policyLogitsMaxAbsError: Number(response.result.policyLogitsMaxAbsError.toExponential(6)),
      policyLogitsRmsError: Number(response.result.policyLogitsRmsError.toExponential(6)),
      mappedPolicyMaxAbsError: Number(response.result.mappedPolicyMaxAbsError.toExponential(6)),
      mappedPolicyRmsError: Number(response.result.mappedPolicyRmsError.toExponential(6)),
      valueEmbedMaxAbsError: Number(response.result.valueEmbedMaxAbsError.toExponential(6)),
      valueEmbedRmsError: Number(response.result.valueEmbedRmsError.toExponential(6)),
      wgslWdlMaxAbsError: Number(response.result.wgslWdlMaxAbsError.toExponential(6)),
      wgslWdlRmsError: Number(response.result.wgslWdlRmsError.toExponential(6)),
      policyDenseSample: response.result.policyDenseSample.map((value) => Number(value.toFixed(8))),
      policyLogitsSample: response.result.policyLogitsSample.map((value) => Number(value.toFixed(8))),
      mappedPolicySample: response.result.mappedPolicySample.map((value) => Number(value.toFixed(8))),
      valueEmbedSample: response.result.valueEmbedSample.map((value) => Number(value.toFixed(8))),
      wgslWdl: response.result.wgslWdl.map((value) => Number(value.toFixed(8))),
      ortHeads: {
        ...response.result.ortHeads,
        runMs: Number(response.result.ortHeads.runMs.toFixed(3)),
        mappedPolicySample: response.result.ortHeads.mappedPolicySample.map((value) => Number(value.toFixed(8))),
        wdl: response.result.ortHeads.wdl.map((value) => Number(value.toFixed(8))),
        wdlMaxAbsError: Number(response.result.ortHeads.wdlMaxAbsError.toExponential(6)),
      },
    };
    el('benchResult').textContent = JSON.stringify(rounded);
    el('message').textContent = `WGSL_HEADS_PROBE_DONE policy dense |err| ${rounded.policyDenseMaxAbsError.toExponential(2)} · policy logits |err| ${rounded.policyLogitsMaxAbsError.toExponential(2)} · mapped policy |err| ${rounded.mappedPolicyMaxAbsError.toExponential(2)} · value embed |err| ${rounded.valueEmbedMaxAbsError.toExponential(2)} · WGSL WDL |err| ${rounded.wgslWdlMaxAbsError.toExponential(2)} · nonzero/nonuniform ${rounded.nonzero.policyDense && rounded.nonzero.policyLogits && rounded.nonzero.mappedPolicy && rounded.nonzero.valueEmbed && rounded.nonzero.wgslWdl && rounded.nonuniform.policyDense && rounded.nonuniform.policyLogits && rounded.nonuniform.mappedPolicy && rounded.nonuniform.valueEmbed && rounded.nonuniform.wgslWdl ? 'yes' : 'no'}`;
  } catch (error) {
    el('benchResult').textContent = `WGSL_HEADS_PROBE_FAILED ${(error as Error).message}`;
    el('message').textContent = `WGSL heads probe failed: ${(error as Error).message}`;
    throw error;
  } finally {
    setBusy(false);
  }
}

async function runWgslHeadsVsOrtFixtures(): Promise<void> {
  if (!searchWorker) throw new Error('WGSL heads vs ORT fixture comparison requires LC0 worker');
  const limit = Math.min(16, Math.max(1, Math.floor(Number(params.get('fixtureLimit') ?? params.get('wgslHeadsLimit') ?? '9') || 9)));
  const layers = Math.min(32, Math.max(1, Math.floor(Number(params.get('encoderLayers') ?? params.get('layers') ?? '10') || 10)));
  el('benchResult').textContent = 'WGSL_HEADS_VS_ORT_FIXTURES_RUNNING';
  setBusy(true, `Comparing WGSL heads against ORT heads on ${limit} real hybrid encoder fixture output(s)…`);
  try {
    const records = [
      ...await fetchNativeRecords('/lc0/native_fen_only_blas.jsonl'),
      ...await fetchNativeRecords('/lc0/native_history_blas.jsonl'),
    ].slice(0, limit);
    const fixtures = records.map((native) => ({
      id: native.id,
      input: native.moves ? { positions: buildBoardHistoryFromMoves(native.moves, native.startFen) } : native.fen,
    }));
    const response = await postWorkerRequest<{ type: 'wgslHeadsVsOrtFixturesResult'; result: WgslHeadsVsOrtFixturesResult }>({
      type: 'wgslHeadsVsOrtFixtures',
      packUrl: PACK_URL,
      ep: requestedWorkerEp(),
      fixtures,
      layers,
      verifyShards: params.get('packVerify') !== '0',
      mappedPolicyTolerance: Number(params.get('mappedPolicyTolerance') ?? '0.001'),
      wdlTolerance: Number(params.get('wdlTolerance') ?? '0.001'),
    });
    const rounded = {
      ...response.result,
      maxMappedPolicyAbsDiff: Number(response.result.maxMappedPolicyAbsDiff.toExponential(6)),
      maxWdlAbsDiff: Number(response.result.maxWdlAbsDiff.toExponential(6)),
      evaluations: response.result.evaluations.map((entry) => ({
        ...entry,
        encoderDispatchSyncedMs: Number(entry.encoderDispatchSyncedMs.toFixed(3)),
        wgslDispatchSyncedMs: Number(entry.wgslDispatchSyncedMs.toFixed(3)),
        wgslReadbackSyncedMs: Number(entry.wgslReadbackSyncedMs.toFixed(3)),
        ortRunMs: Number(entry.ortRunMs.toFixed(3)),
        mappedPolicyMaxAbsDiff: Number(entry.mappedPolicyMaxAbsDiff.toExponential(6)),
        mappedPolicyRmsDiff: Number(entry.mappedPolicyRmsDiff.toExponential(6)),
        wdlMaxAbsDiff: Number(entry.wdlMaxAbsDiff.toExponential(6)),
        wdlRmsDiff: Number(entry.wdlRmsDiff.toExponential(6)),
        wgslWdl: entry.wgslWdl.map((value) => Number(value.toFixed(8))),
        ortWdl: entry.ortWdl.map((value) => Number(value.toFixed(8))),
        wgslMappedPolicySample: entry.wgslMappedPolicySample.map((value) => Number(value.toFixed(8))),
        ortMappedPolicySample: entry.ortMappedPolicySample.map((value) => Number(value.toFixed(8))),
      })),
    };
    el('benchResult').textContent = JSON.stringify(rounded);
    el('message').textContent = `WGSL_HEADS_VS_ORT_FIXTURES_DONE ${rounded.bestMoveMatches}/${rounded.fixtures} best moves · mapped max |diff| ${rounded.maxMappedPolicyAbsDiff.toExponential(2)} · WDL max |diff| ${rounded.maxWdlAbsDiff.toExponential(2)}`;
  } catch (error) {
    el('benchResult').textContent = `WGSL_HEADS_VS_ORT_FIXTURES_FAILED ${(error as Error).message}`;
    el('message').textContent = `WGSL heads vs ORT fixture comparison failed: ${(error as Error).message}`;
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
  const rawFfnKernelVariant = params.get('encoder0FfnKernel') ?? params.get('ffnKernel');
  const ffnKernelVariant = rawFfnKernelVariant === 'tvm-packed-f16' || rawFfnKernelVariant === 'hand-shader-f16-accum-f32' ? rawFfnKernelVariant : 'hand';
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
      ffnKernelVariant,
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
  const fusedScoreSoftmax = params.get('attentionFusion') === 'score-softmax' || params.get('fusedScoreSoftmax') === '1';
  const attentionQkvKernelVariant = params.get('attentionQkvKernel') === 'tvm-packed-f16' || params.get('attnQkvKernel') === 'tvm-packed-f16' ? 'tvm-packed-f16' : 'hand';
  el('benchResult').textContent = 'ATTENTION_BLOCK_BENCH_RUNNING';
  setBusy(true, `Benchmarking lc0web WGSL attention block: ${warmup} warmup + ${iterations} queued QKV/QK/softmax/value blocks, one final readback…`);
  try {
    const response = await postWorkerRequest<{ type: 'attentionBlockBenchmarkResult'; result: AttentionBlockBenchmarkResult }>({
      type: 'attentionBlockBenchmark',
      packUrl: PACK_URL,
      iterations,
      warmup,
      verifyShards: params.get('packVerify') !== '0',
      fusedScoreSoftmax,
      attentionQkvKernelVariant,
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

function boundedQueryInt(names: string[], fallback: number, min: number, max: number): number {
  for (const name of names) {
    const raw = params.get(name);
    if (raw === null) continue;
    const value = Math.floor(Number(raw));
    if (Number.isFinite(value)) return Math.min(max, Math.max(min, value));
  }
  return fallback;
}

function queryBool(names: string[], fallback = false): boolean {
  for (const name of names) {
    const raw = params.get(name);
    if (raw === null) continue;
    return !['0', 'false', 'no', 'off'].includes(raw.toLowerCase());
  }
  return fallback;
}

function queryIntList(names: string[], fallback: number[], min: number, max: number): number[] {
  for (const name of names) {
    const raw = params.get(name);
    if (raw === null) continue;
    const values = raw.split(',')
      .map((entry) => Math.floor(Number(entry.trim())))
      .filter((value) => Number.isFinite(value))
      .map((value) => Math.min(max, Math.max(min, value)));
    if (values.length > 0) return [...new Set(values)];
  }
  return fallback;
}

async function resetSearchTreeState(): Promise<void> {
  searcher?.resetTree();
  if (searchWorkerReady) await postWorkerRequest<{ type: 'searchReset' }>({ type: 'resetSearch' });
}

type SearchStatsSnapshot = NonNullable<RenderableSearchResult['stats']>;

function aggregateSearchStats(samples: SearchStatsSnapshot[]) {
  const sum = (key: keyof SearchStatsSnapshot) => samples.reduce((total, stats) => total + (Number(stats[key]) || 0), 0);
  const rootReusedCount = samples.filter((stats) => stats.rootReused).length;
  const cacheHits = sum('cacheHits');
  const neuralEvalMisses = sum('neuralEvalMisses');
  const evalCalls = sum('evalCalls');
  const completedVisits = sum('completedVisits');
  const evalBatchSizeHistogram = samples.reduce<Record<string, number>>((histogram, stats) => {
    for (const [size, count] of Object.entries(stats.evalBatchSizeHistogram ?? {})) histogram[size] = (histogram[size] ?? 0) + count;
    return histogram;
  }, {});
  const evalBatchItems = Object.entries(evalBatchSizeHistogram).reduce((total, [size, count]) => total + Number(size) * count, 0);
  const evalBatchCalls = Object.values(evalBatchSizeHistogram).reduce((total, count) => total + count, 0);
  const evalBackendTimingSamples = sum('evalBackendTimingSamples');
  const evalBackendTimingPositions = sum('evalBackendTimingPositions');
  const evalBackendTimingTotals = samples.reduce<Record<string, number>>((totals, stats) => {
    for (const [key, value] of Object.entries(stats.evalBackendTimingTotals ?? {})) {
      if (typeof value === 'number' && Number.isFinite(value)) totals[key] = (totals[key] ?? 0) + value;
    }
    return totals;
  }, {});
  const evalBackendTimingMeans = evalBackendTimingSamples > 0
    ? Object.fromEntries(Object.entries(evalBackendTimingTotals).map(([key, value]) => [key, roundReportMs(value / evalBackendTimingSamples)]))
    : undefined;
  const evalBackendTimingPerPositionMeans = evalBackendTimingPositions > 0
    ? Object.fromEntries(Object.entries(evalBackendTimingTotals).map(([key, value]) => [key, roundReportMs(value / evalBackendTimingPositions)]))
    : undefined;
  return {
    samples: samples.length,
    rootReusedCount,
    completedVisits,
    evalCalls,
    batchEvalCalls: sum('batchEvalCalls'),
    maxEvalBatch: samples.reduce((max, stats) => Math.max(max, stats.maxEvalBatch ?? 0), 0),
    evalBatchSizeHistogram,
    averageEvalBatchSize: Number((evalBatchItems / Math.max(1, evalBatchCalls)).toFixed(4)),
    cacheHits,
    neuralEvalMisses,
    cacheHitRate: Number((cacheHits / Math.max(1, cacheHits + neuralEvalMisses)).toFixed(6)),
    expansions: sum('expansions'),
    terminalHits: sum('terminalHits'),
    batchLeafCollisions: sum('batchLeafCollisions'),
    batchLeafRetries: sum('batchLeafRetries'),
    evalBackendTimingSamples,
    evalBackendTimingPositions,
    evalBackendTimingTotals: roundedNumericRecord(evalBackendTimingTotals),
    evalBackendTimingMeans,
    evalBackendTimingPerPositionMeans,
    requestedVisits: sum('requestedVisits'),
    stopReasons: samples.reduce<Record<string, number>>((counts, stats) => {
      const reason = stats.stopReason ?? 'unknown';
      counts[reason] = (counts[reason] ?? 0) + 1;
      return counts;
    }, {}),
    effectiveVisitsPerSecondDenominator: completedVisits > 0 ? 'completedVisits' : 'requestedVisits',
  };
}

async function runWorkerEvalBenchmark(): Promise<void> {
  if (!searchWorkerReady) throw new Error('benchmark requires ready LC0 worker');
  const input = currentEvaluationInput();
  const times: number[] = [];
  const backendTimingSamples: Record<string, number[]> = {};
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
      recordNumericTimingSamples(backendTimingSamples, (last.evaluation as { timing?: unknown }).timing);
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
      lastBackendTiming: roundedNumericRecord((last?.evaluation as { timing?: unknown } | undefined)?.timing),
      phaseTimingStats: summarizeNumericTimingSamples(backendTimingSamples, 'onnx worker eval backend timing'),
      ...stats,
    } as EvalBenchResult & Record<string, unknown>);
  } catch (error) {
    el('benchResult').textContent = `BENCH_FAILED ${(error as Error).message}`;
    el('message').textContent = `Benchmark failed: ${(error as Error).message}`;
    throw error;
  } finally {
    setBusy(false);
  }
}

async function runHybridEncoderProfile(): Promise<void> {
  if (!searchWorkerReady) throw new Error('hybrid encoder profile requires ready LC0 worker');
  const input = currentEvaluationInput();
  const iterations = boundedQueryInt(['hybridEncoderProfileIters', 'profileIters'], 1, 1, 20);
  const warmup = boundedQueryInt(['hybridEncoderProfileWarmup', 'profileWarmup'], 1, 0, 10);
  const profileModeParam = params.get('hybridEncoderProfileMode') ?? params.get('encoderProfileMode') ?? params.get('profileMode');
  const profileMode: HybridEncoderProfileMode = profileModeParam === 'sync-staged' || profileModeParam === 'sync' ? 'sync-staged' : 'gpu-timestamp';
  setBusy(true, `Profiling hybrid encoder stages: ${iterations} ${profileMode} pass(es)…`);
  el('benchResult').textContent = 'HYBRID_ENCODER_PROFILE_RUNNING';
  try {
    const response = await postWorkerRequest<{ type: 'hybridEncoderProfileResult'; result: HybridEncoderProfileResult }>({
      type: 'hybridEncoderProfile',
      packUrl: PACK_URL,
      input,
      layers: boundedQueryInt(['encoderLayers', 'layers'], 10, 1, 32),
      iterations,
      warmup,
      verifyShards: params.get('packVerify') !== '0',
      inputBackend: HYBRID_INPUT_BACKEND,
      encoderKernelVariant: HYBRID_ENCODER_KERNEL_VARIANT,
      profileMode,
    });
    const result = {
      ...response.result,
      packLoadMs: roundReportMs(response.result.packLoadMs),
      profiledStageTotalMs: roundReportMs(response.result.profiledStageTotalMs),
      readbackSyncedMs: roundReportMs(response.result.readbackSyncedMs),
      aggregateStageTimings: response.result.aggregateStageTimings.map((timing) => ({
        ...timing,
        totalMs: Number(timing.totalMs.toFixed(4)),
        avgMs: Number(timing.avgMs.toFixed(4)),
        percentOfProfiledStageMs: Number(timing.percentOfProfiledStageMs.toFixed(2)),
      })),
      layerTimings: response.result.layerTimings.map((layer) => ({
        ...layer,
        totalMs: Number(layer.totalMs.toFixed(4)),
        stages: layer.stages.map((timing) => ({
          ...timing,
          totalMs: Number(timing.totalMs.toFixed(4)),
          avgMs: Number(timing.avgMs.toFixed(4)),
          percentOfProfiledStageMs: Number(timing.percentOfProfiledStageMs.toFixed(2)),
        })),
      })),
    };
    el('benchResult').textContent = JSON.stringify(result);
    const top = result.aggregateStageTimings.slice(0, 3).map((timing) => `${timing.stage} ${timing.avgMs.toFixed(2)} ms`).join(' · ');
    el('message').textContent = `HYBRID_ENCODER_PROFILE_DONE ${result.encoderKernelVariant} · top stages ${top}`;
  } catch (error) {
    el('benchResult').textContent = `HYBRID_ENCODER_PROFILE_FAILED ${(error as Error).message}`;
    el('message').textContent = `Hybrid encoder profile failed: ${(error as Error).message}`;
    throw error;
  } finally {
    setBusy(false);
  }
}

async function runHybridSearchBenchmark(): Promise<void> {
  if (!searchWorkerReady) throw new Error('hybrid search benchmark requires ready LC0 worker');
  const input = currentEvaluationInput();
  const evalWarmup = boundedQueryInt(['hybridEvalBenchWarmup', 'evalWarmup'], 1, 0, 20);
  const evalIterations = boundedQueryInt(['hybridEvalBenchIters', 'evalIters'], 3, 0, 100);
  const batchEvalWarmup = boundedQueryInt(['hybridBatchEvalWarmup', 'batchEvalWarmup'], 0, 0, 20);
  const batchEvalIterations = boundedQueryInt(['hybridBatchEvalIters', 'batchEvalIters'], 0, 0, 100);
  const searchWarmup = boundedQueryInt(['hybridSearchWarmup', 'searchWarmup'], 1, 0, 10);
  const searchIterations = boundedQueryInt(['hybridSearchIters', 'searchIters'], 3, 1, 50);
  const reuseTree = queryBool(['reuseTree', 'searchReuseTree', 'treeReuse'], false);
  const resetBetweenSearches = queryBool(['resetBetweenSearches', 'resetSearchTree'], !reuseTree);
  const evalTimes: number[] = [];
  const evalTimingSamples: Record<string, number[]> = {};
  const batchEvalTimes: number[] = [];
  const batchEvalTimingSamples: Record<string, number[]> = {};
  let lastBatchEval: BrowserEvaluationChoice[] | undefined;
  let lastBatchEvalTiming: Record<string, number> | undefined;
  const searchTimes: number[] = [];
  const searchStatsSamples: SearchStatsSnapshot[] = [];
  let lastEval: BrowserEvaluationChoice | undefined;
  let lastSearch: RenderableSearchResult | undefined;
  setBusy(true, `Benchmarking hybrid LC0 eval/search: ${evalIterations} evals + ${searchIterations} searches at ${searchVisits} visits…`);
  el('benchResult').textContent = 'HYBRID_SEARCH_BENCH_RUNNING';
  try {
    for (let i = 0; i < evalWarmup; i++) {
      lastEval = await evaluateWithWorker(input);
      el('benchResult').textContent = `HYBRID_SEARCH_BENCH_EVAL_WARMUP ${i + 1}/${evalWarmup}`;
      await new Promise<void>((resolve) => setTimeout(resolve, 0));
    }
    for (let i = 0; i < evalIterations; i++) {
      const started = performance.now();
      lastEval = await evaluateWithWorker(input);
      evalTimes.push(performance.now() - started);
      recordNumericTimingSamples(evalTimingSamples, (lastEval.evaluation as { timing?: unknown }).timing);
      el('benchResult').textContent = `HYBRID_SEARCH_BENCH_EVAL_TIMED ${i + 1}/${evalIterations}`;
      await new Promise<void>((resolve) => setTimeout(resolve, 0));
    }
    const batchInputs = Array.from({ length: searchBatchSize }, () => input);
    for (let i = 0; i < batchEvalWarmup; i++) {
      lastBatchEval = await evaluateBatchWithWorker(batchInputs);
      el('benchResult').textContent = `HYBRID_SEARCH_BENCH_BATCH_EVAL_WARMUP ${i + 1}/${batchEvalWarmup}`;
      await new Promise<void>((resolve) => setTimeout(resolve, 0));
    }
    for (let i = 0; i < batchEvalIterations; i++) {
      const started = performance.now();
      lastBatchEval = await evaluateBatchWithWorker(batchInputs);
      batchEvalTimes.push(performance.now() - started);
      lastBatchEvalTiming = aggregateBatchEvaluationTiming(lastBatchEval);
      recordNumericTimingSamples(batchEvalTimingSamples, lastBatchEvalTiming);
      el('benchResult').textContent = `HYBRID_SEARCH_BENCH_BATCH_EVAL_TIMED ${i + 1}/${batchEvalIterations}`;
      await new Promise<void>((resolve) => setTimeout(resolve, 0));
    }
    await resetSearchTreeState();
    for (let i = 0; i < searchWarmup; i++) {
      if (resetBetweenSearches) await resetSearchTreeState();
      const response = await postWorkerRequest<{ type: 'searchResult'; result: RenderableSearchResult }>({
        type: 'search', input, visits: searchVisits, batchSize: searchBatchSize, batchPipelineDepth: searchBatchPipelineDepth, multiPv: searchMultiPv, reuseTree,
      });
      lastSearch = response.result;
      el('benchResult').textContent = `HYBRID_SEARCH_BENCH_SEARCH_WARMUP ${i + 1}/${searchWarmup}`;
      await new Promise<void>((resolve) => setTimeout(resolve, 0));
    }
    for (let i = 0; i < searchIterations; i++) {
      if (resetBetweenSearches) await resetSearchTreeState();
      const started = performance.now();
      const response = await postWorkerRequest<{ type: 'searchResult'; result: RenderableSearchResult }>({
        type: 'search', input, visits: searchVisits, batchSize: searchBatchSize, batchPipelineDepth: searchBatchPipelineDepth, multiPv: searchMultiPv, reuseTree,
      });
      lastSearch = response.result;
      searchTimes.push(performance.now() - started);
      if (lastSearch.stats) searchStatsSamples.push(lastSearch.stats);
      el('benchResult').textContent = `HYBRID_SEARCH_BENCH_SEARCH_TIMED ${i + 1}/${searchIterations}`;
      await new Promise<void>((resolve) => setTimeout(resolve, 0));
    }
    const totalSearchMs = searchTimes.reduce((sum, value) => sum + value, 0);
    const result = {
      status: 'HYBRID_SEARCH_BENCH_DONE',
      backend: searchWorkerBackend,
      packUrl: PACK_URL,
      model: MODEL_URL,
      layers: boundedQueryInt(['encoderLayers', 'layers'], 10, 1, 32),
      browserInfo: browserReportInfo(),
      workerInitMs: roundReportMs(searchWorkerInitMs),
      requestedEp: requestedWorkerEp(),
      packVerification: params.get('packVerify') === '0' ? 'disabled' : 'enabled',
      visits: searchVisits,
      batchSize: searchBatchSize,
      batchPipelineDepth: searchBatchPipelineDepth,
      wgslBatchMode: HYBRID_WGSL_HEADS_REQUESTED ? HYBRID_WGSL_BATCH_MODE : undefined,
      inputBackend: HYBRID_INPUT_BACKEND,
      legalPriorsBackend: HYBRID_LEGAL_PRIORS_BACKEND,
      encoderKernelVariant: HYBRID_ENCODER_KERNEL_VARIANT,
      multiPv: searchMultiPv,
      reuseTree,
      resetBetweenSearches,
      evalCacheEntries: HYBRID_EVAL_CACHE_ENTRIES,
      eval: {
        warmup: evalWarmup,
        iterations: evalIterations,
        timingStats: sampleTimingStats(evalTimes, 'hybrid warm eval round trips'),
        phaseTimingStats: summarizeNumericTimingSamples(evalTimingSamples, 'hybrid warm eval backend timing'),
        lastBackendTiming: roundedNumericRecord((lastEval?.evaluation as { timing?: unknown } | undefined)?.timing),
        timesMs: evalTimes.map((time) => roundReportMs(time)),
        bestMove: lastEval?.move,
        q: lastEval?.evaluation.q === undefined ? undefined : Number(lastEval.evaluation.q.toFixed(8)),
        mlh: lastEval?.evaluation.mlh === undefined ? undefined : Number(lastEval.evaluation.mlh.toFixed(3)),
      },
      batchEval: {
        warmup: batchEvalWarmup,
        iterations: batchEvalIterations,
        batchSize: searchBatchSize,
        timingStats: sampleTimingStats(batchEvalTimes, 'hybrid physical batch eval round trips'),
        phaseTimingStats: summarizeNumericTimingSamples(batchEvalTimingSamples, 'hybrid physical batch eval backend timing'),
        lastBackendTiming: roundedNumericRecord(lastBatchEvalTiming),
        timesMs: batchEvalTimes.map((time) => roundReportMs(time)),
        bestMoves: lastBatchEval?.map((entry) => entry.move),
        allBestMovesMatch: lastBatchEval ? lastBatchEval.every((entry) => entry.move === lastBatchEval?.[0]?.move) : undefined,
      },
      search: {
        warmup: searchWarmup,
        iterations: searchIterations,
        timingStats: sampleTimingStats(searchTimes, 'hybrid fixed-visit search round trips'),
        timesMs: searchTimes.map((time) => roundReportMs(time)),
        visitsPerSecond: Number(((searchVisits * searchIterations) / Math.max(1e-9, totalSearchMs / 1000)).toFixed(3)),
        completedVisitsPerSecond: Number((aggregateSearchStats(searchStatsSamples).completedVisits / Math.max(1e-9, totalSearchMs / 1000)).toFixed(3)),
        aggregateStats: aggregateSearchStats(searchStatsSamples),
        statsSamples: searchStatsSamples,
        bestMove: lastSearch?.move,
        value: lastSearch?.value === undefined ? undefined : Number(lastSearch.value.toFixed(8)),
        pv: lastSearch?.pv,
        stats: lastSearch?.stats,
      },
    };
    el('benchResult').textContent = JSON.stringify(result);
    const evalMean = (result.eval.timingStats?.meanMs as number | undefined) ?? 0;
    const searchMean = (result.search.timingStats?.meanMs as number | undefined) ?? 0;
    el('message').textContent = `HYBRID_SEARCH_BENCH_DONE eval ${evalMean.toFixed(1)} ms · search ${searchMean.toFixed(1)} ms · ${result.search.visitsPerSecond.toFixed(1)} visits/s · ${searchWorkerBackend}`;
  } catch (error) {
    el('benchResult').textContent = `HYBRID_SEARCH_BENCH_FAILED ${(error as Error).message}`;
    el('message').textContent = `Hybrid search benchmark failed: ${(error as Error).message}`;
    throw error;
  } finally {
    setBusy(false);
  }
}

type HybridInputBenchFixture = { id: string; kind: 'fen' | 'history'; input: Lc0EvaluatorInput };

async function loadRepresentativeInputFixtures(): Promise<HybridInputBenchFixture[]> {
  const [fenFixtures, historyFixtures] = await Promise.all([
    fetch('/fixtures/lc0/fen_only.json', { cache: 'no-store' }).then((response) => {
      if (!response.ok) throw new Error(`failed to load FEN fixtures: HTTP ${response.status}`);
      return response.json() as Promise<Array<{ id: string; fen: string }>>;
    }),
    fetch('/fixtures/lc0/history.json', { cache: 'no-store' }).then((response) => {
      if (!response.ok) throw new Error(`failed to load history fixtures: HTTP ${response.status}`);
      return response.json() as Promise<Array<{ id: string; startFen?: string; moves: string[] }>>;
    }),
  ]);
  return [
    ...fenFixtures.map((fixture) => ({ id: fixture.id, kind: 'fen' as const, input: fixture.fen })),
    ...historyFixtures.map((fixture) => ({ id: fixture.id, kind: 'history' as const, input: { positions: buildBoardHistoryFromMoves(fixture.moves, fixture.startFen ?? START_FEN) } })),
  ];
}

async function runHybridDeferredReadbackBenchmark(): Promise<void> {
  const fixtures = await loadRepresentativeInputFixtures();
  const iterations = boundedQueryInt(['deferredReadbackIters', 'iters'], 4, 1, 50);
  const warmup = boundedQueryInt(['deferredReadbackWarmup', 'warmup'], 1, 0, 10);
  const batchSize = boundedQueryInt(['deferredReadbackBatch', 'batch', 'batchSize'], 4, 1, 32);
  const fixtureLimit = boundedQueryInt(['fixtureLimit', 'fixtures'], Math.min(4, fixtures.length), 1, fixtures.length);
  const inputs = fixtures.slice(0, fixtureLimit).map((fixture) => fixture.input);
  setBusy(true, `Benchmarking deferred WGSL-head readback over ${fixtureLimit} fixtures…`);
  el('benchResult').textContent = 'WGSL_DEFERRED_READBACK_BENCH_RUNNING';
  try {
    const response = await postWorkerRequest<{ type: 'wgslDeferredReadbackBenchmarkResult'; result: WgslDeferredReadbackBenchmarkResult }>({
      type: 'wgslDeferredReadbackBenchmark',
      packUrl: PACK_URL,
      inputs,
      layers: boundedQueryInt(['encoderLayers', 'layers'], 10, 1, 32),
      verifyShards: params.get('packVerify') !== '0',
      inputBackend: HYBRID_INPUT_BACKEND,
      legalPriorsBackend: HYBRID_LEGAL_PRIORS_BACKEND,
      batchSize,
      iterations,
      warmup,
    });
    const result = response.result;
    el('benchResult').textContent = JSON.stringify(result);
    el('message').textContent = `WGSL_DEFERRED_READBACK_BENCH_DONE immediate ${result.immediate.evalsPerSecond.toFixed(1)} eval/s · deferred ${result.deferred.evalsPerSecond.toFixed(1)} eval/s · best moves match ${result.allBestMovesMatch ? 'yes' : 'no'}`;
  } catch (error) {
    el('benchResult').textContent = `WGSL_DEFERRED_READBACK_BENCH_FAILED ${(error as Error).message}`;
    el('message').textContent = `Deferred readback benchmark failed: ${(error as Error).message}`;
    throw error;
  } finally {
    setBusy(false);
  }
}

async function browserMemorySample(): Promise<BrowserMemorySample> {
  const perf = performance as Performance & {
    memory?: { usedJSHeapSize?: number; totalJSHeapSize?: number; jsHeapSizeLimit?: number };
    measureUserAgentSpecificMemory?: () => Promise<{ bytes?: number }>;
  };
  const sample: BrowserMemorySample = {};
  if (perf.memory) {
    sample.usedJSHeapSize = perf.memory.usedJSHeapSize;
    sample.totalJSHeapSize = perf.memory.totalJSHeapSize;
    sample.jsHeapSizeLimit = perf.memory.jsHeapSizeLimit;
  }
  if (perf.measureUserAgentSpecificMemory) {
    try {
      const measured = await perf.measureUserAgentSpecificMemory();
      if (typeof measured.bytes === 'number') sample.userAgentSpecificBytes = measured.bytes;
    } catch (error) {
      sample.unavailableReason = (error as Error).message;
    }
  }
  if (!perf.memory && !perf.measureUserAgentSpecificMemory) sample.unavailableReason = 'browser memory APIs unavailable';
  return sample;
}

async function runHybridDeferredReadbackLifecycleSmoke(): Promise<void> {
  const fixtures = await loadRepresentativeInputFixtures();
  const cycles = boundedQueryInt(['lifecycleCycles', 'cycles'], 3, 1, 20);
  const iterations = boundedQueryInt(['deferredReadbackIters', 'iters'], 4, 1, 50);
  const warmup = boundedQueryInt(['deferredReadbackWarmup', 'warmup'], 1, 0, 10);
  const batchSize = boundedQueryInt(['deferredReadbackBatch', 'batch', 'batchSize'], 4, 1, 32);
  const fixtureLimit = boundedQueryInt(['fixtureLimit', 'fixtures'], Math.min(4, fixtures.length), 1, fixtures.length);
  const pauseMs = boundedQueryInt(['lifecyclePauseMs', 'pauseMs'], 0, 0, 5000);
  const inputs = fixtures.slice(0, fixtureLimit).map((fixture) => fixture.input);
  const cycleResults = [];
  const memorySamples: Array<{ cycle: number; phase: 'before' | 'after'; sample: BrowserMemorySample }> = [];
  setBusy(true, `Running WGSL deferred-readback lifecycle smoke: ${cycles} cycle(s), ${fixtureLimit} fixture(s)…`);
  el('benchResult').textContent = 'WGSL_DEFERRED_READBACK_LIFECYCLE_RUNNING';
  try {
    for (let cycle = 1; cycle <= cycles; cycle++) {
      memorySamples.push({ cycle, phase: 'before', sample: await browserMemorySample() });
      const response = await postWorkerRequest<{ type: 'wgslDeferredReadbackBenchmarkResult'; result: WgslDeferredReadbackBenchmarkResult }>({
        type: 'wgslDeferredReadbackBenchmark',
        packUrl: PACK_URL,
        inputs,
        layers: boundedQueryInt(['encoderLayers', 'layers'], 10, 1, 32),
        verifyShards: params.get('packVerify') !== '0',
        inputBackend: HYBRID_INPUT_BACKEND,
        legalPriorsBackend: HYBRID_LEGAL_PRIORS_BACKEND,
        batchSize,
        iterations,
        warmup,
      });
      const result = response.result;
      memorySamples.push({ cycle, phase: 'after', sample: await browserMemorySample() });
      cycleResults.push({
        cycle,
        allBestMovesMatch: result.allBestMovesMatch,
        immediate: result.immediate,
        deferred: result.deferred,
      });
      el('benchResult').textContent = `WGSL_DEFERRED_READBACK_LIFECYCLE ${cycle}/${cycles}`;
      if (pauseMs > 0) await new Promise<void>((resolve) => setTimeout(resolve, pauseMs));
    }
    const allCyclesBestMovesMatch = cycleResults.every((entry) => entry.allBestMovesMatch);
    const result = {
      status: 'WGSL_DEFERRED_READBACK_LIFECYCLE_DONE',
      backend: searchWorkerBackend,
      stableBackend: 'lc0web-wgsl-encoder-ort-heads',
      packUrl: PACK_URL,
      layers: boundedQueryInt(['encoderLayers', 'layers'], 10, 1, 32),
      inputBackend: HYBRID_INPUT_BACKEND,
      legalPriorsBackend: HYBRID_LEGAL_PRIORS_BACKEND,
      cycles,
      batchSize,
      iterations,
      warmup,
      inputCount: inputs.length,
      allCyclesBestMovesMatch,
      failedCycles: cycleResults.filter((entry) => !entry.allBestMovesMatch).map((entry) => entry.cycle),
      memorySamples,
      cycleResults,
    };
    el('benchResult').textContent = JSON.stringify(result);
    el('message').textContent = `WGSL_DEFERRED_READBACK_LIFECYCLE_DONE cycles ${cycles} · best moves match ${allCyclesBestMovesMatch ? 'yes' : 'no'}`;
  } catch (error) {
    el('benchResult').textContent = `WGSL_DEFERRED_READBACK_LIFECYCLE_FAILED ${(error as Error).message}`;
    el('message').textContent = `Deferred readback lifecycle smoke failed: ${(error as Error).message}`;
    throw error;
  } finally {
    setBusy(false);
  }
}

function parseJsonlRecords<T>(text: string): T[] {
  return text.trim().split('\n').filter(Boolean).map((line) => JSON.parse(line) as T);
}

async function loadNativeSearchRecords(visits: number, limit: number): Promise<NativeRecord[]> {
  const [fenResponse, historyResponse] = await Promise.all([
    fetch(`/lc0/native_search_fen_only_blas_nodes${visits}.jsonl`, { cache: 'no-store' }),
    fetch(`/lc0/native_search_history_blas_nodes${visits}.jsonl`, { cache: 'no-store' }),
  ]);
  if (!fenResponse.ok) throw new Error(`failed to load native FEN search fixtures for nodes${visits}: HTTP ${fenResponse.status}`);
  if (!historyResponse.ok) throw new Error(`failed to load native history search fixtures for nodes${visits}: HTTP ${historyResponse.status}`);
  const records = [
    ...parseJsonlRecords<NativeRecord>(await fenResponse.text()),
    ...parseJsonlRecords<NativeRecord>(await historyResponse.text()),
  ];
  return records.slice(0, Math.min(limit, records.length));
}

function nativeSearchInput(record: NativeRecord): Lc0EvaluatorInput {
  if (record.moves) return { positions: buildBoardHistoryFromMoves(record.moves, record.startFen ?? START_FEN) };
  return record.fen;
}

function rootVisitShare(children: Lc0SearchChild[]): Map<string, number> {
  const total = children.reduce((sum, child) => sum + Math.max(0, child.visits), 0);
  const denom = total > 0 ? total : 1;
  return new Map(children.map((child) => [child.uci, Math.max(0, child.visits) / denom]));
}

function rootVisitDistributionL1(a: Lc0SearchChild[] | undefined, b: Lc0SearchChild[] | undefined): number | undefined {
  if (!a || !b) return undefined;
  const aa = rootVisitShare(a);
  const bb = rootVisitShare(b);
  const ucis = new Set([...aa.keys(), ...bb.keys()]);
  let l1 = 0;
  for (const uci of ucis) l1 += Math.abs((aa.get(uci) ?? 0) - (bb.get(uci) ?? 0));
  return Number(l1.toFixed(6));
}

function rootTopVisitShare(children: Lc0SearchChild[]): number | undefined {
  const total = children.reduce((sum, child) => sum + Math.max(0, child.visits), 0);
  if (total <= 0) return undefined;
  const top = children.reduce((max, child) => Math.max(max, child.visits), 0);
  return Number((top / total).toFixed(6));
}

function rootChildTrace(children: Lc0SearchChild[] | undefined): Array<{ uci: string; visits: number; prior: number; q: number; probability: number }> | undefined {
  if (!children) return undefined;
  return children.map((child) => ({
    uci: child.uci,
    visits: child.visits,
    prior: roundReportMs(child.prior) ?? child.prior,
    q: roundReportMs(child.q) ?? child.q,
    probability: roundReportMs(child.probability) ?? child.probability,
  }));
}

function pipelineSearchSemantics(depth: number): 'serial-parity' | 'speculative-pipelined' {
  return depth > 1 ? 'speculative-pipelined' : 'serial-parity';
}

async function runHybridSearchFixtureParity(): Promise<void> {
  if (!searchWorkerReady) throw new Error('hybrid search fixture parity requires ready LC0 worker');
  const visitsList = queryIntList(['searchFixtureVisits', 'fixtureVisits', 'visitsList', 'visits'], [32], 1, 100000);
  const requestedDepths = queryIntList(['batchPipelineDepths', 'pipelineDepths'], [1], 1, 16);
  const depths = [1, ...requestedDepths.filter((depth) => depth !== 1)];
  const repeats = boundedQueryInt(['searchFixtureRepeats', 'fixtureRepeats', 'repeats'], 1, 1, 10);
  const fixtureLimit = boundedQueryInt(['fixtureLimit', 'fixtures'], 16, 1, 16);
  const fixtureIds = (params.get('fixtureIds') ?? '').split(',').map((value) => value.trim()).filter(Boolean);
  const fixtureIdSet = fixtureIds.length ? new Set(fixtureIds) : undefined;
  const traceRootChildren = params.get('traceRootChildren') === '1';
  const traceSearchVisits = params.get('traceSearchVisits') === '1';
  const batchSize = searchBatchSize;
  const cells = [];
  const depthBaselines = new Map<string, { bestMove: string | undefined; children: Lc0SearchChild[] }>();
  setBusy(true, `Running hybrid search fixture parity: visits ${visitsList.join(',')} · depths ${depths.join(',')} · ${fixtureLimit} fixtures…`);
  el('benchResult').textContent = 'HYBRID_SEARCH_FIXTURE_PARITY_RUNNING';
  try {
    for (const visits of visitsList) {
      const loadedRecords = await loadNativeSearchRecords(visits, 16);
      const filteredRecords = fixtureIdSet ? loadedRecords.filter((record) => fixtureIdSet.has(record.id)) : loadedRecords;
      if (fixtureIdSet && filteredRecords.length !== fixtureIdSet.size) {
        const loadedIds = new Set(loadedRecords.map((record) => record.id));
        const missing = fixtureIds.filter((id) => !loadedIds.has(id));
        throw new Error(`requested fixture ID(s) not found for visits ${visits}: ${missing.join(', ')}`);
      }
      const records = filteredRecords.slice(0, fixtureLimit);
      for (const record of records) {
        const input = nativeSearchInput(record);
        const expectedNativeBestMove = nativeCastlingToStandard(record.bestmove);
        for (const depth of depths) {
          for (let repeat = 1; repeat <= repeats; repeat++) {
            await resetSearchTreeState();
            const started = performance.now();
            const response = await postWorkerRequest<{ type: 'searchResult'; result: RenderableSearchResult }>({
              type: 'search',
              input,
              visits,
              batchSize,
              batchPipelineDepth: depth,
              traceSearchVisits,
              multiPv: 1,
              reuseTree: false,
            });
            const result = response.result;
            const baselineKey = `${visits}\t${record.id}\t${repeat}`;
            if (depth === 1) depthBaselines.set(baselineKey, { bestMove: result.move, children: result.children });
            const depthBaseline = depthBaselines.get(baselineKey);
            const depthBaselineBestMove = depthBaseline?.bestMove;
            cells.push({
              visits,
              batchSize,
              batchPipelineDepth: depth,
              pipelineSearchSemantics: pipelineSearchSemantics(depth),
              repeat,
              id: record.id,
              kind: record.moves ? 'history' : 'fen',
              expectedNativeBestMove,
              bestMove: result.move,
              matchesNative: result.move === expectedNativeBestMove,
              depthBaselineBestMove,
              matchesDepthBaseline: result.move === depthBaselineBestMove,
              depthBaselineVisitL1: rootVisitDistributionL1(result.children, depthBaseline?.children),
              topVisitShare: rootTopVisitShare(result.children),
              depthBaselineTopVisitShare: depthBaseline ? rootTopVisitShare(depthBaseline.children) : undefined,
              rootChildren: traceRootChildren ? rootChildTrace(result.children) : undefined,
              depthBaselineRootChildren: traceRootChildren ? rootChildTrace(depthBaseline?.children) : undefined,
              searchTrace: traceSearchVisits ? result.stats?.searchTrace : undefined,
              completedVisits: result.stats?.completedVisits,
              stopReason: result.stats?.stopReason,
              elapsedMs: roundReportMs(performance.now() - started),
              searchElapsedMs: roundReportMs(result.elapsedMs),
              evalCalls: result.stats?.evalCalls,
              batchEvalCalls: result.stats?.batchEvalCalls,
              maxEvalBatch: result.stats?.maxEvalBatch,
              evalBatchSizeHistogram: result.stats?.evalBatchSizeHistogram,
              batchPipelineFlushes: result.stats?.batchPipelineFlushes,
              maxBatchPipelineBatches: result.stats?.maxBatchPipelineBatches,
              totalEvalMs: result.stats?.evalBackendTimingMeans?.totalEvalMs,
              totalEvalMsPerPosition: result.stats?.evalBackendTimingPerPositionMeans?.totalEvalMs,
              legalPriorsMs: result.stats?.evalBackendTimingMeans?.legalPriorsMs,
              legalPriorsMsPerPosition: result.stats?.evalBackendTimingPerPositionMeans?.legalPriorsMs,
              legalPriorsBridgeCopyMs: result.stats?.evalBackendTimingMeans?.legalPriorsBridgeCopyMs,
              legalPriorsWasmRunMs: result.stats?.evalBackendTimingMeans?.legalPriorsWasmRunMs,
              legalPriorsWasmTotalMs: result.stats?.evalBackendTimingMeans?.legalPriorsWasmTotalMs,
              readbackSyncedMs: result.stats?.evalBackendTimingMeans?.readbackSyncedMs,
              readbackSyncedMsPerPosition: result.stats?.evalBackendTimingPerPositionMeans?.readbackSyncedMs,
            });
            el('benchResult').textContent = `HYBRID_SEARCH_FIXTURE_PARITY ${cells.length}/${visitsList.length * records.length * depths.length * repeats}`;
            await new Promise<void>((resolve) => setTimeout(resolve, 0));
          }
        }
      }
    }
    const mismatches = cells.filter((cell) => !cell.matchesNative || !cell.matchesDepthBaseline);
    const result = {
      status: 'HYBRID_SEARCH_FIXTURE_PARITY_DONE',
      backend: searchWorkerBackend,
      headBackend: HYBRID_WGSL_HEADS_REQUESTED ? 'wgsl' : 'ort',
      inputBackend: HYBRID_INPUT_BACKEND,
      legalPriorsBackend: HYBRID_LEGAL_PRIORS_BACKEND,
      encoderKernelVariant: HYBRID_ENCODER_KERNEL_VARIANT,
      visitsList,
      batchSize,
      batchPipelineDepths: depths,
      pipelineSearchSemanticsByDepth: Object.fromEntries(depths.map((depth) => [String(depth), pipelineSearchSemantics(depth)])),
      repeats,
      fixtureLimit,
      fixtureIds,
      traceRootChildren,
      traceSearchVisits,
      cells: cells.length,
      nativeMatches: cells.filter((cell) => cell.matchesNative).length,
      depthBaselineMatches: cells.filter((cell) => cell.matchesDepthBaseline).length,
      maxDepthBaselineVisitL1: Math.max(0, ...cells.map((cell) => typeof cell.depthBaselineVisitL1 === 'number' ? cell.depthBaselineVisitL1 : 0)),
      mismatches,
      results: cells,
    };
    el('benchResult').textContent = JSON.stringify(result);
    el('message').textContent = `HYBRID_SEARCH_FIXTURE_PARITY_DONE native ${result.nativeMatches}/${result.cells} · depth baseline ${result.depthBaselineMatches}/${result.cells}`;
  } catch (error) {
    el('benchResult').textContent = `HYBRID_SEARCH_FIXTURE_PARITY_FAILED ${(error as Error).message}`;
    el('message').textContent = `Hybrid search fixture parity failed: ${(error as Error).message}`;
    throw error;
  } finally {
    setBusy(false);
  }
}

async function runHybridInputBenchmark(): Promise<void> {
  const fixtures = await loadRepresentativeInputFixtures();
  const iterations = boundedQueryInt(['hybridInputBenchIters', 'inputBenchIters', 'iters'], 1, 1, 20);
  const warmup = boundedQueryInt(['hybridInputBenchWarmup', 'inputBenchWarmup', 'warmup'], 1, 0, 10);
  const backendParam = params.get('inputBenchBackends') ?? params.get('hybridInputBackends') ?? 'js,wasm';
  const requestedBackends = backendParam.split(',').map((entry) => entry.trim()).filter(Boolean).map((entry) => {
    if (entry !== 'js' && entry !== 'wgsl' && entry !== 'wasm') throw new Error(`invalid hybrid input benchmark backend: ${entry}`);
    return entry;
  }) as Array<'js' | 'wgsl' | 'wasm'>;
  const backends: Array<'js' | 'wgsl' | 'wasm'> = requestedBackends.includes('js')
    ? ['js', ...requestedBackends.filter((backend) => backend !== 'js')]
    : requestedBackends;
  if (fixtures.length !== 16) throw new Error(`expected 16 representative fixtures, loaded ${fixtures.length}`);
  setBusy(true, `Benchmarking LC0 hybrid input backends over ${fixtures.length} fixtures…`);
  el('benchResult').textContent = 'HYBRID_INPUT_BENCH_RUNNING';
  try {
    const byBackend: Record<string, unknown> = {};
    const baselineBestMoves = new Map<string, string | undefined>();
    for (const backend of backends) {
      await initHybridWorkerWithInputBackend(backend);
      const roundTripSamples: number[] = [];
      const backendTimingSamples: Record<string, number[]> = {};
      const fixtureResults = [];
      for (let i = 0; i < warmup; i++) {
        await evaluateWithWorker(fixtures[i % fixtures.length].input);
        el('benchResult').textContent = `HYBRID_INPUT_BENCH_${backend.toUpperCase()}_WARMUP ${i + 1}/${warmup}`;
        await new Promise<void>((resolve) => setTimeout(resolve, 0));
      }
      for (let fixtureIndex = 0; fixtureIndex < fixtures.length; fixtureIndex++) {
        const fixture = fixtures[fixtureIndex];
        const fixtureTimes: number[] = [];
        let last: BrowserEvaluationChoice | undefined;
        for (let iter = 0; iter < iterations; iter++) {
          const started = performance.now();
          last = await evaluateWithWorker(fixture.input);
          const elapsed = performance.now() - started;
          fixtureTimes.push(elapsed);
          roundTripSamples.push(elapsed);
          recordNumericTimingSamples(backendTimingSamples, (last.evaluation as { timing?: unknown }).timing);
          el('benchResult').textContent = `HYBRID_INPUT_BENCH_${backend.toUpperCase()} ${fixtureIndex + 1}/${fixtures.length} iter ${iter + 1}/${iterations}`;
          await new Promise<void>((resolve) => setTimeout(resolve, 0));
        }
        if (backend === 'js') baselineBestMoves.set(fixture.id, last?.move);
        fixtureResults.push({
          id: fixture.id,
          kind: fixture.kind,
          timingStats: sampleTimingStats(fixtureTimes, `${backend} ${fixture.id} input eval round trips`),
          lastBackendTiming: roundedNumericRecord((last?.evaluation as { timing?: unknown } | undefined)?.timing),
          bestMove: last?.move,
          bestMoveMatchesJs: backend === 'js' ? true : (baselineBestMoves.has(fixture.id) ? last?.move === baselineBestMoves.get(fixture.id) : undefined),
        });
      }
      byBackend[backend] = {
        workerBackend: searchWorkerBackend,
        workerInitMs: roundReportMs(searchWorkerInitMs),
        modelCache: workerModelCacheStatus,
        timingStats: sampleTimingStats(roundTripSamples, `${backend} input eval round trips over representative fixtures`),
        phaseTimingStats: summarizeNumericTimingSamples(backendTimingSamples, `${backend} backend input timing over representative fixtures`),
        fixtures: fixtureResults,
      };
    }
    const result = {
      status: 'HYBRID_INPUT_BENCH_DONE',
      packUrl: PACK_URL,
      layers: boundedQueryInt(['encoderLayers', 'layers'], 10, 1, 32),
      headBackend: HYBRID_WGSL_HEADS_REQUESTED ? 'wgsl' : 'ort',
      legalPriorsBackend: HYBRID_LEGAL_PRIORS_BACKEND,
      backends,
      fixtureCount: fixtures.length,
      iterations,
      warmup,
      browserInfo: browserReportInfo(),
      packVerification: params.get('packVerify') === '0' ? 'disabled' : 'enabled',
      byBackend,
    };
    el('benchResult').textContent = JSON.stringify(result);
    el('message').textContent = `HYBRID_INPUT_BENCH_DONE ${fixtures.length} fixtures · ${backends.join(' vs ')}`;
  } catch (error) {
    el('benchResult').textContent = `HYBRID_INPUT_BENCH_FAILED ${(error as Error).message}`;
    el('message').textContent = `Hybrid input benchmark failed: ${(error as Error).message}`;
    throw error;
  } finally {
    setBusy(false);
  }
}

async function searchWithWorker(): Promise<RenderableSearchResult> {
  const response = await postWorkerRequest<{ type: 'searchResult'; result: RenderableSearchResult }>({
    type: 'search',
    input: currentEvaluationInput(),
    ...currentSearchOptions(),
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
  const search = await searcher!.search(currentEvaluationInput(), currentSearchOptions({
    signal: mainSearchAbort!.signal,
    yieldEveryMs: 16,
  }));
  return { ...search, stats: search.search.stats, elapsedMs: performance.now() - started };
}

async function searchRootPosition() {
  if (!searchAvailable() || busy) return;
  beginSearch();
  setBusy(true, `LC0 PUCT search running (${currentSearchLimitLabel()}, ${searchModeLabel()})… press Stop to cancel.`);
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
      el('message').textContent = `Search selected ${result.move ?? '—'} (${result.visits} visits, batch ${searchBatchSize}, PUCT via ${searchModeLabel()}).`;
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
    ? `LC0 engine replying with ${currentSearchLimitLabel()} search (${searchModeLabel()})… press Stop to cancel.`
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

async function runHybridDriftFixtures() {
  if (!searchWorkerReady) throw new Error('hybrid drift requires initialized LC0 worker');
  setBusy(true, 'Running hybrid WGSL encoder + ORT heads fixture evaluations in browser…');
  el('benchResult').textContent = 'HYBRID_DRIFT_RUNNING';
  try {
    const limit = Math.min(100, Math.max(1, Math.floor(Number(params.get('hybridDriftLimit') ?? params.get('fixtureLimit') ?? '9') || 9)));
    const records = [
      ...await fetchNativeRecords('/lc0/native_fen_only_blas.jsonl'),
      ...await fetchNativeRecords('/lc0/native_history_blas.jsonl'),
    ].slice(0, limit);
    const started = performance.now();
    const evaluations = [];
    for (const native of records) {
      const input = native.moves ? { positions: buildBoardHistoryFromMoves(native.moves, native.startFen) } : native.fen;
      const choice = await choosePolicyMove(input);
      evaluations.push({
        id: native.id,
        fen: native.fen,
        startFen: native.startFen,
        moves: native.moves,
        bestMove: choice.evaluation.bestMove,
        wdl: choice.evaluation.wdl,
        q: choice.evaluation.q,
        topPriors: choice.evaluation.legalPriors.slice(0, 10).map(({ uci, index, prior }) => ({ uci, index, prior })),
      });
    }
    const elapsedMs = performance.now() - started;
    const result = {
      status: 'HYBRID_DRIFT_DONE',
      backend: searchWorkerBackend,
      packUrl: PACK_URL,
      layers: Math.min(32, Math.max(1, Math.floor(Number(params.get('encoderLayers') ?? params.get('layers') ?? '10') || 10))),
      encoderKernelVariant: HYBRID_ENCODER_KERNEL_VARIANT,
      fixtures: evaluations.length,
      elapsedMs: Number(elapsedMs.toFixed(3)),
      evaluations,
    };
    el('benchResult').textContent = JSON.stringify(result);
    el('message').textContent = `HYBRID_DRIFT_DONE ${evaluations.length} fixture(s) · ${(elapsedMs / Math.max(1, evaluations.length)).toFixed(1)} ms/eval`;
  } catch (error) {
    el('benchResult').textContent = `HYBRID_DRIFT_FAILED ${(error as Error).message}`;
    el('message').textContent = `Hybrid drift failed: ${(error as Error).message}`;
    throw error;
  } finally {
    setBusy(false);
  }
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

async function resetBattleSearchTree(): Promise<void> {
  searcher?.resetTree();
  if (searchWorkerReady) {
    await postWorkerRequest<{ type: 'searchReset' }>({ type: 'resetSearch' });
  }
}

// LC0 search move, run in the worker when available so the board keeps
// animating; falls back to a cancellable main-thread search otherwise.
async function battleSearchMove(positions: BoardState[]): Promise<string | null> {
  if (searchWorkerReady) {
    const response = await postWorkerRequest<{ type: 'searchResult'; result: RenderableSearchResult }>(
      { type: 'search', input: { positions }, ...currentSearchOptions({ reuseTree: true }) },
      (id) => { activeWorkerSearchId = id; },
    );
    return response.result.cancelled ? null : (response.result.move ?? null);
  }
  const result = await searcher!.search({ positions }, currentSearchOptions({ signal: battleAbort!.signal, yieldEveryMs: 16, reuseTree: true }));
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
  await resetBattleSearchTree();
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
  const lc0Label = `LC0 search ${currentSearchLimitLabel()}`;
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

function disposeRuntimeResources(): void {
  mainSearchAbort?.abort();
  battleAbort?.abort();
  if (activeWorkerSearchId !== null) searchWorker?.postMessage({ type: 'cancel', target: activeWorkerSearchId });
  activeWorkerSearchId = null;
  searchWorker?.terminate();
  searchWorker = null;
  searchWorkerReady = false;
  for (const pending of workerPending.values()) pending.reject(new Error('LC0 worker disposed'));
  workerPending.clear();
  stockfish?.dispose();
  stockfish = null;
  void mainEvaluator?.dispose();
  mainEvaluator = null;
  player = null;
  searcher = null;
}

async function init() {
  window.addEventListener('pagehide', (event) => {
    if (!(event as PageTransitionEvent).persisted) disposeRuntimeResources();
  });
  el('message').textContent = SHADER_F16_PROBE_REQUESTED ? 'Preparing WebGPU shader-f16 probe…' : PACK_PROBE_REQUESTED ? 'Preparing dedicated worker for lc0web pack probe…' : WORKER_ONLY_MODEL ? 'Loading LC0 model in dedicated worker…' : 'Loading LC0 ONNX model…';
  renderStatic();
  try {
    if (SHADER_F16_PROBE_REQUESTED) {
      mainModelCacheStatus = 'shader-f16 feature probe (no model loaded)';
      workerModelCacheStatus = 'not used';
      el('backend').textContent = 'webgpu-shader-f16-probe';
      renderStatic();
      await runShaderF16Probe();
      return;
    }
    if (PACK_PROBE_REQUESTED) {
      mainModelCacheStatus = 'pack-probe worker-only (no ONNX session)';
      workerModelCacheStatus = 'pack shards worker-owned';
      useSearchWorker = true;
      await initSearchWorker({ initModel: false });
      searchWorkerBackend = MAPPED_POLICY_PROBE_REQUESTED ? 'lc0web-wgsl-mapped-policy-probe' : WGSL_HEADS_VS_ORT_FIXTURES_REQUESTED ? 'lc0web-wgsl-encoder-wgsl-heads-probe' : WGSL_HEADS_PROBE_REQUESTED ? 'lc0web-wgsl-heads-probe' : ENCODER_STACK_BENCH_REQUESTED ? 'lc0web-wgsl-encoder-stack-bench' : ENCODER0_BLOCK_ORT_BENCH_REQUESTED ? 'ort-tiny-encoder0-block-bench' : ENCODER0_BLOCK_BENCH_REQUESTED ? 'lc0web-wgsl-encoder0-block-bench' : ENCODER0_FFN_ORT_BENCH_REQUESTED ? 'ort-tiny-encoder0-ffn-bench' : ENCODER0_FFN_BENCH_REQUESTED ? 'lc0web-wgsl-encoder0-ffn-bench' : ATTENTION_OUTPUT_ORT_BENCH_REQUESTED ? 'ort-tiny-attention-output-bench' : ATTENTION_OUTPUT_BENCH_REQUESTED ? 'lc0web-wgsl-attention-output-bench' : ATTENTION_BLOCK_BENCH_REQUESTED ? 'lc0web-wgsl-attention-block-bench' : ATTENTION_VALUE_ORT_BENCH_REQUESTED ? 'ort-tiny-attention-value-bench' : ATTENTION_VALUE_BENCH_REQUESTED ? 'lc0web-wgsl-attention-value-bench' : SOFTMAX_BENCH_REQUESTED ? 'lc0web-wgsl-softmax-bench' : ATTENTION_SCORE_ORT_BENCH_REQUESTED ? 'ort-tiny-attention-score-bench' : ATTENTION_SCORE_BENCH_REQUESTED ? 'lc0web-wgsl-attention-score-bench' : QKV_BENCH_REQUESTED ? 'lc0web-wgsl-qkv-bench' : QKV_PROBE_REQUESTED ? 'lc0web-wgsl-qkv-probe' : ORT_OP_BENCH_REQUESTED ? 'ort-tiny-matmul-add-bench' : KERNEL_BENCH_REQUESTED ? 'lc0web-wgsl-kernel-bench' : KERNEL_PROBE_REQUESTED ? 'lc0web-wgsl-kernel' : 'lc0web-pack-loader';
      renderStatic();
      if (MAPPED_POLICY_PROBE_REQUESTED) await runMappedPolicyProbe();
      else if (WGSL_HEADS_VS_ORT_FIXTURES_REQUESTED) await runWgslHeadsVsOrtFixtures();
      else if (WGSL_HEADS_PROBE_REQUESTED) await runWgslHeadsProbe();
      else if (ENCODER_STACK_BENCH_REQUESTED) await runEncoderStackBenchmark();
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
      mainEvaluator = evaluator;
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
    if (HYBRID_DRIFT_REQUESTED) {
      await runHybridDriftFixtures();
      return;
    }
    if (HYBRID_DEFERRED_READBACK_LIFECYCLE_REQUESTED) {
      await runHybridDeferredReadbackLifecycleSmoke();
      return;
    }
    if (HYBRID_DEFERRED_READBACK_BENCH_REQUESTED) {
      await runHybridDeferredReadbackBenchmark();
      return;
    }
    if (HYBRID_ENCODER_PROFILE_REQUESTED) {
      await runHybridEncoderProfile();
      return;
    }
    if (HYBRID_SEARCH_FIXTURE_PARITY_REQUESTED) {
      await runHybridSearchFixtureParity();
      return;
    }
    if (HYBRID_SEARCH_BENCH_REQUESTED) {
      await runHybridSearchBenchmark();
      return;
    }
    if (HYBRID_INPUT_BENCH_REQUESTED) {
      await runHybridInputBenchmark();
      return;
    }
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
  selectEl('collisionSelect').value = searchBatchCollisionMode;
  inputEl('multiPvInput').value = String(searchMultiPv);
  selectEl('earlyStopSelect').value = searchEarlyStop;
  inputEl('movetimeInput').value = String(searchMovetimeMs);
  inputEl('cpuctInput').value = String(searchCpuct);
  selectEl('cpuctScheduleSelect').value = searchCpuctSchedule;
  selectEl('fpuStrategySelect').value = searchFpuStrategy;
  inputEl('fpuReductionInput').value = String(searchFpuReduction);
  inputEl('temperatureInput').value = String(searchTemperature);
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
selectEl('collisionSelect').addEventListener('change', () => {
  searchBatchCollisionMode = parseBatchCollisionMode(selectEl('collisionSelect').value);
  selectEl('collisionSelect').value = searchBatchCollisionMode;
  renderStatic();
});
inputEl('multiPvInput').addEventListener('change', () => {
  searchMultiPv = clampInt(inputEl('multiPvInput').value, 1, 20, searchMultiPv);
  inputEl('multiPvInput').value = String(searchMultiPv);
  renderStatic();
});
selectEl('earlyStopSelect').addEventListener('change', () => {
  searchEarlyStop = parseEarlyStop(selectEl('earlyStopSelect').value);
  selectEl('earlyStopSelect').value = searchEarlyStop;
  renderStatic();
});
inputEl('movetimeInput').addEventListener('change', () => {
  searchMovetimeMs = clampInt(inputEl('movetimeInput').value, 0, 600000, searchMovetimeMs);
  inputEl('movetimeInput').value = String(searchMovetimeMs);
  renderStatic();
});
inputEl('cpuctInput').addEventListener('change', () => {
  searchCpuct = clampFloat(inputEl('cpuctInput').value, 0, 100, searchCpuct);
  inputEl('cpuctInput').value = String(searchCpuct);
  renderStatic();
});
selectEl('cpuctScheduleSelect').addEventListener('change', () => {
  searchCpuctSchedule = parseCpuctSchedule(selectEl('cpuctScheduleSelect').value);
  selectEl('cpuctScheduleSelect').value = searchCpuctSchedule;
  renderStatic();
});
selectEl('fpuStrategySelect').addEventListener('change', () => {
  searchFpuStrategy = parseFpuStrategy(selectEl('fpuStrategySelect').value);
  selectEl('fpuStrategySelect').value = searchFpuStrategy;
  renderStatic();
});
inputEl('fpuReductionInput').addEventListener('change', () => {
  searchFpuReduction = clampFloat(inputEl('fpuReductionInput').value, 0, 5, searchFpuReduction);
  inputEl('fpuReductionInput').value = String(searchFpuReduction);
  renderStatic();
});
inputEl('temperatureInput').addEventListener('change', () => {
  searchTemperature = clampFloat(inputEl('temperatureInput').value, 0, 10, searchTemperature);
  inputEl('temperatureInput').value = String(searchTemperature);
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
  el('message').textContent = `Engine reply mode: ${engineReplyMode === 'search' ? 'PUCT search' : 'policy-only'}.`;
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
