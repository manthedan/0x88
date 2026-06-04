import { collectOrtRuntimeDiagnostics, setRequestedOrtExecutionProviderForCurrentThread, type OrtExecutionProviderPreference } from '../nn/ortRuntime.ts';
import { describeLc0ModelLoad, loadLc0ModelForOrt } from './modelCache.ts';
import { loadLc0WebModelPack } from './modelPack.ts';
import { CachedLc0Evaluator, Lc0OnnxEvaluator, type Lc0Evaluation, type Lc0EvaluationProvider, type Lc0EvaluatorInput } from './onnxEvaluator.ts';
import {
  runLc0WebAttentionBlockBenchmark,
  runLc0WebAttentionOutputBenchmark,
  runLc0WebAttentionOutputOrtBenchmark,
  runLc0WebAttentionScoreBenchmark,
  runLc0WebEncoder0BlockBenchmark,
  runLc0WebEncoder0BlockOrtBenchmark,
  runLc0WebEncoder0FfnBenchmark,
  runLc0WebEncoderStackBenchmark,
  runLc0WebEncoder0FfnOrtBenchmark,
  runLc0WebWgslHeadsProbe,
  runLc0WebMappedPolicyProbe,
  runLc0WebWgslHeadsVsOrtFixtures,
  Lc0WebHybridEvaluator,
  runLc0WebHybridEvaluation,
  runLc0WebHybridEncoderProfile,
  runLc0WebWgslDeferredReadbackBenchmark,
  runLc0WebAttentionScoreOrtBenchmark,
  runLc0WebAttentionValueBenchmark,
  runLc0WebAttentionValueOrtBenchmark,
  runLc0WebMatmulAddKernelBenchmark,
  runLc0WebMatmulAddKernelProbe,
  runLc0WebMatmulAddOrtBenchmark,
  runLc0WebQkvProjectionBenchmark,
  runLc0WebQkvProjectionProbe,
  runLc0WebSoftmaxBenchmark,
  type Lc0WebAttentionBlockBenchmarkResult,
  type Lc0WebAttentionQkvKernelVariant,
  type Lc0WebAttentionOutputBenchmarkResult,
  type Lc0WebAttentionOutProjKernelVariant,
  type Lc0WebAttentionOutputOrtBenchmarkResult,
  type Lc0WebAttentionScoreBenchmarkResult,
  type Lc0WebAttentionScoreOrtBenchmarkResult,
  type Lc0WebAttentionValueBenchmarkResult,
  type Lc0WebAttentionValueOrtBenchmarkResult,
  type Lc0WebEncoder0BlockBenchmarkResult,
  type Lc0WebEncoder0BlockOrtBenchmarkResult,
  type Lc0WebEncoder0FfnBenchmarkResult,
  type Lc0WebEncoderStackBenchmarkResult,
  type Lc0WebEncoderKernelVariant,
  type Lc0WebFfnKernelVariant,
  type Lc0WebEncoder0FfnOrtBenchmarkResult,
  type Lc0WebHybridEvaluationResult,
  type Lc0WebHybridEncoderProfileResult,
  type Lc0WebHybridEncoderProfileMode,
  type Lc0WebWgslDeferredReadbackBenchResult,
  type Lc0WebWgslHeadsProbeResult,
  type Lc0WebMappedPolicyProbeResult,
  type Lc0WebWgslHeadsVsOrtFixturesResult,
  type Lc0WebWgslHeadsVsOrtFixtureInput,
  type Lc0WebMatmulAddKernelBenchmarkResult,
  type Lc0WebMatmulAddKernelProbeResult,
  type Lc0WebMatmulAddOrtBenchmarkResult,
  type Lc0WebQkvProjectionBenchmarkResult,
  type Lc0WebQkvProjectionProbeResult,
  type Lc0WebSoftmaxBenchmarkResult,
} from './wgslMatmulAddProbe.ts';
import { Lc0PuctSearcher, type Lc0SearchOptions, type Lc0SearchResult } from './search.ts';
import type { CpuctSchedule, FpuStrategy, SearchBatchCollisionMode, SearchEarlyStop } from '../search/puct.ts';

type InitMessage = {
  type: 'init';
  id: number;
  modelUrl: string;
  ep: OrtExecutionProviderPreference;
  cacheModel: boolean;
  runtime?: 'onnx' | 'hybrid';
  packUrl?: string;
  layers?: number;
  verifyShards?: boolean;
  headBackend?: 'ort' | 'wgsl';
  wgslBatchMode?: 'physical' | 'serial';
  inputBackend?: 'js' | 'wgsl' | 'wasm';
  encoderKernelVariant?: Lc0WebEncoderKernelVariant;
  evalCacheEntries?: number;
};

type SearchMessage = {
  type: 'search';
  id: number;
  input: Lc0EvaluatorInput;
  visits?: number;
  movetimeMs?: number;
  batchSize?: number;
  batchCollisionMode?: SearchBatchCollisionMode;
  batchPipelineDepth?: number;
  multiPv?: number;
  reuseTree?: boolean;
  earlyStop?: SearchEarlyStop;
  cpuct?: number;
  cpuctSchedule?: CpuctSchedule;
  fpuStrategy?: FpuStrategy;
  fpuReduction?: number;
  temperature?: number;
};

type ResetSearchMessage = {
  type: 'resetSearch';
  id: number;
};

type EvaluateMessage = {
  type: 'evaluate';
  id: number;
  input: Lc0EvaluatorInput;
};

type EvaluateBatchMessage = {
  type: 'evaluateBatch';
  id: number;
  inputs: Lc0EvaluatorInput[];
};

type HybridEvaluateMessage = {
  type: 'hybridEvaluate';
  id: number;
  packUrl: string;
  input: Lc0EvaluatorInput;
  layers?: number;
  verifyShards?: boolean;
  headBackend?: 'ort' | 'wgsl';
  wgslBatchMode?: 'physical' | 'serial';
  inputBackend?: 'js' | 'wgsl' | 'wasm';
};

type HybridEncoderProfileMessage = {
  type: 'hybridEncoderProfile';
  id: number;
  packUrl: string;
  input: Lc0EvaluatorInput;
  layers?: number;
  iterations?: number;
  warmup?: number;
  verifyShards?: boolean;
  inputBackend?: 'js' | 'wgsl' | 'wasm';
  encoderKernelVariant?: Lc0WebEncoderKernelVariant;
  profileMode?: Lc0WebHybridEncoderProfileMode;
};

type WgslDeferredReadbackBenchmarkMessage = {
  type: 'wgslDeferredReadbackBenchmark';
  id: number;
  packUrl: string;
  inputs: Lc0EvaluatorInput[];
  layers?: number;
  verifyShards?: boolean;
  inputBackend?: 'js' | 'wgsl' | 'wasm';
  batchSize?: number;
  iterations?: number;
  warmup?: number;
};

type LoadPackMessage = {
  type: 'loadPack';
  id: number;
  packUrl: string;
  loadWeights?: boolean;
  verifyShards?: boolean;
  tensorNames?: string[];
};

type KernelProbeMessage = {
  type: 'kernelProbe';
  id: number;
  packUrl: string;
  weightTensorName?: string;
  biasTensorName?: string;
  iterations?: number;
  warmup?: number;
  verifyShards?: boolean;
  variant?: 'scalar' | 'tiled16' | 'scalar-transposed';
};

type KernelBenchmarkMessage = {
  type: 'kernelBenchmark';
  id: number;
  packUrl: string;
  weightTensorName?: string;
  biasTensorName?: string;
  iterations?: number;
  warmup?: number;
  verifyShards?: boolean;
  variant?: 'scalar' | 'tiled16' | 'scalar-transposed';
};

type QkvProbeMessage = {
  type: 'qkvProbe';
  id: number;
  packUrl: string;
  iterations?: number;
  warmup?: number;
  verifyShards?: boolean;
};

type QkvBenchmarkMessage = {
  type: 'qkvBenchmark';
  id: number;
  packUrl: string;
  iterations?: number;
  warmup?: number;
  verifyShards?: boolean;
};

type AttentionScoreBenchmarkMessage = {
  type: 'attentionScoreBenchmark';
  id: number;
  packUrl: string;
  iterations?: number;
  warmup?: number;
  verifyShards?: boolean;
};

type AttentionScoreOrtBenchmarkMessage = {
  type: 'attentionScoreOrtBenchmark';
  id: number;
  packUrl: string;
  ep: OrtExecutionProviderPreference;
  iterations?: number;
  warmup?: number;
  verifyShards?: boolean;
};

type SoftmaxBenchmarkMessage = {
  type: 'softmaxBenchmark';
  id: number;
  packUrl: string;
  iterations?: number;
  warmup?: number;
  verifyShards?: boolean;
};

type AttentionValueBenchmarkMessage = {
  type: 'attentionValueBenchmark';
  id: number;
  packUrl: string;
  iterations?: number;
  warmup?: number;
  verifyShards?: boolean;
};

type AttentionValueOrtBenchmarkMessage = {
  type: 'attentionValueOrtBenchmark';
  id: number;
  packUrl: string;
  ep: OrtExecutionProviderPreference;
  iterations?: number;
  warmup?: number;
  verifyShards?: boolean;
};

type AttentionBlockBenchmarkMessage = {
  type: 'attentionBlockBenchmark';
  id: number;
  packUrl: string;
  iterations?: number;
  warmup?: number;
  verifyShards?: boolean;
  fusedScoreSoftmax?: boolean;
  attentionQkvKernelVariant?: Lc0WebAttentionQkvKernelVariant;
};

type AttentionOutputBenchmarkMessage = {
  type: 'attentionOutputBenchmark';
  id: number;
  packUrl: string;
  iterations?: number;
  warmup?: number;
  verifyShards?: boolean;
  encoderPrefix?: string;
  attentionOutProjKernelVariant?: Lc0WebAttentionOutProjKernelVariant;
};

type AttentionOutputOrtBenchmarkMessage = {
  type: 'attentionOutputOrtBenchmark';
  id: number;
  packUrl: string;
  ep: OrtExecutionProviderPreference;
  iterations?: number;
  warmup?: number;
  verifyShards?: boolean;
  encoderPrefix?: string;
};

type Encoder0FfnBenchmarkMessage = {
  type: 'encoder0FfnBenchmark';
  id: number;
  packUrl: string;
  iterations?: number;
  warmup?: number;
  verifyShards?: boolean;
  encoderPrefix?: string;
  ffnKernelVariant?: Lc0WebFfnKernelVariant;
};

type Encoder0FfnOrtBenchmarkMessage = {
  type: 'encoder0FfnOrtBenchmark';
  id: number;
  packUrl: string;
  ep: OrtExecutionProviderPreference;
  iterations?: number;
  warmup?: number;
  verifyShards?: boolean;
  encoderPrefix?: string;
};

type Encoder0BlockBenchmarkMessage = {
  type: 'encoder0BlockBenchmark';
  id: number;
  packUrl: string;
  iterations?: number;
  warmup?: number;
  verifyShards?: boolean;
  encoderPrefix?: string;
};

type Encoder0BlockOrtBenchmarkMessage = {
  type: 'encoder0BlockOrtBenchmark';
  id: number;
  packUrl: string;
  ep: OrtExecutionProviderPreference;
  iterations?: number;
  warmup?: number;
  verifyShards?: boolean;
  encoderPrefix?: string;
};

type EncoderStackBenchmarkMessage = {
  type: 'encoderStackBenchmark';
  id: number;
  packUrl: string;
  ep: OrtExecutionProviderPreference;
  layers?: number;
  warmup?: number;
  verifyShards?: boolean;
  compareOrt?: boolean;
  compareHeads?: boolean;
};

type WgslHeadsProbeMessage = {
  type: 'wgslHeadsProbe';
  id: number;
  packUrl: string;
  ep: OrtExecutionProviderPreference;
  verifyShards?: boolean;
};

type WgslHeadsVsOrtFixturesMessage = {
  type: 'wgslHeadsVsOrtFixtures';
  id: number;
  packUrl: string;
  ep: OrtExecutionProviderPreference;
  fixtures: Lc0WebWgslHeadsVsOrtFixtureInput[];
  layers?: number;
  verifyShards?: boolean;
  mappedPolicyTolerance?: number;
  wdlTolerance?: number;
};

type MappedPolicyProbeMessage = {
  type: 'mappedPolicyProbe';
  id: number;
};

type OrtBenchmarkMessage = {
  type: 'ortBenchmark';
  id: number;
  packUrl: string;
  ep: OrtExecutionProviderPreference;
  weightTensorName?: string;
  biasTensorName?: string;
  iterations?: number;
  warmup?: number;
  verifyShards?: boolean;
};

type CancelMessage = {
  type: 'cancel';
  id: number;
  /** Optional target request id; when omitted, cancels any in-flight search. */
  target?: number;
};

type WorkerRequest = InitMessage | SearchMessage | ResetSearchMessage | EvaluateMessage | EvaluateBatchMessage | HybridEvaluateMessage | HybridEncoderProfileMessage | WgslDeferredReadbackBenchmarkMessage | LoadPackMessage | KernelProbeMessage | KernelBenchmarkMessage | OrtBenchmarkMessage | WgslHeadsProbeMessage | WgslHeadsVsOrtFixturesMessage | MappedPolicyProbeMessage | QkvProbeMessage | QkvBenchmarkMessage | AttentionScoreBenchmarkMessage | AttentionScoreOrtBenchmarkMessage | SoftmaxBenchmarkMessage | AttentionValueBenchmarkMessage | AttentionValueOrtBenchmarkMessage | AttentionBlockBenchmarkMessage | AttentionOutputBenchmarkMessage | AttentionOutputOrtBenchmarkMessage | Encoder0FfnBenchmarkMessage | Encoder0FfnOrtBenchmarkMessage | Encoder0BlockBenchmarkMessage | EncoderStackBenchmarkMessage | Encoder0BlockOrtBenchmarkMessage | CancelMessage;

type SearchWorkerResult = Omit<Lc0SearchResult, 'search'> & {
  stats?: Lc0SearchResult['search']['stats'];
  elapsedMs: number;
  cancelled?: boolean;
};

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

type WorkerResponse =
  | { type: 'ready'; id: number; backend: string; modelCache: string }
  | { type: 'evaluationResult'; id: number; result: Lc0Evaluation }
  | { type: 'evaluationBatchResult'; id: number; result: Lc0Evaluation[] }
  | { type: 'hybridEvaluationResult'; id: number; result: Lc0WebHybridEvaluationResult }
  | { type: 'hybridEncoderProfileResult'; id: number; result: Lc0WebHybridEncoderProfileResult }
  | { type: 'wgslDeferredReadbackBenchmarkResult'; id: number; result: Lc0WebWgslDeferredReadbackBenchResult }
  | { type: 'packLoadResult'; id: number; result: PackLoadResult }
  | { type: 'kernelProbeResult'; id: number; result: Lc0WebMatmulAddKernelProbeResult }
  | { type: 'kernelBenchmarkResult'; id: number; result: Lc0WebMatmulAddKernelBenchmarkResult }
  | { type: 'ortBenchmarkResult'; id: number; result: Lc0WebMatmulAddOrtBenchmarkResult }
  | { type: 'qkvProbeResult'; id: number; result: Lc0WebQkvProjectionProbeResult }
  | { type: 'qkvBenchmarkResult'; id: number; result: Lc0WebQkvProjectionBenchmarkResult }
  | { type: 'attentionScoreBenchmarkResult'; id: number; result: Lc0WebAttentionScoreBenchmarkResult }
  | { type: 'attentionScoreOrtBenchmarkResult'; id: number; result: Lc0WebAttentionScoreOrtBenchmarkResult }
  | { type: 'softmaxBenchmarkResult'; id: number; result: Lc0WebSoftmaxBenchmarkResult }
  | { type: 'attentionValueBenchmarkResult'; id: number; result: Lc0WebAttentionValueBenchmarkResult }
  | { type: 'attentionValueOrtBenchmarkResult'; id: number; result: Lc0WebAttentionValueOrtBenchmarkResult }
  | { type: 'attentionBlockBenchmarkResult'; id: number; result: Lc0WebAttentionBlockBenchmarkResult }
  | { type: 'attentionOutputBenchmarkResult'; id: number; result: Lc0WebAttentionOutputBenchmarkResult }
  | { type: 'attentionOutputOrtBenchmarkResult'; id: number; result: Lc0WebAttentionOutputOrtBenchmarkResult }
  | { type: 'encoder0FfnBenchmarkResult'; id: number; result: Lc0WebEncoder0FfnBenchmarkResult }
  | { type: 'encoder0FfnOrtBenchmarkResult'; id: number; result: Lc0WebEncoder0FfnOrtBenchmarkResult }
  | { type: 'encoder0BlockBenchmarkResult'; id: number; result: Lc0WebEncoder0BlockBenchmarkResult }
  | { type: 'encoder0BlockOrtBenchmarkResult'; id: number; result: Lc0WebEncoder0BlockOrtBenchmarkResult }
  | { type: 'encoderStackBenchmarkResult'; id: number; result: Lc0WebEncoderStackBenchmarkResult }
  | { type: 'wgslHeadsProbeResult'; id: number; result: Lc0WebWgslHeadsProbeResult }
  | { type: 'wgslHeadsVsOrtFixturesResult'; id: number; result: Lc0WebWgslHeadsVsOrtFixturesResult }
  | { type: 'mappedPolicyProbeResult'; id: number; result: Lc0WebMappedPolicyProbeResult }
  | { type: 'searchResult'; id: number; result: SearchWorkerResult }
  | { type: 'searchReset'; id: number }
  | { type: 'error'; id: number; error: string };

type WorkerEvaluator = Lc0EvaluationProvider & {
  evaluateBatch(inputs: Lc0EvaluatorInput[]): Promise<Lc0Evaluation[]> | Lc0Evaluation[];
};

let evaluator: WorkerEvaluator | null = null;
let searcher: Lc0PuctSearcher | null = null;
let configuredModelUrl: string | null = null;
let configuredInitKey: string | null = null;
let configuredBackend = '';
let configuredModelCacheStatus = '';
/** In-flight search abort controllers keyed by request id, so cancel messages can stop them. */
const activeSearches = new Map<number, AbortController>();
// This worker owns exactly one ORT session. Queue all model operations so the
// page can broker repeated eval/search requests here without concurrent
// session.run() calls against the same static batch-1 WebGPU/WASM session.
let operationQueue: Promise<void> = Promise.resolve();

function nowMs(): number {
  return typeof performance === 'undefined' ? Date.now() : performance.now();
}

function post(message: WorkerResponse): void {
  self.postMessage(message);
}

function enqueueModelOperation(work: () => Promise<void>): Promise<void> {
  const run = operationQueue.then(work, work);
  operationQueue = run.catch(() => undefined);
  return run;
}

async function handleInit(message: InitMessage): Promise<void> {
  const initKey = JSON.stringify({
    runtime: message.runtime ?? 'onnx',
    modelUrl: message.modelUrl,
    ep: message.ep,
    cacheModel: message.cacheModel,
    packUrl: message.packUrl,
    layers: message.layers,
    verifyShards: message.verifyShards,
    headBackend: message.headBackend,
    wgslBatchMode: message.wgslBatchMode,
    inputBackend: message.inputBackend,
    encoderKernelVariant: message.encoderKernelVariant,
    evalCacheEntries: message.evalCacheEntries ?? 0,
  });
  if (evaluator && configuredInitKey === initKey) {
    post({ type: 'ready', id: message.id, backend: configuredBackend, modelCache: `${configuredModelCacheStatus} · reused existing worker session` });
    return;
  }

  const evalCacheEntries = Math.max(0, Math.floor(message.evalCacheEntries ?? 0));
  const cacheLabel = evalCacheEntries > 0 ? ` · eval-cache ${evalCacheEntries}` : '';
  setRequestedOrtExecutionProviderForCurrentThread(message.ep);
  if (message.runtime === 'hybrid') {
    if (!message.packUrl) throw new Error('hybrid LC0 worker init requires packUrl');
    const baseEvaluator: WorkerEvaluator = new Lc0WebHybridEvaluator({
      packUrl: message.packUrl,
      layers: message.layers,
      verifyShards: message.verifyShards,
      headBackend: message.headBackend,
      wgslBatchMode: message.wgslBatchMode,
      inputBackend: message.inputBackend,
      encoderKernelVariant: message.encoderKernelVariant,
    });
    const nextEvaluator: WorkerEvaluator = evalCacheEntries > 0
      ? new CachedLc0Evaluator(baseEvaluator, { maxEntries: evalCacheEntries })
      : baseEvaluator;
    const previousEvaluator = evaluator;
    evaluator = nextEvaluator;
    searcher = new Lc0PuctSearcher(nextEvaluator);
    configuredModelUrl = message.packUrl;
    configuredInitKey = initKey;
    configuredBackend = message.headBackend === 'wgsl' ? 'lc0web-wgsl-encoder-wgsl-heads' : 'lc0web-wgsl-encoder-ort-heads';
    configuredModelCacheStatus = `hybrid-pack-lazy${cacheLabel}`;
    await previousEvaluator?.dispose?.();
    post({ type: 'ready', id: message.id, backend: configuredBackend, modelCache: configuredModelCacheStatus });
    return;
  }
  const modelLoad = await loadLc0ModelForOrt(message.modelUrl, { cache: message.cacheModel });
  const baseEvaluator = await Lc0OnnxEvaluator.create(modelLoad.model);
  const nextEvaluator: WorkerEvaluator = evalCacheEntries > 0
    ? new CachedLc0Evaluator(baseEvaluator, { maxEntries: evalCacheEntries })
    : baseEvaluator;
  const nextSearcher = new Lc0PuctSearcher(nextEvaluator);
  const diagnostics = await collectOrtRuntimeDiagnostics();
  const previousEvaluator = evaluator;
  evaluator = nextEvaluator;
  searcher = nextSearcher;
  configuredModelUrl = message.modelUrl;
  configuredInitKey = initKey;
  configuredBackend = diagnostics.describe;
  configuredModelCacheStatus = `${describeLc0ModelLoad(modelLoad)}${cacheLabel}`;
  await previousEvaluator?.dispose?.();
  post({ type: 'ready', id: message.id, backend: configuredBackend, modelCache: configuredModelCacheStatus });
}

async function handleEvaluate(message: EvaluateMessage): Promise<void> {
  if (!evaluator) throw new Error('LC0 search worker evaluator is not initialized');
  post({ type: 'evaluationResult', id: message.id, result: await evaluator.evaluate(message.input) });
}

async function handleEvaluateBatch(message: EvaluateBatchMessage): Promise<void> {
  if (!evaluator) throw new Error('LC0 search worker evaluator is not initialized');
  post({ type: 'evaluationBatchResult', id: message.id, result: await evaluator.evaluateBatch(message.inputs) });
}

async function handleLoadPack(message: LoadPackMessage): Promise<void> {
  const pack = await loadLc0WebModelPack(message.packUrl, {
    loadWeights: message.loadWeights,
    verifyShards: message.verifyShards,
    tensorNames: message.tensorNames,
  });
  let loadedTensorBytes = 0;
  for (const tensor of pack.tensors.values()) loadedTensorBytes += tensor.bytes.byteLength;
  post({
    type: 'packLoadResult',
    id: message.id,
    result: {
      packUrl: pack.manifestUrl,
      modelName: pack.manifest.model.name,
      sourceSha256: pack.manifest.model.sourceSha256,
      layout: pack.manifest.model.layout,
      recommendedRuntime: pack.manifest.model.recommendedRuntime,
      tensorCount: pack.manifest.weights.tensorCount,
      loadedTensorCount: pack.tensors.size,
      loadedTensorBytes,
      shardCount: pack.manifest.weights.shards.length,
      verifiedShardCount: pack.verifiedShards.length,
      shardBytes: pack.manifest.weights.shards.reduce((sum, shard) => sum + shard.bytes, 0),
      elapsedMs: pack.elapsedMs,
    },
  });
}

async function handleKernelProbe(message: KernelProbeMessage): Promise<void> {
  const result = await runLc0WebMatmulAddKernelProbe({
    packUrl: message.packUrl,
    weightTensorName: message.weightTensorName,
    biasTensorName: message.biasTensorName,
    iterations: message.iterations,
    warmup: message.warmup,
    verifyShards: message.verifyShards,
    variant: message.variant,
  });
  post({ type: 'kernelProbeResult', id: message.id, result });
}

async function handleKernelBenchmark(message: KernelBenchmarkMessage): Promise<void> {
  const result = await runLc0WebMatmulAddKernelBenchmark({
    packUrl: message.packUrl,
    weightTensorName: message.weightTensorName,
    biasTensorName: message.biasTensorName,
    iterations: message.iterations,
    warmup: message.warmup,
    verifyShards: message.verifyShards,
    variant: message.variant,
  });
  post({ type: 'kernelBenchmarkResult', id: message.id, result });
}

async function handleQkvProbe(message: QkvProbeMessage): Promise<void> {
  const result = await runLc0WebQkvProjectionProbe({
    packUrl: message.packUrl,
    iterations: message.iterations,
    warmup: message.warmup,
    verifyShards: message.verifyShards,
  });
  post({ type: 'qkvProbeResult', id: message.id, result });
}

async function handleQkvBenchmark(message: QkvBenchmarkMessage): Promise<void> {
  const result = await runLc0WebQkvProjectionBenchmark({
    packUrl: message.packUrl,
    iterations: message.iterations,
    warmup: message.warmup,
    verifyShards: message.verifyShards,
  });
  post({ type: 'qkvBenchmarkResult', id: message.id, result });
}

async function handleAttentionScoreBenchmark(message: AttentionScoreBenchmarkMessage): Promise<void> {
  const result = await runLc0WebAttentionScoreBenchmark({
    packUrl: message.packUrl,
    iterations: message.iterations,
    warmup: message.warmup,
    verifyShards: message.verifyShards,
  });
  post({ type: 'attentionScoreBenchmarkResult', id: message.id, result });
}

async function handleAttentionScoreOrtBenchmark(message: AttentionScoreOrtBenchmarkMessage): Promise<void> {
  setRequestedOrtExecutionProviderForCurrentThread(message.ep);
  const result = await runLc0WebAttentionScoreOrtBenchmark({
    packUrl: message.packUrl,
    iterations: message.iterations,
    warmup: message.warmup,
    verifyShards: message.verifyShards,
  });
  post({ type: 'attentionScoreOrtBenchmarkResult', id: message.id, result });
}

async function handleSoftmaxBenchmark(message: SoftmaxBenchmarkMessage): Promise<void> {
  const result = await runLc0WebSoftmaxBenchmark({
    packUrl: message.packUrl,
    iterations: message.iterations,
    warmup: message.warmup,
    verifyShards: message.verifyShards,
  });
  post({ type: 'softmaxBenchmarkResult', id: message.id, result });
}

async function handleAttentionValueBenchmark(message: AttentionValueBenchmarkMessage): Promise<void> {
  const result = await runLc0WebAttentionValueBenchmark({
    packUrl: message.packUrl,
    iterations: message.iterations,
    warmup: message.warmup,
    verifyShards: message.verifyShards,
  });
  post({ type: 'attentionValueBenchmarkResult', id: message.id, result });
}

async function handleAttentionValueOrtBenchmark(message: AttentionValueOrtBenchmarkMessage): Promise<void> {
  setRequestedOrtExecutionProviderForCurrentThread(message.ep);
  const result = await runLc0WebAttentionValueOrtBenchmark({
    packUrl: message.packUrl,
    iterations: message.iterations,
    warmup: message.warmup,
    verifyShards: message.verifyShards,
  });
  post({ type: 'attentionValueOrtBenchmarkResult', id: message.id, result });
}

async function handleAttentionBlockBenchmark(message: AttentionBlockBenchmarkMessage): Promise<void> {
  const result = await runLc0WebAttentionBlockBenchmark({
    packUrl: message.packUrl,
    iterations: message.iterations,
    warmup: message.warmup,
    verifyShards: message.verifyShards,
    fusedScoreSoftmax: message.fusedScoreSoftmax,
    attentionQkvKernelVariant: message.attentionQkvKernelVariant,
  });
  post({ type: 'attentionBlockBenchmarkResult', id: message.id, result });
}

async function handleAttentionOutputBenchmark(message: AttentionOutputBenchmarkMessage): Promise<void> {
  const result = await runLc0WebAttentionOutputBenchmark({
    packUrl: message.packUrl,
    iterations: message.iterations,
    warmup: message.warmup,
    verifyShards: message.verifyShards,
    encoderPrefix: message.encoderPrefix,
    attentionOutProjKernelVariant: message.attentionOutProjKernelVariant,
  });
  post({ type: 'attentionOutputBenchmarkResult', id: message.id, result });
}

async function handleAttentionOutputOrtBenchmark(message: AttentionOutputOrtBenchmarkMessage): Promise<void> {
  setRequestedOrtExecutionProviderForCurrentThread(message.ep);
  const result = await runLc0WebAttentionOutputOrtBenchmark({
    packUrl: message.packUrl,
    iterations: message.iterations,
    warmup: message.warmup,
    verifyShards: message.verifyShards,
    encoderPrefix: message.encoderPrefix,
  });
  post({ type: 'attentionOutputOrtBenchmarkResult', id: message.id, result });
}

async function handleEncoder0FfnBenchmark(message: Encoder0FfnBenchmarkMessage): Promise<void> {
  const result = await runLc0WebEncoder0FfnBenchmark({
    packUrl: message.packUrl,
    iterations: message.iterations,
    warmup: message.warmup,
    verifyShards: message.verifyShards,
    encoderPrefix: message.encoderPrefix,
    ffnKernelVariant: message.ffnKernelVariant,
  });
  post({ type: 'encoder0FfnBenchmarkResult', id: message.id, result });
}

async function handleEncoder0FfnOrtBenchmark(message: Encoder0FfnOrtBenchmarkMessage): Promise<void> {
  setRequestedOrtExecutionProviderForCurrentThread(message.ep);
  const result = await runLc0WebEncoder0FfnOrtBenchmark({
    packUrl: message.packUrl,
    iterations: message.iterations,
    warmup: message.warmup,
    verifyShards: message.verifyShards,
    encoderPrefix: message.encoderPrefix,
  });
  post({ type: 'encoder0FfnOrtBenchmarkResult', id: message.id, result });
}

async function handleEncoder0BlockBenchmark(message: Encoder0BlockBenchmarkMessage): Promise<void> {
  const result = await runLc0WebEncoder0BlockBenchmark({
    packUrl: message.packUrl,
    iterations: message.iterations,
    warmup: message.warmup,
    verifyShards: message.verifyShards,
    encoderPrefix: message.encoderPrefix,
  });
  post({ type: 'encoder0BlockBenchmarkResult', id: message.id, result });
}

async function handleEncoder0BlockOrtBenchmark(message: Encoder0BlockOrtBenchmarkMessage): Promise<void> {
  setRequestedOrtExecutionProviderForCurrentThread(message.ep);
  const result = await runLc0WebEncoder0BlockOrtBenchmark({
    packUrl: message.packUrl,
    iterations: message.iterations,
    warmup: message.warmup,
    verifyShards: message.verifyShards,
    encoderPrefix: message.encoderPrefix,
  });
  post({ type: 'encoder0BlockOrtBenchmarkResult', id: message.id, result });
}

async function handleEncoderStackBenchmark(message: EncoderStackBenchmarkMessage): Promise<void> {
  setRequestedOrtExecutionProviderForCurrentThread(message.ep);
  const result = await runLc0WebEncoderStackBenchmark({
    packUrl: message.packUrl,
    layers: message.layers,
    warmup: message.warmup,
    verifyShards: message.verifyShards,
    compareOrt: message.compareOrt,
    compareHeads: message.compareHeads,
  });
  post({ type: 'encoderStackBenchmarkResult', id: message.id, result });
}

async function handleWgslHeadsProbe(message: WgslHeadsProbeMessage): Promise<void> {
  setRequestedOrtExecutionProviderForCurrentThread(message.ep);
  const result = await runLc0WebWgslHeadsProbe({
    packUrl: message.packUrl,
    verifyShards: message.verifyShards,
  });
  post({ type: 'wgslHeadsProbeResult', id: message.id, result });
}

async function handleWgslHeadsVsOrtFixtures(message: WgslHeadsVsOrtFixturesMessage): Promise<void> {
  setRequestedOrtExecutionProviderForCurrentThread(message.ep);
  const result = await runLc0WebWgslHeadsVsOrtFixtures({
    packUrl: message.packUrl,
    fixtures: message.fixtures,
    layers: message.layers,
    verifyShards: message.verifyShards,
    mappedPolicyTolerance: message.mappedPolicyTolerance,
    wdlTolerance: message.wdlTolerance,
  });
  post({ type: 'wgslHeadsVsOrtFixturesResult', id: message.id, result });
}

async function handleMappedPolicyProbe(message: MappedPolicyProbeMessage): Promise<void> {
  const result = await runLc0WebMappedPolicyProbe();
  post({ type: 'mappedPolicyProbeResult', id: message.id, result });
}

async function handleOrtBenchmark(message: OrtBenchmarkMessage): Promise<void> {
  setRequestedOrtExecutionProviderForCurrentThread(message.ep);
  const result = await runLc0WebMatmulAddOrtBenchmark({
    packUrl: message.packUrl,
    weightTensorName: message.weightTensorName,
    biasTensorName: message.biasTensorName,
    iterations: message.iterations,
    warmup: message.warmup,
    verifyShards: message.verifyShards,
  });
  post({ type: 'ortBenchmarkResult', id: message.id, result });
}

function isAbortError(error: unknown): boolean {
  return error instanceof Error && error.name === 'AbortError';
}

async function handleSearch(message: SearchMessage): Promise<void> {
  if (!searcher) throw new Error('LC0 search worker is not initialized');
  const started = nowMs();
  const controller = new AbortController();
  activeSearches.set(message.id, controller);
  try {
    const searchOptions: Lc0SearchOptions = {
      visits: message.visits,
      movetimeMs: message.movetimeMs,
      batchSize: message.batchSize ?? 1,
      batchCollisionMode: message.batchCollisionMode,
      batchPipelineDepth: message.batchPipelineDepth,
      multiPv: message.multiPv,
      reuseTree: message.reuseTree,
      earlyStop: message.earlyStop,
      cpuct: message.cpuct,
      cpuctSchedule: message.cpuctSchedule,
      fpuStrategy: message.fpuStrategy,
      fpuReduction: message.fpuReduction,
      temperature: message.temperature,
      signal: controller.signal,
      yieldEveryMs: 16,
    };
    const result = await searcher.search(message.input, searchOptions);
    post({
      type: 'searchResult',
      id: message.id,
      result: {
        fen: result.fen,
        move: result.move,
        visits: result.visits,
        value: result.value,
        children: result.children,
        pv: result.pv,
        multiPv: result.multiPv,
        stats: result.search.stats,
        elapsedMs: nowMs() - started,
        cancelled: controller.signal.aborted,
      },
    });
  } catch (error) {
    if (!isAbortError(error)) throw error;
    // Cancellation discards the partial tree; report an empty cancelled result.
    post({
      type: 'searchResult',
      id: message.id,
      result: { fen: '', visits: 0, value: 0, children: [], pv: [], cancelled: true, elapsedMs: nowMs() - started },
    });
  } finally {
    activeSearches.delete(message.id);
  }
}

async function handleHybridEvaluate(message: HybridEvaluateMessage): Promise<void> {
  setRequestedOrtExecutionProviderForCurrentThread('wasm');
  const result = await runLc0WebHybridEvaluation({
    packUrl: message.packUrl,
    input: message.input,
    layers: message.layers,
    verifyShards: message.verifyShards,
    headBackend: message.headBackend,
    wgslBatchMode: message.wgslBatchMode,
    inputBackend: message.inputBackend,
  });
  post({ type: 'hybridEvaluationResult', id: message.id, result });
}

async function handleHybridEncoderProfile(message: HybridEncoderProfileMessage): Promise<void> {
  setRequestedOrtExecutionProviderForCurrentThread('wasm');
  const result = await runLc0WebHybridEncoderProfile({
    packUrl: message.packUrl,
    input: message.input,
    layers: message.layers,
    iterations: message.iterations,
    warmup: message.warmup,
    verifyShards: message.verifyShards,
    inputBackend: message.inputBackend,
    encoderKernelVariant: message.encoderKernelVariant,
    profileMode: message.profileMode,
  });
  post({ type: 'hybridEncoderProfileResult', id: message.id, result });
}

async function handleWgslDeferredReadbackBenchmark(message: WgslDeferredReadbackBenchmarkMessage): Promise<void> {
  setRequestedOrtExecutionProviderForCurrentThread('wasm');
  const result = await runLc0WebWgslDeferredReadbackBenchmark({
    packUrl: message.packUrl,
    inputs: message.inputs,
    layers: message.layers,
    verifyShards: message.verifyShards,
    inputBackend: message.inputBackend,
    batchSize: message.batchSize,
    iterations: message.iterations,
    warmup: message.warmup,
  });
  post({ type: 'wgslDeferredReadbackBenchmarkResult', id: message.id, result });
}

function handleCancel(message: CancelMessage): void {
  if (message.target !== undefined) {
    activeSearches.get(message.target)?.abort();
    return;
  }
  for (const controller of activeSearches.values()) controller.abort();
}

self.addEventListener('message', (event: MessageEvent<WorkerRequest>) => {
  const message = event.data;
  // Cancellation is synchronous and must not be wrapped in the error reporter,
  // so an abort never posts a spurious error for the cancel request itself.
  if (message.type === 'cancel') {
    handleCancel(message);
    return;
  }
  void enqueueModelOperation(async () => {
    try {
      if (message.type === 'init') await handleInit(message);
      else if (message.type === 'loadPack') await handleLoadPack(message);
      else if (message.type === 'kernelProbe') await handleKernelProbe(message);
      else if (message.type === 'kernelBenchmark') await handleKernelBenchmark(message);
      else if (message.type === 'ortBenchmark') await handleOrtBenchmark(message);
      else if (message.type === 'qkvProbe') await handleQkvProbe(message);
      else if (message.type === 'qkvBenchmark') await handleQkvBenchmark(message);
      else if (message.type === 'attentionScoreBenchmark') await handleAttentionScoreBenchmark(message);
      else if (message.type === 'attentionScoreOrtBenchmark') await handleAttentionScoreOrtBenchmark(message);
      else if (message.type === 'softmaxBenchmark') await handleSoftmaxBenchmark(message);
      else if (message.type === 'attentionValueBenchmark') await handleAttentionValueBenchmark(message);
      else if (message.type === 'attentionValueOrtBenchmark') await handleAttentionValueOrtBenchmark(message);
      else if (message.type === 'attentionBlockBenchmark') await handleAttentionBlockBenchmark(message);
      else if (message.type === 'attentionOutputBenchmark') await handleAttentionOutputBenchmark(message);
      else if (message.type === 'attentionOutputOrtBenchmark') await handleAttentionOutputOrtBenchmark(message);
      else if (message.type === 'encoder0FfnBenchmark') await handleEncoder0FfnBenchmark(message);
      else if (message.type === 'encoder0FfnOrtBenchmark') await handleEncoder0FfnOrtBenchmark(message);
      else if (message.type === 'encoder0BlockBenchmark') await handleEncoder0BlockBenchmark(message);
      else if (message.type === 'encoder0BlockOrtBenchmark') await handleEncoder0BlockOrtBenchmark(message);
      else if (message.type === 'encoderStackBenchmark') await handleEncoderStackBenchmark(message);
      else if (message.type === 'wgslHeadsProbe') await handleWgslHeadsProbe(message);
      else if (message.type === 'wgslHeadsVsOrtFixtures') await handleWgslHeadsVsOrtFixtures(message);
      else if (message.type === 'mappedPolicyProbe') await handleMappedPolicyProbe(message);
      else if (message.type === 'hybridEvaluate') await handleHybridEvaluate(message);
      else if (message.type === 'hybridEncoderProfile') await handleHybridEncoderProfile(message);
      else if (message.type === 'wgslDeferredReadbackBenchmark') await handleWgslDeferredReadbackBenchmark(message);
      else if (message.type === 'evaluate') {
        if (!configuredModelUrl) throw new Error('LC0 search worker missing model URL');
        await handleEvaluate(message);
      } else if (message.type === 'evaluateBatch') {
        if (!configuredModelUrl) throw new Error('LC0 search worker missing model URL');
        await handleEvaluateBatch(message);
      } else if (message.type === 'resetSearch') {
        searcher?.resetTree();
        post({ type: 'searchReset', id: message.id });
      } else if (message.type === 'search') {
        if (!configuredModelUrl) throw new Error('LC0 search worker missing model URL');
        await handleSearch(message);
      }
    } catch (error) {
      post({ type: 'error', id: message.id, error: (error as Error).message });
    }
  });
});
