import { collectOrtRuntimeDiagnostics, setRequestedOrtExecutionProviderForCurrentThread, type OrtExecutionProviderPreference } from '../nn/ortRuntime.ts';
import { describeLc0ModelLoad, loadLc0ModelForOrt } from './modelCache.ts';
import { loadLc0WebModelPack } from './modelPack.ts';
import { Lc0OnnxEvaluator, type Lc0Evaluation, type Lc0EvaluatorInput } from './onnxEvaluator.ts';
import {
  runLc0WebMatmulAddKernelBenchmark,
  runLc0WebMatmulAddKernelProbe,
  runLc0WebMatmulAddOrtBenchmark,
  type Lc0WebMatmulAddKernelBenchmarkResult,
  type Lc0WebMatmulAddKernelProbeResult,
  type Lc0WebMatmulAddOrtBenchmarkResult,
} from './wgslMatmulAddProbe.ts';
import { Lc0PuctSearcher, type Lc0SearchResult } from './search.ts';

type InitMessage = {
  type: 'init';
  id: number;
  modelUrl: string;
  ep: OrtExecutionProviderPreference;
  cacheModel: boolean;
};

type SearchMessage = {
  type: 'search';
  id: number;
  input: Lc0EvaluatorInput;
  visits: number;
  batchSize?: number;
  multiPv?: number;
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

type WorkerRequest = InitMessage | SearchMessage | EvaluateMessage | EvaluateBatchMessage | LoadPackMessage | KernelProbeMessage | KernelBenchmarkMessage | OrtBenchmarkMessage | CancelMessage;

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
  | { type: 'packLoadResult'; id: number; result: PackLoadResult }
  | { type: 'kernelProbeResult'; id: number; result: Lc0WebMatmulAddKernelProbeResult }
  | { type: 'kernelBenchmarkResult'; id: number; result: Lc0WebMatmulAddKernelBenchmarkResult }
  | { type: 'ortBenchmarkResult'; id: number; result: Lc0WebMatmulAddOrtBenchmarkResult }
  | { type: 'searchResult'; id: number; result: SearchWorkerResult }
  | { type: 'error'; id: number; error: string };

let evaluator: Lc0OnnxEvaluator | null = null;
let searcher: Lc0PuctSearcher | null = null;
let configuredModelUrl: string | null = null;
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
  setRequestedOrtExecutionProviderForCurrentThread(message.ep);
  const modelLoad = await loadLc0ModelForOrt(message.modelUrl, { cache: message.cacheModel });
  evaluator = await Lc0OnnxEvaluator.create(modelLoad.model);
  searcher = new Lc0PuctSearcher(evaluator);
  configuredModelUrl = message.modelUrl;
  const diagnostics = await collectOrtRuntimeDiagnostics();
  post({ type: 'ready', id: message.id, backend: diagnostics.describe, modelCache: describeLc0ModelLoad(modelLoad) });
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
  });
  post({ type: 'kernelBenchmarkResult', id: message.id, result });
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
    const result = await searcher.search(message.input, {
      visits: message.visits,
      batchSize: message.batchSize ?? 1,
      multiPv: message.multiPv,
      signal: controller.signal,
      yieldEveryMs: 16,
    });
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
      else if (message.type === 'evaluate') {
        if (!configuredModelUrl) throw new Error('LC0 search worker missing model URL');
        await handleEvaluate(message);
      } else if (message.type === 'evaluateBatch') {
        if (!configuredModelUrl) throw new Error('LC0 search worker missing model URL');
        await handleEvaluateBatch(message);
      } else if (message.type === 'search') {
        if (!configuredModelUrl) throw new Error('LC0 search worker missing model URL');
        await handleSearch(message);
      }
    } catch (error) {
      post({ type: 'error', id: message.id, error: (error as Error).message });
    }
  });
});
