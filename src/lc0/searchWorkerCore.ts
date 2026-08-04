import '../nn/ortConsoleFilter.ts';
import {
  collectOrtRuntimeDiagnostics,
  type OrtExecutionProviderPreference,
  type OrtRuntimeDiagnosticOptions,
  type OrtRuntimeDiagnostics,
  type OrtWasmArtifactSelection,
  setOrtRuntimeDiagnosticOptionsForCurrentThread,
  setRequestedOrtWasmArtifactForCurrentThread,
  setRequestedOrtWasmThreadsForCurrentThread,
} from '../nn/ortRuntime.ts';
import type { CpuctSchedule, FpuStrategy, SearchBatchCollisionMode, SearchEarlyStop } from '../search/puct.ts';
import type { Lc0WebEncoderKernelVariant, Lc0WebExecutionFootprint, Lc0WebHybridLegalPriorsBackend } from './hybridEvaluator.ts';
import { describeLc0ModelLoad, loadLc0ModelForOrt } from './modelCache.ts';
import {
  CachedLc0Evaluator,
  type Lc0Evaluation,
  type Lc0EvaluationCacheFootprint,
  type Lc0EvaluationProvider,
  type Lc0EvaluatorInput,
  Lc0OnnxEvaluator,
} from './onnxEvaluator.ts';
import { applyOrtExecutionProvider } from './ortWorkerExecutionProvider.ts';
import { Lc0PuctSearcher, type Lc0SearchOptions, type Lc0SearchProgress, type Lc0SearchResult } from './search.ts';
import { Lc0WholeOnnxWebgpuEvaluator } from './wholeOnnxWebgpuEvaluator.ts';

type InitMessage = {
  type: 'init';
  id: number;
  modelUrl: string;
  ep: OrtExecutionProviderPreference;
  cacheModel: boolean;
  runtime?: 'onnx' | 'hybrid' | 'whole-onnx-webgpu';
  packUrl?: string;
  wholeModelManifestUrl?: string;
  wholeModelBatch?: number;
  wholeModelTensorCache?: boolean;
  layers?: number;
  verifyShards?: boolean;
  headBackend?: 'ort' | 'wgsl';
  wgslBatchMode?: 'physical' | 'serial';
  inputBackend?: 'js' | 'wgsl' | 'wasm';
  legalPriorsBackend?: Lc0WebHybridLegalPriorsBackend;
  encoderKernelVariant?: Lc0WebEncoderKernelVariant;
  evalCacheEntries?: number;
  ortDiagnostics?: OrtRuntimeDiagnosticOptions;
  ortWasmArtifact?: OrtWasmArtifactSelection;
  ortWasmThreads?: number;
  /** Stream model download progress back as 'downloadProgress' messages. */
  reportDownloadProgress?: boolean;
  requestPersistentModelStorage?: boolean;
  minimumFreeBytesAfterModelCache?: number;
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
  traceSearchVisits?: boolean;
  multiPv?: number;
  reuseTree?: boolean;
  earlyStop?: SearchEarlyStop;
  cpuct?: number;
  fpu?: number;
  cpuctSchedule?: CpuctSchedule;
  fpuStrategy?: FpuStrategy;
  fpuReduction?: number;
  temperature?: number;
  drawScore?: number;
  contemptElo?: number;
  searchContemptLimit?: number;
  reportProgress?: boolean;
  progressEveryMs?: number;
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

type CancelMessage = {
  type: 'cancel';
  id: number;
  /** Optional target request id; when omitted, cancels any in-flight search. */
  target?: number;
};

type WorkerRequest = InitMessage | SearchMessage | ResetSearchMessage | EvaluateMessage | EvaluateBatchMessage | CancelMessage;

type WebGpuBufferAllocationTelemetry = {
  installed: boolean;
  createBufferCount: number;
  createBufferBytes: number;
  maxBufferBytes: number;
  failures: number;
  byUsage: Record<string, { count: number; bytes: number }>;
  note: string;
};

type SearchWorkerResult = Omit<Lc0SearchResult, 'search'> & {
  stats?: Lc0SearchResult['search']['stats'];
  elapsedMs: number;
  cancelled?: boolean;
  executionFootprint?: Lc0WebExecutionFootprint;
  cacheFootprint?: Lc0EvaluationCacheFootprint;
  gpuBufferAllocation?: WebGpuBufferAllocationTelemetry;
};

type SearchWorkerProgress = Lc0SearchProgress;

type WorkerResponse =
  | { type: 'ready'; id: number; backend: string; modelCache: string; ortWasm?: OrtRuntimeDiagnostics['wasm']; ortFallback?: OrtRuntimeDiagnostics['fallback'] }
  | { type: 'evaluationResult'; id: number; result: Lc0Evaluation }
  | { type: 'evaluationBatchResult'; id: number; result: Lc0Evaluation[] }
  | { type: 'searchProgress'; id: number; progress: SearchWorkerProgress }
  | { type: 'searchResult'; id: number; result: SearchWorkerResult }
  | { type: 'searchReset'; id: number }
  | { type: 'downloadProgress'; id: number; loadedBytes: number; totalBytes?: number }
  | { type: 'error'; id: number; error: string };

type WorkerEvaluator = Lc0EvaluationProvider & {
  evaluateBatch(inputs: Lc0EvaluatorInput[]): Promise<Lc0Evaluation[]> | Lc0Evaluation[];
  executionFootprint?(): Lc0WebExecutionFootprint | undefined;
  cacheFootprint?(): Lc0EvaluationCacheFootprint | undefined;
};

type HybridEvaluatorFactory = (options: {
  packUrl: string;
  layers?: number;
  verifyShards?: boolean;
  headBackend?: 'ort' | 'wgsl';
  wgslBatchMode?: 'physical' | 'serial';
  inputBackend?: 'js' | 'wgsl' | 'wasm';
  legalPriorsBackend?: Lc0WebHybridLegalPriorsBackend;
  encoderKernelVariant?: Lc0WebEncoderKernelVariant;
}) => WorkerEvaluator;

let hybridEvaluatorFactory: HybridEvaluatorFactory | undefined;

const webGpuBufferAllocationTelemetry: WebGpuBufferAllocationTelemetry = {
  installed: false,
  createBufferCount: 0,
  createBufferBytes: 0,
  maxBufferBytes: 0,
  failures: 0,
  byUsage: {},
  note: 'GPUDevice.createBuffer request telemetry only; counts allocation requests visible to this worker monkeypatch, not live GPU residency.',
};
const patchedAdapters = new WeakSet<object>();
const patchedDevices = new WeakSet<object>();

function recordWebGpuBufferAllocation(descriptor: unknown): void {
  const maybeDescriptor = descriptor as { size?: unknown; usage?: unknown } | undefined;
  const size = typeof maybeDescriptor?.size === 'bigint' ? Number(maybeDescriptor.size) : Number(maybeDescriptor?.size ?? 0);
  const bytes = Number.isFinite(size) && size > 0 ? Math.floor(size) : 0;
  const usage = String(maybeDescriptor?.usage ?? 'unknown');
  webGpuBufferAllocationTelemetry.createBufferCount += 1;
  webGpuBufferAllocationTelemetry.createBufferBytes += bytes;
  webGpuBufferAllocationTelemetry.maxBufferBytes = Math.max(webGpuBufferAllocationTelemetry.maxBufferBytes, bytes);
  const bucket = webGpuBufferAllocationTelemetry.byUsage[usage] ?? { count: 0, bytes: 0 };
  bucket.count += 1;
  bucket.bytes += bytes;
  webGpuBufferAllocationTelemetry.byUsage[usage] = bucket;
}

function patchWebGpuDevice(device: unknown): void {
  if (!device || (typeof device !== 'object' && typeof device !== 'function')) return;
  const target = device as { createBuffer?: (...args: unknown[]) => unknown };
  if (patchedDevices.has(target) || typeof target.createBuffer !== 'function') return;
  const originalCreateBuffer = target.createBuffer;
  patchedDevices.add(target);
  target.createBuffer = function patchedCreateBuffer(this: unknown, descriptor: unknown, ...rest: unknown[]) {
    try {
      recordWebGpuBufferAllocation(descriptor);
    } catch {
      webGpuBufferAllocationTelemetry.failures += 1;
    }
    return originalCreateBuffer.call(this, descriptor, ...rest);
  };
}

function patchWebGpuAdapter(adapter: unknown): void {
  if (!adapter || (typeof adapter !== 'object' && typeof adapter !== 'function')) return;
  const target = adapter as { requestDevice?: (...args: unknown[]) => Promise<unknown> };
  if (patchedAdapters.has(target) || typeof target.requestDevice !== 'function') return;
  const originalRequestDevice = target.requestDevice;
  patchedAdapters.add(target);
  target.requestDevice = async function patchedRequestDevice(this: unknown, ...args: unknown[]) {
    const device = await originalRequestDevice.apply(this, args);
    patchWebGpuDevice(device);
    return device;
  };
}

function installWebGpuBufferAllocationProbe(): void {
  try {
    const gpu = (globalThis.navigator as { gpu?: { requestAdapter?: (...args: unknown[]) => Promise<unknown> } } | undefined)?.gpu;
    if (!gpu || typeof gpu.requestAdapter !== 'function') return;
    const originalRequestAdapter = gpu.requestAdapter;
    gpu.requestAdapter = async function patchedRequestAdapter(this: unknown, ...args: unknown[]) {
      const adapter = await originalRequestAdapter.apply(this, args);
      patchWebGpuAdapter(adapter);
      return adapter;
    };
    webGpuBufferAllocationTelemetry.installed = true;
  } catch {
    webGpuBufferAllocationTelemetry.failures += 1;
  }
}

function currentWebGpuBufferAllocationTelemetry(): WebGpuBufferAllocationTelemetry {
  return {
    ...webGpuBufferAllocationTelemetry,
    byUsage: Object.fromEntries(Object.entries(webGpuBufferAllocationTelemetry.byUsage).map(([usage, stats]) => [usage, { ...stats }])),
  };
}

installWebGpuBufferAllocationProbe();

let evaluator: WorkerEvaluator | null = null;
let searcher: Lc0PuctSearcher | null = null;
let configuredModelUrl: string | null = null;
let configuredInitKey: string | null = null;
let configuredBackend = '';
let configuredModelCacheStatus = '';
let configuredOrtWasm: OrtRuntimeDiagnostics['wasm'] | undefined;
let configuredOrtFallback: OrtRuntimeDiagnostics['fallback'];
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
    wholeModelManifestUrl: message.wholeModelManifestUrl,
    wholeModelBatch: message.wholeModelBatch,
    wholeModelTensorCache: message.wholeModelTensorCache,
    layers: message.layers,
    verifyShards: message.verifyShards,
    headBackend: message.headBackend,
    wgslBatchMode: message.wgslBatchMode,
    inputBackend: message.inputBackend,
    legalPriorsBackend: message.legalPriorsBackend,
    encoderKernelVariant: message.encoderKernelVariant,
    evalCacheEntries: message.evalCacheEntries ?? 0,
    requestPersistentModelStorage: message.requestPersistentModelStorage === true,
    minimumFreeBytesAfterModelCache: message.minimumFreeBytesAfterModelCache ?? null,
    ortDiagnostics: message.ortDiagnostics ?? null,
    ortWasmArtifact: message.ortWasmArtifact ?? null,
    ortWasmThreads: message.ortWasmThreads ?? null,
  });
  if (evaluator && configuredInitKey === initKey) {
    post({
      type: 'ready',
      id: message.id,
      backend: configuredBackend,
      modelCache: `${configuredModelCacheStatus} · reused existing worker session`,
      ortWasm: configuredOrtWasm,
      ortFallback: configuredOrtFallback,
    });
    return;
  }

  const evalCacheEntries = Math.max(0, Math.floor(message.evalCacheEntries ?? 0));
  const cacheLabel = evalCacheEntries > 0 ? ` · eval-cache ${evalCacheEntries}` : '';
  applyOrtExecutionProvider(message.ep);
  setOrtRuntimeDiagnosticOptionsForCurrentThread(message.ortDiagnostics ?? null);
  setRequestedOrtWasmArtifactForCurrentThread(message.ortWasmArtifact ?? null);
  setRequestedOrtWasmThreadsForCurrentThread(message.ortWasmThreads ?? null);
  if (message.runtime === 'whole-onnx-webgpu') {
    if (!message.wholeModelManifestUrl) throw new Error('whole-model LC0 worker init requires manifest URL');
    const baseEvaluator: WorkerEvaluator = await Lc0WholeOnnxWebgpuEvaluator.create({
      manifestUrl: message.wholeModelManifestUrl,
      batch: message.wholeModelBatch,
      fetchTensorCache: message.wholeModelTensorCache,
      logger: (line) => console.info('[lc0 whole-model worker]', line),
    });
    const nextEvaluator: WorkerEvaluator = evalCacheEntries > 0 ? new CachedLc0Evaluator(baseEvaluator, { maxEntries: evalCacheEntries }) : baseEvaluator;
    const previousEvaluator = evaluator;
    evaluator = nextEvaluator;
    searcher = new Lc0PuctSearcher(nextEvaluator);
    configuredModelUrl = message.wholeModelManifestUrl;
    configuredInitKey = initKey;
    configuredBackend = 'whole-onnx-webgpu';
    configuredModelCacheStatus = `whole-model-webgpu${cacheLabel}`;
    configuredOrtWasm = undefined;
    configuredOrtFallback = undefined;
    await previousEvaluator?.dispose?.();
    post({ type: 'ready', id: message.id, backend: configuredBackend, modelCache: configuredModelCacheStatus });
    return;
  }
  if (message.runtime === 'hybrid') {
    if (!message.packUrl) throw new Error('hybrid LC0 worker init requires packUrl');
    if (!hybridEvaluatorFactory) throw new Error('hybrid LC0 runtime requires the dedicated hybrid search worker');
    const baseEvaluator = hybridEvaluatorFactory({
      packUrl: message.packUrl,
      layers: message.layers,
      verifyShards: message.verifyShards,
      headBackend: message.headBackend,
      wgslBatchMode: message.wgslBatchMode,
      inputBackend: message.inputBackend,
      legalPriorsBackend: message.legalPriorsBackend,
      encoderKernelVariant: message.encoderKernelVariant,
    });
    const nextEvaluator: WorkerEvaluator = evalCacheEntries > 0 ? new CachedLc0Evaluator(baseEvaluator, { maxEntries: evalCacheEntries }) : baseEvaluator;
    const previousEvaluator = evaluator;
    evaluator = nextEvaluator;
    searcher = new Lc0PuctSearcher(nextEvaluator);
    configuredModelUrl = message.packUrl;
    configuredInitKey = initKey;
    configuredBackend = message.headBackend === 'wgsl' ? 'lc0web-wgsl-encoder-wgsl-heads' : 'lc0web-wgsl-encoder-ort-heads';
    configuredModelCacheStatus = `hybrid-pack-lazy${cacheLabel}`;
    configuredOrtWasm = undefined;
    configuredOrtFallback = undefined;
    await previousEvaluator?.dispose?.();
    post({ type: 'ready', id: message.id, backend: configuredBackend, modelCache: configuredModelCacheStatus });
    return;
  }
  // Throttle progress posts to ~2MB steps so a large net does not flood the
  // main thread with one message per network chunk.
  let lastReportedBytes = -Infinity;
  const modelLoad = await loadLc0ModelForOrt(message.modelUrl, {
    cache: message.cacheModel,
    requestPersistentStorage: message.requestPersistentModelStorage,
    minimumFreeBytesAfterCache: message.minimumFreeBytesAfterModelCache,
    onProgress: message.reportDownloadProgress
      ? (loadedBytes, totalBytes) => {
          if (loadedBytes - lastReportedBytes < 2_000_000 && loadedBytes !== totalBytes) return;
          lastReportedBytes = loadedBytes;
          post({ type: 'downloadProgress', id: message.id, loadedBytes, totalBytes });
        }
      : undefined,
  });
  const baseEvaluator = await Lc0OnnxEvaluator.create(modelLoad.model);
  const nextEvaluator: WorkerEvaluator = evalCacheEntries > 0 ? new CachedLc0Evaluator(baseEvaluator, { maxEntries: evalCacheEntries }) : baseEvaluator;
  const nextSearcher = new Lc0PuctSearcher(nextEvaluator);
  const diagnostics = await collectOrtRuntimeDiagnostics();
  const previousEvaluator = evaluator;
  evaluator = nextEvaluator;
  searcher = nextSearcher;
  configuredModelUrl = message.modelUrl;
  configuredInitKey = initKey;
  configuredBackend = diagnostics.describe;
  configuredModelCacheStatus = `${describeLc0ModelLoad(modelLoad)}${cacheLabel}`;
  configuredOrtWasm = diagnostics.wasm;
  configuredOrtFallback = diagnostics.fallback;
  await previousEvaluator?.dispose?.();
  post({
    type: 'ready',
    id: message.id,
    backend: configuredBackend,
    modelCache: configuredModelCacheStatus,
    ortWasm: configuredOrtWasm,
    ortFallback: configuredOrtFallback,
  });
}

async function handleEvaluate(message: EvaluateMessage): Promise<void> {
  if (!evaluator) throw new Error('LC0 search worker evaluator is not initialized');
  post({ type: 'evaluationResult', id: message.id, result: await evaluator.evaluate(message.input) });
}

async function handleEvaluateBatch(message: EvaluateBatchMessage): Promise<void> {
  if (!evaluator) throw new Error('LC0 search worker evaluator is not initialized');
  post({ type: 'evaluationBatchResult', id: message.id, result: await evaluator.evaluateBatch(message.inputs) });
}

function isAbortError(error: unknown): boolean {
  return error instanceof Error && error.name === 'AbortError';
}

function currentExecutionFootprint(): Lc0WebExecutionFootprint | undefined {
  const direct = evaluator?.executionFootprint?.();
  if (direct) return direct;
  const inner = (evaluator as { inner?: { executionFootprint?: () => Lc0WebExecutionFootprint | undefined } } | null)?.inner;
  return inner?.executionFootprint?.();
}

function currentCacheFootprint(): Lc0EvaluationCacheFootprint | undefined {
  const direct = evaluator?.cacheFootprint?.();
  if (direct) return direct;
  const inner = (evaluator as { inner?: { cacheFootprint?: () => Lc0EvaluationCacheFootprint | undefined } } | null)?.inner;
  return inner?.cacheFootprint?.();
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
      traceSearchVisits: message.traceSearchVisits,
      multiPv: message.multiPv,
      reuseTree: message.reuseTree,
      earlyStop: message.earlyStop,
      cpuct: message.cpuct,
      fpu: message.fpu,
      cpuctSchedule: message.cpuctSchedule,
      fpuStrategy: message.fpuStrategy,
      fpuReduction: message.fpuReduction,
      temperature: message.temperature,
      drawScore: message.drawScore,
      contemptElo: message.contemptElo,
      searchContemptLimit: message.searchContemptLimit,
      progressEveryMs: message.progressEveryMs,
      onProgress: message.reportProgress ? (progress) => post({ type: 'searchProgress', id: message.id, progress }) : undefined,
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
        executionFootprint: currentExecutionFootprint(),
        cacheFootprint: currentCacheFootprint(),
        gpuBufferAllocation: currentWebGpuBufferAllocationTelemetry(),
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

export function startSearchWorker(options: { createHybridEvaluator?: HybridEvaluatorFactory } = {}): void {
  hybridEvaluatorFactory = options.createHybridEvaluator;
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
        post({ type: 'error', id: message.id, error: error instanceof Error ? error.message : String(error) });
      }
    });
  });
}
