import { createOrtDeviceLossMonitor, type OrtDeviceLoss } from '../nn/ortDeviceLoss.ts';
import { releaseOrtWorkerResources } from '../nn/ortWorkerDispose.ts';
import { ExclusiveQueue } from './exclusiveQueue.ts';
import '../nn/ortConsoleFilter.ts';
import {
  collectOrtRuntimeDiagnostics,
  type OrtExecutionProviderPreference,
  type OrtRuntimeDiagnosticOptions,
  type OrtRuntimeDiagnostics,
  type OrtSessionDiagnostics,
  type OrtWasmArtifactSelection,
  setOrtDequantFoldForCurrentThread,
  setOrtRuntimeDiagnosticOptionsForCurrentThread,
  setRequestedOrtWasmArtifactForCurrentThread,
  setRequestedOrtWasmThreadsForCurrentThread,
} from '../nn/ortRuntime.ts';
import type { Lc0WebEncoderKernelVariant, Lc0WebExecutionFootprint, Lc0WebHybridLegalPriorsBackend } from './hybridEvaluator.ts';
import { createLc0OnnxEvaluatorFromUrl, DEFAULT_LC0_ORT_EVALUATOR_DEPS } from './lc0SmallModelSelection.ts';
import { describeLc0ModelLoad, type Lc0ModelLoadResult } from './modelCache.ts';
import {
  CachedLc0Evaluator,
  defaultLc0OrtIoMode,
  type Lc0Evaluation,
  type Lc0EvaluationCacheFootprint,
  type Lc0EvaluationProvider,
  type Lc0EvaluatorInput,
  type Lc0OrtIoMode,
} from './onnxEvaluator.ts';
import { applyOrtExecutionProvider } from './ortWorkerExecutionProvider.ts';
import { noteSearchCancel, takeSearchCancel } from './pendingSearchCancel.ts';
import { Lc0PuctSearcher, type Lc0SearchOptions, type Lc0SearchProgress, type Lc0SearchResult } from './search.ts';
import { Lc0WholeOnnxWebgpuEvaluator } from './wholeOnnxWebgpuEvaluator.ts';

type InitMessage = {
  type: 'init';
  id: number;
  modelUrl: string;
  /**
   * Let the worker swap the default Lc0 Small model for its split-K rewrite on
   * Apple WebGPU (src/lc0/lc0SmallModelSelection.ts). Product callers set it
   * when no ?model= override was given; omitted = load exactly modelUrl.
   */
  deviceModelSelection?: boolean;
  ep: OrtExecutionProviderPreference;
  cacheModel: boolean;
  /** Explicit lab opt-out; product callers leave verification enabled. */
  allowUnverifiedModel?: boolean;
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
  /** Fold weight-only DequantizeLinear at load; omitted = runtime default (on unless ?ortDequantFold=0). */
  dequantFold?: boolean;
  /**
   * Lab override for dynamic-batch graphs. Omitted keeps the product path
   * (symbolic leading dim runs as serial batch-1). 'logical' is one session.run.
   */
  physicalBatchMode?: 'declared' | 'logical';
  /** ORT IO mode for the ONNX runtime (Lc0OnnxEvaluatorOptions.ioMode); omitted = defaultLc0OrtIoMode(deviceModelSelection). */
  ortIoMode?: Lc0OrtIoMode;
  /** GPU-only shape policy; no model bytes or cache identity change. */
  ortFreeDimensionOverrides?: Readonly<Record<string, number>>;
  ortWasmArtifact?: OrtWasmArtifactSelection;
  ortWasmThreads?: number;
  /** Stream model download progress back as 'downloadProgress' messages. */
  reportDownloadProgress?: boolean;
  requestPersistentModelStorage?: boolean;
  minimumFreeBytesAfterModelCache?: number;
  /** Load from Cache Storage only (a cache-hit approval); downloading fails instead (modelCache.ts noDownload). */
  noModelDownload?: boolean;
  prefetchedModel?: { url: string; load?: Lc0ModelLoadResult };
};

type SearchMessage = {
  type: 'search';
  id: number;
  input: Lc0EvaluatorInput;
  reportProgress?: boolean;
} & Omit<Lc0SearchOptions, 'onProgress' | 'signal' | 'yieldEveryMs' | 'root' | 'rootMoves'>;

type ResetSearchMessage = {
  type: 'resetSearch';
  id: number;
  clearEvalCache?: boolean;
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

type DisposeMessage = { type: 'dispose'; id: number };
type WorkerRequest = InitMessage | SearchMessage | ResetSearchMessage | EvaluateMessage | EvaluateBatchMessage | CancelMessage | DisposeMessage;

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

type ReadyPayload = {
  backend: string;
  modelCache: string;
  ortWasm?: OrtRuntimeDiagnostics['wasm'];
  ortFallback?: OrtRuntimeDiagnostics['fallback'];
  ortRuntime?: OrtSessionDiagnostics;
  /** ONNX runtime only: the model the session runs (differs from modelUrl when the split-K model was selected). */
  servedModelUrl?: string;
  /** ONNX runtime only: the ORT IO mode in use (a requested GPU mode may have fallen back to 'cpu'). */
  ortIoMode?: Lc0OrtIoMode;
};

type WorkerResponse =
  | ({ type: 'ready'; id: number } & ReadyPayload)
  | { type: 'evaluationResult'; id: number; result: Lc0Evaluation }
  | { type: 'evaluationBatchResult'; id: number; result: Lc0Evaluation[] }
  | { type: 'searchProgress'; id: number; progress: Lc0SearchProgress }
  | { type: 'searchResult'; id: number; result: SearchWorkerResult }
  | { type: 'searchReset'; id: number }
  | { type: 'disposed'; id: number }
  | { type: 'downloadProgress'; id: number; loadedBytes: number; totalBytes?: number }
  | { type: 'deviceLost'; id?: never; loss: OrtDeviceLoss }
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

const gpuBufferTelemetry: WebGpuBufferAllocationTelemetry = {
  installed: false,
  createBufferCount: 0,
  createBufferBytes: 0,
  maxBufferBytes: 0,
  failures: 0,
  byUsage: {},
  note: 'GPUDevice.createBuffer request telemetry only; counts allocation requests visible to this worker monkeypatch, not live GPU residency.',
};
const patchedWebGpuObjects = new WeakSet<object>();

try {
  const gpu = (globalThis.navigator as { gpu?: { requestAdapter?: (...args: unknown[]) => Promise<unknown> } } | undefined)?.gpu;
  if (gpu && typeof gpu.requestAdapter === 'function') {
    const originalRequestAdapter = gpu.requestAdapter;
    gpu.requestAdapter = async function patchedRequestAdapter(this: unknown, ...args: unknown[]) {
      const adapter = (await originalRequestAdapter.apply(this, args)) as { requestDevice?: (...args: unknown[]) => Promise<unknown> } | null | undefined;
      if (adapter && typeof adapter.requestDevice === 'function' && !patchedWebGpuObjects.has(adapter)) {
        patchedWebGpuObjects.add(adapter);
        const originalRequestDevice = adapter.requestDevice;
        adapter.requestDevice = async function patchedRequestDevice(this: unknown, ...args: unknown[]) {
          const device = (await originalRequestDevice.apply(this, args)) as { createBuffer?: (...args: unknown[]) => unknown } | null | undefined;
          if (device && typeof device.createBuffer === 'function' && !patchedWebGpuObjects.has(device)) {
            patchedWebGpuObjects.add(device);
            const originalCreateBuffer = device.createBuffer;
            device.createBuffer = function patchedCreateBuffer(this: unknown, descriptor: unknown, ...rest: unknown[]) {
              try {
                const desc = descriptor as { size?: unknown; usage?: unknown } | undefined;
                const size = Number(desc?.size ?? 0);
                const bytes = Number.isFinite(size) && size > 0 ? Math.floor(size) : 0;
                const usage = String(desc?.usage ?? 'unknown');
                gpuBufferTelemetry.createBufferCount += 1;
                gpuBufferTelemetry.createBufferBytes += bytes;
                gpuBufferTelemetry.maxBufferBytes = Math.max(gpuBufferTelemetry.maxBufferBytes, bytes);
                const bucket = gpuBufferTelemetry.byUsage[usage] ?? { count: 0, bytes: 0 };
                bucket.count += 1;
                bucket.bytes += bytes;
                gpuBufferTelemetry.byUsage[usage] = bucket;
              } catch {
                gpuBufferTelemetry.failures += 1;
              }
              return originalCreateBuffer.call(this, descriptor, ...rest);
            };
          }
          return device;
        };
      }
      return adapter;
    };
    gpuBufferTelemetry.installed = true;
  }
} catch {
  gpuBufferTelemetry.failures += 1;
}

let evaluator: WorkerEvaluator | null = null;
let searcher: Lc0PuctSearcher | null = null;
let configured: ({ initKey: string } & ReadyPayload) | null = null;
/** In-flight search abort controllers keyed by request id, so cancel messages can stop them. */
const activeSearches = new Map<number, AbortController>();
/** The running model load. Disposal aborts it so release need not wait out a large download. */
let activeInit: AbortController | null = null;
/** Cancels whose search message is still queued behind the search that is running. */
const pendingSearchCancels: number[] = [];
// This worker owns exactly one ORT session. Queue all model operations so the
// page can broker repeated eval/search requests here without concurrent
// session.run() calls against the same static batch-1 WebGPU/WASM session.
const operationQueue = new ExclusiveQueue();
let retiring = false;

function nowMs(): number {
  return typeof performance === 'undefined' ? Date.now() : performance.now();
}

function post(message: WorkerResponse): void {
  self.postMessage(message);
}

const deviceLossMonitor = createOrtDeviceLossMonitor((loss) => post({ type: 'deviceLost', loss }));

async function handleInit(message: InitMessage): Promise<void> {
  const selectedModelUrl = message.prefetchedModel?.url;
  const physicalBatchMode = message.physicalBatchMode === 'logical' ? 'logical' : 'declared';
  const initKey = JSON.stringify({
    runtime: message.runtime ?? 'onnx',
    modelUrl: message.modelUrl,
    selectedModelUrl,
    deviceModelSelection: message.deviceModelSelection === true,
    ep: message.ep,
    cacheModel: message.cacheModel,
    allowUnverifiedModel: message.allowUnverifiedModel === true,
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
    dequantFold: message.dequantFold ?? null,
    ortWasmArtifact: message.ortWasmArtifact ?? null,
    ortWasmThreads: message.ortWasmThreads ?? null,
    physicalBatchMode,
    ortIoMode: message.ortIoMode ?? null,
    ortFreeDimensionOverrides: message.ortFreeDimensionOverrides ?? null,
  });
  if (evaluator && configured?.initKey === initKey) {
    const { initKey: _key, ...ready } = configured;
    post({ type: 'ready', id: message.id, ...ready, modelCache: `${configured.modelCache} · reused existing worker session` });
    return;
  }

  const evalCacheEntries = Math.max(0, Math.floor(message.evalCacheEntries ?? 0));
  const cacheLabel = evalCacheEntries > 0 ? ` · eval-cache ${evalCacheEntries}` : '';
  applyOrtExecutionProvider(message.ep);
  setOrtRuntimeDiagnosticOptionsForCurrentThread(message.ortDiagnostics ?? null);
  setOrtDequantFoldForCurrentThread(message.dequantFold ?? null);
  setRequestedOrtWasmArtifactForCurrentThread(message.ortWasmArtifact ?? null);
  setRequestedOrtWasmThreadsForCurrentThread(message.ortWasmThreads ?? null);

  let baseEvaluator: WorkerEvaluator;
  let nextEvaluator: WorkerEvaluator;
  let nextSearcher: Lc0PuctSearcher;
  let ready: ReadyPayload;
  const isOnnx = message.runtime !== 'whole-onnx-webgpu' && message.runtime !== 'hybrid';

  if (message.runtime === 'whole-onnx-webgpu') {
    if (!message.wholeModelManifestUrl) throw new Error('whole-model LC0 worker init requires manifest URL');
    baseEvaluator = await Lc0WholeOnnxWebgpuEvaluator.create({
      manifestUrl: message.wholeModelManifestUrl,
      batch: message.wholeModelBatch,
      fetchTensorCache: message.wholeModelTensorCache,
      logger: (line) => console.info('[lc0 whole-model worker]', line),
    });
    nextEvaluator = evalCacheEntries > 0 ? new CachedLc0Evaluator(baseEvaluator, { maxEntries: evalCacheEntries }) : baseEvaluator;
    nextSearcher = new Lc0PuctSearcher(nextEvaluator);
    ready = { backend: 'whole-onnx-webgpu', modelCache: `whole-model-webgpu${cacheLabel}` };
  } else if (message.runtime === 'hybrid') {
    if (!message.packUrl) throw new Error('hybrid LC0 worker init requires packUrl');
    if (!hybridEvaluatorFactory) throw new Error('hybrid LC0 runtime requires the dedicated hybrid search worker');
    baseEvaluator = hybridEvaluatorFactory({
      packUrl: message.packUrl,
      layers: message.layers,
      verifyShards: message.verifyShards,
      headBackend: message.headBackend,
      wgslBatchMode: message.wgslBatchMode,
      inputBackend: message.inputBackend,
      legalPriorsBackend: message.legalPriorsBackend,
      encoderKernelVariant: message.encoderKernelVariant,
    });
    nextEvaluator = evalCacheEntries > 0 ? new CachedLc0Evaluator(baseEvaluator, { maxEntries: evalCacheEntries }) : baseEvaluator;
    nextSearcher = new Lc0PuctSearcher(nextEvaluator);
    ready = {
      backend: message.headBackend === 'wgsl' ? 'lc0web-wgsl-encoder-wgsl-heads' : 'lc0web-wgsl-encoder-ort-heads',
      modelCache: `hybrid-pack-lazy${cacheLabel}`,
    };
  } else {
    // Throttle progress posts to ~2MB steps so a large net does not flood the
    // main thread with one message per network chunk.
    let lastReportedBytes = -Infinity;
    const initAbort = new AbortController();
    activeInit = initAbort;
    let prefetchedLoad = message.prefetchedModel?.load;
    message.prefetchedModel = undefined;
    const deps = prefetchedLoad && {
      ...DEFAULT_LC0_ORT_EVALUATOR_DEPS,
      load: (url: string, options: Parameters<typeof DEFAULT_LC0_ORT_EVALUATOR_DEPS.load>[1]) => {
        const load = url === selectedModelUrl ? prefetchedLoad : undefined;
        prefetchedLoad = undefined;
        return load ? Promise.resolve(load) : DEFAULT_LC0_ORT_EVALUATOR_DEPS.load(url, options);
      },
    };
    const created = await createLc0OnnxEvaluatorFromUrl(
      message.modelUrl,
      {
        deviceSelection: message.deviceModelSelection === true,
        selectedModelUrl,
        evaluator: {
          physicalBatchMode,
          ioMode: message.ortIoMode ?? defaultLc0OrtIoMode(message.deviceModelSelection === true),
          freeDimensionOverrides: message.ortFreeDimensionOverrides,
        },
        load: {
          cache: message.cacheModel,
          allowUnverified: message.allowUnverifiedModel,
          requestPersistentStorage: message.requestPersistentModelStorage,
          minimumFreeBytesAfterCache: message.minimumFreeBytesAfterModelCache,
          noDownload: message.noModelDownload,
          signal: initAbort.signal,
          onProgress: message.reportDownloadProgress
            ? (loadedBytes, totalBytes) => {
                if (loadedBytes - lastReportedBytes < 2_000_000 && loadedBytes !== totalBytes) return;
                lastReportedBytes = loadedBytes;
                post({ type: 'downloadProgress', id: message.id, loadedBytes, totalBytes });
              }
            : undefined,
        },
      },
      deps,
    ).finally(() => {
      if (activeInit === initAbort) activeInit = null;
    });
    const onnxEvaluator = created.evaluator;
    baseEvaluator = onnxEvaluator;
    try {
      nextEvaluator = evalCacheEntries > 0 ? new CachedLc0Evaluator(baseEvaluator, { maxEntries: evalCacheEntries }) : baseEvaluator;
      nextSearcher = new Lc0PuctSearcher(nextEvaluator);
      const diagnostics = await collectOrtRuntimeDiagnostics();
      ready = {
        backend: diagnostics.describe,
        modelCache: `${describeLc0ModelLoad(created.modelLoad)}${cacheLabel}`,
        ortWasm: diagnostics.wasm,
        ortFallback: diagnostics.fallback,
        ortRuntime: {
          graphOptimizationLevel: diagnostics.graphOptimizationLevel,
          ioMode: onnxEvaluator.ioMode,
          fixedBatchSize: onnxEvaluator.fixedBatchSize,
          dequantFoldEnabled: diagnostics.dequantFoldEnabled,
          dequantFold: diagnostics.dequantFold,
        },
        servedModelUrl: created.servedUrl,
        ortIoMode: onnxEvaluator.ioMode,
      };
    } catch (error) {
      await baseEvaluator.dispose?.();
      throw error;
    }
  }

  // Keep the working session if replacement creation or diagnostics fail.
  const previousEvaluator = evaluator;
  evaluator = nextEvaluator;
  searcher = nextSearcher;
  configured = { initKey, ...ready };
  if (isOnnx) {
    deviceLossMonitor.refresh();
  } else {
    deviceLossMonitor.stop();
  }
  await previousEvaluator?.dispose?.();
  post({ type: 'ready', id: message.id, ...ready });
}

async function handleSearch(message: SearchMessage): Promise<void> {
  if (!searcher) throw new Error('LC0 search worker is not initialized');
  if (takeSearchCancel(pendingSearchCancels, message.id)) {
    post({
      type: 'searchResult',
      id: message.id,
      result: { fen: '', visits: 0, value: 0, children: [], pv: [], cancelled: true, elapsedMs: 0 },
    });
    return;
  }
  const started = nowMs();
  const controller = new AbortController();
  activeSearches.set(message.id, controller);
  try {
    const { type: _type, id, input, reportProgress, ...rest } = message;
    const { search, ...summary } = await searcher.search(input, {
      ...rest,
      batchSize: rest.batchSize ?? 1,
      onProgress: reportProgress ? (progress) => post({ type: 'searchProgress', id, progress }) : undefined,
      signal: controller.signal,
      yieldEveryMs: 16,
    });
    const innerEvaluator = (evaluator as { inner?: WorkerEvaluator } | null)?.inner;
    post({
      type: 'searchResult',
      id,
      result: {
        ...summary,
        stats: search.stats,
        elapsedMs: nowMs() - started,
        cancelled: controller.signal.aborted,
        executionFootprint: evaluator?.executionFootprint?.() ?? innerEvaluator?.executionFootprint?.(),
        cacheFootprint: evaluator?.cacheFootprint?.() ?? innerEvaluator?.cacheFootprint?.(),
        gpuBufferAllocation: {
          ...gpuBufferTelemetry,
          byUsage: Object.fromEntries(Object.entries(gpuBufferTelemetry.byUsage).map(([usage, stats]) => [usage, { ...stats }])),
        },
      },
    });
  } catch (error) {
    if (!(error instanceof Error && error.name === 'AbortError')) throw error;
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

function cancelRequests(target?: number): void {
  if (target !== undefined) {
    const controller = activeSearches.get(target);
    if (controller) controller.abort();
    else noteSearchCancel(pendingSearchCancels, target);
    return;
  }
  for (const controller of activeSearches.values()) controller.abort();
  activeInit?.abort();
}

export function startSearchWorker(options: { createHybridEvaluator?: HybridEvaluatorFactory } = {}): void {
  hybridEvaluatorFactory = options.createHybridEvaluator;
  retiring = false;
  self.addEventListener('message', (event: MessageEvent<WorkerRequest>) => {
    const message = event.data;
    // Cancellation is synchronous and must not be wrapped in the error reporter,
    // so an abort never posts a spurious error for the cancel request itself.
    if (message.type === 'cancel') {
      cancelRequests(message.target);
      return;
    }
    // Seal admission immediately, but release only behind already admitted model work.
    if (retiring) return;
    if (message.type === 'dispose') {
      retiring = true;
      cancelRequests();
    }
    void operationQueue.run(async () => {
      try {
        // Work queued before dispose would only delay release (an init would start a
        // fresh download that the 1.5 s disposal timeout then waits on).
        if (retiring && message.type !== 'dispose') throw new Error('LC0 search worker is shutting down');
        if (message.type === 'dispose') {
          deviceLossMonitor.stop();
          const previous = evaluator;
          evaluator = null;
          searcher?.resetTree();
          searcher = null;
          configured = null;
          pendingSearchCancels.length = 0;
          await releaseOrtWorkerResources(async () => {
            await previous?.dispose?.();
          });
          post({ type: 'disposed', id: message.id });
        } else if (message.type === 'init') {
          await handleInit(message);
        } else if (message.type === 'resetSearch') {
          searcher?.resetTree();
          if (message.clearEvalCache) (evaluator as CachedLc0Evaluator | null)?.clearCache?.();
          post({ type: 'searchReset', id: message.id });
        } else if (!configured || !evaluator) {
          throw new Error('LC0 search worker missing model URL');
        } else if (message.type === 'evaluate') {
          post({ type: 'evaluationResult', id: message.id, result: await evaluator.evaluate(message.input) });
        } else if (message.type === 'evaluateBatch') {
          post({ type: 'evaluationBatchResult', id: message.id, result: await evaluator.evaluateBatch(message.inputs) });
        } else if (message.type === 'search') {
          await handleSearch(message);
        }
      } catch (error) {
        post({ type: 'error', id: message.id, error: error instanceof Error ? error.message : String(error) });
      }
    });
  });
}
