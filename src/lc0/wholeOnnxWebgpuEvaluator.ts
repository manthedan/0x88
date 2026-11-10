import * as v from 'valibot';
import type { BoardState } from '../chess/board.ts';
import { f16BitsToF32, f32ToF16Bits } from '../nn/float16.ts';
import { encodeLc0Classical112 } from './encoder112.ts';
import {
  currentBoardAndFen,
  LC0_DEFAULT_POLICY_TEMPERATURE,
  type Lc0Evaluation,
  type Lc0EvaluationProvider,
  type Lc0EvaluatorInput,
  legalPolicyPriors,
} from './onnxEvaluator.ts';
import { WholeOnnxRuntimeManifestSchema } from './wholeOnnxRuntimeManifestSchema.ts';

const STAGED_RUNTIME_PREFIX = '/runtimes/lc0-' + 'tvm' + 'js-webgpu/';
const DEFAULT_BUNDLE = 'tvm' + 'js.bundle.js';
const RUNTIME_GLOBAL = 'tvm' + 'js';
const LC0_POLICY_SIZE = 1858;
const LC0_INPUT_PLANES_SIZE = 112 * 8 * 8;
const EVAL_TIMING_KEYS = [
  'encodeMs',
  'inputConvertMs',
  'inputTensorAllocMs',
  'inputUploadMs',
  'setInputMs',
  'tvmInvokeMs',
  'outputHandleMs',
  'outputCopyEnqueueMs',
  'outputReadbackMs',
  'outputDecodeMs',
] as const;

interface RuntimeManifestModel {
  batch: number;
  wasm: string;
  bytes?: number;
  sha256?: string;
}

interface RuntimeManifest {
  runtime?: Record<string, string | undefined>;
  parameterStrategy?: { current?: string };
  tensorCache?: { directory?: string; manifest?: string };
  models: RuntimeManifestModel[];
  files?: Array<{ path: string; sha256?: string }>;
}

interface WholeModelTensor {
  shape: number[];
  dtype: string;
  device: { sync(): Promise<void> };
  copyFromRawBytes(bytes: Uint8Array): void;
  copyFrom(other: WholeModelTensor): void;
  toRawBytes(): Uint8Array;
  dispose(): void;
}

interface WholeModelModule {
  getFunction(name: string, queryOnly?: boolean): (...args: unknown[]) => unknown;
}

interface WholeModelVm {
  getInternalModule(): WholeModelModule;
}

interface WholeModelRuntime {
  beginScope(): void;
  endScope(): void;
  detachFromCurrentScope<T>(obj: T): T;
  dispose(): void;
  cpu(): unknown;
  webgpu(): unknown;
  initWebGPU(device: unknown): void;
  empty(shape: number[], dtype: string, device: unknown): WholeModelTensor;
  scalar(value: number, dtype: string): unknown;
  systemLib(): WholeModelModule;
  asyncLoadWebGPUPipelines(module: WholeModelModule): Promise<void>;
  createVirtualMachine(device: unknown): WholeModelVm;
  fetchTensorCache?(baseUrl: string, device: unknown, cacheScope?: string, cacheType?: string, signal?: AbortSignal): Promise<void>;
  tensorCacheGet?(name: string): WholeModelTensor | undefined;
  cacheMetadata?: unknown;
}

interface WholeModelRuntimeApi {
  instantiate(bytes: ArrayBuffer, wasi: unknown, logger?: (message: string) => void): Promise<WholeModelRuntime>;
  createPolyfillWASI(): unknown;
}

interface GpuAdapterLike {
  features: { has(feature: string): boolean };
  requestDevice(options?: { requiredFeatures?: string[] }): Promise<unknown>;
}

interface NavigatorWithGpu extends Navigator {
  gpu?: { requestAdapter(): Promise<GpuAdapterLike | null> };
}

interface SubmittedBatch {
  boards: BoardState[];
  fens: string[];
  preparedLegalMoves: Array<ReturnType<typeof currentBoardAndFen>['preparedLegalMoves']>;
  staged: Array<WholeModelTensor | null>;
  /** Tensors detached from the submit scope; released by releaseBatch once readback is decoded. */
  owned: WholeModelTensor[];
  device: { sync(): Promise<void> };
  timing: Record<string, number>;
}

export interface Lc0WholeOnnxWebgpuEvaluatorOptions {
  manifestUrl: string;
  batch?: number;
  fetchTensorCache?: boolean;
  logger?: (message: string) => void;
  signal?: AbortSignal;
}

function sumFinite(values: Array<number | undefined>): number {
  return values.reduce<number>((sum, value) => sum + (Number.isFinite(value) ? (value as number) : 0), 0);
}

function float32ToFloat16Bytes(values: Float32Array): Uint8Array {
  const out = new Uint16Array(values.length);
  for (let i = 0; i < values.length; i++) out[i] = f32ToF16Bits(values[i]);
  return new Uint8Array(out.buffer);
}

function f16BytesToFloat32Array(raw: Uint8Array): Float32Array {
  const view = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
  const out = new Float32Array(raw.byteLength / 2);
  for (let i = 0; i < out.length; i++) out[i] = f16BitsToF32(view.getUint16(i * 2, true));
  return out;
}

async function sha256Hex(bytes: ArrayBuffer): Promise<string> {
  const digest = await crypto.subtle.digest('SHA-256', bytes);
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, '0')).join('');
}

function checkedStagedRuntimeUrl(value: string, baseUrl: string | URL, label: string): URL {
  const url = new URL(value, baseUrl);
  if (url.origin !== location.origin) throw new Error(`${label} must be same-origin: ${url}`);
  if (!url.pathname.startsWith(STAGED_RUNTIME_PREFIX)) throw new Error(`${label} must stay under ${STAGED_RUNTIME_PREFIX}: ${url.pathname}`);
  return url;
}

async function fetchChecked(baseUrl: URL, rel: string, expected?: { bytes?: number; sha256?: string }, signal?: AbortSignal): Promise<ArrayBuffer> {
  signal?.throwIfAborted();
  const url = checkedStagedRuntimeUrl(rel, baseUrl, `artifact ${rel}`).toString();
  const response = await fetch(url, { signal });
  if (!response.ok) throw new Error(`Fetch failed ${response.status}: ${url}`);
  const bytes = await response.arrayBuffer();
  signal?.throwIfAborted();
  if (expected?.bytes != null && bytes.byteLength !== expected.bytes) throw new Error(`Size mismatch for ${rel}: ${bytes.byteLength} != ${expected.bytes}`);
  if (expected?.sha256) {
    const actual = await sha256Hex(bytes);
    signal?.throwIfAborted();
    if (actual !== expected.sha256) throw new Error(`SHA-256 mismatch for ${rel}: ${actual} != ${expected.sha256}`);
  }
  return bytes;
}

async function loadRuntimeBundle(bundleUrl: string, baseUrl: URL, expectedSha256: string, signal?: AbortSignal): Promise<WholeModelRuntimeApi> {
  signal?.throwIfAborted();
  const url = checkedStagedRuntimeUrl(bundleUrl, baseUrl, 'runtime bundle').toString();
  const globals = globalThis as Record<string, unknown>;
  const loadedSha256 = globals.__LC0_TVMJS_BUNDLE_SHA256__;
  if (globals[RUNTIME_GLOBAL]) {
    if (loadedSha256 === expectedSha256) return globals[RUNTIME_GLOBAL] as WholeModelRuntimeApi;
    throw new Error(`TVMJS runtime bundle mismatch: loaded ${String(loadedSha256 ?? 'unknown')} != expected ${expectedSha256}`);
  }
  await import(/* @vite-ignore */ url);
  // Record the integrity stamp before honouring cancellation: the import has
  // already installed the runtime global, and an unstamped global would make
  // every later create in this realm fail the mismatch check.
  globals.lc0WholeModelRuntimeBundleUrl = url;
  globals.__LC0_TVMJS_BUNDLE_SHA256__ = expectedSha256;
  const api = globals[RUNTIME_GLOBAL] as WholeModelRuntimeApi | undefined;
  if (!api) throw new Error(`runtime global missing after loading ${url}`);
  signal?.throwIfAborted();
  return api;
}

function tensorCacheUrl(manifest: RuntimeManifest, baseUrl: URL): string | undefined {
  const tensorCache = manifest.tensorCache;
  if (!tensorCache) return undefined;
  const directory = tensorCache.directory ?? (tensorCache.manifest ? tensorCache.manifest.replace(/[^/]*$/, '') : 'tensor-cache/');
  const path = String(directory).replace(/\/?$/, '/');
  return checkedStagedRuntimeUrl(path, baseUrl, 'tensor cache').toString();
}

/**
 * WebGPU buffer copies must be 4-byte multiples, and tvmjs reads a tensor back
 * at its exact byte size. The f16 policy [b,1858] always qualifies, but WDL
 * [b,3] (6b bytes) and MLH [b,1] (2b bytes) only do for even physical batches.
 */
function readbackAlignedBatch(batch: number): boolean {
  return (batch * 3 * 2) % 4 === 0 && (batch * 2) % 4 === 0;
}

let batchEvalSequence = 0;

export class Lc0WholeOnnxWebgpuEvaluator implements Lc0EvaluationProvider {
  readonly tensorCacheInfo?: { url: string; metadata: unknown };
  readonly startupTimings: Record<string, number>;
  private disposed = false;
  private readonly inFlight = new Set<Promise<unknown>>();
  // Explicit fields rather than constructor parameter properties so the module
  // loads under Node's strip-only TypeScript in unit tests.
  private readonly runtime: WholeModelRuntime;
  private readonly device: { destroy?: () => void } | null;
  private readonly vmMod: WholeModelModule;
  private readonly physicalBatchSize: number;
  private readonly detachedParams: WholeModelTensor[];

  private constructor(
    runtime: WholeModelRuntime,
    device: { destroy?: () => void } | null,
    vmMod: WholeModelModule,
    physicalBatchSize: number,
    detachedParams: WholeModelTensor[],
    startupTimings: Record<string, number>,
    tensorCacheInfo?: { url: string; metadata: unknown },
  ) {
    this.runtime = runtime;
    this.device = device;
    this.vmMod = vmMod;
    this.physicalBatchSize = physicalBatchSize;
    this.detachedParams = detachedParams;
    this.startupTimings = startupTimings;
    this.tensorCacheInfo = tensorCacheInfo;
  }

  static async create(options: Lc0WholeOnnxWebgpuEvaluatorOptions): Promise<Lc0WholeOnnxWebgpuEvaluator> {
    options.signal?.throwIfAborted();
    const startupTimings: Record<string, number> = {};
    const timed = async <T>(name: string, fn: () => Promise<T>): Promise<T> => {
      const started = performance.now();
      try {
        return await fn();
      } finally {
        startupTimings[name] = (startupTimings[name] ?? 0) + performance.now() - started;
      }
    };
    const timedSync = <T>(name: string, fn: () => T): T => {
      const started = performance.now();
      try {
        return fn();
      } finally {
        startupTimings[name] = (startupTimings[name] ?? 0) + performance.now() - started;
      }
    };

    const base = checkedStagedRuntimeUrl(options.manifestUrl, location.href, 'manifest');
    const manifestResponse = await timed('manifestFetchMs', () => fetch(base, { signal: options.signal }));
    if (!manifestResponse.ok) throw new Error(`Manifest fetch failed: ${manifestResponse.status}`);
    const manifest = await timed('manifestJsonMs', async () => v.parse(WholeOnnxRuntimeManifestSchema, await manifestResponse.json()));
    options.signal?.throwIfAborted();
    const requestedBatch = Math.max(1, Math.floor(options.batch ?? 8));
    const model = manifest.models.find((item) => item.batch === requestedBatch);
    if (!model) throw new Error(`Batch ${requestedBatch} not in manifest`);
    if (!readbackAlignedBatch(model.batch)) {
      const usable = manifest.models.map((item) => item.batch).filter(readbackAlignedBatch);
      throw new Error(
        `Whole-model batch ${model.batch} is unsupported: its f16 WDL/MLH outputs (${model.batch * 6}/${model.batch * 2} bytes) cannot be read back ` +
          `because WebGPU copies need 4-byte multiples; use an even batch${usable.length ? ` (${usable.join(', ')})` : ''}`,
      );
    }
    const bundle = manifest.runtime?.['tvm' + 'jsBundle'] ?? DEFAULT_BUNDLE;
    const bundleFile = manifest.files?.find((entry) => entry.path === bundle);
    if (!bundleFile?.sha256) throw new Error(`Runtime bundle integrity metadata missing for ${bundle}`);
    const api = await timed('runtimeBundleLoadMs', () => loadRuntimeBundle(bundle, base, bundleFile.sha256!, options.signal));
    const nav = navigator as NavigatorWithGpu;
    if (!nav.gpu) throw new Error('navigator.gpu missing');
    const adapter = await timed('requestAdapterMs', () => nav.gpu!.requestAdapter());
    options.signal?.throwIfAborted();
    if (!adapter) throw new Error('WebGPU adapter unavailable');
    const requiredFeatures = ['shader-f16'];
    for (const feature of requiredFeatures) if (!adapter.features.has(feature)) throw new Error(`WebGPU adapter missing required feature: ${feature}`);
    const device = await timed('requestDeviceMs', () => adapter.requestDevice({ requiredFeatures }));
    let ownedRuntime: WholeModelRuntime | undefined;
    try {
      options.signal?.throwIfAborted();
      const wasmBytes = await timed('wasmFetchVerifyMs', () => fetchChecked(base, model.wasm, model, options.signal));
      const runtime = await timed('runtimeInstantiateMs', () => api.instantiate(wasmBytes, api.createPolyfillWASI(), options.logger));
      options.signal?.throwIfAborted();
      ownedRuntime = runtime;
      timedSync('runtimeInitWebGpuMs', () => runtime.initWebGPU(device));
      runtime.beginScope();
      try {
        const sys = timedSync('systemLibMs', () => runtime.systemLib());
        await timed('webgpuPipelinePrebuildMs', () => runtime.asyncLoadWebGPUPipelines(sys));
        options.signal?.throwIfAborted();
        const vm = timedSync('createVirtualMachineMs', () => runtime.createVirtualMachine(runtime.webgpu()));
        const vmMod = vm.getInternalModule();
        const cacheUrl = tensorCacheUrl(manifest, base);
        const needsDetachedParams = manifest.parameterStrategy?.current === 'detached-tensor-cache';
        let tensorCacheInfo: { url: string; metadata: unknown } | undefined;
        if (options.fetchTensorCache || needsDetachedParams) {
          if (!cacheUrl) throw new Error('tensor cache requested but manifest.tensorCache is missing');
          if (!runtime.fetchTensorCache) throw new Error('runtime tensor-cache fetch API is missing');
          await timed('tensorCacheFetchMs', () => runtime.fetchTensorCache!(cacheUrl, runtime.webgpu(), 'tvmjs', 'cache', options.signal));
          options.signal?.throwIfAborted();
          tensorCacheInfo = { url: cacheUrl, metadata: runtime.cacheMetadata ?? null };
        }
        const detachedParams: WholeModelTensor[] = [];
        if (needsDetachedParams) {
          if (!runtime.tensorCacheGet) throw new Error('detached params require tensor-cache lookup API');
          for (let i = 0; ; i++) {
            const tensor = runtime.tensorCacheGet(`param_${i}`);
            if (!tensor) break;
            detachedParams.push(tensor);
          }
          if (!detachedParams.length) throw new Error('detached tensor-cache manifest but no param_N tensors loaded');
        }
        return new Lc0WholeOnnxWebgpuEvaluator(
          runtime,
          device as { destroy?: () => void } | null,
          vmMod,
          model.batch,
          detachedParams,
          startupTimings,
          tensorCacheInfo,
        );
      } catch (error) {
        runtime.endScope();
        throw error;
      }
    } catch (error) {
      // Nothing else owns this runtime or device until create() returns; release
      // them on any failure, including cancellation, so a superseded load frees
      // wasm and GPU memory.
      try {
        ownedRuntime?.dispose();
      } catch {
        /* best effort */
      }
      (device as { destroy?: () => void } | null)?.destroy?.();
      throw error;
    }
  }

  async evaluate(input: Lc0EvaluatorInput): Promise<Lc0Evaluation> {
    return (await this.runBatchEvaluation([input]))[0];
  }

  async evaluateBatch(inputs: Lc0EvaluatorInput[]): Promise<Lc0Evaluation[]> {
    const out: Lc0Evaluation[] = [];
    for (let i = 0; i < inputs.length; i += this.physicalBatchSize) out.push(...(await this.runBatchEvaluation(inputs.slice(i, i + this.physicalBatchSize))));
    return out;
  }

  async evaluateBatchSequence(batches: Lc0EvaluatorInput[][]): Promise<Lc0Evaluation[][]> {
    const chunks: Array<{ batchIndex: number; inputs: Lc0EvaluatorInput[] }> = [];
    batches.forEach((inputs, batchIndex) => {
      for (let i = 0; i < inputs.length; i += this.physicalBatchSize) chunks.push({ batchIndex, inputs: inputs.slice(i, i + this.physicalBatchSize) });
    });
    const chunkRows = await this.runBatchSequence(chunks.map((chunk) => chunk.inputs));
    const out = batches.map((): Lc0Evaluation[] => []);
    for (let i = 0; i < chunkRows.length; i++) out[chunks[i].batchIndex].push(...chunkRows[i]);
    return out;
  }

  /**
   * Releases the runtime once every in-flight evaluation has settled: the
   * create()-time scope (VM, pipelines, detached params), the wasm instance and
   * the GPUDevice this evaluator owns. Evaluations started afterwards reject.
   */
  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    const release = () => {
      try {
        this.runtime.endScope();
      } catch {
        /* already unwound */
      }
      try {
        this.runtime.dispose();
      } catch {
        /* best effort */
      }
      try {
        this.device?.destroy?.();
      } catch {
        /* best effort */
      }
    };
    void Promise.allSettled([...this.inFlight]).then(release);
  }

  private track<T>(run: () => Promise<T>): Promise<T> {
    if (this.disposed) return Promise.reject(new Error('Whole-model evaluator is disposed'));
    const promise = run();
    this.inFlight.add(promise);
    const settle = () => this.inFlight.delete(promise);
    promise.then(settle, settle);
    return promise;
  }

  private releaseBatch(submitted: SubmittedBatch): void {
    for (const tensor of submitted.owned) tensor.dispose();
    submitted.owned.length = 0;
  }

  /**
   * Encodes and enqueues one physical batch. Every TVM object it allocates
   * (packed funcs, shape tuples, tensors) lands in a scope opened and closed
   * synchronously here, so concurrent evaluations never interleave the
   * runtime's scope stack. The tensors the pending readback still writes into
   * or reads from are detached and handed back in `owned`.
   */
  private submitBatch(inputs: Lc0EvaluatorInput[]): SubmittedBatch {
    if (!inputs.length) throw new Error('cannot submit empty batch');
    this.runtime.beginScope();
    try {
      return this.submitBatchInScope(inputs);
    } finally {
      this.runtime.endScope();
    }
  }

  private submitBatchInScope(inputs: Lc0EvaluatorInput[]): SubmittedBatch {
    const setInput = this.vmMod.getFunction('set_input');
    const invokeStateful = this.vmMod.getFunction('invoke_stateful');
    const getOutput = this.vmMod.getFunction('get_output');
    const shape = [this.physicalBatchSize, 112, 8, 8];
    const encoded = new Float32Array(this.physicalBatchSize * LC0_INPUT_PLANES_SIZE);
    const boards: BoardState[] = [];
    const fens: string[] = [];
    const preparedLegalMoves: Array<ReturnType<typeof currentBoardAndFen>['preparedLegalMoves']> = [];
    let phaseStarted = performance.now();
    for (let i = 0; i < inputs.length; i++) {
      const current = currentBoardAndFen(inputs[i]);
      boards.push(current.board);
      fens.push(current.fen);
      preparedLegalMoves.push(current.preparedLegalMoves);
      encoded.set(encodeLc0Classical112(inputs[i], { historyFill: 'fen_only' }).planes, i * LC0_INPUT_PLANES_SIZE);
    }
    for (let i = inputs.length; i < this.physicalBatchSize; i++) {
      encoded.copyWithin(i * LC0_INPUT_PLANES_SIZE, (inputs.length - 1) * LC0_INPUT_PLANES_SIZE, inputs.length * LC0_INPUT_PLANES_SIZE);
    }
    const encodeMs = performance.now() - phaseStarted;
    phaseStarted = performance.now();
    const inputBytes = float32ToFloat16Bytes(encoded);
    const inputConvertMs = performance.now() - phaseStarted;
    phaseStarted = performance.now();
    const input = this.runtime.empty(shape, 'float16', this.runtime.webgpu());
    const inputTensorAllocMs = performance.now() - phaseStarted;
    phaseStarted = performance.now();
    input.copyFromRawBytes(inputBytes);
    const inputUploadMs = performance.now() - phaseStarted;
    phaseStarted = performance.now();
    if (this.detachedParams.length) setInput('main', input, ...this.detachedParams);
    else setInput('main', input);
    const setInputMs = performance.now() - phaseStarted;
    phaseStarted = performance.now();
    invokeStateful('main');
    const tvmInvokeMs = performance.now() - phaseStarted;
    phaseStarted = performance.now();
    const outs = [0, 1, 2].map((i) => getOutput('main', this.runtime.scalar(i, 'int32')) as WholeModelTensor);
    const outputHandleMs = performance.now() - phaseStarted;
    phaseStarted = performance.now();
    const staged = outs.map((out) => {
      const elementCount = out.shape.reduce((a, b) => a * b, 1);
      const rawBytes = elementCount * (out.dtype === 'float16' ? 2 : out.dtype === 'float32' ? 4 : 1);
      if (rawBytes % 4 !== 0) return null;
      const cpu = this.runtime.empty(out.shape, out.dtype, this.runtime.cpu());
      cpu.copyFrom(out);
      return cpu;
    });
    const outputCopyEnqueueMs = performance.now() - phaseStarted;
    // The CPU staging tensors receive the GPU readback asynchronously (at
    // device.sync()), so they must outlive this scope; the input and output
    // handles are kept alongside them so nothing the queued work touches is
    // released before finishBatch has decoded the results.
    const owned = [input, ...outs, ...staged.filter((tensor): tensor is WholeModelTensor => tensor !== null)];
    for (const tensor of owned) this.runtime.detachFromCurrentScope(tensor);
    return {
      boards,
      fens,
      preparedLegalMoves,
      staged,
      owned,
      device: outs[0].device,
      timing: { encodeMs, inputConvertMs, inputTensorAllocMs, inputUploadMs, setInputMs, tvmInvokeMs, outputHandleMs, outputCopyEnqueueMs },
    };
  }

  private finishBatch(submitted: SubmittedBatch, timingExtra: Record<string, number | string>): Lc0Evaluation[] {
    const { boards, fens, preparedLegalMoves, staged, timing } = submitted;
    const phaseStarted = performance.now();
    const policyRaw = staged[0]?.toRawBytes();
    if (!policyRaw) throw new Error('Whole-model policy output readback unavailable');
    const policy = f16BytesToFloat32Array(policyRaw);
    if (!staged[1]) throw new Error('Whole-model WDL output readback unavailable');
    const wdl = f16BytesToFloat32Array(staged[1].toRawBytes());
    const mlh = staged[2] ? f16BytesToFloat32Array(staged[2].toRawBytes()) : undefined;
    const outputDecodeMs = performance.now() - phaseStarted;
    const phases: Record<string, number | string> = { ...timing, outputDecodeMs, ...timingExtra };
    const sharedTiming = {
      backend: 'whole-onnx-webgpu',
      tvmBatchEvalId: ++batchEvalSequence,
      tvmBatchEvalMs: sumFinite(EVAL_TIMING_KEYS.map((key) => (typeof phases[key] === 'number' ? (phases[key] as number) : undefined))),
      ...phases,
      physicalBatchSize: this.physicalBatchSize,
      logicalBatchSize: boards.length,
    };
    return boards.map((board, i) => {
      const rawWdl = Array.from(wdl.subarray(i * 3, i * 3 + 3));
      const wdlSlice: [number, number, number] = [rawWdl[0] ?? NaN, rawWdl[1] ?? NaN, rawWdl[2] ?? NaN];
      const logits = policy.subarray(i * LC0_POLICY_SIZE, (i + 1) * LC0_POLICY_SIZE);
      const legalStarted = performance.now();
      const legalPriors = legalPolicyPriors(board, logits, LC0_DEFAULT_POLICY_TEMPERATURE, preparedLegalMoves[i]);
      const legalPriorsMs = performance.now() - legalStarted;
      return {
        fen: fens[i],
        wdl: wdlSlice,
        q: wdlSlice[0] - wdlSlice[2],
        mlh: mlh ? mlh[i] : NaN,
        legalPriors,
        bestMove: legalPriors[0]?.uci,
        timing: { ...sharedTiming, legalPriorsMs, batchPosition: i },
      };
    });
  }

  private async runBatchEvaluation(inputs: Lc0EvaluatorInput[]): Promise<Lc0Evaluation[]> {
    if (!inputs.length) return [];
    return this.track(async () => {
      const submitted = this.submitBatch(inputs);
      try {
        const syncStarted = performance.now();
        await submitted.device.sync();
        const outputReadbackMs = performance.now() - syncStarted;
        return this.finishBatch(submitted, { timingScope: 'whole-model-batch-invocation', outputReadbackMs });
      } finally {
        this.releaseBatch(submitted);
      }
    });
  }

  private runBatchSequence(batchesInputs: Lc0EvaluatorInput[][]): Promise<Lc0Evaluation[][]> {
    return this.track(async () => {
      const submitted: Array<SubmittedBatch | null> = [];
      try {
        for (const inputs of batchesInputs) submitted.push(inputs.length ? this.submitBatch(inputs) : null);
        const device = submitted.find((batch): batch is SubmittedBatch => !!batch)?.device;
        let syncMs = 0;
        if (device) {
          const syncStarted = performance.now();
          await device.sync();
          syncMs = performance.now() - syncStarted;
        }
        const inFlight = submitted.filter(Boolean).length || 1;
        return submitted.map((batch) =>
          batch
            ? this.finishBatch(batch, {
                timingScope: 'whole-model-pipelined-batch',
                outputReadbackMs: syncMs / inFlight,
                pipelineFlushBatches: inFlight,
                pipelineFlushSyncMs: syncMs,
              })
            : [],
        );
      } finally {
        for (const batch of submitted) if (batch) this.releaseBatch(batch);
      }
    });
  }
}
