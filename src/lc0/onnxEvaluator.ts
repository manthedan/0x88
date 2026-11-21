import { f16BitsToF32 } from '../nn/float16.ts';
import { OrtGpuIoBinding, type OrtGpuIoOutputs, type OrtGpuIoTensorSpec, ortGraphCaptureSessionOptions } from '../nn/ortGpuIo.ts';
import * as ort from '../nn/ortRuntime.ts';
import { encodeLc0PlanesInto, type Lc0EncoderInput, type Lc0HistoryFill } from './encoder112.ts';
import { trackEngineRetirement } from './engineRetirement.ts';
import {
  CachedLc0Evaluator,
  currentBoardAndFen,
  LC0_DEFAULT_POLICY_TEMPERATURE,
  LC0_ONNX_INPUT_PLANES,
  LC0_ONNX_OUTPUT_MLH,
  LC0_ONNX_OUTPUT_POLICY,
  LC0_ONNX_OUTPUT_WDL,
  type Lc0Evaluation,
  type Lc0EvaluationCacheFootprint,
  type Lc0EvaluationCacheMetrics,
  type Lc0EvaluationCacheOptions,
  type Lc0EvaluationProvider,
  type Lc0EvaluatorInput,
  type Lc0LegalPrior,
  type Lc0PreparedEvaluatorInput,
  type Lc0PreparedLegalMove,
  legalPolicyPriors,
  prepareLc0EvaluatorInput,
} from './lc0EvaluatorBase.ts';
import { untilAborted } from './modelCache.ts';
import { applyOnnxFreeDimensionOverrides, type OnnxModelIo, readOnnxModelIo } from './onnxModelIo.ts';
import { isOnnxModelParseError, OrtSessionFallbackError, ortErrorMessage } from './onnxModelRecovery.ts';
import { type OrtPhysicalBatchMode, resolveOrtPhysicalBatchSize } from './ortPhysicalBatch.ts';

export {
  CachedLc0Evaluator,
  currentBoardAndFen,
  LC0_DEFAULT_POLICY_TEMPERATURE,
  LC0_ONNX_INPUT_PLANES,
  LC0_ONNX_OUTPUT_MLH,
  LC0_ONNX_OUTPUT_POLICY,
  LC0_ONNX_OUTPUT_WDL,
  type Lc0Evaluation,
  type Lc0EvaluationCacheFootprint,
  type Lc0EvaluationCacheMetrics,
  type Lc0EvaluationCacheOptions,
  type Lc0EvaluationProvider,
  type Lc0EvaluatorInput,
  type Lc0LegalPrior,
  type Lc0PreparedEvaluatorInput,
  type Lc0PreparedLegalMove,
  legalPolicyPriors,
  prepareLc0EvaluatorInput,
};

const LC0_POLICY_SIZE = 1858;
const LC0_WDL_SIZE = 3;
const LC0_MLH_SIZE = 1;
const LC0_INPUT_PLANES_SIZE = 112 * 8 * 8;

export interface Lc0OnnxEvaluatorOptions {
  /** Initialization owner only; admitted inference is drained by dispose(). */
  signal?: AbortSignal;
  policyTemperature?: number;
  historyFill?: Lc0HistoryFill;
  /**
   * Lab/query override. Omitted and 'declared' keep today's serial batch-1
   * path for a dynamic leading dimension. 'logical' submits that whole
   * logical batch as one session.run. Fixed batch-N graphs are unaffected.
   */
  physicalBatchMode?: OrtPhysicalBatchMode;
  /** Passed to createOrtSession: the model for its WebGPU -> WASM fallback attempt. */
  wasmFallbackModel?: () => Promise<string | Uint8Array | ArrayBuffer>;
  /** Verified loader owns one-shot metadata-cache recovery before provider fallback. */
  rejectModelParseErrors?: boolean | (() => boolean);
  /** Verified lazy reload of the same bytes, so folded-session creation need not pin the original. */
  reloadModel?: () => Promise<string | Uint8Array | ArrayBuffer>;
  /**
   * How invokes move data (docs/ort_invoke_overhead_2026-11.md). 'cpu' (default):
   * CPU input and output tensors. 'gpu': persistent GPU input/output buffers and
   * one staging readback per invoke. 'gpu-capture': 'gpu' plus ORT graph
   * capture. Both GPU modes need a fixed-shape model on WebGPU; anything else
   * (WASM placement, dynamic batch, session failure) falls back to 'cpu'.
   */
  ioMode?: Lc0OrtIoMode;
  /** Pin named dimensions only for GPU IO/capture. CPU/WASM fallback retains the shipped dynamic graph. */
  freeDimensionOverrides?: Readonly<Record<string, number>>;
}

export type Lc0OrtIoMode = 'cpu' | 'gpu' | 'gpu-capture';

/**
 * The IO mode a search-worker init gets when it names none. The device-selected
 * path (Small and big nets) requests graph-captured GPU IO: fewer CPU-side
 * WebGPU calls per invoke, same outputs (docs/ort_invoke_overhead_2026-11.md).
 * Fixed-shape WebGPU models use it; explicit per-net dimension overrides can
 * make a dynamic model eligible (LQO pins batch=1). Unresolved dynamic shapes
 * use CPU IO without a probe session.
 * Without device selection the default is CPU IO. Explicit ioMode wins.
 */
export function defaultLc0OrtIoMode(deviceModelSelection: boolean): Lc0OrtIoMode {
  return deviceModelSelection ? 'gpu-capture' : 'cpu';
}

function outputValue(data: ArrayLike<number>, index: number): number {
  const value = data[index];
  return data instanceof Uint16Array && value !== undefined ? f16BitsToF32(value) : Number(value);
}

async function tensorDataTimed(
  outputs: Awaited<ReturnType<ort.InferenceSession['run']>>,
  name: string,
  timingKey: string,
  timing: Record<string, number>,
): Promise<ArrayLike<number>> {
  const tensor = outputs[name];
  if (!tensor) throw new Error(`LC0 ONNX output ${name} missing`);
  const started = ort.tinyLeelaNowMs();
  try {
    return (await ort.ortTensorData(tensor)) as ArrayLike<number>;
  } finally {
    timing[`${timingKey}GetDataMs`] = ort.tinyLeelaNowMs() - started;
  }
}

function sessionInputMetadata(session: ort.InferenceSession): { type?: string; shape?: unknown[] } | undefined {
  return (session.inputMetadata?.find?.((entry: { name?: string }) => entry.name === LC0_ONNX_INPUT_PLANES) ?? session.inputMetadata?.[0]) as
    | { type?: string; shape?: unknown[] }
    | undefined;
}

function gpuIoOutput(outputs: OrtGpuIoOutputs, name: string): ArrayLike<number> {
  const data = outputs[name];
  if (!data) throw new Error(`LC0 ONNX output ${name} missing`);
  return data;
}

/**
 * Tensor specs for GPU-resident IO, or the reason the session cannot use it.
 * Needs one float input and float outputs, all with fixed (numeric) shapes.
 * Exported for tests.
 */
export function lc0GpuIoSpecs(
  session: OnnxModelIo,
  { inferOutputBatch = false }: { inferOutputBatch?: boolean } = {},
): { input: OrtGpuIoTensorSpec; outputs: OrtGpuIoTensorSpec[] } | { reason: string } {
  const spec = (meta: unknown, name: string): OrtGpuIoTensorSpec | string => {
    const m = meta as { isTensor?: boolean; type?: string; shape?: readonly unknown[] } | undefined;
    if (!m) return `${name}: no metadata`;
    if (m.type !== 'float16' && m.type !== 'float32') return `${name}: type ${m.type}`;
    if (!m.shape?.length || !m.shape.every((d) => typeof d === 'number' && Number.isInteger(d) && d > 0)) return `${name}: dynamic shape`;
    return { name, type: m.type, dims: m.shape as number[] };
  };
  if (session.inputNames.length !== 1 || session.inputNames[0] !== LC0_ONNX_INPUT_PLANES) return { reason: `inputs ${session.inputNames.join(',')}` };
  const input = spec(
    session.inputMetadata.find((m) => m.name === LC0_ONNX_INPUT_PLANES),
    LC0_ONNX_INPUT_PLANES,
  );
  if (typeof input === 'string') return { reason: input };
  const outputs: OrtGpuIoTensorSpec[] = [];
  for (const name of session.outputNames) {
    const meta = session.outputMetadata.find((m) => m.name === name);
    // Small declares policy as ['batch', 1858] even with fixed batch-1 input;
    // ORT infers that head's batch. Permit that known LC0 convention only in
    // preflight; the post-create check still requires actual numeric shapes.
    const inferBatch = inferOutputBatch && meta?.shape?.[0] === 'batch' && [LC0_ONNX_OUTPUT_POLICY, LC0_ONNX_OUTPUT_WDL, LC0_ONNX_OUTPUT_MLH].includes(name);
    const out = spec(inferBatch ? { ...meta, shape: [input.dims[0], ...meta!.shape!.slice(1)] } : meta, name);
    if (typeof out === 'string') return { reason: out };
    outputs.push(out);
  }
  if (!outputs.some((o) => o.name === LC0_ONNX_OUTPUT_POLICY) || !outputs.some((o) => o.name === LC0_ONNX_OUTPUT_WDL))
    return { reason: 'missing policy/wdl output' };
  return { input, outputs };
}

type Lc0FixedGpuIo = { input: OrtGpuIoTensorSpec; outputs: OrtGpuIoTensorSpec[] };

function gpuIoModelSpecs(modelPath: string | Uint8Array | ArrayBuffer, mode: string, options: Lc0OnnxEvaluatorOptions): Lc0FixedGpuIo | null {
  // No unverified shape-probe download for URL-only callers.
  if (typeof modelPath === 'string') return null;
  try {
    const declared = lc0GpuIoSpecs(
      applyOnnxFreeDimensionOverrides(readOnnxModelIo(modelPath instanceof Uint8Array ? modelPath : new Uint8Array(modelPath)), options.freeDimensionOverrides),
      { inferOutputBatch: true },
    );
    if ('reason' in declared) throw new Error(`model not eligible: ${declared.reason}`);
    return declared;
  } catch (error) {
    const rejectParse = typeof options.rejectModelParseErrors === 'function' ? options.rejectModelParseErrors() : options.rejectModelParseErrors;
    if (rejectParse && isOnnxModelParseError(error)) throw error;
    console.warn(`Lc0: GPU-resident ORT IO (${mode}) unavailable, using CPU IO. ${error instanceof Error ? error.message : String(error)}`);
    return null;
  }
}

/**
 * Try GPU-resident IO. Preserve the failure for the caller's ordinary session
 * fallback, in case that also fails.
 */
async function createLc0GpuIoSession(
  takeModel: () => string | Uint8Array | ArrayBuffer,
  mode: 'gpu' | 'gpu-capture',
  options: Lc0OnnxEvaluatorOptions,
  declared: Lc0FixedGpuIo,
): Promise<{ session: ort.InferenceSession; binding: OrtGpuIoBinding } | { error: unknown }> {
  let modelPath = takeModel();
  let session: ort.InferenceSession | undefined;
  let binding: OrtGpuIoBinding | undefined;
  try {
    options.signal?.throwIfAborted();
    const creating = ort.createOrtSession(modelPath, {
      signal: options.signal,
      webgpuOnly: true,
      rejectModelParseErrors: options.rejectModelParseErrors,
      webgpuSessionOptions: {
        preferredOutputLocation: 'gpu-buffer',
        ...(mode === 'gpu-capture' ? ortGraphCaptureSessionOptions() : {}),
        ...(options.freeDimensionOverrides ? { freeDimensionOverrides: { ...options.freeDimensionOverrides } } : {}),
      },
    });
    modelPath = undefined as never;
    session = await creating;
    const specs = lc0GpuIoSpecs(applyOnnxFreeDimensionOverrides(session, options.freeDimensionOverrides));
    if ('reason' in specs) throw new Error(`model not eligible: ${specs.reason}`);
    const same = (a: OrtGpuIoTensorSpec | undefined, b: OrtGpuIoTensorSpec) =>
      a?.name === b.name && a.type === b.type && a.dims.length === b.dims.length && a.dims.every((dim, i) => dim === b.dims[i]);
    if (
      !same(specs.input, declared.input) ||
      declared.outputs.some(
        (output) =>
          !same(
            specs.outputs.find((actual) => actual.name === output.name),
            output,
          ),
      )
    ) {
      throw new Error('GPU IO metadata differs from the declared/overridden shape');
    }
    const device = (await untilAborted(Promise.resolve((ort.env.webgpu as unknown as { device?: unknown }).device), options.signal)) as
      | ConstructorParameters<typeof OrtGpuIoBinding>[1]
      | undefined;
    options.signal?.throwIfAborted();
    if (!device) throw new Error('ORT exposed no WebGPU device');
    // Captured sessions keep ORT's own output buffers bound across replays; preallocated
    // fetches leak a native OrtValue per output per replay there (OrtGpuIoOutputMode).
    binding = new OrtGpuIoBinding(session, device, ort.Tensor as never, specs.input, specs.outputs, {
      outputMode: mode === 'gpu-capture' ? 'session' : 'preallocated',
    });
    // One zero-input invoke: ORT allocates (and, with capture, records) on the first
    // run, so a rejection surfaces here, where the CPU-IO fallback can still run.
    const elements = specs.input.dims.reduce((a, b) => a * b, 1);
    await binding.run(specs.input.type === 'float16' ? new Uint16Array(elements) : new Float32Array(elements));
    options.signal?.throwIfAborted();
    return { session, binding };
  } catch (error) {
    const owned = session;
    // Tracked: a cancelled owner's replacement still waits for this release.
    if (owned) await trackEngineRetirement(Promise.resolve(binding?.dispose()).finally(() => ort.releaseOrtSession(owned)));
    options.signal?.throwIfAborted();
    const rejectParse = typeof options.rejectModelParseErrors === 'function' ? options.rejectModelParseErrors() : options.rejectModelParseErrors;
    if (rejectParse && isOnnxModelParseError(error)) throw error;
    console.warn(`Lc0: GPU-resident ORT IO (${mode}) unavailable, using CPU IO. ${ortErrorMessage(error)}`);
    return { error };
  }
}

export class Lc0OnnxEvaluator implements Lc0EvaluationProvider {
  readonly policyTemperature: number;
  readonly historyFill: Lc0HistoryFill;
  private readonly physicalBatchMode: OrtPhysicalBatchMode;
  private readonly session: ort.InferenceSession;
  private readonly gpuIo: OrtGpuIoBinding | null;
  /** IO mode actually in use (a requested GPU mode can fall back to 'cpu'). */
  readonly ioMode: Lc0OrtIoMode;
  private idleF16Planes?: Uint16Array<ArrayBuffer>;
  private disposed = false;
  private closePromise: Promise<void> | null = null;
  private admitted = new Set<Promise<unknown>>();

  constructor(session: ort.InferenceSession, options: Lc0OnnxEvaluatorOptions = {}, gpuIo: { binding: OrtGpuIoBinding; mode: Lc0OrtIoMode } | null = null) {
    this.session = session;
    this.gpuIo = gpuIo?.binding ?? null;
    this.ioMode = gpuIo?.mode ?? 'cpu';
    this.policyTemperature = options.policyTemperature ?? LC0_DEFAULT_POLICY_TEMPERATURE;
    this.historyFill = options.historyFill ?? 'fen_only';
    this.physicalBatchMode = options.physicalBatchMode === 'logical' ? 'logical' : 'declared';
  }

  static create(modelPath: string | Uint8Array | ArrayBuffer, options: Lc0OnnxEvaluatorOptions = {}): Promise<Lc0OnnxEvaluator> {
    return Lc0OnnxEvaluator.createFromSource(() => {
      const input = modelPath;
      modelPath = undefined as never;
      return input;
    }, options);
  }

  private static async createFromSource(takeModel: () => string | Uint8Array | ArrayBuffer, options: Lc0OnnxEvaluatorOptions): Promise<Lc0OnnxEvaluator> {
    options.signal?.throwIfAborted();
    let modelPath = takeModel();
    let gpuIoFailure: { error: unknown } | undefined;
    const declared = options.ioMode === 'gpu' || options.ioMode === 'gpu-capture' ? gpuIoModelSpecs(modelPath, options.ioMode, options) : null;
    if ((options.ioMode === 'gpu' || options.ioMode === 'gpu-capture') && declared) {
      const placement = await untilAborted(ort.resolveOrtSessionPlacement(), options.signal);
      options.signal?.throwIfAborted();
      if (placement.providers[0] === 'webgpu') {
        const gpu = await createLc0GpuIoSession(() => modelPath, options.ioMode, options, declared);
        if ('session' in gpu) return new Lc0OnnxEvaluator(gpu.session, options, { binding: gpu.binding, mode: options.ioMode });
        gpuIoFailure = gpu;
      }
    }
    try {
      const creating = ort.createOrtSession(modelPath, {
        signal: options.signal,
        wasmFallbackModel: options.wasmFallbackModel,
        reloadModel: options.reloadModel,
        rejectModelParseErrors: options.rejectModelParseErrors,
      });
      modelPath = undefined as never;
      return new Lc0OnnxEvaluator(await creating, options);
    } catch (error) {
      if (!gpuIoFailure) throw error;
      options.signal?.throwIfAborted();
      throw new OrtSessionFallbackError(ortErrorMessage(gpuIoFailure.error), error);
    }
  }

  dispose(): Promise<void> {
    if (!this.closePromise) {
      this.disposed = true;
      this.closePromise = (async () => {
        await Promise.allSettled(this.admitted);
        this.idleF16Planes = undefined;
        try {
          await ort.releaseOrtSession(this.session);
        } finally {
          await this.gpuIo?.dispose();
        }
      })();
    }
    return this.closePromise;
  }

  /** Effective fixed physical IO batch; capture bindings may resolve symbolic ORT metadata. */
  get fixedBatchSize(): number | undefined {
    const first = this.gpuIo?.input.dims[0] ?? sessionInputMetadata(this.session)?.shape?.[0];
    return typeof first === 'number' && first > 0 ? first : undefined;
  }

  async evaluate(boardOrFen: Lc0EvaluatorInput): Promise<Lc0Evaluation> {
    return (await this.evaluateBatch([boardOrFen]))[0];
  }

  // Admit whole logical calls so later physical batches can finish during close.
  private admit<T>(work: () => Promise<T>): Promise<T> {
    if (this.disposed) throw new Error('LC0 ONNX evaluator has been disposed');
    const pending = Promise.resolve().then(work);
    this.admitted.add(pending);
    const settled = () => {
      this.admitted.delete(pending);
    };
    void pending.then(settled, settled);
    return pending;
  }

  private async runPhysicalBatch(inputs: Lc0EvaluatorInput[], physicalBatchSize: number): Promise<Lc0Evaluation[]> {
    if (!inputs.length) return [];
    const totalStarted = ort.tinyLeelaNowMs();
    const elements = physicalBatchSize * LC0_INPUT_PLANES_SIZE;
    const type = this.gpuIo?.input.type ?? sessionInputMetadata(this.session)?.type;
    if (type !== 'float16') return this.runWithPlanes(inputs, physicalBatchSize, new Float32Array(elements), totalStarted);
    const planes = this.idleF16Planes?.length === elements ? this.idleF16Planes : new Uint16Array(elements);
    this.idleF16Planes = undefined;
    try {
      return await this.runWithPlanes(inputs, physicalBatchSize, planes, totalStarted);
    } finally {
      this.idleF16Planes ??= planes;
    }
  }

  private async runWithPlanes(
    inputs: Lc0EvaluatorInput[],
    physicalBatchSize: number,
    encodedPlanes: Float32Array<ArrayBuffer> | Uint16Array<ArrayBuffer>,
    totalStarted: number,
  ): Promise<Lc0Evaluation[]> {
    const encodeStarted = ort.tinyLeelaNowMs();
    const currents = inputs.map((input, i) => {
      encodeLc0PlanesInto(input as Lc0EncoderInput, encodedPlanes, i * LC0_INPUT_PLANES_SIZE, this.historyFill);
      return currentBoardAndFen(input);
    });
    // Fixed batch-N artifacts require a full physical batch. Pad by copying the
    // last encoded position rather than re-encoding it for every unused slot;
    // callers only receive the real `inputs.length` outputs below.
    for (let i = inputs.length; i < physicalBatchSize; i++) {
      encodedPlanes.copyWithin(i * LC0_INPUT_PLANES_SIZE, (inputs.length - 1) * LC0_INPUT_PLANES_SIZE, inputs.length * LC0_INPUT_PLANES_SIZE);
    }
    const encodeMs = ort.tinyLeelaNowMs() - encodeStarted;
    const webgpuBefore = ort.getOrtWebGpuDiagnosticsSnapshot();
    const ortRunStarted = ort.tinyLeelaNowMs();
    const downloadTiming: Record<string, number> = {};
    let policy: ArrayLike<number>;
    let wdlRaw: ArrayLike<number>;
    let mlhRaw: ArrayLike<number> | undefined;
    let ortRunMs: number;
    let allGetDataMs: number;
    if (this.gpuIo) {
      const gpuOutputs = await this.gpuIo.run(encodedPlanes);
      ortRunMs = ort.tinyLeelaNowMs() - ortRunStarted;
      const allGetDataStarted = ort.tinyLeelaNowMs();
      policy = gpuIoOutput(gpuOutputs, LC0_ONNX_OUTPUT_POLICY);
      wdlRaw = gpuIoOutput(gpuOutputs, LC0_ONNX_OUTPUT_WDL);
      mlhRaw = gpuOutputs[LC0_ONNX_OUTPUT_MLH];
      allGetDataMs = ort.tinyLeelaNowMs() - allGetDataStarted;
    } else {
      const inputTensor = encodedPlanes instanceof Uint16Array
        ? new ort.Tensor('float16', encodedPlanes, [physicalBatchSize, 112, 8, 8])
        : new ort.Tensor('float32', encodedPlanes, [physicalBatchSize, 112, 8, 8]);
      const outputs = await this.session.run({ [LC0_ONNX_INPUT_PLANES]: inputTensor });
      ortRunMs = ort.tinyLeelaNowMs() - ortRunStarted;
      const allGetDataStarted = ort.tinyLeelaNowMs();
      // Older/smaller nets (e.g. Maia) have no moves-left head; treat MLH as 0.
      try {
        [policy, wdlRaw, mlhRaw] = await Promise.all([
          tensorDataTimed(outputs, LC0_ONNX_OUTPUT_POLICY, 'ortPolicy', downloadTiming),
          tensorDataTimed(outputs, LC0_ONNX_OUTPUT_WDL, 'ortWdl', downloadTiming),
          LC0_ONNX_OUTPUT_MLH in outputs ? tensorDataTimed(outputs, LC0_ONNX_OUTPUT_MLH, 'ortMlh', downloadTiming) : Promise.resolve(undefined),
        ]);
      } finally {
        ort.releaseOrtGpuOutputs(outputs);
      }
      allGetDataMs = ort.tinyLeelaNowMs() - allGetDataStarted;
    }
    await ort.waitForOrtWebGpuDiagnostics();
    const postprocessStarted = ort.tinyLeelaNowMs();
    const webgpuDelta = ort.subtractOrtWebGpuDiagnosticsSnapshot(ort.getOrtWebGpuDiagnosticsSnapshot(), webgpuBefore);
    const baseTiming: Record<string, number | string | null | undefined> = {
      backend: 'ort-onnx',
      ortIoMode: this.ioMode,
      inputBuildMs: encodeMs,
      ortRunMs,
      sessionRunMs: ortRunMs,
      ortAllGetDataMs: allGetDataMs,
      readbackSyncedMs: allGetDataMs,
      readbackBytes: physicalBatchSize * (LC0_POLICY_SIZE + LC0_WDL_SIZE + LC0_MLH_SIZE) * 4,
      readbackMapCount: webgpuDelta.api.mapAsyncCount,
      ortKernelCount: webgpuDelta.profiling.eventCount,
      ortKernelProfileStatus: webgpuDelta.profiling.status,
      ortKernelProfileReason: webgpuDelta.profiling.reason,
      ortKernelGpuMs: webgpuDelta.profiling.kernelGpuMsTotal,
      webgpuSubmitCount: webgpuDelta.api.submitCount,
      webgpuSubmittedCommandBufferCount: webgpuDelta.api.submittedCommandBufferCount,
      webgpuMapAsyncCount: webgpuDelta.api.mapAsyncCount,
      webgpuMapAsyncMs: webgpuDelta.api.mapAsyncMsTotal,
      webgpuCopyBufferToBufferCount: webgpuDelta.api.copyBufferToBufferCount,
      webgpuCopyBufferToBufferBytes: webgpuDelta.api.copyBufferToBufferBytes,
      webgpuMapReadBufferCount: webgpuDelta.api.mapReadBufferCount,
      webgpuMapReadBufferBytes: webgpuDelta.api.mapReadBufferBytes,
      webgpuCreateBufferCount: webgpuDelta.api.createBufferCount,
      webgpuCreateBufferBytes: webgpuDelta.api.createBufferBytes,
      webgpuComputePipelineCreateCount: webgpuDelta.api.computePipelineCreateCount,
      webgpuComputePipelineCreateAsyncCount: webgpuDelta.api.computePipelineCreateAsyncCount,
      batchPosition: 0,
      physicalBatchSize: inputs.length,
      ortPhysicalBatchSize: physicalBatchSize,
      ...downloadTiming,
    };
    const results: Lc0Evaluation[] = currents.map((current, i) => {
      const legalPriorsStarted = ort.tinyLeelaNowMs();
      const wdlOffset = i * LC0_WDL_SIZE;
      const wdl: [number, number, number] = [outputValue(wdlRaw, wdlOffset), outputValue(wdlRaw, wdlOffset + 1), outputValue(wdlRaw, wdlOffset + 2)];
      const legalPriors = legalPolicyPriors(current.board, policy, this.policyTemperature, current.preparedLegalMoves, i * LC0_POLICY_SIZE);
      const legalPriorsMs = ort.tinyLeelaNowMs() - legalPriorsStarted;
      return {
        fen: current.fen,
        wdl,
        q: wdl[0] - wdl[2],
        mlh: mlhRaw === undefined ? 0 : outputValue(mlhRaw, i * LC0_MLH_SIZE),
        legalPriors,
        bestMove: legalPriors[0]?.uci,
        timing: { ...baseTiming, batchPosition: i, legalPriorsMs },
      };
    });
    const postprocessMs = ort.tinyLeelaNowMs() - postprocessStarted;
    const totalEvalMs = ort.tinyLeelaNowMs() - totalStarted;
    for (const result of results) {
      const timing = result.timing as Record<string, unknown>;
      timing.postprocessMs = postprocessMs;
      timing.totalEvalMs = totalEvalMs;
    }
    return results;
  }

  async evaluateBatch(inputs: Lc0EvaluatorInput[]): Promise<Lc0Evaluation[]> {
    return this.admit(() => this.evaluateBatchNow(inputs));
  }

  private async evaluateBatchNow(inputs: Lc0EvaluatorInput[]): Promise<Lc0Evaluation[]> {
    if (!inputs.length) return [];
    const declared = this.gpuIo?.input.dims[0] ?? sessionInputMetadata(this.session)?.shape?.[0];
    const out: Lc0Evaluation[] = [];
    let offset = 0;
    while (offset < inputs.length) {
      const logicalCount = inputs.length - offset;
      const physicalBatchSize = resolveOrtPhysicalBatchSize(declared, logicalCount, this.physicalBatchMode);
      const count = Math.min(logicalCount, physicalBatchSize);
      out.push(...(await this.runPhysicalBatch(inputs.slice(offset, offset + count), physicalBatchSize)));
      offset += count;
    }
    return out;
  }

  async evaluateBatchSequence(batches: Lc0EvaluatorInput[][]): Promise<Lc0Evaluation[][]> {
    return this.admit(async () => {
      const out: Lc0Evaluation[][] = [];
      for (const batch of batches) {
        out.push(await this.evaluateBatchNow(batch));
      }
      return out;
    });
  }
}
