import { boardToFen, parseFen, type BoardState } from '../chess/board.ts';
import { legalMoves } from '../chess/movegen.ts';
import { moveToUci, type Move } from '../chess/moveCodec.ts';
import { encodeLc0Classical112, type Lc0PositionHistoryInput } from './encoder112.ts';
import { LC0_DEFAULT_POLICY_TEMPERATURE, type Lc0Evaluation, type Lc0EvaluationProvider, type Lc0EvaluatorInput } from './onnxEvaluator.ts';
import { LC0_MIRROR_TRANSFORM, uciToLc0PolicyIndex } from './policyMap.ts';

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
}

interface WholeModelTensor {
  shape: number[];
  dtype: string;
  device: { sync(): Promise<void> };
  copyFromRawBytes(bytes: Uint8Array): void;
  copyFrom(other: WholeModelTensor): void;
  toRawBytes(): Uint8Array;
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
  cpu(): unknown;
  webgpu(): unknown;
  initWebGPU(device: unknown): void;
  empty(shape: number[], dtype: string, device: unknown): WholeModelTensor;
  scalar(value: number, dtype: string): unknown;
  systemLib(): WholeModelModule;
  asyncLoadWebGPUPipelines(module: WholeModelModule): Promise<void>;
  createVirtualMachine(device: unknown): WholeModelVm;
  fetchTensorCache?(baseUrl: string, device: unknown): Promise<void>;
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
  staged: Array<WholeModelTensor | null>;
  device: { sync(): Promise<void> };
  timing: Record<string, number>;
}

export interface Lc0WholeOnnxWebgpuEvaluatorOptions {
  manifestUrl: string;
  batch?: number;
  fetchTensorCache?: boolean;
  logger?: (message: string) => void;
}

function sumFinite(values: Array<number | undefined>): number {
  return values.reduce<number>((sum, value) => sum + (Number.isFinite(value) ? value as number : 0), 0);
}

function f32ToF16Bits(value: number): number {
  if (Number.isNaN(value)) return 0x7e00;
  if (value === Infinity) return 0x7c00;
  if (value === -Infinity) return 0xfc00;
  const sign = value < 0 || Object.is(value, -0) ? 0x8000 : 0;
  const abs = Math.abs(value);
  if (abs === 0) return sign;
  if (abs >= 65504) return sign | 0x7bff;
  if (abs < 2 ** -24) return sign;
  if (abs < 2 ** -14) return sign | Math.round(abs / 2 ** -24);
  const exponent = Math.floor(Math.log2(abs));
  const fraction = abs / 2 ** exponent - 1;
  let halfExponent = exponent + 15;
  let halfFraction = Math.round(fraction * 1024);
  if (halfFraction === 1024) { halfExponent += 1; halfFraction = 0; }
  if (halfExponent >= 31) return sign | 0x7bff;
  return sign | (halfExponent << 10) | (halfFraction & 0x03ff);
}

function f16BitsToF32(bits: number): number {
  const sign = (bits & 0x8000) ? -1 : 1;
  const exponent = (bits >>> 10) & 0x1f;
  const fraction = bits & 0x03ff;
  if (exponent === 0) return sign * (fraction === 0 ? 0 : 2 ** -14 * (fraction / 1024));
  if (exponent === 0x1f) return fraction === 0 ? sign * Infinity : NaN;
  return sign * 2 ** (exponent - 15) * (1 + fraction / 1024);
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

async function fetchChecked(baseUrl: URL, rel: string, expected?: { bytes?: number; sha256?: string }): Promise<ArrayBuffer> {
  const url = checkedStagedRuntimeUrl(rel, baseUrl, `artifact ${rel}`).toString();
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Fetch failed ${response.status}: ${url}`);
  const bytes = await response.arrayBuffer();
  if (expected?.bytes != null && bytes.byteLength !== expected.bytes) throw new Error(`Size mismatch for ${rel}: ${bytes.byteLength} != ${expected.bytes}`);
  if (expected?.sha256) {
    const actual = await sha256Hex(bytes);
    if (actual !== expected.sha256) throw new Error(`SHA-256 mismatch for ${rel}: ${actual} != ${expected.sha256}`);
  }
  return bytes;
}

async function loadRuntimeBundle(bundleUrl: string, baseUrl: URL): Promise<WholeModelRuntimeApi> {
  const url = checkedStagedRuntimeUrl(bundleUrl, baseUrl, 'runtime bundle').toString();
  const globals = globalThis as Record<string, unknown>;
  if (globals[RUNTIME_GLOBAL] && globals.lc0WholeModelRuntimeBundleUrl === url) return globals[RUNTIME_GLOBAL] as WholeModelRuntimeApi;
  await import(/* @vite-ignore */ url);
  globals.lc0WholeModelRuntimeBundleUrl = url;
  const api = globals[RUNTIME_GLOBAL] as WholeModelRuntimeApi | undefined;
  if (!api) throw new Error(`runtime global missing after loading ${url}`);
  return api;
}

function tensorCacheUrl(manifest: RuntimeManifest, baseUrl: URL): string | undefined {
  const tensorCache = manifest.tensorCache;
  if (!tensorCache) return undefined;
  const directory = tensorCache.directory ?? (tensorCache.manifest ? tensorCache.manifest.replace(/[^/]*$/, '') : 'tensor-cache/');
  const path = String(directory).replace(/\/?$/, '/');
  return checkedStagedRuntimeUrl(path, baseUrl, 'tensor cache').toString();
}

function fileOf(square: number): number { return square % 8; }

function isStandardCastlingMove(board: BoardState, move: Move): boolean {
  const piece = board.squares[move.from];
  return piece?.[1] === 'k' && Math.abs(fileOf(move.to) - fileOf(move.from)) === 2;
}

function legalPolicyPriors(board: BoardState, logits: Float32Array): Lc0Evaluation['legalPriors'] {
  const moveTransform = board.turn === 'b' ? LC0_MIRROR_TRANSFORM : 0;
  const legal = legalMoves(board).map((move) => {
    const uci = moveToUci(move);
    const index = uciToLc0PolicyIndex(uci, moveTransform, { standardCastling: isStandardCastlingMove(board, move) });
    if (index === undefined) throw new Error(`No LC0 policy index for legal move ${uci}`);
    return { uci, index, logit: Number(logits[index]) / LC0_DEFAULT_POLICY_TEMPERATURE };
  });
  if (!legal.length) return [];
  const max = Math.max(...legal.map((entry) => entry.logit));
  const sum = legal.reduce((acc, entry) => acc + Math.exp(entry.logit - max), 0);
  return legal.map((entry) => ({ ...entry, prior: Math.exp(entry.logit - max) / sum })).sort((a, b) => b.prior - a.prior);
}

function currentBoardAndFen(input: Lc0EvaluatorInput): { board: BoardState; fen: string } {
  if (typeof input === 'object' && input !== null && 'positions' in input) {
    const history = input as Lc0PositionHistoryInput;
    if (!history.positions.length) throw new Error('history input requires at least one position');
    const last = history.positions[history.positions.length - 1];
    const board = typeof last === 'string' ? parseFen(last) : last;
    return { board, fen: boardToFen(board) };
  }
  const board = typeof input === 'string' ? parseFen(input) : input;
  return { board, fen: boardToFen(board) };
}

function batchTimingFromRowTiming(timing: unknown): Record<string, unknown> | undefined {
  if (!timing || typeof timing !== 'object') return undefined;
  const { legalPriorsMs: _legalPriorsMs, batchPosition: _batchPosition, ...batchTiming } = timing as Record<string, unknown>;
  return batchTiming;
}

let batchEvalSequence = 0;

export class Lc0WholeOnnxWebgpuEvaluator implements Lc0EvaluationProvider {
  readonly evaluationTimings: Record<string, unknown>[] = [];
  readonly positionTimings: unknown[] = [];
  readonly tensorCacheInfo?: { url: string; metadata: unknown };
  readonly startupTimings: Record<string, number>;

  private constructor(
    private readonly runtime: WholeModelRuntime,
    private readonly vmMod: WholeModelModule,
    private readonly physicalBatchSize: number,
    private readonly detachedParams: WholeModelTensor[],
    startupTimings: Record<string, number>,
    tensorCacheInfo?: { url: string; metadata: unknown },
  ) {
    this.startupTimings = startupTimings;
    this.tensorCacheInfo = tensorCacheInfo;
  }

  static async create(options: Lc0WholeOnnxWebgpuEvaluatorOptions): Promise<Lc0WholeOnnxWebgpuEvaluator> {
    const startupTimings: Record<string, number> = {};
    const timed = async <T>(name: string, fn: () => Promise<T>): Promise<T> => {
      const started = performance.now();
      try { return await fn(); }
      finally { startupTimings[name] = (startupTimings[name] ?? 0) + performance.now() - started; }
    };
    const timedSync = <T>(name: string, fn: () => T): T => {
      const started = performance.now();
      try { return fn(); }
      finally { startupTimings[name] = (startupTimings[name] ?? 0) + performance.now() - started; }
    };

    const base = checkedStagedRuntimeUrl(options.manifestUrl, location.href, 'manifest');
    const manifestResponse = await timed('manifestFetchMs', () => fetch(base));
    if (!manifestResponse.ok) throw new Error(`Manifest fetch failed: ${manifestResponse.status}`);
    const manifest = await timed('manifestJsonMs', () => manifestResponse.json() as Promise<RuntimeManifest>);
    const requestedBatch = Math.max(1, Math.floor(options.batch ?? 8));
    const model = manifest.models.find((item) => item.batch === requestedBatch);
    if (!model) throw new Error(`Batch ${requestedBatch} not in manifest`);
    const bundle = manifest.runtime?.['tvm' + 'jsBundle'] ?? DEFAULT_BUNDLE;
    const api = await timed('runtimeBundleLoadMs', () => loadRuntimeBundle(bundle, base));
    const nav = navigator as NavigatorWithGpu;
    if (!nav.gpu) throw new Error('navigator.gpu missing');
    const adapter = await timed('requestAdapterMs', () => nav.gpu!.requestAdapter());
    if (!adapter) throw new Error('WebGPU adapter unavailable');
    const requiredFeatures = ['shader-f16'];
    for (const feature of requiredFeatures) if (!adapter.features.has(feature)) throw new Error(`WebGPU adapter missing required feature: ${feature}`);
    const device = await timed('requestDeviceMs', () => adapter.requestDevice({ requiredFeatures }));
    const wasmBytes = await timed('wasmFetchVerifyMs', () => fetchChecked(base, model.wasm, model));
    const runtime = await timed('runtimeInstantiateMs', () => api.instantiate(wasmBytes, api.createPolyfillWASI(), options.logger));
    timedSync('runtimeInitWebGpuMs', () => runtime.initWebGPU(device));
    runtime.beginScope();
    try {
      const sys = timedSync('systemLibMs', () => runtime.systemLib());
      await timed('webgpuPipelinePrebuildMs', () => runtime.asyncLoadWebGPUPipelines(sys));
      const vm = timedSync('createVirtualMachineMs', () => runtime.createVirtualMachine(runtime.webgpu()));
      const vmMod = vm.getInternalModule();
      const cacheUrl = tensorCacheUrl(manifest, base);
      const needsDetachedParams = manifest.parameterStrategy?.current === 'detached-tensor-cache';
      let tensorCacheInfo: { url: string; metadata: unknown } | undefined;
      if (options.fetchTensorCache || needsDetachedParams) {
        if (!cacheUrl) throw new Error('tensor cache requested but manifest.tensorCache is missing');
        if (!runtime.fetchTensorCache) throw new Error('runtime tensor-cache fetch API is missing');
        await timed('tensorCacheFetchMs', () => runtime.fetchTensorCache!(cacheUrl, runtime.webgpu()));
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
      return new Lc0WholeOnnxWebgpuEvaluator(runtime, vmMod, model.batch, detachedParams, startupTimings, tensorCacheInfo);
    } catch (error) {
      runtime.endScope();
      throw error;
    }
  }

  async evaluate(input: Lc0EvaluatorInput): Promise<Lc0Evaluation> {
    return (await this.remember(await this.runBatchEvaluation([input])))[0];
  }

  async evaluateBatch(inputs: Lc0EvaluatorInput[]): Promise<Lc0Evaluation[]> {
    const out: Lc0Evaluation[] = [];
    for (let i = 0; i < inputs.length; i += this.physicalBatchSize) out.push(...await this.remember(await this.runBatchEvaluation(inputs.slice(i, i + this.physicalBatchSize))));
    return out;
  }

  async evaluateBatchSequence(batches: Lc0EvaluatorInput[][]): Promise<Lc0Evaluation[][]> {
    const chunks: Array<{ batchIndex: number; inputs: Lc0EvaluatorInput[] }> = [];
    batches.forEach((inputs, batchIndex) => {
      for (let i = 0; i < inputs.length; i += this.physicalBatchSize) chunks.push({ batchIndex, inputs: inputs.slice(i, i + this.physicalBatchSize) });
    });
    const chunkRows = await this.runBatchSequence(chunks.map((chunk) => chunk.inputs));
    const out = batches.map((): Lc0Evaluation[] => []);
    for (let i = 0; i < chunkRows.length; i++) out[chunks[i].batchIndex].push(...await this.remember(chunkRows[i]));
    return out;
  }

  dispose(): void {
    this.runtime.endScope();
  }

  private remember(rows: Lc0Evaluation[]): Lc0Evaluation[] {
    const batchTiming = batchTimingFromRowTiming(rows[0]?.timing);
    if (batchTiming) this.evaluationTimings.push(batchTiming);
    for (const row of rows) this.positionTimings.push(row.timing);
    return rows;
  }

  private submitBatch(inputs: Lc0EvaluatorInput[]): SubmittedBatch {
    if (!inputs.length) throw new Error('cannot submit empty batch');
    const setInput = this.vmMod.getFunction('set_input');
    const invokeStateful = this.vmMod.getFunction('invoke_stateful');
    const getOutput = this.vmMod.getFunction('get_output');
    const shape = [this.physicalBatchSize, 112, 8, 8];
    const encoded = new Float32Array(this.physicalBatchSize * LC0_INPUT_PLANES_SIZE);
    const boards: BoardState[] = [];
    const fens: string[] = [];
    let phaseStarted = performance.now();
    for (let i = 0; i < inputs.length; i++) {
      const { board, fen } = currentBoardAndFen(inputs[i]);
      boards.push(board);
      fens.push(fen);
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
    return { boards, fens, staged, device: outs[0].device, timing: { encodeMs, inputConvertMs, inputTensorAllocMs, inputUploadMs, setInputMs, tvmInvokeMs, outputHandleMs, outputCopyEnqueueMs } };
  }

  private finishBatch(submitted: SubmittedBatch, timingExtra: Record<string, number | string>): Lc0Evaluation[] {
    const { boards, fens, staged, timing } = submitted;
    const phaseStarted = performance.now();
    const policyRaw = staged[0]?.toRawBytes();
    if (!policyRaw) throw new Error('Whole-model policy output readback unavailable');
    const policy = f16BytesToFloat32Array(policyRaw);
    const wdl = staged[1] ? f16BytesToFloat32Array(staged[1].toRawBytes()) : undefined;
    const mlh = staged[2] ? f16BytesToFloat32Array(staged[2].toRawBytes()) : undefined;
    const outputDecodeMs = performance.now() - phaseStarted;
    const phases: Record<string, number | string> = { ...timing, outputDecodeMs, ...timingExtra };
    const sharedTiming = {
      backend: 'whole-onnx-webgpu',
      tvmBatchEvalId: ++batchEvalSequence,
      tvmBatchEvalMs: sumFinite(EVAL_TIMING_KEYS.map((key) => typeof phases[key] === 'number' ? phases[key] as number : undefined)),
      ...phases,
      physicalBatchSize: this.physicalBatchSize,
      logicalBatchSize: boards.length,
    };
    return boards.map((board, i) => {
      const rawWdl = wdl ? Array.from(wdl.subarray(i * 3, i * 3 + 3)) : [];
      const wdlSlice: [number, number, number] = [rawWdl[0] ?? NaN, rawWdl[1] ?? NaN, rawWdl[2] ?? NaN];
      const logits = policy.subarray(i * LC0_POLICY_SIZE, (i + 1) * LC0_POLICY_SIZE);
      const legalStarted = performance.now();
      const legalPriors = legalPolicyPriors(board, logits);
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
    const submitted = this.submitBatch(inputs);
    const syncStarted = performance.now();
    await submitted.device.sync();
    const outputReadbackMs = performance.now() - syncStarted;
    return this.finishBatch(submitted, { timingScope: 'whole-model-batch-invocation', outputReadbackMs });
  }

  private async runBatchSequence(batchesInputs: Lc0EvaluatorInput[][]): Promise<Lc0Evaluation[][]> {
    const submitted = batchesInputs.map((inputs) => inputs.length ? this.submitBatch(inputs) : null);
    const device = submitted.find((batch): batch is SubmittedBatch => !!batch)?.device;
    let syncMs = 0;
    if (device) {
      const syncStarted = performance.now();
      await device.sync();
      syncMs = performance.now() - syncStarted;
    }
    const inFlight = submitted.filter(Boolean).length || 1;
    return submitted.map((batch) => batch ? this.finishBatch(batch, {
      timingScope: 'whole-model-pipelined-batch',
      outputReadbackMs: syncMs / inFlight,
      pipelineFlushBatches: inFlight,
      pipelineFlushSyncMs: syncMs,
    }) : []);
  }
}
