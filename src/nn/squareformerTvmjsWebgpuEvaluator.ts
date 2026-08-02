import * as v from 'valibot';
import type { BoardState } from '../chess/board.ts';
import { isStmWhiteRankflip, normalizedMoveToOriginal, normalizePositionForStmWhite } from '../chess/boardNormalization.ts';
import { moveToActionId } from '../chess/moveCodec.ts';
import { moveToSquareformerPolicyIndex } from '../chess/moveEncodings.ts';
import { legalMoves } from '../chess/movegen.ts';
import { isTrustedExecutableAssetUrl } from '../lc0/assetUrls.ts';
import type { Evaluation, EvaluationContext, Evaluator } from './evaluator.ts';
import { softmax } from './numerics.ts';
import {
  type SquareFormerMeta,
  squareformerCompactInput,
  THREATGRAPH_SQUARE_SUMMARY_V1_FEATURES,
  threatgraphSquareSummaryV1,
} from './squareformerEvaluator.ts';
import { TvmjsManifestSchema } from './tvmjsManifestSchema.ts';

type TvmTensor = {
  shape: readonly number[];
  dtype: string;
  device: { sync(): Promise<void> };
  copyFromRawBytes(bytes: Uint8Array): void;
  copyFrom(tensor: TvmTensor): void;
  toRawBytes(): Uint8Array;
};

type TvmPackedFunction = (...args: unknown[]) => unknown;

type TvmRuntime = {
  initWebGPU(device: unknown): void;
  webgpu(): unknown;
  cpu(): unknown;
  beginScope(): void;
  endScope(): void;
  systemLib(): unknown;
  asyncLoadWebGPUPipelines(module: unknown): Promise<void>;
  createVirtualMachine(device: unknown): { getInternalModule(): { getFunction(name: string): TvmPackedFunction } };
  empty(shape: readonly number[], dtype: string, device: unknown): TvmTensor;
  scalar(value: number, dtype: string): unknown;
  dispose(): void;
};

type TvmjsApi = {
  instantiate(wasmBytes: ArrayBuffer, wasi: unknown, logger?: (message: string) => void): Promise<TvmRuntime>;
  createPolyfillWASI(): unknown;
};

type GpuAdapter = {
  features: { has(feature: string): boolean };
  limits: Record<string, number>;
  requestDevice(descriptor?: unknown): Promise<GpuDevice>;
};

type GpuDevice = {
  lost?: Promise<{ message?: string }>;
  addEventListener?(type: string, listener: (event: { error?: { message?: string } }) => void): void;
  destroy(): void;
};

type TvmjsManifest = {
  schema: string;
  modelFamily: string;
  dtype: string;
  target: string;
  requiredFeatures?: string[];
  runtime?: { tvmjsBundle?: string; tvmjsRuntimeWasm?: string };
  models?: Array<{ batch: number; wasm: string; bytes?: number; sha256?: string }>;
  files?: Array<{ path: string; bytes?: number; sha256?: string }>;
};

type PreparedRow = {
  board: BoardState;
  historyFens: string[];
  legalMoves?: ReturnType<typeof legalMoves>;
  flipped: boolean;
  attackSummaryChannelMask?: ArrayLike<number>;
};

const EXPECTED_SCHEMA = 'lc0_browser.lc0_tvmjs_webgpu_bundle.v1';
const EXPECTED_MODEL_FAMILY = 'bt4-soap-rem-c19000-final';
const OUTPUT_POLICY = 0;
const OUTPUT_WDL = 1;
const BUNDLE_SHA_GLOBAL = '__LC0_TVMJS_BUNDLE_SHA256__';
let bundleLoad: { sha256: string; promise: Promise<TvmjsApi> } | undefined;

function currentTvmjsApi(): TvmjsApi | undefined {
  return (globalThis as typeof globalThis & { tvmjs?: TvmjsApi }).tvmjs;
}

function loadedTvmjsBundleSha(): string | undefined {
  return (globalThis as typeof globalThis & { [BUNDLE_SHA_GLOBAL]?: string })[BUNDLE_SHA_GLOBAL];
}

function sha256Integrity(sha256: string): string {
  const bytes = new Uint8Array(sha256.match(/../g)!.map((byte) => Number.parseInt(byte, 16)));
  return `sha256-${btoa(String.fromCharCode(...bytes))}`;
}

async function sha256Hex(bytes: ArrayBuffer): Promise<string> {
  if (!globalThis.crypto?.subtle) throw new Error('WebCrypto SHA-256 is unavailable');
  const digest = await globalThis.crypto.subtle.digest('SHA-256', bytes);
  return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, '0')).join('');
}

async function fetchVerified(url: URL, expected?: { bytes?: number; sha256?: string }): Promise<ArrayBuffer> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Fetch failed ${response.status}: ${url}`);
  const bytes = await response.arrayBuffer();
  if (expected?.bytes !== undefined && bytes.byteLength !== expected.bytes) {
    throw new Error(`Artifact size mismatch for ${url}: ${bytes.byteLength} != ${expected.bytes}`);
  }
  if (expected?.sha256) {
    const actual = await sha256Hex(bytes);
    if (actual !== expected.sha256) throw new Error(`Artifact SHA-256 mismatch for ${url}: ${actual} != ${expected.sha256}`);
  }
  return bytes;
}

async function loadTvmjsBundle(url: URL, expected: { sha256: string }): Promise<TvmjsApi> {
  if (!isTrustedExecutableAssetUrl(url.toString())) throw new Error(`Untrusted TVMJS runtime URL: ${url}`);
  const existing = currentTvmjsApi();
  const loadedSha = loadedTvmjsBundleSha();
  if (existing) {
    if (loadedSha !== expected.sha256) {
      throw new Error(`TVMJS runtime bundle mismatch: loaded ${loadedSha ?? 'unknown'} != expected ${expected.sha256}`);
    }
    return existing;
  }
  if (bundleLoad && bundleLoad.sha256 !== expected.sha256) {
    throw new Error(`TVMJS runtime bundle load conflict: ${bundleLoad.sha256} != ${expected.sha256}`);
  }
  if (!bundleLoad) {
    const promise = new Promise<TvmjsApi>((resolve, reject) => {
      const script = document.createElement('script');
      script.src = url.toString();
      script.async = true;
      script.crossOrigin = 'anonymous';
      script.integrity = sha256Integrity(expected.sha256);
      script.onload = () => {
        const api = currentTvmjsApi();
        if (api) {
          (globalThis as typeof globalThis & { [BUNDLE_SHA_GLOBAL]?: string })[BUNDLE_SHA_GLOBAL] = expected.sha256;
          resolve(api);
        } else reject(new Error('TVMJS runtime global missing after bundle load'));
      };
      script.onerror = () => reject(new Error(`TVMJS bundle load failed: ${url}`));
      document.head.appendChild(script);
    }).catch((error) => {
      bundleLoad = undefined;
      throw error;
    });
    bundleLoad = { sha256: expected.sha256, promise };
  }
  return bundleLoad.promise;
}

function validateManifest(manifest: TvmjsManifest): void {
  if (manifest.schema !== EXPECTED_SCHEMA) throw new Error(`Unsupported TVMJS manifest schema: ${manifest.schema}`);
  if (manifest.modelFamily !== EXPECTED_MODEL_FAMILY) throw new Error(`Unexpected TVMJS model family: ${manifest.modelFamily}`);
  if (manifest.dtype !== 'f32') throw new Error(`Unexpected TVMJS dtype: ${manifest.dtype}`);
  if (manifest.target !== 'webgpu') throw new Error(`Unexpected TVMJS target: ${manifest.target}`);
}

function prepareRows(boards: BoardState[], contexts: EvaluationContext[], meta: SquareFormerMeta): PreparedRow[] {
  return boards.map((board, index) => {
    const context = contexts[index] ?? {};
    const normalized = isStmWhiteRankflip(meta.board_normalization)
      ? normalizePositionForStmWhite(board, context.historyFens ?? [], context.legalMoves)
      : { board, historyFens: context.historyFens ?? [], legalMoves: context.legalMoves, flipped: false };
    return { ...normalized, attackSummaryChannelMask: context.attackSummaryChannelMask };
  });
}

function floatArray(bytes: Uint8Array): Float32Array {
  const copy = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
  return new Float32Array(copy);
}

export class SquareformerTvmjsWebgpuEvaluator implements Evaluator {
  private runTail: Promise<unknown> = Promise.resolve();
  private failureReason: string | undefined;
  private destroyed = false;
  private readonly runtime: TvmRuntime;
  private readonly device: GpuDevice;
  private readonly vmModule: { getFunction(name: string): TvmPackedFunction };
  private readonly meta: SquareFormerMeta;
  readonly physicalBatchSize: number;

  private constructor(
    runtime: TvmRuntime,
    device: GpuDevice,
    vmModule: { getFunction(name: string): TvmPackedFunction },
    meta: SquareFormerMeta,
    physicalBatchSize: number,
  ) {
    this.runtime = runtime;
    this.device = device;
    this.vmModule = vmModule;
    this.meta = meta;
    this.physicalBatchSize = physicalBatchSize;
    this.device.addEventListener?.('uncapturederror', (event) => {
      this.failureReason = event.error?.message ?? 'Uncaptured WebGPU error';
    });
    void this.device.lost?.then((info) => {
      this.failureReason = info?.message || 'WebGPU device lost';
    });
  }

  static async create(manifestUrl: string, meta: SquareFormerMeta): Promise<SquareformerTvmjsWebgpuEvaluator> {
    const manifestResolved = new URL(manifestUrl, location.href);
    if (!isTrustedExecutableAssetUrl(manifestResolved.toString())) throw new Error(`Untrusted TVMJS manifest URL: ${manifestResolved}`);
    const response = await fetch(manifestResolved, { cache: 'no-cache' });
    if (!response.ok) throw new Error(`TVMJS manifest fetch failed ${response.status}: ${manifestResolved}`);
    const manifest = v.parse(TvmjsManifestSchema, await response.json());
    validateManifest(manifest);
    const model = manifest.models?.find((entry) => Number(entry.batch) === 16) ?? manifest.models?.[0];
    if (!model || !Number.isInteger(model.batch) || model.batch <= 0) throw new Error('TVMJS manifest has no fixed-batch model');
    const baseUrl = new URL('.', manifestResolved);
    const bundlePath = manifest.runtime?.tvmjsBundle ?? 'tvmjs.bundle.js';
    const bundleFile = manifest.files?.find((entry) => entry.path === bundlePath);
    if (!bundleFile?.sha256) throw new Error(`TVMJS manifest missing bundle integrity metadata: ${bundlePath}`);
    const bundleUrl = new URL(bundlePath, baseUrl);
    const api = await loadTvmjsBundle(bundleUrl, { sha256: bundleFile.sha256 });
    const gpu = (navigator as Navigator & { gpu?: { requestAdapter(options?: unknown): Promise<GpuAdapter | null> } }).gpu;
    if (!gpu) throw new Error('WebGPU is unavailable');
    const adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) throw new Error('WebGPU adapter unavailable');
    const requiredFeatures = (manifest.requiredFeatures ?? []).filter((feature) => feature !== 'webgpu');
    for (const feature of requiredFeatures) {
      if (!adapter.features.has(feature)) throw new Error(`WebGPU adapter missing required feature: ${feature}`);
    }
    const requiredLimits: Record<string, number> = {};
    for (const key of [
      'maxStorageBuffersPerShaderStage',
      'maxStorageBufferBindingSize',
      'maxBufferSize',
      'maxComputeWorkgroupStorageSize',
      'maxComputeInvocationsPerWorkgroup',
    ]) {
      const value = adapter.limits[key];
      if (value !== undefined) requiredLimits[key] = value;
    }
    const device = await adapter.requestDevice({ requiredFeatures, requiredLimits });
    let runtime: TvmRuntime | undefined;
    try {
      const wasmUrl = new URL(model.wasm, baseUrl);
      if (!isTrustedExecutableAssetUrl(wasmUrl.toString())) throw new Error(`Untrusted TVMJS model URL: ${wasmUrl}`);
      const wasmBytes = await fetchVerified(wasmUrl, model);
      runtime = await api.instantiate(wasmBytes, api.createPolyfillWASI(), () => {});
      runtime.initWebGPU(device);
      runtime.beginScope();
      const systemLib = runtime.systemLib();
      await runtime.asyncLoadWebGPUPipelines(systemLib);
      const vm = runtime.createVirtualMachine(runtime.webgpu());
      return new SquareformerTvmjsWebgpuEvaluator(runtime, device, vm.getInternalModule(), meta, model.batch);
    } catch (error) {
      runtime?.dispose();
      device.destroy();
      throw error;
    }
  }

  private runExclusive<T>(fn: () => Promise<T>): Promise<T> {
    const run = this.runTail.then(fn, fn);
    this.runTail = run.catch(() => undefined);
    return run;
  }

  async evaluate(board: BoardState, context: EvaluationContext = {}): Promise<Evaluation> {
    return (await this.evaluateBatch([board], [context]))[0]!;
  }

  async evaluateBatch(boards: BoardState[], contexts: EvaluationContext[] = []): Promise<Evaluation[]> {
    if (!boards.length) return [];
    const output: Evaluation[] = [];
    for (let offset = 0; offset < boards.length; offset += this.physicalBatchSize) {
      output.push(
        ...(await this.evaluatePhysicalBatch(boards.slice(offset, offset + this.physicalBatchSize), contexts.slice(offset, offset + this.physicalBatchSize))),
      );
    }
    return output;
  }

  private async evaluatePhysicalBatch(boards: BoardState[], contexts: EvaluationContext[]): Promise<Evaluation[]> {
    return this.runExclusive(async () => {
      if (this.destroyed) throw new Error('TVMJS evaluator is destroyed');
      if (this.failureReason) throw new Error(`TVMJS WebGPU runtime is unavailable: ${this.failureReason}`);
      const rows = prepareRows(boards, contexts, this.meta);
      const padded = [...rows];
      while (padded.length < this.physicalBatchSize) padded.push(rows[rows.length - 1]!);
      const stride = this.meta.token_features ?? this.meta.history_plies + 9;
      const attackFeatures = Number(this.meta.attack_summary_feature_count ?? 0);
      if (attackFeatures !== THREATGRAPH_SQUARE_SUMMARY_V1_FEATURES || this.meta.attack_summary_schema !== 'threatgraph_square_summary_v1') {
        throw new Error(`Unsupported TVMJS attack summary: ${this.meta.attack_summary_schema}/${attackFeatures}`);
      }
      const tokens = new Int32Array(this.physicalBatchSize * 64 * stride);
      const attack = new Float32Array(this.physicalBatchSize * 64 * attackFeatures);
      for (let index = 0; index < padded.length; index++) {
        const row = padded[index]!;
        tokens.set(squareformerCompactInput(row.board, this.meta, row.historyFens, 'int32') as Int32Array, index * 64 * stride);
        const attackRow = threatgraphSquareSummaryV1(row.board);
        if (row.attackSummaryChannelMask) {
          if (row.attackSummaryChannelMask.length !== attackFeatures) {
            throw new Error(`attackSummaryChannelMask length ${row.attackSummaryChannelMask.length} does not match ${attackFeatures}`);
          }
          for (let square = 0; square < 64; square++) {
            const base = square * attackFeatures;
            for (let channel = 0; channel < attackFeatures; channel++) {
              attackRow[base + channel] *= Number(row.attackSummaryChannelMask[channel] ?? 0);
            }
          }
        }
        attack.set(attackRow, index * 64 * attackFeatures);
      }

      const setInput = this.vmModule.getFunction('set_input');
      const invokeStateful = this.vmModule.getFunction('invoke_stateful');
      const getOutput = this.vmModule.getFunction('get_output');
      this.runtime.beginScope();
      try {
        const tokensTensor = this.runtime.empty([this.physicalBatchSize, 64, stride], 'int32', this.runtime.webgpu());
        tokensTensor.copyFromRawBytes(new Uint8Array(tokens.buffer));
        const attackTensor = this.runtime.empty([this.physicalBatchSize, 64, attackFeatures], 'float32', this.runtime.webgpu());
        attackTensor.copyFromRawBytes(new Uint8Array(attack.buffer));
        setInput('main', tokensTensor, attackTensor);
        invokeStateful('main');
        const policyGpu = getOutput('main', this.runtime.scalar(OUTPUT_POLICY, 'int32')) as TvmTensor;
        const wdlGpu = getOutput('main', this.runtime.scalar(OUTPUT_WDL, 'int32')) as TvmTensor;
        const policyCpu = this.runtime.empty(policyGpu.shape, policyGpu.dtype, this.runtime.cpu());
        const wdlCpu = this.runtime.empty(wdlGpu.shape, wdlGpu.dtype, this.runtime.cpu());
        policyCpu.copyFrom(policyGpu);
        wdlCpu.copyFrom(wdlGpu);
        await wdlGpu.device.sync();
        if (this.failureReason) throw new Error(`TVMJS WebGPU runtime failed: ${this.failureReason}`);
        const policyRaw = floatArray(policyCpu.toRawBytes());
        const wdlRaw = floatArray(wdlCpu.toRawBytes());
        const expectedPolicy = this.physicalBatchSize * this.meta.policy_size;
        if (policyRaw.length !== expectedPolicy) throw new Error(`TVMJS policy output length ${policyRaw.length} != ${expectedPolicy}`);
        if (wdlRaw.length !== this.physicalBatchSize * 3) throw new Error(`TVMJS WDL output length ${wdlRaw.length} != ${this.physicalBatchSize * 3}`);
        return boards.map((board, index) => {
          const row = rows[index]!;
          const evalLegal = row.legalMoves ?? legalMoves(row.board);
          const policyRow = policyRaw.subarray(index * this.meta.policy_size, (index + 1) * this.meta.policy_size);
          const probabilities = softmax(evalLegal.map((move) => Number(policyRow[moveToSquareformerPolicyIndex(move)] ?? -100)));
          const policy = new Map<number, number>();
          evalLegal.forEach((move, moveIndex) => {
            const originalMove = normalizedMoveToOriginal(move, row.flipped);
            policy.set(moveToActionId(originalMove), probabilities[moveIndex] ?? 0);
          });
          const wdl = softmax(wdlRaw.subarray(index * 3, index * 3 + 3));
          return {
            policy,
            wdl: [wdl[0] ?? 0, wdl[1] ?? 0, wdl[2] ?? 0] as [number, number, number],
          };
        });
      } finally {
        this.runtime.endScope();
      }
    });
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    const dispose = () => {
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
        this.device.destroy();
      } catch {
        /* best effort */
      }
    };
    void this.runTail.then(dispose, dispose);
  }
}
