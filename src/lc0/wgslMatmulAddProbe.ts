import * as ort from '../nn/ortRuntime.ts';
import { boardToFen, type BoardState } from '../chess/board.ts';
import { legalMoves } from '../chess/movegen.ts';
import { moveToUci, type Move } from '../chess/moveCodec.ts';
import { encodeLc0Classical112, type Lc0HistoryFill } from './encoder112.ts';
import { currentBoardAndFen, LC0_DEFAULT_POLICY_TEMPERATURE, legalPolicyPriors, type Lc0Evaluation, type Lc0EvaluatorInput } from './onnxEvaluator.ts';
import { LC0_MIRROR_TRANSFORM, uciToLc0PolicyIndex } from './policyMap.ts';
import { loadLc0WebModelPack, type Lc0WebTensorView } from './modelPack.ts';
import { createLc0WasmInputEncoder, type Lc0WasmInputEncoder, type Lc0WasmInputEncoderTiming } from './wasmInputEncoder.ts';
import { createLc0WasmLegalPriors, type Lc0WasmLegalPriors, type Lc0WasmLegalPriorTiming } from './wasmLegalPriors.ts';
import { ATTENTION_BLOCK_QKV_TVM_PACKED_F16_WGSL, ATTENTION_OUTPUT_PROJ_TVM_PACKED_F16_WGSL, FFN_DENSE1_TVM_PACKED_F16_WGSL, FFN_DENSE2_TVM_PACKED_F16_WGSL } from './generated/tvmPackedF16Wgsl.ts';

const DEFAULT_WEIGHT_TENSOR = '/encoder0/mha/Q/w/w';
const DEFAULT_BIAS_TENSOR = '/encoder0/mha/Q/b/w';
const DEFAULT_QKV_TENSORS = {
  qWeight: '/encoder0/mha/Q/w/w',
  qBias: '/encoder0/mha/Q/b/w',
  kWeight: '/encoder0/mha/K/w/w',
  kBias: '/encoder0/mha/K/b/w',
  vWeight: '/encoder0/mha/V/w/w',
  vBias: '/encoder0/mha/V/b/w',
} as const;
const DEFAULT_SCALE_TENSOR = '/encoder0/mha/QK/scale/w';
const DEFAULT_SMOLGEN_TENSORS = {
  compressWeight: '/encoder0/smolgen/compress/w',
  dense1Weight: '/encoder0/smolgen/dense1/w/w',
  dense1Bias: '/encoder0/smolgen/dense1/b/w',
  ln1Scale: '/encoder0/smolgen/ln1/w/scale',
  ln1Bias: '/encoder0/smolgen/ln1/w/bias',
  dense2Weight: '/encoder0/smolgen/dense2/w/w',
  dense2Bias: '/encoder0/smolgen/dense2/b/w',
  ln2Scale: '/encoder0/smolgen/ln2/w/scale',
  ln2Bias: '/encoder0/smolgen/ln2/w/bias',
  smolgenWeight: '/const/smolgen_w',
} as const;
const DEFAULT_K = 256;
const DEFAULT_N = 256;
const DEFAULT_TOKENS = 64;
const DEFAULT_HEADS = 8;
const DEFAULT_HEAD_DIM = DEFAULT_N / DEFAULT_HEADS;
const DEFAULT_SMOLGEN_COMPRESSED = 32;
const DEFAULT_SMOLGEN_HIDDEN = 256;
const DEFAULT_SMOLGEN_FLAT = DEFAULT_TOKENS * DEFAULT_SMOLGEN_COMPRESSED;
const DEFAULT_SMOLGEN_EPSILON = 1e-3;

export type Lc0WebMatmulAddKernelVariant = 'scalar' | 'tiled16' | 'scalar-transposed';

export interface Lc0WebMatmulAddKernelProbeOptions {
  packUrl: string;
  weightTensorName?: string;
  biasTensorName?: string;
  iterations?: number;
  warmup?: number;
  verifyShards?: boolean;
  variant?: Lc0WebMatmulAddKernelVariant;
}

export interface Lc0WebMatmulAddKernelBenchmarkOptions extends Lc0WebMatmulAddKernelProbeOptions {
  /** Optional extra dispatches submitted before timing, with no readback. */
  warmup?: number;
}

export interface Lc0WebMatmulAddOrtBenchmarkOptions extends Lc0WebMatmulAddKernelProbeOptions {
  iterations?: number;
  warmup?: number;
}

export interface Lc0WebAttentionScoreBenchmarkOptions {
  packUrl: string;
  iterations?: number;
  warmup?: number;
  verifyShards?: boolean;
}

export interface Lc0WebMatmulAddKernelProbeResult {
  status: 'KERNEL_DONE';
  packUrl: string;
  modelName: string;
  adapterInfo?: Record<string, unknown>;
  weightTensor: string;
  biasTensor: string;
  variant: Lc0WebMatmulAddKernelVariant;
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
}

export interface Lc0WebMatmulAddKernelBenchmarkResult {
  status: 'KERNEL_BENCH_DONE';
  packUrl: string;
  modelName: string;
  adapterInfo?: Record<string, unknown>;
  weightTensor: string;
  biasTensor: string;
  variant: Lc0WebMatmulAddKernelVariant;
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
}

export interface Lc0WebQkvProjectionProbeOptions {
  packUrl: string;
  iterations?: number;
  warmup?: number;
  verifyShards?: boolean;
}

export interface Lc0WebQkvProjectionBenchmarkOptions extends Lc0WebQkvProjectionProbeOptions {
  /** Optional extra dispatches submitted before timing, with no readback. */
  warmup?: number;
}

export interface Lc0WebMatmulAddOrtBenchmarkResult {
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
}

export interface Lc0WebQkvProjectionProbeResult {
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
}

export interface Lc0WebQkvProjectionBenchmarkResult {
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
}

export interface Lc0WebAttentionScoreBenchmarkResult {
  status: 'ATTENTION_SCORE_BENCH_DONE';
  packUrl: string;
  modelName: string;
  adapterInfo?: Record<string, unknown>;
  tokens: number;
  channels: number;
  heads: number;
  headDim: number;
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
}

export interface Lc0WebSmolgenBenchmarkOptions {
  packUrl: string;
  iterations?: number;
  warmup?: number;
  verifyShards?: boolean;
  encoderPrefix?: string;
}

export interface Lc0WebSmolgenBenchmarkResult {
  status: 'SMOLGEN_BENCH_DONE';
  packUrl: string;
  modelName: string;
  adapterInfo?: Record<string, unknown>;
  encoderPrefix: string;
  tokens: number;
  channels: number;
  compressed: number;
  hidden: number;
  heads: number;
  epsilon: number;
  warmup: number;
  iterations: number;
  packLoadMs: number;
  uploadSetupMs: number;
  dispatchLoopMs: number;
  dispatchLoopAvgMs: number;
  stageDispatchAvgMs: Record<'compress' | 'dense1' | 'ln1' | 'dense2' | 'ln2' | 'project', number>;
  readbackSyncedMs: number;
  endToEndMs: number;
  maxAbsError: number;
  rmsError: number;
  outputSample: number[];
}

export interface Lc0WebAttentionScoreOrtBenchmarkResult {
  status: 'ATTENTION_SCORE_ORT_BENCH_DONE';
  packUrl: string;
  modelName: string;
  tokens: number;
  channels: number;
  heads: number;
  headDim: number;
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
}

type GpuGlobals = typeof globalThis & {
  navigator?: { gpu?: unknown };
  GPUBufferUsage?: Record<string, number>;
  GPUMapMode?: Record<string, number>;
};

type GpuLike = {
  requestAdapter: () => Promise<unknown>;
};

type AdapterLike = {
  features?: { has: (feature: string) => boolean };
  requestDevice: (descriptor?: Record<string, unknown>) => Promise<DeviceLike>;
  requestAdapterInfo?: () => Promise<Record<string, unknown>>;
  info?: Record<string, unknown>;
};

type DeviceLike = {
  queue: {
    writeBuffer: (buffer: unknown, bufferOffset: number, data: unknown) => void;
    submit: (commandBuffers: unknown[]) => void;
    onSubmittedWorkDone?: () => Promise<void>;
  };
  createBuffer: (descriptor: Record<string, unknown>) => BufferLike;
  createQuerySet?: (descriptor: Record<string, unknown>) => QuerySetLike;
  createShaderModule: (descriptor: Record<string, unknown>) => unknown;
  createComputePipeline: (descriptor: Record<string, unknown>) => PipelineLike;
  createBindGroup: (descriptor: Record<string, unknown>) => unknown;
  createCommandEncoder: () => CommandEncoderLike;
};

type QuerySetLike = {
  destroy?: () => void;
};

type BufferLike = {
  getMappedRange: () => ArrayBuffer;
  unmap: () => void;
  mapAsync: (mode: number) => Promise<void>;
  destroy?: () => void;
};

type PipelineLike = {
  getBindGroupLayout: (index: number) => unknown;
};

type CommandEncoderLike = {
  beginComputePass: (descriptor?: Record<string, unknown>) => ComputePassLike;
  copyBufferToBuffer: (source: unknown, sourceOffset: number, destination: unknown, destinationOffset: number, size: number) => void;
  writeTimestamp?: (querySet: unknown, queryIndex: number) => void;
  resolveQuerySet?: (querySet: unknown, firstQuery: number, queryCount: number, destination: unknown, destinationOffset: number) => void;
  finish: () => unknown;
};

type ComputePassLike = {
  setPipeline: (pipeline: unknown) => void;
  setBindGroup: (index: number, bindGroup: unknown) => void;
  dispatchWorkgroups: (x: number, y?: number, z?: number) => void;
  end: () => void;
};

type DispatchCounter = { count: number };

function beginCountedComputePass(encoder: CommandEncoderLike, counter: DispatchCounter, descriptor?: Record<string, unknown>): ComputePassLike {
  const pass = encoder.beginComputePass(descriptor);
  return {
    setPipeline: (pipeline) => pass.setPipeline(pipeline),
    setBindGroup: (index, bindGroup) => pass.setBindGroup(index, bindGroup),
    dispatchWorkgroups: (x, y, z) => { counter.count += 1; pass.dispatchWorkgroups(x, y, z); },
    end: () => pass.end(),
  };
}

function gpuGlobals(): GpuGlobals {
  return globalThis as GpuGlobals;
}

function nowMs(): number {
  return typeof performance === 'undefined' ? Date.now() : performance.now();
}

function clampInteger(value: unknown, fallback: number, min: number, max: number): number {
  const numeric = Number(value ?? fallback);
  const finite = Number.isFinite(numeric) ? numeric : fallback;
  return Math.min(max, Math.max(min, Math.floor(finite)));
}

function normalizeKernelVariant(value: unknown): Lc0WebMatmulAddKernelVariant {
  return value === 'tiled16' || value === 'scalar-transposed' ? value : 'scalar';
}

export function f16BitsToF32(bits: number): number {
  const sign = (bits & 0x8000) ? -1 : 1;
  const exp = (bits >>> 10) & 0x1f;
  const frac = bits & 0x03ff;
  if (exp === 0) return sign * (frac === 0 ? 0 : Math.pow(2, -14) * (frac / 1024));
  if (exp === 0x1f) return frac === 0 ? sign * Infinity : NaN;
  return sign * Math.pow(2, exp - 15) * (1 + frac / 1024);
}

function readF16At(bytes: Uint8Array, index: number): number {
  const byteIndex = index * 2;
  return f16BitsToF32(bytes[byteIndex] | (bytes[byteIndex + 1] << 8));
}

function makeInputVector(k: number): Float32Array {
  const input = new Float32Array(k);
  for (let i = 0; i < k; i++) {
    // Deterministic non-trivial signal with enough variation to catch layout bugs.
    input[i] = Math.sin(i * 0.071) * 0.5 + Math.cos(i * 0.013) * 0.25;
  }
  return input;
}

function makeInputTokenMatrix(tokens: number, channels: number): Float32Array<ArrayBufferLike> {
  const input = new Float32Array(tokens * channels);
  for (let token = 0; token < tokens; token++) {
    for (let channel = 0; channel < channels; channel++) {
      const i = token * channels + channel;
      input[i] = Math.sin((token + 1) * 0.037 + channel * 0.071) * 0.5 + Math.cos(token * 0.019 - channel * 0.013) * 0.25;
    }
  }
  return input;
}

function cpuMatmulAdd(input: Float32Array<ArrayBufferLike>, weight: Uint8Array, bias: Uint8Array, k: number, n: number): Float32Array<ArrayBufferLike> {
  const output = new Float32Array(n);
  for (let col = 0; col < n; col++) {
    let sum = readF16At(bias, col);
    for (let row = 0; row < k; row++) sum += input[row] * readF16At(weight, row * n + col);
    output[col] = sum;
  }
  return output;
}

function cpuProjectTokens(input: Float32Array<ArrayBufferLike>, weight: Uint8Array, bias: Uint8Array, tokens: number, k: number, n: number): Float32Array<ArrayBufferLike> {
  const output = new Float32Array(tokens * n);
  for (let token = 0; token < tokens; token++) {
    const tokenInput = input.subarray(token * k, (token + 1) * k);
    output.set(cpuMatmulAdd(tokenInput, weight, bias, k, n), token * n);
  }
  return output;
}

function cpuProjectTokensNoBias(input: Float32Array<ArrayBufferLike>, weight: Uint8Array, tokens: number, k: number, n: number): Float32Array<ArrayBufferLike> {
  const output = new Float32Array(tokens * n);
  for (let token = 0; token < tokens; token++) {
    const tokenBase = token * k;
    for (let col = 0; col < n; col++) {
      let sum = 0;
      for (let row = 0; row < k; row++) sum += input[tokenBase + row] * readF16At(weight, row * n + col);
      output[token * n + col] = sum;
    }
  }
  return output;
}

function cpuMatmulAddVector(input: Float32Array<ArrayBufferLike>, weight: Uint8Array, bias: Uint8Array, k: number, n: number): Float32Array<ArrayBufferLike> {
  const output = new Float32Array(n);
  for (let col = 0; col < n; col++) {
    let sum = readF16At(bias, col);
    for (let row = 0; row < k; row++) sum += input[row] * readF16At(weight, row * n + col);
    output[col] = sum;
  }
  return output;
}

function cpuMatmulVectorNoBias(input: Float32Array<ArrayBufferLike>, weight: Uint8Array, k: number, n: number): Float32Array<ArrayBufferLike> {
  const output = new Float32Array(n);
  for (let col = 0; col < n; col++) {
    let sum = 0;
    for (let row = 0; row < k; row++) sum += input[row] * readF16At(weight, row * n + col);
    output[col] = sum;
  }
  return output;
}

function cpuSwish(input: Float32Array<ArrayBufferLike>): Float32Array<ArrayBufferLike> {
  const output = new Float32Array(input.length);
  for (let i = 0; i < input.length; i++) output[i] = input[i] / (1 + Math.exp(-input[i]));
  return output;
}

function cpuLayerNormVector(input: Float32Array<ArrayBufferLike>, scale: Uint8Array, bias: Uint8Array, epsilon: number): Float32Array<ArrayBufferLike> {
  const output = new Float32Array(input.length);
  let mean = 0;
  for (const value of input) mean += value;
  mean /= input.length;
  let variance = 0;
  for (const value of input) {
    const centered = value - mean;
    variance += centered * centered;
  }
  const invStd = 1 / Math.sqrt(variance / input.length + epsilon);
  for (let i = 0; i < input.length; i++) output[i] = (input[i] - mean) * invStd * readF16At(scale, i) + readF16At(bias, i);
  return output;
}

function cpuAttentionScores(q: Float32Array<ArrayBufferLike>, k: Float32Array<ArrayBufferLike>, scale: number, tokens: number, channels: number, heads: number): Float32Array<ArrayBufferLike> {
  const headDim = channels / heads;
  const output = new Float32Array(heads * tokens * tokens);
  for (let head = 0; head < heads; head++) {
    const channelOffset = head * headDim;
    for (let row = 0; row < tokens; row++) {
      for (let col = 0; col < tokens; col++) {
        let sum = 0;
        for (let channel = 0; channel < headDim; channel++) {
          sum += q[row * channels + channelOffset + channel] * k[col * channels + channelOffset + channel];
        }
        output[(head * tokens + row) * tokens + col] = sum * scale;
      }
    }
  }
  return output;
}

function packHeadsQ(input: Float32Array<ArrayBufferLike>, tokens: number, channels: number, heads: number): Float32Array<ArrayBufferLike> {
  const headDim = channels / heads;
  const out = new Float32Array(input.length);
  for (let head = 0; head < heads; head++) {
    for (let token = 0; token < tokens; token++) {
      for (let channel = 0; channel < headDim; channel++) {
        out[(head * tokens + token) * headDim + channel] = input[token * channels + head * headDim + channel];
      }
    }
  }
  return out;
}

function packHeadsKt(input: Float32Array<ArrayBufferLike>, tokens: number, channels: number, heads: number): Float32Array<ArrayBufferLike> {
  const headDim = channels / heads;
  const out = new Float32Array(input.length);
  for (let head = 0; head < heads; head++) {
    for (let channel = 0; channel < headDim; channel++) {
      for (let token = 0; token < tokens; token++) {
        out[(head * headDim + channel) * tokens + token] = input[token * channels + head * headDim + channel];
      }
    }
  }
  return out;
}

function assertTensorShapeAndBytes(tensor: Lc0WebTensorView, expected: number[], bytesPerElement: number, label: string): void {
  const got = tensor.info.shape;
  if (got.length !== expected.length || got.some((value, i) => value !== expected[i])) {
    throw new Error(`${label} tensor ${tensor.info.name} shape mismatch: got [${got.join(',')}], expected [${expected.join(',')}]`);
  }
  const expectedBytes = expected.reduce((product, dim) => product * dim, 1) * bytesPerElement;
  if (tensor.bytes.byteLength !== expectedBytes || tensor.info.byteLength !== expectedBytes) {
    throw new Error(`${label} tensor ${tensor.info.name} byte length mismatch: got ${tensor.bytes.byteLength}/${tensor.info.byteLength}, expected ${expectedBytes}`);
  }
}

function createStorageBuffer(device: DeviceLike, data: ArrayBufferView | ArrayBufferLike, usage: number): BufferLike {
  const byteLength = ArrayBuffer.isView(data) ? data.byteLength : data.byteLength;
  const paddedSize = Math.max(4, Math.ceil(byteLength / 4) * 4);
  const buffer = device.createBuffer({ size: paddedSize, usage, mappedAtCreation: true });
  const mapped = new Uint8Array(buffer.getMappedRange());
  const source = ArrayBuffer.isView(data)
    ? new Uint8Array(data.buffer, data.byteOffset, data.byteLength)
    : new Uint8Array(data);
  mapped.set(source);
  buffer.unmap();
  return buffer;
}

class ProtoWriter {
  private readonly chunks: Uint8Array[] = [];

  finish(): Uint8Array {
    const total = this.chunks.reduce((sum, chunk) => sum + chunk.byteLength, 0);
    const out = new Uint8Array(total);
    let offset = 0;
    for (const chunk of this.chunks) {
      out.set(chunk, offset);
      offset += chunk.byteLength;
    }
    return out;
  }

  varint(value: number | bigint): void {
    let v = BigInt(value);
    const bytes: number[] = [];
    while (v >= 0x80n) {
      bytes.push(Number((v & 0x7fn) | 0x80n));
      v >>= 7n;
    }
    bytes.push(Number(v));
    this.chunks.push(Uint8Array.from(bytes));
  }

  tag(field: number, wireType: number): void {
    this.varint((field << 3) | wireType);
  }

  int32(field: number, value: number): void {
    this.tag(field, 0);
    this.varint(value);
  }

  int64(field: number, value: number): void {
    this.tag(field, 0);
    this.varint(BigInt(value));
  }

  float32(field: number, value: number): void {
    const bytes = new Uint8Array(4);
    new DataView(bytes.buffer).setFloat32(0, value, true);
    this.tag(field, 5);
    this.chunks.push(bytes);
  }

  string(field: number, value: string): void {
    const bytes = new TextEncoder().encode(value);
    this.bytes(field, bytes);
  }

  bytes(field: number, bytes: Uint8Array): void {
    this.tag(field, 2);
    this.varint(bytes.byteLength);
    this.chunks.push(bytes);
  }

  message(field: number, write: (writer: ProtoWriter) => void): void {
    const nested = new ProtoWriter();
    write(nested);
    this.bytes(field, nested.finish());
  }
}

function f32ArrayToBytes(values: Float32Array<ArrayBufferLike>): Uint8Array {
  return new Uint8Array(new Uint8Array(values.buffer, values.byteOffset, values.byteLength));
}

function f16BytesToF32Array(bytes: Uint8Array, elements: number): Float32Array<ArrayBufferLike> {
  const out = new Float32Array(elements);
  for (let i = 0; i < elements; i++) out[i] = readF16At(bytes, i);
  return out;
}

function transposeF16MatrixBytes(bytes: Uint8Array, rows: number, cols: number): Uint8Array {
  const out = new Uint8Array(bytes.byteLength);
  for (let row = 0; row < rows; row++) {
    for (let col = 0; col < cols; col++) {
      const source = (row * cols + col) * 2;
      const target = (col * rows + row) * 2;
      out[target] = bytes[source];
      out[target + 1] = bytes[source + 1];
    }
  }
  return out;
}

function createTransposedF16StorageBuffer(device: DeviceLike, bytes: Uint8Array, rows: number, cols: number, usage: number): BufferLike {
  return createStorageBuffer(device, transposeF16MatrixBytes(bytes, rows, cols), usage);
}

function onnxDim(value: number): Uint8Array {
  const writer = new ProtoWriter();
  writer.int64(1, value);
  return writer.finish();
}

function onnxShape(dims: number[]): Uint8Array {
  const writer = new ProtoWriter();
  for (const dim of dims) writer.bytes(1, onnxDim(dim));
  return writer.finish();
}

function onnxTensorType(elemType: number, dims: number[]): Uint8Array {
  const writer = new ProtoWriter();
  writer.int32(1, elemType);
  writer.bytes(2, onnxShape(dims));
  return writer.finish();
}

function onnxValueInfo(name: string, elemType: number, dims: number[]): Uint8Array {
  const writer = new ProtoWriter();
  writer.string(1, name);
  writer.message(2, (type) => type.bytes(1, onnxTensorType(elemType, dims)));
  return writer.finish();
}

function onnxTensor(name: string, dims: number[], values: Float32Array<ArrayBufferLike>): Uint8Array {
  const writer = new ProtoWriter();
  for (const dim of dims) writer.int64(1, dim);
  writer.int32(2, 1); // TensorProto.FLOAT
  writer.string(8, name);
  writer.bytes(9, f32ArrayToBytes(values));
  return writer.finish();
}

function onnxInt64Tensor(name: string, dims: number[], values: readonly number[]): Uint8Array {
  const raw = new Uint8Array(values.length * 8);
  const view = new DataView(raw.buffer);
  values.forEach((value, index) => view.setBigInt64(index * 8, BigInt(value), true));
  const writer = new ProtoWriter();
  for (const dim of dims) writer.int64(1, dim);
  writer.int32(2, 7); // TensorProto.INT64
  writer.string(8, name);
  writer.bytes(9, raw);
  return writer.finish();
}

function onnxInt32Tensor(name: string, dims: number[], values: readonly number[]): Uint8Array {
  const raw = new Uint8Array(values.length * 4);
  const view = new DataView(raw.buffer);
  values.forEach((value, index) => view.setInt32(index * 4, value, true));
  const writer = new ProtoWriter();
  for (const dim of dims) writer.int64(1, dim);
  writer.int32(2, 6); // TensorProto.INT32
  writer.string(8, name);
  writer.bytes(9, raw);
  return writer.finish();
}

function onnxFloatAttribute(name: string, value: number): Uint8Array {
  const writer = new ProtoWriter();
  writer.string(1, name);
  writer.float32(2, value);
  writer.int32(20, 1); // AttributeProto.FLOAT
  return writer.finish();
}

function onnxIntAttribute(name: string, value: number): Uint8Array {
  const writer = new ProtoWriter();
  writer.string(1, name);
  writer.int64(3, value);
  writer.int32(20, 2); // AttributeProto.INT
  return writer.finish();
}

function onnxIntsAttribute(name: string, values: readonly number[]): Uint8Array {
  const writer = new ProtoWriter();
  writer.string(1, name);
  for (const value of values) writer.int64(8, value);
  writer.int32(20, 7); // AttributeProto.INTS
  return writer.finish();
}

function onnxNode(opType: string, inputs: string[], outputs: string[], name: string, attributes: Uint8Array[] = []): Uint8Array {
  const writer = new ProtoWriter();
  for (const input of inputs) writer.string(1, input);
  for (const output of outputs) writer.string(2, output);
  writer.string(3, name);
  writer.string(4, opType);
  for (const attribute of attributes) writer.bytes(5, attribute);
  return writer.finish();
}

export function createTinyMatmulAddOnnxForTest(weightF32: Float32Array<ArrayBufferLike>, biasF32: Float32Array<ArrayBufferLike>): Uint8Array {
  const writer = new ProtoWriter();
  writer.int64(1, 8); // IR_VERSION_2021_7_30; new enough for opset 13.
  writer.string(2, 'lc0web');
  writer.message(7, (graph) => {
    graph.bytes(1, onnxNode('MatMul', ['input', 'weight'], ['matmul_out'], 'matmul'));
    graph.bytes(1, onnxNode('Add', ['matmul_out', 'bias'], ['output'], 'add'));
    graph.string(2, 'lc0web_matmul_add_256');
    graph.bytes(5, onnxTensor('weight', [DEFAULT_K, DEFAULT_N], weightF32));
    graph.bytes(5, onnxTensor('bias', [DEFAULT_N], biasF32));
    graph.bytes(11, onnxValueInfo('input', 1, [1, DEFAULT_K]));
    graph.bytes(12, onnxValueInfo('output', 1, [1, DEFAULT_N]));
  });
  writer.message(8, (opset) => opset.int64(2, 13));
  return writer.finish();
}

export function createTinyAttentionScoreOnnxForTest(scale: number, withBias = false): Uint8Array {
  const writer = new ProtoWriter();
  writer.int64(1, 8);
  writer.string(2, 'lc0web');
  writer.message(7, (graph) => {
    graph.bytes(1, onnxNode('MatMul', ['q', 'kt'], ['matmul_out'], 'matmul'));
    graph.bytes(1, onnxNode('Mul', ['matmul_out', 'scale'], [withBias ? 'scaled' : 'output'], 'scale'));
    if (withBias) graph.bytes(1, onnxNode('Add', ['scaled', 'bias'], ['output'], 'smolgen_bias_add'));
    graph.string(2, withBias ? 'lc0web_attention_score_heads_smolgen_bias' : 'lc0web_attention_score_heads');
    graph.bytes(5, onnxTensor('scale', [1], new Float32Array([scale])));
    graph.bytes(11, onnxValueInfo('q', 1, [DEFAULT_HEADS, DEFAULT_TOKENS, DEFAULT_HEAD_DIM]));
    graph.bytes(11, onnxValueInfo('kt', 1, [DEFAULT_HEADS, DEFAULT_HEAD_DIM, DEFAULT_TOKENS]));
    if (withBias) graph.bytes(11, onnxValueInfo('bias', 1, [DEFAULT_HEADS, DEFAULT_TOKENS, DEFAULT_TOKENS]));
    graph.bytes(12, onnxValueInfo('output', 1, [DEFAULT_HEADS, DEFAULT_TOKENS, DEFAULT_TOKENS]));
  });
  writer.message(8, (opset) => opset.int64(2, 13));
  return writer.finish();
}

const WGSL_HEADER = `
@group(0) @binding(0) var<storage, read> inputVec: array<f32>;
@group(0) @binding(1) var<storage, read> weightsF16: array<u32>;
@group(0) @binding(2) var<storage, read> biasF16: array<u32>;
@group(0) @binding(3) var<storage, read_write> outputVec: array<f32>;

fn pick_lane(word: u32, index: u32) -> f32 {
  let pair = unpack2x16float(word);
  return select(pair.x, pair.y, (index & 1u) == 1u);
}

fn load_weight(index: u32) -> f32 {
  return pick_lane(weightsF16[index >> 1u], index);
}

fn load_bias(index: u32) -> f32 {
  return pick_lane(biasF16[index >> 1u], index);
}
`;

const SCALAR_WGSL = `${WGSL_HEADER}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let col = gid.x;
  if (col >= 256u) { return; }
  var sum = load_bias(col);
  for (var row = 0u; row < 256u; row = row + 1u) {
    sum = sum + inputVec[row] * load_weight(row * 256u + col);
  }
  outputVec[col] = sum;
}
`;

const SCALAR_TRANSPOSED_WGSL = `${WGSL_HEADER}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let col = gid.x;
  if (col >= 256u) { return; }
  var sum = load_bias(col);
  for (var row = 0u; row < 256u; row = row + 1u) {
    sum = sum + inputVec[row] * load_weight(col * 256u + row);
  }
  outputVec[col] = sum;
}
`;

const TILED16_WGSL = `${WGSL_HEADER}
var<workgroup> partial: array<f32, 256>;

@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let col = wid.x * 16u + lid.x;
  let local_index = lid.y * 16u + lid.x;
  var sum = 0.0;
  if (col < 256u) {
    for (var row = lid.y; row < 256u; row = row + 16u) {
      sum = sum + inputVec[row] * load_weight(row * 256u + col);
    }
  }
  partial[local_index] = sum;
  workgroupBarrier();

  var stride = 8u;
  loop {
    if (lid.y < stride) {
      partial[lid.y * 16u + lid.x] = partial[lid.y * 16u + lid.x] + partial[(lid.y + stride) * 16u + lid.x];
    }
    workgroupBarrier();
    if (stride == 1u) { break; }
    stride = stride / 2u;
  }

  if (lid.y == 0u && col < 256u) {
    outputVec[col] = load_bias(col) + partial[lid.x];
  }
}
`;

function wgslForVariant(variant: Lc0WebMatmulAddKernelVariant): string {
  if (variant === 'tiled16') return TILED16_WGSL;
  if (variant === 'scalar-transposed') return SCALAR_TRANSPOSED_WGSL;
  return SCALAR_WGSL;
}


function cloneableAdapterInfo(info: unknown): Record<string, unknown> | undefined {
  if (!info || typeof info !== 'object') return undefined;
  const source = info as Record<string, unknown>;
  const out: Record<string, unknown> = {};
  for (const key of ['vendor', 'architecture', 'device', 'description', 'isFallbackAdapter']) {
    const value = source[key];
    if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') out[key] = value;
  }
  return Object.keys(out).length ? out : undefined;
}

async function requestDevice(options: { timestampQuery?: boolean } = {}): Promise<{ device: DeviceLike; adapterInfo?: Record<string, unknown>; timestampQuerySupported: boolean }> {
  const globals = gpuGlobals();
  const gpu = globals.navigator?.gpu as GpuLike | undefined;
  if (!gpu) throw new Error('WebGPU unavailable for lc0web kernel probe');
  const adapter = await gpu.requestAdapter() as AdapterLike | null;
  if (!adapter) throw new Error('WebGPU adapter unavailable for lc0web kernel probe');
  const rawAdapterInfo = adapter.requestAdapterInfo ? await adapter.requestAdapterInfo() : adapter.info;
  const timestampQuerySupported = adapter.features?.has('timestamp-query') === true;
  const device = await adapter.requestDevice(options.timestampQuery && timestampQuerySupported ? { requiredFeatures: ['timestamp-query'] } : undefined);
  return { device, adapterInfo: cloneableAdapterInfo(rawAdapterInfo), timestampQuerySupported };
}

function dispatchKernel(pass: ComputePassLike, variant: Lc0WebMatmulAddKernelVariant, n: number): void {
  if (variant === 'tiled16') pass.dispatchWorkgroups(Math.ceil(n / 16));
  else pass.dispatchWorkgroups(Math.ceil(n / 64));
}

function encodeKernelDispatches(device: DeviceLike, pipeline: PipelineLike, bindGroup: unknown, n: number, iterations: number, variant: Lc0WebMatmulAddKernelVariant): unknown {
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  for (let i = 0; i < iterations; i++) dispatchKernel(pass, variant, n);
  pass.end();
  return encoder.finish();
}

async function readOutputOnce(device: DeviceLike, outputBuffer: BufferLike, readbackBuffer: BufferLike, n: number): Promise<Float32Array<ArrayBufferLike>> {
  const globals = gpuGlobals();
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(outputBuffer, 0, readbackBuffer, 0, n * 4);
  device.queue.submit([encoder.finish()]);
  await readbackBuffer.mapAsync(globals.GPUMapMode!.READ);
  const copy = new Float32Array(readbackBuffer.getMappedRange().slice(0));
  readbackBuffer.unmap();
  return copy;
}

async function runKernelOnce(device: DeviceLike, pipeline: PipelineLike, bindGroup: unknown, outputBuffer: BufferLike, readbackBuffer: BufferLike, n: number, variant: Lc0WebMatmulAddKernelVariant): Promise<Float32Array<ArrayBufferLike>> {
  device.queue.submit([encodeKernelDispatches(device, pipeline, bindGroup, n, 1, variant)]);
  return readOutputOnce(device, outputBuffer, readbackBuffer, n);
}

function computeErrorStats(gpuOutput: Float32Array<ArrayBufferLike>, cpu: Float32Array<ArrayBufferLike>, n: number): { maxAbsError: number; rmsError: number } {
  let maxAbsError = 0;
  let sq = 0;
  for (let i = 0; i < n; i++) {
    const error = Math.abs(gpuOutput[i] - cpu[i]);
    maxAbsError = Math.max(maxAbsError, error);
    sq += error * error;
  }
  return { maxAbsError, rmsError: Math.sqrt(sq / n) };
}

function assertErrorInTolerance(maxAbsError: number): void {
  const tolerance = 1e-3;
  if (!Number.isFinite(maxAbsError) || maxAbsError > tolerance) {
    throw new Error(`lc0web MatMul+Add kernel verification failed: maxAbsError=${maxAbsError}, tolerance=${tolerance}`);
  }
}

function createMatmulAddPipeline(device: DeviceLike, inputBuffer: BufferLike, weightBuffer: BufferLike, biasBuffer: BufferLike, outputBuffer: BufferLike, variant: Lc0WebMatmulAddKernelVariant): { pipeline: PipelineLike; bindGroup: unknown } {
  const module = device.createShaderModule({ label: `lc0web matmul+add ${variant}`, code: wgslForVariant(variant) });
  const pipeline = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'main' } }) as PipelineLike;
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: inputBuffer } },
      { binding: 1, resource: { buffer: weightBuffer } },
      { binding: 2, resource: { buffer: biasBuffer } },
      { binding: 3, resource: { buffer: outputBuffer } },
    ],
  });
  return { pipeline, bindGroup };
}

export async function runLc0WebMatmulAddKernelProbe(options: Lc0WebMatmulAddKernelProbeOptions): Promise<Lc0WebMatmulAddKernelProbeResult> {
  const weightTensorName = options.weightTensorName ?? DEFAULT_WEIGHT_TENSOR;
  const biasTensorName = options.biasTensorName ?? DEFAULT_BIAS_TENSOR;
  const variant = normalizeKernelVariant(options.variant);
  const warmup = clampInteger(options.warmup, 2, 0, 50);
  const iterations = clampInteger(options.iterations, 10, 1, 1000);
  // Request WebGPU before fetching pack shards so unsupported browsers fail
  // without downloading/verifying model weights.
  const { device, adapterInfo } = await requestDevice();
  const pack = await loadLc0WebModelPack(options.packUrl, {
    verifyShards: options.verifyShards ?? true,
    tensorNames: [weightTensorName, biasTensorName],
  });
  const packLoadMs = pack.elapsedMs;
  const weight = pack.tensors.get(weightTensorName);
  const bias = pack.tensors.get(biasTensorName);
  if (!weight || !bias) throw new Error('lc0web kernel probe tensors were not loaded');
  assertTensorShapeAndBytes(weight, [DEFAULT_K, DEFAULT_N], 2, 'weight');
  assertTensorShapeAndBytes(bias, [DEFAULT_N], 2, 'bias');
  if (weight.info.dtype !== 'f16' || bias.info.dtype !== 'f16') {
    throw new Error(`lc0web kernel probe expects f16 tensors, got ${weight.info.dtype}/${bias.info.dtype}`);
  }

  const globals = gpuGlobals();
  const usage = globals.GPUBufferUsage!;
  const input = makeInputVector(DEFAULT_K);
  const cpu = cpuMatmulAdd(input, weight.bytes, bias.bytes, DEFAULT_K, DEFAULT_N);

  const buffers: BufferLike[] = [];
  try {
    const inputBuffer = createStorageBuffer(device, input, usage.STORAGE | usage.COPY_DST);
    const kernelWeightBytes = variant === 'scalar-transposed' ? transposeF16MatrixBytes(weight.bytes, DEFAULT_K, DEFAULT_N) : weight.bytes;
    const weightBuffer = createStorageBuffer(device, kernelWeightBytes, usage.STORAGE | usage.COPY_DST);
    const biasBuffer = createStorageBuffer(device, bias.bytes, usage.STORAGE | usage.COPY_DST);
    const outputBuffer = device.createBuffer({ size: DEFAULT_N * 4, usage: usage.STORAGE | usage.COPY_SRC });
    const readbackBuffer = device.createBuffer({ size: DEFAULT_N * 4, usage: usage.MAP_READ | usage.COPY_DST });
    buffers.push(inputBuffer, weightBuffer, biasBuffer, outputBuffer, readbackBuffer);
    const { pipeline, bindGroup } = createMatmulAddPipeline(device, inputBuffer, weightBuffer, biasBuffer, outputBuffer, variant);

    let gpuOutput: Float32Array<ArrayBufferLike> = new Float32Array(DEFAULT_N);
    for (let i = 0; i < warmup; i++) gpuOutput = await runKernelOnce(device, pipeline, bindGroup, outputBuffer, readbackBuffer, DEFAULT_N, variant);
    const times: number[] = [];
    for (let i = 0; i < iterations; i++) {
      const started = nowMs();
      gpuOutput = await runKernelOnce(device, pipeline, bindGroup, outputBuffer, readbackBuffer, DEFAULT_N, variant);
      times.push(nowMs() - started);
    }

    const { maxAbsError, rmsError } = computeErrorStats(gpuOutput, cpu, DEFAULT_N);
    assertErrorInTolerance(maxAbsError);
    const avgMs = times.reduce((sum, value) => sum + value, 0) / times.length;

    return {
      status: 'KERNEL_DONE',
      packUrl: pack.manifestUrl,
      modelName: pack.manifest.model.name,
      adapterInfo,
      weightTensor: weightTensorName,
      biasTensor: biasTensorName,
      variant,
      k: DEFAULT_K,
      n: DEFAULT_N,
      warmup,
      iterations,
      packLoadMs,
      avgMs,
      minMs: Math.min(...times),
      maxMs: Math.max(...times),
      firstMs: times[0],
      timesMs: times,
      maxAbsError,
      rmsError,
      outputSample: Array.from(gpuOutput.slice(0, 8)),
    };
  } finally {
    for (const buffer of buffers) buffer.destroy?.();
  }
}

export async function runLc0WebMatmulAddKernelBenchmark(options: Lc0WebMatmulAddKernelBenchmarkOptions): Promise<Lc0WebMatmulAddKernelBenchmarkResult> {
  const totalStarted = nowMs();
  const weightTensorName = options.weightTensorName ?? DEFAULT_WEIGHT_TENSOR;
  const biasTensorName = options.biasTensorName ?? DEFAULT_BIAS_TENSOR;
  const variant = normalizeKernelVariant(options.variant);
  const warmup = clampInteger(options.warmup, 10, 0, 1000);
  const iterations = clampInteger(options.iterations, 1000, 1, 100_000);
  const { device, adapterInfo } = await requestDevice();
  const pack = await loadLc0WebModelPack(options.packUrl, {
    verifyShards: options.verifyShards ?? true,
    tensorNames: [weightTensorName, biasTensorName],
  });
  const weight = pack.tensors.get(weightTensorName);
  const bias = pack.tensors.get(biasTensorName);
  if (!weight || !bias) throw new Error('lc0web kernel benchmark tensors were not loaded');
  assertTensorShapeAndBytes(weight, [DEFAULT_K, DEFAULT_N], 2, 'weight');
  assertTensorShapeAndBytes(bias, [DEFAULT_N], 2, 'bias');
  if (weight.info.dtype !== 'f16' || bias.info.dtype !== 'f16') {
    throw new Error(`lc0web kernel benchmark expects f16 tensors, got ${weight.info.dtype}/${bias.info.dtype}`);
  }

  const globals = gpuGlobals();
  const usage = globals.GPUBufferUsage!;
  const input = makeInputVector(DEFAULT_K);
  const cpu = cpuMatmulAdd(input, weight.bytes, bias.bytes, DEFAULT_K, DEFAULT_N);
  const buffers: BufferLike[] = [];
  try {
    const setupStarted = nowMs();
    const inputBuffer = createStorageBuffer(device, input, usage.STORAGE | usage.COPY_DST);
    const kernelWeightBytes = variant === 'scalar-transposed' ? transposeF16MatrixBytes(weight.bytes, DEFAULT_K, DEFAULT_N) : weight.bytes;
    const weightBuffer = createStorageBuffer(device, kernelWeightBytes, usage.STORAGE | usage.COPY_DST);
    const biasBuffer = createStorageBuffer(device, bias.bytes, usage.STORAGE | usage.COPY_DST);
    const outputBuffer = device.createBuffer({ size: DEFAULT_N * 4, usage: usage.STORAGE | usage.COPY_SRC });
    const readbackBuffer = device.createBuffer({ size: DEFAULT_N * 4, usage: usage.MAP_READ | usage.COPY_DST });
    buffers.push(inputBuffer, weightBuffer, biasBuffer, outputBuffer, readbackBuffer);
    const { pipeline, bindGroup } = createMatmulAddPipeline(device, inputBuffer, weightBuffer, biasBuffer, outputBuffer, variant);
    const uploadSetupMs = nowMs() - setupStarted;

    if (warmup > 0) {
      device.queue.submit([encodeKernelDispatches(device, pipeline, bindGroup, DEFAULT_N, warmup, variant)]);
      // Prefer a no-readback warmup barrier when supported so timed readback
      // is not polluted by pending warmup dispatches. Some browsers omit it;
      // correctness still holds, but readbackSyncedMs may include warmup work.
      await device.queue.onSubmittedWorkDone?.();
    }

    const dispatchStarted = nowMs();
    device.queue.submit([encodeKernelDispatches(device, pipeline, bindGroup, DEFAULT_N, iterations, variant)]);
    const dispatchLoopMs = nowMs() - dispatchStarted;

    const readbackStarted = nowMs();
    const gpuOutput = await readOutputOnce(device, outputBuffer, readbackBuffer, DEFAULT_N);
    const readbackSyncedMs = nowMs() - readbackStarted;
    const { maxAbsError, rmsError } = computeErrorStats(gpuOutput, cpu, DEFAULT_N);
    assertErrorInTolerance(maxAbsError);

    return {
      status: 'KERNEL_BENCH_DONE',
      packUrl: pack.manifestUrl,
      modelName: pack.manifest.model.name,
      adapterInfo,
      weightTensor: weightTensorName,
      biasTensor: biasTensorName,
      variant,
      k: DEFAULT_K,
      n: DEFAULT_N,
      warmup,
      iterations,
      packLoadMs: pack.elapsedMs,
      uploadSetupMs,
      dispatchLoopMs,
      dispatchLoopAvgMs: dispatchLoopMs / iterations,
      readbackSyncedMs,
      endToEndMs: nowMs() - totalStarted,
      maxAbsError,
      rmsError,
      outputSample: Array.from(gpuOutput.slice(0, 8)),
    };
  } finally {
    for (const buffer of buffers) buffer.destroy?.();
  }
}

export async function runLc0WebMatmulAddOrtBenchmark(options: Lc0WebMatmulAddOrtBenchmarkOptions): Promise<Lc0WebMatmulAddOrtBenchmarkResult> {
  const weightTensorName = options.weightTensorName ?? DEFAULT_WEIGHT_TENSOR;
  const biasTensorName = options.biasTensorName ?? DEFAULT_BIAS_TENSOR;
  const warmup = clampInteger(options.warmup, 5, 0, 100);
  const iterations = clampInteger(options.iterations, 25, 1, 1000);
  const pack = await loadLc0WebModelPack(options.packUrl, {
    verifyShards: options.verifyShards ?? true,
    tensorNames: [weightTensorName, biasTensorName],
  });
  const weight = pack.tensors.get(weightTensorName);
  const bias = pack.tensors.get(biasTensorName);
  if (!weight || !bias) throw new Error('lc0web ORT benchmark tensors were not loaded');
  assertTensorShapeAndBytes(weight, [DEFAULT_K, DEFAULT_N], 2, 'weight');
  assertTensorShapeAndBytes(bias, [DEFAULT_N], 2, 'bias');
  if (weight.info.dtype !== 'f16' || bias.info.dtype !== 'f16') {
    throw new Error(`lc0web ORT benchmark expects f16 tensors, got ${weight.info.dtype}/${bias.info.dtype}`);
  }

  const input = makeInputVector(DEFAULT_K);
  const cpu = cpuMatmulAdd(input, weight.bytes, bias.bytes, DEFAULT_K, DEFAULT_N);
  const modelBuildStarted = nowMs();
  const weightF32 = f16BytesToF32Array(weight.bytes, DEFAULT_K * DEFAULT_N);
  const biasF32 = f16BytesToF32Array(bias.bytes, DEFAULT_N);
  const tinyOnnx = createTinyMatmulAddOnnxForTest(weightF32, biasF32);
  const modelBuildMs = nowMs() - modelBuildStarted;

  const sessionStarted = nowMs();
  const session = await ort.createOrtSession(tinyOnnx);
  const sessionCreateMs = nowMs() - sessionStarted;
  const feeds = { input: new ort.Tensor('float32', input, [1, DEFAULT_K]) };

  let output: Float32Array<ArrayBufferLike> = new Float32Array(DEFAULT_N);
  for (let i = 0; i < warmup; i++) {
    const outputs = await session.run(feeds);
    output = outputs.output.data as Float32Array<ArrayBufferLike>;
  }
  const times: number[] = [];
  for (let i = 0; i < iterations; i++) {
    const started = nowMs();
    const outputs = await session.run(feeds);
    times.push(nowMs() - started);
    output = outputs.output.data as Float32Array<ArrayBufferLike>;
  }
  const { maxAbsError, rmsError } = computeErrorStats(output, cpu, DEFAULT_N);
  assertErrorInTolerance(maxAbsError);
  const avgMs = times.reduce((sum, value) => sum + value, 0) / times.length;

  return {
    status: 'ORT_BENCH_DONE',
    packUrl: pack.manifestUrl,
    modelName: pack.manifest.model.name,
    weightTensor: weightTensorName,
    biasTensor: biasTensorName,
    k: DEFAULT_K,
    n: DEFAULT_N,
    warmup,
    iterations,
    packLoadMs: pack.elapsedMs,
    modelBuildMs,
    sessionCreateMs,
    avgMs,
    minMs: Math.min(...times),
    maxMs: Math.max(...times),
    firstMs: times[0],
    timesMs: times,
    runsPerSecond: 1000 / avgMs,
    maxAbsError,
    rmsError,
    outputSample: Array.from(output.slice(0, 8)),
  };
}

const QKV_WGSL = `${WGSL_HEADER}
@group(0) @binding(4) var<storage, read> kWeightsF16: array<u32>;
@group(0) @binding(5) var<storage, read> kBiasF16: array<u32>;
@group(0) @binding(6) var<storage, read> vWeightsF16: array<u32>;
@group(0) @binding(7) var<storage, read> vBiasF16: array<u32>;

fn load_k_weight(index: u32) -> f32 {
  return pick_lane(kWeightsF16[index >> 1u], index);
}

fn load_k_bias(index: u32) -> f32 {
  return pick_lane(kBiasF16[index >> 1u], index);
}

fn load_v_weight(index: u32) -> f32 {
  return pick_lane(vWeightsF16[index >> 1u], index);
}

fn load_v_bias(index: u32) -> f32 {
  return pick_lane(vBiasF16[index >> 1u], index);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let col = gid.x;
  if (col >= 256u) { return; }
  var q_sum = load_bias(col);
  var k_sum = load_k_bias(col);
  var v_sum = load_v_bias(col);
  for (var row = 0u; row < 256u; row = row + 1u) {
    let x = inputVec[row];
    let base = row * 256u + col;
    q_sum = q_sum + x * load_weight(base);
    k_sum = k_sum + x * load_k_weight(base);
    v_sum = v_sum + x * load_v_weight(base);
  }
  outputVec[col] = q_sum;
  outputVec[256u + col] = k_sum;
  outputVec[512u + col] = v_sum;
}
`;

function createQkvPipeline(device: DeviceLike, buffers: {
  input: BufferLike;
  qWeight: BufferLike;
  qBias: BufferLike;
  kWeight: BufferLike;
  kBias: BufferLike;
  vWeight: BufferLike;
  vBias: BufferLike;
  output: BufferLike;
}): { pipeline: PipelineLike; bindGroup: unknown } {
  const module = device.createShaderModule({ label: 'lc0web qkv projection probe', code: QKV_WGSL });
  const pipeline = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'main' } }) as PipelineLike;
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: buffers.input } },
      { binding: 1, resource: { buffer: buffers.qWeight } },
      { binding: 2, resource: { buffer: buffers.qBias } },
      { binding: 3, resource: { buffer: buffers.output } },
      { binding: 4, resource: { buffer: buffers.kWeight } },
      { binding: 5, resource: { buffer: buffers.kBias } },
      { binding: 6, resource: { buffer: buffers.vWeight } },
      { binding: 7, resource: { buffer: buffers.vBias } },
    ],
  });
  return { pipeline, bindGroup };
}

function encodeQkvDispatches(device: DeviceLike, pipeline: PipelineLike, bindGroup: unknown, iterations: number): unknown {
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  for (let i = 0; i < iterations; i++) pass.dispatchWorkgroups(Math.ceil(DEFAULT_N / 64));
  pass.end();
  return encoder.finish();
}

async function readQkvOutputOnce(device: DeviceLike, outputBuffer: BufferLike, readbackBuffer: BufferLike): Promise<{ q: Float32Array<ArrayBufferLike>; k: Float32Array<ArrayBufferLike>; v: Float32Array<ArrayBufferLike> }> {
  const globals = gpuGlobals();
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(outputBuffer, 0, readbackBuffer, 0, DEFAULT_N * 3 * 4);
  device.queue.submit([encoder.finish()]);
  await readbackBuffer.mapAsync(globals.GPUMapMode!.READ);
  const all = new Float32Array(readbackBuffer.getMappedRange().slice(0));
  readbackBuffer.unmap();
  return { q: all.slice(0, DEFAULT_N), k: all.slice(DEFAULT_N, DEFAULT_N * 2), v: all.slice(DEFAULT_N * 2, DEFAULT_N * 3) };
}

async function runQkvOnce(device: DeviceLike, pipeline: PipelineLike, bindGroup: unknown, outputBuffer: BufferLike, readbackBuffer: BufferLike): Promise<{ q: Float32Array<ArrayBufferLike>; k: Float32Array<ArrayBufferLike>; v: Float32Array<ArrayBufferLike> }> {
  device.queue.submit([encodeQkvDispatches(device, pipeline, bindGroup, 1)]);
  return readQkvOutputOnce(device, outputBuffer, readbackBuffer);
}

export async function runLc0WebQkvProjectionProbe(options: Lc0WebQkvProjectionProbeOptions): Promise<Lc0WebQkvProjectionProbeResult> {
  const warmup = clampInteger(options.warmup, 2, 0, 50);
  const iterations = clampInteger(options.iterations, 10, 1, 1000);
  const { device, adapterInfo } = await requestDevice();
  const tensorNames = Object.values(DEFAULT_QKV_TENSORS);
  const pack = await loadLc0WebModelPack(options.packUrl, {
    verifyShards: options.verifyShards ?? true,
    tensorNames,
  });
  const tensors = Object.fromEntries(Object.entries(DEFAULT_QKV_TENSORS).map(([key, name]) => [key, pack.tensors.get(name)])) as Record<keyof typeof DEFAULT_QKV_TENSORS, Lc0WebTensorView | undefined>;
  for (const [key, tensor] of Object.entries(tensors)) {
    if (!tensor) throw new Error(`lc0web QKV projection tensor missing: ${key}`);
    const isBias = key.endsWith('Bias');
    assertTensorShapeAndBytes(tensor, isBias ? [DEFAULT_N] : [DEFAULT_K, DEFAULT_N], 2, key);
    if (tensor.info.dtype !== 'f16') throw new Error(`lc0web QKV projection expects f16 tensor ${tensor.info.name}, got ${tensor.info.dtype}`);
  }

  const globals = gpuGlobals();
  const usage = globals.GPUBufferUsage!;
  const input = makeInputVector(DEFAULT_K);
  const cpuQ = cpuMatmulAdd(input, tensors.qWeight!.bytes, tensors.qBias!.bytes, DEFAULT_K, DEFAULT_N);
  const cpuK = cpuMatmulAdd(input, tensors.kWeight!.bytes, tensors.kBias!.bytes, DEFAULT_K, DEFAULT_N);
  const cpuV = cpuMatmulAdd(input, tensors.vWeight!.bytes, tensors.vBias!.bytes, DEFAULT_K, DEFAULT_N);
  const liveBuffers: BufferLike[] = [];
  try {
    const inputBuffer = createStorageBuffer(device, input, usage.STORAGE | usage.COPY_DST);
    const qWeight = createStorageBuffer(device, tensors.qWeight!.bytes, usage.STORAGE | usage.COPY_DST);
    const qBias = createStorageBuffer(device, tensors.qBias!.bytes, usage.STORAGE | usage.COPY_DST);
    const kWeight = createStorageBuffer(device, tensors.kWeight!.bytes, usage.STORAGE | usage.COPY_DST);
    const kBias = createStorageBuffer(device, tensors.kBias!.bytes, usage.STORAGE | usage.COPY_DST);
    const vWeight = createStorageBuffer(device, tensors.vWeight!.bytes, usage.STORAGE | usage.COPY_DST);
    const vBias = createStorageBuffer(device, tensors.vBias!.bytes, usage.STORAGE | usage.COPY_DST);
    const outputBuffer = device.createBuffer({ size: DEFAULT_N * 3 * 4, usage: usage.STORAGE | usage.COPY_SRC });
    const readbackBuffer = device.createBuffer({ size: DEFAULT_N * 3 * 4, usage: usage.MAP_READ | usage.COPY_DST });
    liveBuffers.push(inputBuffer, qWeight, qBias, kWeight, kBias, vWeight, vBias, outputBuffer, readbackBuffer);
    const { pipeline, bindGroup } = createQkvPipeline(device, { input: inputBuffer, qWeight, qBias, kWeight, kBias, vWeight, vBias, output: outputBuffer });

    let outputs: { q: Float32Array<ArrayBufferLike>; k: Float32Array<ArrayBufferLike>; v: Float32Array<ArrayBufferLike> } = { q: new Float32Array(DEFAULT_N), k: new Float32Array(DEFAULT_N), v: new Float32Array(DEFAULT_N) };
    for (let i = 0; i < warmup; i++) outputs = await runQkvOnce(device, pipeline, bindGroup, outputBuffer, readbackBuffer);
    const times: number[] = [];
    for (let i = 0; i < iterations; i++) {
      const started = nowMs();
      outputs = await runQkvOnce(device, pipeline, bindGroup, outputBuffer, readbackBuffer);
      times.push(nowMs() - started);
    }
    const qErr = computeErrorStats(outputs.q, cpuQ, DEFAULT_N);
    const kErr = computeErrorStats(outputs.k, cpuK, DEFAULT_N);
    const vErr = computeErrorStats(outputs.v, cpuV, DEFAULT_N);
    assertErrorInTolerance(Math.max(qErr.maxAbsError, kErr.maxAbsError, vErr.maxAbsError));
    const avgMs = times.reduce((sum, value) => sum + value, 0) / times.length;
    return {
      status: 'QKV_DONE',
      packUrl: pack.manifestUrl,
      modelName: pack.manifest.model.name,
      adapterInfo,
      k: DEFAULT_K,
      n: DEFAULT_N,
      warmup,
      iterations,
      packLoadMs: pack.elapsedMs,
      avgMs,
      minMs: Math.min(...times),
      maxMs: Math.max(...times),
      firstMs: times[0],
      timesMs: times,
      maxAbsError: { q: qErr.maxAbsError, k: kErr.maxAbsError, v: vErr.maxAbsError },
      rmsError: { q: qErr.rmsError, k: kErr.rmsError, v: vErr.rmsError },
      outputSample: { q: Array.from(outputs.q.slice(0, 8)), k: Array.from(outputs.k.slice(0, 8)), v: Array.from(outputs.v.slice(0, 8)) },
    };
  } finally {
    for (const buffer of liveBuffers) buffer.destroy?.();
  }
}


export async function runLc0WebQkvProjectionBenchmark(options: Lc0WebQkvProjectionBenchmarkOptions): Promise<Lc0WebQkvProjectionBenchmarkResult> {
  const totalStarted = nowMs();
  const warmup = clampInteger(options.warmup, 10, 0, 1000);
  const iterations = clampInteger(options.iterations, 1000, 1, 100_000);
  const { device, adapterInfo } = await requestDevice();
  const tensorNames = Object.values(DEFAULT_QKV_TENSORS);
  const pack = await loadLc0WebModelPack(options.packUrl, {
    verifyShards: options.verifyShards ?? true,
    tensorNames,
  });
  const tensors = Object.fromEntries(Object.entries(DEFAULT_QKV_TENSORS).map(([key, name]) => [key, pack.tensors.get(name)])) as Record<keyof typeof DEFAULT_QKV_TENSORS, Lc0WebTensorView | undefined>;
  for (const [key, tensor] of Object.entries(tensors)) {
    if (!tensor) throw new Error(`lc0web QKV benchmark tensor missing: ${key}`);
    const isBias = key.endsWith('Bias');
    assertTensorShapeAndBytes(tensor, isBias ? [DEFAULT_N] : [DEFAULT_K, DEFAULT_N], 2, key);
    if (tensor.info.dtype !== 'f16') throw new Error(`lc0web QKV benchmark expects f16 tensor ${tensor.info.name}, got ${tensor.info.dtype}`);
  }

  const globals = gpuGlobals();
  const usage = globals.GPUBufferUsage!;
  const input = makeInputVector(DEFAULT_K);
  const cpuQ = cpuMatmulAdd(input, tensors.qWeight!.bytes, tensors.qBias!.bytes, DEFAULT_K, DEFAULT_N);
  const cpuK = cpuMatmulAdd(input, tensors.kWeight!.bytes, tensors.kBias!.bytes, DEFAULT_K, DEFAULT_N);
  const cpuV = cpuMatmulAdd(input, tensors.vWeight!.bytes, tensors.vBias!.bytes, DEFAULT_K, DEFAULT_N);
  const liveBuffers: BufferLike[] = [];
  try {
    const setupStarted = nowMs();
    const inputBuffer = createStorageBuffer(device, input, usage.STORAGE | usage.COPY_DST);
    const qWeight = createStorageBuffer(device, tensors.qWeight!.bytes, usage.STORAGE | usage.COPY_DST);
    const qBias = createStorageBuffer(device, tensors.qBias!.bytes, usage.STORAGE | usage.COPY_DST);
    const kWeight = createStorageBuffer(device, tensors.kWeight!.bytes, usage.STORAGE | usage.COPY_DST);
    const kBias = createStorageBuffer(device, tensors.kBias!.bytes, usage.STORAGE | usage.COPY_DST);
    const vWeight = createStorageBuffer(device, tensors.vWeight!.bytes, usage.STORAGE | usage.COPY_DST);
    const vBias = createStorageBuffer(device, tensors.vBias!.bytes, usage.STORAGE | usage.COPY_DST);
    const outputBuffer = device.createBuffer({ size: DEFAULT_N * 3 * 4, usage: usage.STORAGE | usage.COPY_SRC });
    const readbackBuffer = device.createBuffer({ size: DEFAULT_N * 3 * 4, usage: usage.MAP_READ | usage.COPY_DST });
    liveBuffers.push(inputBuffer, qWeight, qBias, kWeight, kBias, vWeight, vBias, outputBuffer, readbackBuffer);
    const { pipeline, bindGroup } = createQkvPipeline(device, { input: inputBuffer, qWeight, qBias, kWeight, kBias, vWeight, vBias, output: outputBuffer });
    const uploadSetupMs = nowMs() - setupStarted;

    if (warmup > 0) {
      device.queue.submit([encodeQkvDispatches(device, pipeline, bindGroup, warmup)]);
      await device.queue.onSubmittedWorkDone?.();
    }

    const dispatchStarted = nowMs();
    device.queue.submit([encodeQkvDispatches(device, pipeline, bindGroup, iterations)]);
    const dispatchLoopMs = nowMs() - dispatchStarted;

    const readbackStarted = nowMs();
    const outputs = await readQkvOutputOnce(device, outputBuffer, readbackBuffer);
    const readbackSyncedMs = nowMs() - readbackStarted;
    const qErr = computeErrorStats(outputs.q, cpuQ, DEFAULT_N);
    const kErr = computeErrorStats(outputs.k, cpuK, DEFAULT_N);
    const vErr = computeErrorStats(outputs.v, cpuV, DEFAULT_N);
    assertErrorInTolerance(Math.max(qErr.maxAbsError, kErr.maxAbsError, vErr.maxAbsError));

    return {
      status: 'QKV_BENCH_DONE',
      packUrl: pack.manifestUrl,
      modelName: pack.manifest.model.name,
      adapterInfo,
      k: DEFAULT_K,
      n: DEFAULT_N,
      warmup,
      iterations,
      packLoadMs: pack.elapsedMs,
      uploadSetupMs,
      dispatchLoopMs,
      dispatchLoopAvgMs: dispatchLoopMs / iterations,
      readbackSyncedMs,
      endToEndMs: nowMs() - totalStarted,
      maxAbsError: { q: qErr.maxAbsError, k: kErr.maxAbsError, v: vErr.maxAbsError },
      rmsError: { q: qErr.rmsError, k: kErr.rmsError, v: vErr.rmsError },
      outputSample: { q: Array.from(outputs.q.slice(0, 8)), k: Array.from(outputs.k.slice(0, 8)), v: Array.from(outputs.v.slice(0, 8)) },
    };
  } finally {
    for (const buffer of liveBuffers) buffer.destroy?.();
  }
}

const ATTENTION_SCORE_WGSL = `
@group(0) @binding(0) var<storage, read> qVec: array<f32>;
@group(0) @binding(1) var<storage, read> kVec: array<f32>;
@group(0) @binding(2) var<storage, read> scaleF16: array<u32>;
@group(0) @binding(3) var<storage, read> smolgenBiasVec: array<f32>;
@group(0) @binding(4) var<storage, read_write> scoreVec: array<f32>;

fn pick_lane(word: u32, index: u32) -> f32 {
  let pair = unpack2x16float(word);
  return select(pair.x, pair.y, (index & 1u) == 1u);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let col = gid.x;
  let row = gid.y;
  let head = gid.z;
  if (row >= 64u || col >= 64u || head >= 8u) { return; }
  let channel_offset = head * 32u;
  var sum = 0.0;
  for (var channel = 0u; channel < 32u; channel = channel + 1u) {
    sum = sum + qVec[row * 256u + channel_offset + channel] * kVec[col * 256u + channel_offset + channel];
  }
  let index = (head * 64u + row) * 64u + col;
  scoreVec[index] = sum * pick_lane(scaleF16[0], 0u) + smolgenBiasVec[index];
}
`;

function createAttentionScorePipeline(device: DeviceLike, buffers: { q: BufferLike; k: BufferLike; scale: BufferLike; smolgenBias: BufferLike; output: BufferLike }): { pipeline: PipelineLike; bindGroup: unknown } {
  const module = device.createShaderModule({ label: 'lc0web attention score probe with smolgen bias', code: ATTENTION_SCORE_WGSL });
  const pipeline = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'main' } }) as PipelineLike;
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: buffers.q } },
      { binding: 1, resource: { buffer: buffers.k } },
      { binding: 2, resource: { buffer: buffers.scale } },
      { binding: 3, resource: { buffer: buffers.smolgenBias } },
      { binding: 4, resource: { buffer: buffers.output } },
    ],
  });
  return { pipeline, bindGroup };
}

function encodeAttentionScoreDispatches(device: DeviceLike, pipeline: PipelineLike, bindGroup: unknown, iterations: number): unknown {
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  for (let i = 0; i < iterations; i++) pass.dispatchWorkgroups(Math.ceil(DEFAULT_TOKENS / 8), Math.ceil(DEFAULT_TOKENS / 8), DEFAULT_HEADS);
  pass.end();
  return encoder.finish();
}

async function readAttentionScoreOutputOnce(device: DeviceLike, outputBuffer: BufferLike, readbackBuffer: BufferLike): Promise<Float32Array<ArrayBufferLike>> {
  const globals = gpuGlobals();
  const bytes = DEFAULT_HEADS * DEFAULT_TOKENS * DEFAULT_TOKENS * 4;
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(outputBuffer, 0, readbackBuffer, 0, bytes);
  device.queue.submit([encoder.finish()]);
  await readbackBuffer.mapAsync(globals.GPUMapMode!.READ);
  const output = new Float32Array(readbackBuffer.getMappedRange().slice(0));
  readbackBuffer.unmap();
  return output;
}

type Encoder0SmolgenTensors = {
  compressWeight: Lc0WebTensorView;
  dense1Weight: Lc0WebTensorView;
  dense1Bias: Lc0WebTensorView;
  ln1Scale: Lc0WebTensorView;
  ln1Bias: Lc0WebTensorView;
  dense2Weight: Lc0WebTensorView;
  dense2Bias: Lc0WebTensorView;
  ln2Scale: Lc0WebTensorView;
  ln2Bias: Lc0WebTensorView;
  smolgenWeight: Lc0WebTensorView;
};

type AttentionScoreInputs = {
  qWeight: Lc0WebTensorView;
  qBias: Lc0WebTensorView;
  kWeight: Lc0WebTensorView;
  kBias: Lc0WebTensorView;
  scale: Lc0WebTensorView;
  smolgen: Encoder0SmolgenTensors;
};

function loadEncoder0SmolgenTensors(pack: Awaited<ReturnType<typeof loadLc0WebModelPack>>, tensorNames: Record<keyof typeof DEFAULT_SMOLGEN_TENSORS, string> = DEFAULT_SMOLGEN_TENSORS): Encoder0SmolgenTensors {
  const tensors = Object.fromEntries(Object.entries(tensorNames).map(([key, name]) => [key, pack.tensors.get(name)])) as Record<keyof typeof DEFAULT_SMOLGEN_TENSORS, Lc0WebTensorView | undefined>;
  for (const [key, tensor] of Object.entries(tensors)) if (!tensor) throw new Error(`lc0web encoder0 smolgen tensor missing: ${key}`);
  assertTensorShapeAndBytes(tensors.compressWeight!, [DEFAULT_N, DEFAULT_SMOLGEN_COMPRESSED], 2, 'smolgen.compressWeight');
  assertTensorShapeAndBytes(tensors.dense1Weight!, [DEFAULT_SMOLGEN_FLAT, DEFAULT_SMOLGEN_HIDDEN], 2, 'smolgen.dense1Weight');
  assertTensorShapeAndBytes(tensors.dense1Bias!, [DEFAULT_SMOLGEN_HIDDEN], 2, 'smolgen.dense1Bias');
  assertTensorShapeAndBytes(tensors.ln1Scale!, [DEFAULT_SMOLGEN_HIDDEN], 2, 'smolgen.ln1Scale');
  assertTensorShapeAndBytes(tensors.ln1Bias!, [DEFAULT_SMOLGEN_HIDDEN], 2, 'smolgen.ln1Bias');
  assertTensorShapeAndBytes(tensors.dense2Weight!, [DEFAULT_SMOLGEN_HIDDEN, DEFAULT_SMOLGEN_FLAT], 2, 'smolgen.dense2Weight');
  assertTensorShapeAndBytes(tensors.dense2Bias!, [DEFAULT_SMOLGEN_FLAT], 2, 'smolgen.dense2Bias');
  assertTensorShapeAndBytes(tensors.ln2Scale!, [DEFAULT_SMOLGEN_FLAT], 2, 'smolgen.ln2Scale');
  assertTensorShapeAndBytes(tensors.ln2Bias!, [DEFAULT_SMOLGEN_FLAT], 2, 'smolgen.ln2Bias');
  assertTensorShapeAndBytes(tensors.smolgenWeight!, [DEFAULT_SMOLGEN_HIDDEN, DEFAULT_TOKENS * DEFAULT_TOKENS], 2, 'smolgen.smolgenWeight');
  for (const tensor of Object.values(tensors)) {
    if (tensor!.info.dtype !== 'f16') throw new Error(`lc0web encoder0 smolgen expects f16 tensor ${tensor!.info.name}, got ${tensor!.info.dtype}`);
  }
  return tensors as Encoder0SmolgenTensors;
}

function loadAttentionScoreInputs(pack: Awaited<ReturnType<typeof loadLc0WebModelPack>>, tensorNames: Pick<Lc0WebEncoderBlockTensorNames, 'qkv' | 'scaleTensor' | 'smolgen'> = { qkv: DEFAULT_QKV_TENSORS, scaleTensor: DEFAULT_SCALE_TENSOR, smolgen: DEFAULT_SMOLGEN_TENSORS }): AttentionScoreInputs {
  const qWeight = pack.tensors.get(tensorNames.qkv.qWeight);
  const qBias = pack.tensors.get(tensorNames.qkv.qBias);
  const kWeight = pack.tensors.get(tensorNames.qkv.kWeight);
  const kBias = pack.tensors.get(tensorNames.qkv.kBias);
  const scale = pack.tensors.get(tensorNames.scaleTensor);
  if (!qWeight || !qBias || !kWeight || !kBias || !scale) throw new Error('lc0web attention score tensors were not loaded');
  assertTensorShapeAndBytes(qWeight, [DEFAULT_K, DEFAULT_N], 2, 'qWeight');
  assertTensorShapeAndBytes(qBias, [DEFAULT_N], 2, 'qBias');
  assertTensorShapeAndBytes(kWeight, [DEFAULT_K, DEFAULT_N], 2, 'kWeight');
  assertTensorShapeAndBytes(kBias, [DEFAULT_N], 2, 'kBias');
  assertTensorShapeAndBytes(scale, [1], 2, 'scale');
  for (const tensor of [qWeight, qBias, kWeight, kBias, scale]) {
    if (tensor.info.dtype !== 'f16') throw new Error(`lc0web attention score expects f16 tensor ${tensor.info.name}, got ${tensor.info.dtype}`);
  }
  return { qWeight, qBias, kWeight, kBias, scale, smolgen: loadEncoder0SmolgenTensors(pack, tensorNames.smolgen) };
}

function paddedF16ScalarBytes(bytes: Uint8Array): Uint8Array {
  const out = new Uint8Array(4);
  out[0] = bytes[0];
  out[1] = bytes[1];
  return out;
}

function cpuEncoder0SmolgenBias(input: Float32Array<ArrayBufferLike>, tensors: Encoder0SmolgenTensors): Float32Array<ArrayBufferLike> {
  const compressed = cpuProjectTokensNoBias(input, tensors.compressWeight.bytes, DEFAULT_TOKENS, DEFAULT_N, DEFAULT_SMOLGEN_COMPRESSED);
  const dense1 = cpuMatmulAddVector(compressed, tensors.dense1Weight.bytes, tensors.dense1Bias.bytes, DEFAULT_SMOLGEN_FLAT, DEFAULT_SMOLGEN_HIDDEN);
  const ln1 = cpuLayerNormVector(cpuSwish(dense1), tensors.ln1Scale.bytes, tensors.ln1Bias.bytes, DEFAULT_SMOLGEN_EPSILON);
  const dense2 = cpuMatmulAddVector(ln1, tensors.dense2Weight.bytes, tensors.dense2Bias.bytes, DEFAULT_SMOLGEN_HIDDEN, DEFAULT_SMOLGEN_FLAT);
  const ln2 = cpuLayerNormVector(cpuSwish(dense2), tensors.ln2Scale.bytes, tensors.ln2Bias.bytes, DEFAULT_SMOLGEN_EPSILON);
  const bias = new Float32Array(DEFAULT_HEADS * DEFAULT_TOKENS * DEFAULT_TOKENS);
  for (let head = 0; head < DEFAULT_HEADS; head++) {
    const headInput = ln2.subarray(head * DEFAULT_SMOLGEN_HIDDEN, (head + 1) * DEFAULT_SMOLGEN_HIDDEN);
    bias.set(cpuMatmulVectorNoBias(headInput, tensors.smolgenWeight.bytes, DEFAULT_SMOLGEN_HIDDEN, DEFAULT_TOKENS * DEFAULT_TOKENS), head * DEFAULT_TOKENS * DEFAULT_TOKENS);
  }
  return bias;
}

function addElementwise(a: Float32Array<ArrayBufferLike>, b: Float32Array<ArrayBufferLike>): Float32Array<ArrayBufferLike> {
  const out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = a[i] + b[i];
  return out;
}

function buildAttentionScoreReference(tensors: ReturnType<typeof loadAttentionScoreInputs>, input: Float32Array<ArrayBufferLike> = makeInputTokenMatrix(DEFAULT_TOKENS, DEFAULT_K)): { input: Float32Array<ArrayBufferLike>; q: Float32Array<ArrayBufferLike>; k: Float32Array<ArrayBufferLike>; scale: number; qkScores: Float32Array<ArrayBufferLike>; smolgenBias: Float32Array<ArrayBufferLike>; scores: Float32Array<ArrayBufferLike> } {
  const q = cpuProjectTokens(input, tensors.qWeight.bytes, tensors.qBias.bytes, DEFAULT_TOKENS, DEFAULT_K, DEFAULT_N);
  const k = cpuProjectTokens(input, tensors.kWeight.bytes, tensors.kBias.bytes, DEFAULT_TOKENS, DEFAULT_K, DEFAULT_N);
  const scale = readF16At(tensors.scale.bytes, 0);
  const qkScores = cpuAttentionScores(q, k, scale, DEFAULT_TOKENS, DEFAULT_N, DEFAULT_HEADS);
  const smolgenBias = cpuEncoder0SmolgenBias(input, tensors.smolgen);
  const scores = addElementwise(qkScores, smolgenBias);
  return { input, q, k, scale, qkScores, smolgenBias, scores };
}

export async function runLc0WebAttentionScoreBenchmark(options: Lc0WebAttentionScoreBenchmarkOptions): Promise<Lc0WebAttentionScoreBenchmarkResult> {
  const totalStarted = nowMs();
  const warmup = clampInteger(options.warmup, 10, 0, 1000);
  const iterations = clampInteger(options.iterations, 1000, 1, 100_000);
  const { device, adapterInfo } = await requestDevice();
  const pack = await loadLc0WebModelPack(options.packUrl, {
    verifyShards: options.verifyShards ?? true,
    tensorNames: [DEFAULT_QKV_TENSORS.qWeight, DEFAULT_QKV_TENSORS.qBias, DEFAULT_QKV_TENSORS.kWeight, DEFAULT_QKV_TENSORS.kBias, DEFAULT_SCALE_TENSOR, ...Object.values(DEFAULT_SMOLGEN_TENSORS)],
  });
  const tensors = loadAttentionScoreInputs(pack);
  const reference = buildAttentionScoreReference(tensors);
  const globals = gpuGlobals();
  const usage = globals.GPUBufferUsage!;
  const buffers: BufferLike[] = [];
  try {
    const setupStarted = nowMs();
    const qBuffer = createStorageBuffer(device, reference.q, usage.STORAGE | usage.COPY_DST);
    const kBuffer = createStorageBuffer(device, reference.k, usage.STORAGE | usage.COPY_DST);
    const scaleBuffer = createStorageBuffer(device, paddedF16ScalarBytes(tensors.scale.bytes), usage.STORAGE | usage.COPY_DST);
    const smolgenBiasBuffer = createStorageBuffer(device, reference.smolgenBias, usage.STORAGE | usage.COPY_DST);
    const outputBuffer = device.createBuffer({ size: DEFAULT_HEADS * DEFAULT_TOKENS * DEFAULT_TOKENS * 4, usage: usage.STORAGE | usage.COPY_SRC });
    const readbackBuffer = device.createBuffer({ size: DEFAULT_HEADS * DEFAULT_TOKENS * DEFAULT_TOKENS * 4, usage: usage.MAP_READ | usage.COPY_DST });
    buffers.push(qBuffer, kBuffer, scaleBuffer, smolgenBiasBuffer, outputBuffer, readbackBuffer);
    const { pipeline, bindGroup } = createAttentionScorePipeline(device, { q: qBuffer, k: kBuffer, scale: scaleBuffer, smolgenBias: smolgenBiasBuffer, output: outputBuffer });
    const uploadSetupMs = nowMs() - setupStarted;

    if (warmup > 0) {
      device.queue.submit([encodeAttentionScoreDispatches(device, pipeline, bindGroup, warmup)]);
      await device.queue.onSubmittedWorkDone?.();
    }
    const dispatchStarted = nowMs();
    device.queue.submit([encodeAttentionScoreDispatches(device, pipeline, bindGroup, iterations)]);
    const dispatchLoopMs = nowMs() - dispatchStarted;
    const readbackStarted = nowMs();
    const output = await readAttentionScoreOutputOnce(device, outputBuffer, readbackBuffer);
    const readbackSyncedMs = nowMs() - readbackStarted;
    const { maxAbsError, rmsError } = computeErrorStats(output, reference.scores, output.length);
    assertErrorInTolerance(maxAbsError);
    return {
      status: 'ATTENTION_SCORE_BENCH_DONE',
      packUrl: pack.manifestUrl,
      modelName: pack.manifest.model.name,
      adapterInfo,
      tokens: DEFAULT_TOKENS,
      channels: DEFAULT_N,
      heads: DEFAULT_HEADS,
      headDim: DEFAULT_HEAD_DIM,
      scale: reference.scale,
      smolgen: { enabled: true, epsilon: DEFAULT_SMOLGEN_EPSILON },
      warmup,
      iterations,
      packLoadMs: pack.elapsedMs,
      uploadSetupMs,
      dispatchLoopMs,
      dispatchLoopAvgMs: dispatchLoopMs / iterations,
      readbackSyncedMs,
      endToEndMs: nowMs() - totalStarted,
      maxAbsError,
      rmsError,
      outputSample: Array.from(output.slice(0, 8)),
    };
  } finally {
    for (const buffer of buffers) buffer.destroy?.();
  }
}

export async function runLc0WebAttentionScoreOrtBenchmark(options: Lc0WebAttentionScoreBenchmarkOptions): Promise<Lc0WebAttentionScoreOrtBenchmarkResult> {
  const warmup = clampInteger(options.warmup, 5, 0, 100);
  const iterations = clampInteger(options.iterations, 25, 1, 1000);
  const pack = await loadLc0WebModelPack(options.packUrl, {
    verifyShards: options.verifyShards ?? true,
    tensorNames: [DEFAULT_QKV_TENSORS.qWeight, DEFAULT_QKV_TENSORS.qBias, DEFAULT_QKV_TENSORS.kWeight, DEFAULT_QKV_TENSORS.kBias, DEFAULT_SCALE_TENSOR, ...Object.values(DEFAULT_SMOLGEN_TENSORS)],
  });
  const tensors = loadAttentionScoreInputs(pack);
  const reference = buildAttentionScoreReference(tensors);
  const modelBuildStarted = nowMs();
  const tinyOnnx = createTinyAttentionScoreOnnxForTest(reference.scale, true);
  const modelBuildMs = nowMs() - modelBuildStarted;
  const sessionStarted = nowMs();
  const session = await ort.createOrtSession(tinyOnnx);
  const sessionCreateMs = nowMs() - sessionStarted;
  const feeds = { q: new ort.Tensor('float32', packHeadsQ(reference.q, DEFAULT_TOKENS, DEFAULT_N, DEFAULT_HEADS), [DEFAULT_HEADS, DEFAULT_TOKENS, DEFAULT_HEAD_DIM]), kt: new ort.Tensor('float32', packHeadsKt(reference.k, DEFAULT_TOKENS, DEFAULT_N, DEFAULT_HEADS), [DEFAULT_HEADS, DEFAULT_HEAD_DIM, DEFAULT_TOKENS]), bias: new ort.Tensor('float32', reference.smolgenBias, [DEFAULT_HEADS, DEFAULT_TOKENS, DEFAULT_TOKENS]) };
  let output: Float32Array<ArrayBufferLike> = new Float32Array(DEFAULT_TOKENS * DEFAULT_TOKENS);
  for (let i = 0; i < warmup; i++) {
    const outputs = await session.run(feeds);
    output = outputs.output.data as Float32Array<ArrayBufferLike>;
  }
  const times: number[] = [];
  for (let i = 0; i < iterations; i++) {
    const started = nowMs();
    const outputs = await session.run(feeds);
    times.push(nowMs() - started);
    output = outputs.output.data as Float32Array<ArrayBufferLike>;
  }
  const { maxAbsError, rmsError } = computeErrorStats(output, reference.scores, output.length);
  assertErrorInTolerance(maxAbsError);
  const avgMs = times.reduce((sum, value) => sum + value, 0) / times.length;
  return {
    status: 'ATTENTION_SCORE_ORT_BENCH_DONE',
    packUrl: pack.manifestUrl,
    modelName: pack.manifest.model.name,
    tokens: DEFAULT_TOKENS,
    channels: DEFAULT_N,
    heads: DEFAULT_HEADS,
    headDim: DEFAULT_HEAD_DIM,
    scale: reference.scale,
    smolgen: { enabled: true, epsilon: DEFAULT_SMOLGEN_EPSILON },
    warmup,
    iterations,
    packLoadMs: pack.elapsedMs,
    modelBuildMs,
    sessionCreateMs,
    avgMs,
    minMs: Math.min(...times),
    maxMs: Math.max(...times),
    firstMs: times[0],
    timesMs: times,
    runsPerSecond: 1000 / avgMs,
    maxAbsError,
    rmsError,
    outputSample: Array.from(output.slice(0, 8)),
  };
}

export interface Lc0WebSoftmaxBenchmarkOptions {
  packUrl: string;
  iterations?: number;
  warmup?: number;
  verifyShards?: boolean;
}

export interface Lc0WebSoftmaxBenchmarkResult {
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
}

function cpuSoftmaxRows(input: Float32Array<ArrayBufferLike>, rows: number, cols: number): Float32Array<ArrayBufferLike> {
  const out = new Float32Array(input.length);
  for (let row = 0; row < rows; row++) {
    const base = row * cols;
    let maxValue = -Infinity;
    for (let col = 0; col < cols; col++) maxValue = Math.max(maxValue, input[base + col]);
    let sum = 0;
    for (let col = 0; col < cols; col++) {
      const value = Math.exp(input[base + col] - maxValue);
      out[base + col] = value;
      sum += value;
    }
    const inv = 1 / sum;
    for (let col = 0; col < cols; col++) out[base + col] *= inv;
  }
  return out;
}

const SOFTMAX_WGSL = `
@group(0) @binding(0) var<storage, read> inputScores: array<f32>;
@group(0) @binding(1) var<storage, read_write> outputProbs: array<f32>;

var<workgroup> scratch: array<f32, 64>;

fn reduce_max(col: u32) {
  for (var stride = 32u; stride > 0u; stride = stride / 2u) {
    if (col < stride) {
      scratch[col] = max(scratch[col], scratch[col + stride]);
    }
    workgroupBarrier();
  }
}

fn reduce_sum(col: u32) {
  for (var stride = 32u; stride > 0u; stride = stride / 2u) {
    if (col < stride) {
      scratch[col] = scratch[col] + scratch[col + stride];
    }
    workgroupBarrier();
  }
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let row = wid.x;
  let col = lid.x;
  if (row >= 512u) { return; }
  let index = row * 64u + col;
  scratch[col] = inputScores[index];
  workgroupBarrier();
  reduce_max(col);
  let max_value = scratch[0];

  let value = exp(inputScores[index] - max_value);
  outputProbs[index] = value;
  scratch[col] = value;
  workgroupBarrier();
  reduce_sum(col);
  outputProbs[index] = value / scratch[0];
}
`;

function createSoftmaxPipeline(device: DeviceLike, buffers: { input: BufferLike; output: BufferLike }): { pipeline: PipelineLike; bindGroup: unknown } {
  const module = device.createShaderModule({ label: 'lc0web attention softmax probe', code: SOFTMAX_WGSL });
  const pipeline = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'main' } }) as PipelineLike;
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: buffers.input } },
      { binding: 1, resource: { buffer: buffers.output } },
    ],
  });
  return { pipeline, bindGroup };
}

function encodeSoftmaxDispatches(device: DeviceLike, pipeline: PipelineLike, bindGroup: unknown, rows: number, iterations: number): unknown {
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  for (let i = 0; i < iterations; i++) pass.dispatchWorkgroups(rows);
  pass.end();
  return encoder.finish();
}

async function readF32OutputOnce(device: DeviceLike, outputBuffer: BufferLike, readbackBuffer: BufferLike, elements: number): Promise<Float32Array<ArrayBufferLike>> {
  const globals = gpuGlobals();
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(outputBuffer, 0, readbackBuffer, 0, elements * 4);
  device.queue.submit([encoder.finish()]);
  await readbackBuffer.mapAsync(globals.GPUMapMode!.READ);
  const output = new Float32Array(readbackBuffer.getMappedRange().slice(0));
  readbackBuffer.unmap();
  return output;
}

async function measureGpuTimestampMs(device: DeviceLike, commandBuffers: unknown[]): Promise<number | undefined> {
  const globals = gpuGlobals();
  const usage = globals.GPUBufferUsage!;
  if (!device.createQuerySet || !usage.QUERY_RESOLVE) return undefined;
  const startEncoder = device.createCommandEncoder();
  const finishEncoder = device.createCommandEncoder();
  if (!startEncoder.writeTimestamp || !finishEncoder.writeTimestamp || !finishEncoder.resolveQuerySet) return undefined;
  const querySet = device.createQuerySet({ type: 'timestamp', count: 2 });
  const resolveBuffer = device.createBuffer({ size: 16, usage: usage.QUERY_RESOLVE | usage.COPY_SRC });
  const readbackBuffer = device.createBuffer({ size: 16, usage: usage.MAP_READ | usage.COPY_DST });
  try {
    startEncoder.writeTimestamp(querySet, 0);
    finishEncoder.writeTimestamp(querySet, 1);
    finishEncoder.resolveQuerySet(querySet, 0, 2, resolveBuffer, 0);
    finishEncoder.copyBufferToBuffer(resolveBuffer, 0, readbackBuffer, 0, 16);
    device.queue.submit([startEncoder.finish(), ...commandBuffers, finishEncoder.finish()]);
    await readbackBuffer.mapAsync(globals.GPUMapMode!.READ);
    const timestamps = new BigUint64Array(readbackBuffer.getMappedRange().slice(0));
    readbackBuffer.unmap();
    const elapsedNs = timestamps[1] > timestamps[0] ? timestamps[1] - timestamps[0] : 0n;
    return Number(elapsedNs) / 1_000_000;
  } finally {
    querySet.destroy?.();
    resolveBuffer.destroy?.();
    readbackBuffer.destroy?.();
  }
}

export async function runLc0WebSoftmaxBenchmark(options: Lc0WebSoftmaxBenchmarkOptions): Promise<Lc0WebSoftmaxBenchmarkResult> {
  const totalStarted = nowMs();
  const warmup = clampInteger(options.warmup, 10, 0, 1000);
  const iterations = clampInteger(options.iterations, 1000, 1, 100_000);
  const { device, adapterInfo } = await requestDevice();
  const pack = await loadLc0WebModelPack(options.packUrl, {
    verifyShards: options.verifyShards ?? true,
    tensorNames: [DEFAULT_QKV_TENSORS.qWeight, DEFAULT_QKV_TENSORS.qBias, DEFAULT_QKV_TENSORS.kWeight, DEFAULT_QKV_TENSORS.kBias, DEFAULT_SCALE_TENSOR, ...Object.values(DEFAULT_SMOLGEN_TENSORS)],
  });
  const tensors = loadAttentionScoreInputs(pack);
  const reference = buildAttentionScoreReference(tensors);
  const rows = DEFAULT_HEADS * DEFAULT_TOKENS;
  const elements = rows * DEFAULT_TOKENS;
  const cpu = cpuSoftmaxRows(reference.scores, rows, DEFAULT_TOKENS);
  const globals = gpuGlobals();
  const usage = globals.GPUBufferUsage!;
  const buffers: BufferLike[] = [];
  try {
    const setupStarted = nowMs();
    const inputBuffer = createStorageBuffer(device, reference.scores, usage.STORAGE | usage.COPY_DST);
    const outputBuffer = device.createBuffer({ size: elements * 4, usage: usage.STORAGE | usage.COPY_SRC });
    const readbackBuffer = device.createBuffer({ size: elements * 4, usage: usage.MAP_READ | usage.COPY_DST });
    buffers.push(inputBuffer, outputBuffer, readbackBuffer);
    const { pipeline, bindGroup } = createSoftmaxPipeline(device, { input: inputBuffer, output: outputBuffer });
    const uploadSetupMs = nowMs() - setupStarted;

    if (warmup > 0) {
      device.queue.submit([encodeSoftmaxDispatches(device, pipeline, bindGroup, rows, warmup)]);
      await device.queue.onSubmittedWorkDone?.();
    }
    const dispatchStarted = nowMs();
    device.queue.submit([encodeSoftmaxDispatches(device, pipeline, bindGroup, rows, iterations)]);
    const dispatchLoopMs = nowMs() - dispatchStarted;
    const readbackStarted = nowMs();
    const output = await readF32OutputOnce(device, outputBuffer, readbackBuffer, elements);
    const readbackSyncedMs = nowMs() - readbackStarted;
    const { maxAbsError, rmsError } = computeErrorStats(output, cpu, elements);
    assertErrorInTolerance(maxAbsError);
    return {
      status: 'SOFTMAX_BENCH_DONE',
      packUrl: pack.manifestUrl,
      modelName: pack.manifest.model.name,
      adapterInfo,
      tokens: DEFAULT_TOKENS,
      heads: DEFAULT_HEADS,
      rows,
      warmup,
      iterations,
      packLoadMs: pack.elapsedMs,
      uploadSetupMs,
      dispatchLoopMs,
      dispatchLoopAvgMs: dispatchLoopMs / iterations,
      readbackSyncedMs,
      endToEndMs: nowMs() - totalStarted,
      maxAbsError,
      rmsError,
      outputSample: Array.from(output.slice(0, 8)),
    };
  } finally {
    for (const buffer of buffers) buffer.destroy?.();
  }
}

export interface Lc0WebAttentionValueBenchmarkOptions {
  packUrl: string;
  iterations?: number;
  warmup?: number;
  verifyShards?: boolean;
}

export interface Lc0WebAttentionValueBenchmarkResult {
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
}

function cpuAttentionValues(probs: Float32Array<ArrayBufferLike>, v: Float32Array<ArrayBufferLike>, tokens: number, channels: number, heads: number): Float32Array<ArrayBufferLike> {
  const headDim = channels / heads;
  const output = new Float32Array(tokens * channels);
  for (let head = 0; head < heads; head++) {
    const channelOffset = head * headDim;
    for (let row = 0; row < tokens; row++) {
      for (let channel = 0; channel < headDim; channel++) {
        let sum = 0;
        for (let col = 0; col < tokens; col++) {
          sum += probs[(head * tokens + row) * tokens + col] * v[col * channels + channelOffset + channel];
        }
        output[row * channels + channelOffset + channel] = sum;
      }
    }
  }
  return output;
}

const ATTENTION_VALUE_WGSL = `
@group(0) @binding(0) var<storage, read> probsVec: array<f32>;
@group(0) @binding(1) var<storage, read> valueVec: array<f32>;
@group(0) @binding(2) var<storage, read_write> outputVec: array<f32>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let channel = gid.x;
  let row = gid.y;
  let head = gid.z;
  if (channel >= 32u || row >= 64u || head >= 8u) { return; }
  let channel_offset = head * 32u;
  var sum = 0.0;
  for (var col = 0u; col < 64u; col = col + 1u) {
    sum = sum + probsVec[(head * 64u + row) * 64u + col] * valueVec[col * 256u + channel_offset + channel];
  }
  outputVec[row * 256u + channel_offset + channel] = sum;
}
`;

function createAttentionValuePipeline(device: DeviceLike, buffers: { probs: BufferLike; v: BufferLike; output: BufferLike }): { pipeline: PipelineLike; bindGroup: unknown } {
  const module = device.createShaderModule({ label: 'lc0web attention value probe', code: ATTENTION_VALUE_WGSL });
  const pipeline = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'main' } }) as PipelineLike;
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: buffers.probs } },
      { binding: 1, resource: { buffer: buffers.v } },
      { binding: 2, resource: { buffer: buffers.output } },
    ],
  });
  return { pipeline, bindGroup };
}

function encodeAttentionValueDispatches(device: DeviceLike, pipeline: PipelineLike, bindGroup: unknown, iterations: number): unknown {
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  for (let i = 0; i < iterations; i++) pass.dispatchWorkgroups(Math.ceil(DEFAULT_HEAD_DIM / 8), Math.ceil(DEFAULT_TOKENS / 8), DEFAULT_HEADS);
  pass.end();
  return encoder.finish();
}

function loadAttentionValueInputs(pack: Awaited<ReturnType<typeof loadLc0WebModelPack>>, tensorNames: Pick<Lc0WebEncoderBlockTensorNames, 'qkv' | 'scaleTensor' | 'smolgen'> = { qkv: DEFAULT_QKV_TENSORS, scaleTensor: DEFAULT_SCALE_TENSOR, smolgen: DEFAULT_SMOLGEN_TENSORS }): {
  qWeight: Lc0WebTensorView;
  qBias: Lc0WebTensorView;
  kWeight: Lc0WebTensorView;
  kBias: Lc0WebTensorView;
  vWeight: Lc0WebTensorView;
  vBias: Lc0WebTensorView;
  scale: Lc0WebTensorView;
  smolgen: Encoder0SmolgenTensors;
} {
  const qWeight = pack.tensors.get(tensorNames.qkv.qWeight);
  const qBias = pack.tensors.get(tensorNames.qkv.qBias);
  const kWeight = pack.tensors.get(tensorNames.qkv.kWeight);
  const kBias = pack.tensors.get(tensorNames.qkv.kBias);
  const vWeight = pack.tensors.get(tensorNames.qkv.vWeight);
  const vBias = pack.tensors.get(tensorNames.qkv.vBias);
  const scale = pack.tensors.get(tensorNames.scaleTensor);
  if (!qWeight || !qBias || !kWeight || !kBias || !vWeight || !vBias || !scale) throw new Error('lc0web attention value tensors were not loaded');
  for (const [label, tensor] of Object.entries({ qWeight, kWeight, vWeight })) assertTensorShapeAndBytes(tensor, [DEFAULT_K, DEFAULT_N], 2, label);
  for (const [label, tensor] of Object.entries({ qBias, kBias, vBias })) assertTensorShapeAndBytes(tensor, [DEFAULT_N], 2, label);
  assertTensorShapeAndBytes(scale, [1], 2, 'scale');
  for (const tensor of [qWeight, qBias, kWeight, kBias, vWeight, vBias, scale]) {
    if (tensor.info.dtype !== 'f16') throw new Error(`lc0web attention value expects f16 tensor ${tensor.info.name}, got ${tensor.info.dtype}`);
  }
  return { qWeight, qBias, kWeight, kBias, vWeight, vBias, scale, smolgen: loadEncoder0SmolgenTensors(pack, tensorNames.smolgen) };
}

function buildAttentionValueReference(tensors: ReturnType<typeof loadAttentionValueInputs>, input: Float32Array<ArrayBufferLike> = makeInputTokenMatrix(DEFAULT_TOKENS, DEFAULT_K)): { probs: Float32Array<ArrayBufferLike>; v: Float32Array<ArrayBufferLike>; output: Float32Array<ArrayBufferLike>; scale: number; smolgenBias: Float32Array<ArrayBufferLike> } {
  const q = cpuProjectTokens(input, tensors.qWeight.bytes, tensors.qBias.bytes, DEFAULT_TOKENS, DEFAULT_K, DEFAULT_N);
  const k = cpuProjectTokens(input, tensors.kWeight.bytes, tensors.kBias.bytes, DEFAULT_TOKENS, DEFAULT_K, DEFAULT_N);
  const v = cpuProjectTokens(input, tensors.vWeight.bytes, tensors.vBias.bytes, DEFAULT_TOKENS, DEFAULT_K, DEFAULT_N);
  const scale = readF16At(tensors.scale.bytes, 0);
  const qkScores = cpuAttentionScores(q, k, scale, DEFAULT_TOKENS, DEFAULT_N, DEFAULT_HEADS);
  const smolgenBias = cpuEncoder0SmolgenBias(input, tensors.smolgen);
  const scores = addElementwise(qkScores, smolgenBias);
  const probs = cpuSoftmaxRows(scores, DEFAULT_HEADS * DEFAULT_TOKENS, DEFAULT_TOKENS);
  const output = cpuAttentionValues(probs, v, DEFAULT_TOKENS, DEFAULT_N, DEFAULT_HEADS);
  return { probs, v, output, scale, smolgenBias };
}

export async function runLc0WebAttentionValueBenchmark(options: Lc0WebAttentionValueBenchmarkOptions): Promise<Lc0WebAttentionValueBenchmarkResult> {
  const totalStarted = nowMs();
  const warmup = clampInteger(options.warmup, 10, 0, 1000);
  const iterations = clampInteger(options.iterations, 1000, 1, 100_000);
  const { device, adapterInfo } = await requestDevice();
  const pack = await loadLc0WebModelPack(options.packUrl, {
    verifyShards: options.verifyShards ?? true,
    tensorNames: [...Object.values(DEFAULT_QKV_TENSORS), DEFAULT_SCALE_TENSOR, ...Object.values(DEFAULT_SMOLGEN_TENSORS)],
  });
  const tensors = loadAttentionValueInputs(pack);
  const reference = buildAttentionValueReference(tensors);
  const outputElements = DEFAULT_TOKENS * DEFAULT_N;
  const globals = gpuGlobals();
  const usage = globals.GPUBufferUsage!;
  const buffers: BufferLike[] = [];
  try {
    const setupStarted = nowMs();
    const probsBuffer = createStorageBuffer(device, reference.probs, usage.STORAGE | usage.COPY_DST);
    const vBuffer = createStorageBuffer(device, reference.v, usage.STORAGE | usage.COPY_DST);
    const outputBuffer = device.createBuffer({ size: outputElements * 4, usage: usage.STORAGE | usage.COPY_SRC });
    const readbackBuffer = device.createBuffer({ size: outputElements * 4, usage: usage.MAP_READ | usage.COPY_DST });
    buffers.push(probsBuffer, vBuffer, outputBuffer, readbackBuffer);
    const { pipeline, bindGroup } = createAttentionValuePipeline(device, { probs: probsBuffer, v: vBuffer, output: outputBuffer });
    const uploadSetupMs = nowMs() - setupStarted;

    if (warmup > 0) {
      device.queue.submit([encodeAttentionValueDispatches(device, pipeline, bindGroup, warmup)]);
      await device.queue.onSubmittedWorkDone?.();
    }
    const dispatchStarted = nowMs();
    device.queue.submit([encodeAttentionValueDispatches(device, pipeline, bindGroup, iterations)]);
    const dispatchLoopMs = nowMs() - dispatchStarted;
    const readbackStarted = nowMs();
    const output = await readF32OutputOnce(device, outputBuffer, readbackBuffer, outputElements);
    const readbackSyncedMs = nowMs() - readbackStarted;
    const { maxAbsError, rmsError } = computeErrorStats(output, reference.output, outputElements);
    assertErrorInTolerance(maxAbsError);
    return {
      status: 'ATTENTION_VALUE_BENCH_DONE',
      packUrl: pack.manifestUrl,
      modelName: pack.manifest.model.name,
      adapterInfo,
      tokens: DEFAULT_TOKENS,
      channels: DEFAULT_N,
      heads: DEFAULT_HEADS,
      headDim: DEFAULT_HEAD_DIM,
      warmup,
      iterations,
      packLoadMs: pack.elapsedMs,
      uploadSetupMs,
      dispatchLoopMs,
      dispatchLoopAvgMs: dispatchLoopMs / iterations,
      readbackSyncedMs,
      endToEndMs: nowMs() - totalStarted,
      maxAbsError,
      rmsError,
      outputSample: Array.from(output.slice(0, 8)),
    };
  } finally {
    for (const buffer of buffers) buffer.destroy?.();
  }
}

export type Lc0WebAttentionQkvKernelVariant = 'hand' | 'tvm-packed-f16';

export interface Lc0WebAttentionBlockBenchmarkOptions {
  packUrl: string;
  iterations?: number;
  warmup?: number;
  verifyShards?: boolean;
  fusedScoreSoftmax?: boolean;
  attentionQkvKernelVariant?: Lc0WebAttentionQkvKernelVariant;
}

export interface Lc0WebAttentionBlockBenchmarkResult {
  status: 'ATTENTION_BLOCK_BENCH_DONE';
  packUrl: string;
  modelName: string;
  adapterInfo?: Record<string, unknown>;
  tokens: number;
  channels: number;
  heads: number;
  headDim: number;
  fusedScoreSoftmax?: boolean;
  qkvKernelVariant?: Lc0WebAttentionQkvKernelVariant;
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
}


const ATTENTION_BLOCK_QKV_WGSL = `
@group(0) @binding(0) var<storage, read> inputMat: array<f32>;
@group(0) @binding(1) var<storage, read> qWeightsF16: array<u32>;
@group(0) @binding(2) var<storage, read> qBiasF16: array<u32>;
@group(0) @binding(3) var<storage, read> kWeightsF16: array<u32>;
@group(0) @binding(4) var<storage, read> kBiasF16: array<u32>;
@group(0) @binding(5) var<storage, read> vWeightsF16: array<u32>;
@group(0) @binding(6) var<storage, read> vBiasF16: array<u32>;
@group(0) @binding(7) var<storage, read_write> qkvOut: array<f32>;

fn pick_lane(word: u32, index: u32) -> f32 {
  let pair = unpack2x16float(word);
  return select(pair.x, pair.y, (index & 1u) == 1u);
}

fn load_q_weight(index: u32) -> f32 { return pick_lane(qWeightsF16[index >> 1u], index); }
fn load_q_bias(index: u32) -> f32 { return pick_lane(qBiasF16[index >> 1u], index); }
fn load_k_weight(index: u32) -> f32 { return pick_lane(kWeightsF16[index >> 1u], index); }
fn load_k_bias(index: u32) -> f32 { return pick_lane(kBiasF16[index >> 1u], index); }
fn load_v_weight(index: u32) -> f32 { return pick_lane(vWeightsF16[index >> 1u], index); }
fn load_v_bias(index: u32) -> f32 { return pick_lane(vBiasF16[index >> 1u], index); }

var<workgroup> qkvInputTile: array<f32, 128>;
var<workgroup> qWeightTile: array<f32, 128>;
var<workgroup> kWeightTile: array<f32, 128>;
var<workgroup> vWeightTile: array<f32, 128>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let col = wid.x * 8u + lid.x;
  let token = wid.y * 8u + lid.y;
  let local_index = lid.y * 8u + lid.x;
  var q_sum = load_q_bias(col);
  var k_sum = load_k_bias(col);
  var v_sum = load_v_bias(col);
  for (var tile = 0u; tile < 256u; tile = tile + 16u) {
    for (var i = local_index; i < 128u; i = i + 64u) {
      let tile_row = i / 16u;
      let tile_k = i % 16u;
      let input_token = wid.y * 8u + tile_row;
      let weight_col = wid.x * 8u + (i % 8u);
      let weight_k = tile + (i / 8u);
      qkvInputTile[i] = select(inputMat[input_token * 256u + tile + tile_k], 0.0, input_token >= 64u);
      qWeightTile[i] = select(load_q_weight(weight_col * 256u + weight_k), 0.0, weight_col >= 256u);
      kWeightTile[i] = select(load_k_weight(weight_col * 256u + weight_k), 0.0, weight_col >= 256u);
      vWeightTile[i] = select(load_v_weight(weight_col * 256u + weight_k), 0.0, weight_col >= 256u);
    }
    workgroupBarrier();
    for (var k = 0u; k < 16u; k = k + 1u) {
      let x = qkvInputTile[lid.y * 16u + k];
      let weight_index = k * 8u + lid.x;
      q_sum = q_sum + x * qWeightTile[weight_index];
      k_sum = k_sum + x * kWeightTile[weight_index];
      v_sum = v_sum + x * vWeightTile[weight_index];
    }
    workgroupBarrier();
  }
  if (col < 256u && token < 64u) {
    qkvOut[token * 256u + col] = q_sum;
    qkvOut[16384u + token * 256u + col] = k_sum;
    qkvOut[32768u + token * 256u + col] = v_sum;
  }
}
`;

const ATTENTION_BLOCK_SCORE_WGSL = `
@group(0) @binding(0) var<storage, read> qkvVec: array<f32>;
@group(0) @binding(1) var<storage, read> scaleF16: array<u32>;
@group(0) @binding(2) var<storage, read> smolgenBiasVec: array<f32>;
@group(0) @binding(3) var<storage, read_write> scoreVec: array<f32>;

fn pick_lane(word: u32, index: u32) -> f32 {
  let pair = unpack2x16float(word);
  return select(pair.x, pair.y, (index & 1u) == 1u);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let col = gid.x;
  let row = gid.y;
  let head = gid.z;
  if (row >= 64u || col >= 64u || head >= 8u) { return; }
  let channel_offset = head * 32u;
  var sum = 0.0;
  for (var channel = 0u; channel < 32u; channel = channel + 1u) {
    let q = qkvVec[row * 256u + channel_offset + channel];
    let k = qkvVec[16384u + col * 256u + channel_offset + channel];
    sum = sum + q * k;
  }
  let index = (head * 64u + row) * 64u + col;
  scoreVec[index] = sum * pick_lane(scaleF16[0], 0u) + smolgenBiasVec[index];
}
`;

const ATTENTION_BLOCK_VALUE_WGSL = `
@group(0) @binding(0) var<storage, read> probsVec: array<f32>;
@group(0) @binding(1) var<storage, read> qkvVec: array<f32>;
@group(0) @binding(2) var<storage, read_write> outputVec: array<f32>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let channel = gid.x;
  let row = gid.y;
  let head = gid.z;
  if (channel >= 32u || row >= 64u || head >= 8u) { return; }
  let channel_offset = head * 32u;
  var sum = 0.0;
  for (var col = 0u; col < 64u; col = col + 1u) {
    let prob = probsVec[(head * 64u + row) * 64u + col];
    let value = qkvVec[32768u + col * 256u + channel_offset + channel];
    sum = sum + prob * value;
  }
  outputVec[row * 256u + channel_offset + channel] = sum;
}
`;

const ATTENTION_BLOCK_SCORE_SOFTMAX_WGSL = `
@group(0) @binding(0) var<storage, read> qkvVec: array<f32>;
@group(0) @binding(1) var<storage, read> scaleF16: array<u32>;
@group(0) @binding(2) var<storage, read> smolgenBiasVec: array<f32>;
@group(0) @binding(3) var<storage, read_write> probsVec: array<f32>;

fn pick_lane(word: u32, index: u32) -> f32 {
  let pair = unpack2x16float(word);
  return select(pair.x, pair.y, (index & 1u) == 1u);
}

fn score_for(head: u32, row: u32, col: u32) -> f32 {
  let channel_offset = head * 32u;
  var sum = 0.0;
  for (var channel = 0u; channel < 32u; channel = channel + 1u) {
    let q = qkvVec[row * 256u + channel_offset + channel];
    let k = qkvVec[16384u + col * 256u + channel_offset + channel];
    sum = sum + q * k;
  }
  let index = (head * 64u + row) * 64u + col;
  return sum * pick_lane(scaleF16[0], 0u) + smolgenBiasVec[index];
}

@compute @workgroup_size(1, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.x;
  let head = gid.y;
  if (row >= 64u || head >= 8u) { return; }
  var max_value = score_for(head, row, 0u);
  for (var col = 1u; col < 64u; col = col + 1u) {
    max_value = max(max_value, score_for(head, row, col));
  }
  var sum_exp = 0.0;
  for (var col = 0u; col < 64u; col = col + 1u) {
    sum_exp = sum_exp + exp(score_for(head, row, col) - max_value);
  }
  for (var col = 0u; col < 64u; col = col + 1u) {
    let index = (head * 64u + row) * 64u + col;
    probsVec[index] = exp(score_for(head, row, col) - max_value) / sum_exp;
  }
}
`;

function createAttentionQkvStage(device: DeviceLike, buffers: {
  input: BufferLike;
  qWeight: BufferLike;
  qBias: BufferLike;
  kWeight: BufferLike;
  kBias: BufferLike;
  vWeight: BufferLike;
  vBias: BufferLike;
  qkv: BufferLike;
  podArgs?: BufferLike;
}, qkvKernelVariant: Lc0WebAttentionQkvKernelVariant): { qkv: PipelineLike; qkvBinds: unknown[]; qkvKernelVariant: Lc0WebAttentionQkvKernelVariant } {
  if (qkvKernelVariant === 'tvm-packed-f16') {
    if (!buffers.podArgs) throw new Error('TVM packed-f16 attention QKV kernels require a POD args uniform buffer');
    const qkvModule = device.createShaderModule({ label: 'lc0web attention block QKV TVM packed-f16 projection', code: ATTENTION_BLOCK_QKV_TVM_PACKED_F16_WGSL });
    const qkv = device.createComputePipeline({ layout: 'auto', compute: { module: qkvModule, entryPoint: 'matmul_kernel' } }) as PipelineLike;
    const outputBytes = DEFAULT_TOKENS * DEFAULT_N * 4;
    const qkvBindFor = (weight: BufferLike, bias: BufferLike, outputOffset: number) => device.createBindGroup({
      layout: qkv.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: buffers.qkv, offset: outputOffset, size: outputBytes } },
        { binding: 1, resource: { buffer: weight } },
        { binding: 2, resource: { buffer: buffers.input } },
        { binding: 3, resource: { buffer: buffers.podArgs! } },
        { binding: 4, resource: { buffer: bias } },
      ],
    });
    return { qkv, qkvKernelVariant, qkvBinds: [
      qkvBindFor(buffers.qWeight, buffers.qBias, 0),
      qkvBindFor(buffers.kWeight, buffers.kBias, outputBytes),
      qkvBindFor(buffers.vWeight, buffers.vBias, outputBytes * 2),
    ] };
  }
  const qkvModule = device.createShaderModule({ label: 'lc0web attention block qkv', code: ATTENTION_BLOCK_QKV_WGSL });
  const qkv = device.createComputePipeline({ layout: 'auto', compute: { module: qkvModule, entryPoint: 'main' } }) as PipelineLike;
  const qkvBind = device.createBindGroup({
    layout: qkv.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: buffers.input } },
      { binding: 1, resource: { buffer: buffers.qWeight } },
      { binding: 2, resource: { buffer: buffers.qBias } },
      { binding: 3, resource: { buffer: buffers.kWeight } },
      { binding: 4, resource: { buffer: buffers.kBias } },
      { binding: 5, resource: { buffer: buffers.vWeight } },
      { binding: 6, resource: { buffer: buffers.vBias } },
      { binding: 7, resource: { buffer: buffers.qkv } },
    ],
  });
  return { qkv, qkvKernelVariant, qkvBinds: [qkvBind] };
}

function createAttentionBlockPipelines(device: DeviceLike, buffers: {
  input: BufferLike;
  qWeight: BufferLike;
  qBias: BufferLike;
  kWeight: BufferLike;
  kBias: BufferLike;
  vWeight: BufferLike;
  vBias: BufferLike;
  scale: BufferLike;
  smolgenBias: BufferLike;
  qkv: BufferLike;
  scores: BufferLike;
  probs: BufferLike;
  output: BufferLike;
  podArgs?: BufferLike;
}, qkvKernelVariant: Lc0WebAttentionQkvKernelVariant = 'hand'): { qkv: PipelineLike; qkvBind: unknown; qkvBinds: unknown[]; qkvKernelVariant: Lc0WebAttentionQkvKernelVariant; score: PipelineLike; scoreBind: unknown; softmax: PipelineLike; softmaxBind: unknown; value: PipelineLike; valueBind: unknown } {
  const qkvStage = createAttentionQkvStage(device, buffers, qkvKernelVariant);
  const { qkv, qkvBinds } = qkvStage;
  const scoreModule = device.createShaderModule({ label: 'lc0web attention block score', code: ATTENTION_BLOCK_SCORE_WGSL });
  const score = device.createComputePipeline({ layout: 'auto', compute: { module: scoreModule, entryPoint: 'main' } }) as PipelineLike;
  const scoreBind = device.createBindGroup({
    layout: score.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: buffers.qkv } },
      { binding: 1, resource: { buffer: buffers.scale } },
      { binding: 2, resource: { buffer: buffers.smolgenBias } },
      { binding: 3, resource: { buffer: buffers.scores } },
    ],
  });
  const { pipeline: softmax, bindGroup: softmaxBind } = createSoftmaxPipeline(device, { input: buffers.scores, output: buffers.probs });
  const valueModule = device.createShaderModule({ label: 'lc0web attention block value', code: ATTENTION_BLOCK_VALUE_WGSL });
  const value = device.createComputePipeline({ layout: 'auto', compute: { module: valueModule, entryPoint: 'main' } }) as PipelineLike;
  const valueBind = device.createBindGroup({
    layout: value.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: buffers.probs } },
      { binding: 1, resource: { buffer: buffers.qkv } },
      { binding: 2, resource: { buffer: buffers.output } },
    ],
  });
  return { qkv, qkvBind: qkvBinds[0], qkvBinds, qkvKernelVariant: qkvStage.qkvKernelVariant, score, scoreBind, softmax, softmaxBind, value, valueBind };
}

function createAttentionBlockFusedPipelines(device: DeviceLike, buffers: {
  input: BufferLike;
  qWeight: BufferLike;
  qBias: BufferLike;
  kWeight: BufferLike;
  kBias: BufferLike;
  vWeight: BufferLike;
  vBias: BufferLike;
  scale: BufferLike;
  smolgenBias: BufferLike;
  qkv: BufferLike;
  probs: BufferLike;
  output: BufferLike;
  podArgs?: BufferLike;
}, qkvKernelVariant: Lc0WebAttentionQkvKernelVariant = 'hand'): { qkv: PipelineLike; qkvBind: unknown; qkvBinds: unknown[]; qkvKernelVariant: Lc0WebAttentionQkvKernelVariant; scoreSoftmax: PipelineLike; scoreSoftmaxBind: unknown; value: PipelineLike; valueBind: unknown } {
  const qkvStage = createAttentionQkvStage(device, buffers, qkvKernelVariant);
  const { qkv, qkvBinds } = qkvStage;
  const scoreSoftmaxModule = device.createShaderModule({ label: 'lc0web attention block fused score+softmax', code: ATTENTION_BLOCK_SCORE_SOFTMAX_WGSL });
  const scoreSoftmax = device.createComputePipeline({ layout: 'auto', compute: { module: scoreSoftmaxModule, entryPoint: 'main' } }) as PipelineLike;
  const scoreSoftmaxBind = device.createBindGroup({
    layout: scoreSoftmax.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: buffers.qkv } },
      { binding: 1, resource: { buffer: buffers.scale } },
      { binding: 2, resource: { buffer: buffers.smolgenBias } },
      { binding: 3, resource: { buffer: buffers.probs } },
    ],
  });
  const valueModule = device.createShaderModule({ label: 'lc0web attention block fused value', code: ATTENTION_BLOCK_VALUE_WGSL });
  const value = device.createComputePipeline({ layout: 'auto', compute: { module: valueModule, entryPoint: 'main' } }) as PipelineLike;
  const valueBind = device.createBindGroup({
    layout: value.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: buffers.probs } },
      { binding: 1, resource: { buffer: buffers.qkv } },
      { binding: 2, resource: { buffer: buffers.output } },
    ],
  });
  return { qkv, qkvBind: qkvBinds[0], qkvBinds, qkvKernelVariant: qkvStage.qkvKernelVariant, scoreSoftmax, scoreSoftmaxBind, value, valueBind };
}

function encodeAttentionQkvPass(pass: ComputePassLike, pipelines: { qkv: PipelineLike; qkvBinds: unknown[]; qkvKernelVariant: Lc0WebAttentionQkvKernelVariant }): void {
  pass.setPipeline(pipelines.qkv);
  for (const qkvBind of pipelines.qkvBinds) {
    pass.setBindGroup(0, qkvBind);
    if (pipelines.qkvKernelVariant === 'tvm-packed-f16') pass.dispatchWorkgroups(2, 8, 1);
    else pass.dispatchWorkgroups(Math.ceil(DEFAULT_N / 8), Math.ceil(DEFAULT_TOKENS / 8));
  }
}

function encodeAttentionBlockDispatches(device: DeviceLike, pipelines: ReturnType<typeof createAttentionBlockPipelines>, iterations: number): unknown {
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  for (let i = 0; i < iterations; i++) {
    encodeAttentionQkvPass(pass, pipelines);
    pass.setPipeline(pipelines.score);
    pass.setBindGroup(0, pipelines.scoreBind);
    pass.dispatchWorkgroups(Math.ceil(DEFAULT_TOKENS / 8), Math.ceil(DEFAULT_TOKENS / 8), DEFAULT_HEADS);
    pass.setPipeline(pipelines.softmax);
    pass.setBindGroup(0, pipelines.softmaxBind);
    pass.dispatchWorkgroups(DEFAULT_HEADS * DEFAULT_TOKENS);
    pass.setPipeline(pipelines.value);
    pass.setBindGroup(0, pipelines.valueBind);
    pass.dispatchWorkgroups(Math.ceil(DEFAULT_HEAD_DIM / 8), Math.ceil(DEFAULT_TOKENS / 8), DEFAULT_HEADS);
  }
  pass.end();
  return encoder.finish();
}

function encodeAttentionBlockFusedDispatches(device: DeviceLike, pipelines: ReturnType<typeof createAttentionBlockFusedPipelines>, iterations: number): unknown {
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  for (let i = 0; i < iterations; i++) {
    encodeAttentionQkvPass(pass, pipelines);
    pass.setPipeline(pipelines.scoreSoftmax);
    pass.setBindGroup(0, pipelines.scoreSoftmaxBind);
    pass.dispatchWorkgroups(DEFAULT_TOKENS, DEFAULT_HEADS);
    pass.setPipeline(pipelines.value);
    pass.setBindGroup(0, pipelines.valueBind);
    pass.dispatchWorkgroups(Math.ceil(DEFAULT_HEAD_DIM / 8), Math.ceil(DEFAULT_TOKENS / 8), DEFAULT_HEADS);
  }
  pass.end();
  return encoder.finish();
}

export async function runLc0WebAttentionBlockBenchmark(options: Lc0WebAttentionBlockBenchmarkOptions): Promise<Lc0WebAttentionBlockBenchmarkResult> {
  const totalStarted = nowMs();
  const warmup = clampInteger(options.warmup, 5, 0, 1000);
  const iterations = clampInteger(options.iterations, 100, 1, 10_000);
  const fusedScoreSoftmax = options.fusedScoreSoftmax ?? false;
  const qkvKernelVariant = options.attentionQkvKernelVariant ?? 'hand';
  const { device, adapterInfo } = await requestDevice();
  const pack = await loadLc0WebModelPack(options.packUrl, {
    verifyShards: options.verifyShards ?? true,
    tensorNames: [...Object.values(DEFAULT_QKV_TENSORS), DEFAULT_SCALE_TENSOR, ...Object.values(DEFAULT_SMOLGEN_TENSORS)],
  });
  const tensors = loadAttentionValueInputs(pack);
  const input = makeInputTokenMatrix(DEFAULT_TOKENS, DEFAULT_K);
  const reference = buildAttentionValueReference(tensors);
  const outputElements = DEFAULT_TOKENS * DEFAULT_N;
  const globals = gpuGlobals();
  const usage = globals.GPUBufferUsage!;
  const buffers: BufferLike[] = [];
  try {
    const setupStarted = nowMs();
    const inputBuffer = createStorageBuffer(device, input, usage.STORAGE | usage.COPY_DST);
    const qWeight = createTransposedF16StorageBuffer(device, tensors.qWeight.bytes, DEFAULT_K, DEFAULT_N, usage.STORAGE | usage.COPY_DST);
    const qBias = createStorageBuffer(device, tensors.qBias.bytes, usage.STORAGE | usage.COPY_DST);
    const kWeight = createTransposedF16StorageBuffer(device, tensors.kWeight.bytes, DEFAULT_K, DEFAULT_N, usage.STORAGE | usage.COPY_DST);
    const kBias = createStorageBuffer(device, tensors.kBias.bytes, usage.STORAGE | usage.COPY_DST);
    const vWeight = createTransposedF16StorageBuffer(device, tensors.vWeight.bytes, DEFAULT_K, DEFAULT_N, usage.STORAGE | usage.COPY_DST);
    const vBias = createStorageBuffer(device, tensors.vBias.bytes, usage.STORAGE | usage.COPY_DST);
    const scale = createStorageBuffer(device, paddedF16ScalarBytes(tensors.scale.bytes), usage.STORAGE | usage.COPY_DST);
    const smolgenBias = createStorageBuffer(device, reference.smolgenBias, usage.STORAGE | usage.COPY_DST);
    const qkvBuffer = device.createBuffer({ size: DEFAULT_TOKENS * DEFAULT_N * 3 * 4, usage: usage.STORAGE });
    const scoreBuffer = device.createBuffer({ size: DEFAULT_HEADS * DEFAULT_TOKENS * DEFAULT_TOKENS * 4, usage: usage.STORAGE });
    const probBuffer = device.createBuffer({ size: DEFAULT_HEADS * DEFAULT_TOKENS * DEFAULT_TOKENS * 4, usage: usage.STORAGE });
    const outputBuffer = device.createBuffer({ size: outputElements * 4, usage: usage.STORAGE | usage.COPY_SRC });
    const readbackBuffer = device.createBuffer({ size: outputElements * 4, usage: usage.MAP_READ | usage.COPY_DST });
    const podArgs = qkvKernelVariant === 'tvm-packed-f16' ? createU32UniformBuffer(device, [1], usage.UNIFORM | usage.COPY_DST) : undefined;
    buffers.push(inputBuffer, qWeight, qBias, kWeight, kBias, vWeight, vBias, scale, smolgenBias, qkvBuffer, scoreBuffer, probBuffer, outputBuffer, readbackBuffer);
    if (podArgs) buffers.push(podArgs);
    const pipelines = fusedScoreSoftmax
      ? createAttentionBlockFusedPipelines(device, { input: inputBuffer, qWeight, qBias, kWeight, kBias, vWeight, vBias, scale, smolgenBias, qkv: qkvBuffer, probs: probBuffer, output: outputBuffer, podArgs }, qkvKernelVariant)
      : createAttentionBlockPipelines(device, { input: inputBuffer, qWeight, qBias, kWeight, kBias, vWeight, vBias, scale, smolgenBias, qkv: qkvBuffer, scores: scoreBuffer, probs: probBuffer, output: outputBuffer, podArgs }, qkvKernelVariant);
    const encode = fusedScoreSoftmax
      ? (count: number) => encodeAttentionBlockFusedDispatches(device, pipelines as ReturnType<typeof createAttentionBlockFusedPipelines>, count)
      : (count: number) => encodeAttentionBlockDispatches(device, pipelines as ReturnType<typeof createAttentionBlockPipelines>, count);
    const uploadSetupMs = nowMs() - setupStarted;

    if (warmup > 0) {
      device.queue.submit([encode(warmup)]);
      await device.queue.onSubmittedWorkDone?.();
    }
    const dispatchStarted = nowMs();
    device.queue.submit([encode(iterations)]);
    const dispatchLoopMs = nowMs() - dispatchStarted;
    const readbackStarted = nowMs();
    const output = await readF32OutputOnce(device, outputBuffer, readbackBuffer, outputElements);
    const readbackSyncedMs = nowMs() - readbackStarted;
    const { maxAbsError, rmsError } = computeErrorStats(output, reference.output, outputElements);
    assertErrorInTolerance(maxAbsError);
    return {
      status: 'ATTENTION_BLOCK_BENCH_DONE',
      packUrl: pack.manifestUrl,
      modelName: pack.manifest.model.name,
      adapterInfo,
      tokens: DEFAULT_TOKENS,
      channels: DEFAULT_N,
      heads: DEFAULT_HEADS,
      headDim: DEFAULT_HEAD_DIM,
      fusedScoreSoftmax,
      qkvKernelVariant,
      dispatchesPerIteration: (fusedScoreSoftmax ? 2 : 3) + (qkvKernelVariant === 'tvm-packed-f16' ? 3 : 1),
      warmup,
      iterations,
      packLoadMs: pack.elapsedMs,
      uploadSetupMs,
      dispatchLoopMs,
      dispatchLoopAvgMs: dispatchLoopMs / iterations,
      readbackSyncedMs,
      endToEndMs: nowMs() - totalStarted,
      maxAbsError,
      rmsError,
      outputSample: Array.from(output.slice(0, 8)),
    };
  } finally {
    for (const buffer of buffers) buffer.destroy?.();
  }
}

const DEFAULT_OUT_DENSE_WEIGHT = '/encoder0/mha/out/dense/w/w';
const DEFAULT_OUT_DENSE_BIAS = '/encoder0/mha/out/dense/b/w';
const DEFAULT_OUT_ALPHA = '/encoder0/alpha*input/w';
const DEFAULT_LN1_SCALE = '/encoder0/ln1/w/scale';
const DEFAULT_LN1_BIAS = '/encoder0/ln1/w/bias';
const DEFAULT_LN_EPSILON = 9.999999974752427e-7;

export type Lc0WebAttentionOutProjKernelVariant = 'hand' | 'tvm-packed-f16';
export type Lc0WebEncoderKernelVariant = 'hand' | 'tvm-packed-f16' | 'mixed-tvm-ffn' | 'mixed-tvm-ffn-outproj' | 'mixed-tvm-ffn-smolgen-project';

function encoderUsesTvmPackedF16Ffn(variant: Lc0WebEncoderKernelVariant): boolean {
  return variant === 'tvm-packed-f16' || variant === 'mixed-tvm-ffn' || variant === 'mixed-tvm-ffn-outproj' || variant === 'mixed-tvm-ffn-smolgen-project';
}

function encoderUsesTiledSmolgenProject(variant: Lc0WebEncoderKernelVariant): boolean {
  return variant === 'mixed-tvm-ffn-smolgen-project';
}

function encoderUsesTvmPackedF16Qkv(variant: Lc0WebEncoderKernelVariant): boolean {
  return variant === 'tvm-packed-f16';
}

function encoderUsesTvmPackedF16AttentionOutProj(variant: Lc0WebEncoderKernelVariant): boolean {
  return variant === 'tvm-packed-f16' || variant === 'mixed-tvm-ffn-outproj';
}

export interface Lc0WebAttentionOutputBenchmarkOptions {
  packUrl: string;
  iterations?: number;
  warmup?: number;
  verifyShards?: boolean;
  encoderPrefix?: string;
  attentionOutProjKernelVariant?: Lc0WebAttentionOutProjKernelVariant;
}

export interface Lc0WebAttentionOutputBenchmarkResult {
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
  outProjKernelVariant?: Lc0WebAttentionOutProjKernelVariant;
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
}

export interface Lc0WebAttentionOutputOrtBenchmarkOptions {
  packUrl: string;
  iterations?: number;
  warmup?: number;
  verifyShards?: boolean;
  encoderPrefix?: string;
}

export interface Lc0WebAttentionOutputOrtBenchmarkResult {
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
  timesMs: number[];
  runsPerSecond: number;
  maxAbsError: number;
  rmsError: number;
  outputSample: number[];
}

function cpuLayerNormRows(input: Float32Array<ArrayBufferLike>, scale: Uint8Array, bias: Uint8Array, rows: number, cols: number, epsilon: number): Float32Array<ArrayBufferLike> {
  const output = new Float32Array(input.length);
  for (let row = 0; row < rows; row++) {
    const base = row * cols;
    let mean = 0;
    for (let col = 0; col < cols; col++) mean += input[base + col];
    mean /= cols;
    let variance = 0;
    for (let col = 0; col < cols; col++) {
      const centered = input[base + col] - mean;
      variance += centered * centered;
    }
    const invStd = 1 / Math.sqrt(variance / cols + epsilon);
    for (let col = 0; col < cols; col++) output[base + col] = (input[base + col] - mean) * invStd * readF16At(scale, col) + readF16At(bias, col);
  }
  return output;
}

function loadAttentionOutputInputs(pack: Awaited<ReturnType<typeof loadLc0WebModelPack>>, tensorNames: Lc0WebEncoderBlockTensorNames = lc0WebEncoderBlockTensorNames()): ReturnType<typeof loadAttentionValueInputs> & {
  outWeight: Lc0WebTensorView;
  outBias: Lc0WebTensorView;
  alpha: Lc0WebTensorView;
  lnScale: Lc0WebTensorView;
  lnBias: Lc0WebTensorView;
} {
  const base = loadAttentionValueInputs(pack, tensorNames);
  const outWeight = pack.tensors.get(tensorNames.outDenseWeight);
  const outBias = pack.tensors.get(tensorNames.outDenseBias);
  const alpha = pack.tensors.get(tensorNames.outAlpha);
  const lnScale = pack.tensors.get(tensorNames.ln1Scale);
  const lnBias = pack.tensors.get(tensorNames.ln1Bias);
  if (!outWeight || !outBias || !alpha || !lnScale || !lnBias) throw new Error('lc0web attention output tensors were not loaded');
  assertTensorShapeAndBytes(outWeight, [DEFAULT_N, DEFAULT_N], 2, 'outWeight');
  assertTensorShapeAndBytes(outBias, [DEFAULT_N], 2, 'outBias');
  assertTensorShapeAndBytes(alpha, [1], 2, 'alpha');
  assertTensorShapeAndBytes(lnScale, [DEFAULT_N], 2, 'lnScale');
  assertTensorShapeAndBytes(lnBias, [DEFAULT_N], 2, 'lnBias');
  for (const tensor of [outWeight, outBias, alpha, lnScale, lnBias]) {
    if (tensor.info.dtype !== 'f16') throw new Error(`lc0web attention output expects f16 tensor ${tensor.info.name}, got ${tensor.info.dtype}`);
  }
  return { ...base, outWeight, outBias, alpha, lnScale, lnBias };
}

function buildAttentionOutputReference(tensors: ReturnType<typeof loadAttentionOutputInputs>, input: Float32Array<ArrayBufferLike> = makeInputTokenMatrix(DEFAULT_TOKENS, DEFAULT_K)): { input: Float32Array<ArrayBufferLike>; output: Float32Array<ArrayBufferLike>; alpha: number; smolgenBias: Float32Array<ArrayBufferLike> } {
  const attentionReference = buildAttentionValueReference(tensors, input);
  const attention = attentionReference.output;
  const projected = cpuProjectTokens(attention, tensors.outWeight.bytes, tensors.outBias.bytes, DEFAULT_TOKENS, DEFAULT_N, DEFAULT_N);
  const alpha = readF16At(tensors.alpha.bytes, 0);
  const skip = new Float32Array(projected.length);
  for (let i = 0; i < projected.length; i++) skip[i] = projected[i] * alpha + input[i];
  const output = cpuLayerNormRows(skip, tensors.lnScale.bytes, tensors.lnBias.bytes, DEFAULT_TOKENS, DEFAULT_N, DEFAULT_LN_EPSILON);
  return { input, output, alpha, smolgenBias: attentionReference.smolgenBias };
}

const ATTENTION_OUTPUT_PROJ_WGSL = `
@group(0) @binding(0) var<storage, read> attnVec: array<f32>;
@group(0) @binding(1) var<storage, read> residualVec: array<f32>;
@group(0) @binding(2) var<storage, read> weightF16: array<u32>;
@group(0) @binding(3) var<storage, read> biasF16: array<u32>;
@group(0) @binding(4) var<storage, read> alphaF16: array<u32>;
@group(0) @binding(5) var<storage, read_write> skipVec: array<f32>;

fn pick_lane(word: u32, index: u32) -> f32 {
  let pair = unpack2x16float(word);
  return select(pair.x, pair.y, (index & 1u) == 1u);
}
fn load_weight(index: u32) -> f32 { return pick_lane(weightF16[index >> 1u], index); }
fn load_bias(index: u32) -> f32 { return pick_lane(biasF16[index >> 1u], index); }

var<workgroup> projInputTile: array<f32, 128>;
var<workgroup> projWeightTile: array<f32, 128>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let col = wid.x * 8u + lid.x;
  let token = wid.y * 8u + lid.y;
  let local_index = lid.y * 8u + lid.x;
  var sum = load_bias(col);
  for (var tile = 0u; tile < 256u; tile = tile + 16u) {
    for (var i = local_index; i < 128u; i = i + 64u) {
      let tile_row = i / 16u;
      let tile_k = i % 16u;
      let input_token = wid.y * 8u + tile_row;
      let weight_col = wid.x * 8u + (i % 8u);
      let weight_k = tile + (i / 8u);
      projInputTile[i] = attnVec[input_token * 256u + tile + tile_k];
      projWeightTile[i] = load_weight(weight_col * 256u + weight_k);
    }
    workgroupBarrier();
    for (var k = 0u; k < 16u; k = k + 1u) {
      sum = sum + projInputTile[lid.y * 16u + k] * projWeightTile[k * 8u + lid.x];
    }
    workgroupBarrier();
  }
  let index = token * 256u + col;
  skipVec[index] = sum * pick_lane(alphaF16[0], 0u) + residualVec[index];
}
`;


const ATTENTION_OUTPUT_NORM_WGSL = `
@group(0) @binding(0) var<storage, read> skipVec: array<f32>;
@group(0) @binding(1) var<storage, read> scaleF16: array<u32>;
@group(0) @binding(2) var<storage, read> biasF16: array<u32>;
@group(0) @binding(3) var<storage, read_write> outputVec: array<f32>;

fn pick_lane(word: u32, index: u32) -> f32 {
  let pair = unpack2x16float(word);
  return select(pair.x, pair.y, (index & 1u) == 1u);
}
fn load_scale(index: u32) -> f32 { return pick_lane(scaleF16[index >> 1u], index); }
fn load_bias(index: u32) -> f32 { return pick_lane(biasF16[index >> 1u], index); }

var<workgroup> lnPartial: array<f32, 64>;
var<workgroup> lnMean: f32;
var<workgroup> lnInvStd: f32;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let token = wid.x;
  let lane = lid.x;
  let base = token * 256u;
  let col0 = lane;
  let col1 = lane + 64u;
  let col2 = lane + 128u;
  let col3 = lane + 192u;
  let v0 = skipVec[base + col0];
  let v1 = skipVec[base + col1];
  let v2 = skipVec[base + col2];
  let v3 = skipVec[base + col3];
  lnPartial[lane] = v0 + v1 + v2 + v3;
  workgroupBarrier();

  for (var stride = 32u; stride > 0u; stride = stride / 2u) {
    if (lane < stride) { lnPartial[lane] = lnPartial[lane] + lnPartial[lane + stride]; }
    workgroupBarrier();
  }
  if (lane == 0u) { lnMean = lnPartial[0] / 256.0; }
  workgroupBarrier();
  let c0 = v0 - lnMean;
  let c1 = v1 - lnMean;
  let c2 = v2 - lnMean;
  let c3 = v3 - lnMean;
  lnPartial[lane] = c0 * c0 + c1 * c1 + c2 * c2 + c3 * c3;
  workgroupBarrier();

  for (var stride = 32u; stride > 0u; stride = stride / 2u) {
    if (lane < stride) { lnPartial[lane] = lnPartial[lane] + lnPartial[lane + stride]; }
    workgroupBarrier();
  }
  if (lane == 0u) { lnInvStd = inverseSqrt(lnPartial[0] / 256.0 + 0.000001); }
  workgroupBarrier();
  outputVec[base + col0] = c0 * lnInvStd * load_scale(col0) + load_bias(col0);
  outputVec[base + col1] = c1 * lnInvStd * load_scale(col1) + load_bias(col1);
  outputVec[base + col2] = c2 * lnInvStd * load_scale(col2) + load_bias(col2);
  outputVec[base + col3] = c3 * lnInvStd * load_scale(col3) + load_bias(col3);
}
`;

const SMOLGEN_COMPRESS_WGSL = `
@group(0) @binding(0) var<storage, read> inputVec: array<f32>;
@group(0) @binding(1) var<storage, read> weightF16: array<u32>;
@group(0) @binding(2) var<storage, read_write> outputVec: array<f32>;

fn pick_lane(word: u32, index: u32) -> f32 {
  let pair = unpack2x16float(word);
  return select(pair.x, pair.y, (index & 1u) == 1u);
}
fn load_weight(index: u32) -> f32 { return pick_lane(weightF16[index >> 1u], index); }

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let col = gid.x;
  let token = gid.y;
  if (col >= 32u || token >= 64u) { return; }
  var sum = 0.0;
  for (var channel = 0u; channel < 256u; channel = channel + 1u) {
    sum = sum + inputVec[token * 256u + channel] * load_weight(channel * 32u + col);
  }
  outputVec[token * 32u + col] = sum;
}
`;

const SMOLGEN_DENSE1_WGSL = `
@group(0) @binding(0) var<storage, read> inputVec: array<f32>;
@group(0) @binding(1) var<storage, read> weightF16: array<u32>;
@group(0) @binding(2) var<storage, read> biasF16: array<u32>;
@group(0) @binding(3) var<storage, read_write> outputVec: array<f32>;
var<workgroup> partial: array<f32, 128>;

fn pick_lane(word: u32, index: u32) -> f32 {
  let pair = unpack2x16float(word);
  return select(pair.x, pair.y, (index & 1u) == 1u);
}
fn load_weight(index: u32) -> f32 { return pick_lane(weightF16[index >> 1u], index); }
fn load_bias(index: u32) -> f32 { return pick_lane(biasF16[index >> 1u], index); }

@compute @workgroup_size(16, 8, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let col = wid.x * 16u + lid.x;
  var sum = 0.0;
  for (var row = lid.y; row < 2048u; row = row + 16u) {
    sum = sum + inputVec[row] * load_weight(row * 256u + col);
    let row2 = row + 8u;
    sum = sum + inputVec[row2] * load_weight(row2 * 256u + col);
  }
  partial[lid.y * 16u + lid.x] = sum;
  workgroupBarrier();
  for (var stride = 4u; stride > 0u; stride = stride / 2u) {
    if (lid.y < stride) {
      partial[lid.y * 16u + lid.x] = partial[lid.y * 16u + lid.x] + partial[(lid.y + stride) * 16u + lid.x];
    }
    workgroupBarrier();
  }
  if (lid.y == 0u) {
    outputVec[col] = partial[lid.x] + load_bias(col);
  }
}
`;

const SMOLGEN_SWISH_LN1_WGSL = `
@group(0) @binding(0) var<storage, read> inputVec: array<f32>;
@group(0) @binding(1) var<storage, read> scaleF16: array<u32>;
@group(0) @binding(2) var<storage, read> biasF16: array<u32>;
@group(0) @binding(3) var<storage, read_write> outputVec: array<f32>;
var<workgroup> partial: array<f32, 256>;
var<workgroup> meanValue: f32;
var<workgroup> invStdValue: f32;

fn pick_lane(word: u32, index: u32) -> f32 {
  let pair = unpack2x16float(word);
  return select(pair.x, pair.y, (index & 1u) == 1u);
}
fn load_scale(index: u32) -> f32 { return pick_lane(scaleF16[index >> 1u], index); }
fn load_bias(index: u32) -> f32 { return pick_lane(biasF16[index >> 1u], index); }
fn swish(x: f32) -> f32 { return x / (1.0 + exp(-x)); }

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let lane = lid.x;
  let value = swish(inputVec[lane]);
  partial[lane] = value;
  workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride = stride / 2u) {
    if (lane < stride) { partial[lane] = partial[lane] + partial[lane + stride]; }
    workgroupBarrier();
  }
  if (lane == 0u) { meanValue = partial[0] / 256.0; }
  workgroupBarrier();
  let centered = value - meanValue;
  partial[lane] = centered * centered;
  workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride = stride / 2u) {
    if (lane < stride) { partial[lane] = partial[lane] + partial[lane + stride]; }
    workgroupBarrier();
  }
  if (lane == 0u) { invStdValue = inverseSqrt(partial[0] / 256.0 + 0.001); }
  workgroupBarrier();
  outputVec[lane] = centered * invStdValue * load_scale(lane) + load_bias(lane);
}
`;

const SMOLGEN_DENSE2_WGSL = `
@group(0) @binding(0) var<storage, read> inputVec: array<f32>;
@group(0) @binding(1) var<storage, read> weightF16: array<u32>;
@group(0) @binding(2) var<storage, read> biasF16: array<u32>;
@group(0) @binding(3) var<storage, read_write> outputVec: array<f32>;

fn pick_lane(word: u32, index: u32) -> f32 {
  let pair = unpack2x16float(word);
  return select(pair.x, pair.y, (index & 1u) == 1u);
}
fn load_weight(index: u32) -> f32 { return pick_lane(weightF16[index >> 1u], index); }
fn load_bias(index: u32) -> f32 { return pick_lane(biasF16[index >> 1u], index); }

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let col = gid.x;
  if (col >= 2048u) { return; }
  var sum = load_bias(col);
  for (var row = 0u; row < 256u; row = row + 1u) {
    sum = sum + inputVec[row] * load_weight(row * 2048u + col);
  }
  outputVec[col] = sum;
}
`;

const SMOLGEN_SWISH_LN2_WGSL = `
@group(0) @binding(0) var<storage, read> inputVec: array<f32>;
@group(0) @binding(1) var<storage, read> scaleF16: array<u32>;
@group(0) @binding(2) var<storage, read> biasF16: array<u32>;
@group(0) @binding(3) var<storage, read_write> outputVec: array<f32>;
var<workgroup> partial: array<f32, 256>;
var<workgroup> meanValue: f32;
var<workgroup> invStdValue: f32;

fn pick_lane(word: u32, index: u32) -> f32 {
  let pair = unpack2x16float(word);
  return select(pair.x, pair.y, (index & 1u) == 1u);
}
fn load_scale(index: u32) -> f32 { return pick_lane(scaleF16[index >> 1u], index); }
fn load_bias(index: u32) -> f32 { return pick_lane(biasF16[index >> 1u], index); }
fn swish(x: f32) -> f32 { return x / (1.0 + exp(-x)); }

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let lane = lid.x;
  var sum = 0.0;
  var sq = 0.0;
  for (var i = lane; i < 2048u; i = i + 256u) {
    let value = swish(inputVec[i]);
    sum = sum + value;
    sq = sq + value * value;
  }
  partial[lane] = sum;
  workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride = stride / 2u) {
    if (lane < stride) { partial[lane] = partial[lane] + partial[lane + stride]; }
    workgroupBarrier();
  }
  if (lane == 0u) { meanValue = partial[0] / 2048.0; }
  workgroupBarrier();
  partial[lane] = sq;
  workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride = stride / 2u) {
    if (lane < stride) { partial[lane] = partial[lane] + partial[lane + stride]; }
    workgroupBarrier();
  }
  if (lane == 0u) {
    let variance = partial[0] / 2048.0 - meanValue * meanValue;
    invStdValue = inverseSqrt(max(variance, 0.0) + 0.001);
  }
  workgroupBarrier();
  for (var i = lane; i < 2048u; i = i + 256u) {
    let centered = swish(inputVec[i]) - meanValue;
    outputVec[i] = centered * invStdValue * load_scale(i) + load_bias(i);
  }
}
`;

const SMOLGEN_PROJECT_WGSL = `
@group(0) @binding(0) var<storage, read> inputVec: array<f32>;
@group(0) @binding(1) var<storage, read> weightF16: array<u32>;
@group(0) @binding(2) var<storage, read_write> outputVec: array<f32>;

fn pick_lane(word: u32, index: u32) -> f32 {
  let pair = unpack2x16float(word);
  return select(pair.x, pair.y, (index & 1u) == 1u);
}
fn load_weight(index: u32) -> f32 { return pick_lane(weightF16[index >> 1u], index); }

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let col = gid.x;
  let token = gid.y;
  let head = gid.z;
  if (col >= 64u || token >= 64u || head >= 8u) { return; }
  let outCol = token * 64u + col;
  var sum = 0.0;
  for (var row = 0u; row < 256u; row = row + 1u) {
    sum = sum + inputVec[head * 256u + row] * load_weight(row * 4096u + outCol);
  }
  outputVec[head * 4096u + outCol] = sum;
}
`;

const SMOLGEN_PROJECT_TILED_F16_WGSL = `
@group(0) @binding(0) var<storage, read> inputVec: array<f32>;
@group(0) @binding(1) var<storage, read> weightF16: array<u32>;
@group(0) @binding(2) var<storage, read_write> outputVec: array<f32>;
var<workgroup> inputTile: array<f32, 64>;

fn pick_lane(word: u32, index: u32) -> f32 {
  let pair = unpack2x16float(word);
  return select(pair.x, pair.y, (index & 1u) == 1u);
}
fn load_weight(index: u32) -> f32 { return pick_lane(weightF16[index >> 1u], index); }

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let col = wid.x * 64u + lid.x;
  let head = wid.z;
  var sum = 0.0;
  for (var tile = 0u; tile < 256u; tile = tile + 64u) {
    inputTile[lid.x] = inputVec[head * 256u + tile + lid.x];
    workgroupBarrier();
    for (var k = 0u; k < 64u; k = k + 1u) {
      sum = sum + inputTile[k] * load_weight((tile + k) * 4096u + col);
    }
    workgroupBarrier();
  }
  outputVec[head * 4096u + col] = sum;
}
`;

type Lc0WebSmolgenKernelVariant = 'hand' | 'tiled-project-f16';

type SmolgenPipelines = {
  compress: PipelineLike;
  compressBind: unknown;
  dense1: PipelineLike;
  dense1Bind: unknown;
  ln1: PipelineLike;
  ln1Bind: unknown;
  dense2: PipelineLike;
  dense2Bind: unknown;
  ln2: PipelineLike;
  ln2Bind: unknown;
  projectKernelVariant: Lc0WebSmolgenKernelVariant;
  project: PipelineLike;
  projectBind: unknown;
};

function createSmolgenPipelines(device: DeviceLike, buffers: {
  input: BufferLike;
  compressWeight: BufferLike;
  compressed: BufferLike;
  dense1Weight: BufferLike;
  dense1Bias: BufferLike;
  dense1: BufferLike;
  ln1Scale: BufferLike;
  ln1Bias: BufferLike;
  ln1: BufferLike;
  dense2Weight: BufferLike;
  dense2Bias: BufferLike;
  dense2: BufferLike;
  ln2Scale: BufferLike;
  ln2Bias: BufferLike;
  ln2: BufferLike;
  smolgenWeight: BufferLike;
  output: BufferLike;
}, projectKernelVariant: Lc0WebSmolgenKernelVariant = 'hand'): SmolgenPipelines {
  const compress = device.createComputePipeline({ layout: 'auto', compute: { module: device.createShaderModule({ label: 'lc0web smolgen compress', code: SMOLGEN_COMPRESS_WGSL }), entryPoint: 'main' } }) as PipelineLike;
  const compressBind = device.createBindGroup({ layout: compress.getBindGroupLayout(0), entries: [
    { binding: 0, resource: { buffer: buffers.input } },
    { binding: 1, resource: { buffer: buffers.compressWeight } },
    { binding: 2, resource: { buffer: buffers.compressed } },
  ] });
  const dense1 = device.createComputePipeline({ layout: 'auto', compute: { module: device.createShaderModule({ label: 'lc0web smolgen dense1', code: SMOLGEN_DENSE1_WGSL }), entryPoint: 'main' } }) as PipelineLike;
  const dense1Bind = device.createBindGroup({ layout: dense1.getBindGroupLayout(0), entries: [
    { binding: 0, resource: { buffer: buffers.compressed } },
    { binding: 1, resource: { buffer: buffers.dense1Weight } },
    { binding: 2, resource: { buffer: buffers.dense1Bias } },
    { binding: 3, resource: { buffer: buffers.dense1 } },
  ] });
  const ln1 = device.createComputePipeline({ layout: 'auto', compute: { module: device.createShaderModule({ label: 'lc0web smolgen ln1', code: SMOLGEN_SWISH_LN1_WGSL }), entryPoint: 'main' } }) as PipelineLike;
  const ln1Bind = device.createBindGroup({ layout: ln1.getBindGroupLayout(0), entries: [
    { binding: 0, resource: { buffer: buffers.dense1 } },
    { binding: 1, resource: { buffer: buffers.ln1Scale } },
    { binding: 2, resource: { buffer: buffers.ln1Bias } },
    { binding: 3, resource: { buffer: buffers.ln1 } },
  ] });
  const dense2 = device.createComputePipeline({ layout: 'auto', compute: { module: device.createShaderModule({ label: 'lc0web smolgen dense2', code: SMOLGEN_DENSE2_WGSL }), entryPoint: 'main' } }) as PipelineLike;
  const dense2Bind = device.createBindGroup({ layout: dense2.getBindGroupLayout(0), entries: [
    { binding: 0, resource: { buffer: buffers.ln1 } },
    { binding: 1, resource: { buffer: buffers.dense2Weight } },
    { binding: 2, resource: { buffer: buffers.dense2Bias } },
    { binding: 3, resource: { buffer: buffers.dense2 } },
  ] });
  const ln2 = device.createComputePipeline({ layout: 'auto', compute: { module: device.createShaderModule({ label: 'lc0web smolgen ln2', code: SMOLGEN_SWISH_LN2_WGSL }), entryPoint: 'main' } }) as PipelineLike;
  const ln2Bind = device.createBindGroup({ layout: ln2.getBindGroupLayout(0), entries: [
    { binding: 0, resource: { buffer: buffers.dense2 } },
    { binding: 1, resource: { buffer: buffers.ln2Scale } },
    { binding: 2, resource: { buffer: buffers.ln2Bias } },
    { binding: 3, resource: { buffer: buffers.ln2 } },
  ] });
  const project = device.createComputePipeline({ layout: 'auto', compute: { module: device.createShaderModule({ label: projectKernelVariant === 'tiled-project-f16' ? 'lc0web smolgen project tiled f16' : 'lc0web smolgen project', code: projectKernelVariant === 'tiled-project-f16' ? SMOLGEN_PROJECT_TILED_F16_WGSL : SMOLGEN_PROJECT_WGSL }), entryPoint: 'main' } }) as PipelineLike;
  const projectBind = device.createBindGroup({ layout: project.getBindGroupLayout(0), entries: [
    { binding: 0, resource: { buffer: buffers.ln2 } },
    { binding: 1, resource: { buffer: buffers.smolgenWeight } },
    { binding: 2, resource: { buffer: buffers.output } },
  ] });
  return { compress, compressBind, dense1, dense1Bind, ln1, ln1Bind, dense2, dense2Bind, ln2, ln2Bind, projectKernelVariant, project, projectBind };
}

function encodeSmolgenCompressPass(pass: ComputePassLike, pipelines: SmolgenPipelines): void {
  pass.setPipeline(pipelines.compress);
  pass.setBindGroup(0, pipelines.compressBind);
  pass.dispatchWorkgroups(Math.ceil(DEFAULT_SMOLGEN_COMPRESSED / 8), Math.ceil(DEFAULT_TOKENS / 8));
}

function encodeSmolgenDense1Pass(pass: ComputePassLike, pipelines: SmolgenPipelines): void {
  pass.setPipeline(pipelines.dense1);
  pass.setBindGroup(0, pipelines.dense1Bind);
  pass.dispatchWorkgroups(Math.ceil(DEFAULT_SMOLGEN_HIDDEN / 16));
}

function encodeSmolgenLn1Pass(pass: ComputePassLike, pipelines: SmolgenPipelines): void {
  pass.setPipeline(pipelines.ln1);
  pass.setBindGroup(0, pipelines.ln1Bind);
  pass.dispatchWorkgroups(1);
}

function encodeSmolgenDense2Pass(pass: ComputePassLike, pipelines: SmolgenPipelines): void {
  pass.setPipeline(pipelines.dense2);
  pass.setBindGroup(0, pipelines.dense2Bind);
  pass.dispatchWorkgroups(Math.ceil(DEFAULT_SMOLGEN_FLAT / 64));
}

function encodeSmolgenLn2Pass(pass: ComputePassLike, pipelines: SmolgenPipelines): void {
  pass.setPipeline(pipelines.ln2);
  pass.setBindGroup(0, pipelines.ln2Bind);
  pass.dispatchWorkgroups(1);
}

function encodeSmolgenProjectPass(pass: ComputePassLike, pipelines: SmolgenPipelines): void {
  pass.setPipeline(pipelines.project);
  pass.setBindGroup(0, pipelines.projectBind);
  if (pipelines.projectKernelVariant === 'tiled-project-f16') pass.dispatchWorkgroups(Math.ceil((DEFAULT_TOKENS * DEFAULT_TOKENS) / 64), 1, DEFAULT_HEADS);
  else pass.dispatchWorkgroups(Math.ceil(DEFAULT_TOKENS / 8), Math.ceil(DEFAULT_TOKENS / 8), DEFAULT_HEADS);
}

function encodeSmolgenPass(pass: ComputePassLike, pipelines: SmolgenPipelines): void {
  encodeSmolgenCompressPass(pass, pipelines);
  encodeSmolgenDense1Pass(pass, pipelines);
  encodeSmolgenLn1Pass(pass, pipelines);
  encodeSmolgenDense2Pass(pass, pipelines);
  encodeSmolgenLn2Pass(pass, pipelines);
  encodeSmolgenProjectPass(pass, pipelines);
}

function encodeSmolgenBenchmarkDispatches(device: DeviceLike, pipelines: SmolgenPipelines, iterations: number): unknown {
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  for (let i = 0; i < iterations; i++) encodeSmolgenPass(pass, pipelines);
  pass.end();
  return encoder.finish();
}

function encodeSmolgenBenchmarkStageDispatches(device: DeviceLike, iterations: number, encodeStage: (pass: ComputePassLike) => void): unknown {
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  for (let i = 0; i < iterations; i++) encodeStage(pass);
  pass.end();
  return encoder.finish();
}

async function measureSmolgenBenchmarkStage(device: DeviceLike, warmup: number, iterations: number, encodeStage: (pass: ComputePassLike) => void): Promise<number> {
  if (warmup > 0) {
    device.queue.submit([encodeSmolgenBenchmarkStageDispatches(device, warmup, encodeStage)]);
    await device.queue.onSubmittedWorkDone?.();
  }
  const started = nowMs();
  device.queue.submit([encodeSmolgenBenchmarkStageDispatches(device, iterations, encodeStage)]);
  await device.queue.onSubmittedWorkDone?.();
  return (nowMs() - started) / iterations;
}

export async function runLc0WebSmolgenBenchmark(options: Lc0WebSmolgenBenchmarkOptions): Promise<Lc0WebSmolgenBenchmarkResult> {
  const totalStarted = nowMs();
  const warmup = clampInteger(options.warmup, 3, 0, 1000);
  const iterations = clampInteger(options.iterations, 50, 1, 100_000);
  const tensorNames = lc0WebEncoderBlockTensorNames(options.encoderPrefix).smolgen;
  const { device, adapterInfo } = await requestDevice();
  const pack = await loadLc0WebModelPack(options.packUrl, {
    verifyShards: options.verifyShards ?? true,
    tensorNames: Object.values(tensorNames),
  });
  const tensors = loadEncoder0SmolgenTensors(pack, tensorNames);
  const input = makeInputTokenMatrix(DEFAULT_TOKENS, DEFAULT_N);
  const reference = cpuEncoder0SmolgenBias(input, tensors);
  const globals = gpuGlobals();
  const usage = globals.GPUBufferUsage!;
  const buffers: BufferLike[] = [];
  try {
    const setupStarted = nowMs();
    const inputBuffer = createStorageBuffer(device, input, usage.STORAGE | usage.COPY_DST);
    const compressWeight = createStorageBuffer(device, tensors.compressWeight.bytes, usage.STORAGE | usage.COPY_DST);
    const dense1Weight = createStorageBuffer(device, tensors.dense1Weight.bytes, usage.STORAGE | usage.COPY_DST);
    const dense1Bias = createStorageBuffer(device, tensors.dense1Bias.bytes, usage.STORAGE | usage.COPY_DST);
    const ln1Scale = createStorageBuffer(device, tensors.ln1Scale.bytes, usage.STORAGE | usage.COPY_DST);
    const ln1Bias = createStorageBuffer(device, tensors.ln1Bias.bytes, usage.STORAGE | usage.COPY_DST);
    const dense2Weight = createStorageBuffer(device, tensors.dense2Weight.bytes, usage.STORAGE | usage.COPY_DST);
    const dense2Bias = createStorageBuffer(device, tensors.dense2Bias.bytes, usage.STORAGE | usage.COPY_DST);
    const ln2Scale = createStorageBuffer(device, tensors.ln2Scale.bytes, usage.STORAGE | usage.COPY_DST);
    const ln2Bias = createStorageBuffer(device, tensors.ln2Bias.bytes, usage.STORAGE | usage.COPY_DST);
    const smolgenWeight = createStorageBuffer(device, tensors.smolgenWeight.bytes, usage.STORAGE | usage.COPY_DST);
    const compressed = device.createBuffer({ size: DEFAULT_SMOLGEN_FLAT * 4, usage: usage.STORAGE });
    const dense1 = device.createBuffer({ size: DEFAULT_SMOLGEN_HIDDEN * 4, usage: usage.STORAGE });
    const ln1 = device.createBuffer({ size: DEFAULT_SMOLGEN_HIDDEN * 4, usage: usage.STORAGE });
    const dense2 = device.createBuffer({ size: DEFAULT_SMOLGEN_FLAT * 4, usage: usage.STORAGE });
    const ln2 = device.createBuffer({ size: DEFAULT_SMOLGEN_FLAT * 4, usage: usage.STORAGE });
    const output = device.createBuffer({ size: DEFAULT_HEADS * DEFAULT_TOKENS * DEFAULT_TOKENS * 4, usage: usage.STORAGE | usage.COPY_SRC });
    const readbackBuffer = device.createBuffer({ size: DEFAULT_HEADS * DEFAULT_TOKENS * DEFAULT_TOKENS * 4, usage: usage.MAP_READ | usage.COPY_DST });
    buffers.push(inputBuffer, compressWeight, dense1Weight, dense1Bias, ln1Scale, ln1Bias, dense2Weight, dense2Bias, ln2Scale, ln2Bias, smolgenWeight, compressed, dense1, ln1, dense2, ln2, output, readbackBuffer);
    const pipelines = createSmolgenPipelines(device, {
      input: inputBuffer,
      compressWeight,
      compressed,
      dense1Weight,
      dense1Bias,
      dense1,
      ln1Scale,
      ln1Bias,
      ln1,
      dense2Weight,
      dense2Bias,
      dense2,
      ln2Scale,
      ln2Bias,
      ln2,
      smolgenWeight,
      output,
    }, 'tiled-project-f16');
    const uploadSetupMs = nowMs() - setupStarted;

    if (warmup > 0) {
      device.queue.submit([encodeSmolgenBenchmarkDispatches(device, pipelines, warmup)]);
      await device.queue.onSubmittedWorkDone?.();
    }
    const dispatchStarted = nowMs();
    device.queue.submit([encodeSmolgenBenchmarkDispatches(device, pipelines, iterations)]);
    const dispatchLoopMs = nowMs() - dispatchStarted;
    const readbackStarted = nowMs();
    const outputValues = await readF32OutputOnce(device, output, readbackBuffer, reference.length);
    const readbackSyncedMs = nowMs() - readbackStarted;
    const { maxAbsError, rmsError } = computeErrorStats(outputValues, reference, reference.length);
    assertErrorInTolerance(maxAbsError);

    // Re-initialize the pipeline once, then time each smolgen stage in isolation.
    device.queue.submit([encodeSmolgenBenchmarkDispatches(device, pipelines, 1)]);
    await device.queue.onSubmittedWorkDone?.();
    const stageDispatchAvgMs = {
      compress: await measureSmolgenBenchmarkStage(device, warmup, iterations, (pass) => encodeSmolgenCompressPass(pass, pipelines)),
      dense1: await measureSmolgenBenchmarkStage(device, warmup, iterations, (pass) => encodeSmolgenDense1Pass(pass, pipelines)),
      ln1: await measureSmolgenBenchmarkStage(device, warmup, iterations, (pass) => encodeSmolgenLn1Pass(pass, pipelines)),
      dense2: await measureSmolgenBenchmarkStage(device, warmup, iterations, (pass) => encodeSmolgenDense2Pass(pass, pipelines)),
      ln2: await measureSmolgenBenchmarkStage(device, warmup, iterations, (pass) => encodeSmolgenLn2Pass(pass, pipelines)),
      project: await measureSmolgenBenchmarkStage(device, warmup, iterations, (pass) => encodeSmolgenProjectPass(pass, pipelines)),
    };

    return {
      status: 'SMOLGEN_BENCH_DONE',
      packUrl: pack.manifestUrl,
      modelName: pack.manifest.model.name,
      adapterInfo,
      encoderPrefix: normalizeEncoderPrefix(options.encoderPrefix),
      tokens: DEFAULT_TOKENS,
      channels: DEFAULT_N,
      compressed: DEFAULT_SMOLGEN_COMPRESSED,
      hidden: DEFAULT_SMOLGEN_HIDDEN,
      heads: DEFAULT_HEADS,
      epsilon: DEFAULT_SMOLGEN_EPSILON,
      warmup,
      iterations,
      packLoadMs: pack.elapsedMs,
      uploadSetupMs,
      dispatchLoopMs,
      dispatchLoopAvgMs: dispatchLoopMs / iterations,
      stageDispatchAvgMs,
      readbackSyncedMs,
      endToEndMs: nowMs() - totalStarted,
      maxAbsError,
      rmsError,
      outputSample: Array.from(outputValues.slice(0, 8)),
    };
  } finally {
    for (const buffer of buffers) buffer.destroy?.();
  }
}

function createAttentionOutputPipelines(device: DeviceLike, buffers: {
  input: BufferLike;
  qWeight: BufferLike;
  qBias: BufferLike;
  kWeight: BufferLike;
  kBias: BufferLike;
  vWeight: BufferLike;
  vBias: BufferLike;
  scale: BufferLike;
  smolgenBias: BufferLike;
  qkv: BufferLike;
  scores: BufferLike;
  probs: BufferLike;
  attn: BufferLike;
  outWeight: BufferLike;
  outBias: BufferLike;
  alpha: BufferLike;
  skip: BufferLike;
  lnScale: BufferLike;
  lnBias: BufferLike;
  output: BufferLike;
  podArgs?: BufferLike;
}, outProjKernelVariant: Lc0WebAttentionOutProjKernelVariant = 'hand', qkvKernelVariant: Lc0WebAttentionQkvKernelVariant = 'hand'): ReturnType<typeof createAttentionBlockPipelines> & { outProjKernelVariant: Lc0WebAttentionOutProjKernelVariant; outProj: PipelineLike; outProjBind: unknown; norm: PipelineLike; normBind: unknown } {
  const base = createAttentionBlockPipelines(device, { ...buffers, output: buffers.attn }, qkvKernelVariant);
  let outProj: PipelineLike;
  let outProjBind: unknown;
  if (outProjKernelVariant === 'tvm-packed-f16') {
    if (!buffers.podArgs) throw new Error('TVM packed-f16 attention output projection requires a POD args uniform buffer');
    const outModule = device.createShaderModule({ label: 'lc0web attention output projection TVM packed-f16 residual', code: ATTENTION_OUTPUT_PROJ_TVM_PACKED_F16_WGSL });
    outProj = device.createComputePipeline({ layout: 'auto', compute: { module: outModule, entryPoint: 'matmul_kernel' } }) as PipelineLike;
    outProjBind = device.createBindGroup({
      layout: outProj.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: buffers.skip } },
        { binding: 1, resource: { buffer: buffers.outWeight } },
        { binding: 2, resource: { buffer: buffers.attn } },
        { binding: 3, resource: { buffer: buffers.podArgs } },
        { binding: 4, resource: { buffer: buffers.outBias } },
        { binding: 5, resource: { buffer: buffers.input } },
        { binding: 6, resource: { buffer: buffers.alpha } },
      ],
    });
  } else {
    const outModule = device.createShaderModule({ label: 'lc0web attention output projection residual', code: ATTENTION_OUTPUT_PROJ_WGSL });
    outProj = device.createComputePipeline({ layout: 'auto', compute: { module: outModule, entryPoint: 'main' } }) as PipelineLike;
    outProjBind = device.createBindGroup({
      layout: outProj.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: buffers.attn } },
        { binding: 1, resource: { buffer: buffers.input } },
        { binding: 2, resource: { buffer: buffers.outWeight } },
        { binding: 3, resource: { buffer: buffers.outBias } },
        { binding: 4, resource: { buffer: buffers.alpha } },
        { binding: 5, resource: { buffer: buffers.skip } },
      ],
    });
  }
  const normModule = device.createShaderModule({ label: 'lc0web attention output layernorm', code: ATTENTION_OUTPUT_NORM_WGSL });
  const norm = device.createComputePipeline({ layout: 'auto', compute: { module: normModule, entryPoint: 'main' } }) as PipelineLike;
  const normBind = device.createBindGroup({
    layout: norm.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: buffers.skip } },
      { binding: 1, resource: { buffer: buffers.lnScale } },
      { binding: 2, resource: { buffer: buffers.lnBias } },
      { binding: 3, resource: { buffer: buffers.output } },
    ],
  });
  return { ...base, outProjKernelVariant, outProj, outProjBind, norm, normBind };
}

function encodeAttentionOutputDispatches(device: DeviceLike, pipelines: ReturnType<typeof createAttentionOutputPipelines>, iterations: number): unknown {
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  for (let i = 0; i < iterations; i++) {
    encodeAttentionQkvPass(pass, pipelines);
    pass.setPipeline(pipelines.score);
    pass.setBindGroup(0, pipelines.scoreBind);
    pass.dispatchWorkgroups(Math.ceil(DEFAULT_TOKENS / 8), Math.ceil(DEFAULT_TOKENS / 8), DEFAULT_HEADS);
    pass.setPipeline(pipelines.softmax);
    pass.setBindGroup(0, pipelines.softmaxBind);
    pass.dispatchWorkgroups(DEFAULT_HEADS * DEFAULT_TOKENS);
    pass.setPipeline(pipelines.value);
    pass.setBindGroup(0, pipelines.valueBind);
    pass.dispatchWorkgroups(Math.ceil(DEFAULT_HEAD_DIM / 8), Math.ceil(DEFAULT_TOKENS / 8), DEFAULT_HEADS);
    pass.setPipeline(pipelines.outProj);
    pass.setBindGroup(0, pipelines.outProjBind);
    if (pipelines.outProjKernelVariant === 'tvm-packed-f16') pass.dispatchWorkgroups(2, 8, 1);
    else pass.dispatchWorkgroups(Math.ceil(DEFAULT_N / 8), Math.ceil(DEFAULT_TOKENS / 8));
    pass.setPipeline(pipelines.norm);
    pass.setBindGroup(0, pipelines.normBind);
    pass.dispatchWorkgroups(DEFAULT_TOKENS);
  }
  pass.end();
  return encoder.finish();
}

export async function runLc0WebAttentionOutputBenchmark(options: Lc0WebAttentionOutputBenchmarkOptions): Promise<Lc0WebAttentionOutputBenchmarkResult> {
  const totalStarted = nowMs();
  const warmup = clampInteger(options.warmup, 3, 0, 1000);
  const iterations = clampInteger(options.iterations, 50, 1, 10_000);
  const outProjKernelVariant = options.attentionOutProjKernelVariant ?? 'hand';
  const { device, adapterInfo } = await requestDevice();
  const tensorNames = lc0WebEncoderBlockTensorNames(options.encoderPrefix);
  const pack = await loadLc0WebModelPack(options.packUrl, {
    verifyShards: options.verifyShards ?? true,
    tensorNames: [
      ...Object.values(tensorNames.qkv), tensorNames.scaleTensor, ...Object.values(tensorNames.smolgen),
      tensorNames.outDenseWeight, tensorNames.outDenseBias, tensorNames.outAlpha, tensorNames.ln1Scale, tensorNames.ln1Bias,
    ],
  });
  const tensors = loadAttentionOutputInputs(pack, tensorNames);
  const reference = buildAttentionOutputReference(tensors);
  const outputElements = DEFAULT_TOKENS * DEFAULT_N;
  const globals = gpuGlobals();
  const usage = globals.GPUBufferUsage!;
  const buffers: BufferLike[] = [];
  try {
    const setupStarted = nowMs();
    const inputBuffer = createStorageBuffer(device, reference.input, usage.STORAGE | usage.COPY_DST);
    const qWeight = createTransposedF16StorageBuffer(device, tensors.qWeight.bytes, DEFAULT_K, DEFAULT_N, usage.STORAGE | usage.COPY_DST);
    const qBias = createStorageBuffer(device, tensors.qBias.bytes, usage.STORAGE | usage.COPY_DST);
    const kWeight = createTransposedF16StorageBuffer(device, tensors.kWeight.bytes, DEFAULT_K, DEFAULT_N, usage.STORAGE | usage.COPY_DST);
    const kBias = createStorageBuffer(device, tensors.kBias.bytes, usage.STORAGE | usage.COPY_DST);
    const vWeight = createTransposedF16StorageBuffer(device, tensors.vWeight.bytes, DEFAULT_K, DEFAULT_N, usage.STORAGE | usage.COPY_DST);
    const vBias = createStorageBuffer(device, tensors.vBias.bytes, usage.STORAGE | usage.COPY_DST);
    const scale = createStorageBuffer(device, paddedF16ScalarBytes(tensors.scale.bytes), usage.STORAGE | usage.COPY_DST);
    const smolgenBias = createStorageBuffer(device, reference.smolgenBias, usage.STORAGE | usage.COPY_DST);
    const outWeight = createTransposedF16StorageBuffer(device, tensors.outWeight.bytes, DEFAULT_N, DEFAULT_N, usage.STORAGE | usage.COPY_DST);
    const outBias = createStorageBuffer(device, tensors.outBias.bytes, usage.STORAGE | usage.COPY_DST);
    const alpha = createStorageBuffer(device, paddedF16ScalarBytes(tensors.alpha.bytes), usage.STORAGE | usage.COPY_DST);
    const lnScale = createStorageBuffer(device, tensors.lnScale.bytes, usage.STORAGE | usage.COPY_DST);
    const lnBias = createStorageBuffer(device, tensors.lnBias.bytes, usage.STORAGE | usage.COPY_DST);
    const qkvBuffer = device.createBuffer({ size: DEFAULT_TOKENS * DEFAULT_N * 3 * 4, usage: usage.STORAGE });
    const scoreBuffer = device.createBuffer({ size: DEFAULT_HEADS * DEFAULT_TOKENS * DEFAULT_TOKENS * 4, usage: usage.STORAGE });
    const probBuffer = device.createBuffer({ size: DEFAULT_HEADS * DEFAULT_TOKENS * DEFAULT_TOKENS * 4, usage: usage.STORAGE });
    const attnBuffer = device.createBuffer({ size: outputElements * 4, usage: usage.STORAGE });
    const skipBuffer = device.createBuffer({ size: outputElements * 4, usage: usage.STORAGE });
    const outputBuffer = device.createBuffer({ size: outputElements * 4, usage: usage.STORAGE | usage.COPY_SRC });
    const readbackBuffer = device.createBuffer({ size: outputElements * 4, usage: usage.MAP_READ | usage.COPY_DST });
    const podArgs = outProjKernelVariant === 'tvm-packed-f16' ? createU32UniformBuffer(device, [1], usage.UNIFORM | usage.COPY_DST) : undefined;
    buffers.push(inputBuffer, qWeight, qBias, kWeight, kBias, vWeight, vBias, scale, smolgenBias, outWeight, outBias, alpha, lnScale, lnBias, qkvBuffer, scoreBuffer, probBuffer, attnBuffer, skipBuffer, outputBuffer, readbackBuffer);
    if (podArgs) buffers.push(podArgs);
    const pipelines = createAttentionOutputPipelines(device, { input: inputBuffer, qWeight, qBias, kWeight, kBias, vWeight, vBias, scale, smolgenBias, qkv: qkvBuffer, scores: scoreBuffer, probs: probBuffer, attn: attnBuffer, outWeight, outBias, alpha, skip: skipBuffer, lnScale, lnBias, output: outputBuffer, podArgs }, outProjKernelVariant);
    const uploadSetupMs = nowMs() - setupStarted;

    if (warmup > 0) {
      device.queue.submit([encodeAttentionOutputDispatches(device, pipelines, warmup)]);
      await device.queue.onSubmittedWorkDone?.();
    }
    const dispatchStarted = nowMs();
    device.queue.submit([encodeAttentionOutputDispatches(device, pipelines, iterations)]);
    const dispatchLoopMs = nowMs() - dispatchStarted;
    const readbackStarted = nowMs();
    const output = await readF32OutputOnce(device, outputBuffer, readbackBuffer, outputElements);
    const readbackSyncedMs = nowMs() - readbackStarted;
    const { maxAbsError, rmsError } = computeErrorStats(output, reference.output, outputElements);
    assertErrorInTolerance(maxAbsError);
    return {
      status: 'ATTENTION_OUTPUT_BENCH_DONE',
      packUrl: pack.manifestUrl,
      modelName: pack.manifest.model.name,
      adapterInfo,
      tokens: DEFAULT_TOKENS,
      channels: DEFAULT_N,
      heads: DEFAULT_HEADS,
      headDim: DEFAULT_HEAD_DIM,
      epsilon: DEFAULT_LN_EPSILON,
      alpha: reference.alpha,
      outProjKernelVariant,
      dispatchesPerIteration: 6,
      warmup,
      iterations,
      packLoadMs: pack.elapsedMs,
      uploadSetupMs,
      dispatchLoopMs,
      dispatchLoopAvgMs: dispatchLoopMs / iterations,
      readbackSyncedMs,
      endToEndMs: nowMs() - totalStarted,
      maxAbsError,
      rmsError,
      outputSample: Array.from(output.slice(0, 8)),
    };
  } finally {
    for (const buffer of buffers) buffer.destroy?.();
  }
}

export function createTinyAttentionOutputOnnxForTest(
  outWeight: Float32Array<ArrayBufferLike>,
  outBias: Float32Array<ArrayBufferLike>,
  alpha: number,
  lnScale: Float32Array<ArrayBufferLike>,
  lnBias: Float32Array<ArrayBufferLike>,
): Uint8Array {
  const writer = new ProtoWriter();
  writer.int64(1, 8);
  writer.string(2, 'lc0web');
  writer.message(7, (graph) => {
    graph.bytes(1, onnxNode('MatMul', ['attention', 'outWeight'], ['projected'], 'attention_output_matmul'));
    graph.bytes(1, onnxNode('Add', ['projected', 'outBias'], ['biased'], 'attention_output_bias'));
    graph.bytes(1, onnxNode('Mul', ['biased', 'alpha'], ['scaled'], 'attention_output_alpha'));
    graph.bytes(1, onnxNode('Add', ['scaled', 'residual'], ['skip'], 'attention_output_residual'));
    graph.bytes(1, onnxNode('LayerNormalization', ['skip', 'lnScale', 'lnBias'], ['output'], 'attention_output_ln1', [onnxFloatAttribute('epsilon', DEFAULT_LN_EPSILON)]));
    graph.string(2, 'lc0web_attention_output_projection_residual_ln1');
    graph.bytes(5, onnxTensor('outWeight', [DEFAULT_N, DEFAULT_N], outWeight));
    graph.bytes(5, onnxTensor('outBias', [DEFAULT_N], outBias));
    graph.bytes(5, onnxTensor('alpha', [1], new Float32Array([alpha])));
    graph.bytes(5, onnxTensor('lnScale', [DEFAULT_N], lnScale));
    graph.bytes(5, onnxTensor('lnBias', [DEFAULT_N], lnBias));
    graph.bytes(11, onnxValueInfo('attention', 1, [DEFAULT_TOKENS, DEFAULT_N]));
    graph.bytes(11, onnxValueInfo('residual', 1, [DEFAULT_TOKENS, DEFAULT_N]));
    graph.bytes(12, onnxValueInfo('output', 1, [DEFAULT_TOKENS, DEFAULT_N]));
  });
  writer.message(8, (opset) => opset.int64(2, 17));
  return writer.finish();
}

export async function runLc0WebAttentionOutputOrtBenchmark(options: Lc0WebAttentionOutputOrtBenchmarkOptions): Promise<Lc0WebAttentionOutputOrtBenchmarkResult> {
  const warmup = clampInteger(options.warmup, 5, 0, 100);
  const iterations = clampInteger(options.iterations, 25, 1, 1000);
  const tensorNames = lc0WebEncoderBlockTensorNames(options.encoderPrefix);
  const pack = await loadLc0WebModelPack(options.packUrl, {
    verifyShards: options.verifyShards ?? true,
    tensorNames: [
      ...Object.values(tensorNames.qkv), tensorNames.scaleTensor, ...Object.values(tensorNames.smolgen),
      tensorNames.outDenseWeight, tensorNames.outDenseBias, tensorNames.outAlpha, tensorNames.ln1Scale, tensorNames.ln1Bias,
    ],
  });
  const tensors = loadAttentionOutputInputs(pack, tensorNames);
  const attentionReference = buildAttentionValueReference(tensors);
  const reference = buildAttentionOutputReference(tensors);
  const outputElements = DEFAULT_TOKENS * DEFAULT_N;
  const modelBuildStarted = nowMs();
  const tinyOnnx = createTinyAttentionOutputOnnxForTest(
    f16BytesToF32Array(tensors.outWeight.bytes, DEFAULT_N * DEFAULT_N),
    f16BytesToF32Array(tensors.outBias.bytes, DEFAULT_N),
    reference.alpha,
    f16BytesToF32Array(tensors.lnScale.bytes, DEFAULT_N),
    f16BytesToF32Array(tensors.lnBias.bytes, DEFAULT_N),
  );
  const modelBuildMs = nowMs() - modelBuildStarted;
  const sessionStarted = nowMs();
  const session = await ort.createOrtSession(tinyOnnx);
  const sessionCreateMs = nowMs() - sessionStarted;
  const feeds = {
    attention: new ort.Tensor('float32', attentionReference.output, [DEFAULT_TOKENS, DEFAULT_N]),
    residual: new ort.Tensor('float32', reference.input, [DEFAULT_TOKENS, DEFAULT_N]),
  };
  let output: Float32Array<ArrayBufferLike> = new Float32Array(outputElements);
  for (let i = 0; i < warmup; i++) {
    const outputs = await session.run(feeds);
    output = outputs.output.data as Float32Array<ArrayBufferLike>;
  }
  const times: number[] = [];
  for (let i = 0; i < iterations; i++) {
    const started = nowMs();
    const outputs = await session.run(feeds);
    times.push(nowMs() - started);
    output = outputs.output.data as Float32Array<ArrayBufferLike>;
  }
  const { maxAbsError, rmsError } = computeErrorStats(output, reference.output, outputElements);
  assertErrorInTolerance(maxAbsError);
  const avgMs = times.reduce((sum, value) => sum + value, 0) / times.length;
  return {
    status: 'ATTENTION_OUTPUT_ORT_BENCH_DONE',
    packUrl: pack.manifestUrl,
    modelName: pack.manifest.model.name,
    tokens: DEFAULT_TOKENS,
    channels: DEFAULT_N,
    epsilon: DEFAULT_LN_EPSILON,
    alpha: reference.alpha,
    warmup,
    iterations,
    packLoadMs: pack.elapsedMs,
    modelBuildMs,
    sessionCreateMs,
    avgMs,
    minMs: Math.min(...times),
    maxMs: Math.max(...times),
    firstMs: times[0],
    timesMs: times,
    runsPerSecond: 1000 / avgMs,
    maxAbsError,
    rmsError,
    outputSample: Array.from(output.slice(0, 8)),
  };
}

const DEFAULT_FFN_DENSE1_WEIGHT = '/encoder0/ffn/dense1/w/w';
const DEFAULT_FFN_DENSE1_BIAS = '/encoder0/ffn/dense1/b/w';
const DEFAULT_FFN_DENSE2_WEIGHT = '/encoder0/ffn/dense2/w/w';
const DEFAULT_FFN_DENSE2_BIAS = '/encoder0/ffn/dense2/b/w';
const DEFAULT_FFN_ALPHA = '/encoder0/ffn/alpha/w';
const DEFAULT_LN2_SCALE = '/encoder0/ln2/w/scale';
const DEFAULT_LN2_BIAS = '/encoder0/ln2/w/bias';
const DEFAULT_FFN_HIDDEN = 1024;
const DEFAULT_ENCODER_PREFIX = '/encoder0';

type Lc0WebEncoderBlockTensorNames = {
  qkv: Record<keyof typeof DEFAULT_QKV_TENSORS, string>;
  scaleTensor: string;
  smolgen: Record<keyof typeof DEFAULT_SMOLGEN_TENSORS, string>;
  outDenseWeight: string;
  outDenseBias: string;
  outAlpha: string;
  ln1Scale: string;
  ln1Bias: string;
  ffnDense1Weight: string;
  ffnDense1Bias: string;
  ffnDense2Weight: string;
  ffnDense2Bias: string;
  ffnAlpha: string;
  ln2Scale: string;
  ln2Bias: string;
};

function normalizeEncoderPrefix(prefix: string | undefined): string {
  const raw = (prefix ?? DEFAULT_ENCODER_PREFIX).trim().replace(/\/+$/, '');
  if (!raw || !raw.startsWith('/')) throw new Error(`Invalid lc0web encoder prefix: ${prefix}`);
  return raw;
}

export function lc0WebEncoderBlockTensorNames(prefix?: string): Lc0WebEncoderBlockTensorNames {
  const p = normalizeEncoderPrefix(prefix);
  return {
    qkv: {
      qWeight: `${p}/mha/Q/w/w`,
      qBias: `${p}/mha/Q/b/w`,
      kWeight: `${p}/mha/K/w/w`,
      kBias: `${p}/mha/K/b/w`,
      vWeight: `${p}/mha/V/w/w`,
      vBias: `${p}/mha/V/b/w`,
    },
    scaleTensor: `${p}/mha/QK/scale/w`,
    smolgen: {
      compressWeight: `${p}/smolgen/compress/w`,
      dense1Weight: `${p}/smolgen/dense1/w/w`,
      dense1Bias: `${p}/smolgen/dense1/b/w`,
      ln1Scale: `${p}/smolgen/ln1/w/scale`,
      ln1Bias: `${p}/smolgen/ln1/w/bias`,
      dense2Weight: `${p}/smolgen/dense2/w/w`,
      dense2Bias: `${p}/smolgen/dense2/b/w`,
      ln2Scale: `${p}/smolgen/ln2/w/scale`,
      ln2Bias: `${p}/smolgen/ln2/w/bias`,
      smolgenWeight: DEFAULT_SMOLGEN_TENSORS.smolgenWeight,
    },
    outDenseWeight: `${p}/mha/out/dense/w/w`,
    outDenseBias: `${p}/mha/out/dense/b/w`,
    outAlpha: `${p}/alpha*input/w`,
    ln1Scale: `${p}/ln1/w/scale`,
    ln1Bias: `${p}/ln1/w/bias`,
    ffnDense1Weight: `${p}/ffn/dense1/w/w`,
    ffnDense1Bias: `${p}/ffn/dense1/b/w`,
    ffnDense2Weight: `${p}/ffn/dense2/w/w`,
    ffnDense2Bias: `${p}/ffn/dense2/b/w`,
    ffnAlpha: `${p}/ffn/alpha/w`,
    ln2Scale: `${p}/ln2/w/scale`,
    ln2Bias: `${p}/ln2/w/bias`,
  };
}

function encoderBlockTensorNameList(names: Lc0WebEncoderBlockTensorNames): string[] {
  return [
    ...Object.values(names.qkv), names.scaleTensor, ...Object.values(names.smolgen),
    names.outDenseWeight, names.outDenseBias, names.outAlpha, names.ln1Scale, names.ln1Bias,
    names.ffnDense1Weight, names.ffnDense1Bias, names.ffnDense2Weight, names.ffnDense2Bias,
    names.ffnAlpha, names.ln2Scale, names.ln2Bias,
  ];
}

export type Lc0WebFfnKernelVariant = 'hand' | 'tvm-packed-f16';

export interface Lc0WebEncoder0FfnBenchmarkOptions {
  packUrl: string;
  iterations?: number;
  warmup?: number;
  verifyShards?: boolean;
  encoderPrefix?: string;
  ffnKernelVariant?: Lc0WebFfnKernelVariant;
}

export interface Lc0WebEncoder0FfnBenchmarkResult {
  status: 'FFN_BENCH_DONE';
  packUrl: string;
  modelName: string;
  adapterInfo?: Record<string, unknown>;
  tokens: number;
  channels: number;
  hidden: number;
  epsilon: number;
  alpha: number;
  ffnKernelVariant: Lc0WebFfnKernelVariant;
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
}

export interface Lc0WebEncoder0FfnOrtBenchmarkResult {
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
  timesMs: number[];
  runsPerSecond: number;
  maxAbsError: number;
  rmsError: number;
  outputSample: number[];
}

type Encoder0FfnTensors = ReturnType<typeof loadAttentionOutputInputs> & {
  ffnDense1Weight: Lc0WebTensorView;
  ffnDense1Bias: Lc0WebTensorView;
  ffnDense2Weight: Lc0WebTensorView;
  ffnDense2Bias: Lc0WebTensorView;
  ffnAlpha: Lc0WebTensorView;
  ln2Scale: Lc0WebTensorView;
  ln2Bias: Lc0WebTensorView;
};

function loadEncoder0FfnInputs(pack: Awaited<ReturnType<typeof loadLc0WebModelPack>>, tensorNames: Lc0WebEncoderBlockTensorNames = lc0WebEncoderBlockTensorNames()): Encoder0FfnTensors {
  const base = loadAttentionOutputInputs(pack, tensorNames);
  const ffnDense1Weight = pack.tensors.get(tensorNames.ffnDense1Weight);
  const ffnDense1Bias = pack.tensors.get(tensorNames.ffnDense1Bias);
  const ffnDense2Weight = pack.tensors.get(tensorNames.ffnDense2Weight);
  const ffnDense2Bias = pack.tensors.get(tensorNames.ffnDense2Bias);
  const ffnAlpha = pack.tensors.get(tensorNames.ffnAlpha);
  const ln2Scale = pack.tensors.get(tensorNames.ln2Scale);
  const ln2Bias = pack.tensors.get(tensorNames.ln2Bias);
  if (!ffnDense1Weight || !ffnDense1Bias || !ffnDense2Weight || !ffnDense2Bias || !ffnAlpha || !ln2Scale || !ln2Bias) throw new Error('lc0web encoder0 FFN tensors were not loaded');
  assertTensorShapeAndBytes(ffnDense1Weight, [DEFAULT_N, DEFAULT_FFN_HIDDEN], 2, 'ffnDense1Weight');
  assertTensorShapeAndBytes(ffnDense1Bias, [DEFAULT_FFN_HIDDEN], 2, 'ffnDense1Bias');
  assertTensorShapeAndBytes(ffnDense2Weight, [DEFAULT_FFN_HIDDEN, DEFAULT_N], 2, 'ffnDense2Weight');
  assertTensorShapeAndBytes(ffnDense2Bias, [DEFAULT_N], 2, 'ffnDense2Bias');
  assertTensorShapeAndBytes(ffnAlpha, [1], 2, 'ffnAlpha');
  assertTensorShapeAndBytes(ln2Scale, [DEFAULT_N], 2, 'ln2Scale');
  assertTensorShapeAndBytes(ln2Bias, [DEFAULT_N], 2, 'ln2Bias');
  for (const tensor of [ffnDense1Weight, ffnDense1Bias, ffnDense2Weight, ffnDense2Bias, ffnAlpha, ln2Scale, ln2Bias]) {
    if (tensor.info.dtype !== 'f16') throw new Error(`lc0web encoder0 FFN expects f16 tensor ${tensor.info.name}, got ${tensor.info.dtype}`);
  }
  return { ...base, ffnDense1Weight, ffnDense1Bias, ffnDense2Weight, ffnDense2Bias, ffnAlpha, ln2Scale, ln2Bias };
}

function cpuSqrRelu(input: Float32Array<ArrayBufferLike>): Float32Array<ArrayBufferLike> {
  const output = new Float32Array(input.length);
  for (let i = 0; i < input.length; i++) {
    const value = Math.max(0, input[i]);
    output[i] = value * value;
  }
  return output;
}

function cpuEncoder0FfnFromLn1(input: Float32Array<ArrayBufferLike>, tensors: Encoder0FfnTensors): { output: Float32Array<ArrayBufferLike>; alpha: number } {
  const hidden = cpuSqrRelu(cpuProjectTokens(input, tensors.ffnDense1Weight.bytes, tensors.ffnDense1Bias.bytes, DEFAULT_TOKENS, DEFAULT_N, DEFAULT_FFN_HIDDEN));
  const projected = cpuProjectTokens(hidden, tensors.ffnDense2Weight.bytes, tensors.ffnDense2Bias.bytes, DEFAULT_TOKENS, DEFAULT_FFN_HIDDEN, DEFAULT_N);
  const alpha = readF16At(tensors.ffnAlpha.bytes, 0);
  const skip = new Float32Array(projected.length);
  for (let i = 0; i < projected.length; i++) skip[i] = projected[i] * alpha + input[i];
  return { output: cpuLayerNormRows(skip, tensors.ln2Scale.bytes, tensors.ln2Bias.bytes, DEFAULT_TOKENS, DEFAULT_N, DEFAULT_LN_EPSILON), alpha };
}

function buildEncoder0FfnReference(tensors: Encoder0FfnTensors): { input: Float32Array<ArrayBufferLike>; output: Float32Array<ArrayBufferLike>; alpha: number } {
  const input = buildAttentionOutputReference(tensors).output;
  const ffn = cpuEncoder0FfnFromLn1(input, tensors);
  return { input, output: ffn.output, alpha: ffn.alpha };
}

export function createTinyEncoder0FfnOnnxForTest(
  dense1Weight: Float32Array<ArrayBufferLike>,
  dense1Bias: Float32Array<ArrayBufferLike>,
  dense2Weight: Float32Array<ArrayBufferLike>,
  dense2Bias: Float32Array<ArrayBufferLike>,
  alpha: number,
  ln2Scale: Float32Array<ArrayBufferLike>,
  ln2Bias: Float32Array<ArrayBufferLike>,
): Uint8Array {
  const writer = new ProtoWriter();
  writer.int64(1, 8);
  writer.string(2, 'lc0web');
  writer.message(7, (graph) => {
    graph.bytes(1, onnxNode('MatMul', ['input', 'dense1Weight'], ['dense1'], 'encoder0_ffn_dense1_matmul'));
    graph.bytes(1, onnxNode('Add', ['dense1', 'dense1Bias'], ['dense1Biased'], 'encoder0_ffn_dense1_bias'));
    graph.bytes(1, onnxNode('Relu', ['dense1Biased'], ['dense1Relu'], 'encoder0_ffn_relu'));
    graph.bytes(1, onnxNode('Mul', ['dense1Relu', 'dense1Relu'], ['hidden'], 'encoder0_ffn_sqrrelu'));
    graph.bytes(1, onnxNode('MatMul', ['hidden', 'dense2Weight'], ['dense2'], 'encoder0_ffn_dense2_matmul'));
    graph.bytes(1, onnxNode('Add', ['dense2', 'dense2Bias'], ['dense2Biased'], 'encoder0_ffn_dense2_bias'));
    graph.bytes(1, onnxNode('Mul', ['dense2Biased', 'alpha'], ['scaled'], 'encoder0_ffn_alpha'));
    graph.bytes(1, onnxNode('Add', ['scaled', 'input'], ['skip'], 'encoder0_ffn_residual'));
    graph.bytes(1, onnxNode('LayerNormalization', ['skip', 'ln2Scale', 'ln2Bias'], ['output'], 'encoder0_ffn_ln2', [onnxFloatAttribute('epsilon', DEFAULT_LN_EPSILON)]));
    graph.string(2, 'lc0web_encoder0_ffn_sqrrelu_residual_ln2');
    graph.bytes(5, onnxTensor('dense1Weight', [DEFAULT_N, DEFAULT_FFN_HIDDEN], dense1Weight));
    graph.bytes(5, onnxTensor('dense1Bias', [DEFAULT_FFN_HIDDEN], dense1Bias));
    graph.bytes(5, onnxTensor('dense2Weight', [DEFAULT_FFN_HIDDEN, DEFAULT_N], dense2Weight));
    graph.bytes(5, onnxTensor('dense2Bias', [DEFAULT_N], dense2Bias));
    graph.bytes(5, onnxTensor('alpha', [1], new Float32Array([alpha])));
    graph.bytes(5, onnxTensor('ln2Scale', [DEFAULT_N], ln2Scale));
    graph.bytes(5, onnxTensor('ln2Bias', [DEFAULT_N], ln2Bias));
    graph.bytes(11, onnxValueInfo('input', 1, [DEFAULT_TOKENS, DEFAULT_N]));
    graph.bytes(12, onnxValueInfo('output', 1, [DEFAULT_TOKENS, DEFAULT_N]));
  });
  writer.message(8, (opset) => opset.int64(2, 17));
  return writer.finish();
}



const FFN_DENSE1_WGSL = `${WGSL_HEADER}
var<workgroup> dense1InputTile: array<f32, 128>;
var<workgroup> dense1WeightTile: array<f32, 128>;

@compute @workgroup_size(8, 8)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let col = wid.x * 8u + lid.x;
  let token = wid.y * 8u + lid.y;
  let local_index = lid.y * 8u + lid.x;
  var sum = pick_lane(biasF16[col >> 1u], col);
  for (var tile = 0u; tile < 256u; tile = tile + 16u) {
    for (var i = local_index; i < 128u; i = i + 64u) {
      let tile_row = i / 16u;
      let tile_k = i % 16u;
      let input_token = wid.y * 8u + tile_row;
      let weight_col = wid.x * 8u + (i % 8u);
      let weight_k = tile + (i / 8u);
      dense1InputTile[i] = inputVec[input_token * 256u + tile + tile_k];
      dense1WeightTile[i] = pick_lane(weightsF16[(weight_col * 256u + weight_k) >> 1u], weight_col * 256u + weight_k);
    }
    workgroupBarrier();
    for (var k = 0u; k < 16u; k = k + 1u) {
      sum = sum + dense1InputTile[lid.y * 16u + k] * dense1WeightTile[k * 8u + lid.x];
    }
    workgroupBarrier();
  }
  let value = max(sum, 0.0);
  outputVec[token * 1024u + col] = value * value;
}
`;

const FFN_DENSE2_WGSL = `${WGSL_HEADER}
@group(0) @binding(4) var<storage, read> residualVec: array<f32>;
@group(0) @binding(5) var<storage, read> alphaF16: array<u32>;
var<workgroup> dense2InputTile: array<f32, 128>;
var<workgroup> dense2WeightTile: array<f32, 128>;

@compute @workgroup_size(8, 8)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let col = wid.x * 8u + lid.x;
  let token = wid.y * 8u + lid.y;
  let local_index = lid.y * 8u + lid.x;
  var sum = pick_lane(biasF16[col >> 1u], col);
  for (var tile = 0u; tile < 1024u; tile = tile + 16u) {
    for (var i = local_index; i < 128u; i = i + 64u) {
      let tile_row = i / 16u;
      let tile_k = i % 16u;
      let input_token = wid.y * 8u + tile_row;
      let weight_col = wid.x * 8u + (i % 8u);
      let weight_k = tile + (i / 8u);
      dense2InputTile[i] = inputVec[input_token * 1024u + tile + tile_k];
      dense2WeightTile[i] = pick_lane(weightsF16[(weight_col * 1024u + weight_k) >> 1u], weight_col * 1024u + weight_k);
    }
    workgroupBarrier();
    for (var k = 0u; k < 16u; k = k + 1u) {
      sum = sum + dense2InputTile[lid.y * 16u + k] * dense2WeightTile[k * 8u + lid.x];
    }
    workgroupBarrier();
  }
  outputVec[token * 256u + col] = sum * pick_lane(alphaF16[0], 0u) + residualVec[token * 256u + col];
}
`;

const FFN_LN2_WGSL = ATTENTION_OUTPUT_NORM_WGSL;

function createEncoder0FfnPipelines(device: DeviceLike, buffers: {
  input: BufferLike;
  dense1Weight: BufferLike;
  dense1Bias: BufferLike;
  hidden: BufferLike;
  dense2Weight: BufferLike;
  dense2Bias: BufferLike;
  alpha: BufferLike;
  skip: BufferLike;
  ln2Scale: BufferLike;
  ln2Bias: BufferLike;
  output: BufferLike;
  podArgs?: BufferLike;
}, ffnKernelVariant: Lc0WebFfnKernelVariant = 'hand'): { ffnKernelVariant: Lc0WebFfnKernelVariant; dense1: PipelineLike; dense1Bind: unknown; dense2: PipelineLike; dense2Bind: unknown; ln2: PipelineLike; ln2Bind: unknown } {
  const useTvmPackedF16 = ffnKernelVariant === 'tvm-packed-f16';
  if (useTvmPackedF16 && !buffers.podArgs) throw new Error('TVM packed-f16 FFN kernels require a POD args uniform buffer');
  const dense1Module = device.createShaderModule({
    label: useTvmPackedF16 ? 'lc0web encoder0 FFN dense1 TVM packed-f16 sqrrelu' : 'lc0web encoder0 FFN dense1 sqrrelu',
    code: useTvmPackedF16 ? FFN_DENSE1_TVM_PACKED_F16_WGSL : FFN_DENSE1_WGSL,
  });
  const dense1 = device.createComputePipeline({ layout: 'auto', compute: { module: dense1Module, entryPoint: useTvmPackedF16 ? 'matmul_kernel' : 'main' } }) as PipelineLike;
  const dense1Bind = device.createBindGroup({ layout: dense1.getBindGroupLayout(0), entries: useTvmPackedF16 ? [
    { binding: 0, resource: { buffer: buffers.hidden } },
    { binding: 1, resource: { buffer: buffers.dense1Weight } },
    { binding: 2, resource: { buffer: buffers.input } },
    { binding: 3, resource: { buffer: buffers.podArgs! } },
    { binding: 4, resource: { buffer: buffers.dense1Bias } },
  ] : [
    { binding: 0, resource: { buffer: buffers.input } },
    { binding: 1, resource: { buffer: buffers.dense1Weight } },
    { binding: 2, resource: { buffer: buffers.dense1Bias } },
    { binding: 3, resource: { buffer: buffers.hidden } },
  ] });
  const dense2Module = device.createShaderModule({
    label: useTvmPackedF16 ? 'lc0web encoder0 FFN dense2 TVM packed-f16 residual' : 'lc0web encoder0 FFN dense2 residual',
    code: useTvmPackedF16 ? FFN_DENSE2_TVM_PACKED_F16_WGSL : FFN_DENSE2_WGSL,
  });
  const dense2 = device.createComputePipeline({ layout: 'auto', compute: { module: dense2Module, entryPoint: useTvmPackedF16 ? 'matmul_kernel' : 'main' } }) as PipelineLike;
  const dense2Bind = device.createBindGroup({ layout: dense2.getBindGroupLayout(0), entries: useTvmPackedF16 ? [
    { binding: 0, resource: { buffer: buffers.skip } },
    { binding: 1, resource: { buffer: buffers.dense2Weight } },
    { binding: 2, resource: { buffer: buffers.hidden } },
    { binding: 3, resource: { buffer: buffers.podArgs! } },
    { binding: 4, resource: { buffer: buffers.dense2Bias } },
    { binding: 5, resource: { buffer: buffers.input } },
    { binding: 6, resource: { buffer: buffers.alpha } },
  ] : [
    { binding: 0, resource: { buffer: buffers.hidden } },
    { binding: 1, resource: { buffer: buffers.dense2Weight } },
    { binding: 2, resource: { buffer: buffers.dense2Bias } },
    { binding: 3, resource: { buffer: buffers.skip } },
    { binding: 4, resource: { buffer: buffers.input } },
    { binding: 5, resource: { buffer: buffers.alpha } },
  ] });
  const ln2Module = device.createShaderModule({ label: 'lc0web encoder0 FFN ln2', code: FFN_LN2_WGSL });
  const ln2 = device.createComputePipeline({ layout: 'auto', compute: { module: ln2Module, entryPoint: 'main' } }) as PipelineLike;
  const ln2Bind = device.createBindGroup({ layout: ln2.getBindGroupLayout(0), entries: [
    { binding: 0, resource: { buffer: buffers.skip } },
    { binding: 1, resource: { buffer: buffers.ln2Scale } },
    { binding: 2, resource: { buffer: buffers.ln2Bias } },
    { binding: 3, resource: { buffer: buffers.output } },
  ] });
  return { ffnKernelVariant, dense1, dense1Bind, dense2, dense2Bind, ln2, ln2Bind };
}

function encodeEncoder0FfnPass(pass: ComputePassLike, pipelines: ReturnType<typeof createEncoder0FfnPipelines>): void {
  pass.setPipeline(pipelines.dense1);
  pass.setBindGroup(0, pipelines.dense1Bind);
  if (pipelines.ffnKernelVariant === 'tvm-packed-f16') pass.dispatchWorkgroups(2, 32, 1);
  else pass.dispatchWorkgroups(Math.ceil(DEFAULT_FFN_HIDDEN / 8), Math.ceil(DEFAULT_TOKENS / 8));
  pass.setPipeline(pipelines.dense2);
  pass.setBindGroup(0, pipelines.dense2Bind);
  if (pipelines.ffnKernelVariant === 'tvm-packed-f16') pass.dispatchWorkgroups(2, 8, 1);
  else pass.dispatchWorkgroups(Math.ceil(DEFAULT_N / 8), Math.ceil(DEFAULT_TOKENS / 8));
  pass.setPipeline(pipelines.ln2);
  pass.setBindGroup(0, pipelines.ln2Bind);
  pass.dispatchWorkgroups(DEFAULT_TOKENS);
}

function encodeEncoder0FfnDispatches(device: DeviceLike, pipelines: ReturnType<typeof createEncoder0FfnPipelines>, iterations: number): unknown {
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  for (let i = 0; i < iterations; i++) encodeEncoder0FfnPass(pass, pipelines);
  pass.end();
  return encoder.finish();
}

function encodeAttentionScoresPass(pass: ComputePassLike, attentionPipelines: ReturnType<typeof createAttentionOutputPipelines>): void {
  pass.setPipeline(attentionPipelines.score);
  pass.setBindGroup(0, attentionPipelines.scoreBind);
  pass.dispatchWorkgroups(Math.ceil(DEFAULT_TOKENS / 8), Math.ceil(DEFAULT_TOKENS / 8), DEFAULT_HEADS);
}

function encodeAttentionSoftmaxPass(pass: ComputePassLike, attentionPipelines: ReturnType<typeof createAttentionOutputPipelines>): void {
  pass.setPipeline(attentionPipelines.softmax);
  pass.setBindGroup(0, attentionPipelines.softmaxBind);
  pass.dispatchWorkgroups(DEFAULT_HEADS * DEFAULT_TOKENS);
}

function encodeAttentionValuePass(pass: ComputePassLike, attentionPipelines: ReturnType<typeof createAttentionOutputPipelines>): void {
  pass.setPipeline(attentionPipelines.value);
  pass.setBindGroup(0, attentionPipelines.valueBind);
  pass.dispatchWorkgroups(Math.ceil(DEFAULT_HEAD_DIM / 8), Math.ceil(DEFAULT_TOKENS / 8), DEFAULT_HEADS);
}

function encodeAttentionOutputProjectionPass(pass: ComputePassLike, attentionPipelines: ReturnType<typeof createAttentionOutputPipelines>): void {
  pass.setPipeline(attentionPipelines.outProj);
  pass.setBindGroup(0, attentionPipelines.outProjBind);
  if (attentionPipelines.outProjKernelVariant === 'tvm-packed-f16') pass.dispatchWorkgroups(2, 8, 1);
  else pass.dispatchWorkgroups(Math.ceil(DEFAULT_N / 8), Math.ceil(DEFAULT_TOKENS / 8));
}

function encodeAttentionNormPass(pass: ComputePassLike, attentionPipelines: ReturnType<typeof createAttentionOutputPipelines>): void {
  pass.setPipeline(attentionPipelines.norm);
  pass.setBindGroup(0, attentionPipelines.normBind);
  pass.dispatchWorkgroups(DEFAULT_TOKENS);
}

function encodeFfnDense1Pass(pass: ComputePassLike, ffnPipelines: ReturnType<typeof createEncoder0FfnPipelines>): void {
  pass.setPipeline(ffnPipelines.dense1);
  pass.setBindGroup(0, ffnPipelines.dense1Bind);
  if (ffnPipelines.ffnKernelVariant === 'tvm-packed-f16') pass.dispatchWorkgroups(2, 32, 1);
  else pass.dispatchWorkgroups(Math.ceil(DEFAULT_FFN_HIDDEN / 8), Math.ceil(DEFAULT_TOKENS / 8));
}

function encodeFfnDense2ResidualPass(pass: ComputePassLike, ffnPipelines: ReturnType<typeof createEncoder0FfnPipelines>): void {
  pass.setPipeline(ffnPipelines.dense2);
  pass.setBindGroup(0, ffnPipelines.dense2Bind);
  if (ffnPipelines.ffnKernelVariant === 'tvm-packed-f16') pass.dispatchWorkgroups(2, 8, 1);
  else pass.dispatchWorkgroups(Math.ceil(DEFAULT_N / 8), Math.ceil(DEFAULT_TOKENS / 8));
}

function encodeFfnLn2Pass(pass: ComputePassLike, ffnPipelines: ReturnType<typeof createEncoder0FfnPipelines>): void {
  pass.setPipeline(ffnPipelines.ln2);
  pass.setBindGroup(0, ffnPipelines.ln2Bind);
  pass.dispatchWorkgroups(DEFAULT_TOKENS);
}

function encodeLc0WebEncoderBlockPass(pass: ComputePassLike, attentionPipelines: ReturnType<typeof createAttentionOutputPipelines>, ffnPipelines: ReturnType<typeof createEncoder0FfnPipelines>): void {
  encodeAttentionQkvPass(pass, attentionPipelines);
  encodeAttentionScoresPass(pass, attentionPipelines);
  encodeAttentionSoftmaxPass(pass, attentionPipelines);
  encodeAttentionValuePass(pass, attentionPipelines);
  encodeAttentionOutputProjectionPass(pass, attentionPipelines);
  encodeAttentionNormPass(pass, attentionPipelines);
  encodeEncoder0FfnPass(pass, ffnPipelines);
}

function encodeLc0WebEncoderBlockDispatches(device: DeviceLike, attentionPipelines: ReturnType<typeof createAttentionOutputPipelines>, ffnPipelines: ReturnType<typeof createEncoder0FfnPipelines>, iterations: number): unknown {
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  for (let i = 0; i < iterations; i++) encodeLc0WebEncoderBlockPass(pass, attentionPipelines, ffnPipelines);
  pass.end();
  return encoder.finish();
}

export async function runLc0WebEncoder0FfnBenchmark(options: Lc0WebEncoder0FfnBenchmarkOptions): Promise<Lc0WebEncoder0FfnBenchmarkResult> {
  const totalStarted = nowMs();
  const warmup = clampInteger(options.warmup, 2, 0, 1000);
  const iterations = clampInteger(options.iterations, 10, 1, 10_000);
  const ffnKernelVariant = options.ffnKernelVariant ?? 'hand';
  const { device, adapterInfo } = await requestDevice();
  const tensorNames = lc0WebEncoderBlockTensorNames(options.encoderPrefix);
  const pack = await loadLc0WebModelPack(options.packUrl, {
    verifyShards: options.verifyShards ?? true,
    tensorNames: encoderBlockTensorNameList(tensorNames),
  });
  const tensors = loadEncoder0FfnInputs(pack, tensorNames);
  const reference = buildEncoder0FfnReference(tensors);
  const outputElements = DEFAULT_TOKENS * DEFAULT_N;
  const globals = gpuGlobals();
  const usage = globals.GPUBufferUsage!;
  const buffers: BufferLike[] = [];
  try {
    const setupStarted = nowMs();
    const input = createStorageBuffer(device, reference.input, usage.STORAGE | usage.COPY_DST);
    const dense1Weight = createTransposedF16StorageBuffer(device, tensors.ffnDense1Weight.bytes, DEFAULT_N, DEFAULT_FFN_HIDDEN, usage.STORAGE | usage.COPY_DST);
    const dense1Bias = createStorageBuffer(device, tensors.ffnDense1Bias.bytes, usage.STORAGE | usage.COPY_DST);
    const dense2Weight = createTransposedF16StorageBuffer(device, tensors.ffnDense2Weight.bytes, DEFAULT_FFN_HIDDEN, DEFAULT_N, usage.STORAGE | usage.COPY_DST);
    const dense2Bias = createStorageBuffer(device, tensors.ffnDense2Bias.bytes, usage.STORAGE | usage.COPY_DST);
    const alpha = createStorageBuffer(device, paddedF16ScalarBytes(tensors.ffnAlpha.bytes), usage.STORAGE | usage.COPY_DST);
    const ln2Scale = createStorageBuffer(device, tensors.ln2Scale.bytes, usage.STORAGE | usage.COPY_DST);
    const ln2Bias = createStorageBuffer(device, tensors.ln2Bias.bytes, usage.STORAGE | usage.COPY_DST);
    const hidden = device.createBuffer({ size: DEFAULT_TOKENS * DEFAULT_FFN_HIDDEN * 4, usage: usage.STORAGE });
    const skip = device.createBuffer({ size: outputElements * 4, usage: usage.STORAGE });
    const output = device.createBuffer({ size: outputElements * 4, usage: usage.STORAGE | usage.COPY_SRC });
    const readback = device.createBuffer({ size: outputElements * 4, usage: usage.MAP_READ | usage.COPY_DST });
    const podArgs = ffnKernelVariant === 'tvm-packed-f16' ? createU32UniformBuffer(device, [1], usage.UNIFORM | usage.COPY_DST) : undefined;
    buffers.push(input, dense1Weight, dense1Bias, dense2Weight, dense2Bias, alpha, ln2Scale, ln2Bias, hidden, skip, output, readback);
    if (podArgs) buffers.push(podArgs);
    const pipelines = createEncoder0FfnPipelines(device, { input, dense1Weight, dense1Bias, hidden, dense2Weight, dense2Bias, alpha, skip, ln2Scale, ln2Bias, output, podArgs }, ffnKernelVariant);
    const uploadSetupMs = nowMs() - setupStarted;
    if (warmup > 0) {
      device.queue.submit([encodeEncoder0FfnDispatches(device, pipelines, warmup)]);
      await device.queue.onSubmittedWorkDone?.();
    }
    const dispatchStarted = nowMs();
    device.queue.submit([encodeEncoder0FfnDispatches(device, pipelines, iterations)]);
    const dispatchLoopMs = nowMs() - dispatchStarted;
    const readbackStarted = nowMs();
    const gpuOutput = await readF32OutputOnce(device, output, readback, outputElements);
    const readbackSyncedMs = nowMs() - readbackStarted;
    const { maxAbsError, rmsError } = computeErrorStats(gpuOutput, reference.output, outputElements);
    assertErrorInTolerance(maxAbsError);
    return {
      status: 'FFN_BENCH_DONE',
      packUrl: pack.manifestUrl,
      modelName: pack.manifest.model.name,
      adapterInfo,
      tokens: DEFAULT_TOKENS,
      channels: DEFAULT_N,
      hidden: DEFAULT_FFN_HIDDEN,
      epsilon: DEFAULT_LN_EPSILON,
      alpha: reference.alpha,
      ffnKernelVariant,
      warmup,
      iterations,
      packLoadMs: pack.elapsedMs,
      uploadSetupMs,
      dispatchLoopMs,
      dispatchLoopAvgMs: dispatchLoopMs / iterations,
      readbackSyncedMs,
      endToEndMs: nowMs() - totalStarted,
      maxAbsError,
      rmsError,
      outputSample: Array.from(gpuOutput.slice(0, 8)),
    };
  } finally {
    for (const buffer of buffers) buffer.destroy?.();
  }
}

export async function runLc0WebEncoder0FfnOrtBenchmark(options: Lc0WebEncoder0FfnBenchmarkOptions): Promise<Lc0WebEncoder0FfnOrtBenchmarkResult> {
  const warmup = clampInteger(options.warmup, 5, 0, 100);
  const iterations = clampInteger(options.iterations, 25, 1, 1000);
  const tensorNames = lc0WebEncoderBlockTensorNames(options.encoderPrefix);
  const pack = await loadLc0WebModelPack(options.packUrl, {
    verifyShards: options.verifyShards ?? true,
    tensorNames: encoderBlockTensorNameList(tensorNames),
  });
  const tensors = loadEncoder0FfnInputs(pack, tensorNames);
  const reference = buildEncoder0FfnReference(tensors);
  const outputElements = DEFAULT_TOKENS * DEFAULT_N;
  const modelBuildStarted = nowMs();
  const tinyOnnx = createTinyEncoder0FfnOnnxForTest(
    f16BytesToF32Array(tensors.ffnDense1Weight.bytes, DEFAULT_N * DEFAULT_FFN_HIDDEN),
    f16BytesToF32Array(tensors.ffnDense1Bias.bytes, DEFAULT_FFN_HIDDEN),
    f16BytesToF32Array(tensors.ffnDense2Weight.bytes, DEFAULT_FFN_HIDDEN * DEFAULT_N),
    f16BytesToF32Array(tensors.ffnDense2Bias.bytes, DEFAULT_N),
    reference.alpha,
    f16BytesToF32Array(tensors.ln2Scale.bytes, DEFAULT_N),
    f16BytesToF32Array(tensors.ln2Bias.bytes, DEFAULT_N),
  );
  const modelBuildMs = nowMs() - modelBuildStarted;
  const sessionStarted = nowMs();
  const session = await ort.createOrtSession(tinyOnnx);
  const sessionCreateMs = nowMs() - sessionStarted;
  const feeds = { input: new ort.Tensor('float32', reference.input, [DEFAULT_TOKENS, DEFAULT_N]) };
  let output: Float32Array<ArrayBufferLike> = new Float32Array(outputElements);
  for (let i = 0; i < warmup; i++) {
    const outputs = await session.run(feeds);
    output = outputs.output.data as Float32Array<ArrayBufferLike>;
  }
  const times: number[] = [];
  for (let i = 0; i < iterations; i++) {
    const started = nowMs();
    const outputs = await session.run(feeds);
    times.push(nowMs() - started);
    output = outputs.output.data as Float32Array<ArrayBufferLike>;
  }
  const { maxAbsError, rmsError } = computeErrorStats(output, reference.output, outputElements);
  assertErrorInTolerance(maxAbsError);
  const avgMs = times.reduce((sum, value) => sum + value, 0) / times.length;
  return {
    status: 'FFN_ORT_BENCH_DONE',
    packUrl: pack.manifestUrl,
    modelName: pack.manifest.model.name,
    tokens: DEFAULT_TOKENS,
    channels: DEFAULT_N,
    hidden: DEFAULT_FFN_HIDDEN,
    epsilon: DEFAULT_LN_EPSILON,
    alpha: reference.alpha,
    warmup,
    iterations,
    packLoadMs: pack.elapsedMs,
    modelBuildMs,
    sessionCreateMs,
    avgMs,
    minMs: Math.min(...times),
    maxMs: Math.max(...times),
    firstMs: times[0],
    timesMs: times,
    runsPerSecond: 1000 / avgMs,
    maxAbsError,
    rmsError,
    outputSample: Array.from(output.slice(0, 8)),
  };
}

export function createTinyEncoder0BlockOnnxForTest(
  outWeight: Float32Array<ArrayBufferLike>,
  outBias: Float32Array<ArrayBufferLike>,
  attentionAlpha: number,
  ln1Scale: Float32Array<ArrayBufferLike>,
  ln1Bias: Float32Array<ArrayBufferLike>,
  dense1Weight: Float32Array<ArrayBufferLike>,
  dense1Bias: Float32Array<ArrayBufferLike>,
  dense2Weight: Float32Array<ArrayBufferLike>,
  dense2Bias: Float32Array<ArrayBufferLike>,
  ffnAlpha: number,
  ln2Scale: Float32Array<ArrayBufferLike>,
  ln2Bias: Float32Array<ArrayBufferLike>,
): Uint8Array {
  const writer = new ProtoWriter();
  writer.int64(1, 8);
  writer.string(2, 'lc0web');
  writer.message(7, (graph) => {
    graph.bytes(1, onnxNode('MatMul', ['attention', 'outWeight'], ['projected'], 'encoder0_attention_output_matmul'));
    graph.bytes(1, onnxNode('Add', ['projected', 'outBias'], ['biased'], 'encoder0_attention_output_bias'));
    graph.bytes(1, onnxNode('Mul', ['biased', 'attentionAlpha'], ['attentionScaled'], 'encoder0_attention_output_alpha'));
    graph.bytes(1, onnxNode('Add', ['attentionScaled', 'residual'], ['attentionSkip'], 'encoder0_attention_output_residual'));
    graph.bytes(1, onnxNode('LayerNormalization', ['attentionSkip', 'ln1Scale', 'ln1Bias'], ['ln1'], 'encoder0_ln1', [onnxFloatAttribute('epsilon', DEFAULT_LN_EPSILON)]));
    graph.bytes(1, onnxNode('MatMul', ['ln1', 'dense1Weight'], ['dense1'], 'encoder0_ffn_dense1_matmul'));
    graph.bytes(1, onnxNode('Add', ['dense1', 'dense1Bias'], ['dense1Biased'], 'encoder0_ffn_dense1_bias'));
    graph.bytes(1, onnxNode('Relu', ['dense1Biased'], ['dense1Relu'], 'encoder0_ffn_relu'));
    graph.bytes(1, onnxNode('Mul', ['dense1Relu', 'dense1Relu'], ['hidden'], 'encoder0_ffn_sqrrelu'));
    graph.bytes(1, onnxNode('MatMul', ['hidden', 'dense2Weight'], ['dense2'], 'encoder0_ffn_dense2_matmul'));
    graph.bytes(1, onnxNode('Add', ['dense2', 'dense2Bias'], ['dense2Biased'], 'encoder0_ffn_dense2_bias'));
    graph.bytes(1, onnxNode('Mul', ['dense2Biased', 'ffnAlpha'], ['ffnScaled'], 'encoder0_ffn_alpha'));
    graph.bytes(1, onnxNode('Add', ['ffnScaled', 'ln1'], ['ffnSkip'], 'encoder0_ffn_residual'));
    graph.bytes(1, onnxNode('LayerNormalization', ['ffnSkip', 'ln2Scale', 'ln2Bias'], ['output'], 'encoder0_ln2', [onnxFloatAttribute('epsilon', DEFAULT_LN_EPSILON)]));
    graph.string(2, 'lc0web_encoder0_attention_output_plus_ffn');
    graph.bytes(5, onnxTensor('outWeight', [DEFAULT_N, DEFAULT_N], outWeight));
    graph.bytes(5, onnxTensor('outBias', [DEFAULT_N], outBias));
    graph.bytes(5, onnxTensor('attentionAlpha', [1], new Float32Array([attentionAlpha])));
    graph.bytes(5, onnxTensor('ln1Scale', [DEFAULT_N], ln1Scale));
    graph.bytes(5, onnxTensor('ln1Bias', [DEFAULT_N], ln1Bias));
    graph.bytes(5, onnxTensor('dense1Weight', [DEFAULT_N, DEFAULT_FFN_HIDDEN], dense1Weight));
    graph.bytes(5, onnxTensor('dense1Bias', [DEFAULT_FFN_HIDDEN], dense1Bias));
    graph.bytes(5, onnxTensor('dense2Weight', [DEFAULT_FFN_HIDDEN, DEFAULT_N], dense2Weight));
    graph.bytes(5, onnxTensor('dense2Bias', [DEFAULT_N], dense2Bias));
    graph.bytes(5, onnxTensor('ffnAlpha', [1], new Float32Array([ffnAlpha])));
    graph.bytes(5, onnxTensor('ln2Scale', [DEFAULT_N], ln2Scale));
    graph.bytes(5, onnxTensor('ln2Bias', [DEFAULT_N], ln2Bias));
    graph.bytes(11, onnxValueInfo('attention', 1, [DEFAULT_TOKENS, DEFAULT_N]));
    graph.bytes(11, onnxValueInfo('residual', 1, [DEFAULT_TOKENS, DEFAULT_N]));
    graph.bytes(12, onnxValueInfo('output', 1, [DEFAULT_TOKENS, DEFAULT_N]));
  });
  writer.message(8, (opset) => opset.int64(2, 17));
  return writer.finish();
}

export interface Lc0WebEncoder0BlockBenchmarkOptions {
  packUrl: string;
  iterations?: number;
  warmup?: number;
  verifyShards?: boolean;
  encoderPrefix?: string;
}

export type Lc0WebEncoder0BlockStageName =
  | 'qkvProjection'
  | 'attentionScores'
  | 'softmax'
  | 'attentionValue'
  | 'outputProjectionLn1'
  | 'ffnDense1'
  | 'ffnDense2Residual'
  | 'ln2';

export interface Lc0WebEncoder0BlockStageTiming {
  stage: Lc0WebEncoder0BlockStageName;
  label: string;
  iterations: number;
  totalMs: number;
  avgMs: number;
}

export interface Lc0WebEncoder0BlockBenchmarkResult {
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
  smolgen: { enabled: boolean; epsilon: number };
  warmup: number;
  iterations: number;
  packLoadMs: number;
  uploadSetupMs: number;
  dispatchLoopMs: number;
  dispatchLoopAvgMs: number;
  readbackSyncedMs: number;
  gpuTimestampSupported: boolean;
  gpuTimestampMs?: number;
  stageTimings: Lc0WebEncoder0BlockStageTiming[];
  stageTimingTotalMs: number;
  endToEndMs: number;
  maxAbsError: number;
  rmsError: number;
  outputSample: number[];
}

export interface Lc0WebEncoder0BlockOrtBenchmarkResult {
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
  smolgen: { enabled: boolean; epsilon: number };
  warmup: number;
  iterations: number;
  packLoadMs: number;
  modelBuildMs: number;
  sessionCreateMs: number;
  avgMs: number;
  minMs: number;
  maxMs: number;
  firstMs: number;
  timesMs: number[];
  runsPerSecond: number;
  ortDiagnostics?: Record<string, unknown>;
  maxAbsError: number;
  rmsError: number;
  outputSample: number[];
}

export interface Lc0WebEncoderStackBenchmarkOptions {
  packUrl: string;
  layers?: number;
  warmup?: number;
  verifyShards?: boolean;
  compareOrt?: boolean;
  compareHeads?: boolean;
  /** Optional real model activation entering /encoder0; defaults to synthetic benchmark input. */
  input?: Float32Array<ArrayBufferLike>;
  /** Include full policy/head arrays in policyValueHeads for evaluator/drift callers. */
  includeHeadOutputs?: boolean;
}

export interface Lc0WebEncoderStackBlockResult {
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
}

export interface Lc0WebEncoderStackHeadsResult {
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
  /** Optional full 64x64 policy logits. Omitted by benchmark callers to keep reports compact. */
  policy?: number[];
  /** Optional full final 1858 LC0 policy logits. */
  mappedPolicy?: number[];
}

export interface Lc0WebEncoderStackBenchmarkResult {
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
  policyValueHeads?: Lc0WebEncoderStackHeadsResult;
  blocks: Lc0WebEncoderStackBlockResult[];
}

function buildEncoder0BlockReference(tensors: Encoder0FfnTensors, input: Float32Array<ArrayBufferLike> = makeInputTokenMatrix(DEFAULT_TOKENS, DEFAULT_K)): { input: Float32Array<ArrayBufferLike>; output: Float32Array<ArrayBufferLike>; attentionAlpha: number; ffnAlpha: number; smolgenBias: Float32Array<ArrayBufferLike> } {
  const attention = buildAttentionOutputReference(tensors, input);
  const ffn = cpuEncoder0FfnFromLn1(attention.output, tensors);
  return { input: attention.input, output: ffn.output, attentionAlpha: attention.alpha, ffnAlpha: ffn.alpha, smolgenBias: attention.smolgenBias };
}

const DEFAULT_POLICY_HEAD_TENSORS = {
  dense1Weight: '/policy/dense1/matmul/w',
  dense1Bias: '/policy/dense1/add/w',
  qWeight: '/policy/Q/matmul/w',
  qBias: '/policy/Q/add/w',
  kWeight: '/policy/K/matmul/w',
  kBias: '/policy/K/add/w',
  scale: '/policy/scale/w',
  promotionWeight: '/policy/promotion/matmul/w',
  mappingTable: '/const/mapping_table',
} as const;
const DEFAULT_VALUE_HEAD_TENSORS = {
  embedWeight: '/value/embed/matmul/w',
  embedBias: '/value/embed/add/w',
  dense1Weight: '/value/dense1/matmul/w',
  dense1Bias: '/value/dense1/add/w',
  dense2Weight: '/value/dense2/matmul/w',
  dense2Bias: '/value/dense2/add/w',
} as const;
const DEFAULT_POLICY_OUTPUTS = DEFAULT_TOKENS * DEFAULT_TOKENS;
const DEFAULT_POLICY_MAPPED_OUTPUTS = 1858;
const DEFAULT_POLICY_FLAT = 4288;
const DEFAULT_VALUE_EMBED = 32;
const DEFAULT_VALUE_HIDDEN = 128;

type Lc0WebPolicyValueHeadTensors = {
  policyDense1Weight: Lc0WebTensorView;
  policyDense1Bias: Lc0WebTensorView;
  policyQWeight: Lc0WebTensorView;
  policyQBias: Lc0WebTensorView;
  policyKWeight: Lc0WebTensorView;
  policyKBias: Lc0WebTensorView;
  policyScale: Lc0WebTensorView;
  policyPromotionWeight: Lc0WebTensorView;
  policyMappingTable: Lc0WebTensorView;
  valueEmbedWeight: Lc0WebTensorView;
  valueEmbedBias: Lc0WebTensorView;
  valueDense1Weight: Lc0WebTensorView;
  valueDense1Bias: Lc0WebTensorView;
  valueDense2Weight: Lc0WebTensorView;
  valueDense2Bias: Lc0WebTensorView;
};

function policyValueHeadTensorNameList(): string[] {
  return [...Object.values(DEFAULT_POLICY_HEAD_TENSORS), ...Object.values(DEFAULT_VALUE_HEAD_TENSORS)];
}

function loadPolicyValueHeadTensors(pack: Awaited<ReturnType<typeof loadLc0WebModelPack>>): Lc0WebPolicyValueHeadTensors {
  const get = (name: string): Lc0WebTensorView => {
    const tensor = pack.tensors.get(name);
    if (!tensor) throw new Error(`lc0web policy/value head tensor was not loaded: ${name}`);
    return tensor;
  };
  const tensors: Lc0WebPolicyValueHeadTensors = {
    policyDense1Weight: get(DEFAULT_POLICY_HEAD_TENSORS.dense1Weight),
    policyDense1Bias: get(DEFAULT_POLICY_HEAD_TENSORS.dense1Bias),
    policyQWeight: get(DEFAULT_POLICY_HEAD_TENSORS.qWeight),
    policyQBias: get(DEFAULT_POLICY_HEAD_TENSORS.qBias),
    policyKWeight: get(DEFAULT_POLICY_HEAD_TENSORS.kWeight),
    policyKBias: get(DEFAULT_POLICY_HEAD_TENSORS.kBias),
    policyScale: get(DEFAULT_POLICY_HEAD_TENSORS.scale),
    policyPromotionWeight: get(DEFAULT_POLICY_HEAD_TENSORS.promotionWeight),
    policyMappingTable: get(DEFAULT_POLICY_HEAD_TENSORS.mappingTable),
    valueEmbedWeight: get(DEFAULT_VALUE_HEAD_TENSORS.embedWeight),
    valueEmbedBias: get(DEFAULT_VALUE_HEAD_TENSORS.embedBias),
    valueDense1Weight: get(DEFAULT_VALUE_HEAD_TENSORS.dense1Weight),
    valueDense1Bias: get(DEFAULT_VALUE_HEAD_TENSORS.dense1Bias),
    valueDense2Weight: get(DEFAULT_VALUE_HEAD_TENSORS.dense2Weight),
    valueDense2Bias: get(DEFAULT_VALUE_HEAD_TENSORS.dense2Bias),
  };
  assertTensorShapeAndBytes(tensors.policyDense1Weight, [DEFAULT_N, DEFAULT_N], 2, 'policyDense1Weight');
  assertTensorShapeAndBytes(tensors.policyDense1Bias, [DEFAULT_N], 2, 'policyDense1Bias');
  assertTensorShapeAndBytes(tensors.policyQWeight, [DEFAULT_N, DEFAULT_N], 2, 'policyQWeight');
  assertTensorShapeAndBytes(tensors.policyQBias, [DEFAULT_N], 2, 'policyQBias');
  assertTensorShapeAndBytes(tensors.policyKWeight, [DEFAULT_N, DEFAULT_N], 2, 'policyKWeight');
  assertTensorShapeAndBytes(tensors.policyKBias, [DEFAULT_N], 2, 'policyKBias');
  assertTensorShapeAndBytes(tensors.policyScale, [1], 2, 'policyScale');
  assertTensorShapeAndBytes(tensors.policyPromotionWeight, [DEFAULT_N, 4], 2, 'policyPromotionWeight');
  assertTensorShapeAndBytes(tensors.policyMappingTable, [DEFAULT_POLICY_MAPPED_OUTPUTS], 4, 'policyMappingTable');
  assertTensorShapeAndBytes(tensors.valueEmbedWeight, [DEFAULT_N, DEFAULT_VALUE_EMBED], 2, 'valueEmbedWeight');
  assertTensorShapeAndBytes(tensors.valueEmbedBias, [DEFAULT_VALUE_EMBED], 2, 'valueEmbedBias');
  assertTensorShapeAndBytes(tensors.valueDense1Weight, [DEFAULT_TOKENS * DEFAULT_VALUE_EMBED, DEFAULT_VALUE_HIDDEN], 2, 'valueDense1Weight');
  assertTensorShapeAndBytes(tensors.valueDense1Bias, [DEFAULT_VALUE_HIDDEN], 2, 'valueDense1Bias');
  assertTensorShapeAndBytes(tensors.valueDense2Weight, [DEFAULT_VALUE_HIDDEN, 3], 2, 'valueDense2Weight');
  assertTensorShapeAndBytes(tensors.valueDense2Bias, [3], 2, 'valueDense2Bias');
  return tensors;
}

function readI32Array(bytes: Uint8Array, elements: number): Int32Array<ArrayBufferLike> {
  const out = new Int32Array(elements);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  for (let i = 0; i < elements; i++) out[i] = view.getInt32(i * 4, true);
  return out;
}

function cpuMish(input: Float32Array<ArrayBufferLike>): Float32Array<ArrayBufferLike> {
  const output = new Float32Array(input.length);
  for (let i = 0; i < input.length; i++) {
    const x = input[i];
    output[i] = x * Math.tanh(Math.log1p(Math.exp(x)));
  }
  return output;
}

function cpuSoftmaxVector(input: Float32Array<ArrayBufferLike>): Float32Array<ArrayBufferLike> {
  const output = new Float32Array(input.length);
  let max = -Infinity;
  for (const value of input) max = Math.max(max, value);
  let denom = 0;
  for (let i = 0; i < input.length; i++) {
    const value = Math.exp(input[i] - max);
    output[i] = value;
    denom += value;
  }
  for (let i = 0; i < output.length; i++) output[i] /= denom;
  return output;
}

function cpuPolicyHead(input: Float32Array<ArrayBufferLike>, tensors: Lc0WebPolicyValueHeadTensors): { policy: Float32Array<ArrayBufferLike>; mappedPolicy: Float32Array<ArrayBufferLike> } {
  const dense = cpuMish(cpuProjectTokens(input, tensors.policyDense1Weight.bytes, tensors.policyDense1Bias.bytes, DEFAULT_TOKENS, DEFAULT_N, DEFAULT_N));
  const q = cpuProjectTokens(dense, tensors.policyQWeight.bytes, tensors.policyQBias.bytes, DEFAULT_TOKENS, DEFAULT_N, DEFAULT_N);
  const k = cpuProjectTokens(dense, tensors.policyKWeight.bytes, tensors.policyKBias.bytes, DEFAULT_TOKENS, DEFAULT_N, DEFAULT_N);
  const scaleValue = readF16At(tensors.policyScale.bytes, 0);
  const policy = new Float32Array(DEFAULT_POLICY_OUTPUTS);
  for (let row = 0; row < DEFAULT_TOKENS; row++) {
    for (let col = 0; col < DEFAULT_TOKENS; col++) {
      let sum = 0;
      for (let channel = 0; channel < DEFAULT_N; channel++) sum += q[row * DEFAULT_N + channel] * k[col * DEFAULT_N + channel];
      policy[row * DEFAULT_TOKENS + col] = sum * scaleValue;
    }
  }

  const promotionMatmul = new Float32Array(8 * 4);
  for (let token = 0; token < 8; token++) {
    const kToken = 56 + token;
    for (let col = 0; col < 4; col++) {
      let sum = 0;
      for (let channel = 0; channel < DEFAULT_N; channel++) {
        sum += k[kToken * DEFAULT_N + channel] * readF16At(tensors.policyPromotionWeight.bytes, channel * 4 + col);
      }
      promotionMatmul[token * 4 + col] = sum;
    }
  }
  const promotionBias = new Float32Array(8 * 3);
  for (let token = 0; token < 8; token++) {
    for (let under = 0; under < 3; under++) promotionBias[token * 3 + under] = promotionMatmul[token * 4 + under] + promotionMatmul[token * 4 + 3];
  }
  const promotionBaseReshaped = new Float32Array(8 * 24);
  for (let from = 0; from < 8; from++) {
    for (let to = 0; to < 8; to++) {
      const base = policy[(48 + from) * DEFAULT_TOKENS + 56 + to];
      const flat = from * 8 + to;
      for (let under = 0; under < 3; under++) {
        const concatFlat = flat * 3 + under;
        promotionBaseReshaped[concatFlat] = base;
      }
    }
  }
  const promotionRows = new Float32Array(3 * DEFAULT_TOKENS);
  for (let i = 0; i < promotionRows.length; i++) {
    promotionRows[i] = promotionBaseReshaped[i] + promotionBias[i % 24];
  }

  const policyFlat = new Float32Array(DEFAULT_POLICY_FLAT);
  policyFlat.set(policy);
  policyFlat.set(promotionRows, DEFAULT_POLICY_OUTPUTS);
  const mapping = readI32Array(tensors.policyMappingTable.bytes, DEFAULT_POLICY_MAPPED_OUTPUTS);
  const mappedPolicy = new Float32Array(DEFAULT_POLICY_MAPPED_OUTPUTS);
  for (let i = 0; i < DEFAULT_POLICY_MAPPED_OUTPUTS; i++) mappedPolicy[i] = policyFlat[mapping[i]];
  return { policy, mappedPolicy };
}

function cpuValueHead(input: Float32Array<ArrayBufferLike>, tensors: Lc0WebPolicyValueHeadTensors): Float32Array<ArrayBufferLike> {
  const embed = cpuMish(cpuProjectTokens(input, tensors.valueEmbedWeight.bytes, tensors.valueEmbedBias.bytes, DEFAULT_TOKENS, DEFAULT_N, DEFAULT_VALUE_EMBED));
  const dense1 = cpuMish(cpuMatmulAddVector(embed, tensors.valueDense1Weight.bytes, tensors.valueDense1Bias.bytes, DEFAULT_TOKENS * DEFAULT_VALUE_EMBED, DEFAULT_VALUE_HIDDEN));
  const logits = cpuMatmulAddVector(dense1, tensors.valueDense2Weight.bytes, tensors.valueDense2Bias.bytes, DEFAULT_VALUE_HIDDEN, 3);
  return cpuSoftmaxVector(logits);
}

function buildPolicyValueHeadReference(input: Float32Array<ArrayBufferLike>, tensors: Lc0WebPolicyValueHeadTensors): { policy: Float32Array<ArrayBufferLike>; mappedPolicy: Float32Array<ArrayBufferLike>; wdl: Float32Array<ArrayBufferLike> } {
  const policy = cpuPolicyHead(input, tensors);
  return { policy: policy.policy, mappedPolicy: policy.mappedPolicy, wdl: cpuValueHead(input, tensors) };
}

export interface Lc0WebMappedPolicyProbeResult {
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
}

export interface Lc0WebWgslHeadsProbeResult {
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
  /** Optional full 64x64 policy logits for WGSL-vs-ORT real-encoder comparisons. */
  policyLogits?: number[];
  /** Optional full final 1858 LC0 policy logits for WGSL-vs-ORT real-encoder comparisons. */
  mappedPolicy?: number[];
  nonzero: { policyDense: boolean; policyLogits: boolean; mappedPolicy: boolean; valueEmbed: boolean; wgslWdl: boolean };
  nonuniform: { policyDense: boolean; policyLogits: boolean; mappedPolicy: boolean; valueEmbed: boolean; wgslWdl: boolean };
  ortHeads: {
    mode: 'ort-policy-value';
    runMs: number;
    mappedPolicySample: number[];
    wdl: number[];
    wdlMaxAbsError: number;
    /** Optional full final 1858 LC0 policy logits from ORT. */
    mappedPolicy?: number[];
  };
}

const WGSL_HEADS_DENSE_PROBE = `
struct MatrixShape { outChannels: u32, activation: u32, _pad0: u32, _pad1: u32 };
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> shape: MatrixShape;

fn mish(x: f32) -> f32 {
  return x * tanh(log(1.0 + exp(x)));
}

@compute @workgroup_size(16, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let channel = gid.x;
  let token = gid.y;
  if (token >= 64u || channel >= shape.outChannels) { return; }
  var sum = bias[channel];
  for (var k = 0u; k < 256u; k = k + 1u) {
    sum = sum + input[token * 256u + k] * weight[k * shape.outChannels + channel];
  }
  output[token * shape.outChannels + channel] = select(sum, mish(sum), shape.activation == 1u);
}
`;

const WGSL_HEADS_POLICY_LOGITS_PROBE = `
@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k: array<f32>;
@group(0) @binding(2) var<storage, read> scale: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let col = gid.x;
  let row = gid.y;
  if (row >= 64u || col >= 64u) { return; }
  var sum = 0.0;
  for (var channel = 0u; channel < 256u; channel = channel + 1u) {
    sum = sum + q[row * 256u + channel] * k[col * 256u + channel];
  }
  output[row * 64u + col] = sum * scale[0];
}
`;

const WGSL_MAPPED_POLICY_PROBE = `
@group(0) @binding(0) var<storage, read> policy: array<f32>;
@group(0) @binding(1) var<storage, read> k: array<f32>;
@group(0) @binding(2) var<storage, read> promotionWeight: array<f32>;
@group(0) @binding(3) var<storage, read> mapping: array<i32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

fn promotion_value(flat_index: u32) -> f32 {
  let promotion_index = flat_index - 4096u;
  let move_flat = promotion_index / 3u;
  let under = promotion_index % 3u;
  let from_file = move_flat / 8u;
  let to_file = move_flat % 8u;
  let base = policy[(48u + from_file) * 64u + 56u + to_file];
  let bias_token = (promotion_index % 24u) / 3u;
  var bias = 0.0;
  for (var channel = 0u; channel < 256u; channel = channel + 1u) {
    let offset = channel * 4u;
    bias = bias + k[(56u + bias_token) * 256u + channel] * (promotionWeight[offset + under] + promotionWeight[offset + 3u]);
  }
  return base + bias;
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let out_index = gid.x;
  if (out_index >= 1858u) { return; }
  let flat_index = u32(mapping[out_index]);
  if (flat_index < 4096u) {
    output[out_index] = policy[flat_index];
  } else {
    output[out_index] = promotion_value(flat_index);
  }
}
`;

const WGSL_HEADS_VECTOR_DENSE_PROBE = `
struct VectorShape { inChannels: u32, outChannels: u32, activation: u32, _pad0: u32 };
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> shape: VectorShape;

fn mish(x: f32) -> f32 {
  return x * tanh(log(1.0 + exp(x)));
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let channel = gid.x;
  if (channel >= shape.outChannels) { return; }
  var sum = bias[channel];
  for (var k = 0u; k < shape.inChannels; k = k + 1u) {
    sum = sum + input[k] * weight[k * shape.outChannels + channel];
  }
  output[channel] = select(sum, mish(sum), shape.activation == 1u);
}
`;

const WGSL_GPU_LEGAL_MAX_MOVES = 256;
const WGSL_GPU_LEGAL_OUTPUT_FLOATS = WGSL_GPU_LEGAL_MAX_MOVES * 3;
const WGSL_GPU_LEGAL_READBACK_FLOATS = WGSL_GPU_LEGAL_OUTPUT_FLOATS + 3;
const WGSL_GPU_LEGAL_READBACK_BYTES = WGSL_GPU_LEGAL_READBACK_FLOATS * 4;

const WGSL_HEADS_SOFTMAX3_PROBE = `
@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(1, 1, 1)
fn main() {
  let m = max(logits[0], max(logits[1], logits[2]));
  let e0 = exp(logits[0] - m);
  let e1 = exp(logits[1] - m);
  let e2 = exp(logits[2] - m);
  let denom = e0 + e1 + e2;
  output[0] = e0 / denom;
  output[1] = e1 / denom;
  output[2] = e2 / denom;
}
`;

const WGSL_LEGAL_PRIORS_PROBE = `
@group(0) @binding(0) var<storage, read> mappedPolicy: array<f32>;
@group(0) @binding(1) var<storage, read> legalIndices: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> args: vec4<f32>;

@compute @workgroup_size(1, 1, 1)
fn main() {
  let count = min(u32(args.x), ${WGSL_GPU_LEGAL_MAX_MOVES}u);
  let temperature = max(args.y, 0.000001);
  if (count == 0u) { return; }
  var maxLogit = -3.402823e38;
  for (var i = 0u; i < count; i = i + 1u) {
    let policyIndex = legalIndices[i];
    let logit = mappedPolicy[policyIndex] / temperature;
    output[i * 3u] = f32(i);
    output[i * 3u + 1u] = logit;
    output[i * 3u + 2u] = 0.0;
    maxLogit = max(maxLogit, logit);
  }
  var sum = 0.0;
  for (var i = 0u; i < count; i = i + 1u) {
    let prior = exp(output[i * 3u + 1u] - maxLogit);
    output[i * 3u + 2u] = prior;
    sum = sum + prior;
  }
  let invSum = 1.0 / sum;
  for (var i = 0u; i < count; i = i + 1u) {
    output[i * 3u + 2u] = output[i * 3u + 2u] * invSum;
  }
  for (var i = 0u; i < count; i = i + 1u) {
    for (var j = i + 1u; j < count; j = j + 1u) {
      if (output[j * 3u + 2u] > output[i * 3u + 2u]) {
        let slot = output[i * 3u];
        let logit = output[i * 3u + 1u];
        let prior = output[i * 3u + 2u];
        output[i * 3u] = output[j * 3u];
        output[i * 3u + 1u] = output[j * 3u + 1u];
        output[i * 3u + 2u] = output[j * 3u + 2u];
        output[j * 3u] = slot;
        output[j * 3u + 1u] = logit;
        output[j * 3u + 2u] = prior;
      }
    }
  }
}
`;

function arrayHasNonzero(values: Float32Array<ArrayBufferLike>): boolean {
  for (const value of values) if (Math.abs(value) > 1e-8) return true;
  return false;
}

function arrayHasVariation(values: Float32Array<ArrayBufferLike>): boolean {
  if (!values.length) return false;
  let min = values[0];
  let max = values[0];
  for (const value of values) {
    min = Math.min(min, value);
    max = Math.max(max, value);
  }
  return Math.abs(max - min) > 1e-8;
}

function arrayHasNonzeroAndVariation(values: Float32Array<ArrayBufferLike>): boolean {
  if (!values.length) return false;
  let hasNonzero = false;
  let min = values[0];
  let max = values[0];
  for (const value of values) {
    if (Math.abs(value) > 1e-8) hasNonzero = true;
    min = Math.min(min, value);
    max = Math.max(max, value);
  }
  return hasNonzero && Math.abs(max - min) > 1e-8;
}

function createU32UniformBuffer(device: DeviceLike, values: number[], usage: number): BufferLike {
  const data = new Uint32Array(4);
  data.set(values.slice(0, 4));
  return createStorageBuffer(device, data, usage);
}

function createWgslHeadsDenseBindGroup(device: DeviceLike, pipeline: PipelineLike, inputBuffer: BufferLike, weightBuffer: BufferLike, biasBuffer: BufferLike, outputBuffer: BufferLike, shapeBuffer: BufferLike): unknown {
  return device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: inputBuffer } },
      { binding: 1, resource: { buffer: weightBuffer } },
      { binding: 2, resource: { buffer: biasBuffer } },
      { binding: 3, resource: { buffer: outputBuffer } },
      { binding: 4, resource: { buffer: shapeBuffer } },
    ],
  });
}

function createWgslHeadsPolicyLogitsBindGroup(device: DeviceLike, pipeline: PipelineLike, qBuffer: BufferLike, kBuffer: BufferLike, scaleBuffer: BufferLike, outputBuffer: BufferLike): unknown {
  return device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: qBuffer } },
      { binding: 1, resource: { buffer: kBuffer } },
      { binding: 2, resource: { buffer: scaleBuffer } },
      { binding: 3, resource: { buffer: outputBuffer } },
    ],
  });
}

function createMappedPolicyBindGroup(device: DeviceLike, pipeline: PipelineLike, policyBuffer: BufferLike, kBuffer: BufferLike, promotionWeightBuffer: BufferLike, mappingBuffer: BufferLike, outputBuffer: BufferLike): unknown {
  return device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: policyBuffer } },
      { binding: 1, resource: { buffer: kBuffer } },
      { binding: 2, resource: { buffer: promotionWeightBuffer } },
      { binding: 3, resource: { buffer: mappingBuffer } },
      { binding: 4, resource: { buffer: outputBuffer } },
    ],
  });
}

function createWgslHeadsSoftmaxBindGroup(device: DeviceLike, pipeline: PipelineLike, logitsBuffer: BufferLike, outputBuffer: BufferLike): unknown {
  return device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: logitsBuffer } },
      { binding: 1, resource: { buffer: outputBuffer } },
    ],
  });
}

function createWgslLegalPriorsBindGroup(device: DeviceLike, pipeline: PipelineLike, mappedPolicyBuffer: BufferLike, legalIndicesBuffer: BufferLike, outputBuffer: BufferLike, argsBuffer: BufferLike): unknown {
  return device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: mappedPolicyBuffer } },
      { binding: 1, resource: { buffer: legalIndicesBuffer } },
      { binding: 2, resource: { buffer: outputBuffer } },
      { binding: 3, resource: { buffer: argsBuffer } },
    ],
  });
}

function makeSyntheticMappedPolicyInputs(): { policy: Float32Array<ArrayBufferLike>; k: Float32Array<ArrayBufferLike>; promotionWeight: Float32Array<ArrayBufferLike>; mapping: Int32Array<ArrayBufferLike> } {
  const policy = new Float32Array(DEFAULT_POLICY_OUTPUTS);
  for (let i = 0; i < policy.length; i++) policy[i] = Math.sin(i * 0.013) * 0.5 + (i % 97) * 0.001;
  const k = new Float32Array(DEFAULT_TOKENS * DEFAULT_N);
  for (let token = 0; token < DEFAULT_TOKENS; token++) {
    for (let channel = 0; channel < DEFAULT_N; channel++) {
      k[token * DEFAULT_N + channel] = Math.cos(token * 0.17 + channel * 0.019) * 0.02 + ((token + channel) % 11) * 0.0007;
    }
  }
  const promotionWeight = new Float32Array(DEFAULT_N * 4);
  for (let channel = 0; channel < DEFAULT_N; channel++) {
    for (let col = 0; col < 4; col++) {
      promotionWeight[channel * 4 + col] = Math.sin(channel * 0.023 + col * 0.37) * 0.015 + (col + 1) * 0.0009;
    }
  }
  const mapping = new Int32Array(DEFAULT_POLICY_MAPPED_OUTPUTS);
  for (let i = 0; i < 1792; i++) mapping[i] = (i * 37 + 1) % DEFAULT_POLICY_OUTPUTS;
  for (let i = 1792; i < DEFAULT_POLICY_MAPPED_OUTPUTS; i++) mapping[i] = DEFAULT_POLICY_OUTPUTS + ((i - 1792) * 17) % (3 * DEFAULT_TOKENS);
  return { policy, k, promotionWeight, mapping };
}

function cpuMappedPolicyFromSyntheticInputs(policy: Float32Array<ArrayBufferLike>, k: Float32Array<ArrayBufferLike>, promotionWeight: Float32Array<ArrayBufferLike>, mapping: Int32Array<ArrayBufferLike>): Float32Array<ArrayBufferLike> {
  const output = new Float32Array(DEFAULT_POLICY_MAPPED_OUTPUTS);
  for (let out = 0; out < output.length; out++) {
    const flat = mapping[out];
    if (flat < DEFAULT_POLICY_OUTPUTS) {
      output[out] = policy[flat];
      continue;
    }
    const promotionIndex = flat - DEFAULT_POLICY_OUTPUTS;
    const moveFlat = Math.floor(promotionIndex / 3);
    const under = promotionIndex % 3;
    const from = Math.floor(moveFlat / 8);
    const to = moveFlat % 8;
    const base = policy[(48 + from) * DEFAULT_TOKENS + 56 + to];
    const biasToken = Math.floor((promotionIndex % 24) / 3);
    let bias = 0;
    for (let channel = 0; channel < DEFAULT_N; channel++) {
      const offset = channel * 4;
      bias += k[(56 + biasToken) * DEFAULT_N + channel] * (promotionWeight[offset + under] + promotionWeight[offset + 3]);
    }
    output[out] = base + bias;
  }
  return output;
}

export async function runLc0WebMappedPolicyProbe(): Promise<Lc0WebMappedPolicyProbeResult> {
  const { policy, k, promotionWeight, mapping } = makeSyntheticMappedPolicyInputs();
  const expected = cpuMappedPolicyFromSyntheticInputs(policy, k, promotionWeight, mapping);
  const { device, adapterInfo } = await requestDevice();
  const usage = gpuGlobals().GPUBufferUsage!;
  const buffers: BufferLike[] = [];
  try {
    const compileStarted = nowMs();
    const module = device.createShaderModule({ label: 'lc0web WGSL mapped policy synthetic probe', code: WGSL_MAPPED_POLICY_PROBE });
    const compilationInfo = await (module as { getCompilationInfo?: () => Promise<{ messages: Array<{ type: string; message: string; lineNum?: number; linePos?: number }> }> }).getCompilationInfo?.();
    const compilationErrors = compilationInfo?.messages.filter((message) => message.type === 'error') ?? [];
    if (compilationErrors.length) throw new Error(`mapped-policy probe shader compilation failed: ${compilationErrors.map((message) => `${message.lineNum}:${message.linePos} ${message.message}`).join('; ')}`);
    const scopedCompileDevice = device as DeviceLike & { pushErrorScope?: (filter: string) => void; popErrorScope?: () => Promise<unknown> };
    scopedCompileDevice.pushErrorScope?.('validation');
    const pipeline = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'main' } }) as PipelineLike;
    const pipelineError = await scopedCompileDevice.popErrorScope?.();
    if (pipelineError) throw new Error(`mapped-policy probe pipeline validation error: ${String((pipelineError as { message?: unknown }).message ?? pipelineError)}`);
    const pipelineCompileMs = nowMs() - compileStarted;
    const policyBuffer = createStorageBuffer(device, policy, usage.STORAGE | usage.COPY_DST);
    const kBuffer = createStorageBuffer(device, k, usage.STORAGE | usage.COPY_DST);
    const promotionWeightBuffer = createStorageBuffer(device, promotionWeight, usage.STORAGE | usage.COPY_DST);
    const mappingBuffer = createStorageBuffer(device, mapping, usage.STORAGE | usage.COPY_DST);
    const outputBuffer = device.createBuffer({ size: DEFAULT_POLICY_MAPPED_OUTPUTS * 4, usage: usage.STORAGE | usage.COPY_SRC }) as BufferLike;
    const readbackBuffer = device.createBuffer({ size: DEFAULT_POLICY_MAPPED_OUTPUTS * 4, usage: usage.MAP_READ | usage.COPY_DST }) as BufferLike;
    buffers.push(policyBuffer, kBuffer, promotionWeightBuffer, mappingBuffer, outputBuffer, readbackBuffer);
    const bindGroup = createMappedPolicyBindGroup(device, pipeline, policyBuffer, kBuffer, promotionWeightBuffer, mappingBuffer, outputBuffer);
    const scopedDevice = device as DeviceLike & { pushErrorScope?: (filter: string) => void; popErrorScope?: () => Promise<unknown> };
    scopedDevice.pushErrorScope?.('validation');
    const dispatchStarted = nowMs();
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(DEFAULT_POLICY_MAPPED_OUTPUTS / 64));
    pass.end();
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone?.();
    const validationError = await scopedDevice.popErrorScope?.();
    if (validationError) throw new Error(`mapped-policy probe WebGPU validation error: ${String((validationError as { message?: unknown }).message ?? validationError)}`);
    const dispatchSyncedMs = nowMs() - dispatchStarted;
    const readbackStarted = nowMs();
    const output = await readF32OutputOnce(device, outputBuffer, readbackBuffer, DEFAULT_POLICY_MAPPED_OUTPUTS);
    const readbackSyncedMs = nowMs() - readbackStarted;
    const error = computeErrorStats(output, expected, DEFAULT_POLICY_MAPPED_OUTPUTS);
    let normalMaxAbsError = 0;
    let promotionMaxAbsError = 0;
    let normalOutputs = 0;
    let promotionOutputs = 0;
    const normalSample: number[] = [];
    const promotionSample: number[] = [];
    for (let i = 0; i < DEFAULT_POLICY_MAPPED_OUTPUTS; i++) {
      const absError = Math.abs(output[i] - expected[i]);
      if (mapping[i] < DEFAULT_POLICY_OUTPUTS) {
        normalOutputs++;
        normalMaxAbsError = Math.max(normalMaxAbsError, absError);
        if (normalSample.length < 8) normalSample.push(output[i]);
      } else {
        promotionOutputs++;
        promotionMaxAbsError = Math.max(promotionMaxAbsError, absError);
        if (promotionSample.length < 8) promotionSample.push(output[i]);
      }
    }
    if (error.maxAbsError > 1e-5) throw new Error(`mapped-policy probe maxAbsError=${error.maxAbsError}, normalMaxAbsError=${normalMaxAbsError}, promotionMaxAbsError=${promotionMaxAbsError}, normalSample=${normalSample.slice(0, 3).join('/')}, promotionSample=${promotionSample.slice(0, 3).join('/')}, tolerance=0.00001`);
    if (!arrayHasNonzero(output) || !arrayHasVariation(output)) throw new Error('mapped-policy probe produced zero or uniform output');
    return {
      status: 'MAPPED_POLICY_PROBE_DONE',
      adapterInfo,
      outputs: DEFAULT_POLICY_MAPPED_OUTPUTS,
      normalOutputs,
      promotionOutputs,
      pipelineCompileMs,
      dispatchSyncedMs,
      readbackSyncedMs,
      maxAbsError: error.maxAbsError,
      rmsError: error.rmsError,
      normalMaxAbsError,
      promotionMaxAbsError,
      normalSample,
      promotionSample,
      outputSample: Array.from(output.slice(0, 8)),
      nonzero: arrayHasNonzero(output),
      nonuniform: arrayHasVariation(output),
    };
  } finally {
    for (const buffer of buffers) buffer.destroy?.();
  }
}

export async function runLc0WebWgslHeadsProbe(options: { packUrl: string; verifyShards?: boolean; input?: Float32Array<ArrayBufferLike>; includeOutputs?: boolean }): Promise<Lc0WebWgslHeadsProbeResult> {
  const packLoadStarted = nowMs();
  const pack = await loadLc0WebModelPack(options.packUrl, {
    verifyShards: options.verifyShards,
    tensorNames: policyValueHeadTensorNameList(),
  });
  const packLoadMs = nowMs() - packLoadStarted;
  const tensors = loadPolicyValueHeadTensors(pack);
  const input = options.input ?? makeInputTokenMatrix(DEFAULT_TOKENS, DEFAULT_N);
  if (input.length !== DEFAULT_TOKENS * DEFAULT_N) throw new Error(`WGSL heads input length ${input.length} != ${DEFAULT_TOKENS * DEFAULT_N}`);
  const policyRef = cpuPolicyHead(input, tensors);
  const policyDenseRef = cpuMish(cpuProjectTokens(input, tensors.policyDense1Weight.bytes, tensors.policyDense1Bias.bytes, DEFAULT_TOKENS, DEFAULT_N, DEFAULT_N));
  const valueEmbedRef = cpuMish(cpuProjectTokens(input, tensors.valueEmbedWeight.bytes, tensors.valueEmbedBias.bytes, DEFAULT_TOKENS, DEFAULT_N, DEFAULT_VALUE_EMBED));
  const valueWdlRef = cpuValueHead(input, tensors);
  const { device, adapterInfo } = await requestDevice();
  const usage = gpuGlobals().GPUBufferUsage!;
  const buffers: BufferLike[] = [];
  try {
    const compileStarted = nowMs();
    const module = device.createShaderModule({ label: 'lc0web WGSL policy/value head dense probe', code: WGSL_HEADS_DENSE_PROBE });
    const pipeline = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'main' } }) as PipelineLike;
    const policyLogitsModule = device.createShaderModule({ label: 'lc0web WGSL policy head logits probe', code: WGSL_HEADS_POLICY_LOGITS_PROBE });
    const policyLogitsPipeline = device.createComputePipeline({ layout: 'auto', compute: { module: policyLogitsModule, entryPoint: 'main' } }) as PipelineLike;
    const mappedPolicyModule = device.createShaderModule({ label: 'lc0web WGSL mapped policy probe', code: WGSL_MAPPED_POLICY_PROBE });
    const mappedPolicyPipeline = device.createComputePipeline({ layout: 'auto', compute: { module: mappedPolicyModule, entryPoint: 'main' } }) as PipelineLike;
    const vectorModule = device.createShaderModule({ label: 'lc0web WGSL value head vector dense probe', code: WGSL_HEADS_VECTOR_DENSE_PROBE });
    const vectorPipeline = device.createComputePipeline({ layout: 'auto', compute: { module: vectorModule, entryPoint: 'main' } }) as PipelineLike;
    const softmaxModule = device.createShaderModule({ label: 'lc0web WGSL value head softmax probe', code: WGSL_HEADS_SOFTMAX3_PROBE });
    const softmaxPipeline = device.createComputePipeline({ layout: 'auto', compute: { module: softmaxModule, entryPoint: 'main' } }) as PipelineLike;
    const pipelineCompileMs = nowMs() - compileStarted;

    const inputBuffer = createStorageBuffer(device, input, usage.STORAGE | usage.COPY_DST);
    const policyWeightBuffer = createStorageBuffer(device, f16BytesToF32Array(tensors.policyDense1Weight.bytes, DEFAULT_N * DEFAULT_N), usage.STORAGE | usage.COPY_DST);
    const policyBiasBuffer = createStorageBuffer(device, f16BytesToF32Array(tensors.policyDense1Bias.bytes, DEFAULT_N), usage.STORAGE | usage.COPY_DST);
    const policyQWeightBuffer = createStorageBuffer(device, f16BytesToF32Array(tensors.policyQWeight.bytes, DEFAULT_N * DEFAULT_N), usage.STORAGE | usage.COPY_DST);
    const policyQBiasBuffer = createStorageBuffer(device, f16BytesToF32Array(tensors.policyQBias.bytes, DEFAULT_N), usage.STORAGE | usage.COPY_DST);
    const policyKWeightBuffer = createStorageBuffer(device, f16BytesToF32Array(tensors.policyKWeight.bytes, DEFAULT_N * DEFAULT_N), usage.STORAGE | usage.COPY_DST);
    const policyKBiasBuffer = createStorageBuffer(device, f16BytesToF32Array(tensors.policyKBias.bytes, DEFAULT_N), usage.STORAGE | usage.COPY_DST);
    const policyScaleBuffer = createStorageBuffer(device, f16BytesToF32Array(tensors.policyScale.bytes, 1), usage.STORAGE | usage.COPY_DST);
    const policyPromotionWeightBuffer = createStorageBuffer(device, f16BytesToF32Array(tensors.policyPromotionWeight.bytes, DEFAULT_N * 4), usage.STORAGE | usage.COPY_DST);
    const policyMappingBuffer = createStorageBuffer(device, readI32Array(tensors.policyMappingTable.bytes, DEFAULT_POLICY_MAPPED_OUTPUTS), usage.STORAGE | usage.COPY_DST);
    const valueWeightBuffer = createStorageBuffer(device, f16BytesToF32Array(tensors.valueEmbedWeight.bytes, DEFAULT_N * DEFAULT_VALUE_EMBED), usage.STORAGE | usage.COPY_DST);
    const valueBiasBuffer = createStorageBuffer(device, f16BytesToF32Array(tensors.valueEmbedBias.bytes, DEFAULT_VALUE_EMBED), usage.STORAGE | usage.COPY_DST);
    const valueDense1WeightBuffer = createStorageBuffer(device, f16BytesToF32Array(tensors.valueDense1Weight.bytes, DEFAULT_TOKENS * DEFAULT_VALUE_EMBED * DEFAULT_VALUE_HIDDEN), usage.STORAGE | usage.COPY_DST);
    const valueDense1BiasBuffer = createStorageBuffer(device, f16BytesToF32Array(tensors.valueDense1Bias.bytes, DEFAULT_VALUE_HIDDEN), usage.STORAGE | usage.COPY_DST);
    const valueDense2WeightBuffer = createStorageBuffer(device, f16BytesToF32Array(tensors.valueDense2Weight.bytes, DEFAULT_VALUE_HIDDEN * 3), usage.STORAGE | usage.COPY_DST);
    const valueDense2BiasBuffer = createStorageBuffer(device, f16BytesToF32Array(tensors.valueDense2Bias.bytes, 3), usage.STORAGE | usage.COPY_DST);
    const policyOutputBuffer = device.createBuffer({ size: DEFAULT_TOKENS * DEFAULT_N * 4, usage: usage.STORAGE | usage.COPY_SRC }) as BufferLike;
    const policyQBuffer = device.createBuffer({ size: DEFAULT_TOKENS * DEFAULT_N * 4, usage: usage.STORAGE | usage.COPY_SRC }) as BufferLike;
    const policyKBuffer = device.createBuffer({ size: DEFAULT_TOKENS * DEFAULT_N * 4, usage: usage.STORAGE | usage.COPY_SRC }) as BufferLike;
    const policyLogitsBuffer = device.createBuffer({ size: DEFAULT_POLICY_OUTPUTS * 4, usage: usage.STORAGE | usage.COPY_SRC }) as BufferLike;
    const mappedPolicyBuffer = device.createBuffer({ size: DEFAULT_POLICY_MAPPED_OUTPUTS * 4, usage: usage.STORAGE | usage.COPY_SRC }) as BufferLike;
    const valueOutputBuffer = device.createBuffer({ size: DEFAULT_TOKENS * DEFAULT_VALUE_EMBED * 4, usage: usage.STORAGE | usage.COPY_SRC }) as BufferLike;
    const valueHiddenBuffer = device.createBuffer({ size: DEFAULT_VALUE_HIDDEN * 4, usage: usage.STORAGE | usage.COPY_SRC }) as BufferLike;
    const valueLogitsBuffer = device.createBuffer({ size: 3 * 4, usage: usage.STORAGE | usage.COPY_SRC }) as BufferLike;
    const valueWdlBuffer = device.createBuffer({ size: 3 * 4, usage: usage.STORAGE | usage.COPY_SRC }) as BufferLike;
    const policyReadbackBuffer = device.createBuffer({ size: DEFAULT_TOKENS * DEFAULT_N * 4, usage: usage.MAP_READ | usage.COPY_DST }) as BufferLike;
    const policyLogitsReadbackBuffer = device.createBuffer({ size: DEFAULT_POLICY_OUTPUTS * 4, usage: usage.MAP_READ | usage.COPY_DST }) as BufferLike;
    const mappedPolicyReadbackBuffer = device.createBuffer({ size: DEFAULT_POLICY_MAPPED_OUTPUTS * 4, usage: usage.MAP_READ | usage.COPY_DST }) as BufferLike;
    const valueReadbackBuffer = device.createBuffer({ size: DEFAULT_TOKENS * DEFAULT_VALUE_EMBED * 4, usage: usage.MAP_READ | usage.COPY_DST }) as BufferLike;
    const valueWdlReadbackBuffer = device.createBuffer({ size: 3 * 4, usage: usage.MAP_READ | usage.COPY_DST }) as BufferLike;
    const policyShapeBuffer = createU32UniformBuffer(device, [DEFAULT_N, 1], usage.UNIFORM | usage.COPY_DST);
    const policyLinearShapeBuffer = createU32UniformBuffer(device, [DEFAULT_N, 0], usage.UNIFORM | usage.COPY_DST);
    const valueShapeBuffer = createU32UniformBuffer(device, [DEFAULT_VALUE_EMBED, 1], usage.UNIFORM | usage.COPY_DST);
    const valueDense1ShapeBuffer = createU32UniformBuffer(device, [DEFAULT_TOKENS * DEFAULT_VALUE_EMBED, DEFAULT_VALUE_HIDDEN, 1], usage.UNIFORM | usage.COPY_DST);
    const valueDense2ShapeBuffer = createU32UniformBuffer(device, [DEFAULT_VALUE_HIDDEN, 3, 0], usage.UNIFORM | usage.COPY_DST);
    buffers.push(inputBuffer, policyWeightBuffer, policyBiasBuffer, policyQWeightBuffer, policyQBiasBuffer, policyKWeightBuffer, policyKBiasBuffer, policyScaleBuffer, policyPromotionWeightBuffer, policyMappingBuffer, valueWeightBuffer, valueBiasBuffer, valueDense1WeightBuffer, valueDense1BiasBuffer, valueDense2WeightBuffer, valueDense2BiasBuffer, policyOutputBuffer, policyQBuffer, policyKBuffer, policyLogitsBuffer, mappedPolicyBuffer, valueOutputBuffer, valueHiddenBuffer, valueLogitsBuffer, valueWdlBuffer, policyReadbackBuffer, policyLogitsReadbackBuffer, mappedPolicyReadbackBuffer, valueReadbackBuffer, valueWdlReadbackBuffer, policyShapeBuffer, policyLinearShapeBuffer, valueShapeBuffer, valueDense1ShapeBuffer, valueDense2ShapeBuffer);

    const policyBindGroup = createWgslHeadsDenseBindGroup(device, pipeline, inputBuffer, policyWeightBuffer, policyBiasBuffer, policyOutputBuffer, policyShapeBuffer);
    const policyQBindGroup = createWgslHeadsDenseBindGroup(device, pipeline, policyOutputBuffer, policyQWeightBuffer, policyQBiasBuffer, policyQBuffer, policyLinearShapeBuffer);
    const policyKBindGroup = createWgslHeadsDenseBindGroup(device, pipeline, policyOutputBuffer, policyKWeightBuffer, policyKBiasBuffer, policyKBuffer, policyLinearShapeBuffer);
    const policyLogitsBindGroup = createWgslHeadsPolicyLogitsBindGroup(device, policyLogitsPipeline, policyQBuffer, policyKBuffer, policyScaleBuffer, policyLogitsBuffer);
    const mappedPolicyBindGroup = createMappedPolicyBindGroup(device, mappedPolicyPipeline, policyLogitsBuffer, policyKBuffer, policyPromotionWeightBuffer, policyMappingBuffer, mappedPolicyBuffer);
    const valueBindGroup = createWgslHeadsDenseBindGroup(device, pipeline, inputBuffer, valueWeightBuffer, valueBiasBuffer, valueOutputBuffer, valueShapeBuffer);
    const valueDense1BindGroup = createWgslHeadsDenseBindGroup(device, vectorPipeline, valueOutputBuffer, valueDense1WeightBuffer, valueDense1BiasBuffer, valueHiddenBuffer, valueDense1ShapeBuffer);
    const valueDense2BindGroup = createWgslHeadsDenseBindGroup(device, vectorPipeline, valueHiddenBuffer, valueDense2WeightBuffer, valueDense2BiasBuffer, valueLogitsBuffer, valueDense2ShapeBuffer);
    const valueSoftmaxBindGroup = createWgslHeadsSoftmaxBindGroup(device, softmaxPipeline, valueLogitsBuffer, valueWdlBuffer);
    const dispatchStarted = nowMs();
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, policyBindGroup);
    pass.dispatchWorkgroups(Math.ceil(DEFAULT_N / 16), DEFAULT_TOKENS);
    pass.setBindGroup(0, valueBindGroup);
    pass.dispatchWorkgroups(Math.ceil(DEFAULT_VALUE_EMBED / 16), DEFAULT_TOKENS);
    pass.end();
    const policyProjectionPass = encoder.beginComputePass();
    policyProjectionPass.setPipeline(pipeline);
    policyProjectionPass.setBindGroup(0, policyQBindGroup);
    policyProjectionPass.dispatchWorkgroups(Math.ceil(DEFAULT_N / 16), DEFAULT_TOKENS);
    policyProjectionPass.setBindGroup(0, policyKBindGroup);
    policyProjectionPass.dispatchWorkgroups(Math.ceil(DEFAULT_N / 16), DEFAULT_TOKENS);
    policyProjectionPass.end();
    const policyLogitsPass = encoder.beginComputePass();
    policyLogitsPass.setPipeline(policyLogitsPipeline);
    policyLogitsPass.setBindGroup(0, policyLogitsBindGroup);
    policyLogitsPass.dispatchWorkgroups(Math.ceil(DEFAULT_TOKENS / 8), Math.ceil(DEFAULT_TOKENS / 8));
    policyLogitsPass.end();
    const mappedPolicyPass = encoder.beginComputePass();
    mappedPolicyPass.setPipeline(mappedPolicyPipeline);
    mappedPolicyPass.setBindGroup(0, mappedPolicyBindGroup);
    mappedPolicyPass.dispatchWorkgroups(Math.ceil(DEFAULT_POLICY_MAPPED_OUTPUTS / 64));
    mappedPolicyPass.end();
    const valueDense1Pass = encoder.beginComputePass();
    valueDense1Pass.setPipeline(vectorPipeline);
    valueDense1Pass.setBindGroup(0, valueDense1BindGroup);
    valueDense1Pass.dispatchWorkgroups(Math.ceil(DEFAULT_VALUE_HIDDEN / 64));
    valueDense1Pass.end();
    const valueDense2Pass = encoder.beginComputePass();
    valueDense2Pass.setPipeline(vectorPipeline);
    valueDense2Pass.setBindGroup(0, valueDense2BindGroup);
    valueDense2Pass.dispatchWorkgroups(1);
    valueDense2Pass.end();
    const valueSoftmaxPass = encoder.beginComputePass();
    valueSoftmaxPass.setPipeline(softmaxPipeline);
    valueSoftmaxPass.setBindGroup(0, valueSoftmaxBindGroup);
    valueSoftmaxPass.dispatchWorkgroups(1);
    valueSoftmaxPass.end();
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone?.();
    const dispatchSyncedMs = nowMs() - dispatchStarted;

    const readbackStarted = nowMs();
    const policyDense = await readF32OutputOnce(device, policyOutputBuffer, policyReadbackBuffer, DEFAULT_TOKENS * DEFAULT_N);
    const policyLogits = await readF32OutputOnce(device, policyLogitsBuffer, policyLogitsReadbackBuffer, DEFAULT_POLICY_OUTPUTS);
    const mappedPolicy = await readF32OutputOnce(device, mappedPolicyBuffer, mappedPolicyReadbackBuffer, DEFAULT_POLICY_MAPPED_OUTPUTS);
    const valueEmbed = await readF32OutputOnce(device, valueOutputBuffer, valueReadbackBuffer, DEFAULT_TOKENS * DEFAULT_VALUE_EMBED);
    const wgslWdl = await readF32OutputOnce(device, valueWdlBuffer, valueWdlReadbackBuffer, 3);
    const readbackSyncedMs = nowMs() - readbackStarted;
    const policyErr = computeErrorStats(policyDense, policyDenseRef, DEFAULT_TOKENS * DEFAULT_N);
    const policyLogitsErr = computeErrorStats(policyLogits, policyRef.policy, DEFAULT_POLICY_OUTPUTS);
    const mappedPolicyErr = computeErrorStats(mappedPolicy, policyRef.mappedPolicy, DEFAULT_POLICY_MAPPED_OUTPUTS);
    const valueErr = computeErrorStats(valueEmbed, valueEmbedRef, DEFAULT_TOKENS * DEFAULT_VALUE_EMBED);
    const wdlErr = computeErrorStats(wgslWdl, valueWdlRef, 3);
    assertErrorInTolerance(Math.max(policyErr.maxAbsError, policyLogitsErr.maxAbsError, mappedPolicyErr.maxAbsError, valueErr.maxAbsError, wdlErr.maxAbsError));
    if (!arrayHasNonzero(policyDense) || !arrayHasNonzero(policyLogits) || !arrayHasNonzero(mappedPolicy) || !arrayHasNonzero(valueEmbed) || !arrayHasNonzero(wgslWdl) || !arrayHasVariation(policyDense) || !arrayHasVariation(policyLogits) || !arrayHasVariation(mappedPolicy) || !arrayHasVariation(valueEmbed) || !arrayHasVariation(wgslWdl)) {
      throw new Error('WGSL heads probe produced zero or uniform intermediate output');
    }
    const ortHeads = await runPolicyValueHeadsOrt(input, tensors, { includeOutputs: options.includeOutputs });
    return {
      status: 'WGSL_HEADS_PROBE_DONE',
      packUrl: pack.manifestUrl,
      modelName: pack.manifest.model.name,
      adapterInfo,
      tokens: DEFAULT_TOKENS,
      channels: DEFAULT_N,
      valueEmbedChannels: DEFAULT_VALUE_EMBED,
      packLoadMs,
      pipelineCompileMs,
      dispatchSyncedMs,
      readbackSyncedMs,
      policyDenseMaxAbsError: policyErr.maxAbsError,
      policyDenseRmsError: policyErr.rmsError,
      policyLogitsMaxAbsError: policyLogitsErr.maxAbsError,
      policyLogitsRmsError: policyLogitsErr.rmsError,
      mappedPolicyMaxAbsError: mappedPolicyErr.maxAbsError,
      mappedPolicyRmsError: mappedPolicyErr.rmsError,
      valueEmbedMaxAbsError: valueErr.maxAbsError,
      valueEmbedRmsError: valueErr.rmsError,
      wgslWdlMaxAbsError: wdlErr.maxAbsError,
      wgslWdlRmsError: wdlErr.rmsError,
      policyDenseSample: Array.from(policyDense.slice(0, 8)),
      policyLogitsSample: Array.from(policyLogits.slice(0, 8)),
      mappedPolicySample: Array.from(mappedPolicy.slice(0, 8)),
      valueEmbedSample: Array.from(valueEmbed.slice(0, 8)),
      wgslWdl: Array.from(wgslWdl),
      ...(options.includeOutputs ? { policyLogits: Array.from(policyLogits), mappedPolicy: Array.from(mappedPolicy) } : {}),
      nonzero: { policyDense: arrayHasNonzero(policyDense), policyLogits: arrayHasNonzero(policyLogits), mappedPolicy: arrayHasNonzero(mappedPolicy), valueEmbed: arrayHasNonzero(valueEmbed), wgslWdl: arrayHasNonzero(wgslWdl) },
      nonuniform: { policyDense: arrayHasVariation(policyDense), policyLogits: arrayHasVariation(policyLogits), mappedPolicy: arrayHasVariation(mappedPolicy), valueEmbed: arrayHasVariation(valueEmbed), wgslWdl: arrayHasVariation(wgslWdl) },
      ortHeads: {
        mode: ortHeads.mode,
        runMs: ortHeads.runMs,
        mappedPolicySample: ortHeads.mappedPolicySample,
        wdl: ortHeads.wdl,
        wdlMaxAbsError: ortHeads.wdlMaxAbsError,
        ...(options.includeOutputs && ortHeads.mappedPolicy ? { mappedPolicy: ortHeads.mappedPolicy } : {}),
      },
    };
  } finally {
    for (const buffer of buffers) buffer.destroy?.();
  }
}

export function createTinyPolicyValueHeadsOnnxForTest(tensors: Lc0WebPolicyValueHeadTensors): Uint8Array {
  const writer = new ProtoWriter();
  writer.int64(1, 8);
  writer.string(2, 'lc0web');
  writer.message(7, (graph) => {
    graph.bytes(1, onnxNode('MatMul', ['input', 'policyDense1Weight'], ['policyDense1'], 'policy_dense1_matmul'));
    graph.bytes(1, onnxNode('Add', ['policyDense1', 'policyDense1Bias'], ['policyDense1Biased'], 'policy_dense1_bias'));
    graph.bytes(1, onnxNode('Softplus', ['policyDense1Biased'], ['policyDense1Softplus'], 'policy_dense1_softplus'));
    graph.bytes(1, onnxNode('Tanh', ['policyDense1Softplus'], ['policyDense1Tanh'], 'policy_dense1_tanh'));
    graph.bytes(1, onnxNode('Mul', ['policyDense1Biased', 'policyDense1Tanh'], ['policyDense1Mish'], 'policy_dense1_mish'));
    graph.bytes(1, onnxNode('MatMul', ['policyDense1Mish', 'policyQWeight'], ['policyQMatmul'], 'policy_q_matmul'));
    graph.bytes(1, onnxNode('Add', ['policyQMatmul', 'policyQBias'], ['policyQ'], 'policy_q_bias'));
    graph.bytes(1, onnxNode('MatMul', ['policyDense1Mish', 'policyKWeight'], ['policyKMatmul'], 'policy_k_matmul'));
    graph.bytes(1, onnxNode('Add', ['policyKMatmul', 'policyKBias'], ['policyK'], 'policy_k_bias'));
    graph.bytes(1, onnxNode('Transpose', ['policyK'], ['policyKt'], 'policy_k_transpose', [onnxIntsAttribute('perm', [1, 0])]));
    graph.bytes(1, onnxNode('MatMul', ['policyQ', 'policyKt'], ['policyMatmul'], 'policy_matmul'));
    graph.bytes(1, onnxNode('Mul', ['policyMatmul', 'policyScale'], ['policy'], 'policy_scale'));
    graph.bytes(1, onnxNode('Slice', ['policyK', 'policyPromStarts', 'policyPromEnds', 'policyPromAxes'], ['policyPromK'], 'policy_promotion_slice'));
    graph.bytes(1, onnxNode('MatMul', ['policyPromK', 'policyPromotionWeight'], ['policyPromMatmul'], 'policy_promotion_matmul'));
    graph.bytes(1, onnxNode('Transpose', ['policyPromMatmul'], ['policyPromT'], 'policy_promotion_transpose', [onnxIntsAttribute('perm', [1, 0])]));
    graph.bytes(1, onnxNode('Slice', ['policyPromT', 'policyPromUnderStarts', 'policyPromUnderEnds', 'policyPromUnderAxes'], ['policyPromUnder'], 'policy_promotion_under_slice'));
    graph.bytes(1, onnxNode('Slice', ['policyPromT', 'policyPromQueenStarts', 'policyPromQueenEnds', 'policyPromUnderAxes'], ['policyPromQueen'], 'policy_promotion_queen_slice'));
    graph.bytes(1, onnxNode('Add', ['policyPromUnder', 'policyPromQueen'], ['policyPromAdd'], 'policy_promotion_add'));
    graph.bytes(1, onnxNode('Transpose', ['policyPromAdd'], ['policyPromAddT'], 'policy_promotion_transpose2', [onnxIntsAttribute('perm', [1, 0])]));
    graph.bytes(1, onnxNode('Reshape', ['policyPromAddT', 'policyPromotionShape1'], ['policyPromBias'], 'policy_promotion_reshape'));
    graph.bytes(1, onnxNode('Slice', ['policy', 'policyBaseStarts', 'policyBaseEnds', 'policyBaseAxes'], ['policyPromotionBase'], 'policy_promotion_base_slice'));
    graph.bytes(1, onnxNode('Reshape', ['policyPromotionBase', 'policyPromotionShape2'], ['policyPromotionBaseFlat'], 'policy_promotion_base_reshape'));
    graph.bytes(1, onnxNode('Concat', ['policyPromotionBaseFlat', 'policyPromotionBaseFlat', 'policyPromotionBaseFlat'], ['policyPromotionBaseTripled'], 'policy_promotion_base_concat', [onnxIntAttribute('axis', 1)]));
    graph.bytes(1, onnxNode('Reshape', ['policyPromotionBaseTripled', 'policyPromotionShape3'], ['policyPromotionBaseReshaped'], 'policy_promotion_base_reshape2'));
    graph.bytes(1, onnxNode('Add', ['policyPromotionBaseReshaped', 'policyPromBias'], ['policyPromotionAdd'], 'policy_promotion_add2'));
    graph.bytes(1, onnxNode('Reshape', ['policyPromotionAdd', 'policyPromotionShape4'], ['policyPromotionRows'], 'policy_promotion_reshape4'));
    graph.bytes(1, onnxNode('Concat', ['policy', 'policyPromotionRows'], ['policyConcat'], 'policy_concat', [onnxIntAttribute('axis', 0)]));
    graph.bytes(1, onnxNode('Reshape', ['policyConcat', 'policyFlatShape'], ['policyFlat'], 'policy_reshape'));
    graph.bytes(1, onnxNode('Gather', ['policyFlat', 'policyMappingTable'], ['mappedPolicy'], 'policy_mapping_gather', [onnxIntAttribute('axis', 1)]));

    graph.bytes(1, onnxNode('MatMul', ['input', 'valueEmbedWeight'], ['valueEmbed'], 'value_embed_matmul'));
    graph.bytes(1, onnxNode('Add', ['valueEmbed', 'valueEmbedBias'], ['valueEmbedBiased'], 'value_embed_bias'));
    graph.bytes(1, onnxNode('Softplus', ['valueEmbedBiased'], ['valueEmbedSoftplus'], 'value_embed_softplus'));
    graph.bytes(1, onnxNode('Tanh', ['valueEmbedSoftplus'], ['valueEmbedTanh'], 'value_embed_tanh'));
    graph.bytes(1, onnxNode('Mul', ['valueEmbedBiased', 'valueEmbedTanh'], ['valueEmbedMish'], 'value_embed_mish'));
    graph.bytes(1, onnxNode('Reshape', ['valueEmbedMish', 'valueShape'], ['valueFlat'], 'value_reshape'));
    graph.bytes(1, onnxNode('MatMul', ['valueFlat', 'valueDense1Weight'], ['valueDense1'], 'value_dense1_matmul'));
    graph.bytes(1, onnxNode('Add', ['valueDense1', 'valueDense1Bias'], ['valueDense1Biased'], 'value_dense1_bias'));
    graph.bytes(1, onnxNode('Softplus', ['valueDense1Biased'], ['valueDense1Softplus'], 'value_dense1_softplus'));
    graph.bytes(1, onnxNode('Tanh', ['valueDense1Softplus'], ['valueDense1Tanh'], 'value_dense1_tanh'));
    graph.bytes(1, onnxNode('Mul', ['valueDense1Biased', 'valueDense1Tanh'], ['valueDense1Mish'], 'value_dense1_mish'));
    graph.bytes(1, onnxNode('MatMul', ['valueDense1Mish', 'valueDense2Weight'], ['valueDense2'], 'value_dense2_matmul'));
    graph.bytes(1, onnxNode('Add', ['valueDense2', 'valueDense2Bias'], ['valueLogits'], 'value_dense2_bias'));
    graph.bytes(1, onnxNode('Softmax', ['valueLogits'], ['wdl'], 'output_wdl', [onnxIntAttribute('axis', 1)]));

    graph.string(2, 'lc0web_policy_value_heads');
    graph.bytes(5, onnxTensor('policyDense1Weight', [DEFAULT_N, DEFAULT_N], f16BytesToF32Array(tensors.policyDense1Weight.bytes, DEFAULT_N * DEFAULT_N)));
    graph.bytes(5, onnxTensor('policyDense1Bias', [DEFAULT_N], f16BytesToF32Array(tensors.policyDense1Bias.bytes, DEFAULT_N)));
    graph.bytes(5, onnxTensor('policyQWeight', [DEFAULT_N, DEFAULT_N], f16BytesToF32Array(tensors.policyQWeight.bytes, DEFAULT_N * DEFAULT_N)));
    graph.bytes(5, onnxTensor('policyQBias', [DEFAULT_N], f16BytesToF32Array(tensors.policyQBias.bytes, DEFAULT_N)));
    graph.bytes(5, onnxTensor('policyKWeight', [DEFAULT_N, DEFAULT_N], f16BytesToF32Array(tensors.policyKWeight.bytes, DEFAULT_N * DEFAULT_N)));
    graph.bytes(5, onnxTensor('policyKBias', [DEFAULT_N], f16BytesToF32Array(tensors.policyKBias.bytes, DEFAULT_N)));
    graph.bytes(5, onnxTensor('policyScale', [1], f16BytesToF32Array(tensors.policyScale.bytes, 1)));
    graph.bytes(5, onnxTensor('policyPromotionWeight', [DEFAULT_N, 4], f16BytesToF32Array(tensors.policyPromotionWeight.bytes, DEFAULT_N * 4)));
    graph.bytes(5, onnxInt64Tensor('policyPromStarts', [1], [56]));
    graph.bytes(5, onnxInt64Tensor('policyPromEnds', [1], [64]));
    graph.bytes(5, onnxInt64Tensor('policyPromAxes', [1], [0]));
    graph.bytes(5, onnxInt64Tensor('policyPromUnderStarts', [1], [0]));
    graph.bytes(5, onnxInt64Tensor('policyPromUnderEnds', [1], [3]));
    graph.bytes(5, onnxInt64Tensor('policyPromQueenStarts', [1], [3]));
    graph.bytes(5, onnxInt64Tensor('policyPromQueenEnds', [1], [4]));
    graph.bytes(5, onnxInt64Tensor('policyPromUnderAxes', [1], [0]));
    graph.bytes(5, onnxInt64Tensor('policyBaseStarts', [2], [48, 56]));
    graph.bytes(5, onnxInt64Tensor('policyBaseEnds', [2], [56, 64]));
    graph.bytes(5, onnxInt64Tensor('policyBaseAxes', [2], [0, 1]));
    graph.bytes(5, onnxInt64Tensor('policyPromotionShape1', [2], [1, 24]));
    graph.bytes(5, onnxInt64Tensor('policyPromotionShape2', [2], [64, 1]));
    graph.bytes(5, onnxInt64Tensor('policyPromotionShape3', [2], [8, 24]));
    graph.bytes(5, onnxInt64Tensor('policyPromotionShape4', [2], [3, 64]));
    graph.bytes(5, onnxInt64Tensor('policyFlatShape', [2], [1, DEFAULT_POLICY_FLAT]));
    graph.bytes(5, onnxInt64Tensor('policyMappingTable', [DEFAULT_POLICY_MAPPED_OUTPUTS], Array.from(readI32Array(tensors.policyMappingTable.bytes, DEFAULT_POLICY_MAPPED_OUTPUTS))));
    graph.bytes(5, onnxInt64Tensor('valueShape', [2], [1, DEFAULT_TOKENS * DEFAULT_VALUE_EMBED]));
    graph.bytes(5, onnxTensor('valueEmbedWeight', [DEFAULT_N, DEFAULT_VALUE_EMBED], f16BytesToF32Array(tensors.valueEmbedWeight.bytes, DEFAULT_N * DEFAULT_VALUE_EMBED)));
    graph.bytes(5, onnxTensor('valueEmbedBias', [DEFAULT_VALUE_EMBED], f16BytesToF32Array(tensors.valueEmbedBias.bytes, DEFAULT_VALUE_EMBED)));
    graph.bytes(5, onnxTensor('valueDense1Weight', [DEFAULT_TOKENS * DEFAULT_VALUE_EMBED, DEFAULT_VALUE_HIDDEN], f16BytesToF32Array(tensors.valueDense1Weight.bytes, DEFAULT_TOKENS * DEFAULT_VALUE_EMBED * DEFAULT_VALUE_HIDDEN)));
    graph.bytes(5, onnxTensor('valueDense1Bias', [DEFAULT_VALUE_HIDDEN], f16BytesToF32Array(tensors.valueDense1Bias.bytes, DEFAULT_VALUE_HIDDEN)));
    graph.bytes(5, onnxTensor('valueDense2Weight', [DEFAULT_VALUE_HIDDEN, 3], f16BytesToF32Array(tensors.valueDense2Weight.bytes, DEFAULT_VALUE_HIDDEN * 3)));
    graph.bytes(5, onnxTensor('valueDense2Bias', [3], f16BytesToF32Array(tensors.valueDense2Bias.bytes, 3)));
    graph.bytes(11, onnxValueInfo('input', 1, [DEFAULT_TOKENS, DEFAULT_N]));
    graph.bytes(12, onnxValueInfo('policy', 1, [DEFAULT_TOKENS, DEFAULT_TOKENS]));
    graph.bytes(12, onnxValueInfo('mappedPolicy', 1, [1, DEFAULT_POLICY_MAPPED_OUTPUTS]));
    graph.bytes(12, onnxValueInfo('wdl', 1, [1, 3]));
  });
  writer.message(8, (opset) => opset.int64(2, 17));
  return writer.finish();
}

async function runPolicyValueHeadsOrt(input: Float32Array<ArrayBufferLike>, tensors: Lc0WebPolicyValueHeadTensors, options: { includeOutputs?: boolean } = {}): Promise<{
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
  policy?: number[];
  mappedPolicy?: number[];
  mode: 'ort-policy-value';
}> {
  const reference = buildPolicyValueHeadReference(input, tensors);
  const modelBuildStarted = nowMs();
  const tinyOnnx = createTinyPolicyValueHeadsOnnxForTest(tensors);
  const modelBuildMs = nowMs() - modelBuildStarted;
  const sessionStarted = nowMs();
  const session = await ort.createOrtSession(tinyOnnx);
  const sessionCreateMs = nowMs() - sessionStarted;
  const runStarted = nowMs();
  const outputs = await session.run({ input: new ort.Tensor('float32', input, [DEFAULT_TOKENS, DEFAULT_N]) });
  const runMs = nowMs() - runStarted;
  const policy = outputs.policy.data as Float32Array<ArrayBufferLike>;
  const mappedPolicy = outputs.mappedPolicy.data as Float32Array<ArrayBufferLike>;
  const wdl = outputs.wdl.data as Float32Array<ArrayBufferLike>;
  const policyError = computeErrorStats(policy, reference.policy, DEFAULT_POLICY_OUTPUTS);
  const mappedPolicyError = computeErrorStats(mappedPolicy, reference.mappedPolicy, DEFAULT_POLICY_MAPPED_OUTPUTS);
  const wdlError = computeErrorStats(wdl, reference.wdl, 3);
  // The policy logits are a diagnostic CPU-vs-ORT comparison for the ORT head
  // handoff path. They are not used as the WGSL stack correctness gate because
  // this CPU helper accumulates large policy matmuls in JS number precision;
  // the f32 ORT head output itself is the baseline consumed by the hybrid path.
  assertErrorInTolerance(wdlError.maxAbsError);
  return {
    mode: 'ort-policy-value',
    modelBuildMs,
    sessionCreateMs,
    runMs,
    policyMaxAbsError: policyError.maxAbsError,
    policyRmsError: policyError.rmsError,
    mappedPolicyMaxAbsError: mappedPolicyError.maxAbsError,
    mappedPolicyRmsError: mappedPolicyError.rmsError,
    wdlMaxAbsError: wdlError.maxAbsError,
    wdlRmsError: wdlError.rmsError,
    policySample: Array.from(policy.slice(0, 8)),
    mappedPolicySample: Array.from(mappedPolicy.slice(0, 8)),
    wdl: Array.from(wdl),
    ...(options.includeOutputs ? { policy: Array.from(policy), mappedPolicy: Array.from(mappedPolicy) } : {}),
  };
}

type CachedPolicyValueHeadSession = {
  session: ort.InferenceSession;
  modelBuildMs: number;
  sessionCreateMs: number;
};

async function createCachedPolicyValueHeadSession(tensors: Lc0WebPolicyValueHeadTensors): Promise<CachedPolicyValueHeadSession> {
  const modelBuildStarted = nowMs();
  const tinyOnnx = createTinyPolicyValueHeadsOnnxForTest(tensors);
  const modelBuildMs = nowMs() - modelBuildStarted;
  const sessionStarted = nowMs();
  const session = await ort.createOrtSession(tinyOnnx);
  return { session, modelBuildMs, sessionCreateMs: nowMs() - sessionStarted };
}

async function runCachedPolicyValueHeadsOrt(input: Float32Array<ArrayBufferLike>, cached: CachedPolicyValueHeadSession): Promise<{
  runMs: number;
  mappedPolicy: number[];
  wdl: number[];
}> {
  const runStarted = nowMs();
  const outputs = await cached.session.run({ input: new ort.Tensor('float32', input, [DEFAULT_TOKENS, DEFAULT_N]) });
  const runMs = nowMs() - runStarted;
  const mappedPolicy = outputs.mappedPolicy.data as Float32Array<ArrayBufferLike>;
  const wdl = outputs.wdl.data as Float32Array<ArrayBufferLike>;
  return { runMs, mappedPolicy: Array.from(mappedPolicy), wdl: Array.from(wdl) };
}

type Lc0WebHybridHeadBackend = 'ort' | 'wgsl';
type Lc0WebHybridWgslBatchMode = 'physical' | 'serial';
type Lc0WebHybridInputBackend = 'js' | 'wgsl' | 'wasm';
export type Lc0WebHybridLegalPriorsBackend = 'js' | 'wasm' | 'gpu';

const WGSL_HEADS_READBACK_FLOATS = DEFAULT_POLICY_MAPPED_OUTPUTS + 3;
const WGSL_HEADS_READBACK_BYTES = WGSL_HEADS_READBACK_FLOATS * 4;

interface LegalPolicyCandidate {
  uci: string;
  index: number;
}

function lc0SquareFile(square: number): number {
  return square % 8;
}

function isLc0StandardCastlingMove(board: BoardState, move: Move): boolean {
  const piece = board.squares[move.from];
  return piece?.[1] === 'k' && Math.abs(lc0SquareFile(move.to) - lc0SquareFile(move.from)) === 2;
}

function legalPolicyCandidates(board: BoardState): LegalPolicyCandidate[] {
  const moveTransform = board.turn === 'b' ? LC0_MIRROR_TRANSFORM : 0;
  return legalMoves(board).map((move) => {
    const uci = moveToUci(move);
    const index = uciToLc0PolicyIndex(uci, moveTransform, { standardCastling: isLc0StandardCastlingMove(board, move) });
    if (index === undefined) throw new Error(`No LC0 policy index for legal move ${uci}`);
    return { uci, index };
  });
}

function legalPolicyPriorsFromCandidates(candidates: LegalPolicyCandidate[], logits: ArrayLike<number>, policyTemperature: number): Lc0Evaluation['legalPriors'] {
  if (!candidates.length) return [];
  const legal = candidates.map((entry) => ({ ...entry, logit: Number(logits[entry.index]) / policyTemperature }));
  const max = Math.max(...legal.map((entry) => entry.logit));
  const sum = legal.reduce((acc, entry) => acc + Math.exp(entry.logit - max), 0);
  return legal
    .map((entry) => ({ ...entry, prior: Math.exp(entry.logit - max) / sum }))
    .sort((a, b) => b.prior - a.prior);
}

type WgslPolicyValueHeadRuntime = {
  densePipeline: PipelineLike;
  policyLogitsPipeline: PipelineLike;
  mappedPolicyPipeline: PipelineLike;
  vectorPipeline: PipelineLike;
  softmaxPipeline: PipelineLike;
  legalPriorsPipeline: PipelineLike;
  policyBindGroup: unknown;
  policyQBindGroup: unknown;
  policyKBindGroup: unknown;
  policyLogitsBindGroup: unknown;
  mappedPolicyBindGroup: unknown;
  valueBindGroup: unknown;
  valueDense1BindGroup: unknown;
  valueDense2BindGroup: unknown;
  valueSoftmaxBindGroup: unknown;
  legalPriorsBindGroup: unknown;
  mappedPolicyBuffer: BufferLike;
  valueWdlBuffer: BufferLike;
  legalIndicesBuffer: BufferLike;
  legalArgsBuffer: BufferLike;
  legalOutputBuffer: BufferLike;
  headsReadbackBuffer: BufferLike;
  buffers: BufferLike[];
};

function createWgslPolicyValueHeadRuntime(device: DeviceLike, tensors: Lc0WebPolicyValueHeadTensors, inputBuffer: BufferLike, usage: Record<string, number>): WgslPolicyValueHeadRuntime {
  const denseModule = device.createShaderModule({ label: 'lc0web hybrid WGSL policy/value head dense', code: WGSL_HEADS_DENSE_PROBE });
  const densePipeline = device.createComputePipeline({ layout: 'auto', compute: { module: denseModule, entryPoint: 'main' } }) as PipelineLike;
  const policyLogitsModule = device.createShaderModule({ label: 'lc0web hybrid WGSL policy logits', code: WGSL_HEADS_POLICY_LOGITS_PROBE });
  const policyLogitsPipeline = device.createComputePipeline({ layout: 'auto', compute: { module: policyLogitsModule, entryPoint: 'main' } }) as PipelineLike;
  const mappedPolicyModule = device.createShaderModule({ label: 'lc0web hybrid WGSL mapped policy', code: WGSL_MAPPED_POLICY_PROBE });
  const mappedPolicyPipeline = device.createComputePipeline({ layout: 'auto', compute: { module: mappedPolicyModule, entryPoint: 'main' } }) as PipelineLike;
  const vectorModule = device.createShaderModule({ label: 'lc0web hybrid WGSL value vector dense', code: WGSL_HEADS_VECTOR_DENSE_PROBE });
  const vectorPipeline = device.createComputePipeline({ layout: 'auto', compute: { module: vectorModule, entryPoint: 'main' } }) as PipelineLike;
  const softmaxModule = device.createShaderModule({ label: 'lc0web hybrid WGSL WDL softmax', code: WGSL_HEADS_SOFTMAX3_PROBE });
  const softmaxPipeline = device.createComputePipeline({ layout: 'auto', compute: { module: softmaxModule, entryPoint: 'main' } }) as PipelineLike;
  const legalPriorsModule = device.createShaderModule({ label: 'lc0web hybrid WGSL legal priors', code: WGSL_LEGAL_PRIORS_PROBE });
  const legalPriorsPipeline = device.createComputePipeline({ layout: 'auto', compute: { module: legalPriorsModule, entryPoint: 'main' } }) as PipelineLike;

  const buffers: BufferLike[] = [];
  const policyWeightBuffer = createStorageBuffer(device, f16BytesToF32Array(tensors.policyDense1Weight.bytes, DEFAULT_N * DEFAULT_N), usage.STORAGE | usage.COPY_DST);
  const policyBiasBuffer = createStorageBuffer(device, f16BytesToF32Array(tensors.policyDense1Bias.bytes, DEFAULT_N), usage.STORAGE | usage.COPY_DST);
  const policyQWeightBuffer = createStorageBuffer(device, f16BytesToF32Array(tensors.policyQWeight.bytes, DEFAULT_N * DEFAULT_N), usage.STORAGE | usage.COPY_DST);
  const policyQBiasBuffer = createStorageBuffer(device, f16BytesToF32Array(tensors.policyQBias.bytes, DEFAULT_N), usage.STORAGE | usage.COPY_DST);
  const policyKWeightBuffer = createStorageBuffer(device, f16BytesToF32Array(tensors.policyKWeight.bytes, DEFAULT_N * DEFAULT_N), usage.STORAGE | usage.COPY_DST);
  const policyKBiasBuffer = createStorageBuffer(device, f16BytesToF32Array(tensors.policyKBias.bytes, DEFAULT_N), usage.STORAGE | usage.COPY_DST);
  const policyScaleBuffer = createStorageBuffer(device, f16BytesToF32Array(tensors.policyScale.bytes, 1), usage.STORAGE | usage.COPY_DST);
  const policyPromotionWeightBuffer = createStorageBuffer(device, f16BytesToF32Array(tensors.policyPromotionWeight.bytes, DEFAULT_N * 4), usage.STORAGE | usage.COPY_DST);
  const policyMappingBuffer = createStorageBuffer(device, readI32Array(tensors.policyMappingTable.bytes, DEFAULT_POLICY_MAPPED_OUTPUTS), usage.STORAGE | usage.COPY_DST);
  const valueWeightBuffer = createStorageBuffer(device, f16BytesToF32Array(tensors.valueEmbedWeight.bytes, DEFAULT_N * DEFAULT_VALUE_EMBED), usage.STORAGE | usage.COPY_DST);
  const valueBiasBuffer = createStorageBuffer(device, f16BytesToF32Array(tensors.valueEmbedBias.bytes, DEFAULT_VALUE_EMBED), usage.STORAGE | usage.COPY_DST);
  const valueDense1WeightBuffer = createStorageBuffer(device, f16BytesToF32Array(tensors.valueDense1Weight.bytes, DEFAULT_TOKENS * DEFAULT_VALUE_EMBED * DEFAULT_VALUE_HIDDEN), usage.STORAGE | usage.COPY_DST);
  const valueDense1BiasBuffer = createStorageBuffer(device, f16BytesToF32Array(tensors.valueDense1Bias.bytes, DEFAULT_VALUE_HIDDEN), usage.STORAGE | usage.COPY_DST);
  const valueDense2WeightBuffer = createStorageBuffer(device, f16BytesToF32Array(tensors.valueDense2Weight.bytes, DEFAULT_VALUE_HIDDEN * 3), usage.STORAGE | usage.COPY_DST);
  const valueDense2BiasBuffer = createStorageBuffer(device, f16BytesToF32Array(tensors.valueDense2Bias.bytes, 3), usage.STORAGE | usage.COPY_DST);
  const policyOutputBuffer = device.createBuffer({ size: DEFAULT_TOKENS * DEFAULT_N * 4, usage: usage.STORAGE }) as BufferLike;
  const policyQBuffer = device.createBuffer({ size: DEFAULT_TOKENS * DEFAULT_N * 4, usage: usage.STORAGE }) as BufferLike;
  const policyKBuffer = device.createBuffer({ size: DEFAULT_TOKENS * DEFAULT_N * 4, usage: usage.STORAGE }) as BufferLike;
  const policyLogitsBuffer = device.createBuffer({ size: DEFAULT_POLICY_OUTPUTS * 4, usage: usage.STORAGE }) as BufferLike;
  const mappedPolicyBuffer = device.createBuffer({ size: DEFAULT_POLICY_MAPPED_OUTPUTS * 4, usage: usage.STORAGE | usage.COPY_SRC }) as BufferLike;
  const valueOutputBuffer = device.createBuffer({ size: DEFAULT_TOKENS * DEFAULT_VALUE_EMBED * 4, usage: usage.STORAGE }) as BufferLike;
  const valueHiddenBuffer = device.createBuffer({ size: DEFAULT_VALUE_HIDDEN * 4, usage: usage.STORAGE }) as BufferLike;
  const valueLogitsBuffer = device.createBuffer({ size: 3 * 4, usage: usage.STORAGE }) as BufferLike;
  const valueWdlBuffer = device.createBuffer({ size: 3 * 4, usage: usage.STORAGE | usage.COPY_SRC }) as BufferLike;
  const legalIndicesBuffer = device.createBuffer({ size: WGSL_GPU_LEGAL_MAX_MOVES * 4, usage: usage.STORAGE | usage.COPY_DST }) as BufferLike;
  const legalArgsBuffer = device.createBuffer({ size: 4 * 4, usage: usage.UNIFORM | usage.COPY_DST }) as BufferLike;
  const legalOutputBuffer = device.createBuffer({ size: WGSL_GPU_LEGAL_OUTPUT_FLOATS * 4, usage: usage.STORAGE | usage.COPY_SRC }) as BufferLike;
  const headsReadbackBuffer = device.createBuffer({ size: Math.max(WGSL_HEADS_READBACK_BYTES, WGSL_GPU_LEGAL_READBACK_BYTES), usage: usage.MAP_READ | usage.COPY_DST }) as BufferLike;
  const policyShapeBuffer = createU32UniformBuffer(device, [DEFAULT_N, 1], usage.UNIFORM | usage.COPY_DST);
  const policyLinearShapeBuffer = createU32UniformBuffer(device, [DEFAULT_N, 0], usage.UNIFORM | usage.COPY_DST);
  const valueShapeBuffer = createU32UniformBuffer(device, [DEFAULT_VALUE_EMBED, 1], usage.UNIFORM | usage.COPY_DST);
  const valueDense1ShapeBuffer = createU32UniformBuffer(device, [DEFAULT_TOKENS * DEFAULT_VALUE_EMBED, DEFAULT_VALUE_HIDDEN, 1], usage.UNIFORM | usage.COPY_DST);
  const valueDense2ShapeBuffer = createU32UniformBuffer(device, [DEFAULT_VALUE_HIDDEN, 3, 0], usage.UNIFORM | usage.COPY_DST);
  buffers.push(policyWeightBuffer, policyBiasBuffer, policyQWeightBuffer, policyQBiasBuffer, policyKWeightBuffer, policyKBiasBuffer, policyScaleBuffer, policyPromotionWeightBuffer, policyMappingBuffer, valueWeightBuffer, valueBiasBuffer, valueDense1WeightBuffer, valueDense1BiasBuffer, valueDense2WeightBuffer, valueDense2BiasBuffer, policyOutputBuffer, policyQBuffer, policyKBuffer, policyLogitsBuffer, mappedPolicyBuffer, valueOutputBuffer, valueHiddenBuffer, valueLogitsBuffer, valueWdlBuffer, legalIndicesBuffer, legalArgsBuffer, legalOutputBuffer, headsReadbackBuffer, policyShapeBuffer, policyLinearShapeBuffer, valueShapeBuffer, valueDense1ShapeBuffer, valueDense2ShapeBuffer);

  return {
    densePipeline,
    policyLogitsPipeline,
    mappedPolicyPipeline,
    vectorPipeline,
    softmaxPipeline,
    legalPriorsPipeline,
    policyBindGroup: createWgslHeadsDenseBindGroup(device, densePipeline, inputBuffer, policyWeightBuffer, policyBiasBuffer, policyOutputBuffer, policyShapeBuffer),
    policyQBindGroup: createWgslHeadsDenseBindGroup(device, densePipeline, policyOutputBuffer, policyQWeightBuffer, policyQBiasBuffer, policyQBuffer, policyLinearShapeBuffer),
    policyKBindGroup: createWgslHeadsDenseBindGroup(device, densePipeline, policyOutputBuffer, policyKWeightBuffer, policyKBiasBuffer, policyKBuffer, policyLinearShapeBuffer),
    policyLogitsBindGroup: createWgslHeadsPolicyLogitsBindGroup(device, policyLogitsPipeline, policyQBuffer, policyKBuffer, policyScaleBuffer, policyLogitsBuffer),
    mappedPolicyBindGroup: createMappedPolicyBindGroup(device, mappedPolicyPipeline, policyLogitsBuffer, policyKBuffer, policyPromotionWeightBuffer, policyMappingBuffer, mappedPolicyBuffer),
    valueBindGroup: createWgslHeadsDenseBindGroup(device, densePipeline, inputBuffer, valueWeightBuffer, valueBiasBuffer, valueOutputBuffer, valueShapeBuffer),
    valueDense1BindGroup: createWgslHeadsDenseBindGroup(device, vectorPipeline, valueOutputBuffer, valueDense1WeightBuffer, valueDense1BiasBuffer, valueHiddenBuffer, valueDense1ShapeBuffer),
    valueDense2BindGroup: createWgslHeadsDenseBindGroup(device, vectorPipeline, valueHiddenBuffer, valueDense2WeightBuffer, valueDense2BiasBuffer, valueLogitsBuffer, valueDense2ShapeBuffer),
    valueSoftmaxBindGroup: createWgslHeadsSoftmaxBindGroup(device, softmaxPipeline, valueLogitsBuffer, valueWdlBuffer),
    legalPriorsBindGroup: createWgslLegalPriorsBindGroup(device, legalPriorsPipeline, mappedPolicyBuffer, legalIndicesBuffer, legalOutputBuffer, legalArgsBuffer),
    mappedPolicyBuffer,
    valueWdlBuffer,
    legalIndicesBuffer,
    legalArgsBuffer,
    legalOutputBuffer,
    headsReadbackBuffer,
    buffers,
  };
}

function encodeWgslPolicyValueHeads(pass: ComputePassLike, runtime: WgslPolicyValueHeadRuntime): void {
  pass.setPipeline(runtime.densePipeline);
  pass.setBindGroup(0, runtime.policyBindGroup);
  pass.dispatchWorkgroups(Math.ceil(DEFAULT_N / 16), DEFAULT_TOKENS);
  pass.setBindGroup(0, runtime.valueBindGroup);
  pass.dispatchWorkgroups(Math.ceil(DEFAULT_VALUE_EMBED / 16), DEFAULT_TOKENS);
  pass.setBindGroup(0, runtime.policyQBindGroup);
  pass.dispatchWorkgroups(Math.ceil(DEFAULT_N / 16), DEFAULT_TOKENS);
  pass.setBindGroup(0, runtime.policyKBindGroup);
  pass.dispatchWorkgroups(Math.ceil(DEFAULT_N / 16), DEFAULT_TOKENS);
  pass.setPipeline(runtime.policyLogitsPipeline);
  pass.setBindGroup(0, runtime.policyLogitsBindGroup);
  pass.dispatchWorkgroups(Math.ceil(DEFAULT_TOKENS / 8), Math.ceil(DEFAULT_TOKENS / 8));
  pass.setPipeline(runtime.mappedPolicyPipeline);
  pass.setBindGroup(0, runtime.mappedPolicyBindGroup);
  pass.dispatchWorkgroups(Math.ceil(DEFAULT_POLICY_MAPPED_OUTPUTS / 64));
  pass.setPipeline(runtime.vectorPipeline);
  pass.setBindGroup(0, runtime.valueDense1BindGroup);
  pass.dispatchWorkgroups(Math.ceil(DEFAULT_VALUE_HIDDEN / 64));
  pass.setBindGroup(0, runtime.valueDense2BindGroup);
  pass.dispatchWorkgroups(1);
  pass.setPipeline(runtime.softmaxPipeline);
  pass.setBindGroup(0, runtime.valueSoftmaxBindGroup);
  pass.dispatchWorkgroups(1);
}

function copyWgslPolicyValueHeadOutputsTo(encoder: CommandEncoderLike, runtime: WgslPolicyValueHeadRuntime, readbackBuffer: BufferLike, destinationOffset: number): void {
  encoder.copyBufferToBuffer(runtime.mappedPolicyBuffer, 0, readbackBuffer, destinationOffset, DEFAULT_POLICY_MAPPED_OUTPUTS * 4);
  encoder.copyBufferToBuffer(runtime.valueWdlBuffer, 0, readbackBuffer, destinationOffset + DEFAULT_POLICY_MAPPED_OUTPUTS * 4, 3 * 4);
}

function copyWgslPolicyValueHeadOutputs(encoder: CommandEncoderLike, runtime: WgslPolicyValueHeadRuntime): void {
  copyWgslPolicyValueHeadOutputsTo(encoder, runtime, runtime.headsReadbackBuffer, 0);
}

function uploadWgslLegalPriorsInputs(device: DeviceLike, runtime: WgslPolicyValueHeadRuntime, legalCandidates: LegalPolicyCandidate[], policyTemperature: number): void {
  if (legalCandidates.length > WGSL_GPU_LEGAL_MAX_MOVES) throw new Error(`WGSL legal-prior path supports at most ${WGSL_GPU_LEGAL_MAX_MOVES} legal moves, got ${legalCandidates.length}`);
  const indices = new Uint32Array(WGSL_GPU_LEGAL_MAX_MOVES);
  for (let i = 0; i < legalCandidates.length; i++) indices[i] = legalCandidates[i].index;
  device.queue.writeBuffer(runtime.legalIndicesBuffer, 0, indices);
  device.queue.writeBuffer(runtime.legalArgsBuffer, 0, new Float32Array([legalCandidates.length, policyTemperature, 0, 0]));
}

function encodeWgslLegalPriors(pass: ComputePassLike, runtime: WgslPolicyValueHeadRuntime): void {
  pass.setPipeline(runtime.legalPriorsPipeline);
  pass.setBindGroup(0, runtime.legalPriorsBindGroup);
  pass.dispatchWorkgroups(1);
}

function copyWgslLegalPriorsOutputsTo(encoder: CommandEncoderLike, runtime: WgslPolicyValueHeadRuntime, readbackBuffer: BufferLike, destinationOffset: number): void {
  encoder.copyBufferToBuffer(runtime.legalOutputBuffer, 0, readbackBuffer, destinationOffset, WGSL_GPU_LEGAL_OUTPUT_FLOATS * 4);
  encoder.copyBufferToBuffer(runtime.valueWdlBuffer, 0, readbackBuffer, destinationOffset + WGSL_GPU_LEGAL_OUTPUT_FLOATS * 4, 3 * 4);
}

function copyWgslLegalPriorsOutputs(encoder: CommandEncoderLike, runtime: WgslPolicyValueHeadRuntime): void {
  copyWgslLegalPriorsOutputsTo(encoder, runtime, runtime.headsReadbackBuffer, 0);
}

async function mapWgslPolicyValueHeadOutputs(runtime: WgslPolicyValueHeadRuntime): Promise<{
  mappedPolicy: Float32Array<ArrayBufferLike>;
  wdl: Float32Array<ArrayBufferLike>;
  readbackSyncedMs: number;
}> {
  const globals = gpuGlobals();
  const started = nowMs();
  await runtime.headsReadbackBuffer.mapAsync(globals.GPUMapMode!.READ);
  const range = runtime.headsReadbackBuffer.getMappedRange();
  const mappedPolicy = new Float32Array(DEFAULT_POLICY_MAPPED_OUTPUTS);
  mappedPolicy.set(new Float32Array(range, 0, DEFAULT_POLICY_MAPPED_OUTPUTS));
  const wdl = new Float32Array(3);
  wdl.set(new Float32Array(range, DEFAULT_POLICY_MAPPED_OUTPUTS * 4, 3));
  runtime.headsReadbackBuffer.unmap();
  return { mappedPolicy, wdl, readbackSyncedMs: nowMs() - started };
}

function legalPriorsFromGpuOutput(legalCandidates: LegalPolicyCandidate[], output: Float32Array<ArrayBufferLike>): Lc0Evaluation['legalPriors'] {
  const priors: Lc0Evaluation['legalPriors'] = [];
  for (let i = 0; i < legalCandidates.length; i++) {
    const base = i * 3;
    const slot = Math.round(output[base]);
    const candidate = legalCandidates[slot];
    if (!candidate) throw new Error(`WGSL legal-prior output referenced invalid legal slot ${slot}`);
    priors.push({ uci: candidate.uci, index: candidate.index, logit: Number(output[base + 1]), prior: Number(output[base + 2]) });
  }
  return priors;
}

async function mapWgslLegalPriorsOutputs(runtime: WgslPolicyValueHeadRuntime, legalCandidates: LegalPolicyCandidate[]): Promise<{
  legalPriors: Lc0Evaluation['legalPriors'];
  wdl: Float32Array<ArrayBufferLike>;
  readbackSyncedMs: number;
}> {
  const globals = gpuGlobals();
  const started = nowMs();
  await runtime.headsReadbackBuffer.mapAsync(globals.GPUMapMode!.READ);
  const range = runtime.headsReadbackBuffer.getMappedRange();
  const legalOutput = new Float32Array(WGSL_GPU_LEGAL_OUTPUT_FLOATS);
  legalOutput.set(new Float32Array(range, 0, WGSL_GPU_LEGAL_OUTPUT_FLOATS));
  const wdl = new Float32Array(3);
  wdl.set(new Float32Array(range, WGSL_GPU_LEGAL_OUTPUT_FLOATS * 4, 3));
  runtime.headsReadbackBuffer.unmap();
  return { legalPriors: legalPriorsFromGpuOutput(legalCandidates, legalOutput), wdl, readbackSyncedMs: nowMs() - started };
}

async function runCachedWgslPolicyValueHeads(device: DeviceLike, runtime: WgslPolicyValueHeadRuntime): Promise<{ runMs: number; mappedPolicy: number[]; wdl: number[]; readbackSyncedMs: number }> {
  const started = nowMs();
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  encodeWgslPolicyValueHeads(pass, runtime);
  pass.end();
  copyWgslPolicyValueHeadOutputs(encoder, runtime);
  device.queue.submit([encoder.finish()]);
  const { mappedPolicy, wdl, readbackSyncedMs } = await mapWgslPolicyValueHeadOutputs(runtime);
  if (!arrayHasNonzero(mappedPolicy) || !arrayHasVariation(mappedPolicy) || !arrayHasNonzero(wdl) || !arrayHasVariation(wdl)) throw new Error('WGSL hybrid heads produced zero or uniform mapped policy/WDL');
  return { runMs: nowMs() - started, mappedPolicy: Array.from(mappedPolicy), wdl: Array.from(wdl), readbackSyncedMs };
}

const DEFAULT_INPUT_BODY_TENSORS = {
  posEncoding: '/const/pos_encoding',
  inputWeight: '/attn_body/matmul/w',
  inputBias: '/attn_body/add/w',
  mulGate: '/ip_mul_gate/w',
  addGate: '/ip_add_gate/w',
} as const;
const DEFAULT_INPUT_PLANES = 112;
const DEFAULT_POSITIONAL_CHANNELS = 64;
const DEFAULT_PADDED_INPUT_CHANNELS = DEFAULT_INPUT_PLANES + DEFAULT_POSITIONAL_CHANNELS;

interface Lc0WebInitialInputTensors {
  posEncoding: Lc0WebTensorView;
  inputWeight: Lc0WebTensorView;
  inputBias: Lc0WebTensorView;
  mulGate: Lc0WebTensorView;
  addGate: Lc0WebTensorView;
}

interface Lc0WebPreparedInitialInputTensors {
  posEncoding: Float32Array<ArrayBufferLike>;
  inputWeight: Float32Array<ArrayBufferLike>;
  inputBias: Float32Array<ArrayBufferLike>;
  mulGate: Float32Array<ArrayBufferLike>;
  addGate: Float32Array<ArrayBufferLike>;
}

export interface Lc0WebHybridEvaluationOptions {
  packUrl: string;
  input: Lc0EvaluatorInput;
  layers?: number;
  verifyShards?: boolean;
  historyFill?: Lc0HistoryFill;
  policyTemperature?: number;
  headBackend?: Lc0WebHybridHeadBackend;
  wgslBatchMode?: Lc0WebHybridWgslBatchMode;
  inputBackend?: Lc0WebHybridInputBackend;
  legalPriorsBackend?: Lc0WebHybridLegalPriorsBackend;
  encoderKernelVariant?: Lc0WebEncoderKernelVariant;
  timestampQuery?: boolean;
}

export interface Lc0WebHybridTimingBreakdown {
  totalEvalMs: number;
  inputBuildMs: number;
  inputUploadMs: number;
  commandEncodeMs: number;
  queueSubmitMs: number;
  /** Queue drain plus copy/map time for the backend's required readback(s). */
  readbackSyncedMs: number;
  headRunMs: number;
  legalPriorsMs: number;
  readbackBytes: number;
  readbackMapCount: number;
  /** Number of WebGPU compute dispatch calls encoded by this custom WGSL eval, when available. */
  dispatchCount?: number;
  readbackMode?: 'immediate' | 'deferred-double-buffered';
  /** Monotonic runtime submission id for strict deferred-readback diagnostics. */
  wgslSequenceId?: number;
  /** Index of this logical batch in a multi-batch deferred sequence call. */
  batchSequenceIndex?: number;
  /** Readback-buffer ring slot used by deferred/double-buffered WGSL-head evaluation. */
  deferredReadbackSlot?: number;
  /** Wall time between queue submission and starting mapAsync; near-zero when readback mapping is requested eagerly. */
  deferredReadbackDelayMs?: number;
  /** Total mapAsync pending time from request to resolution, separated from CPU copy/postprocess. */
  readbackMapAsyncMs?: number;
  /** Wall time spent awaiting an already-started mapAsync promise during result finalization. */
  readbackMapAsyncWaitMs?: number;
  /** CPU time to copy the mapped readback range and unmap it. */
  readbackMapCopyMs?: number;
  /** CPU legal-move/index preparation run after queue submission while mapAsync is already pending. */
  legalPriorsPrepMs?: number;
  /** CPU work intentionally scheduled while mapAsync is pending to hide under GPU/readback latency. */
  readbackOverlapCpuMs?: number;
  /** Portion of readbackOverlapCpuMs estimated to be hidden by the pending mapAsync interval. */
  readbackOverlapHiddenMs?: number;
  physicalBatchSize?: number;
  batchPosition?: number;
  inputBackend?: Lc0WebHybridInputBackend;
  legalPriorsBackend?: Lc0WebHybridLegalPriorsBackend;
  encoderKernelVariant?: Lc0WebEncoderKernelVariant;
  inputBridgeCopyMs?: number;
  wasmEncodeMs?: number;
  wasmTotalMs?: number;
  legalPriorsBridgeCopyMs?: number;
  legalPriorsWasmRunMs?: number;
  legalPriorsWasmTotalMs?: number;
}

export interface Lc0WebHybridEvaluationResult extends Lc0Evaluation {
  status: 'LC0WEB_HYBRID_EVALUATION_DONE';
  backend: 'lc0web-wgsl-encoder-ort-heads' | 'lc0web-wgsl-encoder-wgsl-heads';
  packUrl: string;
  layers: number;
  encoderKernelVariant?: Lc0WebEncoderKernelVariant;
  packLoadMs: number;
  encoderDispatchSyncedMs: number;
  headRunMs: number;
  mappedPolicy: number[];
  timing: Lc0WebHybridTimingBreakdown;
}

export type Lc0WebHybridEncoderProfileStageName =
  | 'inputBody'
  | 'smolgen'
  | 'smolgenCompress'
  | 'smolgenDense1'
  | 'smolgenLn1'
  | 'smolgenDense2'
  | 'smolgenLn2'
  | 'smolgenProject'
  | 'qkvProjection'
  | 'attentionScores'
  | 'softmax'
  | 'attentionValue'
  | 'outputProjection'
  | 'ln1'
  | 'ffnDense1'
  | 'ffnDense2Residual'
  | 'ln2';

export interface Lc0WebHybridEncoderProfileStageTiming {
  stage: Lc0WebHybridEncoderProfileStageName;
  label: string;
  iterations: number;
  totalMs: number;
  avgMs: number;
  percentOfProfiledStageMs: number;
}

export interface Lc0WebHybridEncoderProfileLayerTiming {
  layer: number;
  totalMs: number;
  stages: Lc0WebHybridEncoderProfileStageTiming[];
}

export type Lc0WebHybridEncoderProfileMode = 'sync-staged' | 'gpu-timestamp';

export interface Lc0WebHybridEncoderProfileOptions {
  packUrl: string;
  input: Lc0EvaluatorInput;
  layers?: number;
  iterations?: number;
  warmup?: number;
  verifyShards?: boolean;
  inputBackend?: Lc0WebHybridInputBackend;
  encoderKernelVariant?: Lc0WebEncoderKernelVariant;
  historyFill?: Lc0HistoryFill;
  profileMode?: Lc0WebHybridEncoderProfileMode;
}

export interface Lc0WebHybridEncoderProfileResult {
  status: 'HYBRID_ENCODER_PROFILE_DONE';
  packUrl: string;
  layers: number;
  encoderKernelVariant: Lc0WebEncoderKernelVariant;
  inputBackend: Lc0WebHybridInputBackend;
  warmup: number;
  iterations: number;
  packLoadMs: number;
  profileMode: Lc0WebHybridEncoderProfileMode;
  requestedProfileMode: Lc0WebHybridEncoderProfileMode;
  gpuTimestampSupported: boolean;
  profiledStageTotalMs: number;
  readbackSyncedMs: number;
  outputSample: number[];
  aggregateStageTimings: Lc0WebHybridEncoderProfileStageTiming[];
  layerTimings: Lc0WebHybridEncoderProfileLayerTiming[];
  note: string;
}

function inputBodyTensorNameList(): string[] {
  return Object.values(DEFAULT_INPUT_BODY_TENSORS);
}

function loadInitialInputTensors(pack: Awaited<ReturnType<typeof loadLc0WebModelPack>>): Lc0WebInitialInputTensors {
  const get = (name: string): Lc0WebTensorView => {
    const tensor = pack.tensors.get(name);
    if (!tensor) throw new Error(`lc0web input tensor was not loaded: ${name}`);
    return tensor;
  };
  const tensors: Lc0WebInitialInputTensors = {
    posEncoding: get(DEFAULT_INPUT_BODY_TENSORS.posEncoding),
    inputWeight: get(DEFAULT_INPUT_BODY_TENSORS.inputWeight),
    inputBias: get(DEFAULT_INPUT_BODY_TENSORS.inputBias),
    mulGate: get(DEFAULT_INPUT_BODY_TENSORS.mulGate),
    addGate: get(DEFAULT_INPUT_BODY_TENSORS.addGate),
  };
  assertTensorShapeAndBytes(tensors.posEncoding, [1, DEFAULT_TOKENS, DEFAULT_POSITIONAL_CHANNELS], 2, 'posEncoding');
  assertTensorShapeAndBytes(tensors.inputWeight, [DEFAULT_PADDED_INPUT_CHANNELS, DEFAULT_N], 2, 'inputWeight');
  assertTensorShapeAndBytes(tensors.inputBias, [DEFAULT_N], 2, 'inputBias');
  assertTensorShapeAndBytes(tensors.mulGate, [DEFAULT_TOKENS, DEFAULT_N], 2, 'mulGate');
  assertTensorShapeAndBytes(tensors.addGate, [DEFAULT_TOKENS, DEFAULT_N], 2, 'addGate');
  return tensors;
}

function prepareInitialInputTensors(tensors: Lc0WebInitialInputTensors): Lc0WebPreparedInitialInputTensors {
  return {
    posEncoding: f16BytesToF32Array(tensors.posEncoding.bytes, DEFAULT_TOKENS * DEFAULT_POSITIONAL_CHANNELS),
    inputWeight: f16BytesToF32Array(tensors.inputWeight.bytes, DEFAULT_PADDED_INPUT_CHANNELS * DEFAULT_N),
    inputBias: f16BytesToF32Array(tensors.inputBias.bytes, DEFAULT_N),
    mulGate: f16BytesToF32Array(tensors.mulGate.bytes, DEFAULT_TOKENS * DEFAULT_N),
    addGate: f16BytesToF32Array(tensors.addGate.bytes, DEFAULT_TOKENS * DEFAULT_N),
  };
}

function cpuProjectTokensF32(input: Float32Array<ArrayBufferLike>, weight: Float32Array<ArrayBufferLike>, bias: Float32Array<ArrayBufferLike>, tokens: number, k: number, n: number): Float32Array<ArrayBufferLike> {
  const output = new Float32Array(tokens * n);
  for (let token = 0; token < tokens; token++) {
    const inputBase = token * k;
    const outputBase = token * n;
    for (let col = 0; col < n; col++) {
      let sum = bias[col];
      for (let row = 0; row < k; row++) sum += input[inputBase + row] * weight[row * n + col];
      output[outputBase + col] = sum;
    }
  }
  return output;
}

function buildInitialEncoderPlanes(input: Lc0EvaluatorInput, historyFill: Lc0HistoryFill): Float32Array<ArrayBufferLike> {
  return encodeLc0Classical112(input, { historyFill }).planes;
}

function boardStateToFen(board: BoardState | string): string {
  return typeof board === 'string' ? board : boardToFen(board);
}

function buildInitialEncoderPlanesWasm(input: Lc0EvaluatorInput, historyFill: Lc0HistoryFill, encoder: Lc0WasmInputEncoder): { planes: Float32Array<ArrayBufferLike>; timing: Lc0WasmInputEncoderTiming } {
  if (typeof input === 'object' && input !== null && 'positions' in input) {
    const fens = input.positions.map(boardStateToFen);
    const result = encoder.encodeFenHistoryTimed(fens);
    return { planes: result.encoded.planes, timing: result.timing };
  }
  const result = encoder.encodeFenTimed(boardStateToFen(input), { historyFill: historyFill !== 'no' });
  return { planes: result.encoded.planes, timing: result.timing };
}

function buildInitialEncoderActivation(input: Lc0EvaluatorInput, tensors: Lc0WebPreparedInitialInputTensors, historyFill: Lc0HistoryFill): Float32Array<ArrayBufferLike> {
  const encoded = buildInitialEncoderPlanes(input, historyFill);
  const padded = new Float32Array(DEFAULT_TOKENS * DEFAULT_PADDED_INPUT_CHANNELS);
  for (let token = 0; token < DEFAULT_TOKENS; token++) {
    const base = token * DEFAULT_PADDED_INPUT_CHANNELS;
    for (let plane = 0; plane < DEFAULT_INPUT_PLANES; plane++) padded[base + plane] = encoded[plane * DEFAULT_TOKENS + token];
    padded.set(tensors.posEncoding.subarray(token * DEFAULT_POSITIONAL_CHANNELS, (token + 1) * DEFAULT_POSITIONAL_CHANNELS), base + DEFAULT_INPUT_PLANES);
  }
  const projected = cpuProjectTokensF32(padded, tensors.inputWeight, tensors.inputBias, DEFAULT_TOKENS, DEFAULT_PADDED_INPUT_CHANNELS, DEFAULT_N);
  const out = new Float32Array(DEFAULT_TOKENS * DEFAULT_N);
  for (let i = 0; i < projected.length; i++) {
    const x = projected[i];
    const mish = x * Math.tanh(Math.log1p(Math.exp(x)));
    out[i] = mish * tensors.mulGate[i] + tensors.addGate[i];
  }
  return out;
}

const INPUT_BODY_WGSL = `
struct InputShape { inputPlanes: u32, positionalChannels: u32, paddedChannels: u32, outputChannels: u32 };
@group(0) @binding(0) var<storage, read> planes: array<f32>;
@group(0) @binding(1) var<storage, read> posEncoding: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read> mulGate: array<f32>;
@group(0) @binding(5) var<storage, read> addGate: array<f32>;
@group(0) @binding(6) var<storage, read_write> output: array<f32>;
@group(0) @binding(7) var<uniform> shape: InputShape;

fn mish(x: f32) -> f32 {
  return x * tanh(log(1.0 + exp(x)));
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let channel = gid.x;
  let token = gid.y;
  if (channel >= shape.outputChannels || token >= 64u) { return; }
  var sum = bias[channel];
  for (var row = 0u; row < shape.inputPlanes; row = row + 1u) {
    sum = sum + planes[row * 64u + token] * weight[row * shape.outputChannels + channel];
  }
  for (var pos = 0u; pos < shape.positionalChannels; pos = pos + 1u) {
    let row = shape.inputPlanes + pos;
    sum = sum + posEncoding[token * shape.positionalChannels + pos] * weight[row * shape.outputChannels + channel];
  }
  let offset = token * shape.outputChannels + channel;
  output[offset] = mish(sum) * mulGate[offset] + addGate[offset];
}
`;

type InputBodyGpuRuntime = {
  planesBuffer: BufferLike;
  pipeline: PipelineLike;
  bindGroup: unknown;
  buffers: BufferLike[];
};

function createInputBodyGpuRuntime(device: DeviceLike, tensors: Lc0WebPreparedInitialInputTensors, outputBuffer: BufferLike, usage: Record<string, number>): InputBodyGpuRuntime {
  const planesBuffer = device.createBuffer({ size: DEFAULT_INPUT_PLANES * DEFAULT_TOKENS * 4, usage: usage.STORAGE | usage.COPY_DST });
  const posEncodingBuffer = createStorageBuffer(device, tensors.posEncoding, usage.STORAGE | usage.COPY_DST);
  const inputWeightBuffer = createStorageBuffer(device, tensors.inputWeight, usage.STORAGE | usage.COPY_DST);
  const inputBiasBuffer = createStorageBuffer(device, tensors.inputBias, usage.STORAGE | usage.COPY_DST);
  const mulGateBuffer = createStorageBuffer(device, tensors.mulGate, usage.STORAGE | usage.COPY_DST);
  const addGateBuffer = createStorageBuffer(device, tensors.addGate, usage.STORAGE | usage.COPY_DST);
  const shape = new Uint32Array([DEFAULT_INPUT_PLANES, DEFAULT_POSITIONAL_CHANNELS, DEFAULT_PADDED_INPUT_CHANNELS, DEFAULT_N]);
  const shapeBuffer = createStorageBuffer(device, shape, usage.UNIFORM | usage.COPY_DST);
  const pipeline = device.createComputePipeline({ layout: 'auto', compute: { module: device.createShaderModule({ label: 'lc0web input body projection', code: INPUT_BODY_WGSL }), entryPoint: 'main' } }) as PipelineLike;
  const bindGroup = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
    { binding: 0, resource: { buffer: planesBuffer } },
    { binding: 1, resource: { buffer: posEncodingBuffer } },
    { binding: 2, resource: { buffer: inputWeightBuffer } },
    { binding: 3, resource: { buffer: inputBiasBuffer } },
    { binding: 4, resource: { buffer: mulGateBuffer } },
    { binding: 5, resource: { buffer: addGateBuffer } },
    { binding: 6, resource: { buffer: outputBuffer } },
    { binding: 7, resource: { buffer: shapeBuffer } },
  ] });
  return { planesBuffer, pipeline, bindGroup, buffers: [planesBuffer, posEncodingBuffer, inputWeightBuffer, inputBiasBuffer, mulGateBuffer, addGateBuffer, shapeBuffer] };
}

function encodeInputBodyPass(pass: ComputePassLike, runtime: InputBodyGpuRuntime): void {
  pass.setPipeline(runtime.pipeline);
  pass.setBindGroup(0, runtime.bindGroup);
  pass.dispatchWorkgroups(Math.ceil(DEFAULT_N / 8), Math.ceil(DEFAULT_TOKENS / 8));
}

type HybridEncoderLayerWeights = {
  qWeight: BufferLike;
  qBias: BufferLike;
  kWeight: BufferLike;
  kBias: BufferLike;
  vWeight: BufferLike;
  vBias: BufferLike;
  scale: BufferLike;
  outWeight: BufferLike;
  outBias: BufferLike;
  attentionAlpha: BufferLike;
  ln1Scale: BufferLike;
  ln1Bias: BufferLike;
  ffnDense1Weight: BufferLike;
  ffnDense1Bias: BufferLike;
  ffnDense2Weight: BufferLike;
  ffnDense2Bias: BufferLike;
  ffnAlpha: BufferLike;
  ln2Scale: BufferLike;
  ln2Bias: BufferLike;
  smolgenCompressWeight: BufferLike;
  smolgenDense1Weight: BufferLike;
  smolgenDense1Bias: BufferLike;
  smolgenLn1Scale: BufferLike;
  smolgenLn1Bias: BufferLike;
  smolgenDense2Weight: BufferLike;
  smolgenDense2Bias: BufferLike;
  smolgenLn2Scale: BufferLike;
  smolgenLn2Bias: BufferLike;
  smolgenWeight: BufferLike;
};

type HybridEncoderLayerSlotRuntime = {
  output: BufferLike;
  smolgenPipelines: SmolgenPipelines;
  attentionPipelines: ReturnType<typeof createAttentionOutputPipelines>;
  ffnPipelines: ReturnType<typeof createEncoder0FfnPipelines>;
};

type HybridEncoderLayerRuntime = HybridEncoderLayerSlotRuntime & {
  weights: HybridEncoderLayerWeights;
};

type HybridWgslBatchSlot = {
  inputBuffer: BufferLike;
  inputBodyGpu?: InputBodyGpuRuntime;
  layerRuntimes: HybridEncoderLayerSlotRuntime[];
  wgslHeads: WgslPolicyValueHeadRuntime;
};

function createHybridEncoderLayerWeights(device: DeviceLike, tensors: ReturnType<typeof loadEncoder0FfnInputs>, usage: Record<string, number>): { weights: HybridEncoderLayerWeights; buffers: BufferLike[] } {
  const qWeight = createTransposedF16StorageBuffer(device, tensors.qWeight.bytes, DEFAULT_K, DEFAULT_N, usage.STORAGE | usage.COPY_DST);
  const qBias = createStorageBuffer(device, tensors.qBias.bytes, usage.STORAGE | usage.COPY_DST);
  const kWeight = createTransposedF16StorageBuffer(device, tensors.kWeight.bytes, DEFAULT_K, DEFAULT_N, usage.STORAGE | usage.COPY_DST);
  const kBias = createStorageBuffer(device, tensors.kBias.bytes, usage.STORAGE | usage.COPY_DST);
  const vWeight = createTransposedF16StorageBuffer(device, tensors.vWeight.bytes, DEFAULT_K, DEFAULT_N, usage.STORAGE | usage.COPY_DST);
  const vBias = createStorageBuffer(device, tensors.vBias.bytes, usage.STORAGE | usage.COPY_DST);
  const scale = createStorageBuffer(device, paddedF16ScalarBytes(tensors.scale.bytes), usage.STORAGE | usage.COPY_DST);
  const outWeight = createTransposedF16StorageBuffer(device, tensors.outWeight.bytes, DEFAULT_N, DEFAULT_N, usage.STORAGE | usage.COPY_DST);
  const outBias = createStorageBuffer(device, tensors.outBias.bytes, usage.STORAGE | usage.COPY_DST);
  const attentionAlpha = createStorageBuffer(device, paddedF16ScalarBytes(tensors.alpha.bytes), usage.STORAGE | usage.COPY_DST);
  const ln1Scale = createStorageBuffer(device, tensors.lnScale.bytes, usage.STORAGE | usage.COPY_DST);
  const ln1Bias = createStorageBuffer(device, tensors.lnBias.bytes, usage.STORAGE | usage.COPY_DST);
  const ffnDense1Weight = createTransposedF16StorageBuffer(device, tensors.ffnDense1Weight.bytes, DEFAULT_N, DEFAULT_FFN_HIDDEN, usage.STORAGE | usage.COPY_DST);
  const ffnDense1Bias = createStorageBuffer(device, tensors.ffnDense1Bias.bytes, usage.STORAGE | usage.COPY_DST);
  const ffnDense2Weight = createTransposedF16StorageBuffer(device, tensors.ffnDense2Weight.bytes, DEFAULT_FFN_HIDDEN, DEFAULT_N, usage.STORAGE | usage.COPY_DST);
  const ffnDense2Bias = createStorageBuffer(device, tensors.ffnDense2Bias.bytes, usage.STORAGE | usage.COPY_DST);
  const ffnAlpha = createStorageBuffer(device, paddedF16ScalarBytes(tensors.ffnAlpha.bytes), usage.STORAGE | usage.COPY_DST);
  const ln2Scale = createStorageBuffer(device, tensors.ln2Scale.bytes, usage.STORAGE | usage.COPY_DST);
  const ln2Bias = createStorageBuffer(device, tensors.ln2Bias.bytes, usage.STORAGE | usage.COPY_DST);
  const smolgenCompressWeight = createStorageBuffer(device, tensors.smolgen.compressWeight.bytes, usage.STORAGE | usage.COPY_DST);
  const smolgenDense1Weight = createStorageBuffer(device, tensors.smolgen.dense1Weight.bytes, usage.STORAGE | usage.COPY_DST);
  const smolgenDense1Bias = createStorageBuffer(device, tensors.smolgen.dense1Bias.bytes, usage.STORAGE | usage.COPY_DST);
  const smolgenLn1Scale = createStorageBuffer(device, tensors.smolgen.ln1Scale.bytes, usage.STORAGE | usage.COPY_DST);
  const smolgenLn1Bias = createStorageBuffer(device, tensors.smolgen.ln1Bias.bytes, usage.STORAGE | usage.COPY_DST);
  const smolgenDense2Weight = createStorageBuffer(device, tensors.smolgen.dense2Weight.bytes, usage.STORAGE | usage.COPY_DST);
  const smolgenDense2Bias = createStorageBuffer(device, tensors.smolgen.dense2Bias.bytes, usage.STORAGE | usage.COPY_DST);
  const smolgenLn2Scale = createStorageBuffer(device, tensors.smolgen.ln2Scale.bytes, usage.STORAGE | usage.COPY_DST);
  const smolgenLn2Bias = createStorageBuffer(device, tensors.smolgen.ln2Bias.bytes, usage.STORAGE | usage.COPY_DST);
  const smolgenWeight = createStorageBuffer(device, tensors.smolgen.smolgenWeight.bytes, usage.STORAGE | usage.COPY_DST);
  const weights = { qWeight, qBias, kWeight, kBias, vWeight, vBias, scale, outWeight, outBias, attentionAlpha, ln1Scale, ln1Bias, ffnDense1Weight, ffnDense1Bias, ffnDense2Weight, ffnDense2Bias, ffnAlpha, ln2Scale, ln2Bias, smolgenCompressWeight, smolgenDense1Weight, smolgenDense1Bias, smolgenLn1Scale, smolgenLn1Bias, smolgenDense2Weight, smolgenDense2Bias, smolgenLn2Scale, smolgenLn2Bias, smolgenWeight };
  return { weights, buffers: Object.values(weights) };
}

function createHybridEncoderLayerSlotRuntime(device: DeviceLike, usage: Record<string, number>, weights: HybridEncoderLayerWeights, layerInput: BufferLike, encoderKernelVariant: Lc0WebEncoderKernelVariant = 'hand'): { runtime: HybridEncoderLayerSlotRuntime; buffers: BufferLike[] } {
  const outputElements = DEFAULT_TOKENS * DEFAULT_N;
  const smolgenBias = device.createBuffer({ size: DEFAULT_HEADS * DEFAULT_TOKENS * DEFAULT_TOKENS * 4, usage: usage.STORAGE | usage.COPY_DST });
  const smolgenCompressed = device.createBuffer({ size: DEFAULT_SMOLGEN_FLAT * 4, usage: usage.STORAGE });
  const smolgenDense1 = device.createBuffer({ size: DEFAULT_SMOLGEN_HIDDEN * 4, usage: usage.STORAGE });
  const smolgenLn1 = device.createBuffer({ size: DEFAULT_SMOLGEN_HIDDEN * 4, usage: usage.STORAGE });
  const smolgenDense2 = device.createBuffer({ size: DEFAULT_SMOLGEN_FLAT * 4, usage: usage.STORAGE });
  const smolgenLn2 = device.createBuffer({ size: DEFAULT_SMOLGEN_FLAT * 4, usage: usage.STORAGE });
  const qkv = device.createBuffer({ size: DEFAULT_TOKENS * DEFAULT_N * 3 * 4, usage: usage.STORAGE | usage.COPY_DST });
  const scores = device.createBuffer({ size: DEFAULT_HEADS * DEFAULT_TOKENS * DEFAULT_TOKENS * 4, usage: usage.STORAGE | usage.COPY_DST });
  const probs = device.createBuffer({ size: DEFAULT_HEADS * DEFAULT_TOKENS * DEFAULT_TOKENS * 4, usage: usage.STORAGE | usage.COPY_DST });
  const attn = device.createBuffer({ size: outputElements * 4, usage: usage.STORAGE | usage.COPY_DST });
  const attentionSkip = device.createBuffer({ size: outputElements * 4, usage: usage.STORAGE | usage.COPY_DST });
  const attentionOutput = device.createBuffer({ size: outputElements * 4, usage: usage.STORAGE | usage.COPY_DST });
  const ffnHidden = device.createBuffer({ size: DEFAULT_TOKENS * DEFAULT_FFN_HIDDEN * 4, usage: usage.STORAGE | usage.COPY_DST });
  const ffnSkip = device.createBuffer({ size: outputElements * 4, usage: usage.STORAGE | usage.COPY_DST });
  const output = device.createBuffer({ size: outputElements * 4, usage: usage.STORAGE | usage.COPY_SRC | usage.COPY_DST });
  const podArgs = encoderUsesTvmPackedF16Qkv(encoderKernelVariant) || encoderUsesTvmPackedF16AttentionOutProj(encoderKernelVariant) || encoderUsesTvmPackedF16Ffn(encoderKernelVariant) ? createU32UniformBuffer(device, [1], usage.UNIFORM | usage.COPY_DST) : undefined;
  const buffers = [smolgenBias, smolgenCompressed, smolgenDense1, smolgenLn1, smolgenDense2, smolgenLn2, qkv, scores, probs, attn, attentionSkip, attentionOutput, ffnHidden, ffnSkip, output];
  if (podArgs) buffers.push(podArgs);
  const smolgenPipelines = createSmolgenPipelines(device, {
    input: layerInput,
    compressWeight: weights.smolgenCompressWeight,
    compressed: smolgenCompressed,
    dense1Weight: weights.smolgenDense1Weight,
    dense1Bias: weights.smolgenDense1Bias,
    dense1: smolgenDense1,
    ln1Scale: weights.smolgenLn1Scale,
    ln1Bias: weights.smolgenLn1Bias,
    ln1: smolgenLn1,
    dense2Weight: weights.smolgenDense2Weight,
    dense2Bias: weights.smolgenDense2Bias,
    dense2: smolgenDense2,
    ln2Scale: weights.smolgenLn2Scale,
    ln2Bias: weights.smolgenLn2Bias,
    ln2: smolgenLn2,
    smolgenWeight: weights.smolgenWeight,
    output: smolgenBias,
  }, encoderUsesTiledSmolgenProject(encoderKernelVariant) ? 'tiled-project-f16' : 'hand');
  const attentionPipelines = createAttentionOutputPipelines(device, {
    input: layerInput, qWeight: weights.qWeight, qBias: weights.qBias, kWeight: weights.kWeight, kBias: weights.kBias, vWeight: weights.vWeight, vBias: weights.vBias, scale: weights.scale, smolgenBias, qkv, scores, probs, attn,
    outWeight: weights.outWeight, outBias: weights.outBias, alpha: weights.attentionAlpha, skip: attentionSkip, lnScale: weights.ln1Scale, lnBias: weights.ln1Bias, output: attentionOutput, podArgs,
  }, encoderUsesTvmPackedF16AttentionOutProj(encoderKernelVariant) ? 'tvm-packed-f16' : 'hand', encoderUsesTvmPackedF16Qkv(encoderKernelVariant) ? 'tvm-packed-f16' : 'hand');
  const ffnPipelines = createEncoder0FfnPipelines(device, {
    input: attentionOutput,
    dense1Weight: weights.ffnDense1Weight,
    dense1Bias: weights.ffnDense1Bias,
    hidden: ffnHidden,
    dense2Weight: weights.ffnDense2Weight,
    dense2Bias: weights.ffnDense2Bias,
    alpha: weights.ffnAlpha,
    skip: ffnSkip,
    ln2Scale: weights.ln2Scale,
    ln2Bias: weights.ln2Bias,
    output,
    podArgs,
  }, encoderUsesTvmPackedF16Ffn(encoderKernelVariant) ? 'tvm-packed-f16' : 'hand');
  return { runtime: { output, smolgenPipelines, attentionPipelines, ffnPipelines }, buffers };
}

interface SubmittedWgslHybridBatchReadbackMapState {
  startedAt: number;
  settledAt?: number;
  promise: Promise<void>;
}

interface SubmittedWgslHybridBatch {
  inputs: Lc0EvaluatorInput[];
  boardsAndFens: Array<ReturnType<typeof currentBoardAndFen>>;
  legalCandidates?: LegalPolicyCandidate[][];
  jsLegalCandidates?: LegalPolicyCandidate[][];
  legalPriorsSetupMs?: number;
  legalPriorsPrepMs?: number;
  readbackBuffer: BufferLike;
  readbackMapState: SubmittedWgslHybridBatchReadbackMapState;
  sequenceId: number;
  batchSequenceIndex?: number;
  deferredReadbackSlot?: number;
  submittedAt: number;
  totalStarted: number;
  encoderStarted: number;
  inputBuildMs: number;
  inputUploadMs: number;
  commandEncodeMs: number;
  queueSubmitMs: number;
  inputBridgeCopyMs: number;
  wasmEncodeMs: number;
  wasmTotalMs: number;
  dispatchCount: number;
}

class Lc0WebHybridRuntime {
  private readonly device: DeviceLike;
  private readonly gpuTimestampSupported: boolean;
  private readonly inputTensors: Lc0WebPreparedInitialInputTensors;
  private readonly inputBackend: Lc0WebHybridInputBackend;
  private readonly legalPriorsBackend: Lc0WebHybridLegalPriorsBackend;
  private readonly encoderKernelVariant: Lc0WebEncoderKernelVariant;
  private readonly inputBodyGpu?: InputBodyGpuRuntime;
  private readonly wasmInputEncoder?: Lc0WasmInputEncoder;
  private readonly wasmLegalPriors?: Lc0WasmLegalPriors;
  private readonly headBackend: Lc0WebHybridHeadBackend;
  private readonly wgslBatchMode: Lc0WebHybridWgslBatchMode;
  private readonly headSession?: CachedPolicyValueHeadSession;
  private readonly wgslHeads?: WgslPolicyValueHeadRuntime;
  private readonly headTensors: Lc0WebPolicyValueHeadTensors;
  private readonly usage: Record<string, number>;
  private readonly inputBuffer: BufferLike;
  private readonly readbackBuffer: BufferLike;
  private readonly layerRuntimes: HybridEncoderLayerRuntime[];
  private readonly buffers: BufferLike[];
  private wgslBatchSlots?: HybridWgslBatchSlot[];
  private wgslBatchUploadBuffer?: BufferLike;
  private wgslBatchUploadCapacity = 0;
  private wgslBatchReadbackBuffer?: BufferLike;
  private wgslBatchReadbackCapacity = 0;
  // Deferred readback can have two command buffers queued before the first map;
  // keep per-ring compute/upload resources separate from immediate batches so
  // overlap experiments do not reuse mutable intermediate buffers in flight.
  private wgslDeferredBatchSlots: HybridWgslBatchSlot[][] = [];
  private wgslDeferredBatchUploadBuffers: BufferLike[] = [];
  private wgslDeferredBatchUploadCapacities: number[] = [];
  private wgslDeferredReadbackBuffers: BufferLike[] = [];
  private wgslDeferredReadbackCapacity = 0;
  private wgslDeferredReadbackInUse = new Set<number>();
  private nextWgslSequenceId = 1;

  private constructor(options: {
    device: DeviceLike;
    gpuTimestampSupported: boolean;
    inputTensors: Lc0WebPreparedInitialInputTensors;
    inputBackend: Lc0WebHybridInputBackend;
    legalPriorsBackend: Lc0WebHybridLegalPriorsBackend;
    encoderKernelVariant: Lc0WebEncoderKernelVariant;
    inputBodyGpu?: InputBodyGpuRuntime;
    wasmInputEncoder?: Lc0WasmInputEncoder;
    wasmLegalPriors?: Lc0WasmLegalPriors;
    headBackend: Lc0WebHybridHeadBackend;
    wgslBatchMode: Lc0WebHybridWgslBatchMode;
    headSession?: CachedPolicyValueHeadSession;
    wgslHeads?: WgslPolicyValueHeadRuntime;
    headTensors: Lc0WebPolicyValueHeadTensors;
    usage: Record<string, number>;
    inputBuffer: BufferLike;
    readbackBuffer: BufferLike;
    layerRuntimes: HybridEncoderLayerRuntime[];
    buffers: BufferLike[];
    packUrl: string;
    packLoadMs: number;
    layers: number;
  }) {
    this.device = options.device;
    this.gpuTimestampSupported = options.gpuTimestampSupported;
    this.inputTensors = options.inputTensors;
    this.inputBackend = options.inputBackend;
    this.legalPriorsBackend = options.legalPriorsBackend;
    this.encoderKernelVariant = options.encoderKernelVariant;
    this.inputBodyGpu = options.inputBodyGpu;
    this.wasmInputEncoder = options.wasmInputEncoder;
    this.wasmLegalPriors = options.wasmLegalPriors;
    this.headBackend = options.headBackend;
    this.wgslBatchMode = options.wgslBatchMode;
    this.headSession = options.headSession;
    this.wgslHeads = options.wgslHeads;
    this.headTensors = options.headTensors;
    this.usage = options.usage;
    this.inputBuffer = options.inputBuffer;
    this.readbackBuffer = options.readbackBuffer;
    this.layerRuntimes = options.layerRuntimes;
    this.buffers = options.buffers;
    this.packUrl = options.packUrl;
    this.packLoadMs = options.packLoadMs;
    this.layers = options.layers;
  }

  readonly packUrl: string;
  readonly packLoadMs: number;
  readonly layers: number;

  static async create(options: Omit<Lc0WebHybridEvaluationOptions, 'input'>): Promise<Lc0WebHybridRuntime> {
    const layers = clampInteger(options.layers, 10, 1, 32);
    const prefixes = Array.from({ length: layers }, (_, layer) => `/encoder${layer}`);
    const layerTensorNames = prefixes.map((prefix) => lc0WebEncoderBlockTensorNames(prefix));
    const pack = await loadLc0WebModelPack(options.packUrl, {
      verifyShards: options.verifyShards ?? true,
      tensorNames: Array.from(new Set([
        ...inputBodyTensorNameList(),
        ...layerTensorNames.flatMap((names) => encoderBlockTensorNameList(names)),
        ...policyValueHeadTensorNameList(),
      ])),
    });
    const inputTensors = prepareInitialInputTensors(loadInitialInputTensors(pack));
    const tensorsByLayer = layerTensorNames.map((names) => loadEncoder0FfnInputs(pack, names));
    const headTensors = loadPolicyValueHeadTensors(pack);
    const headBackend = options.headBackend ?? 'ort';
    const wgslBatchMode = options.wgslBatchMode ?? 'physical';
    const inputBackend = options.inputBackend ?? 'js';
    const legalPriorsBackend = options.legalPriorsBackend ?? 'js';
    if (legalPriorsBackend === 'gpu' && headBackend !== 'wgsl') throw new Error('GPU legal-prior backend requires WGSL heads');
    const encoderKernelVariant = options.encoderKernelVariant ?? 'hand';
    const headSession = headBackend === 'ort' ? await createCachedPolicyValueHeadSession(headTensors) : undefined;
    const { device, timestampQuerySupported } = await requestDevice({ timestampQuery: options.timestampQuery });
    const usage = gpuGlobals().GPUBufferUsage!;
    const outputElements = DEFAULT_TOKENS * DEFAULT_N;
    const buffers: BufferLike[] = [];
    const inputBuffer = device.createBuffer({ size: outputElements * 4, usage: usage.STORAGE | usage.COPY_DST });
    const readbackBuffer = device.createBuffer({ size: outputElements * 4, usage: usage.MAP_READ | usage.COPY_DST });
    buffers.push(inputBuffer, readbackBuffer);
    const usesGpuInputBody = inputBackend === 'wgsl' || inputBackend === 'wasm';
    const wasmInputEncoder = inputBackend === 'wasm' ? await createLc0WasmInputEncoder() : undefined;
    const wasmLegalPriors = legalPriorsBackend === 'wasm' ? await createLc0WasmLegalPriors() : undefined;
    const inputBodyGpu = usesGpuInputBody ? createInputBodyGpuRuntime(device, inputTensors, inputBuffer, usage) : undefined;
    if (inputBodyGpu) buffers.push(...inputBodyGpu.buffers);
    const layerRuntimes: HybridEncoderLayerRuntime[] = [];
    let layerInput = inputBuffer;
    for (const tensors of tensorsByLayer) {
      const { weights, buffers: weightBuffers } = createHybridEncoderLayerWeights(device, tensors, usage);
      buffers.push(...weightBuffers);
      const { runtime, buffers: slotBuffers } = createHybridEncoderLayerSlotRuntime(device, usage, weights, layerInput, encoderKernelVariant);
      buffers.push(...slotBuffers);
      layerRuntimes.push({ ...runtime, weights });
      layerInput = runtime.output;
    }
    const wgslHeads = headBackend === 'wgsl' ? createWgslPolicyValueHeadRuntime(device, headTensors, layerRuntimes[layerRuntimes.length - 1].output, usage) : undefined;
    if (wgslHeads) buffers.push(...wgslHeads.buffers);
    return new Lc0WebHybridRuntime({
      device,
      gpuTimestampSupported: timestampQuerySupported,
      inputTensors,
      inputBackend,
      legalPriorsBackend,
      encoderKernelVariant,
      inputBodyGpu,
      wasmInputEncoder,
      wasmLegalPriors,
      headBackend,
      wgslBatchMode,
      headSession,
      wgslHeads,
      headTensors,
      usage,
      inputBuffer,
      readbackBuffer,
      layerRuntimes,
      buffers,
      packUrl: pack.manifestUrl,
      packLoadMs: pack.elapsedMs,
      layers,
    });
  }

  private createWgslBatchSlot(): HybridWgslBatchSlot {
    if (this.headBackend !== 'wgsl') throw new Error('WGSL batch slots are only available for the WGSL-head backend');
    const outputElements = DEFAULT_TOKENS * DEFAULT_N;
    const inputBuffer = this.device.createBuffer({ size: outputElements * 4, usage: this.usage.STORAGE | this.usage.COPY_DST });
    this.buffers.push(inputBuffer);
    const inputBodyGpu = this.inputBackend === 'wgsl' || this.inputBackend === 'wasm' ? createInputBodyGpuRuntime(this.device, this.inputTensors, inputBuffer, this.usage) : undefined;
    if (inputBodyGpu) this.buffers.push(...inputBodyGpu.buffers);
    const layerRuntimes: HybridEncoderLayerSlotRuntime[] = [];
    let layerInput = inputBuffer;
    for (const layer of this.layerRuntimes) {
      const { runtime, buffers } = createHybridEncoderLayerSlotRuntime(this.device, this.usage, layer.weights, layerInput, this.encoderKernelVariant);
      this.buffers.push(...buffers);
      layerRuntimes.push(runtime);
      layerInput = runtime.output;
    }
    const wgslHeads = createWgslPolicyValueHeadRuntime(this.device, this.headTensors, layerInput, this.usage);
    this.buffers.push(...wgslHeads.buffers);
    return { inputBuffer, inputBodyGpu, layerRuntimes, wgslHeads };
  }

  private ensureWgslBatchSlots(count: number): HybridWgslBatchSlot[] {
    if (this.headBackend !== 'wgsl' || !this.wgslHeads) throw new Error('WGSL batch evaluation requires the WGSL-head backend');
    if (!this.wgslBatchSlots) {
      this.wgslBatchSlots = [{ inputBuffer: this.inputBuffer, inputBodyGpu: this.inputBodyGpu, layerRuntimes: this.layerRuntimes, wgslHeads: this.wgslHeads }];
    }
    while (this.wgslBatchSlots.length < count) this.wgslBatchSlots.push(this.createWgslBatchSlot());
    return this.wgslBatchSlots.slice(0, count);
  }

  private ensureWgslDeferredBatchSlots(count: number, slot: number): HybridWgslBatchSlot[] {
    if (!this.wgslDeferredBatchSlots[slot]) this.wgslDeferredBatchSlots[slot] = [];
    const slots = this.wgslDeferredBatchSlots[slot];
    while (slots.length < count) slots.push(this.createWgslBatchSlot());
    return slots.slice(0, count);
  }

  private ensureWgslBatchUploadBuffer(count: number): BufferLike {
    const outputElements = DEFAULT_TOKENS * DEFAULT_N;
    if (!this.wgslBatchUploadBuffer || this.wgslBatchUploadCapacity < count) {
      this.wgslBatchUploadBuffer = this.device.createBuffer({ size: count * outputElements * 4, usage: this.usage.COPY_SRC | this.usage.COPY_DST });
      this.wgslBatchUploadCapacity = count;
      this.buffers.push(this.wgslBatchUploadBuffer);
    }
    return this.wgslBatchUploadBuffer;
  }

  private ensureWgslDeferredBatchUploadBuffer(count: number, slot: number): BufferLike {
    const outputElements = DEFAULT_TOKENS * DEFAULT_N;
    if (!this.wgslDeferredBatchUploadBuffers[slot] || (this.wgslDeferredBatchUploadCapacities[slot] ?? 0) < count) {
      const buffer = this.device.createBuffer({ size: count * outputElements * 4, usage: this.usage.COPY_SRC | this.usage.COPY_DST });
      this.wgslDeferredBatchUploadBuffers[slot] = buffer;
      this.wgslDeferredBatchUploadCapacities[slot] = count;
      this.buffers.push(buffer);
    }
    return this.wgslDeferredBatchUploadBuffers[slot];
  }

  private ensureWgslBatchReadbackBuffer(count: number): BufferLike {
    if (!this.wgslBatchReadbackBuffer || this.wgslBatchReadbackCapacity < count) {
      this.wgslBatchReadbackBuffer = this.device.createBuffer({ size: count * this.wgslHeadReadbackBytes(), usage: this.usage.MAP_READ | this.usage.COPY_DST });
      this.wgslBatchReadbackCapacity = count;
      this.buffers.push(this.wgslBatchReadbackBuffer);
    }
    return this.wgslBatchReadbackBuffer;
  }

  private ensureWgslDeferredReadbackBuffer(count: number, slot: number): BufferLike {
    if (this.wgslDeferredReadbackCapacity < count) {
      if (this.wgslDeferredReadbackInUse.size) throw new Error('cannot grow WGSL deferred readback buffers while a deferred readback is still in flight');
      for (const buffer of this.wgslDeferredReadbackBuffers) buffer.destroy?.();
      this.wgslDeferredReadbackBuffers = [];
      this.wgslDeferredReadbackCapacity = count;
    }
    while (this.wgslDeferredReadbackBuffers.length <= slot) {
      const buffer = this.device.createBuffer({ size: count * this.wgslHeadReadbackBytes(), usage: this.usage.MAP_READ | this.usage.COPY_DST });
      this.wgslDeferredReadbackBuffers.push(buffer);
      this.buffers.push(buffer);
    }
    return this.wgslDeferredReadbackBuffers[slot];
  }

  private buildInputPayload(input: Lc0EvaluatorInput, historyFill: Lc0HistoryFill): { payload: Float32Array<ArrayBufferLike>; wasmTiming?: Lc0WasmInputEncoderTiming } {
    if (this.inputBackend === 'js') return { payload: buildInitialEncoderActivation(input, this.inputTensors, historyFill) };
    if (this.inputBackend === 'wasm') {
      if (!this.wasmInputEncoder) throw new Error('WASM input encoder is not initialized');
      const wasm = buildInitialEncoderPlanesWasm(input, historyFill, this.wasmInputEncoder);
      return { payload: wasm.planes, wasmTiming: wasm.timing };
    }
    return { payload: buildInitialEncoderPlanes(input, historyFill) };
  }

  private timingWasmFields(wasmTiming: Lc0WasmInputEncoderTiming | undefined): Pick<Lc0WebHybridTimingBreakdown, 'inputBridgeCopyMs' | 'wasmEncodeMs' | 'wasmTotalMs'> {
    return wasmTiming ? { inputBridgeCopyMs: wasmTiming.bridgeCopyMs, wasmEncodeMs: wasmTiming.wasmEncodeMs, wasmTotalMs: wasmTiming.totalMs } : {};
  }

  private timingWasmLegalPriorsFields(wasmTiming: Lc0WasmLegalPriorTiming | undefined): Pick<Lc0WebHybridTimingBreakdown, 'legalPriorsBridgeCopyMs' | 'legalPriorsWasmRunMs' | 'legalPriorsWasmTotalMs'> {
    return wasmTiming ? { legalPriorsBridgeCopyMs: wasmTiming.bridgeCopyMs, legalPriorsWasmRunMs: wasmTiming.wasmRunMs, legalPriorsWasmTotalMs: wasmTiming.totalMs } : {};
  }

  private wgslHeadReadbackBytes(): number {
    return this.legalPriorsBackend === 'gpu' ? WGSL_GPU_LEGAL_READBACK_BYTES : WGSL_HEADS_READBACK_BYTES;
  }

  private computeLegalPriors(board: ReturnType<typeof currentBoardAndFen>['board'], fen: string, mappedPolicy: ArrayLike<number>, policyTemperature: number): { legalPriors: Lc0Evaluation['legalPriors']; bestMove?: string; legalPriorsMs: number; wasmTiming?: Lc0WasmLegalPriorTiming } {
    const legalPriorsStarted = nowMs();
    if (this.legalPriorsBackend === 'wasm') {
      if (!this.wasmLegalPriors) throw new Error('WASM legal-prior backend is not initialized');
      const result = this.wasmLegalPriors.evaluateFen(fen, mappedPolicy, { temperature: policyTemperature });
      return { legalPriors: result.legalPriors, bestMove: result.bestMove, legalPriorsMs: nowMs() - legalPriorsStarted, wasmTiming: result.timing };
    }
    const legalPriors = legalPolicyPriors(board, mappedPolicy, policyTemperature);
    return { legalPriors, bestMove: legalPriors[0]?.uci, legalPriorsMs: nowMs() - legalPriorsStarted };
  }

  async encode(input: Lc0EvaluatorInput, options: { historyFill: Lc0HistoryFill }): Promise<{
    board: ReturnType<typeof currentBoardAndFen>['board'];
    fen: string;
    output: Float32Array<ArrayBufferLike>;
    encoderDispatchSyncedMs: number;
    timing: Pick<Lc0WebHybridTimingBreakdown, 'inputBuildMs' | 'inputUploadMs' | 'commandEncodeMs' | 'queueSubmitMs' | 'readbackSyncedMs' | 'readbackBytes' | 'readbackMapCount' | 'dispatchCount' | 'inputBackend' | 'encoderKernelVariant' | 'inputBridgeCopyMs' | 'wasmEncodeMs' | 'wasmTotalMs'>;
  }> {
    const { board, fen } = currentBoardAndFen(input);
    const inputBuildStarted = nowMs();
    const { payload: inputPayload, wasmTiming } = this.buildInputPayload(input, options.historyFill);
    const inputBuildMs = nowMs() - inputBuildStarted;
    const inputUploadStarted = nowMs();
    if (this.inputBackend !== 'js') {
      if (!this.inputBodyGpu) throw new Error('WGSL/WASM input backend is not initialized');
      this.device.queue.writeBuffer(this.inputBodyGpu.planesBuffer, 0, inputPayload);
    } else {
      this.device.queue.writeBuffer(this.inputBuffer, 0, inputPayload);
    }
    const inputUploadMs = nowMs() - inputUploadStarted;
    const blockStarted = nowMs();
    const commandEncodeStarted = nowMs();
    const encoder = this.device.createCommandEncoder();
    const dispatchCounter: DispatchCounter = { count: 0 };
    if (this.inputBackend !== 'js') {
      if (!this.inputBodyGpu) throw new Error('WGSL/WASM input backend is not initialized');
      const inputPass = beginCountedComputePass(encoder, dispatchCounter);
      encodeInputBodyPass(inputPass, this.inputBodyGpu);
      inputPass.end();
    }
    for (const layer of this.layerRuntimes) {
      const pass = beginCountedComputePass(encoder, dispatchCounter);
      encodeSmolgenPass(pass, layer.smolgenPipelines);
      encodeLc0WebEncoderBlockPass(pass, layer.attentionPipelines, layer.ffnPipelines);
      pass.end();
    }
    const commandBuffer = encoder.finish();
    const commandEncodeMs = nowMs() - commandEncodeStarted;
    const queueSubmitStarted = nowMs();
    this.device.queue.submit([commandBuffer]);
    const queueSubmitMs = nowMs() - queueSubmitStarted;
    const readbackStarted = nowMs();
    const output = await readF32OutputOnce(this.device, this.layerRuntimes[this.layerRuntimes.length - 1].output, this.readbackBuffer, DEFAULT_TOKENS * DEFAULT_N);
    const readbackSyncedMs = nowMs() - readbackStarted;
    return {
      board,
      fen,
      output,
      encoderDispatchSyncedMs: nowMs() - blockStarted,
      timing: { inputBuildMs, inputUploadMs, commandEncodeMs, queueSubmitMs, readbackSyncedMs, readbackBytes: DEFAULT_TOKENS * DEFAULT_N * 4, readbackMapCount: 1, dispatchCount: dispatchCounter.count, inputBackend: this.inputBackend, encoderKernelVariant: this.encoderKernelVariant, ...this.timingWasmFields(wasmTiming) },
    };
  }

  async profileEncoder(input: Lc0EvaluatorInput, options: { historyFill: Lc0HistoryFill; iterations: number; warmup: number; profileMode?: Lc0WebHybridEncoderProfileMode }): Promise<Lc0WebHybridEncoderProfileResult> {
    const iterations = clampInteger(options.iterations, 1, 1, 100);
    const warmup = clampInteger(options.warmup, 1, 0, 20);
    const requestedProfileMode = options.profileMode ?? 'gpu-timestamp';
    const timestampReady = requestedProfileMode === 'gpu-timestamp'
      && this.gpuTimestampSupported
      && !!this.device.createQuerySet
      && !!this.usage.QUERY_RESOLVE;
    const profileMode: Lc0WebHybridEncoderProfileMode = timestampReady ? 'gpu-timestamp' : 'sync-staged';
    for (let i = 0; i < warmup; i++) await this.encode(input, { historyFill: options.historyFill });
    const aggregate = new Map<Lc0WebHybridEncoderProfileStageName, { label: string; totalMs: number; iterations: number }>();
    const byLayer = Array.from({ length: this.layerRuntimes.length }, (_, layer) => ({ layer, totalMs: 0, stages: new Map<Lc0WebHybridEncoderProfileStageName, { label: string; totalMs: number; iterations: number }>() }));
    const addAggregate = (stage: Lc0WebHybridEncoderProfileStageName, label: string, elapsedMs: number): void => {
      const entry = aggregate.get(stage) ?? { label, totalMs: 0, iterations: 0 };
      entry.totalMs += elapsedMs;
      entry.iterations += 1;
      aggregate.set(stage, entry);
    };
    const addLayer = (layerIndex: number, stage: Lc0WebHybridEncoderProfileStageName, label: string, elapsedMs: number): void => {
      const layer = byLayer[layerIndex];
      layer.totalMs += elapsedMs;
      const entry = layer.stages.get(stage) ?? { label, totalMs: 0, iterations: 0 };
      entry.totalMs += elapsedMs;
      entry.iterations += 1;
      layer.stages.set(stage, entry);
      addAggregate(stage, label, elapsedMs);
    };
    const toResult = (profiledStageTotalMs: number, readbackSyncedMs: number, output: Float32Array<ArrayBufferLike>, note: string): Lc0WebHybridEncoderProfileResult => {
      const toStageTiming = (stage: Lc0WebHybridEncoderProfileStageName, entry: { label: string; totalMs: number; iterations: number }): Lc0WebHybridEncoderProfileStageTiming => ({
        stage,
        label: entry.label,
        iterations: entry.iterations,
        totalMs: entry.totalMs,
        avgMs: entry.totalMs / Math.max(1, entry.iterations),
        percentOfProfiledStageMs: profiledStageTotalMs > 0 ? (entry.totalMs / profiledStageTotalMs) * 100 : 0,
      });
      return {
        status: 'HYBRID_ENCODER_PROFILE_DONE',
        packUrl: this.packUrl,
        layers: this.layers,
        encoderKernelVariant: this.encoderKernelVariant,
        inputBackend: this.inputBackend,
        warmup,
        iterations,
        packLoadMs: this.packLoadMs,
        profileMode,
        requestedProfileMode,
        gpuTimestampSupported: this.gpuTimestampSupported,
        profiledStageTotalMs,
        readbackSyncedMs,
        outputSample: Array.from(output.slice(0, 8)),
        aggregateStageTimings: Array.from(aggregate.entries()).map(([stage, entry]) => toStageTiming(stage, entry)).sort((a, b) => b.totalMs - a.totalMs),
        layerTimings: byLayer.map((layer) => ({
          layer: layer.layer,
          totalMs: layer.totalMs,
          stages: Array.from(layer.stages.entries()).map(([stage, entry]) => toStageTiming(stage, entry)).sort((a, b) => b.totalMs - a.totalMs),
        })),
        note,
      };
    };

    if (profileMode === 'gpu-timestamp') {
      const globals = gpuGlobals();
      const timestampStageCount = (this.inputBackend !== 'js' ? 1 : 0) + this.layerRuntimes.length * 15;
      const timestampCount = timestampStageCount * 2;
      const timestampBytes = timestampCount * 8;
      let output: Float32Array<ArrayBufferLike> = new Float32Array(DEFAULT_TOKENS * DEFAULT_N);
      for (let iteration = 0; iteration < iterations; iteration++) {
        const { payload: inputPayload } = this.buildInputPayload(input, options.historyFill);
        if (this.inputBackend !== 'js') {
          if (!this.inputBodyGpu) throw new Error('WGSL/WASM input backend is not initialized');
          this.device.queue.writeBuffer(this.inputBodyGpu.planesBuffer, 0, inputPayload);
        } else {
          this.device.queue.writeBuffer(this.inputBuffer, 0, inputPayload);
        }
        const querySet = this.device.createQuerySet!({ type: 'timestamp', count: timestampCount });
        const resolveBuffer = this.device.createBuffer({ size: timestampBytes, usage: this.usage.QUERY_RESOLVE | this.usage.COPY_SRC });
        const timestampReadback = this.device.createBuffer({ size: timestampBytes, usage: this.usage.MAP_READ | this.usage.COPY_DST });
        const records: Array<{ stage: Lc0WebHybridEncoderProfileStageName; label: string; layerIndex?: number; begin: number; end: number }> = [];
        let queryIndex = 0;
        try {
          const encoder = this.device.createCommandEncoder();
          const encodeTimestampStage = (stage: Lc0WebHybridEncoderProfileStageName, label: string, layerIndex: number | undefined, encode: (pass: ComputePassLike) => void): void => {
            const begin = queryIndex++;
            const end = queryIndex++;
            records.push({ stage, label, layerIndex, begin, end });
            const pass = encoder.beginComputePass({ timestampWrites: { querySet, beginningOfPassWriteIndex: begin, endOfPassWriteIndex: end } });
            encode(pass);
            pass.end();
          };
          if (this.inputBackend !== 'js') encodeTimestampStage('inputBody', 'input body projection', undefined, (pass) => encodeInputBodyPass(pass, this.inputBodyGpu!));
          for (let layerIndex = 0; layerIndex < this.layerRuntimes.length; layerIndex++) {
            const layer = this.layerRuntimes[layerIndex];
            encodeTimestampStage('smolgenCompress', 'smolgen compress', layerIndex, (pass) => encodeSmolgenCompressPass(pass, layer.smolgenPipelines));
            encodeTimestampStage('smolgenDense1', 'smolgen dense1', layerIndex, (pass) => encodeSmolgenDense1Pass(pass, layer.smolgenPipelines));
            encodeTimestampStage('smolgenLn1', 'smolgen ln1', layerIndex, (pass) => encodeSmolgenLn1Pass(pass, layer.smolgenPipelines));
            encodeTimestampStage('smolgenDense2', 'smolgen dense2', layerIndex, (pass) => encodeSmolgenDense2Pass(pass, layer.smolgenPipelines));
            encodeTimestampStage('smolgenLn2', 'smolgen ln2', layerIndex, (pass) => encodeSmolgenLn2Pass(pass, layer.smolgenPipelines));
            encodeTimestampStage('smolgenProject', 'smolgen project', layerIndex, (pass) => encodeSmolgenProjectPass(pass, layer.smolgenPipelines));
            encodeTimestampStage('qkvProjection', 'QKV projection', layerIndex, (pass) => encodeAttentionQkvPass(pass, layer.attentionPipelines));
            encodeTimestampStage('attentionScores', 'attention scores', layerIndex, (pass) => encodeAttentionScoresPass(pass, layer.attentionPipelines));
            encodeTimestampStage('softmax', 'softmax', layerIndex, (pass) => encodeAttentionSoftmaxPass(pass, layer.attentionPipelines));
            encodeTimestampStage('attentionValue', 'attention value', layerIndex, (pass) => encodeAttentionValuePass(pass, layer.attentionPipelines));
            encodeTimestampStage('outputProjection', 'attention output projection', layerIndex, (pass) => encodeAttentionOutputProjectionPass(pass, layer.attentionPipelines));
            encodeTimestampStage('ln1', 'attention ln1', layerIndex, (pass) => encodeAttentionNormPass(pass, layer.attentionPipelines));
            encodeTimestampStage('ffnDense1', 'FFN dense1', layerIndex, (pass) => encodeFfnDense1Pass(pass, layer.ffnPipelines));
            encodeTimestampStage('ffnDense2Residual', 'FFN dense2 + residual', layerIndex, (pass) => encodeFfnDense2ResidualPass(pass, layer.ffnPipelines));
            encodeTimestampStage('ln2', 'FFN ln2', layerIndex, (pass) => encodeFfnLn2Pass(pass, layer.ffnPipelines));
          }
          if (!encoder.resolveQuerySet) throw new Error('WebGPU timestamp query resolve is unavailable');
          encoder.resolveQuerySet(querySet, 0, queryIndex, resolveBuffer, 0);
          encoder.copyBufferToBuffer(resolveBuffer, 0, timestampReadback, 0, timestampBytes);
          this.device.queue.submit([encoder.finish()]);
          await timestampReadback.mapAsync(globals.GPUMapMode!.READ);
          const timestamps = new BigUint64Array(timestampReadback.getMappedRange().slice(0));
          timestampReadback.unmap();
          for (const record of records) {
            const elapsedNs = timestamps[record.end] > timestamps[record.begin] ? timestamps[record.end] - timestamps[record.begin] : 0n;
            const elapsedMs = Number(elapsedNs) / 1_000_000;
            if (record.layerIndex === undefined) addAggregate(record.stage, record.label, elapsedMs);
            else addLayer(record.layerIndex, record.stage, record.label, elapsedMs);
          }
        } finally {
          querySet.destroy?.();
          resolveBuffer.destroy?.();
          timestampReadback.destroy?.();
        }
      }
      const readbackStarted = nowMs();
      output = await readF32OutputOnce(this.device, this.layerRuntimes[this.layerRuntimes.length - 1].output, this.readbackBuffer, DEFAULT_TOKENS * DEFAULT_N);
      const readbackSyncedMs = nowMs() - readbackStarted;
      const profiledStageTotalMs = Array.from(aggregate.values()).reduce((sum, entry) => sum + entry.totalMs, 0);
      return toResult(profiledStageTotalMs, readbackSyncedMs, output, 'GPU timestamp profile encodes the full encoder as one command buffer per iteration and reads timestamp queries once, avoiding per-stage queue waits. Pass boundaries are still inserted around stages, so use this for lower-perturbation attribution rather than exact route latency.');
    }

    const submitAndMeasure = async (encode: (pass: ComputePassLike) => void): Promise<number> => {
      const encoder = this.device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      encode(pass);
      pass.end();
      const started = nowMs();
      this.device.queue.submit([encoder.finish()]);
      await this.device.queue.onSubmittedWorkDone?.();
      return nowMs() - started;
    };
    if (warmup > 0) {
      const { payload: inputPayload } = this.buildInputPayload(input, options.historyFill);
      if (this.inputBackend !== 'js') {
        if (!this.inputBodyGpu) throw new Error('WGSL/WASM input backend is not initialized');
        this.device.queue.writeBuffer(this.inputBodyGpu.planesBuffer, 0, inputPayload);
        await submitAndMeasure((pass) => encodeInputBodyPass(pass, this.inputBodyGpu!));
      } else {
        this.device.queue.writeBuffer(this.inputBuffer, 0, inputPayload);
      }
      for (const layer of this.layerRuntimes) {
        await submitAndMeasure((pass) => encodeSmolgenPass(pass, layer.smolgenPipelines));
        await submitAndMeasure((pass) => encodeAttentionQkvPass(pass, layer.attentionPipelines));
        await submitAndMeasure((pass) => encodeAttentionScoresPass(pass, layer.attentionPipelines));
        await submitAndMeasure((pass) => encodeAttentionSoftmaxPass(pass, layer.attentionPipelines));
        await submitAndMeasure((pass) => encodeAttentionValuePass(pass, layer.attentionPipelines));
        await submitAndMeasure((pass) => encodeAttentionOutputProjectionPass(pass, layer.attentionPipelines));
        await submitAndMeasure((pass) => encodeAttentionNormPass(pass, layer.attentionPipelines));
        await submitAndMeasure((pass) => encodeFfnDense1Pass(pass, layer.ffnPipelines));
        await submitAndMeasure((pass) => encodeFfnDense2ResidualPass(pass, layer.ffnPipelines));
        await submitAndMeasure((pass) => encodeFfnLn2Pass(pass, layer.ffnPipelines));
      }
      await readF32OutputOnce(this.device, this.layerRuntimes[this.layerRuntimes.length - 1].output, this.readbackBuffer, DEFAULT_TOKENS * DEFAULT_N);
    }
    let readbackSyncedMs = 0;
    let output: Float32Array<ArrayBufferLike> = new Float32Array(DEFAULT_TOKENS * DEFAULT_N);
    for (let iteration = 0; iteration < iterations; iteration++) {
      const { payload: inputPayload } = this.buildInputPayload(input, options.historyFill);
      if (this.inputBackend !== 'js') {
        if (!this.inputBodyGpu) throw new Error('WGSL/WASM input backend is not initialized');
        this.device.queue.writeBuffer(this.inputBodyGpu.planesBuffer, 0, inputPayload);
        const elapsedMs = await submitAndMeasure((pass) => encodeInputBodyPass(pass, this.inputBodyGpu!));
        addAggregate('inputBody', 'input body projection', elapsedMs);
      } else {
        this.device.queue.writeBuffer(this.inputBuffer, 0, inputPayload);
      }
      for (let layerIndex = 0; layerIndex < this.layerRuntimes.length; layerIndex++) {
        const layer = this.layerRuntimes[layerIndex];
        addLayer(layerIndex, 'smolgenCompress', 'smolgen compress', await submitAndMeasure((pass) => encodeSmolgenCompressPass(pass, layer.smolgenPipelines)));
        addLayer(layerIndex, 'smolgenDense1', 'smolgen dense1', await submitAndMeasure((pass) => encodeSmolgenDense1Pass(pass, layer.smolgenPipelines)));
        addLayer(layerIndex, 'smolgenLn1', 'smolgen ln1', await submitAndMeasure((pass) => encodeSmolgenLn1Pass(pass, layer.smolgenPipelines)));
        addLayer(layerIndex, 'smolgenDense2', 'smolgen dense2', await submitAndMeasure((pass) => encodeSmolgenDense2Pass(pass, layer.smolgenPipelines)));
        addLayer(layerIndex, 'smolgenLn2', 'smolgen ln2', await submitAndMeasure((pass) => encodeSmolgenLn2Pass(pass, layer.smolgenPipelines)));
        addLayer(layerIndex, 'smolgenProject', 'smolgen project', await submitAndMeasure((pass) => encodeSmolgenProjectPass(pass, layer.smolgenPipelines)));
        addLayer(layerIndex, 'qkvProjection', 'QKV projection', await submitAndMeasure((pass) => encodeAttentionQkvPass(pass, layer.attentionPipelines)));
        addLayer(layerIndex, 'attentionScores', 'attention scores', await submitAndMeasure((pass) => encodeAttentionScoresPass(pass, layer.attentionPipelines)));
        addLayer(layerIndex, 'softmax', 'softmax', await submitAndMeasure((pass) => encodeAttentionSoftmaxPass(pass, layer.attentionPipelines)));
        addLayer(layerIndex, 'attentionValue', 'attention value', await submitAndMeasure((pass) => encodeAttentionValuePass(pass, layer.attentionPipelines)));
        addLayer(layerIndex, 'outputProjection', 'attention output projection', await submitAndMeasure((pass) => encodeAttentionOutputProjectionPass(pass, layer.attentionPipelines)));
        addLayer(layerIndex, 'ln1', 'attention ln1', await submitAndMeasure((pass) => encodeAttentionNormPass(pass, layer.attentionPipelines)));
        addLayer(layerIndex, 'ffnDense1', 'FFN dense1', await submitAndMeasure((pass) => encodeFfnDense1Pass(pass, layer.ffnPipelines)));
        addLayer(layerIndex, 'ffnDense2Residual', 'FFN dense2 + residual', await submitAndMeasure((pass) => encodeFfnDense2ResidualPass(pass, layer.ffnPipelines)));
        addLayer(layerIndex, 'ln2', 'FFN ln2', await submitAndMeasure((pass) => encodeFfnLn2Pass(pass, layer.ffnPipelines)));
      }
      const readbackStarted = nowMs();
      output = await readF32OutputOnce(this.device, this.layerRuntimes[this.layerRuntimes.length - 1].output, this.readbackBuffer, DEFAULT_TOKENS * DEFAULT_N);
      readbackSyncedMs += nowMs() - readbackStarted;
    }
    const profiledStageTotalMs = Array.from(aggregate.values()).reduce((sum, entry) => sum + entry.totalMs, 0);
    return toResult(profiledStageTotalMs, readbackSyncedMs, output, 'Sync-staged profiling intentionally submits and waits after each stage, so totals are sync-perturbed and should be used for relative attribution rather than direct route latency.');
  }

  async evaluate(input: Lc0EvaluatorInput, options: { historyFill: Lc0HistoryFill; policyTemperature: number }): Promise<Lc0WebHybridEvaluationResult> {
    const totalStarted = nowMs();
    if (this.headBackend === 'wgsl') {
      if (!this.wgslHeads) throw new Error('WGSL hybrid heads runtime is not initialized');
      const { board, fen } = currentBoardAndFen(input);
      const inputBuildStarted = nowMs();
      const { payload: inputPayload, wasmTiming } = this.buildInputPayload(input, options.historyFill);
      const inputBuildMs = nowMs() - inputBuildStarted;
      const inputUploadStarted = nowMs();
      if (this.inputBackend !== 'js') {
        if (!this.inputBodyGpu) throw new Error('WGSL/WASM input backend is not initialized');
        this.device.queue.writeBuffer(this.inputBodyGpu.planesBuffer, 0, inputPayload);
      } else {
        this.device.queue.writeBuffer(this.inputBuffer, 0, inputPayload);
      }
      const inputUploadMs = nowMs() - inputUploadStarted;
      const gpuLegalSetupStarted = nowMs();
      const gpuLegalCandidates = this.legalPriorsBackend === 'gpu' ? legalPolicyCandidates(board) : undefined;
      if (gpuLegalCandidates) uploadWgslLegalPriorsInputs(this.device, this.wgslHeads, gpuLegalCandidates, options.policyTemperature);
      const legalPriorsGpuSetupMs = gpuLegalCandidates ? nowMs() - gpuLegalSetupStarted : undefined;
      const encoderStarted = nowMs();
      const commandEncodeStarted = nowMs();
      const encoder = this.device.createCommandEncoder();
      const dispatchCounter: DispatchCounter = { count: 0 };
      if (this.inputBackend !== 'js') {
        if (!this.inputBodyGpu) throw new Error('WGSL/WASM input backend is not initialized');
        const inputPass = beginCountedComputePass(encoder, dispatchCounter);
        encodeInputBodyPass(inputPass, this.inputBodyGpu);
        inputPass.end();
      }
      for (const layer of this.layerRuntimes) {
        const pass = beginCountedComputePass(encoder, dispatchCounter);
        encodeSmolgenPass(pass, layer.smolgenPipelines);
        encodeLc0WebEncoderBlockPass(pass, layer.attentionPipelines, layer.ffnPipelines);
        pass.end();
      }
      const headPass = beginCountedComputePass(encoder, dispatchCounter);
      encodeWgslPolicyValueHeads(headPass, this.wgslHeads);
      if (gpuLegalCandidates) encodeWgslLegalPriors(headPass, this.wgslHeads);
      headPass.end();
      if (gpuLegalCandidates) copyWgslLegalPriorsOutputs(encoder, this.wgslHeads);
      else copyWgslPolicyValueHeadOutputs(encoder, this.wgslHeads);
      const commandBuffer = encoder.finish();
      const commandEncodeMs = nowMs() - commandEncodeStarted;
      const queueSubmitStarted = nowMs();
      this.device.queue.submit([commandBuffer]);
      const queueSubmitMs = nowMs() - queueSubmitStarted;
      const headStarted = nowMs();
      const gpuLegalOutput = gpuLegalCandidates ? await mapWgslLegalPriorsOutputs(this.wgslHeads, gpuLegalCandidates) : undefined;
      const fullHeadOutput = gpuLegalOutput ? undefined : await mapWgslPolicyValueHeadOutputs(this.wgslHeads);
      const headRunMs = nowMs() - headStarted;
      const encoderDispatchSyncedMs = nowMs() - encoderStarted;
      const mappedPolicyF32 = fullHeadOutput?.mappedPolicy;
      const wdlF32 = gpuLegalOutput?.wdl ?? fullHeadOutput!.wdl;
      if (mappedPolicyF32 && !arrayHasNonzeroAndVariation(mappedPolicyF32)) throw new Error('WGSL hybrid heads produced zero or uniform mapped policy');
      if (!arrayHasNonzeroAndVariation(wdlF32)) throw new Error('WGSL hybrid heads produced zero or uniform WDL');
      const mappedPolicy = mappedPolicyF32 ? Array.from(mappedPolicyF32) : [];
      const wdlValues = Array.from(wdlF32);
      const wdl: [number, number, number] = [Number(wdlValues[0]), Number(wdlValues[1]), Number(wdlValues[2])];
      const legalPriorsStarted = nowMs();
      const legalPriorsResult = gpuLegalOutput
        ? { legalPriors: gpuLegalOutput.legalPriors, bestMove: gpuLegalOutput.legalPriors[0]?.uci, legalPriorsMs: (legalPriorsGpuSetupMs ?? 0) + (nowMs() - legalPriorsStarted), wasmTiming: undefined }
        : this.computeLegalPriors(board, fen, mappedPolicy, options.policyTemperature);
      const { legalPriors, bestMove, legalPriorsMs, wasmTiming: legalPriorsWasmTiming } = legalPriorsResult;
      return {
        status: 'LC0WEB_HYBRID_EVALUATION_DONE',
        backend: 'lc0web-wgsl-encoder-wgsl-heads',
        packUrl: this.packUrl,
        layers: this.layers,
        encoderKernelVariant: this.encoderKernelVariant,
        packLoadMs: this.packLoadMs,
        encoderDispatchSyncedMs,
        headRunMs,
        fen,
        wdl,
        q: wdl[0] - wdl[2],
        mlh: 0,
        legalPriors,
        bestMove,
        mappedPolicy,
        timing: {
          totalEvalMs: nowMs() - totalStarted,
          inputBuildMs,
          inputUploadMs,
          commandEncodeMs,
          queueSubmitMs,
          readbackSyncedMs: gpuLegalOutput?.readbackSyncedMs ?? fullHeadOutput!.readbackSyncedMs,
          headRunMs,
          legalPriorsMs,
          readbackBytes: gpuLegalOutput ? WGSL_GPU_LEGAL_READBACK_BYTES : WGSL_HEADS_READBACK_BYTES,
          readbackMapCount: 1,
          dispatchCount: dispatchCounter.count,
          inputBackend: this.inputBackend,
          legalPriorsBackend: this.legalPriorsBackend,
          encoderKernelVariant: this.encoderKernelVariant,
          ...this.timingWasmFields(wasmTiming),
          ...this.timingWasmLegalPriorsFields(legalPriorsWasmTiming),
        },
      };
    }
    if (!this.headSession) throw new Error('ORT hybrid heads session is not initialized');
    const encoded = await this.encode(input, { historyFill: options.historyFill });
    const heads = await runCachedPolicyValueHeadsOrt(encoded.output, this.headSession);
    const wdl: [number, number, number] = [Number(heads.wdl[0]), Number(heads.wdl[1]), Number(heads.wdl[2])];
    const { legalPriors, bestMove, legalPriorsMs, wasmTiming: legalPriorsWasmTiming } = this.computeLegalPriors(encoded.board, encoded.fen, heads.mappedPolicy, options.policyTemperature);
    return {
      status: 'LC0WEB_HYBRID_EVALUATION_DONE',
      backend: 'lc0web-wgsl-encoder-ort-heads',
      packUrl: this.packUrl,
      layers: this.layers,
      encoderKernelVariant: this.encoderKernelVariant,
      packLoadMs: this.packLoadMs,
      encoderDispatchSyncedMs: encoded.encoderDispatchSyncedMs,
      headRunMs: heads.runMs,
      fen: encoded.fen,
      wdl,
      q: wdl[0] - wdl[2],
      mlh: 0,
      legalPriors,
      bestMove,
      mappedPolicy: heads.mappedPolicy,
      timing: {
        totalEvalMs: nowMs() - totalStarted,
        inputBuildMs: encoded.timing.inputBuildMs,
        inputUploadMs: encoded.timing.inputUploadMs,
        commandEncodeMs: encoded.timing.commandEncodeMs,
        queueSubmitMs: encoded.timing.queueSubmitMs,
        readbackSyncedMs: encoded.timing.readbackSyncedMs,
        headRunMs: heads.runMs,
        legalPriorsMs,
        readbackBytes: encoded.timing.readbackBytes,
        readbackMapCount: encoded.timing.readbackMapCount,
        dispatchCount: encoded.timing.dispatchCount,
        inputBackend: this.inputBackend,
        legalPriorsBackend: this.legalPriorsBackend,
        encoderKernelVariant: this.encoderKernelVariant,
        inputBridgeCopyMs: encoded.timing.inputBridgeCopyMs,
        wasmEncodeMs: encoded.timing.wasmEncodeMs,
        wasmTotalMs: encoded.timing.wasmTotalMs,
        ...this.timingWasmLegalPriorsFields(legalPriorsWasmTiming),
      },
    };
  }

  private async submitWgslBatch(inputs: Lc0EvaluatorInput[], options: { historyFill: Lc0HistoryFill; policyTemperature: number }, readbackBuffer: BufferLike, sequenceOptions: { batchSequenceIndex?: number; deferredReadbackSlot?: number } = {}): Promise<SubmittedWgslHybridBatch> {
    const totalStarted = nowMs();
    const slots = sequenceOptions.deferredReadbackSlot === undefined
      ? this.ensureWgslBatchSlots(inputs.length)
      : this.ensureWgslDeferredBatchSlots(inputs.length, sequenceOptions.deferredReadbackSlot);
    const uploadBuffer = sequenceOptions.deferredReadbackSlot === undefined
      ? this.ensureWgslBatchUploadBuffer(inputs.length)
      : this.ensureWgslDeferredBatchUploadBuffer(inputs.length, sequenceOptions.deferredReadbackSlot);
    const outputElements = DEFAULT_TOKENS * DEFAULT_N;
    const inputElements = this.inputBackend !== 'js' ? DEFAULT_INPUT_PLANES * DEFAULT_TOKENS : outputElements;
    const boardsAndFens = inputs.map((input) => currentBoardAndFen(input));
    const legalPriorsSetupStarted = nowMs();
    const legalCandidates = this.legalPriorsBackend === 'gpu' ? boardsAndFens.map(({ board }) => legalPolicyCandidates(board)) : undefined;
    let legalPriorsSetupMs = legalCandidates ? nowMs() - legalPriorsSetupStarted : undefined;
    const inputBuildStarted = nowMs();
    const combinedInput = new Float32Array(inputs.length * inputElements);
    let inputBridgeCopyMs = 0;
    let wasmEncodeMs = 0;
    let wasmTotalMs = 0;
    for (let i = 0; i < inputs.length; i++) {
      const { payload, wasmTiming } = this.buildInputPayload(inputs[i], options.historyFill);
      if (wasmTiming) {
        inputBridgeCopyMs += wasmTiming.bridgeCopyMs;
        wasmEncodeMs += wasmTiming.wasmEncodeMs;
        wasmTotalMs += wasmTiming.totalMs;
      }
      combinedInput.set(payload, i * inputElements);
    }
    const inputBuildMs = nowMs() - inputBuildStarted;
    const inputUploadStarted = nowMs();
    this.device.queue.writeBuffer(uploadBuffer, 0, combinedInput);
    const inputUploadMs = nowMs() - inputUploadStarted;
    const encoderStarted = nowMs();
    const commandEncodeStarted = nowMs();
    const encoder = this.device.createCommandEncoder();
    const dispatchCounter: DispatchCounter = { count: 0 };
    for (let slotIndex = 0; slotIndex < slots.length; slotIndex++) {
      const slot = slots[slotIndex];
      if (this.inputBackend !== 'js') {
        const inputBodyGpu = slot.inputBodyGpu;
        if (!inputBodyGpu) throw new Error('WGSL/WASM input backend batch slot is not initialized');
        encoder.copyBufferToBuffer(uploadBuffer, slotIndex * inputElements * 4, inputBodyGpu.planesBuffer, 0, inputElements * 4);
        const inputPass = beginCountedComputePass(encoder, dispatchCounter);
        encodeInputBodyPass(inputPass, inputBodyGpu);
        inputPass.end();
      } else {
        encoder.copyBufferToBuffer(uploadBuffer, slotIndex * inputElements * 4, slot.inputBuffer, 0, inputElements * 4);
      }
      for (const layer of slot.layerRuntimes) {
        const pass = beginCountedComputePass(encoder, dispatchCounter);
        encodeSmolgenPass(pass, layer.smolgenPipelines);
        encodeLc0WebEncoderBlockPass(pass, layer.attentionPipelines, layer.ffnPipelines);
        pass.end();
      }
      if (legalCandidates) {
        const legalUploadStarted = nowMs();
        uploadWgslLegalPriorsInputs(this.device, slots[slotIndex].wgslHeads, legalCandidates[slotIndex], options.policyTemperature);
        legalPriorsSetupMs = (legalPriorsSetupMs ?? 0) + (nowMs() - legalUploadStarted);
      }
      const headPass = beginCountedComputePass(encoder, dispatchCounter);
      encodeWgslPolicyValueHeads(headPass, slots[slotIndex].wgslHeads);
      if (legalCandidates) encodeWgslLegalPriors(headPass, slots[slotIndex].wgslHeads);
      headPass.end();
      const readbackOffset = slotIndex * this.wgslHeadReadbackBytes();
      if (legalCandidates) copyWgslLegalPriorsOutputsTo(encoder, slots[slotIndex].wgslHeads, readbackBuffer, readbackOffset);
      else copyWgslPolicyValueHeadOutputsTo(encoder, slots[slotIndex].wgslHeads, readbackBuffer, readbackOffset);
    }
    const commandBuffer = encoder.finish();
    const commandEncodeMs = nowMs() - commandEncodeStarted;
    const queueSubmitStarted = nowMs();
    this.device.queue.submit([commandBuffer]);
    const queueSubmitMs = nowMs() - queueSubmitStarted;
    const submittedAt = nowMs();
    const readbackMapState: SubmittedWgslHybridBatchReadbackMapState = { startedAt: nowMs(), promise: Promise.resolve() };
    readbackMapState.promise = readbackBuffer.mapAsync(gpuGlobals().GPUMapMode!.READ).then(() => { readbackMapState.settledAt = nowMs(); });
    let jsLegalCandidates: LegalPolicyCandidate[][] | undefined;
    let legalPriorsPrepMs: number | undefined;
    try {
      if (this.legalPriorsBackend === 'js') {
        const legalPriorsPrepStarted = nowMs();
        jsLegalCandidates = boardsAndFens.map(({ board }) => legalPolicyCandidates(board));
        legalPriorsPrepMs = nowMs() - legalPriorsPrepStarted;
      }
    } catch (error) {
      await this.cleanupStartedReadbackMap(readbackBuffer, readbackMapState);
      throw error;
    }
    return {
      inputs,
      boardsAndFens,
      legalCandidates,
      jsLegalCandidates,
      legalPriorsSetupMs,
      legalPriorsPrepMs,
      readbackBuffer,
      readbackMapState,
      sequenceId: this.nextWgslSequenceId++,
      batchSequenceIndex: sequenceOptions.batchSequenceIndex,
      deferredReadbackSlot: sequenceOptions.deferredReadbackSlot,
      submittedAt,
      totalStarted,
      encoderStarted,
      inputBuildMs,
      inputUploadMs,
      commandEncodeMs,
      queueSubmitMs,
      inputBridgeCopyMs,
      wasmEncodeMs,
      wasmTotalMs,
      dispatchCount: dispatchCounter.count,
    };
  }

  private async cleanupStartedReadbackMap(readbackBuffer: BufferLike, readbackMapState: SubmittedWgslHybridBatchReadbackMapState): Promise<void> {
    try {
      await readbackMapState.promise;
      readbackBuffer.unmap();
    } catch {
      // The caller is already propagating the primary error. If mapAsync also
      // fails, there is no mapped range to release.
    }
  }

  private async finishWgslBatch(submitted: SubmittedWgslHybridBatch, options: { policyTemperature: number }, readbackMode: 'immediate' | 'deferred-double-buffered'): Promise<Lc0WebHybridEvaluationResult[]> {
    let readbackMapped = false;
    try {
      const headStarted = nowMs();
      const readbackStarted = nowMs();
      const deferredReadbackDelayMs = Math.max(0, submitted.readbackMapState.startedAt - submitted.submittedAt);
      const mapAsyncAwaitStarted = nowMs();
      await submitted.readbackMapState.promise;
      readbackMapped = true;
      const readbackMapAsyncWaitMs = nowMs() - mapAsyncAwaitStarted;
      const readbackMapAsyncMs = Math.max(0, (submitted.readbackMapState.settledAt ?? nowMs()) - submitted.readbackMapState.startedAt);
      const readbackBytesPerSlot = this.wgslHeadReadbackBytes();
      const mapCopyStarted = nowMs();
      const readbackRange = submitted.readbackBuffer.getMappedRange();
      const readbackFloats = new Float32Array(readbackRange, 0, submitted.inputs.length * (readbackBytesPerSlot / 4));
      const readbackMapCopyMs = nowMs() - mapCopyStarted;
      const readbackSyncedMs = nowMs() - readbackStarted;
      const headRunMs = nowMs() - headStarted;
      const encoderDispatchSyncedMs = nowMs() - submitted.encoderStarted;
      const totalEvalMs = nowMs() - submitted.totalStarted;
      const out: Lc0WebHybridEvaluationResult[] = [];
      for (let i = 0; i < submitted.inputs.length; i++) {
        const base = i * (readbackBytesPerSlot / 4);
        const gpuLegalCandidates = submitted.legalCandidates?.[i];
        const mappedPolicyF32 = gpuLegalCandidates ? undefined : readbackFloats.slice(base, base + DEFAULT_POLICY_MAPPED_OUTPUTS);
        const wdlF32 = gpuLegalCandidates
          ? readbackFloats.slice(base + WGSL_GPU_LEGAL_OUTPUT_FLOATS, base + WGSL_GPU_LEGAL_OUTPUT_FLOATS + 3)
          : readbackFloats.slice(base + DEFAULT_POLICY_MAPPED_OUTPUTS, base + DEFAULT_POLICY_MAPPED_OUTPUTS + 3);
        const diagnosticFen = submitted.boardsAndFens[i].fen;
        const diagnosticBatchFens = submitted.boardsAndFens.map((entry) => entry.fen).join(' | ');
        const diagnosticSuffix = `slot ${i}, mode ${readbackMode}, sequence ${submitted.sequenceId}, batch ${submitted.batchSequenceIndex ?? 'single'}, position ${submitted.deferredReadbackSlot ?? 'immediate'}, fen ${diagnosticFen}, batchFens ${diagnosticBatchFens}`;
        if (mappedPolicyF32 && !arrayHasNonzeroAndVariation(mappedPolicyF32)) throw new Error(`WGSL hybrid batch heads produced zero or uniform mapped policy for ${diagnosticSuffix}`);
        if (!arrayHasNonzeroAndVariation(wdlF32)) throw new Error(`WGSL hybrid batch heads produced zero or uniform WDL for ${diagnosticSuffix}`);
        const mappedPolicy = mappedPolicyF32 ? Array.from(mappedPolicyF32) : [];
        const wdlValues = Array.from(wdlF32);
        const wdl: [number, number, number] = [Number(wdlValues[0]), Number(wdlValues[1]), Number(wdlValues[2])];
        const legalPriorsStarted = nowMs();
        const legalPriorsSetupMs = gpuLegalCandidates ? (submitted.legalPriorsSetupMs ?? 0) / submitted.inputs.length : 0;
        const legalPriorsPrepMs = submitted.jsLegalCandidates ? (submitted.legalPriorsPrepMs ?? 0) / submitted.inputs.length : 0;
        const readbackOverlapHiddenMs = submitted.jsLegalCandidates ? Math.min(submitted.legalPriorsPrepMs ?? 0, readbackMapAsyncMs) / submitted.inputs.length : 0;
        const jsLegalCandidates = submitted.jsLegalCandidates?.[i];
        const legalPriorsResult = gpuLegalCandidates
          ? { legalPriors: legalPriorsFromGpuOutput(gpuLegalCandidates, readbackFloats.slice(base, base + WGSL_GPU_LEGAL_OUTPUT_FLOATS)), bestMove: undefined as string | undefined, legalPriorsMs: legalPriorsSetupMs + (nowMs() - legalPriorsStarted), wasmTiming: undefined }
          : jsLegalCandidates
            ? (() => {
                const legalPriors = legalPolicyPriorsFromCandidates(jsLegalCandidates, mappedPolicy, options.policyTemperature);
                return { legalPriors, bestMove: legalPriors[0]?.uci, legalPriorsMs: legalPriorsPrepMs + (nowMs() - legalPriorsStarted), wasmTiming: undefined };
              })()
            : this.computeLegalPriors(submitted.boardsAndFens[i].board, submitted.boardsAndFens[i].fen, mappedPolicy, options.policyTemperature);
        const legalPriors = legalPriorsResult.legalPriors;
        const bestMove = gpuLegalCandidates ? legalPriors[0]?.uci : legalPriorsResult.bestMove;
        const { legalPriorsMs, wasmTiming: legalPriorsWasmTiming } = legalPriorsResult;
        out.push({
          status: 'LC0WEB_HYBRID_EVALUATION_DONE',
          backend: 'lc0web-wgsl-encoder-wgsl-heads',
          packUrl: this.packUrl,
          layers: this.layers,
          encoderKernelVariant: this.encoderKernelVariant,
          packLoadMs: this.packLoadMs,
          encoderDispatchSyncedMs,
          headRunMs,
          fen: submitted.boardsAndFens[i].fen,
          wdl,
          q: wdl[0] - wdl[2],
          mlh: 0,
          legalPriors,
          bestMove,
          mappedPolicy,
          timing: {
            totalEvalMs,
            inputBuildMs: submitted.inputBuildMs,
            inputUploadMs: submitted.inputUploadMs,
            commandEncodeMs: submitted.commandEncodeMs,
            queueSubmitMs: submitted.queueSubmitMs,
            readbackSyncedMs,
            headRunMs,
            legalPriorsMs,
            readbackBytes: submitted.inputs.length * readbackBytesPerSlot,
            readbackMapCount: 1,
            dispatchCount: submitted.dispatchCount,
            readbackMode,
            wgslSequenceId: submitted.sequenceId,
            batchSequenceIndex: submitted.batchSequenceIndex,
            deferredReadbackSlot: submitted.deferredReadbackSlot,
            deferredReadbackDelayMs,
            readbackMapAsyncMs,
            readbackMapAsyncWaitMs,
            readbackMapCopyMs,
            ...(submitted.legalPriorsPrepMs !== undefined ? { legalPriorsPrepMs, readbackOverlapCpuMs: legalPriorsPrepMs, readbackOverlapHiddenMs } : {}),
            physicalBatchSize: submitted.inputs.length,
            batchPosition: i,
            inputBackend: this.inputBackend,
            legalPriorsBackend: this.legalPriorsBackend,
            encoderKernelVariant: this.encoderKernelVariant,
            ...(this.inputBackend === 'wasm' ? { inputBridgeCopyMs: submitted.inputBridgeCopyMs, wasmEncodeMs: submitted.wasmEncodeMs, wasmTotalMs: submitted.wasmTotalMs } : {}),
            ...this.timingWasmLegalPriorsFields(legalPriorsWasmTiming),
          },
        });
      }
      return out;
    } finally {
      if (readbackMapped) {
        try { submitted.readbackBuffer.unmap(); }
        catch { /* best-effort cleanup after a failed mapped-range read */ }
      }
      if (submitted.deferredReadbackSlot !== undefined) this.wgslDeferredReadbackInUse.delete(submitted.deferredReadbackSlot);
    }
  }

  async evaluateBatch(inputs: Lc0EvaluatorInput[], options: { historyFill: Lc0HistoryFill; policyTemperature: number }): Promise<Lc0WebHybridEvaluationResult[]> {
    if (inputs.length === 0) return [];
    if (this.headBackend !== 'wgsl' || this.wgslBatchMode === 'serial') {
      const out: Lc0WebHybridEvaluationResult[] = [];
      for (const input of inputs) out.push(await this.evaluate(input, options));
      return out;
    }
    const readbackBuffer = this.ensureWgslBatchReadbackBuffer(inputs.length);
    const submitted = await this.submitWgslBatch(inputs, options, readbackBuffer);
    return this.finishWgslBatch(submitted, options, 'immediate');
  }

  async evaluateWgslBatchesDeferredReadback(batches: Lc0EvaluatorInput[][], options: { historyFill: Lc0HistoryFill; policyTemperature: number }): Promise<Lc0WebHybridEvaluationResult[][]> {
    if (this.headBackend !== 'wgsl') throw new Error('deferred readback benchmark requires the WGSL-head backend');
    const out: Lc0WebHybridEvaluationResult[][] = new Array(batches.length);
    const maxBatchLength = Math.max(0, ...batches.map((batch) => batch.length));
    if (maxBatchLength > 0) {
      for (const slot of [0, 1]) {
        this.ensureWgslDeferredBatchSlots(maxBatchLength, slot);
        this.ensureWgslDeferredBatchUploadBuffer(maxBatchLength, slot);
        this.ensureWgslDeferredReadbackBuffer(maxBatchLength, slot);
      }
    }
    let pending: { index: number; submitted: SubmittedWgslHybridBatch } | undefined;
    for (let i = 0; i < batches.length; i++) {
      const batch = batches[i];
      if (!batch.length) {
        if (pending) {
          out[pending.index] = await this.finishWgslBatch(pending.submitted, options, 'deferred-double-buffered');
          pending = undefined;
        }
        out[i] = [];
        continue;
      }
      if (pending && this.wgslDeferredReadbackCapacity < batch.length) {
        // Growing the readback buffers requires destroying/replacing the ring;
        // finish the pending map first so no in-flight result points at it.
        out[pending.index] = await this.finishWgslBatch(pending.submitted, options, 'deferred-double-buffered');
        pending = undefined;
      }
      const deferredReadbackSlot = i % 2;
      if (this.wgslDeferredReadbackInUse.has(deferredReadbackSlot)) throw new Error(`WGSL deferred readback slot ${deferredReadbackSlot} reused before its prior result was mapped`);
      const readbackBuffer = this.ensureWgslDeferredReadbackBuffer(batch.length, deferredReadbackSlot);
      this.wgslDeferredReadbackInUse.add(deferredReadbackSlot);
      let submitted: SubmittedWgslHybridBatch;
      try {
        submitted = await this.submitWgslBatch(batch, options, readbackBuffer, { batchSequenceIndex: i, deferredReadbackSlot });
      } catch (error) {
        this.wgslDeferredReadbackInUse.delete(deferredReadbackSlot);
        throw error;
      }
      if (pending) {
        try {
          out[pending.index] = await this.finishWgslBatch(pending.submitted, options, 'deferred-double-buffered');
        } catch (error) {
          this.wgslDeferredReadbackInUse.delete(deferredReadbackSlot);
          await this.cleanupStartedReadbackMap(readbackBuffer, submitted.readbackMapState);
          throw error;
        }
      }
      pending = { index: i, submitted };
    }
    if (pending) out[pending.index] = await this.finishWgslBatch(pending.submitted, options, 'deferred-double-buffered');
    return out.map((batch) => batch ?? []);
  }

  destroy(): void {
    for (const buffer of this.buffers) buffer.destroy?.();
  }
}

export async function runLc0WebHybridEvaluation(options: Lc0WebHybridEvaluationOptions): Promise<Lc0WebHybridEvaluationResult> {
  const runtime = await Lc0WebHybridRuntime.create(options);
  try {
    return await runtime.evaluate(options.input, {
      historyFill: options.historyFill ?? 'fen_only',
      policyTemperature: options.policyTemperature ?? LC0_DEFAULT_POLICY_TEMPERATURE,
    });
  } finally {
    runtime.destroy();
  }
}

export async function runLc0WebHybridEncoderProfile(options: Lc0WebHybridEncoderProfileOptions): Promise<Lc0WebHybridEncoderProfileResult> {
  const runtime = await Lc0WebHybridRuntime.create({
    packUrl: options.packUrl,
    layers: options.layers,
    verifyShards: options.verifyShards,
    inputBackend: options.inputBackend,
    encoderKernelVariant: options.encoderKernelVariant,
    headBackend: 'ort',
    timestampQuery: options.profileMode !== 'sync-staged',
  });
  try {
    return await runtime.profileEncoder(options.input, {
      historyFill: options.historyFill ?? 'fen_only',
      iterations: options.iterations ?? 1,
      warmup: options.warmup ?? 1,
      profileMode: options.profileMode ?? 'gpu-timestamp',
    });
  } finally {
    runtime.destroy();
  }
}

export interface Lc0WebWgslDeferredReadbackBenchModeResult {
  mode: 'immediate' | 'deferred-double-buffered';
  wallMs: number;
  batches: number;
  evaluations: number;
  evalsPerSecond: number;
  timingMeans: Partial<Record<keyof Lc0WebHybridTimingBreakdown, number>>;
  bestMoves: Array<string | undefined>;
}

export interface Lc0WebWgslDeferredReadbackBenchResult {
  status: 'WGSL_DEFERRED_READBACK_BENCH_DONE';
  backend: 'lc0web-wgsl-encoder-wgsl-heads';
  stableBackend: 'lc0web-wgsl-encoder-ort-heads';
  packUrl: string;
  layers: number;
  inputBackend: Lc0WebHybridInputBackend;
  legalPriorsBackend: Lc0WebHybridLegalPriorsBackend;
  batchSize: number;
  iterations: number;
  warmup: number;
  inputCount: number;
  immediate: Lc0WebWgslDeferredReadbackBenchModeResult;
  deferred: Lc0WebWgslDeferredReadbackBenchModeResult;
  allBestMovesMatch: boolean;
}

function mean(values: number[]): number {
  return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : 0;
}

function summarizeHybridEvaluations(mode: 'immediate' | 'deferred-double-buffered', wallMs: number, batches: Lc0WebHybridEvaluationResult[][]): Lc0WebWgslDeferredReadbackBenchModeResult {
  const flat = batches.flat();
  const timingKeys: Array<keyof Lc0WebHybridTimingBreakdown> = ['totalEvalMs', 'inputBuildMs', 'inputUploadMs', 'commandEncodeMs', 'queueSubmitMs', 'readbackSyncedMs', 'readbackMapAsyncMs', 'readbackMapAsyncWaitMs', 'readbackMapCopyMs', 'deferredReadbackDelayMs', 'headRunMs', 'legalPriorsMs', 'legalPriorsPrepMs', 'readbackOverlapCpuMs', 'readbackOverlapHiddenMs', 'legalPriorsBridgeCopyMs', 'legalPriorsWasmRunMs', 'legalPriorsWasmTotalMs', 'readbackBytes', 'readbackMapCount', 'dispatchCount'];
  const physicalBatchScopedKeys = new Set<keyof Lc0WebHybridTimingBreakdown>(['totalEvalMs', 'inputBuildMs', 'inputUploadMs', 'commandEncodeMs', 'queueSubmitMs', 'readbackSyncedMs', 'readbackMapAsyncMs', 'readbackMapAsyncWaitMs', 'readbackMapCopyMs', 'deferredReadbackDelayMs', 'headRunMs', 'readbackBytes', 'readbackMapCount', 'dispatchCount']);
  const timingMeans: Partial<Record<keyof Lc0WebHybridTimingBreakdown, number>> = {};
  for (const key of timingKeys) {
    const values = flat.map((entry) => {
      const value = entry.timing[key];
      if (typeof value !== 'number' || !Number.isFinite(value)) return undefined;
      const physicalBatchSize = Math.max(1, Math.floor(Number(entry.timing.physicalBatchSize ?? 1)));
      return physicalBatchScopedKeys.has(key) ? value / physicalBatchSize : value;
    }).filter((value): value is number => typeof value === 'number' && Number.isFinite(value));
    if (values.length) timingMeans[key] = mean(values);
  }
  return {
    mode,
    wallMs,
    batches: batches.length,
    evaluations: flat.length,
    evalsPerSecond: flat.length / Math.max(1e-9, wallMs / 1000),
    timingMeans,
    bestMoves: flat.map((entry) => entry.bestMove),
  };
}

function buildRoundRobinBatches(inputs: Lc0EvaluatorInput[], batchSize: number, batches: number): Lc0EvaluatorInput[][] {
  if (!inputs.length) throw new Error('deferred readback benchmark requires at least one input');
  return Array.from({ length: batches }, (_, batchIndex) => Array.from({ length: batchSize }, (_, slot) => inputs[(batchIndex * batchSize + slot) % inputs.length]));
}

export async function runLc0WebWgslDeferredReadbackBenchmark(options: {
  packUrl: string;
  inputs: Lc0EvaluatorInput[];
  layers?: number;
  verifyShards?: boolean;
  historyFill?: Lc0HistoryFill;
  policyTemperature?: number;
  inputBackend?: Lc0WebHybridInputBackend;
  legalPriorsBackend?: Lc0WebHybridLegalPriorsBackend;
  batchSize?: number;
  iterations?: number;
  warmup?: number;
}): Promise<Lc0WebWgslDeferredReadbackBenchResult> {
  const batchSize = clampInteger(options.batchSize, 4, 1, 64);
  const iterations = clampInteger(options.iterations, 4, 1, 100);
  const warmup = clampInteger(options.warmup, 1, 0, 20);
  const runtime = await Lc0WebHybridRuntime.create({
    packUrl: options.packUrl,
    layers: options.layers,
    verifyShards: options.verifyShards,
    historyFill: options.historyFill,
    policyTemperature: options.policyTemperature,
    headBackend: 'wgsl',
    wgslBatchMode: 'physical',
    inputBackend: options.inputBackend ?? 'js',
    legalPriorsBackend: options.legalPriorsBackend ?? 'js',
  });
  try {
    const runImmediate = async (batches: Lc0EvaluatorInput[][]): Promise<Lc0WebHybridEvaluationResult[][]> => {
      const out: Lc0WebHybridEvaluationResult[][] = [];
      for (const batch of batches) out.push(await runtime.evaluateBatch(batch, { historyFill: options.historyFill ?? 'fen_only', policyTemperature: options.policyTemperature ?? LC0_DEFAULT_POLICY_TEMPERATURE }));
      return out;
    };
    if (warmup) await runImmediate(buildRoundRobinBatches(options.inputs, batchSize, warmup));
    const immediateBatches = buildRoundRobinBatches(options.inputs, batchSize, iterations);
    let started = nowMs();
    const immediateResults = await runImmediate(immediateBatches);
    const immediate = summarizeHybridEvaluations('immediate', nowMs() - started, immediateResults);

    if (warmup) await runtime.evaluateWgslBatchesDeferredReadback(buildRoundRobinBatches(options.inputs, batchSize, warmup), { historyFill: options.historyFill ?? 'fen_only', policyTemperature: options.policyTemperature ?? LC0_DEFAULT_POLICY_TEMPERATURE });
    const deferredBatches = buildRoundRobinBatches(options.inputs, batchSize, iterations);
    started = nowMs();
    const deferredResults = await runtime.evaluateWgslBatchesDeferredReadback(deferredBatches, { historyFill: options.historyFill ?? 'fen_only', policyTemperature: options.policyTemperature ?? LC0_DEFAULT_POLICY_TEMPERATURE });
    const deferred = summarizeHybridEvaluations('deferred-double-buffered', nowMs() - started, deferredResults);

    return {
      status: 'WGSL_DEFERRED_READBACK_BENCH_DONE',
      backend: 'lc0web-wgsl-encoder-wgsl-heads',
      stableBackend: 'lc0web-wgsl-encoder-ort-heads',
      packUrl: runtime.packUrl,
      layers: runtime.layers,
      inputBackend: options.inputBackend ?? 'js',
      legalPriorsBackend: options.legalPriorsBackend ?? 'js',
      batchSize,
      iterations,
      warmup,
      inputCount: options.inputs.length,
      immediate,
      deferred,
      allBestMovesMatch: immediate.bestMoves.length === deferred.bestMoves.length && immediate.bestMoves.every((move, i) => move === deferred.bestMoves[i]),
    };
  } finally {
    runtime.destroy();
  }
}

export class Lc0WebHybridEvaluator {
  readonly packUrl: string;
  readonly layers: number;
  readonly historyFill: Lc0HistoryFill;
  readonly policyTemperature: number;
  readonly verifyShards: boolean;
  readonly headBackend: Lc0WebHybridHeadBackend;
  readonly wgslBatchMode: Lc0WebHybridWgslBatchMode;
  readonly inputBackend: Lc0WebHybridInputBackend;
  readonly legalPriorsBackend: Lc0WebHybridLegalPriorsBackend;
  readonly encoderKernelVariant: Lc0WebEncoderKernelVariant;
  private runtimePromise?: Promise<Lc0WebHybridRuntime>;
  private evaluationQueue: Promise<void> = Promise.resolve();

  constructor(options: Omit<Lc0WebHybridEvaluationOptions, 'input'>) {
    this.packUrl = options.packUrl;
    this.layers = clampInteger(options.layers, 10, 1, 32);
    this.historyFill = options.historyFill ?? 'fen_only';
    this.policyTemperature = options.policyTemperature ?? LC0_DEFAULT_POLICY_TEMPERATURE;
    this.verifyShards = options.verifyShards ?? true;
    this.headBackend = options.headBackend ?? 'ort';
    this.wgslBatchMode = options.wgslBatchMode ?? 'physical';
    this.inputBackend = options.inputBackend ?? 'js';
    this.legalPriorsBackend = options.legalPriorsBackend ?? 'js';
    if (this.legalPriorsBackend === 'gpu' && this.headBackend !== 'wgsl') throw new Error('GPU legal-prior backend requires WGSL heads');
    this.encoderKernelVariant = options.encoderKernelVariant ?? 'hand';
  }

  private runtime(): Promise<Lc0WebHybridRuntime> {
    if (!this.runtimePromise) {
      const runtimePromise = Lc0WebHybridRuntime.create({
        packUrl: this.packUrl,
        layers: this.layers,
        historyFill: this.historyFill,
        policyTemperature: this.policyTemperature,
        verifyShards: this.verifyShards,
        headBackend: this.headBackend,
        wgslBatchMode: this.wgslBatchMode,
        inputBackend: this.inputBackend,
        legalPriorsBackend: this.legalPriorsBackend,
        encoderKernelVariant: this.encoderKernelVariant,
      });
      runtimePromise.catch(() => {
        if (this.runtimePromise === runtimePromise) this.runtimePromise = undefined;
      });
      this.runtimePromise = runtimePromise;
    }
    return this.runtimePromise;
  }

  private enqueueEvaluation<T>(work: () => Promise<T>): Promise<T> {
    const run = this.evaluationQueue.then(work, work);
    this.evaluationQueue = run.then(() => undefined, () => undefined);
    return run;
  }

  async evaluate(input: Lc0EvaluatorInput): Promise<Lc0Evaluation> {
    return this.enqueueEvaluation(async () => (await this.runtime()).evaluate(input, {
      historyFill: this.historyFill,
      policyTemperature: this.policyTemperature,
    }));
  }

  async evaluateBatch(inputs: Lc0EvaluatorInput[]): Promise<Lc0Evaluation[]> {
    return this.enqueueEvaluation(async () => {
      const runtime = await this.runtime();
      return runtime.evaluateBatch(inputs, {
        historyFill: this.historyFill,
        policyTemperature: this.policyTemperature,
      });
    });
  }

  async evaluateBatchSequence(batches: Lc0EvaluatorInput[][]): Promise<Lc0Evaluation[][]> {
    return this.enqueueEvaluation(async () => {
      const runtime = await this.runtime();
      const options = { historyFill: this.historyFill, policyTemperature: this.policyTemperature };
      if (this.headBackend === 'wgsl' && this.wgslBatchMode === 'physical' && batches.length > 1) return runtime.evaluateWgslBatchesDeferredReadback(batches, options);
      const out: Lc0Evaluation[][] = [];
      for (const batch of batches) out.push(await runtime.evaluateBatch(batch, options));
      return out;
    });
  }

  async dispose(): Promise<void> {
    const runtimePromise = this.runtimePromise;
    this.runtimePromise = undefined;
    if (!runtimePromise) return;
    const runtime = await runtimePromise.catch(() => undefined);
    runtime?.destroy();
  }
}

export interface Lc0WebWgslHeadsVsOrtFixtureInput {
  id: string;
  input: Lc0EvaluatorInput;
}

export interface Lc0WebWgslHeadsVsOrtFixtureResult {
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
}

export interface Lc0WebWgslHeadsVsOrtFixturesResult {
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
  evaluations: Lc0WebWgslHeadsVsOrtFixtureResult[];
}

export async function runLc0WebWgslHeadsVsOrtFixtures(options: {
  packUrl: string;
  fixtures: Lc0WebWgslHeadsVsOrtFixtureInput[];
  layers?: number;
  verifyShards?: boolean;
  historyFill?: Lc0HistoryFill;
  policyTemperature?: number;
  mappedPolicyTolerance?: number;
  wdlTolerance?: number;
}): Promise<Lc0WebWgslHeadsVsOrtFixturesResult> {
  const layers = clampInteger(options.layers, 10, 1, 32);
  const mappedPolicyTolerance = options.mappedPolicyTolerance ?? 1e-3;
  const wdlTolerance = options.wdlTolerance ?? 1e-3;
  const policyTemperature = options.policyTemperature ?? LC0_DEFAULT_POLICY_TEMPERATURE;
  const runtime = await Lc0WebHybridRuntime.create({
    packUrl: options.packUrl,
    layers,
    verifyShards: options.verifyShards,
    historyFill: options.historyFill,
    policyTemperature,
  });
  try {
    const evaluations: Lc0WebWgslHeadsVsOrtFixtureResult[] = [];
    for (const fixture of options.fixtures) {
      const encoded = await runtime.encode(fixture.input, { historyFill: options.historyFill ?? 'fen_only' });
      const heads = await runLc0WebWgslHeadsProbe({
        packUrl: options.packUrl,
        verifyShards: options.verifyShards,
        input: encoded.output,
        includeOutputs: true,
      });
      if (!heads.mappedPolicy || !heads.ortHeads.mappedPolicy) throw new Error('WGSL heads fixture comparison missing full mapped-policy outputs');
      const wgslMappedPolicy = new Float32Array(heads.mappedPolicy);
      const ortMappedPolicy = new Float32Array(heads.ortHeads.mappedPolicy);
      const wgslWdl = new Float32Array(heads.wgslWdl);
      const ortWdl = new Float32Array(heads.ortHeads.wdl);
      if (!arrayHasNonzero(wgslMappedPolicy) || !arrayHasVariation(wgslMappedPolicy) || !arrayHasNonzero(wgslWdl) || !arrayHasVariation(wgslWdl)) {
        throw new Error(`WGSL heads fixture ${fixture.id} produced zero or uniform mapped policy/WDL`);
      }
      const mappedPolicyDiff = computeErrorStats(wgslMappedPolicy, ortMappedPolicy, DEFAULT_POLICY_MAPPED_OUTPUTS);
      const wdlDiff = computeErrorStats(wgslWdl, ortWdl, 3);
      const wgslPriors = legalPolicyPriors(encoded.board, heads.mappedPolicy, policyTemperature);
      const ortPriors = legalPolicyPriors(encoded.board, heads.ortHeads.mappedPolicy, policyTemperature);
      const result: Lc0WebWgslHeadsVsOrtFixtureResult = {
        id: fixture.id,
        fen: encoded.fen,
        encoderDispatchSyncedMs: encoded.encoderDispatchSyncedMs,
        wgslDispatchSyncedMs: heads.dispatchSyncedMs,
        wgslReadbackSyncedMs: heads.readbackSyncedMs,
        ortRunMs: heads.ortHeads.runMs,
        mappedPolicyMaxAbsDiff: mappedPolicyDiff.maxAbsError,
        mappedPolicyRmsDiff: mappedPolicyDiff.rmsError,
        wdlMaxAbsDiff: wdlDiff.maxAbsError,
        wdlRmsDiff: wdlDiff.rmsError,
        wgslBestMove: wgslPriors[0]?.uci,
        ortBestMove: ortPriors[0]?.uci,
        bestMoveMatch: wgslPriors[0]?.uci === ortPriors[0]?.uci,
        wgslWdl: heads.wgslWdl,
        ortWdl: heads.ortHeads.wdl,
        wgslMappedPolicySample: heads.mappedPolicySample,
        ortMappedPolicySample: heads.ortHeads.mappedPolicySample,
      };
      if (mappedPolicyDiff.maxAbsError > mappedPolicyTolerance) throw new Error(`WGSL heads fixture ${fixture.id} mapped-policy maxAbsDiff=${mappedPolicyDiff.maxAbsError}, tolerance=${mappedPolicyTolerance}`);
      if (wdlDiff.maxAbsError > wdlTolerance) throw new Error(`WGSL heads fixture ${fixture.id} WDL maxAbsDiff=${wdlDiff.maxAbsError}, tolerance=${wdlTolerance}`);
      if (!result.bestMoveMatch) throw new Error(`WGSL heads fixture ${fixture.id} best move ${result.wgslBestMove} != ORT ${result.ortBestMove}`);
      evaluations.push(result);
    }
    return {
      status: 'WGSL_HEADS_VS_ORT_FIXTURES_DONE',
      backend: 'lc0web-wgsl-encoder-wgsl-heads-probe',
      stableBackend: 'lc0web-wgsl-encoder-ort-heads',
      packUrl: options.packUrl,
      layers,
      fixtures: evaluations.length,
      mappedPolicyTolerance,
      wdlTolerance,
      bestMoveMatches: evaluations.filter((entry) => entry.bestMoveMatch).length,
      maxMappedPolicyAbsDiff: evaluations.reduce((max, entry) => Math.max(max, entry.mappedPolicyMaxAbsDiff), 0),
      maxWdlAbsDiff: evaluations.reduce((max, entry) => Math.max(max, entry.wdlMaxAbsDiff), 0),
      evaluations,
    };
  } finally {
    runtime.destroy();
  }
}

export async function runLc0WebEncoder0BlockOrtBenchmark(options: Lc0WebEncoder0BlockBenchmarkOptions): Promise<Lc0WebEncoder0BlockOrtBenchmarkResult> {
  const warmup = clampInteger(options.warmup, 3, 0, 100);
  const iterations = clampInteger(options.iterations, 10, 1, 1000);
  const tensorNames = lc0WebEncoderBlockTensorNames(options.encoderPrefix);
  const pack = await loadLc0WebModelPack(options.packUrl, {
    verifyShards: options.verifyShards ?? true,
    tensorNames: encoderBlockTensorNameList(tensorNames),
  });
  const tensors = loadEncoder0FfnInputs(pack, tensorNames);
  const attentionValue = buildAttentionValueReference(tensors);
  const reference = buildEncoder0BlockReference(tensors);
  const outputElements = DEFAULT_TOKENS * DEFAULT_N;
  const modelBuildStarted = nowMs();
  const tinyOnnx = createTinyEncoder0BlockOnnxForTest(
    f16BytesToF32Array(tensors.outWeight.bytes, DEFAULT_N * DEFAULT_N),
    f16BytesToF32Array(tensors.outBias.bytes, DEFAULT_N),
    reference.attentionAlpha,
    f16BytesToF32Array(tensors.lnScale.bytes, DEFAULT_N),
    f16BytesToF32Array(tensors.lnBias.bytes, DEFAULT_N),
    f16BytesToF32Array(tensors.ffnDense1Weight.bytes, DEFAULT_N * DEFAULT_FFN_HIDDEN),
    f16BytesToF32Array(tensors.ffnDense1Bias.bytes, DEFAULT_FFN_HIDDEN),
    f16BytesToF32Array(tensors.ffnDense2Weight.bytes, DEFAULT_FFN_HIDDEN * DEFAULT_N),
    f16BytesToF32Array(tensors.ffnDense2Bias.bytes, DEFAULT_N),
    reference.ffnAlpha,
    f16BytesToF32Array(tensors.ln2Scale.bytes, DEFAULT_N),
    f16BytesToF32Array(tensors.ln2Bias.bytes, DEFAULT_N),
  );
  const modelBuildMs = nowMs() - modelBuildStarted;
  const sessionStarted = nowMs();
  const session = await ort.createOrtSession(tinyOnnx);
  const sessionCreateMs = nowMs() - sessionStarted;
  const feeds = {
    attention: new ort.Tensor('float32', attentionValue.output, [DEFAULT_TOKENS, DEFAULT_N]),
    residual: new ort.Tensor('float32', reference.input, [DEFAULT_TOKENS, DEFAULT_N]),
  };
  let output: Float32Array<ArrayBufferLike> = new Float32Array(outputElements);
  for (let i = 0; i < warmup; i++) {
    const outputs = await session.run(feeds);
    output = outputs.output.data as Float32Array<ArrayBufferLike>;
  }
  const times: number[] = [];
  for (let i = 0; i < iterations; i++) {
    const started = nowMs();
    const outputs = await session.run(feeds);
    times.push(nowMs() - started);
    output = outputs.output.data as Float32Array<ArrayBufferLike>;
  }
  const { maxAbsError, rmsError } = computeErrorStats(output, reference.output, outputElements);
  assertErrorInTolerance(maxAbsError);
  const avgMs = times.reduce((sum, value) => sum + value, 0) / times.length;
  const ortDiagnostics = await ort.collectOrtRuntimeDiagnostics();
  return {
    status: 'ENCODER0_BLOCK_ORT_BENCH_DONE',
    packUrl: pack.manifestUrl,
    modelName: pack.manifest.model.name,
    tokens: DEFAULT_TOKENS,
    channels: DEFAULT_N,
    heads: DEFAULT_HEADS,
    headDim: DEFAULT_HEAD_DIM,
    ffnHidden: DEFAULT_FFN_HIDDEN,
    lnEpsilon: DEFAULT_LN_EPSILON,
    attentionAlpha: reference.attentionAlpha,
    ffnAlpha: reference.ffnAlpha,
    smolgen: { enabled: true, epsilon: DEFAULT_SMOLGEN_EPSILON },
    warmup,
    iterations,
    packLoadMs: pack.elapsedMs,
    modelBuildMs,
    sessionCreateMs,
    avgMs,
    minMs: Math.min(...times),
    maxMs: Math.max(...times),
    firstMs: times[0],
    timesMs: times,
    runsPerSecond: 1000 / avgMs,
    ortDiagnostics: ortDiagnostics as unknown as Record<string, unknown>,
    maxAbsError,
    rmsError,
    outputSample: Array.from(output.slice(0, 8)),
  };
}

export async function runLc0WebEncoder0BlockBenchmark(options: Lc0WebEncoder0BlockBenchmarkOptions): Promise<Lc0WebEncoder0BlockBenchmarkResult> {
  const totalStarted = nowMs();
  const warmup = clampInteger(options.warmup, 1, 0, 1000);
  const iterations = clampInteger(options.iterations, 5, 1, 10_000);
  const { device, adapterInfo, timestampQuerySupported } = await requestDevice({ timestampQuery: true });
  const tensorNames = lc0WebEncoderBlockTensorNames(options.encoderPrefix);
  const pack = await loadLc0WebModelPack(options.packUrl, {
    verifyShards: options.verifyShards ?? true,
    tensorNames: encoderBlockTensorNameList(tensorNames),
  });
  const tensors = loadEncoder0FfnInputs(pack, tensorNames);
  const reference = buildEncoder0BlockReference(tensors);
  const outputElements = DEFAULT_TOKENS * DEFAULT_N;
  const globals = gpuGlobals();
  const usage = globals.GPUBufferUsage!;
  const buffers: BufferLike[] = [];
  try {
    const setupStarted = nowMs();
    const input = createStorageBuffer(device, reference.input, usage.STORAGE | usage.COPY_DST);
    const qWeight = createTransposedF16StorageBuffer(device, tensors.qWeight.bytes, DEFAULT_K, DEFAULT_N, usage.STORAGE | usage.COPY_DST);
    const qBias = createStorageBuffer(device, tensors.qBias.bytes, usage.STORAGE | usage.COPY_DST);
    const kWeight = createTransposedF16StorageBuffer(device, tensors.kWeight.bytes, DEFAULT_K, DEFAULT_N, usage.STORAGE | usage.COPY_DST);
    const kBias = createStorageBuffer(device, tensors.kBias.bytes, usage.STORAGE | usage.COPY_DST);
    const vWeight = createTransposedF16StorageBuffer(device, tensors.vWeight.bytes, DEFAULT_K, DEFAULT_N, usage.STORAGE | usage.COPY_DST);
    const vBias = createStorageBuffer(device, tensors.vBias.bytes, usage.STORAGE | usage.COPY_DST);
    const scale = createStorageBuffer(device, paddedF16ScalarBytes(tensors.scale.bytes), usage.STORAGE | usage.COPY_DST);
    const smolgenBias = createStorageBuffer(device, reference.smolgenBias, usage.STORAGE | usage.COPY_DST);
    const outWeight = createTransposedF16StorageBuffer(device, tensors.outWeight.bytes, DEFAULT_N, DEFAULT_N, usage.STORAGE | usage.COPY_DST);
    const outBias = createStorageBuffer(device, tensors.outBias.bytes, usage.STORAGE | usage.COPY_DST);
    const attentionAlpha = createStorageBuffer(device, paddedF16ScalarBytes(tensors.alpha.bytes), usage.STORAGE | usage.COPY_DST);
    const ln1Scale = createStorageBuffer(device, tensors.lnScale.bytes, usage.STORAGE | usage.COPY_DST);
    const ln1Bias = createStorageBuffer(device, tensors.lnBias.bytes, usage.STORAGE | usage.COPY_DST);
    const ffnDense1Weight = createTransposedF16StorageBuffer(device, tensors.ffnDense1Weight.bytes, DEFAULT_N, DEFAULT_FFN_HIDDEN, usage.STORAGE | usage.COPY_DST);
    const ffnDense1Bias = createStorageBuffer(device, tensors.ffnDense1Bias.bytes, usage.STORAGE | usage.COPY_DST);
    const ffnDense2Weight = createTransposedF16StorageBuffer(device, tensors.ffnDense2Weight.bytes, DEFAULT_FFN_HIDDEN, DEFAULT_N, usage.STORAGE | usage.COPY_DST);
    const ffnDense2Bias = createStorageBuffer(device, tensors.ffnDense2Bias.bytes, usage.STORAGE | usage.COPY_DST);
    const ffnAlpha = createStorageBuffer(device, paddedF16ScalarBytes(tensors.ffnAlpha.bytes), usage.STORAGE | usage.COPY_DST);
    const ln2Scale = createStorageBuffer(device, tensors.ln2Scale.bytes, usage.STORAGE | usage.COPY_DST);
    const ln2Bias = createStorageBuffer(device, tensors.ln2Bias.bytes, usage.STORAGE | usage.COPY_DST);
    const qkv = device.createBuffer({ size: DEFAULT_TOKENS * DEFAULT_N * 3 * 4, usage: usage.STORAGE });
    const scores = device.createBuffer({ size: DEFAULT_HEADS * DEFAULT_TOKENS * DEFAULT_TOKENS * 4, usage: usage.STORAGE });
    const probs = device.createBuffer({ size: DEFAULT_HEADS * DEFAULT_TOKENS * DEFAULT_TOKENS * 4, usage: usage.STORAGE });
    const attn = device.createBuffer({ size: outputElements * 4, usage: usage.STORAGE });
    const attentionSkip = device.createBuffer({ size: outputElements * 4, usage: usage.STORAGE });
    const attentionOutput = device.createBuffer({ size: outputElements * 4, usage: usage.STORAGE });
    const ffnHidden = device.createBuffer({ size: DEFAULT_TOKENS * DEFAULT_FFN_HIDDEN * 4, usage: usage.STORAGE });
    const ffnSkip = device.createBuffer({ size: outputElements * 4, usage: usage.STORAGE });
    const output = device.createBuffer({ size: outputElements * 4, usage: usage.STORAGE | usage.COPY_SRC });
    const readback = device.createBuffer({ size: outputElements * 4, usage: usage.MAP_READ | usage.COPY_DST });
    buffers.push(input, qWeight, qBias, kWeight, kBias, vWeight, vBias, scale, smolgenBias, outWeight, outBias, attentionAlpha, ln1Scale, ln1Bias, ffnDense1Weight, ffnDense1Bias, ffnDense2Weight, ffnDense2Bias, ffnAlpha, ln2Scale, ln2Bias, qkv, scores, probs, attn, attentionSkip, attentionOutput, ffnHidden, ffnSkip, output, readback);
    const attentionPipelines = createAttentionOutputPipelines(device, {
      input, qWeight, qBias, kWeight, kBias, vWeight, vBias, scale, smolgenBias, qkv, scores, probs, attn,
      outWeight, outBias, alpha: attentionAlpha, skip: attentionSkip, lnScale: ln1Scale, lnBias: ln1Bias, output: attentionOutput,
    });
    const ffnPipelines = createEncoder0FfnPipelines(device, {
      input: attentionOutput,
      dense1Weight: ffnDense1Weight,
      dense1Bias: ffnDense1Bias,
      hidden: ffnHidden,
      dense2Weight: ffnDense2Weight,
      dense2Bias: ffnDense2Bias,
      alpha: ffnAlpha,
      skip: ffnSkip,
      ln2Scale,
      ln2Bias,
      output,
    });
    const uploadSetupMs = nowMs() - setupStarted;
    const encodePipelineStage = (pipeline: PipelineLike, bindGroup: unknown, dispatch: (pass: ComputePassLike) => void, count: number): unknown => {
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      for (let i = 0; i < count; i++) {
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        dispatch(pass);
      }
      pass.end();
      return encoder.finish();
    };
    const encodeQkv = (count: number) => encodePipelineStage(
      attentionPipelines.qkv,
      attentionPipelines.qkvBind,
      (pass) => pass.dispatchWorkgroups(Math.ceil(DEFAULT_N / 8), Math.ceil(DEFAULT_TOKENS / 8)),
      count,
    );
    const encodeAttentionScores = (count: number) => encodePipelineStage(
      attentionPipelines.score,
      attentionPipelines.scoreBind,
      (pass) => pass.dispatchWorkgroups(Math.ceil(DEFAULT_TOKENS / 8), Math.ceil(DEFAULT_TOKENS / 8), DEFAULT_HEADS),
      count,
    );
    const encodeSoftmax = (count: number) => encodePipelineStage(
      attentionPipelines.softmax,
      attentionPipelines.softmaxBind,
      (pass) => pass.dispatchWorkgroups(DEFAULT_HEADS * DEFAULT_TOKENS),
      count,
    );
    const encodeAttentionValue = (count: number) => encodePipelineStage(
      attentionPipelines.value,
      attentionPipelines.valueBind,
      (pass) => pass.dispatchWorkgroups(Math.ceil(DEFAULT_HEAD_DIM / 8), Math.ceil(DEFAULT_TOKENS / 8), DEFAULT_HEADS),
      count,
    );
    const encodeOutputProjectionLn1 = (count: number): unknown => {
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      for (let i = 0; i < count; i++) {
        pass.setPipeline(attentionPipelines.outProj);
        pass.setBindGroup(0, attentionPipelines.outProjBind);
        pass.dispatchWorkgroups(Math.ceil(DEFAULT_N / 8), Math.ceil(DEFAULT_TOKENS / 8));
        pass.setPipeline(attentionPipelines.norm);
        pass.setBindGroup(0, attentionPipelines.normBind);
        pass.dispatchWorkgroups(DEFAULT_TOKENS);
      }
      pass.end();
      return encoder.finish();
    };
    const encodeFfnDense1 = (count: number) => encodePipelineStage(
      ffnPipelines.dense1,
      ffnPipelines.dense1Bind,
      (pass) => pass.dispatchWorkgroups(Math.ceil(DEFAULT_FFN_HIDDEN / 8), Math.ceil(DEFAULT_TOKENS / 8)),
      count,
    );
    const encodeFfnDense2Residual = (count: number) => encodePipelineStage(
      ffnPipelines.dense2,
      ffnPipelines.dense2Bind,
      (pass) => pass.dispatchWorkgroups(Math.ceil(DEFAULT_N / 8), Math.ceil(DEFAULT_TOKENS / 8)),
      count,
    );
    const encodeLn2 = (count: number) => encodePipelineStage(
      ffnPipelines.ln2,
      ffnPipelines.ln2Bind,
      (pass) => pass.dispatchWorkgroups(DEFAULT_TOKENS),
      count,
    );
    const encodeAttention = (count: number) => {
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      for (let i = 0; i < count; i++) {
        pass.setPipeline(attentionPipelines.qkv);
        pass.setBindGroup(0, attentionPipelines.qkvBind);
        pass.dispatchWorkgroups(Math.ceil(DEFAULT_N / 8), Math.ceil(DEFAULT_TOKENS / 8));
        pass.setPipeline(attentionPipelines.score);
        pass.setBindGroup(0, attentionPipelines.scoreBind);
        pass.dispatchWorkgroups(Math.ceil(DEFAULT_TOKENS / 8), Math.ceil(DEFAULT_TOKENS / 8), DEFAULT_HEADS);
        pass.setPipeline(attentionPipelines.softmax);
        pass.setBindGroup(0, attentionPipelines.softmaxBind);
        pass.dispatchWorkgroups(DEFAULT_HEADS * DEFAULT_TOKENS);
        pass.setPipeline(attentionPipelines.value);
        pass.setBindGroup(0, attentionPipelines.valueBind);
        pass.dispatchWorkgroups(Math.ceil(DEFAULT_HEAD_DIM / 8), Math.ceil(DEFAULT_TOKENS / 8), DEFAULT_HEADS);
        pass.setPipeline(attentionPipelines.outProj);
        pass.setBindGroup(0, attentionPipelines.outProjBind);
        pass.dispatchWorkgroups(Math.ceil(DEFAULT_N / 8), Math.ceil(DEFAULT_TOKENS / 8));
        pass.setPipeline(attentionPipelines.norm);
        pass.setBindGroup(0, attentionPipelines.normBind);
        pass.dispatchWorkgroups(DEFAULT_TOKENS);
      }
      pass.end();
      return encoder.finish();
    };
    const encodeFfn = (count: number) => encodeEncoder0FfnDispatches(device, ffnPipelines, count);
    const measureBlockGpuTimestampMs = async (count: number): Promise<number | undefined> => {
      if (!timestampQuerySupported || !device.createQuerySet || !usage.QUERY_RESOLVE) return undefined;
      const querySet = device.createQuerySet({ type: 'timestamp', count: 2 });
      const resolveBuffer = device.createBuffer({ size: 16, usage: usage.QUERY_RESOLVE | usage.COPY_SRC });
      const timestampReadback = device.createBuffer({ size: 16, usage: usage.MAP_READ | usage.COPY_DST });
      try {
        const encoder = device.createCommandEncoder();
        const attentionPass = encoder.beginComputePass({ timestampWrites: { querySet, beginningOfPassWriteIndex: 0 } });
        for (let i = 0; i < count; i++) {
          attentionPass.setPipeline(attentionPipelines.qkv);
          attentionPass.setBindGroup(0, attentionPipelines.qkvBind);
          attentionPass.dispatchWorkgroups(Math.ceil(DEFAULT_N / 8), Math.ceil(DEFAULT_TOKENS / 8));
          attentionPass.setPipeline(attentionPipelines.score);
          attentionPass.setBindGroup(0, attentionPipelines.scoreBind);
          attentionPass.dispatchWorkgroups(Math.ceil(DEFAULT_TOKENS / 8), Math.ceil(DEFAULT_TOKENS / 8), DEFAULT_HEADS);
          attentionPass.setPipeline(attentionPipelines.softmax);
          attentionPass.setBindGroup(0, attentionPipelines.softmaxBind);
          attentionPass.dispatchWorkgroups(DEFAULT_HEADS * DEFAULT_TOKENS);
          attentionPass.setPipeline(attentionPipelines.value);
          attentionPass.setBindGroup(0, attentionPipelines.valueBind);
          attentionPass.dispatchWorkgroups(Math.ceil(DEFAULT_HEAD_DIM / 8), Math.ceil(DEFAULT_TOKENS / 8), DEFAULT_HEADS);
          attentionPass.setPipeline(attentionPipelines.outProj);
          attentionPass.setBindGroup(0, attentionPipelines.outProjBind);
          attentionPass.dispatchWorkgroups(Math.ceil(DEFAULT_N / 8), Math.ceil(DEFAULT_TOKENS / 8));
          attentionPass.setPipeline(attentionPipelines.norm);
          attentionPass.setBindGroup(0, attentionPipelines.normBind);
          attentionPass.dispatchWorkgroups(DEFAULT_TOKENS);
        }
        attentionPass.end();
        const ffnPass = encoder.beginComputePass({ timestampWrites: { querySet, endOfPassWriteIndex: 1 } });
        for (let i = 0; i < count; i++) {
          ffnPass.setPipeline(ffnPipelines.dense1);
          ffnPass.setBindGroup(0, ffnPipelines.dense1Bind);
          ffnPass.dispatchWorkgroups(Math.ceil(DEFAULT_FFN_HIDDEN / 8), Math.ceil(DEFAULT_TOKENS / 8));
          ffnPass.setPipeline(ffnPipelines.dense2);
          ffnPass.setBindGroup(0, ffnPipelines.dense2Bind);
          ffnPass.dispatchWorkgroups(Math.ceil(DEFAULT_N / 8), Math.ceil(DEFAULT_TOKENS / 8));
          ffnPass.setPipeline(ffnPipelines.ln2);
          ffnPass.setBindGroup(0, ffnPipelines.ln2Bind);
          ffnPass.dispatchWorkgroups(DEFAULT_TOKENS);
        }
        ffnPass.end();
        encoder.resolveQuerySet?.(querySet, 0, 2, resolveBuffer, 0);
        encoder.copyBufferToBuffer(resolveBuffer, 0, timestampReadback, 0, 16);
        device.queue.submit([encoder.finish()]);
        await timestampReadback.mapAsync(globals.GPUMapMode!.READ);
        const timestamps = new BigUint64Array(timestampReadback.getMappedRange().slice(0));
        timestampReadback.unmap();
        const elapsedNs = timestamps[1] > timestamps[0] ? timestamps[1] - timestamps[0] : 0n;
        return Number(elapsedNs) / 1_000_000;
      } catch {
        return undefined;
      } finally {
        querySet.destroy?.();
        resolveBuffer.destroy?.();
        timestampReadback.destroy?.();
      }
    };
    const measureStageTimings = async (count: number): Promise<Lc0WebEncoder0BlockStageTiming[]> => {
      const stageDefs: { stage: Lc0WebEncoder0BlockStageName; label: string; encode: (iterations: number) => unknown }[] = [
        { stage: 'qkvProjection', label: 'QKV projection', encode: encodeQkv },
        { stage: 'attentionScores', label: 'attention scores', encode: encodeAttentionScores },
        { stage: 'softmax', label: 'softmax', encode: encodeSoftmax },
        { stage: 'attentionValue', label: 'attention value', encode: encodeAttentionValue },
        { stage: 'outputProjectionLn1', label: 'output projection + ln1', encode: encodeOutputProjectionLn1 },
        { stage: 'ffnDense1', label: 'FFN dense1', encode: encodeFfnDense1 },
        { stage: 'ffnDense2Residual', label: 'FFN dense2 + residual', encode: encodeFfnDense2Residual },
        { stage: 'ln2', label: 'ln2', encode: encodeLn2 },
      ];
      const timings: Lc0WebEncoder0BlockStageTiming[] = [];
      for (const stageDef of stageDefs) {
        const started = nowMs();
        device.queue.submit([stageDef.encode(count)]);
        await device.queue.onSubmittedWorkDone?.();
        const totalMs = nowMs() - started;
        timings.push({
          stage: stageDef.stage,
          label: stageDef.label,
          iterations: count,
          totalMs,
          avgMs: totalMs / count,
        });
      }
      return timings;
    };
    const submitBlockBatches = (count: number) => {
      // Submit attention-output and FFN command buffers together. WebGPU queue
      // ordering preserves the write/read dependency from ln1 output into FFN
      // dense1 without forcing an intermediate queue-completion sync.
      device.queue.submit([encodeAttention(count), encodeFfn(count)]);
    };
    if (warmup > 0) {
      submitBlockBatches(warmup);
      await device.queue.onSubmittedWorkDone?.();
    }
    const dispatchStarted = nowMs();
    submitBlockBatches(iterations);
    const dispatchLoopMs = nowMs() - dispatchStarted;
    const gpuOutput = await readF32OutputOnce(device, output, readback, outputElements);
    const readbackSyncedMs = nowMs() - dispatchStarted;
    const { maxAbsError, rmsError } = computeErrorStats(gpuOutput, reference.output, outputElements);
    assertErrorInTolerance(maxAbsError);
    const gpuTimestampMs = await measureBlockGpuTimestampMs(iterations);
    const stageTimings = await measureStageTimings(iterations);
    const stageTimingTotalMs = stageTimings.reduce((sum, timing) => sum + timing.totalMs, 0);
    return {
      status: 'ENCODER0_BLOCK_BENCH_DONE',
      packUrl: pack.manifestUrl,
      modelName: pack.manifest.model.name,
      adapterInfo,
      tokens: DEFAULT_TOKENS,
      channels: DEFAULT_N,
      heads: DEFAULT_HEADS,
      headDim: DEFAULT_HEAD_DIM,
      ffnHidden: DEFAULT_FFN_HIDDEN,
      lnEpsilon: DEFAULT_LN_EPSILON,
      attentionAlpha: reference.attentionAlpha,
      ffnAlpha: reference.ffnAlpha,
      smolgen: { enabled: true, epsilon: DEFAULT_SMOLGEN_EPSILON },
      warmup,
      iterations,
      packLoadMs: pack.elapsedMs,
      uploadSetupMs,
      dispatchLoopMs,
      dispatchLoopAvgMs: dispatchLoopMs / iterations,
      readbackSyncedMs,
      gpuTimestampSupported: timestampQuerySupported,
      gpuTimestampMs,
      stageTimings,
      stageTimingTotalMs,
      endToEndMs: nowMs() - totalStarted,
      maxAbsError,
      rmsError,
      outputSample: Array.from(gpuOutput.slice(0, 8)),
    };
  } finally {
    for (const buffer of buffers) buffer.destroy?.();
  }
}

export async function runLc0WebEncoderStackBenchmark(options: Lc0WebEncoderStackBenchmarkOptions): Promise<Lc0WebEncoderStackBenchmarkResult> {
  const totalStarted = nowMs();
  const warmup = clampInteger(options.warmup, 0, 0, 10);
  const layers = clampInteger(options.layers, 2, 1, 32);
  const compareOrt = options.compareOrt ?? true;
  const compareHeads = options.compareHeads ?? false;
  const prefixes = Array.from({ length: layers }, (_, layer) => `/encoder${layer}`);
  const layerTensorNames = prefixes.map((prefix) => lc0WebEncoderBlockTensorNames(prefix));
  const pack = await loadLc0WebModelPack(options.packUrl, {
    verifyShards: options.verifyShards ?? true,
    tensorNames: Array.from(new Set([
      ...layerTensorNames.flatMap((names) => encoderBlockTensorNameList(names)),
      ...(compareHeads ? policyValueHeadTensorNameList() : []),
    ])),
  });
  const tensorsByLayer = layerTensorNames.map((names) => loadEncoder0FfnInputs(pack, names));
  const headTensors = compareHeads ? loadPolicyValueHeadTensors(pack) : undefined;
  const { device, adapterInfo } = await requestDevice();
  const globals = gpuGlobals();
  const usage = globals.GPUBufferUsage!;
  const outputElements = DEFAULT_TOKENS * DEFAULT_N;
  const buffers: BufferLike[] = [];
  const blocks: Lc0WebEncoderStackBlockResult[] = [];
  try {
    const setupStarted = nowMs();
    let cpuInput = options.input ?? makeInputTokenMatrix(DEFAULT_TOKENS, DEFAULT_K);
    if (cpuInput.length !== outputElements) throw new Error(`encoder stack input length ${cpuInput.length} != ${outputElements}`);
    let gpuInput = createStorageBuffer(device, cpuInput, usage.STORAGE | usage.COPY_DST);
    buffers.push(gpuInput);
    let lastGpuOutput = cpuInput;
    for (let layer = 0; layer < layers; layer++) {
      const tensors = tensorsByLayer[layer];
      const reference = buildEncoder0BlockReference(tensors, cpuInput);
      const qWeight = createTransposedF16StorageBuffer(device, tensors.qWeight.bytes, DEFAULT_K, DEFAULT_N, usage.STORAGE | usage.COPY_DST);
      const qBias = createStorageBuffer(device, tensors.qBias.bytes, usage.STORAGE | usage.COPY_DST);
      const kWeight = createTransposedF16StorageBuffer(device, tensors.kWeight.bytes, DEFAULT_K, DEFAULT_N, usage.STORAGE | usage.COPY_DST);
      const kBias = createStorageBuffer(device, tensors.kBias.bytes, usage.STORAGE | usage.COPY_DST);
      const vWeight = createTransposedF16StorageBuffer(device, tensors.vWeight.bytes, DEFAULT_K, DEFAULT_N, usage.STORAGE | usage.COPY_DST);
      const vBias = createStorageBuffer(device, tensors.vBias.bytes, usage.STORAGE | usage.COPY_DST);
      const scale = createStorageBuffer(device, paddedF16ScalarBytes(tensors.scale.bytes), usage.STORAGE | usage.COPY_DST);
      const smolgenBias = createStorageBuffer(device, reference.smolgenBias, usage.STORAGE | usage.COPY_DST);
      const outWeight = createTransposedF16StorageBuffer(device, tensors.outWeight.bytes, DEFAULT_N, DEFAULT_N, usage.STORAGE | usage.COPY_DST);
      const outBias = createStorageBuffer(device, tensors.outBias.bytes, usage.STORAGE | usage.COPY_DST);
      const attentionAlpha = createStorageBuffer(device, paddedF16ScalarBytes(tensors.alpha.bytes), usage.STORAGE | usage.COPY_DST);
      const ln1Scale = createStorageBuffer(device, tensors.lnScale.bytes, usage.STORAGE | usage.COPY_DST);
      const ln1Bias = createStorageBuffer(device, tensors.lnBias.bytes, usage.STORAGE | usage.COPY_DST);
      const ffnDense1Weight = createTransposedF16StorageBuffer(device, tensors.ffnDense1Weight.bytes, DEFAULT_N, DEFAULT_FFN_HIDDEN, usage.STORAGE | usage.COPY_DST);
      const ffnDense1Bias = createStorageBuffer(device, tensors.ffnDense1Bias.bytes, usage.STORAGE | usage.COPY_DST);
      const ffnDense2Weight = createTransposedF16StorageBuffer(device, tensors.ffnDense2Weight.bytes, DEFAULT_FFN_HIDDEN, DEFAULT_N, usage.STORAGE | usage.COPY_DST);
      const ffnDense2Bias = createStorageBuffer(device, tensors.ffnDense2Bias.bytes, usage.STORAGE | usage.COPY_DST);
      const ffnAlpha = createStorageBuffer(device, paddedF16ScalarBytes(tensors.ffnAlpha.bytes), usage.STORAGE | usage.COPY_DST);
      const ln2Scale = createStorageBuffer(device, tensors.ln2Scale.bytes, usage.STORAGE | usage.COPY_DST);
      const ln2Bias = createStorageBuffer(device, tensors.ln2Bias.bytes, usage.STORAGE | usage.COPY_DST);
      const qkv = device.createBuffer({ size: DEFAULT_TOKENS * DEFAULT_N * 3 * 4, usage: usage.STORAGE });
      const scores = device.createBuffer({ size: DEFAULT_HEADS * DEFAULT_TOKENS * DEFAULT_TOKENS * 4, usage: usage.STORAGE });
      const probs = device.createBuffer({ size: DEFAULT_HEADS * DEFAULT_TOKENS * DEFAULT_TOKENS * 4, usage: usage.STORAGE });
      const attn = device.createBuffer({ size: outputElements * 4, usage: usage.STORAGE });
      const attentionSkip = device.createBuffer({ size: outputElements * 4, usage: usage.STORAGE });
      const attentionOutput = device.createBuffer({ size: outputElements * 4, usage: usage.STORAGE });
      const ffnHidden = device.createBuffer({ size: DEFAULT_TOKENS * DEFAULT_FFN_HIDDEN * 4, usage: usage.STORAGE });
      const ffnSkip = device.createBuffer({ size: outputElements * 4, usage: usage.STORAGE });
      const output = device.createBuffer({ size: outputElements * 4, usage: usage.STORAGE | usage.COPY_SRC });
      const readback = device.createBuffer({ size: outputElements * 4, usage: usage.MAP_READ | usage.COPY_DST });
      buffers.push(qWeight, qBias, kWeight, kBias, vWeight, vBias, scale, smolgenBias, outWeight, outBias, attentionAlpha, ln1Scale, ln1Bias, ffnDense1Weight, ffnDense1Bias, ffnDense2Weight, ffnDense2Bias, ffnAlpha, ln2Scale, ln2Bias, qkv, scores, probs, attn, attentionSkip, attentionOutput, ffnHidden, ffnSkip, output, readback);
      const attentionPipelines = createAttentionOutputPipelines(device, {
        input: gpuInput, qWeight, qBias, kWeight, kBias, vWeight, vBias, scale, smolgenBias, qkv, scores, probs, attn,
        outWeight, outBias, alpha: attentionAlpha, skip: attentionSkip, lnScale: ln1Scale, lnBias: ln1Bias, output: attentionOutput,
      });
      const ffnPipelines = createEncoder0FfnPipelines(device, {
        input: attentionOutput,
        dense1Weight: ffnDense1Weight,
        dense1Bias: ffnDense1Bias,
        hidden: ffnHidden,
        dense2Weight: ffnDense2Weight,
        dense2Bias: ffnDense2Bias,
        alpha: ffnAlpha,
        skip: ffnSkip,
        ln2Scale,
        ln2Bias,
        output,
      });
      if (warmup > 0) {
        device.queue.submit([encodeLc0WebEncoderBlockDispatches(device, attentionPipelines, ffnPipelines, warmup)]);
        await device.queue.onSubmittedWorkDone?.();
      }
      const blockStarted = nowMs();
      device.queue.submit([encodeLc0WebEncoderBlockDispatches(device, attentionPipelines, ffnPipelines, 1)]);
      const gpuOutput = await readF32OutputOnce(device, output, readback, outputElements);
      const dispatchSyncedMs = nowMs() - blockStarted;
      const { maxAbsError, rmsError } = computeErrorStats(gpuOutput, reference.output, outputElements);
      assertErrorInTolerance(maxAbsError);
      const block: Lc0WebEncoderStackBlockResult = {
        layer,
        prefix: prefixes[layer],
        dispatchSyncedMs,
        maxAbsError,
        rmsError,
        outputSample: Array.from(gpuOutput.slice(0, 8)),
      };
      if (compareOrt) {
        const attentionValue = buildAttentionValueReference(tensors, cpuInput);
        const tinyOnnx = createTinyEncoder0BlockOnnxForTest(
          f16BytesToF32Array(tensors.outWeight.bytes, DEFAULT_N * DEFAULT_N),
          f16BytesToF32Array(tensors.outBias.bytes, DEFAULT_N),
          reference.attentionAlpha,
          f16BytesToF32Array(tensors.lnScale.bytes, DEFAULT_N),
          f16BytesToF32Array(tensors.lnBias.bytes, DEFAULT_N),
          f16BytesToF32Array(tensors.ffnDense1Weight.bytes, DEFAULT_N * DEFAULT_FFN_HIDDEN),
          f16BytesToF32Array(tensors.ffnDense1Bias.bytes, DEFAULT_FFN_HIDDEN),
          f16BytesToF32Array(tensors.ffnDense2Weight.bytes, DEFAULT_FFN_HIDDEN * DEFAULT_N),
          f16BytesToF32Array(tensors.ffnDense2Bias.bytes, DEFAULT_N),
          reference.ffnAlpha,
          f16BytesToF32Array(tensors.ln2Scale.bytes, DEFAULT_N),
          f16BytesToF32Array(tensors.ln2Bias.bytes, DEFAULT_N),
        );
        const session = await ort.createOrtSession(tinyOnnx);
        const ortOutputs = await session.run({
          attention: new ort.Tensor('float32', attentionValue.output, [DEFAULT_TOKENS, DEFAULT_N]),
          residual: new ort.Tensor('float32', cpuInput, [DEFAULT_TOKENS, DEFAULT_N]),
        });
        const ortOutput = ortOutputs.output.data as Float32Array<ArrayBufferLike>;
        const ortVsGpu = computeErrorStats(gpuOutput, ortOutput, outputElements);
        const ortVsCpu = computeErrorStats(ortOutput, reference.output, outputElements);
        assertErrorInTolerance(ortVsGpu.maxAbsError);
        block.ortMaxAbsError = ortVsGpu.maxAbsError;
        block.ortRmsError = ortVsGpu.rmsError;
        block.ortVsCpuMaxAbsError = ortVsCpu.maxAbsError;
        block.ortVsCpuRmsError = ortVsCpu.rmsError;
      }
      blocks.push(block);
      // Use the actual previous-layer GPU output as the next CPU reference
      // input too, so CPU-side smolgen bias matches the GPU-buffer handoff
      // activation instead of an idealized CPU-only stack activation.
      cpuInput = gpuOutput;
      gpuInput = output;
      lastGpuOutput = gpuOutput;
    }
    const dispatchSyncedMs = blocks.reduce((sum, block) => sum + block.dispatchSyncedMs, 0);
    const maxBlock = blocks.reduce((max, block) => Math.max(max, block.maxAbsError), 0);
    const rmsBlock = Math.sqrt(blocks.reduce((sum, block) => sum + block.rmsError * block.rmsError, 0) / blocks.length);
    const ortMax = compareOrt ? blocks.reduce((max, block) => Math.max(max, block.ortMaxAbsError ?? 0), 0) : undefined;
    const policyValueHeads = headTensors ? await runPolicyValueHeadsOrt(lastGpuOutput, headTensors, { includeOutputs: options.includeHeadOutputs }) : undefined;
    return {
      status: 'ENCODER_STACK_BENCH_DONE',
      packUrl: pack.manifestUrl,
      modelName: pack.manifest.model.name,
      adapterInfo,
      tokens: DEFAULT_TOKENS,
      channels: DEFAULT_N,
      heads: DEFAULT_HEADS,
      headDim: DEFAULT_HEAD_DIM,
      ffnHidden: DEFAULT_FFN_HIDDEN,
      lnEpsilon: DEFAULT_LN_EPSILON,
      warmup,
      layers,
      prefixes,
      compareOrt,
      compareHeads,
      ortCoveredStages: compareHeads
        ? 'attention output projection/ln1 + FFN/ln2 f32 ONNX subgraph per block; final policy/value heads run through a tiny f32 ONNX subgraph from the WGSL stack output; QKV/softmax/attention value are checked against the CPU f32 reference'
        : 'attention output projection/ln1 + FFN/ln2 f32 ONNX subgraph per block; QKV/softmax/attention value are checked against the CPU f32 reference',
      packLoadMs: pack.elapsedMs,
      setupAndDispatchMs: nowMs() - setupStarted,
      dispatchSyncedMs,
      avgBlockDispatchSyncedMs: dispatchSyncedMs / layers,
      maxAbsError: maxBlock,
      rmsError: rmsBlock,
      ortMaxAbsError: ortMax,
      outputSample: Array.from(lastGpuOutput.slice(0, 8)),
      policyValueHeads,
      blocks,
    };
  } finally {
    for (const buffer of buffers) buffer.destroy?.();
  }
}

export interface Lc0WebAttentionValueOrtBenchmarkOptions {
  packUrl: string;
  iterations?: number;
  warmup?: number;
  verifyShards?: boolean;
}

export interface Lc0WebAttentionValueOrtBenchmarkResult {
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
  timesMs: number[];
  runsPerSecond: number;
  maxAbsError: number;
  rmsError: number;
  outputSample: number[];
}

export function createTinyAttentionValueOnnxForTest(): Uint8Array {
  const writer = new ProtoWriter();
  writer.int64(1, 8);
  writer.string(2, 'lc0web');
  writer.message(7, (graph) => {
    graph.bytes(1, onnxNode('MatMul', ['probs', 'v'], ['output'], 'attention_value'));
    graph.string(2, 'lc0web_attention_value_heads');
    graph.bytes(11, onnxValueInfo('probs', 1, [DEFAULT_HEADS, DEFAULT_TOKENS, DEFAULT_TOKENS]));
    graph.bytes(11, onnxValueInfo('v', 1, [DEFAULT_HEADS, DEFAULT_TOKENS, DEFAULT_HEAD_DIM]));
    graph.bytes(12, onnxValueInfo('output', 1, [DEFAULT_HEADS, DEFAULT_TOKENS, DEFAULT_HEAD_DIM]));
  });
  writer.message(8, (opset) => opset.int64(2, 13));
  return writer.finish();
}

function unpackHeadsToTokenChannels(input: Float32Array<ArrayBufferLike>, tokens: number, channels: number, heads: number): Float32Array<ArrayBufferLike> {
  const headDim = channels / heads;
  const out = new Float32Array(input.length);
  for (let head = 0; head < heads; head++) {
    for (let token = 0; token < tokens; token++) {
      for (let channel = 0; channel < headDim; channel++) {
        out[token * channels + head * headDim + channel] = input[(head * tokens + token) * headDim + channel];
      }
    }
  }
  return out;
}

export async function runLc0WebAttentionValueOrtBenchmark(options: Lc0WebAttentionValueOrtBenchmarkOptions): Promise<Lc0WebAttentionValueOrtBenchmarkResult> {
  const warmup = clampInteger(options.warmup, 5, 0, 100);
  const iterations = clampInteger(options.iterations, 25, 1, 1000);
  const pack = await loadLc0WebModelPack(options.packUrl, {
    verifyShards: options.verifyShards ?? true,
    tensorNames: [...Object.values(DEFAULT_QKV_TENSORS), DEFAULT_SCALE_TENSOR, ...Object.values(DEFAULT_SMOLGEN_TENSORS)],
  });
  const tensors = loadAttentionValueInputs(pack);
  const reference = buildAttentionValueReference(tensors);
  const probs = reference.probs;
  const vByHead = packHeadsQ(reference.v, DEFAULT_TOKENS, DEFAULT_N, DEFAULT_HEADS);
  const modelBuildStarted = nowMs();
  const tinyOnnx = createTinyAttentionValueOnnxForTest();
  const modelBuildMs = nowMs() - modelBuildStarted;
  const sessionStarted = nowMs();
  const session = await ort.createOrtSession(tinyOnnx);
  const sessionCreateMs = nowMs() - sessionStarted;
  const feeds = {
    probs: new ort.Tensor('float32', probs, [DEFAULT_HEADS, DEFAULT_TOKENS, DEFAULT_TOKENS]),
    v: new ort.Tensor('float32', vByHead, [DEFAULT_HEADS, DEFAULT_TOKENS, DEFAULT_HEAD_DIM]),
  };
  let outputHeads: Float32Array<ArrayBufferLike> = new Float32Array(DEFAULT_TOKENS * DEFAULT_N);
  for (let i = 0; i < warmup; i++) {
    const outputs = await session.run(feeds);
    outputHeads = outputs.output.data as Float32Array<ArrayBufferLike>;
  }
  const times: number[] = [];
  for (let i = 0; i < iterations; i++) {
    const started = nowMs();
    const outputs = await session.run(feeds);
    times.push(nowMs() - started);
    outputHeads = outputs.output.data as Float32Array<ArrayBufferLike>;
  }
  const output = unpackHeadsToTokenChannels(outputHeads, DEFAULT_TOKENS, DEFAULT_N, DEFAULT_HEADS);
  const { maxAbsError, rmsError } = computeErrorStats(output, reference.output, DEFAULT_TOKENS * DEFAULT_N);
  assertErrorInTolerance(maxAbsError);
  const avgMs = times.reduce((sum, value) => sum + value, 0) / times.length;
  return {
    status: 'ATTENTION_VALUE_ORT_BENCH_DONE',
    packUrl: pack.manifestUrl,
    modelName: pack.manifest.model.name,
    tokens: DEFAULT_TOKENS,
    channels: DEFAULT_N,
    heads: DEFAULT_HEADS,
    headDim: DEFAULT_HEAD_DIM,
    warmup,
    iterations,
    packLoadMs: pack.elapsedMs,
    modelBuildMs,
    sessionCreateMs,
    avgMs,
    minMs: Math.min(...times),
    maxMs: Math.max(...times),
    firstMs: times[0],
    timesMs: times,
    runsPerSecond: 1000 / avgMs,
    maxAbsError,
    rmsError,
    outputSample: Array.from(output.slice(0, 8)),
  };
}
