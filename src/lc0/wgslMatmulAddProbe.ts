import * as ort from '../nn/ortRuntime.ts';
import { loadLc0WebModelPack, type Lc0WebTensorView } from './modelPack.ts';

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
  requestDevice: () => Promise<DeviceLike>;
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
  createShaderModule: (descriptor: Record<string, unknown>) => unknown;
  createComputePipeline: (descriptor: Record<string, unknown>) => PipelineLike;
  createBindGroup: (descriptor: Record<string, unknown>) => unknown;
  createCommandEncoder: () => CommandEncoderLike;
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
  beginComputePass: () => ComputePassLike;
  copyBufferToBuffer: (source: unknown, sourceOffset: number, destination: unknown, destinationOffset: number, size: number) => void;
  finish: () => unknown;
};

type ComputePassLike = {
  setPipeline: (pipeline: unknown) => void;
  setBindGroup: (index: number, bindGroup: unknown) => void;
  dispatchWorkgroups: (x: number, y?: number, z?: number) => void;
  end: () => void;
};

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

function onnxFloatAttribute(name: string, value: number): Uint8Array {
  const writer = new ProtoWriter();
  writer.string(1, name);
  writer.float32(2, value);
  writer.int32(20, 1); // AttributeProto.FLOAT
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

async function requestDevice(): Promise<{ device: DeviceLike; adapterInfo?: Record<string, unknown> }> {
  const globals = gpuGlobals();
  const gpu = globals.navigator?.gpu as GpuLike | undefined;
  if (!gpu) throw new Error('WebGPU unavailable for lc0web kernel probe');
  const adapter = await gpu.requestAdapter() as AdapterLike | null;
  if (!adapter) throw new Error('WebGPU adapter unavailable for lc0web kernel probe');
  const rawAdapterInfo = adapter.requestAdapterInfo ? await adapter.requestAdapterInfo() : adapter.info;
  const device = await adapter.requestDevice();
  return { device, adapterInfo: cloneableAdapterInfo(rawAdapterInfo) };
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

function buildAttentionScoreReference(tensors: ReturnType<typeof loadAttentionScoreInputs>): { input: Float32Array<ArrayBufferLike>; q: Float32Array<ArrayBufferLike>; k: Float32Array<ArrayBufferLike>; scale: number; qkScores: Float32Array<ArrayBufferLike>; smolgenBias: Float32Array<ArrayBufferLike>; scores: Float32Array<ArrayBufferLike> } {
  const input = makeInputTokenMatrix(DEFAULT_TOKENS, DEFAULT_K);
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

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.x;
  if (row >= 512u) { return; }
  let base = row * 64u;
  var max_value = -3.4028234663852886e+38;
  for (var col = 0u; col < 64u; col = col + 1u) {
    max_value = max(max_value, inputScores[base + col]);
  }
  var sum = 0.0;
  for (var col = 0u; col < 64u; col = col + 1u) {
    let value = exp(inputScores[base + col] - max_value);
    outputProbs[base + col] = value;
    sum = sum + value;
  }
  let inv_sum = 1.0 / sum;
  for (var col = 0u; col < 64u; col = col + 1u) {
    outputProbs[base + col] = outputProbs[base + col] * inv_sum;
  }
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
  for (let i = 0; i < iterations; i++) pass.dispatchWorkgroups(Math.ceil(rows / 64));
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

function buildAttentionValueReference(tensors: ReturnType<typeof loadAttentionValueInputs>): { probs: Float32Array<ArrayBufferLike>; v: Float32Array<ArrayBufferLike>; output: Float32Array<ArrayBufferLike>; scale: number; smolgenBias: Float32Array<ArrayBufferLike> } {
  const input = makeInputTokenMatrix(DEFAULT_TOKENS, DEFAULT_K);
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

export interface Lc0WebAttentionBlockBenchmarkOptions {
  packUrl: string;
  iterations?: number;
  warmup?: number;
  verifyShards?: boolean;
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

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let col = gid.x;
  let token = gid.y;
  if (col >= 256u || token >= 64u) { return; }
  var q_sum = load_q_bias(col);
  var k_sum = load_k_bias(col);
  var v_sum = load_v_bias(col);
  for (var row = 0u; row < 256u; row = row + 1u) {
    let x = inputMat[token * 256u + row];
    let base = row * 256u + col;
    q_sum = q_sum + x * load_q_weight(base);
    k_sum = k_sum + x * load_k_weight(base);
    v_sum = v_sum + x * load_v_weight(base);
  }
  qkvOut[token * 256u + col] = q_sum;
  qkvOut[16384u + token * 256u + col] = k_sum;
  qkvOut[32768u + token * 256u + col] = v_sum;
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
}): { qkv: PipelineLike; qkvBind: unknown; score: PipelineLike; scoreBind: unknown; softmax: PipelineLike; softmaxBind: unknown; value: PipelineLike; valueBind: unknown } {
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
  return { qkv, qkvBind, score, scoreBind, softmax, softmaxBind, value, valueBind };
}

function encodeAttentionBlockDispatches(device: DeviceLike, pipelines: ReturnType<typeof createAttentionBlockPipelines>, iterations: number): unknown {
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  for (let i = 0; i < iterations; i++) {
    pass.setPipeline(pipelines.qkv);
    pass.setBindGroup(0, pipelines.qkvBind);
    pass.dispatchWorkgroups(Math.ceil(DEFAULT_N / 8), Math.ceil(DEFAULT_TOKENS / 8));
    pass.setPipeline(pipelines.score);
    pass.setBindGroup(0, pipelines.scoreBind);
    pass.dispatchWorkgroups(Math.ceil(DEFAULT_TOKENS / 8), Math.ceil(DEFAULT_TOKENS / 8), DEFAULT_HEADS);
    pass.setPipeline(pipelines.softmax);
    pass.setBindGroup(0, pipelines.softmaxBind);
    pass.dispatchWorkgroups(Math.ceil((DEFAULT_HEADS * DEFAULT_TOKENS) / 64));
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
    const qWeight = createStorageBuffer(device, tensors.qWeight.bytes, usage.STORAGE | usage.COPY_DST);
    const qBias = createStorageBuffer(device, tensors.qBias.bytes, usage.STORAGE | usage.COPY_DST);
    const kWeight = createStorageBuffer(device, tensors.kWeight.bytes, usage.STORAGE | usage.COPY_DST);
    const kBias = createStorageBuffer(device, tensors.kBias.bytes, usage.STORAGE | usage.COPY_DST);
    const vWeight = createStorageBuffer(device, tensors.vWeight.bytes, usage.STORAGE | usage.COPY_DST);
    const vBias = createStorageBuffer(device, tensors.vBias.bytes, usage.STORAGE | usage.COPY_DST);
    const scale = createStorageBuffer(device, paddedF16ScalarBytes(tensors.scale.bytes), usage.STORAGE | usage.COPY_DST);
    const smolgenBias = createStorageBuffer(device, reference.smolgenBias, usage.STORAGE | usage.COPY_DST);
    const qkvBuffer = device.createBuffer({ size: DEFAULT_TOKENS * DEFAULT_N * 3 * 4, usage: usage.STORAGE });
    const scoreBuffer = device.createBuffer({ size: DEFAULT_HEADS * DEFAULT_TOKENS * DEFAULT_TOKENS * 4, usage: usage.STORAGE });
    const probBuffer = device.createBuffer({ size: DEFAULT_HEADS * DEFAULT_TOKENS * DEFAULT_TOKENS * 4, usage: usage.STORAGE });
    const outputBuffer = device.createBuffer({ size: outputElements * 4, usage: usage.STORAGE | usage.COPY_SRC });
    const readbackBuffer = device.createBuffer({ size: outputElements * 4, usage: usage.MAP_READ | usage.COPY_DST });
    buffers.push(inputBuffer, qWeight, qBias, kWeight, kBias, vWeight, vBias, scale, smolgenBias, qkvBuffer, scoreBuffer, probBuffer, outputBuffer, readbackBuffer);
    const pipelines = createAttentionBlockPipelines(device, { input: inputBuffer, qWeight, qBias, kWeight, kBias, vWeight, vBias, scale, smolgenBias, qkv: qkvBuffer, scores: scoreBuffer, probs: probBuffer, output: outputBuffer });
    const uploadSetupMs = nowMs() - setupStarted;

    if (warmup > 0) {
      device.queue.submit([encodeAttentionBlockDispatches(device, pipelines, warmup)]);
      await device.queue.onSubmittedWorkDone?.();
    }
    const dispatchStarted = nowMs();
    device.queue.submit([encodeAttentionBlockDispatches(device, pipelines, iterations)]);
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

export interface Lc0WebAttentionOutputBenchmarkOptions {
  packUrl: string;
  iterations?: number;
  warmup?: number;
  verifyShards?: boolean;
  encoderPrefix?: string;
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

function buildAttentionOutputReference(tensors: ReturnType<typeof loadAttentionOutputInputs>): { input: Float32Array<ArrayBufferLike>; output: Float32Array<ArrayBufferLike>; alpha: number; smolgenBias: Float32Array<ArrayBufferLike> } {
  const input = makeInputTokenMatrix(DEFAULT_TOKENS, DEFAULT_K);
  const attentionReference = buildAttentionValueReference(tensors);
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

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let col = gid.x;
  let token = gid.y;
  if (col >= 256u || token >= 64u) { return; }
  var sum = load_bias(col);
  for (var row = 0u; row < 256u; row = row + 1u) {
    sum = sum + attnVec[token * 256u + row] * load_weight(row * 256u + col);
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

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let token = gid.x;
  if (token >= 64u) { return; }
  let base = token * 256u;
  var mean = 0.0;
  for (var col = 0u; col < 256u; col = col + 1u) { mean = mean + skipVec[base + col]; }
  mean = mean / 256.0;
  var variance = 0.0;
  for (var col = 0u; col < 256u; col = col + 1u) {
    let centered = skipVec[base + col] - mean;
    variance = variance + centered * centered;
  }
  let inv_std = inverseSqrt(variance / 256.0 + 0.000001);
  for (var col = 0u; col < 256u; col = col + 1u) {
    outputVec[base + col] = (skipVec[base + col] - mean) * inv_std * load_scale(col) + load_bias(col);
  }
}
`;

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
}): ReturnType<typeof createAttentionBlockPipelines> & { outProj: PipelineLike; outProjBind: unknown; norm: PipelineLike; normBind: unknown } {
  const base = createAttentionBlockPipelines(device, { ...buffers, output: buffers.attn });
  const outModule = device.createShaderModule({ label: 'lc0web attention output projection residual', code: ATTENTION_OUTPUT_PROJ_WGSL });
  const outProj = device.createComputePipeline({ layout: 'auto', compute: { module: outModule, entryPoint: 'main' } }) as PipelineLike;
  const outProjBind = device.createBindGroup({
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
  return { ...base, outProj, outProjBind, norm, normBind };
}

function encodeAttentionOutputDispatches(device: DeviceLike, pipelines: ReturnType<typeof createAttentionOutputPipelines>, iterations: number): unknown {
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  for (let i = 0; i < iterations; i++) {
    pass.setPipeline(pipelines.qkv);
    pass.setBindGroup(0, pipelines.qkvBind);
    pass.dispatchWorkgroups(Math.ceil(DEFAULT_N / 8), Math.ceil(DEFAULT_TOKENS / 8));
    pass.setPipeline(pipelines.score);
    pass.setBindGroup(0, pipelines.scoreBind);
    pass.dispatchWorkgroups(Math.ceil(DEFAULT_TOKENS / 8), Math.ceil(DEFAULT_TOKENS / 8), DEFAULT_HEADS);
    pass.setPipeline(pipelines.softmax);
    pass.setBindGroup(0, pipelines.softmaxBind);
    pass.dispatchWorkgroups(Math.ceil((DEFAULT_HEADS * DEFAULT_TOKENS) / 64));
    pass.setPipeline(pipelines.value);
    pass.setBindGroup(0, pipelines.valueBind);
    pass.dispatchWorkgroups(Math.ceil(DEFAULT_HEAD_DIM / 8), Math.ceil(DEFAULT_TOKENS / 8), DEFAULT_HEADS);
    pass.setPipeline(pipelines.outProj);
    pass.setBindGroup(0, pipelines.outProjBind);
    pass.dispatchWorkgroups(Math.ceil(DEFAULT_N / 8), Math.ceil(DEFAULT_TOKENS / 8));
    pass.setPipeline(pipelines.norm);
    pass.setBindGroup(0, pipelines.normBind);
    pass.dispatchWorkgroups(Math.ceil(DEFAULT_TOKENS / 64));
  }
  pass.end();
  return encoder.finish();
}

export async function runLc0WebAttentionOutputBenchmark(options: Lc0WebAttentionOutputBenchmarkOptions): Promise<Lc0WebAttentionOutputBenchmarkResult> {
  const totalStarted = nowMs();
  const warmup = clampInteger(options.warmup, 3, 0, 1000);
  const iterations = clampInteger(options.iterations, 50, 1, 10_000);
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
    const qWeight = createStorageBuffer(device, tensors.qWeight.bytes, usage.STORAGE | usage.COPY_DST);
    const qBias = createStorageBuffer(device, tensors.qBias.bytes, usage.STORAGE | usage.COPY_DST);
    const kWeight = createStorageBuffer(device, tensors.kWeight.bytes, usage.STORAGE | usage.COPY_DST);
    const kBias = createStorageBuffer(device, tensors.kBias.bytes, usage.STORAGE | usage.COPY_DST);
    const vWeight = createStorageBuffer(device, tensors.vWeight.bytes, usage.STORAGE | usage.COPY_DST);
    const vBias = createStorageBuffer(device, tensors.vBias.bytes, usage.STORAGE | usage.COPY_DST);
    const scale = createStorageBuffer(device, paddedF16ScalarBytes(tensors.scale.bytes), usage.STORAGE | usage.COPY_DST);
    const smolgenBias = createStorageBuffer(device, reference.smolgenBias, usage.STORAGE | usage.COPY_DST);
    const outWeight = createStorageBuffer(device, tensors.outWeight.bytes, usage.STORAGE | usage.COPY_DST);
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
    buffers.push(inputBuffer, qWeight, qBias, kWeight, kBias, vWeight, vBias, scale, smolgenBias, outWeight, outBias, alpha, lnScale, lnBias, qkvBuffer, scoreBuffer, probBuffer, attnBuffer, skipBuffer, outputBuffer, readbackBuffer);
    const pipelines = createAttentionOutputPipelines(device, { input: inputBuffer, qWeight, qBias, kWeight, kBias, vWeight, vBias, scale, smolgenBias, qkv: qkvBuffer, scores: scoreBuffer, probs: probBuffer, attn: attnBuffer, outWeight, outBias, alpha, skip: skipBuffer, lnScale, lnBias, output: outputBuffer });
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

export interface Lc0WebEncoder0FfnBenchmarkOptions {
  packUrl: string;
  iterations?: number;
  warmup?: number;
  verifyShards?: boolean;
  encoderPrefix?: string;
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
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let col = gid.x;
  let token = gid.y;
  if (col >= 1024u || token >= 64u) { return; }
  var sum = pick_lane(biasF16[col >> 1u], col);
  for (var row = 0u; row < 256u; row = row + 1u) {
    sum = sum + inputVec[token * 256u + row] * pick_lane(weightsF16[(row * 1024u + col) >> 1u], row * 1024u + col);
  }
  let value = max(sum, 0.0);
  outputVec[token * 1024u + col] = value * value;
}
`;

const FFN_DENSE2_WGSL = `${WGSL_HEADER}
@group(0) @binding(4) var<storage, read> residualVec: array<f32>;
@group(0) @binding(5) var<storage, read> alphaF16: array<u32>;
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let col = gid.x;
  let token = gid.y;
  if (col >= 256u || token >= 64u) { return; }
  var sum = pick_lane(biasF16[col >> 1u], col);
  for (var row = 0u; row < 1024u; row = row + 1u) {
    sum = sum + inputVec[token * 1024u + row] * pick_lane(weightsF16[(row * 256u + col) >> 1u], row * 256u + col);
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
}): { dense1: PipelineLike; dense1Bind: unknown; dense2: PipelineLike; dense2Bind: unknown; ln2: PipelineLike; ln2Bind: unknown } {
  const dense1Module = device.createShaderModule({ label: 'lc0web encoder0 FFN dense1 sqrrelu', code: FFN_DENSE1_WGSL });
  const dense1 = device.createComputePipeline({ layout: 'auto', compute: { module: dense1Module, entryPoint: 'main' } }) as PipelineLike;
  const dense1Bind = device.createBindGroup({ layout: dense1.getBindGroupLayout(0), entries: [
    { binding: 0, resource: { buffer: buffers.input } },
    { binding: 1, resource: { buffer: buffers.dense1Weight } },
    { binding: 2, resource: { buffer: buffers.dense1Bias } },
    { binding: 3, resource: { buffer: buffers.hidden } },
  ] });
  const dense2Module = device.createShaderModule({ label: 'lc0web encoder0 FFN dense2 residual', code: FFN_DENSE2_WGSL });
  const dense2 = device.createComputePipeline({ layout: 'auto', compute: { module: dense2Module, entryPoint: 'main' } }) as PipelineLike;
  const dense2Bind = device.createBindGroup({ layout: dense2.getBindGroupLayout(0), entries: [
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
  return { dense1, dense1Bind, dense2, dense2Bind, ln2, ln2Bind };
}

function encodeEncoder0FfnDispatches(device: DeviceLike, pipelines: ReturnType<typeof createEncoder0FfnPipelines>, iterations: number): unknown {
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  for (let i = 0; i < iterations; i++) {
    pass.setPipeline(pipelines.dense1);
    pass.setBindGroup(0, pipelines.dense1Bind);
    pass.dispatchWorkgroups(Math.ceil(DEFAULT_FFN_HIDDEN / 8), Math.ceil(DEFAULT_TOKENS / 8));
    pass.setPipeline(pipelines.dense2);
    pass.setBindGroup(0, pipelines.dense2Bind);
    pass.dispatchWorkgroups(Math.ceil(DEFAULT_N / 8), Math.ceil(DEFAULT_TOKENS / 8));
    pass.setPipeline(pipelines.ln2);
    pass.setBindGroup(0, pipelines.ln2Bind);
    pass.dispatchWorkgroups(Math.ceil(DEFAULT_TOKENS / 64));
  }
  pass.end();
  return encoder.finish();
}

export async function runLc0WebEncoder0FfnBenchmark(options: Lc0WebEncoder0FfnBenchmarkOptions): Promise<Lc0WebEncoder0FfnBenchmarkResult> {
  const totalStarted = nowMs();
  const warmup = clampInteger(options.warmup, 2, 0, 1000);
  const iterations = clampInteger(options.iterations, 10, 1, 10_000);
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
    const dense1Weight = createStorageBuffer(device, tensors.ffnDense1Weight.bytes, usage.STORAGE | usage.COPY_DST);
    const dense1Bias = createStorageBuffer(device, tensors.ffnDense1Bias.bytes, usage.STORAGE | usage.COPY_DST);
    const dense2Weight = createStorageBuffer(device, tensors.ffnDense2Weight.bytes, usage.STORAGE | usage.COPY_DST);
    const dense2Bias = createStorageBuffer(device, tensors.ffnDense2Bias.bytes, usage.STORAGE | usage.COPY_DST);
    const alpha = createStorageBuffer(device, paddedF16ScalarBytes(tensors.ffnAlpha.bytes), usage.STORAGE | usage.COPY_DST);
    const ln2Scale = createStorageBuffer(device, tensors.ln2Scale.bytes, usage.STORAGE | usage.COPY_DST);
    const ln2Bias = createStorageBuffer(device, tensors.ln2Bias.bytes, usage.STORAGE | usage.COPY_DST);
    const hidden = device.createBuffer({ size: DEFAULT_TOKENS * DEFAULT_FFN_HIDDEN * 4, usage: usage.STORAGE });
    const skip = device.createBuffer({ size: outputElements * 4, usage: usage.STORAGE });
    const output = device.createBuffer({ size: outputElements * 4, usage: usage.STORAGE | usage.COPY_SRC });
    const readback = device.createBuffer({ size: outputElements * 4, usage: usage.MAP_READ | usage.COPY_DST });
    buffers.push(input, dense1Weight, dense1Bias, dense2Weight, dense2Bias, alpha, ln2Scale, ln2Bias, hidden, skip, output, readback);
    const pipelines = createEncoder0FfnPipelines(device, { input, dense1Weight, dense1Bias, hidden, dense2Weight, dense2Bias, alpha, skip, ln2Scale, ln2Bias, output });
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

function buildEncoder0BlockReference(tensors: Encoder0FfnTensors): { input: Float32Array<ArrayBufferLike>; output: Float32Array<ArrayBufferLike>; attentionAlpha: number; ffnAlpha: number; smolgenBias: Float32Array<ArrayBufferLike> } {
  const attention = buildAttentionOutputReference(tensors);
  const ffn = cpuEncoder0FfnFromLn1(attention.output, tensors);
  return { input: attention.input, output: ffn.output, attentionAlpha: attention.alpha, ffnAlpha: ffn.alpha, smolgenBias: attention.smolgenBias };
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
  const { device, adapterInfo } = await requestDevice();
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
    const qWeight = createStorageBuffer(device, tensors.qWeight.bytes, usage.STORAGE | usage.COPY_DST);
    const qBias = createStorageBuffer(device, tensors.qBias.bytes, usage.STORAGE | usage.COPY_DST);
    const kWeight = createStorageBuffer(device, tensors.kWeight.bytes, usage.STORAGE | usage.COPY_DST);
    const kBias = createStorageBuffer(device, tensors.kBias.bytes, usage.STORAGE | usage.COPY_DST);
    const vWeight = createStorageBuffer(device, tensors.vWeight.bytes, usage.STORAGE | usage.COPY_DST);
    const vBias = createStorageBuffer(device, tensors.vBias.bytes, usage.STORAGE | usage.COPY_DST);
    const scale = createStorageBuffer(device, paddedF16ScalarBytes(tensors.scale.bytes), usage.STORAGE | usage.COPY_DST);
    const smolgenBias = createStorageBuffer(device, reference.smolgenBias, usage.STORAGE | usage.COPY_DST);
    const outWeight = createStorageBuffer(device, tensors.outWeight.bytes, usage.STORAGE | usage.COPY_DST);
    const outBias = createStorageBuffer(device, tensors.outBias.bytes, usage.STORAGE | usage.COPY_DST);
    const attentionAlpha = createStorageBuffer(device, paddedF16ScalarBytes(tensors.alpha.bytes), usage.STORAGE | usage.COPY_DST);
    const ln1Scale = createStorageBuffer(device, tensors.lnScale.bytes, usage.STORAGE | usage.COPY_DST);
    const ln1Bias = createStorageBuffer(device, tensors.lnBias.bytes, usage.STORAGE | usage.COPY_DST);
    const ffnDense1Weight = createStorageBuffer(device, tensors.ffnDense1Weight.bytes, usage.STORAGE | usage.COPY_DST);
    const ffnDense1Bias = createStorageBuffer(device, tensors.ffnDense1Bias.bytes, usage.STORAGE | usage.COPY_DST);
    const ffnDense2Weight = createStorageBuffer(device, tensors.ffnDense2Weight.bytes, usage.STORAGE | usage.COPY_DST);
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
      (pass) => pass.dispatchWorkgroups(Math.ceil((DEFAULT_HEADS * DEFAULT_TOKENS) / 64)),
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
        pass.dispatchWorkgroups(Math.ceil(DEFAULT_TOKENS / 64));
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
      (pass) => pass.dispatchWorkgroups(Math.ceil(DEFAULT_TOKENS / 64)),
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
        pass.dispatchWorkgroups(Math.ceil((DEFAULT_HEADS * DEFAULT_TOKENS) / 64));
        pass.setPipeline(attentionPipelines.value);
        pass.setBindGroup(0, attentionPipelines.valueBind);
        pass.dispatchWorkgroups(Math.ceil(DEFAULT_HEAD_DIM / 8), Math.ceil(DEFAULT_TOKENS / 8), DEFAULT_HEADS);
        pass.setPipeline(attentionPipelines.outProj);
        pass.setBindGroup(0, attentionPipelines.outProjBind);
        pass.dispatchWorkgroups(Math.ceil(DEFAULT_N / 8), Math.ceil(DEFAULT_TOKENS / 8));
        pass.setPipeline(attentionPipelines.norm);
        pass.setBindGroup(0, attentionPipelines.normBind);
        pass.dispatchWorkgroups(Math.ceil(DEFAULT_TOKENS / 64));
      }
      pass.end();
      return encoder.finish();
    };
    const encodeFfn = (count: number) => encodeEncoder0FfnDispatches(device, ffnPipelines, count);
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
