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
const DEFAULT_K = 256;
const DEFAULT_N = 256;

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
  maxAbsError: { q: number; k: number; v: number };
  rmsError: { q: number; k: number; v: number };
  outputSample: { q: number[]; k: number[]; v: number[] };
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

function cpuMatmulAdd(input: Float32Array<ArrayBufferLike>, weight: Uint8Array, bias: Uint8Array, k: number, n: number): Float32Array<ArrayBufferLike> {
  const output = new Float32Array(n);
  for (let col = 0; col < n; col++) {
    let sum = readF16At(bias, col);
    for (let row = 0; row < k; row++) sum += input[row] * readF16At(weight, row * n + col);
    output[col] = sum;
  }
  return output;
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

function onnxNode(opType: string, inputs: string[], outputs: string[], name: string): Uint8Array {
  const writer = new ProtoWriter();
  for (const input of inputs) writer.string(1, input);
  for (const output of outputs) writer.string(2, output);
  writer.string(3, name);
  writer.string(4, opType);
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

async function runQkvOnce(device: DeviceLike, pipeline: PipelineLike, bindGroup: unknown, outputBuffer: BufferLike, readbackBuffer: BufferLike): Promise<{ q: Float32Array<ArrayBufferLike>; k: Float32Array<ArrayBufferLike>; v: Float32Array<ArrayBufferLike> }> {
  const globals = gpuGlobals();
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(DEFAULT_N / 64));
  pass.end();
  encoder.copyBufferToBuffer(outputBuffer, 0, readbackBuffer, 0, DEFAULT_N * 3 * 4);
  device.queue.submit([encoder.finish()]);
  await readbackBuffer.mapAsync(globals.GPUMapMode!.READ);
  const all = new Float32Array(readbackBuffer.getMappedRange().slice(0));
  readbackBuffer.unmap();
  return { q: all.slice(0, DEFAULT_N), k: all.slice(DEFAULT_N, DEFAULT_N * 2), v: all.slice(DEFAULT_N * 2, DEFAULT_N * 3) };
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
      maxAbsError: { q: qErr.maxAbsError, k: kErr.maxAbsError, v: vErr.maxAbsError },
      rmsError: { q: qErr.rmsError, k: kErr.rmsError, v: vErr.rmsError },
      outputSample: { q: Array.from(outputs.q.slice(0, 8)), k: Array.from(outputs.k.slice(0, 8)), v: Array.from(outputs.v.slice(0, 8)) },
    };
  } finally {
    for (const buffer of liveBuffers) buffer.destroy?.();
  }
}
