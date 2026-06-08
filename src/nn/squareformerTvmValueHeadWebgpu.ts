import type { SquareformerGpuBuffer, SquareformerTvmFfnShape } from './squareformerTvmFfnWebgpu.ts';

type GPUBuffer = SquareformerGpuBuffer;
type GPUDevice = {
  queue: { writeBuffer(buffer: GPUBuffer, bufferOffset: number, data: unknown): void; submit(commandBuffers: unknown[]): void };
  createBindGroup(descriptor: unknown): unknown;
  createBuffer(descriptor: { label?: string; size: number; usage: number; mappedAtCreation?: boolean }): GPUBuffer;
  createCommandEncoder(descriptor?: unknown): GPUCommandEncoder;
  createComputePipelineAsync(descriptor: unknown): Promise<GPUComputePipeline>;
  createShaderModule(descriptor: { label?: string; code: string }): unknown;
};
type GPUComputePipeline = { getBindGroupLayout(index: number): unknown };
type GPUCommandEncoder = {
  beginComputePass(descriptor?: unknown): {
    setPipeline(pipeline: GPUComputePipeline): void;
    setBindGroup(index: number, bindGroup: unknown): void;
    dispatchWorkgroups(x: number, y?: number, z?: number): void;
    end(): void;
  };
  copyBufferToBuffer(source: GPUBuffer, sourceOffset: number, destination: GPUBuffer, destinationOffset: number, size: number): void;
  finish(): unknown;
};
declare const GPUBufferUsage: Record<'STORAGE' | 'COPY_SRC' | 'COPY_DST' | 'MAP_READ', number>;
declare const GPUMapMode: Record<'READ', number>;

export type SquareformerTvmValueHeadWeights = {
  wdlWeight: Float32Array;
  wdlBias: Float32Array;
  qWeight: Float32Array;
  qBias: Float32Array;
};

const VALUE_HEAD_WGSL = `
@group(0) @binding(0) var<storage, read_write> wdl : array<f32>;
@group(0) @binding(1) var<storage, read_write> q : array<f32>;
@group(0) @binding(2) var<storage, read> hidden : array<f32>;
@group(0) @binding(3) var<storage, read> wdlWeight : array<f32>;
@group(0) @binding(4) var<storage, read> wdlBias : array<f32>;
@group(0) @binding(5) var<storage, read> qWeight : array<f32>;
@group(0) @binding(6) var<storage, read> qBias : array<f32>;
var<workgroup> scratch : array<f32, 128>;

@compute @workgroup_size(128, 1, 1)
fn value_head_kernel(@builtin(workgroup_id) blockIdx : vec3<u32>, @builtin(local_invocation_id) threadIdx : vec3<u32>) {
  let outIdx : u32 = blockIdx.x;
  let lane : u32 = threadIdx.x;
  var sum : f32 = 0.0f;
  for (var row : u32 = 0u; row < 64u; row = row + 1u) {
    sum = sum + hidden[row * 128u + lane];
  }
  let mean : f32 = sum / 64.0f;
  var term : f32;
  if (outIdx < 3u) {
    term = mean * wdlWeight[outIdx * 128u + lane];
  } else {
    term = mean * qWeight[lane];
  }
  scratch[lane] = term;
  workgroupBarrier();
  var stride : u32 = 64u;
  loop {
    if (lane < stride) { scratch[lane] = scratch[lane] + scratch[lane + stride]; }
    workgroupBarrier();
    if (stride == 1u) { break; }
    stride = stride / 2u;
  }
  if (lane == 0u) {
    if (outIdx < 3u) { wdl[outIdx] = scratch[0u] + wdlBias[outIdx]; }
    else { q[0u] = scratch[0u] + qBias[0u]; }
  }
}
`;

function assertLength(name: string, actual: number, expected: number) { if (actual !== expected) throw new Error(`${name} length ${actual} does not match expected ${expected}`); }
function createInitializedBuffer(device: GPUDevice, label: string, data: Float32Array, usage: number): GPUBuffer {
  const buffer = device.createBuffer({ label, size: Math.max(4, data.byteLength), usage, mappedAtCreation: true });
  new Float32Array(buffer.getMappedRange()).set(data);
  buffer.unmap();
  return buffer;
}
function createStorageBuffer(device: GPUDevice, label: string, bytes: number, usage = 0): GPUBuffer {
  return device.createBuffer({ label, size: Math.max(4, bytes), usage: GPUBufferUsage.STORAGE | usage });
}

export class SquareformerTvmValueHeadWebgpuBlock {
  readonly shape: SquareformerTvmFfnShape;
  private pipeline: GPUComputePipeline;
  private input: GPUBuffer;
  private wdl: GPUBuffer;
  private q: GPUBuffer;
  private wdlReadback: GPUBuffer;
  private qReadback: GPUBuffer;
  private wdlWeight: GPUBuffer;
  private wdlBias: GPUBuffer;
  private qWeight: GPUBuffer;
  private qBias: GPUBuffer;

  private constructor(private device: GPUDevice, shape: SquareformerTvmFfnShape, pipeline: GPUComputePipeline, buffers: {
    input: GPUBuffer; wdl: GPUBuffer; q: GPUBuffer; wdlReadback: GPUBuffer; qReadback: GPUBuffer; wdlWeight: GPUBuffer; wdlBias: GPUBuffer; qWeight: GPUBuffer; qBias: GPUBuffer;
  }) {
    this.shape = shape;
    this.pipeline = pipeline;
    this.input = buffers.input;
    this.wdl = buffers.wdl;
    this.q = buffers.q;
    this.wdlReadback = buffers.wdlReadback;
    this.qReadback = buffers.qReadback;
    this.wdlWeight = buffers.wdlWeight;
    this.wdlBias = buffers.wdlBias;
    this.qWeight = buffers.qWeight;
    this.qBias = buffers.qBias;
  }

  static async create(device: GPUDevice, weights: SquareformerTvmValueHeadWeights, shape: Partial<SquareformerTvmFfnShape> = {}): Promise<SquareformerTvmValueHeadWebgpuBlock> {
    const fullShape = { rows: 64, dModel: 128, dFf: 256, ...shape };
    if (fullShape.rows !== 64 || fullShape.dModel !== 128) throw new Error(`value head expects rows=64,dModel=128; got ${JSON.stringify(fullShape)}`);
    assertLength('wdlWeight', weights.wdlWeight.length, 3 * fullShape.dModel);
    assertLength('wdlBias', weights.wdlBias.length, 3);
    assertLength('qWeight', weights.qWeight.length, fullShape.dModel);
    assertLength('qBias', weights.qBias.length, 1);
    const module = device.createShaderModule({ label: 'squareformer-tvm-value-head', code: VALUE_HEAD_WGSL });
    const pipeline = await device.createComputePipelineAsync({ label: 'squareformer-tvm-value-head', layout: 'auto', compute: { module, entryPoint: 'value_head_kernel' } });
    const inputBytes = fullShape.rows * fullShape.dModel * 4;
    const buffers = {
      input: createStorageBuffer(device, 'squareformer-tvm-value-input', inputBytes, GPUBufferUsage.COPY_DST),
      wdl: createStorageBuffer(device, 'squareformer-tvm-value-wdl', 3 * 4, GPUBufferUsage.COPY_SRC),
      q: createStorageBuffer(device, 'squareformer-tvm-value-q', 4, GPUBufferUsage.COPY_SRC),
      wdlReadback: device.createBuffer({ label: 'squareformer-tvm-value-wdl-readback', size: 3 * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }),
      qReadback: device.createBuffer({ label: 'squareformer-tvm-value-q-readback', size: 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }),
      wdlWeight: createInitializedBuffer(device, 'squareformer-tvm-value-wdl-weight', weights.wdlWeight, GPUBufferUsage.STORAGE),
      wdlBias: createInitializedBuffer(device, 'squareformer-tvm-value-wdl-bias', weights.wdlBias, GPUBufferUsage.STORAGE),
      qWeight: createInitializedBuffer(device, 'squareformer-tvm-value-q-weight', weights.qWeight, GPUBufferUsage.STORAGE),
      qBias: createInitializedBuffer(device, 'squareformer-tvm-value-q-bias', weights.qBias, GPUBufferUsage.STORAGE),
    };
    return new SquareformerTvmValueHeadWebgpuBlock(device, fullShape, pipeline, buffers);
  }

  upload(input: Float32Array): void {
    assertLength('input', input.length, this.shape.rows * this.shape.dModel);
    this.device.queue.writeBuffer(this.input, 0, input);
  }

  encode(commandEncoder: GPUCommandEncoder, inputBuffer: GPUBuffer = this.input): { wdl: GPUBuffer; q: GPUBuffer } {
    const pass = commandEncoder.beginComputePass({ label: 'squareformer-tvm-value-head' });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.device.createBindGroup({
      label: 'squareformer-tvm-value-bindings',
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.wdl } },
        { binding: 1, resource: { buffer: this.q } },
        { binding: 2, resource: { buffer: inputBuffer } },
        { binding: 3, resource: { buffer: this.wdlWeight } },
        { binding: 4, resource: { buffer: this.wdlBias } },
        { binding: 5, resource: { buffer: this.qWeight } },
        { binding: 6, resource: { buffer: this.qBias } },
      ],
    }));
    pass.dispatchWorkgroups(4, 1, 1);
    pass.end();
    return { wdl: this.wdl, q: this.q };
  }

  copyOutputToReadback(commandEncoder: GPUCommandEncoder): void {
    commandEncoder.copyBufferToBuffer(this.wdl, 0, this.wdlReadback, 0, 3 * 4);
    commandEncoder.copyBufferToBuffer(this.q, 0, this.qReadback, 0, 4);
  }

  async readOutput(): Promise<{ wdl: Float32Array; q: Float32Array }> {
    await Promise.all([this.wdlReadback.mapAsync(GPUMapMode.READ), this.qReadback.mapAsync(GPUMapMode.READ)]);
    const wdl = new Float32Array(this.wdlReadback.getMappedRange().slice(0));
    const q = new Float32Array(this.qReadback.getMappedRange().slice(0));
    this.wdlReadback.unmap();
    this.qReadback.unmap();
    return { wdl, q };
  }

  async run(input: Float32Array, options: { readback?: boolean } = {}): Promise<{ wdl: Float32Array; q: Float32Array } | { wdl: GPUBuffer; q: GPUBuffer }> {
    this.upload(input);
    const commandEncoder = this.device.createCommandEncoder({ label: 'squareformer-tvm-value-head-run' });
    const output = this.encode(commandEncoder);
    if (options.readback !== false) this.copyOutputToReadback(commandEncoder);
    this.device.queue.submit([commandEncoder.finish()]);
    if (options.readback === false) return output;
    return this.readOutput();
  }

  destroy(): void {
    for (const buffer of [this.input, this.wdl, this.q, this.wdlReadback, this.qReadback, this.wdlWeight, this.wdlBias, this.qWeight, this.qBias]) buffer.destroy();
  }
}
