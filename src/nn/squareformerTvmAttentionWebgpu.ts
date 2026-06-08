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
declare const GPUBufferUsage: Record<'STORAGE' | 'COPY_SRC' | 'COPY_DST' | 'MAP_READ' | 'UNIFORM', number>;
declare const GPUMapMode: Record<'READ', number>;

export type SquareformerTvmAttentionShape = SquareformerTvmFfnShape & {
  heads: number;
};

const DEFAULT_SHAPE: SquareformerTvmAttentionShape = { rows: 64, dModel: 128, dFf: 256, heads: 8 };

const ATTENTION_WGSL = `
@group(0) @binding(0) var<storage, read_write> context : array<f32>;
@group(0) @binding(1) var<storage, read> qkv : array<f32>;
@group(0) @binding(2) var<storage, read> bias : array<f32>;
struct PodArgs { rows: u32, dModel: u32, heads: u32, headDim: u32 }
@group(0) @binding(3) var<uniform> podArgs : PodArgs;
var<workgroup> scores : array<f32, 64>;
var<workgroup> scratch : array<f32, 64>;

fn q_at(row: u32, head: u32, dim: u32) -> f32 {
  return qkv[row * (3u * podArgs.dModel) + head * podArgs.headDim + dim];
}
fn k_at(row: u32, head: u32, dim: u32) -> f32 {
  return qkv[row * (3u * podArgs.dModel) + podArgs.dModel + head * podArgs.headDim + dim];
}
fn v_at(row: u32, head: u32, dim: u32) -> f32 {
  return qkv[row * (3u * podArgs.dModel) + 2u * podArgs.dModel + head * podArgs.headDim + dim];
}

@compute @workgroup_size(64, 1, 1)
fn attention_kernel(@builtin(workgroup_id) blockIdx : vec3<u32>, @builtin(local_invocation_id) threadIdx : vec3<u32>) {
  let row : u32 = blockIdx.x;
  let head : u32 = blockIdx.y;
  let lane : u32 = threadIdx.x;
  var dot : f32 = 0.0f;
  for (var d : u32 = 0u; d < 16u; d = d + 1u) {
    dot = dot + q_at(row, head, d) * k_at(lane, head, d);
  }
  let biasIndex : u32 = head * 4096u + row * 64u + lane;
  scores[lane] = dot * 2.500000000000000e-01f + bias[biasIndex];
  scratch[lane] = scores[lane];
  workgroupBarrier();
  var stride : u32 = 32u;
  loop {
    if (lane < stride) { scratch[lane] = max(scratch[lane], scratch[lane + stride]); }
    workgroupBarrier();
    if (stride == 1u) { break; }
    stride = stride / 2u;
  }
  workgroupBarrier();
  let maxScore : f32 = scratch[0u];
  let unnormalized : f32 = exp(scores[lane] - maxScore);
  scores[lane] = unnormalized;
  workgroupBarrier();
  scratch[lane] = unnormalized;
  workgroupBarrier();
  stride = 32u;
  loop {
    if (lane < stride) { scratch[lane] = scratch[lane] + scratch[lane + stride]; }
    workgroupBarrier();
    if (stride == 1u) { break; }
    stride = stride / 2u;
  }
  workgroupBarrier();
  let invSum : f32 = 1.0f / scratch[0u];
  workgroupBarrier();
  if (lane < podArgs.headDim) {
    var acc : f32 = 0.0f;
    for (var keyRow : u32 = 0u; keyRow < 64u; keyRow = keyRow + 1u) {
      acc = acc + scores[keyRow] * invSum * v_at(keyRow, head, lane);
    }
    context[row * podArgs.dModel + head * podArgs.headDim + lane] = acc;
  }
}
`;

function assertLength(name: string, actual: number, expected: number) {
  if (actual !== expected) throw new Error(`${name} length ${actual} does not match expected ${expected}`);
}

function createInitializedBuffer(device: GPUDevice, label: string, data: Uint32Array, usage: number): GPUBuffer {
  const buffer = device.createBuffer({ label, size: Math.max(4, data.byteLength), usage, mappedAtCreation: true });
  new Uint32Array(buffer.getMappedRange()).set(data);
  buffer.unmap();
  return buffer;
}

function createStorageBuffer(device: GPUDevice, label: string, bytes: number, usage = 0): GPUBuffer {
  return device.createBuffer({ label, size: Math.max(4, bytes), usage: GPUBufferUsage.STORAGE | usage });
}

export class SquareformerTvmAttentionWebgpuBlock {
  readonly shape: SquareformerTvmAttentionShape;
  private pipeline: GPUComputePipeline;
  private qkv: GPUBuffer;
  private bias: GPUBuffer;
  private output: GPUBuffer;
  private readback: GPUBuffer;
  private podArgs: GPUBuffer;

  private constructor(private device: GPUDevice, shape: SquareformerTvmAttentionShape, pipeline: GPUComputePipeline, buffers: { qkv: GPUBuffer; bias: GPUBuffer; output: GPUBuffer; readback: GPUBuffer; podArgs: GPUBuffer }) {
    this.shape = shape;
    this.pipeline = pipeline;
    this.qkv = buffers.qkv;
    this.bias = buffers.bias;
    this.output = buffers.output;
    this.readback = buffers.readback;
    this.podArgs = buffers.podArgs;
  }

  static async create(device: GPUDevice, shape: Partial<SquareformerTvmAttentionShape> = {}): Promise<SquareformerTvmAttentionWebgpuBlock> {
    const fullShape = { ...DEFAULT_SHAPE, ...shape };
    const { rows, dModel, heads } = fullShape;
    if (rows !== 64) throw new Error(`attention shader is fixed to 64 rows, got ${rows}`);
    if (dModel % heads !== 0) throw new Error(`dModel ${dModel} must be divisible by heads ${heads}`);
    const headDim = dModel / heads;
    if (headDim !== 16) throw new Error(`attention shader is fixed to headDim=16, got ${headDim}`);
    const module = device.createShaderModule({ label: 'squareformer-tvm-attention', code: ATTENTION_WGSL });
    const pipeline = await device.createComputePipelineAsync({ label: 'squareformer-tvm-attention', layout: 'auto', compute: { module, entryPoint: 'attention_kernel' } });
    const qkvBytes = rows * 3 * dModel * 4;
    const biasBytes = heads * rows * rows * 4;
    const outputBytes = rows * dModel * 4;
    const buffers = {
      qkv: createStorageBuffer(device, 'squareformer-tvm-attention-qkv', qkvBytes, GPUBufferUsage.COPY_DST),
      bias: createStorageBuffer(device, 'squareformer-tvm-attention-bias', biasBytes, GPUBufferUsage.COPY_DST),
      output: createStorageBuffer(device, 'squareformer-tvm-attention-context', outputBytes, GPUBufferUsage.COPY_SRC),
      readback: device.createBuffer({ label: 'squareformer-tvm-attention-readback', size: outputBytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }),
      podArgs: createInitializedBuffer(device, 'squareformer-tvm-attention-pod', new Uint32Array([rows, dModel, heads, headDim]), GPUBufferUsage.UNIFORM),
    };
    return new SquareformerTvmAttentionWebgpuBlock(device, fullShape, pipeline, buffers);
  }

  upload(qkv: Float32Array, bias: Float32Array): void {
    this.uploadQkv(qkv);
    this.uploadBias(bias);
  }

  uploadQkv(qkv: Float32Array): void {
    const { rows, dModel } = this.shape;
    assertLength('qkv', qkv.length, rows * 3 * dModel);
    this.device.queue.writeBuffer(this.qkv, 0, qkv);
  }

  uploadBias(bias: Float32Array): void {
    const { rows, heads } = this.shape;
    assertLength('bias', bias.length, heads * rows * rows);
    this.device.queue.writeBuffer(this.bias, 0, bias);
  }

  encode(commandEncoder: GPUCommandEncoder, qkvBuffer: GPUBuffer = this.qkv, biasBuffer: GPUBuffer = this.bias, outputBuffer: GPUBuffer = this.output): GPUBuffer {
    const pass = commandEncoder.beginComputePass({ label: 'squareformer-tvm-attention' });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.device.createBindGroup({
      label: 'squareformer-tvm-attention-bindings',
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: outputBuffer } },
        { binding: 1, resource: { buffer: qkvBuffer } },
        { binding: 2, resource: { buffer: biasBuffer } },
        { binding: 3, resource: { buffer: this.podArgs } },
      ],
    }));
    pass.dispatchWorkgroups(this.shape.rows, this.shape.heads, 1);
    pass.end();
    return outputBuffer;
  }

  copyOutputToReadback(commandEncoder: GPUCommandEncoder): void {
    commandEncoder.copyBufferToBuffer(this.output, 0, this.readback, 0, this.shape.rows * this.shape.dModel * 4);
  }

  async readOutput(): Promise<Float32Array> {
    await this.readback.mapAsync(GPUMapMode.READ);
    const copy = new Float32Array(this.readback.getMappedRange().slice(0));
    this.readback.unmap();
    return copy;
  }

  async run(qkv: Float32Array, bias: Float32Array, options: { readback?: boolean } = {}): Promise<Float32Array | GPUBuffer> {
    this.upload(qkv, bias);
    const commandEncoder = this.device.createCommandEncoder({ label: 'squareformer-tvm-attention-run' });
    const output = this.encode(commandEncoder);
    if (options.readback !== false) this.copyOutputToReadback(commandEncoder);
    this.device.queue.submit([commandEncoder.finish()]);
    if (options.readback === false) return output;
    return this.readOutput();
  }

  destroy(): void {
    for (const buffer of [this.qkv, this.bias, this.output, this.readback, this.podArgs]) buffer.destroy();
  }
}
