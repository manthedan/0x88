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

export type SquareformerTvmLayerNormQkvWeights = {
  layerNormWeight: Float32Array;
  layerNormBias: Float32Array;
  qkvWeight: Float32Array;
  qkvBias: Float32Array;
};

export type SquareformerTvmLayerNormQkvKernels = {
  qkvBias: string;
};

const LAYERNORM_WGSL = `
@group(0) @binding(0) var<storage, read_write> norm_out : array<f32>;
@group(0) @binding(1) var<storage, read> x : array<f32>;
@group(0) @binding(2) var<storage, read> gamma : array<f32>;
@group(0) @binding(3) var<storage, read> beta : array<f32>;
struct PodArgs { epsilon: f32, rows: u32, dModel: u32, _pad: u32 }
@group(0) @binding(4) var<uniform> podArgs : PodArgs;
var<workgroup> scratch : array<f32, 128>;
@compute @workgroup_size(128, 1, 1)
fn layernorm_kernel(@builtin(workgroup_id) blockIdx : vec3<u32>, @builtin(local_invocation_id) threadIdx : vec3<u32>) {
  let row : u32 = blockIdx.x;
  let col : u32 = threadIdx.x;
  let base : u32 = row * podArgs.dModel;
  let value : f32 = x[base + col];
  scratch[col] = value;
  workgroupBarrier();
  var stride : u32 = 64u;
  loop {
    if (col < stride) { scratch[col] = scratch[col] + scratch[col + stride]; }
    workgroupBarrier();
    if (stride == 1u) { break; }
    stride = stride / 2u;
  }
  let mean : f32 = scratch[0u] / f32(podArgs.dModel);
  let centered : f32 = value - mean;
  scratch[col] = centered * centered;
  workgroupBarrier();
  stride = 64u;
  loop {
    if (col < stride) { scratch[col] = scratch[col] + scratch[col + stride]; }
    workgroupBarrier();
    if (stride == 1u) { break; }
    stride = stride / 2u;
  }
  let inv_std : f32 = inverseSqrt((scratch[0u] / f32(podArgs.dModel)) + podArgs.epsilon);
  norm_out[base + col] = centered * inv_std * gamma[col] + beta[col];
}
`;

function ceilDiv(a: number, b: number): number { return Math.floor((a + b - 1) / b); }
function assertLength(name: string, actual: number, expected: number) { if (actual !== expected) throw new Error(`${name} length ${actual} does not match expected ${expected}`); }
function createInitializedBuffer(device: GPUDevice, label: string, data: Float32Array | Uint32Array, usage: number): GPUBuffer {
  const buffer = device.createBuffer({ label, size: Math.max(4, data.byteLength), usage, mappedAtCreation: true });
  if (data instanceof Float32Array) new Float32Array(buffer.getMappedRange()).set(data);
  else new Uint32Array(buffer.getMappedRange()).set(data);
  buffer.unmap();
  return buffer;
}
function createStorageBuffer(device: GPUDevice, label: string, bytes: number, usage = 0): GPUBuffer {
  return device.createBuffer({ label, size: Math.max(4, bytes), usage: GPUBufferUsage.STORAGE | usage });
}

export class SquareformerTvmLayerNormQkvWebgpuBlock {
  readonly shape: SquareformerTvmFfnShape;
  private layerNormPipeline: GPUComputePipeline;
  private qkvPipeline: GPUComputePipeline;
  private input: GPUBuffer;
  private normalized: GPUBuffer;
  private output: GPUBuffer;
  private readback: GPUBuffer;
  private layerNormWeight: GPUBuffer;
  private layerNormBias: GPUBuffer;
  private qkvWeight: GPUBuffer;
  private qkvBias: GPUBuffer;
  private layerNormPodArgs: GPUBuffer;
  private qkvPodArgs: GPUBuffer;

  private constructor(private device: GPUDevice, shape: SquareformerTvmFfnShape, layerNormPipeline: GPUComputePipeline, qkvPipeline: GPUComputePipeline, buffers: {
    input: GPUBuffer; normalized: GPUBuffer; output: GPUBuffer; readback: GPUBuffer; layerNormWeight: GPUBuffer; layerNormBias: GPUBuffer; qkvWeight: GPUBuffer; qkvBias: GPUBuffer; layerNormPodArgs: GPUBuffer; qkvPodArgs: GPUBuffer;
  }) {
    this.shape = shape;
    this.layerNormPipeline = layerNormPipeline;
    this.qkvPipeline = qkvPipeline;
    this.input = buffers.input;
    this.normalized = buffers.normalized;
    this.output = buffers.output;
    this.readback = buffers.readback;
    this.layerNormWeight = buffers.layerNormWeight;
    this.layerNormBias = buffers.layerNormBias;
    this.qkvWeight = buffers.qkvWeight;
    this.qkvBias = buffers.qkvBias;
    this.layerNormPodArgs = buffers.layerNormPodArgs;
    this.qkvPodArgs = buffers.qkvPodArgs;
  }

  static async create(device: GPUDevice, kernels: SquareformerTvmLayerNormQkvKernels, weights: SquareformerTvmLayerNormQkvWeights, shape: SquareformerTvmFfnShape, epsilon = 1e-5): Promise<SquareformerTvmLayerNormQkvWebgpuBlock> {
    const { rows, dModel } = shape;
    assertLength('layerNormWeight', weights.layerNormWeight.length, dModel);
    assertLength('layerNormBias', weights.layerNormBias.length, dModel);
    assertLength('qkvWeight', weights.qkvWeight.length, dModel * 3 * dModel);
    assertLength('qkvBias', weights.qkvBias.length, 3 * dModel);
    const layerNormModule = device.createShaderModule({ label: 'squareformer-tvm-n1-layernorm', code: LAYERNORM_WGSL });
    const qkvModule = device.createShaderModule({ label: 'squareformer-tvm-qkv-bias', code: kernels.qkvBias });
    const [layerNormPipeline, qkvPipeline] = await Promise.all([
      device.createComputePipelineAsync({ label: 'squareformer-tvm-n1-layernorm', layout: 'auto', compute: { module: layerNormModule, entryPoint: 'layernorm_kernel' } }),
      device.createComputePipelineAsync({ label: 'squareformer-tvm-qkv-bias', layout: 'auto', compute: { module: qkvModule, entryPoint: 'matmul_kernel' } }),
    ]);
    const inputBytes = rows * dModel * 4;
    const outputBytes = rows * 3 * dModel * 4;
    const layerNormPodData = new Uint32Array(4);
    new Float32Array(layerNormPodData.buffer)[0] = epsilon;
    layerNormPodData[1] = rows;
    layerNormPodData[2] = dModel;
    const buffers = {
      input: createStorageBuffer(device, 'squareformer-tvm-qkv-input', inputBytes, GPUBufferUsage.COPY_DST),
      normalized: createStorageBuffer(device, 'squareformer-tvm-qkv-normalized', inputBytes),
      output: createStorageBuffer(device, 'squareformer-tvm-qkv-output', outputBytes, GPUBufferUsage.COPY_SRC),
      readback: device.createBuffer({ label: 'squareformer-tvm-qkv-readback', size: outputBytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }),
      layerNormWeight: createInitializedBuffer(device, 'squareformer-tvm-n1-ln-weight', weights.layerNormWeight, GPUBufferUsage.STORAGE),
      layerNormBias: createInitializedBuffer(device, 'squareformer-tvm-n1-ln-bias', weights.layerNormBias, GPUBufferUsage.STORAGE),
      qkvWeight: createInitializedBuffer(device, 'squareformer-tvm-qkv-weight', weights.qkvWeight, GPUBufferUsage.STORAGE),
      qkvBias: createInitializedBuffer(device, 'squareformer-tvm-qkv-bias', weights.qkvBias, GPUBufferUsage.STORAGE),
      layerNormPodArgs: createInitializedBuffer(device, 'squareformer-tvm-n1-ln-pod', layerNormPodData, GPUBufferUsage.UNIFORM),
      qkvPodArgs: createInitializedBuffer(device, 'squareformer-tvm-qkv-pod', new Uint32Array([ceilDiv(rows, 32) - 1, 0, 0, 0]), GPUBufferUsage.UNIFORM),
    };
    return new SquareformerTvmLayerNormQkvWebgpuBlock(device, shape, layerNormPipeline, qkvPipeline, buffers);
  }

  upload(input: Float32Array): void {
    const { rows, dModel } = this.shape;
    assertLength('input', input.length, rows * dModel);
    this.device.queue.writeBuffer(this.input, 0, input);
  }

  encode(commandEncoder: GPUCommandEncoder, inputBuffer: GPUBuffer = this.input, outputBuffer: GPUBuffer = this.output): GPUBuffer {
    const { rows, dModel } = this.shape;
    const pass = commandEncoder.beginComputePass({ label: 'squareformer-tvm-n1-qkv' });
    pass.setPipeline(this.layerNormPipeline);
    pass.setBindGroup(0, this.device.createBindGroup({
      label: 'squareformer-tvm-n1-layernorm-bindings',
      layout: this.layerNormPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.normalized } },
        { binding: 1, resource: { buffer: inputBuffer } },
        { binding: 2, resource: { buffer: this.layerNormWeight } },
        { binding: 3, resource: { buffer: this.layerNormBias } },
        { binding: 4, resource: { buffer: this.layerNormPodArgs } },
      ],
    }));
    pass.dispatchWorkgroups(rows, 1, 1);
    pass.setPipeline(this.qkvPipeline);
    pass.setBindGroup(0, this.device.createBindGroup({
      label: 'squareformer-tvm-qkv-bindings',
      layout: this.qkvPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: outputBuffer } },
        { binding: 1, resource: { buffer: this.qkvWeight } },
        { binding: 2, resource: { buffer: this.normalized } },
        { binding: 3, resource: { buffer: this.qkvPodArgs } },
        { binding: 4, resource: { buffer: this.qkvBias } },
      ],
    }));
    pass.dispatchWorkgroups(ceilDiv(rows, 32), ceilDiv(3 * dModel, 32), 1);
    pass.end();
    return outputBuffer;
  }

  copyOutputToReadback(commandEncoder: GPUCommandEncoder): void {
    commandEncoder.copyBufferToBuffer(this.output, 0, this.readback, 0, this.shape.rows * 3 * this.shape.dModel * 4);
  }

  async readOutput(): Promise<Float32Array> {
    await this.readback.mapAsync(GPUMapMode.READ);
    const copy = new Float32Array(this.readback.getMappedRange().slice(0));
    this.readback.unmap();
    return copy;
  }

  async run(input: Float32Array, options: { readback?: boolean } = {}): Promise<Float32Array | GPUBuffer> {
    this.upload(input);
    const commandEncoder = this.device.createCommandEncoder({ label: 'squareformer-tvm-n1-qkv-run' });
    const output = this.encode(commandEncoder);
    if (options.readback !== false) this.copyOutputToReadback(commandEncoder);
    this.device.queue.submit([commandEncoder.finish()]);
    if (options.readback === false) return output;
    return this.readOutput();
  }

  destroy(): void {
    for (const buffer of [this.input, this.normalized, this.output, this.readback, this.layerNormWeight, this.layerNormBias, this.qkvWeight, this.qkvBias, this.layerNormPodArgs, this.qkvPodArgs]) buffer.destroy();
  }
}
