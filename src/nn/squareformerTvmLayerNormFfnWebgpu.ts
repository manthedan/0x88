import {
  SquareformerTvmFfnWebgpuBlock,
  type SquareformerGpuBuffer,
  type SquareformerTvmFfnKernels,
  type SquareformerTvmFfnShape,
  type SquareformerTvmFfnWeights,
} from './squareformerTvmFfnWebgpu.ts';

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
declare const GPUBufferUsage: Record<'STORAGE' | 'COPY_DST' | 'UNIFORM', number>;

export type SquareformerTvmLayerNormFfnWeights = SquareformerTvmFfnWeights & {
  layerNormWeight: Float32Array;
  layerNormBias: Float32Array;
};

export type SquareformerTvmLayerNormFfnRunOptions = {
  readback?: boolean;
};

const LAYERNORM_WGSL = `
@group(0) @binding(0) var<storage, read_write> norm_out : array<f32>;
@group(0) @binding(1) var<storage, read> x : array<f32>;
@group(0) @binding(2) var<storage, read> gamma : array<f32>;
@group(0) @binding(3) var<storage, read> beta : array<f32>;

struct PodArgs {
  epsilon: f32,
  rows: u32,
  dModel: u32,
  _pad: u32,
}
@group(0) @binding(4) var<uniform> podArgs : PodArgs;

var<workgroup> scratch : array<f32, 128>;

@compute @workgroup_size(128, 1, 1)
fn layernorm_kernel(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  let row : u32 = blockIdx.x;
  let col : u32 = threadIdx.x;
  let base : u32 = row * podArgs.dModel;
  let value : f32 = x[base + col];
  scratch[col] = value;
  workgroupBarrier();
  var stride : u32 = 64u;
  loop {
    if (col < stride) {
      scratch[col] = scratch[col] + scratch[col + stride];
    }
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
    if (col < stride) {
      scratch[col] = scratch[col] + scratch[col + stride];
    }
    workgroupBarrier();
    if (stride == 1u) { break; }
    stride = stride / 2u;
  }
  let inv_std : f32 = inverseSqrt((scratch[0u] / f32(podArgs.dModel)) + podArgs.epsilon);
  norm_out[base + col] = centered * inv_std * gamma[col] + beta[col];
}
`;

function assertLength(name: string, actual: number, expected: number) {
  if (actual !== expected) throw new Error(`${name} length ${actual} does not match expected ${expected}`);
}

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

export class SquareformerTvmLayerNormFfnWebgpuBlock {
  readonly shape: SquareformerTvmFfnShape;
  private layerNormPipeline: GPUComputePipeline;
  private residual: GPUBuffer;
  private normalized: GPUBuffer;
  private layerNormWeight: GPUBuffer;
  private layerNormBias: GPUBuffer;
  private podArgs: GPUBuffer;
  private ffn: SquareformerTvmFfnWebgpuBlock;

  private constructor(
    private device: GPUDevice,
    shape: SquareformerTvmFfnShape,
    layerNormPipeline: GPUComputePipeline,
    ffn: SquareformerTvmFfnWebgpuBlock,
    buffers: { residual: GPUBuffer; normalized: GPUBuffer; layerNormWeight: GPUBuffer; layerNormBias: GPUBuffer; podArgs: GPUBuffer },
  ) {
    this.shape = shape;
    this.layerNormPipeline = layerNormPipeline;
    this.ffn = ffn;
    this.residual = buffers.residual;
    this.normalized = buffers.normalized;
    this.layerNormWeight = buffers.layerNormWeight;
    this.layerNormBias = buffers.layerNormBias;
    this.podArgs = buffers.podArgs;
  }

  static async create(device: GPUDevice, kernels: SquareformerTvmFfnKernels, weights: SquareformerTvmLayerNormFfnWeights, shape: SquareformerTvmFfnShape, epsilon = 1e-5): Promise<SquareformerTvmLayerNormFfnWebgpuBlock> {
    const { rows, dModel } = shape;
    assertLength('layerNormWeight', weights.layerNormWeight.length, dModel);
    assertLength('layerNormBias', weights.layerNormBias.length, dModel);
    const module = device.createShaderModule({ label: 'squareformer-tvm-layernorm', code: LAYERNORM_WGSL });
    const [layerNormPipeline, ffn] = await Promise.all([
      device.createComputePipelineAsync({ label: 'squareformer-tvm-layernorm', layout: 'auto', compute: { module, entryPoint: 'layernorm_kernel' } }),
      SquareformerTvmFfnWebgpuBlock.create(device, kernels, weights, shape),
    ]);
    const inputBytes = rows * dModel * 4;
    const podData = new Uint32Array(4);
    new Float32Array(podData.buffer)[0] = epsilon;
    podData[1] = rows;
    podData[2] = dModel;
    const buffers = {
      residual: createStorageBuffer(device, 'squareformer-tvm-lnffn-residual', inputBytes, GPUBufferUsage.COPY_DST),
      normalized: createStorageBuffer(device, 'squareformer-tvm-lnffn-normalized', inputBytes),
      layerNormWeight: createInitializedBuffer(device, 'squareformer-tvm-ln-weight', weights.layerNormWeight, GPUBufferUsage.STORAGE),
      layerNormBias: createInitializedBuffer(device, 'squareformer-tvm-ln-bias', weights.layerNormBias, GPUBufferUsage.STORAGE),
      podArgs: createInitializedBuffer(device, 'squareformer-tvm-ln-pod', podData, GPUBufferUsage.UNIFORM),
    };
    return new SquareformerTvmLayerNormFfnWebgpuBlock(device, shape, layerNormPipeline, ffn, buffers);
  }

  upload(residual: Float32Array): void {
    const { rows, dModel } = this.shape;
    assertLength('residual', residual.length, rows * dModel);
    this.device.queue.writeBuffer(this.residual, 0, residual);
  }

  encode(commandEncoder: GPUCommandEncoder, residualBuffer: GPUBuffer = this.residual): GPUBuffer {
    const pass = commandEncoder.beginComputePass({ label: 'squareformer-tvm-layernorm-ffn' });
    pass.setPipeline(this.layerNormPipeline);
    pass.setBindGroup(0, this.device.createBindGroup({
      label: 'squareformer-tvm-layernorm-bindings',
      layout: this.layerNormPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.normalized } },
        { binding: 1, resource: { buffer: residualBuffer } },
        { binding: 2, resource: { buffer: this.layerNormWeight } },
        { binding: 3, resource: { buffer: this.layerNormBias } },
        { binding: 4, resource: { buffer: this.podArgs } },
      ],
    }));
    pass.dispatchWorkgroups(this.shape.rows, 1, 1);
    pass.end();
    return this.ffn.encode(commandEncoder as never, this.normalized as never, residualBuffer as never) as never;
  }

  copyOutputToReadback(commandEncoder: GPUCommandEncoder): void {
    this.ffn.copyOutputToReadback(commandEncoder as never);
  }

  async readOutput(): Promise<Float32Array> {
    return this.ffn.readOutput();
  }

  async run(residual: Float32Array, options: SquareformerTvmLayerNormFfnRunOptions = {}): Promise<Float32Array | GPUBuffer> {
    this.upload(residual);
    const commandEncoder = this.device.createCommandEncoder({ label: 'squareformer-tvm-lnffn-run' });
    const output = this.encode(commandEncoder);
    if (options.readback !== false) this.copyOutputToReadback(commandEncoder);
    this.device.queue.submit([commandEncoder.finish()]);
    if (options.readback === false) return output;
    return this.readOutput();
  }

  destroy(): void {
    for (const buffer of [this.residual, this.normalized, this.layerNormWeight, this.layerNormBias, this.podArgs]) buffer.destroy();
    this.ffn.destroy();
  }
}
