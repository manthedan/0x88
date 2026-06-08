import {
  SquareformerTvmLayerNormFfnWebgpuBlock,
  type SquareformerTvmLayerNormFfnWeights,
} from './squareformerTvmLayerNormFfnWebgpu.ts';
import type { SquareformerGpuBuffer, SquareformerTvmFfnKernels, SquareformerTvmFfnShape } from './squareformerTvmFfnWebgpu.ts';

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

export type SquareformerTvmAttnOutLnFfnKernels = SquareformerTvmFfnKernels & {
  attnOutResidual: string;
};

export type SquareformerTvmAttnOutLnFfnWeights = SquareformerTvmLayerNormFfnWeights & {
  attnOutWeight: Float32Array;
  attnOutBias: Float32Array;
};

function ceilDiv(a: number, b: number): number {
  return Math.floor((a + b - 1) / b);
}

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

export class SquareformerTvmAttnOutLnFfnWebgpuBlock {
  readonly shape: SquareformerTvmFfnShape;
  private attnOutPipeline: GPUComputePipeline;
  private attnContext: GPUBuffer;
  private attnResidual: GPUBuffer;
  private attnOut: GPUBuffer;
  private attnOutWeight: GPUBuffer;
  private attnOutBias: GPUBuffer;
  private podArgs: GPUBuffer;
  private lnFfn: SquareformerTvmLayerNormFfnWebgpuBlock;

  private constructor(
    private device: GPUDevice,
    shape: SquareformerTvmFfnShape,
    attnOutPipeline: GPUComputePipeline,
    lnFfn: SquareformerTvmLayerNormFfnWebgpuBlock,
    buffers: {
      attnContext: GPUBuffer;
      attnResidual: GPUBuffer;
      attnOut: GPUBuffer;
      attnOutWeight: GPUBuffer;
      attnOutBias: GPUBuffer;
      podArgs: GPUBuffer;
    },
  ) {
    this.shape = shape;
    this.attnOutPipeline = attnOutPipeline;
    this.lnFfn = lnFfn;
    this.attnContext = buffers.attnContext;
    this.attnResidual = buffers.attnResidual;
    this.attnOut = buffers.attnOut;
    this.attnOutWeight = buffers.attnOutWeight;
    this.attnOutBias = buffers.attnOutBias;
    this.podArgs = buffers.podArgs;
  }

  static async create(device: GPUDevice, kernels: SquareformerTvmAttnOutLnFfnKernels, weights: SquareformerTvmAttnOutLnFfnWeights, shape: SquareformerTvmFfnShape, epsilon = 1e-5): Promise<SquareformerTvmAttnOutLnFfnWebgpuBlock> {
    const { rows, dModel } = shape;
    assertLength('attnOutWeight', weights.attnOutWeight.length, dModel * dModel);
    assertLength('attnOutBias', weights.attnOutBias.length, dModel);
    const attnOutModule = device.createShaderModule({ label: 'squareformer-tvm-attn-out-residual', code: kernels.attnOutResidual });
    const [attnOutPipeline, lnFfn] = await Promise.all([
      device.createComputePipelineAsync({ label: 'squareformer-tvm-attn-out-residual', layout: 'auto', compute: { module: attnOutModule, entryPoint: 'matmul_kernel' } }),
      SquareformerTvmLayerNormFfnWebgpuBlock.create(device, kernels, weights, shape, epsilon),
    ]);
    const bytes = rows * dModel * 4;
    const buffers = {
      attnContext: createStorageBuffer(device, 'squareformer-tvm-attn-context', bytes, GPUBufferUsage.COPY_DST),
      attnResidual: createStorageBuffer(device, 'squareformer-tvm-attn-residual', bytes, GPUBufferUsage.COPY_DST),
      attnOut: createStorageBuffer(device, 'squareformer-tvm-attn-out', bytes),
      attnOutWeight: createInitializedBuffer(device, 'squareformer-tvm-attn-out-weight', weights.attnOutWeight, GPUBufferUsage.STORAGE),
      attnOutBias: createInitializedBuffer(device, 'squareformer-tvm-attn-out-bias', weights.attnOutBias, GPUBufferUsage.STORAGE),
      podArgs: createInitializedBuffer(device, 'squareformer-tvm-attn-out-pod-args', new Uint32Array([ceilDiv(rows, 32) - 1, 0, 0, 0]), GPUBufferUsage.UNIFORM),
    };
    return new SquareformerTvmAttnOutLnFfnWebgpuBlock(device, shape, attnOutPipeline, lnFfn, buffers);
  }

  upload(attnContext: Float32Array, attnResidual: Float32Array): void {
    const { rows, dModel } = this.shape;
    assertLength('attnContext', attnContext.length, rows * dModel);
    this.device.queue.writeBuffer(this.attnContext, 0, attnContext);
    this.uploadResidual(attnResidual);
  }

  uploadResidual(attnResidual: Float32Array): void {
    const { rows, dModel } = this.shape;
    assertLength('attnResidual', attnResidual.length, rows * dModel);
    this.device.queue.writeBuffer(this.attnResidual, 0, attnResidual);
  }

  encode(commandEncoder: GPUCommandEncoder, attnContextBuffer: GPUBuffer = this.attnContext, attnResidualBuffer: GPUBuffer = this.attnResidual): GPUBuffer {
    const { rows, dModel } = this.shape;
    const pass = commandEncoder.beginComputePass({ label: 'squareformer-tvm-attn-out-ln-ffn' });
    pass.setPipeline(this.attnOutPipeline);
    pass.setBindGroup(0, this.device.createBindGroup({
      label: 'squareformer-tvm-attn-out-bindings',
      layout: this.attnOutPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.attnOut } },
        { binding: 1, resource: { buffer: this.attnOutWeight } },
        { binding: 2, resource: { buffer: attnContextBuffer } },
        { binding: 3, resource: { buffer: this.podArgs } },
        { binding: 4, resource: { buffer: this.attnOutBias } },
        { binding: 5, resource: { buffer: attnResidualBuffer } },
      ],
    }));
    pass.dispatchWorkgroups(ceilDiv(rows, 32), ceilDiv(dModel, 32), 1);
    pass.end();
    return this.lnFfn.encode(commandEncoder as never, this.attnOut as never) as never;
  }

  copyOutputToReadback(commandEncoder: GPUCommandEncoder): void {
    this.lnFfn.copyOutputToReadback(commandEncoder as never);
  }

  async readOutput(): Promise<Float32Array> {
    return this.lnFfn.readOutput();
  }

  async run(attnContext: Float32Array, attnResidual: Float32Array, options: { readback?: boolean } = {}): Promise<Float32Array | GPUBuffer> {
    this.upload(attnContext, attnResidual);
    const commandEncoder = this.device.createCommandEncoder({ label: 'squareformer-tvm-attn-out-ln-ffn-run' });
    const output = this.encode(commandEncoder);
    if (options.readback !== false) this.copyOutputToReadback(commandEncoder);
    this.device.queue.submit([commandEncoder.finish()]);
    if (options.readback === false) return output;
    return this.readOutput();
  }

  destroy(): void {
    for (const buffer of [this.attnContext, this.attnResidual, this.attnOut, this.attnOutWeight, this.attnOutBias, this.podArgs]) buffer.destroy();
    this.lnFfn.destroy();
  }
}
