import {
  SquareformerTvmLayerNormQkvWebgpuBlock,
  type SquareformerTvmLayerNormQkvKernels,
  type SquareformerTvmLayerNormQkvWeights,
} from './squareformerTvmLayerNormQkvWebgpu.ts';
import {
  SquareformerTvmAttentionWebgpuBlock,
  type SquareformerTvmAttentionShape,
} from './squareformerTvmAttentionWebgpu.ts';
import {
  SquareformerTvmAttnOutLnFfnWebgpuBlock,
  type SquareformerTvmAttnOutLnFfnKernels,
  type SquareformerTvmAttnOutLnFfnWeights,
} from './squareformerTvmAttnOutLnFfnWebgpu.ts';
import type { SquareformerGpuBuffer, SquareformerTvmFfnShape } from './squareformerTvmFfnWebgpu.ts';

type GPUBuffer = SquareformerGpuBuffer;
type GPUDevice = {
  queue: { submit(commandBuffers: unknown[]): void };
  createCommandEncoder(descriptor?: unknown): GPUCommandEncoder;
};
type GPUCommandEncoder = {
  finish(): unknown;
};

export type SquareformerTvmFullLayerKernels = SquareformerTvmLayerNormQkvKernels & SquareformerTvmAttnOutLnFfnKernels;
export type SquareformerTvmFullLayerWeights = Omit<SquareformerTvmLayerNormQkvWeights, 'layerNormWeight' | 'layerNormBias'> & SquareformerTvmAttnOutLnFfnWeights & {
  n1LayerNormWeight: Float32Array;
  n1LayerNormBias: Float32Array;
};

export type SquareformerTvmFullLayerShape = SquareformerTvmFfnShape & {
  heads: number;
};

export class SquareformerTvmFullLayerWebgpuBlock {
  readonly shape: SquareformerTvmFullLayerShape;
  private qkv: SquareformerTvmLayerNormQkvWebgpuBlock;
  private attention: SquareformerTvmAttentionWebgpuBlock;
  private output: SquareformerTvmAttnOutLnFfnWebgpuBlock;

  private constructor(
    private device: GPUDevice,
    shape: SquareformerTvmFullLayerShape,
    qkv: SquareformerTvmLayerNormQkvWebgpuBlock,
    attention: SquareformerTvmAttentionWebgpuBlock,
    output: SquareformerTvmAttnOutLnFfnWebgpuBlock,
  ) {
    this.shape = shape;
    this.qkv = qkv;
    this.attention = attention;
    this.output = output;
  }

  static async create(device: GPUDevice, kernels: SquareformerTvmFullLayerKernels, weights: SquareformerTvmFullLayerWeights, shape: SquareformerTvmFullLayerShape, epsilons: { n1?: number; n2?: number } = {}): Promise<SquareformerTvmFullLayerWebgpuBlock> {
    const qkvWeights: SquareformerTvmLayerNormQkvWeights = {
      layerNormWeight: weights.n1LayerNormWeight,
      layerNormBias: weights.n1LayerNormBias,
      qkvWeight: weights.qkvWeight,
      qkvBias: weights.qkvBias,
    };
    const [qkv, attention, output] = await Promise.all([
      SquareformerTvmLayerNormQkvWebgpuBlock.create(device as never, kernels, qkvWeights, shape, epsilons.n1 ?? 1e-5),
      SquareformerTvmAttentionWebgpuBlock.create(device as never, shape as SquareformerTvmAttentionShape),
      SquareformerTvmAttnOutLnFfnWebgpuBlock.create(device as never, kernels, weights, shape, epsilons.n2 ?? 1e-5),
    ]);
    return new SquareformerTvmFullLayerWebgpuBlock(device, shape, qkv, attention, output);
  }

  upload(input: Float32Array, attentionBias: Float32Array): void {
    this.qkv.upload(input);
    this.attention.uploadBias(attentionBias);
    this.output.uploadResidual(input);
  }

  uploadAttentionBias(attentionBias: Float32Array): void {
    this.attention.uploadBias(attentionBias);
  }

  encode(commandEncoder: GPUCommandEncoder, inputBuffer?: GPUBuffer): GPUBuffer {
    const qkvBuffer = this.qkv.encode(commandEncoder as never, inputBuffer as never);
    const contextBuffer = this.attention.encode(commandEncoder as never, qkvBuffer as never);
    return this.output.encode(commandEncoder as never, contextBuffer as never, inputBuffer as never) as never;
  }

  copyOutputToReadback(commandEncoder: GPUCommandEncoder): void {
    this.output.copyOutputToReadback(commandEncoder as never);
  }

  async readOutput(): Promise<Float32Array> {
    return this.output.readOutput();
  }

  async run(input: Float32Array, attentionBias: Float32Array, options: { readback?: boolean } = {}): Promise<Float32Array | GPUBuffer> {
    this.upload(input, attentionBias);
    const commandEncoder = this.device.createCommandEncoder({ label: 'squareformer-tvm-full-layer-run' });
    const output = this.encode(commandEncoder);
    if (options.readback !== false) this.copyOutputToReadback(commandEncoder);
    this.device.queue.submit([commandEncoder.finish()]);
    if (options.readback === false) return output;
    return this.readOutput();
  }

  destroy(): void {
    this.qkv.destroy();
    this.attention.destroy();
    this.output.destroy();
  }
}
