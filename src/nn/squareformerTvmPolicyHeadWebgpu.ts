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

export type SquareformerTvmPolicyHeadKernels = {
  policyPairProj: string;
  policyPromoProj: string;
};

export type SquareformerTvmPolicyHeadWeights = {
  policyFromWeight: Float32Array;
  policyFromBias: Float32Array;
  policyToWeight: Float32Array;
  policyToBias: Float32Array;
  policyPromoWeight: Float32Array;
  policyPromoBias: Float32Array;
};

export type SquareformerTvmPolicyHeadShape = SquareformerTvmFfnShape & {
  pairDim: number;
  promoDim: number;
};

const DEFAULT_SHAPE: SquareformerTvmPolicyHeadShape = { rows: 64, dModel: 128, dFf: 256, pairDim: 128, promoDim: 256 };
const POLICY_PAIR_SCALE = 1.0 / 11.313708305358887;

const POLICY_CONCAT_WGSL = `
@group(0) @binding(0) var<storage, read_write> policy : array<f32>;
@group(0) @binding(1) var<storage, read> fromProj : array<f32>;
@group(0) @binding(2) var<storage, read> toProj : array<f32>;
@group(0) @binding(3) var<storage, read> promoProj : array<f32>;
struct PodArgs { rows: u32, pairDim: u32, promoDim: u32, _pad: u32 }
@group(0) @binding(4) var<uniform> podArgs : PodArgs;

@compute @workgroup_size(16, 16, 1)
fn policy_concat_kernel(@builtin(global_invocation_id) gid : vec3<u32>) {
  let col : u32 = gid.x;
  let row : u32 = gid.y;
  if (row >= podArgs.rows || col >= podArgs.promoDim) { return; }
  if (col < podArgs.rows) {
    var acc : f32 = 0.0f;
    for (var k : u32 = 0u; k < 128u; k = k + 1u) {
      acc = acc + fromProj[row * podArgs.pairDim + k] * toProj[col * podArgs.pairDim + k];
    }
    policy[row * podArgs.rows + col] = acc * ${POLICY_PAIR_SCALE.toExponential(15)}f;
  }
  policy[podArgs.rows * podArgs.rows + row * podArgs.promoDim + col] = promoProj[row * podArgs.promoDim + col];
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

export class SquareformerTvmPolicyHeadWebgpuBlock {
  readonly shape: SquareformerTvmPolicyHeadShape;
  private pairPipeline: GPUComputePipeline;
  private promoPipeline: GPUComputePipeline;
  private concatPipeline: GPUComputePipeline;
  private input: GPUBuffer;
  private fromProj: GPUBuffer;
  private toProj: GPUBuffer;
  private promoProj: GPUBuffer;
  private policy: GPUBuffer;
  private readback: GPUBuffer;
  private policyFromWeight: GPUBuffer;
  private policyFromBias: GPUBuffer;
  private policyToWeight: GPUBuffer;
  private policyToBias: GPUBuffer;
  private policyPromoWeight: GPUBuffer;
  private policyPromoBias: GPUBuffer;
  private matmulPodArgs: GPUBuffer;
  private concatPodArgs: GPUBuffer;

  private constructor(private device: GPUDevice, shape: SquareformerTvmPolicyHeadShape, pipelines: { pair: GPUComputePipeline; promo: GPUComputePipeline; concat: GPUComputePipeline }, buffers: {
    input: GPUBuffer; fromProj: GPUBuffer; toProj: GPUBuffer; promoProj: GPUBuffer; policy: GPUBuffer; readback: GPUBuffer;
    policyFromWeight: GPUBuffer; policyFromBias: GPUBuffer; policyToWeight: GPUBuffer; policyToBias: GPUBuffer; policyPromoWeight: GPUBuffer; policyPromoBias: GPUBuffer; matmulPodArgs: GPUBuffer; concatPodArgs: GPUBuffer;
  }) {
    this.shape = shape;
    this.pairPipeline = pipelines.pair;
    this.promoPipeline = pipelines.promo;
    this.concatPipeline = pipelines.concat;
    this.input = buffers.input;
    this.fromProj = buffers.fromProj;
    this.toProj = buffers.toProj;
    this.promoProj = buffers.promoProj;
    this.policy = buffers.policy;
    this.readback = buffers.readback;
    this.policyFromWeight = buffers.policyFromWeight;
    this.policyFromBias = buffers.policyFromBias;
    this.policyToWeight = buffers.policyToWeight;
    this.policyToBias = buffers.policyToBias;
    this.policyPromoWeight = buffers.policyPromoWeight;
    this.policyPromoBias = buffers.policyPromoBias;
    this.matmulPodArgs = buffers.matmulPodArgs;
    this.concatPodArgs = buffers.concatPodArgs;
  }

  static async create(device: GPUDevice, kernels: SquareformerTvmPolicyHeadKernels, weights: SquareformerTvmPolicyHeadWeights, shape: Partial<SquareformerTvmPolicyHeadShape> = {}): Promise<SquareformerTvmPolicyHeadWebgpuBlock> {
    const fullShape = { ...DEFAULT_SHAPE, ...shape };
    const { rows, dModel, pairDim, promoDim } = fullShape;
    if (rows !== 64 || dModel !== 128 || pairDim !== 128 || promoDim !== 256) throw new Error(`policy shader expects rows=64,dModel=128,pairDim=128,promoDim=256; got ${JSON.stringify(fullShape)}`);
    assertLength('policyFromWeight', weights.policyFromWeight.length, dModel * pairDim);
    assertLength('policyFromBias', weights.policyFromBias.length, pairDim);
    assertLength('policyToWeight', weights.policyToWeight.length, dModel * pairDim);
    assertLength('policyToBias', weights.policyToBias.length, pairDim);
    assertLength('policyPromoWeight', weights.policyPromoWeight.length, dModel * promoDim);
    assertLength('policyPromoBias', weights.policyPromoBias.length, promoDim);
    const pairModule = device.createShaderModule({ label: 'squareformer-tvm-policy-pair-proj', code: kernels.policyPairProj });
    const promoModule = device.createShaderModule({ label: 'squareformer-tvm-policy-promo-proj', code: kernels.policyPromoProj });
    const concatModule = device.createShaderModule({ label: 'squareformer-tvm-policy-concat', code: POLICY_CONCAT_WGSL });
    const [pairPipeline, promoPipeline, concatPipeline] = await Promise.all([
      device.createComputePipelineAsync({ label: 'squareformer-tvm-policy-pair-proj', layout: 'auto', compute: { module: pairModule, entryPoint: 'matmul_kernel' } }),
      device.createComputePipelineAsync({ label: 'squareformer-tvm-policy-promo-proj', layout: 'auto', compute: { module: promoModule, entryPoint: 'matmul_kernel' } }),
      device.createComputePipelineAsync({ label: 'squareformer-tvm-policy-concat', layout: 'auto', compute: { module: concatModule, entryPoint: 'policy_concat_kernel' } }),
    ]);
    const inputBytes = rows * dModel * 4;
    const pairBytes = rows * pairDim * 4;
    const promoBytes = rows * promoDim * 4;
    const policyBytes = (rows * rows + rows * promoDim) * 4;
    const buffers = {
      input: createStorageBuffer(device, 'squareformer-tvm-policy-input', inputBytes, GPUBufferUsage.COPY_DST),
      fromProj: createStorageBuffer(device, 'squareformer-tvm-policy-from-proj', pairBytes),
      toProj: createStorageBuffer(device, 'squareformer-tvm-policy-to-proj', pairBytes),
      promoProj: createStorageBuffer(device, 'squareformer-tvm-policy-promo-proj', promoBytes),
      policy: createStorageBuffer(device, 'squareformer-tvm-policy-output', policyBytes, GPUBufferUsage.COPY_SRC),
      readback: device.createBuffer({ label: 'squareformer-tvm-policy-readback', size: policyBytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }),
      policyFromWeight: createInitializedBuffer(device, 'squareformer-tvm-policy-from-weight', weights.policyFromWeight, GPUBufferUsage.STORAGE),
      policyFromBias: createInitializedBuffer(device, 'squareformer-tvm-policy-from-bias', weights.policyFromBias, GPUBufferUsage.STORAGE),
      policyToWeight: createInitializedBuffer(device, 'squareformer-tvm-policy-to-weight', weights.policyToWeight, GPUBufferUsage.STORAGE),
      policyToBias: createInitializedBuffer(device, 'squareformer-tvm-policy-to-bias', weights.policyToBias, GPUBufferUsage.STORAGE),
      policyPromoWeight: createInitializedBuffer(device, 'squareformer-tvm-policy-promo-weight', weights.policyPromoWeight, GPUBufferUsage.STORAGE),
      policyPromoBias: createInitializedBuffer(device, 'squareformer-tvm-policy-promo-bias', weights.policyPromoBias, GPUBufferUsage.STORAGE),
      matmulPodArgs: createInitializedBuffer(device, 'squareformer-tvm-policy-matmul-pod', new Uint32Array([ceilDiv(rows, 32) - 1, 0, 0, 0]), GPUBufferUsage.UNIFORM),
      concatPodArgs: createInitializedBuffer(device, 'squareformer-tvm-policy-concat-pod', new Uint32Array([rows, pairDim, promoDim, 0]), GPUBufferUsage.UNIFORM),
    };
    return new SquareformerTvmPolicyHeadWebgpuBlock(device, fullShape, { pair: pairPipeline, promo: promoPipeline, concat: concatPipeline }, buffers);
  }

  upload(input: Float32Array): void {
    const { rows, dModel } = this.shape;
    assertLength('input', input.length, rows * dModel);
    this.device.queue.writeBuffer(this.input, 0, input);
  }

  encode(commandEncoder: GPUCommandEncoder, inputBuffer: GPUBuffer = this.input, outputBuffer: GPUBuffer = this.policy): GPUBuffer {
    const { rows, pairDim, promoDim } = this.shape;
    const pass = commandEncoder.beginComputePass({ label: 'squareformer-tvm-policy-head' });
    pass.setPipeline(this.pairPipeline);
    pass.setBindGroup(0, this.device.createBindGroup({
      label: 'squareformer-tvm-policy-from-bindings',
      layout: this.pairPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.fromProj } },
        { binding: 1, resource: { buffer: this.policyFromWeight } },
        { binding: 2, resource: { buffer: inputBuffer } },
        { binding: 3, resource: { buffer: this.matmulPodArgs } },
        { binding: 4, resource: { buffer: this.policyFromBias } },
      ],
    }));
    pass.dispatchWorkgroups(ceilDiv(rows, 32), ceilDiv(pairDim, 32), 1);
    pass.setBindGroup(0, this.device.createBindGroup({
      label: 'squareformer-tvm-policy-to-bindings',
      layout: this.pairPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.toProj } },
        { binding: 1, resource: { buffer: this.policyToWeight } },
        { binding: 2, resource: { buffer: inputBuffer } },
        { binding: 3, resource: { buffer: this.matmulPodArgs } },
        { binding: 4, resource: { buffer: this.policyToBias } },
      ],
    }));
    pass.dispatchWorkgroups(ceilDiv(rows, 32), ceilDiv(pairDim, 32), 1);
    pass.setPipeline(this.promoPipeline);
    pass.setBindGroup(0, this.device.createBindGroup({
      label: 'squareformer-tvm-policy-promo-bindings',
      layout: this.promoPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.promoProj } },
        { binding: 1, resource: { buffer: this.policyPromoWeight } },
        { binding: 2, resource: { buffer: inputBuffer } },
        { binding: 3, resource: { buffer: this.matmulPodArgs } },
        { binding: 4, resource: { buffer: this.policyPromoBias } },
      ],
    }));
    pass.dispatchWorkgroups(ceilDiv(rows, 32), ceilDiv(promoDim, 32), 1);
    pass.setPipeline(this.concatPipeline);
    pass.setBindGroup(0, this.device.createBindGroup({
      label: 'squareformer-tvm-policy-concat-bindings',
      layout: this.concatPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: outputBuffer } },
        { binding: 1, resource: { buffer: this.fromProj } },
        { binding: 2, resource: { buffer: this.toProj } },
        { binding: 3, resource: { buffer: this.promoProj } },
        { binding: 4, resource: { buffer: this.concatPodArgs } },
      ],
    }));
    pass.dispatchWorkgroups(ceilDiv(promoDim, 16), ceilDiv(rows, 16), 1);
    pass.end();
    return outputBuffer;
  }

  copyOutputToReadback(commandEncoder: GPUCommandEncoder): void {
    commandEncoder.copyBufferToBuffer(this.policy, 0, this.readback, 0, (this.shape.rows * this.shape.rows + this.shape.rows * this.shape.promoDim) * 4);
  }

  async readOutput(): Promise<Float32Array> {
    await this.readback.mapAsync(GPUMapMode.READ);
    const copy = new Float32Array(this.readback.getMappedRange().slice(0));
    this.readback.unmap();
    return copy;
  }

  async run(input: Float32Array, options: { readback?: boolean } = {}): Promise<Float32Array | GPUBuffer> {
    this.upload(input);
    const commandEncoder = this.device.createCommandEncoder({ label: 'squareformer-tvm-policy-head-run' });
    const output = this.encode(commandEncoder);
    if (options.readback !== false) this.copyOutputToReadback(commandEncoder);
    this.device.queue.submit([commandEncoder.finish()]);
    if (options.readback === false) return output;
    return this.readOutput();
  }

  destroy(): void {
    for (const buffer of [this.input, this.fromProj, this.toProj, this.promoProj, this.policy, this.readback, this.policyFromWeight, this.policyFromBias, this.policyToWeight, this.policyToBias, this.policyPromoWeight, this.policyPromoBias, this.matmulPodArgs, this.concatPodArgs]) buffer.destroy();
  }
}
