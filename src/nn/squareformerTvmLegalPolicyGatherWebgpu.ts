import type { SquareformerGpuBuffer } from './squareformerTvmFfnWebgpu.ts';

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

const LEGAL_GATHER_WGSL = `
@group(0) @binding(0) var<storage, read_write> legalPolicy : array<f32>;
@group(0) @binding(1) var<storage, read> policy : array<f32>;
@group(0) @binding(2) var<storage, read> indices : array<u32>;
struct PodArgs { legalCount: u32, policySize: u32, _pad0: u32, _pad1: u32 }
@group(0) @binding(3) var<uniform> podArgs : PodArgs;

@compute @workgroup_size(64, 1, 1)
fn legal_policy_gather_kernel(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i : u32 = gid.x;
  if (i >= podArgs.legalCount) { return; }
  let policyIndex : u32 = indices[i];
  legalPolicy[i] = select(-100.0f, policy[policyIndex], policyIndex < podArgs.policySize);
}
`;

function ceilDiv(a: number, b: number): number { return Math.floor((a + b - 1) / b); }
function assertLength(name: string, actual: number, expected: number) { if (actual !== expected) throw new Error(`${name} length ${actual} does not match expected ${expected}`); }
function validateLegalIndices(legalIndices: Uint32Array, policySize: number) { for (const index of legalIndices) if (index >= policySize) throw new Error(`legal policy index ${index} is out of range for policySize=${policySize}`); }
function createInitializedBuffer(device: GPUDevice, label: string, data: Uint32Array, usage: number): GPUBuffer {
  const buffer = device.createBuffer({ label, size: Math.max(4, data.byteLength), usage, mappedAtCreation: true });
  new Uint32Array(buffer.getMappedRange()).set(data);
  buffer.unmap();
  return buffer;
}
function createStorageBuffer(device: GPUDevice, label: string, bytes: number, usage = 0): GPUBuffer {
  return device.createBuffer({ label, size: Math.max(4, bytes), usage: GPUBufferUsage.STORAGE | usage });
}

export class SquareformerTvmLegalPolicyGatherWebgpuBlock {
  legalCount: number;
  readonly policySize: number;
  readonly maxLegalCount: number;
  private pipeline: GPUComputePipeline;
  private policy: GPUBuffer;
  private indices: GPUBuffer;
  private output: GPUBuffer;
  private readback: GPUBuffer;
  private podArgs: GPUBuffer;

  private constructor(private device: GPUDevice, pipeline: GPUComputePipeline, legalCount: number, maxLegalCount: number, policySize: number, buffers: { policy: GPUBuffer; indices: GPUBuffer; output: GPUBuffer; readback: GPUBuffer; podArgs: GPUBuffer }) {
    this.pipeline = pipeline;
    this.legalCount = legalCount;
    this.maxLegalCount = maxLegalCount;
    this.policySize = policySize;
    this.policy = buffers.policy;
    this.indices = buffers.indices;
    this.output = buffers.output;
    this.readback = buffers.readback;
    this.podArgs = buffers.podArgs;
  }

  static async create(device: GPUDevice, legalIndices: Uint32Array, policySize = 20480): Promise<SquareformerTvmLegalPolicyGatherWebgpuBlock> {
    if (legalIndices.length <= 0) throw new Error('legalIndices must be non-empty');
    validateLegalIndices(legalIndices, policySize);
    const block = await SquareformerTvmLegalPolicyGatherWebgpuBlock.createDynamic(device, legalIndices.length, policySize);
    block.uploadIndices(legalIndices);
    return block;
  }

  static async createDynamic(device: GPUDevice, maxLegalCount = 256, policySize = 20480): Promise<SquareformerTvmLegalPolicyGatherWebgpuBlock> {
    if (maxLegalCount <= 0) throw new Error('maxLegalCount must be positive');
    const module = device.createShaderModule({ label: 'squareformer-tvm-legal-policy-gather', code: LEGAL_GATHER_WGSL });
    const pipeline = await device.createComputePipelineAsync({ label: 'squareformer-tvm-legal-policy-gather', layout: 'auto', compute: { module, entryPoint: 'legal_policy_gather_kernel' } });
    const outputBytes = maxLegalCount * 4;
    const buffers = {
      policy: createStorageBuffer(device, 'squareformer-tvm-legal-policy-input', policySize * 4, GPUBufferUsage.COPY_DST),
      indices: createInitializedBuffer(device, 'squareformer-tvm-legal-policy-indices', new Uint32Array(maxLegalCount), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST),
      output: createStorageBuffer(device, 'squareformer-tvm-legal-policy-output', outputBytes, GPUBufferUsage.COPY_SRC),
      readback: device.createBuffer({ label: 'squareformer-tvm-legal-policy-readback', size: outputBytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }),
      podArgs: createInitializedBuffer(device, 'squareformer-tvm-legal-policy-pod', new Uint32Array([0, policySize, 0, 0]), GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST),
    };
    return new SquareformerTvmLegalPolicyGatherWebgpuBlock(device, pipeline, 0, maxLegalCount, policySize, buffers);
  }

  uploadIndices(legalIndices: Uint32Array): void {
    if (legalIndices.length <= 0) throw new Error('legalIndices must be non-empty');
    if (legalIndices.length > this.maxLegalCount) throw new Error(`legalIndices length ${legalIndices.length} exceeds maxLegalCount=${this.maxLegalCount}`);
    validateLegalIndices(legalIndices, this.policySize);
    this.legalCount = legalIndices.length;
    this.device.queue.writeBuffer(this.indices, 0, legalIndices);
    this.device.queue.writeBuffer(this.podArgs, 0, new Uint32Array([this.legalCount, this.policySize, 0, 0]));
  }

  upload(policy: Float32Array): void {
    assertLength('policy', policy.length, this.policySize);
    this.device.queue.writeBuffer(this.policy, 0, policy);
  }

  encode(commandEncoder: GPUCommandEncoder, policyBuffer: GPUBuffer = this.policy, outputBuffer: GPUBuffer = this.output): GPUBuffer {
    const pass = commandEncoder.beginComputePass({ label: 'squareformer-tvm-legal-policy-gather' });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.device.createBindGroup({
      label: 'squareformer-tvm-legal-policy-gather-bindings',
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: outputBuffer } },
        { binding: 1, resource: { buffer: policyBuffer } },
        { binding: 2, resource: { buffer: this.indices } },
        { binding: 3, resource: { buffer: this.podArgs } },
      ],
    }));
    pass.dispatchWorkgroups(ceilDiv(this.legalCount, 64), 1, 1);
    pass.end();
    return outputBuffer;
  }

  copyOutputToReadback(commandEncoder: GPUCommandEncoder): void {
    commandEncoder.copyBufferToBuffer(this.output, 0, this.readback, 0, this.legalCount * 4);
  }

  async readOutput(): Promise<Float32Array> {
    await this.readback.mapAsync(GPUMapMode.READ);
    const copy = new Float32Array(this.readback.getMappedRange().slice(0, this.legalCount * 4));
    this.readback.unmap();
    return copy;
  }

  async run(policy: Float32Array, options: { readback?: boolean } = {}): Promise<Float32Array | GPUBuffer> {
    this.upload(policy);
    const commandEncoder = this.device.createCommandEncoder({ label: 'squareformer-tvm-legal-policy-gather-run' });
    const output = this.encode(commandEncoder);
    if (options.readback !== false) this.copyOutputToReadback(commandEncoder);
    this.device.queue.submit([commandEncoder.finish()]);
    if (options.readback === false) return output;
    return this.readOutput();
  }

  destroy(): void {
    for (const buffer of [this.policy, this.indices, this.output, this.readback, this.podArgs]) buffer.destroy();
  }
}
