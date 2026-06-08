type GPUBufferUsageFlags = number;
export type SquareformerGpuBuffer = {
  destroy(): void;
  getMappedRange(): ArrayBuffer;
  mapAsync(mode: number): Promise<void>;
  unmap(): void;
};
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

export type SquareformerTvmFfnShape = {
  rows: number;
  dModel: number;
  dFf: number;
};

export type SquareformerTvmFfnKernels = {
  dense1Gelu: string;
  dense2Residual: string;
};

export type SquareformerTvmFfnWeights = {
  dense1Weight: Float32Array;
  dense1Bias: Float32Array;
  dense2Weight: Float32Array;
  dense2Bias: Float32Array;
};

export type SquareformerTvmFfnRunOptions = {
  readback?: boolean;
  residual?: Float32Array;
};

const DEFAULT_SHAPE: SquareformerTvmFfnShape = { rows: 64, dModel: 128, dFf: 256 };

type StorageBuffer = GPUBuffer & { size?: number };

function ceilDiv(a: number, b: number): number {
  return Math.floor((a + b - 1) / b);
}

function assertLength(name: string, actual: number, expected: number) {
  if (actual !== expected) throw new Error(`${name} length ${actual} does not match expected ${expected}`);
}

function createInitializedBuffer(device: GPUDevice, label: string, data: Float32Array | Uint32Array, usage: GPUBufferUsageFlags): GPUBuffer {
  const buffer = device.createBuffer({ label, size: Math.max(4, data.byteLength), usage, mappedAtCreation: true });
  if (data instanceof Float32Array) new Float32Array(buffer.getMappedRange()).set(data);
  else new Uint32Array(buffer.getMappedRange()).set(data);
  buffer.unmap();
  return buffer;
}

function createStorageBuffer(device: GPUDevice, label: string, bytes: number, usage: GPUBufferUsageFlags = 0): GPUBuffer {
  return device.createBuffer({
    label,
    size: Math.max(4, bytes),
    usage: GPUBufferUsage.STORAGE | usage,
  });
}

function approxErf(input: number): number {
  const x = Math.max(-4, Math.min(4, input));
  const x2 = x * x;
  const numerator = x * ((((((-2.726142e-10 * x2 + 2.770681e-8) * x2 - 2.101024e-6) * x2 - 5.692506e-5) * x2 - 7.349906e-4) * x2 - 2.954600e-3) * x2 - 1.609603e-2);
  const denominator = ((((-1.456607e-5 * x2 - 2.133740e-4) * x2 - 1.682827e-3) * x2 - 7.373329e-3) * x2 - 1.426474e-2);
  return numerator / denominator;
}

export function squareformerTvmFfnCpuReference(input: Float32Array, weights: SquareformerTvmFfnWeights, shape: SquareformerTvmFfnShape = DEFAULT_SHAPE, residual: Float32Array = input): Float32Array {
  const { rows, dModel, dFf } = shape;
  assertLength('input', input.length, rows * dModel);
  assertLength('dense1Weight', weights.dense1Weight.length, dModel * dFf);
  assertLength('dense1Bias', weights.dense1Bias.length, dFf);
  assertLength('dense2Weight', weights.dense2Weight.length, dFf * dModel);
  assertLength('dense2Bias', weights.dense2Bias.length, dModel);
  assertLength('residual', residual.length, rows * dModel);
  const hidden = new Float32Array(rows * dFf);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < dFf; c++) {
      let acc = weights.dense1Bias[c] ?? 0;
      for (let k = 0; k < dModel; k++) acc += (input[r * dModel + k] ?? 0) * (weights.dense1Weight[k * dFf + c] ?? 0);
      hidden[r * dFf + c] = acc * (0.5 * (1 + approxErf(acc * Math.SQRT1_2)));
    }
  }
  const out = new Float32Array(rows * dModel);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < dModel; c++) {
      let acc = weights.dense2Bias[c] ?? 0;
      for (let k = 0; k < dFf; k++) acc += (hidden[r * dFf + k] ?? 0) * (weights.dense2Weight[k * dModel + c] ?? 0);
      out[r * dModel + c] = acc + (residual[r * dModel + c] ?? 0);
    }
  }
  return out;
}

export function makeSquareformerTvmFfnFixture(shape: SquareformerTvmFfnShape = DEFAULT_SHAPE): { input: Float32Array; weights: SquareformerTvmFfnWeights } {
  const { rows, dModel, dFf } = shape;
  const input = new Float32Array(rows * dModel);
  const dense1Weight = new Float32Array(dModel * dFf);
  const dense1Bias = new Float32Array(dFf);
  const dense2Weight = new Float32Array(dFf * dModel);
  const dense2Bias = new Float32Array(dModel);
  for (let i = 0; i < input.length; i++) input[i] = Math.sin(i * 0.013) * 0.25;
  for (let i = 0; i < dense1Weight.length; i++) dense1Weight[i] = Math.cos(i * 0.017) * 0.05;
  for (let i = 0; i < dense1Bias.length; i++) dense1Bias[i] = Math.sin(i * 0.019) * 0.1;
  for (let i = 0; i < dense2Weight.length; i++) dense2Weight[i] = Math.cos(i * 0.011) * 0.04;
  for (let i = 0; i < dense2Bias.length; i++) dense2Bias[i] = Math.sin(i * 0.023) * 0.05;
  return { input, weights: { dense1Weight, dense1Bias, dense2Weight, dense2Bias } };
}

export function maxAbsDiff(a: Float32Array, b: Float32Array): number {
  if (a.length !== b.length) throw new Error(`length mismatch ${a.length} != ${b.length}`);
  let max = 0;
  for (let i = 0; i < a.length; i++) max = Math.max(max, Math.abs((a[i] ?? 0) - (b[i] ?? 0)));
  return max;
}

export class SquareformerTvmFfnWebgpuBlock {
  readonly shape: SquareformerTvmFfnShape;
  private dense1Pipeline: GPUComputePipeline;
  private dense2Pipeline: GPUComputePipeline;
  private dense1Weight: GPUBuffer;
  private dense1Bias: GPUBuffer;
  private dense2Weight: GPUBuffer;
  private dense2Bias: GPUBuffer;
  private podArgs: GPUBuffer;
  private input: GPUBuffer;
  private residual: GPUBuffer;
  private hidden: GPUBuffer;
  private output: GPUBuffer;
  private readback: GPUBuffer;

  private constructor(
    private device: GPUDevice,
    shape: SquareformerTvmFfnShape,
    dense1Pipeline: GPUComputePipeline,
    dense2Pipeline: GPUComputePipeline,
    buffers: {
      dense1Weight: GPUBuffer;
      dense1Bias: GPUBuffer;
      dense2Weight: GPUBuffer;
      dense2Bias: GPUBuffer;
      podArgs: GPUBuffer;
      input: GPUBuffer;
      residual: GPUBuffer;
      hidden: GPUBuffer;
      output: GPUBuffer;
      readback: GPUBuffer;
    },
  ) {
    this.shape = shape;
    this.dense1Pipeline = dense1Pipeline;
    this.dense2Pipeline = dense2Pipeline;
    this.dense1Weight = buffers.dense1Weight;
    this.dense1Bias = buffers.dense1Bias;
    this.dense2Weight = buffers.dense2Weight;
    this.dense2Bias = buffers.dense2Bias;
    this.podArgs = buffers.podArgs;
    this.input = buffers.input;
    this.residual = buffers.residual;
    this.hidden = buffers.hidden;
    this.output = buffers.output;
    this.readback = buffers.readback;
  }

  static async create(device: GPUDevice, kernels: SquareformerTvmFfnKernels, weights: SquareformerTvmFfnWeights, shape: Partial<SquareformerTvmFfnShape> = {}): Promise<SquareformerTvmFfnWebgpuBlock> {
    const fullShape = { ...DEFAULT_SHAPE, ...shape };
    const { rows, dModel, dFf } = fullShape;
    assertLength('dense1Weight', weights.dense1Weight.length, dModel * dFf);
    assertLength('dense1Bias', weights.dense1Bias.length, dFf);
    assertLength('dense2Weight', weights.dense2Weight.length, dFf * dModel);
    assertLength('dense2Bias', weights.dense2Bias.length, dModel);
    const dense1Module = device.createShaderModule({ label: 'squareformer-tvm-ffn-dense1-gelu', code: kernels.dense1Gelu });
    const dense2Module = device.createShaderModule({ label: 'squareformer-tvm-ffn-dense2-residual', code: kernels.dense2Residual });
    const [dense1Pipeline, dense2Pipeline] = await Promise.all([
      device.createComputePipelineAsync({ label: 'squareformer-tvm-ffn-dense1-gelu', layout: 'auto', compute: { module: dense1Module, entryPoint: 'matmul_kernel' } }),
      device.createComputePipelineAsync({ label: 'squareformer-tvm-ffn-dense2-residual', layout: 'auto', compute: { module: dense2Module, entryPoint: 'matmul_kernel' } }),
    ]);
    const inputBytes = rows * dModel * 4;
    const hiddenBytes = rows * dFf * 4;
    const outputBytes = inputBytes;
    const buffers = {
      dense1Weight: createInitializedBuffer(device, 'squareformer-tvm-ffn-dense1-weight', weights.dense1Weight, GPUBufferUsage.STORAGE),
      dense1Bias: createInitializedBuffer(device, 'squareformer-tvm-ffn-dense1-bias', weights.dense1Bias, GPUBufferUsage.STORAGE),
      dense2Weight: createInitializedBuffer(device, 'squareformer-tvm-ffn-dense2-weight', weights.dense2Weight, GPUBufferUsage.STORAGE),
      dense2Bias: createInitializedBuffer(device, 'squareformer-tvm-ffn-dense2-bias', weights.dense2Bias, GPUBufferUsage.STORAGE),
      podArgs: createInitializedBuffer(device, 'squareformer-tvm-ffn-pod-args', new Uint32Array([ceilDiv(rows, 32) - 1, 0, 0, 0]), GPUBufferUsage.UNIFORM),
      input: createStorageBuffer(device, 'squareformer-tvm-ffn-input', inputBytes, GPUBufferUsage.COPY_DST),
      residual: createStorageBuffer(device, 'squareformer-tvm-ffn-residual', inputBytes, GPUBufferUsage.COPY_DST),
      hidden: createStorageBuffer(device, 'squareformer-tvm-ffn-hidden', hiddenBytes),
      output: createStorageBuffer(device, 'squareformer-tvm-ffn-output', outputBytes, GPUBufferUsage.COPY_SRC),
      readback: device.createBuffer({ label: 'squareformer-tvm-ffn-readback', size: outputBytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }),
    };
    return new SquareformerTvmFfnWebgpuBlock(device, fullShape, dense1Pipeline, dense2Pipeline, buffers);
  }

  upload(input: Float32Array, residual: Float32Array = input): void {
    const { rows, dModel } = this.shape;
    assertLength('input', input.length, rows * dModel);
    assertLength('residual', residual.length, rows * dModel);
    this.device.queue.writeBuffer(this.input, 0, input);
    this.device.queue.writeBuffer(this.residual, 0, residual);
  }

  encode(commandEncoder: GPUCommandEncoder, inputBuffer: GPUBuffer = this.input, residualBuffer: GPUBuffer = this.residual, outputBuffer: GPUBuffer = this.output): GPUBuffer {
    const { rows, dModel, dFf } = this.shape;
    const pass = commandEncoder.beginComputePass({ label: 'squareformer-tvm-ffn' });
    pass.setPipeline(this.dense1Pipeline);
    pass.setBindGroup(0, this.device.createBindGroup({
      label: 'squareformer-tvm-ffn-dense1-bindings',
      layout: this.dense1Pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.hidden } },
        { binding: 1, resource: { buffer: this.dense1Weight } },
        { binding: 2, resource: { buffer: inputBuffer } },
        { binding: 3, resource: { buffer: this.podArgs } },
        { binding: 4, resource: { buffer: this.dense1Bias } },
      ],
    }));
    pass.dispatchWorkgroups(ceilDiv(rows, 32), ceilDiv(dFf, 32), 1);
    pass.setPipeline(this.dense2Pipeline);
    pass.setBindGroup(0, this.device.createBindGroup({
      label: 'squareformer-tvm-ffn-dense2-bindings',
      layout: this.dense2Pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: outputBuffer } },
        { binding: 1, resource: { buffer: this.dense2Weight } },
        { binding: 2, resource: { buffer: this.hidden } },
        { binding: 3, resource: { buffer: this.podArgs } },
        { binding: 4, resource: { buffer: this.dense2Bias } },
        { binding: 5, resource: { buffer: residualBuffer } },
      ],
    }));
    pass.dispatchWorkgroups(ceilDiv(rows, 32), ceilDiv(dModel, 32), 1);
    pass.end();
    return outputBuffer;
  }

  copyOutputToReadback(commandEncoder: GPUCommandEncoder): void {
    const { rows, dModel } = this.shape;
    commandEncoder.copyBufferToBuffer(this.output, 0, this.readback, 0, rows * dModel * 4);
  }

  async readOutput(): Promise<Float32Array> {
    await this.readback.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(this.readback.getMappedRange()).slice();
    this.readback.unmap();
    return out;
  }

  async run(input: Float32Array, options: SquareformerTvmFfnRunOptions = {}): Promise<Float32Array | GPUBuffer> {
    const residual = options.residual ?? input;
    this.upload(input, residual);
    const commandEncoder = this.device.createCommandEncoder({ label: 'squareformer-tvm-ffn-run' });
    this.encode(commandEncoder, this.input, this.residual, this.output);
    if (options.readback !== false) this.copyOutputToReadback(commandEncoder);
    this.device.queue.submit([commandEncoder.finish()]);
    if (options.readback === false) return this.output;
    return this.readOutput();
  }

  destroy() {
    for (const buffer of [this.dense1Weight, this.dense1Bias, this.dense2Weight, this.dense2Bias, this.podArgs, this.input, this.residual, this.hidden, this.output, this.readback] as StorageBuffer[]) {
      buffer.destroy();
    }
  }
}
