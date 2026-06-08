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
declare const GPUBufferUsage: Record<'STORAGE' | 'COPY_SRC' | 'COPY_DST' | 'MAP_READ', number>;
declare const GPUMapMode: Record<'READ', number>;

export type SquareformerTvmInputStemWeights = {
  pieceEmb: Float32Array;
  stmEmb: Float32Array;
  flagLinearWeight: Float32Array;
  flagLinearBias: Float32Array;
  rankEmb: Float32Array;
  fileEmb: Float32Array;
  colorEmb: Float32Array;
  squareEmb: Float32Array;
  repLinearWeight: Float32Array;
  attackNormWeight: Float32Array;
  attackNormBias: Float32Array;
  attackProjWeight: Float32Array;
  pos: Float32Array;
};

const OFFSETS = {
  pieceEmb: 0,
  stmEmb: 104 * 128,
  flagLinearWeight: 104 * 128 + 3 * 128,
  flagLinearBias: 104 * 128 + 3 * 128 + 6 * 128,
  rankEmb: 104 * 128 + 3 * 128 + 6 * 128 + 128,
  fileEmb: 104 * 128 + 3 * 128 + 6 * 128 + 128 + 8 * 128,
  colorEmb: 104 * 128 + 3 * 128 + 6 * 128 + 128 + 8 * 128 + 8 * 128,
  squareEmb: 104 * 128 + 3 * 128 + 6 * 128 + 128 + 8 * 128 + 8 * 128 + 2 * 128,
  repLinearWeight: 104 * 128 + 3 * 128 + 6 * 128 + 128 + 8 * 128 + 8 * 128 + 2 * 128 + 64 * 128,
  attackNormWeight: 104 * 128 + 3 * 128 + 6 * 128 + 128 + 8 * 128 + 8 * 128 + 2 * 128 + 64 * 128 + 8 * 128,
  attackNormBias: 104 * 128 + 3 * 128 + 6 * 128 + 128 + 8 * 128 + 8 * 128 + 2 * 128 + 64 * 128 + 8 * 128 + 28,
  attackProjWeight: 104 * 128 + 3 * 128 + 6 * 128 + 128 + 8 * 128 + 8 * 128 + 2 * 128 + 64 * 128 + 8 * 128 + 28 + 28,
  pos: 104 * 128 + 3 * 128 + 6 * 128 + 128 + 8 * 128 + 8 * 128 + 2 * 128 + 64 * 128 + 8 * 128 + 28 + 28 + 28 * 128,
} as const;
const PACKED_WEIGHT_FLOATS = OFFSETS.pos + 64 * 128;

const INPUT_STEM_WGSL = `
const D_MODEL : u32 = 128u;
const TOKEN_FEATURES : u32 = 24u;
const PIECE_EMB : u32 = ${OFFSETS.pieceEmb}u;
const STM_EMB : u32 = ${OFFSETS.stmEmb}u;
const FLAG_LINEAR_WEIGHT : u32 = ${OFFSETS.flagLinearWeight}u;
const FLAG_LINEAR_BIAS : u32 = ${OFFSETS.flagLinearBias}u;
const RANK_EMB : u32 = ${OFFSETS.rankEmb}u;
const FILE_EMB : u32 = ${OFFSETS.fileEmb}u;
const COLOR_EMB : u32 = ${OFFSETS.colorEmb}u;
const SQUARE_EMB : u32 = ${OFFSETS.squareEmb}u;
const REP_LINEAR_WEIGHT : u32 = ${OFFSETS.repLinearWeight}u;
const ATTACK_NORM_WEIGHT : u32 = ${OFFSETS.attackNormWeight}u;
const ATTACK_NORM_BIAS : u32 = ${OFFSETS.attackNormBias}u;
const ATTACK_PROJ_WEIGHT : u32 = ${OFFSETS.attackProjWeight}u;
const POS : u32 = ${OFFSETS.pos}u;

@group(0) @binding(0) var<storage, read_write> outputHidden : array<f32>;
@group(0) @binding(1) var<storage, read> tokens : array<u32>;
@group(0) @binding(2) var<storage, read> attackSummary : array<f32>;
@group(0) @binding(3) var<storage, read> weights : array<f32>;

fn clamp_u32(v : u32, hi : u32) -> u32 {
  return min(v, hi);
}

@compute @workgroup_size(128, 1, 1)
fn input_stem_kernel(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx : u32 = gid.x;
  if (idx >= 8192u) { return; }
  let row : u32 = idx / D_MODEL;
  let d : u32 = idx - row * D_MODEL;
  let tokenBase : u32 = row * TOKEN_FEATURES;
  var acc : f32 = 0.0f;

  for (var h : u32 = 0u; h < 8u; h = h + 1u) {
    let piece : u32 = clamp_u32(tokens[tokenBase + h], 12u);
    acc = acc + weights[PIECE_EMB + ((h * 13u + piece) * D_MODEL + d)];
  }

  let stm : u32 = clamp_u32(tokens[tokenBase + 8u], 2u);
  acc = acc + weights[STM_EMB + stm * D_MODEL + d];

  let flags : u32 = tokens[tokenBase + 9u];
  let ep : f32 = f32(tokens[tokenBase + 10u]);
  let half : f32 = f32(tokens[tokenBase + 11u]) / 100.0f;
  let flag0 : f32 = f32(flags & 1u);
  let flag1 : f32 = f32((flags >> 1u) & 1u);
  let flag2 : f32 = f32((flags >> 2u) & 1u);
  let flag3 : f32 = f32((flags >> 3u) & 1u);
  acc = acc + weights[FLAG_LINEAR_BIAS + d]
    + flag0 * weights[FLAG_LINEAR_WEIGHT + 0u * D_MODEL + d]
    + flag1 * weights[FLAG_LINEAR_WEIGHT + 1u * D_MODEL + d]
    + flag2 * weights[FLAG_LINEAR_WEIGHT + 2u * D_MODEL + d]
    + flag3 * weights[FLAG_LINEAR_WEIGHT + 3u * D_MODEL + d]
    + ep * weights[FLAG_LINEAR_WEIGHT + 4u * D_MODEL + d]
    + half * weights[FLAG_LINEAR_WEIGHT + 5u * D_MODEL + d];

  let rank : u32 = clamp_u32(tokens[tokenBase + 12u], 7u);
  let file : u32 = clamp_u32(tokens[tokenBase + 13u], 7u);
  let color : u32 = clamp_u32(tokens[tokenBase + 14u], 1u);
  let square : u32 = clamp_u32(tokens[tokenBase + 15u], 63u);
  acc = acc + weights[RANK_EMB + rank * D_MODEL + d]
    + weights[FILE_EMB + file * D_MODEL + d]
    + weights[COLOR_EMB + color * D_MODEL + d]
    + weights[SQUARE_EMB + square * D_MODEL + d];

  for (var r : u32 = 0u; r < 8u; r = r + 1u) {
    let rep : f32 = f32(clamp_u32(tokens[tokenBase + 16u + r], 1u));
    acc = acc + rep * weights[REP_LINEAR_WEIGHT + r * D_MODEL + d];
  }

  var mean : f32 = 0.0f;
  for (var f : u32 = 0u; f < 28u; f = f + 1u) {
    mean = mean + attackSummary[row * 28u + f] / 8.0f;
  }
  mean = mean / 28.0f;
  var variance : f32 = 0.0f;
  for (var f2 : u32 = 0u; f2 < 28u; f2 = f2 + 1u) {
    let centered : f32 = attackSummary[row * 28u + f2] / 8.0f - mean;
    variance = variance + centered * centered;
  }
  let invStd : f32 = inverseSqrt(variance / 28.0f + 0.00001f);
  var attackAcc : f32 = 0.0f;
  for (var f3 : u32 = 0u; f3 < 28u; f3 = f3 + 1u) {
    let normed : f32 = (attackSummary[row * 28u + f3] / 8.0f - mean) * invStd;
    let projectedInput : f32 = normed * weights[ATTACK_NORM_WEIGHT + f3] + weights[ATTACK_NORM_BIAS + f3];
    attackAcc = attackAcc + projectedInput * weights[ATTACK_PROJ_WEIGHT + f3 * D_MODEL + d];
  }

  outputHidden[idx] = acc + attackAcc + weights[POS + row * D_MODEL + d];
}
`;

function assertLength(name: string, actual: number, expected: number) { if (actual !== expected) throw new Error(`${name} length ${actual} does not match expected ${expected}`); }
function createStorageBuffer(device: GPUDevice, label: string, bytes: number, usage = 0): GPUBuffer { return device.createBuffer({ label, size: Math.max(4, bytes), usage: GPUBufferUsage.STORAGE | usage }); }
function createInitializedF32Buffer(device: GPUDevice, label: string, data: Float32Array, usage = 0): GPUBuffer {
  const buffer = device.createBuffer({ label, size: Math.max(4, data.byteLength), usage: GPUBufferUsage.STORAGE | usage, mappedAtCreation: true });
  new Float32Array(buffer.getMappedRange()).set(data);
  buffer.unmap();
  return buffer;
}
function createInitializedU32Buffer(device: GPUDevice, label: string, data: Uint32Array, usage = 0): GPUBuffer {
  const buffer = device.createBuffer({ label, size: Math.max(4, data.byteLength), usage: GPUBufferUsage.STORAGE | usage, mappedAtCreation: true });
  new Uint32Array(buffer.getMappedRange()).set(data);
  buffer.unmap();
  return buffer;
}
function packWeights(weights: SquareformerTvmInputStemWeights): Float32Array {
  assertLength('pieceEmb', weights.pieceEmb.length, 104 * 128);
  assertLength('stmEmb', weights.stmEmb.length, 3 * 128);
  assertLength('flagLinearWeight', weights.flagLinearWeight.length, 6 * 128);
  assertLength('flagLinearBias', weights.flagLinearBias.length, 128);
  assertLength('rankEmb', weights.rankEmb.length, 8 * 128);
  assertLength('fileEmb', weights.fileEmb.length, 8 * 128);
  assertLength('colorEmb', weights.colorEmb.length, 2 * 128);
  assertLength('squareEmb', weights.squareEmb.length, 64 * 128);
  assertLength('repLinearWeight', weights.repLinearWeight.length, 8 * 128);
  assertLength('attackNormWeight', weights.attackNormWeight.length, 28);
  assertLength('attackNormBias', weights.attackNormBias.length, 28);
  assertLength('attackProjWeight', weights.attackProjWeight.length, 28 * 128);
  assertLength('pos', weights.pos.length, 64 * 128);
  const packed = new Float32Array(PACKED_WEIGHT_FLOATS);
  packed.set(weights.pieceEmb, OFFSETS.pieceEmb);
  packed.set(weights.stmEmb, OFFSETS.stmEmb);
  packed.set(weights.flagLinearWeight, OFFSETS.flagLinearWeight);
  packed.set(weights.flagLinearBias, OFFSETS.flagLinearBias);
  packed.set(weights.rankEmb, OFFSETS.rankEmb);
  packed.set(weights.fileEmb, OFFSETS.fileEmb);
  packed.set(weights.colorEmb, OFFSETS.colorEmb);
  packed.set(weights.squareEmb, OFFSETS.squareEmb);
  packed.set(weights.repLinearWeight, OFFSETS.repLinearWeight);
  packed.set(weights.attackNormWeight, OFFSETS.attackNormWeight);
  packed.set(weights.attackNormBias, OFFSETS.attackNormBias);
  packed.set(weights.attackProjWeight, OFFSETS.attackProjWeight);
  packed.set(weights.pos, OFFSETS.pos);
  return packed;
}

export class SquareformerTvmInputStemWebgpuBlock {
  readonly rows = 64;
  readonly dModel = 128;
  private pipeline: GPUComputePipeline;
  private tokens: GPUBuffer;
  private attackSummary: GPUBuffer;
  private weights: GPUBuffer;
  private output: GPUBuffer;
  private readback: GPUBuffer;

  private constructor(private device: GPUDevice, pipeline: GPUComputePipeline, buffers: { tokens: GPUBuffer; attackSummary: GPUBuffer; weights: GPUBuffer; output: GPUBuffer; readback: GPUBuffer }) {
    this.pipeline = pipeline;
    this.tokens = buffers.tokens;
    this.attackSummary = buffers.attackSummary;
    this.weights = buffers.weights;
    this.output = buffers.output;
    this.readback = buffers.readback;
  }

  static async create(device: GPUDevice, weights: SquareformerTvmInputStemWeights): Promise<SquareformerTvmInputStemWebgpuBlock> {
    const module = device.createShaderModule({ label: 'squareformer-tvm-input-stem', code: INPUT_STEM_WGSL });
    const pipeline = await device.createComputePipelineAsync({ label: 'squareformer-tvm-input-stem', layout: 'auto', compute: { module, entryPoint: 'input_stem_kernel' } });
    const buffers = {
      tokens: createStorageBuffer(device, 'squareformer-tvm-input-stem-tokens', 64 * 24 * 4, GPUBufferUsage.COPY_DST),
      attackSummary: createStorageBuffer(device, 'squareformer-tvm-input-stem-attack-summary', 64 * 28 * 4, GPUBufferUsage.COPY_DST),
      weights: createInitializedF32Buffer(device, 'squareformer-tvm-input-stem-weights', packWeights(weights)),
      output: createStorageBuffer(device, 'squareformer-tvm-input-stem-output', 64 * 128 * 4, GPUBufferUsage.COPY_SRC),
      readback: device.createBuffer({ label: 'squareformer-tvm-input-stem-readback', size: 64 * 128 * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }),
    };
    return new SquareformerTvmInputStemWebgpuBlock(device, pipeline, buffers);
  }

  upload(tokens: Uint32Array, attackSummary: Float32Array): void {
    assertLength('tokens', tokens.length, 64 * 24);
    assertLength('attackSummary', attackSummary.length, 64 * 28);
    this.device.queue.writeBuffer(this.tokens, 0, tokens);
    this.device.queue.writeBuffer(this.attackSummary, 0, attackSummary);
  }

  encode(commandEncoder: GPUCommandEncoder, tokensBuffer: GPUBuffer = this.tokens, attackSummaryBuffer: GPUBuffer = this.attackSummary, outputBuffer: GPUBuffer = this.output): GPUBuffer {
    const pass = commandEncoder.beginComputePass({ label: 'squareformer-tvm-input-stem' });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.device.createBindGroup({
      label: 'squareformer-tvm-input-stem-bindings',
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: outputBuffer } },
        { binding: 1, resource: { buffer: tokensBuffer } },
        { binding: 2, resource: { buffer: attackSummaryBuffer } },
        { binding: 3, resource: { buffer: this.weights } },
      ],
    }));
    pass.dispatchWorkgroups(64, 1, 1);
    pass.end();
    return outputBuffer;
  }

  copyOutputToReadback(commandEncoder: GPUCommandEncoder): void {
    commandEncoder.copyBufferToBuffer(this.output, 0, this.readback, 0, 64 * 128 * 4);
  }

  async readOutput(): Promise<Float32Array> {
    await this.readback.mapAsync(GPUMapMode.READ);
    const copy = new Float32Array(this.readback.getMappedRange().slice(0));
    this.readback.unmap();
    return copy;
  }

  async run(tokens: Uint32Array, attackSummary: Float32Array, options: { readback?: boolean } = {}): Promise<Float32Array | GPUBuffer> {
    this.upload(tokens, attackSummary);
    const commandEncoder = this.device.createCommandEncoder({ label: 'squareformer-tvm-input-stem-run' });
    const output = this.encode(commandEncoder);
    if (options.readback !== false) this.copyOutputToReadback(commandEncoder);
    this.device.queue.submit([commandEncoder.finish()]);
    if (options.readback === false) return output;
    return this.readOutput();
  }

  destroy(): void {
    for (const buffer of [this.tokens, this.attackSummary, this.weights, this.output, this.readback]) buffer.destroy();
  }
}
