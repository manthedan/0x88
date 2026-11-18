import type { BoardState } from '../chess/board.ts';
import { isStmWhiteRankflip, normalizedMoveToOriginal, normalizePositionForStmWhite } from '../chess/boardNormalization.ts';
import { type Move, moveToActionId } from '../chess/moveCodec.ts';
import { moveToSquareformerPolicyIndex } from '../chess/moveEncodings.ts';
import { legalMoves } from '../chess/movegen.ts';
import { ExclusiveQueue } from '../lc0/exclusiveQueue.ts';
import type { Evaluation, EvaluationContext, Evaluator } from './evaluator.ts';
import { softmax } from './numerics.ts';
import {
  type SquareFormerMeta,
  squareformerCompactInput,
  THREATGRAPH_SQUARE_SUMMARY_V1_FEATURES,
  threatgraphSquareSummaryV1,
} from './squareformerEvaluator.ts';
import {
  type SquareformerTvmFullLayerKernels,
  SquareformerTvmFullLayerWebgpuBlock,
  type SquareformerTvmFullLayerWeights,
} from './squareformerTvmFullLayerWebgpu.ts';
import { SquareformerTvmInputStemWebgpuBlock, type SquareformerTvmInputStemWeights } from './squareformerTvmInputStemWebgpu.ts';
import { SquareformerTvmLegalPolicyGatherWebgpuBlock } from './squareformerTvmLegalPolicyGatherWebgpu.ts';
import {
  type SquareformerTvmPolicyHeadKernels,
  SquareformerTvmPolicyHeadWebgpuBlock,
  type SquareformerTvmPolicyHeadWeights,
} from './squareformerTvmPolicyHeadWebgpu.ts';
import { SquareformerTvmValueHeadWebgpuBlock, type SquareformerTvmValueHeadWeights } from './squareformerTvmValueHeadWebgpu.ts';

export type SquareformerTvmHybridRawOutput = {
  legalPolicy: Float32Array;
  wdlLogits: Float32Array;
  q: Float32Array;
  timings: { uploadMs: number; gpuSubmitMs: number; readbackMs: number; totalMs: number };
};

type GPUBuffer = { destroy(): void };
type GPUDevice = {
  queue: { submit(commandBuffers: unknown[]): void };
  createCommandEncoder(descriptor?: unknown): { finish(): unknown };
  lost?: Promise<{ message?: string }>;
  destroy(): void;
};

type KernelSummaryRow = { label: string; epilogue?: string; file: string };
type FixtureRootSummary = { layers: Array<{ path: string; summary: string; shape?: { rows?: number; dModel?: number; dFf?: number } }> };
type HybridRuntimeManifest = {
  schema: 'centipawn.squareformer-tvm-hybrid.v1' | 'tiny-leela.squareformer-tvm-hybrid.v1' | string;
  modelKey?: string;
  bundleVersion?: string;
  modelOnnx?: string;
  modelMeta?: string;
  fallbackModelKey?: string;
  requiredFeatures?: string[];
  kernelBase?: string;
  fixtureRoot?: string;
  expected?: {
    kind?: string;
    layers?: number;
    rows?: number;
    dModel?: number;
    dFf?: number;
    heads?: number;
    policySize?: number;
    inputFormat?: string;
    attackSummarySchema?: string;
    attackSummaryFeatureCount?: number;
  };
  artifacts?: {
    modelOnnxSha256?: string;
    modelMetaSha256?: string;
    kernelSummarySha256?: string;
    fixtureSummarySha256?: string;
    files?: Array<{ path: string; bytes?: number; sha256: string }>;
  };
};
/** Owner cancellation and optional manifest checks for every artifact fetch. */
type ArtifactSource = { signal?: AbortSignal; verify?(url: string, bytes: ArrayBuffer): Promise<void> };
type FixtureArrayKey =
  | 'staticAttentionBias'
  | 'attnQkvWeight'
  | 'attnQkvBias'
  | 'n1LayerNormWeight'
  | 'n1LayerNormBias'
  | 'attnOutWeight'
  | 'attnOutBias'
  | 'dense1Weight'
  | 'dense1Bias'
  | 'dense2Weight'
  | 'dense2Bias'
  | 'layerNormWeight'
  | 'layerNormBias'
  | 'pieceEmb'
  | 'stmEmb'
  | 'flagLinearWeight'
  | 'flagLinearBias'
  | 'rankEmb'
  | 'fileEmb'
  | 'colorEmb'
  | 'squareEmb'
  | 'repLinearWeight'
  | 'attackNormWeight'
  | 'attackNormBias'
  | 'attackProjWeight'
  | 'pos'
  | 'policyFromWeight'
  | 'policyFromBias'
  | 'policyToWeight'
  | 'policyToBias'
  | 'policyPromoWeight'
  | 'policyPromoBias'
  | 'wdlWeight'
  | 'wdlBias'
  | 'qWeight'
  | 'qBias';
type FixtureSummary = {
  layer: number;
  shape: { rows: number; dModel: number; dFf: number; batch?: number };
  n1LayerNormEpsilon?: number;
  layerNormEpsilon?: number;
  arrays: Record<FixtureArrayKey, { file: string }>;
};
async function loadBytes(url: string, source?: ArtifactSource): Promise<ArrayBuffer> {
  const res = await fetch(url, { signal: source?.signal });
  if (!res.ok) throw new Error(`fetch ${url}: ${res.status} ${res.statusText}`);
  const bytes = await res.arrayBuffer();
  await source?.verify?.(url, bytes);
  return bytes;
}
async function loadText(url: string, source?: ArtifactSource): Promise<string> {
  return new TextDecoder().decode(await loadBytes(url, source));
}
async function loadJson<T>(url: string, source?: ArtifactSource): Promise<T> {
  return JSON.parse(await loadText(url, source)) as T;
}
async function loadF32(url: string, source?: ArtifactSource): Promise<Float32Array> {
  return new Float32Array(await loadBytes(url, source));
}
function stripSlash(path: string): string {
  return path.replace(/\/$/, '');
}
function resolveManifestPath(manifestUrl: string, path: string | undefined, fallback: string): string {
  const resolved = path ?? fallback;
  if (/^(?:https?:)?\//.test(resolved)) return stripSlash(resolved);
  const manifestBase = new URL(manifestUrl, typeof location !== 'undefined' ? location.href : 'http://localhost/').toString();
  return stripSlash(new URL(resolved, manifestBase).toString());
}
function assertExpected(label: string, expected: unknown, actual: unknown): void {
  if (expected !== undefined && actual !== undefined && expected !== actual)
    throw new Error(`Hybrid TVM artifact mismatch for ${label}: expected ${expected}, got ${actual}`);
}
function validateManifest(manifest: HybridRuntimeManifest, meta: SquareFormerMeta): void {
  // Accept the legacy 'tiny-leela.*' schema as well so already-cached/CDN manifests keep validating after the Centipawn rename.
  if (manifest.schema !== 'centipawn.squareformer-tvm-hybrid.v1' && manifest.schema !== 'tiny-leela.squareformer-tvm-hybrid.v1')
    throw new Error(`Unsupported hybrid TVM manifest schema: ${manifest.schema}`);
  const expected = manifest.expected;
  if (!expected) return;
  assertExpected('kind', expected.kind, meta.kind);
  assertExpected('policySize', expected.policySize, meta.policy_size);
  assertExpected('inputFormat', expected.inputFormat, meta.input_format);
  assertExpected('attackSummarySchema', expected.attackSummarySchema, meta.attack_summary_schema);
  assertExpected('attackSummaryFeatureCount', expected.attackSummaryFeatureCount, meta.attack_summary_feature_count);
  assertExpected('heads', expected.heads, 8);
}
async function sha256Hex(bytes: ArrayBuffer): Promise<string> {
  if (!globalThis.crypto?.subtle) throw new Error('WebCrypto SHA-256 is unavailable; cannot verify hybrid TVM artifact manifest');
  const digest = await globalThis.crypto.subtle.digest('SHA-256', bytes);
  return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, '0')).join('');
}
async function verifySha256(label: string, url: string, expected: string | undefined, signal?: AbortSignal): Promise<void> {
  if (!expected) return;
  const actual = await sha256Hex(await loadBytes(url, { signal }));
  if (actual !== expected) throw new Error(`Hybrid TVM artifact SHA-256 mismatch for ${label}: expected ${expected}, got ${actual}`);
}
function artifactUrl(manifestUrl: string, path: string): string {
  return new URL(path, new URL(manifestUrl, typeof location !== 'undefined' ? location.href : 'http://localhost/')).toString();
}
function createArtifactSource(manifest: HybridRuntimeManifest | undefined, manifestUrl: string | undefined, signal?: AbortSignal): ArtifactSource {
  if (!manifest?.artifacts?.files?.length || !manifestUrl) return { signal };
  const byUrl = new Map(manifest.artifacts.files.map((file) => [artifactUrl(manifestUrl, file.path), file]));
  return {
    signal,
    async verify(url: string, bytes: ArrayBuffer): Promise<void> {
      const resolved = new URL(url, typeof location !== 'undefined' ? location.href : 'http://localhost/').toString();
      const expected = byUrl.get(resolved);
      if (!expected) return;
      if (expected.bytes !== undefined && expected.bytes !== bytes.byteLength)
        throw new Error(`Hybrid TVM artifact size mismatch for ${expected.path}: expected ${expected.bytes}, got ${bytes.byteLength}`);
      const actual = await sha256Hex(bytes);
      if (actual !== expected.sha256) throw new Error(`Hybrid TVM artifact SHA-256 mismatch for ${expected.path}: expected ${expected.sha256}, got ${actual}`);
    },
  };
}
async function verifyManifestArtifacts(
  manifest: HybridRuntimeManifest | undefined,
  kernelBase: string,
  fixtureRoot: string,
  signal?: AbortSignal,
): Promise<void> {
  if (!manifest?.artifacts || manifest.artifacts.files?.length) return;
  await Promise.all([
    verifySha256('kernel summary', `${kernelBase}/summary.json`, manifest.artifacts.kernelSummarySha256, signal),
    verifySha256('fixture summary', `${fixtureRoot}/summary.json`, manifest.artifacts.fixtureSummarySha256, signal),
  ]);
}
function nowMs(): number {
  return typeof performance !== 'undefined' ? performance.now() : Date.now();
}
function toU32(input: ArrayLike<number | bigint>): Uint32Array {
  const out = new Uint32Array(input.length);
  for (let i = 0; i < input.length; i++) out[i] = Number(input[i]) >>> 0;
  return out;
}
function findKernel(summary: KernelSummaryRow[], labelPrefix: string, epilogue: string): KernelSummaryRow {
  const row = summary.find((item) => item.label.startsWith(labelPrefix) && item.epilogue === epilogue);
  if (!row) throw new Error(`No ${labelPrefix}/${epilogue} kernel in kernel summary`);
  return row;
}
async function loadKernels(kernelBase: string, source?: ArtifactSource): Promise<SquareformerTvmFullLayerKernels & SquareformerTvmPolicyHeadKernels> {
  const base = stripSlash(kernelBase);
  const summary = await loadJson<KernelSummaryRow[]>(`${base}/summary.json`, source);
  const qkv = findKernel(summary, 'attn_qkv_', 'bias');
  const dense1 = findKernel(summary, 'ffn_dense1_gelu_', 'bias-gelu');
  const dense2 = findKernel(summary, 'ffn_dense2_residual_', 'bias-residual');
  const attnOut = findKernel(summary, 'attn_out_residual_', 'bias-residual');
  const policyPair = findKernel(summary, 'policy_pair_proj_', 'bias');
  const policyPromo = findKernel(summary, 'policy_proj_', 'bias');
  return {
    qkvBias: await loadText(`${base}/${qkv.file}`, source),
    dense1Gelu: await loadText(`${base}/${dense1.file}`, source),
    dense2Residual: await loadText(`${base}/${dense2.file}`, source),
    attnOutResidual: await loadText(`${base}/${attnOut.file}`, source),
    policyPairProj: await loadText(`${base}/${policyPair.file}`, source),
    policyPromoProj: await loadText(`${base}/${policyPromo.file}`, source),
  };
}

export class SquareformerTvmHybridEvaluator implements Evaluator {
  private readonly queue = new ExclusiveQueue();
  private destroyed?: Promise<void>;
  private failureReason: string | undefined;
  private device: GPUDevice;
  private meta: SquareFormerMeta;
  private stem: SquareformerTvmInputStemWebgpuBlock;
  private layers: SquareformerTvmFullLayerWebgpuBlock[];
  private policy: SquareformerTvmPolicyHeadWebgpuBlock;
  private value: SquareformerTvmValueHeadWebgpuBlock;
  private legal: SquareformerTvmLegalPolicyGatherWebgpuBlock;

  private constructor(
    device: GPUDevice,
    meta: SquareFormerMeta,
    stem: SquareformerTvmInputStemWebgpuBlock,
    layers: SquareformerTvmFullLayerWebgpuBlock[],
    policy: SquareformerTvmPolicyHeadWebgpuBlock,
    value: SquareformerTvmValueHeadWebgpuBlock,
    legal: SquareformerTvmLegalPolicyGatherWebgpuBlock,
  ) {
    this.device = device;
    this.meta = meta;
    this.stem = stem;
    this.layers = layers;
    this.policy = policy;
    this.value = value;
    this.legal = legal;
    // Surface a lost device as an evaluation error so a wrapping
    // RuntimeFallbackEvaluator can switch to ORT.
    void this.device.lost?.then((info) => {
      this.failureReason = info?.message || 'WebGPU device lost';
    });
  }

  /**
   * Builds the evaluator on `device`. On success the evaluator owns the device
   * and destroys it in destroy(); on failure the caller still owns it.
   */
  static async create(
    device: GPUDevice,
    meta: SquareFormerMeta,
    urls: { kernelBase?: string; fixtureRoot?: string; manifestUrl?: string } = {},
    signal?: AbortSignal,
  ): Promise<SquareformerTvmHybridEvaluator> {
    let kernelBase = stripSlash(urls.kernelBase ?? '/tvm/squareformer-ffn');
    let fixtureRoot = stripSlash(urls.fixtureRoot ?? '/tvm/ffn-fixtures');
    let manifest: HybridRuntimeManifest | undefined;
    if (urls.manifestUrl) {
      manifest = await loadJson<HybridRuntimeManifest>(urls.manifestUrl, { signal });
      validateManifest(manifest, meta);
      kernelBase = resolveManifestPath(urls.manifestUrl, urls.kernelBase ?? manifest.kernelBase, kernelBase);
      fixtureRoot = resolveManifestPath(urls.manifestUrl, urls.fixtureRoot ?? manifest.fixtureRoot, fixtureRoot);
      await verifyManifestArtifacts(manifest, kernelBase, fixtureRoot, signal);
    }
    const source = createArtifactSource(manifest, urls.manifestUrl, signal);
    const [kernels, root] = await Promise.all([loadKernels(kernelBase, source), loadJson<FixtureRootSummary>(`${fixtureRoot}/summary.json`, source)]);
    if (!root.layers.length) throw new Error(`No layers in ${fixtureRoot}/summary.json`);
    assertExpected('layers', manifest?.expected?.layers, root.layers.length);
    const firstShape = root.layers[0]?.shape;
    assertExpected('rows', manifest?.expected?.rows, firstShape?.rows);
    assertExpected('dModel', manifest?.expected?.dModel, firstShape?.dModel);
    assertExpected('dFf', manifest?.expected?.dFf, firstShape?.dFf);
    const layerFixtures = await Promise.all(
      root.layers.map(async (layerInfo) => {
        const base = `${fixtureRoot}/${layerInfo.path}`;
        const summary = await loadJson<FixtureSummary>(`${base}/${layerInfo.summary}`, source);
        const weights: SquareformerTvmFullLayerWeights = {
          layerNormWeight: await loadF32(`${base}/${summary.arrays.layerNormWeight.file}`, source),
          layerNormBias: await loadF32(`${base}/${summary.arrays.layerNormBias.file}`, source),
          dense1Weight: await loadF32(`${base}/${summary.arrays.dense1Weight.file}`, source),
          dense1Bias: await loadF32(`${base}/${summary.arrays.dense1Bias.file}`, source),
          dense2Weight: await loadF32(`${base}/${summary.arrays.dense2Weight.file}`, source),
          dense2Bias: await loadF32(`${base}/${summary.arrays.dense2Bias.file}`, source),
          attnOutWeight: await loadF32(`${base}/${summary.arrays.attnOutWeight.file}`, source),
          attnOutBias: await loadF32(`${base}/${summary.arrays.attnOutBias.file}`, source),
          qkvWeight: await loadF32(`${base}/${summary.arrays.attnQkvWeight.file}`, source),
          qkvBias: await loadF32(`${base}/${summary.arrays.attnQkvBias.file}`, source),
          n1LayerNormWeight: await loadF32(`${base}/${summary.arrays.n1LayerNormWeight.file}`, source),
          n1LayerNormBias: await loadF32(`${base}/${summary.arrays.n1LayerNormBias.file}`, source),
        };
        const block = await SquareformerTvmFullLayerWebgpuBlock.create(
          device as never,
          kernels,
          weights,
          { ...summary.shape, heads: 8 },
          { n1: summary.n1LayerNormEpsilon ?? 1e-5, n2: summary.layerNormEpsilon ?? 1e-5 },
        );
        // The static attention bias is a read-only per-layer buffer: upload it once here, not per evaluation.
        block.uploadAttentionBias(await loadF32(`${base}/${summary.arrays.staticAttentionBias.file}`, source));
        return { summary, base, block };
      }),
    );
    layerFixtures.sort((a, b) => a.summary.layer - b.summary.layer);
    const first = layerFixtures[0]!;
    const last = layerFixtures[layerFixtures.length - 1]!;
    const stemWeights: SquareformerTvmInputStemWeights = {
      pieceEmb: await loadF32(`${first.base}/${first.summary.arrays.pieceEmb.file}`, source),
      stmEmb: await loadF32(`${first.base}/${first.summary.arrays.stmEmb.file}`, source),
      flagLinearWeight: await loadF32(`${first.base}/${first.summary.arrays.flagLinearWeight.file}`, source),
      flagLinearBias: await loadF32(`${first.base}/${first.summary.arrays.flagLinearBias.file}`, source),
      rankEmb: await loadF32(`${first.base}/${first.summary.arrays.rankEmb.file}`, source),
      fileEmb: await loadF32(`${first.base}/${first.summary.arrays.fileEmb.file}`, source),
      colorEmb: await loadF32(`${first.base}/${first.summary.arrays.colorEmb.file}`, source),
      squareEmb: await loadF32(`${first.base}/${first.summary.arrays.squareEmb.file}`, source),
      repLinearWeight: await loadF32(`${first.base}/${first.summary.arrays.repLinearWeight.file}`, source),
      attackNormWeight: await loadF32(`${first.base}/${first.summary.arrays.attackNormWeight.file}`, source),
      attackNormBias: await loadF32(`${first.base}/${first.summary.arrays.attackNormBias.file}`, source),
      attackProjWeight: await loadF32(`${first.base}/${first.summary.arrays.attackProjWeight.file}`, source),
      pos: await loadF32(`${first.base}/${first.summary.arrays.pos.file}`, source),
    };
    const policyWeights: SquareformerTvmPolicyHeadWeights = {
      policyFromWeight: await loadF32(`${last.base}/${last.summary.arrays.policyFromWeight.file}`, source),
      policyFromBias: await loadF32(`${last.base}/${last.summary.arrays.policyFromBias.file}`, source),
      policyToWeight: await loadF32(`${last.base}/${last.summary.arrays.policyToWeight.file}`, source),
      policyToBias: await loadF32(`${last.base}/${last.summary.arrays.policyToBias.file}`, source),
      policyPromoWeight: await loadF32(`${last.base}/${last.summary.arrays.policyPromoWeight.file}`, source),
      policyPromoBias: await loadF32(`${last.base}/${last.summary.arrays.policyPromoBias.file}`, source),
    };
    const valueWeights: SquareformerTvmValueHeadWeights = {
      wdlWeight: await loadF32(`${last.base}/${last.summary.arrays.wdlWeight.file}`, source),
      wdlBias: await loadF32(`${last.base}/${last.summary.arrays.wdlBias.file}`, source),
      qWeight: await loadF32(`${last.base}/${last.summary.arrays.qWeight.file}`, source),
      qBias: await loadF32(`${last.base}/${last.summary.arrays.qBias.file}`, source),
    };
    const [stem, policy, value, legal] = await Promise.all([
      SquareformerTvmInputStemWebgpuBlock.create(device as never, stemWeights),
      SquareformerTvmPolicyHeadWebgpuBlock.create(device as never, kernels, policyWeights, { ...last.summary.shape, pairDim: 128, promoDim: 256 }),
      SquareformerTvmValueHeadWebgpuBlock.create(device as never, valueWeights, last.summary.shape),
      SquareformerTvmLegalPolicyGatherWebgpuBlock.createDynamic(device as never, Math.max(1, Number(meta.max_legal_moves ?? 256)), meta.policy_size),
    ]);
    // Shader compilation is not abortable; never hand a cancelled owner an evaluator.
    // The caller still owns the device here and destroys it, freeing every block.
    signal?.throwIfAborted();
    return new SquareformerTvmHybridEvaluator(
      device,
      meta,
      stem,
      layerFixtures.map(({ block }) => block),
      policy,
      value,
      legal,
    );
  }

  private async ensureLegalCapacity(legalCount: number): Promise<void> {
    if (legalCount <= this.legal.maxLegalCount) return;
    this.legal.destroy();
    this.legal = await SquareformerTvmLegalPolicyGatherWebgpuBlock.createDynamic(this.device as never, legalCount, this.meta.policy_size);
  }

  private runExclusive<T>(fn: () => Promise<T>): Promise<T> {
    return this.queue.run(fn);
  }

  async evaluateEncoded(tokens: Uint32Array, attackSummary: Float32Array, legalIndices: Uint32Array): Promise<SquareformerTvmHybridRawOutput> {
    return this.runExclusive(async () => {
      if (this.destroyed) throw new Error('Custom WebGPU evaluator is destroyed');
      if (this.failureReason) throw new Error(`Custom WebGPU runtime is unavailable: ${this.failureReason}`);
      const effectiveLegal = legalIndices.length ? legalIndices : new Uint32Array([0]);
      await this.ensureLegalCapacity(effectiveLegal.length);
      const t0 = nowMs();
      this.stem.upload(tokens, attackSummary);
      this.legal.uploadIndices(effectiveLegal);
      const tUpload = nowMs();
      const commandEncoder = this.device.createCommandEncoder({ label: 'squareformer-tvm-hybrid-evaluate' });
      let inputBuffer = this.stem.encode(commandEncoder as never) as GPUBuffer;
      for (const layer of this.layers) inputBuffer = layer.encode(commandEncoder as never, inputBuffer as never) as GPUBuffer;
      const policyBuffer = this.policy.encode(commandEncoder as never, inputBuffer as never) as GPUBuffer;
      this.legal.encode(commandEncoder as never, policyBuffer as never);
      this.value.encode(commandEncoder as never, inputBuffer as never);
      this.legal.copyOutputToReadback(commandEncoder as never);
      this.value.copyOutputToReadback(commandEncoder as never);
      this.device.queue.submit([commandEncoder.finish()]);
      const tSubmit = nowMs();
      const [legalPolicy, valueOut] = await Promise.all([this.legal.readOutput(), this.value.readOutput()]);
      if (this.failureReason) throw new Error(`Custom WebGPU runtime failed: ${this.failureReason}`);
      const tDone = nowMs();
      return {
        legalPolicy: legalIndices.length ? legalPolicy : legalPolicy.subarray(0, 0),
        wdlLogits: valueOut.wdl,
        q: valueOut.q,
        timings: { uploadMs: tUpload - t0, gpuSubmitMs: tSubmit - tUpload, readbackMs: tDone - tSubmit, totalMs: tDone - t0 },
      };
    });
  }

  async evaluate(board: BoardState, context: EvaluationContext = {}): Promise<Evaluation> {
    const normalized = isStmWhiteRankflip(this.meta.board_normalization)
      ? normalizePositionForStmWhite(board, context.historyFens ?? [], context.legalMoves)
      : { board, historyFens: context.historyFens ?? [], legalMoves: context.legalMoves, flipped: false };
    if (this.meta.input_format !== 'compact_uint8_tokens') throw new Error(`Hybrid TVM evaluator requires compact_uint8_tokens, got ${this.meta.input_format}`);
    if (
      this.meta.attack_summary_schema !== 'threatgraph_square_summary_v1' ||
      Number(this.meta.attack_summary_feature_count ?? 0) !== THREATGRAPH_SQUARE_SUMMARY_V1_FEATURES
    ) {
      throw new Error(`Hybrid TVM evaluator requires threatgraph_square_summary_v1/${THREATGRAPH_SQUARE_SUMMARY_V1_FEATURES}`);
    }
    const tokens = toU32(squareformerCompactInput(normalized.board, this.meta, normalized.historyFens));
    const attackSummary = threatgraphSquareSummaryV1(normalized.board);
    const evalLegal = normalized.legalMoves ?? legalMoves(normalized.board);
    const legalIndices = new Uint32Array(evalLegal.map(moveToSquareformerPolicyIndex));
    const raw = await this.evaluateEncoded(tokens, attackSummary, legalIndices);
    const probs = softmax(Array.from(raw.legalPolicy));
    const policy = new Map<number, number>();
    evalLegal.forEach((move: Move, i: number) => {
      const originalMove = normalizedMoveToOriginal(move, normalized.flipped);
      policy.set(moveToActionId(originalMove), probs[i] ?? 0);
    });
    const wdl = softmax(raw.wdlLogits);
    return { policy, wdl: [wdl[0] ?? 0, wdl[1] ?? 0, wdl[2] ?? 0] };
  }

  async evaluateBatch(boards: BoardState[], contexts: EvaluationContext[] = []): Promise<Evaluation[]> {
    const out: Evaluation[] = [];
    for (let i = 0; i < boards.length; i++) out.push(await this.evaluate(boards[i]!, contexts[i] ?? {}));
    return out;
  }

  /** Frees the GPU buffers and the owned device once in-flight evaluations have
   * settled; resolves only then, so a retirement barrier holds until the device is gone. */
  destroy(): Promise<void> {
    if (this.destroyed) return this.destroyed;
    const release = () => {
      for (const block of [this.stem, ...this.layers, this.policy, this.value, this.legal]) {
        try {
          block.destroy();
        } catch {
          /* best effort */
        }
      }
      try {
        this.device.destroy();
      } catch {
        /* best effort */
      }
    };
    this.destroyed = this.queue.idle().then(release);
    return this.destroyed;
  }
}
