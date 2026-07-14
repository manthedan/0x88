import * as ort from '../nn/ortRuntime.ts';
import { boardToFen, parseFen, type BoardState } from '../chess/board.ts';
import { legalMoves } from '../chess/movegen.ts';
import { moveToActionId, moveToUci, type Move } from '../chess/moveCodec.ts';
import { encodeLc0Classical112, type Lc0EncoderInput, type Lc0HistoryFill, type Lc0PositionHistoryInput } from './encoder112.ts';
import { LC0_MIRROR_TRANSFORM, uciToLc0PolicyIndex } from './policyMap.ts';

export const LC0_ONNX_INPUT_PLANES = '/input/planes';
export const LC0_ONNX_OUTPUT_POLICY = '/output/policy';
export const LC0_ONNX_OUTPUT_WDL = '/output/wdl';
export const LC0_ONNX_OUTPUT_MLH = '/output/mlh';
export const LC0_DEFAULT_POLICY_TEMPERATURE = 1.359;
const LC0_POLICY_SIZE = 1858;
const LC0_WDL_SIZE = 3;
const LC0_MLH_SIZE = 1;
const LC0_INPUT_PLANES_SIZE = 112 * 8 * 8;

export interface Lc0LegalPrior {
  uci: string;
  index: number;
  /** Search-native action ID, present when legal moves were prepared upstream. */
  actionId?: number;
  logit: number;
  prior: number;
}

export interface Lc0PreparedLegalMove {
  move: Move;
  uci: string;
  actionId: number;
  policyIndex: number;
}

/**
 * Search-native evaluator input. It retains the already-owned current board,
 * legal move mappings, and canonical cache key so evaluator adapters do not
 * repeat move generation, FEN parsing, or policy/action conversion.
 */
export interface Lc0PreparedEvaluatorInput extends Lc0PositionHistoryInput {
  prepared: {
    board: BoardState;
    fen: string;
    legalMoves: readonly Lc0PreparedLegalMove[];
    /** Preserve fen-only synthetic history semantics for a bare root input. */
    explicitHistory: boolean;
    cacheKey: string;
  };
}

export interface Lc0Evaluation {
  fen: string;
  wdl: [number, number, number];
  q: number;
  mlh: number;
  legalPriors: Lc0LegalPrior[];
  bestMove?: string;
  /** Optional backend timing diagnostics consumed by search/benchmark reporters. */
  timing?: unknown;
}

export interface Lc0OnnxEvaluatorOptions {
  policyTemperature?: number;
  historyFill?: Lc0HistoryFill;
}

export type Lc0EvaluatorInput = BoardState | string | Lc0PositionHistoryInput | Lc0PreparedEvaluatorInput;

export interface Lc0EvaluationProvider {
  evaluate(input: Lc0EvaluatorInput): Promise<Lc0Evaluation> | Lc0Evaluation;
  evaluateBatch?(inputs: Lc0EvaluatorInput[]): Promise<Lc0Evaluation[]> | Lc0Evaluation[];
  /** Optional ordered multi-batch API used by deferred/double-buffered browser paths. */
  evaluateBatchSequence?(batches: Lc0EvaluatorInput[][]): Promise<Lc0Evaluation[][]> | Lc0Evaluation[][];
  dispose?(): Promise<void> | void;
}

export interface Lc0EvaluationCacheMetrics {
  hits: number;
  misses: number;
  entries: number;
  maxEntries: number;
}

export interface Lc0EvaluationCacheFootprint {
  entries: number;
  maxEntries: number;
  approxBytes: number;
  approxKeyBytes: number;
  approxEvaluationBytes: number;
  note: string;
}

export interface Lc0EvaluationCacheOptions {
  maxEntries?: number;
}

function fileOf(square: number): number {
  return square % 8;
}

function isStandardCastlingMove(board: BoardState, move: Move): boolean {
  const piece = board.squares[move.from];
  return piece?.[1] === 'k' && Math.abs(fileOf(move.to) - fileOf(move.from)) === 2;
}

function f16ToF32(bits: number): number {
  const sign = (bits & 0x8000) ? -1 : 1;
  const exponent = (bits >>> 10) & 0x1f;
  const fraction = bits & 0x03ff;
  if (exponent === 0) return sign * (fraction === 0 ? 0 : 2 ** -14 * (fraction / 1024));
  if (exponent === 0x1f) return fraction === 0 ? sign * Infinity : NaN;
  return sign * 2 ** (exponent - 15) * (1 + fraction / 1024);
}

function f32ToF16Bits(value: number): number {
  if (Number.isNaN(value)) return 0x7e00;
  if (value === Infinity) return 0x7c00;
  if (value === -Infinity) return 0xfc00;
  const sign = value < 0 || Object.is(value, -0) ? 0x8000 : 0;
  const abs = Math.abs(value);
  if (abs === 0) return sign;
  if (abs >= 65504) return sign | 0x7bff;
  if (abs < 2 ** -24) return sign;
  if (abs < 2 ** -14) return sign | Math.round(abs / 2 ** -24);
  const exponent = Math.floor(Math.log2(abs));
  const fraction = abs / 2 ** exponent - 1;
  let halfExponent = exponent + 15;
  let halfFraction = Math.round(fraction * 1024);
  if (halfFraction === 1024) {
    halfExponent += 1;
    halfFraction = 0;
  }
  if (halfExponent >= 31) return sign | 0x7bff;
  return sign | (halfExponent << 10) | (halfFraction & 0x03ff);
}

function float32ToFloat16Array(values: Float32Array): Uint16Array {
  const out = new Uint16Array(values.length);
  for (let i = 0; i < values.length; i++) out[i] = f32ToF16Bits(values[i]);
  return out;
}

function coerceTensorData(tensor: { type?: string }, rawData: unknown): number[] | Float32Array {
  if (tensor.type === 'float16') {
    const data = rawData as ArrayLike<number> & { constructor?: { name?: string } };
    // ORT-web returns Uint16Array float16 bits in Node today, while modern
    // browsers may expose Float16Array values directly. Handle both forms.
    if (data.constructor?.name === 'Float16Array') return Array.from(data);
    return Array.from(data, f16ToF32);
  }
  return rawData as Float32Array | number[];
}

function tensorData(outputs: Awaited<ReturnType<ort.InferenceSession['run']>>, name: string): number[] | Float32Array {
  const tensor = outputs[name];
  if (!tensor) throw new Error(`LC0 ONNX output ${name} missing`);
  return coerceTensorData(tensor, tensor.data);
}

async function tensorDataTimed(outputs: Awaited<ReturnType<ort.InferenceSession['run']>>, name: string, timingKey: string, timing: Record<string, number>): Promise<number[] | Float32Array> {
  const tensor = outputs[name] as (ort.Tensor & { location?: string; getData?: () => Promise<unknown> }) | undefined;
  if (!tensor) throw new Error(`LC0 ONNX output ${name} missing`);
  const started = ort.tinyLeelaNowMs();
  try {
    const rawData = tensor.location === 'gpu-buffer' && typeof tensor.getData === 'function'
      ? await tensor.getData()
      : tensor.data;
    return coerceTensorData(tensor, rawData);
  } finally {
    timing[`${timingKey}GetDataMs`] = ort.tinyLeelaNowMs() - started;
  }
}

function sessionInputMetadata(session: ort.InferenceSession): { type?: string; shape?: unknown[] } | undefined {
  return (session.inputMetadata?.find?.((entry: { name?: string }) => entry.name === LC0_ONNX_INPUT_PLANES) ?? session.inputMetadata?.[0]) as { type?: string; shape?: unknown[] } | undefined;
}

function sessionInputType(session: ort.InferenceSession): 'float32' | 'float16' {
  return sessionInputMetadata(session)?.type === 'float16' ? 'float16' : 'float32';
}

function sessionFixedInputBatchSize(session: ort.InferenceSession): number {
  const firstDim = sessionInputMetadata(session)?.shape?.[0];
  return typeof firstDim === 'number' && Number.isFinite(firstDim) && firstDim > 0 ? Math.floor(firstDim) : 1;
}

function arraySlice<T extends ArrayLike<number>>(values: T, start: number, length: number): ArrayLike<number> {
  const end = start + length;
  const maybe = values as T & { subarray?: (start: number, end?: number) => ArrayLike<number>; slice?: (start: number, end?: number) => ArrayLike<number> };
  if (typeof maybe.subarray === 'function') return maybe.subarray(start, end);
  if (typeof maybe.slice === 'function') return maybe.slice(start, end);
  return Array.from({ length }, (_, i) => Number(values[start + i]));
}

function inputHistoryKey(input: Lc0EvaluatorInput): string {
  if (typeof input === 'object' && input !== null && 'prepared' in input) return input.prepared.cacheKey;
  if (typeof input === 'object' && input !== null && 'positions' in input) {
    const positions = input.positions.map((position) => typeof position === 'string' ? boardToFen(parseFen(position)) : boardToFen(position));
    return `history:${positions.length}\n${positions.join('\n')}`;
  }
  const fen = typeof input === 'string' ? boardToFen(parseFen(input)) : boardToFen(input);
  return `single\n${fen}`;
}

function cloneEvaluation(evaluation: Lc0Evaluation): Lc0Evaluation {
  return { ...evaluation, wdl: [...evaluation.wdl] as [number, number, number], legalPriors: evaluation.legalPriors.map((prior) => ({ ...prior })) };
}

function cloneCachedEvaluation(evaluation: Lc0Evaluation): Lc0Evaluation {
  const cloned = cloneEvaluation(evaluation);
  delete cloned.timing;
  return cloned;
}

function approximateCacheKeyBytes(key: string): number {
  return key.length * 2;
}

function approximateCachedEvaluationBytes(evaluation: Lc0Evaluation): number {
  const stringBytes = evaluation.fen.length * 2 + (evaluation.bestMove?.length ?? 0) * 2;
  const scalarBytes = 5 * 8; // WDL[3], q, mlh as JS numbers; object overhead is excluded.
  const legalPriorBytes = evaluation.legalPriors.reduce((sum, prior) => sum + prior.uci.length * 2 + 3 * 8, 0);
  return stringBytes + scalarBytes + legalPriorBytes;
}

export function prepareLc0EvaluatorInput(
  board: BoardState,
  historyNewestFirst: readonly (BoardState | string)[] = [],
  preparedMoves: readonly Move[] = legalMoves(board),
  historyFensNewestFirst?: readonly string[],
  explicitHistory: boolean = historyNewestFirst.length > 0,
): Lc0PreparedEvaluatorInput {
  const fen = boardToFen(board);
  const moveTransform = board.turn === 'b' ? LC0_MIRROR_TRANSFORM : 0;
  const mappedMoves = preparedMoves.map((move) => {
    const uci = moveToUci(move);
    const policyIndex = uciToLc0PolicyIndex(uci, moveTransform, { standardCastling: isStandardCastlingMove(board, move) });
    if (policyIndex === undefined) throw new Error(`No LC0 policy index for legal move ${uci}`);
    return { move, uci, actionId: moveToActionId(move), policyIndex };
  });
  const chronological = [...historyNewestFirst].reverse();
  const cacheHistory = historyFensNewestFirst
    ? [...historyFensNewestFirst].reverse()
    : chronological.map((position) => typeof position === 'string' ? boardToFen(parseFen(position)) : boardToFen(position));
  return {
    positions: chronological.concat(board),
    prepared: {
      board,
      fen,
      legalMoves: mappedMoves,
      explicitHistory,
      cacheKey: `history:${cacheHistory.length + 1}\n${cacheHistory.concat(fen).join('\n')}`,
    },
  };
}

export function currentBoardAndFen(input: Lc0EvaluatorInput): { board: BoardState; fen: string; preparedLegalMoves?: readonly Lc0PreparedLegalMove[] } {
  if (typeof input === 'object' && input !== null && 'prepared' in input) {
    return { board: input.prepared.board, fen: input.prepared.fen, preparedLegalMoves: input.prepared.legalMoves };
  }
  if (typeof input === 'object' && input !== null && 'positions' in input) {
    if (input.positions.length === 0) throw new Error('LC0 evaluator history input requires at least one position');
    const last = input.positions[input.positions.length - 1];
    const board = typeof last === 'string' ? parseFen(last) : last;
    return { board, fen: boardToFen(board) };
  }
  const board = typeof input === 'string' ? parseFen(input) : input;
  return { board, fen: typeof input === 'string' ? input : boardToFen(board) };
}

export function legalPolicyPriors(board: BoardState, logits: ArrayLike<number>, policyTemperature: number, prepared?: readonly Lc0PreparedLegalMove[]): Lc0LegalPrior[] {
  const moveTransform = board.turn === 'b' ? LC0_MIRROR_TRANSFORM : 0;
  const legal = (prepared ?? legalMoves(board)).map((entry) => {
    if ('policyIndex' in entry) {
      return { uci: entry.uci, index: entry.policyIndex, actionId: entry.actionId, logit: Number(logits[entry.policyIndex]) / policyTemperature };
    }
    const move = entry;
    const uci = moveToUci(move);
    const index = uciToLc0PolicyIndex(uci, moveTransform, { standardCastling: isStandardCastlingMove(board, move) });
    if (index === undefined) throw new Error(`No LC0 policy index for legal move ${uci}`);
    return { uci, index, logit: Number(logits[index]) / policyTemperature };
  });
  if (legal.length === 0) return [];
  const max = Math.max(...legal.map((entry) => entry.logit));
  const sum = legal.reduce((acc, entry) => acc + Math.exp(entry.logit - max), 0);
  return legal
    .map((entry) => ({ ...entry, prior: Math.exp(entry.logit - max) / sum }))
    .sort((a, b) => b.prior - a.prior);
}

export class CachedLc0Evaluator implements Lc0EvaluationProvider {
  readonly inner: Lc0EvaluationProvider;
  private maxEntries: number;
  private hits = 0;
  private misses = 0;
  private readonly cache = new Map<string, Lc0Evaluation>();

  constructor(inner: Lc0EvaluationProvider, options: Lc0EvaluationCacheOptions = {}) {
    this.inner = inner;
    this.maxEntries = Math.max(0, Math.floor(options.maxEntries ?? 2048));
  }

  setMaxEntries(maxEntries: number): void {
    this.maxEntries = Math.max(0, Math.floor(maxEntries));
    this.evictIfNeeded();
  }

  clearCache(): void {
    this.cache.clear();
    this.hits = 0;
    this.misses = 0;
  }

  async dispose(): Promise<void> {
    this.clearCache();
    await this.inner.dispose?.();
  }

  metrics(): Lc0EvaluationCacheMetrics {
    return { hits: this.hits, misses: this.misses, entries: this.cache.size, maxEntries: this.maxEntries };
  }

  cacheFootprint(): Lc0EvaluationCacheFootprint {
    let approxKeyBytes = 0;
    let approxEvaluationBytes = 0;
    for (const [key, evaluation] of this.cache.entries()) {
      approxKeyBytes += approximateCacheKeyBytes(key);
      approxEvaluationBytes += approximateCachedEvaluationBytes(evaluation);
    }
    return {
      entries: this.cache.size,
      maxEntries: this.maxEntries,
      approxBytes: approxKeyBytes + approxEvaluationBytes,
      approxKeyBytes,
      approxEvaluationBytes,
      note: 'Approximate JS evaluator-cache payload bytes for keys, FENs, scalar outputs, and legal-prior entries; excludes JS object/map overhead and cached backend/runtime resources.',
    };
  }

  async evaluate(input: Lc0EvaluatorInput): Promise<Lc0Evaluation> {
    return (await this.evaluateBatch([input]))[0];
  }

  async evaluateBatch(inputs: Lc0EvaluatorInput[]): Promise<Lc0Evaluation[]> {
    if (!inputs.length) return [];
    const results = new Array<Lc0Evaluation>(inputs.length);
    const groups = new Map<string, { input: Lc0EvaluatorInput; slots: number[]; promise?: Promise<Lc0Evaluation> }>();

    for (let i = 0; i < inputs.length; i++) {
      const key = inputHistoryKey(inputs[i]);
      const cached = this.cachedValue(key);
      if (cached) {
        results[i] = cached;
        continue;
      }
      this.misses += 1;
      const group = groups.get(key);
      if (group) group.slots.push(i);
      else groups.set(key, { input: inputs[i], slots: [i], promise: this.inFlight.get(key) });
    }

    const fresh = [...groups.entries()].filter(([, group]) => !group.promise);
    if (fresh.length) {
      const deferred = fresh.map(([key, group]) => {
        const pending = this.createInFlight(key);
        group.promise = pending.promise;
        return { key, input: group.input, ...pending };
      });
      void this.runMissBatch(deferred);
    }

    await Promise.all([...groups.entries()].map(async ([, group]) => {
      const value = await group.promise!;
      for (const slot of group.slots) results[slot] = cloneEvaluation(value);
    }));
    return results;
  }

  async evaluateBatchSequence(batches: Lc0EvaluatorInput[][]): Promise<Lc0Evaluation[][]> {
    const out = batches.map((batch) => new Array<Lc0Evaluation>(batch.length));
    const groups = new Map<string, { input: Lc0EvaluatorInput; slots: Array<{ batch: number; slot: number }>; sourceBatch: number; promise?: Promise<Lc0Evaluation> }>();

    for (let batchIndex = 0; batchIndex < batches.length; batchIndex++) {
      for (let slot = 0; slot < batches[batchIndex].length; slot++) {
        const input = batches[batchIndex][slot];
        const key = inputHistoryKey(input);
        const cached = this.cachedValue(key);
        if (cached) {
          out[batchIndex][slot] = cached;
          continue;
        }
        this.misses += 1;
        const group = groups.get(key);
        if (group) group.slots.push({ batch: batchIndex, slot });
        else groups.set(key, { input, slots: [{ batch: batchIndex, slot }], sourceBatch: batchIndex, promise: this.inFlight.get(key) });
      }
    }

    const newBySourceBatch = new Map<number, Array<{ key: string; input: Lc0EvaluatorInput; promise: Promise<Lc0Evaluation>; resolve: (value: Lc0Evaluation) => void; reject: (error: unknown) => void }>>();
    for (const [key, group] of groups) {
      if (group.promise) continue;
      const pending = this.createInFlight(key);
      group.promise = pending.promise;
      const entries = newBySourceBatch.get(group.sourceBatch) ?? [];
      entries.push({ key, input: group.input, ...pending });
      newBySourceBatch.set(group.sourceBatch, entries);
    }
    if (newBySourceBatch.size) void this.runMissSequence([...newBySourceBatch.values()]);

    await Promise.all([...groups.values()].map(async (group) => {
      const value = await group.promise!;
      for (const target of group.slots) out[target.batch][target.slot] = cloneEvaluation(value);
    }));
    return out;
  }

  private readonly inFlight = new Map<string, Promise<Lc0Evaluation>>();

  private cachedValue(key: string): Lc0Evaluation | undefined {
    const cached = this.cache.get(key);
    if (!cached) return undefined;
    this.hits += 1;
    this.cache.delete(key);
    this.cache.set(key, cached);
    return cloneCachedEvaluation(cached);
  }

  private createInFlight(key: string): { promise: Promise<Lc0Evaluation>; resolve: (value: Lc0Evaluation) => void; reject: (error: unknown) => void } {
    let resolve!: (value: Lc0Evaluation) => void;
    let reject!: (error: unknown) => void;
    const promise = new Promise<Lc0Evaluation>((res, rej) => { resolve = res; reject = rej; });
    this.inFlight.set(key, promise);
    return { promise, resolve, reject };
  }

  private settleMiss(key: string, value: Lc0Evaluation): Lc0Evaluation {
    const cloned = cloneEvaluation(value);
    this.store(key, cloned);
    this.inFlight.delete(key);
    return cloned;
  }

  private async runMissBatch(entries: Array<{ key: string; input: Lc0EvaluatorInput; resolve: (value: Lc0Evaluation) => void; reject: (error: unknown) => void }>): Promise<void> {
    try {
      const evals = this.inner.evaluateBatch
        ? await this.inner.evaluateBatch(entries.map((entry) => entry.input))
        : await Promise.all(entries.map((entry) => this.inner.evaluate(entry.input)));
      if (evals.length !== entries.length) throw new Error(`LC0 evaluator returned ${evals.length} result(s), expected ${entries.length}`);
      for (let i = 0; i < entries.length; i++) entries[i].resolve(this.settleMiss(entries[i].key, evals[i]));
    } catch (error) {
      for (const entry of entries) {
        this.inFlight.delete(entry.key);
        entry.reject(error);
      }
    }
  }

  private async runMissSequence(batchEntries: Array<Array<{ key: string; input: Lc0EvaluatorInput; resolve: (value: Lc0Evaluation) => void; reject: (error: unknown) => void }>>): Promise<void> {
    try {
      const inputs = batchEntries.map((batch) => batch.map((entry) => entry.input));
      const results = this.inner.evaluateBatchSequence ? await this.inner.evaluateBatchSequence(inputs) : [];
      if (!this.inner.evaluateBatchSequence) {
        for (const batch of inputs) {
          results.push(this.inner.evaluateBatch
            ? await this.inner.evaluateBatch(batch)
            : await Promise.all(batch.map((input) => this.inner.evaluate(input))));
        }
      }
      if (results.length !== batchEntries.length) throw new Error(`LC0 evaluator returned ${results.length} sequence batch(es), expected ${batchEntries.length}`);
      for (let batch = 0; batch < batchEntries.length; batch++) {
        if (results[batch].length !== batchEntries[batch].length) throw new Error(`LC0 evaluator returned ${results[batch].length} result(s) for sequence batch ${batch}, expected ${batchEntries[batch].length}`);
        for (let i = 0; i < batchEntries[batch].length; i++) {
          const entry = batchEntries[batch][i];
          entry.resolve(this.settleMiss(entry.key, results[batch][i]));
        }
      }
    } catch (error) {
      for (const entries of batchEntries) {
        for (const entry of entries) {
          this.inFlight.delete(entry.key);
          entry.reject(error);
        }
      }
    }
  }

  private store(key: string, value: Lc0Evaluation): void {
    if (this.maxEntries <= 0) return;
    this.cache.set(key, cloneCachedEvaluation(value));
    this.evictIfNeeded();
  }

  private evictIfNeeded(): void {
    while (this.cache.size > this.maxEntries) {
      const oldest = this.cache.keys().next().value;
      if (oldest === undefined) break;
      this.cache.delete(oldest);
    }
  }
}

export class Lc0OnnxEvaluator implements Lc0EvaluationProvider {
  readonly policyTemperature: number;
  readonly historyFill: Lc0HistoryFill;
  private readonly session: ort.InferenceSession;
  private disposed = false;

  constructor(session: ort.InferenceSession, options: Lc0OnnxEvaluatorOptions = {}) {
    this.session = session;
    this.policyTemperature = options.policyTemperature ?? LC0_DEFAULT_POLICY_TEMPERATURE;
    this.historyFill = options.historyFill ?? 'fen_only';
  }

  static async create(modelPath: string | Uint8Array | ArrayBuffer, options: Lc0OnnxEvaluatorOptions = {}): Promise<Lc0OnnxEvaluator> {
    return new Lc0OnnxEvaluator(await ort.createOrtSession(modelPath), options);
  }

  async dispose(): Promise<void> {
    if (this.disposed) return;
    this.disposed = true;
    await ort.releaseOrtSession(this.session);
  }

  private assertNotDisposed(): void {
    if (this.disposed) throw new Error('LC0 ONNX evaluator has been disposed');
  }

  async evaluate(boardOrFen: Lc0EvaluatorInput): Promise<Lc0Evaluation> {
    return (await this.evaluateBatch([boardOrFen]))[0];
  }

  private async runPhysicalBatch(inputs: Lc0EvaluatorInput[], physicalBatchSize: number): Promise<Lc0Evaluation[]> {
    this.assertNotDisposed();
    if (!inputs.length) return [];
    const totalStarted = ort.tinyLeelaNowMs();
    const encodedPlanes = new Float32Array(physicalBatchSize * LC0_INPUT_PLANES_SIZE);
    const boards: BoardState[] = [];
    const fens: string[] = [];
    const preparedLegalMoves: Array<readonly Lc0PreparedLegalMove[] | undefined> = [];
    const encodeStarted = ort.tinyLeelaNowMs();
    for (let i = 0; i < inputs.length; i++) {
      const input = inputs[i];
      const current = currentBoardAndFen(input);
      boards.push(current.board);
      fens.push(current.fen);
      preparedLegalMoves.push(current.preparedLegalMoves);
      encodedPlanes.set(encodeLc0Classical112(input as Lc0EncoderInput, { historyFill: this.historyFill }).planes, i * LC0_INPUT_PLANES_SIZE);
    }
    // Fixed batch-N artifacts require a full physical batch. Pad by copying the
    // last encoded position rather than re-encoding it for every unused slot;
    // callers only receive the real `inputs.length` outputs below.
    for (let i = inputs.length; i < physicalBatchSize; i++) {
      encodedPlanes.copyWithin(i * LC0_INPUT_PLANES_SIZE, (inputs.length - 1) * LC0_INPUT_PLANES_SIZE, inputs.length * LC0_INPUT_PLANES_SIZE);
    }
    const encodeMs = ort.tinyLeelaNowMs() - encodeStarted;
    const inputType = sessionInputType(this.session);
    const inputTensor = inputType === 'float16'
      ? new ort.Tensor('float16', float32ToFloat16Array(encodedPlanes), [physicalBatchSize, 112, 8, 8])
      : new ort.Tensor('float32', encodedPlanes, [physicalBatchSize, 112, 8, 8]);
    const webgpuBefore = ort.getOrtWebGpuDiagnosticsSnapshot();
    const ortRunStarted = ort.tinyLeelaNowMs();
    const outputs = await this.session.run({
      [LC0_ONNX_INPUT_PLANES]: inputTensor,
    });
    const ortRunMs = ort.tinyLeelaNowMs() - ortRunStarted;
    const downloadTiming: Record<string, number> = {};
    const allGetDataStarted = ort.tinyLeelaNowMs();
    // Older/smaller nets (e.g. Maia) have no moves-left head; treat MLH as 0.
    const hasMlh = LC0_ONNX_OUTPUT_MLH in outputs;
    const [policy, wdlRaw, mlhRaw] = await Promise.all([
      tensorDataTimed(outputs, LC0_ONNX_OUTPUT_POLICY, 'ortPolicy', downloadTiming),
      tensorDataTimed(outputs, LC0_ONNX_OUTPUT_WDL, 'ortWdl', downloadTiming),
      hasMlh ? tensorDataTimed(outputs, LC0_ONNX_OUTPUT_MLH, 'ortMlh', downloadTiming) : Promise.resolve(undefined),
    ]);
    const allGetDataMs = ort.tinyLeelaNowMs() - allGetDataStarted;
    await ort.waitForOrtWebGpuDiagnostics();
    const postprocessStarted = ort.tinyLeelaNowMs();
    const webgpuDelta = ort.subtractOrtWebGpuDiagnosticsSnapshot(ort.getOrtWebGpuDiagnosticsSnapshot(), webgpuBefore);
    const baseTiming: Record<string, number | string | undefined> = {
      backend: 'ort-onnx',
      inputBuildMs: encodeMs,
      ortRunMs,
      sessionRunMs: ortRunMs,
      ortAllGetDataMs: allGetDataMs,
      readbackSyncedMs: allGetDataMs,
      readbackBytes: physicalBatchSize * (LC0_POLICY_SIZE + LC0_WDL_SIZE + LC0_MLH_SIZE) * 4,
      readbackMapCount: webgpuDelta.api.mapAsyncCount,
      ortKernelCount: webgpuDelta.profiling.eventCount,
      ortKernelGpuMs: webgpuDelta.profiling.kernelGpuMsTotal,
      webgpuSubmitCount: webgpuDelta.api.submitCount,
      webgpuSubmittedCommandBufferCount: webgpuDelta.api.submittedCommandBufferCount,
      webgpuMapAsyncCount: webgpuDelta.api.mapAsyncCount,
      webgpuMapAsyncMs: webgpuDelta.api.mapAsyncMsTotal,
      webgpuCopyBufferToBufferCount: webgpuDelta.api.copyBufferToBufferCount,
      webgpuCopyBufferToBufferBytes: webgpuDelta.api.copyBufferToBufferBytes,
      webgpuMapReadBufferCount: webgpuDelta.api.mapReadBufferCount,
      webgpuMapReadBufferBytes: webgpuDelta.api.mapReadBufferBytes,
      webgpuCreateBufferCount: webgpuDelta.api.createBufferCount,
      webgpuCreateBufferBytes: webgpuDelta.api.createBufferBytes,
      webgpuComputePipelineCreateCount: webgpuDelta.api.computePipelineCreateCount,
      webgpuComputePipelineCreateAsyncCount: webgpuDelta.api.computePipelineCreateAsyncCount,
      batchPosition: 0,
      physicalBatchSize: inputs.length,
      ortPhysicalBatchSize: physicalBatchSize,
      ...downloadTiming,
    };
    const results: Lc0Evaluation[] = inputs.map((_, i) => {
      const legalPriorsStarted = ort.tinyLeelaNowMs();
      const wdlSlice = arraySlice(wdlRaw, i * LC0_WDL_SIZE, LC0_WDL_SIZE);
      const wdl: [number, number, number] = [Number(wdlSlice[0]), Number(wdlSlice[1]), Number(wdlSlice[2])];
      const legalPriors = legalPolicyPriors(boards[i], arraySlice(policy, i * LC0_POLICY_SIZE, LC0_POLICY_SIZE), this.policyTemperature, preparedLegalMoves[i]);
      const legalPriorsMs = ort.tinyLeelaNowMs() - legalPriorsStarted;
      return {
        fen: fens[i],
        wdl,
        q: wdl[0] - wdl[2],
        mlh: mlhRaw === undefined ? 0 : Number(arraySlice(mlhRaw, i * LC0_MLH_SIZE, LC0_MLH_SIZE)[0]),
        legalPriors,
        bestMove: legalPriors[0]?.uci,
        timing: { ...baseTiming, batchPosition: i, legalPriorsMs },
      };
    });
    const postprocessMs = ort.tinyLeelaNowMs() - postprocessStarted;
    const totalEvalMs = ort.tinyLeelaNowMs() - totalStarted;
    for (const result of results) {
      const timing = result.timing as Record<string, unknown>;
      timing.postprocessMs = postprocessMs;
      timing.totalEvalMs = totalEvalMs;
    }
    return results;
  }

  async evaluateBatch(inputs: Lc0EvaluatorInput[]): Promise<Lc0Evaluation[]> {
    this.assertNotDisposed();
    if (!inputs.length) return [];
    const physicalBatchSize = sessionFixedInputBatchSize(this.session);
    const out: Lc0Evaluation[] = [];
    for (let offset = 0; offset < inputs.length; offset += physicalBatchSize) {
      out.push(...await this.runPhysicalBatch(inputs.slice(offset, offset + physicalBatchSize), physicalBatchSize));
    }
    return out;
  }

  async evaluateBatchSequence(batches: Lc0EvaluatorInput[][]): Promise<Lc0Evaluation[][]> {
    const out: Lc0Evaluation[][] = [];
    for (const batch of batches) out.push(await this.evaluateBatch(batch));
    return out;
  }
}
