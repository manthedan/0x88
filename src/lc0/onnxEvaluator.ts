import * as ort from '../nn/ortRuntime.ts';
import { boardToFen, parseFen, type BoardState } from '../chess/board.ts';
import { legalMoves } from '../chess/movegen.ts';
import { moveToUci, type Move } from '../chess/moveCodec.ts';
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
  logit: number;
  prior: number;
}

export interface Lc0Evaluation {
  fen: string;
  wdl: [number, number, number];
  q: number;
  mlh: number;
  legalPriors: Lc0LegalPrior[];
  bestMove?: string;
}

export interface Lc0OnnxEvaluatorOptions {
  policyTemperature?: number;
  historyFill?: Lc0HistoryFill;
}

export type Lc0EvaluatorInput = BoardState | string | Lc0PositionHistoryInput;

export interface Lc0EvaluationProvider {
  evaluate(input: Lc0EvaluatorInput): Promise<Lc0Evaluation> | Lc0Evaluation;
  evaluateBatch?(inputs: Lc0EvaluatorInput[]): Promise<Lc0Evaluation[]> | Lc0Evaluation[];
  dispose?(): Promise<void> | void;
}

export interface Lc0EvaluationCacheMetrics {
  hits: number;
  misses: number;
  entries: number;
  maxEntries: number;
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

function tensorData(outputs: Awaited<ReturnType<ort.InferenceSession['run']>>, name: string): number[] | Float32Array {
  const tensor = outputs[name];
  if (!tensor) throw new Error(`LC0 ONNX output ${name} missing`);
  if (tensor.type === 'float16') {
    const data = tensor.data as ArrayLike<number> & { constructor?: { name?: string } };
    // ORT-web returns Uint16Array float16 bits in Node today, while modern
    // browsers may expose Float16Array values directly. Handle both forms.
    if (data.constructor?.name === 'Float16Array') return Array.from(data);
    return Array.from(data, f16ToF32);
  }
  return tensor.data as Float32Array | number[];
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

export function currentBoardAndFen(input: Lc0EvaluatorInput): { board: BoardState; fen: string } {
  if (typeof input === 'object' && input !== null && 'positions' in input) {
    if (input.positions.length === 0) throw new Error('LC0 evaluator history input requires at least one position');
    const last = input.positions[input.positions.length - 1];
    const board = typeof last === 'string' ? parseFen(last) : last;
    return { board, fen: boardToFen(board) };
  }
  const board = typeof input === 'string' ? parseFen(input) : input;
  return { board, fen: typeof input === 'string' ? input : boardToFen(board) };
}

export function legalPolicyPriors(board: BoardState, logits: ArrayLike<number>, policyTemperature: number): Lc0LegalPrior[] {
  const moveTransform = board.turn === 'b' ? LC0_MIRROR_TRANSFORM : 0;
  const legal = legalMoves(board).map((move) => {
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

  async evaluate(input: Lc0EvaluatorInput): Promise<Lc0Evaluation> {
    return (await this.evaluateBatch([input]))[0];
  }

  async evaluateBatch(inputs: Lc0EvaluatorInput[]): Promise<Lc0Evaluation[]> {
    if (!inputs.length) return [];
    const results = new Array<Lc0Evaluation>(inputs.length);
    const missInputs: Lc0EvaluatorInput[] = [];
    const missSlots: number[] = [];
    const missKeys: string[] = [];
    for (let i = 0; i < inputs.length; i++) {
      const key = inputHistoryKey(inputs[i]);
      const cached = this.cache.get(key);
      if (cached) {
        this.hits += 1;
        this.cache.delete(key);
        this.cache.set(key, cached);
        results[i] = cloneEvaluation(cached);
      } else {
        this.misses += 1;
        missInputs.push(inputs[i]);
        missSlots.push(i);
        missKeys.push(key);
      }
    }
    if (missInputs.length) {
      const evals = this.inner.evaluateBatch
        ? await this.inner.evaluateBatch(missInputs)
        : await Promise.all(missInputs.map((input) => this.inner.evaluate(input)));
      for (let i = 0; i < evals.length; i++) {
        const value = cloneEvaluation(evals[i]);
        this.store(missKeys[i], value);
        results[missSlots[i]] = cloneEvaluation(value);
      }
    }
    return results;
  }

  private store(key: string, value: Lc0Evaluation): void {
    if (this.maxEntries <= 0) return;
    this.cache.set(key, cloneEvaluation(value));
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
    const encodedPlanes = new Float32Array(physicalBatchSize * LC0_INPUT_PLANES_SIZE);
    const boards: BoardState[] = [];
    const fens: string[] = [];
    for (let i = 0; i < inputs.length; i++) {
      const input = inputs[i];
      const { board, fen } = currentBoardAndFen(input);
      boards.push(board);
      fens.push(fen);
      encodedPlanes.set(encodeLc0Classical112(input as Lc0EncoderInput, { historyFill: this.historyFill }).planes, i * LC0_INPUT_PLANES_SIZE);
    }
    // Fixed batch-N artifacts require a full physical batch. Pad by copying the
    // last encoded position rather than re-encoding it for every unused slot;
    // callers only receive the real `inputs.length` outputs below.
    for (let i = inputs.length; i < physicalBatchSize; i++) {
      encodedPlanes.copyWithin(i * LC0_INPUT_PLANES_SIZE, (inputs.length - 1) * LC0_INPUT_PLANES_SIZE, inputs.length * LC0_INPUT_PLANES_SIZE);
    }
    const inputType = sessionInputType(this.session);
    const inputTensor = inputType === 'float16'
      ? new ort.Tensor('float16', float32ToFloat16Array(encodedPlanes), [physicalBatchSize, 112, 8, 8])
      : new ort.Tensor('float32', encodedPlanes, [physicalBatchSize, 112, 8, 8]);
    const outputs = await this.session.run({
      [LC0_ONNX_INPUT_PLANES]: inputTensor,
    });
    const policy = tensorData(outputs, LC0_ONNX_OUTPUT_POLICY);
    const wdlRaw = tensorData(outputs, LC0_ONNX_OUTPUT_WDL);
    const mlhRaw = tensorData(outputs, LC0_ONNX_OUTPUT_MLH);
    return inputs.map((_, i) => {
      const wdlSlice = arraySlice(wdlRaw, i * LC0_WDL_SIZE, LC0_WDL_SIZE);
      const wdl: [number, number, number] = [Number(wdlSlice[0]), Number(wdlSlice[1]), Number(wdlSlice[2])];
      const legalPriors = legalPolicyPriors(boards[i], arraySlice(policy, i * LC0_POLICY_SIZE, LC0_POLICY_SIZE), this.policyTemperature);
      return {
        fen: fens[i],
        wdl,
        q: wdl[0] - wdl[2],
        mlh: Number(arraySlice(mlhRaw, i * LC0_MLH_SIZE, LC0_MLH_SIZE)[0]),
        legalPriors,
        bestMove: legalPriors[0]?.uci,
      };
    });
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
}
