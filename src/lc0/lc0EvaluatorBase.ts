import { type BoardState, boardToFen, parseFen } from '../chess/board.ts';
import { type Move, moveToActionId, moveToUci } from '../chess/moveCodec.ts';
import { legalMoves } from '../chess/movegen.ts';
import { f16BitsToF32 } from '../nn/float16.ts';
import type { Lc0PositionHistoryInput } from './encoder112.ts';
import { LC0_MIRROR_TRANSFORM, uciToLc0PolicyIndex } from './policyMap.ts';

export const LC0_ONNX_INPUT_PLANES = '/input/planes';
export const LC0_ONNX_OUTPUT_POLICY = '/output/policy';
export const LC0_ONNX_OUTPUT_WDL = '/output/wdl';
export const LC0_ONNX_OUTPUT_MLH = '/output/mlh';
export const LC0_DEFAULT_POLICY_TEMPERATURE = 1.359;

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

function isStandardCastlingMove(board: BoardState, move: Move): boolean {
  const piece = board.squares[move.from];
  return piece?.[1] === 'k' && Math.abs((move.to % 8) - (move.from % 8)) === 2;
}

function inputHistoryKey(input: Lc0EvaluatorInput): string {
  if (typeof input === 'object' && input !== null && 'prepared' in input) return input.prepared.cacheKey;
  if (typeof input === 'object' && input !== null && 'positions' in input) {
    const positions = input.positions.map((position) => (typeof position === 'string' ? boardToFen(parseFen(position)) : boardToFen(position)));
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
  // Keep all eight real slots and each slot's own reversible repetition
  // horizon. Short histories stay whole: their oldest board controls FEN fill.
  let cacheLength = Math.max(7, board.halfmove);
  for (let i = 0; i < Math.min(7, historyNewestFirst.length); i++) {
    const position = historyNewestFirst[i];
    const previous = typeof position === 'string' ? parseFen(position) : position;
    cacheLength = Math.max(cacheLength, i + 1 + previous.halfmove);
  }
  const cacheStart = Math.max(0, chronological.length - cacheLength);
  const cacheHistory = historyFensNewestFirst
    ? historyFensNewestFirst.slice(0, chronological.length - cacheStart).reverse()
    : chronological.slice(cacheStart).map((position) => (typeof position === 'string' ? boardToFen(parseFen(position)) : boardToFen(position)));
  return {
    positions: chronological.concat(board),
    prepared: {
      board,
      fen,
      legalMoves: mappedMoves,
      explicitHistory,
      cacheKey: `history:${chronological.length + 1}:${explicitHistory}:${cacheStart}\n${cacheHistory.concat(fen).join('\n')}`,
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

export function legalPolicyPriors(
  board: BoardState,
  logits: ArrayLike<number>,
  policyTemperature: number,
  prepared?: readonly Lc0PreparedLegalMove[],
  policyOffset = 0,
  f16Bits = false,
): Lc0LegalPrior[] {
  const logitAt = (index: number) => {
    const value = logits[policyOffset + index];
    return (f16Bits && value !== undefined ? f16BitsToF32(value) : Number(value)) / policyTemperature;
  };
  const moveTransform = board.turn === 'b' ? LC0_MIRROR_TRANSFORM : 0;
  const legal = (prepared ?? legalMoves(board)).map((entry) => {
    if ('policyIndex' in entry) {
      return { uci: entry.uci, index: entry.policyIndex, actionId: entry.actionId, logit: logitAt(entry.policyIndex) };
    }
    const uci = moveToUci(entry);
    const index = uciToLc0PolicyIndex(uci, moveTransform, { standardCastling: isStandardCastlingMove(board, entry) });
    if (index === undefined) throw new Error(`No LC0 policy index for legal move ${uci}`);
    return { uci, index, logit: logitAt(index) };
  });
  if (legal.length === 0) return [];
  // Same contract as softmax() in src/nn/numerics.ts: fail here rather than hand NaN priors to search.
  const invalid = legal.find((entry) => Number.isNaN(entry.logit) || entry.logit === Infinity);
  if (invalid) throw new Error(`LC0 policy logit for ${invalid.uci} is not finite (${invalid.logit}) in ${boardToFen(board)}`);
  const max = Math.max(...legal.map((entry) => entry.logit));
  if (max === -Infinity) throw new Error(`LC0 policy has no finite legal-move logit in ${boardToFen(board)}`);
  const sum = legal.reduce((acc, entry) => acc + Math.exp(entry.logit - max), 0);
  return legal.map((entry) => ({ ...entry, prior: Math.exp(entry.logit - max) / sum })).sort((a, b) => b.prior - a.prior);
}

type PendingMiss = {
  key: string;
  input: Lc0EvaluatorInput;
  promise: Promise<Lc0Evaluation>;
  resolve: (value: Lc0Evaluation) => void;
  reject: (error: unknown) => void;
};

export class CachedLc0Evaluator implements Lc0EvaluationProvider {
  readonly inner: Lc0EvaluationProvider;
  private maxEntries: number;
  private hits = 0;
  private misses = 0;
  private readonly cache = new Map<string, Lc0Evaluation>();
  private readonly inFlight = new Map<string, Promise<Lc0Evaluation>>();

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
      approxKeyBytes += key.length * 2;
      // WDL[3], q, mlh as JS numbers (5 * 8); object overhead is excluded.
      approxEvaluationBytes +=
        evaluation.fen.length * 2 +
        (evaluation.bestMove?.length ?? 0) * 2 +
        40 +
        evaluation.legalPriors.reduce((sum, prior) => sum + prior.uci.length * 2 + 24, 0);
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
    if (!groups.size) return results;

    const deferred: PendingMiss[] = [];
    for (const [key, group] of groups) {
      if (group.promise) continue;
      const pending = this.createInFlight(key, group.input);
      group.promise = pending.promise;
      deferred.push(pending);
    }
    if (deferred.length) void this.runMissBatch(deferred);

    await Promise.all(
      [...groups.values()].map(async (group) => {
        const value = await group.promise!;
        for (const slot of group.slots) results[slot] = cloneEvaluation(value);
      }),
    );
    return results;
  }

  async evaluateBatchSequence(batches: Lc0EvaluatorInput[][]): Promise<Lc0Evaluation[][]> {
    const out = batches.map((batch) => new Array<Lc0Evaluation>(batch.length));
    const groups = new Map<
      string,
      { input: Lc0EvaluatorInput; slots: Array<{ batch: number; slot: number }>; sourceBatch: number; promise?: Promise<Lc0Evaluation> }
    >();

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
    if (!groups.size) return out;

    const newBySourceBatch = new Map<number, PendingMiss[]>();
    for (const [key, group] of groups) {
      if (group.promise) continue;
      const pending = this.createInFlight(key, group.input);
      group.promise = pending.promise;
      const entries = newBySourceBatch.get(group.sourceBatch) ?? [];
      entries.push(pending);
      newBySourceBatch.set(group.sourceBatch, entries);
    }
    if (newBySourceBatch.size) void this.runMissSequence([...newBySourceBatch.values()]);

    await Promise.all(
      [...groups.values()].map(async (group) => {
        const value = await group.promise!;
        for (const target of group.slots) out[target.batch][target.slot] = cloneEvaluation(value);
      }),
    );
    return out;
  }

  private cachedValue(key: string): Lc0Evaluation | undefined {
    const cached = this.cache.get(key);
    if (!cached) return undefined;
    this.hits += 1;
    this.cache.delete(key);
    this.cache.set(key, cached);
    return cloneCachedEvaluation(cached);
  }

  private createInFlight(key: string, input: Lc0EvaluatorInput): PendingMiss {
    let resolve!: (value: Lc0Evaluation) => void;
    let reject!: (error: unknown) => void;
    const promise = new Promise<Lc0Evaluation>((res, rej) => {
      resolve = res;
      reject = rej;
    });
    this.inFlight.set(key, promise);
    return { key, input, promise, resolve, reject };
  }

  private settleMissBatch(entries: PendingMiss[], evals: Lc0Evaluation[]): void {
    for (let i = 0; i < entries.length; i++) {
      const entry = entries[i];
      const cloned = cloneEvaluation(evals[i]);
      if (this.maxEntries > 0) {
        this.cache.set(entry.key, cloneCachedEvaluation(cloned));
        this.evictIfNeeded();
      }
      this.inFlight.delete(entry.key);
      entry.resolve(cloned);
    }
  }

  private rejectMissBatch(entries: PendingMiss[], error: unknown): void {
    for (const entry of entries) {
      this.inFlight.delete(entry.key);
      entry.reject(error);
    }
  }

  private async runMissBatch(entries: PendingMiss[]): Promise<void> {
    try {
      const evals = this.inner.evaluateBatch
        ? await this.inner.evaluateBatch(entries.map((entry) => entry.input))
        : await Promise.all(entries.map((entry) => this.inner.evaluate(entry.input)));
      if (evals.length !== entries.length) throw new Error(`LC0 evaluator returned ${evals.length} result(s), expected ${entries.length}`);
      this.settleMissBatch(entries, evals);
    } catch (error) {
      this.rejectMissBatch(entries, error);
    }
  }

  private async runMissSequence(batchEntries: PendingMiss[][]): Promise<void> {
    try {
      const inputs = batchEntries.map((batch) => batch.map((entry) => entry.input));
      let results: Lc0Evaluation[][];
      if (this.inner.evaluateBatchSequence) {
        results = await this.inner.evaluateBatchSequence(inputs);
      } else {
        results = [];
        for (const batch of inputs) {
          results.push(this.inner.evaluateBatch ? await this.inner.evaluateBatch(batch) : await Promise.all(batch.map((input) => this.inner.evaluate(input))));
        }
      }
      if (results.length !== batchEntries.length) {
        throw new Error(`LC0 evaluator returned ${results.length} sequence batch(es), expected ${batchEntries.length}`);
      }
      for (let batch = 0; batch < batchEntries.length; batch++) {
        if (results[batch].length !== batchEntries[batch].length) {
          throw new Error(`LC0 evaluator returned ${results[batch].length} result(s) for sequence batch ${batch}, expected ${batchEntries[batch].length}`);
        }
        this.settleMissBatch(batchEntries[batch], results[batch]);
      }
    } catch (error) {
      for (const entries of batchEntries) this.rejectMissBatch(entries, error);
    }
  }

  private evictIfNeeded(): void {
    while (this.cache.size > this.maxEntries) {
      const oldest = this.cache.keys().next().value;
      if (oldest === undefined) break;
      this.cache.delete(oldest);
    }
  }
}
