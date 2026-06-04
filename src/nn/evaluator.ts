import { boardToFen, type BoardState } from '../chess/board.ts';
import { legalMoves } from '../chess/movegen.ts';
import { moveToActionId, type Move } from '../chess/moveCodec.ts';

export interface Evaluation {
  policy: Map<number, number>;
  wdl: [win: number, draw: number, loss: number];
  /** Optional auxiliary WDL heads keyed by ONNX/training head name, e.g. wdl_sf18. */
  auxiliaryWdls?: Record<string, [win: number, draw: number, loss: number]>;
  /** Optional per-legal-action value from the current side-to-move perspective, usually in [-1, 1]. */
  actionValues?: Map<number, number>;
  /** Optional per-legal-action rank/regret/risk/uncertainty signals for experimental search policies. */
  rankScores?: Map<number, number>;
  regrets?: Map<number, number>;
  risks?: Map<number, number>;
  uncertainties?: Map<number, number>;
}

export interface EvaluationContext {
  /** Previous position FENs, newest first. */
  historyFens?: string[];
  /** Optional precomputed legal moves for this board, used to avoid duplicate movegen in search hot paths. */
  legalMoves?: Move[];
  /** Optional diagnostic-only multiplier for ThreatGraph attack-summary channels. Not used by search. */
  attackSummaryChannelMask?: ArrayLike<number>;
}

export interface EvaluationBatchRequest {
  boards: BoardState[];
  contexts?: EvaluationContext[];
}

export interface Evaluator {
  evaluate(board: BoardState, context?: EvaluationContext): Promise<Evaluation> | Evaluation;
  evaluateBatch?(boards: BoardState[], contexts?: EvaluationContext[]): Promise<Evaluation[]> | Evaluation[];
  /**
   * Optional sequence API for evaluators that can submit/read back multiple
   * physical batches more efficiently than separate awaited evaluateBatch calls.
   * Search still consumes results in request order.
   */
  evaluateBatchSequence?(batches: EvaluationBatchRequest[]): Promise<Evaluation[][]> | Evaluation[][];
}

export class UniformEvaluator implements Evaluator {
  evaluate(board: BoardState): Evaluation {
    const moves = legalMoves(board);
    const policy = new Map<number, number>();
    const p = moves.length ? 1 / moves.length : 0;
    for (const move of moves) policy.set(moveToActionId(move), p);
    return { policy, wdl: [0.33, 0.34, 0.33] };
  }
  evaluateBatch(boards: BoardState[]): Evaluation[] { return boards.map((board) => this.evaluate(board)); }
}

export interface CachedEvaluatorMetrics {
  hits: number;
  misses: number;
  inserts: number;
  evictions: number;
}

export interface CachedEvaluatorOptions {
  /** Maximum cached positions. Defaults to 8192 for browser memory safety. */
  maxEntries?: number;
  /** Include history FENs in the key. Keep true for h8/history-aware encoders. */
  includeHistory?: boolean;
  /** Include legal move action IDs in the key. Keep true for move-token models. */
  includeLegalMoves?: boolean;
  /** Human/debug label. */
  label?: string;
}

function cloneEvaluation(evaln: Evaluation): Evaluation {
  return {
    policy: new Map(evaln.policy),
    wdl: [evaln.wdl[0], evaln.wdl[1], evaln.wdl[2]],
    ...(evaln.auxiliaryWdls ? { auxiliaryWdls: Object.fromEntries(Object.entries(evaln.auxiliaryWdls).map(([k, v]) => [k, [v[0], v[1], v[2]] as [number, number, number]])) } : {}),
    ...(evaln.actionValues ? { actionValues: new Map(evaln.actionValues) } : {}),
    ...(evaln.rankScores ? { rankScores: new Map(evaln.rankScores) } : {}),
    ...(evaln.regrets ? { regrets: new Map(evaln.regrets) } : {}),
    ...(evaln.risks ? { risks: new Map(evaln.risks) } : {}),
    ...(evaln.uncertainties ? { uncertainties: new Map(evaln.uncertainties) } : {}),
  };
}

function legalMoveKey(board: BoardState, context: EvaluationContext | undefined): string {
  const moves = context?.legalMoves ?? legalMoves(board);
  // Preserve order: move-token exports can consume legal moves as an ordered input sequence.
  return moves.map((move) => moveToActionId(move)).join(',');
}

function evalCacheKey(board: BoardState, context: EvaluationContext | undefined, includeHistory: boolean, includeLegalMoves: boolean): string {
  const fen = boardToFen(board);
  const history = includeHistory ? (context?.historyFens ?? []).join('|') : '';
  const moves = includeLegalMoves ? legalMoveKey(board, context) : '';
  const attackMask = context?.attackSummaryChannelMask ? Array.from(context.attackSummaryChannelMask).join(',') : '';
  return `${fen}\nh:${history}\nl:${moves}\na:${attackMask}`;
}

/**
 * Browser-side neural-eval cache for playable frontend search/render paths.
 *
 * Keyed by current FEN + history FENs + ordered legal move action IDs by default.
 * Including legal moves matters for MoveFormer/SquareFormer move-token exports where
 * the provided legal list is part of the evaluator input, not just a convenience.
 */
export interface BrokeredEvaluatorMetrics {
  logicalRequests: number;
  logicalPositions: number;
  flushes: number;
  positionsEvaluated: number;
  maxQueueDepth: number;
  batchSizes: Record<number, number>;
  waitMs: number[];
  runMs: number[];
}

export interface BrokeredEvaluatorOptions {
  /** Maximum physical batch size sent to the wrapped evaluator. */
  maxBatchSize?: number;
  /** Milliseconds to wait for more concurrent requests before flushing. 0 flushes on next microtask. */
  maxWaitMs?: number;
  /** Human/debug label. */
  label?: string;
}

export class CachedEvaluator implements Evaluator {
  readonly inner: Evaluator;
  readonly label: string;
  private readonly maxEntries: number;
  private readonly includeHistory: boolean;
  private readonly includeLegalMoves: boolean;
  private cache = new Map<string, Evaluation>();
  private metricsState: CachedEvaluatorMetrics = { hits: 0, misses: 0, inserts: 0, evictions: 0 };

  constructor(inner: Evaluator, options: CachedEvaluatorOptions = {}) {
    this.inner = inner;
    this.maxEntries = Math.max(1, Math.floor(options.maxEntries ?? 8192));
    this.includeHistory = options.includeHistory ?? true;
    this.includeLegalMoves = options.includeLegalMoves ?? true;
    this.label = options.label ?? 'eval';
  }

  metrics(): CachedEvaluatorMetrics & { entries: number; hitRate: number } {
    const total = this.metricsState.hits + this.metricsState.misses;
    return { ...this.metricsState, entries: this.cache.size, hitRate: total ? this.metricsState.hits / total : 0 };
  }

  clear() {
    this.cache.clear();
  }

  private get(key: string): Evaluation | undefined {
    const cached = this.cache.get(key);
    if (!cached) return undefined;
    // Refresh insertion order for approximate LRU.
    this.cache.delete(key);
    this.cache.set(key, cached);
    this.metricsState.hits++;
    return cloneEvaluation(cached);
  }

  private put(key: string, evaln: Evaluation) {
    if (this.cache.has(key)) this.cache.delete(key);
    this.cache.set(key, cloneEvaluation(evaln));
    this.metricsState.inserts++;
    while (this.cache.size > this.maxEntries) {
      const oldest = this.cache.keys().next().value as string | undefined;
      if (oldest === undefined) break;
      this.cache.delete(oldest);
      this.metricsState.evictions++;
    }
  }

  async evaluate(board: BoardState, context: EvaluationContext = {}): Promise<Evaluation> {
    return (await this.evaluateBatch([board], [context]))[0];
  }

  async evaluateBatch(boards: BoardState[], contexts: EvaluationContext[] = []): Promise<Evaluation[]> {
    const out = new Array<Evaluation>(boards.length);
    const missingBoards: BoardState[] = [];
    const missingContexts: EvaluationContext[] = [];
    const missingKeys: string[] = [];
    const firstMissingSlotByKey = new Map<string, number>();
    const duplicateSlots: { slot: number; firstSlot: number }[] = [];

    for (let i = 0; i < boards.length; i++) {
      const context = contexts[i] ?? {};
      const key = evalCacheKey(boards[i], context, this.includeHistory, this.includeLegalMoves);
      const cached = this.get(key);
      if (cached) {
        out[i] = cached;
        continue;
      }
      const firstSlot = firstMissingSlotByKey.get(key);
      if (firstSlot !== undefined) {
        duplicateSlots.push({ slot: i, firstSlot });
        continue;
      }
      this.metricsState.misses++;
      firstMissingSlotByKey.set(key, i);
      missingKeys.push(key);
      missingBoards.push(boards[i]);
      missingContexts.push(context);
    }

    if (missingBoards.length) {
      const evals = this.inner.evaluateBatch
        ? await this.inner.evaluateBatch(missingBoards, missingContexts)
        : await Promise.all(missingBoards.map((board, i) => this.inner.evaluate(board, missingContexts[i])));
      for (let j = 0; j < evals.length; j++) {
        const slot = firstMissingSlotByKey.get(missingKeys[j]);
        if (slot === undefined) continue;
        this.put(missingKeys[j], evals[j]);
        out[slot] = cloneEvaluation(evals[j]);
      }
    }

    for (const dup of duplicateSlots) out[dup.slot] = cloneEvaluation(out[dup.firstSlot]);
    return out;
  }
}

type BrokerPending = {
  boards: BoardState[];
  contexts: EvaluationContext[];
  enqueuedAt: number;
  resolve: (evals: Evaluation[]) => void;
  reject: (err: unknown) => void;
};

function currentMs(): number {
  return typeof performance === 'undefined' ? Date.now() : performance.now();
}

/**
 * Coalesces concurrent evaluate/evaluateBatch calls into larger physical batches.
 *
 * This is intentionally separate from CachedEvaluator: wrap as
 * `new BrokeredEvaluator(new CachedEvaluator(baseEval, ...), ...)` so concurrent
 * searches/tuner candidates share one queue and one cache-backed evaluator.
 */
export class BrokeredEvaluator implements Evaluator {
  readonly inner: Evaluator;
  readonly label: string;
  private readonly maxBatchSize: number;
  private readonly maxWaitMs: number;
  private pending: BrokerPending[] = [];
  private flushTimer: ReturnType<typeof setTimeout> | null = null;
  private flushing = false;
  private metricsState: BrokeredEvaluatorMetrics = {
    logicalRequests: 0,
    logicalPositions: 0,
    flushes: 0,
    positionsEvaluated: 0,
    maxQueueDepth: 0,
    batchSizes: {},
    waitMs: [],
    runMs: [],
  };

  constructor(inner: Evaluator, options: BrokeredEvaluatorOptions = {}) {
    this.inner = inner;
    this.maxBatchSize = Math.max(1, Math.floor(options.maxBatchSize ?? 64));
    this.maxWaitMs = Math.max(0, Number(options.maxWaitMs ?? 0));
    this.label = options.label ?? 'brokered-eval';
  }

  metrics(): BrokeredEvaluatorMetrics & { avgBatch: number } {
    return {
      ...this.metricsState,
      batchSizes: { ...this.metricsState.batchSizes },
      waitMs: [...this.metricsState.waitMs],
      runMs: [...this.metricsState.runMs],
      avgBatch: this.metricsState.positionsEvaluated / Math.max(1, this.metricsState.flushes),
    };
  }

  async evaluate(board: BoardState, context: EvaluationContext = {}): Promise<Evaluation> {
    return (await this.evaluateBatch([board], [context]))[0];
  }

  evaluateBatch(boards: BoardState[], contexts: EvaluationContext[] = []): Promise<Evaluation[]> {
    this.metricsState.logicalRequests += 1;
    this.metricsState.logicalPositions += boards.length;
    return new Promise((resolve, reject) => {
      this.pending.push({ boards, contexts, enqueuedAt: currentMs(), resolve, reject });
      this.metricsState.maxQueueDepth = Math.max(this.metricsState.maxQueueDepth, this.pending.reduce((s, p) => s + p.boards.length, 0));
      this.scheduleFlush();
    });
  }

  async drain(): Promise<void> {
    while (this.pending.length || this.flushing) {
      if (this.flushTimer) {
        clearTimeout(this.flushTimer);
        this.flushTimer = null;
      }
      if (this.pending.length && !this.flushing) void this.flush();
      await new Promise((resolve) => setTimeout(resolve, 0));
    }
  }

  private scheduleFlush() {
    const queued = this.pending.reduce((s, p) => s + p.boards.length, 0);
    if (queued >= this.maxBatchSize) {
      if (this.flushTimer) clearTimeout(this.flushTimer);
      this.flushTimer = null;
      void this.flush();
      return;
    }
    if (this.flushTimer || this.flushing) return;
    this.flushTimer = setTimeout(() => {
      this.flushTimer = null;
      void this.flush();
    }, this.maxWaitMs);
  }

  private takeBatch(): BrokerPending[] {
    const batch: BrokerPending[] = [];
    let n = 0;
    while (this.pending.length) {
      const next = this.pending[0];
      if (batch.length && n + next.boards.length > this.maxBatchSize) break;
      batch.push(this.pending.shift() as BrokerPending);
      n += next.boards.length;
      if (n >= this.maxBatchSize) break;
    }
    // If a single logical request is larger than maxBatchSize, allow it through;
    // inner evaluators already support their own fixed/static batching fallbacks.
    if (!batch.length && this.pending.length) batch.push(this.pending.shift() as BrokerPending);
    return batch;
  }

  private async flush() {
    if (this.flushing) return;
    this.flushing = true;
    try {
      while (this.pending.length) {
        const batch = this.takeBatch();
        const now = currentMs();
        const boards = batch.flatMap((p) => p.boards);
        const contexts = batch.flatMap((p) => p.boards.map((_, i) => p.contexts[i] ?? {}));
        for (const p of batch) this.metricsState.waitMs.push(now - p.enqueuedAt);
        const t0 = currentMs();
        try {
          const evals = this.inner.evaluateBatch
            ? await this.inner.evaluateBatch(boards, contexts)
            : await Promise.all(boards.map((board, i) => this.inner.evaluate(board, contexts[i])));
          this.metricsState.flushes += 1;
          this.metricsState.positionsEvaluated += boards.length;
          this.metricsState.batchSizes[boards.length] = (this.metricsState.batchSizes[boards.length] ?? 0) + 1;
          this.metricsState.runMs.push(currentMs() - t0);
          let offset = 0;
          for (const p of batch) {
            const slice = evals.slice(offset, offset + p.boards.length).map(cloneEvaluation);
            offset += p.boards.length;
            p.resolve(slice);
          }
        } catch (err) {
          for (const p of batch) p.reject(err);
        }
      }
    } finally {
      this.flushing = false;
      if (this.pending.length) this.scheduleFlush();
    }
  }
}
