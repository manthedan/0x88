import { boardToFen, parseFen, type BoardState } from '../chess/board.ts';
import { legalMoves } from '../chess/movegen.ts';
import { moveToActionId, moveToUci, type Move } from '../chess/moveCodec.ts';
import { searchRoot, type Node as PuctNode, type SearchOptions, type SearchResult } from '../search/puct.ts';
import type { Evaluation, EvaluationBatchRequest, EvaluationContext, Evaluator } from '../nn/evaluator.ts';
import { Lc0OnnxEvaluator, type Lc0EvaluationCacheMetrics, type Lc0EvaluationProvider, type Lc0EvaluatorInput, type Lc0Evaluation, type Lc0OnnxEvaluatorOptions } from './onnxEvaluator.ts';
import type { Lc0PositionHistoryInput } from './encoder112.ts';

export interface Lc0SearchOptions extends SearchOptions {
  /** Fixed PUCT visits. Kept explicit here because Phase 2 starts with fixed-visit search parity. */
  visits?: number;
  /** Reuse the previous compatible search subtree, mirroring native LC0's practical tree reuse between moves. */
  reuseTree?: boolean;
}

export interface Lc0SearchResult {
  fen: string;
  move?: string;
  visits: number;
  value: number;
  children: Lc0SearchChild[];
  /** Principal variation as UCI move strings from the root, best line first. */
  pv: string[];
  /** MultiPV lines (UCI), one per top root move, present only when multiPv > 1. */
  multiPv?: string[][];
  search: SearchResult;
}

export interface Lc0SearchChild {
  uci: string;
  visits: number;
  prior: number;
  q: number;
  probability: number;
}

function currentBoardAndHistory(input: Lc0EvaluatorInput): { board: BoardState; historyFens: string[] } {
  if (typeof input === 'object' && input !== null && 'positions' in input) {
    if (input.positions.length === 0) throw new Error('LC0 search history input requires at least one position');
    const boards = input.positions.map((position) => typeof position === 'string' ? parseFen(position) : position);
    const board = boards[boards.length - 1];
    return { board, historyFens: boards.slice(0, -1).reverse().map(boardToFen) };
  }
  return { board: typeof input === 'string' ? parseFen(input) : input, historyFens: [] };
}

function contextInput(board: BoardState, context?: EvaluationContext): BoardState | Lc0PositionHistoryInput {
  const history = context?.historyFens ?? [];
  if (!history.length) return board;
  return { positions: [...history].reverse().map(parseFen).concat(board) };
}

function lc0ToSearchEvaluation(board: BoardState, lc0: Lc0Evaluation, context?: EvaluationContext): Evaluation {
  const policy = new Map<number, number>();
  const legalByUci = new Map<string, Move>((context?.legalMoves ?? legalMoves(board)).map((move) => [moveToUci(move), move]));
  for (const prior of lc0.legalPriors) {
    const move = legalByUci.get(prior.uci);
    if (move) policy.set(moveToActionId(move), prior.prior);
  }
  const evaluation: Evaluation & { timing?: unknown } = { policy, wdl: lc0.wdl };
  if (Number.isFinite(lc0.mlh)) evaluation.movesLeft = lc0.mlh;
  const timing = (lc0 as { timing?: unknown }).timing;
  if (timing !== undefined) evaluation.timing = timing;
  return evaluation;
}

export class Lc0SearchEvaluator implements Evaluator {
  readonly inner: Lc0EvaluationProvider;

  constructor(inner: Lc0EvaluationProvider) {
    this.inner = inner;
  }

  async evaluate(board: BoardState, context?: EvaluationContext): Promise<Evaluation> {
    const lc0 = await this.inner.evaluate(contextInput(board, context));
    return lc0ToSearchEvaluation(board, lc0, context);
  }

  async evaluateBatch(boards: BoardState[], contexts: EvaluationContext[] = []): Promise<Evaluation[]> {
    const inputs = boards.map((board, i) => contextInput(board, contexts[i]));
    const evals = this.inner.evaluateBatch
      ? await this.inner.evaluateBatch(inputs)
      : await Promise.all(inputs.map((input) => this.inner.evaluate(input)));
    return evals.map((lc0, i) => lc0ToSearchEvaluation(boards[i], lc0, contexts[i]));
  }

  async evaluateBatchSequence(batches: EvaluationBatchRequest[]): Promise<Evaluation[][]> {
    const inputBatches = batches.map((batch) => batch.boards.map((board, i) => contextInput(board, batch.contexts?.[i])));
    const lc0Batches = this.inner.evaluateBatchSequence ? await this.inner.evaluateBatchSequence(inputBatches) : [];
    if (!this.inner.evaluateBatchSequence) {
      for (const inputs of inputBatches) {
        lc0Batches.push(this.inner.evaluateBatch
          ? await this.inner.evaluateBatch(inputs)
          : await Promise.all(inputs.map((input) => this.inner.evaluate(input))));
      }
    }
    return lc0Batches.map((evals, batchIndex) => evals.map((lc0, i) => lc0ToSearchEvaluation(batches[batchIndex].boards[i], lc0, batches[batchIndex].contexts?.[i])));
  }

  metrics(): Lc0EvaluationCacheMetrics | undefined {
    return (this.inner as Lc0EvaluationProvider & { metrics?: () => Lc0EvaluationCacheMetrics }).metrics?.();
  }
}

export class Lc0PuctSearcher {
  private readonly evaluator: Lc0SearchEvaluator;
  private cachedRoot: PuctNode | null = null;

  constructor(evaluator: Lc0SearchEvaluator | Lc0EvaluationProvider) {
    this.evaluator = evaluator instanceof Lc0SearchEvaluator ? evaluator : new Lc0SearchEvaluator(evaluator);
  }

  resetTree(): void {
    this.cachedRoot = null;
  }

  private compatibleCachedRoot(board: BoardState, historyFens: string[]): PuctNode | null {
    const targetFen = boardToFen(board);
    const sameHistory = (candidate: string[]) => candidate.length === historyFens.length && candidate.every((fen, i) => fen === historyFens[i]);
    const root = this.cachedRoot;
    if (!root) return null;
    const seen = new Set<PuctNode>();
    const find = (node: PuctNode): PuctNode | null => {
      if (seen.has(node)) return null;
      seen.add(node);
      if (boardToFen(node.board) === targetFen && sameHistory(node.historyFens)) return node;
      if (!node.expanded) return null;
      for (const edge of node.edges) {
        if (!edge.child) continue;
        const found = find(edge.child);
        if (found) return found;
      }
      return null;
    };
    const found = find(root);
    if (!found) return null;
    root.isRoot = false;
    found.isRoot = true;
    return found;
  }

  static async create(modelPath: string | Uint8Array | ArrayBuffer, options: Lc0OnnxEvaluatorOptions = {}): Promise<Lc0PuctSearcher> {
    return new Lc0PuctSearcher(await Lc0OnnxEvaluator.create(modelPath, options));
  }

  async search(input: Lc0EvaluatorInput, options: Lc0SearchOptions = {}): Promise<Lc0SearchResult> {
    const { board, historyFens } = currentBoardAndHistory(input);
    const { reuseTree = false, ...searchOptions } = options;
    const hasExplicitRoot = Object.prototype.hasOwnProperty.call(searchOptions, 'root');
    const useInternalTree = reuseTree && !hasExplicitRoot && !searchOptions.rootMoves;
    const root = hasExplicitRoot
      ? searchOptions.root
      : useInternalTree ? this.compatibleCachedRoot(board, historyFens) : undefined;
    const result = await searchRoot(board, this.evaluator, {
      ...searchOptions,
      visits: options.visits ?? (options.movetimeMs && options.movetimeMs > 0 ? Number.MAX_SAFE_INTEGER : 32),
      temperature: options.temperature ?? 0,
      cpuctSchedule: options.cpuctSchedule ?? 'lc0-log',
      fpuStrategy: options.fpuStrategy ?? 'lc0-reduction',
      includePv: options.includePv ?? true,
      root,
      // History belongs to the LC0 input. Do not let generic SearchOptions
      // accidentally replace it and desynchronize the 112-plane encoder.
      historyFens,
    });
    if (useInternalTree) this.cachedRoot = result.root ?? null;
    const children = result.policy
      .map((entry) => ({ uci: moveToUci(entry.move), visits: entry.visits, prior: entry.prior, q: entry.q, probability: entry.probability }))
      .sort((a, b) => b.visits - a.visits || b.prior - a.prior);
    const pv = (result.principalVariation ?? []).map((entry) => moveToUci(entry.move));
    const multiPv = result.multiPvLines?.map((line) => line.map((entry) => moveToUci(entry.move)));
    return {
      fen: boardToFen(board),
      move: result.move ? moveToUci(result.move) : undefined,
      visits: result.visits,
      value: result.value,
      children,
      pv,
      ...(multiPv ? { multiPv } : {}),
      search: result,
    };
  }
}
