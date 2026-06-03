import { boardToFen, parseFen, type BoardState } from '../chess/board.ts';
import { legalMoves } from '../chess/movegen.ts';
import { moveToActionId, moveToUci, type Move } from '../chess/moveCodec.ts';
import { searchRoot, type SearchOptions, type SearchResult } from '../search/puct.ts';
import type { Evaluation, EvaluationContext, Evaluator } from '../nn/evaluator.ts';
import { Lc0OnnxEvaluator, type Lc0EvaluatorInput, type Lc0Evaluation, type Lc0OnnxEvaluatorOptions } from './onnxEvaluator.ts';
import type { Lc0PositionHistoryInput } from './encoder112.ts';

export interface Lc0SearchOptions extends SearchOptions {
  /** Fixed PUCT visits. Kept explicit here because Phase 2 starts with fixed-visit search parity. */
  visits?: number;
}

export interface Lc0SearchResult {
  fen: string;
  move?: string;
  visits: number;
  value: number;
  children: Lc0SearchChild[];
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

type Lc0EvaluationProvider = {
  evaluate(input: Lc0EvaluatorInput): Promise<Lc0Evaluation> | Lc0Evaluation;
  evaluateBatch?(inputs: Lc0EvaluatorInput[]): Promise<Lc0Evaluation[]> | Lc0Evaluation[];
};

function lc0ToSearchEvaluation(board: BoardState, lc0: Lc0Evaluation, context?: EvaluationContext): Evaluation {
  const policy = new Map<number, number>();
  const legalByUci = new Map<string, Move>((context?.legalMoves ?? legalMoves(board)).map((move) => [moveToUci(move), move]));
  for (const prior of lc0.legalPriors) {
    const move = legalByUci.get(prior.uci);
    if (move) policy.set(moveToActionId(move), prior.prior);
  }
  return { policy, wdl: lc0.wdl };
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
}

export class Lc0PuctSearcher {
  private readonly evaluator: Lc0SearchEvaluator;

  constructor(evaluator: Lc0SearchEvaluator | Lc0EvaluationProvider) {
    this.evaluator = evaluator instanceof Lc0SearchEvaluator ? evaluator : new Lc0SearchEvaluator(evaluator);
  }

  static async create(modelPath: string | Uint8Array | ArrayBuffer, options: Lc0OnnxEvaluatorOptions = {}): Promise<Lc0PuctSearcher> {
    return new Lc0PuctSearcher(await Lc0OnnxEvaluator.create(modelPath, options));
  }

  async search(input: Lc0EvaluatorInput, options: Lc0SearchOptions = {}): Promise<Lc0SearchResult> {
    const { board, historyFens } = currentBoardAndHistory(input);
    const result = await searchRoot(board, this.evaluator, {
      ...options,
      visits: options.visits ?? 32,
      temperature: options.temperature ?? 0,
      cpuctSchedule: options.cpuctSchedule ?? 'lc0-log',
      fpuStrategy: options.fpuStrategy ?? 'lc0-reduction',
      includePv: options.includePv ?? true,
      // History belongs to the LC0 input. Do not let generic SearchOptions
      // accidentally replace it and desynchronize the 112-plane encoder.
      historyFens,
    });
    const children = result.policy
      .map((entry) => ({ uci: moveToUci(entry.move), visits: entry.visits, prior: entry.prior, q: entry.q, probability: entry.probability }))
      .sort((a, b) => b.visits - a.visits || b.prior - a.prior);
    return {
      fen: boardToFen(board),
      move: result.move ? moveToUci(result.move) : undefined,
      visits: result.visits,
      value: result.value,
      children,
      search: result,
    };
  }
}
