import type { BoardState } from '../chess/board.ts';
import { legalMoves } from '../chess/movegen.ts';
import { moveToActionId, type Move } from '../chess/moveCodec.ts';

export interface Evaluation {
  policy: Map<number, number>;
  wdl: [win: number, draw: number, loss: number];
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
}

export interface Evaluator {
  evaluate(board: BoardState, context?: EvaluationContext): Promise<Evaluation> | Evaluation;
  evaluateBatch?(boards: BoardState[], contexts?: EvaluationContext[]): Promise<Evaluation[]> | Evaluation[];
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
