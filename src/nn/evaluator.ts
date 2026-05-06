import type { BoardState } from '../chess/board.ts';
import { legalMoves } from '../chess/movegen.ts';
import { moveToActionId } from '../chess/moveCodec.ts';

export interface Evaluation {
  policy: Map<number, number>;
  wdl: [win: number, draw: number, loss: number];
}

export interface EvaluationContext {
  /** Previous position FENs, newest first. */
  historyFens?: string[];
}

export interface Evaluator {
  evaluate(board: BoardState, context?: EvaluationContext): Promise<Evaluation> | Evaluation;
}

export class UniformEvaluator implements Evaluator {
  evaluate(board: BoardState): Evaluation {
    const moves = legalMoves(board);
    const policy = new Map<number, number>();
    const p = moves.length ? 1 / moves.length : 0;
    for (const move of moves) policy.set(moveToActionId(move), p);
    return { policy, wdl: [0.33, 0.34, 0.33] };
  }
}
