import type { BoardState } from '../chess/board.js';
import { makeMove, pseudoLegalMoves } from '../chess/movegen.js';
import { moveToActionId, type Move } from '../chess/moveCodec.js';
import type { Evaluator } from '../nn/evaluator.js';

export interface SearchResult { move: Move | null; visits: number; value: number; }

export async function chooseMove(board: BoardState, evaluator: Evaluator): Promise<SearchResult> {
  // Minimum shared substrate: one neural-guided root selection. Full tree reuse,
  // virtual loss, FPU, batching, and time management belong to later research lanes.
  const legal = pseudoLegalMoves(board);
  if (!legal.length) return { move: null, visits: 0, value: 0 };
  const evaln = await evaluator.evaluate(board);
  let best = legal[0];
  let bestPrior = -Infinity;
  for (const move of legal) {
    const prior = evaln.policy.get(moveToActionId(move)) ?? 0;
    if (prior > bestPrior) { best = move; bestPrior = prior; }
  }
  // Exercise makeMove so search consumers share the same transition code.
  makeMove(board, best);
  return { move: best, visits: 1, value: evaln.wdl[0] - evaln.wdl[2] };
}
