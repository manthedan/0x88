import { boardToFen, parseFen, START_FEN, type BoardState } from '../chess/board.ts';
import { legalMoves, makeMove } from '../chess/movegen.ts';
import { moveToUci } from '../chess/moveCodec.ts';

export interface Lc0MoveHistoryFixture {
  id: string;
  description?: string;
  startFen?: string;
  moves: string[];
  finalFen?: string;
}

export function buildBoardHistoryFromMoves(moves: readonly string[], startFen = START_FEN): BoardState[] {
  let board = parseFen(startFen);
  const history = [board];
  for (const uci of moves) {
    const move = legalMoves(board).find((candidate) => moveToUci(candidate) === uci);
    if (!move) throw new Error(`Illegal fixture move ${uci} from ${boardToFen(board)}`);
    board = makeMove(board, move);
    history.push(board);
  }
  return history;
}

export function finalFenFromMoveHistory(moves: readonly string[], startFen = START_FEN): string {
  const history = buildBoardHistoryFromMoves(moves, startFen);
  return boardToFen(history[history.length - 1]);
}
