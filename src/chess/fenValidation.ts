import { type BoardState, opposite, parseFen } from './board.ts';
import { inCheck } from './movegen.ts';

/** Cheap position invariants required before sending a user-supplied FEN to an engine. */
export function validateFenPosition(board: BoardState): void {
  for (const color of ['w', 'b'] as const) {
    const kings = board.squares.filter((piece) => piece === `${color}k`).length;
    if (kings !== 1) throw new Error(`Invalid FEN: expected exactly one ${color === 'w' ? 'white' : 'black'} king (found ${kings})`);
  }
  for (let square = 0; square < 64; square++) {
    if ((square < 8 || square >= 56) && board.squares[square]?.[1] === 'p') {
      throw new Error('Invalid FEN: pawns cannot be on rank 1 or 8');
    }
  }
  // The player who just moved must have left their own king safe. The side to
  // move may be in check (including checkmate); adjacent kings fail this too.
  const previousMover = opposite(board.turn);
  if (inCheck(board, previousMover)) {
    throw new Error(`Invalid FEN: ${previousMover === 'w' ? 'white' : 'black'} king is in check, but that side is not to move`);
  }
}

/** Keep parseFen syntax-only for encoders/fixtures; validate at user-facing roots. */
export function parseValidatedFen(fen: string): BoardState {
  const board = parseFen(fen);
  validateFenPosition(board);
  return board;
}
