import { type BoardState, boardToFen, parseFen } from '../chess/board.ts';
import { moveToUci } from '../chess/moveCodec.ts';
import { legalMoves, makeMove, pseudoLegalMoves } from '../chess/movegen.ts';

/** A UCI `position fen <fen> moves <moves...>` payload. */
export interface UciGameHistory {
  /** The game's start position (or the earliest position the move chain reaches back to). */
  fen: string;
  /** UCI moves from `fen` to the position to search. */
  moves: string[];
}

function samePosition(a: BoardState, b: BoardState): boolean {
  if (a.turn !== b.turn || a.castling !== b.castling || a.epSquare !== b.epSquare) return false;
  for (let i = 0; i < 64; i += 1) if (a.squares[i] !== b.squares[i]) return false;
  return true;
}

/** The UCI move that turns `before` into `after`, or null if no single move does. */
export function uciMoveBetween(before: BoardState, after: BoardState): string | null {
  if (before.turn === after.turn) return null;
  for (const move of pseudoLegalMoves(before)) {
    // Cheap filter first: the mover leaves its square and its piece (or the
    // promoted piece) lands on the target. Castling and en passant also pass.
    if (after.squares[move.from] !== null) continue;
    const landed = move.promotion ? `${before.turn}${move.promotion}` : before.squares[move.from];
    if (after.squares[move.to] !== landed) continue;
    if (samePosition(makeMove(before, move), after)) return moveToUci(move);
  }
  return null;
}

/**
 * Turn a game's position history (start position first, position to search
 * last) into a start FEN plus UCI moves, so a UCI engine sees repetitions and
 * the 50-move clock the same way Lc0 does when given the whole history.
 * Castling is encoded king-two-squares (e1g1) and promotions lowercase
 * (e7e8q), as standard UCI engines expect. If two neighbouring positions are
 * not linked by a single move, history before that break is dropped and the
 * later position becomes the start FEN.
 */
export function uciGameHistory(positions: readonly BoardState[]): UciGameHistory {
  if (!positions.length) throw new Error('uciGameHistory needs at least one position');
  const moves: string[] = [];
  let start = positions.length - 1;
  while (start > 0) {
    const move = uciMoveBetween(positions[start - 1], positions[start]);
    if (!move) break;
    moves.push(move);
    start -= 1;
  }
  return { fen: boardToFen(positions[start]), moves: moves.reverse() };
}

/**
 * The same position history re-rooted at the last capture or pawn move (where
 * the halfmove clock is 0). No repetition reaches back past such a move and
 * the root FEN carries the halfmove clock, so the engine sees the same
 * repetitions and 50-move count from a shorter `position` command.
 */
export function uciHistoryFromLastIrreversibleMove(fen: string, moves: readonly string[]): UciGameHistory {
  let board = parseFen(fen);
  let root = { fen, index: 0 };
  for (let i = 0; i < moves.length; i += 1) {
    const move = legalMoves(board).find((candidate) => moveToUci(candidate) === moves[i]);
    if (!move) throw new Error(`Illegal move ${moves[i]} in UCI history from ${fen}`);
    board = makeMove(board, move);
    if (board.halfmove === 0) root = { fen: boardToFen(board), index: i + 1 };
  }
  return { fen: root.fen, moves: moves.slice(root.index) };
}

/** Apply legal UCI `moves` to `fen` and return the resulting FEN. */
export function fenAfterUciMoves(fen: string, moves: readonly string[] = []): string {
  if (!moves.length) return fen;
  let board = parseFen(fen);
  for (const uci of moves) {
    const move = legalMoves(board).find((candidate) => moveToUci(candidate) === uci);
    if (!move) throw new Error(`Illegal move ${uci} in UCI history from ${fen}`);
    board = makeMove(board, move);
  }
  return boardToFen(board);
}
