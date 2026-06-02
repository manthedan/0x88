import { boardToFen, cloneBoard, parseFen, squareName, type BoardState, type Piece } from './board.ts';
import { type Move } from './moveCodec.ts';

export const STM_WHITE_RANKFLIP_V1 = 'stm_white_rankflip_v1';
// Rank-flip + color-swap normalization.  Black-to-move boards are transformed
// so the original side to move is represented as white.  Side-to-move-relative
// WDL/Q/AV targets are therefore invariant under this transform; do not swap
// WDL win/loss or negate Q/AV merely because `flipped` is true.
export type BoardNormalization = typeof STM_WHITE_RANKFLIP_V1;

export interface NormalizedPosition {
  board: BoardState;
  historyFens: string[];
  legalMoves?: Move[];
  flipped: boolean;
}

export function isStmWhiteRankflip(value: unknown): value is BoardNormalization {
  return value === STM_WHITE_RANKFLIP_V1;
}

export function rankFlipSquare(square: number): number {
  if (!Number.isInteger(square) || square < 0 || square >= 64) throw new Error(`square out of range: ${square}`);
  const file = square % 8;
  const rank = Math.floor(square / 8);
  return file + (7 - rank) * 8;
}

function swapPieceColor(piece: Piece | null): Piece | null {
  if (!piece) return null;
  return `${piece[0] === 'w' ? 'b' : 'w'}${piece[1]}` as Piece;
}

function normalizeCastling(castling: string): string {
  const src = castling && castling !== '-' ? castling : '';
  let out = '';
  if (src.includes('k')) out += 'K';
  if (src.includes('q')) out += 'Q';
  if (src.includes('K')) out += 'k';
  if (src.includes('Q')) out += 'q';
  return out || '-';
}

export function rankFlipColorSwapBoard(board: BoardState): BoardState {
  const squares: BoardState['squares'] = Array(64).fill(null);
  for (let sq = 0; sq < 64; sq++) squares[rankFlipSquare(sq)] = swapPieceColor(board.squares[sq]);
  return {
    squares,
    turn: board.turn === 'w' ? 'b' : 'w',
    castling: normalizeCastling(board.castling),
    epSquare: board.epSquare === null ? null : rankFlipSquare(board.epSquare),
    halfmove: board.halfmove,
    fullmove: board.fullmove,
  };
}

export function normalizeBoardForStmWhite(board: BoardState): { board: BoardState; flipped: boolean } {
  if (board.turn === 'w') return { board: cloneBoard(board), flipped: false };
  return { board: rankFlipColorSwapBoard(board), flipped: true };
}

export function normalizeFenForStmWhite(fen: string): { fen: string; flipped: boolean } {
  const { board, flipped } = normalizeBoardForStmWhite(parseFen(fen));
  return { fen: boardToFen(board), flipped };
}

export function normalizeHistoryForStmWhite(historyFens: string[] = [], flipped: boolean): string[] {
  if (!flipped) return [...historyFens];
  return historyFens.map((fen) => boardToFen(rankFlipColorSwapBoard(parseFen(fen))));
}

export function rankFlipMove(move: Move): Move {
  return {
    from: rankFlipSquare(move.from),
    to: rankFlipSquare(move.to),
    ...(move.promotion ? { promotion: move.promotion } : {}),
  };
}

export const unrankFlipMove = rankFlipMove;

export function normalizeMovesForStmWhite(moves: Move[] | undefined, flipped: boolean): Move[] | undefined {
  if (!moves) return undefined;
  return flipped ? moves.map(rankFlipMove) : moves.map((m) => ({ ...m }));
}

export function normalizePositionForStmWhite(board: BoardState, historyFens: string[] = [], legal: Move[] | undefined = undefined): NormalizedPosition {
  const normalized = normalizeBoardForStmWhite(board);
  return {
    board: normalized.board,
    historyFens: normalizeHistoryForStmWhite(historyFens, normalized.flipped),
    legalMoves: normalizeMovesForStmWhite(legal, normalized.flipped),
    flipped: normalized.flipped,
  };
}

export function normalizedMoveToOriginal(move: Move, flipped: boolean): Move {
  return flipped ? unrankFlipMove(move) : move;
}

export function debugSquareTransform(square: number): string {
  return `${squareName(square)}->${squareName(rankFlipSquare(square))}`;
}
