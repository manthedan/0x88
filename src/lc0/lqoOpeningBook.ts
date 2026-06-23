import { type BoardState } from '../chess/board.ts';
import { legalMoves } from '../chess/movegen.ts';
import { moveToUci } from '../chess/moveCodec.ts';
import { LQO_BLACK_OPENING_BOOK, LQO_OPENING_BOOK_MAX_PLIES, LQO_WHITE_OPENING_BOOK } from './lqoOpeningBookData.ts';

export interface LqoBookMove {
  uci: string;
  weight: number;
}

function parseBook(rows: readonly (readonly [string, string])[]): Map<string, LqoBookMove[]> {
  return new Map(rows.map(([key, packed]) => [
    key,
    packed ? packed.split(',').map((entry) => {
      const [uci, weight] = entry.split(':');
      return { uci, weight: Number.parseInt(weight, 36) };
    }) : [],
  ]));
}

const lqoBlackBook = new Map<string, LqoBookMove[]>(
  parseBook(LQO_BLACK_OPENING_BOOK),
);
const lqoWhiteBook = new Map<string, LqoBookMove[]>(
  parseBook(LQO_WHITE_OPENING_BOOK),
);

function sequenceKey(ply: number, historyUcis: readonly string[]): string | null {
  if (historyUcis.length < ply) return null;
  return historyUcis.slice(0, ply).join(' ');
}

export function lqoBlackBookMoves(board: BoardState, ply: number, historyUcis: readonly string[] = []): LqoBookMove[] {
  if (board.turn !== 'b' || ply >= LQO_OPENING_BOOK_MAX_PLIES) return [];
  const key = sequenceKey(ply, historyUcis);
  if (key === null) return [];
  const legal = new Set(legalMoves(board).map(moveToUci));
  return (lqoBlackBook.get(key) ?? []).filter((move) => legal.has(move.uci));
}

export function lqoBlackBookMove(board: BoardState, ply: number, historyUcis: readonly string[] = [], random = Math.random): string | null {
  const moves = lqoBlackBookMoves(board, ply, historyUcis);
  const total = moves.reduce((sum, move) => sum + Math.max(0, move.weight), 0);
  if (total <= 0) return null;
  let pick = Math.max(0, Math.min(1, random())) * total;
  for (const move of moves) {
    pick -= Math.max(0, move.weight);
    if (pick <= 0) return move.uci;
  }
  return moves.at(-1)?.uci ?? null;
}

export function lqoWhiteBookMoves(board: BoardState, ply: number, historyUcis: readonly string[] = []): LqoBookMove[] {
  if (board.turn !== 'w' || ply >= LQO_OPENING_BOOK_MAX_PLIES) return [];
  const key = sequenceKey(ply, historyUcis);
  if (key === null) return [];
  const legal = new Set(legalMoves(board).map(moveToUci));
  return (lqoWhiteBook.get(key) ?? []).filter((move) => legal.has(move.uci));
}

export function lqoWhiteFirstPolicyBookMove(
  board: BoardState,
  ply: number,
  historyUcis: readonly string[],
  legalPriors: readonly { uci: string; prior: number }[],
  random = Math.random,
  topN = 8,
): string | null {
  if (board.turn !== 'w' || ply !== 0) return null;
  const legal = new Set(legalMoves(board).map(moveToUci));
  const bookMoves = new Set(lqoWhiteBookMoves(board, ply, historyUcis).map((move) => move.uci));
  const moves = legalPriors
    .filter((move) => legal.has(move.uci))
    .sort((a, b) => b.prior - a.prior)
    .slice(0, Math.max(1, topN))
    .filter((move) => bookMoves.has(move.uci))
    .map((move) => move.uci);
  if (!moves.length) return null;
  const index = Math.min(moves.length - 1, Math.floor(Math.max(0, Math.min(0.999999999, random())) * moves.length));
  return moves[index];
}

export function lqoWhiteBookMove(board: BoardState, ply: number, historyUcis: readonly string[] = [], random = Math.random): string | null {
  const moves = lqoWhiteBookMoves(board, ply, historyUcis);
  const total = moves.reduce((sum, move) => sum + Math.max(0, move.weight), 0);
  if (total <= 0) return null;
  let pick = Math.max(0, Math.min(1, random())) * total;
  for (const move of moves) {
    pick -= Math.max(0, move.weight);
    if (pick <= 0) return move.uci;
  }
  return moves.at(-1)?.uci ?? null;
}
