import { type BoardState, boardToFen } from './board.ts';

export type AutomaticDrawReason = 'threefold' | 'fiftyMove' | 'insufficientMaterial';

function placementPieceAt(placement: string, file: number, rank: number): string | null {
  const rows = placement.split('/');
  const row = rows[7 - rank];
  if (row === undefined) return null;
  let f = 0;
  for (const ch of row) {
    if (ch >= '1' && ch <= '8') f += Number(ch);
    else if (f++ === file) return ch;
    if (f > file) return null;
  }
  return null;
}

// An en-passant square only distinguishes positions when a pawn of the side to
// move could capture onto it (FIDE 9.2.3; LC0 board.cc only records it then).
// movegen sets epSquare after every double push, so drop the inert ones here.
function capturableEpField(placement: string, turn: string, ep: string): string {
  if (ep === '-' || ep.length !== 2) return ep;
  const file = ep.charCodeAt(0) - 97;
  const rank = ep.charCodeAt(1) - 49;
  const capturerRank = turn === 'w' ? rank - 1 : rank + 1;
  const capturer = turn === 'w' ? 'P' : 'p';
  for (const df of [-1, 1]) {
    const f = file + df;
    if (f >= 0 && f < 8 && capturerRank >= 0 && capturerRank < 8 && placementPieceAt(placement, f, capturerRank) === capturer) return ep;
  }
  return '-';
}

export function repetitionKeyFromFen(fen: string): string {
  const fields = fen.trim().split(/\s+/).slice(0, 4);
  if (fields.length === 4) fields[3] = capturableEpField(fields[0], fields[1], fields[3]);
  return fields.join(' ');
}

/**
 * Key for the earliest position of a history. That position is the FEN root
 * LC0 is given (`position fen ... moves ...`), and LC0's SetFromFen keeps any
 * en-passant square as written, so it is compared as given rather than
 * normalised. Later positions only carry a capturable en-passant square.
 */
export function rootRepetitionKeyFromFen(fen: string): string {
  return fen.trim().split(/\s+/).slice(0, 4).join(' ');
}

// Published board states are immutable; movegen's legality make/undo is synchronous
// and never calls these helpers while the board is temporarily modified.
const boardRepetitionKeys = new WeakMap<BoardState, { normal: string; root: string }>();

function cachedBoardRepetitionKeys(board: BoardState): { normal: string; root: string } {
  let keys = boardRepetitionKeys.get(board);
  if (!keys) {
    const fen = boardToFen(board);
    keys = { normal: repetitionKeyFromFen(fen), root: rootRepetitionKeyFromFen(fen) };
    boardRepetitionKeys.set(board, keys);
  }
  return keys;
}

export function repetitionKey(board: BoardState): string {
  return cachedBoardRepetitionKeys(board).normal;
}

export function rootRepetitionKey(board: BoardState): string {
  return cachedBoardRepetitionKeys(board).root;
}

/**
 * Occurrences of `board` among itself and the earlier positions. The search and
 * encoder pass history newest first, so the last entry is the FEN root and is
 * keyed as given (see rootRepetitionKeyFromFen). Oldest-first callers are
 * unaffected: their last entry has the other side to move and never matches.
 */
export function repetitionCount(board: BoardState, historyFens: string[] = []): number {
  const key = repetitionKey(board);
  let count = 1;
  for (let i = 0; i < historyFens.length; i++) {
    const fen = historyFens[i];
    if ((i === historyFens.length - 1 ? rootRepetitionKeyFromFen(fen) : repetitionKeyFromFen(fen)) === key) count += 1;
  }
  return count;
}

export function insufficientMaterial(board: BoardState): boolean {
  const bishops: boolean[] = [];
  let whiteMinors = 0;
  let blackMinors = 0;
  let whiteKnights = 0;
  let blackKnights = 0;
  for (let sq = 0; sq < board.squares.length; sq++) {
    const piece = board.squares[sq];
    if (!piece) continue;
    const color = piece[0];
    const role = piece[1];
    if (role === 'p' || role === 'r' || role === 'q') return false;
    if (role === 'n') {
      if (color === 'w') {
        whiteMinors++;
        whiteKnights++;
      } else {
        blackMinors++;
        blackKnights++;
      }
    } else if (role === 'b') {
      if (color === 'w') whiteMinors++;
      else blackMinors++;
      bishops.push(((sq % 8) + Math.floor(sq / 8)) % 2 === 0);
    }
  }
  if (whiteMinors === 0 && blackMinors === 0) return true;
  if (whiteMinors === 1 && blackMinors === 0) return true;
  if (whiteMinors === 0 && blackMinors === 1) return true;
  if (whiteKnights === 0 && blackKnights === 0 && bishops.length >= 2 && bishops.every((color) => color === bishops[0])) return true;
  return false;
}

export function automaticDrawReason(board: BoardState, historyFens: string[] = []): AutomaticDrawReason | null {
  if (board.halfmove >= 100) return 'fiftyMove';
  if (repetitionCount(board, historyFens) >= 3) return 'threefold';
  if (insufficientMaterial(board)) return 'insufficientMaterial';
  return null;
}
