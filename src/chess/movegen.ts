import { cloneBoard, opposite, squareName, type BoardState, type Color, type Piece } from './board.ts';
import type { Move } from './moveCodec.ts';

const KNIGHT = [[1, 2], [2, 1], [-1, 2], [-2, 1], [1, -2], [2, -1], [-1, -2], [-2, -1]];
const KING = [[1, 1], [1, 0], [1, -1], [0, 1], [0, -1], [-1, 1], [-1, 0], [-1, -1]];
const BISHOP = [[1, 1], [1, -1], [-1, 1], [-1, -1]];
const ROOK = [[1, 0], [-1, 0], [0, 1], [0, -1]];

const fileOf = (sq: number) => sq % 8;
const rankOf = (sq: number) => Math.floor(sq / 8);
const on = (f: number, r: number) => f >= 0 && f < 8 && r >= 0 && r < 8;
const idx = (f: number, r: number) => f + r * 8;
const colorOf = (piece: Piece | null): Color | null => piece?.[0] as Color | undefined ?? null;

function addStep(board: BoardState, moves: Move[], from: number, df: number, dr: number): void {
  const f = fileOf(from) + df, r = rankOf(from) + dr;
  if (!on(f, r)) return;
  const to = idx(f, r);
  if (colorOf(board.squares[to]) !== board.turn) moves.push({ from, to });
}

function addSlides(board: BoardState, moves: Move[], from: number, dirs: number[][]): void {
  for (const [df, dr] of dirs) {
    let f = fileOf(from) + df, r = rankOf(from) + dr;
    while (on(f, r)) {
      const to = idx(f, r);
      const occ = board.squares[to];
      if (!occ) moves.push({ from, to });
      else {
        if (colorOf(occ) !== board.turn) moves.push({ from, to });
        break;
      }
      f += df; r += dr;
    }
  }
}

function canCastle(board: BoardState, side: 'K' | 'Q' | 'k' | 'q'): boolean {
  if (!board.castling.includes(side)) return false;
  const white = side === 'K' || side === 'Q';
  const color: Color = white ? 'w' : 'b';
  const rank = white ? 0 : 7;
  const kingFrom = idx(4, rank);
  const rookFrom = side === 'K' || side === 'k' ? idx(7, rank) : idx(0, rank);
  if (board.turn !== color || board.squares[kingFrom] !== `${color}k` || board.squares[rookFrom] !== `${color}r`) return false;
  const between = side === 'K' || side === 'k' ? [idx(5, rank), idx(6, rank)] : [idx(1, rank), idx(2, rank), idx(3, rank)];
  if (between.some((sq) => board.squares[sq])) return false;
  const pass = side === 'K' || side === 'k' ? [idx(4, rank), idx(5, rank), idx(6, rank)] : [idx(4, rank), idx(3, rank), idx(2, rank)];
  return pass.every((sq) => !isSquareAttacked(board, sq, opposite(color)));
}

export function pseudoLegalMoves(board: BoardState): Move[] {
  const moves: Move[] = [];
  for (let from = 0; from < 64; from++) {
    const piece = board.squares[from];
    if (!piece || colorOf(piece) !== board.turn) continue;
    const role = piece[1];
    if (role === 'p') {
      const dir = board.turn === 'w' ? 1 : -1;
      const startRank = board.turn === 'w' ? 1 : 6;
      const promoteRank = board.turn === 'w' ? 7 : 0;
      const f = fileOf(from), r = rankOf(from);
      const oneR = r + dir;
      if (on(f, oneR) && !board.squares[idx(f, oneR)]) {
        const to = idx(f, oneR);
        if (oneR === promoteRank) for (const promotion of ['q', 'r', 'b', 'n'] as const) moves.push({ from, to, promotion });
        else moves.push({ from, to });
        const twoR = r + 2 * dir;
        if (r === startRank && on(f, twoR) && !board.squares[idx(f, twoR)]) moves.push({ from, to: idx(f, twoR) });
      }
      for (const df of [-1, 1]) {
        const cf = f + df;
        if (!on(cf, oneR)) continue;
        const to = idx(cf, oneR);
        if (colorOf(board.squares[to]) === opposite(board.turn) || to === board.epSquare) {
          if (oneR === promoteRank) for (const promotion of ['q', 'r', 'b', 'n'] as const) moves.push({ from, to, promotion });
          else moves.push({ from, to });
        }
      }
    } else if (role === 'n') KNIGHT.forEach(([df, dr]) => addStep(board, moves, from, df, dr));
    else if (role === 'b') addSlides(board, moves, from, BISHOP);
    else if (role === 'r') addSlides(board, moves, from, ROOK);
    else if (role === 'q') addSlides(board, moves, from, [...BISHOP, ...ROOK]);
    else if (role === 'k') {
      KING.forEach(([df, dr]) => addStep(board, moves, from, df, dr));
      if (canCastle(board, board.turn === 'w' ? 'K' : 'k')) moves.push({ from, to: idx(6, board.turn === 'w' ? 0 : 7) });
      if (canCastle(board, board.turn === 'w' ? 'Q' : 'q')) moves.push({ from, to: idx(2, board.turn === 'w' ? 0 : 7) });
    }
  }
  return moves;
}

export function isSquareAttacked(board: BoardState, square: number, by: Color): boolean {
  const f = fileOf(square), r = rankOf(square);
  const pawnFromRank = r + (by === 'w' ? -1 : 1);
  for (const df of [-1, 1]) {
    const pf = f + df;
    if (on(pf, pawnFromRank) && board.squares[idx(pf, pawnFromRank)] === `${by}p`) return true;
  }
  for (const [df, dr] of KNIGHT) {
    const nf = f + df, nr = r + dr;
    if (on(nf, nr) && board.squares[idx(nf, nr)] === `${by}n`) return true;
  }
  for (const [df, dr] of BISHOP) {
    let sf = f + df, sr = r + dr;
    while (on(sf, sr)) {
      const piece = board.squares[idx(sf, sr)];
      if (piece) { if (piece[0] === by && (piece[1] === 'b' || piece[1] === 'q')) return true; break; }
      sf += df; sr += dr;
    }
  }
  for (const [df, dr] of ROOK) {
    let sf = f + df, sr = r + dr;
    while (on(sf, sr)) {
      const piece = board.squares[idx(sf, sr)];
      if (piece) { if (piece[0] === by && (piece[1] === 'r' || piece[1] === 'q')) return true; break; }
      sf += df; sr += dr;
    }
  }
  for (const [df, dr] of KING) {
    const kf = f + df, kr = r + dr;
    if (on(kf, kr) && board.squares[idx(kf, kr)] === `${by}k`) return true;
  }
  return false;
}

export function kingSquare(board: BoardState, color: Color): number | null {
  const target = `${color}k`;
  const sq = board.squares.findIndex((piece) => piece === target);
  return sq >= 0 ? sq : null;
}

export function inCheck(board: BoardState, color: Color = board.turn): boolean {
  const king = kingSquare(board, color);
  return king === null ? true : isSquareAttacked(board, king, opposite(color));
}

export function legalMoves(board: BoardState): Move[] {
  return pseudoLegalMoves(board).filter((move) => !inCheck(makeMove(board, move), board.turn));
}

function castlingWithout(castling: string, removed: string): string {
  const next = castling.split('').filter((ch) => !removed.includes(ch)).join('');
  return next || '-';
}

export function makeMove(board: BoardState, move: Move): BoardState {
  const next = cloneBoard(board);
  const piece = next.squares[move.from];
  if (!piece) throw new Error('No piece on source square');
  const captured = next.squares[move.to];
  const fromName = squareName(move.from), toName = squareName(move.to);
  const isPawn = piece[1] === 'p';
  const isCastle = piece[1] === 'k' && Math.abs(fileOf(move.to) - fileOf(move.from)) === 2;
  const isEp = isPawn && move.to === board.epSquare && !captured && fileOf(move.from) !== fileOf(move.to);

  next.squares[move.from] = null;
  if (isEp) next.squares[idx(fileOf(move.to), rankOf(move.from))] = null;
  next.squares[move.to] = move.promotion ? `${board.turn}${move.promotion}` as Piece : piece;
  if (isCastle) {
    const rank = board.turn === 'w' ? 0 : 7;
    if (fileOf(move.to) === 6) {
      next.squares[idx(5, rank)] = next.squares[idx(7, rank)];
      next.squares[idx(7, rank)] = null;
    } else {
      next.squares[idx(3, rank)] = next.squares[idx(0, rank)];
      next.squares[idx(0, rank)] = null;
    }
  }

  let castling = board.castling === '-' ? '' : board.castling;
  if (piece === 'wk') castling = castlingWithout(castling, 'KQ');
  if (piece === 'bk') castling = castlingWithout(castling, 'kq');
  if (fromName === 'a1' || toName === 'a1') castling = castlingWithout(castling, 'Q');
  if (fromName === 'h1' || toName === 'h1') castling = castlingWithout(castling, 'K');
  if (fromName === 'a8' || toName === 'a8') castling = castlingWithout(castling, 'q');
  if (fromName === 'h8' || toName === 'h8') castling = castlingWithout(castling, 'k');
  next.castling = castling || '-';

  next.turn = opposite(board.turn);
  next.epSquare = isPawn && Math.abs(rankOf(move.to) - rankOf(move.from)) === 2 ? idx(fileOf(move.from), (rankOf(move.from) + rankOf(move.to)) / 2) : null;
  next.halfmove = isPawn || captured || isEp ? 0 : board.halfmove + 1;
  next.fullmove += board.turn === 'b' ? 1 : 0;
  return next;
}
