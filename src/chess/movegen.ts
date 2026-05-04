import { cloneBoard, opposite, type BoardState, type Color, type Piece } from './board.js';
import type { Move } from './moveCodec.js';

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
        if (r === startRank && !board.squares[idx(f, twoR)]) moves.push({ from, to: idx(f, twoR) });
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
    else if (role === 'k') KING.forEach(([df, dr]) => addStep(board, moves, from, df, dr));
  }
  return moves;
}

export function makeMove(board: BoardState, move: Move): BoardState {
  const next = cloneBoard(board);
  const piece = next.squares[move.from];
  if (!piece) throw new Error('No piece on source square');
  next.squares[move.from] = null;
  next.squares[move.to] = move.promotion ? `${board.turn}${move.promotion}` as Piece : piece;
  next.turn = opposite(board.turn);
  next.epSquare = null;
  next.fullmove += board.turn === 'b' ? 1 : 0;
  return next;
}
