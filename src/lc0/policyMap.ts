import { moveFromUci, moveToUci, type Move } from '../chess/moveCodec.ts';
import { squareIndex, squareName } from '../chess/board.ts';
import { LC0_POLICY_MAP, LC0_POLICY_MOVES, LC0_POLICY_SIZE } from './generatedPolicyMap.ts';

export { LC0_POLICY_MAP, LC0_POLICY_MOVES, LC0_POLICY_SIZE };

export const LC0_NO_TRANSFORM = 0;
export const LC0_FLIP_TRANSFORM = 1;
export const LC0_MIRROR_TRANSFORM = 2;
export const LC0_TRANSPOSE_TRANSFORM = 4;
export type Lc0BoardTransform = 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7;

export const LC0_POLICY_INDEX = new Map<string, number>(
  LC0_POLICY_MOVES.map((uci, i) => [uci, i]),
);

function transformSquareIndex(square: number, transform: number): number {
  if (square < 0 || square >= 64 || !Number.isInteger(square)) throw new Error(`Invalid square index: ${square}`);
  let file = square % 8;
  let rank = Math.floor(square / 8);
  // Mirrors LC0 src/neural/encoder.cc Transform(Square, int) exactly.
  if ((transform & (LC0_MIRROR_TRANSFORM | LC0_TRANSPOSE_TRANSFORM)) !== 0) rank = 7 - rank;
  if ((transform & (LC0_FLIP_TRANSFORM | LC0_TRANSPOSE_TRANSFORM)) !== 0) file = 7 - file;
  return rank * 8 + file;
}

export function transformSquareName(square: string, transform: number): string {
  return squareName(transformSquareIndex(squareIndex(square), transform));
}

function inverseTransform(transform: number): number {
  // Mirrors LC0 MoveFromNNIndex inverse-transform logic.
  if ((transform & LC0_TRANSPOSE_TRANSFORM) === 0) return transform;
  let inv = LC0_TRANSPOSE_TRANSFORM;
  if ((transform & LC0_FLIP_TRANSFORM) !== 0) inv |= LC0_MIRROR_TRANSFORM;
  if ((transform & LC0_MIRROR_TRANSFORM) !== 0) inv |= LC0_FLIP_TRANSFORM;
  return inv;
}

function standardCastlingUciToLc0Internal(uci: string): string {
  switch (uci) {
    case 'e1g1': return 'e1h1';
    case 'e1c1': return 'e1a1';
    case 'e8g8': return 'e8h8';
    case 'e8c8': return 'e8a8';
    default: return uci;
  }
}

function lc0InternalCastlingToStandardUci(uci: string): string {
  switch (uci) {
    case 'e1h1': return 'e1g1';
    case 'e1a1': return 'e1c1';
    case 'e8h8': return 'e8g8';
    case 'e8a8': return 'e8c8';
    default: return uci;
  }
}

function normalizeKnightPromotion(uci: string): string {
  // LC0's 1858 policy omits explicit knight-promotion suffixes. In LC0 C++,
  // MoveToNNIndex packs a knight promotion to the same slot as the corresponding
  // unsuffixed knight move.
  return uci.endsWith('n') ? uci.slice(0, 4) : uci;
}

export interface Lc0PolicyMoveOptions {
  /**
   * Standard UCI writes castling as e1g1/e1c1/e8g8/e8c8, while LC0's
   * internal Move representation encodes castling as king-takes-rook
   * e1h1/e1a1/e8h8/e8a8. Enable this only when the caller knows the move is
   * castling; raw UCI alone is ambiguous for pieces such as a queen on e1.
   */
  standardCastling?: boolean;
}

export function transformPolicyUci(uci: string, transform: number, options?: Lc0PolicyMoveOptions): string {
  const maybeCastling = options?.standardCastling ? standardCastlingUciToLc0Internal(uci) : uci;
  const normalized = normalizeKnightPromotion(maybeCastling);
  const from = transformSquareName(normalized.slice(0, 2), transform);
  const to = transformSquareName(normalized.slice(2, 4), transform);
  return `${from}${to}${normalized[4] ?? ''}`;
}

export function inverseTransformPolicyUci(policyUci: string, transform: number, options?: Lc0PolicyMoveOptions): string {
  const inv = inverseTransform(transform);
  const from = transformSquareName(policyUci.slice(0, 2), inv);
  const to = transformSquareName(policyUci.slice(2, 4), inv);
  const raw = `${from}${to}${policyUci[4] ?? ''}`;
  return options?.standardCastling ? lc0InternalCastlingToStandardUci(raw) : raw;
}

export function uciToLc0PolicyIndex(uci: string, transform: number = LC0_NO_TRANSFORM, options?: Lc0PolicyMoveOptions): number | undefined {
  return LC0_POLICY_INDEX.get(transformPolicyUci(uci, transform, options));
}

export function moveToLc0PolicyIndex(move: Move, transform: number = LC0_NO_TRANSFORM, options?: Lc0PolicyMoveOptions): number | undefined {
  return uciToLc0PolicyIndex(moveToUci(move), transform, options);
}

export function lc0PolicyIndexToUci(index: number, transform: number = LC0_NO_TRANSFORM, options?: Lc0PolicyMoveOptions): string {
  const policyUci = LC0_POLICY_MOVES[index];
  if (policyUci === undefined) throw new Error(`Invalid LC0 policy index: ${index}`);
  return inverseTransformPolicyUci(policyUci, transform, options);
}

export function lc0PolicyIndexToMove(index: number, transform: number = LC0_NO_TRANSFORM): Move {
  return moveFromUci(lc0PolicyIndexToUci(index, transform));
}
