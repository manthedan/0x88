import { squareIndex, squareName, type PieceRole } from './board.js';

export interface Move {
  from: number;
  to: number;
  promotion?: Exclude<PieceRole, 'p' | 'k'>;
}

export function moveToUci(move: Move): string {
  return `${squareName(move.from)}${squareName(move.to)}${move.promotion ?? ''}`;
}

export function moveFromUci(uci: string): Move {
  if (!/^[a-h][1-8][a-h][1-8][nbrq]?$/.test(uci)) throw new Error(`Invalid UCI move: ${uci}`);
  return {
    from: squareIndex(uci.slice(0, 2)),
    to: squareIndex(uci.slice(2, 4)),
    promotion: uci[4] as Move['promotion'] | undefined,
  };
}

// Stable AlphaZero-like action id for source/destination/promotion. It is deliberately
// simple for v1; training/export/inference must all use this exact mapping.
export function moveToActionId(move: Move): number {
  const promo = move.promotion ? { n: 1, b: 2, r: 3, q: 4 }[move.promotion] : 0;
  return (move.from * 64 + move.to) * 5 + promo;
}

export const ACTION_SPACE = 64 * 64 * 5;
