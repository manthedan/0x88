import type { BoardState, Color, PieceRole } from '../chess/board.js';

const ROLES: PieceRole[] = ['p', 'n', 'b', 'r', 'q', 'k'];
const COLORS: Color[] = ['w', 'b'];
export const FEATURE_PLANES = 12 + 1;

export function encodeFeatures(board: BoardState): Float32Array {
  const out = new Float32Array(FEATURE_PLANES * 64);
  for (let sq = 0; sq < 64; sq++) {
    const piece = board.squares[sq];
    if (!piece) continue;
    const colorIndex = COLORS.indexOf(piece[0] as Color);
    const roleIndex = ROLES.indexOf(piece[1] as PieceRole);
    out[(colorIndex * 6 + roleIndex) * 64 + sq] = 1;
  }
  if (board.turn === 'w') out.fill(1, 12 * 64, 13 * 64);
  return out;
}
