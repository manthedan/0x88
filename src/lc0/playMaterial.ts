import type { BoardState, Color } from '../chess/board.ts';

const PIECE_VALUES: Record<string, number> = { p: 1, n: 3, b: 3, r: 5, q: 9 };
const PIECE_NAMES: Record<string, string> = { p: 'pawn', n: 'knight', b: 'bishop', r: 'rook', q: 'queen' };
const CAPTURED_GLYPHS: Record<Color, Record<string, string>> = {
  w: { p: '♙', n: '♘', b: '♗', r: '♖', q: '♕' },
  b: { p: '♟', n: '♞', b: '♝', r: '♜', q: '♛' },
};

function materialCounts(board: BoardState): Record<Color, Record<string, number>> {
  const counts: Record<Color, Record<string, number>> = { w: { p: 0, n: 0, b: 0, r: 0, q: 0 }, b: { p: 0, n: 0, b: 0, r: 0, q: 0 } };
  for (const piece of board.squares) {
    if (!piece) continue;
    const role = piece[1];
    if (role in PIECE_VALUES) counts[piece[0] as Color][role] += 1;
  }
  return counts;
}

/** Count captures only through the displayed ply, using the existing board history.
 * Enemy material lost on this side's turn is a capture; a pawn disappearing on
 * the enemy's own turn is a promotion. Custom-root odds aren't captures either.
 */
export function playMaterial(positions: readonly BoardState[], color: Color, ply = positions.length - 1): { html: string; label: string } {
  const enemy: Color = color === 'w' ? 'b' : 'w';
  const takenCounts: Record<string, number> = { p: 0, n: 0, b: 0, r: 0, q: 0 };
  let now = materialCounts(positions[0]);
  for (let i = 1; i <= ply; i++) {
    const before = now;
    now = materialCounts(positions[i]);
    if (positions[i - 1].turn === color) {
      for (const role of Object.keys(takenCounts)) takenCounts[role] += before[enemy][role] - now[enemy][role];
    }
  }
  let glyphs = '';
  const captured: string[] = [];
  for (const role of ['p', 'n', 'b', 'r', 'q']) {
    const taken = takenCounts[role];
    glyphs += CAPTURED_GLYPHS[enemy][role].repeat(taken);
    if (taken) captured.push(`${taken} ${PIECE_NAMES[role]}${taken > 1 ? 's' : ''}`);
  }
  const score = (side: Color) => Object.entries(now[side]).reduce((sum, [role, n]) => sum + PIECE_VALUES[role] * n, 0);
  const lead = score(color) - score(enemy);
  return {
    html: `${glyphs ? `<span class="taken">${glyphs}</span>` : ''}${lead > 0 ? `<span class="adv">+${lead}</span>` : ''}`,
    label: `${captured.length ? `Captured: ${captured.join(', ')}` : 'No captures'}${lead > 0 ? `; ahead by ${lead} material point${lead === 1 ? '' : 's'}` : ''}`,
  };
}
