import type { Color } from '../chess/board.ts';

/** Use the live game's samples, never the chart currently selected for review.
 * Samples start at ply zero after the opening, which may leave Black to move.
 * Background CPU games already have samples when they become the live focus.
 */
export function arenaSideTimeMs(samples: readonly { ply: number; moveMs: number }[], firstTurn: Color): Record<Color, number> {
  const totals = { w: 0, b: 0 };
  const other = firstTurn === 'w' ? 'b' : 'w';
  for (const sample of samples) totals[sample.ply % 2 === 0 ? firstTurn : other] += sample.moveMs;
  return totals;
}
