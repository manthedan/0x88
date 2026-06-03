import type { DrawShape } from 'chessground/draw';
import type { Key } from 'chessground/types';

// Chessground draws analysis arrows from auto-shapes. These helpers turn engine
// output (UCI moves) into shapes so the board reflects what was analyzed.

function uciToShape(uci: string | undefined, brush: string): DrawShape | null {
  if (!uci || uci.length < 4) return null;
  const orig = uci.slice(0, 2) as Key;
  const dest = uci.slice(2, 4) as Key;
  return { orig, dest, brush };
}

/** A single green arrow for the chosen/best move (e.g. the policy pick). */
export function bestMoveShapes(uci?: string): DrawShape[] {
  const shape = uciToShape(uci, 'green');
  return shape ? [shape] : [];
}

/**
 * Search arrows: the best move in green plus the other MultiPV root moves in
 * blue, so the board shows the engine's top candidates from the current
 * position. PV continuations stay in the side panel (they start from future
 * positions, so drawing them on this board would be misleading).
 */
export function searchShapes(bestMove?: string, multiPv?: string[][]): DrawShape[] {
  const shapes: DrawShape[] = [];
  const best = uciToShape(bestMove, 'green');
  if (best) shapes.push(best);
  for (const line of (multiPv ?? []).slice(1)) {
    const alt = uciToShape(line[0], 'blue');
    if (alt && !(best && alt.orig === best.orig && alt.dest === best.dest)) shapes.push(alt);
  }
  return shapes;
}
