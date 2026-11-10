// Board-arrow geometry, in board squares (1 = one square). The owner chose (2026-11-08) a
// chess.com-style tail that starts near the edge of the origin square, so the moving piece stays
// visible, and a fixed head (0.40 long, 0.55 wide) whose tip sits on the target square's centre.
// The head is sized in squares, not in line widths, so its base stays 0.10 inside the target square
// whatever the arrow width. Chessground's own head scales with the width (3 x 4 line widths), which
// put its base almost exactly on the square boundary at width 11.
//
// vite.chessground-arrows.ts injects arrowLayout() into chessground, and the Home preview draws
// its arrows with it, so every board arrow in the app (and arrows users draw) shares this shape.

export interface ArrowGeometry {
  /** Distance from the origin square's centre to where the shaft starts. */
  tail: number;
  /** Arrowhead length, from its base to its tip. */
  headLength: number;
  /** Arrowhead width at its base. */
  headWidth: number;
  /** How far the shaft runs under the head, so no seam shows between them. */
  overlap: number;
  /** How far the tip stops short of the centre when several arrows end on one square. */
  shortenBy: number;
}

export const ARROW_GEOMETRY: ArrowGeometry = {
  tail: 0.42,
  headLength: 0.4,
  headWidth: 0.55,
  overlap: 0.04,
  shortenBy: 10 / 64,
};

export interface ArrowLayout {
  /** Shaft start and end. The end lies `overlap` inside the head. */
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  tip: [number, number];
  /** Unit direction from origin to target. */
  ux: number;
  uy: number;
}

/**
 * Lays out one arrow between two square centres. This function is serialised into chessground's
 * svg.js at build time, so it must stay self-contained: no imports and no module-level references.
 */
export function arrowLayout(from: [number, number], to: [number, number], shorten: boolean, g: ArrowGeometry): ArrowLayout {
  const dx = to[0] - from[0], dy = to[1] - from[1];
  const length = Math.hypot(dx, dy) || 1;
  const ux = dx / length, uy = dy / length;
  const tipAt = length - (shorten ? g.shortenBy : 0);
  const shaftEnd = tipAt - g.headLength + g.overlap;
  // Keep a visible shaft on one-square moves to a shortened (shared) target.
  const tailAt = Math.max(0, Math.min(g.tail, shaftEnd - g.overlap - 0.06));
  return {
    x1: from[0] + ux * tailAt,
    y1: from[1] + uy * tailAt,
    x2: from[0] + ux * shaftEnd,
    y2: from[1] + uy * shaftEnd,
    tip: [from[0] + ux * tipAt, from[1] + uy * tipAt],
    ux,
    uy,
  };
}

/** Arrowhead path with its base on x = 0 and its tip at x = headLength, centred on y = 0. */
export function arrowHeadPath(g: ArrowGeometry): string {
  const half = g.headWidth / 2;
  return `M0,${-half} L${g.headLength},0 L0,${half} Z`;
}
