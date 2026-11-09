import { ARROW_GEOMETRY, arrowHeadPath, arrowLayout } from './arrowGeometry.ts';

// Build-time patch for chessground's dist/svg.js (applied by the Vite plugin in vite.config.ts).
// Chessground has no setting for arrowhead size: its marker scales with the line width and its
// tip position is tuned for lichess's width. This swaps in arrowGeometry.ts so the tail, tip and
// head are set in board squares. Every replacement must match exactly once, so a chessground
// update that changes this code fails the build instead of silently dropping the patch.

const REPLACEMENTS: [string, string][] = [
  [
    'const m = arrowMargin(shorten && !current), dx = to[0] - from[0], dy = to[1] - from[1], angle = Math.atan2(dy, dx), xo = Math.cos(angle) * m, yo = Math.sin(angle) * m;',
    'const cgArrow = __cgArrowLayout(from, to, !!(shorten && !current), __CG_ARROW_GEOMETRY);',
  ],
  ["'stroke-linecap': 'round',", "'stroke-linecap': 'butt',"],
  [
    'x1: from[0],\n            y1: from[1],\n            x2: to[0] - xo,\n            y2: to[1] - yo,',
    'x1: cgArrow.x1,\n            y1: cgArrow.y1,\n            x2: cgArrow.x2,\n            y2: cgArrow.y2,',
  ],
  [
    `markerWidth: 4,
        markerHeight: 4,
        refX: brush.key.startsWith('hilite') ? 1.86 : 2.05,
        refY: 2,
    });
    marker.appendChild(setAttributes(createElement('path'), {
        d: 'M0,0 V4 L3,2 Z',
        fill: brush.color,
    }));`,
    `markerUnits: 'userSpaceOnUse',
        markerWidth: __CG_ARROW_GEOMETRY.headLength,
        markerHeight: __CG_ARROW_GEOMETRY.headWidth,
        refX: __CG_ARROW_GEOMETRY.overlap,
        refY: 0,
    });
    const hiliteHead = brush.key.startsWith('hilite');
    marker.appendChild(setAttributes(createElement('path'), {
        d: __CG_ARROW_HEAD,
        fill: brush.color,
        stroke: hiliteHead ? brush.color : 'none',
        'stroke-width': hiliteHead ? 0.04 : 0,
        'stroke-linejoin': 'round',
    }));`,
  ],
];

export const CHESSGROUND_SVG_PATCH_MARKER = '/* 0x88 arrow geometry */';

export function patchChessgroundSvg(code: string): string {
  let patched = code;
  for (const [from, to] of REPLACEMENTS) {
    const count = patched.split(from).length - 1;
    if (count !== 1) {
      throw new Error(`chessground arrow patch: expected 1 match, found ${count}, for: ${from.split('\n')[0]}`);
    }
    patched = patched.replace(from, () => to);
  }
  const header = [
    CHESSGROUND_SVG_PATCH_MARKER,
    `const __CG_ARROW_GEOMETRY = ${JSON.stringify(ARROW_GEOMETRY)};`,
    `const __CG_ARROW_HEAD = ${JSON.stringify(arrowHeadPath(ARROW_GEOMETRY))};`,
    `const __cgArrowLayout = (${arrowLayout.toString()});`,
  ].join('\n');
  return `${header}\n${patched}`;
}
