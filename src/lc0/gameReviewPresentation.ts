import { parseFen } from '../chess/board.ts';
import { htmlEscape } from './browserDom.ts';
import { lineChartSvg } from './charts.ts';
import { type GameReview, type ReviewedMove } from './gameReview.ts';

export const REVIEW_CLASS_LABEL: Record<ReviewedMove['class'], string> = {
  best: 'Best',
  good: 'Good',
  inaccuracy: 'Inaccuracy',
  mistake: 'Mistake',
  blunder: 'Blunder',
  forced: 'Forced',
};
export const REVIEW_CLASS_GLYPH: Record<ReviewedMove['class'], string> = {
  best: '✓',
  good: '·',
  inaccuracy: '?!',
  mistake: '?',
  blunder: '??',
  forced: '□',
};

/** Move numbering is based on the starting FEN, not the relative review ply. */
export function reviewMoveLabel(startFen: string, move: ReviewedMove): string {
  const start = parseFen(startFen);
  const absolutePly = (start.fullmove - 1) * 2 + (start.turn === 'b' ? 1 : 0) + move.ply - 1;
  return `${Math.floor(absolutePly / 2) + 1}${move.side === 'w' ? '.' : '…'} ${move.san}`;
}

/** Same 36/8 padding and 360-unit viewBox as lineChartSvg. */
export function reviewPlyAtRatio(ratio: number, moveCount: number): number {
  return Math.max(0, Math.min(moveCount, Math.round(((ratio * 360 - 36) / 316) * moveCount)));
}

export function reviewChartSvg(review: GameReview, currentPly: number): string {
  if (!review.moves.length) return '';
  const values = [review.moves[0].winBefore, ...review.moves.map((move) => move.winAfter)];
  const x = (ply: number) => 36 + (ply / review.moves.length) * 316;
  const y = (value: number) => 6 + (1 - value) * 98;
  const svg = lineChartSvg([{ label: 'White advantage', color: 'var(--ink)', points: values.map((value, ply) => ({ x: ply, y: value })) }], {
    yMin: 0,
    yMax: 1,
    midline: 0.5,
    formatY: (v) => `${Math.round(v * 100)}%`,
    height: 118,
  });
  const errors = review.moves
    .filter((move) => ['inaccuracy', 'mistake', 'blunder'].includes(move.class))
    .map(
      (move) =>
        `<circle cx="${x(move.ply)}" cy="${y(move.winAfter)}" r="2.5" class="review-point ${move.class}" fill="var(--warn)"><title>${htmlEscape(`${move.san}: ${REVIEW_CLASS_LABEL[move.class]}`)}</title></circle>`,
    )
    .join('');
  const cursor =
    currentPly >= 0 && currentPly < values.length
      ? `<g class="review-cursor"><line x1="${x(currentPly)}" x2="${x(currentPly)}" y1="6" y2="104" stroke="var(--accent)" stroke-width="1.5"/><circle cx="${x(currentPly)}" cy="${y(values[currentPly])}" r="4" fill="var(--accent)" stroke="var(--panel)" stroke-width="1.5"/></g>`
      : '';
  // The focusable parent owns slider semantics. The SVG is decorative to assistive tech.
  return svg.replace('role="img"', 'aria-hidden="true"').replace('</svg>', `${errors}${cursor}</svg>`);
}

export function reviewSummaryHtml(review: GameReview): string {
  const side = (label: string, accuracy: number, counts: GameReview['counts']['white']) => {
    const measured = Object.entries(counts).some(([key, count]) => key !== 'forced' && count > 0);
    const count = (n: number, singular: string, plural = `${singular}s`) => `${n} ${n === 1 ? singular : plural}`;
    return (
      `<div class="review-side"><strong>${label}</strong><span class="review-accuracy">${measured ? `${accuracy.toFixed(1)}%` : '—'}</span>` +
      `<span class="review-counts">${count(counts.inaccuracy, 'inaccuracy', 'inaccuracies')} · ${count(counts.mistake, 'mistake')} · ${count(counts.blunder, 'blunder')}</span></div>`
    );
  };
  return `<div class="review-summary">${side('White', review.accuracy.white, review.counts.white)}${side('Black', review.accuracy.black, review.counts.black)}</div>`;
}
