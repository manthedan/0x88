import { type AnalysisLine, evalBarWhitePercent } from './analysisFormat.ts';
import type { EngineRow } from './engineCatalog.ts';
import type { ReviewPosition } from './gameReview.ts';

export function analysisResultCacheKey(fen: string, history: string[], row: EngineRow, settings: { multiPv: number; runtime: string; cpu: string }): string {
  return JSON.stringify([fen, history, row.family, row.variant, row.strength, settings.multiPv, settings.runtime, settings.cpu]);
}

/** Analysis lines are already White-relative; do not flip Black scores twice. */
export function cachedReviewPosition(lines: AnalysisLine[] | undefined, legalMoves: number): ReviewPosition | undefined {
  const first = lines?.find((line) => line.multipv === 1 && !line.error);
  if (!first || (first.scoreCp === undefined && first.mateIn === undefined)) return undefined;
  return { winWhite: evalBarWhitePercent(first.scoreCp, first.mateIn) / 100, bestUci: first.pvUci[0] ?? null, legalMoves };
}
