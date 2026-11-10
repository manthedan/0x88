import { START_FEN } from '../chess/board.ts';
import { parseValidatedFen } from '../chess/fenValidation.ts';

/** URL input must never throw during module evaluation or discard a saved workspace. */
export function analysisUrlFen(raw: string | null): { fen: string; error: string | null } {
  if (raw === null) return { fen: START_FEN, error: null };
  try {
    parseValidatedFen(raw);
    return { fen: raw, error: null };
  } catch (error) {
    return { fen: START_FEN, error: `Could not load linked FEN: ${(error as Error).message}` };
  }
}
