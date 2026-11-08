import { boardToFen } from '../chess/board.ts';
import { parseValidatedFen } from '../chess/fenValidation.ts';

/** Share only the position, never developer flags, model URLs or the sender's game data. */
export function positionShareUrl(fen: string, pageUrl: string): string {
  const url = new URL('/app/analysis/', new URL(pageUrl).origin);
  url.searchParams.set('fen', boardToFen(parseValidatedFen(fen)));
  return url.href;
}
