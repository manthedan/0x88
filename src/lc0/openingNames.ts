import { parseFen } from '../chess/board.ts';
import { legalMoves } from '../chess/movegen.ts';
import type { GameNode } from './gameTree.ts';

export type OpeningName = readonly [eco: string, name: string];
export type OpeningNameIndex = Record<string, OpeningName>;

/** EPD identity, with an en-passant square only when a legal capture exists (Lichess convention). */
export function openingPositionKey(fen: string): string {
  const fields = fen.split(/\s+/).slice(0, 4);
  if (fields[3] !== '-') {
    const board = parseFen(fen);
    const legalEp = legalMoves(board).some((move) => move.to === board.epSquare && board.squares[move.from]?.endsWith('p') && move.from % 8 !== move.to % 8);
    if (!legalEp) fields[3] = '-';
  }
  return fields.join(' ');
}

/** Walk backwards on the current line: never leak a later opening into an earlier ply.
 * Off-book continuations retain their last named position; variations use their own ancestry.
 */
export function openingNameAt(index: OpeningNameIndex, current: GameNode): OpeningName | null {
  for (let node: GameNode | null = current; node; node = node.parent) {
    const opening = index[openingPositionKey(node.fen)];
    if (opening) return opening;
  }
  return null;
}
