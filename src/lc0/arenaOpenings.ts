import { boardToFen, parseFen, START_FEN } from '../chess/board.ts';

export interface ArenaOpening {
  name: string;
  fen: string;
}

export const BUILTIN_ARENA_OPENINGS: ArenaOpening[] = [
  { name: 'Start position', fen: START_FEN },
  { name: 'Italian Game', fen: 'r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 2 3' },
  { name: 'Ruy Lopez', fen: 'r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 2 3' },
  { name: 'Sicilian Open', fen: 'rnbqkbnr/pp1ppppp/8/2p5/3PP3/8/PPP2PPP/RNBQKBNR b KQkq d3 0 2' },
  { name: 'French Advance', fen: 'rnbqkbnr/ppp2ppp/4p3/3pP3/3P4/8/PPP2PPP/RNBQKBNR b KQkq - 0 3' },
  { name: 'Caro-Kann Advance', fen: 'rnbqkbnr/pp2pppp/2p5/3pP3/3P4/8/PPP2PPP/RNBQKBNR b KQkq - 0 3' },
  { name: "Queen's Gambit Declined", fen: 'rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq c3 0 2' },
  { name: "King's Indian", fen: 'rnbqkb1r/pppppp1p/5np1/8/2PPP3/8/PP3PPP/RNBQKBNR b KQkq - 0 3' },
  { name: 'English Opening', fen: 'rnbqkbnr/pppppppp/8/8/2P5/8/PP1PPPPP/RNBQKBNR b KQkq c3 0 1' },
  { name: 'Reti Opening', fen: 'rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R b KQkq - 1 1' },
  { name: 'Scandinavian Defense', fen: 'rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2' },
  { name: 'Pirc Defense', fen: 'rnbqkbnr/ppp1pppp/3p4/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2' },
  { name: 'Slav Defense', fen: 'rnbqkbnr/pp2pppp/2p5/3p4/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3' },
  { name: 'Benoni Defense', fen: 'rnbqkbnr/pp1ppppp/8/2pP4/8/8/PPP1PPPP/RNBQKBNR b KQkq - 0 2' },
  { name: 'Dutch Defense', fen: 'rnbqkbnr/ppppp1pp/8/5p2/2PP4/8/PP2PPPP/RNBQKBNR b KQkq c3 0 2' },
  { name: 'Four Knights', fen: 'r1bqkb1r/pppp1ppp/2n2n2/4p3/4P3/2N2N2/PPPP1PPP/R1BQKB1R w KQkq - 4 4' },
].map((opening) => ({ ...opening, fen: normalizeFen(opening.fen) }));

export function normalizeFen(fen: string): string {
  return boardToFen(parseFen(fen));
}

function stripInlineComment(line: string): string {
  return line.replace(/(^|\s+)#.*$/, '').trim();
}

function parseOpeningLine(line: string, index: number): ArenaOpening {
  const delimiter = line.includes('|') ? '|' : line.includes(';') ? ';' : null;
  const parts = delimiter ? line.split(delimiter) : null;
  const name = parts && parts.length >= 2 ? parts[0].trim() : `Position ${index}`;
  const fen = parts && parts.length >= 2 ? parts.slice(1).join(delimiter!).trim() : line;
  return { name: name || `Position ${index}`, fen: normalizeFen(fen) };
}

/**
 * Parse custom arena starting positions. Each non-empty line can be either:
 *   - a raw FEN / EPD-like position
 *   - "Name | FEN"
 *   - "Name; FEN"
 */
export function parseArenaOpenings(text: string): ArenaOpening[] {
  const openings: ArenaOpening[] = [];
  for (const raw of text.replace(/\r\n?/g, '\n').split('\n')) {
    const line = stripInlineComment(raw);
    if (!line) continue;
    openings.push(parseOpeningLine(line, openings.length + 1));
  }
  return openings;
}

export function scheduleOpenings<T extends { white: string; black: string }>(pairings: readonly T[], openings: readonly ArenaOpening[]): (T & { opening: ArenaOpening })[] {
  const selected = openings.length ? openings : [{ name: 'Start position', fen: START_FEN }];
  const games: (T & { opening: ArenaOpening })[] = [];
  for (const opening of selected) {
    for (const pairing of pairings) games.push({ ...pairing, opening });
  }
  return games;
}
