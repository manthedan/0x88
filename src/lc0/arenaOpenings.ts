import { boardToFen, parseFen, START_FEN, type BoardState } from '../chess/board.ts';
import { moveToUci } from '../chess/moveCodec.ts';
import { parsePgnGame } from '../chess/pgn.ts';
import { buildBoardHistoryFromMoves } from './history.ts';

export interface ArenaOpening {
  name: string;
  /** Final position where engine play begins. */
  fen: string;
  /** Known replay start. Present when the opening was loaded from moves/PGN. */
  startFen?: string;
  /** UCI moves from startFen to fen. Present when true LC0 history is available. */
  moves?: string[];
  /** Board states from startFen through fen, for LC0 112-plane history input. */
  positions?: BoardState[];
}

interface ArenaOpeningDefinition {
  name: string;
  fen?: string;
  moves?: string[];
  startFen?: string;
}

const BUILTIN_ARENA_OPENING_DEFS: ArenaOpeningDefinition[] = [
  { name: 'Start position', moves: [] },
  { name: 'Italian Game', moves: ['e2e4', 'e7e5', 'g1f3', 'b8c6', 'f1c4'] },
  { name: 'Ruy Lopez', moves: ['e2e4', 'e7e5', 'g1f3', 'b8c6', 'f1b5'] },
  { name: 'Sicilian Open', moves: ['e2e4', 'c7c5', 'd2d4'] },
  { name: 'French Advance', moves: ['e2e4', 'e7e6', 'd2d4', 'd7d5', 'e4e5'] },
  { name: 'Caro-Kann Advance', moves: ['e2e4', 'c7c6', 'd2d4', 'd7d5', 'e4e5'] },
  { name: "Queen's Gambit Declined", moves: ['d2d4', 'd7d5', 'c2c4'] },
  { name: "King's Indian", moves: ['d2d4', 'g8f6', 'c2c4', 'g7g6', 'e2e4'] },
  { name: 'English Opening', moves: ['c2c4'] },
  { name: 'Reti Opening', moves: ['g1f3'] },
  { name: 'Scandinavian Defense', moves: ['e2e4', 'd7d5'] },
  { name: 'Pirc Defense', moves: ['e2e4', 'd7d6'] },
  { name: 'Slav Defense', moves: ['d2d4', 'd7d5', 'c2c4', 'c7c6'] },
  { name: 'Benoni Defense', moves: ['d2d4', 'c7c5', 'd4d5'] },
  { name: 'Dutch Defense', moves: ['d2d4', 'f7f5', 'c2c4'] },
  { name: 'Four Knights', moves: ['e2e4', 'e7e5', 'g1f3', 'b8c6', 'b1c3', 'g8f6'] },
];

export const BUILTIN_ARENA_OPENINGS: ArenaOpening[] = BUILTIN_ARENA_OPENING_DEFS.map((opening) => {
  if (opening.moves) return openingFromUciMoves(opening.name, opening.moves, opening.startFen);
  return { name: opening.name, fen: normalizeFen(opening.fen ?? START_FEN) };
});

export function normalizeFen(fen: string): string {
  return boardToFen(parseFen(fen));
}

export function openingFromUciMoves(name: string, moves: readonly string[], startFen = START_FEN): ArenaOpening {
  const positions = buildBoardHistoryFromMoves(moves, startFen);
  return {
    name,
    fen: boardToFen(positions[positions.length - 1]),
    startFen: normalizeFen(startFen),
    moves: [...moves],
    positions,
  };
}

function stripInlineComment(line: string): string {
  return line.replace(/(^|\s+)#.*$/, '').trim();
}

function splitOpeningLine(line: string): { name?: string; body: string } {
  const delimiter = line.includes('|') ? '|' : line.includes(';') ? ';' : null;
  if (!delimiter) return { body: line };
  const parts = line.split(delimiter);
  const name = parts[0].trim();
  return { name: name || undefined, body: parts.slice(1).join(delimiter).trim() };
}

function parseFenOpening(name: string, body: string): ArenaOpening | null {
  try {
    return { name, fen: normalizeFen(body) };
  } catch {
    return null;
  }
}

function uciTokens(body: string): string[] | null {
  const tokens = body.split(/[\s,]+/).map((token) => token.trim()).filter(Boolean);
  if (!tokens.length) return null;
  return tokens.every((token) => /^[a-h][1-8][a-h][1-8][qrbn]?$/i.test(token))
    ? tokens.map((token) => token.toLowerCase())
    : null;
}

function parseUciOpening(name: string, body: string): ArenaOpening | null {
  const moves = uciTokens(body);
  if (!moves) return null;
  return openingFromUciMoves(name, moves);
}

function parsePgnOpening(name: string, body: string): ArenaOpening | null {
  const game = parsePgnGame(body);
  const line = game.tree.mainlineFrom(game.tree.root);
  if (!line.length) return null;
  const moves = line.map((node) => {
    if (!node.move) throw new Error(`Missing move in parsed PGN opening ${name}`);
    return moveToUci(node.move);
  });
  return openingFromUciMoves(name, moves, game.tree.root.fen);
}

function parseOpeningLine(line: string, index: number): ArenaOpening {
  const { name: explicitName, body } = splitOpeningLine(line);
  const name = explicitName || `Position ${index}`;
  const fen = parseFenOpening(name, body);
  if (fen) return fen;
  const uci = parseUciOpening(name, body);
  if (uci) return uci;
  const pgn = parsePgnOpening(name, body);
  if (pgn) return pgn;
  throw new Error(`Opening ${name} is not a valid FEN, UCI move list, or PGN/SAN line`);
}

/**
 * Parse custom arena starting positions. Each non-empty line can be either:
 *   - a raw FEN / EPD-like position (no known history)
 *   - "Name | FEN" or "Name; FEN"
 *   - a UCI replay from the normal start position, e.g. "e2e4 e7e5 g1f3"
 *   - "Name | UCI..." or "Name | PGN/SAN movetext..." for true LC0 history planes
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
