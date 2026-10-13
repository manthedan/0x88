import { type BoardState, parseFen } from '../chess/board.ts';
import { type ArenaOpening, BUILTIN_ARENA_OPENINGS, openingFromUciMoves } from './arenaOpenings.ts';
import { LQO_BLACK_OPENING_BOOK, LQO_WHITE_OPENING_BOOK } from './lqoOpeningBookData.ts';

/** Product Play FEN when the human is White and the bot (Black) is missing the queen. */
export const LQO_BLACK_MISSING_QUEEN_FEN = 'rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';
/** Product Play FEN when the human is Black and the bot (White) is missing the queen. */
export const LQO_WHITE_MISSING_QUEEN_FEN = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1';

export const SMALL_NET_EPD_PATH = '/lc0/opening_books/queen500.epd';

export interface StrengthOpening extends ArenaOpening {
  /** Side missing its queen, or null for an equal-material opening. */
  queenLessColor: 'w' | 'b' | null;
  positions: BoardState[];
}

function queenCounts(board: BoardState): { w: number; b: number } {
  let w = 0;
  let b = 0;
  for (const piece of board.squares) {
    if (piece === 'wq') w += 1;
    else if (piece === 'bq') b += 1;
  }
  return { w, b };
}

function topBookReply(packed: string): string | null {
  let bestUci: string | null = null;
  let bestWeight = -1;
  for (const entry of packed.split(',')) {
    if (!entry) continue;
    const colon = entry.lastIndexOf(':');
    if (colon <= 0) continue;
    const uci = entry.slice(0, colon);
    const weight = Number.parseInt(entry.slice(colon + 1), 36);
    if (!uci || !Number.isFinite(weight) || weight <= bestWeight) continue;
    bestUci = uci;
    bestWeight = weight;
  }
  return bestUci;
}

function bookLines(book: readonly (readonly [string, string])[], startFen: string, queenLessColor: 'w' | 'b', minHistoryPlies: number): StrengthOpening[] {
  const out: StrengthOpening[] = [];
  const seen = new Set<string>();
  const rows = [...book].sort((a, b) => a[0].localeCompare(b[0]) || a[1].localeCompare(b[1]));
  for (const [key, packed] of rows) {
    const history = key.length ? key.split(' ') : [];
    if (history.length < minHistoryPlies) continue;
    const reply = topBookReply(packed);
    if (!reply) continue;
    const moves = history.concat(reply);
    let opening: ArenaOpening;
    try {
      opening = openingFromUciMoves(`lqo-${queenLessColor}-${moves.join('.')}`, moves, startFen);
    } catch {
      continue;
    }
    const counts = queenCounts(opening.positions?.at(-1) ?? parseFen(opening.fen));
    const keepsOdds = queenLessColor === 'b' ? counts.b === 0 && counts.w === 1 : counts.w === 0 && counts.b === 1;
    if (!keepsOdds || seen.has(opening.fen) || !opening.positions) continue;
    seen.add(opening.fen);
    out.push({ ...opening, queenLessColor, positions: opening.positions });
  }
  return out;
}

function stride<T>(items: readonly T[], count: number): T[] {
  if (count <= 0 || items.length === 0) return [];
  if (count >= items.length) return [...items];
  const picked: T[] = [];
  const strideWidth = items.length / count;
  for (let i = 0; i < count; i++) picked.push(items[Math.min(items.length - 1, Math.floor(i * strideWidth))]);
  return picked;
}

function bareStart(name: string, fen: string, queenLessColor: 'w' | 'b'): StrengthOpening {
  const opening = openingFromUciMoves(name, [], fen);
  if (!opening.positions) throw new Error(`missing positions for ${name}`);
  return { ...opening, queenLessColor, positions: opening.positions };
}

/**
 * Queen-odds openings. Both product start FENs are included. The rest are
 * deterministic highest-weight lines from the product LQO books, so the
 * searched position varies and the book itself is not part of the match.
 */
export function lqoQueenOddsOpenings(limit: number): StrengthOpening[] {
  const want = Math.max(2, Math.floor(limit));
  const blackStart = bareStart('lqo-product-black-missing', LQO_BLACK_MISSING_QUEEN_FEN, 'b');
  const whiteStart = bareStart('lqo-product-white-missing', LQO_WHITE_MISSING_QUEEN_FEN, 'w');
  const blackLines = bookLines(LQO_BLACK_OPENING_BOOK, LQO_BLACK_MISSING_QUEEN_FEN, 'b', 3);
  const whiteLines = bookLines(LQO_WHITE_OPENING_BOOK, LQO_WHITE_MISSING_QUEEN_FEN, 'w', 2);
  const remaining = want - 2;
  const whiteBudget = Math.min(whiteLines.length, Math.max(1, Math.round(remaining * 0.2)));
  const blackBudget = Math.min(blackLines.length, remaining - whiteBudget);
  const extraWhite = Math.min(whiteLines.length - whiteBudget, remaining - blackBudget - whiteBudget);
  return [blackStart, whiteStart, ...stride(blackLines, blackBudget), ...stride(whiteLines, whiteBudget + Math.max(0, extraWhite))];
}

export function openingsFromEpd(text: string, limit: number): StrengthOpening[] {
  const out: StrengthOpening[] = [];
  const seen = new Set<string>();
  for (const rawLine of text.split(/\r?\n/)) {
    const line = rawLine.replace(/(^|\s+)#.*$/, '').trim();
    if (!line || line.startsWith('#')) continue;
    let opening: ArenaOpening;
    try {
      opening = openingFromUciMoves(`epd-${out.length + 1}`, [], line);
    } catch {
      continue;
    }
    if (seen.has(opening.fen) || !opening.positions) continue;
    const counts = queenCounts(opening.positions[opening.positions.length - 1]);
    if (counts.w !== 1 || counts.b !== 1) continue;
    seen.add(opening.fen);
    out.push({ ...opening, name: `epd-${out.length + 1}`, queenLessColor: null, positions: opening.positions });
    if (out.length >= limit) break;
  }
  return out;
}

/** Arena's built-in lines first (real history), then equal-material EPD fills. */
export function smallNetOpenings(epdText: string, limit: number): StrengthOpening[] {
  const out: StrengthOpening[] = [];
  const seen = new Set<string>();
  for (const opening of BUILTIN_ARENA_OPENINGS) {
    if (!opening.positions || seen.has(opening.fen)) continue;
    seen.add(opening.fen);
    out.push({ ...opening, queenLessColor: null, positions: opening.positions });
    if (out.length >= limit) return out;
  }
  for (const opening of openingsFromEpd(epdText, limit)) {
    if (seen.has(opening.fen)) continue;
    seen.add(opening.fen);
    out.push(opening);
    if (out.length >= limit) break;
  }
  return out;
}

export function validateStrengthOpenings(openings: readonly StrengthOpening[], requested: number, gamesTarget: number): void {
  if (openings.length !== requested) throw new Error(`requested ${requested} distinct openings, got ${openings.length}; supply a suitable opening suite`);
  if (gamesTarget > openings.length * 2) throw new Error('game target would repeat opening pairs; increase --opening-count');
}

export function scheduleGame(openingCount: number, gameIndex: number): { pair: number; openingIndex: number; candidateWhite: boolean } {
  const pair = Math.floor(gameIndex / 2);
  const count = Math.max(1, openingCount);
  return { pair, openingIndex: pair % count, candidateWhite: gameIndex % 2 === 0 };
}
