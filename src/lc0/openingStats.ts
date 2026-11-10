import { parseFen, squareName } from '../chess/board.ts';
import { moveToUci } from '../chess/moveCodec.ts';
import { legalMoves } from '../chess/movegen.ts';
import type { GameTree } from './gameTree.ts';

/** Mainline statistics. A game contributes only its first departure from each position. */
export interface OpeningMoveStat {
  uci: string;
  san: string;
  count: number;
  whiteWins: number;
  blackWins: number;
  draws: number;
  unknown: number;
}

export interface ImportedGame {
  tree: GameTree;
  result: string;
}

export type OpeningPositionIndex = Record<string, OpeningMoveStat[]>;
export const OPENING_INDEX_VERSION = 2;

/** Ignore clocks, canonicalize castling, and retain EP only if a legal capture exists. */
export function positionKey(fen: string): string {
  const [placement, turn, rights = '-', ep = '-'] = fen.trim().split(/\s+/);
  const board = parseFen(fen);
  const homes = { K: [4, 7, 'w'], Q: [4, 0, 'w'], k: [60, 63, 'b'], q: [60, 56, 'b'] } as const;
  const castling =
    Object.entries(homes)
      .filter(([right, [king, rook, color]]) => rights.includes(right) && board.squares[king] === `${color}k` && board.squares[rook] === `${color}r`)
      .map(([right]) => right)
      .join('') || '-';
  const target = board.epSquare;
  const legalEp =
    ep !== '-' &&
    target !== null &&
    Math.floor(target / 8) === (turn === 'w' ? 5 : 2) &&
    !board.squares[target] &&
    board.squares[target + (turn === 'w' ? -8 : 8)] === (turn === 'w' ? 'bp' : 'wp') &&
    legalMoves(board).some((move) => move.to === target && board.squares[move.from] === `${turn}p` && move.from % 8 !== move.to % 8);
  return `${placement} ${turn} ${castling} ${legalEp ? squareName(board.epSquare!) : '-'}`;
}

export function mergeOpeningMoveStats(groups: readonly OpeningMoveStat[][]): OpeningMoveStat[] {
  const byUci = new Map<string, OpeningMoveStat>();
  for (const stats of groups) {
    for (const source of stats) {
      const stat = byUci.get(source.uci) ?? { uci: source.uci, san: source.san, count: 0, whiteWins: 0, blackWins: 0, draws: 0, unknown: 0 };
      stat.count += source.count;
      stat.whiteWins += source.whiteWins;
      stat.blackWins += source.blackWins;
      stat.draws += source.draws;
      stat.unknown += source.unknown;
      byUci.set(source.uci, stat);
    }
  }
  return [...byUci.values()].sort((a, b) => b.count - a.count || a.san.localeCompare(b.san));
}

export function buildOpeningPositionIndex(games: ImportedGame[]): OpeningPositionIndex {
  const byPosition = new Map<string, Map<string, OpeningMoveStat>>();
  for (const game of games) {
    const seen = new Set<string>();
    let node = game.tree.root;
    while (node.children[0]) {
      const child = node.children[0];
      const key = positionKey(node.fen);
      if (child.move && !seen.has(key)) {
        seen.add(key);
        let byUci = byPosition.get(key);
        if (!byUci) {
          byUci = new Map();
          byPosition.set(key, byUci);
        }
        const uci = moveToUci(child.move);
        const stat = byUci.get(uci) ?? { uci, san: child.san ?? uci, count: 0, whiteWins: 0, blackWins: 0, draws: 0, unknown: 0 };
        stat.count++;
        if (game.result === '1-0') stat.whiteWins++;
        else if (game.result === '0-1') stat.blackWins++;
        else if (game.result === '1/2-1/2') stat.draws++;
        else stat.unknown++;
        byUci.set(uci, stat);
      }
      node = child;
    }
  }
  const index: OpeningPositionIndex = {};
  for (const [key, stats] of byPosition) index[key] = mergeOpeningMoveStats([[...stats.values()]]);
  return index;
}

export function openingStatsFromIndex(index: OpeningPositionIndex | null | undefined, fen: string): OpeningMoveStat[] {
  return index?.[positionKey(fen)] ?? [];
}

export function openingStatsForPosition(games: ImportedGame[], fen: string): OpeningMoveStat[] {
  return openingStatsFromIndex(buildOpeningPositionIndex(games), fen);
}

/** Unique games with a continuation, not terminal visits to the position. */
export function openingSummary(stats: OpeningMoveStat[]): { total: number; whiteWins: number; blackWins: number; draws: number; unknown: number } {
  return stats.reduce(
    (acc, stat) => ({
      total: acc.total + stat.count,
      whiteWins: acc.whiteWins + stat.whiteWins,
      blackWins: acc.blackWins + stat.blackWins,
      draws: acc.draws + stat.draws,
      unknown: acc.unknown + stat.unknown,
    }),
    { total: 0, whiteWins: 0, blackWins: 0, draws: 0, unknown: 0 },
  );
}
