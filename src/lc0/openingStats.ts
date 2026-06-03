import { moveToUci } from '../chess/moveCodec.ts';
import type { GameNode, GameTree } from './gameTree.ts';

/**
 * OpeningTree-style aggregation: given many imported games, summarize the moves
 * played from a given position with their counts and white/draw/loss results.
 * Pure and unit-testable.
 */

export interface OpeningMoveStat {
  uci: string;
  san: string;
  count: number;
  whiteWins: number;
  blackWins: number;
  draws: number;
}

export interface ImportedGame {
  tree: GameTree;
  result: string;
}

/** Position key ignoring move clocks (piece placement, side, castling, en passant). */
export function positionKey(fen: string): string {
  return fen.split(' ').slice(0, 4).join(' ');
}

/**
 * Moves played from the position matching `fen` across all games (transpositions
 * included), most frequent first. Results are from White's perspective.
 */
export function openingStatsForPosition(games: ImportedGame[], fen: string): OpeningMoveStat[] {
  const target = positionKey(fen);
  const byUci = new Map<string, OpeningMoveStat>();
  const tally = (node: GameNode, result: string) => {
    const atTarget = positionKey(node.fen) === target;
    for (const child of node.children) {
      if (atTarget && child.move) {
        const uci = moveToUci(child.move);
        let stat = byUci.get(uci);
        if (!stat) {
          stat = { uci, san: child.san ?? uci, count: 0, whiteWins: 0, blackWins: 0, draws: 0 };
          byUci.set(uci, stat);
        }
        stat.count += 1;
        if (result === '1-0') stat.whiteWins += 1;
        else if (result === '0-1') stat.blackWins += 1;
        else if (result === '1/2-1/2') stat.draws += 1;
      }
      tally(child, result);
    }
  };
  for (const game of games) tally(game.tree.root, game.result);
  return [...byUci.values()].sort((a, b) => b.count - a.count || a.san.localeCompare(b.san));
}

/** Total games and aggregate result split for a position (sum over its moves). */
export function openingSummary(stats: OpeningMoveStat[]): { total: number; whiteWins: number; blackWins: number; draws: number } {
  return stats.reduce(
    (acc, stat) => ({
      total: acc.total + stat.count,
      whiteWins: acc.whiteWins + stat.whiteWins,
      blackWins: acc.blackWins + stat.blackWins,
      draws: acc.draws + stat.draws,
    }),
    { total: 0, whiteWins: 0, blackWins: 0, draws: 0 },
  );
}
