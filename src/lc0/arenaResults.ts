import { evalBarWhitePercent } from './analysisFormat.ts';
import type { ArenaChartSample } from './arenaCheckpoint.ts';
import type { StockfishInfoLine } from './stockfishEngine.ts';
import type { GameResultText, ScheduledGame } from './tournament.ts';

export function formatArenaPoints(value: number): string {
  return value % 1 ? `${Math.floor(value) || ''}½` : String(value);
}

export function arenaPointsForSeat(game: { whiteId: string }, seatId: string, result: GameResultText): number {
  return result === '1/2-1/2' ? 0.5 : (game.whiteId === seatId) === (result === '1-0') ? 1 : 0;
}

/** Capture each CPU game's own search info, even when another game owns the board. */
export function arenaUciChartSample(
  ply: number,
  engine: { id: string; name: string },
  fen: string,
  lines: readonly StockfishInfoLine[],
  moveMs: number,
): ArenaChartSample {
  const info = lines[0];
  const sign = fen.split(' ')[1] === 'b' ? -1 : 1;
  const cp = info?.scoreCp === undefined ? undefined : sign * info.scoreCp;
  const mate = info?.mateIn === undefined ? undefined : sign * info.mateIn;
  const whiteScore = mate !== undefined ? (mate > 0 ? 1 : 0) : cp !== undefined ? evalBarWhitePercent(cp, undefined) / 100 : undefined;
  return { ply, engineId: engine.id, engineName: engine.name, whiteScore, moveMs: Math.round(moveMs), nps: info?.nps };
}

export interface ArenaOpeningPair<T> {
  pairingKey: string;
  a: string;
  b: string;
  opening: T;
  openingIndex: number;
  round: number;
  /** A's White game, then A's Black game. The latter is absent for odd run lengths. */
  indices: number[];
  results: (GameResultText | null)[];
  points: [number, number];
  played: number;
}

/** Pair by suite slot and repetition, never completion order, name, or final FEN. */
export function arenaOpeningPairs<T>(
  schedule: readonly ScheduledGame<T>[],
  completed: readonly { scheduleIndex: number; result: GameResultText }[],
): ArenaOpeningPair<T>[] {
  const results = new Map(completed.map((game) => [game.scheduleIndex, game.result]));
  const groups = new Map<string, ArenaOpeningPair<T>[]>();
  const pairs: ArenaOpeningPair<T>[] = [];
  for (const [index, game] of schedule.entries()) {
    const key = `${game.pairingKey}:${game.openingIndex}`;
    const group = groups.get(key) ?? [];
    let pair = group.at(-1);
    if (!pair || pair.indices.length === 2) {
      pair = {
        pairingKey: game.pairingKey,
        a: game.whiteId,
        b: game.blackId,
        opening: game.opening,
        openingIndex: game.openingIndex,
        round: group.length + 1,
        indices: [],
        results: [],
        points: [0, 0],
        played: 0,
      };
      group.push(pair);
      groups.set(key, group);
      pairs.push(pair);
    }
    const result = results.get(index) ?? null;
    pair.indices.push(index);
    pair.results.push(result);
    if (result) {
      const points = arenaPointsForSeat(game, pair.a, result);
      pair.points[0] += points;
      pair.points[1] += 1 - points;
      pair.played++;
    }
  }
  return pairs;
}
