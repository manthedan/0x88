import type { GameResultCode } from './engineBattle.ts';

/**
 * Pure tournament bookkeeping for the arena: pairing generation and standings.
 * Kept free of engines/DOM so it is unit-testable.
 */

export interface ArenaPairing {
  white: string;
  black: string;
}

export interface Standing {
  id: string;
  name: string;
  wins: number;
  losses: number;
  draws: number;
  score: number;
  games: number;
}

/**
 * Round-robin pairings: every unordered pair plays `gamesPerPair` games with
 * alternating colors (so an even count is color-balanced).
 */
export function roundRobinPairings(ids: string[], gamesPerPair = 2): ArenaPairing[] {
  const pairings: ArenaPairing[] = [];
  for (let i = 0; i < ids.length; i++) {
    for (let j = i + 1; j < ids.length; j++) {
      for (let g = 0; g < gamesPerPair; g++) {
        pairings.push(g % 2 === 0 ? { white: ids[i], black: ids[j] } : { white: ids[j], black: ids[i] });
      }
    }
  }
  return pairings;
}

/**
 * Gauntlet pairings: each champion plays every challenger `gamesPerPair` games
 * with alternating colors. Champions do not play each other.
 */
export function gauntletPairings(championIds: string[], challengerIds: string[], gamesPerPair = 2): ArenaPairing[] {
  const pairings: ArenaPairing[] = [];
  for (const champion of championIds) {
    for (const challenger of challengerIds) {
      if (champion === challenger) continue;
      for (let g = 0; g < gamesPerPair; g++) {
        pairings.push(g % 2 === 0 ? { white: champion, black: challenger } : { white: challenger, black: champion });
      }
    }
  }
  return pairings;
}

export function initStandings(engines: { id: string; name: string }[]): Map<string, Standing> {
  const standings = new Map<string, Standing>();
  for (const engine of engines) {
    standings.set(engine.id, { id: engine.id, name: engine.name, wins: 0, losses: 0, draws: 0, score: 0, games: 0 });
  }
  return standings;
}

/** Record a finished game (result is from White's perspective) into standings. */
export function applyGameResult(standings: Map<string, Standing>, whiteId: string, blackId: string, result: GameResultCode): void {
  const white = standings.get(whiteId);
  const black = standings.get(blackId);
  if (!white || !black) return;
  white.games += 1;
  black.games += 1;
  if (result === '1/2-1/2') {
    white.draws += 1; black.draws += 1;
    white.score += 0.5; black.score += 0.5;
  } else if (result === '1-0') {
    white.wins += 1; black.losses += 1;
    white.score += 1;
  } else {
    black.wins += 1; white.losses += 1;
    black.score += 1;
  }
}

/** Standings sorted by score (then wins, then fewer games), for a leaderboard. */
export function rankedStandings(standings: Map<string, Standing>): Standing[] {
  return [...standings.values()].sort((a, b) => b.score - a.score || b.wins - a.wins || a.games - b.games || a.name.localeCompare(b.name));
}
