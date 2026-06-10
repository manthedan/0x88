/**
 * Pure tournament scheduling and standings for the arena: round-robin and
 * gauntlet pairings expanded over openings × games-per-pairing with color
 * alternation, plus standings with W-D-L, per-opponent records, and an
 * approximate Elo-vs-pool estimate with ~95% error bars (logistic Elo on the
 * score fraction with a normal approximation — Ordo-lite, not Ordo).
 * See docs/arena_analysis_roadmap.md stage 1.
 */

export type TournamentMode = 'match' | 'round-robin' | 'gauntlet';

export interface TournamentPairing {
  /** Participant ids; `a` holds white in even-numbered games of the pairing. */
  a: string;
  b: string;
}

export interface ScheduledGame<TOpening> {
  whiteId: string;
  blackId: string;
  opening: TOpening;
  /** Stable key identifying the pairing this game belongs to. */
  pairingKey: string;
}

/** All unordered pairs, in seat order: (1,2), (1,3), (2,3), … */
export function roundRobinPairings(ids: readonly string[]): TournamentPairing[] {
  const pairings: TournamentPairing[] = [];
  for (let i = 0; i < ids.length; i++) {
    for (let j = i + 1; j < ids.length; j++) pairings.push({ a: ids[i], b: ids[j] });
  }
  return pairings;
}

/** Seat 1 challenges every other seat: (1,2), (1,3), … */
export function gauntletPairings(ids: readonly string[]): TournamentPairing[] {
  return ids.slice(1).map((id) => ({ a: ids[0], b: id }));
}

export function tournamentPairings(mode: TournamentMode, ids: readonly string[]): TournamentPairing[] {
  if (ids.length < 2) return [];
  if (mode === 'match') return [{ a: ids[0], b: ids[1] }];
  if (mode === 'gauntlet') return gauntletPairings(ids);
  return roundRobinPairings(ids);
}

/**
 * Expand pairings into a per-game schedule: for each pairing, every opening
 * is played `gamesPerOpening` times with colors alternating per game index —
 * the same color-balance rule as the existing two-engine arena loop.
 */
export function buildSchedule<TOpening>(
  pairings: readonly TournamentPairing[],
  openings: readonly TOpening[],
  gamesPerOpening: number,
): ScheduledGame<TOpening>[] {
  const games: ScheduledGame<TOpening>[] = [];
  const perPairing = Math.max(1, Math.floor(gamesPerOpening));
  for (const pairing of pairings) {
    const pairingKey = `${pairing.a}|${pairing.b}`;
    for (let g = 0; g < perPairing; g++) {
      for (const opening of openings) {
        const aIsWhite = g % 2 === 0;
        games.push({
          whiteId: aIsWhite ? pairing.a : pairing.b,
          blackId: aIsWhite ? pairing.b : pairing.a,
          opening,
          pairingKey,
        });
      }
    }
  }
  return games;
}

export type GameResultText = '1-0' | '0-1' | '1/2-1/2';

export interface OpponentRecord {
  wins: number;
  draws: number;
  losses: number;
}

export interface StandingsRow {
  id: string;
  name: string;
  games: number;
  wins: number;
  draws: number;
  losses: number;
  points: number;
  /** Score fraction over played games, 0..1; null before any game. */
  score: number | null;
  /** Approximate Elo difference vs the pool average, from the score fraction. */
  eloDiff: number | null;
  /** Approximate ±95% error on eloDiff; null when score is degenerate (0 or 1). */
  eloError: number | null;
  perOpponent: Map<string, OpponentRecord>;
}

const ELO_CLAMP = 1200;

/** Logistic Elo difference from a score fraction, clamped for degenerate scores. */
export function eloFromScore(score: number): number {
  if (score <= 0) return -ELO_CLAMP;
  if (score >= 1) return ELO_CLAMP;
  // `|| 0` normalizes the -0 that the logistic produces at score 0.5.
  return Math.max(-ELO_CLAMP, Math.min(ELO_CLAMP, -400 * Math.log10(1 / score - 1))) || 0;
}

/**
 * Approximate ±95% Elo error from a binomial normal approximation on the
 * score fraction (draws treated as half-points; slightly conservative for
 * draw-heavy samples). Null for degenerate scores where the logistic slope
 * blows up.
 */
export function eloError95(score: number, games: number): number | null {
  if (games <= 0 || score <= 0 || score >= 1) return null;
  const sigma = Math.sqrt((score * (1 - score)) / games);
  const slope = 400 / (Math.LN10 * score * (1 - score));
  return Math.min(ELO_CLAMP, 1.96 * sigma * slope);
}

export class TournamentStandings {
  private readonly rows = new Map<string, StandingsRow>();

  constructor(participants: readonly { id: string; name: string }[]) {
    for (const participant of participants) {
      this.rows.set(participant.id, {
        id: participant.id,
        name: participant.name,
        games: 0,
        wins: 0,
        draws: 0,
        losses: 0,
        points: 0,
        score: null,
        eloDiff: null,
        eloError: null,
        perOpponent: new Map(),
      });
    }
  }

  /** Record one finished game. Unknown ids are ignored (defensive for replays). */
  record(whiteId: string, blackId: string, result: GameResultText): void {
    const white = this.rows.get(whiteId);
    const black = this.rows.get(blackId);
    if (!white || !black || whiteId === blackId) return;
    const whitePoints = result === '1-0' ? 1 : result === '0-1' ? 0 : 0.5;
    this.apply(white, black.id, whitePoints);
    this.apply(black, white.id, 1 - whitePoints);
  }

  private apply(row: StandingsRow, opponentId: string, points: number): void {
    row.games += 1;
    row.points += points;
    const record = row.perOpponent.get(opponentId) ?? { wins: 0, draws: 0, losses: 0 };
    if (points === 1) { row.wins += 1; record.wins += 1; }
    else if (points === 0) { row.losses += 1; record.losses += 1; }
    else { row.draws += 1; record.draws += 1; }
    row.perOpponent.set(opponentId, record);
    row.score = row.points / row.games;
    row.eloDiff = eloFromScore(row.score);
    row.eloError = eloError95(row.score, row.games);
  }

  /** Rows sorted by points, then score fraction, then name; stable for ties. */
  table(): StandingsRow[] {
    return [...this.rows.values()].sort((a, b) =>
      b.points - a.points
      || (b.score ?? 0) - (a.score ?? 0)
      || a.name.localeCompare(b.name));
  }

  totalGames(): number {
    let total = 0;
    for (const row of this.rows.values()) total += row.games;
    return total / 2;
  }
}
