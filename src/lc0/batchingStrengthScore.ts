import { boardToFen, parseFen, START_FEN } from '../chess/board.ts';
import { moveToUci } from '../chess/moveCodec.ts';
import { legalMoves, makeMove } from '../chess/movegen.ts';
import { moveToSan } from '../chess/san.ts';
import type { GameResultCode } from './engineBattle.ts';

export const BATCHING_STRENGTH_GENERATOR_VERSION = 1;

export interface ScoredGame {
  index: number;
  result: GameResultCode;
  candidateWhite: boolean;
}

export interface PairOutcomeCounts {
  n0: number;
  n05: number;
  n1: number;
  n15: number;
  n2: number;
}

export interface EloEstimate {
  games: number;
  pairs: number;
  unpairedGames: number;
  wins: number;
  draws: number;
  losses: number;
  score: number;
  drawRate: number;
  elo: number;
  ci95: [number, number];
  pairCounts: PairOutcomeCounts;
  method: string;
}

export interface GateResult {
  pass: boolean;
  elo: number;
  ciLow: number;
  ciHigh: number;
  reason: string;
}

const LLR_95 = 3.841458820694124;

export function candidatePoints(result: GameResultCode, candidateWhite: boolean): 0 | 0.5 | 1 {
  if (result === '1/2-1/2') return 0.5;
  return (result === '1-0') === candidateWhite ? 1 : 0;
}

/** Point for the side that still has a queen. Null when the opening is equal material. */
export function queenSidePoints(result: GameResultCode, queenLessColor: 'w' | 'b' | null): 0 | 0.5 | 1 | null {
  if (queenLessColor === null) return null;
  if (result === '1/2-1/2') return 0.5;
  const queenIsWhite = queenLessColor === 'b';
  return (result === '1-0') === queenIsWhite ? 1 : 0;
}

export function pairOutcomeCounts(games: readonly ScoredGame[]): { counts: PairOutcomeCounts; pairs: number; unpaired: number; used: ScoredGame[] } {
  const byPair = new Map<number, ScoredGame[]>();
  for (const game of games) {
    const pair = Math.floor(game.index / 2);
    const list = byPair.get(pair) ?? [];
    list.push(game);
    byPair.set(pair, list);
  }
  const counts: PairOutcomeCounts = { n0: 0, n05: 0, n1: 0, n15: 0, n2: 0 };
  const used: ScoredGame[] = [];
  let pairs = 0;
  let unpaired = 0;
  for (const list of byPair.values()) {
    const even = list.find((game) => game.index % 2 === 0);
    const odd = list.find((game) => game.index % 2 === 1);
    if (!even || !odd) {
      unpaired += list.length;
      continue;
    }
    const points = candidatePoints(even.result, even.candidateWhite) + candidatePoints(odd.result, odd.candidateWhite);
    if (points === 0) counts.n0 += 1;
    else if (points === 0.5) counts.n05 += 1;
    else if (points === 1) counts.n1 += 1;
    else if (points === 1.5) counts.n15 += 1;
    else if (points === 2) counts.n2 += 1;
    else unpaired += 2;
    if (points === 0 || points === 0.5 || points === 1 || points === 1.5 || points === 2) {
      pairs += 1;
      used.push(even, odd);
    }
  }
  return { counts, pairs, unpaired, used };
}

function scoreFromElo(elo: number): number {
  return 1 / (1 + 10 ** (-elo / 400));
}

function eloFromScore(score: number): number {
  const p = Math.min(1 - 1e-6, Math.max(1e-6, score));
  return -400 * Math.log10(1 / p - 1);
}

/**
 * Profile the unrestricted five-category pair distribution at a fixed mean.
 * The two games in an opening pair are NOT assumed independent. The dual
 * optimum gives p_i = (n_i / N) / (1 + t * (x_i - mean)). Bounds include
 * unobserved categories, which can absorb probability at a boundary; an
 * all-split sample therefore has a finite-width interval, not zero variance.
 */
function pairLikelihoodRatio(mean: number, counts: PairOutcomeCounts): number {
  const ns = [counts.n0, counts.n05, counts.n1, counts.n15, counts.n2];
  const xs = [0, 0.25, 0.5, 0.75, 1];
  let low = -1 / (1 - mean);
  let high = 1 / mean;
  for (let step = 0; step < 80; step++) {
    const t = (low + high) / 2;
    const slope = ns.reduce((sum, n, i) => (n ? sum + (n * (xs[i] - mean)) / (1 + t * (xs[i] - mean)) : sum), 0);
    if (slope > 0) low = t;
    else high = t;
  }
  const t = (low + high) / 2;
  return 2 * ns.reduce((sum, n, i) => (n ? sum + n * Math.log(1 + t * (xs[i] - mean)) : sum), 0);
}

export function pentanomialElo(games: readonly ScoredGame[]): EloEstimate {
  const { counts, pairs, unpaired, used } = pairOutcomeCounts(games);
  let wins = 0;
  let draws = 0;
  let losses = 0;
  for (const game of used) {
    const points = candidatePoints(game.result, game.candidateWhite);
    if (points === 1) wins += 1;
    else if (points === 0.5) draws += 1;
    else losses += 1;
  }
  const gamesUsed = wins + draws + losses;
  const score = gamesUsed ? (wins + draws * 0.5) / gamesUsed : 0.5;
  const drawRate = gamesUsed ? draws / gamesUsed : 0;
  const bestElo = Math.round(eloFromScore(score) * 10) / 10 || 0;
  let low = bestElo;
  let high = bestElo;
  if (pairs) {
    for (let elo = -2400; elo <= 2400; elo += 0.5) {
      if (pairLikelihoodRatio(scoreFromElo(elo), counts) <= LLR_95) {
        low = Math.min(low, elo);
        high = Math.max(high, elo);
      }
    }
  }
  return {
    games: gamesUsed,
    pairs,
    unpairedGames: unpaired,
    wins,
    draws,
    losses,
    score,
    drawRate,
    elo: gamesUsed ? bestElo : 0,
    ci95: gamesUsed ? [low, high] : [-2400, 2400],
    pairCounts: counts,
    method:
      'unrestricted pentanomial pair-mean profile likelihood (no within-pair independence assumption); logistic score Elo; nominal 95% LR cutoff 3.841 on a 0.5 Elo grid, clipped at +/-2400',
  };
}

/**
 * Historical-suite sensitivity: retain the first complete pair per exact
 * opening/history, chosen by index (never by outcome). Repeated deterministic
 * games are not independent opening units. This discards later stochastic
 * repeats rather than pretending they add new opening diversity.
 */
export function distinctOpeningElo(games: readonly (ScoredGame & { startFen: string; bookMoves?: readonly string[] })[]): EloEstimate {
  const sorted = [...games].sort((a, b) => a.index - b.index);
  const byIndex = new Map(sorted.map((game) => [game.index, game]));
  const seen = new Set<string>();
  const first: ScoredGame[] = [];
  for (const even of sorted) {
    if (even.index % 2 !== 0) continue;
    const odd = byIndex.get(even.index + 1);
    if (!odd) continue;
    const key = JSON.stringify([even.startFen, even.bookMoves ?? []]);
    if (seen.has(key)) continue;
    seen.add(key);
    first.push(even, odd);
  }
  return pentanomialElo(first);
}

/** Arithmetic gate for one equal-visit level, not validation of its sample design. */
export function equalVisitGate(estimate: EloEstimate): GateResult {
  const [ciLow, ciHigh] = estimate.ci95;
  if (estimate.pairs < 1) return { pass: false, elo: estimate.elo, ciLow, ciHigh, reason: 'no complete pairs' };
  if (estimate.elo < -10) return { pass: false, elo: estimate.elo, ciLow, ciHigh, reason: 'point estimate below -10' };
  if (ciHigh < 0) return { pass: false, elo: estimate.elo, ciLow, ciHigh, reason: '95% CI is entirely below 0' };
  return { pass: true, elo: estimate.elo, ciLow, ciHigh, reason: 'point estimate >= -10 and 95% CI upper bound >= 0' };
}

export function sameGate(estimate: EloEstimate): GateResult {
  return equalVisitGate(estimate);
}

export function uciLineToPgn(options: {
  startFen: string;
  moves: readonly string[];
  white: string;
  black: string;
  result: GameResultCode | '*';
  headers?: Record<string, string>;
}): string {
  let board = parseFen(options.startFen);
  const sans: string[] = [];
  for (const uci of options.moves) {
    const move = legalMoves(board).find((candidate) => moveToUci(candidate) === uci);
    if (!move) {
      sans.push(`{illegal ${uci}}`);
      break;
    }
    const san = moveToSan(board, move);
    if (board.turn === 'w') sans.push(`${board.fullmove}. ${san}`);
    else if (sans.length === 0) sans.push(`${board.fullmove}... ${san}`);
    else sans.push(san);
    board = makeMove(board, move);
  }
  const tags: Record<string, string> = {
    Event: 'Batching strength',
    Site: 'leelaweb',
    Date: '2026.10.11',
    Round: '?',
    White: options.white,
    Black: options.black,
    Result: options.result,
    ...(options.headers ?? {}),
  };
  const normalizedStart = boardToFen(parseFen(options.startFen));
  if (normalizedStart !== START_FEN) {
    tags.SetUp = '1';
    tags.FEN = normalizedStart;
  }
  const header = Object.entries(tags)
    .map(([key, value]) => `[${key} "${value.replaceAll('"', '')}"]`)
    .join('\n');
  const movetext = `${sans.join(' ')} ${options.result}`.trim();
  return `${header}\n\n${movetext}\n`;
}

export function eloFromScoreForTest(score: number): number {
  return eloFromScore(score);
}
