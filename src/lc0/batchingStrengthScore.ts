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

function pairProbs(elo: number, drawRate: number): Record<keyof PairOutcomeCounts, number> | null {
  const p = scoreFromElo(elo);
  const d = Math.min(1, Math.max(0, drawRate));
  const w = p - d / 2;
  const l = 1 - p - d / 2;
  if (w < -1e-12 || l < -1e-12) return null;
  const win = Math.max(0, w);
  const loss = Math.max(0, l);
  return {
    n2: win * win,
    n15: 2 * win * d,
    n1: 2 * win * loss + d * d,
    n05: 2 * d * loss,
    n0: loss * loss,
  };
}

function logLikelihood(elo: number, drawRate: number, counts: PairOutcomeCounts): number {
  const probs = pairProbs(elo, drawRate);
  if (!probs) return Number.NEGATIVE_INFINITY;
  let ll = 0;
  const terms: Array<[number, number]> = [
    [counts.n0, probs.n0],
    [counts.n05, probs.n05],
    [counts.n1, probs.n1],
    [counts.n15, probs.n15],
    [counts.n2, probs.n2],
  ];
  for (const [count, prob] of terms) {
    if (count === 0) continue;
    if (prob <= 0) return Number.NEGATIVE_INFINITY;
    ll += count * Math.log(prob);
  }
  return ll;
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
  let bestElo = 0;
  let bestLl = Number.NEGATIVE_INFINITY;
  for (let elo = -1500; elo <= 1500; elo += 0.5) {
    const ll = logLikelihood(elo, drawRate, counts);
    if (ll > bestLl) {
      bestLl = ll;
      bestElo = elo;
    }
  }
  let low = bestElo;
  let high = bestElo;
  if (Number.isFinite(bestLl)) {
    for (let elo = -1500; elo <= 1500; elo += 0.5) {
      const ll = logLikelihood(elo, drawRate, counts);
      if (!Number.isFinite(ll)) continue;
      if (2 * (bestLl - ll) <= LLR_95) {
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
    ci95: gamesUsed ? [low, high] : [0, 0],
    pairCounts: counts,
    method:
      'pentanomial pair likelihood, draw rate fixed at the observed per-game rate, independent-game pair probabilities, 95% CI from a likelihood-ratio cutoff of 3.841 on a 0.5 Elo grid',
  };
}

/** Ship gate for one equal-visit level. Elo is the candidate (faster) arm. */
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
