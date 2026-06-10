/**
 * Pure game-review math: move classification and accuracy from a sequence of
 * white-perspective win probabilities along the mainline, plus an annotated
 * PGN generator. Lichess-style approximations — the per-move accuracy curve
 * is lichess's published formula, the game accuracy is a per-side mean, and
 * Brilliant/Great detection (sacrifice analysis) is intentionally out of
 * scope. See docs/arena_analysis_roadmap.md stage 3.
 */

export type MoveClass = 'best' | 'good' | 'inaccuracy' | 'mistake' | 'blunder' | 'forced';

export interface ReviewPosition {
  /** White win probability 0..1 for the position before the move. */
  winWhite: number;
  /** Engine best move (UCI) in this position; null when unavailable. */
  bestUci: string | null;
  /** Number of legal moves; 1 marks the played move as forced. */
  legalMoves?: number;
}

export interface ReviewedMove {
  ply: number;
  side: 'w' | 'b';
  san: string;
  uci: string;
  /** White win probability before/after the move. */
  winBefore: number;
  winAfter: number;
  /** Win-probability loss from the mover's perspective, >= 0. */
  moverLoss: number;
  class: MoveClass;
  bestUci: string | null;
  /** Lichess-style per-move accuracy 0..100. */
  accuracy: number;
}

export interface GameReview {
  moves: ReviewedMove[];
  /** Per-side accuracy 0..100 (mean of move accuracies; approximate). */
  accuracy: { white: number; black: number };
  counts: { white: Record<MoveClass, number>; black: Record<MoveClass, number> };
  /** Largest mover losses at or above the inaccuracy threshold, descending. */
  criticalMoves: ReviewedMove[];
}

export interface ReviewInputMove {
  san: string;
  uci: string;
}

const INACCURACY = 0.05;
const MISTAKE = 0.1;
const BLUNDER = 0.2;
const BEST_TOLERANCE = 0.005;

export function classifyMoverLoss(moverLoss: number, playedBest: boolean, forced: boolean): MoveClass {
  if (forced) return 'forced';
  if (playedBest || moverLoss <= BEST_TOLERANCE) return 'best';
  if (moverLoss < INACCURACY) return 'good';
  if (moverLoss < MISTAKE) return 'inaccuracy';
  if (moverLoss < BLUNDER) return 'mistake';
  return 'blunder';
}

/** Lichess per-move accuracy from a mover win-probability loss (0..1). */
export function moveAccuracy(moverLoss: number): number {
  const winDiffPercent = Math.max(0, Math.min(1, moverLoss)) * 100;
  const accuracy = 103.1668 * Math.exp(-0.04354 * winDiffPercent) - 3.1669;
  return Math.max(0, Math.min(100, accuracy));
}

function emptyCounts(): Record<MoveClass, number> {
  return { best: 0, good: 0, inaccuracy: 0, mistake: 0, blunder: 0, forced: 0 };
}

/**
 * Review a mainline: `positions` has one entry per position including the
 * final one (positions.length === moves.length + 1); position i is the board
 * the i-th move was played from. The side to move at position 0 is
 * `startTurn` and alternates from there.
 */
export function reviewGame(positions: readonly ReviewPosition[], moves: readonly ReviewInputMove[], startTurn: 'w' | 'b' = 'w'): GameReview {
  if (positions.length !== moves.length + 1) {
    throw new Error(`review needs one position per move plus the final position (got ${positions.length} positions for ${moves.length} moves)`);
  }
  const reviewed: ReviewedMove[] = moves.map((move, index) => {
    const side: 'w' | 'b' = index % 2 === 0 ? startTurn : startTurn === 'w' ? 'b' : 'w';
    const winBefore = positions[index].winWhite;
    const winAfter = positions[index + 1].winWhite;
    const moverLoss = Math.max(0, side === 'w' ? winBefore - winAfter : winAfter - winBefore);
    const playedBest = positions[index].bestUci !== null && positions[index].bestUci === move.uci;
    const forced = positions[index].legalMoves === 1;
    return {
      ply: index + 1,
      side,
      san: move.san,
      uci: move.uci,
      winBefore,
      winAfter,
      moverLoss,
      class: classifyMoverLoss(moverLoss, playedBest, forced),
      bestUci: positions[index].bestUci,
      accuracy: moveAccuracy(moverLoss),
    };
  });

  const counts = { white: emptyCounts(), black: emptyCounts() };
  const accSums = { w: 0, b: 0 };
  const accCounts = { w: 0, b: 0 };
  for (const move of reviewed) {
    counts[move.side === 'w' ? 'white' : 'black'][move.class] += 1;
    // Forced moves carry no information about the player; skip them in accuracy.
    if (move.class === 'forced') continue;
    accSums[move.side] += move.accuracy;
    accCounts[move.side] += 1;
  }
  const accuracy = {
    white: accCounts.w ? Math.round((accSums.w / accCounts.w) * 10) / 10 : 100,
    black: accCounts.b ? Math.round((accSums.b / accCounts.b) * 10) / 10 : 100,
  };
  const criticalMoves = reviewed
    .filter((move) => move.class === 'inaccuracy' || move.class === 'mistake' || move.class === 'blunder')
    .sort((a, b) => b.moverLoss - a.moverLoss)
    .slice(0, 5);
  return { moves: reviewed, accuracy, counts, criticalMoves };
}

const CLASS_NAG: Partial<Record<MoveClass, string>> = {
  inaccuracy: '$6',
  mistake: '$2',
  blunder: '$4',
};

const CLASS_SUFFIX: Partial<Record<MoveClass, string>> = {
  inaccuracy: '?!',
  mistake: '?',
  blunder: '??',
};

export interface AnnotatedPgnOptions {
  tags?: Record<string, string>;
  result?: string;
  startFullmove?: number;
  startTurn?: 'w' | 'b';
}

/**
 * Standalone annotated PGN for a reviewed mainline: NAGs for inaccuracies/
 * mistakes/blunders plus eval comments (white win %) on every move, and the
 * engine best move noted where the played move lost meaningful ground.
 */
export function annotatedPgn(review: GameReview, options: AnnotatedPgnOptions = {}): string {
  const tags = { Event: 'Game review', ...options.tags };
  const header = Object.entries(tags).map(([key, value]) => `[${key} "${value}"]`).join('\n');
  const startTurn = options.startTurn ?? 'w';
  let fullmove = options.startFullmove ?? 1;
  const tokens: string[] = [];
  review.moves.forEach((move, index) => {
    if (move.side === 'w') tokens.push(`${fullmove}.`);
    else if (index === 0 && startTurn === 'b') tokens.push(`${fullmove}...`);
    tokens.push(`${move.san}${CLASS_SUFFIX[move.class] ?? ''}`);
    const nag = CLASS_NAG[move.class];
    if (nag) tokens.push(nag);
    const parts = [`[%win ${Math.round(move.winAfter * 100)}%]`];
    if (move.bestUci && move.class !== 'best' && move.class !== 'forced' && move.moverLoss >= INACCURACY) {
      parts.push(`best ${move.bestUci}`);
    }
    tokens.push(`{ ${parts.join(' ')} }`);
    if (move.side === 'b') fullmove += 1;
  });
  if (options.result) tokens.push(options.result);
  return `${header}\n\n${tokens.join(' ')}\n`;
}
