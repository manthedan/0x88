import { parseFen } from '../chess/board.ts';
import { moveToUci } from '../chess/moveCodec.ts';
import { legalMoves } from '../chess/movegen.ts';
import { criticalReviewPlies, type GameReview, type ReviewedMove } from './gameReview.ts';
import { GameTree, type GameNode } from './gameTree.ts';

/** Maximum mover-relative expected-score loss against a fresh best-move check. */
export const RETRY_SCORE_TOLERANCE = 0.02;
export type RetrySide = 'w' | 'b';
export type RetryPhase = 'attempt' | 'checking' | 'wrong' | 'solved' | 'revealed' | 'complete';
export interface RetryProblem { move: ReviewedMove; before: GameNode }
export type RetryEvaluator = (tree: GameTree, signal: AbortSignal) => Promise<number>;

/** PGN names alone do not identify the user. Only explicit human/user tags do. */
export function retrySideFromTags(tags: Record<string, string>): RetrySide | undefined {
  const tag = (name: string) => Object.entries(tags).find(([key]) => key.toLowerCase() === name.toLowerCase())?.[1]?.toLowerCase();
  const explicit = tag('PlayerColor') ?? tag('UserColor');
  if (explicit === 'white' || explicit === 'w') return 'w';
  if (explicit === 'black' || explicit === 'b') return 'b';
  const white = ['human', 'you'].includes(tag('WhiteType') ?? '') || tag('White') === 'you';
  const black = ['human', 'you'].includes(tag('BlackType') ?? '') || tag('Black') === 'you';
  return white !== black ? white ? 'w' : 'b' : undefined;
}

export function retryProblems(review: GameReview, nodes: readonly GameNode[], side: RetrySide): RetryProblem[] {
  // Filter before limiting: the other side's errors must not consume this side's five slots.
  const eligible = { ...review, moves: review.moves.filter((move) => move.side === side && (move.class === 'mistake' || move.class === 'blunder')) };
  return criticalReviewPlies(eligible, eligible.moves.length).flatMap((ply) => {
    const before = nodes[ply - 1];
    const move = eligible.moves.find((entry) => entry.ply === ply)!;
    if (!before || !move.bestUci) return [];
    const legal = legalMoves(parseFen(before.fen));
    if (legal.length <= 1 || !legal.some((candidate) => moveToUci(candidate) === move.bestUci)) return [];
    return [{ move, before }];
  }).slice(0, 5);
}

/** Replay history into a completely independent tree, without touching even original current. */
export function retryTreeAt(source: GameTree, node: GameNode): GameTree {
  const copy = new GameTree(source.root.fen);
  for (const entry of source.lineTo(node).slice(1)) {
    if (!entry.move || !copy.addUci(moveToUci(entry.move))) throw new Error('Cannot replay retry position.');
  }
  return copy;
}

export function acceptableRetryScore(side: RetrySide, bestWhite: number, attemptWhite: number): boolean {
  if (![bestWhite, attemptWhite].every((score) => Number.isFinite(score) && score >= 0 && score <= 1)) throw new Error('Verification needs a scored result.');
  const loss = side === 'w' ? bestWhite - attemptWhite : attemptWhite - bestWhite;
  return loss <= RETRY_SCORE_TOLERANCE + 1e-9;
}

export class MistakeRetry {
  readonly source: GameTree;
  readonly problems: RetryProblem[];
  index = 0;
  solved = 0;
  phase: RetryPhase = 'attempt';
  tree: GameTree;
  attemptedUci: string | null = null;
  private controller: AbortController | null = null;

  constructor(source: GameTree, problems: RetryProblem[]) {
    if (!problems.length) throw new Error('No retry positions for this side.');
    this.source = source;
    this.problems = problems;
    this.tree = retryTreeAt(source, problems[0].before);
  }
  get problem(): RetryProblem { return this.problems[this.index]; }
  get isLast(): boolean { return this.index === this.problems.length - 1; }
  get answered(): boolean { return this.phase === 'solved' || this.phase === 'revealed'; }
  get remaining(): number { return Math.max(0, this.problems.length - this.index - (this.answered ? 1 : 0)); }
  cancel(): void { this.controller?.abort(); this.controller = null; }
  reveal(): void { this.cancel(); this.phase = 'revealed'; }
  next(): void {
    this.cancel();
    this.index += 1;
    this.attemptedUci = null;
    if (this.index === this.problems.length) { this.phase = 'complete'; return; }
    this.tree = retryTreeAt(this.source, this.problem.before);
    this.phase = 'attempt';
  }
  async attempt(uci: string, evaluate: RetryEvaluator): Promise<void> {
    if (this.answered || this.phase === 'checking' || this.phase === 'complete') return;
    const candidate = retryTreeAt(this.source, this.problem.before);
    if (!candidate.addUci(uci)) return;
    this.attemptedUci = uci;
    if (uci === this.problem.move.bestUci) { this.phase = 'solved'; this.solved += 1; return; }
    const best = retryTreeAt(this.source, this.problem.before);
    best.addUci(this.problem.move.bestUci!);
    const controller = new AbortController();
    this.controller = controller;
    this.phase = 'checking';
    try {
      // Sequential checks share one engine and the same depth, history and settings.
      const bestWhite = await evaluate(best, controller.signal);
      if (controller.signal.aborted) return;
      const attemptWhite = await evaluate(candidate, controller.signal);
      if (controller.signal.aborted) return;
      const accepted = acceptableRetryScore(this.problem.move.side, bestWhite, attemptWhite);
      this.phase = accepted ? 'solved' : 'wrong';
      if (accepted) this.solved += 1;
    } catch (error) {
      if (!controller.signal.aborted) { this.phase = 'attempt'; throw error; }
    } finally {
      if (this.controller === controller) this.controller = null;
    }
  }
}
