import { evalBarWhitePercent } from './analysisFormat.ts';

/** All inputs and positions on the track are from White's perspective. */
export interface EvalScore {
  scoreCp?: number;
  mateIn?: number;
}

// Percentage points on the existing logistic bar, NOT calibrated engine WDL.
// +0.20 vs +1.80 spans 15.3 points; +5 vs +6 spans only 3.6.
export const MEANINGFUL_SPREAD = 12;
export const SIDE_THRESHOLD = 5;

/** Missing / non-finite results must not masquerade as equality. Mate wins over cp. */
export function evalPercent(score: EvalScore): number | undefined {
  if (score.mateIn !== undefined && Number.isFinite(score.mateIn)) return evalBarWhitePercent(undefined, score.mateIn);
  if (score.scoreCp !== undefined && Number.isFinite(score.scoreCp)) return evalBarWhitePercent(score.scoreCp, undefined);
  return undefined;
}

export function compareEvals(scores: readonly EvalScore[]) {
  const results = scores.flatMap((score) => {
    const percent = evalPercent(score);
    return percent === undefined ? [] : [{ score, percent }];
  });
  if (results.length < 2) return undefined;
  const min = Math.min(...results.map((entry) => entry.percent));
  const max = Math.max(...results.map((entry) => entry.percent));
  const spread = max - min;
  const oppositeSides = min <= 50 - SIDE_THRESHOLD && max >= 50 + SIDE_THRESHOLD;
  const allCp = results.every(({ score }) => score.mateIn === undefined && Number.isFinite(score.scoreCp));
  const pawnSpread = allCp
    ? (Math.max(...results.map(({ score }) => score.scoreCp!)) - Math.min(...results.map(({ score }) => score.scoreCp!))) / 100
    : undefined;
  return {
    min,
    max,
    spread,
    pawnSpread,
    oppositeSides,
    meaningful: spread >= MEANINGFUL_SPREAD,
    // Never subtract mate distances as though they were pawns.
    label: pawnSpread === undefined ? `${spread.toFixed(0)} points on bar` : `${pawnSpread.toFixed(2)} pawns`,
  };
}

/** Zero-based position ply, including custom starting positions. */
export function fenPly(fen: string): number | undefined {
  const fields = fen.trim().split(/\s+/);
  const fullmove = Number(fields[5]);
  if (!['w', 'b'].includes(fields[1]) || !Number.isSafeInteger(fullmove) || fullmove < 1) return undefined;
  return (fullmove - 1) * 2 + (fields[1] === 'b' ? 1 : 0);
}

function position(fen: string): string {
  return fen.trim().split(/\s+/).slice(0, 4).join(' ');
}

export function searchAge(fen: string | undefined, currentFen: string): number | undefined {
  if (!fen) return undefined;
  const searched = fenPly(fen);
  const current = fenPly(currentFen);
  if (searched === undefined || current === undefined) return undefined;
  // Same move number on a different position is not a current-position result.
  if (searched === current && position(fen) !== position(currentFen)) return undefined;
  return current - searched;
}

export function searchAgeLabel(age: number | undefined): string {
  if (age === undefined) return 'No search for this position';
  if (age === 0) return 'Current position';
  if (age < 0) return `${-age} ${age === -1 ? 'ply' : 'plies'} later than the board`;
  return `${age} ${age === 1 ? 'ply' : 'plies'} ago`;
}

/**
 * Alternating Arena searches are not simultaneous opinions. We allow an explicitly
 * labelled "adjacent searches" comparison only when both are at most TWO plies
 * behind the displayed board and at most ONE ply apart (the normal post-move
 * ages are 1 and 2). Older, future/review, missing and unrelated results never
 * earn a delta or disagreement emphasis. Same-ply results must match positions.
 * Full strength belongs to the current/most recent search, never an old pair.
 */
export function compareArenaEvals(scores: readonly (EvalScore & { fen?: string })[], currentFen: string) {
  const ages = scores.map((score) => (evalPercent(score) === undefined ? undefined : searchAge(score.fen, currentFen)));
  const recent = ages.filter((age): age is number => age !== undefined && age >= 0 && age <= 2);
  const freshest = recent.length ? Math.min(...recent) : undefined;
  const fresh = ages.map((age) => age !== undefined && age === freshest);
  const fair =
    scores.length === 2 &&
    ages.every((age) => age !== undefined && age >= 0 && age <= 2) &&
    Math.abs(ages[0]! - ages[1]!) <= 1 &&
    (ages[0] !== ages[1] || position(scores[0].fen!) === position(scores[1].fen!));
  // Only a completed, ineligible pair gets the old-lane treatment. An ordinary
  // alternating search (or a missing first result) must retain full contrast.
  // Equally aged but unrelated positions have no authoritative lane: dim both.
  const complete = scores.every((score) => evalPercent(score) !== undefined);
  const stale = ages.map((age, index) => !fair && complete && (!fresh[index] || ages.every((other) => other === age)));
  return { ages, fresh, stale, fair, adjacent: fair && ages[0] !== ages[1], comparison: fair ? compareEvals(scores) : undefined };
}

/** Compact text retains the unit for mate comparisons; a mate distance is not pawns. */
export function compactEvalSpread(comparison: NonNullable<ReturnType<typeof compareEvals>>): string {
  return comparison.pawnSpread === undefined ? `${comparison.spread.toFixed(0)} pts` : comparison.pawnSpread.toFixed(2);
}

/** Visible verdicts use the same eligibility/scale rules as the bar, including pending results. */
export function arenaComparisonStatus(state: ReturnType<typeof compareArenaEvals>, available: readonly boolean[], names: readonly string[]) {
  if (!available.some(Boolean)) return { text: 'Waiting for first search', emphasis: false };
  if (available.some((ready) => !ready)) return { text: `Waiting for ${available[0] ? 'Black' : 'White'}’s search`, emphasis: false };
  if (!state.comparison) {
    const old = state.ages.findIndex((age) => age !== undefined && age > 2);
    const age = old >= 0 ? state.ages[old] : undefined;
    return { text: age !== undefined ? `not comparable, ${names[old]}’s eval is ${age} plies old` : 'not comparable', emphasis: false };
  }
  const comparison = state.comparison;
  if (comparison.oppositeSides) return { text: 'disagree on who’s better', emphasis: true };
  if (comparison.meaningful) return { text: `${compactEvalSpread(comparison)} apart`, emphasis: true };
  return { text: state.adjacent ? 'similar, one move apart' : 'similar, same position', emphasis: false };
}

/**
 * Arena may use one engine ID for both seats. Retain each side's own last
 * completed search instead of borrowing the other side's shared-map snapshot.
 * A missing latest output resets the lane: Arena clears its output map before
 * rendering a new game (even when both seats keep the same IDs).
 */
export function arenaLaneOutput<T extends { engineId: string; fen: string }>(
  previous: T | undefined,
  latest: T | undefined,
  engineId: string | null,
  side: 'w' | 'b',
): T | undefined {
  if (!engineId || !latest) return undefined;
  if (latest.engineId === engineId && latest.fen.split(/\s+/)[1] === side) return latest;
  return previous?.engineId === engineId && previous.fen.split(/\s+/)[1] === side ? previous : undefined;
}
