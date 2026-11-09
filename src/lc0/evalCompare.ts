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
// Includes the outlined markers' footprint, especially on the short mobile track.
export const TICK_GROUP_PX = 6;

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
  const pawnSpread = allCp ? (Math.max(...results.map(({ score }) => score.scoreCp!)) - Math.min(...results.map(({ score }) => score.scoreCp!))) / 100 : undefined;
  return { min, max, spread, oppositeSides, meaningful: spread >= MEANINGFUL_SPREAD,
    // Never subtract mate distances as though they were pawns.
    label: pawnSpread === undefined ? `${spread.toFixed(0)} points on bar` : `${pawnSpread.toFixed(2)} pawns` };
}

/** Zero-based position ply, including custom starting positions. */
export function fenPly(fen: string): number | undefined {
  const fields = fen.trim().split(/\s+/);
  const fullmove = Number(fields[5]);
  if (!['w', 'b'].includes(fields[1]) || !Number.isSafeInteger(fullmove) || fullmove < 1) return undefined;
  return (fullmove - 1) * 2 + (fields[1] === 'b' ? 1 : 0);
}

function position(fen: string): string { return fen.trim().split(/\s+/).slice(0, 4).join(' '); }

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
  const ages = scores.map((score) => evalPercent(score) === undefined ? undefined : searchAge(score.fen, currentFen));
  const recent = ages.filter((age): age is number => age !== undefined && age >= 0 && age <= 2);
  const freshest = recent.length ? Math.min(...recent) : undefined;
  const fresh = ages.map((age) => age !== undefined && age === freshest);
  const fair = scores.length === 2 && ages.every((age) => age !== undefined && age >= 0 && age <= 2)
    && Math.abs(ages[0]! - ages[1]!) <= 1
    && (ages[0] !== ages[1] || position(scores[0].fen!) === position(scores[1].fen!));
  return { ages, fresh, fair, adjacent: fair && ages[0] !== ages[1], comparison: fair ? compareEvals(scores) : undefined };
}

export interface EvalTick extends EvalScore { id: string }

/**
 * Cluster nearby ticks by diameter, then coalesce colliding count badges. A badge
 * is 13px plus its outline; treating it as a 2px tick would create stacked counts.
 * The chosen engine anchors its group; other groups use their mean position.
 */
export function groupEvalTicks<T extends EvalTick>(scores: readonly T[], selectedId: string | undefined, trackPixels: number, tolerance = TICK_GROUP_PX) {
  const sorted = scores.flatMap((score) => {
    const percent = evalPercent(score);
    return percent === undefined ? [] : [{ score, percent }];
  }).sort((a, b) => a.percent - b.percent || a.score.id.localeCompare(b.score.id));
  const groups: { members: T[]; percent: number; selected: boolean }[] = [];
  const pixels = Math.max(1, trackPixels);
  for (let i = 0; i < sorted.length;) {
    const start = i++;
    while (i < sorted.length && (sorted[i].percent - sorted[start].percent) * pixels / 100 <= tolerance) i++;
    const entries = sorted.slice(start, i);
    const selected = entries.find(({ score }) => score.id === selectedId);
    groups.push({ members: entries.map(({ score }) => score), selected: !!selected,
      percent: selected?.percent ?? entries.reduce((sum, entry) => sum + entry.percent, 0) / entries.length });
  }
  const footprint = (group: typeof groups[number]) => group.members.length > 1 ? 15 : group.selected ? 8 : 6;
  for (let i = 1; i < groups.length;) {
    const a = groups[i - 1];
    const b = groups[i];
    const gap = (b.percent - a.percent) * pixels / 100;
    // Plain ticks were already grouped by the requested tolerance. Only badges
    // need extra room; don't chain a row of otherwise distinct single ticks.
    if ((a.members.length > 1 || b.members.length > 1) && gap < (footprint(a) + footprint(b)) / 2 + 2) {
      const members = [...a.members, ...b.members];
      const selected = members.find((member) => member.id === selectedId);
      groups.splice(i - 1, 2, { members, selected: !!selected,
        percent: selected ? evalPercent(selected)! : members.reduce((sum, member) => sum + evalPercent(member)!, 0) / members.length });
      i = Math.max(1, i - 1);
    } else i++;
  }
  return groups;
}
