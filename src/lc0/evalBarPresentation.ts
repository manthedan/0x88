import { engineMonogram } from './analysisPresentation.ts';
import { htmlEscape } from './browserDom.ts';
import { type EvalScore, evalPercent, evalTickBadgeShift, evalTickFocusKey, evalTooltipShift, groupEvalTicks } from './evalCompare.ts';

/** Pure markup/layout for the evaluation rail; no document, worker or engine state. */
export interface EvalMarker extends EvalScore {
  id: string;
  name: string;
  color: string;
  scoreText: string;
  detail: string;
}
export function evalEngineTag(name: string): string { return engineMonogram(name).replace('L0', 'Lc0'); }

export interface EvalMarkerLayout {
  pixels: number;
  mobile: boolean;
  left: number;
  viewport: number;
  focusedId?: string;
}

type MarkerGroup = ReturnType<typeof groupEvalTicks<EvalMarker>>[number];

function markerLabels(group: MarkerGroup, ordered: readonly EvalMarker[], pixels: number) {
  const shift = group.members.length > 1 ? evalTickBadgeShift(group.percent, pixels, group.selected, 46) : 0;
  const ends = ordered.length > 1 ? [ordered[0], ordered.at(-1)!] : ordered;
  return ends.flatMap((tick, index) => {
    if (!group.members.includes(tick)) return [];
    const center = group.percent * pixels / 100 + shift + (index ? 17 : -17);
    return [{ tick, position: Math.max(7, Math.min(pixels - 7, center)) }];
  });
}

/** Endpoint tags have real opaque bounds too, including singleton/mate labels.
 * Coalesce any labels/counters that would obscure a different group's true tick.
 */
export function evalMarkerGroups(ticks: readonly EvalMarker[], selectedId: string | undefined, layout: EvalMarkerLayout) {
  const valid = ticks.filter((tick) => evalPercent(tick) !== undefined);
  const ordered = [...valid].sort((a, b) => evalPercent(a)! - evalPercent(b)! || a.id.localeCompare(b.id));
  const groups = groupEvalTicks(valid, selectedId, layout.pixels, 6, layout.mobile ? 20 : 46);
  if (layout.mobile) return groups;
  const bounds = (group: MarkerGroup) => {
    const point = group.percent * layout.pixels / 100;
    const half = group.selected ? 3 : 2.5;
    const boxes = [{ min: point - half, max: point + half, decorated: false }];
    if (group.members.length > 1) {
      const center = point + evalTickBadgeShift(group.percent, layout.pixels, group.selected, 46);
      boxes.push({ min: center - 9, max: center + 9, decorated: true });
    }
    for (const label of markerLabels(group, ordered, layout.pixels)) boxes.push({ min: label.position - 6, max: label.position + 6, decorated: true });
    return boxes;
  };
  let merged: boolean;
  do {
    merged = false;
    outer: for (let i = 0; i < groups.length; i++) {
      for (let j = i + 1; j < groups.length; j++) {
        if (!bounds(groups[i]).some((a) => bounds(groups[j]).some((b) => (a.decorated || b.decorated) && a.min <= b.max + 1 && b.min <= a.max + 1))) continue;
        const members = groups.slice(i, j + 1).flatMap((group) => group.members);
        const selected = members.find((member) => member.id === selectedId);
        groups.splice(i, j - i + 1, { members, selected: !!selected,
          percent: selected ? evalPercent(selected)! : members.reduce((sum, member) => sum + evalPercent(member)!, 0) / members.length });
        merged = true;
        break outer;
      }
    }
  } while (merged);
  return groups;
}

export function evalMarkerMarkup(ticks: readonly EvalMarker[], selectedId: string | undefined, layout: EvalMarkerLayout): string {
  const valid = ticks.filter((tick) => evalPercent(tick) !== undefined);
  const ordered = [...valid].sort((a, b) => evalPercent(a)! - evalPercent(b)! || a.id.localeCompare(b.id));
  // Count + two outward endpoint labels fit in 46px on desktop. Phone counters
  // use 20px and open the full value list instead of squeezing labels into them.
  const footprint = layout.mobile ? 20 : 46;
  const groups = evalMarkerGroups(valid, selectedId, layout);
  const markup = groups.map((group) => {
    const details = group.members.map((tick) => `${evalEngineTag(tick.name)} · ${tick.name} ${tick.scoreText}${tick.id === selectedId ? ' (shown on bar)' : ''} · ${tick.detail}`).join('; ');
    const focusKey = evalTickFocusKey(group.members.map((tick) => tick.id), selectedId, layout.focusedId);
    const selected = group.members.find((tick) => tick.id === selectedId);
    const color = selected?.color ?? group.members[0].color;
    const shift = group.members.length > 1 ? evalTickBadgeShift(group.percent, layout.pixels, group.selected, footprint) : 0;
    const anchor = layout.left + layout.pixels * group.percent / 100;
    const hitShift = layout.mobile ? evalTooltipShift(anchor, layout.viewport, 44, 2) : 0;
    const tipShift = layout.mobile ? evalTooltipShift(anchor, layout.viewport, 220) - hitShift : 0;
    const segments = group.members.map((tick) => `<i style="background:${htmlEscape(tick.color)}"></i>`).join('');
    const edges = layout.mobile ? '' : markerLabels(group, ordered, layout.pixels).map(({ tick, position }) =>
      `<span class="eval-edge-tag" style="--edge:${position}px;--edge-color:${htmlEscape(tick.color)}" aria-hidden="true">${htmlEscape(evalEngineTag(tick.name))}</span>`).join('');
    // The actual tick remains at the true score. Only the larger counter/labels
    // shift inwards at endpoints, so mate scores still visibly pin to the end.
    return `<button type="button" class="eval-tick${group.selected ? ' chosen' : ''}${group.members.length > 1 ? ' grouped' : ''}" popovertarget="evalCompareDetails" aria-haspopup="dialog" data-focus-key="${htmlEscape(focusKey)}" style="--pos:${group.percent}%;--tick-color:${htmlEscape(color)};--tip-shift:${tipShift}px;--badge-shift:${shift}px;--hit-shift:${hitShift}px" title="${htmlEscape(details)}" aria-label="${htmlEscape(`${details}. Open engine evaluations`)}"><span class="eval-tick-line" aria-hidden="true">${segments}</span>${group.members.length > 1 ? `<span class="eval-tick-count" aria-hidden="true"><b>${group.members.length}</b><span class="eval-segments">${segments}</span></span>` : ''}<span class="eval-tick-tip" aria-hidden="true">${htmlEscape(details)}</span></button>${edges}`;
  }).join('');
  const selected = valid.find((tick) => tick.id === selectedId);
  return markup + (selected ? `<span class="eval-chosen-mark" aria-hidden="true" style="--pos:${evalPercent(selected)}%;--tick-color:${htmlEscape(selected.color)}"></span>` : '');
}

/** Tap details contain every available current-position value, not just colour keys. */
export function evalDetailsMarkup(ticks: readonly EvalMarker[], selectedId: string | undefined): string {
  if (!ticks.length) return '<p class="small">Waiting for this position’s evaluations.</p>';
  return ticks.map((tick) => `<div class="eval-detail-row"><span class="eval-detail-tag" style="--engine-color:${htmlEscape(tick.color)}" aria-hidden="true">${htmlEscape(evalEngineTag(tick.name))}</span><span><strong>${htmlEscape(tick.name)}</strong><small>${htmlEscape(tick.detail)}${tick.id === selectedId ? ' · shown on bar' : ''}</small></span><b>${htmlEscape(tick.scoreText)}</b></div>`).join('');
}
