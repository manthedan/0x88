import { engineMonogram } from './analysisPresentation.ts';
import { htmlEscape } from './browserDom.ts';
import { compareEvals, type EvalScore, evalPercent } from './evalCompare.ts';

/** Pure markup for the evaluation rail; overlapping ticks are deliberately not clustered. */
export interface EvalMarker extends EvalScore {
  id: string;
  name: string;
  color: string;
  scoreText: string;
}
export const evalEngineTag = engineMonogram;

export function evalMarkerMarkup(ticks: readonly EvalMarker[], selectedId: string | undefined, layout: { pixels: number; mobile: boolean }): string {
  const ordered = ticks.filter((tick) => evalPercent(tick) !== undefined).sort((a, b) => evalPercent(a)! - evalPercent(b)! || a.id.localeCompare(b.id));
  const selected = ordered.find((tick) => tick.id === selectedId);
  const comparison = compareEvals(ordered);
  const label = (tick: EvalMarker, chosen = false) =>
    `<span class="eval-engine-label${chosen ? ' selected' : ''}" style="--pos:${evalPercent(tick)}%;--tick-color:${htmlEscape(tick.color)}" aria-hidden="true">${htmlEscape(evalEngineTag(tick.name))}</span>`;
  // On phones the source chip already carries the chosen engine's identity.
  const labels = selected && !layout.mobile ? [label(selected, true)] : [];
  // Only range ends get secondary text, and only with a full 14px of clear space
  // between labels. No displacement: every label stays at its actual value.
  if (comparison?.meaningful && selected) {
    const labelled = [evalPercent(selected)!];
    for (const tick of [ordered[0], ordered.at(-1)!]) {
      const percent = evalPercent(tick)!;
      const clearance = layout.mobile ? 44 : 26;
      if (tick.id !== selectedId && labelled.every((other) => (Math.abs(percent - other) * layout.pixels) / 100 >= clearance)) {
        labels.push(label(tick));
        labelled.push(percent);
      }
    }
  }
  const lines = ordered
    .map((tick) => `<span class="eval-tick" aria-hidden="true" style="--pos:${evalPercent(tick)}%;--tick-color:${htmlEscape(tick.color)}"></span>`)
    .join('');
  // Draw the full-width selected notch last, at the fill edge, above every tick.
  return (
    lines +
    labels.join('') +
    (selected
      ? `<span class="eval-chosen-mark" aria-hidden="true" style="--pos:${evalPercent(selected)}%;--tick-color:${htmlEscape(selected.color)}"></span>`
      : '')
  );
}

/** One compact list for the whole bar, ordered from Black-favouring to White-favouring. */
export function evalDetailsMarkup(ticks: readonly EvalMarker[], selectedId: string | undefined): string {
  const ordered = ticks.filter((tick) => evalPercent(tick) !== undefined).sort((a, b) => evalPercent(a)! - evalPercent(b)! || a.id.localeCompare(b.id));
  if (!ordered.length) return '<p class="small">Waiting for this position’s evaluations.</p>';
  return ordered
    .map(
      (tick) =>
        `<div class="eval-detail-row"><span class="eval-detail-tag" style="--engine-color:${htmlEscape(tick.color)}" aria-hidden="true">${htmlEscape(evalEngineTag(tick.name))}</span><span><strong>${htmlEscape(tick.name)}</strong>${tick.id === selectedId ? '<small>Shown on bar</small>' : ''}</span><b>${htmlEscape(tick.scoreText)}</b></div>`,
    )
    .join('');
}
