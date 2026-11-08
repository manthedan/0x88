import { htmlEscape } from './browserDom.ts';
import { analysisAgreementHeadline, analysisMoveGroups, firstAnalysisMove } from './analysisPresentation.ts';
import { formatScore, type AnalysisLine } from './analysisFormat.ts';
import type { CriticalReport } from './criticalMoments.ts';
import { REVIEW_CLASS_LABEL, reviewMoveLabel } from './gameReviewPresentation.ts';

export function criticalAgreement(lines: AnalysisLine[]): string {
  const best = lines.filter((line) => !line.error && line.pvUci[0]);
  if (best.length < 2) return 'Not enough engine results to compare moves.';
  // Cross-engine scales are not interchangeable. Compare moves, not a synthetic cp average/spread.
  return analysisAgreementHeadline(analysisMoveGroups(best), best.length);
}

export function criticalReportMarkup(report: CriticalReport): string {
  const disabled = report.stale ? ' disabled' : '';
  return report.moments.map((moment, index) => {
    const engineCells = report.lanes.filter((lane) => lane.id !== 'maia').map((lane) => ({ lane, cell: report.cells[lane.id][index] }));
    const lines = engineCells.flatMap(({ cell }) => cell.result?.line ? [cell.result.line] : []);
    const rows = engineCells.map(({ lane, cell }) => {
      const line = cell.result?.line;
      const provenance = cell.result?.provenance ?? lane.provenance;
      const scale = lane.id === 'lc0' ? 'Lc0 Q-mapped pawns' : 'Stockfish pawns';
      const status = line ? `${htmlEscape(firstAnalysisMove(line))} · <strong>${htmlEscape(formatScore(line.scoreCp, line.mateIn))}</strong>`
        : cell.status === 'failed' ? `Failed: ${htmlEscape(cell.error ?? 'Unknown error')}`
        : cell.status === 'cancelled' ? 'Cancelled' : cell.status === 'running' ? 'Running…' : 'Waiting';
      const detail = !line ? '' : lane.id === 'lc0' ? `Best child: ${line.detail}` : line.detail.replace(/^depth \d+( · )?/, '');
      return `<div class="critical-engine"><strong>${htmlEscape(provenance.engine)}</strong><span>${status}</span><small>Budget: ${htmlEscape(provenance.budget)} · ${scale}${cell.cached ? ' · cached' : ''}${detail ? ` · ${htmlEscape(detail)}` : ''}</small></div>`;
    }).join('');
    const human = report.cells.maia?.[index];
    let humanMarkup = '';
    if (human) {
      const result = human.result;
      const percent = (uci: string) => result?.policy?.[uci] === undefined ? 'unavailable' : `${(result.policy[uci] * 100).toFixed(1)}%`;
      const bestMoves = [...new Set(lines.map((line) => line.pvUci[0]))];
      humanMarkup = `<div class="critical-human"><strong>Human plausibility${result ? ` · Maia3 at ${result.rating}` : ' · Maia3'}</strong><p>${result
        ? `Played ${htmlEscape(moment.move.san)}: ${percent(moment.move.uci)}${bestMoves.map((uci) => ` · ${htmlEscape(firstAnalysisMove(lines.find((line) => line.pvUci[0] === uci)!))}: ${percent(uci)}`).join('')}`
        : human.status === 'failed' ? `Failed: ${htmlEscape(human.error ?? '')}` : human.status === 'cancelled' ? 'Cancelled' : 'Waiting for prediction…'}</p></div>`;
    }
    const provenance = report.lanes.flatMap((lane) => {
      const result = report.cells[lane.id][index].result;
      if (!result) return [];
      const p = result.provenance;
      return [`<li><strong>${htmlEscape(p.engine)}</strong>: ${htmlEscape(p.model)} · ${htmlEscape(p.build)} · ${htmlEscape(p.runtime)} · ${htmlEscape(p.budget)} · ${htmlEscape(p.options)}</li>`];
    }).join('');
    return `<article class="critical-moment"><h3><button type="button" data-critical-index="${index}" data-critical-target="board" data-focus-key="critical-${index}-board" title="Show the position before this move"${disabled}>Played ${htmlEscape(reviewMoveLabel(moment.history[0], moment.move))}</button></h3>` +
      `<p class="critical-baseline">Baseline review: <strong>${REVIEW_CLASS_LABEL[moment.move.class]}</strong> · −${Math.round(moment.move.moverLoss * 100)} percentage points</p>` +
      `<p class="critical-agreement">${htmlEscape(criticalAgreement(lines))}</p>${rows}${humanMarkup}` +
      `<div class="critical-links"><button type="button" data-critical-index="${index}" data-critical-target="comparison" data-focus-key="critical-${index}-compare"${disabled}>Compare</button><button type="button" data-critical-index="${index}" data-critical-target="human" data-focus-key="critical-${index}-human"${disabled}>Human</button><button type="button" data-critical-index="${index}" data-critical-target="play" data-focus-key="critical-${index}-play"${disabled}>Play from here</button></div>` +
      (provenance ? `<details class="critical-provenance" data-result-key="critical-${index}-provenance"><summary>Result provenance</summary><ul>${provenance}</ul><p>Known history: ${moment.history.length} positions, retained in each result’s cache identity. All engine evaluations are from White’s perspective.</p></details>` : '') + '</article>';
  }).join('');
}
