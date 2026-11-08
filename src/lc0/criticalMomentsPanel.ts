import { el, htmlEscape, inputEl, selectEl } from './browserDom.ts';
import { createCriticalReport, criticalMoments, criticalReportIsStale, criticalReportSignature, runCriticalReport, type CriticalReport, type CriticalResult, type CriticalMoment } from './criticalMoments.ts';
import { createCriticalEngines } from './criticalMomentsEngines.ts';
import { criticalReportMarkup } from './criticalMomentsPresentation.ts';
import { patchAnalysisResults } from './analysisPresentation.ts';
import type { GameReview } from './gameReview.ts';
import type { EngineResourceBroker } from './resourceBroker.ts';
import type { AnalysisLine } from './analysisFormat.ts';

interface PanelOptions {
  broker: EngineResourceBroker;
  baseline(): { review: GameReview; fens: string[]; signature: string; source: string } | undefined;
  settings(): unknown;
  pauseAnalysis(): void;
  settled(): void;
  navigate(moment: CriticalMoment, target: string, lines: AnalysisLine[]): void;
}

/** DOM-only adapter; the runner and report model do not depend on this panel or the board. */
export function mountCriticalMomentsPanel(options: PanelOptions) {
  let report: CriticalReport | undefined;
  let controller: AbortController | undefined;
  let disposed = false;
  const cache = new Map<string, CriticalResult>();
  const listeners = new AbortController();
  const ownSettings = () => ({ engines: options.settings(), maia: inputEl('criticalMaia').checked, rating: selectEl('criticalRating').value });
  const signature = () => criticalReportSignature(options.baseline()?.signature ?? '', ownSettings());
  // Keep ownership until every aborted lane has settled; a replacement job must not
  // have its broker registrations removed by the old job's finally.
  const isBusy = () => !!controller;
  const render = () => {
    if (disposed) return;
    const baseline = options.baseline();
    el('criticalMomentsArea').hidden = !baseline && !report;
    el('criticalCompare').toggleAttribute('disabled', isBusy() || !baseline || !criticalMoments(baseline.review, baseline.fens).length);
    el('criticalCancel').hidden = !isBusy();
    el('criticalCancel').toggleAttribute('disabled', !isBusy());
    const status = !report ? 'Compare up to five of the largest baseline errors. No extra engine work starts until you confirm.'
      : report.stale ? 'Stale report. The mainline or settings changed. Run a fresh review if needed, then compare again. Old results are shown below with navigation disabled.'
      : isBusy() ? 'Comparing critical moments. You can cancel without losing completed results.'
      : report.state === 'cancelled' ? 'Comparison cancelled. Completed results are retained.' : 'Comparison finished. Any engine failures are listed separately.';
    el('criticalStatus').textContent = baseline && !criticalMoments(baseline.review, baseline.fens).length && !report ? 'No baseline errors to compare at this review depth.' : status;
    if (report) {
      el('criticalSource').textContent = `Baseline severity: ${report.baseline}. Engine move agreement and Maia probabilities are independent of this severity.`;
      el('criticalEngineProgress').innerHTML = report.lanes.map((lane) => `<div><strong>${htmlEscape(lane.provenance.engine)}</strong><span>${lane.completed}/${report!.moments.length} · ${htmlEscape(lane.progress)}</span><progress max="${report!.moments.length}" value="${lane.completed}" aria-label="${htmlEscape(lane.provenance.engine)} progress"></progress></div>`).join('');
      patchAnalysisResults(el('criticalReport'), criticalReportMarkup(report));
    }
  };
  const invalidate = () => {
    if (report) report.stale = true;
    controller?.abort();
    render();
  };
  const sync = () => {
    if (report && criticalReportIsStale(report, signature())) invalidate();
    else render();
    // An open consent is no longer an authorization for a changed game/setup.
    if (!options.baseline()) el('criticalConsent').hidden = true;
  };
  const consentText = () => {
    const count = options.baseline() ? criticalMoments(options.baseline()!.review, options.baseline()!.fens).length : 0;
    el('criticalConsentText').textContent = `This runs Stockfish Lite (depth 12, one CPU thread) and Lc0 Small (400 visits, ONNX WebGPU with WASM fallback) on ${count} pre-move positions, not the whole game. If uncached, allow approximately 2 MB for Stockfish, 21 MB for Lc0 and up to 24 MB for the shared inference runtime.${inputEl('criticalMaia').checked ? ` Maia3 adds approximately 28 MB (46 MB with the fp16 fallback) and one prediction per moment at rating ${selectEl('criticalRating').value}, for both players.` : ''} Downloads are cached where supported. This uses extra CPU/GPU time and battery. Position analysis pauses while it runs.`;
  };
  const on = (id: string, event: string, callback: () => void) => el(id).addEventListener(event, callback, { signal: listeners.signal });
  on('criticalCompare', 'click', () => { consentText(); el('criticalConsent').hidden = false; el('criticalStart').focus(); });
  on('criticalDecline', 'click', () => { el('criticalConsent').hidden = true; el('criticalCompare').focus(); });
  on('criticalCancel', 'click', () => { controller?.abort(); render(); });
  for (const id of ['criticalMaia', 'criticalRating']) on(id, 'change', () => { sync(); consentText(); });
  on('criticalStart', 'click', () => { void (async () => {
    const baseline = options.baseline();
    if (isBusy() || !baseline) return;
    const moments = criticalMoments(baseline.review, baseline.fens);
    if (!moments.length) return;
    const current = new AbortController();
    controller = current;
    const engines = createCriticalEngines(options.broker, inputEl('criticalMaia').checked ? Number(selectEl('criticalRating').value) : undefined, current.signal);
    report = createCriticalReport(signature(), baseline.source, moments, engines.lanes);
    const currentReport = report;
    options.pauseAnalysis();
    el('criticalConsent').hidden = true;
    render();
    try { await runCriticalReport(currentReport, engines.lanes, cache, current.signal, render); }
    finally {
      engines.dispose();
      if (controller === current) controller = undefined;
      render();
      if (!disposed) options.settled();
    }
  })(); });
  el('criticalReport').addEventListener('click', (event) => {
    const button = (event.target as HTMLElement).closest<HTMLButtonElement>('[data-critical-index]');
    if (!button || !report || button.disabled) return;
    if (criticalReportIsStale(report, signature())) { invalidate(); return; }
    const index = Number(button.dataset.criticalIndex);
    const moment = report.moments[index];
    if (!moment) return;
    const lines = report.lanes.flatMap((lane) => {
      const line = report!.cells[lane.id][index].result?.line;
      return line ? [line] : [];
    });
    options.navigate(moment, button.dataset.criticalTarget ?? 'board', lines);
  }, { signal: listeners.signal });
  render();
  return { sync, invalidate, isBusy, dispose: () => { disposed = true; controller?.abort(); listeners.abort(); cache.clear(); } };
}
