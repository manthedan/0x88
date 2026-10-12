/**
 * Pure summary for the low-end memory harness.
 *
 * Bytes come from performance.measureUserAgentSpecificMemory(). That API waits
 * for a garbage collection, so latency is part of the sample and is not engine
 * time. A resident is treated as unreleased when the peak it caused is still
 * mostly present at a later checkpoint in the same browser session.
 */

export const SCHEMA = 'lowend-memory-harness/v1';
export const NOISE_BYTES = 8 * 1024 * 1024;
export const RETAIN_FRACTION = 0.5;

export const CPU_PLAY_ENGINE_IDS = Object.freeze(['sf-lite', 'sf-full', 'reckless', 'viridithas', 'berserk', 'plentychess', 'stormphrax']);
export const CPU_ANALYSIS_FAMILIES = Object.freeze(['sf', 'reckless', 'viridithas', 'berserk', 'plentychess', 'stormphrax']);
export const DEFAULT_PLAY_ENGINE_ID = 'maia3';
export const NEURAL_PLAY_ENGINE_ID = 'lc0-small';

const PHASES = ['play', 'analysis', 'arena', 'maia'];

function checkpoint(id, phase, surface, resident = null) {
  return { id, phase, surface, resident };
}

export function checkpointPlan() {
  const play = [
    checkpoint('landing', 'play', 'landing'),
    checkpoint('play-default-after-first-move', 'play', 'play', DEFAULT_PLAY_ENGINE_ID),
    ...CPU_PLAY_ENGINE_IDS.map((id) => checkpoint(`play-cpu-${id}`, 'play', 'play', id)),
    checkpoint(`play-neural-${NEURAL_PLAY_ENGINE_ID}`, 'play', 'play', NEURAL_PLAY_ENGINE_ID),
    checkpoint('landing-after-play', 'play', 'landing'),
  ];
  const analysis = [
    checkpoint('analysis-shell', 'analysis', 'analysis'),
    checkpoint('analysis-lc0-30s', 'analysis', 'analysis', 'lc0'),
    checkpoint('analysis-lc0-stopped', 'analysis', 'analysis', 'lc0'),
    ...CPU_ANALYSIS_FAMILIES.map((family) => checkpoint(`analysis-cpu-${family}`, 'analysis', 'analysis', family)),
    checkpoint('analysis-after-cpu', 'analysis', 'analysis', 'lc0'),
  ];
  const arena = [
    checkpoint('arena-shell', 'arena', 'arena'),
    checkpoint('arena-cpu-match', 'arena', 'arena', 'sf+reckless'),
    checkpoint('arena-cpu-stopped', 'arena', 'arena'),
  ];
  const maia = [
    checkpoint('maia-before', 'maia', 'analysis'),
    checkpoint('maia-panel', 'maia', 'analysis', 'maia3'),
    checkpoint('landing-after-maia', 'maia', 'landing'),
  ];
  return {
    play,
    analysis,
    arena,
    maia,
    all: [...play, ...analysis, ...arena, ...maia],
  };
}

export function retentionComparisons(plan = checkpointPlan()) {
  const ids = (phase, prefix) => plan[phase].map((entry) => entry.id).filter((id) => !prefix || id.startsWith(prefix));
  return [
    {
      id: 'play-cpu-tour',
      baselineId: 'play-default-after-first-move',
      afterId: `play-neural-${NEURAL_PLAY_ENGINE_ID}`,
      duringIds: ids('play', 'play-cpu-'),
      meaning: 'CPU engines switched on Play, then a neural engine (Lc0 small) replaced the selection',
    },
    {
      id: 'play-left-surface',
      baselineId: 'landing',
      afterId: 'landing-after-play',
      duringIds: ids('play', 'play-'),
      meaning: 'Play was left via in-app navigation back to the landing page',
    },
    {
      id: 'analysis-lc0-stop',
      baselineId: 'analysis-shell',
      afterId: 'analysis-lc0-stopped',
      duringIds: ['analysis-lc0-30s'],
      meaning: 'Analysis Stop after Lc0 had been running',
    },
    {
      id: 'analysis-cpu-tour',
      baselineId: 'analysis-lc0-stopped',
      afterId: 'analysis-after-cpu',
      duringIds: ids('analysis', 'analysis-cpu-'),
      meaning: 'Each CPU family was analyzed, then the row was switched back to Lc0',
    },
    {
      id: 'arena-stop',
      baselineId: 'arena-shell',
      afterId: 'arena-cpu-stopped',
      duringIds: ['arena-cpu-match'],
      meaning: 'A short Stockfish Lite vs Reckless match was stopped',
    },
    {
      id: 'maia-left-surface',
      baselineId: 'maia-before',
      afterId: 'landing-after-maia',
      duringIds: ['maia-panel'],
      meaning: 'The Maia3 panel was loaded, then the app navigated back to landing',
    },
  ];
}

/**
 * True once Analysis has rendered an engine row. The Maia button is static
 * markup; its listener is attached with those rows. Playwright serializes
 * only this function.
 */
export function analysisControllerReady(state) {
  const input = state ?? {};
  let hasRows = input.hasRows;
  let hasMaiaButton = input.hasMaiaButton;
  if ((hasRows == null || hasMaiaButton == null) && typeof document !== 'undefined') {
    if (hasRows == null) hasRows = !!document.querySelector('.row-fam');
    if (hasMaiaButton == null) hasMaiaButton = !!document.querySelector('#maia3Enable');
  }
  return { ok: hasRows === true && hasMaiaButton === true };
}

/**
 * Opens the Maia panel and clicks Load when the click has not taken effect.
 * A click before the controller is wired leaves the button enabled and the
 * status empty, so the caller can poll this until ok. Playwright serializes
 * only this function.
 */
export function requestMaiaLoad() {
  const details = document.querySelector('details.human-moves');
  if (details instanceof HTMLDetailsElement) details.open = true;
  const button = document.querySelector('#maia3Enable');
  const grid = document.querySelector('#maia3Grid');
  const read = () => {
    const status = document.querySelector('#maia3PanelStatus')?.textContent ?? '';
    const current = document.querySelector('#maia3Enable');
    const engaged =
      (current instanceof HTMLButtonElement && (current.disabled || current.hidden)) ||
      /loading|download|failed|error/i.test(status) ||
      (!!grid && grid.hidden === false);
    return { ok: engaged, status };
  };
  const before = read();
  if (!before.ok && button instanceof HTMLButtonElement) button.click();
  return read();
}

/**
 * True when this analysis run has left the start banner.
 *
 * Playwright passes this function to page.evaluate, which sends only its
 * source. Family patterns stay inside the function. The "Analyzing (…)" banner
 * is set before the worker loads, so it is not ready. A progress or completion
 * line must name the selected family. A changed failure line is ready even
 * when the error text omits the family, including "Model load failed" and
 * "LC0 … load failed".
 */
export function analysisSampleReady(state) {
  const patterns = {
    lc0: /lc0|leela/i,
    sf: /\bsf\b|stockfish/i,
    reckless: /reckless/i,
    viridithas: /viridithas/i,
    berserk: /berserk/i,
    plentychess: /plenty/i,
    stormphrax: /stormphrax/i,
  };
  const family = String(state?.family ?? '');
  let message = state?.message;
  let progress = state?.progress;
  if ((message == null || progress == null) && typeof document !== 'undefined') {
    if (message == null) message = document.querySelector('#message')?.textContent ?? '';
    if (progress == null) progress = document.querySelector('#analysisSearchProgress')?.textContent ?? '';
  }
  message = String(message ?? '');
  progress = String(progress ?? '');
  const previous = state?.previousMessage == null ? null : String(state.previousMessage);
  const known = patterns[family];
  const pattern = known ?? new RegExp(family.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'i');
  const settled = /analysis complete|\d+\s*\/\s*\d+\s*(visits|nodes|search)|:\s*searching/i;
  const failedPattern = /analysis failed|engine failures|could not download|asset missing|unavailable|model load failed|\bload failed\b/i;
  const namesSettled = (text) => pattern.test(text) && settled.test(text);
  const changed = previous == null || message !== previous;
  const failed = failedPattern.test(message) || failedPattern.test(progress);
  return {
    ok: changed && (namesSettled(message) || namesSettled(progress) || failed),
    failed,
    message,
    progress,
    family,
  };
}

/** True when Lc0 has reported a visit count and the timed window may start. */
export function lc0TimedRunStarted(ready) {
  if (!ready?.ok || ready.failed) return false;
  const message = String(ready.message ?? '');
  const progress = String(ready.progress ?? '');
  if (/analysis complete|analysis failed|engine failures|stopped/i.test(message)) return false;
  const text = `${message}\n${progress}`;
  return /lc0|leela/i.test(text) && /\d+\s*\/\s*\d+\s*visits/i.test(text);
}

/**
 * True while the timed Lc0 search is still producing visit counts.
 * The start banner and an enabled Stop button do not count. Playwright
 * serializes only this function.
 */
export function analysisLc0StillRunning(state) {
  const input = state ?? {};
  let message = input.message;
  let progress = input.progress;
  let stopEnabled = input.stopEnabled;
  if ((message == null || progress == null || stopEnabled == null) && typeof document !== 'undefined') {
    if (message == null) message = document.querySelector('#message')?.textContent ?? '';
    if (progress == null) progress = document.querySelector('#analysisSearchProgress')?.textContent ?? '';
    if (stopEnabled == null) {
      const stop = document.querySelector('#stop');
      stopEnabled = !!stop && !stop.disabled;
    }
  }
  message = String(message ?? '');
  progress = String(progress ?? '');
  const text = `${message}\n${progress}`;
  const failed = /analysis failed|engine failures|could not download|asset missing|unavailable|model load failed|\bload failed\b/i.test(message);
  const ended = failed || /analysis complete|stopped/i.test(message);
  const live = /lc0|leela/i.test(text) && /\d+\s*\/\s*\d+\s*visits/i.test(text) && !/complete|stopped|failed/i.test(text);
  return { running: stopEnabled === true && live && !ended, failed, message };
}

/** Failure metadata for a checkpoint whose wait treated an error string as done. */
export function checkpointEngineFailure(message) {
  const text = String(message ?? '');
  const failed = /failed|error|unavailable|could not download|asset missing|engine failures/i.test(text);
  return failed ? { engineFailed: true, engineError: text } : { engineFailed: false, engineError: undefined };
}

/**
 * The memory call can outlast a search. pageStatus() is read after that call,
 * so a failure that lands during the wait has to be copied onto the sample.
 */
export function applyObservedEngineFailure(reading) {
  if (!reading || reading.engineFailed) return reading;
  const observed = checkpointEngineFailure(reading.status?.analysisMessage);
  if (!observed.engineFailed) return reading;
  const suffix = `failed during sample: ${observed.engineError}`;
  return {
    ...reading,
    engineFailed: true,
    engineError: observed.engineError,
    note: reading.note ? `${reading.note} (${suffix})` : suffix,
  };
}

export function previousCheckpointId(id, plan = checkpointPlan()) {
  for (const phase of PHASES) {
    const ids = plan[phase].map((entry) => entry.id);
    const index = ids.indexOf(id);
    if (index > 0) return ids[index - 1];
    if (index === 0) return null;
  }
  return null;
}

export function median(values) {
  const sorted = values.filter((value) => Number.isFinite(value)).sort((a, b) => a - b);
  if (!sorted.length) return null;
  const mid = Math.floor(sorted.length / 2);
  if (sorted.length % 2 === 1) return sorted[mid];
  return (sorted[mid - 1] + sorted[mid]) / 2;
}

export function spread(values) {
  const sorted = values.filter((value) => Number.isFinite(value)).sort((a, b) => a - b);
  if (!sorted.length) return { min: null, max: null, range: null };
  const min = sorted[0];
  const max = sorted[sorted.length - 1];
  return { min, max, range: max - min };
}

export function workerCount(sample) {
  const live = sample?.workers?.live;
  if (Number.isInteger(live)) return live;
  const scopes = sample?.workers?.breakdownScopes;
  if (Number.isInteger(scopes)) return scopes;
  return null;
}

export function formatMib(bytes) {
  if (!Number.isFinite(bytes)) return 'n/a';
  const sign = bytes < 0 ? '-' : '';
  return `${sign}${(Math.abs(bytes) / (1024 * 1024)).toFixed(1)} MiB`;
}

export function compareRetention(samples, spec, { noiseBytes = NOISE_BYTES, retainFraction = RETAIN_FRACTION } = {}) {
  const find = (id) => samples.find((sample) => sample.id === id && sample.ok !== false && Number.isFinite(sample.userAgentSpecificBytes));
  const baseline = find(spec.baselineId);
  const after = find(spec.afterId);
  const during = spec.duringIds.map(find).filter(Boolean);
  if (!baseline || !after || !during.length) {
    return {
      id: spec.id,
      baselineId: spec.baselineId,
      afterId: spec.afterId,
      meaning: spec.meaning,
      missing: true,
      unreleased: false,
      keptBytes: null,
      gainedBytes: null,
      releasedBytes: null,
      baselineBytes: baseline?.userAgentSpecificBytes ?? null,
      afterBytes: after?.userAgentSpecificBytes ?? null,
      peakBytes: null,
      peakId: null,
    };
  }
  const peakSample = during.reduce((best, sample) => (sample.userAgentSpecificBytes > best.userAgentSpecificBytes ? sample : best));
  const gainedBytes = peakSample.userAgentSpecificBytes - baseline.userAgentSpecificBytes;
  const keptBytes = after.userAgentSpecificBytes - baseline.userAgentSpecificBytes;
  const releasedBytes = peakSample.userAgentSpecificBytes - after.userAgentSpecificBytes;
  const unreleased = gainedBytes > noiseBytes && keptBytes > Math.max(noiseBytes, gainedBytes * retainFraction);
  return {
    id: spec.id,
    baselineId: spec.baselineId,
    afterId: spec.afterId,
    meaning: spec.meaning,
    missing: false,
    unreleased,
    keptBytes,
    gainedBytes,
    releasedBytes,
    baselineBytes: baseline.userAgentSpecificBytes,
    afterBytes: after.userAgentSpecificBytes,
    peakBytes: peakSample.userAgentSpecificBytes,
    peakId: peakSample.id,
  };
}

export function aggregateRetention(perRunFindings) {
  const runCount = perRunFindings.length;
  const grouped = new Map();
  for (const findings of perRunFindings) {
    for (const finding of findings) {
      let row = grouped.get(finding.id);
      if (!row) {
        row = {
          id: finding.id,
          baselineId: finding.baselineId,
          afterId: finding.afterId,
          meaning: finding.meaning,
          results: [],
        };
        grouped.set(finding.id, row);
      }
      row.results.push(finding);
    }
  }
  return [...grouped.values()].map((row) => {
    const comparable = row.results.filter((finding) => !finding.missing);
    const flagged = comparable.filter((finding) => finding.unreleased);
    return {
      id: row.id,
      baselineId: row.baselineId,
      afterId: row.afterId,
      meaning: row.meaning,
      runs: runCount,
      comparableRuns: comparable.length,
      flaggedRuns: flagged.length,
      // Strict majority. A 1-of-2 or 2-of-4 tie stays unflagged.
      repeated: flagged.length * 2 > runCount,
      medianKeptBytes: median(comparable.map((finding) => finding.keptBytes)),
      medianGainedBytes: median(comparable.map((finding) => finding.gainedBytes)),
      medianReleasedBytes: median(comparable.map((finding) => finding.releasedBytes)),
      medianPeakBytes: median(comparable.map((finding) => finding.peakBytes)),
    };
  });
}

function closestSample(samples, key) {
  const present = samples.filter((sample) => Number.isFinite(sample[key]));
  const middle = median(present.map((sample) => sample[key]));
  if (middle == null) return null;
  return present.reduce((best, sample) => (Math.abs(sample[key] - middle) < Math.abs(best[key] - middle) ? sample : best));
}

export function summarizeMeasurements(runs, options = {}) {
  const plan = options.plan ?? checkpointPlan();
  const comparisons = options.comparisons ?? retentionComparisons(plan);
  const noiseBytes = options.noiseBytes ?? NOISE_BYTES;
  const retainFraction = options.retainFraction ?? RETAIN_FRACTION;
  const byId = new Map();
  for (const run of runs) {
    for (const sample of run.checkpoints ?? []) {
      const bucket = byId.get(sample.id) ?? [];
      bucket.push({ ...sample, run: run.run, phase: sample.phase ?? run.phase });
      byId.set(sample.id, bucket);
    }
  }
  const order = plan.all.map((entry) => entry.id);
  const checkpoints = [...byId.entries()]
    .map(([id, samples]) => {
      const bytes = samples.map((sample) => sample.userAgentSpecificBytes);
      const heap = samples.map((sample) => sample.usedJSHeapSize);
      const workers = samples.map((sample) => workerCount(sample));
      const latency = samples.map((sample) => sample.measureLatencyMs);
      const previousId = previousCheckpointId(id, plan);
      const deltas = samples.map((sample) => {
        if (!previousId || !Number.isFinite(sample.userAgentSpecificBytes)) return null;
        const previous = (byId.get(previousId) ?? []).find((candidate) => candidate.run === sample.run);
        if (!previous || !Number.isFinite(previous.userAgentSpecificBytes)) return null;
        return sample.userAgentSpecificBytes - previous.userAgentSpecificBytes;
      });
      const example = closestSample(samples, 'userAgentSpecificBytes');
      const meta = plan.all.find((entry) => entry.id === id);
      return {
        id,
        phase: meta?.phase ?? samples[0].phase ?? null,
        surface: meta?.surface ?? samples[0].surface ?? null,
        resident: meta?.resident ?? samples[0].resident ?? null,
        previousId,
        runs: samples.length,
        okRuns: samples.filter((sample) => sample.ok !== false).length,
        median: {
          userAgentSpecificBytes: median(bytes),
          usedJSHeapSize: median(heap),
          totalJSHeapSize: median(samples.map((sample) => sample.totalJSHeapSize)),
          workerCount: median(workers),
          measureLatencyMs: median(latency),
        },
        spread: {
          userAgentSpecificBytes: spread(bytes),
          usedJSHeapSize: spread(heap),
          workerCount: spread(workers),
          measureLatencyMs: spread(latency),
        },
        medianDeltaBytes: median(deltas.filter((value) => Number.isFinite(value))),
        stillRunningRuns: samples.filter((sample) => sample.stillRunning === true).length,
        notes: [...new Set(samples.map((sample) => sample.note).filter(Boolean))],
        engineFailures: samples
          .filter((sample) => sample.engineFailed === true)
          .map((sample) => ({ run: sample.run, message: sample.engineError || sample.note || 'engine failed' })),
        errors: samples.filter((sample) => sample.ok === false).map((sample) => ({ run: sample.run, error: sample.error ?? 'failed' })),
        userAgentSpecificErrors: [...new Set(samples.map((sample) => sample.userAgentSpecificError).filter(Boolean))],
        exampleBreakdown: example?.breakdownTop ?? [],
        crossOriginIsolated: samples.every((sample) => sample.crossOriginIsolated !== false),
      };
    })
    .sort((a, b) => {
      const ai = order.indexOf(a.id);
      const bi = order.indexOf(b.id);
      return (ai === -1 ? order.length : ai) - (bi === -1 ? order.length : bi);
    });

  const runIds = [...new Set(runs.map((run) => run.run))].sort((a, b) => a - b);
  const perRun = runIds.map((runId) => {
    const samples = runs.filter((run) => run.run === runId).flatMap((run) => run.checkpoints ?? []);
    return comparisons.map((spec) => compareRetention(samples, spec, { noiseBytes, retainFraction }));
  });

  const largest = [...checkpoints]
    .filter((checkpoint) => Number.isFinite(checkpoint.median.userAgentSpecificBytes))
    .sort((a, b) => b.median.userAgentSpecificBytes - a.median.userAgentSpecificBytes);
  const largestSteps = [...checkpoints]
    .filter((checkpoint) => Number.isFinite(checkpoint.medianDeltaBytes))
    .sort((a, b) => b.medianDeltaBytes - a.medianDeltaBytes);

  return {
    schema: SCHEMA,
    runs: runIds.length,
    repeatable: runIds.length >= 3,
    noiseBytes,
    retainFraction,
    checkpoints,
    largest,
    largestSteps,
    retention: aggregateRetention(perRun),
  };
}

function cell(value) {
  return value == null || value === '' ? 'n/a' : String(value);
}

export function renderMarkdown(summary) {
  const lines = [];
  lines.push('# Low-end memory harness');
  lines.push('');
  lines.push(
    `${summary.runs} run${summary.runs === 1 ? '' : 's'}${summary.repeatable ? ' (median; repeatable bar is 3)' : ' (fewer than 3 runs, so this is not the repeatable median)'}.`,
  );
  lines.push(
    `User-agent memory is \`performance.measureUserAgentSpecificMemory()\` on a COOP/COEP page. The call waits for garbage collection; \`measure latency\` is that wait. JS heap is \`performance.memory.usedJSHeapSize\` when precise heap info is enabled. Worker count prefers main-thread Worker/SharedWorker objects that have not been terminated, and otherwise counts memory-breakdown scopes whose name contains "Worker".`,
  );
  lines.push(
    `A later checkpoint counts as unreleased when the peak gained since the baseline is above ${formatMib(summary.noiseBytes ?? NOISE_BYTES)} and more than ${Math.round((summary.retainFraction ?? RETAIN_FRACTION) * 100)}% of that gain is still present. Equal-to-threshold gains are not flagged. A finding is repeated only on a strict majority of runs; a tie is not.`,
  );
  if (summary.baseUrl) lines.push(`Base URL: ${summary.baseUrl}.`);
  if (summary.host?.cpu) {
    lines.push(
      `Host: ${summary.host.cpu}, ${formatMib(summary.host.totalMemBytes)} RAM, ${summary.host.platform}/${summary.host.arch}, Node ${summary.host.node}.`,
    );
  }
  lines.push('');
  lines.push('## Checkpoints');
  lines.push('');
  lines.push('| Checkpoint | Surface | Resident | UA median | UA min–max | Delta | JS heap | Workers | Measure latency |');
  lines.push('|---|---|---|---:|---:|---:|---:|---:|---:|');
  for (const checkpoint of summary.checkpoints) {
    const range = checkpoint.spread.userAgentSpecificBytes;
    const rangeText = Number.isFinite(range?.min) ? `${formatMib(range.min)}–${formatMib(range.max)}` : 'n/a';
    lines.push(
      `| ${checkpoint.id} | ${cell(checkpoint.surface)} | ${cell(checkpoint.resident)} | ${formatMib(checkpoint.median.userAgentSpecificBytes)} | ${rangeText} | ${formatMib(checkpoint.medianDeltaBytes)} | ${formatMib(checkpoint.median.usedJSHeapSize)} | ${cell(checkpoint.median.workerCount)} | ${checkpoint.median.measureLatencyMs == null ? 'n/a' : `${Math.round(checkpoint.median.measureLatencyMs)} ms`} |`,
    );
  }
  const engineFailures = summary.checkpoints.flatMap((checkpoint) => (checkpoint.engineFailures ?? []).map((failure) => ({ id: checkpoint.id, ...failure })));
  if (engineFailures.length) {
    lines.push('');
    lines.push('## Engine failures');
    lines.push('');
    lines.push('These checkpoints still have a memory reading. The engine did not settle, so the number can be the previous resident.');
    for (const failure of engineFailures) lines.push(`- \`${failure.id}\` run ${failure.run}: ${failure.message}`);
  }
  lines.push('');
  lines.push('## Largest residents');
  lines.push('');
  const top = (summary.largest ?? []).slice(0, 5);
  if (!top.length) lines.push('No user-agent byte samples.');
  for (const [index, checkpoint] of top.entries()) {
    lines.push(
      `${index + 1}. \`${checkpoint.id}\` ${formatMib(checkpoint.median.userAgentSpecificBytes)} median (${formatMib(checkpoint.spread.userAgentSpecificBytes.min)}–${formatMib(checkpoint.spread.userAgentSpecificBytes.max)}).`,
    );
  }
  lines.push('');
  lines.push('## Largest step-ups');
  lines.push('');
  const steps = (summary.largestSteps ?? []).filter((checkpoint) => checkpoint.medianDeltaBytes > 0).slice(0, 5);
  if (!steps.length) lines.push('No positive step from the previous checkpoint in the same session.');
  for (const [index, checkpoint] of steps.entries()) {
    lines.push(`${index + 1}. \`${checkpoint.id}\` added ${formatMib(checkpoint.medianDeltaBytes)} over \`${checkpoint.previousId}\`.`);
  }
  lines.push('');
  lines.push('## Still resident after switching away');
  lines.push('');
  const repeated = (summary.retention ?? []).filter((row) => row.repeated);
  const quiet = (summary.retention ?? []).filter((row) => !row.repeated);
  if (!repeated.length) lines.push('No retention comparison stayed above the threshold in a strict majority of runs.');
  for (const row of repeated) {
    lines.push(
      `- \`${row.id}\` (${row.flaggedRuns}/${row.runs} runs): ${row.meaning}. Peak gain ${formatMib(row.medianGainedBytes)}, still held ${formatMib(row.medianKeptBytes)}, dropped ${formatMib(row.medianReleasedBytes)} by \`${row.afterId}\`.`,
    );
  }
  if (quiet.length) {
    lines.push('');
    lines.push('Not repeated:');
    for (const row of quiet) {
      const detail = row.comparableRuns === 0 ? 'missing samples' : `${row.flaggedRuns}/${row.runs} runs flagged`;
      lines.push(`- \`${row.id}\`: ${detail}. ${row.meaning}.`);
    }
  }
  const failures = summary.checkpoints.filter((checkpoint) => checkpoint.errors.length || checkpoint.userAgentSpecificErrors.length);
  if (failures.length) {
    lines.push('');
    lines.push('## Sample failures');
    lines.push('');
    for (const checkpoint of failures) {
      const reasons = [
        ...checkpoint.errors.map((error) => `run ${error.run}: ${error.error}`),
        ...checkpoint.userAgentSpecificErrors.map((error) => `memory API: ${error}`),
      ];
      lines.push(`- \`${checkpoint.id}\`: ${reasons.join('; ')}`);
    }
  }
  lines.push('');
  return lines.join('\n');
}
