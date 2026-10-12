import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { runInNewContext } from 'node:vm';
import { installWorkerHook, samplePage } from '../scripts/memory_harness/page_hooks.mjs';
import {
  aggregateRetention,
  analysisControllerReady,
  analysisLc0StillRunning,
  analysisSampleReady,
  CPU_ANALYSIS_FAMILIES,
  CPU_PLAY_ENGINE_IDS,
  checkpointPlan,
  compareRetention,
  formatMib,
  lc0TimedRunStarted,
  median,
  NOISE_BYTES,
  renderMarkdown,
  requestMaiaLoad,
  retentionComparisons,
  SCHEMA,
  spread,
  summarizeMeasurements,
  workerCount,
} from '../scripts/memory_harness/summary.mjs';

const script = 'scripts/memory_harness/run.mjs';
const NOISE = NOISE_BYTES;

function run(args) {
  return spawnSync(process.execPath, [script, ...args], { cwd: process.cwd(), encoding: 'utf8' });
}

function sample(id, bytes, extra = {}) {
  return {
    id,
    ok: true,
    userAgentSpecificBytes: bytes,
    usedJSHeapSize: Math.round(bytes / 2),
    totalJSHeapSize: bytes,
    measureLatencyMs: 1000,
    workers: { live: extra.workers ?? 1, breakdownScopes: 0 },
    ...extra,
  };
}

test('median and spread ignore non-finite values', () => {
  assert.equal(median([5, 1, 3]), 3);
  assert.equal(median([4, 1, 3, 2]), 2.5);
  assert.equal(median([Number.NaN, 8]), 8);
  assert.equal(median([]), null);
  assert.deepEqual(spread([2, 9, Number.POSITIVE_INFINITY]), { min: 2, max: 9, range: 7 });
  assert.deepEqual(spread([]), { min: null, max: null, range: null });
});

test('worker count prefers the live hook and otherwise uses breakdown scopes', () => {
  assert.equal(workerCount({ workers: { live: 0, breakdownScopes: 4 } }), 0);
  assert.equal(workerCount({ workers: { live: null, breakdownScopes: 4 } }), 4);
  assert.equal(workerCount({}), null);
});

test('checkpoint plan covers every requested surface', () => {
  const plan = checkpointPlan();
  const ids = plan.all.map((entry) => entry.id);
  assert.ok(ids.includes('landing'));
  assert.ok(ids.includes('play-default-after-first-move'));
  for (const engineId of CPU_PLAY_ENGINE_IDS) assert.ok(ids.includes(`play-cpu-${engineId}`));
  assert.ok(ids.includes('play-neural-lc0-small'));
  assert.ok(ids.includes('analysis-lc0-30s'));
  for (const family of CPU_ANALYSIS_FAMILIES) assert.ok(ids.includes(`analysis-cpu-${family}`));
  assert.ok(ids.includes('arena-cpu-match'));
  assert.ok(ids.includes('maia-panel'));
  assert.deepEqual(
    retentionComparisons().map((entry) => entry.id),
    ['play-cpu-tour', 'play-left-surface', 'analysis-lc0-stop', 'analysis-cpu-tour', 'arena-stop', 'maia-left-surface'],
  );
});

test('retention flags memory that stays and ignores a released peak', () => {
  const spec = {
    id: 'play-left-surface',
    baselineId: 'landing',
    afterId: 'landing-after-play',
    duringIds: ['play-cpu-stormphrax'],
    meaning: 'left play',
  };
  const kept = compareRetention([sample('landing', 20 * NOISE), sample('play-cpu-stormphrax', 80 * NOISE), sample('landing-after-play', 70 * NOISE)], spec);
  assert.equal(kept.unreleased, true);
  assert.equal(kept.peakId, 'play-cpu-stormphrax');
  const released = compareRetention([sample('landing', 20 * NOISE), sample('play-cpu-stormphrax', 80 * NOISE), sample('landing-after-play', 22 * NOISE)], spec);
  assert.equal(released.unreleased, false);
  assert.ok(released.releasedBytes > released.keptBytes);
});

test('retention stays quiet at the noise floor and when the kept bytes only equal the threshold', () => {
  const spec = { id: 'arena-stop', baselineId: 'arena-shell', afterId: 'arena-cpu-stopped', duringIds: ['arena-cpu-match'], meaning: 'stop' };
  const baseline = 32 * NOISE;
  const atNoise = compareRetention(
    [sample('arena-shell', baseline), sample('arena-cpu-match', baseline + NOISE), sample('arena-cpu-stopped', baseline + NOISE)],
    spec,
  );
  assert.equal(atNoise.unreleased, false);
  const gained = NOISE + 1024;
  const equalThreshold = compareRetention(
    [sample('arena-shell', baseline), sample('arena-cpu-match', baseline + gained), sample('arena-cpu-stopped', baseline + NOISE)],
    spec,
  );
  assert.equal(equalThreshold.gainedBytes > NOISE, true);
  assert.equal(equalThreshold.keptBytes, NOISE);
  assert.equal(equalThreshold.unreleased, false);
});

test('a missing checkpoint does not count as unreleased', () => {
  const spec = retentionComparisons()[0];
  const result = compareRetention([sample('play-default-after-first-move', 10 * NOISE)], spec);
  assert.equal(result.missing, true);
  assert.equal(result.unreleased, false);
});

test('summary takes the median of three runs and repeats a retention flag only on a majority', () => {
  const planIds = ['landing', 'play-default-after-first-move', 'play-cpu-stormphrax', 'play-neural-lc0-small', 'landing-after-play'];
  const makeRun = (run, afterLanding) => ({
    run,
    phase: 'play',
    checkpoints: [
      sample('landing', 10 * NOISE, { workers: { live: run, breakdownScopes: 0 } }),
      sample('play-default-after-first-move', 12 * NOISE),
      sample('play-cpu-stormphrax', 40 * NOISE),
      sample('play-neural-lc0-small', 30 * NOISE),
      sample('landing-after-play', afterLanding),
    ].map((entry) => ({ ...entry, phase: 'play', surface: entry.id.startsWith('landing') ? 'landing' : 'play' })),
  });
  const summary = summarizeMeasurements([makeRun(1, 36 * NOISE), makeRun(2, 38 * NOISE), makeRun(3, 11 * NOISE)]);
  assert.equal(summary.repeatable, true);
  const landing = summary.checkpoints.find((entry) => entry.id === 'landing');
  assert.equal(landing.median.userAgentSpecificBytes, 10 * NOISE);
  assert.equal(landing.median.workerCount, 2);
  assert.deepEqual(
    summary.checkpoints.map((entry) => entry.id).filter((id) => planIds.includes(id)),
    planIds,
  );
  const storm = summary.checkpoints.find((entry) => entry.id === 'play-cpu-stormphrax');
  const firstMove = summary.checkpoints.find((entry) => entry.id === 'play-default-after-first-move');
  assert.equal(firstMove.medianDeltaBytes, 2 * NOISE);
  assert.equal(storm.median.userAgentSpecificBytes, 40 * NOISE);
  const left = summary.retention.find((entry) => entry.id === 'play-left-surface');
  assert.equal(left.flaggedRuns, 2);
  assert.equal(left.repeated, true);
  const tour = summary.retention.find((entry) => entry.id === 'play-cpu-tour');
  assert.equal(tour.repeated, true);
  const markdown = renderMarkdown(summary);
  assert.match(markdown, /play-cpu-stormphrax/);
  assert.match(markdown, /play-left-surface/);
  assert.match(markdown, /Still resident after switching away/);
  assert.equal(formatMib(NOISE), '8.0 MiB');
  assert.equal(summary.largest[0].id, 'play-cpu-stormphrax');
});

test('analysis sample waits for a changed family-named result or a new failure', () => {
  const ready = (state) => analysisSampleReady(state).ok;
  assert.equal(ready({ family: 'reckless', previousMessage: null, message: 'Make a move or press Analyze to begin.' }), false);
  assert.equal(
    ready({
      family: 'reckless',
      previousMessage: 'Analysis complete · Reckless Relaxed SIMD',
      message: 'Analysis complete · Reckless Relaxed SIMD',
    }),
    false,
  );
  assert.equal(
    ready({
      family: 'reckless',
      previousMessage: 'Analysis complete · SF Lite',
      message: 'Analysis complete · Reckless Relaxed SIMD',
    }),
    true,
  );
  assert.equal(ready({ family: 'reckless', previousMessage: '', message: 'Downloading Reckless …' }), false);
  assert.equal(ready({ family: 'reckless', previousMessage: '', message: 'Analyzing (Reckless Relaxed SIMD, 1 lines)…' }), false);
  assert.equal(ready({ family: 'lc0', previousMessage: 'Analysis complete · Stormphrax', message: 'Lc0: 100/100000 visits' }), true);
  assert.equal(ready({ family: 'reckless', previousMessage: '', message: 'Analysis complete · SF Lite' }), false);
  assert.equal(ready({ family: 'reckless', previousMessage: 'Downloading Reckless', message: 'Reckless asset missing' }), true);
  assert.equal(ready({ family: 'sf', previousMessage: 'idle', message: 'Analysis complete · SF Lite' }), true);
  assert.equal(ready({ family: 'plentychess', previousMessage: 'idle', message: 'Analysis complete · PlentyChess' }), true);
  assert.equal(ready({ family: 'viridithas', previousMessage: 'idle', message: 'Analysis failed: HTTP 404' }), true);
  assert.equal(
    ready({
      family: 'sf',
      previousMessage: 'Analysis complete · Reckless Relaxed SIMD',
      message: 'Analyzing (SF Lite, 1 lines)…',
      progress: 'SF Lite: searching…',
    }),
    true,
  );
  assert.equal(
    ready({
      family: 'reckless',
      previousMessage: 'Analysis complete · SF Lite',
      message: 'Analyzing (Reckless Relaxed SIMD, 1 lines)…',
      progress: 'SF Lite: 2/2 nodes',
    }),
    false,
  );
});

test('the Lc0 timer starts on visit progress, not the start banner or a failure', () => {
  const banner = analysisSampleReady({
    family: 'lc0',
    previousMessage: '',
    message: 'Analyzing (Lc0, 1 lines)…',
    stopEnabled: true,
  });
  assert.equal(banner.ok, false);
  assert.equal(lc0TimedRunStarted(banner), false);
  assert.equal(analysisLc0StillRunning({ message: 'Analyzing (Lc0, 1 lines)…', stopEnabled: true }).running, false);
  const visits = analysisSampleReady({
    family: 'lc0',
    previousMessage: 'Analyzing (Lc0, 1 lines)…',
    message: 'Lc0: 12/100000 visits 0%',
  });
  assert.equal(visits.ok, true);
  assert.equal(visits.failed, false);
  assert.equal(lc0TimedRunStarted(visits), true);
  assert.equal(analysisLc0StillRunning({ message: visits.message, stopEnabled: true }).running, true);
  assert.equal(analysisLc0StillRunning({ message: visits.message, stopEnabled: false }).running, false);
  const fromProgress = analysisSampleReady({
    family: 'lc0',
    previousMessage: '',
    message: 'Analyzing (Lc0, 1 lines)…',
    progress: 'Lc0: 4/100000 visits 0%',
  });
  assert.equal(lc0TimedRunStarted(fromProgress), true);
  const failed = analysisSampleReady({ family: 'lc0', previousMessage: '', message: 'Analysis failed: model missing' });
  assert.equal(failed.ok, true);
  assert.equal(failed.failed, true);
  assert.equal(lc0TimedRunStarted(failed), false);
  assert.equal(analysisLc0StillRunning({ message: failed.message, stopEnabled: true }).running, false);
  for (const message of ['Model load failed: missing net', 'LC0 ORT ONNX load failed: worker exited']) {
    const backend = analysisSampleReady({ family: 'lc0', previousMessage: 'Ready.', message });
    assert.equal(backend.ok, true, message);
    assert.equal(backend.failed, true, message);
    assert.equal(lc0TimedRunStarted(backend), false, message);
    assert.equal(analysisLc0StillRunning({ message, stopEnabled: true }).running, false, message);
  }
  const staleBackend = analysisSampleReady({
    family: 'lc0',
    previousMessage: 'Model load failed: missing net',
    message: 'Model load failed: missing net',
  });
  assert.equal(staleBackend.ok, false);
  const done = analysisSampleReady({ family: 'lc0', previousMessage: '', message: 'Analysis complete · Lc0' });
  assert.equal(lc0TimedRunStarted(done), false);
  assert.equal(analysisLc0StillRunning({ message: done.message, stopEnabled: true }).running, false);
});

test('Maia waits for the analysis controller and retries a click that did not land', () => {
  assert.equal(analysisControllerReady({ hasRows: false, hasMaiaButton: true }).ok, false);
  assert.equal(analysisControllerReady({ hasRows: true, hasMaiaButton: false }).ok, false);
  assert.equal(analysisControllerReady({ hasRows: true, hasMaiaButton: true }).ok, true);

  const HTMLButtonElement = class HTMLButtonElement {};
  const HTMLDetailsElement = class HTMLDetailsElement {};
  let clicks = 0;
  const idleButton = new HTMLButtonElement();
  idleButton.disabled = false;
  idleButton.hidden = false;
  idleButton.click = () => {
    clicks += 1;
  };
  const idle = runInNewContext(`(${requestMaiaLoad.toString()})`, {
    HTMLDetailsElement,
    HTMLButtonElement,
    document: {
      querySelector(selector) {
        if (selector === 'details.human-moves') return null;
        if (selector === '#maia3Enable') return idleButton;
        if (selector === '#maia3Grid') return { hidden: true };
        if (selector === '#maia3PanelStatus') return { textContent: '' };
        throw new Error(`unexpected selector ${selector}`);
      },
    },
  });
  assert.equal(idle().ok, false);
  assert.equal(clicks, 1);

  const status = { textContent: '' };
  const button = new HTMLButtonElement();
  button.disabled = false;
  button.hidden = false;
  button.click = function click() {
    this.disabled = true;
    status.textContent = 'Loading Maia3…';
  };
  const started = runInNewContext(`(${requestMaiaLoad.toString()})`, {
    HTMLDetailsElement,
    HTMLButtonElement,
    document: {
      querySelector(selector) {
        if (selector === 'details.human-moves') return null;
        if (selector === '#maia3Enable') return button;
        if (selector === '#maia3Grid') return { hidden: true };
        if (selector === '#maia3PanelStatus') return status;
        throw new Error(`unexpected selector ${selector}`);
      },
    },
  });
  const result = started();
  assert.equal(result.ok, true);
  assert.equal(result.status, 'Loading Maia3…');
  assert.equal(started().ok, true);
});

test('analysis sample predicate serializes without module bindings', () => {
  const source = analysisSampleReady.toString();
  const fn = runInNewContext(`(${source})`, {
    document: {
      querySelector(selector) {
        if (selector === '#message') return { textContent: 'Analyzing (Lc0, 1 lines)…' };
        if (selector === '#analysisSearchProgress') return { textContent: 'Lc0: 3/100000 visits 0%' };
        throw new Error(`unexpected selector ${selector}`);
      },
    },
  });
  const ready = fn({ family: 'lc0', previousMessage: 'Analysis complete · SF Lite' });
  assert.equal(ready.ok, true);
  assert.equal(ready.failed, false);
  assert.equal(ready.message, 'Analyzing (Lc0, 1 lines)…');
  assert.equal(lc0TimedRunStarted(ready), true);
  const stillRunning = runInNewContext(`(${analysisLc0StillRunning.toString()})`, {
    document: {
      querySelector(selector) {
        if (selector === '#message') return { textContent: 'Lc0: 9/100000 visits 0%' };
        if (selector === '#analysisSearchProgress') return { textContent: '' };
        if (selector === '#stop') return { disabled: false };
        throw new Error(`unexpected selector ${selector}`);
      },
    },
  });
  assert.equal(stillRunning().running, true);
});

function retentionRun(unreleased) {
  return [
    {
      id: 'arena-stop',
      baselineId: 'a',
      afterId: 'b',
      meaning: 'stop',
      missing: false,
      unreleased,
      keptBytes: unreleased ? 3 : 0,
      gainedBytes: 4,
      releasedBytes: unreleased ? 1 : 4,
      peakBytes: 5,
    },
  ];
}

test('repeated retention requires a strict majority, so ties stay quiet', () => {
  assert.equal(aggregateRetention([retentionRun(true), retentionRun(false)])[0].repeated, false);
  assert.equal(aggregateRetention([retentionRun(true), retentionRun(true), retentionRun(false), retentionRun(false)])[0].repeated, false);
  assert.equal(aggregateRetention([retentionRun(true), retentionRun(true), retentionRun(true), retentionRun(false)])[0].repeated, true);
  assert.equal(aggregateRetention([retentionRun(true), retentionRun(true), retentionRun(false)])[0].repeated, true);
});

test('summary keeps an engine failure beside its memory sample', () => {
  const summary = summarizeMeasurements([
    {
      run: 1,
      phase: 'analysis',
      checkpoints: [
        sample('analysis-cpu-reckless', 20 * NOISE, {
          engineFailed: true,
          engineError: 'Reckless asset missing',
          note: 'depth 2 failed: Reckless asset missing',
        }),
        sample('analysis-cpu-sf', 12 * NOISE, { note: 'depth 2, sampled after this family settled' }),
      ],
    },
  ]);
  const reckless = summary.checkpoints.find((entry) => entry.id === 'analysis-cpu-reckless');
  const sf = summary.checkpoints.find((entry) => entry.id === 'analysis-cpu-sf');
  assert.equal(reckless.median.userAgentSpecificBytes, 20 * NOISE);
  assert.deepEqual(reckless.engineFailures, [{ run: 1, message: 'Reckless asset missing' }]);
  assert.deepEqual(sf.engineFailures, []);
  assert.match(renderMarkdown(summary), /## Engine failures/);
  assert.match(renderMarkdown(summary), /analysis-cpu-reckless` run 1: Reckless asset missing/);
});

test('one flagged run out of three is not called repeated', () => {
  const findings = aggregateRetention([
    [
      {
        id: 'arena-stop',
        baselineId: 'a',
        afterId: 'b',
        meaning: 'stop',
        missing: false,
        unreleased: true,
        keptBytes: 3,
        gainedBytes: 4,
        releasedBytes: 1,
        peakBytes: 5,
      },
    ],
    [
      {
        id: 'arena-stop',
        baselineId: 'a',
        afterId: 'b',
        meaning: 'stop',
        missing: false,
        unreleased: false,
        keptBytes: 0,
        gainedBytes: 4,
        releasedBytes: 4,
        peakBytes: 5,
      },
    ],
    [
      {
        id: 'arena-stop',
        baselineId: 'a',
        afterId: 'b',
        meaning: 'stop',
        missing: false,
        unreleased: false,
        keptBytes: 0,
        gainedBytes: 4,
        releasedBytes: 4,
        peakBytes: 5,
      },
    ],
  ]);
  assert.equal(findings[0].repeated, false);
  assert.equal(findings[0].flaggedRuns, 1);
  assert.equal(findings[0].medianKeptBytes, 0);
});

test('memory harness dry-run lists the checkpoint plan without a browser', () => {
  const result = run(['--base-url', 'https://example.test/path', '--dry-run', '--runs', '3']);
  assert.equal(result.status, 0, result.stderr);
  const report = JSON.parse(result.stdout);
  assert.equal(report.status, 'MEMORY_HARNESS_DRY_RUN');
  assert.equal(report.schema, SCHEMA);
  assert.equal(report.baseUrl, 'https://example.test');
  assert.equal(report.runs, 3);
  assert.ok(report.phases.play.includes('play-default-after-first-move'));
  assert.ok(report.phases.analysis.includes('analysis-lc0-30s'));
  assert.ok(report.phases.arena.includes('arena-cpu-match'));
  assert.ok(report.phases.maia.includes('maia-panel'));
});

test('memory harness rejects a bad phase, a bad repeat count, and a non-HTTP origin', () => {
  const phase = run(['--phase', 'phone', '--dry-run']);
  assert.notEqual(phase.status, 0);
  assert.match(phase.stderr, /Invalid --phase: phone/);
  const runs = run(['--runs', '0', '--dry-run']);
  assert.notEqual(runs.status, 0);
  assert.match(runs.stderr, /Invalid --runs: 0/);
  const origin = run(['--base-url', 'file:///tmp/app', '--dry-run']);
  assert.notEqual(origin.status, 0);
  assert.match(origin.stderr, /Unsupported --base-url protocol: file:/);
});

test('combine writes a markdown summary from phase JSON files', () => {
  const directory = mkdtempSync(join(tmpdir(), 'mem-harness-'));
  const artifact = {
    schema: SCHEMA,
    kind: 'phase-run',
    run: 1,
    phase: 'arena',
    baseUrl: 'http://127.0.0.1:5186',
    host: { platform: 'darwin', arch: 'arm64', cpu: 'Test CPU', totalMemBytes: 8 * 1024 * 1024 * 1024, node: 'v1' },
    loadavg: { start: [1.5], end: [1.7] },
    durationMs: 1000,
    checkpoints: [sample('arena-shell', 10 * NOISE), sample('arena-cpu-match', 30 * NOISE), sample('arena-cpu-stopped', 28 * NOISE)],
  };
  writeFileSync(join(directory, 'run-1-arena.json'), JSON.stringify(artifact));
  const summaryPath = join(directory, 'summary.json');
  const markdownPath = join(directory, 'summary.md');
  const result = run(['--combine', directory, '--summary-json', summaryPath, '--summary-md', markdownPath]);
  assert.equal(result.status, 0, result.stderr);
  const summary = JSON.parse(readFileSync(summaryPath, 'utf8'));
  assert.equal(summary.schema, SCHEMA);
  assert.equal(summary.runs, 1);
  assert.equal(summary.retention.find((entry) => entry.id === 'arena-stop').repeated, true);
  const markdown = readFileSync(markdownPath, 'utf8');
  assert.match(markdown, /arena-cpu-match/);
  assert.match(markdown, /Test CPU/);
});

test('page sample source includes its helpers so Playwright can serialize it', () => {
  const source = samplePage.toString();
  assert.match(source, /async function measureUserAgentMemory/);
  assert.match(source, /function pageStatus/);
  assert.match(source, /function breakdownTop/);
  assert.doesNotMatch(source, /\bimport\b/);
  const hook = installWorkerHook.toString();
  assert.match(hook, /__memprobeWorkers/);
  assert.match(hook, /Worker/);
});
