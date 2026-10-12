#!/usr/bin/env node
/**
 * Low-end memory harness.
 *
 * Drives the BUILT app (COOP/COEP static server) with Playwright, which is
 * already a dependency. Each sample records measureUserAgentSpecificMemory()
 * (it waits for GC), the JS heap, and worker counts.
 *
 * One invocation is one phase of one or more runs. Wrap each invocation in
 * the shared bench lock. A full Play tour loads several WASM engines and can
 * approach the 20 minute chunk budget; do not pass --runs 3 under one lock
 * unless a single run of that phase is known to be short.
 *
 *   PORT=5186 node scripts/serve_isolated_static.mjs dist-client 5186
 *   node scripts/memory_harness/run.mjs --phase play --runs 1 --run-index 1 \
 *     --base-url http://127.0.0.1:5186 --out /tmp/mem-harness/run-1-play.json
 *   node scripts/memory_harness/run.mjs --combine /tmp/mem-harness \
 *     --summary-json docs/lowend_memory_harness_2026-10.json \
 *     --summary-md /tmp/mem-harness/summary.md
 */
import { mkdir, readdir, readFile, writeFile } from 'node:fs/promises';
import { arch, cpus, freemem, loadavg, platform, totalmem } from 'node:os';
import { dirname, join } from 'node:path';
import { setTimeout as delay } from 'node:timers/promises';
import { pathToFileURL } from 'node:url';
import { parseScriptArgs } from '../lib/cli.mjs';
import { installWorkerHook, samplePage } from './page_hooks.mjs';
import {
  analysisControllerReady,
  analysisLc0StillRunning,
  analysisSampleReady,
  applyObservedEngineFailure,
  CPU_ANALYSIS_FAMILIES,
  CPU_PLAY_ENGINE_IDS,
  checkpointEngineFailure,
  checkpointPlan,
  DEFAULT_PLAY_ENGINE_ID,
  lc0TimedRunStarted,
  NEURAL_PLAY_ENGINE_ID,
  renderMarkdown,
  requestMaiaLoad,
  retentionComparisons,
  SCHEMA,
  summarizeMeasurements,
} from './summary.mjs';

const DEFAULT_BASE_URL = 'http://127.0.0.1:5186';
const DEFAULT_TIMEOUT_MS = 180_000;
const PHASE_NAMES = ['play', 'analysis', 'arena', 'maia', 'overlay'];

const USAGE = `Usage: node scripts/memory_harness/run.mjs [options]

Records resident memory at Play, Analysis, Arena, and Maia checkpoints on the
built COOP/COEP app. Use --phase to keep one bench-lock chunk to a single
surface tour. --combine merges phase JSON files into a median summary.

Options:
  --base-url URL       Built app origin (default ${DEFAULT_BASE_URL})
  --phase NAME         play, analysis, arena, maia, or overlay (default play)
  --runs N             Repeats inside this process (default 3; prefer 1 under benchlock)
  --run-index N        Run number written into the JSON when --runs is 1 (default 1)
  --timeout MS         Per-engine wait (default ${DEFAULT_TIMEOUT_MS})
  --lc0-seconds N      How long Analysis Lc0 should run before the sample (default 30)
  --out PATH           Phase JSON path (required for a measurement)
  --combine DIR        Merge *.json phase files in DIR and do not launch a browser
  --summary-json PATH  Combined summary JSON
  --summary-md PATH    Combined summary markdown
  --screenshot-dir PATH  Write failure and overlay screenshots here
  --headed             Show the browser window
  --dry-run            Print the checkpoint plan and exit
  -h, --help           Show this help
`;

export function parseArgs(argv) {
  const args = parseScriptArgs(argv, {
    options: {
      'base-url': { type: 'string', default: DEFAULT_BASE_URL },
      phase: { type: 'string', default: 'play' },
      runs: { type: 'string', default: '3' },
      'run-index': { type: 'string', default: '1' },
      timeout: { type: 'string', default: String(DEFAULT_TIMEOUT_MS) },
      'lc0-seconds': { type: 'string', default: '30' },
      out: { type: 'string' },
      combine: { type: 'string' },
      'summary-json': { type: 'string' },
      'summary-md': { type: 'string' },
      'screenshot-dir': { type: 'string' },
      headed: { type: 'boolean', default: false },
      'dry-run': { type: 'boolean', default: false },
    },
    usage: USAGE,
  });
  args.runs = Number(args.runs);
  args.runIndex = Number(args.runIndex);
  args.timeoutMs = Number(args.timeout);
  args.lc0Seconds = Number(args.lc0Seconds);
  delete args.timeout;
  if (!Number.isInteger(args.runs) || args.runs < 1) throw new Error(`Invalid --runs: ${args.runs}`);
  if (!Number.isInteger(args.runIndex) || args.runIndex < 1) throw new Error(`Invalid --run-index: ${args.runIndex}`);
  if (!Number.isFinite(args.timeoutMs) || args.timeoutMs <= 0) throw new Error(`Invalid --timeout: ${args.timeoutMs}`);
  if (!Number.isFinite(args.lc0Seconds) || args.lc0Seconds <= 0) throw new Error(`Invalid --lc0-seconds: ${args.lc0Seconds}`);
  if (!PHASE_NAMES.includes(args.phase)) throw new Error(`Invalid --phase: ${args.phase}`);
  let base;
  try {
    base = new URL(args.baseUrl);
  } catch {
    throw new Error(`Invalid --base-url: ${args.baseUrl}`);
  }
  if (!['http:', 'https:'].includes(base.protocol)) throw new Error(`Unsupported --base-url protocol: ${base.protocol}`);
  args.baseUrl = base.origin;
  return args;
}

function log(message) {
  process.stderr.write(`[mem-harness] ${message}\n`);
}

async function waitFor(page, fn, arg, timeoutMs, label) {
  const deadline = Date.now() + timeoutMs;
  let last;
  let lastError;
  while (Date.now() < deadline) {
    try {
      last = await page.evaluate(fn, arg);
      if (last?.ok) return last;
    } catch (error) {
      lastError = error;
    }
    await delay(400);
  }
  throw new Error(`Timed out waiting for ${label}. Last=${JSON.stringify(last)} Error=${lastError?.message ?? 'none'}`);
}

async function shot(page, dir, name) {
  if (!dir) return;
  await mkdir(dir, { recursive: true });
  await page.screenshot({ path: join(dir, name), fullPage: false }).catch(() => undefined);
}

function hostInfo() {
  return {
    platform: platform(),
    arch: arch(),
    cpu: cpus()[0]?.model ?? 'unknown',
    cpuCount: cpus().length,
    totalMemBytes: totalmem(),
    freeMemBytes: freemem(),
    node: process.version,
  };
}

async function openSession(chromium, args) {
  // Playwright's default headless binary is chrome-headless-shell. That shell
  // rejects measureUserAgentSpecificMemory ("is not available") because
  // PerformanceManagerInstrumentation is off. channel "chromium" is the full
  // Chrome-for-Testing binary, including when headless.
  const browser = await chromium.launch({
    channel: 'chromium',
    headless: !args.headed,
    timeout: 60_000,
    protocolTimeout: 300_000,
    args: ['--enable-precise-memory-info'],
  });
  const context = await browser.newContext({ viewport: { width: 1280, height: 800 }, serviceWorkers: 'allow' });
  await context.addInitScript(installWorkerHook);
  const page = await context.newPage();
  page.setDefaultTimeout(240_000);
  page.setDefaultNavigationTimeout(45_000);
  const errors = [];
  page.on('pageerror', (error) => errors.push(String(error?.message ?? error)));
  page.on('dialog', (dialog) => {
    dialog.dismiss().catch(() => undefined);
  });
  return { browser, context, page, errors };
}

async function gotoPath(page, baseUrl, pathname) {
  const url = new URL(pathname, baseUrl);
  await page.goto(url.href, { waitUntil: 'domcontentloaded', timeout: 45_000 });
  return url.href;
}

async function clientNavigate(page, pathname, timeoutMs) {
  const clicked = await page.evaluate((target) => {
    const link = document.querySelector(`.site-header nav.primary a[href="${target}"]`) || document.querySelector(`a[href="${target}"]`);
    if (!link) return false;
    link.click();
    return true;
  }, pathname);
  if (!clicked) throw new Error(`No in-app link for ${pathname}`);
  await waitFor(
    page,
    (target) => {
      const path = location.pathname.length > 1 && location.pathname.endsWith('/') ? location.pathname.slice(0, -1) : location.pathname;
      const want = target.length > 1 && target.endsWith('/') ? target.slice(0, -1) : target;
      return { ok: path === want };
    },
    pathname,
    timeoutMs,
    `navigation to ${pathname}`,
  );
}

async function sample(page, meta) {
  return page.evaluate(samplePage, meta);
}

async function record(page, meta, bucket, args) {
  log(`run ${meta.run} ${meta.phase} sample ${meta.id}`);
  try {
    const reading = applyObservedEngineFailure(await sample(page, meta));
    bucket.push(reading);
    const mb = Number.isFinite(reading.userAgentSpecificBytes) ? `${(reading.userAgentSpecificBytes / 1048576).toFixed(1)} MiB` : 'n/a';
    log(`  ${meta.id}: UA ${mb}, heap ${reading.usedJSHeapSize ?? 'n/a'}, workers ${reading.workers?.live ?? 'n/a'}, measure ${reading.measureLatencyMs} ms`);
    return reading;
  } catch (error) {
    await shot(page, args.screenshotDir, `fail-${meta.phase}-${meta.id}.png`);
    const failed = { ...meta, ok: false, error: error?.message ?? String(error) };
    bucket.push(failed);
    log(`  ${meta.id} FAILED: ${failed.error}`);
    return failed;
  }
}

async function playEngineFirstMove(page, engineId, timeoutMs) {
  await waitFor(
    page,
    (id) => {
      const select = document.querySelector('#engineSelect');
      const option = select && [...select.options].find((entry) => entry.value === id);
      return { ok: !!select && !select.disabled && !!option && !option.disabled, engine: select?.value ?? null, disabled: !!option?.disabled };
    },
    engineId,
    timeoutMs,
    `${engineId} to become selectable`,
  );
  await page.evaluate((id) => {
    const engine = document.querySelector('#engineSelect');
    const level = document.querySelector('#levelSelect');
    const color = document.querySelector('#colorSelect');
    if (color && color.value !== 'white') {
      color.value = 'white';
      color.dispatchEvent(new Event('change', { bubbles: true }));
    }
    if (level && level.value !== '0') {
      level.value = '0';
      level.dispatchEvent(new Event('change', { bubbles: true }));
    }
    if (engine && engine.value !== id) {
      engine.value = id;
      engine.dispatchEvent(new Event('change', { bubbles: true }));
    }
  }, engineId);
  await delay(300);
  await page.evaluate((id) => {
    const banner = document.querySelector('#restartBanner');
    if (banner && !banner.hidden) document.querySelector('#confirmRestart')?.click();
    else if ((document.querySelector('#moveList')?.textContent?.trim() ?? '') !== 'No moves yet') document.querySelector('#newGame')?.click();
    const engine = document.querySelector('#engineSelect');
    if (engine && engine.value !== id) {
      engine.value = id;
      engine.dispatchEvent(new Event('change', { bubbles: true }));
    }
    const level = document.querySelector('#levelSelect');
    if (level && level.value !== '0') {
      level.value = '0';
      level.dispatchEvent(new Event('change', { bubbles: true }));
    }
  }, engineId);
  await waitFor(
    page,
    (id) => {
      const engine = document.querySelector('#engineSelect')?.value;
      const level = document.querySelector('#levelSelect')?.value;
      const note = document.querySelector('#engineNote')?.textContent ?? '';
      const moves = document.querySelector('#moveList')?.textContent?.trim() ?? '';
      const banner = document.querySelector('#restartBanner');
      if (banner && !banner.hidden) document.querySelector('#confirmRestart')?.click();
      return { ok: engine === id && level === '0' && note.includes('is ready') && moves === 'No moves yet', engine, level, note, moves };
    },
    engineId,
    timeoutMs,
    `${engineId} ready for a new game`,
  );
  await page.evaluate(() => {
    const color = document.querySelector('#colorSelect');
    color.value = 'black';
    color.dispatchEvent(new Event('change', { bubbles: true }));
  });
  await waitFor(
    page,
    (id) => {
      const engine = document.querySelector('#engineSelect')?.value;
      const status = document.querySelector('#status')?.textContent ?? '';
      const moves = document.querySelector('#moveList')?.textContent?.trim() ?? '';
      return { ok: engine === id && status.includes('Your move') && moves.length > 0 && moves !== 'No moves yet', engine, status, moves };
    },
    engineId,
    timeoutMs,
    `${engineId} first move`,
  );
}

async function runPlay(page, args, run) {
  const checkpoints = [];
  const meta = (id, resident = null) => ({ id, phase: 'play', surface: id.startsWith('landing') ? 'landing' : 'play', resident, run });
  await gotoPath(page, args.baseUrl, '/');
  await waitFor(page, () => ({ ok: !!document.querySelector('h1') }), null, args.timeoutMs, 'landing');
  await record(page, meta('landing'), checkpoints, args);

  await gotoPath(page, args.baseUrl, '/app/play/');
  await waitFor(
    page,
    (id) => ({ ok: !!document.querySelector(`#engineSelect option[value="${id}"]`) }),
    DEFAULT_PLAY_ENGINE_ID,
    args.timeoutMs,
    'play engine select',
  );
  const selected = await page.evaluate(() => document.querySelector('#engineSelect')?.value ?? null);
  if (selected !== DEFAULT_PLAY_ENGINE_ID) {
    checkpoints.push({
      ...meta('play-default-after-first-move', DEFAULT_PLAY_ENGINE_ID),
      ok: false,
      error: `Fresh Play context selected ${selected}, expected ${DEFAULT_PLAY_ENGINE_ID}`,
    });
  } else {
    try {
      await playEngineFirstMove(page, DEFAULT_PLAY_ENGINE_ID, args.timeoutMs);
      await record(
        page,
        { ...meta('play-default-after-first-move', DEFAULT_PLAY_ENGINE_ID), note: 'fastest level, engine played the first move' },
        checkpoints,
        args,
      );
    } catch (error) {
      await shot(page, args.screenshotDir, `fail-play-${DEFAULT_PLAY_ENGINE_ID}.png`);
      checkpoints.push({ ...meta('play-default-after-first-move', DEFAULT_PLAY_ENGINE_ID), ok: false, error: error?.message ?? String(error) });
    }
  }

  for (const engineId of CPU_PLAY_ENGINE_IDS) {
    try {
      await playEngineFirstMove(page, engineId, args.timeoutMs);
      await record(page, { ...meta(`play-cpu-${engineId}`, engineId), note: 'fastest level after a completed first engine move' }, checkpoints, args);
    } catch (error) {
      await shot(page, args.screenshotDir, `fail-play-${engineId}.png`);
      checkpoints.push({ ...meta(`play-cpu-${engineId}`, engineId), ok: false, error: error?.message ?? String(error) });
      log(`play ${engineId} failed, continuing: ${error?.message ?? error}`);
    }
  }

  try {
    await playEngineFirstMove(page, NEURAL_PLAY_ENGINE_ID, args.timeoutMs);
    await record(page, { ...meta(`play-neural-${NEURAL_PLAY_ENGINE_ID}`, NEURAL_PLAY_ENGINE_ID), note: 'neural engine after the CPU tour' }, checkpoints, args);
  } catch (error) {
    await shot(page, args.screenshotDir, `fail-play-${NEURAL_PLAY_ENGINE_ID}.png`);
    checkpoints.push({ ...meta(`play-neural-${NEURAL_PLAY_ENGINE_ID}`, NEURAL_PLAY_ENGINE_ID), ok: false, error: error?.message ?? String(error) });
  }

  try {
    await clientNavigate(page, '/', args.timeoutMs);
    await waitFor(page, () => ({ ok: !!document.querySelector('h1') }), null, args.timeoutMs, 'landing after play');
    await record(page, { ...meta('landing-after-play'), note: 'in-app navigation away from Play; module caches may remain' }, checkpoints, args);
  } catch (error) {
    checkpoints.push({ ...meta('landing-after-play'), ok: false, error: error?.message ?? String(error) });
  }
  return checkpoints;
}

async function setAnalysisFamily(page, family, strength) {
  await page.evaluate((nextFamily) => {
    const select = document.querySelector('.row-fam');
    if (!select) return;
    select.value = nextFamily;
    select.dispatchEvent(new Event('change', { bubbles: true }));
  }, family);
  await delay(50);
  await page.evaluate((nextStrength) => {
    const input = document.querySelector('.row-strength');
    if (!input) return;
    input.value = String(nextStrength);
    input.dispatchEvent(new Event('input', { bubbles: true }));
    input.dispatchEvent(new Event('change', { bubbles: true }));
  }, strength);
}

async function disableAutoAnalyze(page) {
  await page.evaluate(() => {
    const box = document.querySelector('#autoAnalyze');
    if (!box) return;
    box.checked = false;
    box.dispatchEvent(new Event('change', { bubbles: true }));
  });
}

async function stopAnalysis(page, timeoutMs) {
  await page.evaluate(() => document.querySelector('#stop')?.click());
  await waitFor(
    page,
    () => {
      const stop = document.querySelector('#stop');
      const message = document.querySelector('#message')?.textContent ?? '';
      return { ok: !stop || stop.disabled || /complete|stopped|failed/i.test(message), message };
    },
    null,
    timeoutMs,
    'analysis to stop',
  ).catch(() => undefined);
}

async function runAnalysis(page, args, run) {
  const checkpoints = [];
  const meta = (id, resident = null) => ({ id, phase: 'analysis', surface: 'analysis', resident, run });
  await gotoPath(page, args.baseUrl, '/app/analysis/');
  await disableAutoAnalyze(page);
  await waitFor(page, () => ({ ok: !!document.querySelector('.row-fam') && !!document.querySelector('#analyze') }), null, args.timeoutMs, 'analysis controls');
  await stopAnalysis(page, 10_000);
  await record(page, { ...meta('analysis-shell'), note: 'analysis open, auto-analyze unchecked before the timed Lc0 run' }, checkpoints, args);

  try {
    const previousMessage = await page.evaluate(() => document.querySelector('#message')?.textContent ?? '');
    await setAnalysisFamily(page, 'lc0', 100_000);
    await page.evaluate(() => document.querySelector('#analyze')?.click());
    const ready = await waitFor(page, analysisSampleReady, { family: 'lc0', previousMessage }, args.timeoutMs, 'Lc0 search progress');
    if (!lc0TimedRunStarted(ready)) {
      const note = ready.failed ? `Lc0 failed before search progress: ${ready.message}` : `Lc0 finished before a running visit count: ${ready.message}`;
      await record(
        page,
        {
          ...meta('analysis-lc0-30s', 'lc0'),
          note,
          stillRunning: false,
          engineFailed: ready.failed === true,
          engineError: ready.failed ? ready.message : undefined,
        },
        checkpoints,
        args,
      );
    } else {
      const started = Date.now();
      const runMs = args.lc0Seconds * 1000;
      let stillRunning = true;
      let elapsed = 0;
      let lastState = null;
      while (elapsed < runMs) {
        await delay(1000);
        elapsed = Date.now() - started;
        lastState = await page.evaluate(analysisLc0StillRunning);
        if (!lastState.running) {
          stillRunning = false;
          break;
        }
      }
      const failure = !stillRunning && lastState?.failed ? checkpointEngineFailure(lastState.message) : { engineFailed: false, engineError: undefined };
      const note = stillRunning
        ? `Lc0 still running after ${Math.round(elapsed / 1000)}s at 100000 visits`
        : failure.engineFailed
          ? `Lc0 failed during the timed run: ${failure.engineError}`
          : `Lc0 search ended after ${Math.round(elapsed / 1000)}s, before ${args.lc0Seconds}s`;
      await record(page, { ...meta('analysis-lc0-30s', 'lc0'), note, stillRunning, ...failure }, checkpoints, args);
    }
  } catch (error) {
    await shot(page, args.screenshotDir, 'fail-analysis-lc0.png');
    checkpoints.push({ ...meta('analysis-lc0-30s', 'lc0'), ok: false, error: error?.message ?? String(error) });
  }

  try {
    await stopAnalysis(page, args.timeoutMs);
    await delay(500);
    await record(page, { ...meta('analysis-lc0-stopped', 'lc0'), note: 'Stop clicked after the Lc0 run' }, checkpoints, args);
  } catch (error) {
    checkpoints.push({ ...meta('analysis-lc0-stopped', 'lc0'), ok: false, error: error?.message ?? String(error) });
  }

  for (const family of CPU_ANALYSIS_FAMILIES) {
    try {
      const previousMessage = await page.evaluate(() => document.querySelector('#message')?.textContent ?? '');
      await setAnalysisFamily(page, family, 2);
      await page.evaluate(() => document.querySelector('#analyze')?.click());
      const ready = await waitFor(page, analysisSampleReady, { family, previousMessage }, args.timeoutMs, `${family} analysis output`);
      await record(
        page,
        {
          ...meta(`analysis-cpu-${family}`, family),
          note: ready.failed ? `depth 2 failed: ${ready.message}` : 'depth 2, sampled after this family settled',
          engineFailed: ready.failed === true,
          engineError: ready.failed ? ready.message : undefined,
        },
        checkpoints,
        args,
      );
    } catch (error) {
      await shot(page, args.screenshotDir, `fail-analysis-${family}.png`);
      checkpoints.push({ ...meta(`analysis-cpu-${family}`, family), ok: false, error: error?.message ?? String(error) });
      log(`analysis ${family} failed, continuing: ${error?.message ?? error}`);
    }
  }

  try {
    const previousMessage = await page.evaluate(() => document.querySelector('#message')?.textContent ?? '');
    await setAnalysisFamily(page, 'lc0', 8);
    await page.evaluate(() => document.querySelector('#analyze')?.click());
    const ready = await waitFor(page, analysisSampleReady, { family: 'lc0', previousMessage }, args.timeoutMs, 'return to Lc0');
    await stopAnalysis(page, 15_000);
    await record(
      page,
      {
        ...meta('analysis-after-cpu', 'lc0'),
        note: ready.failed ? `return to Lc0 failed: ${ready.message}` : 'row switched back to Lc0 after every CPU family',
        engineFailed: ready.failed === true,
        engineError: ready.failed ? ready.message : undefined,
      },
      checkpoints,
      args,
    );
  } catch (error) {
    checkpoints.push({ ...meta('analysis-after-cpu', 'lc0'), ok: false, error: error?.message ?? String(error) });
  }
  return checkpoints;
}

async function runArena(page, args, run) {
  const checkpoints = [];
  const meta = (id, resident = null) => ({ id, phase: 'arena', surface: 'arena', resident, run });
  const query = '/app/arena/?seatA=sf:lite:2&seatB=reckless::2&games=1&budget=movetime&movetime=200&openings=start&delay=0&sfThreads=1';
  await gotoPath(page, args.baseUrl, query);
  await waitFor(
    page,
    () => {
      const families = [...document.querySelectorAll('.seat-fam')].map((select) => select.value);
      const start = document.querySelector('#start');
      return { ok: families[0] === 'sf' && families[1] === 'reckless' && !!start && !start.disabled, families, disabled: start?.disabled ?? null };
    },
    null,
    args.timeoutMs,
    'arena CPU seats ready',
  );
  await record(page, { ...meta('arena-shell'), note: 'Stockfish Lite vs Reckless configured, match not started' }, checkpoints, args);
  try {
    await page.evaluate(() => document.querySelector('#start')?.click());
    const ready = await waitFor(
      page,
      () => {
        const moves = document.querySelector('#gameMoves')?.innerText?.trim() ?? '';
        const score = document.querySelector('#matchScore')?.textContent?.trim() ?? '';
        const message = document.querySelector('#message')?.textContent ?? '';
        const started = moves.length > 0 || (score.length > 0 && !/No games played yet/.test(score));
        const failed = /failed|error|unavailable/i.test(message);
        return { ok: started || failed, moves: moves.slice(0, 80), score: score.slice(0, 80), message: message.slice(0, 120) };
      },
      null,
      args.timeoutMs,
      'arena match to make a move',
    );
    const failure = checkpointEngineFailure(ready.message);
    await record(
      page,
      {
        ...meta('arena-cpu-match', 'sf+reckless'),
        note: failure.engineFailed ? `match failed: ${failure.engineError}` : '200 ms movetime, 1 game, sampled once a move or a result exists',
        ...failure,
      },
      checkpoints,
      args,
    );
  } catch (error) {
    await shot(page, args.screenshotDir, 'fail-arena-match.png');
    checkpoints.push({ ...meta('arena-cpu-match', 'sf+reckless'), ok: false, error: error?.message ?? String(error) });
  }
  try {
    await page.evaluate(() => document.querySelector('#stop')?.click());
    await delay(750);
    await record(page, { ...meta('arena-cpu-stopped'), note: 'Stop clicked after the CPU match sample' }, checkpoints, args);
  } catch (error) {
    checkpoints.push({ ...meta('arena-cpu-stopped'), ok: false, error: error?.message ?? String(error) });
  }
  return checkpoints;
}

async function runMaia(page, args, run) {
  const checkpoints = [];
  const meta = (id, surface, resident = null) => ({ id, phase: 'maia', surface, resident, run });
  await gotoPath(page, args.baseUrl, '/app/analysis/');
  await disableAutoAnalyze(page);
  await waitFor(page, analysisControllerReady, null, args.timeoutMs, 'analysis controller');
  await stopAnalysis(page, 10_000);
  await record(page, { ...meta('maia-before', 'analysis'), note: 'analysis shell before Load Maia3; auto-analyze unchecked' }, checkpoints, args);
  try {
    await waitFor(page, requestMaiaLoad, null, args.timeoutMs, 'Maia load click');
    const loaded = await waitFor(
      page,
      () => {
        const grid = document.querySelector('#maia3Grid');
        const status = document.querySelector('#maia3PanelStatus')?.textContent ?? '';
        const button = document.querySelector('#maia3Enable');
        const text = grid?.innerText?.trim() ?? '';
        const failed = /failed|error/i.test(status);
        const modelLoaded = !!button?.hidden && !!grid && grid.hidden === false;
        return { ok: text.length > 0 || failed || modelLoaded, status, hidden: grid?.hidden ?? null, text: text.slice(0, 80) };
      },
      null,
      args.timeoutMs,
      'Maia model',
    );
    const painted = await waitFor(
      page,
      () => {
        const text = document.querySelector('#maia3Grid')?.innerText?.trim() ?? '';
        const status = document.querySelector('#maia3PanelStatus')?.textContent ?? '';
        return { ok: text.length > 0 || /failed|error/i.test(status), text: text.slice(0, 80), status };
      },
      null,
      30_000,
      'Maia grid',
    ).catch(() => null);
    const status = painted?.status || loaded.status || '';
    const failure = checkpointEngineFailure(status);
    const gridNote = failure.engineFailed
      ? `Maia3 failed: ${failure.engineError}`
      : painted?.text
        ? 'Maia3 model loaded and the rating grid painted'
        : painted
          ? `Maia3 load reported: ${painted.status}`
          : 'Maia3 model loaded; the rating grid had not painted within 30s, so this sample is the loaded evaluator';
    await shot(page, args.screenshotDir, `maia-panel-run${run}.png`);
    await record(page, { ...meta('maia-panel', 'analysis', 'maia3'), note: gridNote, ...failure }, checkpoints, args);
  } catch (error) {
    await shot(page, args.screenshotDir, 'fail-maia.png');
    checkpoints.push({ ...meta('maia-panel', 'analysis', 'maia3'), ok: false, error: error?.message ?? String(error) });
  }
  try {
    await clientNavigate(page, '/', args.timeoutMs);
    await waitFor(page, () => ({ ok: !!document.querySelector('h1') }), null, args.timeoutMs, 'landing after maia');
    await record(page, { ...meta('landing-after-maia', 'landing'), note: 'in-app navigation away from the Maia panel' }, checkpoints, args);
  } catch (error) {
    checkpoints.push({ ...meta('landing-after-maia', 'landing'), ok: false, error: error?.message ?? String(error) });
  }
  return checkpoints;
}

async function runOverlay(page, args) {
  await gotoPath(page, args.baseUrl, '/?memprobe=1');
  await waitFor(page, () => ({ ok: !!document.querySelector('#memprobe') }), null, args.timeoutMs, 'memprobe overlay');
  await shot(page, args.screenshotDir, 'memprobe-overlay.png');
  await page.evaluate(() => document.querySelector('#memprobe button')?.click());
  await waitFor(
    page,
    () => {
      const text = document.querySelector('#memprobe pre')?.textContent ?? '';
      return { ok: /UA |unavailable|not cross-origin/.test(text), text: text.slice(0, 200) };
    },
    null,
    args.timeoutMs,
    'memprobe sample text',
  );
  await shot(page, args.screenshotDir, 'memprobe-sample.png');
  return page.evaluate(() => ({
    text: document.querySelector('#memprobe pre')?.textContent ?? '',
    isolated: crossOriginIsolated,
    hooked: typeof globalThis.__memprobe?.sample === 'function',
  }));
}

async function runPhase(args) {
  if (!args.out) throw new Error('Missing --out for a measurement run');
  const { chromium } = await import('playwright');
  const files = [];
  for (let offset = 0; offset < args.runs; offset += 1) {
    const run = args.runs === 1 ? args.runIndex : offset + 1;
    const startedAt = new Date().toISOString();
    const loadStart = loadavg();
    log(`start run ${run} phase ${args.phase}`);
    const session = await openSession(chromium, args);
    const browserVersion = session.browser.version();
    let checkpoints = [];
    let overlay = null;
    try {
      if (args.phase === 'play') checkpoints = await runPlay(session.page, args, run);
      else if (args.phase === 'analysis') checkpoints = await runAnalysis(session.page, args, run);
      else if (args.phase === 'arena') checkpoints = await runArena(session.page, args, run);
      else if (args.phase === 'maia') checkpoints = await runMaia(session.page, args, run);
      else overlay = await runOverlay(session.page, args);
    } finally {
      await session.context.close().catch(() => undefined);
      await session.browser.close().catch(() => undefined);
    }
    const artifact = {
      schema: SCHEMA,
      kind: 'phase-run',
      run,
      phase: args.phase,
      baseUrl: args.baseUrl,
      startedAt,
      finishedAt: new Date().toISOString(),
      durationMs: Date.now() - Date.parse(startedAt),
      loadavg: { start: loadStart, end: loadavg() },
      host: { ...hostInfo(), browser: browserVersion },
      pageErrors: session.errors.slice(0, 40),
      checkpoints,
      overlay,
    };
    const out = args.runs === 1 ? args.out : args.out.replace(/\.json$/i, '') + `-run${run}.json`;
    await mkdir(dirname(out), { recursive: true });
    await writeFile(out, `${JSON.stringify(artifact, null, 2)}\n`);
    files.push(out);
    log(`wrote ${out} in ${Math.round(artifact.durationMs / 1000)}s`);
  }
  return files;
}

export function combineArtifacts(artifacts, context = {}) {
  const runs = artifacts
    .filter((artifact) => artifact?.kind === 'phase-run' && artifact.phase !== 'overlay')
    .map((artifact) => ({
      run: artifact.run,
      phase: artifact.phase,
      checkpoints: artifact.checkpoints ?? [],
    }));
  const summary = summarizeMeasurements(runs);
  const hosts = artifacts.map((artifact) => artifact.host).filter(Boolean);
  const loadStarts = artifacts.flatMap((artifact) => artifact.loadavg?.start ?? []);
  summary.generatedAt = context.generatedAt ?? new Date().toISOString();
  summary.baseUrl = artifacts.find((artifact) => artifact.baseUrl)?.baseUrl ?? null;
  summary.host = hosts[0] ?? null;
  summary.loadavg = loadStarts.length ? { min: Math.min(...loadStarts.map(Number)), max: Math.max(...loadStarts.map(Number)), samples: loadStarts } : null;
  summary.phaseDurations = artifacts
    .filter((artifact) => artifact?.kind === 'phase-run')
    .map((artifact) => ({ run: artifact.run, phase: artifact.phase, durationMs: artifact.durationMs ?? null }));
  return summary;
}

async function combineDir(args) {
  const names = (await readdir(args.combine)).filter((name) => name.endsWith('.json')).sort();
  if (!names.length) throw new Error(`No JSON files in ${args.combine}`);
  const artifacts = [];
  for (const name of names) {
    const parsed = JSON.parse(await readFile(join(args.combine, name), 'utf8'));
    if (parsed?.schema !== SCHEMA) throw new Error(`${name} is not a ${SCHEMA} artifact`);
    artifacts.push(parsed);
  }
  const summary = combineArtifacts(artifacts);
  const markdown = renderMarkdown(summary);
  if (args.summaryJson) {
    await mkdir(dirname(args.summaryJson), { recursive: true });
    await writeFile(args.summaryJson, `${JSON.stringify(summary, null, 2)}\n`);
  }
  if (args.summaryMd) {
    await mkdir(dirname(args.summaryMd), { recursive: true });
    await writeFile(args.summaryMd, markdown);
  }
  if (!args.summaryJson && !args.summaryMd) process.stdout.write(markdown);
  else log(`combined ${artifacts.length} files into ${args.summaryJson ?? args.summaryMd}`);
  return summary;
}

export function dryRunReport(args) {
  const plan = checkpointPlan();
  return {
    schema: SCHEMA,
    status: 'MEMORY_HARNESS_DRY_RUN',
    baseUrl: args.baseUrl,
    runs: args.runs,
    phase: args.phase,
    phases: {
      play: plan.play.map((entry) => entry.id),
      analysis: plan.analysis.map((entry) => entry.id),
      arena: plan.arena.map((entry) => entry.id),
      maia: plan.maia.map((entry) => entry.id),
    },
    retentionComparisons: retentionComparisons(plan).map((entry) => entry.id),
  };
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.dryRun) {
    process.stdout.write(`${JSON.stringify(dryRunReport(args), null, 2)}\n`);
    return;
  }
  if (args.combine) {
    await combineDir(args);
    return;
  }
  await runPhase(args);
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((error) => {
    process.stderr.write(`${error?.stack ?? error}\n`);
    process.exitCode = 1;
  });
}
