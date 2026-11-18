#!/usr/bin/env node
// Records dense screenshot timelines of real user journeys against a built preview, for
// vision-model UAT review of motion and state transitions (see README.md next to this file).
import { execFileSync, spawn } from 'node:child_process';
import { existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { createServer } from 'node:net';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { ExclusiveQueue } from '../../src/lc0/exclusiveQueue.ts';
import { parseScriptArgs } from '../lib/cli.mjs';
import { sleep } from '../perf_matrix/common.mjs';
import { startCountingProxy } from '../perf_matrix/proxy.mjs';
import { acquireLock, claimOutput, watchShutdown } from '../perf_routes/lifecycle.mjs';
import { JOURNEYS } from './journeys.mjs';
import {
  assembleTimeline,
  cadenceFor,
  chunk,
  collapseFrames,
  DEFAULT_VIEWPORTS,
  FrameStore,
  parseViewports,
  renderIndexHtml,
  renderJourneyHtml,
  renderSheetHtml,
  sheetLayout,
} from './lib.mjs';
import { PROBE_OPTIONS, probeScript } from './probe.mjs';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const USAGE = `Usage: node scripts/uat/record_journeys.mjs [options]

Drives the built preview through UAT journeys and records screenshot timelines.

Target (one of):
  --dist DIR            Serve a built preview with scripts/serve_isolated_static.mjs (default dist-client)
  --base-url URL        Already-served origin (e.g. http://127.0.0.1:5181)

Options:
  --browser NAME        chrome (Playwright + installed Chrome) or safari (safaridriver) [chrome]
  --driver URL          safaridriver endpoint [http://127.0.0.1:4444]
  --journeys LIST       Comma list or "all" [all]: ${Object.keys(JOURNEYS).join(', ')}
  --viewports LIST      WIDTHxHEIGHT list; < 700 px wide is emulated as a touch phone [${DEFAULT_VIEWPORTS}]
  --cadence MS          Fixed screenshot cadence [250]
  --burst-gap MS        Minimum gap between mutation-burst captures [100]
  --soft-cap N          Unique frames per run before cadence drops 4x [400]
  --hard-cap N          Unique frames per run before only step/burst captures remain [900]
  --format FMT          jpeg or png (Chrome; Safari is always png) [jpeg]
  --quality N           JPEG quality [70]
  --memory-class CLASS  Append ?memoryClass=phone|desktop to entry URLs (phone large-net policy)
  --video               Also record a Chrome screencast video per run
  --headed              Visible Chrome window
  --run-timeout MS      Abort one journey run after this long [900000]
  --lock-dir DIR        Hold this mutex directory while recording (shared with perf:routes)
  --out DIR             Output directory [.local-dev-artifacts/uat/<timestamp>]
  --sheets-only DIR     Re-render index pages and contact sheets for an existing output directory
  --dry-run             Print the plan without starting servers or browsers
  -h, --help            Show this help
`;

export function parseArgs(argv) {
  const args = parseScriptArgs(argv, {
    usage: USAGE,
    options: {
      dist: { type: 'string' },
      'base-url': { type: 'string' },
      browser: { type: 'string', default: 'chrome' },
      driver: { type: 'string', default: 'http://127.0.0.1:4444' },
      journeys: { type: 'string', default: 'all' },
      viewports: { type: 'string', default: DEFAULT_VIEWPORTS },
      cadence: { type: 'string', default: '250' },
      'burst-gap': { type: 'string', default: '100' },
      'soft-cap': { type: 'string', default: '400' },
      'hard-cap': { type: 'string', default: '900' },
      format: { type: 'string', default: 'jpeg' },
      quality: { type: 'string', default: '70' },
      'memory-class': { type: 'string' },
      video: { type: 'boolean', default: false },
      headed: { type: 'boolean', default: false },
      'run-timeout': { type: 'string', default: '900000' },
      'lock-dir': { type: 'string' },
      out: { type: 'string' },
      'sheets-only': { type: 'string' },
      'dry-run': { type: 'boolean', default: false },
    },
  });
  for (const key of ['cadence', 'burstGap', 'softCap', 'hardCap', 'quality', 'runTimeout']) {
    args[key] = Number(args[key]);
    if (!Number.isFinite(args[key]) || args[key] <= 0) throw Error(`Invalid --${key.replace(/[A-Z]/g, (c) => `-${c.toLowerCase()}`)}`);
  }
  if (!['chrome', 'safari'].includes(args.browser)) throw Error('--browser must be chrome or safari');
  if (!['jpeg', 'png'].includes(args.format)) throw Error('--format must be jpeg or png');
  if (args.memoryClass && !['phone', 'desktop'].includes(args.memoryClass)) throw Error('--memory-class must be phone or desktop');
  if (args.dist && args.baseUrl) throw Error('Use --dist or --base-url, not both');
  if (args.baseUrl) args.baseUrl = new URL(args.baseUrl).origin;
  const names = args.journeys === 'all' ? Object.keys(JOURNEYS) : args.journeys.split(',').map((s) => s.trim());
  for (const name of names) if (!JOURNEYS[name]) throw Error(`Unknown journey ${name}; known: ${Object.keys(JOURNEYS).join(', ')}`);
  args.journeyNames = names;
  args.viewportList = parseViewports(args.viewports);
  return args;
}

export function plan(args) {
  return args.journeyNames.flatMap((journey) =>
    args.viewportList.map((viewport) => ({ journey, viewport, dir: `${journey}--${viewport.name}`, entry: entryPath(journey, args.memoryClass) })),
  );
}

export function entryPath(journey, memoryClass) {
  // Append rather than re-serialize: URLSearchParams would rewrite the FEN's %20 as '+'.
  const entry = JOURNEYS[journey].entry;
  return memoryClass ? `${entry}${entry.includes('?') ? '&' : '?'}memoryClass=${memoryClass}` : entry;
}

async function bounded(promise, label, ms) {
  let timer;
  try {
    return await Promise.race([promise, new Promise((_, reject) => (timer = setTimeout(() => reject(Error(`${label} timed out after ${ms}ms`)), ms)))]);
  } finally {
    clearTimeout(timer);
  }
}

async function serveDist(dist, signal) {
  if (!existsSync(join(dist, 'app/analysis/index.html'))) throw Error(`Not a built preview (run npm run build:client): ${dist}`);
  const socket = createServer();
  await new Promise((r) => socket.listen(0, '127.0.0.1', r));
  const port = socket.address().port;
  await new Promise((r) => socket.close(r));
  const child = spawn(process.execPath, [join(root, 'scripts/serve_isolated_static.mjs'), dist, String(port)], {
    stdio: 'ignore',
    env: { ...process.env, PORT: String(port) },
  });
  const closed = new Promise((r) => child.once('close', r));
  const stop = async () => {
    child.kill('SIGTERM'); // only the PID we spawned
    await closed;
  };
  const base = `http://127.0.0.1:${port}`;
  for (let i = 0; i < 100; i++) {
    signal.throwIfAborted();
    const ok = await fetch(`${base}/app/analysis/`, { signal: AbortSignal.timeout(1000) })
      .then(async (r) => {
        await r.body?.cancel();
        return r.ok;
      })
      .catch(() => false);
    if (ok) return { base, stop };
    if (child.exitCode !== null) break;
    await sleep(100);
  }
  await stop();
  throw Error('Static server did not become ready');
}

async function launchSheetBrowser(existing) {
  if (existing) return { browser: existing, close: async () => {} };
  const { chromium } = await import('playwright');
  const browser = await chromium.launch({ channel: 'chrome', headless: true }).catch(() => chromium.launch({ headless: true }));
  return { browser, close: () => browser.close() };
}

async function chromeDriver(args) {
  const { chromium } = await import('playwright');
  const browser = await chromium.launch({ channel: 'chrome', headless: !args.headed, handleSIGINT: false, handleSIGTERM: false });
  return {
    name: 'chrome',
    version: browser.version(),
    browser,
    ext: args.format === 'png' ? 'png' : 'jpg',
    captureConsoleInPage: false, // CDP sees console errors, including browser-generated ones
    async open(viewport, runDir, { onEvent, onBurst }) {
      const context = await browser.newContext({
        viewport: { width: viewport.width, height: viewport.height },
        deviceScaleFactor: viewport.deviceScaleFactor,
        isMobile: viewport.mobile,
        hasTouch: viewport.mobile,
        ...(args.video ? { recordVideo: { dir: join(runDir, '.video'), size: { width: viewport.width, height: viewport.height } } } : {}),
      });
      const page = await context.newPage();
      await page.exposeBinding('__uatBurst', () => onBurst());
      page.on('console', (m) => {
        if (m.type() === 'error' || m.type() === 'warning') onEvent({ type: `console-${m.type()}`, text: m.text().slice(0, 600) });
      });
      page.on('pageerror', (e) => onEvent({ type: 'pageerror', text: String(e?.stack ?? e).slice(0, 600) }));
      page.on('dialog', (d) => {
        onEvent({ type: 'dialog', text: `${d.type()}: ${d.message()}` });
        d.accept().catch(() => {});
      });
      page.on('requestfailed', (r) => {
        const error = r.failure()?.errorText ?? 'failed';
        // Page-initiated aborts (route changes cancelling prefetches) are not transport failures.
        if (error !== 'net::ERR_ABORTED') onEvent({ type: 'request-failed', text: `${error} ${r.url()}` });
      });
      page.on('response', (r) => {
        if (r.status() >= 400) onEvent({ type: 'request-failed', text: `HTTP ${r.status()} ${r.url()}` });
      });
      page.on('crash', () => onEvent({ type: 'pageerror', text: 'Renderer crashed' }));
      const cdp = await context.newCDPSession(page);
      const shot = { format: args.format, ...(args.format === 'jpeg' ? { quality: args.quality } : {}), optimizeForSpeed: true };
      return {
        evaluate: (expression) => bounded(page.evaluate(expression), 'page evaluation', 15000),
        goto: (url) => page.goto(url, { waitUntil: 'domcontentloaded', timeout: 90000 }),
        screenshot: async () => Buffer.from((await bounded(cdp.send('Page.captureScreenshot', shot), 'screenshot', 15000)).data, 'base64'),
        click: (selector) => page.click(selector, { timeout: 5000 }),
        async close() {
          const video = page.video();
          await context.close();
          if (!video) return null;
          await video.saveAs(join(runDir, 'video.webm'));
          await video.delete();
          return 'video.webm';
        },
      };
    },
    close: () => browser.close(),
  };
}

const ELEMENT = 'element-6066-11e4-a52e-4f735466cecf';
async function safariDriver(args) {
  async function wd(method, path, body) {
    const r = await fetch(args.driver + path, {
      method,
      headers: { 'content-type': 'application/json' },
      body: body === undefined ? undefined : JSON.stringify(body),
      signal: AbortSignal.timeout(120000),
    });
    const data = await r.json();
    if (!r.ok || data.value?.error)
      throw Object.assign(Error(`${method} ${path}: ${r.status} ${JSON.stringify(data.value)?.slice(0, 300)}`), { webDriverError: data.value?.error });
    return data.value;
  }
  return {
    name: 'safari',
    version: null,
    browser: null,
    ext: 'png',
    captureConsoleInPage: true, // WebDriver has no console access
    async open(viewport, _runDir, { onEvent }) {
      const session = await wd('POST', '/session', { capabilities: { alwaysMatch: { browserName: 'safari' } } });
      const sid = session.sessionId;
      this.version = session.capabilities?.browserVersion ?? null;
      const s = `/session/${encodeURIComponent(sid)}`;
      // safaridriver handles one command at a time; the journey and the capture loop share this queue.
      const queue = new ExclusiveQueue();
      const call = (method, path, body) =>
        queue.run(async () => {
          try {
            return await wd(method, s + path, body);
          } catch (error) {
            if (error.webDriverError !== 'unexpected alert open') throw error;
            const text = await wd('GET', `${s}/alert/text`).catch(() => '');
            onEvent({ type: 'dialog', text: `alert: ${text}` });
            await wd('POST', `${s}/alert/accept`, {});
            return wd(method, s + path, body);
          }
        });
      await call('POST', '/timeouts', { pageLoad: 90000, script: 120000 });
      await call('POST', '/window/rect', { width: viewport.width, height: viewport.height });
      // An occluded automation window pauses rAF and auto-analysis (see perf_routes/session.mjs).
      execFileSync('osascript', ['-e', 'tell application "Safari" to activate']);
      const evaluate = (expression) => call('POST', '/execute/sync', { script: `return (${expression});`, args: [] });
      return {
        evaluate,
        goto: (url) => call('POST', '/url', { url }),
        screenshot: async () => Buffer.from(await call('GET', '/screenshot'), 'base64'),
        async click(selector) {
          const el = await call('POST', '/element', { using: 'css selector', value: selector });
          await call('POST', `/element/${encodeURIComponent(el[ELEMENT])}/click`, {});
        },
        // Window rect is the outer size; grow it by the chrome so the page viewport matches.
        async fit() {
          const inner = await evaluate('[innerWidth, innerHeight]');
          const rect = await call('GET', '/window/rect');
          await call('POST', '/window/rect', { width: rect.width + viewport.width - inner[0], height: rect.height + viewport.height - inner[1] });
          const actual = await evaluate('[innerWidth, innerHeight]');
          if (actual[0] !== viewport.width)
            onEvent({ type: 'warning', text: `Safari viewport is ${actual.join('x')}, not ${viewport.name} (window minimum size?)` });
        },
        async close() {
          await call('DELETE', '').catch((error) => {
            if (error.webDriverError !== 'invalid session id') throw error;
          });
          return null;
        },
      };
    },
    close: async () => {},
  };
}

async function recordRun(driver, base, item, args, outDir, signal) {
  const runDir = join(outDir, item.dir);
  mkdirSync(join(runDir, 'frames'), { recursive: true });
  const store = new FrameStore((file, buffer) => writeFileSync(join(runDir, file), buffer), { ext: driver.ext });
  const samples = [],
    steps = [],
    events = [];
  const startedAt = Date.now();
  const onEvent = (e) => events.push({ t: Date.now(), ...e });
  const queue = new ExclusiveQueue();
  let inFlight = 0,
    lastCaptureAt = 0,
    failed = false,
    session,
    proxy;
  const capture = (reason, label) => {
    inFlight++;
    return queue
      .run(async () => {
        const sample = { reason, ...(label ? { label } : {}) };
        try {
          const page = await session.evaluate('window.__uat ? window.__uat.sample() : null');
          if (page) Object.assign(sample, page, { sampledAt: page.t });
          else sample.error = 'probe unavailable (navigation in progress?)';
        } catch (error) {
          sample.error = `sample: ${String(error.message ?? error).slice(0, 300)}`;
        }
        sample.t = Date.now();
        try {
          sample.frame = store.add(await session.screenshot());
        } catch (error) {
          sample.error = `${sample.error ? `${sample.error}; ` : ''}screenshot: ${String(error.message ?? error).slice(0, 300)}`;
        }
        samples.push(sample);
      })
      .finally(() => {
        inFlight--;
        lastCaptureAt = Date.now();
      });
  };
  const onBurst = () => {
    const gap = store.unique >= args.softCap ? args.burstGap * 4 : args.burstGap;
    if (inFlight === 0 && store.unique < args.hardCap && Date.now() - lastCaptureAt >= gap) capture('burst').catch(() => {});
  };
  let recording = true;
  const ctx = {
    viewport: item.viewport,
    evaluate: (expression) => session.evaluate(expression),
    pause: (ms) => sleep(ms),
    note: (type, text) => onEvent({ type, text }),
    async click(selector) {
      try {
        await session.click(selector);
      } catch (error) {
        // A control the pointer cannot reach (covered, off-screen, zero-size) is itself a UAT finding.
        onEvent({ type: 'warning', text: `Pointer click on ${selector} failed (${String(error.message ?? error).split('\n')[0]}); used DOM click` });
        await session.evaluate(
          `(() => { const el = document.querySelector(${JSON.stringify(selector)}); if (!el || el.disabled) throw Error('Missing/disabled: ' + ${JSON.stringify(selector)}); el.click(); return true; })()`,
        );
      }
    },
    async step(label, fn) {
      if (failed) return;
      signal.throwIfAborted();
      const step = { label, t: Date.now(), status: 'running' };
      steps.push(step);
      await capture('step', label);
      try {
        await fn();
        step.status = 'ok';
      } catch (error) {
        step.status = 'failed';
        step.error = String(error.message ?? error).slice(0, 1000);
        failed = true;
      }
      step.endT = Date.now();
      await capture(step.status === 'ok' ? 'step-end' : 'failure', `${label}: ${step.status === 'ok' ? 'done' : 'FAILED'}`);
    },
  };
  let status = 'ok',
    error = null,
    video = null;
  try {
    proxy = await startCountingProxy(base, { htmlScript: probeScript({ ...PROBE_OPTIONS, captureConsole: driver.captureConsoleInPage }) });
    session = await driver.open(item.viewport, runDir, { onEvent, onBurst });
    await session.goto(proxy.url + item.entry);
    await session.fit?.();
    const loop = (async () => {
      while (recording) {
        const ms = cadenceFor(store.unique, { cadenceMs: args.cadence, softCap: args.softCap, hardCap: args.hardCap });
        await sleep(ms ?? args.cadence);
        if (recording && ms !== null && inFlight === 0) await capture('cadence').catch(() => {});
      }
    })();
    try {
      await bounded(JOURNEYS[item.journey].run(ctx), `${item.journey} run`, args.runTimeout).catch((e) => {
        failed = true; // stop a timed-out journey's remaining steps
        throw e;
      });
    } finally {
      failed ||= steps.some((s) => s.status === 'failed');
      await capture('final', 'end of journey').catch(() => {});
      recording = false;
      await loop;
      await queue.idle();
    }
    if (failed) status = 'failed';
  } catch (e) {
    status = 'failed';
    error = String(e.stack ?? e).slice(0, 2000);
    console.error(`[uat] ${item.dir}: ${error}`);
  } finally {
    recording = false;
    // A step still waiting when the run timed out must not keep capturing into a closed session.
    failed = true;
    if (session)
      video = await session.close().catch((e) => {
        onEvent({ type: 'warning', text: `close: ${e.message}` });
        return null;
      });
    if (proxy) await proxy.close();
  }
  const meta = {
    journey: item.journey,
    about: JOURNEYS[item.journey].about,
    viewport: item.viewport,
    browser: driver.name,
    browserVersion: driver.version,
    entry: item.entry,
    startedAt,
    durationMs: Date.now() - startedAt,
    cadenceMs: args.cadence,
    status,
    ...(error ? { error } : {}),
    ...(video ? { video } : {}),
    frameBytes: store.bytes,
  };
  const timeline = assembleTimeline({ meta, samples, steps, events });
  writeFileSync(join(runDir, 'timeline.json'), JSON.stringify(timeline, null, 1));
  return summarize(item.dir, timeline);
}

function summarize(dir, timeline) {
  const { meta, digest } = timeline;
  return {
    dir,
    journey: meta.journey,
    viewport: meta.viewport.name,
    status: meta.status,
    uniqueFrames: digest.uniqueFrames,
    overflowSpans: digest.overflowSpans.length,
    controlResizes: digest.controlResizes.length,
    errors: digest.errors.length,
    error: meta.error ?? digest.failedSteps.map((s) => `${s.label}: ${s.error}`).join('; '),
  };
}

/** Writes each run's index.html and contact-sheet images (rendered by Chromium from HTML grids). */
async function renderReview(outDir, dirs, browser) {
  const sheetBrowser = await launchSheetBrowser(browser).catch((error) => {
    console.error(`[uat] No Chromium for contact sheets (${error.message}); writing HTML pages only`);
    return null;
  });
  try {
    for (const dir of dirs) {
      const runDir = join(outDir, dir);
      const timeline = JSON.parse(readFileSync(join(runDir, 'timeline.json'), 'utf8'));
      const layout = sheetLayout(timeline.meta.viewport);
      const pages = chunk(collapseFrames(timeline.frames), layout.cols * layout.rows);
      const sheets = [];
      if (sheetBrowser) {
        mkdirSync(join(runDir, 'sheets'), { recursive: true });
        const page = await sheetBrowser.browser.newPage({ viewport: { width: layout.cols * (layout.thumbWidth + 10) + 24, height: 600 } });
        try {
          for (const [i, entries] of pages.entries()) {
            const name = `sheets/sheet-${String(i + 1).padStart(2, '0')}`;
            const title = `${timeline.meta.journey} · ${timeline.meta.viewport.name} · sheet ${i + 1}/${pages.length} · t=${(entries[0].t / 1000).toFixed(1)}–${(entries.at(-1).until / 1000).toFixed(1)}s`;
            writeFileSync(join(runDir, `${name}.html`), renderSheetHtml(entries, { title, ...layout }));
            await page.goto(pathToFileURL(join(runDir, `${name}.html`)).href, { waitUntil: 'load' });
            await page.screenshot({ path: join(runDir, `${name}.jpg`), fullPage: true, type: 'jpeg', quality: 80 });
            sheets.push(`${name}.jpg`);
          }
        } finally {
          await page.close();
        }
      }
      writeFileSync(join(runDir, 'index.html'), renderJourneyHtml(timeline, { sheets }));
    }
  } finally {
    await sheetBrowser?.close();
  }
}

function findRunDirs(outDir) {
  return readdirSync(outDir, { withFileTypes: true })
    .filter((d) => d.isDirectory() && existsSync(join(outDir, d.name, 'timeline.json')))
    .map((d) => d.name)
    .sort();
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.sheetsOnly) {
    const outDir = resolve(args.sheetsOnly);
    const dirs = findRunDirs(outDir);
    await renderReview(outDir, dirs);
    const runs = dirs.map((dir) => summarize(dir, JSON.parse(readFileSync(join(outDir, dir, 'timeline.json'), 'utf8'))));
    writeFileSync(join(outDir, 'index.html'), renderIndexHtml(runs));
    console.log(`[uat] re-rendered ${dirs.length} runs: ${join(outDir, 'index.html')}`);
    return;
  }
  const runs = plan(args);
  if (args.dryRun) {
    console.log(JSON.stringify({ browser: args.browser, target: args.baseUrl ?? resolve(args.dist ?? 'dist-client'), runs }, null, 2));
    return;
  }
  const { signal, dispose } = watchShutdown();
  const outDir = resolve(args.out ?? `.local-dev-artifacts/uat/${new Date().toISOString().replace(/[:.]/g, '-')}`);
  claimOutput(outDir);
  const release = args.lockDir ? await acquireLock(resolve(args.lockDir), signal) : () => {};
  let server, driver;
  const results = [];
  try {
    if (!args.baseUrl) server = await serveDist(resolve(args.dist ?? 'dist-client'), signal);
    const base = args.baseUrl ?? server.base;
    driver = args.browser === 'chrome' ? await chromeDriver(args) : await safariDriver(args);
    for (const [i, item] of runs.entries()) {
      signal.throwIfAborted();
      console.log(`[uat] [${i + 1}/${runs.length}] ${item.dir}`);
      const summary = await recordRun(driver, base, item, args, outDir, signal);
      console.log(`[uat]   ${summary.status}: ${summary.uniqueFrames} frames, ${summary.overflowSpans} overflow spans, ${summary.errors} errors`);
      results.push(summary);
      writeFileSync(join(outDir, 'run.json'), JSON.stringify({ args: { ...args, journeyNames: undefined }, runs: results }, null, 2));
    }
    await renderReview(
      outDir,
      results.map((r) => r.dir),
      driver.browser,
    );
    writeFileSync(join(outDir, 'index.html'), renderIndexHtml(results));
    console.log(`[uat] review index: ${join(outDir, 'index.html')}`);
    process.exitCode = results.every((r) => r.status === 'ok') ? 0 : 2;
  } finally {
    await driver?.close();
    await server?.stop();
    release();
    dispose();
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((error) => {
    console.error(error?.stack ?? error);
    process.exitCode = 1;
  });
}
