// One owned browser session / cold origin / real route journey. Shared matrix OS and proxy helpers.
import { execFileSync } from 'node:child_process';
import { existsSync, readFileSync, realpathSync, unlinkSync, writeFileSync } from 'node:fs';
import { hostname, loadavg, platform, release } from 'node:os';
import { median } from '../lib/perf_canary_stats.mjs';
import { pgrep, readFootprint, sleep, WORKER_READY_SCRIPT } from '../perf_matrix/common.mjs';
import { startCountingProxy } from '../perf_matrix/proxy.mjs';
import {
  browserIdentity,
  chromeFootprint,
  footprintSum,
  incompleteJsTransfers,
  isJsTransfer,
  jsBytesByRoute,
  naturalVisibilityValid,
  peakFootprint,
  platformProblem,
  requestFailures,
  safariWebContentPid,
} from './gate.mjs';
import { DOCS_READY, MIDGAME, routeFor, runJourney, waitFor } from './journeys.mjs';
import { checkSessionCancellation, cleanupMarker, safariCreationRefused, shutdownDelay, waitForCleanup, watchShutdown } from './lifecycle.mjs';
import { LINUX_CHROME_ARGS, LINUX_LIMITATION, linuxFootprintReader, warmWebGpu, webgpuBackendProblem } from './linux.mjs';

const [specFile, outFile] = process.argv.slice(2);
const spec = JSON.parse(readFileSync(specFile, 'utf8'));
const probe = `(() => {${WORKER_READY_SCRIPT}})();\n${readFileSync(new URL('./probe.js', import.meta.url), 'utf8')}`;
const linux = platform() === 'linux';
// The Linux reader keeps per-PID sampled peaks, so it is created once per session.
const readChromeFootprint = linux ? linuxFootprintReader() : readFootprint;
const result = {
  ...spec,
  platform: platform(),
  at: new Date().toISOString(),
  host: { hostname: hostname(), platform: platform(), release: release(), loadBefore: loadavg() },
  status: 'invalid',
  metrics: {},
  samples: [],
  limitations: [],
  externalRequests: [],
};
const { signal, dispose } = watchShutdown();
const pause = (ms) => shutdownDelay(ms, signal);
const guarded =
  (fn) =>
  async (...args) => {
    signal.throwIfAborted();
    const value = await fn(...args);
    signal.throwIfAborted();
    return value;
  };
let proxy, browser, sid, cdp, page, evaluate, navigate, close, sampling, safariPid;
const consoleWarnings = [];
const routeMarks = [{ route: routeFor(spec.journey), at: Date.now() }];
result.routeMarks = routeMarks;
const mark = (value) => {
  if (value === 'route-ready') routeMarks.at(-1).readyAt = Date.now();
  else {
    phase = value;
    if (value.startsWith('route:')) routeMarks.push({ route: value.slice(6), at: Date.now() });
  }
};
const checkJsDeadline = (requests) => {
  // Copy at the cutoff: proxy records keep mutating while diagnostics and shutdown run.
  result.incompleteJsTransfers = incompleteJsTransfers(requests).map((r) => ({ ...r }));
  if (result.incompleteJsTransfers.length) throw Error('Required JS did not finish before the measurement deadline; see incompleteJsTransfers');
};
let awayAt;
const wc = 'com.apple.WebKit.WebContent';
const pendingCleanup = cleanupMarker(outFile);
const markBrowserStarting = () => {
  checkSessionCancellation(outFile, signal);
  writeFileSync(pendingCleanup, `${spec.browser} session process ${process.pid}\n`, { flag: 'wx', mode: 0o600 });
};
let phase = 'journey',
  busy = false,
  samplingError;
const t0 = Date.now();
async function bounded(promise, label, ms = 15000) {
  let timer;
  try {
    return await Promise.race([
      promise,
      new Promise((_, reject) => {
        timer = setTimeout(() => reject(Error(`${label} timed out after ${ms}ms`)), ms);
      }),
    ]);
  } finally {
    clearTimeout(timer);
  }
}
async function wd(method, path, body) {
  const r = await fetch(spec.driver + path, {
    method,
    redirect: 'error',
    headers: { 'content-type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
    signal: AbortSignal.timeout(120000),
  });
  const data = await r.json();
  if (!r.ok || data.value?.error) {
    throw Object.assign(Error(`${method} ${path}: ${r.status} ${JSON.stringify(data.value)?.slice(0, 300)}`), {
      webDriverError: data.value?.error,
    });
  }
  return data.value;
}
const quantile = (xs, q) => (xs.length ? [...xs].sort((a, b) => a - b)[Math.ceil(xs.length * q) - 1] : 0);
async function chromePids() {
  const { processInfo } = await bounded(cdp.send('SystemInfo.getProcessInfo'), 'Chrome process discovery');
  return {
    rendererPids: processInfo.filter((p) => p.type === 'renderer').map((p) => p.id),
    gpuPids: processInfo.filter((p) => ['GPU', 'gpu-process'].includes(p.type)).map((p) => p.id),
  };
}
async function sample() {
  if (busy) return;
  busy = true;
  const sampledPhase = phase;
  try {
    const { render, gpu } = cdp
      ? await chromeFootprint(chromePids, readChromeFootprint)
      : { render: footprintSum([safariPid], readFootprint([safariPid], { reset: true }), 'Safari WebContent'), gpu: null };
    result.samples.push({
      ms: Date.now() - t0,
      phase: sampledPhase,
      rendererMb: render.cur,
      gpuMb: gpu?.cur ?? null,
      rendererIntervalMb: render.interval,
      gpuIntervalMb: gpu?.interval ?? null,
      ...(gpu?.vram === undefined ? {} : { gpuVramMb: gpu.vram, gpuVramPeakMb: gpu.vramPeak }),
    });
  } catch (e) {
    samplingError ??= e;
  } finally {
    busy = false;
  }
}
function entryUrl() {
  return `${proxy.url}/app/${routeFor(spec.journey)}/${spec.journey.startsWith('analysis') ? `?fen=${encodeURIComponent(MIDGAME)}` : ''}`;
}
// A cohort keeps one browser build (compareRuns refuses a mix). Check right after launch, before
// measuring, and keep a mismatched identity out of browserVersion so the row cannot poison the cohort;
// the coordinator stops and a resume re-runs the cell once the cohort's browser is restored.
function claimBrowser(browserVersion, browserExecutable) {
  const launched = browserExecutable ? { browserVersion, browserExecutable } : { browserVersion };
  if (spec.expectedBrowser && browserIdentity(launched) !== spec.expectedBrowser) {
    result.browserMismatch = browserIdentity(launched);
    throw Error(`Browser build changed mid-cohort: cohort ran ${spec.expectedBrowser}, this launch is ${result.browserMismatch}`);
  }
  Object.assign(result, launched);
}
async function uaMemory(label) {
  if (spec.browser !== 'chrome') return;
  const value = await evaluate(`(async () => {
    if (!performance.measureUserAgentSpecificMemory) return {error:'unsupported'};
    return Promise.race([performance.measureUserAgentSpecificMemory().then(v => ({bytes:v.bytes})).catch(e => ({error:String(e)})),new Promise(r => setTimeout(() => r({error:'timeout 5s'}),5000))]);
  })()`);
  result.uaMemory ??= {};
  result.uaMemory[label] = value;
  if (Number.isFinite(value?.bytes)) result.metrics[`memory.ua.${label}Mb`] = value.bytes / 1048576;
}
try {
  checkSessionCancellation(outFile, signal);
  const unsupported = platformProblem(spec);
  if (unsupported) throw Error(unsupported);
  if (linux) result.limitations.push(LINUX_LIMITATION);
  proxy = await startCountingProxy(spec.base, { htmlScript: probe });
  signal.throwIfAborted();
  if (spec.browser === 'chrome') {
    const { chromium } = await import('playwright');
    signal.throwIfAborted();
    markBrowserStarting();
    browser = await chromium
      .launch({
        // Linux: Playwright's Chromium in new-headless mode (not headless-shell); PERF_ROUTES_CHROMIUM
        // points at a specific build when the installed Playwright expects a revision that is absent.
        ...(linux ? { channel: 'chromium', executablePath: process.env.PERF_ROUTES_CHROMIUM || undefined } : { channel: 'chrome' }),
        headless: true,
        handleSIGINT: false,
        handleSIGTERM: false,
        args: ['--enable-blink-features=ForceEagerMeasureMemory', ...(linux ? LINUX_CHROME_ARGS : [])],
      })
      .catch((error) => {
        // Playwright closes a failed launch before rejecting.
        unlinkSync(pendingCleanup);
        throw error;
      });
    close = () => browser.close();
    claimBrowser(browser.version(), linux ? realpathSync(process.env.PERF_ROUTES_CHROMIUM || chromium.executablePath()) : undefined);
    cdp = await browser.newBrowserCDPSession();
    if (linux) result.webgpuWarmup = await warmWebGpu(browser, async () => !(await chromePids()).rendererPids.length, { signal });
    const context = await browser.newContext({ viewport: { width: 1280, height: 900 } });
    context.on('request', (request) => {
      const url = new URL(request.url());
      if (/^https?:$/.test(url.protocol) && url.origin !== proxy.url) result.externalRequests.push(url.href);
    });
    page = await context.newPage();
    page.on('dialog', (dialog) => dialog.accept()); // only local journey leave confirmations
    // Main-thread ORT sessions (Arena's Lc0) post no worker ready message; their console line names the provider.
    result.ortSessions = [];
    page.on('console', (message) => {
      const text = message.text();
      const ready = /^(?:Centipawn )?ORT: session ready with (WebGPU|WASM) provider/.exec(text);
      if (ready) result.ortSessions.push(ready[1].toLowerCase());
      if (linux && message.type() === 'warning') consoleWarnings.push(text);
    });
    page.setDefaultTimeout(120000);
    evaluate = guarded((expression) => bounded(page.evaluate(expression), 'Chrome page evaluation'));
    navigate = guarded((url) => page.goto(url, { waitUntil: 'domcontentloaded', timeout: 90000 }));
    sampling = setInterval(sample, 100);
    await sample();
  } else {
    // Never create/delete another owner's WebDriver session or change JSC flags.
    result.jscFlags = Object.fromEntries(
      ['__XPC_JSC_useOMGJIT', '__XPC_JSC_useBBQJIT', '__XPC_JSC_useJIT'].map((key) => {
        try {
          return [key, execFileSync('launchctl', ['getenv', key], { encoding: 'utf8' }).trim()];
        } catch {
          return [key, ''];
        }
      }),
    );
    if (Object.values(result.jscFlags).some(Boolean)) throw Error('Normal-JSC cohort refuses launchctl overrides');
    const before = new Set(pgrep(wc, { strict: true }));
    signal.throwIfAborted();
    markBrowserStarting();
    const session = await wd('POST', '/session', { capabilities: { alwaysMatch: { browserName: 'safari' } } }).catch((error) => {
      // A refusal creates no session. A lost reply may hide an owned session.
      if (safariCreationRefused(error)) unlinkSync(pendingCleanup);
      throw error;
    });
    sid = session?.sessionId;
    if (typeof sid !== 'string' || !sid) throw Error('Safari did not return a session ID');
    close = () =>
      wd('DELETE', `/session/${encodeURIComponent(sid)}`).catch((error) => {
        if (error.webDriverError !== 'invalid session id') throw error;
      });
    writeFileSync(pendingCleanup, `Owned Safari session ${sid} at ${spec.driver}\n`);
    signal.throwIfAborted();
    claimBrowser(session.capabilities?.browserVersion);
    await wd('POST', `/session/${sid}/timeouts`, { pageLoad: 90000, script: 120000 });
    await wd('POST', `/session/${sid}/window/rect`, { width: 1280, height: 900 });
    // Safari's automation window can be occluded by other mini apps: background rAF and
    // auto-analysis are paused. Foreground Safari only after acquiring our owned session.
    execFileSync('osascript', ['-e', 'tell application "Safari" to activate']);
    evaluate = guarded((expression) => wd('POST', `/session/${sid}/execute/sync`, { script: `return (${expression});`, args: [] }));
    navigate = guarded((url) => wd('POST', `/session/${sid}/url`, { url }));
    result.limitations.push(
      'Safari longtask and UA memory APIs unavailable if absent; worker counts are main-realm-created workers, not nested workers. GPU process is shared and deliberately not attributed.',
    );
    await navigate(entryUrl());
    const after = pgrep(wc, { strict: true });
    result.candidatePids = after.filter((pid) => !before.has(pid));
    safariPid = safariWebContentPid(before, after);
    // Navigation may already have allocated before discovery. A fresh process's lifetime high-water covers that gap.
    result.navigationPeakMb = footprintSum([safariPid], readFootprint([safariPid]), 'Safari WebContent').life;
    result.pid = safariPid;
    sampling = setInterval(sample, 100);
    await sample();
  }
  if (spec.browser === 'chrome') await navigate(entryUrl());
  await waitFor(evaluate, `!!window.__routePerf && window.__routePerf.boardInteractiveMs !== null`);
  result.context = await evaluate(`({secure:isSecureContext,isolated:crossOriginIsolated,ua:navigator.userAgent,visibility:document.visibilityState})`);
  if (result.context.visibility !== 'visible') throw Error('Benchmark page is hidden: auto-analysis/input timing would be invalid');
  if (!result.context.secure || !result.context.isolated) throw Error('Page is not secure and cross-origin isolated');
  result.evidence = await runJourney(evaluate, spec.journey, mark, pause);
  result.probe = await evaluate(`JSON.parse(JSON.stringify(window.__routePerf))`);
  result.workerReady = await evaluate(`window.__perfWorkerReady`);
  if (linux) {
    const problem = webgpuBackendProblem(spec.journey, result.workerReady, result.ortSessions, consoleWarnings);
    if (problem) throw Error(`${problem}: CPU numbers would stand in for the Linux WebGPU datapoint`);
  }
  result.transitions = await evaluate(`window.__analysisTransitionLog?.() ?? []`);
  const p = result.probe;
  result.metrics.boardInteractiveMs = p.boardInteractiveMs;
  result.metrics.firstLineMs = p.firstLineMs;
  result.metrics['workers.peak'] = p.peakWorkers;
  for (const [kind, rates] of Object.entries(p.rates)) result.metrics[`speed.${kind}`] = median(rates);
  if (p.supportsLongTasks) {
    const tasks = p.longTasks.filter((t) => t.analysis).map((t) => t.ms);
    result.metrics.longTaskCount = tasks.length;
    result.metrics.longTaskTotalMs = tasks.reduce((a, b) => a + b, 0);
    result.metrics.longTaskP95Ms = quantile(tasks, 0.95);
  } else result.limitations.push('Long Tasks API unsupported: no Safari zeroes substituted.');
  if (result.evidence.moves.length) result.metrics.inputLatencyMs = median(result.evidence.moves.map((m) => m.latencyMs));
  // Freeze route bytes before navigation away; workers' JS glue is included, model/WASM bytes are not.
  result.requests = proxy.requests.map((r) => ({ ...r }));
  checkJsDeadline(result.requests);
  result.metrics.jsBytes = result.requests.filter(isJsTransfer).reduce((n, r) => n + r.bytes, 0);
  result.jsBytesByRoute = jsBytesByRoute(result.requests, routeMarks);
  for (const [route, bytes] of Object.entries(result.jsBytesByRoute)) result.metrics[`jsBytes.${route}`] = bytes;
  if (p.errors.length) throw Error(`Worker errors: ${p.errors.join('; ')}`);
  const active = result.samples.filter((s) => s.phase !== 'residual');
  result.metrics['memory.rendererPeakMb'] = Math.max(result.navigationPeakMb ?? 0, ...active.map((s) => s.rendererIntervalMb));
  if (cdp) result.metrics['memory.gpuPeakMb'] = Math.max(...active.map((s) => s.gpuIntervalMb));
  if (linux) result.metrics['memory.gpuVramPeakMb'] = Math.max(...active.map((s) => s.gpuVramPeakMb));
  // Total is the simultaneous sampled sum, NOT the sum of independent high-water marks.
  result.metrics['memory.peakMb'] = peakFootprint(spec.browser, active, result.navigationPeakMb ?? 0);
  const settled = active.filter((s) => s.ms >= active.at(-1).ms - 2000);
  result.metrics['memory.settledMb'] = median(settled.map((s) => s.rendererMb + (s.gpuMb ?? 0)));
  // Client-side navigation to a non-engine route observes retention without destroying the page process.
  const documentId = await evaluate(`window.__routePerf.documentId`);
  const away = '/docs/';
  if (spec.journey.startsWith('play')) {
    // The real leave confirmation is owned by this test flow. Avoid Safari unhandled-prompt deadlocks.
    await evaluate(`(window.confirm = () => true, true)`);
  }
  mark('route:docs');
  awayAt = routeMarks.at(-1).at;
  await evaluate(
    `(() => { const a=document.querySelector('.site-header nav.primary a[href="${away}"]'); if(!a) throw Error('Docs link missing'); a.click(); return true; })()`,
  );
  await waitFor(evaluate, DOCS_READY);
  if ((await evaluate(`window.__routePerf.documentId`)) !== documentId) throw Error('Residual navigation destroyed document');
  mark('route-ready');
  phase = 'residual';
  await pause(20000);
  clearInterval(sampling);
  while (busy) await sleep(10);
  await sample();
  if (samplingError) throw samplingError;
  const residual = result.samples.filter((s) => s.phase === 'residual' && s.ms >= result.samples.at(-1).ms - 2000);
  result.metrics['memory.residualMb'] = median(residual.map((s) => s.rendererMb + (s.gpuMb ?? 0)));
  result.metrics['workers.residual'] = await evaluate(`window.__routePerf.workers`);
  // Errors during client-side teardown matter even if a failed disposal terminates
  // the worker and leaves a superficially healthy zero-worker residual count.
  result.finalWorkerErrors = await evaluate(`[...window.__routePerf.errors]`);
  if (result.finalWorkerErrors.length) throw Error(`Worker errors through residual teardown: ${result.finalWorkerErrors.join('; ')}`);
  result.naturalVisibility = await evaluate(`({events:window.__routePerf.visibility,final:document.visibilityState})`);
  if (!naturalVisibilityValid(result.naturalVisibility.events, result.naturalVisibility.final))
    throw Error('Page became hidden during natural measurement windows; see naturalVisibility.');
  result.externalAssetUrls = [
    ...new Set([
      ...result.externalRequests,
      ...(await evaluate(`window.__routePerf.externalAssets`)),
      ...(await evaluate(
        `performance.getEntriesByType('resource').map(r=>r.name).filter(raw=>{const u=new URL(raw,location.href);return /^https?:$/.test(u.protocol)&&u.origin!==location.origin;})`,
      )),
    ]),
  ];
  if (result.externalAssetUrls.length)
    throw Error('Unobserved cross-origin assets bypass the JS counting proxy; stage same-origin assets before comparing (see externalAssetUrls).');
  result.networkCutoff = { at: Date.now(), requestCount: proxy.requests.length };
  checkJsDeadline(proxy.requests);
  // UA measurement triggers GC (eager with our Chrome flag). Never request it until
  // ALL natural footprint, worker, speed, input and long-task observations are frozen.
  await uaMemory('postResidualGc');
  result.status = 'ok';
} catch (error) {
  if (evaluate && !signal.aborted)
    result.diagnostic = await evaluate(
      `({url:location.href,title:document.title,message:document.querySelector('#message')?.textContent,body:document.body?.textContent?.slice(0,1000),probe:window.__routePerf ?? null,hook:typeof window.__analysisUserMove,board:!!document.querySelector('#ground cg-board'),visibility:document.visibilityState,scheduling:window.__analysisSchedulingState?.(),lines:[...document.querySelectorAll('#engineSummaries [data-pv-fen]')].map(el=>({fen:el.dataset.pvFen,engine:el.dataset.engine}))})`,
    ).catch((e) => ({ error: String(e) }));
  result.error = String(error.stack ?? error);
  console.error(result.error);
} finally {
  clearInterval(sampling);
  while (busy) await sleep(10);
  if (close) {
    try {
      await close();
      unlinkSync(pendingCleanup);
    } catch (error) {
      result.cleanupError = String(error);
      result.status = 'invalid';
    }
  }
  if (proxy) {
    await proxy.close();
    // The browser's close can supply the terminal event for requests still open at the cutoff.
    result.requests ??= proxy.requests;
    const cutoff = result.networkCutoff;
    if (awayAt !== undefined) result.postNavigationRequests = proxy.requests.slice(0, cutoff?.requestCount).filter((r) => r.at >= awayAt);
    if (cutoff) result.postMeasurementRequests = proxy.requests.slice(cutoff.requestCount);
    const failures = requestFailures(proxy.requests, spec.journey, { routeMarks, cutoff });
    result.optionalPrefetchFailures = failures.optional;
    result.teardownAborts = failures.teardownAborts;
    result.networkFailures = failures.unexpected;
    if (result.networkFailures.length && result.status === 'ok') {
      result.status = 'invalid';
      result.error = 'Failed or incomplete asset request; see networkFailures';
      console.error(result.error);
    }
  }
  if (signal.aborted) {
    result.status = 'invalid';
    result.error = signal.reason.message;
  }
  try {
    result.host.loadAfter = loadavg();
    result.durationMs = Date.now() - t0;
    writeFileSync(outFile, JSON.stringify(result, null, 2));
  } finally {
    if (spec.externalLock && existsSync(pendingCleanup)) {
      console.error(`Browser cleanup unconfirmed; holding benchlock. See ${pendingCleanup}`);
      await waitForCleanup(outFile, spec.browser === 'safari' ? close : undefined);
    }
    dispose();
  }
}
console.log(`${spec.arm}/${spec.browser}/${spec.journey}/r${spec.repeat}: ${result.status} ${JSON.stringify(result.metrics)}`);
process.exitCode = result.status === 'ok' ? 0 : /already paired/.test(result.error ?? '') ? 75 : 2;
