// One owned browser session / cold origin / real route journey. Shared matrix OS and proxy helpers.
import { execFileSync } from 'node:child_process';
import { readFileSync, writeFileSync } from 'node:fs';
import { hostname, loadavg, platform, release } from 'node:os';
import { median } from '../lib/perf_canary_stats.mjs';
import { pgrep, readFootprint, sleep, WORKER_READY_SCRIPT } from '../perf_matrix/common.mjs';
import { startCountingProxy } from '../perf_matrix/proxy.mjs';
import { jsBytesByRoute, peakFootprint } from './gate.mjs';
import { MIDGAME, routeFor, runJourney, waitFor } from './journeys.mjs';

const [specFile, outFile] = process.argv.slice(2);
const spec = JSON.parse(readFileSync(specFile, 'utf8'));
const probe = `(() => {${WORKER_READY_SCRIPT}})();\n${readFileSync(new URL('./probe.js', import.meta.url), 'utf8')}`;
const result = {
  ...spec,
  at: new Date().toISOString(),
  host: { hostname: hostname(), platform: platform(), release: release(), loadBefore: loadavg() },
  status: 'invalid',
  metrics: {},
  samples: [],
  limitations: [],
  externalRequests: [],
};
let proxy, browser, sid, cdp, page, evaluate, navigate, close, sampling;
const wc = 'com.apple.WebKit.WebContent';
let rendererPids = [],
  gpuPids = [],
  phase = 'journey',
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
    headers: { 'content-type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
    signal: AbortSignal.timeout(120000),
  });
  const data = await r.json();
  if (!r.ok) throw Error(`${method} ${path}: ${r.status} ${JSON.stringify(data.value).slice(0, 300)}`);
  return data.value;
}
const quantile = (xs, q) => (xs.length ? [...xs].sort((a, b) => a - b)[Math.ceil(xs.length * q) - 1] : 0);
async function sample() {
  if (busy) return;
  busy = true;
  try {
    if (cdp) {
      const procs = (await bounded(cdp.send('SystemInfo.getProcessInfo'), 'Chrome process discovery')).processInfo;
      rendererPids = procs.filter((p) => p.type === 'renderer').map((p) => p.id);
      gpuPids = procs.filter((p) => ['GPU', 'gpu-process'].includes(p.type)).map((p) => p.id);
    }
    const fp = readFootprint([...rendererPids, ...gpuPids], { reset: true });
    if (!rendererPids.length) return;
    const render = rendererPids.map((pid) => fp.get(pid)).filter(Boolean);
    if (!render.length) throw Error('Owned page process disappeared');
    const gpu = gpuPids.map((pid) => fp.get(pid)).filter(Boolean);
    const sum = (rows, key) => rows.reduce((n, v) => n + v[key], 0);
    result.samples.push({
      ms: Date.now() - t0,
      phase,
      rendererMb: sum(render, 'cur'),
      gpuMb: gpu.length ? sum(gpu, 'cur') : null,
      rendererIntervalMb: sum(render, 'interval'),
      gpuIntervalMb: gpu.length ? sum(gpu, 'interval') : null,
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
  if (platform() !== 'darwin') throw Error('OS footprint gate currently requires macOS pidfp; no RSS substitution');
  proxy = await startCountingProxy(spec.base, { htmlScript: probe });
  if (spec.browser === 'chrome') {
    const { chromium } = await import('playwright');
    browser = await chromium.launch({ channel: 'chrome', headless: true, args: ['--enable-blink-features=ForceEagerMeasureMemory'] });
    close = () => browser.close();
    cdp = await browser.newBrowserCDPSession();
    const context = await browser.newContext({ viewport: { width: 1280, height: 900 } });
    context.on('request', (request) => {
      const url = new URL(request.url());
      if (/^https?:$/.test(url.protocol) && url.origin !== proxy.url) result.externalRequests.push(url.href);
    });
    page = await context.newPage();
    page.on('dialog', (dialog) => dialog.accept()); // only local journey leave confirmations
    page.setDefaultTimeout(120000);
    evaluate = (expression) => bounded(page.evaluate(expression), 'Chrome page evaluation');
    navigate = (url) => page.goto(url, { waitUntil: 'domcontentloaded', timeout: 90000 });
    result.browserVersion = browser.version();
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
    const before = readFootprint(pgrep(wc));
    const session = await wd('POST', '/session', { capabilities: { alwaysMatch: { browserName: 'safari' } } });
    sid = session.sessionId;
    close = () => wd('DELETE', `/session/${sid}`);
    result.browserVersion = session.capabilities?.browserVersion;
    await wd('POST', `/session/${sid}/timeouts`, { pageLoad: 90000, script: 120000 });
    await wd('POST', `/session/${sid}/window/rect`, { width: 1280, height: 900 });
    // Safari's automation window can be occluded by other mini apps: background rAF and
    // auto-analysis are paused. Foreground Safari only after acquiring our owned session.
    execFileSync('osascript', ['-e', 'tell application "Safari" to activate']);
    evaluate = (expression) => wd('POST', `/session/${sid}/execute/sync`, { script: `return (${expression});`, args: [] });
    navigate = (url) => wd('POST', `/session/${sid}/url`, { url });
    result.limitations.push(
      'Safari longtask and UA memory APIs unavailable if absent; worker counts are main-realm-created workers, not nested workers. GPU process is shared and deliberately not attributed.',
    );
    await navigate(entryUrl());
    const after = readFootprint(pgrep(wc));
    const candidates = [...after].filter(([pid]) => !before.has(pid) || before.get(pid).cur < 40).sort((a, b) => b[1].cur - a[1].cur);
    result.candidatePids = candidates.map(([pid, v]) => [pid, v.cur]);
    if (!candidates.length || candidates.filter(([, v]) => v.cur > 40).length > 1) throw Error('Ambiguous/missing fresh Safari WebContent process');
    rendererPids = [candidates[0][0]];
    // Navigation may already have allocated before discovery. A fresh process's lifetime high-water covers that gap.
    result.navigationPeakMb = candidates[0][1].life;
    result.pid = rendererPids[0];
    sampling = setInterval(sample, 100);
    await sample();
  }
  if (spec.browser === 'chrome') await navigate(entryUrl());
  await waitFor(evaluate, `!!window.__routePerf && window.__routePerf.boardInteractiveMs !== null`);
  result.context = await evaluate(`({secure:isSecureContext,isolated:crossOriginIsolated,ua:navigator.userAgent,visibility:document.visibilityState})`);
  if (result.context.visibility !== 'visible') throw Error('Benchmark page is hidden: auto-analysis/input timing would be invalid');
  if (!result.context.secure || !result.context.isolated) throw Error('Page is not secure and cross-origin isolated');
  result.routeMarks = [{ route: routeFor(spec.journey), at: t0 }];
  result.evidence = await runJourney(evaluate, spec.journey, (value) => {
    phase = value;
    if (value.startsWith('route:')) result.routeMarks.push({ route: value.slice(6), at: Date.now() });
  });
  result.probe = await evaluate(`JSON.parse(JSON.stringify(window.__routePerf))`);
  result.workerReady = await evaluate(`window.__perfWorkerReady`);
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
  result.metrics.jsBytes = result.requests.filter((r) => /\.(m?js)(?:\?|$)/.test(r.path)).reduce((n, r) => n + r.bytes, 0);
  result.jsBytesByRoute = jsBytesByRoute(result.requests, result.routeMarks);
  for (const [route, bytes] of Object.entries(result.jsBytesByRoute)) result.metrics[`jsBytes.${route}`] = bytes;
  if (p.errors.length) throw Error(`Worker errors: ${p.errors.join('; ')}`);
  const active = result.samples.filter((s) => s.phase !== 'residual');
  result.metrics['memory.rendererPeakMb'] = Math.max(result.navigationPeakMb ?? 0, ...active.map((s) => s.rendererIntervalMb));
  if (gpuPids.length) result.metrics['memory.gpuPeakMb'] = Math.max(...active.map((s) => s.gpuIntervalMb ?? 0));
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
  await evaluate(
    `(() => { const a=document.querySelector('.site-header nav.primary a[href="${away}"]'); if(!a) throw Error('Docs link missing'); a.click(); return true; })()`,
  );
  await waitFor(evaluate, `location.pathname === '${away}'`);
  if ((await evaluate(`window.__routePerf.documentId`)) !== documentId) throw Error('Residual navigation destroyed document');
  phase = 'residual';
  await sleep(20000);
  await sample();
  const residual = result.samples.filter((s) => s.phase === 'residual' && s.ms >= result.samples.at(-1).ms - 2000);
  result.metrics['memory.residualMb'] = median(residual.map((s) => s.rendererMb + (s.gpuMb ?? 0)));
  result.metrics['workers.residual'] = await evaluate(`window.__routePerf.workers`);
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
  // UA measurement triggers GC (eager with our Chrome flag). Never request it until
  // ALL natural footprint, worker, speed, input and long-task observations are frozen.
  phase = 'ua-after-natural';
  await uaMemory('postResidualGc');
  if (samplingError) throw samplingError;
  const failed = result.requests.filter((r) => r.method !== 'HEAD' && r.status >= 400);
  // Adding a row speculatively prefetches the next suggested family before the user changes
  // it. Reckless is not selected in any gate journey; its absent optional local assets must
  // not invalidate four successfully searched engines. Keep those failures as evidence.
  result.optionalPrefetchFailures = failed.filter((r) => r.path.startsWith('/reckless/'));
  if (failed.some((r) => !r.path.startsWith('/reckless/'))) throw Error('Failed asset request; see requests');
  result.status = 'ok';
} catch (error) {
  if (evaluate)
    result.diagnostic = await evaluate(
      `({url:location.href,title:document.title,message:document.querySelector('#message')?.textContent,body:document.body?.textContent?.slice(0,1000),probe:window.__routePerf ?? null,hook:typeof window.__analysisUserMove,board:!!document.querySelector('#ground cg-board'),visibility:document.visibilityState})`,
    ).catch((e) => ({ error: String(e) }));
  if (proxy) result.requests = proxy.requests.map((r) => ({ ...r }));
  result.error = String(error.stack ?? error);
  console.error(result.error);
} finally {
  clearInterval(sampling);
  while (busy) await sleep(10);
  if (close)
    await close().catch((e) => {
      result.cleanupError = String(e);
      result.status = 'invalid';
    });
  if (proxy) await proxy.close();
  result.host.loadAfter = loadavg();
  result.durationMs = Date.now() - t0;
  writeFileSync(outFile, JSON.stringify(result, null, 2));
}
console.log(`${spec.arm}/${spec.browser}/${spec.journey}/r${spec.repeat}: ${result.status} ${JSON.stringify(result.metrics)}`);
process.exitCode = result.status === 'ok' ? 0 : /already paired/.test(result.error ?? '') ? 75 : 2;
