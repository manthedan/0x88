// One headless-Chrome browser instance (launched by agent-browser) = one or more measured loads.
//
//   node chrome_session.mjs <spec.json> <out.json>
//
// Same spec/plan vocabulary as safari_session.mjs. Differences:
//  - cold: a new browser instance + a new proxy port (new origin) => empty profile storage by construction.
//  - warm-fresh: closes the previous tab and opens a new one (new renderer process under site isolation).
//  - Memory is sampled for this instance's renderer(s) and its GPU process (pids from CDP
//    SystemInfo.getProcessInfo, so other Chrome instances on the machine are never counted).
//  - Chrome is launched with --enable-blink-features=ForceEagerMeasureMemory, so the soak page's own
//    performance.measureUserAgentSpecificMemory() samples (before-load, resident-after-load, after-search,
//    after-dispose) resolve immediately instead of waiting for a GC.
import { execFileSync } from 'node:child_process';
import { readFileSync, writeFileSync } from 'node:fs';
import { digestReport, PROGRESS_SCRIPT, phaseOf, readFootprint, round, sleep, soakUrl, WORKER_READY_SCRIPT } from './common.mjs';
import { startCountingProxy } from './proxy.mjs';

const [specPath, outPath] = process.argv.slice(2);
const spec = JSON.parse(readFileSync(specPath, 'utf8'));
const settleMs = spec.settleMs ?? 10_000;
const sampleMs = spec.sampleMs ?? 100;
const timeoutMs = spec.timeoutMs ?? 240_000;
const CODE = { load: 'l', search: 'r', dispose: 'd', settle: 't' };
const log = (...a) => console.log(new Date().toISOString().slice(11, 23), ...a);
const ab = (...args) => execFileSync('agent-browser', ['--session', session, ...args], { encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] });
const session = `perfmx_${process.pid}_${Date.now()}`;

class Cdp {
  constructor(url) {
    this.ws = new WebSocket(url);
    this.id = 0;
    this.pending = new Map();
    this.ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      const p = msg.id !== undefined && this.pending.get(msg.id);
      if (!p) return;
      this.pending.delete(msg.id);
      clearTimeout(p.timer);
      if (msg.error) p.reject(new Error(`${p.method}: ${msg.error.message}`));
      else p.resolve(msg.result);
    };
    this.ws.onclose = () => {
      for (const p of this.pending.values()) {
        clearTimeout(p.timer);
        p.reject(new Error(`${p.method}: owned Chrome connection closed`));
      }
      this.pending.clear();
    };
  }
  open() {
    return new Promise((resolve, reject) => {
      this.ws.onopen = resolve;
      this.ws.onerror = reject;
    });
  }
  send(method, params = {}, sessionId) {
    const id = ++this.id;
    this.ws.send(JSON.stringify({ id, method, params, ...(sessionId ? { sessionId } : {}) }));
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error(`${method}: CDP timed out`));
      }, 120000);
      this.pending.set(id, { resolve, reject, method, timer });
    });
  }
}

const proxy = await startCountingProxy(spec.base);
const args = ['--enable-blink-features=ForceEagerMeasureMemory', ...(spec.chromeArgs ?? [])].join(',');
// Native-tool callers may supply their own fresh, owned browser instead of invoking the CLI.
if (!spec.cdpUrl) ab('--args', args, 'open', 'about:blank');
const cdp = new Cdp(
  spec.cdpUrl ??
    ab('get', 'cdp-url')
      .trim()
      .split('\n')
      .filter((l) => l.startsWith('ws://'))
      .at(-1),
);
await cdp.open();
const processes = async () => (await cdp.send('SystemInfo.getProcessInfo')).processInfo;
const runs = [];
let previous = null; // { targetId, pid }
try {
  for (const [index, kind] of spec.plan.entries()) {
    if (kind === 'warm-same-tab') throw new Error('warm-same-tab is a Safari-only diagnostic');
    if (previous) {
      await cdp.send('Target.closeTarget', { targetId: previous.targetId });
      const until = Date.now() + 15_000;
      while ((await processes()).some((p) => p.id === previous.pid) && Date.now() < until) await sleep(250);
    }
    const run = await measure(kind, index);
    runs.push(run);
    previous = { targetId: run.targetId, pid: run.pid };
  }
} catch (error) {
  log(`session error: ${error.message}`);
  runs.push({
    browser: 'chrome',
    net: spec.net,
    params: spec.params ?? {},
    cache: spec.plan[runs.length] ?? 'unknown',
    sessionIndex: runs.length,
    report: { status: 'harness-error', error: String(error.message).slice(0, 300) },
  });
} finally {
  try {
    if (!spec.cdpUrl) ab('close');
    else await cdp.send('Browser.close'); // this instance was explicitly leased by the caller
  } catch {}
  cdp.ws.close();
  await proxy.close();
}
writeFileSync(outPath, JSON.stringify(runs, null, 1));

async function measure(kind, index) {
  const record = { browser: 'chrome', net: spec.net, params: spec.params ?? {}, cache: kind, sessionIndex: index, at: new Date().toISOString() };
  const before = new Set((await processes()).filter((p) => p.type === 'renderer').map((p) => p.id));
  const { targetId } = await cdp.send('Target.createTarget', { url: 'about:blank', newWindow: false });
  const { sessionId } = await cdp.send('Target.attachToTarget', { targetId, flatten: true });
  const evaluate = async (expression, awaitPromise = false) => {
    const r = await cdp.send('Runtime.evaluate', { expression, returnByValue: true, awaitPromise }, sessionId);
    if (r.exceptionDetails) throw new Error(r.exceptionDetails.exception?.description ?? r.exceptionDetails.text);
    return r.result.value;
  };
  const url = soakUrl(proxy.url, spec.net, spec.params ?? {});
  record.url = url;
  record.targetId = targetId;
  const navStarted = Date.now();
  await cdp.send('Page.navigate', { url }, sessionId);
  for (let i = 0; i < 160; i++) {
    const ok = await evaluate('!!(window.__mobileSoak && window.__mobileSoak.results && window.__mobileSoak.results.env)').catch(() => false);
    if (ok) break;
    await sleep(250);
  }
  // Which renderer hosts the page is decided at the end (largest peak among this instance's renderers):
  // Chrome may move the navigation into a renderer that did not exist when the tab was created.
  let procs = await processes();
  const renderers = new Set(procs.filter((p) => p.type === 'renderer').map((p) => p.id));
  const gpuPid = procs.find((p) => p.type === 'GPU' || p.type === 'gpu-process')?.id;
  record.newRenderers = [...renderers].filter((p) => !before.has(p));
  record.gpuPid = gpuPid;
  const refresh = setInterval(async () => {
    try {
      for (const p of await processes()) if (p.type === 'renderer') renderers.add(p.id);
    } catch {}
  }, 1000);
  const cache = await evaluate(
    `(async () => { const out = {}; for (const k of await caches.keys()) out[k] = (await (await caches.open(k)).keys()).length; try { localStorage.removeItem('0x88-mobile-soak-v1'); } catch {} return out; })()`,
    true,
  );
  record.modelCacheEntriesBefore = Object.entries(cache ?? {})
    .filter(([k]) => !k.startsWith('0x88-app'))
    .reduce((a, [, n]) => a + n, 0);
  await sleep(1000);
  const allPids = () => [...renderers, ...(gpuPid ? [gpuPid] : [])];
  const base = readFootprint(allPids(), { reset: true });
  record.pageReadyMs = Date.now() - navStarted;
  record.gpuBaselineMb = gpuPid ? round(base.get(gpuPid).cur) : null;
  const perPidPeaks = new Map(); // pid -> {phase: interval peak}
  const recordPeaks = (read, ph) => {
    for (const [p, v] of read) if (p !== gpuPid) (perPidPeaks.get(p) ?? perPidPeaks.set(p, {}).get(p))[ph] = round(v.interval);
  };

  await evaluate(`(() => {${WORKER_READY_SCRIPT}})()`);
  const t0 = Date.now();
  await evaluate('void window.__mobileSoak.start(); true');
  let phase = 'load';
  let progress = null;
  let polling = false;
  let settleAt = null;
  let crashed = false;
  const samples = [];
  const phasePeaks = {};
  const gpuPhasePeaks = {};
  const marks = {};
  const timer = setInterval(async () => {
    if (polling) return;
    polling = true;
    try {
      progress = await evaluate(`(() => {${PROGRESS_SCRIPT}})()`);
    } catch (error) {
      progress = { state: 'poll-error', error: String(error.message).slice(0, 120) };
    } finally {
      polling = false;
    }
  }, 250);
  try {
    while (Date.now() - t0 < timeoutMs) {
      const fp = readFootprint(allPids());
      const t = Date.now() - t0;
      const rendererTop = Math.max(0, ...[...fp].filter(([p]) => p !== gpuPid).map(([, v]) => v.cur));
      const next = progress?.state === 'poll-error' ? phase : phaseOf(progress, true);
      const order = ['load', 'search', 'dispose', 'settle'];
      if (next !== phase && order.indexOf(next) > order.indexOf(phase)) {
        const read = readFootprint(allPids(), { reset: true });
        recordPeaks(read, phase);
        if (gpuPid) gpuPhasePeaks[phase] = round(read.get(gpuPid)?.interval);
        marks[next] = { t, rendererTopMb: round(rendererTop), gpuMb: gpuPid ? round(read.get(gpuPid)?.cur) : null };
        log(`${kind} ${phase} -> ${next} at ${t} ms; gpu ${phase} peak ${gpuPhasePeaks[phase]} MB`);
        if (next === 'settle') settleAt = Date.now();
        phase = next;
      }
      samples.push([t, CODE[phase], round(rendererTop), gpuPid ? round(fp.get(gpuPid)?.cur) : null]);
      if (settleAt && Date.now() - settleAt >= settleMs) break;
      await sleep(sampleMs);
    }
  } finally {
    clearInterval(timer);
    clearInterval(refresh);
  }
  const end = readFootprint(allPids(), { reset: true });
  recordPeaks(end, phase);
  if (gpuPid) gpuPhasePeaks[phase] = round(end.get(gpuPid)?.interval);
  const pagePeak = (p) => Math.max(0, ...Object.values(perPidPeaks.get(p) ?? {}).filter(Number.isFinite));
  const pid = [...perPidPeaks.keys()].sort((a, b) => pagePeak(b) - pagePeak(a))[0];
  record.pid = pid;
  record.pidReused = pid === previous?.pid;
  record.pidWasNew = record.newRenderers.includes(pid);
  Object.assign(phasePeaks, perPidPeaks.get(pid) ?? {});
  record.baselineMb = round(base.get(pid)?.cur);
  record.lifeMaxAtBaselineMb = round(base.get(pid)?.life);
  record.crashed = crashed;
  record.timedOut = !crashed && !settleAt;
  record.phasePeaksMb = phasePeaks;
  record.gpuPhasePeaksMb = gpuPhasePeaks;
  record.marks = marks;
  record.peakMb = round(end.get(pid)?.life);
  record.peakThroughSearchMb = round(Math.max(phasePeaks.load ?? 0, phasePeaks.search ?? 0));
  record.gpuPeakThroughSearchMb = gpuPid ? round(Math.max(gpuPhasePeaks.load ?? 0, gpuPhasePeaks.search ?? 0)) : null;
  // Renderer + GPU, each at its own peak: an upper bound on the simultaneous total.
  record.peakThroughSearchSumMb = round(record.peakThroughSearchMb + (record.gpuPeakThroughSearchMb ?? 0));
  const med = (xs) => (xs.length ? [...xs].sort((a, b) => a - b)[xs.length >> 1] : null);
  record.searchMedianMb = med(samples.filter((s) => s[1] === 'r').map((s) => s[2]));
  record.doneMb = marks.settle?.rendererTopMb ?? null;
  const tailFrom = samples.length ? samples.at(-1)[0] - 2000 : 0;
  record.settledMb = med(samples.filter((s) => s[1] === 't' && s[0] >= tailFrom).map((s) => s[2]));
  record.samples = samples;
  try {
    const report = JSON.parse(await evaluate('JSON.stringify(window.__mobileSoak.results)'));
    record.report = digestReport(report);
    record.workerReady = await evaluate('window.__perfWorkerReady');
    const step = report.steps.at(-1);
    record.uaMemoryMb = Object.fromEntries(
      (step?.memory ?? []).map((m) => [m.phase, m.userAgentBytes == null ? (m.error ?? null) : round(m.userAgentBytes / 1048576)]),
    );
  } catch (error) {
    record.report = { status: crashed ? 'crashed' : 'report-error', error: String(error.message).slice(0, 200) };
  }
  const fetched = proxy.modelRequests(navStarted);
  record.modelRequests = fetched.map((r) => ({ path: r.path, status: r.status, mb: round(r.bytes / 1048576, 1) }));
  record.modelBytesFetchedMb = round(fetched.reduce((a, r) => a + r.bytes, 0) / 1048576, 1);
  record.cacheVerified = kind === 'cold' ? record.modelBytesFetchedMb > 1 : record.modelBytesFetchedMb < 1;
  log(
    `${kind}: renderer peakThroughSearch ${record.peakThroughSearchMb} MB, gpu ${record.gpuPeakThroughSearchMb} MB, lifetime ${record.peakMb}, UA ${JSON.stringify(record.uaMemoryMb)}, fetched ${record.modelBytesFetchedMb} MB, status ${record.report.status}`,
  );
  return record;
}
