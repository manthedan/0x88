// One desktop-Safari WebDriver session = one or more measured loads, each in a known WebContent process.
//
//   node safari_session.mjs <spec.json> <out.json>
//
// spec: { base, net, params, plan: ['cold', 'warm-fresh' | 'warm-same-tab', ...], settleMs, sampleMs, timeoutMs, driver }
//  - cold:          first load of a new automation session. Safari automation sessions use an ephemeral
//                   website-data store, so Cache Storage starts empty (asserted, and cleared if not).
//  - warm-fresh:    opens a NEW window, closes the previous one, waits for its WebContent process to exit,
//                   then loads in the new window's process. Cache Storage (same session) is populated.
//  - warm-same-tab: legacy method; navigates the same tab again, so the page process still holds whatever
//                   the previous load left behind. Kept only to quantify that artifact.
// Run under the machine's Safari lock (the matrix runner wraps each session in `benchlock`).
import { execFile, execFileSync } from 'node:child_process';
import { readFileSync, writeFileSync } from 'node:fs';
import { digestReport, footprintCategories, PROGRESS_SCRIPT, pgrep, phaseOf, readFootprint, round, sleep, soakUrl, WORKER_READY_SCRIPT } from './common.mjs';
import { startCountingProxy } from './proxy.mjs';

const [specPath, outPath] = process.argv.slice(2);
const spec = JSON.parse(readFileSync(specPath, 'utf8'));
const driver = spec.driver ?? 'http://127.0.0.1:4445';
const settleMs = spec.settleMs ?? 10_000;
const sampleMs = spec.sampleMs ?? 100;
const timeoutMs = spec.timeoutMs ?? 240_000;
const WC = 'com.apple.WebKit.WebContent';
const fired = new Set(); // diag hooks already run
const CODE = { load: 'l', search: 'r', dispose: 'd', settle: 't' }; // sample phase codes
const GPU = 'com.apple.WebKit.GPU';
const log = (...a) => console.log(new Date().toISOString().slice(11, 23), ...a);

async function wd(method, path, body) {
  const res = await fetch(driver + path, {
    method,
    headers: { 'content-type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
    // Bound every command: a hung command must not strand the shared Safari automation session.
    signal: AbortSignal.timeout(method === 'POST' && path === '/session' ? 30_000 : 120_000),
  });
  const j = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(`${method} ${path} ${res.status} ${JSON.stringify(j.value ?? j).slice(0, 300)}`);
  return j.value;
}
async function createSession() {
  const deadline = Date.now() + (spec.sessionWaitMs ?? 15 * 60_000);
  for (let attempt = 0; ; attempt++) {
    try {
      return (await wd('POST', '/session', { capabilities: { alwaysMatch: { browserName: 'safari' } } })).sessionId;
    } catch (error) {
      // Never delete someone else's session: wait it out.
      if (Date.now() >= deadline) throw error;
      log(`session create failed (attempt ${attempt}), retrying in 10 s: ${String(error.message).slice(0, 160)}`);
      await sleep(10_000);
    }
  }
}

const proxy = await startCountingProxy(spec.base);
const sid = await createSession().catch(async (error) => {
  await proxy.close();
  if (spec.sessionWaitMs === 0 && String(error.message).includes('already paired with another WebDriver session')) {
    console.error('Safari paired elsewhere; yielding benchlock before retry.');
    process.exit(75);
  }
  throw error;
});
log(`webdriver session ${sid}`); // printed so a killed run's session can still be deleted by hand
// Explicit page-load timeout; navigation errors surface instead of hanging.
await wd('POST', `/session/${sid}/timeouts`, { pageLoad: 90_000, script: 60_000 }).catch(() => {});
const exec = (script, args = []) => wd('POST', `/session/${sid}/execute/sync`, { script, args });
const jscFlags = Object.fromEntries(
  ['__XPC_JSC_useOMGJIT', '__XPC_JSC_useBBQJIT', '__XPC_JSC_useJIT'].map((key) => {
    try {
      return [key, execFileSync('launchctl', ['getenv', key], { encoding: 'utf8' }).trim()];
    } catch {
      return [key, ''];
    }
  }),
);
if (Object.values(jscFlags).some((value) => value !== '')) {
  await wd('DELETE', `/session/${sid}`).catch(() => {});
  await proxy.close();
  throw new Error(`Normal-JSC cohort refuses launchctl overrides: ${JSON.stringify(jscFlags)}`);
}
const runs = [];
let previousPagePid = null;
try {
  for (const [index, kind] of spec.plan.entries()) {
    if (kind === 'warm-fresh') {
      const oldHandle = await wd('GET', `/session/${sid}/window`);
      const { handle } = await wd('POST', `/session/${sid}/window/new`, { type: 'window' });
      await wd('DELETE', `/session/${sid}/window`); // closes the current (old) window
      await wd('POST', `/session/${sid}/window`, { handle });
      log(`opened window ${handle}, closed ${oldHandle}`);
      const until = Date.now() + 15_000;
      while (previousPagePid && pgrep(WC).includes(previousPagePid) && Date.now() < until) await sleep(250);
    }
    const previousExited = previousPagePid ? !pgrep(WC).includes(previousPagePid) : null;
    runs.push(await measure(kind, index));
    runs.at(-1).previousPageProcessExited = previousExited;
    runs.at(-1).jscFlags = jscFlags;
    previousPagePid = runs.at(-1).pid ?? previousPagePid;
  }
} catch (error) {
  // Keep completed loads; record the failed one so the runner marks it invalid.
  log(`session error: ${error.message}`);
  runs.push({
    browser: 'safari',
    net: spec.net,
    params: spec.params ?? {},
    cache: spec.plan[runs.length] ?? 'unknown',
    sessionIndex: runs.length,
    report: { status: 'harness-error', error: String(error.message).slice(0, 300) },
  });
} finally {
  await wd('DELETE', `/session/${sid}`).catch(() => {});
  await proxy.close();
}
writeFileSync(outPath, JSON.stringify(runs, null, 1));

async function measure(kind, index) {
  const record = { browser: 'safari', net: spec.net, params: spec.params ?? {}, cache: kind, sessionIndex: index, at: new Date().toISOString() };
  // Candidate processes: new ones, or tiny pre-existing ones (WebKit's prewarmed spare process).
  const before = readFootprint(pgrep(WC));
  const gpuPids = pgrep(GPU);
  const url = soakUrl(proxy.url, spec.net, { ...(spec.params ?? {}), ...(kind === 'warm-same-tab' ? { _nav: String(index) } : {}) });
  record.url = url;
  const navStarted = Date.now();
  await wd('POST', `/session/${sid}/url`, { url });
  // Wait for the soak controller and its environment probe.
  for (let i = 0; i < 120; i++) {
    const ok = await exec('const s=window.__mobileSoak; return !!(s && s.results && s.results.env);').catch(() => false);
    if (ok) break;
    await sleep(250);
  }
  const now = readFootprint(pgrep(WC));
  const candidates = [...now.entries()].filter(
    ([pid]) => !before.has(pid) || before.get(pid).cur < 40 || (kind === 'warm-same-tab' && pid === previousPagePid),
  );
  candidates.sort((a, b) => b[1].cur - a[1].cur);
  const pid = candidates[0]?.[0];
  record.pid = pid;
  record.context = await exec('return { secure: isSecureContext, crossOriginIsolated, ua: navigator.userAgent };').catch(() => null);
  record.pidReused = pid === previousPagePid;
  record.candidatePids = candidates.map(([p, v]) => [p, round(v.cur)]);
  if (!pid) throw new Error('could not identify page WebContent process');

  // Cache state check; cold must start empty.
  const cache = await exec(`return (async () => {
    const out = {};
    for (const k of await caches.keys()) out[k] = (await (await caches.open(k)).keys()).length;
    try { localStorage.removeItem('0x88-mobile-soak-v1'); } catch {}
    return out; })();`);
  record.cacheBefore = cache;
  // The app shell's service-worker cache ("0x88-app-*") is irrelevant; model caches decide cold/warm.
  const modelCaches = Object.entries(cache ?? {}).filter(([k]) => !k.startsWith('0x88-app'));
  record.modelCacheEntriesBefore = modelCaches.reduce((a, [, n]) => a + n, 0);
  if (kind === 'cold' && record.modelCacheEntriesBefore) {
    await exec(`return (async () => { for (const k of await caches.keys()) if (!k.startsWith('0x88-app')) await caches.delete(k); return true; })();`);
    record.coldCacheCleared = record.modelCacheEntriesBefore;
  }
  await sleep(1000);
  const base = readFootprint([pid], { reset: true }).get(pid);
  const gpuBase = readFootprint(gpuPids);
  record.pageReadyMs = Date.now() - navStarted;
  record.baselineMb = round(base.cur);
  record.lifeMaxAtBaselineMb = round(base.life); // ≫ baseline means the process carried an earlier load
  record.cats = { baseline: await footprintCategories(pid) };

  await exec(WORKER_READY_SCRIPT);
  const t0 = Date.now();
  await exec('void window.__mobileSoak.start(); return true;');
  let phase = 'load';
  let progress = null;
  let polling = false;
  let settleAt = null;
  let crashed = false;
  const samples = [];
  const phasePeaks = {};
  const marks = {};
  const pending = [];
  const gpuMax = new Map();
  const timer = setInterval(async () => {
    if (polling) return;
    polling = true;
    try {
      progress = await exec(PROGRESS_SCRIPT);
    } catch (error) {
      progress = { state: 'poll-error', error: String(error.message).slice(0, 120) };
    } finally {
      polling = false;
    }
  }, 250);
  try {
    while (Date.now() - t0 < timeoutMs) {
      const fp = readFootprint([pid], { threads: true }).get(pid);
      const t = Date.now() - t0;
      if (!fp) {
        crashed = true;
        log(`page process ${pid} vanished at ${t} ms`);
        break;
      }
      for (const [gp, v] of readFootprint(gpuPids)) gpuMax.set(gp, Math.max(gpuMax.get(gp) ?? 0, v.cur - (gpuBase.get(gp)?.cur ?? v.cur)));
      const next = progress?.state === 'poll-error' ? phase : phaseOf(progress, true);
      if (next !== phase && ['load', 'search', 'dispose', 'settle'].indexOf(next) > ['load', 'search', 'dispose', 'settle'].indexOf(phase)) {
        const read = readFootprint([pid], { reset: true }).get(pid);
        phasePeaks[phase] = round(read.interval);
        marks[next] = { t, mb: round(read.cur) };
        log(`${kind} ${phase} -> ${next} at ${t} ms; ${phase} peak ${round(read.interval)} MB, now ${round(read.cur)} MB`);
        if (next === 'search') pending.push(footprintCategories(pid).then((c) => (record.cats.loaded = c)));
        if (next === 'settle') {
          settleAt = Date.now();
          pending.push(footprintCategories(pid).then((c) => (record.cats.done = c)));
        }
        phase = next;
      }
      // [t ms, phase, footprint MB, task CPU ms, Wasm tier-up compiler CPU ms, JS JIT compiler CPU ms]
      samples.push([t, CODE[phase], round(fp.cur), round(fp.cpuMs), round(fp.wasmJitMs), round(fp.jsJitMs)]);
      if (settleAt) diagHooks(Date.now() - settleAt, pid, record, pending, samples);
      else if (spec.diag?.loadSampleAtMs !== undefined && t >= spec.diag.loadSampleAtMs && !fired.has(`${index}:loadsample`)) {
        fired.add(`${index}:loadsample`);
        const file = `${spec.diag.outPrefix ?? '/tmp/perf'}-${index}-${kind}.load-sample.txt`;
        record.loadStackSample = { file, at: t, phase };
        pending.push(new Promise((r) => execFile('sample', [String(pid), '1', '-file', file], () => r())));
      }
      if (settleAt && Date.now() - settleAt >= settleMs) break;
      await sleep(sampleMs);
    }
  } finally {
    clearInterval(timer);
  }
  const end = readFootprint([pid], { reset: true }).get(pid);
  if (end) {
    phasePeaks[phase] = round(end.interval);
    record.cats.settled = await footprintCategories(pid);
  }
  await Promise.all(pending);
  record.crashed = crashed;
  record.timedOut = !crashed && !settleAt;
  record.phasePeaksMb = phasePeaks;
  record.marks = marks;
  record.peakMb = round(end?.life ?? Math.max(...samples.map((s) => s[2])));
  record.sampledPeakMb = Math.max(...samples.map((s) => s[2]));
  record.peakThroughSearchMb = round(Math.max(phasePeaks.load ?? 0, phasePeaks.search ?? 0));
  const inPhase = (p) => samples.filter((s) => s[1] === p).map((s) => s[2]);
  const med = (xs) => (xs.length ? [...xs].sort((a, b) => a - b)[xs.length >> 1] : null);
  record.searchMedianMb = med(inPhase('r'));
  record.doneMb = marks.settle?.mb ?? null;
  const tailFrom = samples.length ? samples.at(-1)[0] - 2000 : 0;
  record.settledMb = med(samples.filter((s) => s[1] === 't' && s[0] >= tailFrom).map((s) => s[2]));
  record.gpuProcessMaxDeltaMb = round(Math.max(0, ...gpuMax.values()));
  record.samples = samples;
  try {
    record.report = digestReport(JSON.parse(await exec('return JSON.stringify(window.__mobileSoak.results);')));
    record.workerReady = JSON.parse(await exec('return JSON.stringify(window.__perfWorkerReady);'));
  } catch (error) {
    record.report = { status: crashed ? 'crashed' : 'report-error', error: String(error.message).slice(0, 200) };
  }
  const fetched = proxy.modelRequests(navStarted);
  record.modelRequests = fetched.map((r) => ({ path: r.path, status: r.status, mb: round(r.bytes / 1048576, 1) }));
  record.modelBytesFetchedMb = round(fetched.reduce((a, r) => a + r.bytes, 0) / 1048576, 1);
  record.cacheVerified = kind === 'cold' ? record.modelBytesFetchedMb > 1 : record.modelBytesFetchedMb < 1;
  log(`${kind}: fetched ${record.modelBytesFetchedMb} MB of models; cacheVerified ${record.cacheVerified}`);
  log(
    `${kind}: peak ${record.peakMb} MB (sampled ${record.sampledPeakMb}), baseline ${record.baselineMb}, lifeAtBaseline ${record.lifeMaxAtBaselineMb}, settled ${record.settledMb}, status ${record.report.status}`,
  );
  return record;
}

/** Optional diagnostics during the settle window (spec.diag): stack sample, CPU, simulated memory pressure. */
function diagHooks(sinceSettle, pid, record, pending, samples) {
  const d = spec.diag;
  if (!d) return;
  const once = (key, at, fn) => {
    if (sinceSettle >= at && !fired.has(`${record.sessionIndex}:${key}`)) {
      fired.add(`${record.sessionIndex}:${key}`);
      fn();
    }
  };
  if (d.cpuEveryMs)
    for (let at = 0; at < settleMs; at += d.cpuEveryMs)
      once(`cpu${at}`, at, () => {
        try {
          const cpu = Number(execFileSync('ps', ['-o', '%cpu=', '-p', String(pid)], { encoding: 'utf8' }).trim());
          record.cpu = record.cpu ?? [];
          record.cpu.push([Math.round(sinceSettle), cpu]);
        } catch {}
      });
  if (d.sampleAtMs !== undefined)
    once('sample', d.sampleAtMs, () => {
      const file = `${d.outPrefix ?? '/tmp/perf'}-${record.sessionIndex}-${record.cache}.sample.txt`;
      record.stackSample = file;
      pending.push(new Promise((r) => execFile('sample', [String(pid), String(d.sampleSeconds ?? 2), '-file', file], () => r())));
    });
  // Post lowMemory only once the page process is idle: task CPU < idleCpuPct over the last idleForMs.
  if (d.lowMemoryWhenIdle && !fired.has(`${record.sessionIndex}:lowmemidle`)) {
    const { idleCpuPct = 10, idleForMs = 3000, afterMs = 0 } = d.lowMemoryWhenIdle;
    const last = samples.at(-1);
    const from = samples.findLast((s) => s[0] <= last[0] - idleForMs);
    const pct = from ? ((last[3] - from[3]) / (last[0] - from[0])) * 100 : Infinity;
    if (sinceSettle >= afterMs && pct < idleCpuPct) {
      fired.add(`${record.sessionIndex}:lowmemidle`);
      record.lowMemoryIdle = { postedAtSettleMs: Math.round(sinceSettle), t: last[0], beforeMb: last[2], cpuPctOverWindow: round(pct, 1) };
      execFileSync('notifyutil', ['-p', 'org.WebKit.lowMemory']);
      log(`posted org.WebKit.lowMemory at settle+${Math.round(sinceSettle)} ms (idle ${round(pct, 1)}% CPU), ${last[2]} MB`);
    }
  }
  if (d.lowMemoryAtMs !== undefined)
    once('lowmem', d.lowMemoryAtMs, () => {
      // WebKit's documented simulation hook (WTF MemoryPressureHandlerCocoa): all WebKit processes purge.
      execFileSync('notifyutil', ['-p', 'org.WebKit.lowMemory']);
      record.lowMemoryPostedAt = Math.round(sinceSettle);
      log(`posted org.WebKit.lowMemory at settle+${Math.round(sinceSettle)} ms`);
    });
}
