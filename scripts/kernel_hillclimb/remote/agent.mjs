#!/usr/bin/env node
// Remote-side bench agent. Zero dependencies (node >= 22: global WebSocket + fetch).
//
// Serves a static directory with COOP/COEP headers on 127.0.0.1:<free port>,
// launches headless Chrome with the given flags, opens the page, waits for
// window.__kernelBenchResult (object) or window.__kernelBenchError (string),
// and prints ONE JSON line to stdout. Cleans up only the processes it started
// (its Chrome process group) and its own temp profile.
//
// Usage:
//   node agent.mjs --root <dir> --page <path?query> --chrome <binary>
//                  [--flags-json '["--flag", ...]'] [--timeout 120] [--gpu-info]
//                  [--out result.json] [--keep-profile] [--watch-stdin]
//                  [--lock <dir>] [--wait-idle] [--load-max 3] [--gpu-max 10]
//                  [--hold 60] [--poll 30] [--max-wait 21600]
// --lock: mkdir mutex so our own jobs on this host serialize (stale owner PID reclaimed).
// --wait-idle: before launching Chrome, wait until 1-min loadavg < load-max AND GPU
//   utilisation < gpu-max for `hold` seconds; exit 75 after max-wait seconds.
// --watch-stdin: clean up and exit when stdin closes (controller ssh gone).
// Exit codes: 0 result, 2 page error/timeout, 3 agent failure, 75 never started (busy).
import { spawn, execFileSync } from 'node:child_process';
import { createServer } from 'node:http';
import { mkdtempSync, readFileSync, rmSync, statSync, existsSync, createReadStream, mkdirSync, writeFileSync, renameSync } from 'node:fs';
import { join, resolve, extname, normalize, sep } from 'node:path';
import os from 'node:os';

const args = parseArgs(process.argv.slice(2));
const root = resolve(args.root ?? '.');
const page = args.page ?? '/';
const chromeBin = args.chrome;
const timeoutS = Number(args.timeout ?? 120);
const extraFlags = args['flags-json'] ? JSON.parse(args['flags-json']) : [];
const wantGpuInfo = Boolean(args['gpu-info']);
const outFile = args.out ? resolve(args.out) : null;
const IDLE = {
  loadMax: Number(args['load-max'] ?? 3), gpuMax: Number(args['gpu-max'] ?? 10),
  holdS: Number(args.hold ?? 60), pollS: Number(args.poll ?? 30), maxWaitS: Number(args['max-wait'] ?? 21600),
};
const tStart = Date.now();
if (!chromeBin) fail('missing --chrome');

const MIME = {
  '.html': 'text/html; charset=utf-8', '.htm': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8', '.mjs': 'text/javascript; charset=utf-8',
  '.cjs': 'text/javascript; charset=utf-8', '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8', '.map': 'application/json; charset=utf-8',
  '.wasm': 'application/wasm', '.wgsl': 'text/plain; charset=utf-8',
  '.txt': 'text/plain; charset=utf-8', '.svg': 'image/svg+xml', '.png': 'image/png',
  '.ico': 'image/x-icon', '.onnx': 'application/octet-stream', '.bin': 'application/octet-stream',
  '.webmanifest': 'application/manifest+json',
};

const state = { chrome: null, server: null, profileDir: null, cleaned: false, console: [], lockDir: null, sampler: null, samples: [], chromeStderr: '' };

async function main() {
  if (args.lock) await takeLock(resolve(args.lock));
  const idleWait = args['wait-idle'] ? await waitIdle() : null;
  armDeadline();
  const loadStart = await sampleLoad();
  const port = await startServer();
  state.profileDir = mkdtempSync(join(os.tmpdir(), 'khc-chrome-'));
  const flags = [
    ...extraFlags,
    `--user-data-dir=${state.profileDir}`,
    '--remote-debugging-port=0',
    '--no-first-run', '--no-default-browser-check', '--disable-extensions',
    '--disable-background-timer-throttling', '--disable-renderer-backgrounding',
    '--disable-backgrounding-occluded-windows',
    'about:blank',
  ];
  state.chrome = spawn(chromeBin, flags, { detached: true, stdio: ['ignore', 'ignore', 'pipe'] });
  state.chrome.stderr.on('data', (d) => { state.chromeStderr = (state.chromeStderr + d).slice(-8000); });
  const wsPath = await waitDevToolsPort(state.profileDir, 30_000);
  const cdp = await Cdp.connect(`ws://127.0.0.1:${wsPath.port}${wsPath.path}`);
  const version = await cdp.send('Browser.getVersion');
  let gpuInfo = null;
  if (wantGpuInfo) {
    try { gpuInfo = await cdp.send('SystemInfo.getInfo'); } catch (e) { gpuInfo = { error: String(e) }; }
  }
  const url = `http://127.0.0.1:${port}${page.startsWith('/') ? '' : '/'}${page}`;
  const { targetId } = await cdp.send('Target.createTarget', { url: 'about:blank' });
  const { sessionId } = await cdp.send('Target.attachToTarget', { targetId, flatten: true });
  cdp.on('Runtime.consoleAPICalled', (p, sid) => {
    if (sid !== sessionId) return;
    const text = p.args.map((a) => a.value ?? a.description ?? a.type).join(' ');
    pushConsole(`[${p.type}] ${text}`);
  });
  cdp.on('Runtime.exceptionThrown', (p, sid) => {
    if (sid !== sessionId) return;
    const d = p.exceptionDetails;
    pushConsole(`[exception] ${d.exception?.description ?? d.text}`);
  });
  await cdp.send('Runtime.enable', {}, sessionId);
  await cdp.send('Page.enable', {}, sessionId);
  await cdp.send('Page.navigate', { url }, sessionId);

  state.sampler = setInterval(() => { state.samples.push(foreignLoad()); }, 2000);
  const deadline = Date.now() + timeoutS * 1000;
  let outcome = null;
  while (Date.now() < deadline) {
    await sleep(250);
    let r;
    try {
      r = await cdp.send('Runtime.evaluate', {
        expression: `(() => { const r = window.__kernelBenchResult, e = window.__kernelBenchError;
          if (r !== undefined && r !== null) return { kind: 'result', value: JSON.parse(JSON.stringify(r)) };
          if (e !== undefined && e !== null) return { kind: 'error', value: String(e) };
          return null; })()`,
        returnByValue: true,
      }, sessionId);
    } catch { continue; } // navigation in flight
    const v = r?.result?.value;
    if (v) { outcome = v; break; }
  }
  clearInterval(state.sampler);
  try { await Promise.race([cdp.send('Browser.close'), sleep(3000)]); } catch {}
  cdp.close();
  await cleanup();
  await sleep(3000); // let GPU utilisation counters decay past our own work
  const loadEnd = await sampleLoad(); // after Chrome is gone: GPU util here is other people's
  const during = summarizeSamples(state.samples);
  const contendedReasons = [];
  if (args['wait-idle'] || os.platform() === 'darwin') {
    for (const [tag, l] of [['start', loadStart], ['end', loadEnd]]) {
      if (l.loadavg[0] >= IDLE.loadMax) contendedReasons.push(`loadavg1 ${l.loadavg[0]} >= ${IDLE.loadMax} at ${tag}`);
      const g = gpuUtil(l.gpu); if (g !== null && g >= IDLE.gpuMax) contendedReasons.push(`gpu util ${g}% >= ${IDLE.gpuMax}% at ${tag}`);
    }
    if (during.foreignCpuCoresMax !== null && during.foreignCpuCoresMax >= IDLE.loadMax)
      contendedReasons.push(`foreign CPU ${during.foreignCpuCoresMax} cores >= ${IDLE.loadMax} during run`);
  } else {
    for (const [tag, l] of [['start', loadStart], ['end', loadEnd]]) {
      const apps = l.gpu?.computeApps ?? [];
      if (apps.length) contendedReasons.push(`GPU compute apps at ${tag}: ${apps.join('; ')}`);
      const g = gpuUtil(l.gpu); if (g !== null && g >= IDLE.gpuMax) contendedReasons.push(`gpu util ${g}% at ${tag}`);
    }
  }
  const out = {
    ok: outcome?.kind === 'result',
    result: outcome?.kind === 'result' ? outcome.value : null,
    error: outcome?.kind === 'error' ? outcome.value : outcome ? null : `timeout after ${timeoutS}s`,
    meta: {
      host: os.hostname(), platform: `${os.platform()} ${os.release()} ${os.arch()}`,
      cpus: os.cpus().length, cpuModel: os.cpus()[0]?.model,
      browser: { binary: chromeBin, product: version.product, userAgent: version.userAgent },
      flags: extraFlags, url, wallMs: Date.now() - tStart,
      loadStart, loadEnd, duringRun: during, idleWait,
      contended: contendedReasons.length > 0, contendedReasons,
      console: state.console.slice(-60),
      ...(gpuInfo ? { gpuInfo } : {}),
      ...(outcome ? {} : { chromeStderrTail: state.chromeStderr.slice(-3000) }),
    },
  };
  emit(out);
  releaseLock();
  process.exit(out.ok ? 0 : 2);
}

function emit(out) {
  const line = JSON.stringify(out) + '\n';
  if (outFile) { writeFileSync(outFile + '.tmp', line); renameSync(outFile + '.tmp', outFile); }
  process.stdout.write(line);
}

function gpuUtil(g) { return g?.utilGpuPct ?? g?.deviceUtilPct ?? null; }

async function takeLock(dir) {
  const end = Date.now() + IDLE.maxWaitS * 1000;
  let lastLog = 0;
  while (true) {
    try {
      mkdirSync(dir);
      writeFileSync(join(dir, 'owner'), `${process.pid} ${new Date().toISOString()} ${page}\n`);
      state.lockDir = dir; return;
    } catch {}
    let owner = null; try { owner = Number(readFileSync(join(dir, 'owner'), 'utf8').split(' ')[0]); } catch {}
    if (owner && !pidAlive(owner)) { try { rmSync(dir, { recursive: true, force: true }); } catch {} continue; }
    if (Date.now() > end) { busyExit(`lock ${dir} held by pid ${owner} for > max-wait`); }
    if (Date.now() - lastLog > 300_000) { logErr(`waiting on lock ${dir} (owner ${owner})`); lastLog = Date.now(); }
    await sleep(5000);
  }
}
function releaseLock() {
  if (!state.lockDir) return;
  try { if (Number(readFileSync(join(state.lockDir, 'owner'), 'utf8').split(' ')[0]) === process.pid) rmSync(state.lockDir, { recursive: true, force: true }); } catch {}
  state.lockDir = null;
}
function pidAlive(pid) { try { process.kill(pid, 0); return true; } catch (e) { return e.code === 'EPERM'; } }

async function waitIdle() {
  const t0 = Date.now(), end = t0 + IDLE.maxWaitS * 1000;
  let idleSince = null, polls = 0, last = null;
  while (true) {
    const load1 = +os.loadavg()[0].toFixed(2), g = gpuUtil(gpuLoad());
    last = { load1, gpuUtilPct: g };
    const ok = load1 < IDLE.loadMax && (g === null || g < IDLE.gpuMax);
    polls++;
    if (ok) {
      idleSince ??= Date.now();
      if (Date.now() - idleSince >= IDLE.holdS * 1000) break;
    } else {
      idleSince = null;
      logErr(`waiting for idle: load1=${load1} (max ${IDLE.loadMax}) gpu=${g}% (max ${IDLE.gpuMax}%) waited ${Math.round((Date.now() - t0) / 1000)}s`);
    }
    if (Date.now() > end) busyExit(`host never idle for ${IDLE.holdS}s within ${IDLE.maxWaitS}s (last load1=${load1}, gpu=${g}%)`);
    await sleep((ok ? Math.min(IDLE.pollS, 10) : IDLE.pollS) * 1000);
  }
  return { waitedS: Math.round((Date.now() - t0) / 1000), polls, thresholds: IDLE, last };
}
function busyExit(msg) {
  emit({ ok: false, result: null, error: `busy: ${msg}`, busy: true, meta: { host: os.hostname(), wallMs: Date.now() - tStart, loadavg: os.loadavg(), gpu: gpuLoad() } });
  releaseLock();
  process.exit(75);
}
function logErr(s) { process.stderr.write(`[agent ${new Date().toISOString()}] ${s}\n`); }

// CPU used by processes outside our Chrome process group and this agent (in cores).
function foreignLoad() {
  const s = { t: Date.now(), load1: +os.loadavg()[0].toFixed(2), foreignCpuCores: null };
  try {
    const txt = execFileSync('ps', ['-A', '-o', 'pid=,pgid=,%cpu='], { encoding: 'utf8', timeout: 3000 });
    const pg = state.chrome?.pid; let sum = 0;
    for (const l of txt.trim().split('\n')) {
      const [pid, pgid, cpu] = l.trim().split(/\s+/).map(Number);
      if (pid === process.pid || pgid === pg) continue;
      sum += cpu || 0;
    }
    s.foreignCpuCores = +(sum / 100).toFixed(2);
  } catch {}
  return s;
}
function summarizeSamples(xs) {
  const f = xs.map((x) => x.foreignCpuCores).filter((x) => x !== null);
  return { samples: xs.length, load1Max: xs.length ? Math.max(...xs.map((x) => x.load1)) : null,
    foreignCpuCoresMax: f.length ? Math.max(...f) : null, foreignCpuCoresMean: f.length ? +(f.reduce((a, b) => a + b, 0) / f.length).toFixed(2) : null };
}

function pushConsole(s) {
  state.console.push(s.length > 2000 ? s.slice(0, 2000) + '…' : s);
  if (state.console.length > 500) state.console.shift();
}

function startServer() {
  return new Promise((res, rej) => {
    const server = createServer((req, resp) => {
      let p;
      try { p = decodeURIComponent(new URL(req.url, 'http://x').pathname); } catch { p = '/'; }
      let file = normalize(join(root, p));
      if (file !== root && !file.startsWith(root + sep)) { resp.writeHead(403); return resp.end(); }
      try { if (statSync(file).isDirectory()) file = join(file, 'index.html'); } catch {}
      if (!existsSync(file)) {
        // SvelteKit static fallbacks: /foo -> /foo.html
        if (existsSync(file + '.html')) file += '.html';
        else { resp.writeHead(404, headers('text/plain')); return resp.end('not found'); }
      }
      const size = statSync(file).size;
      resp.writeHead(200, { ...headers(MIME[extname(file).toLowerCase()] ?? 'application/octet-stream'), 'Content-Length': size });
      if (req.method === 'HEAD') return resp.end();
      createReadStream(file).pipe(resp);
    });
    server.on('error', rej);
    server.listen(0, '127.0.0.1', () => { state.server = server; res(server.address().port); });
  });
}

function headers(type) {
  return {
    'Content-Type': type,
    'Cross-Origin-Opener-Policy': 'same-origin',
    'Cross-Origin-Embedder-Policy': 'require-corp',
    'Cross-Origin-Resource-Policy': 'same-origin',
    'Cache-Control': 'no-store',
  };
}

async function waitDevToolsPort(dir, ms) {
  const f = join(dir, 'DevToolsActivePort');
  const end = Date.now() + ms;
  while (Date.now() < end) {
    if (state.chrome.exitCode !== null) throw new Error(`chrome exited early (code ${state.chrome.exitCode})`);
    try {
      const [port, path] = readFileSync(f, 'utf8').trim().split('\n');
      if (port && path) return { port: Number(port), path };
    } catch {}
    await sleep(100);
  }
  throw new Error('DevToolsActivePort not written');
}

class Cdp {
  static connect(url) {
    return new Promise((res, rej) => {
      const ws = new WebSocket(url);
      const c = new Cdp(ws);
      ws.addEventListener('open', () => res(c));
      ws.addEventListener('error', (e) => rej(new Error(`cdp ws error ${e.message ?? ''}`)));
    });
  }
  constructor(ws) {
    this.ws = ws; this.id = 0; this.pending = new Map(); this.handlers = new Map();
    ws.addEventListener('message', (ev) => {
      const m = JSON.parse(typeof ev.data === 'string' ? ev.data : Buffer.from(ev.data).toString());
      if (m.id !== undefined) {
        const p = this.pending.get(m.id); if (!p) return; this.pending.delete(m.id);
        m.error ? p.rej(new Error(`${m.error.message} ${m.error.data ?? ''}`)) : p.res(m.result);
      } else for (const h of this.handlers.get(m.method) ?? []) h(m.params, m.sessionId);
    });
    ws.addEventListener('close', () => { for (const p of this.pending.values()) p.rej(new Error('cdp closed')); this.pending.clear(); });
  }
  send(method, params = {}, sessionId) {
    const id = ++this.id;
    const msg = { id, method, params }; if (sessionId) msg.sessionId = sessionId;
    return new Promise((res, rej) => { this.pending.set(id, { res, rej }); this.ws.send(JSON.stringify(msg)); });
  }
  on(method, fn) { (this.handlers.get(method) ?? this.handlers.set(method, []).get(method)).push(fn); }
  close() { try { this.ws.close(); } catch {} }
}

async function sampleLoad() {
  const s = { at: new Date().toISOString(), loadavg: os.loadavg().map((x) => +x.toFixed(2)), cpus: os.cpus().length };
  // 500 ms CPU busy sample across all cores.
  const a = os.cpus(); await sleep(500); const b = os.cpus();
  let busy = 0, total = 0;
  for (let i = 0; i < a.length; i++) {
    const ta = a[i].times, tb = b[i].times;
    const tot = (tb.user + tb.nice + tb.sys + tb.idle + tb.irq) - (ta.user + ta.nice + ta.sys + ta.idle + ta.irq);
    busy += tot - (tb.idle - ta.idle); total += tot;
  }
  s.cpuBusyPct = total ? +(100 * busy / total).toFixed(1) : null;
  s.gpu = gpuLoad();
  return s;
}

function gpuLoad() {
  try {
    if (os.platform() === 'linux') {
      const q = 'name,utilization.gpu,utilization.memory,memory.used,memory.total,clocks.sm,power.draw,temperature.gpu';
      const line = execFileSync('nvidia-smi', [`--query-gpu=${q}`, '--format=csv,noheader,nounits'], { encoding: 'utf8', timeout: 5000 }).trim().split('\n')[0];
      const v = line.split(',').map((x) => x.trim());
      const procs = execFileSync('nvidia-smi', ['--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader,nounits'], { encoding: 'utf8', timeout: 5000 })
        .trim().split('\n').filter(Boolean).map((l) => l.split(',').map((x) => x.trim()).join(' '));
      return { source: 'nvidia-smi', name: v[0], utilGpuPct: +v[1], utilMemPct: +v[2], memUsedMiB: +v[3], memTotalMiB: +v[4], smClockMHz: +v[5], powerW: +v[6], tempC: +v[7], computeApps: procs };
    }
    if (os.platform() === 'darwin') {
      const txt = execFileSync('ioreg', ['-r', '-d', '1', '-w', '0', '-c', 'IOAccelerator'], { encoding: 'utf8', timeout: 5000 });
      const pick = (k) => { const m = txt.match(new RegExp(`"${k}"=(\\d+)`)); return m ? +m[1] : null; };
      return { source: 'ioreg', deviceUtilPct: pick('Device Utilization %'), rendererUtilPct: pick('Renderer Utilization %'), tilerUtilPct: pick('Tiler Utilization %'), inUseSysMemBytes: pick('In use system memory') };
    }
  } catch (e) { return { error: String(e.message ?? e).slice(0, 200) }; }
  return null;
}

async function cleanup() {
  if (state.cleaned) return; state.cleaned = true;
  const ch = state.chrome;
  if (ch && ch.exitCode === null && ch.signalCode === null) {
    try { process.kill(-ch.pid, 'SIGTERM'); } catch {}
    const end = Date.now() + 5000;
    while (ch.exitCode === null && ch.signalCode === null && Date.now() < end) await sleep(100);
  }
  // Kill any leftovers in our own process group (helpers outlive the browser briefly).
  if (ch) { try { process.kill(-ch.pid, 'SIGKILL'); } catch {} }
  if (state.server) state.server.close();
  if (state.profileDir && !args['keep-profile']) { await sleep(200); try { rmSync(state.profileDir, { recursive: true, force: true }); } catch {} }
}

function fail(msg) {
  const out = { ok: false, result: null, error: msg, meta: { host: os.hostname(), wallMs: Date.now() - tStart, flags: extraFlags, console: state.console.slice(-60),
    chromeExit: state.chrome ? { code: state.chrome.exitCode, signal: state.chrome.signalCode } : null, chromeStderrTail: state.chromeStderr.slice(-3000) } };
  clearInterval(state.sampler);
  cleanup().finally(() => { emit(out); releaseLock(); process.exit(3); });
}

function parseArgs(argv) {
  const o = {};
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i]; if (!a.startsWith('--')) continue;
    const eq = a.indexOf('=');
    if (eq > 0) { o[a.slice(2, eq)] = a.slice(eq + 1); continue; }
    const k = a.slice(2); const n = argv[i + 1];
    if (n === undefined || n.startsWith('--')) o[k] = true; else { o[k] = n; i++; }
  }
  return o;
}
function sleep(ms) { return new Promise((r) => setTimeout(r, ms)); }

for (const sig of ['SIGINT', 'SIGTERM', 'SIGHUP']) process.on(sig, () => fail(`agent got ${sig}`));
if (args['watch-stdin']) { process.stdin.on('end', () => fail('stdin closed (controller gone)')); process.stdin.resume(); }
let hardDeadline = null;
const armDeadline = () => { hardDeadline = setTimeout(() => fail('agent hard deadline'), (timeoutS + 90) * 1000); hardDeadline.unref(); };
main().catch((e) => fail(String(e?.stack ?? e)));
