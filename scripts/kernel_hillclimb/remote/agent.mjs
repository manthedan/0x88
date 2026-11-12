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
//                  [--wait-idle] [--load-max 3] [--gpu-max 10]
//                  [--hold 60] [--poll 30] [--max-wait 21600] [--wait-until <epoch ms>]
// Host serialisation is NOT done here: job.sh wraps the agent in a kernel advisory
// lock (lockf/flock, see lib.mjs lockCommand), which dies with its holder.
// --wait-idle: before launching Chrome, wait until 1-min loadavg < load-max AND GPU
//   utilisation < gpu-max (a known reading) for `hold` seconds; exit 75 once
//   max-wait seconds (or --wait-until) pass without that.
// --watch-stdin: clean up and exit when stdin closes (controller ssh gone).
// Exit codes: 0 result, 2 page error/timeout, 3 agent failure, 75 never started (busy).
import { execFileSync, spawn } from 'node:child_process';
import { createReadStream, mkdtempSync, readFileSync, realpathSync, renameSync, rmSync, writeFileSync } from 'node:fs';
import { createServer } from 'node:http';
import os from 'node:os';
import { extname, join, resolve } from 'node:path';
import { assessRun, parseAgxClients, resolveStatic, waitForIdle } from './lib.mjs';

const args = parseArgs(process.argv.slice(2));
const root = resolve(args.root ?? '.');
const page = args.page ?? '/';
const chromeBin = args.chrome;
const timeoutS = Number(args.timeout ?? 120);
const extraFlags = args['flags-json'] ? JSON.parse(args['flags-json']) : [];
const wantGpuInfo = Boolean(args['gpu-info']);
const outFile = args.out ? resolve(args.out) : null;
const IDLE = {
  loadMax: Number(args['load-max'] ?? 3),
  gpuMax: Number(args['gpu-max'] ?? 10),
  holdS: Number(args.hold ?? 60),
  pollS: Number(args.poll ?? 30),
  maxWaitS: Number(args['max-wait'] ?? 21600),
  waitUntil: args['wait-until'] ? Number(args['wait-until']) : null,
};
const tStart = Date.now();

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.htm': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.cjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.map': 'application/json; charset=utf-8',
  '.wasm': 'application/wasm',
  '.wgsl': 'text/plain; charset=utf-8',
  '.txt': 'text/plain; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.ico': 'image/x-icon',
  '.onnx': 'application/octet-stream',
  '.bin': 'application/octet-stream',
  '.webmanifest': 'application/manifest+json',
};

const state = { chrome: null, server: null, profileDir: null, cleanupPromise: null, exiting: false, console: [], sampler: null, samples: [], chromeStderr: '' };

async function main() {
  if (!chromeBin) throw new Error('missing --chrome');
  const idleWait = args['wait-idle'] ? await waitIdle() : null;
  armDeadline();
  const loadStart = await sampleLoad();
  const port = await startServer();
  state.profileDir = mkdtempSync(join(os.tmpdir(), 'khc-chrome-'));
  const flags = [
    ...extraFlags,
    `--user-data-dir=${state.profileDir}`,
    '--remote-debugging-port=0',
    '--no-first-run',
    '--no-default-browser-check',
    '--disable-extensions',
    '--disable-background-timer-throttling',
    '--disable-renderer-backgrounding',
    '--disable-backgrounding-occluded-windows',
    'about:blank',
  ];
  state.chrome = spawn(chromeBin, flags, { detached: true, stdio: ['ignore', 'ignore', 'pipe'] });
  state.chrome.stderr.on('data', (d) => {
    state.chromeStderr = (state.chromeStderr + d).slice(-8000);
  });
  const wsPath = await waitDevToolsPort(state.profileDir, 30_000);
  const cdp = await Cdp.connect(`ws://127.0.0.1:${wsPath.port}${wsPath.path}`);
  const version = await cdp.send('Browser.getVersion');
  let gpuInfo = null;
  if (wantGpuInfo) {
    try {
      gpuInfo = await cdp.send('SystemInfo.getInfo');
    } catch (e) {
      gpuInfo = { error: String(e) };
    }
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

  state.samples.push(foreignLoad());
  state.sampler = setInterval(() => {
    state.samples.push(foreignLoad());
  }, 1000);
  const deadline = Date.now() + timeoutS * 1000;
  let outcome = null;
  while (Date.now() < deadline) {
    await sleep(250);
    let r;
    try {
      r = await cdp.send(
        'Runtime.evaluate',
        {
          expression: `(() => { const r = window.__kernelBenchResult, e = window.__kernelBenchError;
          if (r !== undefined && r !== null) return { kind: 'result', value: JSON.parse(JSON.stringify(r)) };
          if (e !== undefined && e !== null) return { kind: 'error', value: String(e) };
          return null; })()`,
          returnByValue: true,
        },
        sessionId,
      );
    } catch {
      continue;
    } // navigation in flight
    const v = r?.result?.value;
    if (v) {
      outcome = v;
      break;
    }
  }
  clearInterval(state.sampler);
  state.samples.push(foreignLoad());
  try {
    await Promise.race([cdp.send('Browser.close'), sleep(3000)]);
  } catch {}
  cdp.close();
  await cleanup();
  // After Chrome is gone, GPU util should be other people's, but the counters (ioreg
  // especially) lag our own work by several seconds: wait up to 15 s for them to settle.
  await sleep(3000);
  const settle = { waitedMs: 3000, readings: [] };
  for (let i = 0; i < 12; i++) {
    const g = gpuUtil(gpuLoad());
    settle.readings.push(g);
    if (g === null || g < IDLE.gpuMax) break;
    await sleep(1000);
    settle.waitedMs += 1000;
  }
  const loadEnd = await sampleLoad();
  loadEnd.gpuSettle = settle;
  // During-run contention uses only signals attributable to *other* processes (foreign
  // CPU from ps; foreign GPU from nvidia-smi compute apps on Linux and per-client
  // accumulatedGPUTime from ioreg on macOS). Raw load1 and device-wide GPU utilisation
  // taken mid-run include our own Chrome, so they are deliberately not used mid-run.
  // Any failed or missing reading makes the run *unverified* (never silently clean).
  const { during, contendedReasons, unverifiedReasons } = assessRun({
    platform: os.platform(),
    samples: state.samples,
    loadStart,
    loadEnd,
    gated: Boolean(args['wait-idle']) || os.platform() === 'darwin',
    th: IDLE,
  });
  const out = {
    ok: outcome?.kind === 'result',
    result: outcome?.kind === 'result' ? outcome.value : null,
    error: outcome?.kind === 'error' ? outcome.value : outcome ? null : `timeout after ${timeoutS}s`,
    meta: {
      host: os.hostname(),
      platform: `${os.platform()} ${os.release()} ${os.arch()}`,
      cpus: os.cpus().length,
      cpuModel: os.cpus()[0]?.model,
      browser: { binary: chromeBin, product: version.product, userAgent: version.userAgent },
      flags: extraFlags,
      url,
      wallMs: Date.now() - tStart,
      loadStart,
      loadEnd,
      duringRun: during,
      idleWait,
      contended: contendedReasons.length > 0,
      contendedReasons,
      verified: unverifiedReasons.length === 0,
      unverifiedReasons,
      noWait: !args['wait-idle'],
      console: state.console.slice(-60),
      ...(gpuInfo ? { gpuInfo } : {}),
      ...(outcome ? {} : { chromeStderrTail: state.chromeStderr.slice(-3000) }),
    },
  };
  await finish(out, out.ok ? 0 : 2);
}

// Exactly one terminal outcome per process: the first caller wins, later ones
// (a signal during cleanup, the hard deadline, a second error) wait on it.
let finishing = null;
function finish(out, code) {
  if (finishing) return finishing;
  finishing = (async () => {
    const line = JSON.stringify(out) + '\n';
    if (outFile) {
      try {
        writeFileSync(outFile + '.tmp', line);
        renameSync(outFile + '.tmp', outFile);
      } catch (e) {
        logErr(`writing ${outFile}: ${e}`);
      }
    }
    // Wait for stdout to drain: process.exit() can truncate a piped write.
    await new Promise((r) => process.stdout.write(line, () => r()));
    process.exit(code);
  })();
  return finishing;
}

function gpuUtil(g) {
  return g?.utilGpuPct ?? g?.deviceUtilPct ?? null;
}

async function waitIdle() {
  const t0 = Date.now(),
    end = t0 + IDLE.maxWaitS * 1000;
  const deadline = IDLE.waitUntil ? Math.min(end, IDLE.waitUntil) : end;
  const r = await waitForIdle({
    read: () => ({ load1: +os.loadavg()[0].toFixed(2), gpuUtilPct: gpuUtil(gpuLoad()) }),
    th: IDLE,
    deadline,
    holdMs: IDLE.holdS * 1000,
    pollMs: IDLE.pollS * 1000,
    sleep,
    log: logErr,
  });
  if (!r.ok)
    return busyExit(`host never idle for ${IDLE.holdS}s before the wait deadline (last load1=${r.last?.load1}, gpu=${r.last?.gpuUtilPct}%: ${r.reason})`);
  return { waitedS: Math.round(r.waitedMs / 1000), polls: r.polls, thresholds: IDLE, last: r.last };
}
function busyExit(msg) {
  // Never started: no Chrome or server exists yet. Returns a promise that never settles
  // past process exit, so callers can `return busyExit()`.
  return finish(
    {
      ok: false,
      result: null,
      error: `busy: ${msg}`,
      busy: true,
      meta: { host: os.hostname(), wallMs: Date.now() - tStart, loadavg: os.loadavg(), gpu: gpuLoad() },
    },
    75,
  ).then(() => new Promise(() => {}));
}
function logErr(s) {
  process.stderr.write(`[agent ${new Date().toISOString()}] ${s}\n`);
}

// Load attributable to processes outside this agent and its Chrome process group:
// CPU (cores, from ps %cpu), and GPU: NVIDIA compute apps on Linux, per-client
// accumulated GPU time (ioreg AGXDeviceUserClient) on macOS. cpuOk / gpuOk record
// whether each query succeeded, so a failed sample is never read as "idle".
// Caveat: Linux ps %cpu is a lifetime average; macOS's is a decaying recent average.
function foreignLoad() {
  const s = { t: Date.now(), load1: +os.loadavg()[0].toFixed(2), cpuOk: false, foreignCpuCores: null, gpuOk: false };
  const pg = state.chrome?.pid;
  const pgidOf = new Map();
  const ours = (pid) => pid === process.pid || (pg !== undefined && pgidOf.get(pid) === pg);
  try {
    const txt = execFileSync('ps', ['-A', '-o', 'pid=,pgid=,%cpu='], { encoding: 'utf8', timeout: 3000 });
    let sum = 0;
    for (const l of txt.trim().split('\n')) {
      const [pid, pgid, cpu] = l.trim().split(/\s+/).map(Number);
      pgidOf.set(pid, pgid);
      if (!ours(pid)) sum += cpu || 0;
    }
    s.foreignCpuCores = +(sum / 100).toFixed(2);
    s.cpuOk = pgidOf.size > 0;
  } catch (e) {
    s.cpuError = String(e.message ?? e).slice(0, 120);
  }
  try {
    if (!s.cpuOk) throw new Error('no process table to attribute GPU use');
    if (os.platform() === 'linux') {
      const apps = execFileSync('nvidia-smi', ['--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader,nounits'], {
        encoding: 'utf8',
        timeout: 3000,
      })
        .trim()
        .split('\n')
        .filter(Boolean)
        .map((l) => l.split(',').map((x) => x.trim()));
      s.foreignGpuApps = apps.filter(([pid]) => !ours(Number(pid))).map((a) => a.join(' '));
      s.gpuOk = true;
    } else if (os.platform() === 'darwin') {
      const clients = parseAgxClients(
        execFileSync('ioreg', ['-r', '-c', 'AGXDeviceUserClient', '-l', '-w', '0'], { encoding: 'utf8', timeout: 3000, maxBuffer: 64 << 20 }),
      );
      if (!clients.length) throw new Error('no AGXDeviceUserClient entries');
      s.foreignGpuClients = Object.fromEntries(clients.filter((c) => !ours(c.pid)).map((c) => [`${c.id}/${c.pid}`, c.gpuNs]));
      s.gpuOk = true;
    }
  } catch (e) {
    s.gpuError = String(e.message ?? e).slice(0, 120);
  }
  return s;
}

function pushConsole(s) {
  state.console.push(s.length > 2000 ? s.slice(0, 2000) + '…' : s);
  if (state.console.length > 500) state.console.shift();
}

function startServer() {
  const rootReal = realpathSync(root);
  return new Promise((res, rej) => {
    const server = createServer((req, resp) => {
      // Any filesystem/stream failure answers this request only; it must never throw
      // out of the handler (that would kill the agent and skip Chrome teardown).
      try {
        const r = resolveStatic(rootReal, req.url);
        if (!r.file) {
          resp.writeHead(r.status, headers('text/plain'));
          return resp.end(String(r.status));
        }
        resp.writeHead(200, { ...headers(MIME[extname(r.file).toLowerCase()] ?? 'application/octet-stream'), 'Content-Length': r.size });
        if (req.method === 'HEAD') return resp.end();
        const stream = createReadStream(r.file);
        stream.on('error', (e) => {
          logErr(`static ${req.url}: ${e.message}`);
          resp.destroy(e);
        });
        stream.pipe(resp);
      } catch (e) {
        logErr(`static ${req.url}: ${e?.message ?? e}`);
        try {
          if (!resp.headersSent) resp.writeHead(500, headers('text/plain'));
          resp.end();
        } catch {
          resp.destroy();
        }
      }
    });
    server.on('clientError', (_e, sock) => {
      try {
        sock.destroy();
      } catch {}
    });
    server.on('error', rej);
    server.listen(0, '127.0.0.1', () => {
      state.server = server;
      res(server.address().port);
    });
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
    this.ws = ws;
    this.id = 0;
    this.pending = new Map();
    this.handlers = new Map();
    ws.addEventListener('message', (ev) => {
      const m = JSON.parse(typeof ev.data === 'string' ? ev.data : Buffer.from(ev.data).toString());
      if (m.id !== undefined) {
        const p = this.pending.get(m.id);
        if (!p) return;
        this.pending.delete(m.id);
        m.error ? p.rej(new Error(`${m.error.message} ${m.error.data ?? ''}`)) : p.res(m.result);
      } else for (const h of this.handlers.get(m.method) ?? []) h(m.params, m.sessionId);
    });
    ws.addEventListener('close', () => {
      for (const p of this.pending.values()) p.rej(new Error('cdp closed'));
      this.pending.clear();
    });
  }
  send(method, params = {}, sessionId) {
    const id = ++this.id;
    const msg = { id, method, params };
    if (sessionId) msg.sessionId = sessionId;
    return new Promise((res, rej) => {
      this.pending.set(id, { res, rej });
      this.ws.send(JSON.stringify(msg));
    });
  }
  on(method, fn) {
    (this.handlers.get(method) ?? this.handlers.set(method, []).get(method)).push(fn);
  }
  close() {
    try {
      this.ws.close();
    } catch {}
  }
}

async function sampleLoad() {
  const s = { at: new Date().toISOString(), loadavg: os.loadavg().map((x) => +x.toFixed(2)), cpus: os.cpus().length };
  // 500 ms CPU busy sample across all cores.
  const a = os.cpus();
  await sleep(500);
  const b = os.cpus();
  let busy = 0,
    total = 0;
  for (let i = 0; i < a.length; i++) {
    const ta = a[i].times,
      tb = b[i].times;
    const tot = tb.user + tb.nice + tb.sys + tb.idle + tb.irq - (ta.user + ta.nice + ta.sys + ta.idle + ta.irq);
    busy += tot - (tb.idle - ta.idle);
    total += tot;
  }
  s.cpuBusyPct = total ? +((100 * busy) / total).toFixed(1) : null;
  s.gpu = gpuLoad();
  return s;
}

function gpuLoad() {
  try {
    if (os.platform() === 'linux') {
      const q = 'name,utilization.gpu,utilization.memory,memory.used,memory.total,clocks.sm,power.draw,temperature.gpu';
      const line = execFileSync('nvidia-smi', [`--query-gpu=${q}`, '--format=csv,noheader,nounits'], { encoding: 'utf8', timeout: 5000 })
        .trim()
        .split('\n')[0];
      const v = line.split(',').map((x) => x.trim());
      const procs = execFileSync('nvidia-smi', ['--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader,nounits'], {
        encoding: 'utf8',
        timeout: 5000,
      })
        .trim()
        .split('\n')
        .filter(Boolean)
        .map((l) =>
          l
            .split(',')
            .map((x) => x.trim())
            .join(' '),
        );
      return {
        source: 'nvidia-smi',
        name: v[0],
        utilGpuPct: +v[1],
        utilMemPct: +v[2],
        memUsedMiB: +v[3],
        memTotalMiB: +v[4],
        smClockMHz: +v[5],
        powerW: +v[6],
        tempC: +v[7],
        computeApps: procs,
      };
    }
    if (os.platform() === 'darwin') {
      const txt = execFileSync('ioreg', ['-r', '-d', '1', '-w', '0', '-c', 'IOAccelerator'], { encoding: 'utf8', timeout: 5000 });
      const pick = (k) => {
        const m = txt.match(new RegExp(`"${k}"=(\\d+)`));
        return m ? +m[1] : null;
      };
      return {
        source: 'ioreg',
        deviceUtilPct: pick('Device Utilization %'),
        rendererUtilPct: pick('Renderer Utilization %'),
        tilerUtilPct: pick('Tiler Utilization %'),
        inUseSysMemBytes: pick('In use system memory'),
      };
    }
  } catch (e) {
    return { error: String(e.message ?? e).slice(0, 200) };
  }
  return null;
}

// Idempotent: every caller gets the same in-flight promise, so a second fail()
// cannot exit before the first cleanup has reached SIGKILL.
function cleanup() {
  state.cleanupPromise ??= doCleanup();
  return state.cleanupPromise;
}
async function doCleanup() {
  const ch = state.chrome;
  if (ch && ch.exitCode === null && ch.signalCode === null) {
    try {
      process.kill(-ch.pid, 'SIGTERM');
    } catch {}
    const end = Date.now() + 5000;
    while (ch.exitCode === null && ch.signalCode === null && Date.now() < end) await sleep(100);
  }
  // Kill any leftovers in our own process group (helpers outlive the browser briefly).
  if (ch) {
    try {
      process.kill(-ch.pid, 'SIGKILL');
    } catch {}
  }
  if (state.server) state.server.close();
  if (state.profileDir && !args['keep-profile']) {
    await sleep(200);
    try {
      rmSync(state.profileDir, { recursive: true, force: true });
    } catch {}
  }
}

function fail(msg) {
  const out = {
    ok: false,
    result: null,
    error: msg,
    meta: {
      host: os.hostname(),
      wallMs: Date.now() - tStart,
      flags: extraFlags,
      console: state.console.slice(-60),
      chromeExit: state.chrome ? { code: state.chrome.exitCode, signal: state.chrome.signalCode } : null,
      chromeStderrTail: state.chromeStderr.slice(-3000),
    },
  };
  clearInterval(state.sampler);
  if (finishing) return finishing;
  return cleanup()
    .catch(() => {})
    .then(() => finish(out, 3));
}

function parseArgs(argv) {
  const o = {};
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (!a.startsWith('--')) continue;
    const eq = a.indexOf('=');
    if (eq > 0) {
      o[a.slice(2, eq)] = a.slice(eq + 1);
      continue;
    }
    const k = a.slice(2);
    const n = argv[i + 1];
    if (n === undefined || n.startsWith('--')) o[k] = true;
    else {
      o[k] = n;
      i++;
    }
  }
  return o;
}
function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

for (const sig of ['SIGINT', 'SIGTERM', 'SIGHUP']) process.on(sig, () => fail(`agent got ${sig}`));
process.on('uncaughtException', (e) => fail(`uncaught: ${e?.stack ?? e}`));
process.on('unhandledRejection', (e) => fail(`unhandled rejection: ${e?.stack ?? e}`));
if (args['watch-stdin']) {
  process.stdin.on('end', () => fail('stdin closed (controller gone)'));
  process.stdin.resume();
}
let hardDeadline = null;
const armDeadline = () => {
  hardDeadline = setTimeout(() => fail('agent hard deadline'), (timeoutS + 90) * 1000);
  hardDeadline.unref();
};
main().catch((e) => fail(String(e?.stack ?? e)));
