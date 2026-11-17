// Shared helpers for the perf matrix runner: OS footprint sampling (macOS), stats, report shaping.
import { execFile, execFileSync } from 'node:child_process';
import { existsSync, mkdirSync, statSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
export const repoRoot = resolve(here, '..', '..');
export const MB = 1024 * 1024;
export const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/** Compile the pidfp helper into the ignored artifacts dir on first use. */
export function ensurePidfp() {
  const src = join(here, 'pidfp.c');
  const bin = join(repoRoot, '.local-dev-artifacts', 'perf-matrix', 'bin', 'pidfp');
  if (!existsSync(bin) || statSync(bin).mtimeMs < statSync(src).mtimeMs) {
    mkdirSync(dirname(bin), { recursive: true });
    execFileSync('clang', ['-O2', '-o', bin, src], { stdio: 'inherit' });
  }
  return bin;
}

let pidfpBin;
// pid -> Map(threadId -> max CPU ns seen): keeps compiler threads' time after they exit.
const threadCpu = new Map();
function threadSum(pid, kind, field) {
  let seen = threadCpu.get(`${pid}:${kind}`);
  if (!seen) {
    seen = new Map();
    threadCpu.set(`${pid}:${kind}`, seen);
  }
  for (const entry of field.slice(2).split(',').filter(Boolean)) {
    const [tid, ns] = entry.split('=');
    seen.set(tid, Math.max(seen.get(tid) ?? 0, Number(ns)));
  }
  let sum = 0;
  for (const ns of seen.values()) sum += ns;
  return sum / 1e6;
}
/**
 * Kernel footprint counters, MB (1 MB = 2^20 bytes, matching `footprint`'s units).
 * reset=true resets each pid's interval high-water mark after reading it.
 */
export function readFootprint(pids, { reset = false, threads = false } = {}) {
  pidfpBin ??= ensurePidfp();
  const out = new Map();
  if (!pids.length) return out;
  const txt = execFileSync(pidfpBin, [...(reset ? ['-r'] : []), ...(threads ? ['-t'] : []), ...pids.map(String)], { encoding: 'utf8' });
  for (const line of txt.trim().split('\n')) {
    const [pid, cur, life, interval, cpu, wasmJit, jsJit] = line.split(' ');
    if (cur === 'err' || cur === undefined) continue;
    const v = { cur: Number(cur) / MB, life: Number(life) / MB, interval: Number(interval) / MB };
    // threads=true: cumulative CPU ms (whole task; JSC Wasm / JS tier-up compiler threads).
    if (cpu !== undefined) Object.assign(v, { cpuMs: Number(cpu) / 1e6, wasmJitMs: threadSum(pid, 'w', wasmJit), jsJitMs: threadSum(pid, 'j', jsJit) });
    out.set(Number(pid), v);
  }
  return out;
}

export function pgrep(pattern) {
  try {
    return execFileSync('pgrep', ['-f', pattern], { encoding: 'utf8' }).trim().split('\n').filter(Boolean).map(Number);
  } catch {
    return [];
  }
}

/** Per-category breakdown via `footprint` (slow: 0.1-1 s; call at a few marks only, async). */
export function footprintCategories(pid) {
  return new Promise((resolveCats) => {
    const started = Date.now();
    execFile('footprint', ['-p', String(pid)], { encoding: 'utf8', maxBuffer: 16 * MB }, (error, txt) => {
      if (error || !txt) return resolveCats(null);
      const unit = (v, u) => Number(v) * (u === 'GB' ? 1024 : u === 'KB' ? 1 / 1024 : u === 'B' ? 1 / MB : 1);
      const cats = {};
      for (const line of txt.split('\n')) {
        const m = /^\s*([\d.]+) (B|KB|MB|GB)\s+[\d.]+ (?:B|KB|MB|GB)\s+[\d.]+ (?:B|KB|MB|GB)\s+\d+\s+(.+)$/.exec(line);
        if (m) {
          const v = unit(m[1], m[2]);
          if (v >= 5) cats[m[3].trim()] = Math.round(v);
        }
      }
      const total = /Footprint: ([\d.]+) (KB|MB|GB)/.exec(txt);
      const peak = /phys_footprint_peak: ([\d.]+) (KB|MB|GB)/.exec(txt);
      resolveCats({
        totalMb: total ? Math.round(unit(total[1], total[2])) : null,
        peakMb: peak ? Math.round(unit(peak[1], peak[2])) : null,
        cats,
        tookMs: Date.now() - started,
      });
    });
  });
}

export const median = (xs) => {
  const v = xs.filter((x) => Number.isFinite(x)).sort((a, b) => a - b);
  if (!v.length) return null;
  const m = v.length >> 1;
  return v.length % 2 ? v[m] : (v[m - 1] + v[m]) / 2;
};
export function summarize(xs) {
  const v = xs.filter((x) => Number.isFinite(x));
  if (!v.length) return { n: 0, median: null, min: null, max: null, range: null, mad: null };
  const med = median(v);
  return {
    n: v.length,
    median: Math.round(med),
    min: Math.round(Math.min(...v)),
    max: Math.round(Math.max(...v)),
    range: Math.round(Math.max(...v) - Math.min(...v)),
    mad: Math.round(median(v.map((x) => Math.abs(x - med)))),
  };
}
export const round = (x, d = 0) => (Number.isFinite(x) ? Math.round(x * 10 ** d) / 10 ** d : null);

/** Soak lab URL for a cell. `params` are passed through verbatim (ortDequantFold, ortGraphOpt, lqoModel, assetBase...). */
export function soakUrl(base, net, params = {}, auto = false) {
  const q = new URLSearchParams({ auto: auto ? '1' : '0', steps: net, ...params });
  return `${base.replace(/\/$/, '')}/lab/mobile-soak/?${q}`;
}

/** Observe effective worker settings without changing init payloads or model loading. */
export const WORKER_READY_SCRIPT = `
window.__perfWorkerReady = [];
const NativeWorker = Worker;
window.Worker = class extends NativeWorker {
  constructor(...args) {
    super(...args);
    this.addEventListener('message', e => {
      if (e.data?.type === 'ready') {
        const {backend, ortIoMode, modelCache, ortRuntime, servedModelUrl} = e.data;
        window.__perfWorkerReady.push({backend, ortIoMode, modelCache, ortRuntime, servedModelUrl});
      }
    });
  }
}; return true;`;

/** In-page probe: compact progress, safe to run every ~250 ms. */
export const PROGRESS_SCRIPT = `
const s = window.__mobileSoak; if (!s) return null;
const r = s.results;
return { state: r.state, inProgress: r.inProgress && r.inProgress.id,
  steps: (r.steps || []).map((x) => ({ id: x.id, status: x.status, phases: (x.memory || []).map((m) => m.phase), searches: (x.searches || []).length })) };`;

/** Phase from progress: baseline -> load -> search -> dispose -> settle. */
export function phaseOf(progress, started) {
  if (!started || !progress) return 'baseline';
  if (progress.state === 'done' || progress.state === 'crashed-awaiting-choice') return 'settle';
  const step = progress.steps?.at(-1);
  if (!step) return 'load';
  const ph = step.phases ?? [];
  if (ph.includes('after-search')) return 'dispose';
  if (ph.includes('resident-after-load') || ph.includes('after-prepare')) return 'search';
  return 'load';
}

/** Extract timing/speed/cache evidence from the soak report's single step. */
export function digestReport(report) {
  const step = report?.steps?.at(-1);
  if (!step) return { status: 'no-report' };
  const rt = JSON.stringify(step.runtime ?? {});
  const mid = step.searches?.find((s) => s.fen && !s.fen.startsWith('rnbqkbnr/pppppppp')) ?? step.searches?.[1];
  return {
    status: step.status,
    error: step.error,
    loadMs: round(step.loadMs),
    prepareMs: round(step.prepareMs),
    loadDeferred: step.loadDeferred,
    firstSearchMs: round(step.firstSearchMs),
    firstSearchPerSec: round(step.searches?.[0]?.perSecond, 1),
    midgamePerSec: round(mid?.perSecond, 1),
    disposeMs: round(step.disposeMs),
    deviceLost: step.deviceLost?.length ? step.deviceLost : undefined,
    // Timed searches (heat, labSearchSeconds): throughput per ~10 s slice.
    slices: step.slices?.length ? step.slices.map((x) => [Math.round(x.elapsedMs), round(x.perSecond, 1)]) : undefined,
    dequantFold: /"dequantFoldEnabled":(true|false)/.exec(rt)?.[1] ?? null,
    runtime: step.runtime,
  };
}
