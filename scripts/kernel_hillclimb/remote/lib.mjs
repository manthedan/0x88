// Pure helpers shared by run.mjs (controller) and agent.mjs (remote side).
// Kept dependency-free and side-effect-free so tests/kernel_remote.test.mjs can cover them.

import { randomBytes } from 'node:crypto';
import { realpathSync, statSync } from 'node:fs';
import { join, normalize, sep } from 'node:path';

/** Collision-resistant run id: UTC second + slug + 48 random bits. */
export function makeRunId(name, now = new Date(), rand = () => randomBytes(6).toString('hex')) {
  const ts = now
    .toISOString()
    .replace(/[-:]/g, '')
    .replace(/\.\d+Z$/, 'Z');
  return `${ts}-${slug(name)}-${rand()}`;
}

export function slug(s) {
  return (
    String(s)
      .replace(/\.[a-z]+$/i, '')
      .replace(/[^A-Za-z0-9_.-]+/g, '-')
      .slice(0, 40) || 'page'
  );
}

/** gpu-queue -e value: a positive number of minutes, returned as a canonical string. */
export function validateEstimate(v) {
  const s = String(v ?? '5').trim();
  if (!/^\d+(\.\d+)?$/.test(s) || !(Number(s) > 0) || Number(s) > 24 * 60) {
    throw new Error(`--estimate-min must be a positive number of minutes (got ${JSON.stringify(v)})`);
  }
  return String(Number(s));
}

/** POSIX single-quote for a remote shell. */
export function q(s) {
  return `'${String(s).replace(/'/g, `'\\''`)}'`;
}

/**
 * Map a request path to a regular file inside rootReal (already realpath'd).
 * Containment is enforced on the *resolved* target, so symlinks (which rsync -a
 * preserves) cannot escape the served root. Returns { file, size } or { status }.
 * Fallbacks: directory -> index.html; missing /foo -> /foo.html.
 */
export function resolveStatic(rootReal, urlPath) {
  let p;
  try {
    p = decodeURIComponent(new URL(urlPath, 'http://x').pathname);
  } catch {
    return { status: 400 };
  }
  if (p.includes('\0')) return { status: 400 };
  const lexical = normalize(join(rootReal, p));
  if (!inside(rootReal, lexical)) return { status: 403 };
  for (const cand of candidates(lexical)) {
    let real;
    try {
      real = realpathSync(cand);
    } catch {
      continue;
    } // missing or dangling
    if (!inside(rootReal, real)) return { status: 403 };
    let st;
    try {
      st = statSync(real);
    } catch {
      continue;
    }
    if (st.isFile()) return { file: real, size: st.size };
  }
  return { status: 404 };
}
function candidates(lexical) {
  const out = [];
  let isDir = false;
  try {
    isDir = statSync(lexical).isDirectory();
  } catch {}
  if (isDir) out.push(join(lexical, 'index.html'));
  else out.push(lexical, `${lexical}.html`);
  return out;
}
function inside(root, p) {
  return p === root || p.startsWith(root.endsWith(sep) ? root : root + sep);
}

/**
 * Host lock wrapper. A kernel advisory lock (BSD lockf(1) on macOS, util-linux
 * flock(1) on Linux) is released by the kernel when the holder dies, so there is
 * no stale-owner reclamation (and no reclaim race) at all. Exit 75 on timeout.
 */
/**
 * Include patterns that name something the default excludes drop. rsync takes the first matching
 * rule and the default excludes come before the includes, so such an include silently syncs
 * nothing (a page then fails to load its model). A pattern names an excluded item when one of its
 * path segments, read as a literal string, matches an exclude glob: an excluded directory (exclude
 * ending in `/`, e.g. `models/`) or an excluded file glob (e.g. `net.onnx`, `*.onnx`,
 * `params_shard_*.bin`). The pattern's own wildcards can only match the exclude's `*` spans, so a
 * flagged segment loses every file it could match. Directory excludes only match directories, so
 * they apply to the last segment only when the include ends in `/`. Broad globs such as `assets/**`
 * or `x/*` are not flagged: the default excludes are meant to apply inside them. The check is a
 * best-effort guard, not a glob intersection: a bracket expression (`*.p[bt]`) or a pattern covered
 * only by several excludes together is not flagged.
 */
export function excludedIncludes(includes, excludes) {
  const toRe = (glob) =>
    new RegExp(
      `^${glob
        .replace(/[.+^${}()|[\]\\]/g, '\\$&')
        .replace(/\*/g, '[^/]*')
        .replace(/\?/g, '[^/]')}$`,
    );
  const dirs = excludes.filter((e) => e.endsWith('/')).map((e) => toRe(e.slice(0, -1)));
  const files = excludes.filter((e) => !e.endsWith('/')).map(toRe);
  return includes.filter((p) => {
    const segs = p.replace(/^\/+/, '').split('/').filter(Boolean);
    const lastIsDir = p.endsWith('/');
    return segs.some((seg, i) => ((i < segs.length - 1 || lastIsDir) && dirs.some((re) => re.test(seg))) || files.some((re) => re.test(seg)));
  });
}

export function lockCommand(platform, lockFile, timeoutS) {
  const t = String(Math.max(0, Math.floor(Number(timeoutS))));
  if (platform === 'darwin') return ['lockf', '-k', '-t', t, lockFile]; // lockf exits 75 (EX_TEMPFAIL) on timeout
  return ['flock', '-E', '75', '-w', t, lockFile];
}

/** Parse `gpu-queue list` (task-spooler `tsp -l`) rows. */
export function parseQueueList(txt) {
  return txt
    .split('\n')
    .slice(1)
    .filter(Boolean)
    .map((l) => {
      const m = l.match(/^(\d+)\s+(\w+)\s+(\S+)\s+(?:(-?\d+)\s+(\S+)\s+)?(.*)$/);
      if (!m) return null;
      const label = (m[6].match(/\[[^\]]*:[^\]]*:[\d.]+m\]/) ?? [''])[0];
      return { id: m[1], state: m[2], elevel: m[4] ?? null, label, raw: l };
    })
    .filter(Boolean);
}

/**
 * Idle gate decision for one poll. An unknown GPU reading never counts as idle:
 * it resets the hold, so a broken probe ends in the never-started (75) outcome.
 */
export function idleVerdict({ load1, gpuUtilPct }, { loadMax, gpuMax }) {
  if (gpuUtilPct === null || gpuUtilPct === undefined || !Number.isFinite(gpuUtilPct)) return { ok: false, reason: 'gpu utilisation unavailable' };
  if (!(load1 < loadMax)) return { ok: false, reason: `load1=${load1} >= ${loadMax}` };
  if (!(gpuUtilPct < gpuMax)) return { ok: false, reason: `gpu=${gpuUtilPct}% >= ${gpuMax}%` };
  return { ok: true, reason: null };
}

/**
 * Wait until `read()` passes `verdict` continuously for holdMs, or give up at `deadline`
 * (epoch ms). The deadline is checked before a completed hold is accepted, and every
 * sleep is bounded by the remaining budget, so nothing starts after the deadline.
 * Returns { ok, polls, last, reason, waitedMs }. Clock and sleep are injectable for tests.
 */
export async function waitForIdle({
  read,
  th,
  deadline,
  holdMs,
  pollMs,
  okPollMs = Math.min(pollMs, 10_000),
  verdict = idleVerdict,
  now = Date.now,
  sleep,
  log = () => {},
}) {
  const t0 = now();
  let idleSince = null,
    polls = 0,
    last = null,
    reason = null;
  while (true) {
    last = read();
    polls++;
    const v = verdict(last, th);
    const t = now();
    if (t > deadline) return { ok: false, polls, last, reason: v.reason ?? `hold of ${holdMs / 1000}s not completed`, waitedMs: t - t0 };
    if (v.ok) {
      idleSince ??= t;
      if (t - idleSince >= holdMs) return { ok: true, polls, last, reason: null, waitedMs: t - t0 };
    } else {
      idleSince = null;
      reason = v.reason;
      log(`waiting for idle (${reason}): load1=${last?.load1} gpu=${last?.gpuUtilPct}% waited ${Math.round((t - t0) / 1000)}s`);
    }
    // +1 ms: wake just past the deadline so the next poll returns the busy outcome.
    await sleep(Math.max(1, Math.min(v.ok ? okPollMs : pollMs, deadline - t + 1)));
  }
}

/**
 * Parse `ioreg -r -c AGXDeviceUserClient -l -w0` (Apple GPU user clients): one entry per
 * client with its creator pid and accumulated GPU time in ns (sum over its AppUsage
 * entries). No root needed; this is the per-process GPU attribution macOS exposes.
 */
export function parseAgxClients(txt) {
  const out = [];
  for (const block of String(txt)
    .split(/^\+-o /m)
    .slice(1)) {
    const id = block.match(/\bid (0x[0-9a-f]+)/i)?.[1];
    const pid = Number(block.match(/"IOUserClientCreator" = "pid (\d+)/)?.[1]);
    if (!id || !Number.isFinite(pid)) continue;
    const gpuNs = [...block.matchAll(/"accumulatedGPUTime"=(\d+)/g)].reduce((a, m) => a + Number(m[1]), 0);
    out.push({ id, pid, gpuNs });
  }
  return out;
}

/**
 * Summarise during-run samples. Each sample is
 *   { t, load1, cpuOk, foreignCpuCores, gpuOk, foreignGpuApps?: string[], foreignGpuClients?: {id: ns} }
 * where cpuOk/gpuOk say whether that sample's CPU / GPU-attribution query succeeded.
 * foreignGpuPct (macOS): GPU time accrued by non-Chrome clients between the first and last
 * valid GPU samples, as % of wall time. A client first seen after the baseline counts from 0
 * (it started during the run). Work by a pre-existing client that exits in its final
 * (<1 s) sampling interval is not counted.
 */
export function summarizeSamples(xs) {
  const cpu = xs.filter((x) => x.cpuOk).map((x) => x.foreignCpuCores);
  const gpu = xs.filter((x) => x.gpuOk);
  let foreignGpuPct = null,
    foreignGpuTop = [],
    foreignGpuPctMax = null,
    foreignGpuPeakAt = null;
  const cl = gpu.filter((x) => x.foreignGpuClients);
  // Per-interval utilisation between consecutive valid samples: a short foreign burst must
  // not hide in the whole-run average. A client absent from the previous sample started
  // during the interval and counts from 0.
  for (let i = 1; i < cl.length; i++) {
    const a = cl[i - 1],
      b = cl[i],
      dt = b.t - a.t;
    if (!(dt > 0)) continue;
    let ns = 0;
    for (const [id, v] of Object.entries(b.foreignGpuClients)) ns += Math.max(0, v - (a.foreignGpuClients[id] ?? 0));
    const pct = +(ns / 1e4 / dt).toFixed(2);
    if (foreignGpuPctMax === null || pct > foreignGpuPctMax) {
      foreignGpuPctMax = pct;
      foreignGpuPeakAt = b.t;
    }
  }
  if (cl.length >= 2) {
    const first = cl[0],
      last = cl[cl.length - 1],
      seen = new Map();
    for (const s of cl)
      for (const [id, ns] of Object.entries(s.foreignGpuClients)) {
        const e = seen.get(id);
        if (e) {
          e.last = ns;
        } else seen.set(id, { first: s === first ? ns : 0, last: ns });
      }
    const ms = last.t - first.t;
    const per = [...seen]
      .map(([id, e]) => [id, Math.max(0, e.last - e.first)])
      .filter(([, ns]) => ns > 0)
      .sort((a, b) => b[1] - a[1]);
    const ns = per.reduce((a, [, n]) => a + n, 0);
    if (ms > 0) {
      foreignGpuPct = +(ns / 1e4 / ms).toFixed(2); // ns / 1e6 / ms * 100
      foreignGpuTop = per.slice(0, 5).map(([id, n]) => `${id}:${+(n / 1e4 / ms).toFixed(2)}%`);
    }
  }
  return {
    samples: xs.length,
    cpuFailures: xs.length - cpu.length,
    gpuFailures: xs.length - gpu.length,
    load1Max: xs.length ? Math.max(...xs.map((x) => x.load1)) : null,
    foreignCpuCoresMax: cpu.length ? Math.max(...cpu) : null,
    foreignCpuCoresMean: cpu.length ? +(cpu.reduce((a, b) => a + b, 0) / cpu.length).toFixed(2) : null,
    foreignGpuApps: [...new Set(gpu.flatMap((x) => x.foreignGpuApps ?? []))],
    foreignGpuPct,
    foreignGpuPctMax,
    foreignGpuPeakAt,
    foreignGpuTop,
  };
}

/**
 * Contention verdict for a finished run. `contendedReasons`: someone else measurably used the
 * machine. `unverifiedReasons`: we could not check (a failed sample, an unknown reading, too
 * few samples). A run's timing is valid only when both lists are empty.
 * gated: the host uses the idle gate (mac-mini, or --wait-idle), so start/end loadavg and GPU
 * thresholds apply; otherwise (Yukon, inside gpu-queue) only foreign GPU apps and util count.
 */
export function assessRun({ platform, samples, loadStart, loadEnd, gated, th }) {
  const d = summarizeSamples(samples);
  const contended = [],
    unverified = [];
  if (d.samples < 2) unverified.push(`only ${d.samples} during-run samples`);
  if (d.cpuFailures) unverified.push(`${d.cpuFailures}/${d.samples} during-run CPU samples failed`);
  if (d.gpuFailures) unverified.push(`${d.gpuFailures}/${d.samples} during-run GPU attribution samples failed`);
  if (d.foreignGpuApps.length) contended.push(`foreign GPU compute apps during run: ${d.foreignGpuApps.join('; ')}`);
  if (platform === 'darwin') {
    if (d.foreignGpuPctMax === null) unverified.push('no attributable GPU reading during run');
    else if (d.foreignGpuPctMax >= th.gpuMax)
      contended.push(
        `foreign GPU peaked at ${d.foreignGpuPctMax}% >= ${th.gpuMax}% in one sampling interval (run mean ${d.foreignGpuPct}%; ${d.foreignGpuTop.join(', ')})`,
      );
  }
  const util = (g) => g?.utilGpuPct ?? g?.deviceUtilPct ?? null;
  for (const [tag, l] of [
    ['start', loadStart],
    ['end', loadEnd],
  ]) {
    const g = util(l?.gpu);
    if (g === null || !Number.isFinite(g)) unverified.push(`gpu util unknown at ${tag}${l?.gpu?.error ? `: ${l.gpu.error}` : ''}`);
    else if (g >= th.gpuMax) contended.push(`gpu util ${g}% >= ${th.gpuMax}% at ${tag}`);
    if (gated) {
      if (l?.loadavg?.[0] >= th.loadMax) contended.push(`loadavg1 ${l.loadavg[0]} >= ${th.loadMax} at ${tag}`);
    } else {
      const apps = l?.gpu?.computeApps ?? [];
      if (apps.length) contended.push(`GPU compute apps at ${tag}: ${apps.join('; ')}`);
    }
  }
  if (gated && d.foreignCpuCoresMax !== null && d.foreignCpuCoresMax >= th.loadMax)
    contended.push(`foreign CPU ${d.foreignCpuCoresMax} cores >= ${th.loadMax} during run`);
  return { during: d, contendedReasons: contended, unverifiedReasons: unverified };
}

/**
 * Whether a result's timing can be trusted. Requires positive evidence, never the absence of
 * a negative: the agent verified the run (meta.verified === true, meta.contended === false),
 * and on idle-gated hosts the agent itself reports that the idle gate ran
 * (meta.noWait === false plus an idleWait record). Controller-side run.json is only ever
 * allowed to *invalidate* (noWait: true), since fetching it can fail.
 */
export function timingValidity({ scheduler, j, run }) {
  const m = j?.meta;
  if (!(m?.verified === true && m?.contended === false)) return false;
  if (scheduler === 'idle-wait') return m.noWait === false && m.idleWait != null && run?.noWait !== true;
  return true;
}
