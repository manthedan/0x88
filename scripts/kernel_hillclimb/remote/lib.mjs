// Pure helpers shared by run.mjs (controller) and agent.mjs (remote side).
// Kept dependency-free and side-effect-free so tests/kernel_remote.test.mjs can cover them.
import { realpathSync, statSync } from 'node:fs';
import { join, normalize, sep } from 'node:path';
import { randomBytes } from 'node:crypto';

/** Collision-resistant run id: UTC second + slug + 48 random bits. */
export function makeRunId(name, now = new Date(), rand = () => randomBytes(6).toString('hex')) {
  const ts = now.toISOString().replace(/[-:]/g, '').replace(/\.\d+Z$/, 'Z');
  return `${ts}-${slug(name)}-${rand()}`;
}

export function slug(s) {
  return String(s).replace(/\.[a-z]+$/i, '').replace(/[^A-Za-z0-9_.-]+/g, '-').slice(0, 40) || 'page';
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
export function q(s) { return `'${String(s).replace(/'/g, `'\\''`)}'`; }

/**
 * Map a request path to a regular file inside rootReal (already realpath'd).
 * Containment is enforced on the *resolved* target, so symlinks (which rsync -a
 * preserves) cannot escape the served root. Returns { file, size } or { status }.
 * Fallbacks: directory -> index.html; missing /foo -> /foo.html.
 */
export function resolveStatic(rootReal, urlPath) {
  let p;
  try { p = decodeURIComponent(new URL(urlPath, 'http://x').pathname); } catch { return { status: 400 }; }
  if (p.includes('\0')) return { status: 400 };
  const lexical = normalize(join(rootReal, p));
  if (!inside(rootReal, lexical)) return { status: 403 };
  for (const cand of candidates(lexical)) {
    let real;
    try { real = realpathSync(cand); } catch { continue; } // missing or dangling
    if (!inside(rootReal, real)) return { status: 403 };
    let st;
    try { st = statSync(real); } catch { continue; }
    if (st.isFile()) return { file: real, size: st.size };
  }
  return { status: 404 };
}
function candidates(lexical) {
  const out = [];
  let isDir = false;
  try { isDir = statSync(lexical).isDirectory(); } catch {}
  if (isDir) out.push(join(lexical, 'index.html'));
  else out.push(lexical, `${lexical}.html`);
  return out;
}
function inside(root, p) { return p === root || p.startsWith(root.endsWith(sep) ? root : root + sep); }

/**
 * Host lock wrapper. A kernel advisory lock (BSD lockf(1) on macOS, util-linux
 * flock(1) on Linux) is released by the kernel when the holder dies, so there is
 * no stale-owner reclamation (and no reclaim race) at all. Exit 75 on timeout.
 */
export function lockCommand(platform, lockFile, timeoutS) {
  const t = String(Math.max(0, Math.floor(Number(timeoutS))));
  if (platform === 'darwin') return ['lockf', '-k', '-t', t, lockFile]; // lockf exits 75 (EX_TEMPFAIL) on timeout
  return ['flock', '-E', '75', '-w', t, lockFile];
}

/** Parse `gpu-queue list` (task-spooler `tsp -l`) rows. */
export function parseQueueList(txt) {
  return txt.split('\n').slice(1).filter(Boolean).map((l) => {
    const m = l.match(/^(\d+)\s+(\w+)\s+(\S+)\s+(?:(-?\d+)\s+(\S+)\s+)?(.*)$/);
    if (!m) return null;
    const label = (m[6].match(/\[[^\]]*:[^\]]*:[\d.]+m\]/) ?? [''])[0];
    return { id: m[1], state: m[2], elevel: m[4] ?? null, label, raw: l };
  }).filter(Boolean);
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
