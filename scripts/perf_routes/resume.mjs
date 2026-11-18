// Resume bookkeeping: a run directory may only continue the exact cohort it started.
import { existsSync, readFileSync, renameSync, writeFileSync } from 'node:fs';
import { dirname, relative, resolve } from 'node:path';
import { cleanupMarker } from './lifecycle.mjs';

export const sessionId = (cell, i) => `${String(i).padStart(3, '0')}-${cell.browser}-${cell.journey}-${cell.arm}-r${cell.repeat}`;

export function readJson(file) {
  try {
    return JSON.parse(readFileSync(file, 'utf8'));
  } catch {
    return null; // missing or truncated
  }
}

// A crash mid-write must not leave a truncated record that a resume trusts or chokes on.
export function writeJsonAtomic(file, value) {
  writeFileSync(`${file}.tmp`, JSON.stringify(value, null, 2));
  renameSync(`${file}.tmp`, file);
}

// Everything that changes which sessions exist or what they measure. Lock/driver/output paths do not.
// Arms are identified by resolved bytes (SHA, prebuilt version, URL); the typed ref is provenance only.
export function runManifest(options, arms, { host, harness }) {
  // An externally served arm has no resolvable bytes: its operator-supplied identifier is the identity.
  const identity = ({ input, ...rest }) => (rest.kind === 'externally-served' ? { input, ...rest } : rest);
  return {
    version: 1,
    options: { browser: options.browser, repeats: options.repeats, journeys: options.journeys },
    arms: { baseline: identity(arms.baseline), candidate: identity(arms.candidate) },
    inputs: { baseline: arms.baseline.input, candidate: arms.candidate.input },
    host,
    harness,
  };
}

const flatten = (value, prefix = '', out = {}) => {
  if (value && typeof value === 'object' && !Array.isArray(value))
    for (const [key, v] of Object.entries(value)) flatten(v, prefix ? `${prefix}.${key}` : key, out);
  else out[prefix] = JSON.stringify(value);
  return out;
};

export function manifestMismatches(saved, current) {
  const a = flatten({ ...saved, inputs: null }),
    b = flatten({ ...current, inputs: null });
  return [...new Set([...Object.keys(a), ...Object.keys(b)])].filter((key) => a[key] !== b[key]).sort();
}

// Source files a session's measurements depend on: session.mjs's relative import graph (including
// the chess modules that choose moves), files it reads by URL (probe.js), plus `extras` that are
// used without an import (pidfp.c, the arm static server). Missing files are skipped (mini workspace).
export function measuringFiles(root, entry, extras = []) {
  const seen = new Set();
  const visit = (file) => {
    if (seen.has(file) || !existsSync(file)) return;
    seen.add(file);
    const source = readFileSync(file, 'utf8');
    for (const match of source.matchAll(/\bfrom\s+'(\.{1,2}\/[^']+)'|new URL\('(\.{1,2}\/[^']+)', import\.meta\.url\)/g))
      visit(resolve(dirname(file), match[1] ?? match[2]));
  };
  for (const file of [entry, ...extras]) visit(resolve(root, file));
  return [...seen].map((file) => relative(root, file)).sort();
}

// A build record is written last (atomically); anything unreadable or for another commit is a miss.
export function reusableBuild(recordFile, commit) {
  const built = readJson(recordFile);
  return built?.commit === commit && existsSync(resolve(built.dist, 'app/analysis/index.html')) ? built : null;
}

// Operator stops and Safari-busy attempts are not outcomes (a fresh run would have retried them). A
// `.cancelled` marker alone is not evidence: a benchlock cancel can land after the browser started.
const notAnOutcome = (result) => result.status !== 'ok' && /^(?:Error: )?Interrupted\b|already paired/.test(result.error ?? '');

// Reuse every recorded outcome, failures included: retrying only failures until clean would be
// selective outlier removal. Missing, unreadable or interrupted sessions run again.
export function selectSessions(cells, inspect) {
  return cells.map((cell, i) => {
    const id = sessionId(cell, i);
    const { result, cleanupPending, owner } = inspect(id);
    if (owner)
      throw Error(
        `Session ${id} is still owned by live process ${owner} (orphaned by an earlier coordinator); stop it before resuming, or delete its .owner record if that PID is unrelated`,
      );
    if (cleanupPending)
      throw Error(`Browser cleanup unconfirmed for ${id}; confirm the browser is closed and remove its .cleanup-pending marker before resuming`);
    if (!result || notAnOutcome(result)) return { cell, id, reuse: null };
    for (const key of ['browser', 'journey', 'arm', 'repeat'])
      if (result[key] !== cell[key]) throw Error(`Session file ${id} records a different ${key}: ${result[key]}`);
    return { cell, id, reuse: result };
  });
}

// The coordinator's verdict on one attempt. Failures only the coordinator saw (no result file, cleanup
// unconfirmed) are written into the session file, which is what a resume reads.
export function recordAttempt(cell, resultFile, { status, cleanupConfirmed, interruption, lockDir }) {
  const written = readJson(resultFile);
  const row = written ?? { ...cell, status: 'invalid', error: interruption ?? `session process exited ${status}` };
  if (!cleanupConfirmed) {
    row.status = 'invalid';
    row.error = `Browser cleanup unconfirmed; see ${cleanupMarker(resultFile)}${lockDir ? `; mutex retained at ${lockDir}` : ''}`;
  }
  if (!written || !cleanupConfirmed) writeJsonAtomic(resultFile, row);
  return row;
}
