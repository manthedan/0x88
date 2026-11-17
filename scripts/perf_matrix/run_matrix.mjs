#!/usr/bin/env node
// Browser memory/speed matrix runner (desktop Safari + headless Chrome on macOS).
// Protocol and rationale: docs/mobile_memory_2026-11-17.md, Phase 2.
// Copied from lab/perf-harness (docs/perf_measurement_protocol_2026-11-17.md).
//
//   node scripts/perf_matrix/run_matrix.mjs --cells cells.json --base http://127.0.0.1:5181 --out <dir> \
//       [--repeats 5] [--settle-ms 10000] [--benchlock <path>] [--dry-run]
//   node scripts/perf_matrix/run_matrix.mjs --report <dir>     # re-aggregate existing runs only
//
// cells.json: { "repeats": 5, "cells": [
//   { "browser": "safari", "net": "lqo", "cache": ["cold", "warm"], "params": {}, "repeats": 5 },
//   { "browser": "chrome", "net": "lqo", "cache": ["cold"], "params": { "ortDequantFold": "0" } },
//   { "browser": "safari", "net": "t3", "cache": ["cold"], "build": "720aaf4a", "base": "http://127.0.0.1:5182" } ] }
// Shorthand instead of a file: --cell safari:lqo:cold+warm[:k=v&k=v] (repeatable).
//
// Each repeat of a (browser, net, params) group is one browser session: a cold load, then (if warm is
// requested) a warm load in a NEW page process with the model already in Cache Storage. Groups are
// interleaved across repeats so drift (thermal, background load) spreads over all cells. Safari sessions
// are wrapped in the machine-wide benchlock; Chrome sessions are independent browser instances.
// Output: <out>/runs.jsonl (one line per measured load, appended as sessions finish; reruns resume),
// <out>/summary.json and <out>/summary.md.
import { spawnSync } from 'node:child_process';
import { appendFileSync, existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { hostname } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { ensurePidfp, summarize } from './common.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const argv = process.argv.slice(2);
const opt = (name, fallback) => {
  const i = argv.indexOf(`--${name}`);
  return i >= 0 ? argv[i + 1] : fallback;
};
const optAll = (name) => argv.flatMap((a, i) => (a === `--${name}` ? [argv[i + 1]] : []));

if (opt('report')) {
  aggregate(resolve(opt('report')));
  process.exit(0);
}

const out = resolve(opt('out', `.local-dev-artifacts/perf-matrix/${new Date().toISOString().replace(/[:.]/g, '-')}`));
const base = opt('base');
const fileSpec = opt('cells') ? JSON.parse(readFileSync(opt('cells'), 'utf8')) : { cells: [] };
const cells = [
  ...fileSpec.cells,
  ...optAll('cell').map((s) => {
    const [browser, net, cache, query] = s.split(':');
    return { browser, net, cache: cache.split('+'), params: Object.fromEntries(new URLSearchParams(query ?? '')) };
  }),
];
if (!cells.length) throw new Error('no cells: pass --cells <file> or --cell browser:net:cold+warm[:k=v]');
const repeats = Number(opt('repeats', fileSpec.repeats ?? 5));
const settleMs = Number(opt('settle-ms', fileSpec.settleMs ?? 10_000));
const benchlock = opt('benchlock', process.env.PERF_BENCHLOCK);
const lockDir = opt('lock-dir', process.env.PERF_LOCK_DIR);
const driver = opt('driver', process.env.PERF_SAFARI_DRIVER);
if (benchlock && lockDir) throw new Error('pass --benchlock or --lock-dir, not both');
if (benchlock && !existsSync(benchlock)) throw new Error(`benchlock not found: ${benchlock}`);
mkdirSync(join(out, 'sessions'), { recursive: true });
ensurePidfp();

// Group cells by (browser, net, params); a session covers cold and/or warm of one group.
const groups = new Map();
for (const c of cells) {
  // `build` labels a cell's server (e.g. two builds of different commits on two ports via per-cell `base`).
  const cellBase = c.base ?? base;
  if (!cellBase) throw new Error('--base http://127.0.0.1:<port> (isolated static server for dist-client) or a per-cell "base" is required');
  const key = JSON.stringify([c.browser, c.net, c.params ?? {}, c.build ?? '']);
  const g = groups.get(key) ?? { browser: c.browser, net: c.net, params: c.params ?? {}, build: c.build ?? '', base: cellBase, warm: false, repeats: 0 };
  g.warm ||= (c.cache ?? ['cold']).includes('warm');
  g.repeats = Math.max(g.repeats, c.repeats ?? repeats);
  groups.set(key, g);
}
const plan = [];
const maxRepeats = Math.max(...[...groups.values()].map((g) => g.repeats));
for (let r = 0; r < maxRepeats; r++)
  for (const g of groups.values()) if (r < g.repeats) plan.push({ ...g, repeat: r, plan: g.warm ? ['cold', 'warm-fresh'] : ['cold'] });

const runsFile = join(out, 'runs.jsonl');
const done = new Set(
  existsSync(runsFile)
    ? readFileSync(runsFile, 'utf8')
        .trim()
        .split('\n')
        .filter(Boolean)
        .map((l) => JSON.parse(l).sessionId)
    : [],
);
console.log(`${plan.length} sessions (${groups.size} groups, up to ${maxRepeats} repeats) -> ${out}${done.size ? `; ${done.size} sessions already done` : ''}`);
if (argv.includes('--dry-run')) {
  for (const s of plan) console.log(s.browser, s.net, JSON.stringify(s.params), s.plan.join('+'), `repeat ${s.repeat}`);
  process.exit(0);
}
for (const [i, s] of plan.entries()) {
  const sessionId = `${String(i).padStart(3, '0')}-${s.browser}-${s.build ? `${s.build}-` : ''}${s.net}-${new URLSearchParams(s.params).toString().replace(/[^\w=-]/g, '_') || 'default'}-r${s.repeat}`;
  if (done.has(sessionId)) continue;
  const specPath = join(out, 'sessions', `${sessionId}.spec.json`);
  const resultPath = join(out, 'sessions', `${sessionId}.json`);
  writeFileSync(
    specPath,
    JSON.stringify(
      {
        base: s.base,
        build: s.build,
        net: s.net,
        params: s.params,
        plan: s.plan,
        settleMs,
        ...(driver ? { driver } : {}),
        ...(fileSpec.sessionDefaults ?? {}),
        ...(s.browser === 'safari' && (benchlock || lockDir) ? { sessionWaitMs: 0 } : {}),
      },
      null,
      1,
    ),
  );
  const script = join(here, s.browser === 'safari' ? 'safari_session.mjs' : 'chrome_session.mjs');
  const cmd = benchlock ? [benchlock, process.execPath, script, specPath, resultPath] : [process.execPath, script, specPath, resultPath];
  console.log(`[${i + 1}/${plan.length}] ${sessionId}`);
  let res;
  do {
    const locked = s.browser === 'safari' && lockDir ? acquireLock(lockDir) : false;
    try {
      res = spawnSync(cmd[0], cmd.slice(1), { stdio: 'inherit' });
    } finally {
      if (locked) releaseLock(lockDir);
    }
    if (s.browser === 'safari' && (benchlock || lockDir) && res.status === 75) {
      // No session was created. Do not monopolize the mutex while another owner
      // may need it to finish its existing session; retry outside the lock.
      console.log('Shared Safari busy; benchlock released, retrying in 30 s.');
      Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, 30000);
    }
  } while (s.browser === 'safari' && (benchlock || lockDir) && res.status === 75);
  if (res.status !== 0 || !existsSync(resultPath)) {
    appendFileSync(
      runsFile,
      `${JSON.stringify({ sessionId, build: s.build, browser: s.browser, net: s.net, params: s.params, cache: 'session-failed', exit: res.status, error: res.error?.message })}\n`,
    );
    continue;
  }
  for (const run of JSON.parse(readFileSync(resultPath, 'utf8'))) {
    const { samples, ...rest } = run; // full time series stays in sessions/<id>.json
    appendFileSync(runsFile, `${JSON.stringify({ sessionId, build: s.build, ...rest, samplesFile: resultPath })}\n`);
  }
}
aggregate(out);

/** Shared mini mutex (upstream protocol): never take over a live/unknown owner's lock. */
function acquireLock(dir) {
  for (;;) {
    try {
      mkdirSync(dir);
      writeFileSync(join(dir, 'owner'), `${hostname()} ${process.pid} ${new Date().toISOString()}\n`);
      return true;
    } catch (error) {
      if (error.code !== 'EEXIST') throw error;
    }
    let owner = '';
    try {
      owner = readFileSync(join(dir, 'owner'), 'utf8').trim();
    } catch {}
    const [host, pid] = owner.split(' ');
    let gone = false;
    if (host === hostname() && Number(pid) > 0) {
      try {
        process.kill(Number(pid), 0);
      } catch (error) {
        gone = error.code === 'ESRCH';
      }
    }
    // Do not recursively remove a foreign lock: two stale-lock takers could
    // otherwise erase a newly acquired live lock between checking and unlinking.
    console.log(`mini lock ${gone ? 'has a stale owner (needs owner cleanup)' : 'held by'} ${owner || '(unknown owner)'}; waiting`);
    Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, 5000);
  }
}
function releaseLock(dir) {
  try {
    const [host, pid] = readFileSync(join(dir, 'owner'), 'utf8').split(' ');
    if (host === hostname() && Number(pid) === process.pid) rmSync(dir, { recursive: true, force: true });
  } catch {}
}

/** Validity rules for a measured load (see protocol doc). */
function validity(r) {
  const problems = [];
  if (r.cache === 'session-failed') return ['session failed'];
  if (r.report?.status !== 'ok') problems.push(`status ${r.report?.status}`);
  if (r.crashed) problems.push('page process died');
  if (r.pageReadyMs > 60_000) problems.push(`navigation stalled ${Math.round(r.pageReadyMs / 1000)} s`);
  if (r.timedOut) problems.push('timed out');
  if (r.cacheVerified === false) problems.push(r.cache === 'cold' ? 'cold run fetched no model bytes' : 'warm run fetched model bytes');
  if (r.pidReused) problems.push('page process reused from previous load');
  if (r.context && (r.context.secure !== true || r.context.crossOriginIsolated !== true)) problems.push('page not secure and cross-origin isolated');
  if (r.lifeMaxAtBaselineMb > (r.baselineMb ?? 0) + 100) problems.push(`process already peaked at ${r.lifeMaxAtBaselineMb} MB before start`);
  if (r.browser === 'safari' && r.cache === 'warm-fresh' && r.previousPageProcessExited === false) problems.push('previous page process still alive');
  return problems;
}

function aggregate(dir) {
  const rows = readFileSync(join(dir, 'runs.jsonl'), 'utf8')
    .trim()
    .split('\n')
    .filter(Boolean)
    .map((l) => JSON.parse(l));
  const cellsOut = new Map();
  for (const r of rows) {
    const cache = r.cache === 'warm-fresh' ? 'warm' : r.cache;
    const key = JSON.stringify([r.browser, r.build ?? '', r.net, r.params ?? {}, cache]);
    const c = cellsOut.get(key) ?? { browser: r.browser, build: r.build ?? '', net: r.net, params: r.params ?? {}, cache, runs: [], invalid: [] };
    const problems = validity(r);
    if (problems.length) c.invalid.push({ sessionId: r.sessionId, problems });
    else c.runs.push(r);
    cellsOut.set(key, c);
  }
  const metric = (c, f) => summarize(c.runs.map(f));
  const summary = [...cellsOut.values()].map((c) => ({
    browser: c.browser,
    build: c.build,
    net: c.net,
    params: c.params,
    cache: c.cache,
    n: c.runs.length,
    invalid: c.invalid,
    // Primary: page-process kernel high-water mark from navigation through the end of the test search.
    loadPeakMb: metric(c, (r) => r.peakThroughSearchMb),
    searchMedianMb: metric(c, (r) => r.searchMedianMb),
    // Teardown: Safari keeps compiling ORT wasm (OMG tier) after the worker is terminated; see doc.
    lifetimePeakMb: metric(c, (r) => r.peakMb),
    settledMb: metric(c, (r) => r.settledMb),
    gpuProcessLoadPeakMb: metric(c, (r) => r.gpuPeakThroughSearchMb ?? r.gpuProcessMaxDeltaMb),
    uaResidentMb: metric(c, (r) => (typeof r.uaMemoryMb?.['resident-after-load'] === 'number' ? r.uaMemoryMb['resident-after-load'] : NaN)),
    loadMs: metric(c, (r) => r.report?.loadMs),
    midgameVisitsPerSec: metric(c, (r) => r.report?.midgamePerSec),
  }));
  writeFileSync(join(dir, 'summary.json'), JSON.stringify(summary, null, 1));
  const fmt = (s) => (s.n ? `${s.median} [${s.min}–${s.max}]` : '—');
  const lines = [
    '| browser | build | net | settings | cache | n | load peak MB, median [min–max] | search median MB | lifetime peak MB | GPU proc MB | UA resident MB | load ms | midgame visits/s | invalid |',
    '|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|',
    ...summary
      .map((s) =>
        [
          s.browser,
          s.build || '—',
          s.net,
          new URLSearchParams(s.params).toString() || 'default',
          s.cache,
          s.n,
          fmt(s.loadPeakMb),
          fmt(s.searchMedianMb),
          fmt(s.lifetimePeakMb),
          fmt(s.gpuProcessLoadPeakMb),
          fmt(s.uaResidentMb),
          fmt(s.loadMs),
          s.midgameVisitsPerSec.n ? `${s.midgameVisitsPerSec.median} [${s.midgameVisitsPerSec.min}–${s.midgameVisitsPerSec.max}]` : '—',
          s.invalid.length ? s.invalid.map((x) => `${x.sessionId}: ${x.problems.join('; ')}`).join('<br>') : '',
        ].join(' | '),
      )
      .map((l) => `| ${l} |`),
  ];
  writeFileSync(join(dir, 'summary.md'), `${lines.join('\n')}\n`);
  console.log(lines.join('\n'));
}
