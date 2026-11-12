#!/usr/bin/env node
// Kernel hill-climb bench runner (local agent-browser). Opens one bench page
// on an already-running COOP/COEP static server, waits for
// window.__kernelBenchResult / __kernelBenchError, and writes one JSON result.
//
// Wrap timing runs on the shared Mac in benchlock:
//   notes/kernel-hillclimb/benchlock.sh <label> -- node scripts/kernel_hillclimb/bench.mjs --candidate ort-lc0-small --gate ...
//
// Usage:
//   node scripts/kernel_hillclimb/bench.mjs --candidate NAME [--gate] [--repeat N] [--query k=v&...]
//   node scripts/kernel_hillclimb/bench.mjs --page peak --query 'tests=fma'
// Options:
//   --base-url URL     static server (default http://127.0.0.1:5241)
//   --session NAME     agent-browser session (default $AGENT_BROWSER_SESSION or kernel-hillclimb)
//   --browser-args CSV Chrome launch args (e.g. --enable-unsafe-webgpu,--enable-webgpu-developer-features)
//   --gate             append the candidate's in-page gate query (runs on any device)
//   --command-gate     also run the candidate's local command gate (repo smoke; starts its
//                      own vite dev server, so only on a full checkout, never on remote devices)
//   --repeat N         open the page N times (fresh navigation each), aggregate medians
//   --timeout MS       per-run timeout (default 900000)
//   --out PATH         result JSON (default artifacts/kernel-hillclimb/results/<name>-<stamp>.json)
import { execFileSync } from 'node:child_process';
import { mkdirSync, writeFileSync } from 'node:fs';
import { loadavg, type as osType, release } from 'node:os';
import { dirname, resolve } from 'node:path';
import { setTimeout as delay } from 'node:timers/promises';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import { median } from '../lib/perf_canary_stats.mjs';
import { spawnCapture } from '../lib/process.mjs';
import { CANDIDATES } from './candidates.mjs';

const repo = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const { values: opt } = parseArgs({
  options: {
    candidate: { type: 'string' },
    page: { type: 'string' },
    query: { type: 'string', default: '' },
    'base-url': { type: 'string', default: 'http://127.0.0.1:5241' },
    session: { type: 'string', default: process.env.AGENT_BROWSER_SESSION ?? 'kernel-hillclimb' },
    'browser-args': { type: 'string', default: '' },
    gate: { type: 'boolean', default: false },
    'command-gate': { type: 'boolean', default: false },
    repeat: { type: 'string', default: '1' },
    timeout: { type: 'string', default: '900000' },
    out: { type: 'string' },
    label: { type: 'string' },
  },
});

const candidate = opt.candidate ? CANDIDATES[opt.candidate] : null;
if (opt.candidate && !candidate) throw new Error(`unknown candidate ${opt.candidate}; known: ${Object.keys(CANDIDATES).join(', ')}`);
const page = opt.page ?? candidate?.page;
if (!page) throw new Error('--candidate or --page required');
const name = opt.label ?? opt.candidate ?? page;
const queryParts = [candidate?.query, opt.gate && candidate?.gate?.kind === 'in-page' ? candidate.gate.query : '', opt.query].filter(Boolean);
const url = new URL(`/lab/kernel-hillclimb/${page}.html`, opt['base-url']);
for (const part of queryParts) for (const [k, v] of new URLSearchParams(part)) url.searchParams.set(k, v);
const repeat = Math.max(1, Number(opt.repeat) || 1);
const timeoutMs = Number(opt.timeout);

async function agent(args, { input, timeout = 60_000 } = {}) {
  const launch = opt['browser-args'] ? ['--args', opt['browser-args']] : [];
  const out = await spawnCapture('agent-browser', ['--json', '--session', opt.session, ...launch, ...args], { timeoutMs: timeout, input });
  const parsed = out ? JSON.parse(out) : null;
  if (parsed && parsed.success === false) throw new Error(`agent-browser ${args[0]} failed: ${parsed.error}`);
  return parsed?.data ?? parsed;
}

async function runOnce() {
  const load = { before: loadavg() };
  await agent(['open', String(url)], { timeout: 60_000 });
  const deadline = Date.now() + timeoutMs;
  let last;
  while (Date.now() < deadline) {
    const data = await agent(['eval', '--stdin'], {
      input: `(() => JSON.stringify({ r: window.__kernelBenchResult ?? null, e: window.__kernelBenchError ?? null, tail: (document.getElementById('log')?.textContent ?? '').slice(-400) }))()`,
    });
    const value = typeof data?.result === 'string' ? data.result : typeof data === 'string' ? data : (data?.result ?? data?.value);
    last = typeof value === 'string' ? JSON.parse(value) : value;
    if (last?.e) throw new Error(`bench page error: ${last.e}`);
    if (last?.r) {
      load.after = loadavg();
      return { result: last.r, load };
    }
    await delay(2000);
  }
  throw new Error(`timed out after ${timeoutMs} ms; log tail: ${last?.tail}`);
}

function runCommandGate(gate) {
  if (!gate?.command) return null;
  try {
    const out = execFileSync(gate.command[0], gate.command.slice(1), { cwd: repo, encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'], timeout: 1_800_000 });
    return { command: gate.command.join(' '), pass: true, tail: out.slice(-2000) };
  } catch (error) {
    return { command: gate.command.join(' '), pass: false, tail: String(error.stdout ?? '').slice(-1500) + String(error.stderr ?? '').slice(-1500) };
  }
}

const runs = [];
for (let i = 0; i < repeat; i++) {
  process.stderr.write(`[kernel-bench] ${name} run ${i + 1}/${repeat}: ${url}\n`);
  runs.push(await runOnce());
}
const results = runs.map((r) => r.result);
const medians = results.map((r) => r.timing?.runMs?.median).filter(Number.isFinite);
const commandGate = opt['command-gate'] ? runCommandGate(candidate?.gate) : null;
let uptime = '';
try {
  uptime = execFileSync('uptime', { encoding: 'utf8' }).trim();
} catch {}
const report = {
  schema: 'kernel-hillclimb.bench-run.v1',
  name,
  candidate: opt.candidate ?? null,
  url: String(url),
  browserArgs: opt['browser-args'] || null,
  host: { os: `${osType()} ${release()}`, uptime },
  gate: candidate?.gate
    ? { ...candidate.gate, inPagePass: opt.gate && candidate.gate.kind === 'in-page' ? results.every((r) => r.ok) : null, commandGate }
    : null,
  ok: results.every((r) => r.ok) && (commandGate ? commandGate.pass : true),
  repeat,
  runMedianMsAcrossRepeats: medians.length ? { median: median(medians), samples: medians } : null,
  loads: runs.map((r) => r.load),
  results,
};
const out = resolve(repo, opt.out ?? `artifacts/kernel-hillclimb/results/${name}-${new Date().toISOString().replace(/[:.]/g, '-')}.json`);
mkdirSync(dirname(out), { recursive: true });
writeFileSync(out, `${JSON.stringify(report, null, 2)}\n`);
console.log(JSON.stringify({ status: 'KERNEL_BENCH_DONE', name, ok: report.ok, out, runMedianMs: report.runMedianMsAcrossRepeats?.median ?? null }));
if (!report.ok) process.exitCode = 1;
