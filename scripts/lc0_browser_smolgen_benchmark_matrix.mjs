#!/usr/bin/env node
import { spawn } from 'node:child_process';
import { mkdir, writeFile } from 'node:fs/promises';
import { dirname } from 'node:path';
import { setTimeout as delay } from 'node:timers/promises';

const DEFAULT_HOST = '127.0.0.1';
const DEFAULT_PORT = 5179;
const DEFAULT_TIMEOUT_MS = 120_000;
const PROJECT_KERNELS = ['hand', 'tiled-project-f16-16', 'tiled-project-f16-32', 'tiled-project-f16', 'tiled-project-f16-128', 'tiled-project-f16-256'];

function usage() {
  console.log(`Usage: node scripts/lc0_browser_smolgen_benchmark_matrix.mjs [options]\n\nRuns isolated browser smolgen parity/profiling cells and writes a JSON matrix artifact.\n\nOptions:\n  --out PATH              Matrix artifact path (default /tmp/lc0_smolgen_benchmark_matrix.json)\n  --host HOST             Vite host (default ${DEFAULT_HOST})\n  --port N                Vite port (default ${DEFAULT_PORT})\n  --base-url URL          Use an existing server instead of starting Vite\n  --project-kernels LIST  Comma-separated smolgen project kernels: ${PROJECT_KERNELS.join(',')}\n                          (default ${PROJECT_KERNELS.join(',')})\n  --repeats N             Repeat each cell, alternating kernels in repeat order (default 1)\n  --iters N               Smolgen passes per timed cell (default 50)\n  --warmup N              Warmup passes per cell (default 5)\n  --encoder-prefix NAME   Encoder prefix to benchmark (default /encoder0)\n  --timeout MS            Per-cell browser timeout (default ${DEFAULT_TIMEOUT_MS})\n  --agent-browser BIN     Browser automation binary\n  --dry-run               Print planned cells and URLs without running\n  -h, --help              Show this help\n`);
}

function parseList(raw, name) {
  const values = String(raw ?? '').split(',').map((s) => s.trim()).filter(Boolean);
  if (!values.length) throw new Error(`Invalid --${name}: ${raw}`);
  return values;
}

function parseArgs(argv) {
  const args = {
    out: '/tmp/lc0_smolgen_benchmark_matrix.json',
    host: DEFAULT_HOST,
    port: DEFAULT_PORT,
    projectKernels: PROJECT_KERNELS,
    repeats: 1,
    iters: 50,
    warmup: 5,
    encoderPrefix: '/encoder0',
    timeoutMs: DEFAULT_TIMEOUT_MS,
    agentBrowser: process.env.AGENT_BROWSER_BIN ?? 'agent-browser',
    explicitBaseUrl: false,
    dryRun: false,
  };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    const next = () => {
      if (i + 1 >= argv.length) throw new Error(`${arg} requires a value`);
      return argv[++i];
    };
    if (arg === '--out') args.out = next();
    else if (arg === '--host') args.host = next();
    else if (arg === '--port') args.port = Number(next());
    else if (arg === '--base-url') {
      args.baseUrl = next();
      args.explicitBaseUrl = true;
    } else if (arg === '--project-kernels') args.projectKernels = parseList(next(), 'project-kernels');
    else if (arg === '--repeats') args.repeats = Number(next());
    else if (arg === '--iters') args.iters = Number(next());
    else if (arg === '--warmup') args.warmup = Number(next());
    else if (arg === '--encoder-prefix') args.encoderPrefix = next();
    else if (arg === '--timeout') args.timeoutMs = Number(next());
    else if (arg === '--agent-browser') args.agentBrowser = next();
    else if (arg === '--dry-run') args.dryRun = true;
    else if (arg === '-h' || arg === '--help') args.help = true;
    else throw new Error(`Unknown option: ${arg}`);
  }
  if (!args.baseUrl) args.baseUrl = `http://${args.host}:${args.port}`;
  for (const kernel of args.projectKernels) if (!PROJECT_KERNELS.includes(kernel)) throw new Error(`Invalid project kernel: ${kernel}`);
  for (const [name, value] of [['port', args.port], ['repeats', args.repeats], ['iters', args.iters], ['timeout', args.timeoutMs]]) {
    if (!Number.isFinite(value) || value <= 0) throw new Error(`Invalid --${name}: ${value}`);
  }
  if (!Number.isFinite(args.warmup) || args.warmup < 0) throw new Error(`Invalid --warmup: ${args.warmup}`);
  return args;
}

function startServer(args) {
  if (args.explicitBaseUrl) return null;
  const server = spawn('npm', ['run', 'web:client', '--', '--host', args.host, '--port', String(args.port)], { stdio: ['ignore', 'pipe', 'pipe'] });
  server.stdout.on('data', (chunk) => process.stderr.write(`[vite] ${chunk}`));
  server.stderr.on('data', (chunk) => process.stderr.write(`[vite] ${chunk}`));
  return server;
}

async function waitForServer(baseUrl, timeoutMs = 30_000) {
  const deadline = Date.now() + timeoutMs;
  let lastError;
  while (Date.now() < deadline) {
    try {
      const response = await fetch(new URL('/single-engine', baseUrl), { cache: 'no-store' });
      if (response.ok) return;
      lastError = new Error(`HTTP ${response.status}`);
    } catch (error) {
      lastError = error;
    }
    await delay(250);
  }
  throw new Error(`Vite dev server did not become ready at ${baseUrl}: ${lastError?.message ?? 'timeout'}`);
}

function runAgent(args, commandArgs, timeoutMs = 30_000) {
  const fullArgs = ['--json', ...commandArgs];
  return new Promise((resolve, reject) => {
    const child = spawn(args.agentBrowser, fullArgs, { stdio: ['ignore', 'pipe', 'pipe'] });
    const chunks = { stdout: [], stderr: [] };
    let settled = false;
    const finish = (fn, value) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      fn(value);
    };
    const timer = setTimeout(() => {
      child.kill('SIGKILL');
      finish(reject, new Error(`${args.agentBrowser} ${fullArgs.slice(1).join(' ')} timed out after ${timeoutMs}ms`));
    }, timeoutMs);
    child.stdout.on('data', (chunk) => chunks.stdout.push(chunk));
    child.stderr.on('data', (chunk) => chunks.stderr.push(chunk));
    child.on('error', (error) => finish(reject, error));
    child.on('close', (status) => {
      const stdout = Buffer.concat(chunks.stdout).toString('utf8');
      const stderr = Buffer.concat(chunks.stderr).toString('utf8');
      if (status !== 0) return finish(reject, new Error(`${args.agentBrowser} ${fullArgs.slice(1).join(' ')} failed: ${stderr || stdout}`));
      try {
        const parsed = stdout ? JSON.parse(stdout.trim()) : null;
        if (parsed && typeof parsed === 'object' && 'success' in parsed) {
          if (parsed.success === false) return finish(reject, new Error(`${args.agentBrowser} ${fullArgs.slice(1).join(' ')} failed: ${parsed.error ?? stdout}`));
          return finish(resolve, parsed.data ?? parsed);
        }
        return finish(resolve, parsed);
      } catch (error) {
        return finish(reject, error);
      }
    });
  });
}

async function closeSession(args, session) {
  try {
    await runAgent(args, ['--session', session, 'close'], 5_000);
  } catch (error) {
    process.stderr.write(`[smolgen-matrix] warning: failed to close ${session}: ${error.message ?? error}\n`);
  }
}

function cellUrl(args, combo) {
  const url = new URL('/single-engine', args.baseUrl);
  url.searchParams.set('smolgenBench', '1');
  url.searchParams.set('smolgenIters', String(args.iters));
  url.searchParams.set('smolgenWarmup', String(args.warmup));
  url.searchParams.set('smolgenProjectKernel', combo.projectKernel);
  url.searchParams.set('encoderPrefix', args.encoderPrefix);
  url.searchParams.set('packVerify', '0');
  return String(url);
}

function compactResult(result, combo) {
  return {
    ...combo,
    projectKernelVariant: result.projectKernelVariant,
    dispatchLoopAvgMs: result.dispatchLoopAvgMs,
    syncedMsPerPass: result.readbackSyncedMs / Math.max(1, result.iterations),
    endToEndMs: result.endToEndMs,
    maxAbsError: result.maxAbsError,
    rmsError: result.rmsError,
    smolgenCompressAvgMs: result.stageDispatchAvgMs?.compress,
    smolgenDense1AvgMs: result.stageDispatchAvgMs?.dense1,
    smolgenLn1AvgMs: result.stageDispatchAvgMs?.ln1,
    smolgenDense2AvgMs: result.stageDispatchAvgMs?.dense2,
    smolgenLn2AvgMs: result.stageDispatchAvgMs?.ln2,
    smolgenProjectAvgMs: result.stageDispatchAvgMs?.project,
  };
}

async function runCell(args, combo, index, total) {
  const session = `lc0-smolgen-matrix-${process.pid}-${index}`;
  const url = cellUrl(args, combo);
  process.stderr.write(`[smolgen-matrix] ${index}/${total} repeat=${combo.repeat} project=${combo.projectKernel}\n`);
  const started = Date.now();
  try {
    await runAgent(args, ['--session', session, 'open', url], 30_000);
    const deadline = Date.now() + args.timeoutMs;
    while (Date.now() < deadline) {
      const chunk = Math.min(25_000, Math.max(1000, deadline - Date.now()));
      try {
        await runAgent(args, ['--session', session, 'wait', '--text', 'SMOLGEN_BENCH_DONE', '--timeout', String(chunk)], chunk + 5_000);
        const text = (await runAgent(args, ['--session', session, 'get', 'text', '#benchResult'], 30_000)).text;
        const result = JSON.parse(text);
        if (result.projectKernelVariant !== combo.projectKernel) throw new Error(`unexpected project kernel: ${result.projectKernelVariant}`);
        return { combo, url, elapsedMs: Date.now() - started, result, summary: compactResult(result, combo) };
      } catch (error) {
        if (Date.now() >= deadline) throw error;
      }
    }
    throw new Error(`Timed out waiting for SMOLGEN_BENCH_DONE after ${args.timeoutMs}ms`);
  } finally {
    await closeSession(args, session);
  }
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) return usage();
  const combos = [];
  for (let repeat = 1; repeat <= args.repeats; repeat++) {
    for (const projectKernel of args.projectKernels) combos.push({ repeat, projectKernel });
  }
  if (args.dryRun) {
    console.log(JSON.stringify({ baseUrl: args.baseUrl, combos: combos.map((combo) => ({ ...combo, url: cellUrl(args, combo) })) }, null, 2));
    return;
  }
  const server = startServer(args);
  const startedAt = new Date().toISOString();
  try {
    await waitForServer(args.baseUrl);
    const cells = [];
    for (let i = 0; i < combos.length; i++) cells.push(await runCell(args, combos[i], i + 1, combos.length));
    const artifact = {
      status: 'LC0_SMOLGEN_BENCHMARK_MATRIX_DONE',
      startedAt,
      finishedAt: new Date().toISOString(),
      baseUrl: args.baseUrl,
      encoderPrefix: args.encoderPrefix,
      projectKernels: args.projectKernels,
      repeats: args.repeats,
      benchmark: { warmup: args.warmup, iterations: args.iters },
      cells,
      summary: cells.map((cell) => cell.summary),
    };
    await mkdir(dirname(args.out), { recursive: true });
    await writeFile(args.out, JSON.stringify(artifact, null, 2));
    console.log(JSON.stringify({ status: artifact.status, out: args.out, cells: cells.length, summary: artifact.summary }, null, 2));
  } finally {
    server?.kill('SIGTERM');
  }
}

main().catch((error) => {
  console.error(error.stack ?? error.message);
  process.exit(1);
});
