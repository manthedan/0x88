#!/usr/bin/env node
import { mkdir, writeFile } from 'node:fs/promises';
import { dirname } from 'node:path';
import { parseScriptArgs } from './lib/cli.mjs';
import { runAgent } from './lib/process.mjs';
import { startViteServer, waitForHttp } from './lib/server.mjs';

const DEFAULT_HOST = '127.0.0.1';
const DEFAULT_PORT = 5179;
const DEFAULT_TIMEOUT_MS = 120_000;
const PROJECT_KERNELS = ['hand', 'tiled-project-f16-16', 'tiled-project-f16-32', 'tiled-project-f16', 'tiled-project-f16-128', 'tiled-project-f16-256'];

const USAGE = `Usage: node scripts/lc0_browser_smolgen_benchmark_matrix.mjs [options]\n\nRuns isolated browser smolgen parity/profiling cells and writes a JSON matrix artifact.\n\nOptions:\n  --out PATH              Matrix artifact path (default /tmp/lc0_smolgen_benchmark_matrix.json)\n  --host HOST             Vite host (default ${DEFAULT_HOST})\n  --port N                Vite port (default ${DEFAULT_PORT})\n  --base-url URL          Use an existing server instead of starting Vite\n  --project-kernels LIST  Comma-separated smolgen project kernels: ${PROJECT_KERNELS.join(',')}\n                          (default ${PROJECT_KERNELS.join(',')})\n  --repeats N             Repeat each cell, alternating kernels in repeat order (default 1)\n  --iters N               Smolgen passes per timed cell (default 50)\n  --warmup N              Warmup passes per cell (default 5)\n  --encoder-prefix NAME   Encoder prefix to benchmark (default /encoder0)\n  --timeout MS            Per-cell browser timeout (default ${DEFAULT_TIMEOUT_MS})\n  --agent-browser BIN     Browser automation binary\n  --dry-run               Print planned cells and URLs without running\n  -h, --help              Show this help\n`;

function parseList(raw, name) {
  const values = String(raw ?? '')
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean);
  if (!values.length) throw new Error(`Invalid --${name}: ${raw}`);
  return values;
}

function parseArgs(argv) {
  const args = parseScriptArgs(argv, {
    options: {
      out: { type: 'string', default: '/tmp/lc0_smolgen_benchmark_matrix.json' },
      host: { type: 'string', default: DEFAULT_HOST },
      port: { type: 'string', default: String(DEFAULT_PORT) },
      'project-kernels': { type: 'string', default: PROJECT_KERNELS.join(',') },
      repeats: { type: 'string', default: '1' },
      iters: { type: 'string', default: '50' },
      warmup: { type: 'string', default: '5' },
      'encoder-prefix': { type: 'string', default: '/encoder0' },
      timeout: { type: 'string', default: String(DEFAULT_TIMEOUT_MS) },
      'agent-browser': { type: 'string', default: process.env.AGENT_BROWSER_BIN ?? 'agent-browser' },
      'base-url': { type: 'string' },
      'dry-run': { type: 'boolean', default: false },
    },
    usage: USAGE,
  });
  args.port = Number(args.port);
  args.repeats = Number(args.repeats);
  args.iters = Number(args.iters);
  args.warmup = Number(args.warmup);
  args.timeoutMs = Number(args.timeout);
  delete args.timeout;
  args.projectKernels = parseList(args.projectKernels, 'project-kernels');
  args.explicitBaseUrl = args.baseUrl !== undefined;
  if (!args.baseUrl) args.baseUrl = `http://${args.host}:${args.port}`;
  for (const kernel of args.projectKernels) if (!PROJECT_KERNELS.includes(kernel)) throw new Error(`Invalid project kernel: ${kernel}`);
  for (const [name, value] of [
    ['port', args.port],
    ['repeats', args.repeats],
    ['iters', args.iters],
    ['timeout', args.timeoutMs],
  ]) {
    if (!Number.isFinite(value) || value <= 0) throw new Error(`Invalid --${name}: ${value}`);
  }
  if (!Number.isFinite(args.warmup) || args.warmup < 0) throw new Error(`Invalid --warmup: ${args.warmup}`);
  return args;
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
  const combos = [];
  for (let repeat = 1; repeat <= args.repeats; repeat++) {
    for (const projectKernel of args.projectKernels) combos.push({ repeat, projectKernel });
  }
  if (args.dryRun) {
    console.log(JSON.stringify({ baseUrl: args.baseUrl, combos: combos.map((combo) => ({ ...combo, url: cellUrl(args, combo) })) }, null, 2));
    return;
  }
  const server = startViteServer(args);
  const startedAt = new Date().toISOString();
  try {
    if (server) await server.ready;
    await waitForHttp(args.baseUrl);
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
