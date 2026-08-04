#!/usr/bin/env node
import { mkdir, writeFile } from 'node:fs/promises';
import { dirname } from 'node:path';
import { parseScriptArgs } from './lib/cli.mjs';
import { runAgent } from './lib/process.mjs';
import { startViteServer, waitForHttp } from './lib/server.mjs';

const DEFAULT_HOST = '127.0.0.1';
const DEFAULT_PORT = 5179;
const DEFAULT_TIMEOUT_MS = 180_000;
const ENCODER_KERNELS = ['hand', 'tvm-packed-f16', 'mixed-tvm-ffn', 'mixed-tvm-ffn-outproj', 'mixed-tvm-ffn-smolgen-project'];

const USAGE = `Usage: node scripts/lc0_browser_hybrid_encoder_profile_matrix.mjs [options]\n\nRuns repeated browser hybrid encoder stage profiles over encoder-kernel variants and writes a JSON matrix artifact.\n\nOptions:\n  --out PATH            Matrix artifact path (default /tmp/lc0_hybrid_encoder_profile_matrix.json)\n  --host HOST           Vite host (default ${DEFAULT_HOST})\n  --port N              Vite port (default ${DEFAULT_PORT})\n  --base-url URL        Use an existing server instead of starting Vite\n  --encoder-kernels LIST\n                       Comma-separated encoder kernels: ${ENCODER_KERNELS.join(',')} (default hand)\n  --repeats N           Repeat each variant, alternating variants in repeat order (default 1)\n  --layers N            Encoder layers (default 10)\n  --profile-mode MODE   gpu-timestamp or sync-staged (default gpu-timestamp)\n  --profile-iters N     Profile iterations per cell (default 10)\n  --profile-warmup N    Profile warmup iterations per cell (default 2)\n  --input-backend MODE  Hybrid input backend: js, wgsl, or wasm (default js)\n  --timeout MS          Per-cell browser timeout (default ${DEFAULT_TIMEOUT_MS})\n  --agent-browser BIN   Browser automation binary\n  --dry-run             Print planned cells and URLs without running\n  -h, --help            Show this help\n`;

function parseList(raw, parse, name) {
  const values = String(raw ?? '')
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean)
    .map(parse);
  if (!values.length || values.some((value) => value === undefined || Number.isNaN(value))) throw new Error(`Invalid --${name}: ${raw}`);
  return values;
}

function parseArgs(argv) {
  const args = parseScriptArgs(argv, {
    options: {
      out: { type: 'string', default: '/tmp/lc0_hybrid_encoder_profile_matrix.json' },
      host: { type: 'string', default: DEFAULT_HOST },
      port: { type: 'string', default: String(DEFAULT_PORT) },
      'base-url': { type: 'string' },
      'encoder-kernels': { type: 'string', default: 'hand' },
      repeats: { type: 'string', default: '1' },
      layers: { type: 'string', default: '10' },
      'profile-mode': { type: 'string', default: 'gpu-timestamp' },
      'profile-iters': { type: 'string', default: '10' },
      'profile-warmup': { type: 'string', default: '2' },
      'input-backend': { type: 'string', default: 'js' },
      timeout: { type: 'string', default: String(DEFAULT_TIMEOUT_MS) },
      'agent-browser': { type: 'string', default: process.env.AGENT_BROWSER_BIN ?? 'agent-browser' },
      'dry-run': { type: 'boolean', default: false },
    },
    usage: USAGE,
  });
  args.port = Number(args.port);
  args.repeats = Number(args.repeats);
  args.layers = Number(args.layers);
  args.profileIters = Number(args.profileIters);
  args.profileWarmup = Number(args.profileWarmup);
  args.timeoutMs = Number(args.timeout);
  delete args.timeout;
  args.encoderKernels = parseList(args.encoderKernels, (value) => value, 'encoder-kernels');
  args.explicitBaseUrl = args.baseUrl !== undefined;
  if (!args.baseUrl) args.baseUrl = `http://${args.host}:${args.port}`;
  for (const kernel of args.encoderKernels) if (!ENCODER_KERNELS.includes(kernel)) throw new Error(`Invalid encoder kernel: ${kernel}`);
  if (!['gpu-timestamp', 'sync-staged'].includes(args.profileMode)) throw new Error(`Invalid --profile-mode: ${args.profileMode}`);
  if (!['js', 'wgsl', 'wasm'].includes(args.inputBackend)) throw new Error(`Invalid --input-backend: ${args.inputBackend}`);
  for (const [name, value] of [
    ['port', args.port],
    ['repeats', args.repeats],
    ['layers', args.layers],
    ['profile-iters', args.profileIters],
    ['timeout', args.timeoutMs],
  ]) {
    if (!Number.isFinite(value) || value <= 0) throw new Error(`Invalid --${name}: ${value}`);
  }
  if (!Number.isFinite(args.profileWarmup) || args.profileWarmup < 0) throw new Error(`Invalid --profile-warmup: ${args.profileWarmup}`);
  return args;
}

async function closeSession(args, session) {
  try {
    await runAgent(args, ['--session', session, 'close'], 5_000);
  } catch (error) {
    process.stderr.write(`[profile-matrix] warning: failed to close ${session}: ${error.message ?? error}\n`);
  }
}

function profileUrl(args, combo) {
  const url = new URL('/single-engine', args.baseUrl);
  url.searchParams.set('hybridEncoderProfile', '1');
  url.searchParams.set('runtime', 'hybrid');
  url.searchParams.set('encoderLayers', String(args.layers));
  url.searchParams.set('hybridEncoderProfileMode', args.profileMode);
  url.searchParams.set('hybridEncoderProfileIters', String(args.profileIters));
  url.searchParams.set('hybridEncoderProfileWarmup', String(args.profileWarmup));
  url.searchParams.set('inputBackend', args.inputBackend);
  url.searchParams.set('encoderKernel', combo.encoderKernel);
  url.searchParams.set('packVerify', '0');
  url.searchParams.set('ep', 'wasm');
  return String(url);
}

function compactProfile(result, combo) {
  const stages = Object.fromEntries((result.aggregateStageTimings ?? []).map((stage) => [stage.stage, stage.avgMs]));
  const smolgenSubstageAvgMs = ['smolgenCompress', 'smolgenDense1', 'smolgenLn1', 'smolgenDense2', 'smolgenLn2', 'smolgenProject'].reduce(
    (sum, stage) => sum + (stages[stage] ?? 0),
    0,
  );
  return {
    ...combo,
    encoderKernelVariant: result.encoderKernelVariant,
    requestedProfileMode: result.requestedProfileMode,
    profileMode: result.profileMode,
    gpuTimestampSupported: result.gpuTimestampSupported,
    profiledStageTotalMs: result.profiledStageTotalMs,
    readbackSyncedMs: result.readbackSyncedMs,
    smolgenAvgMs: stages.smolgen ?? smolgenSubstageAvgMs,
    smolgenCompressAvgMs: stages.smolgenCompress,
    smolgenDense1AvgMs: stages.smolgenDense1,
    smolgenLn1AvgMs: stages.smolgenLn1,
    smolgenDense2AvgMs: stages.smolgenDense2,
    smolgenLn2AvgMs: stages.smolgenLn2,
    smolgenProjectAvgMs: stages.smolgenProject,
    qkvProjectionAvgMs: stages.qkvProjection,
    outputProjectionAvgMs: stages.outputProjection,
    ffnDense1AvgMs: stages.ffnDense1,
    ffnDense2ResidualAvgMs: stages.ffnDense2Residual,
    outputSample: result.outputSample,
  };
}

async function runCell(args, combo, index, total) {
  const session = `lc0-hybrid-profile-matrix-${process.pid}-${index}`;
  const url = profileUrl(args, combo);
  process.stderr.write(`[profile-matrix] ${index}/${total} repeat=${combo.repeat} kernel=${combo.encoderKernel} mode=${args.profileMode}\n`);
  const started = Date.now();
  try {
    await runAgent(args, ['--session', session, 'open', url], 30_000);
    const deadline = Date.now() + args.timeoutMs;
    while (Date.now() < deadline) {
      const chunk = Math.min(25_000, Math.max(1000, deadline - Date.now()));
      try {
        await runAgent(args, ['--session', session, 'wait', '--text', 'HYBRID_ENCODER_PROFILE_DONE', '--timeout', String(chunk)], chunk + 5_000);
        const text = (await runAgent(args, ['--session', session, 'get', 'text', '#benchResult'], 30_000)).text;
        const result = JSON.parse(text);
        if (result.encoderKernelVariant !== combo.encoderKernel) throw new Error(`unexpected encoder kernel variant: ${result.encoderKernelVariant}`);
        return { combo, url, elapsedMs: Date.now() - started, result, summary: compactProfile(result, combo) };
      } catch (error) {
        if (Date.now() >= deadline) throw error;
      }
    }
    throw new Error(`Timed out waiting for HYBRID_ENCODER_PROFILE_DONE after ${args.timeoutMs}ms`);
  } finally {
    await closeSession(args, session);
  }
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const combos = [];
  for (let repeat = 1; repeat <= args.repeats; repeat++) {
    for (const encoderKernel of args.encoderKernels) combos.push({ repeat, encoderKernel });
  }
  if (args.dryRun) {
    console.log(JSON.stringify({ baseUrl: args.baseUrl, combos: combos.map((combo) => ({ ...combo, url: profileUrl(args, combo) })) }, null, 2));
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
      status: 'LC0_HYBRID_ENCODER_PROFILE_MATRIX_DONE',
      startedAt,
      finishedAt: new Date().toISOString(),
      baseUrl: args.baseUrl,
      layers: args.layers,
      encoderKernels: args.encoderKernels,
      repeats: args.repeats,
      profile: { mode: args.profileMode, warmup: args.profileWarmup, iterations: args.profileIters, inputBackend: args.inputBackend },
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
