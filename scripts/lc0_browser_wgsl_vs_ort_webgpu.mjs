#!/usr/bin/env node
import { spawn, spawnSync } from 'node:child_process';
import { setTimeout as delay } from 'node:timers/promises';

const DEFAULT_PORT = 5179;
const DEFAULT_HOST = '127.0.0.1';
const DEFAULT_TIMEOUT_MS = 25_000;
const DEFAULT_SAMPLES = 3;
const DEFAULT_MAX_ERROR = 1e-3;

const VARIANTS = {
  wgsl: {
    label: 'wgsl-encoder0-block',
    doneText: 'ENCODER0_BLOCK_BENCH_DONE',
    query: (args) => `encoder0BlockBench=1&encoder0BlockWarmup=${args.wgslWarmup}&encoder0BlockIters=${args.wgslIters}&packVerify=${args.packVerify ? '1' : '0'}`,
  },
  ort: {
    label: 'ort-webgpu-encoder0-block',
    doneText: 'ENCODER0_BLOCK_ORT_BENCH_DONE',
    query: (args) => `encoder0BlockOrtBench=1&encoder0BlockOrtWarmup=${args.ortWarmup}&encoder0BlockOrtIters=${args.ortIters}&ep=webgpu&packVerify=${args.packVerify ? '1' : '0'}`,
  },
};

function usage() {
  console.log(`Usage: node --experimental-strip-types scripts/lc0_browser_wgsl_vs_ort_webgpu.mjs [options]\n\nAlternates fresh browser sessions between custom WGSL encoder0-block and ORT WebGPU encoder0-block benchmarks.\nThis script reports measurements only; it never promotes an implementation based on these numbers.\n\nOptions:\n  --base-url URL        Use an existing dev server, e.g. http://127.0.0.1:5179\n  --port N             Port for the auto-started Vite dev server (default ${DEFAULT_PORT})\n  --host HOST          Host for the auto-started Vite dev server (default ${DEFAULT_HOST})\n  --agent-browser BIN  Browser automation binary (default: AGENT_BROWSER_BIN or agent-browser)\n  --session NAME       Session prefix (default: lc0-wgsl-vs-ort-PID)\n  --timeout MS         Per-run wait timeout (default ${DEFAULT_TIMEOUT_MS})\n  --samples N          Alternating A/B pairs to run (default ${DEFAULT_SAMPLES})\n  --wgsl-iters N       Queued WGSL encoder0 block iterations per sample (default 1)\n  --wgsl-warmup N      WGSL warmup iterations per sample (default 1)\n  --ort-iters N        ORT timed iterations per sample (default 3)\n  --ort-warmup N       ORT warmup iterations per sample (default 1)\n  --max-error N        Max accepted maxAbsError (default ${DEFAULT_MAX_ERROR})\n  --allow-wasm-fallback\n                       Do not fail if ORT WebGPU falls back to WASM; result is marked not-promotable\n  --pack-verify        Enable shard sha256 verification (default skipped for benchmarking)\n  --no-server          Do not auto-start Vite\n  --dry-run            Print planned alternating runs and exit\n  -h, --help           Show this help\n`);
}

function intArg(value, label, min, max) {
  const parsed = Math.floor(Number(value));
  if (!Number.isFinite(parsed) || parsed < min || parsed > max) throw new Error(`Invalid ${label}: ${value}`);
  return parsed;
}

function parseArgs(argv) {
  const args = {
    host: DEFAULT_HOST,
    port: DEFAULT_PORT,
    timeoutMs: DEFAULT_TIMEOUT_MS,
    samples: DEFAULT_SAMPLES,
    wgslIters: 1,
    wgslWarmup: 1,
    ortIters: 3,
    ortWarmup: 1,
    maxError: DEFAULT_MAX_ERROR,
    agentBrowser: process.env.AGENT_BROWSER_BIN ?? 'agent-browser',
    session: process.env.AGENT_BROWSER_SESSION ?? `lc0-wgsl-vs-ort-${process.pid}`,
    allowWasmFallback: false,
    packVerify: false,
    noServer: false,
    dryRun: false,
  };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    const next = () => {
      if (i + 1 >= argv.length) throw new Error(`${arg} requires a value`);
      return argv[++i];
    };
    if (arg === '--base-url') args.baseUrl = next();
    else if (arg === '--port') args.port = intArg(next(), '--port', 1, 65535);
    else if (arg === '--host') args.host = next();
    else if (arg === '--agent-browser') args.agentBrowser = next();
    else if (arg === '--session') args.session = next();
    else if (arg === '--timeout') args.timeoutMs = intArg(next(), '--timeout', 1, 120_000);
    else if (arg === '--samples') args.samples = intArg(next(), '--samples', 1, 50);
    else if (arg === '--wgsl-iters') args.wgslIters = intArg(next(), '--wgsl-iters', 1, 10_000);
    else if (arg === '--wgsl-warmup') args.wgslWarmup = intArg(next(), '--wgsl-warmup', 0, 1000);
    else if (arg === '--ort-iters') args.ortIters = intArg(next(), '--ort-iters', 1, 1000);
    else if (arg === '--ort-warmup') args.ortWarmup = intArg(next(), '--ort-warmup', 0, 100);
    else if (arg === '--max-error') {
      args.maxError = Number(next());
      if (!Number.isFinite(args.maxError) || args.maxError < 0) throw new Error(`Invalid --max-error: ${args.maxError}`);
    } else if (arg === '--allow-wasm-fallback') args.allowWasmFallback = true;
    else if (arg === '--pack-verify') args.packVerify = true;
    else if (arg === '--no-server') args.noServer = true;
    else if (arg === '--dry-run') args.dryRun = true;
    else if (arg === '-h' || arg === '--help') args.help = true;
    else throw new Error(`Unknown option: ${arg}`);
  }
  return args;
}

function startVite(args) {
  const viteBin = process.platform === 'win32' ? 'node_modules/.bin/vite.cmd' : 'node_modules/.bin/vite';
  const child = spawn(viteBin, ['--host', args.host, '--port', String(args.port), '--strictPort'], {
    cwd: process.cwd(),
    stdio: ['ignore', 'pipe', 'pipe'],
    env: { ...process.env, FORCE_COLOR: '0' },
  });
  child.stdout.on('data', (chunk) => process.stderr.write(`[vite] ${chunk}`));
  child.stderr.on('data', (chunk) => process.stderr.write(`[vite] ${chunk}`));
  child.on('exit', (code, signal) => {
    if (code !== null && code !== 0) process.stderr.write(`[vite] exited with code ${code}\n`);
    else if (signal) process.stderr.write(`[vite] exited via signal ${signal}\n`);
  });
  return child;
}

async function waitForServer(baseUrl, timeoutMs) {
  const deadline = Date.now() + timeoutMs;
  let lastError;
  while (Date.now() < deadline) {
    try {
      const response = await fetch(`${baseUrl.replace(/\/$/, '')}/lc0-policy-only.html`, { cache: 'no-store' });
      if (response.ok) return;
      lastError = new Error(`HTTP ${response.status}`);
    } catch (error) {
      lastError = error;
    }
    await delay(250);
  }
  throw new Error(`Vite dev server did not become ready at ${baseUrl}: ${lastError?.message ?? 'timeout'}`);
}

function runJsonCommand(bin, args, timeoutMs, session) {
  const fullArgs = ['--json', '--session', session, ...args];
  const proc = spawnSync(bin, fullArgs, { encoding: 'utf8', timeout: timeoutMs, maxBuffer: 16 * 1024 * 1024 });
  if (proc.error) throw proc.error;
  if (proc.status !== 0) {
    throw new Error(`${bin} ${fullArgs.slice(1).join(' ')} failed with code ${proc.status}\nstdout:\n${proc.stdout}\nstderr:\n${proc.stderr}`);
  }
  const text = proc.stdout.trim();
  let parsed;
  try {
    parsed = JSON.parse(text);
  } catch {
    const start = text.lastIndexOf('{');
    if (start < 0) throw new Error(`Could not parse JSON from ${bin} output:\n${text}`);
    parsed = JSON.parse(text.slice(start));
  }
  if (parsed && typeof parsed === 'object' && 'success' in parsed) {
    if (parsed.success === false) throw new Error(`${bin} ${fullArgs.slice(1).join(' ')} failed: ${parsed.error ?? text}`);
    return parsed.data ?? parsed;
  }
  return parsed;
}

function textFromGetResult(result) {
  if (typeof result === 'string') return result;
  return typeof result?.text === 'string' ? result.text : typeof result?.data?.text === 'string' ? result.data.text : '';
}

function collectNumbers(value, out = []) {
  if (typeof value === 'number' && Number.isFinite(value)) out.push(value);
  else if (Array.isArray(value)) for (const item of value) collectNumbers(item, out);
  else if (value && typeof value === 'object') for (const item of Object.values(value)) collectNumbers(item, out);
  return out;
}

function parseBenchResult(rawText, variant) {
  const result = JSON.parse(rawText);
  if (result.status !== variant.doneText) throw new Error(`expected ${variant.doneText}, got ${result.status}`);
  const maxAbsError = Math.max(...collectNumbers(result.maxAbsError).map(Math.abs));
  if (!Number.isFinite(maxAbsError)) throw new Error(`result missing maxAbsError: ${rawText}`);
  return { ...result, maxAbsError };
}

function runPlan(args) {
  const plan = [];
  for (let pair = 0; pair < args.samples; pair++) {
    const order = pair % 2 === 0 ? ['wgsl', 'ort'] : ['ort', 'wgsl'];
    for (const kind of order) plan.push({ pair, kind, variant: VARIANTS[kind] });
  }
  return plan;
}

function runUrl(baseUrl, args, kind) {
  const variant = VARIANTS[kind];
  return `${baseUrl.replace(/\/$/, '')}/lc0-policy-only.html?${variant.query(args)}`;
}

function metricFor(kind, result) {
  if (kind === 'wgsl') return result.readbackSyncedMs / Math.max(1, result.iterations ?? 1);
  return result.avgMs;
}

function ortUsedWebGpu(result) {
  const providers = result.ortDiagnostics?.resolvedExecutionProviders;
  const attempts = result.ortDiagnostics?.sessionAttempts;
  return (Array.isArray(providers) && providers.includes('webgpu'))
    || (Array.isArray(attempts) && attempts.some((attempt) => attempt?.ok && Array.isArray(attempt.providers) && attempt.providers.includes('webgpu')));
}

function median(values) {
  const sorted = values.filter(Number.isFinite).sort((a, b) => a - b);
  if (!sorted.length) return null;
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

function summarize(rows) {
  const byKind = Object.fromEntries(['wgsl', 'ort'].map((kind) => {
    const items = rows.filter((row) => row.kind === kind);
    const metrics = items.map((row) => row.metricMs);
    return [kind, {
      samples: items.length,
      medianMs: median(metrics),
      minMs: metrics.length ? Math.min(...metrics) : null,
      maxMs: metrics.length ? Math.max(...metrics) : null,
      maxAbsError: items.length ? Math.max(...items.map((row) => row.maxAbsError)) : null,
    }];
  }));
  const wgslMedian = byKind.wgsl.medianMs;
  const ortMedian = byKind.ort.medianMs;
  return {
    ...byKind,
    ratioWgslOverOrt: wgslMedian && ortMedian ? wgslMedian / ortMedian : null,
    note: 'Measurement only. Do not promote custom WGSL or ORT artifacts from this script alone; compare against native/f32 correctness gates separately.',
  };
}

async function runOne(args, baseUrl, stepIndex, step) {
  const session = `${args.session}-${String(stepIndex).padStart(2, '0')}-${step.kind}-${Date.now()}`;
  const url = runUrl(baseUrl, args, step.kind);
  process.stderr.write(`[lc0-wgsl-vs-ort] ${stepIndex + 1}/${args.samples * 2} ${step.variant.label} fresh-session=${session}\n`);
  try {
    runJsonCommand(args.agentBrowser, ['open', url], args.timeoutMs, session);
    runJsonCommand(args.agentBrowser, ['wait', '--text', step.variant.doneText, '--timeout', String(args.timeoutMs)], args.timeoutMs + 5_000, session);
    const bench = runJsonCommand(args.agentBrowser, ['get', 'text', '#benchResult'], args.timeoutMs, session);
    const result = parseBenchResult(textFromGetResult(bench), step.variant);
    if (result.maxAbsError > args.maxError) throw new Error(`${step.variant.label}: maxAbsError ${result.maxAbsError} exceeded ${args.maxError}`);
    const usedWebGpu = step.kind === 'ort' ? ortUsedWebGpu(result) : true;
    if (step.kind === 'ort' && !usedWebGpu && !args.allowWasmFallback) {
      throw new Error('ORT run did not report a WebGPU execution provider; rerun with --allow-wasm-fallback only for non-promotable diagnostics');
    }
    return {
      pair: step.pair,
      kind: step.kind,
      label: step.variant.label,
      session,
      metricMs: metricFor(step.kind, result),
      metricName: step.kind === 'wgsl' ? 'readbackSyncedMs/iterations' : 'avgMs',
      iterations: result.iterations ?? null,
      maxAbsError: result.maxAbsError,
      status: result.status,
      ortUsedWebGpu: step.kind === 'ort' ? usedWebGpu : undefined,
      ortDiagnostics: step.kind === 'ort' ? result.ortDiagnostics : undefined,
      rawTiming: step.kind === 'wgsl'
        ? { readbackSyncedMs: result.readbackSyncedMs, dispatchLoopMs: result.dispatchLoopMs, dispatchLoopAvgMs: result.dispatchLoopAvgMs }
        : { avgMs: result.avgMs, minMs: result.minMs, maxMs: result.maxMs, timesMs: result.timesMs },
    };
  } finally {
    try { runJsonCommand(args.agentBrowser, ['close'], 5_000, session); } catch {}
  }
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) {
    usage();
    return;
  }
  const baseUrl = args.baseUrl ?? `http://${args.host}:${args.port}`;
  const plan = runPlan(args);
  if (args.dryRun) {
    console.log(JSON.stringify({ status: 'LC0_WGSL_VS_ORT_WEBGPU_DRY_RUN', baseUrl, plan: plan.map((step, i) => ({ i, pair: step.pair, kind: step.kind, url: runUrl(baseUrl, args, step.kind) })) }, null, 2));
    return;
  }
  let server;
  if (!args.noServer && !args.baseUrl) server = startVite(args);
  try {
    await waitForServer(baseUrl, args.timeoutMs);
    const rows = [];
    for (let i = 0; i < plan.length; i++) rows.push(await runOne(args, baseUrl, i, plan[i]));
    const summary = summarize(rows);
    const promotable = rows.every((row) => row.kind !== 'ort' || row.ortUsedWebGpu) && rows.length >= 4;
    console.log(JSON.stringify({
      status: 'LC0_WGSL_VS_ORT_WEBGPU_DONE',
      baseUrl,
      samplesPerSide: args.samples,
      freshSessionPerRun: true,
      alternatingOrder: plan.map((step) => step.kind),
      promotableEvidence: promotable ? 'webgpu-provider-present-but-still-measurement-only' : 'not-promotable; ORT WebGPU not confirmed or too few samples',
      rows,
      summary,
    }, null, 2));
  } finally {
    if (server) server.kill('SIGTERM');
  }
}

main().catch((error) => {
  console.error(error.stack ?? error.message);
  process.exitCode = 1;
});
