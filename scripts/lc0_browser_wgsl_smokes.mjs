#!/usr/bin/env node
import { spawn, spawnSync } from 'node:child_process';
import { setTimeout as delay } from 'node:timers/promises';

const DEFAULT_PORT = 5179;
const DEFAULT_HOST = '127.0.0.1';
const DEFAULT_TIMEOUT_MS = 25_000;
const DEFAULT_MAX_ERROR = 1e-3;

const SMOKES = [
  {
    name: 'softmax',
    query: 'softmaxBench=1&softmaxWarmup=1&softmaxIters=3&packVerify=0',
    doneText: 'SOFTMAX_BENCH_DONE',
  },
  {
    name: 'attention-value',
    query: 'attentionValueBench=1&attentionValueWarmup=1&attentionValueIters=3&packVerify=0',
    doneText: 'ATTENTION_VALUE_BENCH_DONE',
  },
  {
    name: 'attention-value-ort-wasm',
    query: 'attentionValueOrtBench=1&attentionValueOrtWarmup=0&attentionValueOrtIters=1&ep=wasm&packVerify=0',
    doneText: 'ATTENTION_VALUE_ORT_BENCH_DONE',
  },
  {
    name: 'attention-block',
    query: 'attentionBlockBench=1&attentionBlockWarmup=1&attentionBlockIters=1&packVerify=0',
    doneText: 'ATTENTION_BLOCK_BENCH_DONE',
  },
  {
    name: 'attention-output',
    query: 'attentionOutputBench=1&attentionOutputWarmup=1&attentionOutputIters=1&packVerify=0',
    doneText: 'ATTENTION_OUTPUT_BENCH_DONE',
  },
  {
    name: 'attention-output-ort-wasm',
    query: 'attentionOutputOrtBench=1&attentionOutputOrtWarmup=0&attentionOutputOrtIters=1&ep=wasm&packVerify=0',
    doneText: 'ATTENTION_OUTPUT_ORT_BENCH_DONE',
  },
  {
    name: 'encoder0-ffn',
    query: 'encoder0FfnBench=1&encoder0FfnWarmup=1&encoder0FfnIters=1&packVerify=0',
    doneText: 'FFN_BENCH_DONE',
  },
  {
    name: 'encoder0-ffn-ort-wasm',
    query: 'encoder0FfnOrtBench=1&encoder0FfnOrtWarmup=0&encoder0FfnOrtIters=1&ep=wasm&packVerify=0',
    doneText: 'FFN_ORT_BENCH_DONE',
  },
  {
    name: 'encoder0-block',
    query: 'encoder0BlockBench=1&encoder0BlockWarmup=1&encoder0BlockIters=1&packVerify=0',
    doneText: 'ENCODER0_BLOCK_BENCH_DONE',
  },
  {
    name: 'encoder0-block-ort-wasm',
    query: 'encoder0BlockOrtBench=1&encoder0BlockOrtWarmup=0&encoder0BlockOrtIters=1&ep=wasm&packVerify=0',
    doneText: 'ENCODER0_BLOCK_ORT_BENCH_DONE',
  },
  {
    name: 'kernel-bench-scalar',
    query: 'kernelBench=1&kernelBenchWarmup=1&kernelBenchIters=3&packVerify=0',
    doneText: 'KERNEL_BENCH_DONE',
  },
  {
    name: 'kernel-bench-tiled16',
    query: 'kernelBench=1&kernelVariant=tiled16&kernelBenchWarmup=1&kernelBenchIters=3&packVerify=0',
    doneText: 'KERNEL_BENCH_DONE',
  },
  {
    name: 'kernel-bench-scalar-transposed',
    query: 'kernelBench=1&kernelVariant=scalar-transposed&kernelBenchWarmup=1&kernelBenchIters=3&packVerify=0',
    doneText: 'KERNEL_BENCH_DONE',
  },
  {
    name: 'qkv-probe',
    query: 'qkvProbe=1&qkvWarmup=1&qkvIters=1&packVerify=0',
    doneText: 'QKV_DONE',
  },
  {
    name: 'qkv-bench',
    query: 'qkvBench=1&qkvBenchWarmup=1&qkvBenchIters=3&packVerify=0',
    doneText: 'QKV_BENCH_DONE',
  },
];

function usage() {
  console.log(`Usage: node --experimental-strip-types scripts/lc0_browser_wgsl_smokes.mjs [options]\n\nRuns lc0-policy-only.html WebGPU smoke benchmarks through agent-browser.\n\nOptions:\n  --base-url URL        Use an existing dev server, e.g. http://127.0.0.1:5179\n  --port N             Port for the auto-started Vite dev server (default ${DEFAULT_PORT})\n  --host HOST          Host for the auto-started Vite dev server (default ${DEFAULT_HOST})\n  --agent-browser BIN  Browser automation binary (default: AGENT_BROWSER_BIN or agent-browser)\n  --session NAME       agent-browser session name (default: lc0-wgsl-smokes-PID)\n  --timeout MS         Per-smoke wait timeout (default ${DEFAULT_TIMEOUT_MS})\n  --max-error N        Max accepted maxAbsError across outputs (default ${DEFAULT_MAX_ERROR})\n  --only a,b,c         Comma-separated smoke names to run\n  --list               Print smoke names and URLs, then exit\n  --no-server          Do not auto-start Vite; requires --base-url or an already-running default URL\n  -h, --help           Show this help\n`);
}

function parseArgs(argv) {
  const args = {
    host: DEFAULT_HOST,
    port: DEFAULT_PORT,
    timeoutMs: DEFAULT_TIMEOUT_MS,
    maxError: DEFAULT_MAX_ERROR,
    agentBrowser: process.env.AGENT_BROWSER_BIN ?? 'agent-browser',
    session: process.env.AGENT_BROWSER_SESSION ?? `lc0-wgsl-smokes-${process.pid}`,
    noServer: false,
    list: false,
  };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    const next = () => {
      if (i + 1 >= argv.length) throw new Error(`${arg} requires a value`);
      return argv[++i];
    };
    if (arg === '--base-url') args.baseUrl = next();
    else if (arg === '--port') args.port = Number(next());
    else if (arg === '--host') args.host = next();
    else if (arg === '--agent-browser') args.agentBrowser = next();
    else if (arg === '--session') args.session = next();
    else if (arg === '--timeout') args.timeoutMs = Number(next());
    else if (arg === '--max-error') args.maxError = Number(next());
    else if (arg === '--only') args.only = new Set(next().split(',').map((s) => s.trim()).filter(Boolean));
    else if (arg === '--list') args.list = true;
    else if (arg === '--no-server') args.noServer = true;
    else if (arg === '-h' || arg === '--help') args.help = true;
    else throw new Error(`Unknown option: ${arg}`);
  }
  if (!Number.isFinite(args.port) || args.port <= 0) throw new Error(`Invalid --port: ${args.port}`);
  if (!Number.isFinite(args.timeoutMs) || args.timeoutMs <= 0) throw new Error(`Invalid --timeout: ${args.timeoutMs}`);
  if (!Number.isFinite(args.maxError) || args.maxError < 0) throw new Error(`Invalid --max-error: ${args.maxError}`);
  return args;
}

function selectedSmokes(args) {
  const selected = args.only ? SMOKES.filter((smoke) => args.only.has(smoke.name)) : SMOKES;
  if (args.only && selected.length !== args.only.size) {
    const known = new Set(SMOKES.map((smoke) => smoke.name));
    const missing = [...args.only].filter((name) => !known.has(name));
    throw new Error(`Unknown smoke name(s): ${missing.join(', ')}. Known: ${SMOKES.map((s) => s.name).join(', ')}`);
  }
  return selected;
}

function smokeUrl(baseUrl, smoke) {
  return `${baseUrl.replace(/\/$/, '')}/lc0-policy-only.html?${smoke.query}`;
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

function runJsonCommand(bin, args, timeoutMs, session) {
  const sessionArgs = session ? ['--session', session] : [];
  const fullArgs = ['--json', ...sessionArgs, ...args];
  const proc = spawnSync(bin, fullArgs, {
    encoding: 'utf8',
    timeout: timeoutMs,
    maxBuffer: 16 * 1024 * 1024,
  });
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
    if (start < 0) throw new Error(`Could not parse JSON from ${bin} ${fullArgs.slice(1).join(' ')} output:\n${text}`);
    parsed = JSON.parse(text.slice(start));
  }
  if (parsed && typeof parsed === 'object' && 'success' in parsed) {
    if (parsed.success === false) throw new Error(`${bin} ${fullArgs.slice(1).join(' ')} failed: ${parsed.error ?? text}`);
    return parsed.data ?? parsed;
  }
  return parsed;
}

function collectNumbers(value, out = []) {
  if (typeof value === 'number' && Number.isFinite(value)) out.push(value);
  else if (Array.isArray(value)) for (const item of value) collectNumbers(item, out);
  else if (value && typeof value === 'object') for (const item of Object.values(value)) collectNumbers(item, out);
  return out;
}

function maxAbsError(result) {
  const numbers = collectNumbers(result.maxAbsError);
  if (!numbers.length) throw new Error(`Result did not include numeric maxAbsError: ${JSON.stringify(result.maxAbsError)}`);
  return Math.max(...numbers.map(Math.abs));
}

function textFromGetResult(result) {
  if (typeof result === 'string') return result;
  return typeof result?.text === 'string' ? result.text : typeof result?.data?.text === 'string' ? result.data.text : '';
}

function parseBenchResult(rawText, smoke) {
  let result;
  try {
    result = JSON.parse(rawText);
  } catch (error) {
    throw new Error(`${smoke.name}: #benchResult was not JSON after ${smoke.doneText}: ${rawText}`);
  }
  if (result.status !== smoke.doneText) {
    throw new Error(`${smoke.name}: expected result.status=${smoke.doneText}, got ${result.status}`);
  }
  return result;
}

async function runSmoke(args, baseUrl, smoke) {
  const url = smokeUrl(baseUrl, smoke);
  runJsonCommand(args.agentBrowser, ['open', url], args.timeoutMs, args.session);
  runJsonCommand(args.agentBrowser, ['wait', '--text', smoke.doneText, '--timeout', String(args.timeoutMs)], args.timeoutMs + 5_000, args.session);
  const bench = runJsonCommand(args.agentBrowser, ['get', 'text', '#benchResult'], args.timeoutMs, args.session);
  const result = parseBenchResult(textFromGetResult(bench), smoke);
  const error = maxAbsError(result);
  if (error > args.maxError) {
    throw new Error(`${smoke.name}: maxAbsError ${error} exceeded threshold ${args.maxError}`);
  }
  const slowestStage = Array.isArray(result.stageTimings) && result.stageTimings.length
    ? result.stageTimings.reduce((best, timing) => timing.avgMs > best.avgMs ? timing : best, result.stageTimings[0])
    : undefined;
  return {
    smoke: smoke.name,
    status: result.status,
    maxAbsError: error,
    iterations: result.iterations ?? null,
    readbackSyncedMs: result.readbackSyncedMs ?? null,
    gpuTimestampSupported: result.gpuTimestampSupported ?? null,
    gpuTimestampMs: result.gpuTimestampMs ?? null,
    stageTimingTotalMs: result.stageTimingTotalMs ?? null,
    slowestStage: slowestStage ? { stage: slowestStage.stage, label: slowestStage.label, avgMs: slowestStage.avgMs } : null,
    stageTimings: result.stageTimings ?? null,
  };
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) {
    usage();
    return;
  }
  const baseUrl = args.baseUrl ?? `http://${args.host}:${args.port}`;
  const smokes = selectedSmokes(args);
  if (args.list) {
    for (const smoke of smokes) console.log(`${smoke.name}\t${smoke.doneText}\t${smokeUrl(baseUrl, smoke)}`);
    return;
  }

  let server;
  if (!args.noServer && !args.baseUrl) server = startVite(args);
  try {
    await waitForServer(baseUrl, args.timeoutMs);
    const rows = [];
    for (const smoke of smokes) {
      process.stderr.write(`[lc0-browser-smoke] ${smoke.name}: ${smokeUrl(baseUrl, smoke)}\n`);
      rows.push(await runSmoke(args, baseUrl, smoke));
    }
    console.log(JSON.stringify({ status: 'LC0_BROWSER_WGSL_SMOKES_DONE', baseUrl, maxError: args.maxError, smokes: rows }, null, 2));
  } finally {
    try { runJsonCommand(args.agentBrowser, ['close'], 5_000, args.session); } catch {}
    if (server) server.kill('SIGTERM');
  }
}

main().catch((error) => {
  console.error(error.stack ?? error.message);
  process.exitCode = 1;
});
