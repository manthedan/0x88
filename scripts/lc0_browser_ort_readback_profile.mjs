#!/usr/bin/env node
import { spawn } from 'node:child_process';
import { setTimeout as delay } from 'node:timers/promises';
import { parseScriptArgs } from './lib/cli.mjs';
import { waitForOutput } from './lib/server.mjs';

const DEFAULT_HOST = '127.0.0.1';
const DEFAULT_PORT = 5179;
const DEFAULT_TIMEOUT_MS = 180_000;

const USAGE = `Usage: node scripts/lc0_browser_ort_readback_profile.mjs [options]\n\nRuns the LC0 ONNX evaluator in browser ORT with diagnostic output enabled.\n\nOptions:\n  --base-url URL        Use an existing dev server\n  --port N             Vite port when auto-starting (default ${DEFAULT_PORT})\n  --host HOST          Vite host when auto-starting (default ${DEFAULT_HOST})\n  --agent-browser BIN  Browser automation binary (default: AGENT_BROWSER_BIN or agent-browser)\n  --session NAME       agent-browser session name\n  --timeout MS         Total browser wait timeout (default ${DEFAULT_TIMEOUT_MS})\n  --model URL          LC0 ONNX model URL\n  --fen FEN            Position to benchmark (default page start position)\n  --iters N            Timed eval iterations (default 10)\n  --warmup N           Warmup eval iterations (default 2)\n  --ep EP              ORT EP: wasm, webgpu, webgpu,wasm, auto (default webgpu)\n  --ort-wasm-variant V ORT WASM artifact: fixed or relaxed\n  --ort-threads N      Pin ORT WASM threads\n  --no-monkey-patch    Disable WebGPU API monkey-patch counts\n  --no-kernel-profile  Disable ORT WebGPU kernel timestamp profiling\n  --no-gpu-outputs     Do not set preferredOutputLocation=gpu-buffer\n  --pack-verify        Kept for symmetry; ignored by ONNX bench\n  --no-server          Do not auto-start Vite\n  --dry-run            Print URL and exit\n  -h, --help           Show this help\n`;

function parseArgs(argv) {
  const args = parseScriptArgs(argv, {
    options: {
      host: { type: 'string', default: DEFAULT_HOST },
      port: { type: 'string', default: String(DEFAULT_PORT) },
      timeout: { type: 'string', default: String(DEFAULT_TIMEOUT_MS) },
      'agent-browser': { type: 'string', default: process.env.AGENT_BROWSER_BIN ?? 'agent-browser' },
      session: { type: 'string', default: process.env.AGENT_BROWSER_SESSION ?? `lc0-ort-readback-profile-${process.pid}` },
      iters: { type: 'string', default: '10' },
      warmup: { type: 'string', default: '2' },
      ep: { type: 'string', default: 'webgpu' },
      'no-monkey-patch': { type: 'boolean', default: false },
      'no-kernel-profile': { type: 'boolean', default: false },
      'no-gpu-outputs': { type: 'boolean', default: false },
      'pack-verify': { type: 'boolean', default: false },
      'base-url': { type: 'string' },
      model: { type: 'string' },
      fen: { type: 'string' },
      'ort-wasm-variant': { type: 'string' },
      'ort-threads': { type: 'string' },
      'no-server': { type: 'boolean', default: false },
      'dry-run': { type: 'boolean', default: false },
    },
    usage: USAGE,
  });
  args.port = Number(args.port);
  args.iters = Number(args.iters);
  args.warmup = Number(args.warmup);
  if (args.ortThreads !== undefined) args.ortThreads = Number(args.ortThreads);
  args.timeoutMs = Number(args.timeout);
  delete args.timeout;
  args.monkeyPatch = !args.noMonkeyPatch;
  delete args.noMonkeyPatch;
  args.kernelProfile = !args.noKernelProfile;
  delete args.noKernelProfile;
  args.gpuOutputs = !args.noGpuOutputs;
  delete args.noGpuOutputs;
  delete args.packVerify;
  args.explicitBaseUrl = args.baseUrl !== undefined;
  if (!args.baseUrl) args.baseUrl = `http://${args.host}:${args.port}`;
  if (args.explicitBaseUrl) args.noServer = true;
  for (const [name, value] of [
    ['port', args.port],
    ['timeout', args.timeoutMs],
    ['iters', args.iters],
    ['warmup', args.warmup],
  ]) {
    if (!Number.isFinite(value) || value < 0 || (name !== 'warmup' && value <= 0)) throw new Error(`Invalid --${name}: ${value}`);
  }
  if (args.ortWasmVariant && !['fixed', 'relaxed'].includes(args.ortWasmVariant)) throw new Error(`Invalid --ort-wasm-variant: ${args.ortWasmVariant}`);
  if (args.ortThreads !== undefined && (!Number.isInteger(args.ortThreads) || args.ortThreads < 1))
    throw new Error(`Invalid --ort-threads: ${args.ortThreads}`);
  return args;
}

function benchmarkUrl(args) {
  const url = new URL('/single-engine', args.baseUrl);
  url.searchParams.set('bench', '1');
  url.searchParams.set('ep', args.ep);
  url.searchParams.set('benchIters', String(args.iters));
  url.searchParams.set('benchWarmup', String(args.warmup));
  if (args.ortWasmVariant) url.searchParams.set('ortWasmVariant', args.ortWasmVariant);
  if (args.ortThreads !== undefined) url.searchParams.set('ortThreads', String(args.ortThreads));
  url.searchParams.set('ortReadbackProfile', '1');
  url.searchParams.set('ortWebGpuProfile', args.kernelProfile ? '1' : '0');
  url.searchParams.set('ortMonkeyPatchWebGpu', args.monkeyPatch ? '1' : '0');
  if (args.gpuOutputs) url.searchParams.set('ortPreferredOutputLocation', 'gpu-buffer');
  else url.searchParams.set('ortGpuOutputs', '0');
  if (args.model) url.searchParams.set('model', args.model);
  if (args.fen) url.searchParams.set('fen', args.fen);
  return String(url);
}

function runAgent(args, commandArgs, timeoutMs = 30_000) {
  const fullArgs = ['--json', '--session', args.session, ...commandArgs];
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
          if (parsed.success === false)
            return finish(reject, new Error(`${args.agentBrowser} ${fullArgs.slice(1).join(' ')} failed: ${parsed.error ?? stdout}`));
          return finish(resolve, parsed.data ?? parsed);
        }
        return finish(resolve, parsed);
      } catch (error) {
        return finish(reject, error);
      }
    });
  });
}

async function closeAgentSession(args) {
  try {
    await runAgent(args, ['close'], 5_000);
  } catch (error) {
    process.stderr.write(`[lc0-ort-readback-profile] warning: failed to close ${args.session}: ${error.message ?? error}\n`);
  }
}

async function waitForServer(baseUrl, timeoutMs) {
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

function startServer(args) {
  if (args.noServer) return null;
  const server = spawn('npm', ['run', 'web:client', '--', '--host', args.host, '--port', String(args.port)], { stdio: ['ignore', 'pipe', 'pipe'] });
  const echoOutput = (chunk) => process.stderr.write(`[vite] ${chunk}`);
  server.stdout.on('data', echoOutput);
  server.stderr.on('data', echoOutput);
  server.ready = waitForOutput(server, {
    match: (text) => /ready in \d+\s*ms/.test(text) || text.includes(`:${args.port}/`),
    timeoutMs: 30_000,
    label: `Vite dev server (port ${args.port})`,
  });
  return server;
}

function textFromGetResult(result) {
  if (typeof result?.text === 'string') return result.text;
  if (typeof result === 'string') return result;
  throw new Error(`agent-browser get text returned unexpected payload: ${JSON.stringify(result)}`);
}

function browserInfoFromEvalResult(result) {
  const value = result?.value ?? result?.result ?? result;
  if (
    !value ||
    typeof value !== 'object' ||
    typeof value.userAgent !== 'string' ||
    typeof value.platform !== 'string' ||
    !Number.isFinite(value.hardwareConcurrency)
  ) {
    throw new Error(`agent-browser eval returned unexpected browser info: ${JSON.stringify(result)}`);
  }
  return {
    userAgent: value.userAgent,
    platform: value.platform,
    hardwareConcurrency: value.hardwareConcurrency,
  };
}

async function readBrowserInfo(args) {
  const result = await runAgent(
    args,
    [
      'eval',
      `(() => ({
    userAgent: navigator.userAgent,
    platform: navigator.platform,
    hardwareConcurrency: navigator.hardwareConcurrency,
  }))()`,
    ],
    30_000,
  );
  return browserInfoFromEvalResult(result);
}

async function runBrowserBenchmark(args) {
  const url = benchmarkUrl(args);
  process.stderr.write(`[lc0-ort-readback-profile] ${url}\n`);
  try {
    await runAgent(args, ['open', url], 30_000);
    const deadline = Date.now() + args.timeoutMs;
    while (Date.now() < deadline) {
      const chunk = Math.min(25_000, Math.max(1000, deadline - Date.now()));
      try {
        await runAgent(args, ['wait', '--text', 'BENCH_DONE', '--timeout', String(chunk)], chunk + 5_000);
        const text = textFromGetResult(await runAgent(args, ['get', 'text', '#benchResult'], 30_000));
        const result = JSON.parse(text);
        if (result.status !== 'BENCH_DONE') throw new Error(`unexpected benchmark status: ${result.status}`);
        if (args.ortWasmVariant && !String(result.backend).includes(`[${args.ortWasmVariant}:`)) {
          throw new Error(`requested ORT WASM ${args.ortWasmVariant}, received backend ${result.backend}`);
        }
        const browserInfo = await readBrowserInfo(args);
        return { ...result, browserInfo };
      } catch (error) {
        if (Date.now() >= deadline) throw error;
      }
    }
    throw new Error(`Timed out waiting for BENCH_DONE after ${args.timeoutMs}ms`);
  } finally {
    await closeAgentSession(args);
  }
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.dryRun) {
    console.log(benchmarkUrl(args));
    return;
  }
  const server = startServer(args);
  try {
    if (server) await server.ready;
    await waitForServer(args.baseUrl, 30_000);
    const result = await runBrowserBenchmark(args);
    console.log(JSON.stringify(result, null, 2));
  } finally {
    server?.kill('SIGTERM');
  }
}

main().catch((error) => {
  console.error(error.stack ?? error.message);
  process.exit(1);
});
