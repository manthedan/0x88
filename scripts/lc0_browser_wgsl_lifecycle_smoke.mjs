#!/usr/bin/env node
import { spawn } from 'node:child_process';
import { mkdir, writeFile } from 'node:fs/promises';
import { dirname } from 'node:path';
import { setTimeout as delay } from 'node:timers/promises';

const DEFAULT_HOST = '127.0.0.1';
const DEFAULT_PORT = 5179;
const DEFAULT_TIMEOUT_MS = 480_000;

function usage() {
  console.log(`Usage: node --experimental-strip-types scripts/lc0_browser_wgsl_lifecycle_smoke.mjs [options]\n\nRuns repeated browser/WebGPU WGSL-head deferred-readback lifecycle cycles. Each cycle asks the worker to create a fresh hybrid runtime, exercise physical WGSL batch buffers plus deferred double-buffer readback, destroy the runtime, and report browser memory samples when available.\n\nOptions:\n  --out PATH            Write full JSON artifact\n  --base-url URL        Use an existing dev server\n  --port N              Vite port when auto-starting (default ${DEFAULT_PORT})\n  --host HOST           Vite host when auto-starting (default ${DEFAULT_HOST})\n  --agent-browser BIN   Browser automation binary (default AGENT_BROWSER_BIN or agent-browser)\n  --session NAME        agent-browser session name\n  --timeout MS          Total browser wait timeout (default ${DEFAULT_TIMEOUT_MS})\n  --cycles N            Runtime create/exercise/destroy cycles (default 3)\n  --layers N            Encoder layers for hybrid path (default 10)\n  --input-backend MODE  js, wgsl, or wasm input path (default js)\n  --legal-priors-backend MODE\n                       Legal-prior backend: js, wasm, or gpu (default js; gpu is opt-in)\n  --batch N             Physical WGSL batch size (default 4)\n  --iters N             Timed batches per immediate/deferred mode and cycle (default 4)\n  --warmup N            Warmup batches per mode and cycle (default 1)\n  --fixture-limit N     Representative fixtures per cycle (default 4)\n  --pause-ms N          Pause between cycles (default 0)\n  --pack-verify         Enable shard sha256 verification (default skipped for smoke speed)\n  --allow-mismatches    Exit 0 even if any immediate/deferred best moves differ\n  --skip-leak-check     Skip final browser/process leak check\n  --no-server           Do not auto-start Vite\n  --dry-run             Print URL and exit\n  -h, --help            Show this help\n`);
}

function parseArgs(argv) {
  const args = {
    host: DEFAULT_HOST,
    port: DEFAULT_PORT,
    timeoutMs: DEFAULT_TIMEOUT_MS,
    agentBrowser: process.env.AGENT_BROWSER_BIN ?? 'agent-browser',
    session: process.env.AGENT_BROWSER_SESSION ?? `lc0-wgsl-lifecycle-${process.pid}`,
    cycles: 3,
    layers: 10,
    inputBackend: 'js',
    legalPriorsBackend: 'js',
    batch: 4,
    iters: 4,
    warmup: 1,
    fixtureLimit: 4,
    pauseMs: 0,
    packVerify: false,
    allowMismatches: false,
    skipLeakCheck: false,
    noServer: false,
    dryRun: false,
    explicitBaseUrl: false,
  };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    const next = () => {
      if (i + 1 >= argv.length) throw new Error(`${arg} requires a value`);
      return argv[++i];
    };
    if (arg === '--out') args.out = next();
    else if (arg === '--base-url') { args.baseUrl = next(); args.explicitBaseUrl = true; }
    else if (arg === '--port') args.port = Number(next());
    else if (arg === '--host') args.host = next();
    else if (arg === '--agent-browser') args.agentBrowser = next();
    else if (arg === '--session') args.session = next();
    else if (arg === '--timeout') args.timeoutMs = Number(next());
    else if (arg === '--cycles') args.cycles = Number(next());
    else if (arg === '--layers') args.layers = Number(next());
    else if (arg === '--input-backend') args.inputBackend = next();
    else if (arg === '--legal-priors-backend' || arg === '--hybrid-legal-priors') args.legalPriorsBackend = next();
    else if (arg === '--batch') args.batch = Number(next());
    else if (arg === '--iters') args.iters = Number(next());
    else if (arg === '--warmup') args.warmup = Number(next());
    else if (arg === '--fixture-limit') args.fixtureLimit = Number(next());
    else if (arg === '--pause-ms') args.pauseMs = Number(next());
    else if (arg === '--pack-verify') args.packVerify = true;
    else if (arg === '--allow-mismatches') args.allowMismatches = true;
    else if (arg === '--skip-leak-check') args.skipLeakCheck = true;
    else if (arg === '--no-server') args.noServer = true;
    else if (arg === '--dry-run') args.dryRun = true;
    else if (arg === '-h' || arg === '--help') args.help = true;
    else throw new Error(`Unknown option: ${arg}`);
  }
  if (!args.baseUrl) args.baseUrl = `http://${args.host}:${args.port}`;
  if (args.explicitBaseUrl) args.noServer = true;
  if (!['js', 'wgsl', 'wasm'].includes(args.inputBackend)) throw new Error(`Invalid --input-backend: ${args.inputBackend}`);
  if (!['js', 'wasm', 'gpu'].includes(args.legalPriorsBackend)) throw new Error(`Invalid --legal-priors-backend: ${args.legalPriorsBackend}`);
  for (const [name, value] of [['port', args.port], ['timeout', args.timeoutMs], ['cycles', args.cycles], ['layers', args.layers], ['batch', args.batch], ['iters', args.iters], ['warmup', args.warmup], ['fixture-limit', args.fixtureLimit], ['pause-ms', args.pauseMs]]) {
    if (!Number.isFinite(value) || value < 0 || (!['warmup', 'pause-ms'].includes(name) && value <= 0)) throw new Error(`Invalid --${name}: ${value}`);
  }
  return args;
}

function lifecycleUrl(args) {
  const url = new URL('/lc0-policy-only.html', args.baseUrl);
  url.searchParams.set('wgslDeferredReadbackLifecycle', '1');
  url.searchParams.set('runtime', 'hybrid');
  url.searchParams.set('headBackend', 'wgsl');
  url.searchParams.set('wgslBatchMode', 'physical');
  url.searchParams.set('inputBackend', args.inputBackend);
  if (args.legalPriorsBackend !== 'js') url.searchParams.set('legalPriorsBackend', args.legalPriorsBackend);
  url.searchParams.set('encoderLayers', String(args.layers));
  url.searchParams.set('lifecycleCycles', String(args.cycles));
  url.searchParams.set('deferredReadbackBatch', String(args.batch));
  url.searchParams.set('deferredReadbackIters', String(args.iters));
  url.searchParams.set('deferredReadbackWarmup', String(args.warmup));
  url.searchParams.set('fixtureLimit', String(args.fixtureLimit));
  url.searchParams.set('lifecyclePauseMs', String(args.pauseMs));
  url.searchParams.set('ep', 'wasm');
  if (!args.packVerify) url.searchParams.set('packVerify', '0');
  return String(url);
}

function runAgent(args, commandArgs, timeoutMs = 30_000) {
  const fullArgs = ['--json', '--session', args.session, ...commandArgs];
  return new Promise((resolve, reject) => {
    const child = spawn(args.agentBrowser, fullArgs, { stdio: ['ignore', 'pipe', 'pipe'] });
    const chunks = { stdout: [], stderr: [] };
    let settled = false;
    const finish = (fn, value) => { if (!settled) { settled = true; clearTimeout(timer); fn(value); } };
    const timer = setTimeout(() => { child.kill('SIGKILL'); finish(reject, new Error(`${args.agentBrowser} ${fullArgs.slice(1).join(' ')} timed out after ${timeoutMs}ms`)); }, timeoutMs);
    child.stdout.on('data', (chunk) => chunks.stdout.push(chunk));
    child.stderr.on('data', (chunk) => chunks.stderr.push(chunk));
    child.on('error', (error) => finish(reject, error));
    child.on('close', (status) => {
      const stdout = Buffer.concat(chunks.stdout).toString('utf8');
      const stderr = Buffer.concat(chunks.stderr).toString('utf8');
      if (status !== 0) return finish(reject, new Error(`${args.agentBrowser} failed: ${stderr || stdout}`));
      try {
        const parsed = stdout ? JSON.parse(stdout.trim()) : null;
        if (parsed && typeof parsed === 'object' && 'success' in parsed) {
          if (parsed.success === false) return finish(reject, new Error(`${args.agentBrowser} failed: ${parsed.error ?? stdout}`));
          return finish(resolve, parsed.data ?? parsed);
        }
        return finish(resolve, parsed);
      } catch (error) { return finish(reject, error); }
    });
  });
}

async function closeAgentSession(args) {
  try { await runAgent(args, ['close'], 5_000); }
  catch (error) { process.stderr.write(`[lc0-wgsl-lifecycle] warning: failed to close session: ${error.message ?? error}\n`); }
}

async function waitForServer(baseUrl, timeoutMs = 30_000) {
  const deadline = Date.now() + timeoutMs;
  let lastError;
  while (Date.now() < deadline) {
    try {
      const response = await fetch(new URL('/lc0-policy-only.html', baseUrl), { cache: 'no-store' });
      if (response.ok) return;
      lastError = new Error(`HTTP ${response.status}`);
    } catch (error) { lastError = error; }
    await delay(250);
  }
  throw new Error(`Vite dev server did not become ready at ${baseUrl}: ${lastError?.message ?? 'timeout'}`);
}

function spawnCapture(command, commandArgs, options = {}) {
  return new Promise((resolve, reject) => {
    const { timeoutMs, echoStderr, ...spawnOptions } = options;
    const child = spawn(command, commandArgs, { stdio: ['ignore', 'pipe', 'pipe'], ...spawnOptions });
    const chunks = { stdout: [], stderr: [] };
    let settled = false;
    const finish = (fn, value) => {
      if (settled) return;
      settled = true;
      if (timer) clearTimeout(timer);
      fn(value);
    };
    const timer = timeoutMs ? setTimeout(() => {
      child.kill('SIGKILL');
      finish(reject, new Error(`${command} ${commandArgs.join(' ')} timed out after ${timeoutMs}ms`));
    }, timeoutMs) : undefined;
    child.stdout.on('data', (chunk) => chunks.stdout.push(chunk));
    child.stderr.on('data', (chunk) => {
      chunks.stderr.push(chunk);
      if (echoStderr) process.stderr.write(chunk);
    });
    child.on('error', (error) => finish(reject, error));
    child.on('close', (status) => {
      const stdout = Buffer.concat(chunks.stdout).toString('utf8');
      const stderr = Buffer.concat(chunks.stderr).toString('utf8');
      if (status !== 0) return finish(reject, new Error(`${command} ${commandArgs.join(' ')} failed with ${status}: ${stderr || stdout}`));
      finish(resolve, { stdout, stderr });
    });
  });
}

function startServer(args) {
  if (args.noServer) return null;
  const server = spawn('npm', ['run', 'web:client', '--', '--host', args.host, '--port', String(args.port), '--strictPort'], { stdio: ['ignore', 'pipe', 'pipe'] });
  server.stdout.on('data', (chunk) => process.stderr.write(`[vite] ${chunk}`));
  server.stderr.on('data', (chunk) => process.stderr.write(`[vite] ${chunk}`));
  return server;
}

async function leakCheck(args, options = {}) {
  await runAgent(args, ['close', '--all'], 10_000).catch((error) => process.stderr.write(`[lc0-wgsl-lifecycle] warning: close --all failed: ${error.message ?? error}\n`));
  const cleanupPatterns = ['agent-browser/bin/agent-browser', 'Google Chrome for Testing'];
  if (options.checkVite !== false) cleanupPatterns.push(`vite .*${args.port}`);
  for (const pattern of cleanupPatterns) await spawnCapture('pkill', ['-f', pattern], { timeoutMs: 10_000 }).catch(() => undefined);
  await delay(1000);
  const { stdout } = await spawnCapture('ps', ['-axo', 'pid,rss,command'], { timeoutMs: 10_000 });
  const pattern = options.checkVite === false
    ? /Google Chrome for Testing|agent-browser|lc0_browser_wgsl_lifecycle|lc0-policy-only/
    : new RegExp(`Google Chrome for Testing|agent-browser|vite .*${args.port}|lc0_browser_wgsl_lifecycle|lc0-policy-only`);
  const leaks = stdout.split('\n').filter((line) => pattern.test(line) && !/lc0_browser_wgsl_lifecycle_smoke|npm run lc0:browser-wgsl-lifecycle-smoke/.test(line));
  if (leaks.length) throw new Error(`WGSL lifecycle browser/process leak check failed:\n${leaks.join('\n')}`);
  return { status: 'LC0_WGSL_LIFECYCLE_LEAK_CHECK_CLEAN' };
}

function textFromGetResult(result) {
  if (typeof result?.text === 'string') return result.text;
  if (typeof result === 'string') return result;
  throw new Error(`agent-browser get text returned unexpected payload: ${JSON.stringify(result)}`);
}

async function runBrowserLifecycle(args) {
  const url = lifecycleUrl(args);
  process.stderr.write(`[lc0-wgsl-lifecycle] ${url}\n`);
  try {
    await runAgent(args, ['open', url], 30_000);
    const deadline = Date.now() + args.timeoutMs;
    while (Date.now() < deadline) {
      const chunk = Math.min(25_000, Math.max(1000, deadline - Date.now()));
      let doneSeen = false;
      try {
        await runAgent(args, ['wait', '--text', 'WGSL_DEFERRED_READBACK_LIFECYCLE_DONE', '--timeout', String(chunk)], chunk + 5_000);
        doneSeen = true;
        const text = textFromGetResult(await runAgent(args, ['get', 'text', '#benchResult'], 30_000));
        const result = JSON.parse(text);
        if (result.status !== 'WGSL_DEFERRED_READBACK_LIFECYCLE_DONE') throw new Error(`unexpected status: ${result.status}`);
        return result;
      } catch (error) {
        if (doneSeen || Date.now() >= deadline) throw error;
      }
    }
    throw new Error(`Timed out waiting for WGSL_DEFERRED_READBACK_LIFECYCLE_DONE after ${args.timeoutMs}ms`);
  } finally { await closeAgentSession(args); }
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) return usage();
  if (args.dryRun) { console.log(lifecycleUrl(args)); return; }
  const server = startServer(args);
  let result;
  let runError;
  try {
    await waitForServer(args.baseUrl);
    result = await runBrowserLifecycle(args);
  } catch (error) {
    runError = error;
  } finally {
    server?.kill('SIGTERM');
    if (server) await delay(1000);
  }
  const leak = args.skipLeakCheck ? { status: 'LC0_WGSL_LIFECYCLE_LEAK_CHECK_SKIPPED' } : await leakCheck(args, { checkVite: !args.noServer });
  if (runError) throw runError;
  const artifact = { ...result, leak };
  if (args.out) {
    await mkdir(dirname(args.out), { recursive: true });
    await writeFile(args.out, JSON.stringify(artifact, null, 2));
  }
  const summary = {
    status: artifact.status,
    out: args.out,
    cycles: artifact.cycles,
    inputCount: artifact.inputCount,
    allCyclesBestMovesMatch: artifact.allCyclesBestMovesMatch,
    failedCycles: artifact.failedCycles,
    lastImmediateEvalsPerSecond: artifact.cycleResults?.at(-1)?.immediate?.evalsPerSecond,
    lastDeferredEvalsPerSecond: artifact.cycleResults?.at(-1)?.deferred?.evalsPerSecond,
    leak: artifact.leak?.status,
  };
  console.log(JSON.stringify(summary, null, 2));
  if (!args.allowMismatches && !artifact.allCyclesBestMovesMatch) {
    throw new Error(`WGSL deferred-readback lifecycle best-move mismatch in cycles: ${artifact.failedCycles?.join(',') || 'unknown'}; pass --allow-mismatches for artifact capture`);
  }
}

main().catch((error) => { console.error(error.stack ?? error.message); process.exit(1); });
