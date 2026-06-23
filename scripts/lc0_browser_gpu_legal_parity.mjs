#!/usr/bin/env node
import { spawn } from 'node:child_process';
import { mkdir, writeFile } from 'node:fs/promises';
import { dirname } from 'node:path';
import { setTimeout as delay } from 'node:timers/promises';
import { createHash } from 'node:crypto';

const DEFAULT_HOST = '127.0.0.1';
const DEFAULT_PORT = 5198;
const DEFAULT_TIMEOUT_MS = 180_000;

function usage() {
  console.log(`Usage: node scripts/lc0_browser_gpu_legal_parity.mjs [options]

Runs an isolated browser parity probe comparing JS legal priors to the opt-in WGSL GPU legal-prior scaffold.

Options:
  --out PATH              Write JSON artifact
  --base-url URL          Use an existing dev server
  --host HOST             Vite host when auto-starting (default ${DEFAULT_HOST})
  --port N                Vite port when auto-starting (default ${DEFAULT_PORT})
  --agent-browser BIN     Browser automation binary (default AGENT_BROWSER_BIN or agent-browser)
  --session NAME          agent-browser session name
  --timeout MS            Browser wait timeout (default ${DEFAULT_TIMEOUT_MS})
  --fixture-limit N       Native fixtures to compare (default 3)
  --top-k N               Top-K order/detail check (default 16)
  --max-prior-diff N      Max allowed prior/top-K prior abs diff (default 0.01)
  --max-wdl-diff N        Max allowed WDL abs diff (default 0.005)
  --max-logit-diff N      Max allowed legal logit abs diff (default 0.05)
  --input-backend MODE    js, wgsl, or wasm (default wasm)
  --encoder-kernel MODE   Encoder kernel variant (default mixed-tvm-ffn)
  --no-server             Do not auto-start Vite
  --dry-run               Print URL and exit
  -h, --help              Show this help
`);
}

function shortSessionName(value) {
  const safe = String(value).replace(/[^A-Za-z0-9_.-]+/g, '-');
  if (safe.length <= 60) return safe;
  const hash = createHash('sha1').update(safe).digest('hex').slice(0, 10);
  return `${safe.slice(0, 49)}-${hash}`;
}

function parseArgs(argv) {
  const args = {
    host: DEFAULT_HOST,
    port: DEFAULT_PORT,
    agentBrowser: process.env.AGENT_BROWSER_BIN ?? 'agent-browser',
    session: process.env.AGENT_BROWSER_SESSION ?? `lc0-gpu-legal-${process.pid}`,
    timeoutMs: DEFAULT_TIMEOUT_MS,
    fixtureLimit: 3,
    topK: 16,
    maxPriorDiff: 0.01,
    maxWdlDiff: 0.005,
    maxLogitDiff: 0.05,
    inputBackend: 'wasm',
    encoderKernel: 'mixed-tvm-ffn',
    out: '',
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
    else if (arg === '--host') args.host = next();
    else if (arg === '--port') args.port = Number(next());
    else if (arg === '--agent-browser') args.agentBrowser = next();
    else if (arg === '--session') args.session = next();
    else if (arg === '--timeout') args.timeoutMs = Number(next());
    else if (arg === '--fixture-limit' || arg === '--fixtures') args.fixtureLimit = Number(next());
    else if (arg === '--top-k' || arg === '--topk') args.topK = Number(next());
    else if (arg === '--max-prior-diff') args.maxPriorDiff = Number(next());
    else if (arg === '--max-wdl-diff') args.maxWdlDiff = Number(next());
    else if (arg === '--max-logit-diff') args.maxLogitDiff = Number(next());
    else if (arg === '--input-backend') args.inputBackend = next();
    else if (arg === '--encoder-kernel') args.encoderKernel = next();
    else if (arg === '--no-server') args.noServer = true;
    else if (arg === '--dry-run') args.dryRun = true;
    else if (arg === '-h' || arg === '--help') args.help = true;
    else throw new Error(`Unknown option: ${arg}`);
  }
  if (!args.baseUrl) args.baseUrl = `http://${args.host}:${args.port}`;
  if (args.explicitBaseUrl) args.noServer = true;
  args.session = shortSessionName(args.session);
  for (const [name, value] of [['port', args.port], ['timeout', args.timeoutMs], ['fixture-limit', args.fixtureLimit], ['top-k', args.topK]]) {
    if (!Number.isFinite(value) || value <= 0) throw new Error(`Invalid --${name}: ${value}`);
  }
  for (const [name, value] of [['max-prior-diff', args.maxPriorDiff], ['max-wdl-diff', args.maxWdlDiff], ['max-logit-diff', args.maxLogitDiff]]) {
    if (!Number.isFinite(value) || value < 0) throw new Error(`Invalid --${name}: ${value}`);
  }
  if (!['js', 'wgsl', 'wasm'].includes(args.inputBackend)) throw new Error(`Invalid --input-backend: ${args.inputBackend}`);
  if (!['hand', 'tvm-packed-f16', 'mixed-tvm-ffn', 'mixed-tvm-ffn-outproj', 'mixed-tvm-ffn-smolgen-project'].includes(args.encoderKernel)) throw new Error(`Invalid --encoder-kernel: ${args.encoderKernel}`);
  return args;
}

function probeUrl(args) {
  const url = new URL('/single-engine', args.baseUrl);
  url.searchParams.set('gpuLegalParity', '1');
  url.searchParams.set('runtime', 'hybrid-wgsl-heads');
  url.searchParams.set('headBackend', 'wgsl');
  url.searchParams.set('wgslBatchMode', 'physical');
  url.searchParams.set('inputBackend', args.inputBackend);
  url.searchParams.set('encoderKernel', args.encoderKernel);
  url.searchParams.set('encoderLayers', '10');
  url.searchParams.set('gpuLegalParityLimit', String(args.fixtureLimit));
  url.searchParams.set('topK', String(args.topK));
  url.searchParams.set('ep', 'wasm');
  url.searchParams.set('packVerify', '0');
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
          if (parsed.success === false) return finish(reject, new Error(`${args.agentBrowser} ${fullArgs.slice(1).join(' ')} failed: ${parsed.error ?? stdout}`));
          return finish(resolve, parsed.data ?? parsed);
        }
        return finish(resolve, parsed);
      } catch (error) { return finish(reject, error); }
    });
  });
}

function textFromGetResult(result) {
  if (typeof result?.text === 'string') return result.text;
  if (typeof result === 'string') return result;
  throw new Error(`agent-browser get text returned unexpected payload: ${JSON.stringify(result)}`);
}

function startServer(args) {
  if (args.noServer) return null;
  const server = spawn('npm', ['run', 'web:client', '--', '--host', args.host, '--port', String(args.port), '--strictPort'], { stdio: ['ignore', 'pipe', 'pipe'] });
  let output = '';
  let readySettled = false;
  server.ready = new Promise((resolve, reject) => {
    const settle = (fn, value) => {
      if (readySettled) return;
      readySettled = true;
      clearTimeout(timer);
      fn(value);
    };
    const timer = setTimeout(() => settle(reject, new Error(`Vite dev server did not report readiness on port ${args.port}: ${output.trim()}`)), 30_000);
    const onOutput = (chunk) => {
      output += chunk.toString('utf8');
      if (/ready in \d+\s*ms/.test(output) || output.includes(`:${args.port}/`)) settle(resolve);
    };
    server.stdout.on('data', (chunk) => { process.stderr.write(`[vite] ${chunk}`); onOutput(chunk); });
    server.stderr.on('data', (chunk) => { process.stderr.write(`[vite] ${chunk}`); onOutput(chunk); });
    server.on('exit', (status, signal) => settle(reject, new Error(`Vite dev server exited before ready (${status ?? signal}): ${output.trim()}`)));
  });
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
    } catch (error) { lastError = error; }
    await delay(250);
  }
  throw new Error(`Vite dev server did not become ready at ${baseUrl}: ${lastError?.message ?? 'timeout'}`);
}

async function closeAgentSession(args) {
  try { await runAgent(args, ['close'], 5_000); }
  catch (error) { process.stderr.write(`[gpu-legal-parity] warning: failed to close session: ${error.message ?? error}\n`); }
}

async function runBrowserProbe(args) {
  const url = probeUrl(args);
  process.stderr.write(`[gpu-legal-parity] ${url}\n`);
  try {
    await runAgent(args, ['open', url], 30_000);
    const deadline = Date.now() + args.timeoutMs;
    let lastText = '';
    while (Date.now() < deadline) {
      try {
        const text = textFromGetResult(await runAgent(args, ['get', 'text', '#benchResult'], 10_000));
        if (text !== lastText) {
          lastText = text;
          process.stderr.write(`[gpu-legal-parity] progress ${text.slice(0, 160).replace(/\s+/g, ' ')}${text.length > 160 ? '…' : ''}\n`);
        }
        if (text.startsWith('GPU_LEGAL_PARITY_FAILED')) throw new Error(text);
        if (text.includes('GPU_LEGAL_PARITY_DONE')) return JSON.parse(text);
      } catch (error) {
        if (Date.now() >= deadline) throw error;
      }
      await delay(500);
    }
    throw new Error(`Timed out waiting for GPU_LEGAL_PARITY_DONE after ${args.timeoutMs}ms (last #benchResult: ${lastText || 'empty'})`);
  } finally { await closeAgentSession(args); }
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) return usage();
  if (args.dryRun) { console.log(probeUrl(args)); return; }
  const server = startServer(args);
  try {
    if (server) await server.ready;
    await waitForServer(args.baseUrl);
    const result = await runBrowserProbe(args);
    if (args.out) {
      await mkdir(dirname(args.out), { recursive: true });
      await writeFile(args.out, JSON.stringify(result, null, 2));
    }
    const summary = {
      status: result.status,
      out: args.out || undefined,
      fixtures: result.fixtures,
      bestMoveMatches: result.bestMoveMatches,
      topK: result.topK,
      topKMatches: result.topKMatches,
      maxPriorAbsDiff: result.maxPriorAbsDiff,
      maxTopKPriorAbsDiff: result.maxTopKPriorAbsDiff,
      maxLogitAbsDiff: result.maxLogitAbsDiff,
      maxWdlAbsDiff: result.maxWdlAbsDiff,
      compactTopKReadbackBytesEstimate: result.compactTopKReadbackBytesEstimate,
      fullGpuLegalReadbackBytes: result.fullGpuLegalReadbackBytes,
      thresholds: {
        maxPriorDiff: args.maxPriorDiff,
        maxWdlDiff: args.maxWdlDiff,
        maxLogitDiff: args.maxLogitDiff,
      },
    };
    console.log(JSON.stringify(summary, null, 2));
    const driftFailures = [];
    if (Number(result.maxPriorAbsDiff) > args.maxPriorDiff) driftFailures.push(`prior ${result.maxPriorAbsDiff} > ${args.maxPriorDiff}`);
    if (Number(result.maxTopKPriorAbsDiff) > args.maxPriorDiff) driftFailures.push(`topK prior ${result.maxTopKPriorAbsDiff} > ${args.maxPriorDiff}`);
    if (Number(result.maxWdlAbsDiff) > args.maxWdlDiff) driftFailures.push(`wdl ${result.maxWdlAbsDiff} > ${args.maxWdlDiff}`);
    if (Number(result.maxLogitAbsDiff) > args.maxLogitDiff) driftFailures.push(`logit ${result.maxLogitAbsDiff} > ${args.maxLogitDiff}`);
    if (result.bestMoveMatches !== result.fixtures || result.topKMatches !== result.fixtures || result.maxMissingFromGpu !== 0 || driftFailures.length) {
      const driftMessage = driftFailures.length ? `, drift ${driftFailures.join('; ')}` : '';
      throw new Error(`GPU legal parity failed: best ${result.bestMoveMatches}/${result.fixtures}, topK ${result.topKMatches}/${result.fixtures}, missing ${result.maxMissingFromGpu}${driftMessage}`);
    }
  } finally { server?.kill('SIGTERM'); }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
