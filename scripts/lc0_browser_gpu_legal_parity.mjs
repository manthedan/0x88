#!/usr/bin/env node
import { spawn } from 'node:child_process';
import { createHash } from 'node:crypto';
import { mkdir, writeFile } from 'node:fs/promises';
import { dirname } from 'node:path';
import { setTimeout as delay } from 'node:timers/promises';
import { parseScriptArgs } from './lib/cli.mjs';
import { waitForOutput } from './lib/server.mjs';

const DEFAULT_HOST = '127.0.0.1';
const DEFAULT_PORT = 5198;
const DEFAULT_TIMEOUT_MS = 180_000;

const USAGE = `Usage: node scripts/lc0_browser_gpu_legal_parity.mjs [options]

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
`;

function shortSessionName(value) {
  const safe = String(value).replace(/[^A-Za-z0-9_.-]+/g, '-');
  if (safe.length <= 60) return safe;
  const hash = createHash('sha1').update(safe).digest('hex').slice(0, 10);
  return `${safe.slice(0, 49)}-${hash}`;
}

function parseArgs(argv) {
  const args = parseScriptArgs(argv, {
    options: {
      out: { type: 'string', default: '' },
      'base-url': { type: 'string' },
      host: { type: 'string', default: DEFAULT_HOST },
      port: { type: 'string', default: String(DEFAULT_PORT) },
      'agent-browser': { type: 'string', default: process.env.AGENT_BROWSER_BIN ?? 'agent-browser' },
      session: { type: 'string', default: process.env.AGENT_BROWSER_SESSION ?? `lc0-gpu-legal-${process.pid}` },
      timeout: { type: 'string', default: String(DEFAULT_TIMEOUT_MS) },
      'fixture-limit': { type: 'string', default: '3' },
      fixtures: { type: 'string' },
      'top-k': { type: 'string', default: '16' },
      topk: { type: 'string' },
      'max-prior-diff': { type: 'string', default: '0.01' },
      'max-wdl-diff': { type: 'string', default: '0.005' },
      'max-logit-diff': { type: 'string', default: '0.05' },
      'input-backend': { type: 'string', default: 'wasm' },
      'encoder-kernel': { type: 'string', default: 'mixed-tvm-ffn' },
      'no-server': { type: 'boolean', default: false },
      'dry-run': { type: 'boolean', default: false },
    },
    usage: USAGE,
  });
  if (args.fixtures !== undefined) args.fixtureLimit = args.fixtures;
  delete args.fixtures;
  if (args.topk !== undefined) args.topK = args.topk;
  delete args.topk;
  args.port = Number(args.port);
  args.timeoutMs = Number(args.timeout);
  delete args.timeout;
  args.fixtureLimit = Number(args.fixtureLimit);
  args.topK = Number(args.topK);
  args.maxPriorDiff = Number(args.maxPriorDiff);
  args.maxWdlDiff = Number(args.maxWdlDiff);
  args.maxLogitDiff = Number(args.maxLogitDiff);
  args.explicitBaseUrl = args.baseUrl !== undefined;
  if (!args.baseUrl) args.baseUrl = `http://${args.host}:${args.port}`;
  if (args.explicitBaseUrl) args.noServer = true;
  args.session = shortSessionName(args.session);
  for (const [name, value] of [
    ['port', args.port],
    ['timeout', args.timeoutMs],
    ['fixture-limit', args.fixtureLimit],
    ['top-k', args.topK],
  ]) {
    if (!Number.isFinite(value) || value <= 0) throw new Error(`Invalid --${name}: ${value}`);
  }
  for (const [name, value] of [
    ['max-prior-diff', args.maxPriorDiff],
    ['max-wdl-diff', args.maxWdlDiff],
    ['max-logit-diff', args.maxLogitDiff],
  ]) {
    if (!Number.isFinite(value) || value < 0) throw new Error(`Invalid --${name}: ${value}`);
  }
  if (!['js', 'wgsl', 'wasm'].includes(args.inputBackend)) throw new Error(`Invalid --input-backend: ${args.inputBackend}`);
  if (!['hand', 'tvm-packed-f16', 'mixed-tvm-ffn', 'mixed-tvm-ffn-outproj', 'mixed-tvm-ffn-smolgen-project'].includes(args.encoderKernel))
    throw new Error(`Invalid --encoder-kernel: ${args.encoderKernel}`);
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

function textFromGetResult(result) {
  if (typeof result?.text === 'string') return result.text;
  if (typeof result === 'string') return result;
  throw new Error(`agent-browser get text returned unexpected payload: ${JSON.stringify(result)}`);
}

function startServer(args) {
  if (args.noServer) return null;
  const server = spawn('npm', ['run', 'web:client', '--', '--host', args.host, '--port', String(args.port), '--strictPort'], {
    stdio: ['ignore', 'pipe', 'pipe'],
  });
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

async function closeAgentSession(args) {
  try {
    await runAgent(args, ['close'], 5_000);
  } catch (error) {
    process.stderr.write(`[gpu-legal-parity] warning: failed to close session: ${error.message ?? error}\n`);
  }
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
  } finally {
    await closeAgentSession(args);
  }
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.dryRun) {
    console.log(probeUrl(args));
    return;
  }
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
      throw new Error(
        `GPU legal parity failed: best ${result.bestMoveMatches}/${result.fixtures}, topK ${result.topKMatches}/${result.fixtures}, missing ${result.maxMissingFromGpu}${driftMessage}`,
      );
    }
  } finally {
    server?.kill('SIGTERM');
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
