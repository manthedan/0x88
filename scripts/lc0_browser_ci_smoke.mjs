#!/usr/bin/env node
import { spawn } from 'node:child_process';
import { mkdir, writeFile } from 'node:fs/promises';
import { dirname } from 'node:path';
import { setTimeout as delay } from 'node:timers/promises';
import { parseScriptArgs } from './lib/cli.mjs';
import { spawnCapture } from './lib/process.mjs';
import { waitForOutput } from './lib/server.mjs';

const DEFAULT_HOST = '127.0.0.1';
const DEFAULT_PORT = 5179;
const DEFAULT_TIMEOUT_MS = 180_000;
const DEFAULT_AGENT_BROWSER = process.env.AGENT_BROWSER_BIN ?? 'agent-browser';

const USAGE = `Usage: node scripts/lc0_browser_ci_smoke.mjs [options]\n\nRuns CI-style browser smokes for the stable hybrid backend, experimental WGSL heads, mapped-policy probe, WGSL-heads-vs-ORT fixtures against both WASM baseline and strict ORT WebGPU, and a final leak check.\n\nOptions:\n  --base-url URL        Use an existing server instead of starting Vite\n  --host HOST           Vite host (default ${DEFAULT_HOST})\n  --port N              Vite port (default ${DEFAULT_PORT})\n  --agent-browser BIN   Browser automation binary (default AGENT_BROWSER_BIN or agent-browser)\n  --timeout MS          Per-smoke timeout (default ${DEFAULT_TIMEOUT_MS})\n  --fixture-limit N     WGSL heads vs ORT fixture limit (default 3)\n  --max-error N         Probe max abs error tolerance (default 0.001)\n  --out PATH            Optional JSON artifact path\n  --no-server           Do not auto-start Vite\n  --skip-leak-check     Skip final browser/process leak check\n  --dry-run             Print planned smokes and URLs without running\n  -h, --help            Show this help\n`;

function parseArgs(argv) {
  const args = parseScriptArgs(argv, {
    options: {
      'base-url': { type: 'string' },
      host: { type: 'string', default: DEFAULT_HOST },
      port: { type: 'string', default: String(DEFAULT_PORT) },
      'agent-browser': { type: 'string', default: DEFAULT_AGENT_BROWSER },
      timeout: { type: 'string', default: String(DEFAULT_TIMEOUT_MS) },
      'fixture-limit': { type: 'string', default: '3' },
      'max-error': { type: 'string', default: '0.001' },
      out: { type: 'string' },
      'no-server': { type: 'boolean', default: false },
      'skip-leak-check': { type: 'boolean', default: false },
      'dry-run': { type: 'boolean', default: false },
    },
    usage: USAGE,
  });
  args.port = Number(args.port);
  args.timeoutMs = Number(args.timeout);
  delete args.timeout;
  args.fixtureLimit = Number(args.fixtureLimit);
  args.maxError = Number(args.maxError);
  args.explicitBaseUrl = args.baseUrl !== undefined;
  if (!args.baseUrl) args.baseUrl = `http://${args.host}:${args.port}`;
  if (args.explicitBaseUrl) args.noServer = true;
  for (const [name, value] of [
    ['port', args.port],
    ['timeout', args.timeoutMs],
    ['fixture-limit', args.fixtureLimit],
    ['max-error', args.maxError],
  ]) {
    if (!Number.isFinite(value) || value <= 0) throw new Error(`Invalid --${name}: ${value}`);
  }
  return args;
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

function parseJsonFromStdout(stdout) {
  const start = stdout.indexOf('{');
  if (start < 0) throw new Error(`No JSON object in command output: ${stdout.slice(0, 200)}`);
  return JSON.parse(stdout.slice(start));
}

function textFromGetResult(result) {
  if (typeof result?.text === 'string') return result.text;
  if (typeof result === 'string') return result;
  throw new Error(`agent-browser get text returned unexpected payload: ${JSON.stringify(result)}`);
}

async function runAgent(args, commandArgs, timeoutMs, session) {
  const fullArgs = ['--json', ...(session ? ['--session', session] : []), ...commandArgs];
  const stdout = await spawnCapture(args.agentBrowser, fullArgs, { timeoutMs });
  const parsed = stdout ? JSON.parse(stdout.trim()) : null;
  if (parsed && typeof parsed === 'object' && 'success' in parsed) {
    if (parsed.success === false) throw new Error(`${args.agentBrowser} ${commandArgs.join(' ')} failed: ${parsed.error ?? stdout}`);
    return parsed.data ?? parsed;
  }
  return parsed;
}

async function closeSession(args, session) {
  try {
    await runAgent(args, ['close'], 5_000, session);
  } catch (error) {
    process.stderr.write(`[ci-smoke] warning: failed to close ${session}: ${error.message ?? error}\n`);
  }
}

async function runUrlSmoke(args, smoke) {
  const session = `lc0-ci-${process.pid}-${smoke.name}`;
  const url = new URL('/single-engine', args.baseUrl);
  for (const [key, value] of Object.entries(smoke.params)) url.searchParams.set(key, String(value));
  process.stderr.write(`[ci-smoke] ${smoke.name}: ${url}\n`);
  try {
    await runAgent(args, ['open', String(url)], 30_000, session);
    const deadline = Date.now() + args.timeoutMs;
    while (Date.now() < deadline) {
      const chunk = Math.min(25_000, Math.max(1000, deadline - Date.now()));
      try {
        await runAgent(args, ['wait', '--text', smoke.doneText, '--timeout', String(chunk)], chunk + 5_000, session);
      } catch (error) {
        const message = String(error?.message ?? error).toLowerCase();
        if (Date.now() >= deadline || !message.includes('timed out')) throw error;
        continue;
      }
      const text = textFromGetResult(await runAgent(args, ['get', 'text', '#benchResult'], 30_000, session));
      const result = JSON.parse(text);
      smoke.validate(result, args);
      return { name: smoke.name, status: result.status, result };
    }
    throw new Error(`${smoke.name} timed out after ${args.timeoutMs}ms`);
  } finally {
    await closeSession(args, session);
  }
}

async function runHybridBenchSmoke(args, headBackend) {
  const session = `lc0-ci-${process.pid}-hybrid-${headBackend}`;
  const commandArgs = [
    'run',
    'lc0:browser-hybrid-search-bench',
    '--',
    '--base-url',
    args.baseUrl,
    '--agent-browser',
    args.agentBrowser,
    '--session',
    session,
    '--head-backend',
    headBackend,
    '--visits',
    '1',
    '--batch',
    '1',
    '--eval-iters',
    '1',
    '--eval-warmup',
    '0',
    '--search-iters',
    '1',
    '--search-warmup',
    '0',
    '--timeout',
    String(args.timeoutMs),
  ];
  process.stderr.write(`[ci-smoke] hybrid-${headBackend}\n`);
  try {
    const stdout = await spawnCapture('npm', commandArgs, { timeoutMs: args.timeoutMs + 60_000 });
    const result = parseJsonFromStdout(stdout);
    const expectedBackend = headBackend === 'wgsl' ? 'lc0web-wgsl-encoder-wgsl-heads' : 'lc0web-wgsl-encoder-ort-heads';
    if (result.status !== 'HYBRID_SEARCH_BENCH_DONE') throw new Error(`hybrid-${headBackend} unexpected status ${result.status}`);
    if (result.backend !== expectedBackend) throw new Error(`hybrid-${headBackend} unexpected backend ${result.backend}`);
    if (!result.search?.bestMove) throw new Error(`hybrid-${headBackend} did not return a best move`);
    return { name: `hybrid-${headBackend}`, status: result.status, result };
  } finally {
    await closeSession(args, session);
  }
}

function assertNumberAtMost(result, key, max) {
  const value = Number(result[key]);
  if (!Number.isFinite(value) || value > max) throw new Error(`${result.status} ${key} ${value} exceeds ${max}`);
}

function smokePlan(args) {
  return [
    { kind: 'hybrid', name: 'stable-ort-heads', headBackend: 'ort' },
    { kind: 'hybrid', name: 'experimental-wgsl-heads', headBackend: 'wgsl' },
    {
      kind: 'url',
      name: 'mapped-policy-probe',
      doneText: 'MAPPED_POLICY_PROBE_DONE',
      params: { mappedPolicyProbe: 1, packVerify: 0 },
      validate(result, runArgs) {
        if (result.status !== 'MAPPED_POLICY_PROBE_DONE') throw new Error(`unexpected mapped-policy status ${result.status}`);
        assertNumberAtMost(result, 'maxAbsError', runArgs.maxError);
        if (!result.nonzero || !result.nonuniform) throw new Error('mapped-policy output was zero or uniform');
      },
    },
    {
      kind: 'url',
      name: 'wgsl-heads-vs-ort-wasm-fixtures',
      doneText: 'WGSL_HEADS_VS_ORT_FIXTURES_DONE',
      params: { wgslHeadsVsOrt: 1, fixtureLimit: args.fixtureLimit, encoderLayers: 10, ep: 'wasm', packVerify: 0 },
      validate(result, runArgs) {
        if (result.status !== 'WGSL_HEADS_VS_ORT_FIXTURES_DONE') throw new Error(`unexpected WGSL heads fixture status ${result.status}`);
        if (result.bestMoveMatches !== result.fixtures) throw new Error(`WGSL heads best moves matched ${result.bestMoveMatches}/${result.fixtures}`);
        if (Number(result.maxMappedPolicyAbsDiff) > runArgs.maxError)
          throw new Error(`mapped policy diff ${result.maxMappedPolicyAbsDiff} exceeds ${runArgs.maxError}`);
        if (Number(result.maxWdlAbsDiff) > runArgs.maxError) throw new Error(`WDL diff ${result.maxWdlAbsDiff} exceeds ${runArgs.maxError}`);
      },
    },
    {
      kind: 'url',
      name: 'wgsl-heads-vs-ort-webgpu-fixtures',
      doneText: 'WGSL_HEADS_VS_ORT_FIXTURES_DONE',
      params: { wgslHeadsVsOrt: 1, fixtureLimit: args.fixtureLimit, encoderLayers: 10, ep: 'webgpu', strictWebGpu: 1, packVerify: 0 },
      validate(result, runArgs) {
        if (result.status !== 'WGSL_HEADS_VS_ORT_FIXTURES_DONE') throw new Error(`unexpected WGSL heads WebGPU fixture status ${result.status}`);
        if (result.bestMoveMatches !== result.fixtures) throw new Error(`WGSL heads WebGPU best moves matched ${result.bestMoveMatches}/${result.fixtures}`);
        if (Number(result.maxMappedPolicyAbsDiff) > runArgs.maxError)
          throw new Error(`WebGPU mapped policy diff ${result.maxMappedPolicyAbsDiff} exceeds ${runArgs.maxError}`);
        if (Number(result.maxWdlAbsDiff) > runArgs.maxError) throw new Error(`WebGPU WDL diff ${result.maxWdlAbsDiff} exceeds ${runArgs.maxError}`);
      },
    },
  ];
}

async function leakCheck(args, options = {}) {
  await runAgent(args, ['close', '--all'], 10_000, undefined).catch((error) =>
    process.stderr.write(`[ci-smoke] warning: close --all failed: ${error.message ?? error}\n`),
  );
  const cleanupPatterns = ['agent-browser/bin/agent-browser', 'Google Chrome for Testing'];
  if (options.checkVite !== false) cleanupPatterns.push(`vite .*${args.port}`);
  for (const pattern of cleanupPatterns) await spawnCapture('pkill', ['-f', pattern], { timeoutMs: 10_000 }).catch(() => undefined);
  await delay(1000);
  const stdout = await spawnCapture('ps', ['-axo', 'pid,rss,command'], { timeoutMs: 10_000 });
  const pattern =
    options.checkVite === false
      ? /Google Chrome for Testing|agent-browser|lc0_browser_hybrid|lc0-policy-only/
      : new RegExp(`Google Chrome for Testing|agent-browser|vite .*${args.port}|lc0_browser_hybrid|lc0-policy-only`);
  const leaks = stdout.split('\n').filter((line) => pattern.test(line) && !/lc0_browser_ci_smoke|npm run lc0:browser-ci-smoke/.test(line));
  if (leaks.length) throw new Error(`Browser/process leak check failed:\n${leaks.join('\n')}`);
  return { status: 'LC0_BROWSER_CI_SMOKE_LEAK_CHECK_CLEAN' };
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const plan = smokePlan(args);
  if (args.dryRun) {
    console.log(
      JSON.stringify({ baseUrl: args.baseUrl, smokes: plan.map((smoke) => ({ name: smoke.name, kind: smoke.kind, params: smoke.params })) }, null, 2),
    );
    return;
  }
  const server = startServer(args);
  const startedAt = new Date().toISOString();
  const rows = [];
  let runError;
  try {
    if (server) await server.ready;
    await waitForServer(args.baseUrl);
    for (const smoke of plan) {
      rows.push(smoke.kind === 'hybrid' ? await runHybridBenchSmoke(args, smoke.headBackend) : await runUrlSmoke(args, smoke));
    }
  } catch (error) {
    runError = error;
  } finally {
    server?.kill('SIGTERM');
    if (server) await delay(1000);
  }
  const leak = args.skipLeakCheck ? { status: 'LC0_BROWSER_CI_SMOKE_LEAK_CHECK_SKIPPED' } : await leakCheck(args, { checkVite: !args.noServer });
  if (runError) throw runError;
  const artifact = { status: 'LC0_BROWSER_CI_SMOKE_DONE', startedAt, finishedAt: new Date().toISOString(), baseUrl: args.baseUrl, rows, leak };
  if (args.out) {
    await mkdir(dirname(args.out), { recursive: true });
    await writeFile(args.out, JSON.stringify(artifact, null, 2));
  }
  console.log(JSON.stringify(artifact, null, 2));
}

main().catch((error) => {
  console.error(error.stack ?? error.message);
  process.exit(1);
});
