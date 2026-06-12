#!/usr/bin/env node
import { spawn } from 'node:child_process';
import { mkdir, writeFile } from 'node:fs/promises';
import { dirname } from 'node:path';
import { setTimeout as delay } from 'node:timers/promises';

const DEFAULT_HOST = '127.0.0.1';
const DEFAULT_PORT = 5203;
const DEFAULT_TIMEOUT_MS = 120_000;
const DEFAULT_AGENT_BROWSER = process.env.AGENT_BROWSER_BIN ?? 'agent-browser';

function usage() {
  console.log(`Usage: node scripts/maia3_browser_smoke.mjs [options]\n\nRuns a browser smoke for the standalone Maia3 evaluator. The page creates and disposes evaluator workers, checks edge-position legal masking, records top-5 human-policy moves and WDL probabilities, and fails on browser errors.\n\nOptions:\n  --base-url URL        Use an existing server instead of starting Vite\n  --host HOST           Vite host (default ${DEFAULT_HOST})\n  --port N              Vite port when auto-starting (default ${DEFAULT_PORT})\n  --agent-browser BIN   Browser automation binary (default AGENT_BROWSER_BIN or agent-browser)\n  --timeout MS          Browser wait timeout (default ${DEFAULT_TIMEOUT_MS})\n  --cycles N            Evaluator create/evaluate/dispose cycles (default 2)\n  --self-elo N          Maia3 self Elo (default 1500)\n  --oppo-elo N          Maia3 opponent Elo (default self Elo)\n  --style MODE          argmax or sample (default argmax)\n  --temperature X       Sampling temperature passed to the page (default 1)\n  --top-p X             Nucleus sampling top-p passed to the page (default 1)\n  --out PATH            Optional JSON artifact path\n  --no-server           Do not auto-start Vite\n  --dry-run             Print smoke URL and exit\n  -h, --help            Show this help\n`);
}

function parseArgs(argv) {
  const args = { host: DEFAULT_HOST, port: DEFAULT_PORT, agentBrowser: DEFAULT_AGENT_BROWSER, timeoutMs: DEFAULT_TIMEOUT_MS, noServer: false, explicitBaseUrl: false, dryRun: false, cycles: 2, selfElo: 1500, style: 'argmax', temperature: 1, topP: 1 };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    const next = () => {
      if (i + 1 >= argv.length) throw new Error(`${arg} requires a value`);
      return argv[++i];
    };
    if (arg === '--base-url') { args.baseUrl = next(); args.explicitBaseUrl = true; }
    else if (arg === '--host') args.host = next();
    else if (arg === '--port') args.port = Number(next());
    else if (arg === '--agent-browser') args.agentBrowser = next();
    else if (arg === '--timeout') args.timeoutMs = Number(next());
    else if (arg === '--cycles') args.cycles = Number(next());
    else if (arg === '--self-elo') args.selfElo = Number(next());
    else if (arg === '--oppo-elo') args.oppoElo = Number(next());
    else if (arg === '--style') args.style = next();
    else if (arg === '--temperature') args.temperature = Number(next());
    else if (arg === '--top-p') args.topP = Number(next());
    else if (arg === '--ort-ep') args.ortEp = next();
    else if (arg === '--grid-size') args.gridSize = Number(next());
    else if (arg === '--out') args.out = next();
    else if (arg === '--no-server') args.noServer = true;
    else if (arg === '--dry-run') args.dryRun = true;
    else if (arg === '-h' || arg === '--help') args.help = true;
    else throw new Error(`Unknown option: ${arg}`);
  }
  if (!args.baseUrl) args.baseUrl = `http://${args.host}:${args.port}`;
  if (args.explicitBaseUrl) args.noServer = true;
  if (!args.oppoElo) args.oppoElo = args.selfElo;
  for (const [name, value] of [['port', args.port], ['timeout', args.timeoutMs], ['cycles', args.cycles], ['self-elo', args.selfElo], ['oppo-elo', args.oppoElo]]) {
    if (!Number.isFinite(value) || value <= 0) throw new Error(`Invalid --${name}: ${value}`);
  }
  if (!['argmax', 'sample'].includes(args.style)) throw new Error(`Invalid --style: ${args.style}`);
  return args;
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
  let output = '';
  let readySettled = false;
  server.ready = new Promise((resolve, reject) => {
    const timer = setTimeout(() => settle(reject, new Error(`Vite dev server did not report readiness on port ${args.port}: ${output.trim()}`)), 30_000);
    const settle = (fn, value) => {
      if (readySettled) return;
      readySettled = true;
      clearTimeout(timer);
      fn(value);
    };
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
      const response = await fetch(new URL('/lc0-maia3-smoke.html', baseUrl), { cache: 'no-store' });
      if (response.ok) return;
      lastError = new Error(`HTTP ${response.status}`);
    } catch (error) {
      lastError = error;
    }
    await delay(250);
  }
  throw new Error(`Vite dev server did not become ready at ${baseUrl}: ${lastError?.message ?? 'timeout'}`);
}

async function runAgent(args, commandArgs, timeoutMs, session) {
  const fullArgs = ['--json', ...(session ? ['--session', session] : []), ...commandArgs];
  const { stdout } = await spawnCapture(args.agentBrowser, fullArgs, { echoStderr: true, timeoutMs });
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
    process.stderr.write(`[maia3-smoke] warning: failed to close ${session}: ${error.message ?? error}\n`);
  }
}

async function waitForText(args, session, selector, pattern, timeoutMs) {
  const deadline = Date.now() + timeoutMs;
  let lastText = '';
  while (Date.now() < deadline) {
    const payload = await runAgent(args, ['get', 'text', selector], Math.min(10_000, timeoutMs), session).catch((error) => ({ error: error.message ?? String(error) }));
    lastText = payload?.text ?? payload?.result ?? '';
    if (typeof lastText === 'string' && pattern.test(lastText)) return lastText;
    await delay(500);
  }
  throw new Error(`Timed out waiting for ${selector} to match ${pattern}; last text=${JSON.stringify(lastText).slice(0, 500)}`);
}

function normalizeEntries(payload) {
  if (Array.isArray(payload)) return payload;
  if (Array.isArray(payload?.entries)) return payload.entries;
  if (Array.isArray(payload?.messages)) return payload.messages;
  if (Array.isArray(payload?.logs)) return payload.logs;
  if (Array.isArray(payload?.result)) return payload.result;
  if (typeof payload?.text === 'string') return payload.text.split('\n').filter(Boolean).map((text) => ({ text }));
  return [];
}

function isActionable(entry) {
  const text = typeof entry === 'string' ? entry : JSON.stringify(entry);
  if (!text) return false;
  if (/\[vite\] connecting|\[vite\] connected|favicon\.ico/i.test(text)) return false;
  return /error|pageerror|uncaught|failed|exception/i.test(text);
}

async function browserErrors(args, session) {
  const errors = [];
  for (const command of [['errors'], ['console']]) {
    try {
      const payload = await runAgent(args, command, 10_000, session);
      errors.push(...normalizeEntries(payload).filter(isActionable));
    } catch (error) {
      errors.push({ error: `${command[0]} inspection failed: ${error.message ?? error}` });
    }
  }
  return errors;
}

function smokeUrl(args) {
  const url = new URL('/lc0-maia3-smoke.html', args.baseUrl);
  url.searchParams.set('cycles', String(args.cycles));
  url.searchParams.set('selfElo', String(args.selfElo));
  url.searchParams.set('oppoElo', String(args.oppoElo));
  url.searchParams.set('style', args.style);
  url.searchParams.set('temperature', String(args.temperature));
  url.searchParams.set('topP', String(args.topP));
  if (args.ortEp) url.searchParams.set('ortEp', args.ortEp);
  if (args.gridSize) url.searchParams.set('gridSize', String(args.gridSize));
  return url;
}

async function runSmoke(args) {
  const session = `maia3-smoke-${process.pid}`;
  const url = smokeUrl(args);
  process.stderr.write(`[maia3-smoke] ${url}\n`);
  try {
    await runAgent(args, ['open', String(url)], 30_000, session);
    await waitForText(args, session, '#status', /MAIA3_BROWSER_SMOKE_(DONE|FAILED)/, args.timeoutMs);
    const payload = await runAgent(args, ['get', 'text', '#benchResult'], 10_000, session);
    const result = JSON.parse(payload.text ?? payload.result ?? '{}');
    const errors = await browserErrors(args, session);
    result.browserErrors = errors;
    if (args.out) {
      await mkdir(dirname(args.out), { recursive: true });
      await writeFile(args.out, `${JSON.stringify(result, null, 2)}\n`);
    }
    if (!result.ok) throw new Error(`Maia3 browser smoke failed: ${(result.errors ?? []).join('; ')}`);
    if (errors.length) throw new Error(`Maia3 browser smoke had actionable browser errors: ${JSON.stringify(errors).slice(0, 1000)}`);
    return result;
  } finally {
    await closeSession(args, session);
  }
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) { usage(); return; }
  if (args.dryRun) { console.log(String(smokeUrl(args))); return; }
  const server = startServer(args);
  try {
    if (server) await server.ready;
    await waitForServer(args.baseUrl);
    const result = await runSmoke(args);
    console.log(JSON.stringify({ ok: true, rows: result.rows?.length ?? 0, elapsedMs: result.elapsedMs, out: args.out }, null, 2));
  } finally {
    if (server) server.kill('SIGTERM');
  }
}

main().catch((error) => {
  console.error(error instanceof Error ? error.stack ?? error.message : error);
  process.exitCode = 1;
});
