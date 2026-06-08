#!/usr/bin/env node
import { spawn } from 'node:child_process';
import { mkdir, writeFile } from 'node:fs/promises';
import { dirname } from 'node:path';
import { setTimeout as delay } from 'node:timers/promises';

const DEFAULT_HOST = '127.0.0.1';
const DEFAULT_PORT = 5196;
const DEFAULT_TIMEOUT_MS = 30_000;
const DEFAULT_AGENT_BROWSER = process.env.AGENT_BROWSER_BIN ?? 'agent-browser';
const REQUIRED_FAMILIES = ['lc0', 'tiny', 'sf', 'reckless', 'viridithas', 'berserk', 'plentychess'];
const REQUIRED_RUNTIME_TOKENS = ['Lc0 BT4:', 'Tiny:', 'Reckless:', 'Viridithas:', 'Berserk:', 'PlentyChess:'];

function usage() {
  console.log(`Usage: node scripts/lc0_analysis_browser_smoke.mjs [options]\n\nRuns a fast browser smoke for lc0-analysis.html UI wiring. It verifies the multi-engine selector, profile controls, local PGN database controls, runtime status text, and actionable browser console errors.\n\nOptions:\n  --base-url URL        Use an existing server instead of starting Vite\n  --host HOST           Vite host (default ${DEFAULT_HOST})\n  --port N              Vite port when auto-starting (default ${DEFAULT_PORT})\n  --agent-browser BIN   Browser automation binary (default AGENT_BROWSER_BIN or agent-browser)\n  --timeout MS          Browser wait timeout (default ${DEFAULT_TIMEOUT_MS})\n  --out PATH            Optional JSON artifact path\n  --no-server           Do not auto-start Vite\n  --dry-run             Print planned smoke URL without running\n  -h, --help            Show this help\n`);
}

function parseArgs(argv) {
  const args = { host: DEFAULT_HOST, port: DEFAULT_PORT, agentBrowser: DEFAULT_AGENT_BROWSER, timeoutMs: DEFAULT_TIMEOUT_MS, noServer: false, explicitBaseUrl: false, dryRun: false };
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
    else if (arg === '--out') args.out = next();
    else if (arg === '--no-server') args.noServer = true;
    else if (arg === '--dry-run') args.dryRun = true;
    else if (arg === '-h' || arg === '--help') args.help = true;
    else throw new Error(`Unknown option: ${arg}`);
  }
  if (!args.baseUrl) args.baseUrl = `http://${args.host}:${args.port}`;
  if (args.explicitBaseUrl) args.noServer = true;
  for (const [name, value] of [['port', args.port], ['timeout', args.timeoutMs]]) {
    if (!Number.isFinite(value) || value <= 0) throw new Error(`Invalid --${name}: ${value}`);
  }
  return args;
}

function spawnCapture(command, commandArgs, options = {}) {
  return new Promise((resolve, reject) => {
    const { timeoutMs, echoStderr, ...spawnOptions } = options;
    const child = spawn(command, commandArgs, { stdio: ['ignore', 'pipe', 'pipe'], ...spawnOptions });
    const chunks = { stdout: [], stderr: [] };
    let settled = false;
    const timer = timeoutMs ? setTimeout(() => {
      child.kill('SIGKILL');
      finish(reject, new Error(`${command} ${commandArgs.join(' ')} timed out after ${timeoutMs}ms`));
    }, timeoutMs) : undefined;
    const finish = (fn, value) => {
      if (settled) return;
      settled = true;
      if (timer) clearTimeout(timer);
      fn(value);
    };
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
      const response = await fetch(new URL('/lc0-analysis.html', baseUrl), { cache: 'no-store' });
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
    process.stderr.write(`[analysis-smoke] warning: failed to close ${session}: ${error.message ?? error}\n`);
  }
}

async function waitForEval(args, session, expression, validate, timeoutMs) {
  const deadline = Date.now() + timeoutMs;
  let lastValue;
  let lastError;
  while (Date.now() < deadline) {
    try {
      const data = await runAgent(args, ['eval', expression], Math.min(10_000, timeoutMs), session);
      lastValue = data?.result ?? data;
      if (validate(lastValue)) return lastValue;
    } catch (error) {
      lastError = error;
    }
    await delay(250);
  }
  throw new Error(`Timed out waiting for browser condition. Last value=${JSON.stringify(lastValue)} Last error=${lastError?.message ?? 'none'}`);
}

function normalizeConsoleEntries(payload) {
  if (Array.isArray(payload)) return payload;
  if (Array.isArray(payload?.entries)) return payload.entries;
  if (Array.isArray(payload?.messages)) return payload.messages;
  if (Array.isArray(payload?.logs)) return payload.logs;
  if (Array.isArray(payload?.result)) return payload.result;
  if (typeof payload?.text === 'string') return payload.text.split('\n').filter(Boolean).map((text) => ({ text }));
  return [];
}

function isActionableConsoleEntry(entry) {
  const text = typeof entry === 'string' ? entry : JSON.stringify(entry);
  if (!text) return false;
  if (/\[vite\] connecting|\[vite\] connected/i.test(text)) return false;
  if (/favicon\.ico/i.test(text)) return false;
  return /error|pageerror|uncaught|failed|exception/i.test(text);
}

async function consoleErrors(args, session) {
  try {
    const payload = await runAgent(args, ['console', '--errors'], 10_000, session);
    return normalizeConsoleEntries(payload).filter(isActionableConsoleEntry);
  } catch (error) {
    return [{ error: `console inspection failed: ${error.message ?? error}` }];
  }
}

async function runSmoke(args) {
  const session = `lc0-analysis-smoke-${process.pid}`;
  const url = new URL('/lc0-analysis.html', args.baseUrl);
  process.stderr.write(`[analysis-smoke] ${url}\n`);
  try {
    await runAgent(args, ['open', String(url)], 30_000, session);
    const snapshot = await waitForEval(args, session, `(() => {
      const optionValues = (selector) => [...document.querySelectorAll(selector)].map((option) => option.value);
      const text = (selector) => document.querySelector(selector)?.textContent ?? '';
      return {
        title: document.title,
        families: optionValues('.row-fam option'),
        hasProfileSelect: !!document.querySelector('#engineProfileSelect'),
        hasProfileName: !!document.querySelector('#engineProfileName'),
        hasSaveProfile: !!document.querySelector('#saveEngineProfile'),
        hasDeleteProfile: !!document.querySelector('#deleteEngineProfile'),
        hasPgnDbSelect: !!document.querySelector('#pgnDbSelect'),
        hasPgnDbName: !!document.querySelector('#pgnDbName'),
        hasSavePgnDb: !!document.querySelector('#savePgnDb'),
        hasLoadPgnDb: !!document.querySelector('#loadPgnDb'),
        hasSearchPgnDbPosition: !!document.querySelector('#searchPgnDbPosition'),
        hasEngineCompare: !!document.querySelector('#engineCompare tbody'),
        runtimeText: text('#recklessRuntimeInfo'),
        pgnDbInfo: text('#pgnDbInfo'),
      };
    })()`, (value) => {
      if (!value || typeof value !== 'object') return false;
      if (!REQUIRED_FAMILIES.every((family) => value.families?.includes(family))) return false;
      if (!value.hasProfileSelect || !value.hasProfileName || !value.hasSaveProfile || !value.hasDeleteProfile) return false;
      if (!value.hasPgnDbSelect || !value.hasPgnDbName || !value.hasSavePgnDb || !value.hasLoadPgnDb || !value.hasSearchPgnDbPosition) return false;
      if (!value.hasEngineCompare) return false;
      if (value.runtimeText === 'Reckless: detecting runtime…') return false;
      return REQUIRED_RUNTIME_TOKENS.every((token) => value.runtimeText?.includes(token));
    }, args.timeoutMs);
    const missingFamilies = REQUIRED_FAMILIES.filter((family) => !snapshot.families.includes(family));
    if (missingFamilies.length) throw new Error(`Missing engine families: ${missingFamilies.join(', ')}`);
    const missingRuntimeTokens = REQUIRED_RUNTIME_TOKENS.filter((token) => !snapshot.runtimeText.includes(token));
    if (missingRuntimeTokens.length) throw new Error(`Missing runtime status tokens: ${missingRuntimeTokens.join(', ')}`);
    const errors = await consoleErrors(args, session);
    if (errors.length) throw new Error(`Actionable console errors found: ${JSON.stringify(errors, null, 2)}`);
    return { status: 'LC0_ANALYSIS_BROWSER_SMOKE_DONE', url: String(url), snapshot, consoleErrors: errors };
  } finally {
    await closeSession(args, session);
  }
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) return usage();
  const smokeUrl = new URL('/lc0-analysis.html', args.baseUrl);
  if (args.dryRun) {
    console.log(JSON.stringify({ status: 'LC0_ANALYSIS_BROWSER_SMOKE_DRY_RUN', url: String(smokeUrl) }, null, 2));
    return;
  }
  const server = startServer(args);
  const startedAt = new Date().toISOString();
  let result;
  let runError;
  try {
    if (server) await server.ready;
    await waitForServer(args.baseUrl);
    result = await runSmoke(args);
  } catch (error) {
    runError = error;
  } finally {
    server?.kill('SIGTERM');
    if (server) await delay(1000);
  }
  if (runError) throw runError;
  const artifact = { ...result, startedAt, finishedAt: new Date().toISOString() };
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
