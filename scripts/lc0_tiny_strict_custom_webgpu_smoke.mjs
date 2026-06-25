#!/usr/bin/env node
import { spawn } from 'node:child_process';
import { mkdir, writeFile } from 'node:fs/promises';
import { dirname } from 'node:path';
import { setTimeout as delay } from 'node:timers/promises';

const DEFAULT_HOST = '127.0.0.1';
const DEFAULT_PORT = 5281;
const DEFAULT_TIMEOUT_MS = 240_000;
const DEFAULT_AGENT_BROWSER = process.env.AGENT_BROWSER_BIN ?? 'agent-browser';
const AUDIT_EVENT = 'lc0-browser-runtime-audit';

function usage() {
  console.log(`Usage: node scripts/lc0_tiny_strict_custom_webgpu_smoke.mjs [options]\n\nRuns strict Centipawn custom WebGPU browser smokes for /app/analysis and /app/arena. The gate fails if the Centipawn runtime audit does not resolve runtime=custom-webgpu without fallback.\n\nOptions:\n  --base-url URL        Use an existing server instead of starting Vite\n  --host HOST           Vite host (default ${DEFAULT_HOST})\n  --port N              Vite port (default ${DEFAULT_PORT})\n  --agent-browser BIN   Browser automation binary (default AGENT_BROWSER_BIN or agent-browser)\n  --timeout MS          Per-surface timeout (default ${DEFAULT_TIMEOUT_MS})\n  --out PATH            Optional JSON artifact path\n  --no-server           Do not auto-start Vite\n  --skip-analysis       Skip /app/analysis smoke\n  --skip-arena          Skip /app/arena smoke\n  --dry-run             Print planned URLs without running\n  -h, --help            Show this help\n`);
}

function parseArgs(argv) {
  const args = { host: DEFAULT_HOST, port: DEFAULT_PORT, agentBrowser: DEFAULT_AGENT_BROWSER, timeoutMs: DEFAULT_TIMEOUT_MS, noServer: false, skipAnalysis: false, skipArena: false, dryRun: false, explicitBaseUrl: false };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    const next = () => { if (i + 1 >= argv.length) throw new Error(`${arg} requires a value`); return argv[++i]; };
    if (arg === '--base-url') { args.baseUrl = next(); args.explicitBaseUrl = true; }
    else if (arg === '--host') args.host = next();
    else if (arg === '--port') args.port = Number(next());
    else if (arg === '--agent-browser') args.agentBrowser = next();
    else if (arg === '--timeout') args.timeoutMs = Number(next());
    else if (arg === '--out') args.out = next();
    else if (arg === '--no-server') args.noServer = true;
    else if (arg === '--skip-analysis') args.skipAnalysis = true;
    else if (arg === '--skip-arena') args.skipArena = true;
    else if (arg === '--dry-run') args.dryRun = true;
    else if (arg === '-h' || arg === '--help') args.help = true;
    else throw new Error(`Unknown option: ${arg}`);
  }
  if (!args.baseUrl) args.baseUrl = `http://${args.host}:${args.port}`;
  if (args.explicitBaseUrl) args.noServer = true;
  if (!Number.isFinite(args.port) || args.port <= 0) throw new Error(`Invalid --port: ${args.port}`);
  if (!Number.isFinite(args.timeoutMs) || args.timeoutMs <= 0) throw new Error(`Invalid --timeout: ${args.timeoutMs}`);
  if (args.skipAnalysis && args.skipArena) throw new Error('Nothing to run: both --skip-analysis and --skip-arena were set');
  return args;
}

function spawnCapture(command, commandArgs, options = {}) {
  return new Promise((resolve, reject) => {
    const { timeoutMs, echoStderr, stdin, ...spawnOptions } = options;
    const child = spawn(command, commandArgs, { stdio: ['pipe', 'pipe', 'pipe'], ...spawnOptions });
    const chunks = { stdout: [], stderr: [] };
    let settled = false;
    const finish = (fn, value) => { if (settled) return; settled = true; if (timer) clearTimeout(timer); fn(value); };
    const timer = timeoutMs ? setTimeout(() => { child.kill('SIGKILL'); finish(reject, new Error(`${command} ${commandArgs.join(' ')} timed out after ${timeoutMs}ms`)); }, timeoutMs) : undefined;
    child.stdout.on('data', (chunk) => chunks.stdout.push(chunk));
    child.stderr.on('data', (chunk) => { chunks.stderr.push(chunk); if (echoStderr) process.stderr.write(chunk); });
    child.on('error', (error) => finish(reject, error));
    child.on('close', (status) => {
      const stdout = Buffer.concat(chunks.stdout).toString('utf8');
      const stderr = Buffer.concat(chunks.stderr).toString('utf8');
      if (status !== 0) return finish(reject, new Error(`${command} ${commandArgs.join(' ')} failed with ${status}: ${stderr || stdout}`));
      finish(resolve, { stdout, stderr });
    });
    if (stdin !== undefined) child.stdin.end(stdin);
    else child.stdin.end();
  });
}

function startServer(args) {
  if (args.noServer) return null;
  const server = spawn('npm', ['run', 'web:client', '--', '--host', args.host, '--port', String(args.port), '--strictPort'], { stdio: ['ignore', 'pipe', 'pipe'] });
  let output = '';
  let settled = false;
  server.ready = new Promise((resolve, reject) => {
    const timer = setTimeout(() => settle(reject, new Error(`Vite did not become ready on ${args.port}: ${output.trim()}`)), 30_000);
    const settle = (fn, value) => { if (settled) return; settled = true; clearTimeout(timer); fn(value); };
    const onOutput = (chunk) => { output += chunk.toString('utf8'); if (/ready in \d+\s*ms/.test(output) || output.includes(`:${args.port}/`)) settle(resolve); };
    server.stdout.on('data', (chunk) => { process.stderr.write(`[vite] ${chunk}`); onOutput(chunk); });
    server.stderr.on('data', (chunk) => { process.stderr.write(`[vite] ${chunk}`); onOutput(chunk); });
    server.on('exit', (status, signal) => settle(reject, new Error(`Vite exited before ready (${status ?? signal}): ${output.trim()}`)));
  });
  return server;
}

async function waitForServer(baseUrl, timeoutMs = 30_000) {
  const deadline = Date.now() + timeoutMs;
  let lastError;
  while (Date.now() < deadline) {
    try {
      const response = await fetch(new URL('/app/analysis', baseUrl), { cache: 'no-store' });
      if (response.ok) return;
      lastError = new Error(`HTTP ${response.status}`);
    } catch (error) { lastError = error; }
    await delay(250);
  }
  throw new Error(`server did not become ready at ${baseUrl}: ${lastError?.message ?? 'timeout'}`);
}

function parseAgentJson(stdout) {
  const parsed = JSON.parse(stdout.trim());
  if (parsed && typeof parsed === 'object' && 'success' in parsed) {
    if (parsed.success === false) throw new Error(parsed.error ?? stdout);
    return parsed.data ?? parsed;
  }
  return parsed;
}

async function runAgent(args, commandArgs, timeoutMs, session, stdin) {
  const fullArgs = ['--json', ...(session ? ['--session', session] : []), ...commandArgs];
  const { stdout } = await spawnCapture(args.agentBrowser, fullArgs, { echoStderr: true, timeoutMs, stdin });
  return parseAgentJson(stdout);
}

async function evalPage(args, session, expression, timeoutMs = 30_000) {
  const payload = await runAgent(args, ['eval', '--stdin'], timeoutMs, session, expression);
  if (payload && typeof payload === 'object') {
    if ('value' in payload) return payload.value;
    if ('result' in payload) return payload.result;
  }
  return payload;
}

async function closeSession(args, session) {
  try { await runAgent(args, ['close'], 5_000, session); } catch { /* best-effort; no process pkill */ }
}

async function pollUntil(args, session, label, expression, validate, timeoutMs) {
  const deadline = Date.now() + timeoutMs;
  let last;
  while (Date.now() < deadline) {
    last = await evalPage(args, session, expression, 30_000);
    const done = validate(last);
    if (done) return last;
    await delay(1000);
  }
  throw new Error(`${label} timed out after ${timeoutMs}ms; last=${JSON.stringify(last)}`);
}

const installAuditCollector = `(() => {
  window.__lc0BrowserRuntimeAuditEvents = [];
  if (!window.__lc0BrowserRuntimeAuditCollectorInstalled) {
    window.addEventListener('${AUDIT_EVENT}', (event) => window.__lc0BrowserRuntimeAuditEvents.push(event.detail));
    window.__lc0BrowserRuntimeAuditCollectorInstalled = true;
  }
  return true;
})()`;

function strictCustomAudit(events) {
  return (Array.isArray(events) ? events : []).filter((event) => event?.family === 'centipawn' && event?.requestedRuntime === 'custom-webgpu');
}

function validateStrictCustomAudit(label, status) {
  if (String(status?.message ?? '').startsWith('Analysis failed') || String(status?.message ?? '').startsWith('Match failed')) {
    throw new Error(`${label} failed in browser: ${status.message}`);
  }
  const customEvents = strictCustomAudit(status?.auditEvents);
  const resolved = customEvents.find((event) => event.resolvedRuntime === 'custom-webgpu' && !event.fallbackReason);
  if (!resolved) return false;
  const fallback = customEvents.find((event) => event.resolvedRuntime !== 'custom-webgpu' || event.fallbackReason);
  if (fallback) throw new Error(`${label} observed strict custom fallback: ${JSON.stringify(fallback)}`);
  return true;
}

async function runAnalysisSmoke(args) {
  const session = `lc0-tiny-strict-${process.pid}-analysis`;
  const url = new URL('/app/analysis', args.baseUrl);
  url.searchParams.set('ep', 'wasm');
  url.searchParams.set('tinyBatch', '1');
  process.stderr.write(`[lc0-tiny-strict] analysis ${url}\n`);
  try {
    await runAgent(args, ['open', String(url)], 30_000, session);
    await runAgent(args, ['wait', '--text', 'Centipawn:', '--timeout', '60000'], 65_000, session);
    await evalPage(args, session, installAuditCollector);
    await evalPage(args, session, `(() => {
      const setValue = (node, value) => { node.value = value; node.dispatchEvent(new Event('change', { bubbles: true })); };
      const fam = document.querySelector('#engineList .row-fam');
      if (!fam) throw new Error('missing analysis engine family select');
      setValue(fam, 'centipawn');
      setTimeout(() => {
        const variant = document.querySelector('#engineList .row-var');
        const strength = document.querySelector('#engineList .row-strength');
        const lines = document.querySelector('#multiPvInput');
        if (variant) setValue(variant, 'bt4-custom');
        if (strength) setValue(strength, '1');
        if (lines) setValue(lines, '1');
        document.querySelector('#analyze')?.click();
      }, 100);
      return true;
    })()`);
    const status = await pollUntil(args, session, 'analysis strict custom WebGPU', `(() => ({
      message: document.querySelector('#message')?.textContent ?? '',
      lines: document.querySelector('#lines')?.textContent ?? '',
      runtimeInfo: document.querySelector('#recklessRuntimeInfo')?.textContent ?? '',
      lc0Audit: document.querySelector('#runtimeAudit')?.textContent ?? '',
      auditEvents: window.__lc0BrowserRuntimeAuditEvents ?? []
    }))()`, (value) => validateStrictCustomAudit('analysis', value) && String(value.lines ?? '').includes('Centipawn · custom WebGPU'), args.timeoutMs);
    return { name: 'analysis', status: 'LC0_TINY_ANALYSIS_STRICT_CUSTOM_WEBGPU_DONE', url: String(url), auditEvents: strictCustomAudit(status.auditEvents), message: status.message, lines: status.lines, runtimeInfo: status.runtimeInfo, lc0Audit: status.lc0Audit };
  } finally {
    await closeSession(args, session);
  }
}

async function runArenaSmoke(args) {
  const session = `lc0-tiny-strict-${process.pid}-arena`;
  const url = new URL('/app/arena', args.baseUrl);
  Object.entries({ seatA: 'centipawn:bt4-custom:1', seatB: 'centipawn:bt4-custom:1', games: 1, delayMs: 0, cacheEntries: 64 }).forEach(([k, v]) => url.searchParams.set(k, String(v)));
  process.stderr.write(`[lc0-tiny-strict] arena ${url}\n`);
  try {
    await runAgent(args, ['open', String(url)], 30_000, session);
    await runAgent(args, ['wait', '--text', 'Centipawn hybrid bundle', '--timeout', '90000'], 95_000, session);
    await evalPage(args, session, installAuditCollector);
    await evalPage(args, session, `(() => { document.querySelector('#start')?.click(); return true; })()`);
    const status = await pollUntil(args, session, 'arena strict custom WebGPU', `(() => ({
      message: document.querySelector('#message')?.textContent ?? '',
      pairing: document.querySelector('#pairing')?.textContent ?? '',
      matchScore: document.querySelector('#matchScore')?.textContent ?? '',
      runtimeBadge: document.querySelector('#runtimeBadge')?.textContent ?? '',
      runtimeInfo: document.querySelector('#runtimeAuditInfo')?.textContent ?? '',
      log: document.querySelector('#log')?.textContent ?? '',
      auditEvents: window.__lc0BrowserRuntimeAuditEvents ?? []
    }))()`, (value) => validateStrictCustomAudit('arena', value) && String(value.message ?? '').startsWith('Match done'), args.timeoutMs);
    return { name: 'arena', status: 'LC0_TINY_ARENA_STRICT_CUSTOM_WEBGPU_DONE', url: String(url), auditEvents: strictCustomAudit(status.auditEvents), message: status.message, pairing: status.pairing, matchScore: status.matchScore, runtimeBadge: status.runtimeBadge, runtimeInfo: status.runtimeInfo };
  } finally {
    await closeSession(args, session);
  }
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) { usage(); return; }
  const plan = [];
  if (!args.skipAnalysis) plan.push({ name: 'analysis', url: `${args.baseUrl}/app/analysis?tinyBatch=1` });
  if (!args.skipArena) plan.push({ name: 'arena', url: `${args.baseUrl}/app/arena?seatA=centipawn:bt4-custom:1&seatB=centipawn:bt4-custom:1&games=1` });
  if (args.dryRun) { console.log(JSON.stringify({ baseUrl: args.baseUrl, plan }, null, 2)); return; }
  const server = startServer(args);
  const rows = [];
  try {
    await server?.ready;
    await waitForServer(args.baseUrl);
    if (!args.skipAnalysis) rows.push(await runAnalysisSmoke(args));
    if (!args.skipArena) rows.push(await runArenaSmoke(args));
    const summary = { schema: 'leelaweb.tiny-strict-custom-webgpu-smoke.v1', createdAt: new Date().toISOString(), baseUrl: args.baseUrl, gatePassed: true, rows };
    if (args.out) { await mkdir(dirname(args.out), { recursive: true }); await writeFile(args.out, `${JSON.stringify(summary, null, 2)}\n`); }
    console.log(JSON.stringify(summary, null, 2));
  } finally {
    server?.kill('SIGTERM');
  }
}

main().catch((error) => { console.error(error.stack ?? error.message ?? error); process.exit(1); });
