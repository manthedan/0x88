#!/usr/bin/env node
import { spawn } from 'node:child_process';
import { writeFile } from 'node:fs/promises';
import { setTimeout as delay } from 'node:timers/promises';

const DEFAULT_HOST = '127.0.0.1';
const DEFAULT_PORT = 5180;
const DEFAULT_TIMEOUT_MS = 10 * 60_000;
const DEFAULT_RUNTIMES = ['onnx', 'hybrid-ort-heads', 'hybrid-wgsl-heads'];

function usage() {
  console.log(`Usage: node --experimental-strip-types scripts/lc0_browser_runtime_arena_bench.mjs [options]\n\nRuns the browser arena as an e2e fixed-time benchmark: LC0 small is matched against a configurable opponent once for each LC0 runtime, then emits one JSON report with match results, diagnostics, search telemetry, engine output snapshots, logs, and PGN.\n\nOptions:\n  --base-url URL        Use an existing dev server (default http://${DEFAULT_HOST}:${DEFAULT_PORT})\n  --port N             Vite port when auto-starting (default ${DEFAULT_PORT})\n  --host HOST          Vite host when auto-starting (default ${DEFAULT_HOST})\n  --agent-browser BIN  Browser automation binary (default: AGENT_BROWSER_BIN or agent-browser)\n  --session NAME       agent-browser session prefix\n  --timeout MS         Per-runtime browser wait timeout (default ${DEFAULT_TIMEOUT_MS})\n  --runtimes LIST      Comma-separated runtimes (default ${DEFAULT_RUNTIMES.join(',')})\n  --movetime MS        Equal movetime per move (default 500)\n  --games N            Games per opening (default 2)\n  --delay MS           UI delay between plies (default 0)\n  --cache N            LC0 NN cache entries (default 2048)\n  --lc0-batch-size N   LC0 PUCT leaf batch size passed to arena search (default 1)\n  --batch-pipeline-depth N  LC0 batch pipeline depth (default 1; >1 is speculative search semantics)\n  --lc0-strength N     LC0 fixed-visit strength field, retained for labels when budget=movetime (default 100)\n  --opponent SPEC      Opponent as family:variant:strength (default sf:lite:8)\n  --sf-threads N       Stockfish threads (default 1)\n  --openings SUITE     start, built-in, or custom (default start)\n  --opening-text TEXT  Custom opening lines; implies --openings custom\n  --out PATH           Write full JSON report to PATH\n  --summary-only       Print only compact summary to stdout; pair with --out for full artifacts\n  --no-server          Do not auto-start Vite\n  --dry-run            Print URLs and exit\n  -h, --help           Show this help\n`);
}

function parseArgs(argv) {
  const args = {
    host: DEFAULT_HOST,
    port: DEFAULT_PORT,
    timeoutMs: DEFAULT_TIMEOUT_MS,
    agentBrowser: process.env.AGENT_BROWSER_BIN ?? 'agent-browser',
    session: process.env.AGENT_BROWSER_SESSION ?? `lc0-runtime-arena-${process.pid}`,
    runtimes: [...DEFAULT_RUNTIMES],
    movetime: 500,
    games: 2,
    delay: 0,
    cache: 2048,
    lc0BatchSize: 1,
    batchPipelineDepth: 1,
    lc0Strength: 100,
    opponent: 'sf:lite:8',
    sfThreads: 1,
    openings: 'start',
    openingText: '',
    out: '',
    summaryOnly: false,
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
    if (arg === '--base-url') { args.baseUrl = next(); args.explicitBaseUrl = true; }
    else if (arg === '--port') args.port = Number(next());
    else if (arg === '--host') args.host = next();
    else if (arg === '--agent-browser') args.agentBrowser = next();
    else if (arg === '--session') args.session = next();
    else if (arg === '--timeout') args.timeoutMs = Number(next());
    else if (arg === '--runtimes') args.runtimes = next().split(',').map((s) => s.trim()).filter(Boolean);
    else if (arg === '--movetime') args.movetime = Number(next());
    else if (arg === '--games' || arg === '--games-per-opening') args.games = Number(next());
    else if (arg === '--delay') args.delay = Number(next());
    else if (arg === '--cache' || arg === '--cache-entries') args.cache = Number(next());
    else if (arg === '--lc0-batch-size' || arg === '--batch-size' || arg === '--batch') args.lc0BatchSize = Number(next());
    else if (arg === '--batch-pipeline-depth' || arg === '--pipeline-depth') args.batchPipelineDepth = Number(next());
    else if (arg === '--lc0-strength') args.lc0Strength = Number(next());
    else if (arg === '--opponent') args.opponent = next();
    else if (arg === '--sf-threads') args.sfThreads = Number(next());
    else if (arg === '--openings') args.openings = next();
    else if (arg === '--opening-text') { args.openingText = next(); args.openings = 'custom'; }
    else if (arg === '--out') args.out = next();
    else if (arg === '--summary-only') args.summaryOnly = true;
    else if (arg === '--no-server') args.noServer = true;
    else if (arg === '--dry-run') args.dryRun = true;
    else if (arg === '-h' || arg === '--help') args.help = true;
    else throw new Error(`Unknown option: ${arg}`);
  }
  if (!args.baseUrl) args.baseUrl = `http://${args.host}:${args.port}`;
  if (args.explicitBaseUrl) args.noServer = true;
  const validRuntimes = new Set(DEFAULT_RUNTIMES);
  for (const runtime of args.runtimes) if (!validRuntimes.has(runtime)) throw new Error(`Invalid runtime: ${runtime}`);
  if (!['start', 'built-in', 'custom'].includes(args.openings)) throw new Error(`Invalid --openings: ${args.openings}`);
  for (const [name, value] of [['timeout', args.timeoutMs], ['movetime', args.movetime], ['games', args.games], ['delay', args.delay], ['cache', args.cache], ['lc0-batch-size', args.lc0BatchSize], ['batch-pipeline-depth', args.batchPipelineDepth], ['lc0-strength', args.lc0Strength], ['sf-threads', args.sfThreads]]) {
    if (!Number.isFinite(value) || value < 0 || (['timeout', 'movetime', 'games', 'lc0-batch-size', 'batch-pipeline-depth', 'lc0-strength', 'sf-threads'].includes(name) && value <= 0)) throw new Error(`Invalid --${name}: ${value}`);
  }
  if (args.batchPipelineDepth > 1) process.stderr.write('[lc0-runtime-arena] warning: batchPipelineDepth > 1 is speculative parallel search; depth=1 is the parity-preserving arena baseline.\n');
  return args;
}

function arenaUrl(args, runtime) {
  const url = new URL('/lc0-arena.html', args.baseUrl);
  url.searchParams.set('arenaBench', '1');
  url.searchParams.set('lc0Runtime', runtime);
  url.searchParams.set('seatA', `lc0:small:${args.lc0Strength}`);
  url.searchParams.set('seatB', args.opponent);
  url.searchParams.set('budgetMode', 'movetime');
  url.searchParams.set('movetimeMs', String(args.movetime));
  url.searchParams.set('gamesPerOpening', String(args.games));
  url.searchParams.set('delayMs', String(args.delay));
  url.searchParams.set('cacheEntries', String(args.cache));
  url.searchParams.set('lc0BatchSize', String(args.lc0BatchSize));
  url.searchParams.set('lc0BatchPipelineDepth', String(args.batchPipelineDepth));
  url.searchParams.set('sfThreads', String(args.sfThreads));
  url.searchParams.set('openingSuite', args.openings);
  if (args.openingText) url.searchParams.set('openingText', args.openingText);
  url.searchParams.set('packVerify', '0');
  return String(url);
}

function runAgent(args, session, commandArgs, timeoutMs = 30_000) {
  const fullArgs = ['--json', '--session', session, ...commandArgs];
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

async function closeAgentSession(args, session) {
  try {
    await runAgent(args, session, ['close'], 5_000);
  } catch (error) {
    process.stderr.write(`[lc0-runtime-arena] warning: failed to close ${session}: ${error.message ?? error}\n`);
  }
}

async function waitForServer(baseUrl, timeoutMs) {
  const deadline = Date.now() + timeoutMs;
  let lastError;
  while (Date.now() < deadline) {
    try {
      const response = await fetch(new URL('/lc0-arena.html', baseUrl), { cache: 'no-store' });
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
  server.stdout.on('data', (chunk) => process.stderr.write(`[vite] ${chunk}`));
  server.stderr.on('data', (chunk) => process.stderr.write(`[vite] ${chunk}`));
  return server;
}

async function runOne(args, runtime) {
  const session = `${args.session}-${runtime}`;
  const url = arenaUrl(args, runtime);
  process.stderr.write(`[lc0-runtime-arena] ${runtime}: ${url}\n`);
  try {
    await runAgent(args, session, ['open', url], 30_000);
    const deadline = Date.now() + args.timeoutMs;
    let lastWaitError;
    while (Date.now() < deadline) {
      const chunk = Math.min(25_000, Math.max(1000, deadline - Date.now()));
      try {
        await runAgent(args, session, ['wait', '--text', 'ARENA_BENCH_DONE', '--timeout', String(chunk)], chunk + 5_000);
        lastWaitError = undefined;
        break;
      } catch (error) {
        lastWaitError = error;
        if (Date.now() >= deadline) throw error;
        await delay(250);
      }
    }
    if (lastWaitError) throw lastWaitError;
    const text = textFromGetResult(await runAgent(args, session, ['get', 'text', '#benchResult'], 30_000));
    const result = JSON.parse(text);
    if (result.status !== 'ARENA_BENCH_DONE') throw new Error(`unexpected benchmark status for ${runtime}: ${result.status}`);
    if (result.runtime !== runtime) throw new Error(`runtime mismatch: expected ${runtime}, got ${result.runtime}`);
    return result;
  } finally {
    await closeAgentSession(args, session);
  }
}

function scoreRateFromMatchScore(text) {
  const match = String(text).match(/\s(\d+|\d+½|½)\s+–\s+(\d+|\d+½|½)\s/);
  if (!match) return null;
  const half = (s) => s === '½' ? 0.5 : s.endsWith('½') ? Number(s.slice(0, -1) || '0') + 0.5 : Number(s);
  const a = half(match[1]);
  const b = half(match[2]);
  return (a + b) > 0 ? a / (a + b) : null;
}

function compactRuntime(result) {
  return {
    runtime: result.runtime,
    elapsedMs: result.elapsedMs,
    matchScore: result.summary?.matchScore,
    lc0ScoreRate: scoreRateFromMatchScore(result.summary?.matchScore),
    runtimeDiagnostics: result.summary?.runtimeDiagnostics,
    searchDiagnostics: result.summary?.searchDiagnostics,
    lc0BatchSize: result.configuration?.lc0BatchSize,
    lc0BatchPipelineDepth: result.configuration?.lc0BatchPipelineDepth,
    lc0Tree: result.telemetry?.lc0Tree,
    uci: result.telemetry?.uci,
    engineOutputCount: result.engineOutputCount,
  };
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) return usage();
  if (args.dryRun) {
    for (const runtime of args.runtimes) console.log(arenaUrl(args, runtime));
    return;
  }
  const server = startServer(args);
  try {
    await waitForServer(args.baseUrl, 30_000);
    const results = [];
    for (const runtime of args.runtimes) results.push(await runOne(args, runtime));
    const report = {
      status: 'LC0_RUNTIME_ARENA_BENCH_DONE',
      startedAt: new Date().toISOString(),
      baseUrl: args.baseUrl,
      config: {
        runtimes: args.runtimes,
        opponent: args.opponent,
        movetimeMs: args.movetime,
        gamesPerOpening: args.games,
        openings: args.openings,
        cacheEntries: args.cache,
        lc0BatchSize: args.lc0BatchSize,
        batchPipelineDepth: args.batchPipelineDepth,
        sfThreads: args.sfThreads,
      },
      summary: results.map(compactRuntime),
      results,
    };
    const fullJson = JSON.stringify(report, null, 2);
    if (args.out) await writeFile(args.out, `${fullJson}\n`);
    console.log(args.summaryOnly ? JSON.stringify({ ...report, results: undefined }, null, 2) : fullJson);
  } finally {
    server?.kill('SIGTERM');
  }
}

main().catch((error) => {
  console.error(error.stack ?? error.message);
  process.exit(1);
});
