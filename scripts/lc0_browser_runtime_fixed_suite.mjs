#!/usr/bin/env node
import { spawn } from 'node:child_process';
import { readFile, writeFile } from 'node:fs/promises';
import { setTimeout as delay } from 'node:timers/promises';
import { parsePgnGames } from '../src/chess/pgn.ts';

const DEFAULT_HOST = '127.0.0.1';
const DEFAULT_PORT = 5180;
const DEFAULT_TIMEOUT_MS = 10 * 60_000;
const DEFAULT_RUNTIMES = ['onnx', 'hybrid-ort-heads', 'hybrid-wgsl-heads'];

function usage() {
  console.log(`Usage: node --experimental-strip-types scripts/lc0_browser_runtime_fixed_suite.mjs [options]\n\nRuns the same fixed LC0-to-move positions in the browser for each LC0 runtime, then scores each LC0 move with Stockfish on the resulting position.\n\nOptions:\n  --source-report PATH      Existing runtime arena JSON to derive positions from\n  --source-runtime NAME     Runtime in source report (default hybrid-wgsl-heads)\n  --source-game N           1-based game number in that runtime PGN (default 2)\n  --max-positions N         Max LC0-to-move positions to extract (default 16)\n  --skip-plies N            Ignore positions before this absolute ply (default 0)\n  --fens FILE               Use newline-separated FENs instead of --source-report\n  --runtimes LIST           Comma-separated runtimes (default ${DEFAULT_RUNTIMES.join(',')})\n  --movetime MS             LC0 movetime per fixed position (default 1000)\n  --stockfish-score-ms MS   Stockfish movetime to score each post-LC0 position (default 500)\n  --stockfish-score-depth N Use fixed Stockfish depth for scoring instead of movetime\n  --cache N                 LC0 NN cache entries (default 2048)\n  --lc0-batch-size N        LC0 PUCT leaf batch size passed to arena search (default 1)\n  --batch-pipeline-depth N  LC0 batch pipeline depth (default 1; >1 is speculative search semantics)\n  --input-backend NAME      Hybrid input backend: js, wgsl, or wasm (default js)\n  --encoder-kernel NAME     Hybrid encoder kernel: hand, tvm-packed-f16, mixed-tvm-ffn, or mixed-tvm-ffn-outproj (default hand)
  --out PATH                Write full JSON report to PATH\n  --summary-only            Print compact summary only\n  --base-url URL            Use existing dev server (default http://${DEFAULT_HOST}:${DEFAULT_PORT})\n  --port N                  Vite port when auto-starting (default ${DEFAULT_PORT})\n  --host HOST               Vite host when auto-starting (default ${DEFAULT_HOST})\n  --agent-browser BIN       Browser automation binary (default agent-browser)\n  --session NAME            agent-browser session prefix\n  --timeout MS              Per-runtime browser wait timeout (default ${DEFAULT_TIMEOUT_MS})\n  --no-server               Do not auto-start Vite\n  -h, --help                Show this help\n`);
}

function parseArgs(argv) {
  const args = {
    host: DEFAULT_HOST,
    port: DEFAULT_PORT,
    timeoutMs: DEFAULT_TIMEOUT_MS,
    agentBrowser: process.env.AGENT_BROWSER_BIN ?? 'agent-browser',
    session: process.env.AGENT_BROWSER_SESSION ?? `lc0-fixed-suite-${process.pid}`,
    runtimes: [...DEFAULT_RUNTIMES],
    sourceReport: '',
    sourceRuntime: 'hybrid-wgsl-heads',
    sourceGame: 2,
    maxPositions: 16,
    skipPlies: 0,
    fensFile: '',
    movetime: 1000,
    stockfishScoreMs: 500,
    stockfishScoreDepth: undefined,
    cache: 2048,
    lc0BatchSize: 1,
    batchPipelineDepth: 1,
    inputBackend: 'js',
    encoderKernel: 'hand',
    out: '',
    summaryOnly: false,
    noServer: false,
    explicitBaseUrl: false,
  };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    const next = () => {
      if (i + 1 >= argv.length) throw new Error(`${arg} requires a value`);
      return argv[++i];
    };
    if (arg === '--source-report') args.sourceReport = next();
    else if (arg === '--source-runtime') args.sourceRuntime = next();
    else if (arg === '--source-game') args.sourceGame = Number(next());
    else if (arg === '--max-positions') args.maxPositions = Number(next());
    else if (arg === '--skip-plies') args.skipPlies = Number(next());
    else if (arg === '--fens') args.fensFile = next();
    else if (arg === '--runtimes') args.runtimes = next().split(',').map((s) => s.trim()).filter(Boolean);
    else if (arg === '--movetime') args.movetime = Number(next());
    else if (arg === '--stockfish-score-ms') args.stockfishScoreMs = Number(next());
    else if (arg === '--stockfish-score-depth') args.stockfishScoreDepth = Number(next());
    else if (arg === '--cache') args.cache = Number(next());
    else if (arg === '--lc0-batch-size' || arg === '--batch-size' || arg === '--batch') args.lc0BatchSize = Number(next());
    else if (arg === '--batch-pipeline-depth' || arg === '--pipeline-depth') args.batchPipelineDepth = Number(next());
    else if (arg === '--input-backend') args.inputBackend = next();
    else if (arg === '--encoder-kernel' || arg === '--encoder-kernel-variant') args.encoderKernel = next();
    else if (arg === '--out') args.out = next();
    else if (arg === '--summary-only') args.summaryOnly = true;
    else if (arg === '--base-url') { args.baseUrl = next(); args.explicitBaseUrl = true; }
    else if (arg === '--port') args.port = Number(next());
    else if (arg === '--host') args.host = next();
    else if (arg === '--agent-browser') args.agentBrowser = next();
    else if (arg === '--session') args.session = next();
    else if (arg === '--timeout') args.timeoutMs = Number(next());
    else if (arg === '--no-server') args.noServer = true;
    else if (arg === '-h' || arg === '--help') args.help = true;
    else throw new Error(`Unknown option: ${arg}`);
  }
  if (!args.baseUrl) args.baseUrl = `http://${args.host}:${args.port}`;
  if (args.explicitBaseUrl) args.noServer = true;
  for (const name of ['sourceGame', 'maxPositions', 'movetime', 'stockfishScoreMs', 'cache', 'timeoutMs', 'lc0BatchSize', 'batchPipelineDepth']) {
    if (!Number.isFinite(args[name]) || args[name] <= 0) throw new Error(`Invalid ${name}: ${args[name]}`);
  }
  if (args.stockfishScoreDepth !== undefined && (!Number.isFinite(args.stockfishScoreDepth) || args.stockfishScoreDepth <= 0)) throw new Error(`Invalid stockfishScoreDepth: ${args.stockfishScoreDepth}`);
  if (!['js', 'wgsl', 'wasm'].includes(args.inputBackend)) throw new Error(`Invalid inputBackend: ${args.inputBackend}`);
  if (!['hand', 'tvm-packed-f16', 'mixed-tvm-ffn', 'mixed-tvm-ffn-outproj'].includes(args.encoderKernel)) throw new Error(`Invalid encoderKernel: ${args.encoderKernel}`);
  if (args.batchPipelineDepth > 1) process.stderr.write('[lc0-fixed-suite] warning: batchPipelineDepth > 1 is speculative parallel search; depth=1 is the parity-preserving arena baseline.\n');
  return args;
}

function lc0SideFromTags(tags) {
  if (/\b(lc0|leela)\b/i.test(tags.White ?? '')) return 'w';
  if (/\b(lc0|leela)\b/i.test(tags.Black ?? '')) return 'b';
  return null;
}

async function loadFixedFens(args) {
  if (args.fensFile) return (await readFile(args.fensFile, 'utf8')).split(/\r?\n/).map((line) => line.trim()).filter(Boolean).slice(0, args.maxPositions);
  if (!args.sourceReport) throw new Error('expected --source-report or --fens');
  const report = JSON.parse(await readFile(args.sourceReport, 'utf8'));
  const result = (report.results ?? []).find((entry) => entry.runtime === args.sourceRuntime) ?? report.results?.[0];
  if (!result) throw new Error(`No results found in ${args.sourceReport}`);
  const game = parsePgnGames(result.pgn)[args.sourceGame - 1];
  if (!game) throw new Error(`No game ${args.sourceGame} in source runtime ${result.runtime}`);
  const lc0Side = lc0SideFromTags(game.tags);
  if (!lc0Side) throw new Error(`Could not identify LC0 side from PGN tags: ${JSON.stringify(game.tags)}`);
  const fens = [];
  for (const node of game.tree.mainlineFrom()) {
    const parent = node.parent;
    if (!parent) continue;
    if (parent.ply < args.skipPlies) continue;
    const turn = parent.fen.split(/\s+/)[1];
    if (turn === lc0Side) fens.push(parent.fen);
    if (fens.length >= args.maxPositions) break;
  }
  return fens;
}

function arenaUrl(args, runtime, fens) {
  const url = new URL('/lc0-arena.html', args.baseUrl);
  url.searchParams.set('fixedSuiteBench', '1');
  url.searchParams.set('lc0Runtime', runtime);
  url.searchParams.set('seatA', 'lc0:small:100');
  url.searchParams.set('seatB', 'sf:lite:8');
  url.searchParams.set('budgetMode', 'movetime');
  url.searchParams.set('movetimeMs', String(args.movetime));
  url.searchParams.set('stockfishScoreMs', String(args.stockfishScoreMs));
  if (args.stockfishScoreDepth !== undefined) url.searchParams.set('stockfishScoreDepth', String(args.stockfishScoreDepth));
  url.searchParams.set('cacheEntries', String(args.cache));
  url.searchParams.set('lc0BatchSize', String(args.lc0BatchSize));
  url.searchParams.set('lc0BatchPipelineDepth', String(args.batchPipelineDepth));
  url.searchParams.set('inputBackend', args.inputBackend);
  url.searchParams.set('encoderKernel', args.encoderKernel);
  url.searchParams.set('sfThreads', '1');
  url.searchParams.set('fixedSuiteFens', fens.join('|'));
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
  try { await runAgent(args, session, ['close'], 5_000); }
  catch (error) { process.stderr.write(`[lc0-fixed-suite] warning: failed to close ${session}: ${error.message ?? error}\n`); }
}

async function waitForServer(baseUrl, timeoutMs) {
  const deadline = Date.now() + timeoutMs;
  let lastError;
  while (Date.now() < deadline) {
    try {
      const response = await fetch(new URL('/lc0-arena.html', baseUrl), { cache: 'no-store' });
      if (response.ok) return;
      lastError = new Error(`HTTP ${response.status}`);
    } catch (error) { lastError = error; }
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

async function runOne(args, runtime, fens) {
  const session = `${args.session}-${runtime}`;
  const url = arenaUrl(args, runtime, fens);
  process.stderr.write(`[lc0-fixed-suite] ${runtime}: ${fens.length} positions\n`);
  try {
    await runAgent(args, session, ['open', url], 30_000);
    const deadline = Date.now() + args.timeoutMs;
    let done = false;
    while (Date.now() < deadline && !done) {
      const chunk = Math.min(25_000, Math.max(1000, deadline - Date.now()));
      try {
        await runAgent(args, session, ['wait', '--text', 'LC0_FIXED_SUITE_DONE', '--timeout', String(chunk)], chunk + 5_000);
        done = true;
      } catch (_error) {
        if (Date.now() >= deadline) throw _error;
        await delay(250);
      }
    }
    const text = textFromGetResult(await runAgent(args, session, ['get', 'text', '#benchResult'], 30_000));
    const result = JSON.parse(text);
    if (result.status !== 'LC0_FIXED_SUITE_DONE') throw new Error(`unexpected benchmark status for ${runtime}: ${result.status} ${result.error ?? ''}`);
    return result;
  } finally {
    await closeAgentSession(args, session);
  }
}

function compactRuntime(result) {
  const cp = result.positions.map((p) => p.stockfish?.lc0PerspectiveCp).filter(Number.isFinite);
  const visits = result.positions.map((p) => p.lc0Search?.visits).filter(Number.isFinite).reduce((a, b) => a + b, 0);
  const evals = result.positions.map((p) => p.lc0Search?.evals).filter(Number.isFinite).reduce((a, b) => a + b, 0);
  const lc0Ms = result.positions.map((p) => p.lc0Search?.elapsedMs).filter(Number.isFinite).reduce((a, b) => a + b, 0);
  return {
    runtime: result.runtime,
    positions: result.positions.length,
    avgStockfishLc0PerspectiveCp: cp.length ? cp.reduce((a, b) => a + b, 0) / cp.length : null,
    visitsPerPosition: visits / result.positions.length,
    evalsPerPosition: evals / result.positions.length,
    evalsPerSecond: lc0Ms ? evals / (lc0Ms / 1000) : null,
    elapsedMs: result.elapsedMs,
    lc0BatchSize: result.configuration?.lc0BatchSize,
    lc0BatchPipelineDepth: result.configuration?.lc0BatchPipelineDepth,
  };
}

function addRelativeLoss(summary, results) {
  const byRuntime = new Map(summary.map((row) => [row.runtime, { ...row, avgRelativeCpLoss: 0, relativeCpLossPositions: 0 }]));
  const n = Math.max(0, ...results.map((result) => result.positions.length));
  for (let i = 0; i < n; i++) {
    const entries = results.map((result) => ({ runtime: result.runtime, position: result.positions[i] }))
      .filter((entry) => Number.isFinite(entry.position?.stockfish?.lc0PerspectiveCp));
    if (entries.length < 2) continue;
    const best = Math.max(...entries.map((entry) => entry.position.stockfish.lc0PerspectiveCp));
    for (const entry of entries) {
      const row = byRuntime.get(entry.runtime);
      row.avgRelativeCpLoss += best - entry.position.stockfish.lc0PerspectiveCp;
      row.relativeCpLossPositions += 1;
    }
  }
  for (const row of byRuntime.values()) if (row.relativeCpLossPositions) row.avgRelativeCpLoss /= row.relativeCpLossPositions;
  return [...byRuntime.values()];
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) return usage();
  const fens = await loadFixedFens(args);
  if (!fens.length) throw new Error('No fixed positions extracted');
  const server = startServer(args);
  try {
    await waitForServer(args.baseUrl, 30_000);
    const results = [];
    for (const runtime of args.runtimes) results.push(await runOne(args, runtime, fens));
    const report = {
      status: 'LC0_FIXED_SUITE_BENCH_DONE',
      startedAt: new Date().toISOString(),
      baseUrl: args.baseUrl,
      config: {
        runtimes: args.runtimes,
        sourceReport: args.sourceReport,
        sourceRuntime: args.sourceRuntime,
        sourceGame: args.sourceGame,
        movetimeMs: args.movetime,
        stockfishScoreMs: args.stockfishScoreDepth === undefined ? args.stockfishScoreMs : undefined,
        stockfishScoreDepth: args.stockfishScoreDepth,
        lc0BatchSize: args.lc0BatchSize,
        batchPipelineDepth: args.batchPipelineDepth,
        positions: fens.length,
      },
      fens,
      summary: addRelativeLoss(results.map(compactRuntime), results),
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
