#!/usr/bin/env node
import { spawn, spawnSync } from 'node:child_process';
import { existsSync, readFileSync } from 'node:fs';
import { setTimeout as delay } from 'node:timers/promises';
import { boardToFen } from '../src/chess/board.ts';
import { buildBoardHistoryFromMoves } from '../src/lc0/history.ts';
import { Lc0OnnxEvaluator } from '../src/lc0/onnxEvaluator.ts';

const DEFAULT_HOST = '127.0.0.1';
const DEFAULT_PORT = 5179;
const DEFAULT_LIMIT = 3;
const DEFAULT_TIMEOUT_MS = 120_000;
const DEFAULT_F32_MODEL = '../models/lc0-bestnets/onnx/t1-256x10-distilled-swa-2432500.batch1.f32.onnx';

function usage() {
  console.log(`Usage: node --experimental-strip-types scripts/lc0_browser_hybrid_drift.mjs [options]\n\nCompares browser hybrid lc0web WGSL encoder + ORT heads output against f32 ONNX and native BLAS fixture priors.\n\nOptions:\n  --base-url URL        Use an existing dev server (default http://${DEFAULT_HOST}:${DEFAULT_PORT})\n  --port N             Vite port when auto-starting (default ${DEFAULT_PORT})\n  --host HOST          Vite host when auto-starting (default ${DEFAULT_HOST})\n  --agent-browser BIN  Browser automation binary (default: AGENT_BROWSER_BIN or agent-browser)\n  --session NAME       agent-browser session name\n  --timeout MS         Total browser wait timeout (default ${DEFAULT_TIMEOUT_MS})\n  --limit N            Number of native fixtures to evaluate (default ${DEFAULT_LIMIT})\n  --layers N           Encoder layers for hybrid path (default 10)\n  --f32-model PATH     f32 ONNX baseline (default ${DEFAULT_F32_MODEL})\n  --no-server          Do not auto-start Vite\n  -h, --help           Show this help\n`);
}

function parseArgs(argv) {
  const args = {
    host: DEFAULT_HOST,
    port: DEFAULT_PORT,
    limit: DEFAULT_LIMIT,
    layers: 10,
    timeoutMs: DEFAULT_TIMEOUT_MS,
    agentBrowser: process.env.AGENT_BROWSER_BIN ?? 'agent-browser',
    session: process.env.AGENT_BROWSER_SESSION ?? `lc0-hybrid-drift-${process.pid}`,
    f32Model: DEFAULT_F32_MODEL,
    noServer: false,
  };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    const next = () => {
      if (i + 1 >= argv.length) throw new Error(`${arg} requires a value`);
      return argv[++i];
    };
    if (arg === '--base-url') args.baseUrl = next();
    else if (arg === '--port') args.port = Number(next());
    else if (arg === '--host') args.host = next();
    else if (arg === '--agent-browser') args.agentBrowser = next();
    else if (arg === '--session') args.session = next();
    else if (arg === '--timeout') args.timeoutMs = Number(next());
    else if (arg === '--limit') args.limit = Number(next());
    else if (arg === '--layers') args.layers = Number(next());
    else if (arg === '--f32-model') args.f32Model = next();
    else if (arg === '--no-server') args.noServer = true;
    else if (arg === '-h' || arg === '--help') args.help = true;
    else throw new Error(`Unknown option: ${arg}`);
  }
  if (!args.baseUrl) args.baseUrl = `http://${args.host}:${args.port}`;
  for (const [name, value] of [['port', args.port], ['limit', args.limit], ['layers', args.layers], ['timeout', args.timeoutMs]]) {
    if (!Number.isFinite(value) || value <= 0) throw new Error(`Invalid --${name}: ${value}`);
  }
  return args;
}

function runAgent(args, commandArgs, timeoutMs = 30_000) {
  const fullArgs = ['--json', '--session', args.session, ...commandArgs];
  const result = spawnSync(args.agentBrowser, fullArgs, { encoding: 'utf8', timeout: timeoutMs, maxBuffer: 16 * 1024 * 1024 });
  if (result.error) throw result.error;
  if (result.status !== 0) throw new Error(`${args.agentBrowser} ${fullArgs.slice(1).join(' ')} failed: ${result.stderr || result.stdout}`);
  const parsed = result.stdout ? JSON.parse(result.stdout.trim()) : null;
  if (parsed && typeof parsed === 'object' && 'success' in parsed) {
    if (parsed.success === false) throw new Error(`${args.agentBrowser} ${fullArgs.slice(1).join(' ')} failed: ${parsed.error ?? result.stdout}`);
    return parsed.data ?? parsed;
  }
  return parsed;
}

async function waitForServer(baseUrl, timeoutMs) {
  const deadline = Date.now() + timeoutMs;
  let lastError;
  while (Date.now() < deadline) {
    try {
      const response = await fetch(`${baseUrl.replace(/\/$/, '')}/lc0-policy-only.html`, { cache: 'no-store' });
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

function readJsonl(path) {
  return readFileSync(path, 'utf8').trim().split('\n').filter(Boolean).map((line) => JSON.parse(line));
}

function nativeCastlingToStandard(uci) {
  return ({ e1h1: 'e1g1', e1a1: 'e1c1', e8h8: 'e8g8', e8a8: 'e8c8' })[uci] ?? uci;
}

function nativeWdl(native) {
  const q = native.node?.q ?? native.node?.wl;
  const d = native.node?.d;
  if (!Number.isFinite(q) || !Number.isFinite(d)) return null;
  return [(1 - d + q) / 2, d, (1 - d - q) / 2];
}

function maxAbs(values) {
  return values.reduce((max, value) => Math.max(max, Math.abs(value)), 0);
}

function priorDiffs(actualPriors, expectedPriors) {
  return expectedPriors.map((expected) => {
    const uci = nativeCastlingToStandard(expected.uci);
    const actual = actualPriors.find((entry) => entry.uci === uci);
    return { uci, expected: expected.prior, actual: actual?.prior ?? null, abs: actual ? Math.abs(actual.prior - expected.prior) : Infinity };
  });
}

async function f32Baselines(args, records) {
  if (!existsSync(args.f32Model)) throw new Error(`f32 model not found: ${args.f32Model}`);
  const originalLog = console.log;
  const originalInfo = console.info;
  console.log = (...values) => console.error(...values);
  console.info = (...values) => console.error(...values);
  try {
    const evaluator = await Lc0OnnxEvaluator.create(readFileSync(args.f32Model));
    const out = [];
    for (const record of records) {
      const positions = record.moves ? buildBoardHistoryFromMoves(record.moves, record.startFen) : undefined;
      const input = positions ? { positions } : record.fen;
      const evaluation = await evaluator.evaluate(input);
      out.push({
        id: record.id,
        fen: record.fen ?? boardToFen(positions[positions.length - 1]),
        bestMove: evaluation.bestMove,
        wdl: evaluation.wdl,
        q: evaluation.q,
        topPriors: evaluation.legalPriors.slice(0, 10).map(({ uci, index, prior }) => ({ uci, index, prior })),
      });
    }
    return out;
  } finally {
    console.log = originalLog;
    console.info = originalInfo;
  }
}

async function browserHybrid(args) {
  const url = `${args.baseUrl.replace(/\/$/, '')}/lc0-policy-only.html?hybridDrift=1&encoderLayers=${args.layers}&hybridDriftLimit=${args.limit}&ep=wasm&packVerify=0`;
  process.stderr.write(`[lc0-hybrid-drift] ${url}\n`);
  runAgent(args, ['open', url], 30_000);
  const deadline = Date.now() + args.timeoutMs;
  while (Date.now() < deadline) {
    const chunk = Math.min(25_000, Math.max(1000, deadline - Date.now()));
    try {
      runAgent(args, ['wait', '--text', 'HYBRID_DRIFT_DONE', '--timeout', String(chunk)], chunk + 5_000);
      const text = runAgent(args, ['get', 'text', '#benchResult'], 30_000).text;
      return JSON.parse(text);
    } catch (error) {
      if (Date.now() >= deadline) throw error;
    }
  }
  throw new Error(`Timed out waiting for HYBRID_DRIFT_DONE after ${args.timeoutMs}ms`);
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) return usage();
  const server = startServer(args);
  try {
    await waitForServer(args.baseUrl, 30_000);
    const nativeRecords = [
      ...readJsonl('fixtures/lc0/native_fen_only_blas.jsonl'),
      ...readJsonl('fixtures/lc0/native_history_blas.jsonl'),
    ].slice(0, args.limit);
    const [f32, hybrid] = await Promise.all([
      f32Baselines(args, nativeRecords),
      browserHybrid(args),
    ]);
    const comparisons = hybrid.evaluations.map((hybridEval) => {
      const f32Eval = f32.find((entry) => entry.id === hybridEval.id);
      const native = nativeRecords.find((entry) => entry.id === hybridEval.id);
      const nativeW = nativeWdl(native);
      const f32Prior = priorDiffs(hybridEval.topPriors, f32Eval.topPriors);
      const nativePrior = priorDiffs(hybridEval.topPriors, native.topPriors.slice(0, 10));
      return {
        id: hybridEval.id,
        hybridBestMove: hybridEval.bestMove,
        f32BestMove: f32Eval.bestMove,
        nativeBestMove: nativeCastlingToStandard(native.bestmove),
        f32WdlMaxAbsDiff: maxAbs(hybridEval.wdl.map((v, i) => v - f32Eval.wdl[i])),
        nativeWdlMaxAbsDiff: nativeW ? maxAbs(hybridEval.wdl.map((v, i) => v - nativeW[i])) : null,
        f32TopPriorMaxAbsDiff: Math.max(...f32Prior.map((entry) => entry.abs)),
        nativeTopPriorMaxAbsDiff: Math.max(...nativePrior.map((entry) => entry.abs)),
        f32Prior,
        nativePrior,
      };
    });
    const result = {
      status: 'LC0_HYBRID_DRIFT_DONE',
      fixtures: comparisons.length,
      browser: { backend: hybrid.backend, layers: hybrid.layers, elapsedMs: hybrid.elapsedMs },
      summary: {
        f32BestMoveMatches: comparisons.filter((c) => c.hybridBestMove === c.f32BestMove).length,
        nativeBestMoveMatches: comparisons.filter((c) => c.hybridBestMove === c.nativeBestMove).length,
        f32WdlMaxAbsDiff: Math.max(...comparisons.map((c) => c.f32WdlMaxAbsDiff)),
        nativeWdlMaxAbsDiff: Math.max(...comparisons.map((c) => c.nativeWdlMaxAbsDiff ?? 0)),
        f32TopPriorMaxAbsDiff: Math.max(...comparisons.map((c) => c.f32TopPriorMaxAbsDiff)),
        nativeTopPriorMaxAbsDiff: Math.max(...comparisons.map((c) => c.nativeTopPriorMaxAbsDiff)),
      },
      comparisons,
    };
    console.log(JSON.stringify(result, null, 2));
  } finally {
    server?.kill('SIGTERM');
  }
}

main().catch((error) => {
  console.error(error.stack ?? error.message);
  process.exit(1);
});
