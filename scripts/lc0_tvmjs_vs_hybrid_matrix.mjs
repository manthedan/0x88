#!/usr/bin/env node
import { spawn } from 'node:child_process';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { parseScriptArgs } from './lib/cli.mjs';
import { startViteServer, waitForHttp } from './lib/server.mjs';

const DEFAULT_HOST = '127.0.0.1';
const DEFAULT_PORT = 5297;
const DEFAULT_TIMEOUT_MS = 10 * 60_000;
const DEFAULT_OUT = 'artifacts/tvm/lc0_tvmjs_vs_hybrid_matrix.json';

const USAGE = `Usage: node scripts/lc0_tvmjs_vs_hybrid_matrix.mjs [options]\n\nRuns a reproducible LC0 research matrix for full-model TVMJS/WebGPU and the custom hybrid TVM/WGSL path using one Vite server.\n\nThis is a research wrapper, not a promotion gate. The current hybrid lane uses its native search fixture records; arbitrary-FEN TVMJS runs are marked as not directly fixture-identical until the hybrid page grows arbitrary-FEN support.\n\nOptions:\n  --out PATH                  Aggregate JSON path (default ${DEFAULT_OUT})\n  --base-url URL              Use existing Vite server\n  --host HOST                 Vite host when auto-starting (default ${DEFAULT_HOST})\n  --port N                    Vite port when auto-starting (default ${DEFAULT_PORT})\n  --timeout MS                Per child command timeout (default ${DEFAULT_TIMEOUT_MS})\n  --agent-browser BIN         Browser automation binary forwarded to child harnesses\n  --batch N                   TVMJS batch artifact, usually 8 (default 8)\n  --hybrid-batch N            Hybrid search leaf batch size, usually 4 (default 4)\n  --fixtures N                Fixture/search rows for both lanes where supported (default 4)\n  --visits N                  Search visits for both lanes; hybrid fixtures currently require 32,64,128 (default 32)\n  --repeats N                 Search repeats for both lanes (default 1)\n  --stockfish-score-depth N   Score TVMJS/ORT post-search moves at fixed Stockfish depth\n  --stockfish-score-ms N      Score TVMJS/ORT post-search moves by Stockfish movetime\n  --fens PATH                 Optional FEN file; with --fens both lanes use the same FEN rows\n  --tvmjs-out PATH            Override child TVMJS artifact path\n  --hybrid-out PATH           Override child hybrid artifact path\n  --hybrid-preset NAME        Hybrid runtime preset (default lc0-webgpu-research-b4)\n  --hybrid-head-backend MODE  Hybrid head backend, ort or wgsl (default wgsl)\n  --hybrid-input-backend MODE Hybrid input backend, js/wgsl/wasm (default wasm)\n  --hybrid-encoder MODE       Hybrid encoder kernel (default mixed-tvm-ffn-smolgen-project)\n  --hybrid-legal MODE         Hybrid legal-priors backend (default js)\n  --no-server                 Do not auto-start Vite\n  --dry-run                   Print child commands only\n  -h, --help                  Show help\n`;

function parseArgs(argv) {
  const args = parseScriptArgs(argv, {
    options: {
      out: { type: 'string', default: DEFAULT_OUT },
      'base-url': { type: 'string' },
      host: { type: 'string', default: DEFAULT_HOST },
      port: { type: 'string', default: String(DEFAULT_PORT) },
      timeout: { type: 'string', default: String(DEFAULT_TIMEOUT_MS) },
      'agent-browser': { type: 'string', default: process.env.AGENT_BROWSER_BIN ?? 'agent-browser' },
      batch: { type: 'string', default: '8' },
      'hybrid-batch': { type: 'string', default: '4' },
      fixtures: { type: 'string', default: '4' },
      'fixture-count': { type: 'string' },
      visits: { type: 'string', default: '32' },
      repeats: { type: 'string', default: '1' },
      'stockfish-score-depth': { type: 'string' },
      'stockfish-score-ms': { type: 'string' },
      fens: { type: 'string', default: '' },
      'tvmjs-out': { type: 'string' },
      'hybrid-out': { type: 'string' },
      'hybrid-preset': { type: 'string', default: 'lc0-webgpu-research-b4' },
      'hybrid-head-backend': { type: 'string', default: 'wgsl' },
      'hybrid-input-backend': { type: 'string', default: 'wasm' },
      'hybrid-encoder': { type: 'string', default: 'mixed-tvm-ffn-smolgen-project' },
      'hybrid-legal': { type: 'string', default: 'js' },
      'no-server': { type: 'boolean', default: false },
      'dry-run': { type: 'boolean', default: false },
    },
    usage: USAGE,
  });
  if (args.fixtureCount !== undefined) args.fixtures = args.fixtureCount;
  delete args.fixtureCount;
  args.fensFile = args.fens;
  delete args.fens;
  args.port = Number(args.port);
  args.timeoutMs = Number(args.timeout);
  delete args.timeout;
  args.batch = Number(args.batch);
  args.hybridBatch = Number(args.hybridBatch);
  args.fixtures = Number(args.fixtures);
  args.visits = Number(args.visits);
  args.repeats = Number(args.repeats);
  if (args.stockfishScoreDepth !== undefined) args.stockfishScoreDepth = Number(args.stockfishScoreDepth);
  if (args.stockfishScoreMs !== undefined) args.stockfishScoreMs = Number(args.stockfishScoreMs);
  args.explicitBaseUrl = args.baseUrl !== undefined;
  if (!args.baseUrl) args.baseUrl = `http://${args.host}:${args.port}`;
  if (args.explicitBaseUrl) args.noServer = true;
  for (const [name, value] of Object.entries({
    port: args.port,
    timeout: args.timeoutMs,
    batch: args.batch,
    hybridBatch: args.hybridBatch,
    fixtures: args.fixtures,
    visits: args.visits,
    repeats: args.repeats,
  })) {
    if (!Number.isFinite(value) || value <= 0) throw new Error(`Invalid --${name}: ${value}`);
  }
  if (![1, 4, 8].includes(args.batch)) throw new Error(`Invalid --batch ${args.batch}; expected 1, 4, or 8`);
  if (args.stockfishScoreDepth !== undefined && (!Number.isFinite(args.stockfishScoreDepth) || args.stockfishScoreDepth <= 0))
    throw new Error(`Invalid --stockfish-score-depth ${args.stockfishScoreDepth}`);
  if (args.stockfishScoreMs !== undefined && (!Number.isFinite(args.stockfishScoreMs) || args.stockfishScoreMs <= 0))
    throw new Error(`Invalid --stockfish-score-ms ${args.stockfishScoreMs}`);
  if (!args.fensFile && ![32, 64, 128].includes(args.visits))
    throw new Error(`Invalid --visits ${args.visits}; current hybrid native search fixtures exist for 32, 64, or 128 unless --fens is provided`);
  if (!['ort', 'wgsl'].includes(args.hybridHeadBackend)) throw new Error(`Invalid --hybrid-head-backend ${args.hybridHeadBackend}`);
  if (!['js', 'wgsl', 'wasm'].includes(args.hybridInputBackend)) throw new Error(`Invalid --hybrid-input-backend ${args.hybridInputBackend}`);
  if (!['js', 'wasm', 'gpu'].includes(args.hybridLegal)) throw new Error(`Invalid --hybrid-legal ${args.hybridLegal}`);
  if (!['hand', 'tvm-packed-f16', 'mixed-tvm-ffn', 'mixed-tvm-ffn-outproj', 'mixed-tvm-ffn-smolgen-project'].includes(args.hybridEncoder))
    throw new Error(`Invalid --hybrid-encoder ${args.hybridEncoder}`);
  const stem = `lc0_tvmjs_vs_hybrid_b${args.batch}_hb${args.hybridBatch}_v${args.visits}_n${args.fixtures}_r${args.repeats}`;
  args.tvmjsOut ??= join(dirname(args.out), `${stem}.tvmjs.json`);
  args.hybridOut ??= join(dirname(args.out), `${stem}.hybrid.json`);
  return args;
}

function runCommand(command, args, options = {}) {
  return new Promise((resolve, reject) => {
    const started = performance.now();
    const child = spawn(command, args, { stdio: ['ignore', 'pipe', 'pipe'], ...options.spawnOptions });
    const chunks = { stdout: [], stderr: [] };
    let settled = false;
    const timer = setTimeout(() => {
      child.kill('SIGKILL');
      finish(reject, new Error(`${command} ${args.join(' ')} timed out after ${options.timeoutMs}ms`));
    }, options.timeoutMs ?? DEFAULT_TIMEOUT_MS);
    const finish = (fn, value) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      fn(value);
    };
    child.stdout.on('data', (chunk) => {
      chunks.stdout.push(chunk);
      if (options.echo) process.stderr.write(chunk);
    });
    child.stderr.on('data', (chunk) => {
      chunks.stderr.push(chunk);
      if (options.echo !== false) process.stderr.write(chunk);
    });
    child.on('error', (error) => finish(reject, error));
    child.on('close', (status) => {
      const stdout = Buffer.concat(chunks.stdout).toString('utf8');
      const stderr = Buffer.concat(chunks.stderr).toString('utf8');
      const elapsedMs = performance.now() - started;
      if (status !== 0) return finish(reject, new Error(`${command} ${args.join(' ')} failed with ${status}: ${stderr || stdout}`));
      finish(resolve, { status, stdout, stderr, elapsedMs });
    });
  });
}

function mean(values) {
  const xs = values.filter((value) => Number.isFinite(value));
  return xs.length ? xs.reduce((sum, value) => sum + value, 0) / xs.length : undefined;
}

function numericStats(values) {
  const xs = values.filter((value) => Number.isFinite(value));
  if (!xs.length) return undefined;
  return { count: xs.length, min: Math.min(...xs), max: Math.max(...xs), mean: mean(xs) };
}

function summarizeTvmjs(artifact) {
  const rows = artifact?.result?.searchParity?.rows ?? [];
  const stockfishDeltas = rows.map((row) => row.stockfishCpDeltaTvmMinusOrt);
  return {
    ok: artifact?.ok === true,
    batch: artifact?.batch,
    fixtureCount: artifact?.result?.fixtureCount,
    invokeMs: artifact?.result?.invokeMs,
    startupTimings: artifact?.result?.startupTimings,
    gpuBufferAllocation: artifact?.result?.gpuBufferAllocation,
    nativeMatches: artifact?.result?.bestMoveMatches,
    nativeComparable: artifact?.result?.nativeComparable,
    ortF16Matches: artifact?.result?.ortComparisons?.f16?.bestMoveMatches,
    ortF16Comparable: artifact?.result?.ortComparisons?.f16?.comparable,
    searchRows: rows.length || artifact?.result?.searchParity?.fixtureCount,
    searchMoveMatches: artifact?.result?.searchParity?.moveMatches,
    stockfishScoredRows: stockfishDeltas.filter((value) => Number.isFinite(value)).length,
    stockfishCpDeltaTvmMinusOrt: numericStats(stockfishDeltas),
    tvmSearchMeanMs: mean(rows.map((row) => row.tvmMs)),
    ortSearchMeanMs: mean(rows.map((row) => row.ortMs)),
  };
}

function summarizeHybrid(artifact) {
  const rows = artifact?.results ?? [];
  const gpuBufferAllocation = artifact?.gpuBufferAllocation ?? rows.at(-1)?.gpuBufferAllocation;
  return {
    ok: artifact?.status === 'HYBRID_SEARCH_FIXTURE_PARITY_DONE',
    backend: artifact?.backend,
    headBackend: artifact?.headBackend,
    inputBackend: artifact?.inputBackend,
    encoderKernelVariant: artifact?.encoderKernelVariant,
    legalPriorsBackend: artifact?.legalPriorsBackend,
    workerInitMs: artifact?.workerInitMs,
    cells: artifact?.cells,
    nativeMatches: artifact?.nativeMatches,
    depthBaselineMatches: artifact?.depthBaselineMatches,
    gpuBufferAllocation,
    searchMeanElapsedMs: mean(rows.map((row) => row.elapsedMs)),
    backendSearchMeanElapsedMs: mean(rows.map((row) => row.searchElapsedMs)),
    totalEvalMsPerPosition: mean(rows.map((row) => row.totalEvalMsPerPosition)),
    readbackSyncedMsPerPosition: mean(rows.map((row) => row.readbackSyncedMsPerPosition)),
  };
}

function rowRepeatKey(row, source) {
  const raw = row?.repeat;
  if (!Number.isFinite(raw)) return 1;
  return source === 'tvmjs' ? raw + 1 : raw;
}

function summarizeHeadToHead(tvmjs, hybrid) {
  const tvmRows = tvmjs?.result?.searchParity?.rows ?? [];
  const hybridRows = hybrid?.results ?? [];
  const hybridByFenRepeat = new Map();
  for (const row of hybridRows) {
    if (!row?.fen) continue;
    hybridByFenRepeat.set(`${row.fen}\t${rowRepeatKey(row, 'hybrid')}`, row);
  }
  const rows = [];
  for (let i = 0; i < tvmRows.length; i++) {
    const tvm = tvmRows[i];
    const repeat = rowRepeatKey(tvm, 'tvmjs');
    const hyb = tvm?.fen ? hybridByFenRepeat.get(`${tvm.fen}\t${repeat}`) : undefined;
    if (!hyb) {
      rows.push({ index: i, fen: tvm?.fen, repeat, tvmjsMove: tvm?.tvmMove, matchedHybridRow: false, tvmjsMs: tvm?.tvmMs });
      continue;
    }
    rows.push({
      index: i,
      fen: tvm.fen,
      repeat,
      matchedHybridRow: true,
      tvmjsMove: tvm.tvmMove,
      hybridMove: hyb.bestMove,
      moveMatches: tvm.tvmMove === hyb.bestMove,
      tvmjsMs: tvm.tvmMs,
      hybridMs: hyb.elapsedMs,
      hybridBackendMs: hyb.searchElapsedMs,
    });
  }
  const comparableRows = rows.filter((row) => row.matchedHybridRow);
  return {
    comparableRows: comparableRows.length,
    unmatchedTvmjsRows: rows.length - comparableRows.length,
    moveMatches: comparableRows.filter((row) => row.moveMatches).length,
    tvmjsMeanMs: mean(comparableRows.map((row) => row.tvmjsMs)),
    hybridMeanMs: mean(comparableRows.map((row) => row.hybridMs)),
    rows,
  };
}

async function loadJson(path) {
  return JSON.parse(await readFile(path, 'utf8'));
}

async function main() {
  const args = parseArgs(process.argv.slice(2));

  const tvmjsCommand = [
    'scripts/lc0_tvmjs_webgpu_smoke.mjs',
    '--base-url',
    args.baseUrl,
    '--batch',
    String(args.batch),
    '--fixture-count',
    String(args.fixtures),
    '--ort-compare',
    'f16',
    '--ort-ep',
    'webgpu',
    '--search-visits',
    String(args.visits),
    '--search-fixtures',
    String(args.fixtures),
    '--search-repeats',
    String(args.repeats),
    '--agent-browser',
    args.agentBrowser,
    '--out',
    args.tvmjsOut,
  ];
  if (args.stockfishScoreDepth !== undefined) tvmjsCommand.push('--stockfish-score-depth', String(args.stockfishScoreDepth));
  if (args.stockfishScoreMs !== undefined) tvmjsCommand.push('--stockfish-score-ms', String(args.stockfishScoreMs));
  if (args.fensFile) tvmjsCommand.push('--fens', args.fensFile);

  const hybridCommand = [
    '--experimental-strip-types',
    'scripts/lc0_browser_hybrid_search_fixture_parity.mjs',
    '--base-url',
    args.baseUrl,
    '--preset',
    args.hybridPreset,
    '--head-backend',
    args.hybridHeadBackend,
    '--input-backend',
    args.hybridInputBackend,
    '--encoder-kernel',
    args.hybridEncoder,
    '--legal-priors-backend',
    args.hybridLegal,
    '--batch',
    String(args.hybridBatch),
    '--visits',
    String(args.visits),
    '--fixture-limit',
    String(args.fixtures),
    '--repeats',
    String(args.repeats),
    '--allow-mismatches',
    '--agent-browser',
    args.agentBrowser,
    '--out',
    args.hybridOut,
  ];
  if (args.fensFile) hybridCommand.push('--fens', args.fensFile);

  if (args.dryRun) {
    console.log(JSON.stringify({ tvmjs: ['node', ...tvmjsCommand], hybrid: ['node', ...hybridCommand] }, null, 2));
    return;
  }

  const server = startViteServer(args);
  try {
    if (server) await server.ready;
    await waitForHttp(args.baseUrl);
    await mkdir(dirname(args.out), { recursive: true });

    process.stderr.write(`[lc0-tvmjs-vs-hybrid] TVMJS child -> ${args.tvmjsOut}\n`);
    const tvmjsRun = await runCommand('node', tvmjsCommand, { timeoutMs: args.timeoutMs });
    process.stderr.write(`[lc0-tvmjs-vs-hybrid] hybrid child -> ${args.hybridOut}\n`);
    const hybridRun = await runCommand('node', hybridCommand, { timeoutMs: args.timeoutMs });

    const tvmjs = await loadJson(args.tvmjsOut);
    const hybrid = await loadJson(args.hybridOut);
    const artifact = {
      schema: 'lc0_browser.tvmjs_vs_hybrid_matrix.v1',
      generatedAt: new Date().toISOString(),
      ok: tvmjs.ok === true && hybrid.status === 'HYBRID_SEARCH_FIXTURE_PARITY_DONE',
      caveat: args.fensFile
        ? 'Both lanes used the requested FEN file. Hybrid arbitrary-FEN mode has no native best-move oracle, so compare TVMJS-vs-ORT and hybrid depth-baseline/search timing rather than native matches.'
        : 'Both lanes use their built-in fixture/search harnesses with matched count/visits/repeats; pass --fens for strict row identity.',
      baseUrl: args.baseUrl,
      parameters: {
        tvmjsBatch: args.batch,
        hybridBatch: args.hybridBatch,
        fixtures: args.fixtures,
        visits: args.visits,
        repeats: args.repeats,
        stockfishScoreDepth: args.stockfishScoreDepth,
        stockfishScoreMs: args.stockfishScoreMs,
        fensFile: args.fensFile || undefined,
        hybridPreset: args.hybridPreset,
        hybridHeadBackend: args.hybridHeadBackend,
        hybridInputBackend: args.hybridInputBackend,
        hybridEncoder: args.hybridEncoder,
        hybridLegal: args.hybridLegal,
      },
      artifacts: { tvmjs: args.tvmjsOut, hybrid: args.hybridOut },
      wallTimeMs: { tvmjsChild: tvmjsRun.elapsedMs, hybridChild: hybridRun.elapsedMs },
      summary: { tvmjs: summarizeTvmjs(tvmjs), hybrid: summarizeHybrid(hybrid), headToHead: summarizeHeadToHead(tvmjs, hybrid) },
    };
    await writeFile(args.out, `${JSON.stringify(artifact, null, 2)}\n`);
    console.log(JSON.stringify(artifact, null, 2));
    if (!artifact.ok) process.exitCode = 1;
  } finally {
    server?.kill('SIGTERM');
  }
}

main().catch((error) => {
  console.error(error?.stack || error?.message || String(error));
  process.exit(1);
});
