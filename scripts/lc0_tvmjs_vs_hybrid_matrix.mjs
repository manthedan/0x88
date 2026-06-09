#!/usr/bin/env node
import { spawn } from 'node:child_process';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { setTimeout as delay } from 'node:timers/promises';

const DEFAULT_HOST = '127.0.0.1';
const DEFAULT_PORT = 5297;
const DEFAULT_TIMEOUT_MS = 10 * 60_000;
const DEFAULT_OUT = 'artifacts/tvm/lc0_tvmjs_vs_hybrid_matrix.json';

function usage() {
  console.log(`Usage: node scripts/lc0_tvmjs_vs_hybrid_matrix.mjs [options]\n\nRuns a reproducible LC0 research matrix for full-model TVMJS/WebGPU and the custom hybrid TVM/WGSL path using one Vite server.\n\nThis is a research wrapper, not a promotion gate. The current hybrid lane uses its native search fixture records; arbitrary-FEN TVMJS runs are marked as not directly fixture-identical until the hybrid page grows arbitrary-FEN support.\n\nOptions:\n  --out PATH                  Aggregate JSON path (default ${DEFAULT_OUT})\n  --base-url URL              Use existing Vite server\n  --host HOST                 Vite host when auto-starting (default ${DEFAULT_HOST})\n  --port N                    Vite port when auto-starting (default ${DEFAULT_PORT})\n  --timeout MS                Per child command timeout (default ${DEFAULT_TIMEOUT_MS})\n  --agent-browser BIN         Browser automation binary forwarded to child harnesses\n  --batch N                   TVMJS batch artifact, usually 8 (default 8)\n  --hybrid-batch N            Hybrid search leaf batch size, usually 4 (default 4)\n  --fixtures N                Fixture/search rows for both lanes where supported (default 4)\n  --visits N                  Search visits for both lanes; hybrid fixtures currently require 32,64,128 (default 32)\n  --repeats N                 Search repeats for both lanes (default 1)\n  --fens PATH                 Optional TVMJS FEN file; hybrid still uses native fixture records\n  --tvmjs-out PATH            Override child TVMJS artifact path\n  --hybrid-out PATH           Override child hybrid artifact path\n  --hybrid-preset NAME        Hybrid runtime preset (default lc0-webgpu-research-b4)\n  --hybrid-head-backend MODE  Hybrid head backend, ort or wgsl (default wgsl)\n  --hybrid-input-backend MODE Hybrid input backend, js/wgsl/wasm (default wasm)\n  --hybrid-encoder MODE       Hybrid encoder kernel (default mixed-tvm-ffn-smolgen-project)\n  --hybrid-legal MODE         Hybrid legal-priors backend (default js)\n  --no-server                 Do not auto-start Vite\n  --dry-run                   Print child commands only\n  -h, --help                  Show help\n`);
}

function parseArgs(argv) {
  const args = {
    out: DEFAULT_OUT,
    host: DEFAULT_HOST,
    port: DEFAULT_PORT,
    timeoutMs: DEFAULT_TIMEOUT_MS,
    agentBrowser: process.env.AGENT_BROWSER_BIN ?? 'agent-browser',
    batch: 8,
    hybridBatch: 4,
    fixtures: 4,
    visits: 32,
    repeats: 1,
    fensFile: '',
    hybridPreset: 'lc0-webgpu-research-b4',
    hybridHeadBackend: 'wgsl',
    hybridInputBackend: 'wasm',
    hybridEncoder: 'mixed-tvm-ffn-smolgen-project',
    hybridLegal: 'js',
    noServer: false,
    explicitBaseUrl: false,
    dryRun: false,
  };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    const next = () => { if (i + 1 >= argv.length) throw new Error(`${arg} requires a value`); return argv[++i]; };
    if (arg === '--out') args.out = next();
    else if (arg === '--base-url') { args.baseUrl = next(); args.explicitBaseUrl = true; }
    else if (arg === '--host') args.host = next();
    else if (arg === '--port') args.port = Number(next());
    else if (arg === '--timeout') args.timeoutMs = Number(next());
    else if (arg === '--agent-browser') args.agentBrowser = next();
    else if (arg === '--batch') args.batch = Number(next());
    else if (arg === '--hybrid-batch') args.hybridBatch = Number(next());
    else if (arg === '--fixtures' || arg === '--fixture-count') args.fixtures = Number(next());
    else if (arg === '--visits') args.visits = Number(next());
    else if (arg === '--repeats') args.repeats = Number(next());
    else if (arg === '--fens') args.fensFile = next();
    else if (arg === '--tvmjs-out') args.tvmjsOut = next();
    else if (arg === '--hybrid-out') args.hybridOut = next();
    else if (arg === '--hybrid-preset') args.hybridPreset = next();
    else if (arg === '--hybrid-head-backend') args.hybridHeadBackend = next();
    else if (arg === '--hybrid-input-backend') args.hybridInputBackend = next();
    else if (arg === '--hybrid-encoder') args.hybridEncoder = next();
    else if (arg === '--hybrid-legal') args.hybridLegal = next();
    else if (arg === '--no-server') args.noServer = true;
    else if (arg === '--dry-run') args.dryRun = true;
    else if (arg === '-h' || arg === '--help') args.help = true;
    else throw new Error(`Unknown option: ${arg}`);
  }
  if (!args.baseUrl) args.baseUrl = `http://${args.host}:${args.port}`;
  if (args.explicitBaseUrl) args.noServer = true;
  for (const [name, value] of Object.entries({ port: args.port, timeout: args.timeoutMs, batch: args.batch, hybridBatch: args.hybridBatch, fixtures: args.fixtures, visits: args.visits, repeats: args.repeats })) {
    if (!Number.isFinite(value) || value <= 0) throw new Error(`Invalid --${name}: ${value}`);
  }
  if (![1, 4, 8].includes(args.batch)) throw new Error(`Invalid --batch ${args.batch}; expected 1, 4, or 8`);
  if (!args.fensFile && ![32, 64, 128].includes(args.visits)) throw new Error(`Invalid --visits ${args.visits}; current hybrid native search fixtures exist for 32, 64, or 128 unless --fens is provided`);
  if (!['ort', 'wgsl'].includes(args.hybridHeadBackend)) throw new Error(`Invalid --hybrid-head-backend ${args.hybridHeadBackend}`);
  if (!['js', 'wgsl', 'wasm'].includes(args.hybridInputBackend)) throw new Error(`Invalid --hybrid-input-backend ${args.hybridInputBackend}`);
  if (!['js', 'wasm', 'gpu'].includes(args.hybridLegal)) throw new Error(`Invalid --hybrid-legal ${args.hybridLegal}`);
  if (!['hand', 'tvm-packed-f16', 'mixed-tvm-ffn', 'mixed-tvm-ffn-outproj', 'mixed-tvm-ffn-smolgen-project'].includes(args.hybridEncoder)) throw new Error(`Invalid --hybrid-encoder ${args.hybridEncoder}`);
  const stem = `lc0_tvmjs_vs_hybrid_b${args.batch}_hb${args.hybridBatch}_v${args.visits}_n${args.fixtures}_r${args.repeats}`;
  args.tvmjsOut ??= join(dirname(args.out), `${stem}.tvmjs.json`);
  args.hybridOut ??= join(dirname(args.out), `${stem}.hybrid.json`);
  return args;
}

function startServer(args) {
  if (args.noServer) return null;
  const child = spawn('npm', ['run', 'web:client', '--', '--host', args.host, '--port', String(args.port), '--strictPort'], { stdio: ['ignore', 'pipe', 'pipe'] });
  let output = '';
  let settled = false;
  child.ready = new Promise((resolve, reject) => {
    const timer = setTimeout(() => settle(reject, new Error(`Vite did not become ready on ${args.port}: ${output.trim()}`)), 30_000);
    const settle = (fn, value) => { if (settled) return; settled = true; clearTimeout(timer); fn(value); };
    const onOutput = (chunk) => { output += chunk.toString('utf8'); if (/ready in \d+\s*ms/.test(output) || output.includes(`:${args.port}/`)) settle(resolve); };
    child.stdout.on('data', (chunk) => { process.stderr.write(`[vite] ${chunk}`); onOutput(chunk); });
    child.stderr.on('data', (chunk) => { process.stderr.write(`[vite] ${chunk}`); onOutput(chunk); });
    child.on('exit', (status, signal) => settle(reject, new Error(`Vite exited before ready (${status ?? signal}): ${output.trim()}`)));
  });
  return child;
}

async function waitForServer(baseUrl, timeoutMs = 30_000) {
  const deadline = Date.now() + timeoutMs;
  let lastError;
  while (Date.now() < deadline) {
    try {
      const response = await fetch(new URL('/lc0-tvmjs-webgpu-smoke.html', baseUrl), { cache: 'no-store' });
      if (response.ok) return;
      lastError = new Error(`HTTP ${response.status}`);
    } catch (error) { lastError = error; }
    await delay(250);
  }
  throw new Error(`server did not become ready at ${baseUrl}: ${lastError?.message ?? 'timeout'}`);
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
    const finish = (fn, value) => { if (settled) return; settled = true; clearTimeout(timer); fn(value); };
    child.stdout.on('data', (chunk) => { chunks.stdout.push(chunk); if (options.echo) process.stderr.write(chunk); });
    child.stderr.on('data', (chunk) => { chunks.stderr.push(chunk); if (options.echo !== false) process.stderr.write(chunk); });
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

function summarizeTvmjs(artifact) {
  const rows = artifact?.result?.searchParity?.rows ?? [];
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
    tvmSearchMeanMs: mean(rows.map((row) => row.tvmMs)),
    ortSearchMeanMs: mean(rows.map((row) => row.ortMs)),
  };
}

function summarizeHybrid(artifact) {
  const rows = artifact?.results ?? [];
  return {
    ok: artifact?.status === 'HYBRID_SEARCH_FIXTURE_PARITY_DONE',
    backend: artifact?.backend,
    headBackend: artifact?.headBackend,
    inputBackend: artifact?.inputBackend,
    encoderKernelVariant: artifact?.encoderKernelVariant,
    legalPriorsBackend: artifact?.legalPriorsBackend,
    cells: artifact?.cells,
    nativeMatches: artifact?.nativeMatches,
    depthBaselineMatches: artifact?.depthBaselineMatches,
    searchMeanElapsedMs: mean(rows.map((row) => row.elapsedMs)),
    backendSearchMeanElapsedMs: mean(rows.map((row) => row.searchElapsedMs)),
    totalEvalMsPerPosition: mean(rows.map((row) => row.totalEvalMsPerPosition)),
    readbackSyncedMsPerPosition: mean(rows.map((row) => row.readbackSyncedMsPerPosition)),
  };
}

function summarizeHeadToHead(tvmjs, hybrid) {
  const tvmRows = tvmjs?.result?.searchParity?.rows ?? [];
  const hybridRows = hybrid?.results ?? [];
  const count = Math.min(tvmRows.length, hybridRows.length);
  const rows = [];
  for (let i = 0; i < count; i++) {
    const tvm = tvmRows[i];
    const hyb = hybridRows[i];
    rows.push({
      index: i,
      fen: tvm?.fen ?? hyb?.fen,
      tvmjsMove: tvm?.tvmMove,
      hybridMove: hyb?.bestMove,
      moveMatches: tvm?.tvmMove === hyb?.bestMove,
      tvmjsMs: tvm?.tvmMs,
      hybridMs: hyb?.elapsedMs,
      hybridBackendMs: hyb?.searchElapsedMs,
    });
  }
  return {
    comparableRows: count,
    moveMatches: rows.filter((row) => row.moveMatches).length,
    tvmjsMeanMs: mean(rows.map((row) => row.tvmjsMs)),
    hybridMeanMs: mean(rows.map((row) => row.hybridMs)),
    rows,
  };
}

async function loadJson(path) {
  return JSON.parse(await readFile(path, 'utf8'));
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) return usage();

  const tvmjsCommand = ['scripts/lc0_tvmjs_webgpu_smoke.mjs', '--base-url', args.baseUrl, '--batch', String(args.batch), '--fixture-count', String(args.fixtures), '--ort-compare', 'f16', '--ort-ep', 'webgpu', '--search-visits', String(args.visits), '--search-fixtures', String(args.fixtures), '--search-repeats', String(args.repeats), '--agent-browser', args.agentBrowser, '--out', args.tvmjsOut];
  if (args.fensFile) tvmjsCommand.push('--fens', args.fensFile);

  const hybridCommand = ['--experimental-strip-types', 'scripts/lc0_browser_hybrid_search_fixture_parity.mjs', '--base-url', args.baseUrl, '--preset', args.hybridPreset, '--head-backend', args.hybridHeadBackend, '--input-backend', args.hybridInputBackend, '--encoder-kernel', args.hybridEncoder, '--legal-priors-backend', args.hybridLegal, '--batch', String(args.hybridBatch), '--visits', String(args.visits), '--fixture-limit', String(args.fixtures), '--repeats', String(args.repeats), '--allow-mismatches', '--agent-browser', args.agentBrowser, '--out', args.hybridOut];
  if (args.fensFile) hybridCommand.push('--fens', args.fensFile);

  if (args.dryRun) {
    console.log(JSON.stringify({ tvmjs: ['node', ...tvmjsCommand], hybrid: ['node', ...hybridCommand] }, null, 2));
    return;
  }

  const server = startServer(args);
  try {
    if (server) await server.ready;
    await waitForServer(args.baseUrl);
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
