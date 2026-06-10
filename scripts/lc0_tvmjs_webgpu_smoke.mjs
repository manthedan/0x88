#!/usr/bin/env node
import { spawn } from 'node:child_process';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname } from 'node:path';
import { setTimeout as delay } from 'node:timers/promises';

const DEFAULT_HOST = '127.0.0.1';
const DEFAULT_PORT = 5291;
const DEFAULT_TIMEOUT_MS = 180_000;
const DEFAULT_AGENT_BROWSER = process.env.AGENT_BROWSER_BIN ?? 'agent-browser';

function usage() {
  console.log(`Usage: node scripts/lc0_tvmjs_webgpu_smoke.mjs [options]\n\nRuns the LC0 whole-model TVMJS/WebGPU browser smoke and saves JSON evidence.\n\nOptions:\n  --batch N           Batch artifact to test: 1, 4, or 8 (default 8)\n  --fixtures          Encode real fixtures and compare best moves (default true)\n  --no-fixtures       Run loader/zero-input invoke only\n  --fixture-offset N  First fixture index for real-fixture mode (default 0)\n  --fixture-count N   Number of fixtures/FEN rows to request (default batch)
  --tensor-cache      Fetch manifest tensor-cache sidecar before VM setup (research-only)
  --fens PATH         Newline-separated FEN suite; implies --fixtures and bypasses fixtures/lc0/fen_only.json\n  --ort-compare MODE  Compare TVMJS outputs against ORT: none, f16, f32, both (default none)\n  --ort-ep EP         ORT execution provider for comparison: webgpu, wasm, webgpu,wasm (default webgpu)\n  --ort-model TPL     ORT comparison model path template with {batch}/{dtype} placeholders (default t1 family)\n  --fixture-baseline PATH  Native fixture baseline JSONL served path (default /fixtures/lc0/native_fen_only_blas.jsonl)\n  --tie-epsilon X     Tolerate best-move mismatches whose competing priors are within X (recorded as tieTolerated; default strict)\n  --game-plies N      Run a same-line tree-reuse A/B game sequence of N plies (fresh-tree leg defines the line)\n  --game-visits N     Visits per game-sequence search (default searchVisits or 16)\n  --game-start-fen F  Game-sequence start position (default startpos)\n  --search-visits N   Also run TVMJS-vs-ORT search parity with fixed visits\n  --search-fixtures N Number of fixtures for search parity (default 2)\n  --search-repeats N  Repeat search parity rows for timing stability (default 1)\n  --search-pipeline-depth N  Evaluate this many TVMJS batches concurrently during search parity (default 1)\n  --stockfish-score-depth N  Score TVMJS/ORT post-search moves at fixed Stockfish depth\n  --stockfish-score-ms N     Score TVMJS/ORT post-search moves by Stockfish movetime\n  --base-url URL      Use existing server instead of starting Vite\n  --host HOST         Vite host (default ${DEFAULT_HOST})\n  --port N            Vite port (default ${DEFAULT_PORT})\n  --timeout MS        Overall timeout (default ${DEFAULT_TIMEOUT_MS})\n  --agent-browser BIN Browser automation binary (default AGENT_BROWSER_BIN or agent-browser)\n  --out PATH          JSON artifact path\n  --no-server         Do not auto-start Vite\n  -h, --help          Show help\n`);
}

function parseArgs(argv) {
  const args = { batch: 8, fixtures: true, fixtureOffset: 0, fixtureCount: undefined, tensorCache: false, fensFile: '', ortCompare: 'none', ortEp: 'webgpu', searchVisits: 0, searchFixtures: 2, searchRepeats: 1, searchPipelineDepth: 1, stockfishScoreDepth: undefined, stockfishScoreMs: undefined, host: DEFAULT_HOST, port: DEFAULT_PORT, timeoutMs: DEFAULT_TIMEOUT_MS, agentBrowser: DEFAULT_AGENT_BROWSER, noServer: false };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    const next = () => { if (i + 1 >= argv.length) throw new Error(`${arg} requires a value`); return argv[++i]; };
    if (arg === '--batch') args.batch = Number(next());
    else if (arg === '--manifest') args.manifest = next();
    else if (arg === '--fixtures') args.fixtures = true;
    else if (arg === '--no-fixtures') args.fixtures = false;
    else if (arg === '--fixture-offset') args.fixtureOffset = Number(next());
    else if (arg === '--fixture-count') args.fixtureCount = Number(next());
    else if (arg === '--tensor-cache') args.tensorCache = true;
    else if (arg === '--fens') { args.fensFile = next(); args.fixtures = true; }
    else if (arg === '--ort-compare') args.ortCompare = next();
    else if (arg === '--ort-ep') args.ortEp = next();
    else if (arg === '--ort-model') args.ortModel = next();
    else if (arg === '--fixture-baseline') args.fixtureBaseline = next();
    else if (arg === '--tie-epsilon') args.tieEpsilon = Number(next());
    else if (arg === '--game-plies') args.gamePlies = Number(next());
    else if (arg === '--game-visits') args.gameVisits = Number(next());
    else if (arg === '--game-start-fen') args.gameStartFen = next();
    else if (arg === '--moves-left-effect') args.movesLeftEffect = Number(next());
    else if (arg === '--search-visits') args.searchVisits = Number(next());
    else if (arg === '--search-fixtures') args.searchFixtures = Number(next());
    else if (arg === '--search-repeats') args.searchRepeats = Number(next());
    else if (arg === '--search-pipeline-depth') args.searchPipelineDepth = Number(next());
    else if (arg === '--pass-coalesce') args.passCoalesce = true;
    else if (arg === '--kernel-profile-invokes') args.kernelProfileInvokes = Number(next());
    else if (arg === '--stockfish-score-depth') args.stockfishScoreDepth = Number(next());
    else if (arg === '--stockfish-score-ms') args.stockfishScoreMs = Number(next());
    else if (arg === '--base-url') { args.baseUrl = next(); args.noServer = true; }
    else if (arg === '--host') args.host = next();
    else if (arg === '--port') args.port = Number(next());
    else if (arg === '--timeout') args.timeoutMs = Number(next());
    else if (arg === '--agent-browser') args.agentBrowser = next();
    else if (arg === '--out') args.out = next();
    else if (arg === '--no-server') args.noServer = true;
    else if (arg === '-h' || arg === '--help') args.help = true;
    else throw new Error(`Unknown option: ${arg}`);
  }
  if (!args.baseUrl) args.baseUrl = `http://${args.host}:${args.port}`;
  if (![1, 4, 8, 16, 32].includes(args.batch)) throw new Error(`Invalid --batch ${args.batch}; expected 1, 4, 8, 16, or 32`);
  if (!Number.isFinite(args.fixtureOffset) || args.fixtureOffset < 0) throw new Error(`Invalid --fixture-offset ${args.fixtureOffset}`);
  if (args.fixtureCount !== undefined && (!Number.isFinite(args.fixtureCount) || args.fixtureCount <= 0)) throw new Error(`Invalid --fixture-count ${args.fixtureCount}`);
  if (!['none', 'f16', 'f32', 'both'].includes(args.ortCompare)) throw new Error(`Invalid --ort-compare ${args.ortCompare}`);
  if (!['webgpu', 'wasm', 'webgpu,wasm'].includes(args.ortEp)) throw new Error(`Invalid --ort-ep ${args.ortEp}`);
  if (!Number.isFinite(args.searchVisits) || args.searchVisits < 0) throw new Error(`Invalid --search-visits ${args.searchVisits}`);
  if (!Number.isFinite(args.searchFixtures) || args.searchFixtures <= 0) throw new Error(`Invalid --search-fixtures ${args.searchFixtures}`);
  if (!Number.isFinite(args.searchRepeats) || args.searchRepeats <= 0) throw new Error(`Invalid --search-repeats ${args.searchRepeats}`);
  if (!Number.isFinite(args.searchPipelineDepth) || args.searchPipelineDepth <= 0) throw new Error(`Invalid --search-pipeline-depth ${args.searchPipelineDepth}`);
  if (args.stockfishScoreDepth !== undefined && (!Number.isFinite(args.stockfishScoreDepth) || args.stockfishScoreDepth <= 0)) throw new Error(`Invalid --stockfish-score-depth ${args.stockfishScoreDepth}`);
  if (args.stockfishScoreMs !== undefined && (!Number.isFinite(args.stockfishScoreMs) || args.stockfishScoreMs <= 0)) throw new Error(`Invalid --stockfish-score-ms ${args.stockfishScoreMs}`);
  if (!Number.isFinite(args.port) || args.port <= 0) throw new Error(`Invalid --port ${args.port}`);
  if (!Number.isFinite(args.timeoutMs) || args.timeoutMs <= 0) throw new Error(`Invalid --timeout ${args.timeoutMs}`);
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
      const response = await fetch(new URL('/lc0-tvmjs-webgpu-smoke.html', baseUrl), { cache: 'no-store' });
      if (response.ok) return;
      lastError = new Error(`HTTP ${response.status}`);
    } catch (error) { lastError = error; }
    await delay(250);
  }
  throw new Error(`server did not become ready at ${baseUrl}: ${lastError?.message ?? 'timeout'}`);
}

async function closeSession(args, session) {
  try { await runAgent(args, ['close'], 5_000, session); } catch { /* best effort */ }
}

async function pollResult(args, session, timeoutMs) {
  const deadline = Date.now() + timeoutMs;
  let last;
  while (Date.now() < deadline) {
    last = await evalPage(args, session, `(() => ({ result: window.lc0TvmjsLastResult ?? null, error: window.lc0TvmjsLastError ?? null, log: document.getElementById('log')?.textContent ?? '' }))()`);
    if (last?.error) throw new Error(`browser smoke failed: ${last.error}`);
    if (last?.result) return last;
    if (String(last?.log ?? '').includes('SMOKE_ERROR')) throw new Error(`browser smoke failed: ${last.log}`);
    await delay(1000);
  }
  throw new Error(`timed out waiting for TVMJS smoke result; last=${JSON.stringify(last)}`);
}

async function loadFenSuite(args) {
  if (!args.fensFile) return [];
  const rows = (await readFile(args.fensFile, 'utf8')).split(/\r?\n/).map((line) => line.trim()).filter((line) => line && !line.startsWith('#'));
  if (!rows.length) throw new Error(`No FEN rows found in ${args.fensFile}`);
  return rows;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) { usage(); return; }
  const server = startServer(args);
  try {
    if (server) await server.ready;
    await waitForServer(args.baseUrl);
    const fixedSuiteFens = await loadFenSuite(args);
    const session = `lc0-tvmjs-${process.pid}`;
    const url = new URL('/lc0-tvmjs-webgpu-smoke.html', args.baseUrl);
    url.searchParams.set('batch', String(args.batch));
    if (args.manifest) url.searchParams.set('manifest', args.manifest);
    if (args.passCoalesce) url.searchParams.set('passCoalesce', '1');
    if (args.kernelProfileInvokes > 1) url.searchParams.set('kernelProfileInvokes', String(Math.floor(args.kernelProfileInvokes)));
    url.searchParams.set('invoke', '1');
    if (args.fixtures) {
      url.searchParams.set('fixtures', '1');
      url.searchParams.set('fixtureOffset', String(Math.floor(args.fixtureOffset)));
      if (fixedSuiteFens.length) {
        const requestedCount = args.fixtureCount !== undefined ? Math.floor(args.fixtureCount) : fixedSuiteFens.length;
        url.searchParams.set('fixtureCount', String(requestedCount));
        url.searchParams.set('fixedSuiteFens', fixedSuiteFens.slice(args.fixtureOffset, args.fixtureOffset + requestedCount).join('|'));
      } else if (args.fixtureCount !== undefined) url.searchParams.set('fixtureCount', String(Math.floor(args.fixtureCount)));
    }
    if (args.tensorCache) url.searchParams.set('tensorCache', '1');
    if (args.ortCompare !== 'none') url.searchParams.set('ortCompare', args.ortCompare);
    if (args.ortEp) url.searchParams.set('ortEp', args.ortEp);
    if (args.ortModel) url.searchParams.set('ortModel', args.ortModel);
    if (args.fixtureBaseline) url.searchParams.set('fixtureBaseline', args.fixtureBaseline);
    if (args.gamePlies) url.searchParams.set('gamePlies', String(Math.floor(args.gamePlies)));
    if (args.gameVisits) url.searchParams.set('gameVisits', String(Math.floor(args.gameVisits)));
    if (args.gameStartFen) url.searchParams.set('gameStartFen', args.gameStartFen);
    if (args.movesLeftEffect) url.searchParams.set('movesLeftEffect', String(args.movesLeftEffect));
    if (args.searchVisits > 0) {
      url.searchParams.set('searchVisits', String(Math.floor(args.searchVisits)));
      url.searchParams.set('searchFixtureCount', String(Math.floor(args.searchFixtures)));
      url.searchParams.set('searchRepeats', String(Math.floor(args.searchRepeats)));
      if (args.searchPipelineDepth > 1) url.searchParams.set('searchPipelineDepth', String(Math.floor(args.searchPipelineDepth)));
      if (args.stockfishScoreDepth !== undefined) url.searchParams.set('stockfishScoreDepth', String(Math.floor(args.stockfishScoreDepth)));
      if (args.stockfishScoreMs !== undefined) url.searchParams.set('stockfishScoreMs', String(Math.floor(args.stockfishScoreMs)));
    }
    url.searchParams.set('autorun', '1');
    process.stderr.write(`[lc0-tvmjs-smoke] ${url}\n`);
    try {
      await runAgent(args, ['open', String(url)], 30_000, session);
      const status = await pollResult(args, session, args.timeoutMs);
      const artifact = {
        schema: 'lc0_browser.tvmjs_webgpu_smoke.v1',
        generatedAt: new Date().toISOString(),
        url: String(url),
        batch: args.batch,
        manifest: args.manifest,
        fixtures: args.fixtures,
        fixtureOffset: args.fixtureOffset,
        fixtureCount: args.fixtureCount ?? args.batch,
        tensorCache: args.tensorCache,
        ortCompare: args.ortCompare,
        ortEp: args.ortEp,
        ortModel: args.ortModel,
        fixtureBaseline: args.fixtureBaseline,
        gamePlies: args.gamePlies,
        gameVisits: args.gameVisits,
        gameStartFen: args.gameStartFen,
        fensFile: args.fensFile || undefined,
        searchVisits: args.searchVisits,
        searchFixtures: args.searchFixtures,
        searchRepeats: args.searchRepeats,
        searchPipelineDepth: args.searchPipelineDepth,
        stockfishScoreDepth: args.stockfishScoreDepth,
        stockfishScoreMs: args.stockfishScoreMs,
        ok: true,
        result: status.result,
        logTail: String(status.log ?? '').split('\n').slice(-20),
      };
      // --tie-epsilon: best-move mismatches whose competing priors sit within
      // epsilon are recorded as tie-tolerated instead of failing the gate.
      // Raw match counts stay untouched in the artifact; default (no flag) is
      // the original strict behavior. This is the research-side expression of
      // the f16 drift/tolerance policy the runbook lists as a promotion blocker.
      const tieEps = args.tieEpsilon;
      const priorOf = (list, uci) => (list ?? []).find((entry) => entry.uci === uci)?.prior;
      artifact.tieEpsilon = tieEps;
      artifact.tieTolerated = { native: [], ort: {}, search: [] };
      if (args.fixtures && artifact.result.nativeComparable > 0 && artifact.result.bestMoveMatches !== artifact.result.nativeComparable) {
        let tolerated = 0;
        for (const row of artifact.result.results ?? []) {
          if (!row.nativeBestMove || row.bestMove === row.nativeBestMove) continue;
          const gap = (priorOf(row.topPriors, row.bestMove) ?? NaN) - (priorOf(row.topPriors, row.nativeBestMove) ?? NaN);
          if (tieEps !== undefined && Number.isFinite(gap) && gap >= 0 && gap <= tieEps) {
            tolerated++;
            artifact.tieTolerated.native.push({ id: row.id, tvm: row.bestMove, native: row.nativeBestMove, priorGap: gap });
          }
        }
        if (artifact.result.bestMoveMatches + tolerated !== artifact.result.nativeComparable) {
          artifact.ok = false;
          artifact.error = `native best-move parity failed: ${artifact.result.bestMoveMatches}/${artifact.result.nativeComparable}`;
        }
      }
      for (const [dtype, comparison] of Object.entries(artifact.result.ortComparisons ?? {})) {
        if (comparison?.skipped) continue;
        if (comparison?.comparable > 0 && comparison.bestMoveMatches !== comparison.comparable) {
          const toleratedRows = (comparison.rows ?? []).filter((row) => row.tvmBestMove && row.ortBestMove
            && row.tvmBestMove !== row.ortBestMove
            && tieEps !== undefined && Number.isFinite(row.maxTopPriorAbsDiff) && row.maxTopPriorAbsDiff <= tieEps);
          if (toleratedRows.length) artifact.tieTolerated.ort[dtype] = toleratedRows.map((row) => ({ id: row.id, tvm: row.tvmBestMove, ort: row.ortBestMove, maxTopPriorAbsDiff: row.maxTopPriorAbsDiff }));
          if (comparison.bestMoveMatches + toleratedRows.length !== comparison.comparable) {
            artifact.ok = false;
            artifact.error = `${dtype} ORT best-move parity failed: ${comparison.bestMoveMatches}/${comparison.comparable}`;
          }
        }
      }
      const search = artifact.result.searchParity;
      const searchComparable = search?.searchRows ?? search?.fixtureCount;
      if (searchComparable > 0 && search.moveMatches !== searchComparable) {
        let tolerated = 0;
        for (const row of search.rows ?? []) {
          if (!row.tvmMove || !row.ortMove || row.tvmMove === row.ortMove) continue;
          const gap = Math.abs((priorOf(row.tvmTop, row.tvmMove) ?? NaN) - (priorOf(row.tvmTop, row.ortMove) ?? NaN));
          // Search picks by visit count; equal visits for both moves in the
          // TVM row's own stats means the flip is pure tie-breaking.
          const visitsOf = (uci) => (row.tvmTop ?? []).find((entry) => entry.uci === uci)?.visits;
          const visitTie = visitsOf(row.tvmMove) !== undefined && visitsOf(row.tvmMove) === visitsOf(row.ortMove);
          if (tieEps !== undefined && (visitTie || (Number.isFinite(gap) && gap <= tieEps))) {
            tolerated++;
            artifact.tieTolerated.search.push({ fen: row.fen, tvm: row.tvmMove, ort: row.ortMove, priorGap: Number.isFinite(gap) ? gap : undefined, visitTie });
          }
        }
        if (search.moveMatches + tolerated !== searchComparable) {
          artifact.ok = false;
          artifact.error = `search parity failed: ${search.moveMatches}/${searchComparable}`;
        }
      }
      if (args.out) {
        await mkdir(dirname(args.out), { recursive: true });
        await writeFile(args.out, `${JSON.stringify(artifact, null, 2)}\n`);
      }
      console.log(JSON.stringify(artifact, null, 2));
      if (!artifact.ok) process.exitCode = 1;
    } finally {
      await closeSession(args, session);
    }
  } finally {
    if (server) server.kill('SIGTERM');
  }
}

main().catch((error) => {
  console.error(error?.stack || error?.message || String(error));
  process.exit(1);
});
