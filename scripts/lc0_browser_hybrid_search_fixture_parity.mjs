#!/usr/bin/env node
import { spawn } from 'node:child_process';
import { mkdir, writeFile } from 'node:fs/promises';
import { dirname } from 'node:path';
import { setTimeout as delay } from 'node:timers/promises';

const DEFAULT_HOST = '127.0.0.1';
const DEFAULT_PORT = 5179;
const DEFAULT_TIMEOUT_MS = 300_000;

function usage() {
  console.log(`Usage: node scripts/lc0_browser_hybrid_search_fixture_parity.mjs [options]\n\nRuns browser/WebGPU LC0 fixed-search fixture parity for hybrid search across batch pipeline depths.\n\nOptions:\n  --out PATH                 Write JSON artifact (default stdout only)\n  --base-url URL             Use an existing dev server\n  --host HOST                Vite host when auto-starting (default ${DEFAULT_HOST})\n  --port N                   Vite port when auto-starting (default ${DEFAULT_PORT})\n  --agent-browser BIN        Browser automation binary (default AGENT_BROWSER_BIN or agent-browser)\n  --session NAME             agent-browser session name\n  --timeout MS               Total browser wait timeout (default ${DEFAULT_TIMEOUT_MS})\n  --visits LIST              Comma-separated fixed-visit fixture sets, e.g. 32,64 (default 32)\n  --batch N                  Search leaf batch size (default 4)\n  --batch-pipeline-depths L  Comma-separated depths (default 1; use 1,2,4 with --allow-mismatches for exploratory pipeline matrices)\n  --repeats N                Repeats per fixture/depth (default 1)\n  --fixture-limit N          Fixtures per visit set (max currently 16, default 16)\n  --fixture-ids LIST         Comma-separated native fixture IDs to run before applying --fixture-limit\n  --trace-root-children      Include depth-baseline/root child visit/prior/q traces in the browser artifact\n  --trace-search-visits      Include per-batch selection/backup search trace in the browser artifact\n  --layers N                 Encoder layers (default 10)\n  --head-backend MODE        ort or wgsl (default ort; use wgsl to opt into experimental WGSL heads)\n  --encoder-kernel MODE      hand, mixed-tvm-ffn, mixed-tvm-ffn-outproj, tvm-packed-f16 (default hand)\n  --input-backend MODE       js, wgsl, or wasm (default js)\n  --legal-priors-backend MODE\n                            Legal-prior backend: js, wasm, or gpu (default js; gpu requires WGSL heads; opt-in)\n  --max-depth-visit-l1 N     Optional fail gate on root visit-distribution L1 vs depth=1 baseline\n  --no-server                Do not auto-start Vite\n  --allow-mismatches         Exit 0 and write/report artifacts even when parity mismatches are found\n  --dry-run                  Print URL and exit\n  -h, --help                 Show this help\n`);
}

function parseList(raw, mapper = Number, label = 'list') {
  const values = String(raw).split(',').map((entry) => entry.trim()).filter(Boolean).map(mapper);
  if (!values.length) throw new Error(`Invalid --${label}: empty list`);
  return values;
}

function parseArgs(argv) {
  const args = {
    host: DEFAULT_HOST,
    port: DEFAULT_PORT,
    timeoutMs: DEFAULT_TIMEOUT_MS,
    agentBrowser: process.env.AGENT_BROWSER_BIN ?? 'agent-browser',
    session: process.env.AGENT_BROWSER_SESSION ?? `lc0-search-fixture-parity-${process.pid}`,
    visits: [32],
    batch: 4,
    batchPipelineDepths: [1],
    repeats: 1,
    fixtureLimit: 16,
    fixtureIds: [],
    traceRootChildren: false,
    traceSearchVisits: false,
    layers: 10,
    headBackend: 'ort',
    encoderKernel: 'hand',
    inputBackend: 'js',
    legalPriorsBackend: 'js',
    packVerify: false,
    maxDepthVisitL1: undefined,
    allowMismatches: false,
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
    if (arg === '--out') args.out = next();
    else if (arg === '--base-url') { args.baseUrl = next(); args.explicitBaseUrl = true; }
    else if (arg === '--host') args.host = next();
    else if (arg === '--port') args.port = Number(next());
    else if (arg === '--agent-browser') args.agentBrowser = next();
    else if (arg === '--session') args.session = next();
    else if (arg === '--timeout') args.timeoutMs = Number(next());
    else if (arg === '--visits') args.visits = parseList(next(), Number, 'visits');
    else if (arg === '--batch') args.batch = Number(next());
    else if (arg === '--batch-pipeline-depths' || arg === '--pipeline-depths') args.batchPipelineDepths = parseList(next(), Number, 'batch-pipeline-depths');
    else if (arg === '--repeats') args.repeats = Number(next());
    else if (arg === '--fixture-limit' || arg === '--fixtures') args.fixtureLimit = Number(next());
    else if (arg === '--fixture-ids') args.fixtureIds = next().split(',').map((value) => value.trim()).filter(Boolean);
    else if (arg === '--trace-root-children') args.traceRootChildren = true;
    else if (arg === '--trace-search-visits') args.traceSearchVisits = true;
    else if (arg === '--layers') args.layers = Number(next());
    else if (arg === '--head-backend') args.headBackend = next();
    else if (arg === '--encoder-kernel') args.encoderKernel = next();
    else if (arg === '--input-backend') args.inputBackend = next();
    else if (arg === '--legal-priors-backend' || arg === '--hybrid-legal-priors') args.legalPriorsBackend = next();
    else if (arg === '--pack-verify') args.packVerify = true;
    else if (arg === '--max-depth-visit-l1' || arg === '--max-depth-baseline-visit-l1') args.maxDepthVisitL1 = Number(next());
    else if (arg === '--allow-mismatches') args.allowMismatches = true;
    else if (arg === '--no-server') args.noServer = true;
    else if (arg === '--dry-run') args.dryRun = true;
    else if (arg === '-h' || arg === '--help') args.help = true;
    else throw new Error(`Unknown option: ${arg}`);
  }
  if (!args.baseUrl) args.baseUrl = `http://${args.host}:${args.port}`;
  if (args.explicitBaseUrl) args.noServer = true;
  if (!['ort', 'wgsl'].includes(args.headBackend)) throw new Error(`Invalid --head-backend: ${args.headBackend}`);
  if (!['js', 'wgsl', 'wasm'].includes(args.inputBackend)) throw new Error(`Invalid --input-backend: ${args.inputBackend}`);
  if (!['js', 'wasm', 'gpu'].includes(args.legalPriorsBackend)) throw new Error(`Invalid --legal-priors-backend: ${args.legalPriorsBackend}`);
  if (args.legalPriorsBackend === 'gpu' && args.headBackend !== 'wgsl') throw new Error('--legal-priors-backend gpu requires --head-backend wgsl');
  if (!['hand', 'tvm-packed-f16', 'mixed-tvm-ffn', 'mixed-tvm-ffn-outproj'].includes(args.encoderKernel)) throw new Error(`Invalid --encoder-kernel: ${args.encoderKernel}`);
  for (const [name, value] of [['port', args.port], ['timeout', args.timeoutMs], ['batch', args.batch], ['repeats', args.repeats], ['fixture-limit', args.fixtureLimit], ['layers', args.layers]]) {
    if (!Number.isFinite(value) || value <= 0) throw new Error(`Invalid --${name}: ${value}`);
  }
  if (args.maxDepthVisitL1 !== undefined && (!Number.isFinite(args.maxDepthVisitL1) || args.maxDepthVisitL1 < 0 || args.maxDepthVisitL1 > 2)) throw new Error(`Invalid --max-depth-visit-l1: ${args.maxDepthVisitL1}`);
  for (const [name, values] of [['visits', args.visits], ['batch-pipeline-depths', args.batchPipelineDepths]]) {
    if (values.some((value) => !Number.isFinite(value) || value <= 0)) throw new Error(`Invalid --${name}: ${values.join(',')}`);
  }
  args.batchPipelineDepths = [1, ...args.batchPipelineDepths.filter((depth) => depth !== 1)];
  return args;
}

function parityUrl(args) {
  const url = new URL('/lc0-policy-only.html', args.baseUrl);
  url.searchParams.set('hybridSearchFixtureParity', '1');
  url.searchParams.set('runtime', 'hybrid');
  url.searchParams.set('headBackend', args.headBackend);
  if (args.headBackend === 'wgsl') url.searchParams.set('wgslBatchMode', 'physical');
  if (args.inputBackend !== 'js') url.searchParams.set('inputBackend', args.inputBackend);
  if (args.legalPriorsBackend !== 'js') url.searchParams.set('legalPriorsBackend', args.legalPriorsBackend);
  if (args.encoderKernel !== 'hand') url.searchParams.set('encoderKernel', args.encoderKernel);
  url.searchParams.set('encoderLayers', String(args.layers));
  url.searchParams.set('visits', args.visits.join(','));
  url.searchParams.set('batch', String(args.batch));
  url.searchParams.set('batchPipelineDepths', args.batchPipelineDepths.join(','));
  url.searchParams.set('repeats', String(args.repeats));
  url.searchParams.set('fixtureLimit', String(args.fixtureLimit));
  if (args.fixtureIds.length) url.searchParams.set('fixtureIds', args.fixtureIds.join(','));
  if (args.traceRootChildren) url.searchParams.set('traceRootChildren', '1');
  if (args.traceSearchVisits) url.searchParams.set('traceSearchVisits', '1');
  url.searchParams.set('ep', 'wasm');
  if (!args.packVerify) url.searchParams.set('packVerify', '0');
  return String(url);
}

function runAgent(args, commandArgs, timeoutMs = 30_000) {
  const fullArgs = ['--json', '--session', args.session, ...commandArgs];
  return new Promise((resolve, reject) => {
    const child = spawn(args.agentBrowser, fullArgs, { stdio: ['ignore', 'pipe', 'pipe'] });
    const chunks = { stdout: [], stderr: [] };
    let settled = false;
    const finish = (fn, value) => { if (!settled) { settled = true; clearTimeout(timer); fn(value); } };
    const timer = setTimeout(() => { child.kill('SIGKILL'); finish(reject, new Error(`${args.agentBrowser} ${fullArgs.slice(1).join(' ')} timed out after ${timeoutMs}ms`)); }, timeoutMs);
    child.stdout.on('data', (chunk) => chunks.stdout.push(chunk));
    child.stderr.on('data', (chunk) => chunks.stderr.push(chunk));
    child.on('error', (error) => finish(reject, error));
    child.on('close', (status) => {
      const stdout = Buffer.concat(chunks.stdout).toString('utf8');
      const stderr = Buffer.concat(chunks.stderr).toString('utf8');
      if (status !== 0) return finish(reject, new Error(`${args.agentBrowser} failed: ${stderr || stdout}`));
      try {
        const parsed = stdout ? JSON.parse(stdout.trim()) : null;
        if (parsed && typeof parsed === 'object' && 'success' in parsed) {
          if (parsed.success === false) return finish(reject, new Error(`${args.agentBrowser} failed: ${parsed.error ?? stdout}`));
          return finish(resolve, parsed.data ?? parsed);
        }
        return finish(resolve, parsed);
      } catch (error) { return finish(reject, error); }
    });
  });
}

async function closeAgentSession(args) {
  try { await runAgent(args, ['close'], 5_000); }
  catch (error) { process.stderr.write(`[lc0-search-fixture-parity] warning: failed to close session: ${error.message ?? error}\n`); }
}

async function waitForServer(baseUrl, timeoutMs = 30_000) {
  const deadline = Date.now() + timeoutMs;
  let lastError;
  while (Date.now() < deadline) {
    try {
      const response = await fetch(new URL('/lc0-policy-only.html', baseUrl), { cache: 'no-store' });
      if (response.ok) return;
      lastError = new Error(`HTTP ${response.status}`);
    } catch (error) { lastError = error; }
    await delay(250);
  }
  throw new Error(`Vite dev server did not become ready at ${baseUrl}: ${lastError?.message ?? 'timeout'}`);
}

function startServer(args) {
  if (args.noServer) return null;
  const server = spawn('npm', ['run', 'web:client', '--', '--host', args.host, '--port', String(args.port), '--strictPort'], { stdio: ['ignore', 'pipe', 'pipe'] });
  server.stdout.on('data', (chunk) => process.stderr.write(`[vite] ${chunk}`));
  server.stderr.on('data', (chunk) => process.stderr.write(`[vite] ${chunk}`));
  return server;
}

function textFromGetResult(result) {
  if (typeof result?.text === 'string') return result.text;
  if (typeof result === 'string') return result;
  throw new Error(`agent-browser get text returned unexpected payload: ${JSON.stringify(result)}`);
}

function omitVerboseTrace(cell) {
  if (!cell || typeof cell !== 'object' || !('searchTrace' in cell)) return cell;
  const { searchTrace: _searchTrace, ...rest } = cell;
  return rest;
}

async function runBrowserParity(args) {
  const url = parityUrl(args);
  process.stderr.write(`[lc0-search-fixture-parity] ${url}\n`);
  try {
    await runAgent(args, ['open', url], 30_000);
    const deadline = Date.now() + args.timeoutMs;
    while (Date.now() < deadline) {
      const chunk = Math.min(25_000, Math.max(1000, deadline - Date.now()));
      let doneSeen = false;
      try {
        await runAgent(args, ['wait', '--text', 'HYBRID_SEARCH_FIXTURE_PARITY_DONE', '--timeout', String(chunk)], chunk + 5_000);
        doneSeen = true;
        const text = textFromGetResult(await runAgent(args, ['get', 'text', '#benchResult'], 30_000));
        const result = JSON.parse(text);
        if (result.status !== 'HYBRID_SEARCH_FIXTURE_PARITY_DONE') throw new Error(`unexpected status: ${result.status}`);
        const expectedBackend = args.headBackend === 'wgsl' ? 'lc0web-wgsl-encoder-wgsl-heads' : 'lc0web-wgsl-encoder-ort-heads';
        if (result.backend !== expectedBackend) throw new Error(`unexpected backend: ${result.backend}`);
        if ((result.encoderKernelVariant ?? 'hand') !== args.encoderKernel) throw new Error(`unexpected encoder kernel: ${result.encoderKernelVariant ?? 'hand'}`);
        return result;
      } catch (error) {
        if (doneSeen || Date.now() >= deadline) throw error;
      }
    }
    throw new Error(`Timed out waiting for HYBRID_SEARCH_FIXTURE_PARITY_DONE after ${args.timeoutMs}ms`);
  } finally { await closeAgentSession(args); }
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) return usage();
  if (args.dryRun) { console.log(parityUrl(args)); return; }
  const server = startServer(args);
  try {
    await waitForServer(args.baseUrl);
    const result = await runBrowserParity(args);
    if (args.out) {
      await mkdir(dirname(args.out), { recursive: true });
      await writeFile(args.out, JSON.stringify(result, null, 2));
    }
    const summary = { status: result.status, out: args.out, cells: result.cells, nativeMatches: result.nativeMatches, depthBaselineMatches: result.depthBaselineMatches, maxDepthBaselineVisitL1: result.maxDepthBaselineVisitL1, mismatches: result.mismatches?.map(omitVerboseTrace) };
    console.log(JSON.stringify(summary, null, 2));
    const visitL1Failed = args.maxDepthVisitL1 !== undefined && Number(result.maxDepthBaselineVisitL1 ?? 0) > args.maxDepthVisitL1;
    if (!args.allowMismatches && (result.mismatches?.length || result.nativeMatches !== result.cells || result.depthBaselineMatches !== result.cells || visitL1Failed)) {
      const visitL1Message = args.maxDepthVisitL1 !== undefined ? `, max depth visit L1 ${result.maxDepthBaselineVisitL1} > ${args.maxDepthVisitL1}` : '';
      throw new Error(`search fixture parity failed: native ${result.nativeMatches}/${result.cells}, depth baseline ${result.depthBaselineMatches}/${result.cells}${visitL1Message}; pass --allow-mismatches for exploratory artifact capture`);
    }
  } finally { server?.kill('SIGTERM'); }
}

main().catch((error) => { console.error(error.stack ?? error.message); process.exit(1); });
