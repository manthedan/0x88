#!/usr/bin/env node
import { spawn } from 'node:child_process';
import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname } from 'node:path';
import { parsePgnGames } from '../src/chess/pgn.ts';

const DEFAULT_OUT = 'artifacts/tvm/lc0_tvmjs_webgpu_fixed_suite_research.json';
const DEFAULT_HOST = '127.0.0.1';
const DEFAULT_PORT = 5291;
const DEFAULT_TIMEOUT_MS = 180_000;
const DEFAULT_AGENT_BROWSER = process.env.AGENT_BROWSER_BIN ?? 'agent-browser';

function usage() {
  console.log(`Usage: node --experimental-strip-types scripts/lc0_tvmjs_webgpu_fixed_suite.mjs [options]\n\nRuns a production-style fixed-position research bridge for the isolated LC0 TVMJS/WebGPU path.\nIt does not add TVMJS to the stable runtime registry or arena runtime UI; it delegates to lc0_tvmjs_webgpu_smoke.mjs and emits a fixed-suite-style report.\n\nOptions:\n  --fens FILE               Newline-separated FEN suite\n  --source-report PATH      Existing runtime arena JSON to derive LC0-to-move FENs from\n  --source-runtime NAME     Runtime in source report (default hybrid-wgsl-heads)\n  --source-game N           1-based game number in source runtime PGN (default 2)\n  --max-positions N         Max fixed positions to run (default 16)\n  --skip-plies N            Ignore source PGN positions before this absolute ply (default 0)\n  --batch N                 TVMJS batch artifact: 1, 4, or 8 (default 8)\n  --visits N                Fixed search visits per position (default 16)\n  --repeats N               Repeat each search row for timing stability (default 1)\n  --ort-compare MODE        ORT comparison: f16, f32, both, none (default f16)\n  --ort-ep EP               ORT EP for comparison: webgpu, wasm, webgpu,wasm (default webgpu)\n  --stockfish-score-depth N Score TVMJS/ORT post-search moves at fixed Stockfish depth\n  --stockfish-score-ms N    Score TVMJS/ORT post-search moves by movetime\n  --suite-out PATH          Write normalized FEN suite to PATH (default next to --out)\n  --smoke-out PATH          Child TVMJS smoke artifact path (default next to --out)\n  --report-out PATH         Fixed-suite-style report path (default next to --out)\n  --out PATH                Aggregate bridge JSON path (default ${DEFAULT_OUT})\n  --base-url URL            Use existing dev server for child smoke\n  --host HOST               Vite host when child starts server (default ${DEFAULT_HOST})\n  --port N                  Vite port when child starts server (default ${DEFAULT_PORT})\n  --timeout MS              Child smoke timeout (default ${DEFAULT_TIMEOUT_MS})\n  --agent-browser BIN       Browser automation binary (default AGENT_BROWSER_BIN or agent-browser)\n  --no-server               Forward --no-server to child smoke\n  --dry-run                 Print resolved commands and artifacts without running or writing\n  -h, --help                Show help\n`);
}

function parseArgs(argv) {
  const args = {
    fensFile: '',
    sourceReport: '',
    sourceRuntime: 'hybrid-wgsl-heads',
    sourceGame: 2,
    maxPositions: 16,
    skipPlies: 0,
    batch: 8,
    visits: 16,
    repeats: 1,
    ortCompare: 'f16',
    ortEp: 'webgpu',
    stockfishScoreDepth: undefined,
    stockfishScoreMs: undefined,
    out: DEFAULT_OUT,
    suiteOut: '',
    smokeOut: '',
    reportOut: '',
    host: DEFAULT_HOST,
    port: DEFAULT_PORT,
    timeoutMs: DEFAULT_TIMEOUT_MS,
    agentBrowser: DEFAULT_AGENT_BROWSER,
    baseUrl: '',
    noServer: false,
    dryRun: false,
  };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    const next = () => { if (i + 1 >= argv.length) throw new Error(`${arg} requires a value`); return argv[++i]; };
    if (arg === '--fens') args.fensFile = next();
    else if (arg === '--source-report') args.sourceReport = next();
    else if (arg === '--source-runtime') args.sourceRuntime = next();
    else if (arg === '--source-game') args.sourceGame = Number(next());
    else if (arg === '--max-positions') args.maxPositions = Number(next());
    else if (arg === '--skip-plies') args.skipPlies = Number(next());
    else if (arg === '--batch') args.batch = Number(next());
    else if (arg === '--visits') args.visits = Number(next());
    else if (arg === '--repeats') args.repeats = Number(next());
    else if (arg === '--ort-compare') args.ortCompare = next();
    else if (arg === '--ort-ep') args.ortEp = next();
    else if (arg === '--stockfish-score-depth') args.stockfishScoreDepth = Number(next());
    else if (arg === '--stockfish-score-ms') args.stockfishScoreMs = Number(next());
    else if (arg === '--suite-out') args.suiteOut = next();
    else if (arg === '--smoke-out') args.smokeOut = next();
    else if (arg === '--report-out') args.reportOut = next();
    else if (arg === '--out') args.out = next();
    else if (arg === '--base-url') { args.baseUrl = next(); args.noServer = true; }
    else if (arg === '--host') args.host = next();
    else if (arg === '--port') args.port = Number(next());
    else if (arg === '--timeout') args.timeoutMs = Number(next());
    else if (arg === '--agent-browser') args.agentBrowser = next();
    else if (arg === '--no-server') args.noServer = true;
    else if (arg === '--dry-run') args.dryRun = true;
    else if (arg === '-h' || arg === '--help') args.help = true;
    else throw new Error(`Unknown option: ${arg}`);
  }
  if (args.help) return args;
  if (!args.fensFile && !args.sourceReport) throw new Error('expected --fens FILE or --source-report PATH');
  if (args.fensFile && args.sourceReport) throw new Error('use only one of --fens or --source-report');
  for (const name of ['sourceGame', 'maxPositions', 'batch', 'visits', 'repeats', 'port', 'timeoutMs']) {
    if (!Number.isFinite(args[name]) || args[name] <= 0) throw new Error(`Invalid ${name}: ${args[name]}`);
  }
  if (!Number.isFinite(args.skipPlies) || args.skipPlies < 0) throw new Error(`Invalid skipPlies: ${args.skipPlies}`);
  if (![1, 4, 8].includes(args.batch)) throw new Error(`Invalid --batch ${args.batch}; expected 1, 4, or 8`);
  if (!['none', 'f16', 'f32', 'both'].includes(args.ortCompare)) throw new Error(`Invalid --ort-compare ${args.ortCompare}`);
  if (!['webgpu', 'wasm', 'webgpu,wasm'].includes(args.ortEp)) throw new Error(`Invalid --ort-ep ${args.ortEp}`);
  if (args.stockfishScoreDepth !== undefined && (!Number.isFinite(args.stockfishScoreDepth) || args.stockfishScoreDepth <= 0)) throw new Error(`Invalid --stockfish-score-depth ${args.stockfishScoreDepth}`);
  if (args.stockfishScoreMs !== undefined && (!Number.isFinite(args.stockfishScoreMs) || args.stockfishScoreMs <= 0)) throw new Error(`Invalid --stockfish-score-ms ${args.stockfishScoreMs}`);
  const stem = args.out.replace(/\.json$/i, '');
  args.suiteOut ||= `${stem}.fens`;
  args.smokeOut ||= `${stem}.smoke.json`;
  args.reportOut ||= `${stem}_report.json`;
  return args;
}

function lc0SideFromTags(tags) {
  if (/\b(lc0|leela)\b/i.test(tags.White ?? '')) return 'w';
  if (/\b(lc0|leela)\b/i.test(tags.Black ?? '')) return 'b';
  return null;
}

async function loadFens(args) {
  if (args.fensFile) {
    return (await readFile(args.fensFile, 'utf8'))
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter((line) => line && !line.startsWith('#'))
      .slice(0, args.maxPositions);
  }
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

function hashLines(lines) {
  return createHash('sha256').update(`${lines.join('\n')}\n`).digest('hex');
}

function spawnCapture(command, commandArgs, options = {}) {
  return new Promise((resolve, reject) => {
    const startedAt = Date.now();
    const { timeoutMs, echoStderr, ...spawnOptions } = options;
    const child = spawn(command, commandArgs, { stdio: ['ignore', 'pipe', 'pipe'], ...spawnOptions });
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
      finish(resolve, { stdout, stderr, elapsedMs: Date.now() - startedAt });
    });
  });
}

function childCommands(args) {
  const smoke = ['scripts/lc0_tvmjs_webgpu_smoke.mjs', '--batch', String(args.batch), '--fixtures', '--fixture-count', String(args.fenCount), '--fens', args.suiteOut, '--ort-compare', args.ortCompare, '--ort-ep', args.ortEp, '--search-visits', String(args.visits), '--search-fixtures', String(args.fenCount), '--search-repeats', String(args.repeats), '--timeout', String(args.timeoutMs), '--agent-browser', args.agentBrowser, '--out', args.smokeOut];
  if (args.stockfishScoreDepth !== undefined) smoke.push('--stockfish-score-depth', String(args.stockfishScoreDepth));
  if (args.stockfishScoreMs !== undefined) smoke.push('--stockfish-score-ms', String(args.stockfishScoreMs));
  if (args.baseUrl) smoke.push('--base-url', args.baseUrl);
  else {
    smoke.push('--host', args.host, '--port', String(args.port));
    if (args.noServer) smoke.push('--no-server');
  }
  const summarize = ['scripts/summarize_lc0_tvmjs_webgpu_smoke.mjs', '--in', args.smokeOut, '--out', args.reportOut];
  return { smoke: ['node', ...smoke], summarize: ['node', ...summarize] };
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) { usage(); return; }
  const fens = await loadFens(args);
  if (!fens.length) throw new Error('No fixed-suite FENs resolved');
  args.fenCount = fens.length;
  const fensSha256 = hashLines(fens);
  const commands = childCommands(args);
  if (args.dryRun) {
    console.log(JSON.stringify({
      schema: 'lc0_browser.tvmjs_webgpu_fixed_suite_research_bridge.dry_run.v1',
      ok: true,
      researchOnly: true,
      noStableRuntimePromotion: true,
      fens: { count: fens.length, sha256: fensSha256, wouldWrite: args.suiteOut },
      artifacts: { aggregate: args.out, smoke: args.smokeOut, report: args.reportOut, suite: args.suiteOut },
      commands,
    }, null, 2));
    return;
  }

  await mkdir(dirname(args.suiteOut), { recursive: true });
  await writeFile(args.suiteOut, `${fens.join('\n')}\n`);
  process.stderr.write(`[lc0-tvmjs-fixed-suite] smoke -> ${args.smokeOut}\n`);
  const smokeRun = await spawnCapture(commands.smoke[0], commands.smoke.slice(1), { timeoutMs: args.timeoutMs + 60_000, echoStderr: true });
  process.stderr.write(`[lc0-tvmjs-fixed-suite] report -> ${args.reportOut}\n`);
  const reportRun = await spawnCapture(commands.summarize[0], commands.summarize.slice(1), { timeoutMs: 30_000, echoStderr: true });
  const smokeArtifact = JSON.parse(await readFile(args.smokeOut, 'utf8'));
  const report = JSON.parse(await readFile(args.reportOut, 'utf8'));
  const aggregate = {
    schema: 'lc0_browser.tvmjs_webgpu_fixed_suite_research_bridge.v1',
    generatedAt: new Date().toISOString(),
    ok: smokeArtifact.ok === true && report.status === 'LC0_TVMJS_FIXED_SUITE_SMOKE_DONE',
    researchOnly: true,
    noStableRuntimePromotion: true,
    caveats: [
      'TVMJS remains isolated in research scripts/docs and is not added to src/nn/runtimeRegistry.ts, browserRuntimeEvaluator, or the stable arena runtime UI.',
      'This bridge mirrors fixed-suite inputs and report shape, but delegates execution to lc0_tvmjs_webgpu_smoke.mjs until a promotion-grade runtime integration exists.',
      'Generated TVMJS wasm/runtime artifacts remain local/ignored unless release policy changes.',
    ],
    source: args.fensFile ? { type: 'fens', path: args.fensFile } : { type: 'source-report', path: args.sourceReport, runtime: args.sourceRuntime, game: args.sourceGame, skipPlies: args.skipPlies },
    config: { batch: args.batch, visits: args.visits, repeats: args.repeats, ortCompare: args.ortCompare, ortEp: args.ortEp, stockfishScoreDepth: args.stockfishScoreDepth, stockfishScoreMs: args.stockfishScoreMs },
    fens: { count: fens.length, sha256: fensSha256, path: args.suiteOut },
    artifacts: { smoke: args.smokeOut, report: args.reportOut, suite: args.suiteOut },
    commands,
    wallTimeMs: { smokeChild: smokeRun.elapsedMs, reportChild: reportRun.elapsedMs },
    summary: report.summary,
    reportSchema: report.schema,
    smokeSchema: smokeArtifact.schema,
  };
  await mkdir(dirname(args.out), { recursive: true });
  await writeFile(args.out, `${JSON.stringify(aggregate, null, 2)}\n`);
  console.log(JSON.stringify(aggregate, null, 2));
  if (!aggregate.ok) process.exitCode = 1;
}

main().catch((error) => {
  console.error(error?.stack ?? error?.message ?? String(error));
  process.exit(1);
});
