#!/usr/bin/env node
import { readdir, readFile, writeFile, stat } from 'node:fs/promises';
import { join, relative } from 'node:path';

const DEFAULT_DIR = 'artifacts/tvm';
const DEFAULT_OUT = 'artifacts/tvm/lc0_tvmjs_webgpu_search_smoke_summary.json';

function usage() {
  console.log(`Usage: node scripts/summarize_lc0_tvmjs_webgpu_evidence.mjs [options]\n\nBuilds an aggregate TVMJS/WebGPU research-evidence summary from local smoke/report artifacts.\n\nOptions:\n  --dir PATH       Artifact directory (default ${DEFAULT_DIR})\n  --out PATH       Output summary JSON (default ${DEFAULT_OUT})\n  --no-write       Print summary only\n  --require-all-matches\n                   Fail unless every discovered search row has matching TVMJS/ORT move\n  --min-search-rows N\n                   Fail unless at least N search rows are present\n  --min-stockfish-scored-runs N\n                   Fail unless at least N Stockfish-scored runs are present\n  --min-fixed-suite-reports N\n                   Fail unless at least N fixed-suite-style reports are present\n  -h, --help       Show help\n`);
}

function parseArgs(argv) {
  const args = { dir: DEFAULT_DIR, out: DEFAULT_OUT, write: true, requireAllMatches: false, minSearchRows: 0, minStockfishScoredRuns: 0, minFixedSuiteReports: 0 };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    const next = () => { if (i + 1 >= argv.length) throw new Error(`${arg} requires a value`); return argv[++i]; };
    if (arg === '--dir') args.dir = next();
    else if (arg === '--out') args.out = next();
    else if (arg === '--no-write') args.write = false;
    else if (arg === '--require-all-matches') args.requireAllMatches = true;
    else if (arg === '--min-search-rows') args.minSearchRows = Number(next());
    else if (arg === '--min-stockfish-scored-runs') args.minStockfishScoredRuns = Number(next());
    else if (arg === '--min-fixed-suite-reports') args.minFixedSuiteReports = Number(next());
    else if (arg === '-h' || arg === '--help') args.help = true;
    else throw new Error(`Unknown option: ${arg}`);
  }
  for (const name of ['minSearchRows', 'minStockfishScoredRuns', 'minFixedSuiteReports']) {
    if (!Number.isFinite(args[name]) || args[name] < 0) throw new Error(`Invalid ${name}: ${args[name]}`);
  }
  return args;
}

async function exists(path) {
  try { await stat(path); return true; } catch { return false; }
}

async function loadJson(path) {
  return JSON.parse(await readFile(path, 'utf8'));
}

function artifactPath(dir, file) {
  return relative(process.cwd(), join(dir, file));
}

async function discoverSmokeArtifacts(dir) {
  const files = await readdir(dir);
  return files
    .filter((file) => file.startsWith('lc0_tvmjs_webgpu_search_smoke_') && file.endsWith('.json') && file !== 'lc0_tvmjs_webgpu_search_smoke_summary.json')
    .sort()
    .map((file) => artifactPath(dir, file));
}

async function discoverReports(dir) {
  const files = await readdir(dir);
  const reports = [];
  for (const file of files.filter((item) => item.startsWith('lc0_tvmjs_webgpu_fixed_suite_') && item.endsWith('_report.json')).sort()) {
    const path = artifactPath(dir, file);
    const report = await loadJson(path);
    reports.push({
      artifact: path,
      sourceArtifact: report.sourceArtifact,
      schema: report.schema,
      caveat: report.caveats?.find((entry) => /Stockfish|No Stockfish/i.test(entry)) ?? report.caveats?.[0],
    });
  }
  return reports;
}

function compactRun(path, artifact) {
  const result = artifact.result ?? {};
  const search = result.searchParity ?? {};
  const ortF16 = result.ortComparisons?.f16;
  return {
    artifact: path,
    fixtureCount: result.fixtureCount,
    nativeMatches: result.bestMoveMatches,
    nativeComparable: result.nativeComparable,
    ortF16Matches: ortF16?.bestMoveMatches,
    ortF16Comparable: ortF16?.comparable,
    searchVisits: search.visits,
    searchRepeats: search.repeats ?? 1,
    searchRows: search.searchRows ?? search.rows?.length ?? search.fixtureCount,
    searchMoveMatches: search.moveMatches,
    searchFixtureCount: search.fixtureCount,
    tvmTiming: search.tvmTiming,
    ortTiming: search.ortTiming,
  };
}

function scoredRun(path, artifact) {
  const result = artifact.result ?? {};
  const search = result.searchParity ?? {};
  if (!search.stockfish) return null;
  return {
    ...compactRun(path, artifact),
    scoreDepth: search.stockfish.scoreDepth,
    scoreMovetimeMs: search.stockfish.scoreMovetimeMs,
    scoredRows: search.stockfish.scoredRows,
    uniqueScoredPositions: search.stockfish.uniqueScoredPositions,
    tvmMinusOrtCp: search.stockfish.tvmMinusOrtCp,
  };
}

async function buildSummary(args) {
  if (!await exists(args.dir)) throw new Error(`Artifact directory not found: ${args.dir}`);
  const inputs = await discoverSmokeArtifacts(args.dir);
  const artifacts = [];
  for (const path of inputs) artifacts.push([path, await loadJson(path)]);
  const searchRows = [];
  for (const [path, artifact] of artifacts) {
    const search = artifact.result?.searchParity ?? {};
    for (const row of search.rows ?? []) searchRows.push({ ...row, artifact: path, visits: search.visits });
  }
  const allCurrentFixtureRuns = [];
  const arbitraryFenRuns = [];
  const stockfishScoredRuns = [];
  for (const [path, artifact] of artifacts) {
    const compact = compactRun(path, artifact);
    if ((artifact.result?.fixtureCount ?? 0) === 10 && !artifact.fensFile) allCurrentFixtureRuns.push(compact);
    if (artifact.fensFile) arbitraryFenRuns.push({ ...compact, fensFile: artifact.fensFile });
    const scored = scoredRun(path, artifact);
    if (scored) stockfishScoredRuns.push(scored);
  }
  const failedInputs = artifacts.filter(([, artifact]) => artifact.ok === false).map(([path]) => path);
  return {
    schema: 'lc0_browser.tvmjs_webgpu_search_smoke_summary.v9',
    generatedAt: new Date().toISOString(),
    inputs,
    failedInputs,
    ok: failedInputs.length === 0,
    searchRows: searchRows.length,
    moveMatches: searchRows.filter((row) => row.moveMatches).length,
    visits: [...new Set(searchRows.map((row) => row.visits).filter((value) => value !== undefined))].sort((a, b) => a - b),
    allCurrentFixtureRuns,
    arbitraryFenRuns,
    stockfishScoredRuns,
    fixedSuiteStyleReports: await discoverReports(args.dir),
    maxTvmMs: Math.max(...searchRows.map((row) => row.tvmMs).filter(Number.isFinite), 0),
    maxOrtMs: Math.max(...searchRows.map((row) => row.ortMs).filter(Number.isFinite), 0),
    rows: searchRows.map((row) => ({
      artifact: row.artifact,
      repeat: row.repeat ?? 0,
      visits: row.visits,
      fen: row.fen,
      tvmMove: row.tvmMove,
      ortMove: row.ortMove,
      moveMatches: row.moveMatches,
      tvmMs: row.tvmMs,
      ortMs: row.ortMs,
      stockfishCpDeltaTvmMinusOrt: row.stockfishCpDeltaTvmMinusOrt,
    })),
  };
}

function checkSummary(summary, args) {
  const failures = [];
  if (!summary.ok) failures.push(`failed smoke artifacts: ${summary.failedInputs.join(', ')}`);
  if (args.requireAllMatches && summary.moveMatches !== summary.searchRows) failures.push(`move matches ${summary.moveMatches}/${summary.searchRows}`);
  if (summary.searchRows < args.minSearchRows) failures.push(`search rows ${summary.searchRows} < ${args.minSearchRows}`);
  if (summary.stockfishScoredRuns.length < args.minStockfishScoredRuns) failures.push(`Stockfish-scored runs ${summary.stockfishScoredRuns.length} < ${args.minStockfishScoredRuns}`);
  if (summary.fixedSuiteStyleReports.length < args.minFixedSuiteReports) failures.push(`fixed-suite-style reports ${summary.fixedSuiteStyleReports.length} < ${args.minFixedSuiteReports}`);
  return failures;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) return usage();
  const summary = await buildSummary(args);
  const failures = checkSummary(summary, args);
  const text = `${JSON.stringify(summary, null, 2)}\n`;
  if (args.write) await writeFile(args.out, text);
  process.stdout.write(text);
  if (failures.length) {
    console.error(`TVMJS evidence summary check failed: ${failures.join('; ')}`);
    process.exitCode = 1;
  }
}

main().catch((error) => {
  console.error(error?.stack ?? error?.message ?? String(error));
  process.exit(1);
});
