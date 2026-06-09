#!/usr/bin/env node
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname } from 'node:path';

function usage() {
  console.log(`Usage: node scripts/summarize_lc0_tvmjs_webgpu_smoke.mjs --in ARTIFACT.json [--out REPORT.json]\n\nConverts a lc0_tvmjs_webgpu_smoke artifact into a compact fixed-suite-style research report.\nThis does not add Stockfish post-move scoring; it preserves that limitation explicitly.\n`);
}

function parseArgs(argv) {
  const args = { in: '', out: '' };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    const next = () => { if (i + 1 >= argv.length) throw new Error(`${arg} requires a value`); return argv[++i]; };
    if (arg === '--in') args.in = next();
    else if (arg === '--out') args.out = next();
    else if (arg === '-h' || arg === '--help') args.help = true;
    else throw new Error(`Unknown option: ${arg}`);
  }
  if (!args.help && !args.in) throw new Error('expected --in ARTIFACT.json');
  return args;
}

function stats(values) {
  const finite = values.filter(Number.isFinite).sort((a, b) => a - b);
  if (!finite.length) return { count: 0, minMs: null, meanMs: null, medianMs: null, maxMs: null };
  const sum = finite.reduce((a, b) => a + b, 0);
  const mid = Math.floor(finite.length / 2);
  return {
    count: finite.length,
    minMs: finite[0],
    meanMs: sum / finite.length,
    medianMs: finite.length % 2 ? finite[mid] : (finite[mid - 1] + finite[mid]) / 2,
    maxMs: finite[finite.length - 1],
  };
}

function groupedPositions(rows) {
  const byFen = new Map();
  for (const row of rows) {
    if (!byFen.has(row.fen)) byFen.set(row.fen, []);
    byFen.get(row.fen).push(row);
  }
  return [...byFen.entries()].map(([fen, entries], index) => ({
    index,
    fen,
    repeats: entries.length,
    tvmMoves: [...new Set(entries.map((row) => row.tvmMove))],
    ortMoves: [...new Set(entries.map((row) => row.ortMove))],
    moveMatches: entries.filter((row) => row.moveMatches).length,
    comparable: entries.length,
    tvmMs: stats(entries.map((row) => row.tvmMs)),
    ortMs: stats(entries.map((row) => row.ortMs)),
    stockfish: entries.some((row) => row.stockfish) ? {
      tvmMinusOrtCp: stats(entries.map((row) => row.stockfishCpDeltaTvmMinusOrt)),
      scoredRows: entries.filter((row) => row.stockfish?.tvm || row.stockfish?.ort).length,
    } : undefined,
    rows: entries.map((row) => ({
      repeat: row.repeat ?? 0,
      tvmMove: row.tvmMove,
      ortMove: row.ortMove,
      moveMatches: row.moveMatches,
      tvmMs: row.tvmMs,
      ortMs: row.ortMs,
      tvmAfterFen: row.tvmAfterFen,
      ortAfterFen: row.ortAfterFen,
      stockfishCpDeltaTvmMinusOrt: row.stockfishCpDeltaTvmMinusOrt,
      stockfish: row.stockfish,
    })),
  }));
}

function buildReport(artifact, sourcePath) {
  const result = artifact.result ?? {};
  const search = result.searchParity ?? {};
  const rows = search.rows ?? [];
  const ortF16 = result.ortComparisons?.f16;
  const searchRows = search.searchRows ?? rows.length;
  const positions = groupedPositions(rows);
  const hasStockfish = Boolean(search.stockfish);
  return {
    schema: hasStockfish ? 'lc0_browser.tvmjs_webgpu_fixed_suite_report.v2' : 'lc0_browser.tvmjs_webgpu_fixed_suite_report.v1',
    status: 'LC0_TVMJS_FIXED_SUITE_SMOKE_DONE',
    generatedAt: new Date().toISOString(),
    sourceArtifact: sourcePath,
    caveats: [
      'Research-only TVMJS/WebGPU path; stable/default ORT WebGPU runtime is unchanged.',
      'This report is derived from the TVMJS smoke harness, not lc0_browser_runtime_fixed_suite.mjs.',
      hasStockfish ? 'Stockfish post-move scoring is included for the search rows, but this is still smoke-harness scoring rather than lc0_browser_runtime_fixed_suite.mjs output.' : 'No Stockfish post-move scoring is included; quality comparison is TVMJS-vs-ORT f16 evaluator/search move parity only.',
      'Timing is warm-page browser wall time and should not be treated as promotion-grade throughput evidence.',
    ],
    config: {
      batch: artifact.batch,
      fensFile: artifact.fensFile,
      fixtureOffset: artifact.fixtureOffset,
      fixtureCount: result.fixtureCount ?? artifact.fixtureCount,
      ortCompare: artifact.ortCompare,
      ortEp: artifact.ortEp,
      searchVisits: search.visits ?? artifact.searchVisits,
      searchRepeats: search.repeats ?? artifact.searchRepeats ?? 1,
      searchFixtureCount: search.fixtureCount ?? artifact.searchFixtures,
      stockfishScoreDepth: search.stockfish?.scoreDepth ?? artifact.stockfishScoreDepth,
      stockfishScoreMs: search.stockfish?.scoreMovetimeMs ?? artifact.stockfishScoreMs,
      url: artifact.url,
    },
    summary: [
      {
        runtime: 'tvmjs-webgpu-f16',
        positions: positions.length,
        searchRows,
        visitsPerSearch: search.visits ?? null,
        moveMatchesVsOrtF16: search.moveMatches ?? null,
        comparableSearchRows: searchRows,
        searchMs: search.tvmTiming ?? stats(rows.map((row) => row.tvmMs)),
        evaluatorBestMoveMatchesNative: result.nativeComparable ? result.bestMoveMatches : null,
        evaluatorNativeComparable: result.nativeComparable ?? 0,
        evaluatorBestMoveMatchesOrtF16: ortF16?.bestMoveMatches ?? null,
        evaluatorOrtF16Comparable: ortF16?.comparable ?? null,
        stockfishScoredRows: search.stockfish?.scoredRows ?? null,
        stockfishTvmMinusOrtCp: search.stockfish?.tvmMinusOrtCp ?? null,
      },
      {
        runtime: 'ort-f16-webgpu',
        positions: positions.length,
        searchRows,
        visitsPerSearch: search.visits ?? null,
        searchMs: search.ortTiming ?? stats(rows.map((row) => row.ortMs)),
      },
    ],
    positions,
    originalArtifactOk: artifact.ok,
  };
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) return usage();
  const artifact = JSON.parse(await readFile(args.in, 'utf8'));
  const report = buildReport(artifact, args.in);
  const text = `${JSON.stringify(report, null, 2)}\n`;
  if (args.out) {
    await mkdir(dirname(args.out), { recursive: true });
    await writeFile(args.out, text);
  }
  process.stdout.write(text);
}

main().catch((error) => {
  console.error(error?.stack ?? error?.message ?? String(error));
  process.exit(1);
});
