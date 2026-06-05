#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

FENS="${LC0_AR_FENS:-artifacts/lc0_runtime_arena_20260605/batch_matrix/fixed_suite_32_fens.txt}"
MAX_POSITIONS="${LC0_AR_MAX_POSITIONS:-16}"
MOVETIME_MS="${LC0_AR_MOVETIME_MS:-500}"
REPS="${LC0_AR_REPS:-3}"
BATCH_SIZE="${LC0_AR_BATCH_SIZE:-4}"
PIPELINE_DEPTH="${LC0_AR_PIPELINE_DEPTH:-1}"
SCORE_DEPTH="${LC0_AR_SCORE_DEPTH:-1}"
TIMEOUT_MS="${LC0_AR_TIMEOUT_MS:-120000}"
RUNTIME="${LC0_AR_RUNTIME:-hybrid-wgsl-heads}"
OUT_DIR="${LC0_AR_OUT_DIR:-/tmp/lc0_autoresearch}"
RUN_ID="${LC0_AR_RUN_ID:-$(date +%Y%m%d_%H%M%S)_$$}"

if [[ ! -f "$FENS" ]]; then
  echo "missing FEN corpus: $FENS" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

# Fast precheck: catch TypeScript syntax/transpile issues in the touched hot file
# before spending browser time. Full type/build checks live in autoresearch.checks.sh.
node --experimental-strip-types -e "import('./src/lc0/wgslMatmulAddProbe.ts').then(()=>{}).catch(e=>{ console.error(e); process.exit(1); })" >/dev/null

json_paths=()
for rep in $(seq 1 "$REPS"); do
  out="$OUT_DIR/${RUN_ID}_rep${rep}.json"
  log="$OUT_DIR/${RUN_ID}_rep${rep}.log"
  session="lc0-autoresearch-${RUN_ID}-rep${rep}"
  json_paths+=("$out")
  node --experimental-strip-types scripts/lc0_browser_runtime_fixed_suite.mjs \
    --fens "$FENS" \
    --max-positions "$MAX_POSITIONS" \
    --runtimes "$RUNTIME" \
    --movetime "$MOVETIME_MS" \
    --stockfish-score-depth "$SCORE_DEPTH" \
    --lc0-batch-size "$BATCH_SIZE" \
    --batch-pipeline-depth "$PIPELINE_DEPTH" \
    --session "$session" \
    --out "$out" \
    --summary-only \
    --timeout "$TIMEOUT_MS" >"$log" 2>&1
  test -s "$out"
done

node - "${json_paths[@]}" <<'NODE'
const fs = require('fs');
const paths = process.argv.slice(2);
const median = (xs) => {
  const ys = xs.filter((x) => Number.isFinite(x)).sort((a, b) => a - b);
  if (!ys.length) return 0;
  const mid = Math.floor(ys.length / 2);
  return ys.length % 2 ? ys[mid] : (ys[mid - 1] + ys[mid]) / 2;
};
const meanTiming = (timings, key) => {
  const xs = timings.map((t) => Number(t?.[key])).filter((x) => Number.isFinite(x));
  return xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0;
};
const perRun = [];
for (const path of paths) {
  const report = JSON.parse(fs.readFileSync(path, 'utf8'));
  const summary = report.summary?.[0];
  if (!summary || !Number.isFinite(Number(summary.evalsPerSecond)) || Number(summary.evalsPerSecond) <= 0) {
    throw new Error(`missing positive evalsPerSecond in ${path}`);
  }
  const timings = [];
  for (const result of report.results ?? []) {
    for (const pos of result.positions ?? []) {
      const t = pos.lc0Search?.evalBackendTimingPerPositionMeans;
      if (t) timings.push(t);
    }
  }
  const evalsPerSecond = Number(summary.evalsPerSecond);
  perRun.push({
    path,
    hybrid_b4_ms_per_eval: 1000 / evalsPerSecond,
    evals_per_second: evalsPerSecond,
    visits_per_position: Number(summary.visitsPerPosition ?? 0),
    evals_per_position: Number(summary.evalsPerPosition ?? 0),
    total_eval_ms: meanTiming(timings, 'totalEvalMs'),
    readback_synced_ms: meanTiming(timings, 'readbackSyncedMs'),
    readback_map_async_ms: meanTiming(timings, 'readbackMapAsyncMs'),
    readback_map_wait_ms: meanTiming(timings, 'readbackMapAsyncWaitMs'),
    readback_overlap_hidden_ms: meanTiming(timings, 'readbackOverlapHiddenMs'),
    legal_priors_prep_ms: meanTiming(timings, 'legalPriorsPrepMs'),
    readback_bytes: meanTiming(timings, 'readbackBytes'),
    readback_maps: meanTiming(timings, 'readbackMapCount'),
    dispatch_count: meanTiming(timings, 'dispatchCount'),
  });
}
const keys = Object.keys(perRun[0]).filter((key) => key !== 'path');
for (const key of keys) console.log(`METRIC ${key}=${median(perRun.map((run) => run[key]))}`);
console.log(`ARTIFACT latest=${perRun.at(-1).path}`);
NODE
