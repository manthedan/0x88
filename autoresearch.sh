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
ATTRIBUTION_MODE="${LC0_AR_ATTRIBUTION_MODE:-}"
ATTR_ITERS="${LC0_AR_ATTR_ITERS:-8}"
ATTR_WARMUP="${LC0_AR_ATTR_WARMUP:-1}"
ATTR_FIXTURE_LIMIT="${LC0_AR_ATTR_FIXTURE_LIMIT:-$MAX_POSITIONS}"
INPUT_BACKEND="${LC0_AR_INPUT_BACKEND:-wasm}"
ENCODER_KERNEL="${LC0_AR_ENCODER_KERNEL:-mixed-tvm-ffn}"
LEGAL_PRIORS_BACKEND="${LC0_AR_LEGAL_PRIORS_BACKEND:-js}"
DRIFT_GUARD="${LC0_AR_DRIFT_GUARD:-1}"
DRIFT_LIMIT="${LC0_AR_DRIFT_LIMIT:-3}"
DRIFT_F32_WDL_MAX="${LC0_AR_DRIFT_F32_WDL_MAX:-0.005}"
DRIFT_F32_TOP_PRIOR_MAX="${LC0_AR_DRIFT_F32_TOP_PRIOR_MAX:-0.01}"
CLEAN_BROWSER_HARNESS="${LC0_AR_CLEAN_BROWSER_HARNESS:-1}"
CLEAN_BROWSER_WAIT_SECONDS="${LC0_AR_CLEAN_BROWSER_WAIT_SECONDS:-20}"

if [[ ! -f "$FENS" ]]; then
  echo "missing FEN corpus: $FENS" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

if [[ "$CLEAN_BROWSER_HARNESS" == "1" ]]; then
  # Browser/WebGPU state can persist across agent-browser sessions and inflate
  # readback/fence timings. Before each autoresearch run, clear only the
  # wrapper-owned agent-browser harness and this benchmark's default Vite ports;
  # the measured metric still covers only the fixed-suite workload below.
  pkill -f '/node_modules/agent-browser/bin/agent-browser-darwin-arm64' >/dev/null 2>&1 || true
  pkill -f '/\.agent-browser/browsers/.*/Google Chrome for Testing' >/dev/null 2>&1 || true
  (lsof -tiTCP:5179 -sTCP:LISTEN 2>/dev/null; lsof -tiTCP:5180 -sTCP:LISTEN 2>/dev/null) | sort -u | xargs -r kill >/dev/null 2>&1 || true
  sleep "$CLEAN_BROWSER_WAIT_SECONDS"
fi

if [[ "$ATTRIBUTION_MODE" == "deferred-readback" ]]; then
  attr_out="$OUT_DIR/${RUN_ID}_deferred_readback_attr.json"
  node --experimental-strip-types scripts/lc0_browser_wgsl_deferred_readback_bench.mjs \
    --input-backend "$INPUT_BACKEND" \
    --batch "$BATCH_SIZE" \
    --iters "$ATTR_ITERS" \
    --warmup "$ATTR_WARMUP" \
    --fixture-limit "$ATTR_FIXTURE_LIMIT" \
    --timeout "$TIMEOUT_MS" >"$attr_out"
  node - "$attr_out" <<'NODE'
const fs = require('fs');
const path = process.argv[2];
const report = JSON.parse(fs.readFileSync(path, 'utf8'));
const timing = (mode, key) => Number(report[mode]?.timingMeans?.[key] ?? 0);
for (const mode of ['immediate', 'deferred']) {
  const evalsPerSecond = Number(report[mode]?.evalsPerSecond ?? 0);
  console.log(`ATTR ${mode}_evals_per_second=${evalsPerSecond}`);
  console.log(`ATTR ${mode}_ms_per_eval=${evalsPerSecond > 0 ? 1000 / evalsPerSecond : 0}`);
  console.log(`ATTR ${mode}_readback_synced_ms=${timing(mode, 'readbackSyncedMs')}`);
  console.log(`ATTR ${mode}_readback_map_async_ms=${timing(mode, 'readbackMapAsyncMs')}`);
  console.log(`ATTR ${mode}_readback_map_wait_ms=${timing(mode, 'readbackMapAsyncWaitMs')}`);
  console.log(`ATTR ${mode}_readback_bytes=${timing(mode, 'readbackBytes')}`);
}
console.log(`ATTR deferred_over_immediate_speedup=${Number(report.deferred?.evalsPerSecond ?? 0) / Math.max(1e-9, Number(report.immediate?.evalsPerSecond ?? 0))}`);
console.log(`ATTR all_best_moves_match=${report.allBestMovesMatch === true ? 1 : 0}`);
console.log(`ARTIFACT deferred_readback_attr=${path}`);
NODE
fi

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
    --input-backend "$INPUT_BACKEND" \
    --encoder-kernel "$ENCODER_KERNEL" \
    --legal-priors-backend "$LEGAL_PRIORS_BACKEND" \
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

if [[ "$DRIFT_GUARD" == "1" && "$RUNTIME" != "onnx" ]]; then
  drift_out="$OUT_DIR/${RUN_ID}_drift.json"
  head_backend="ort"
  if [[ "$RUNTIME" == "hybrid-wgsl-heads" ]]; then head_backend="wgsl"; fi
  node --experimental-strip-types scripts/lc0_browser_hybrid_drift.mjs \
    --limit "$DRIFT_LIMIT" \
    --head-backend "$head_backend" \
    --input-backend "$INPUT_BACKEND" \
    --encoder-kernel "$ENCODER_KERNEL" \
    --legal-priors-backend "$LEGAL_PRIORS_BACKEND" \
    --timeout "$TIMEOUT_MS" >"$drift_out"
  node - "$drift_out" "$DRIFT_F32_WDL_MAX" "$DRIFT_F32_TOP_PRIOR_MAX" <<'NODE'
const fs = require('fs');
const [path, wdlLimitRaw, priorLimitRaw] = process.argv.slice(2);
const raw = fs.readFileSync(path, 'utf8');
const start = raw.indexOf('{');
if (start < 0) throw new Error(`drift report has no JSON object: ${path}`);
const report = JSON.parse(raw.slice(start));
const summary = report.summary ?? {};
const fixtures = Number(report.fixtures ?? 0);
const f32BestMoveMatches = Number(summary.f32BestMoveMatches ?? 0);
const nativeBestMoveMatches = Number(summary.nativeBestMoveMatches ?? 0);
const f32WdlMaxAbsDiff = Number(summary.f32WdlMaxAbsDiff ?? Infinity);
const f32TopPriorMaxAbsDiff = Number(summary.f32TopPriorMaxAbsDiff ?? Infinity);
const nativeTopPriorMaxAbsDiff = Number(summary.nativeTopPriorMaxAbsDiff ?? Infinity);
console.log(`METRIC drift_fixtures=${fixtures}`);
console.log(`METRIC drift_f32_best_move_matches=${f32BestMoveMatches}`);
console.log(`METRIC drift_native_best_move_matches=${nativeBestMoveMatches}`);
console.log(`METRIC drift_f32_wdl_max_abs_diff=${f32WdlMaxAbsDiff}`);
console.log(`METRIC drift_f32_top_prior_max_abs_diff=${f32TopPriorMaxAbsDiff}`);
console.log(`METRIC drift_native_top_prior_max_abs_diff=${nativeTopPriorMaxAbsDiff}`);
console.log(`ARTIFACT drift=${path}`);
const wdlLimit = Number(wdlLimitRaw);
const priorLimit = Number(priorLimitRaw);
if (fixtures <= 0) throw new Error('drift guard evaluated no fixtures');
if (f32BestMoveMatches !== fixtures) throw new Error(`drift guard failed: f32 best-move matches ${f32BestMoveMatches}/${fixtures}`);
if (nativeBestMoveMatches !== fixtures) throw new Error(`drift guard failed: native best-move matches ${nativeBestMoveMatches}/${fixtures}`);
if (!(f32WdlMaxAbsDiff <= wdlLimit)) throw new Error(`drift guard failed: f32 WDL max abs diff ${f32WdlMaxAbsDiff} > ${wdlLimit}`);
if (!(f32TopPriorMaxAbsDiff <= priorLimit)) throw new Error(`drift guard failed: f32 top-prior max abs diff ${f32TopPriorMaxAbsDiff} > ${priorLimit}`);
NODE
fi
