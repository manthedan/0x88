#!/usr/bin/env bash
set -euo pipefail

# Research sweep (BT4-it332): rebuild the b8 detached-params TVMJS wasm with
# dlight Matmul.Config overrides, stage to f16/v2-dlight, run the browser
# kernel-profiler smoke, append per-config results to a JSONL. Parity is the
# native BT4 fixture baseline with --tie-epsilon 0.01 (the known promotion
# near-tie row is tolerated; anything else fails the config).
#
# SWEEP_CONFIGS overrides the config list (space-separated labels) so the
# sweep can run in time-bounded chunks.

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
FAMILY=BT4-1024x15x32h-swa-6147500-policytune-332
BATCH=8
OUT_JSONL="${SWEEP_OUT:-artifacts/tvm/bt4it332_dlight_matmul_config_sweep.jsonl}"
STAGE_DIR="public/runtimes/lc0-tvmjs-webgpu/$FAMILY/f16/v2-dlight"
MANIFEST="/runtimes/lc0-tvmjs-webgpu/$FAMILY/f16/v2-dlight/manifest.json"
FIXTURE_BASELINE="/fixtures/lc0/native_fen_only_blas.$FAMILY.jsonl"
PROBE_DIR="$REPO/artifacts/tvm-dlight-bt4"
mkdir -p artifacts/tvm

config_for() {
  case "$1" in
    baseline) echo '' ;;
    vec2) echo '{"vector_size":2}' ;;
    k16) echo '{"micro_size_k":16}' ;;
    y16) echo '{"block_size_y":16}' ;;
    align) echo '{"storage_align":true}' ;;
    innerx) echo '{"inner_x":true}' ;;
    noshared) echo '{"use_shared":false}' ;;
    micro88) echo '{"micro_size_x":8,"micro_size_y":8}' ;;
    *) echo "[sweep] unknown config label $1" >&2; return 1 ;;
  esac
}
labels="${SWEEP_CONFIGS:-baseline vec2 k16 y16 align innerx noshared micro88}"

for label in $labels; do
  config="$(config_for "$label")"
  echo "[sweep] === $label $config ===" >&2
  PYTHONHASHSEED=0 \
  CAST_INT64_INITIALIZERS_TO_INT32=1 TRUST_NONNEGATIVE_GATHER_INDICES=1 \
  SANITIZE_ONNX_NAMES=1 EXPORT_TVMJS_WASM=1 DLIGHT=1 DETACH_PARAMS=1 \
  TVM_BUILD_DIR=build-tvmjs \
  TVM_HOST_TARGET='{"kind":"llvm","mtriple":"wasm32-unknown-unknown-wasm"}' \
  LC0_TVMJS_MODEL_FAMILY="$FAMILY" LC0_TVMJS_BATCHES="$BATCH" \
  OUT_DIR="$PROBE_DIR" \
  DLIGHT_MATMUL_CONFIG="$config" \
  ./scripts/run_lc0_tvm_whole_onnx_probe.sh >/dev/null 2>&1 || { echo "[sweep] $label BUILD FAILED" >&2; continue; }
  node scripts/stage_lc0_tvmjs_webgpu_artifacts.mjs \
    --artifacts="$PROBE_DIR" --model-family="$FAMILY" --batches="$BATCH" \
    --tensor-cache-dir="$PROBE_DIR/$FAMILY.batch$BATCH.f16.webgpu.tvmjs-wasm.probe.tensor-cache" \
    --params=detached --version=v2-dlight --out="$STAGE_DIR" >/dev/null
  smoke_out="artifacts/tvm/bt4it332_dlight_sweep_${label}.json"
  node scripts/lc0_tvmjs_webgpu_smoke.mjs --batch "$BATCH" --manifest "$MANIFEST" \
    --fixture-count 8 --fixture-baseline "$FIXTURE_BASELINE" --tie-epsilon 0.01 \
    --kernel-profile-invokes 5 --timeout 300000 --out "$smoke_out" >/dev/null 2>&1 || { echo "[sweep] $label SMOKE FAILED" >&2; continue; }
  # BT4 is 341 passes/invoke; the timestamp-query profiler caps at 2048 passes,
  # so 5 invokes (1705) is the max stable profile depth (8 overflows).
  python3 - "$label" "$config" "$smoke_out" "$OUT_JSONL" "$PROBE_DIR/$FAMILY.batch$BATCH.f16.webgpu.tvmjs-wasm.probe.json" <<'EOF'
import json, sys
label, config, smoke_path, out_path, probe_path = sys.argv[1:6]
a = json.load(open(smoke_path))
r = a["result"]
p = r.get("gpuKernelProfile", {})
probe = json.load(open(probe_path))
row = {
    "label": label,
    "config": json.loads(config) if config else None,
    "parityOk": a.get("ok"),
    "bestMoveMatches": f"{r.get('bestMoveMatches')}/{r.get('nativeComparable')}",
    "tieTolerated": len((a.get("tieTolerated") or {}).get("native") or []),
    "maxNativeTopPriorAbsDiff": r.get("maxNativeTopPriorAbsDiff"),
    "totalGpuMs": p.get("totalGpuMs"),
    "gpuMsPerInvoke": p.get("gpuMsPerInvoke"),
    "profileInvokes": p.get("profileInvokes"),
    "passes": p.get("passes"),
    "topKernels": [
        {"name": k["name"], "totalMs": round(k["totalMs"], 3)}
        for k in (p.get("kernels") or [])[:4]
    ],
    "ruleCounts": probe.get("dlight_rule_attribution", {}).get("counts"),
}
with open(out_path, "a") as fh:
    fh.write(json.dumps(row) + "\n")
print(f"[sweep] {label}: gpu/invoke {row['gpuMsPerInvoke']} ms, parity {row['bestMoveMatches']} (+{row['tieTolerated']} tie)", file=sys.stderr)
EOF
done
echo "[sweep] done -> $OUT_JSONL" >&2
