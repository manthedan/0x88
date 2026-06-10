#!/usr/bin/env bash
set -euo pipefail

# Research sweep: rebuild the b16 TVMJS wasm with dlight Matmul.Config
# overrides, stage to f16/v2-dlight, run the browser kernel profiler smoke,
# and append per-config results to a JSONL. Parity (8/8 native best-move) is
# checked by the smoke itself; configs that fail parity fail the run.

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
OUT_JSONL="${SWEEP_OUT:-artifacts/tvm/dlight_matmul_config_sweep.jsonl}"
MANIFEST=/runtimes/lc0-tvmjs-webgpu/t1-256x10-distilled-swa-2432500/f16/v2-dlight/manifest.json
mkdir -p artifacts/tvm

# Single-knob configs: one Matmul config applies to ALL matmuls, including the
# WDL head (N=3) and policy head (N=1858), so vectorization/tiling must remain
# legal for awkward dims; sweep knobs separately to isolate what survives.
configs=(
  'baseline|'
  'vec2|{"vector_size":2}'
  'k16|{"micro_size_k":16}'
  'y16|{"block_size_y":16}'
  'align|{"storage_align":true}'
  'innerx|{"inner_x":true}'
  'noshared|{"use_shared":false}'
  'micro88|{"micro_size_x":8,"micro_size_y":8}'
)

for entry in "${configs[@]}"; do
  label="${entry%%|*}"
  config="${entry#*|}"
  echo "[sweep] === $label $config ===" >&2
  CAST_INT64_INITIALIZERS_TO_INT32=1 TRUST_NONNEGATIVE_GATHER_INDICES=1 \
  SANITIZE_ONNX_NAMES=1 EXPORT_TVMJS_WASM=1 DLIGHT=1 \
  TVM_BUILD_DIR=build-tvmjs \
  TVM_HOST_TARGET='{"kind":"llvm","mtriple":"wasm32-unknown-unknown-wasm"}' \
  LC0_TVMJS_BATCHES=16 OUT_DIR="$REPO/artifacts/tvm-dlight" \
  DLIGHT_MATMUL_CONFIG="$config" \
  ./scripts/run_lc0_tvm_whole_onnx_probe.sh >/dev/null 2>&1 || { echo "[sweep] $label BUILD FAILED" >&2; continue; }
  node scripts/stage_lc0_tvmjs_webgpu_artifacts.mjs --artifacts=artifacts/tvm-dlight --batches=16 \
    --out=public/runtimes/lc0-tvmjs-webgpu/t1-256x10-distilled-swa-2432500/f16/v2-dlight >/dev/null
  smoke_out="artifacts/tvm/dlight_sweep_${label}.json"
  node scripts/lc0_tvmjs_webgpu_smoke.mjs --batch 16 --manifest "$MANIFEST" \
    --fixture-count 8 --kernel-profile-invokes 8 --timeout 240000 --out "$smoke_out" >/dev/null 2>&1 || { echo "[sweep] $label SMOKE FAILED" >&2; continue; }
  python3 - "$label" "$config" "$smoke_out" "$OUT_JSONL" <<'EOF'
import json, sys
label, config, smoke_path, out_path = sys.argv[1:5]
a = json.load(open(smoke_path))
r = a["result"]
p = r.get("gpuKernelProfile", {})
probe = json.load(open("artifacts/tvm-dlight/t1-256x10-distilled-swa-2432500.batch16.f16.webgpu.tvmjs-wasm.probe.json"))
row = {
    "label": label,
    "config": json.loads(config) if config else None,
    "parityOk": a.get("ok"),
    "bestMoveMatches": f"{r.get('bestMoveMatches')}/{r.get('nativeComparable')}",
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
print(f"[sweep] {label}: gpu/invoke {row['gpuMsPerInvoke']} ms, parity {row['bestMoveMatches']}", file=sys.stderr)
EOF
done
echo "[sweep] done -> $OUT_JSONL" >&2
