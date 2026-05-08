#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
OUT_ROOT="artifacts/bench_100m_cnns"
mkdir -p "$OUT_ROOT/logs"
VISITS_LIST="${VISITS_LIST:-1,32,128}"
QUICK_PAIRS="${QUICK_PAIRS:-2}"
FULL_PAIRS="${FULL_PAIRS:-0}"
CPUCT="${CPUCT:-1.5}"
BUCKET_ROWS="${BUCKET_ROWS:-50000}"
WAIT_80X5="${WAIT_80X5:-1}"
MODELS=(cnn_32x4 cnn_48x5 cnn_64x6 cnn_80x5)
model_path() { echo "artifacts/100m_canonical/${1}_100m_e3.onnx"; }
meta_path() { echo "artifacts/100m_canonical/${1}_100m_e3.meta.json"; }
wait_for_model() {
  local name="$1" onnx meta
  onnx="$(model_path "$name")"; meta="$(meta_path "$name")"
  if [[ "$name" != "cnn_80x5" || "$WAIT_80X5" != "1" ]]; then
    [[ -s "$onnx" && -s "$meta" ]] || { echo "missing $name artifacts: $onnx $meta" >&2; return 1; }
    return 0
  fi
  echo "[bench] waiting for $name ONNX/meta export..."
  until [[ -s "$onnx" && -s "$meta" ]]; do
    sleep 60
    tail -n 4 data/datasets/supervised_100m_elite_tcec_v1/reports/train_cnn_80x5.log 2>/dev/null || true
  done
}
run_release_gate() {
  local name="$1"
  wait_for_model "$name"
  echo "[bench] release gate $name $(date -Is)"
  nice -n 5 scripts/run_model_release_gate.sh \
    --name "${name}_100m_e3" \
    --model "$(model_path "$name")" \
    --meta "$(meta_path "$name")" \
    --out-dir "$OUT_ROOT/${name}_release_gate" \
    --bucket-rows "$BUCKET_ROWS" \
    --quick-pairs "$QUICK_PAIRS" \
    --full-pairs "$FULL_PAIRS" \
    --visits-list "$VISITS_LIST" \
    --cpuct "$CPUCT" \
    --skip-build \
    > "$OUT_ROOT/logs/${name}_release_gate.log" 2>&1
}
sync_public_models() {
  mkdir -p public/models
  for name in "${MODELS[@]}"; do
    [[ -s "$(model_path "$name")" && -s "$(meta_path "$name")" ]] || continue
    cp -f "$(model_path "$name")" "$(meta_path "$name")" public/models/
  done
}
run_shootout() {
  sync_public_models
  echo "[bench] ONNX arena shootout $(date -Is)"
  local specs=()
  for name in "${MODELS[@]}"; do
    [[ -s "$(model_path "$name")" && -s "$(meta_path "$name")" ]] && specs+=("${name}:$(model_path "$name"):$(meta_path "$name")")
  done
  specs+=("chessformer_v1_100m:public/models/chessformer_v1_100m_e3_single.onnx:public/models/chessformer_v1_100m_e3_single.meta.json")
  local joined
  joined="$(IFS=,; echo "${specs[*]}")"
  for visits in 1 32 128; do
    nice -n 5 node --experimental-strip-types eval/onnx_round_robin_arena.mjs \
      --models "$joined" \
      --openings-file eval/opening_suite_uho_lite_v1.fen \
      --games-per-pair "${SHOOTOUT_GAMES_PER_PAIR:-4}" \
      --visits "$visits" \
      --cpuct "$CPUCT" \
      --max-plies "${SHOOTOUT_MAX_PLIES:-100}" \
      --out "$OUT_ROOT/arena_shootout_v${visits}.json" \
      > "$OUT_ROOT/logs/arena_shootout_v${visits}.log" 2>&1
  done
}
for name in "${MODELS[@]}"; do run_release_gate "$name"; done
run_shootout
echo "[bench] complete $(date -Is)"
