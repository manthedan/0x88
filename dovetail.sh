#!/usr/bin/env bash
set -euo pipefail
if [[ "${TINY_LEELA_TRAINER:-python}" == "tinygrad" && -d ".venv-tinygrad/lib/python3.12/site-packages/nvidia/cu13" ]]; then
  export CUDA_HOME="$PWD/.venv-tinygrad/lib/python3.12/site-packages/nvidia/cu13"
  export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
fi

mkdir -p artifacts/logs
DOVETAIL_LOG="artifacts/logs/dovetail-$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "$DOVETAIL_LOG") 2>&1
echo "[dovetail] log=$DOVETAIL_LOG"
echo "METRIC dovetail_log_started=1"
echo "[dovetail] phase=selfplay_mix_sweep start ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

# Phase D scaled true-play arena benchmark.
# Evaluate candidate mix weights by actual played result rather than value
# adjudication. Unfinished games at max plies are draws, so this is a fixed-budget
# strength proxy rather than a full Elo estimate.
npm run selfplay:mix-sweep --silent -- \
  --backend=rust \
  --weights=0,0.05,0.1,0.25 \
  --games=2 \
  --selfplay-visits=32 \
  --max-plies=40 \
  --epochs=40 \
  --arena-games=32 \
  --arena-visits=32 \
  --adjudicate=terminal \
  --selection-metric=arena_true_play_score_rate \
  --adjudicate-threshold=0.02 \
  --primary-conv-arch=64x6 \
  --feature-cache=artifacts/cache/conv_features_64x6.json \
  --parallel-candidates \
  --trainer="${TINY_LEELA_TRAINER:-python}" \
  --python="${TINY_LEELA_PYTHON:-python3}" \
  --selfplay=artifacts/selfplay_mix_arena_v1.jsonl \
  --candidate-prefix=artifacts/selfplay_mix_arena_candidate \
  --regenerate

echo "[dovetail] phase=selfplay_mix_sweep done ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[dovetail] phase=playable_suite start ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
npm run eval:playable --silent
echo "[dovetail] phase=playable_suite done ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
DOVETAIL_LOG_BYTES=$(wc -c < "$DOVETAIL_LOG" | tr -d ' ')
echo "METRIC dovetail_log_bytes=$DOVETAIL_LOG_BYTES"
echo "[dovetail] log_path=$DOVETAIL_LOG log_bytes=$DOVETAIL_LOG_BYTES"
DOVETAIL_TAIL_TMP=$(mktemp)
tail -n "${DOVETAIL_LOG_TAIL_LINES:-60}" "$DOVETAIL_LOG" > "$DOVETAIL_TAIL_TMP"
echo "[dovetail] recent_log_tail_begin lines=${DOVETAIL_LOG_TAIL_LINES:-60}"
cat "$DOVETAIL_TAIL_TMP"
echo "[dovetail] recent_log_tail_end"
rm -f "$DOVETAIL_TAIL_TMP"
