#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="/home/ddbb/projects/tiny_leela"
cd "$ROOT"
PY="${PY:-.venv-onnx/bin/python}"
OUT_JSON="${OUT_JSON:-artifacts/analysis/current_model_inventory.json}"
OUT_MD="${OUT_MD:-artifacts/analysis/current_models_efficiency.md}"
"$PY" eval/build_model_manifest.py \
  --out "$OUT_JSON" \
  --md-out "$OUT_MD" \
  --arena-glob 'artifacts/search_mode_arena/*.json' \
  --arena-glob 'artifacts/head_ablation_1m/**/*.json' \
  --arena-glob 'artifacts/arena_10m_guarded/*.json' \
  --arena-glob 'artifacts/bench_100m_cnns/*.json'
