#!/usr/bin/env bash
set -euo pipefail

# Overnight 100M supervised pipeline for Tiny Leela.
# Durable/resumable: each completed step writes a .done marker; logs live under reports/.
# Run with:
#   nohup bash scripts/overnight_100m_pipeline.sh > data/datasets/supervised_100m_elite_tcec_v1/reports/overnight_pipeline.log 2>&1 &

ROOT_DIR=${ROOT_DIR:-$(pwd)}
PY_ONNX=${PY_ONNX:-.venv-onnx/bin/python}
NODE=${NODE:-node}
DATASET=${DATASET:-data/datasets/supervised_100m_elite_tcec_v1}
RAW_DIR=${RAW_DIR:-data/lichess_elite_raw}
ELITE_DIR=${ELITE_DIR:-data/lichess_elite_training}
REPORT_DIR=${REPORT_DIR:-$DATASET/reports}
STATUS_DIR=${STATUS_DIR:-$DATASET/status}
CNN_CACHE=${CNN_CACHE:-$DATASET/cache_h2_state}
SQ_CACHE=${SQ_CACHE:-$DATASET/cache_squareformer_h2}
ART_DIR=${ART_DIR:-artifacts/100m_canonical}

# Data knobs. Months are oldest-to-newest by default so already-ingested 2024/2025 files are reused.
START_YM=${START_YM:-2024-01}
END_YM=${END_YM:-2025-11}
MAX_POSITIONS_PER_MONTH=${MAX_POSITIONS_PER_MONTH:-5000000}
MAX_GAMES_PER_MONTH=${MAX_GAMES_PER_MONTH:-1000000}
REINGEST_LOW_MONTHS=${REINGEST_LOW_MONTHS:-1}
TARGET_TRAIN_ROWS=${TARGET_TRAIN_ROWS:-100000000}
DEV_ROWS=${DEV_ROWS:-1000000}
ROWS_PER_SHARD=${ROWS_PER_SHARD:-1000000}
DATASET_WORKERS=${DATASET_WORKERS:-8}
CACHE_WORKERS=${CACHE_WORKERS:-8}

# Training knobs. Override from env if desired.
CNN_EPOCHS=${CNN_EPOCHS:-3}
CNN_BATCH=${CNN_BATCH:-2048}
SQ_EPOCHS=${SQ_EPOCHS:-3}
SQ_BATCH=${SQ_BATCH:-8192}
DEVICE=${DEVICE:-cuda}
AMP=${AMP:-1}

mkdir -p "$RAW_DIR" "$ELITE_DIR" "$REPORT_DIR" "$STATUS_DIR" "$ART_DIR"

log() { printf '[%(%F %T)T] %s\n' -1 "$*" | tee -a "$REPORT_DIR/pipeline.status.log"; }
mark_done() { mkdir -p "$STATUS_DIR"; date -Is > "$STATUS_DIR/$1.done"; }
is_done() { test -s "$STATUS_DIR/$1.done"; }
run_step() {
  local name="$1"; shift
  if is_done "$name"; then log "skip $name: already done"; return 0; fi
  log "START $name"
  "$@" 2>&1 | tee -a "$REPORT_DIR/${name}.log"
  mark_done "$name"
  log "DONE $name"
}

months_between() {
  "$PY_ONNX" - "$START_YM" "$END_YM" <<'PY'
import sys
from datetime import date
s,e=sys.argv[1:3]
y,m=map(int,s.split('-')); ey,em=map(int,e.split('-'))
out=[]
while (y,m) <= (ey,em):
    out.append(f'{y:04d}-{m:02d}')
    m += 1
    if m == 13: y += 1; m = 1
print(' '.join(out))
PY
}

source_rows() {
  "$PY_ONNX" - "$ELITE_DIR" <<'PY'
import sys, pathlib
root=pathlib.Path(sys.argv[1]); total=0; files=0
for p in sorted(root.glob('lichess_elite_*.jsonl')):
    files += 1
    with p.open('rb') as f:
        total += sum(1 for _ in f)
print(total)
print(files, file=sys.stderr)
PY
}

download_elite_months() {
  for ym in $(months_between); do
    f="$RAW_DIR/lichess_elite_${ym}.zip"
    if [ -s "$f" ]; then log "raw exists $ym"; continue; fi
    url="https://database.nikonoel.fr/lichess_elite_${ym}.zip"
    log "download $url"
    # Some old/new months may be unavailable; continue so the pipeline can use all available months.
    if curl -L -C - --fail --retry 5 --retry-delay 20 -o "$f.part" "$url" && unzip -tq "$f.part" >/dev/null 2>&1; then
      mv "$f.part" "$f"
    else
      log "WARN missing/unavailable or invalid zip $url"
      rm -f "$f.part" "$f"
    fi
  done
}

ingest_elite_months_until_enough() {
  local total
  for ym in $(months_between); do
    local zip="$RAW_DIR/lichess_elite_${ym}.zip"
    local out="$ELITE_DIR/lichess_elite_${ym}.jsonl"
    [ -s "$zip" ] || continue
    local rows=0
    if [ -s "$out" ]; then rows=$(wc -l < "$out"); fi
    if [ ! -s "$out" ] || { [ "$REINGEST_LOW_MONTHS" = "1" ] && [ "$rows" -lt "$MAX_POSITIONS_PER_MONTH" ]; }; then
      log "ingest $ym (existing rows=$rows target=$MAX_POSITIONS_PER_MONTH)"
      rm -f "$out"
      MAX_GAMES="$MAX_GAMES_PER_MONTH" MAX_POSITIONS="$MAX_POSITIONS_PER_MONTH" \
        RAW_DIR="$RAW_DIR" OUT_DIR="$ELITE_DIR" bash scripts/ingest_lichess_elite_months.sh "$ym"
    else
      log "training jsonl exists $ym rows=$rows"
    fi
    total=$(source_rows | tail -1)
    log "elite source rows currently $total"
    if [ "$total" -ge $((TARGET_TRAIN_ROWS + DEV_ROWS)) ]; then
      log "enough source rows for target"
      break
    fi
  done
}

build_dataset() {
  if [ -s "$DATASET/manifest.json" ]; then log "dataset manifest exists"; return 0; fi
  mapfile -t inputs < <(find "$ELITE_DIR" -maxdepth 1 -name 'lichess_elite_*.jsonl' | sort)
  if [ "${#inputs[@]}" -eq 0 ]; then echo "no elite inputs in $ELITE_DIR" >&2; return 1; fi
  log "building 100M dataset from ${#inputs[@]} input files"
  "$PY_ONNX" scripts/build_supervised_dataset_streaming.py \
    --input "${inputs[@]}" \
    --out-dir "$DATASET" \
    --name supervised_100m_elite_tcec_v1 \
    --max-rows "$TARGET_TRAIN_ROWS" \
    --dev-rows "$DEV_ROWS" \
    --rows-per-shard "$ROWS_PER_SHARD" \
    --skip-plies 10 \
    --history-plies 2 \
    --seed 100 \
    --zst
}

validate_dataset_rows() {
  "$PY_ONNX" - "$DATASET/manifest.json" "$TARGET_TRAIN_ROWS" <<'PY'
import json, sys
m=json.load(open(sys.argv[1])); target=int(sys.argv[2])
print('train_rows', m.get('total_train_rows'), 'dev_rows', m.get('total_dev_rows'), 'shards', len(m.get('train_shards', [])))
if int(m.get('total_train_rows', 0)) < target:
    raise SystemExit(f'dataset under target: {m.get("total_train_rows")} < {target}')
PY
}

build_cnn_cache() {
  if [ -s "$CNN_CACHE/cache_manifest.json" ]; then log "CNN cache manifest exists"; return 0; fi
  "$PY_ONNX" scripts/build_residual_cache_from_dataset.py \
    --dataset-dir "$DATASET" \
    --out-dir "$CNN_CACHE" \
    --python "$PY_ONNX" \
    --history-plies 2 \
    --state-planes \
    --workers "$CACHE_WORKERS"
}

build_squareformer_cache() {
  if [ -s "$SQ_CACHE/cache_manifest.json" ]; then log "SquareFormer cache manifest exists"; return 0; fi
  "$PY_ONNX" scripts/build_squareformer_cache_from_dataset.py \
    --dataset-dir "$DATASET" \
    --out-dir "$SQ_CACHE" \
    --python "$PY_ONNX" \
    --history-plies 2 \
    --workers "$CACHE_WORKERS"
}

train_cnn() {
  local name="$1" channels="$2" blocks="$3"
  local done="train_${name}"
  if is_done "$done"; then log "skip $done: already done"; return 0; fi
  log "START $done"
  local amp_arg=(); [ "$AMP" = "1" ] && amp_arg=(--amp)
  "$PY_ONNX" training/train_residual_aux_multicache_torch.py \
    --manifest "$CNN_CACHE/cache_manifest.json" \
    --dev-cache "$CNN_CACHE/dev" \
    --out "$ART_DIR/${name}_100m_e${CNN_EPOCHS}.pt" \
    --onnx-out "$ART_DIR/${name}_100m_e${CNN_EPOCHS}.onnx" \
    --meta-out "$ART_DIR/${name}_100m_e${CNN_EPOCHS}.meta.json" \
    --best-checkpoint "$ART_DIR/${name}_100m_best.pt" \
    --ema-best-checkpoint "$ART_DIR/${name}_100m_best_ema.pt" \
    --epochs "$CNN_EPOCHS" \
    --batch-size "$CNN_BATCH" \
    --channels "$channels" \
    --blocks "$blocks" \
    --lr 0.0001 --min-lr 0.00001 --lr-schedule cosine --warmup-steps 2000 \
    --weight-decay 0.0001 --policy-label-smoothing 0.02 --grad-clip-norm 1.0 --ema-decay 0.999 \
    --amp-dtype bf16 --fused-adamw --matmul-precision high --shuffle-chunk-rows 262144 --progress-every 500 --prefetch-batches 2 \
    --device "$DEVICE" "${amp_arg[@]}" \
    2>&1 | tee -a "$REPORT_DIR/${done}.log"
  mark_done "$done"
  log "DONE $done"
}

train_squareformer() {
  local variant="$1"
  local name="chessformer_${variant}"
  local done="train_${name}"
  if is_done "$done"; then log "skip $done: already done"; return 0; fi
  log "START $done"
  "$PY_ONNX" training/train_squareformer_torch.py \
    --cache-manifest "$SQ_CACHE/cache_manifest.json" \
    --variant "$variant" --relation-bias \
    --epochs "$SQ_EPOCHS" \
    --batch-size "$SQ_BATCH" \
    --max-rows 0 \
    --max-dev-rows 0 \
    --compact-embeddings \
    --shuffle-chunk-rows 262144 \
    --amp \
    --amp-dtype bf16 \
    --torch-compile \
    --fused-adamw \
    --lr-schedule cosine \
    --warmup-frac 0.02 \
    --min-lr-frac 0.1 \
    --matmul-precision high \
    --checkpoint-dir "$ART_DIR/chessformer_v1_100m_checkpoints" \
    --progress-every 500 \
    --prefetch-batches 2 \
    --eval-rows 100000 \
    --device "$DEVICE" \
    --out "$ART_DIR/${name}_100m_e${SQ_EPOCHS}.pt" \
    --onnx-out "$ART_DIR/${name}_100m_e${SQ_EPOCHS}.onnx" \
    --meta-out "$ART_DIR/${name}_100m_e${SQ_EPOCHS}.meta.json" \
    2>&1 | tee -a "$REPORT_DIR/${done}.log"
  mark_done "$done"
  log "DONE $done"
}

main() {
  cd "$ROOT_DIR"
  echo $$ > "$STATUS_DIR/pipeline.pid"
  log "pipeline pid $$"
  log "dataset=$DATASET range=$START_YM..$END_YM target=$TARGET_TRAIN_ROWS dev=$DEV_ROWS"

  run_step download_elite_months download_elite_months
  run_step ingest_elite_months ingest_elite_months_until_enough
  run_step build_dataset build_dataset
  run_step validate_dataset validate_dataset_rows
  run_step build_cnn_cache build_cnn_cache
  run_step build_squareformer_cache build_squareformer_cache

  # Priority order: train the most promising SquareFormer first, then canonical CNNs,
  # then give the known-strong 80x5 family an honest 100M run. Skip chessformer v0
  # for now; v1 already won smoke tests and v0 is lower ROI than stronger CNN/longer-v1 work.
  train_squareformer v1
  train_cnn cnn_32x4 32 4
  train_cnn cnn_48x5 48 5
  train_cnn cnn_64x6 64 6
  train_cnn cnn_80x5 80 5

  log "ALL DONE"
  mark_done all
}

main "$@"
