#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR=${ROOT_DIR:-$(pwd)}
PY_ONNX=${PY_ONNX:-.venv-onnx/bin/python}
DATASET=${DATASET:-data/datasets/supervised_100m_elite_tcec_v1}
REPORT_DIR=${REPORT_DIR:-$DATASET/reports}
STATUS_DIR=${STATUS_DIR:-$DATASET/status}
CNN_CACHE=${CNN_CACHE:-$DATASET/cache_residual_h2}
ART_DIR=${ART_DIR:-artifacts/100m_canonical}
CNN_EPOCHS=${CNN_EPOCHS:-3}
CNN_BATCH=${CNN_BATCH:-2048}
DEVICE=${DEVICE:-cuda}
AMP=${AMP:-1}
mkdir -p "$REPORT_DIR" "$STATUS_DIR" "$ART_DIR"
cd "$ROOT_DIR"
log() { printf '[%(%F %T)T] %s\n' -1 "$*" | tee -a "$REPORT_DIR/pipeline.status.log"; }
is_done() { test -s "$STATUS_DIR/$1.done"; }
mark_done() { date -Is > "$STATUS_DIR/$1.done"; }
wait_done() { local name="$1"; while ! is_done "$name"; do log "wait for $name before train_cnn_80x5"; sleep 300; done; }
train_80x5() {
  local done=train_cnn_80x5
  if is_done "$done"; then log "skip $done: already done"; return 0; fi
  log "START $done"
  local amp_arg=(); [ "$AMP" = "1" ] && amp_arg=(--amp)
  "$PY_ONNX" training/train_residual_aux_multicache_torch.py \
    --manifest "$CNN_CACHE/cache_manifest.json" \
    --dev-cache "$CNN_CACHE/dev" \
    --out "$ART_DIR/cnn_80x5_100m_e${CNN_EPOCHS}.pt" \
    --onnx-out "$ART_DIR/cnn_80x5_100m_e${CNN_EPOCHS}.onnx" \
    --meta-out "$ART_DIR/cnn_80x5_100m_e${CNN_EPOCHS}.meta.json" \
    --best-checkpoint "$ART_DIR/cnn_80x5_100m_best.pt" \
    --ema-best-checkpoint "$ART_DIR/cnn_80x5_100m_best_ema.pt" \
    --epochs "$CNN_EPOCHS" \
    --batch-size "$CNN_BATCH" \
    --channels 80 \
    --blocks 5 \
    --lr 0.0001 --min-lr 0.00001 --lr-schedule cosine --warmup-steps 2000 \
    --weight-decay 0.0001 --policy-label-smoothing 0.02 --grad-clip-norm 1.0 --ema-decay 0.999 \
    --amp-dtype bf16 --fused-adamw --matmul-precision high --shuffle-chunk-rows 262144 --progress-every 500 --prefetch-batches 2 \
    --device "$DEVICE" "${amp_arg[@]}" \
    2>&1 | tee -a "$REPORT_DIR/${done}.log"
  mark_done "$done"
  log "DONE $done"
}
wait_done train_cnn_64x6
# If an older already-running pipeline later marks train_chessformer_v0 done/ALL DONE, this
# watcher still adds the replacement 80x5 run after canonical CNNs complete.
train_80x5
