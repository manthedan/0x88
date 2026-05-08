#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(pwd)}
PY=${PY:-.venv-onnx/bin/python}
SF_RUN=${SF_RUN:-artifacts/squareformer_v2_full_tonight_explicit_budget_overfit}
SF_EPOCH=${SF_EPOCH:-1}
ARCH=${ARCH:-64x6}
CNN_RUN=${CNN_RUN:-artifacts/cnn_av_v2_${ARCH}_after_squareformer_e${SF_EPOCH}}
LOGDIR=${LOGDIR:-artifacts/queue_cnn_av_after_squareformer/logs}
STATUS=${STATUS:-artifacts/queue_cnn_av_after_squareformer/status}
mkdir -p "$LOGDIR" "$STATUS" "$CNN_RUN/logs" "$CNN_RUN/status" "$CNN_RUN/checkpoints"

case "$ARCH" in
  64x6)
    CHANNELS=${CHANNELS:-64}; BLOCKS=${BLOCKS:-6}; RESUME=${RESUME:-artifacts/100m_canonical/cnn_64x6_100m_e3.pt} ;;
  80x5)
    CHANNELS=${CHANNELS:-80}; BLOCKS=${BLOCKS:-5}; RESUME=${RESUME:-artifacts/100m_canonical/cnn_80x5_100m_e3.pt} ;;
  *) echo "Unknown ARCH=$ARCH; expected 64x6 or 80x5" >&2; exit 2 ;;
esac

MANIFEST=${MANIFEST:-data/datasets/supervised_100m_elite_tcec_v1/cache_residual_h2/cache_manifest.json}
DEV_CACHE=${DEV_CACHE:-data/datasets/supervised_100m_elite_tcec_v1/cache_residual_h2/dev}
AV_CACHE=${AV_CACHE:-data/public_teacher_overlays/chessbench_full_policy_value_direct_top8_32shards_v1/collection_manifest.json}
POLICY_ROWS=${POLICY_ROWS:-25000000}
AV_POSITIONS=${AV_POSITIONS:-16286511}
EPOCHS=${EPOCHS:-1}
BATCH_SIZE=${BATCH_SIZE:-2048}
AV_BATCH_SIZE=${AV_BATCH_SIZE:-1024}
LR=${LR:-3e-5}

log(){ echo "[$(date -Is)] $*" | tee -a "$LOGDIR/queue.log"; }
write_state(){ printf '%s\n' "$*" > "$STATUS/state.txt"; }

require_file(){ [[ -f "$1" ]] || { log "missing required file: $1"; exit 2; }; }
require_dir(){ [[ -d "$1" ]] || { log "missing required dir: $1"; exit 2; }; }

stop_squareformer(){
  local pid_file="$SF_RUN/status/train.pid"
  if [[ -f "$pid_file" ]]; then
    local pid; pid=$(cat "$pid_file")
    if ps -p "$pid" >/dev/null 2>&1; then
      log "stopping SquareFormer run pid=$pid after epoch $SF_EPOCH checkpoint"
      pkill -TERM -P "$pid" || true
      kill -TERM "$pid" || true
      sleep 5
      pkill -KILL -P "$pid" || true
      kill -KILL "$pid" 2>/dev/null || true
      date -Is > "$SF_RUN/status/stopped_after_epoch_${SF_EPOCH}_for_cnn_av"
    else
      log "SquareFormer train pid not running: $pid"
    fi
  else
    log "no SquareFormer pid file; assuming not running"
  fi
}

launch_cnn_av(){
  require_file "$MANIFEST"; require_dir "$DEV_CACHE"; require_file "$AV_CACHE"; require_file "$RESUME"; require_file training/train_residual_av_multicache_torch.py
  log "launching CNN-AV V2 ARCH=$ARCH channels=$CHANNELS blocks=$BLOCKS policy_rows=$POLICY_ROWS av_positions=$AV_POSITIONS resume=$RESUME run=$CNN_RUN"
  nohup bash -lc "cd '$ROOT' && '$PY' training/train_residual_av_multicache_torch.py \
    --manifest '$MANIFEST' \
    --dev-cache '$DEV_CACHE' \
    --av-cache '$AV_CACHE' \
    --resume '$RESUME' \
    --out '$CNN_RUN/model.pt' \
    --onnx-out '$CNN_RUN/model.onnx' \
    --meta-out '$CNN_RUN/model.meta.json' \
    --checkpoint-dir '$CNN_RUN/checkpoints' \
    --best-checkpoint '$CNN_RUN/checkpoints/best.pt' \
    --channels '$CHANNELS' --blocks '$BLOCKS' \
    --policy-rows '$POLICY_ROWS' --av-positions '$AV_POSITIONS' --epochs '$EPOCHS' \
    --batch-size '$BATCH_SIZE' --av-batch-size '$AV_BATCH_SIZE' \
    --lr '$LR' --weight-decay 1e-4 \
    --eval-every-steps 10000 --checkpoint-every-steps 10000 --progress-every 500 \
    --max-dev-rows 100000 --max-av-dev-positions 10000 \
    --device cuda --amp --amp-dtype bf16 --export-av-head" \
    > "$CNN_RUN/logs/train.log" 2>&1 &
  echo $! > "$CNN_RUN/status/train.pid"
  date -Is > "$CNN_RUN/status/train.started"
  {
    echo "arch=$ARCH"; echo "channels=$CHANNELS"; echo "blocks=$BLOCKS"; echo "resume=$RESUME"; echo "policy_rows=$POLICY_ROWS"; echo "av_positions=$AV_POSITIONS"; echo "epochs=$EPOCHS";
  } > "$CNN_RUN/status/config.env"
  log "CNN-AV started pid=$(cat "$CNN_RUN/status/train.pid")"
}

main(){
  cd "$ROOT"
  write_state "waiting_for_squareformer_epoch_${SF_EPOCH}"
  local ck="$SF_RUN/checkpoints/epoch_${SF_EPOCH}.pt"
  log "waiting for $ck"
  while [[ ! -s "$ck" ]]; do
    if [[ -f "$SF_RUN/status/train.pid" ]] && ! ps -p "$(cat "$SF_RUN/status/train.pid")" >/dev/null 2>&1; then
      log "SquareFormer process is not running while waiting; continuing to inspect for checkpoint"
    fi
    sleep 60
  done
  log "found checkpoint $ck; waiting for writes to settle"
  sleep 30
  write_state "stopping_squareformer"
  stop_squareformer
  write_state "launching_cnn_av"
  launch_cnn_av
  write_state "cnn_av_launched"
}

main "$@"
