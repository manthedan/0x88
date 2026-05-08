#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(pwd)}
PY=${PY:-.venv-onnx/bin/python}
SRC_RUN=${SRC_RUN:-artifacts/cnn_av_v2_64x6_after_squareformer_e1}
DST_RUN=${DST_RUN:-artifacts/cnn_av_v2_80x5_after_cnn64_av}
LOGDIR=${LOGDIR:-artifacts/queue_cnn80_after_cnn64_av/logs}
STATUS=${STATUS:-artifacts/queue_cnn80_after_cnn64_av/status}
mkdir -p "$LOGDIR" "$STATUS" "$DST_RUN/logs" "$DST_RUN/status" "$DST_RUN/checkpoints"

MANIFEST=${MANIFEST:-data/datasets/supervised_100m_elite_tcec_v1/cache_residual_h2/cache_manifest.json}
DEV_CACHE=${DEV_CACHE:-data/datasets/supervised_100m_elite_tcec_v1/cache_residual_h2/dev}
AV_CACHE=${AV_CACHE:-data/public_teacher_overlays/chessbench_full_policy_value_direct_top8_32shards_v1/collection_manifest.json}
RESUME=${RESUME:-artifacts/100m_canonical/cnn_80x5_100m_e3.pt}
CHANNELS=${CHANNELS:-80}
BLOCKS=${BLOCKS:-5}
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

wait_for_cnn64(){
  write_state "waiting_for_cnn64_pid"
  log "waiting for $SRC_RUN/status/train.pid"
  while [[ ! -s "$SRC_RUN/status/train.pid" ]]; do sleep 60; done
  local pid; pid=$(cat "$SRC_RUN/status/train.pid")
  write_state "waiting_for_cnn64_finish_pid_${pid}"
  log "waiting for CNN-AV 64x6 pid=$pid to finish"
  while ps -p "$pid" >/dev/null 2>&1; do sleep 120; done
  log "CNN-AV 64x6 pid=$pid exited"
  if [[ ! -s "$SRC_RUN/model.pt" ]]; then
    write_state "cnn64_failed_no_model"
    log "not launching 80x5: missing $SRC_RUN/model.pt"
    exit 1
  fi
  write_state "cnn64_complete"
}

launch_cnn80(){
  require_file "$MANIFEST"; require_dir "$DEV_CACHE"; require_file "$AV_CACHE"; require_file "$RESUME"; require_file training/train_residual_av_multicache_torch.py
  write_state "launching_cnn80"
  log "launching CNN-AV V2 80x5 policy_rows=$POLICY_ROWS av_positions=$AV_POSITIONS resume=$RESUME run=$DST_RUN"
  nohup bash -lc "cd '$ROOT' && '$PY' training/train_residual_av_multicache_torch.py \
    --manifest '$MANIFEST' \
    --dev-cache '$DEV_CACHE' \
    --av-cache '$AV_CACHE' \
    --resume '$RESUME' \
    --out '$DST_RUN/model.pt' \
    --onnx-out '$DST_RUN/model.onnx' \
    --meta-out '$DST_RUN/model.meta.json' \
    --checkpoint-dir '$DST_RUN/checkpoints' \
    --best-checkpoint '$DST_RUN/checkpoints/best.pt' \
    --channels '$CHANNELS' --blocks '$BLOCKS' \
    --policy-rows '$POLICY_ROWS' --av-positions '$AV_POSITIONS' --epochs '$EPOCHS' \
    --batch-size '$BATCH_SIZE' --av-batch-size '$AV_BATCH_SIZE' \
    --lr '$LR' --weight-decay 1e-4 \
    --eval-every-steps 10000 --checkpoint-every-steps 10000 --progress-every 500 \
    --max-dev-rows 100000 --max-av-dev-positions 10000 \
    --device cuda --amp --amp-dtype bf16 --export-av-head" \
    > "$DST_RUN/logs/train.log" 2>&1 &
  echo $! > "$DST_RUN/status/train.pid"
  date -Is > "$DST_RUN/status/train.started"
  {
    echo "arch=80x5"; echo "channels=$CHANNELS"; echo "blocks=$BLOCKS"; echo "resume=$RESUME"; echo "policy_rows=$POLICY_ROWS"; echo "av_positions=$AV_POSITIONS"; echo "epochs=$EPOCHS"; echo "depends_on=$SRC_RUN";
  } > "$DST_RUN/status/config.env"
  write_state "cnn80_launched"
  log "CNN-AV 80x5 started pid=$(cat "$DST_RUN/status/train.pid")"
}

main(){
  cd "$ROOT"
  wait_for_cnn64
  launch_cnn80
}
main "$@"
