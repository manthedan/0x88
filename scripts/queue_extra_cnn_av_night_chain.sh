#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(pwd)}
PY=${PY:-.venv-onnx/bin/python}
LOGDIR=${LOGDIR:-artifacts/queue_extra_cnn_av_night_chain/logs}
STATUS=${STATUS:-artifacts/queue_extra_cnn_av_night_chain/status}
mkdir -p "$LOGDIR" "$STATUS"

MANIFEST=${MANIFEST:-data/datasets/supervised_100m_elite_tcec_v1/cache_residual_h2/cache_manifest.json}
DEV_CACHE=${DEV_CACHE:-data/datasets/supervised_100m_elite_tcec_v1/cache_residual_h2/dev}
AV_CACHE=${AV_CACHE:-data/public_teacher_overlays/chessbench_full_policy_value_direct_top8_32shards_v1/collection_manifest.json}
CHANNELS=${CHANNELS:-80}
BLOCKS=${BLOCKS:-5}
POLICY_ROWS=${POLICY_ROWS:-25000000}
AV_POSITIONS=${AV_POSITIONS:-16286511}
EPOCHS=${EPOCHS:-1}
BATCH_SIZE=${BATCH_SIZE:-2048}
AV_BATCH_SIZE=${AV_BATCH_SIZE:-1024}

PHASE1_RUN=${PHASE1_RUN:-artifacts/cnn_av_v2_80x5_after_cnn64_av}
PHASE2_RUN=${PHASE2_RUN:-artifacts/cnn_av_v2_80x5_phase2_low_lr}
PHASE3_RUN=${PHASE3_RUN:-artifacts/cnn_av_v2_80x5_phase3_tiny_lr}

log(){ echo "[$(date -Is)] $*" | tee -a "$LOGDIR/queue.log"; }
write_state(){ printf '%s\n' "$*" > "$STATUS/state.txt"; }
require_file(){ [[ -f "$1" ]] || { log "missing required file: $1"; exit 2; }; }
require_dir(){ [[ -d "$1" ]] || { log "missing required dir: $1"; exit 2; }; }

wait_run_done(){
  local run="$1" name="$2"
  write_state "waiting_for_${name}_pid"
  log "waiting for $name pid: $run/status/train.pid"
  while [[ ! -s "$run/status/train.pid" ]]; do sleep 120; done
  local pid; pid=$(cat "$run/status/train.pid")
  write_state "waiting_for_${name}_finish_pid_${pid}"
  log "waiting for $name pid=$pid to finish"
  while ps -p "$pid" >/dev/null 2>&1; do sleep 180; done
  log "$name pid=$pid exited"
  if [[ ! -s "$run/model.pt" && ! -s "$run/checkpoints/best.pt" ]]; then
    write_state "${name}_failed_no_model"
    log "not continuing: missing $run/model.pt and $run/checkpoints/best.pt"
    exit 1
  fi
}

best_resume_for(){
  local run="$1"
  if [[ -s "$run/checkpoints/best.pt" ]]; then echo "$run/checkpoints/best.pt"; else echo "$run/model.pt"; fi
}

launch_phase(){
  local name="$1" run="$2" resume="$3" lr="$4"
  mkdir -p "$run/logs" "$run/status" "$run/checkpoints"
  require_file "$MANIFEST"; require_dir "$DEV_CACHE"; require_file "$AV_CACHE"; require_file "$resume"; require_file training/train_residual_av_multicache_torch.py
  write_state "launching_${name}"
  log "launching $name run=$run resume=$resume lr=$lr policy_rows=$POLICY_ROWS av_positions=$AV_POSITIONS"
  nohup bash -lc "cd '$ROOT' && '$PY' training/train_residual_av_multicache_torch.py \
    --manifest '$MANIFEST' \
    --dev-cache '$DEV_CACHE' \
    --av-cache '$AV_CACHE' \
    --resume '$resume' \
    --out '$run/model.pt' \
    --onnx-out '$run/model.onnx' \
    --meta-out '$run/model.meta.json' \
    --checkpoint-dir '$run/checkpoints' \
    --best-checkpoint '$run/checkpoints/best.pt' \
    --channels '$CHANNELS' --blocks '$BLOCKS' \
    --policy-rows '$POLICY_ROWS' --av-positions '$AV_POSITIONS' --epochs '$EPOCHS' \
    --batch-size '$BATCH_SIZE' --av-batch-size '$AV_BATCH_SIZE' \
    --lr '$lr' --weight-decay 1e-4 \
    --eval-every-steps 10000 --checkpoint-every-steps 10000 --progress-every 500 \
    --max-dev-rows 100000 --max-av-dev-positions 10000 \
    --device cuda --amp --amp-dtype bf16 --export-av-head" \
    > "$run/logs/train.log" 2>&1 &
  echo $! > "$run/status/train.pid"
  date -Is > "$run/status/train.started"
  {
    echo "arch=80x5"; echo "channels=$CHANNELS"; echo "blocks=$BLOCKS"; echo "resume=$resume"; echo "policy_rows=$POLICY_ROWS"; echo "av_positions=$AV_POSITIONS"; echo "epochs=$EPOCHS"; echo "lr=$lr"; echo "phase=$name";
  } > "$run/status/config.env"
  log "$name started pid=$(cat "$run/status/train.pid")"
}

main(){
  cd "$ROOT"
  wait_run_done "$PHASE1_RUN" "cnn80_phase1"
  local r2; r2=$(best_resume_for "$PHASE1_RUN")
  launch_phase "cnn80_phase2_low_lr" "$PHASE2_RUN" "$r2" "1e-5"
  wait_run_done "$PHASE2_RUN" "cnn80_phase2"
  local r3; r3=$(best_resume_for "$PHASE2_RUN")
  launch_phase "cnn80_phase3_tiny_lr" "$PHASE3_RUN" "$r3" "5e-6"
  write_state "phase3_launched"
}
main "$@"
