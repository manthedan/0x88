#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(pwd)}
PY=${PY:-.venv-onnx/bin/python}
LOGDIR=${LOGDIR:-artifacts/queue_more_phase2_phase3_sf64/logs}
STATUS=${STATUS:-artifacts/queue_more_phase2_phase3_sf64/status}
mkdir -p "$LOGDIR" "$STATUS"

# This queue waits until the existing 80x5 phase3 run is done, then keeps the GPU busy
# with 64x6 CNN-AV phase2/phase3 and SquareFormer V2 phase2/phase3.
WAIT_RUN=${WAIT_RUN:-artifacts/cnn_av_v2_80x5_phase3_tiny_lr}

RES_MANIFEST=${RES_MANIFEST:-data/datasets/supervised_100m_elite_tcec_v1/cache_residual_h2/cache_manifest.json}
RES_DEV_CACHE=${RES_DEV_CACHE:-data/datasets/supervised_100m_elite_tcec_v1/cache_residual_h2/dev}
AV_CACHE=${AV_CACHE:-data/public_teacher_overlays/chessbench_full_policy_value_direct_top8_32shards_v1/collection_manifest.json}
CNN64_PHASE1=${CNN64_PHASE1:-artifacts/cnn_av_v2_64x6_after_squareformer_e1}
CNN64_PHASE2=${CNN64_PHASE2:-artifacts/cnn_av_v2_64x6_phase2_low_lr}
CNN64_PHASE3=${CNN64_PHASE3:-artifacts/cnn_av_v2_64x6_phase3_tiny_lr}
CNN_POLICY_ROWS=${CNN_POLICY_ROWS:-25000000}
CNN_AV_POSITIONS=${CNN_AV_POSITIONS:-16286511}
CNN_BATCH_SIZE=${CNN_BATCH_SIZE:-2048}
CNN_AV_BATCH_SIZE=${CNN_AV_BATCH_SIZE:-1024}

SF_SOURCE=${SF_SOURCE:-artifacts/squareformer_v2_full_tonight_explicit_budget_overfit}
SF_PHASE2=${SF_PHASE2:-artifacts/squareformer_v2_phase2_low_lr}
SF_PHASE3=${SF_PHASE3:-artifacts/squareformer_v2_phase3_tiny_lr}
SF_STREAM_CONFIG=${SF_STREAM_CONFIG:-data/training_streams/squareformer_v2_public_lichess1m_cached_chessbench32shards_direct_cached_v1.json}
SF_VALUE_CACHE=${SF_VALUE_CACHE:-data/public_teacher_overlays/lichess_position_eval_1m_v1/cache_compact_v1}
SF_POLICY_ROWS=${SF_POLICY_ROWS:-100000000}
SF_VALUE_ROWS=${SF_VALUE_ROWS:-5000000}
SF_AV_POSITIONS=${SF_AV_POSITIONS:-16286511}
SF_BATCH_SIZE=${SF_BATCH_SIZE:-512}
SF_AV_BATCH_SIZE=${SF_AV_BATCH_SIZE:-256}

log(){ echo "[$(date -Is)] $*" | tee -a "$LOGDIR/queue.log"; }
write_state(){ printf '%s\n' "$*" > "$STATUS/state.txt"; }
require_file(){ [[ -f "$1" ]] || { log "missing required file: $1"; exit 2; }; }
require_dir(){ [[ -d "$1" ]] || { log "missing required dir: $1"; exit 2; }; }

wait_run_done(){
  local run="$1" name="$2"
  write_state "waiting_for_${name}_pid"
  log "waiting for $name pid: $run/status/train.pid"
  while [[ ! -s "$run/status/train.pid" ]]; do sleep 180; done
  local pid; pid=$(cat "$run/status/train.pid")
  write_state "waiting_for_${name}_finish_pid_${pid}"
  log "waiting for $name pid=$pid to finish"
  while ps -p "$pid" >/dev/null 2>&1; do sleep 180; done
  log "$name pid=$pid exited"
}

best_resume_for(){
  local run="$1"
  if [[ -s "$run/checkpoints/best.pt" ]]; then echo "$run/checkpoints/best.pt"; else echo "$run/model.pt"; fi
}

launch_cnn64_phase(){
  local name="$1" run="$2" resume="$3" lr="$4"
  mkdir -p "$run/logs" "$run/status" "$run/checkpoints"
  require_file "$RES_MANIFEST"; require_dir "$RES_DEV_CACHE"; require_file "$AV_CACHE"; require_file "$resume"; require_file training/train_residual_av_multicache_torch.py
  write_state "launching_${name}"
  log "launching $name run=$run resume=$resume lr=$lr"
  nohup bash -lc "cd '$ROOT' && '$PY' training/train_residual_av_multicache_torch.py \
    --manifest '$RES_MANIFEST' \
    --dev-cache '$RES_DEV_CACHE' \
    --av-cache '$AV_CACHE' \
    --resume '$resume' \
    --out '$run/model.pt' \
    --onnx-out '$run/model.onnx' \
    --meta-out '$run/model.meta.json' \
    --checkpoint-dir '$run/checkpoints' \
    --best-checkpoint '$run/checkpoints/best.pt' \
    --channels 64 --blocks 6 \
    --policy-rows '$CNN_POLICY_ROWS' --av-positions '$CNN_AV_POSITIONS' --epochs 1 \
    --batch-size '$CNN_BATCH_SIZE' --av-batch-size '$CNN_AV_BATCH_SIZE' \
    --lr '$lr' --weight-decay 1e-4 \
    --eval-every-steps 10000 --checkpoint-every-steps 10000 --progress-every 500 \
    --max-dev-rows 100000 --max-av-dev-positions 10000 \
    --device cuda --amp --amp-dtype bf16 --export-av-head" \
    > "$run/logs/train.log" 2>&1 &
  echo $! > "$run/status/train.pid"
  date -Is > "$run/status/train.started"
  { echo "arch=64x6"; echo "phase=$name"; echo "resume=$resume"; echo "lr=$lr"; echo "policy_rows=$CNN_POLICY_ROWS"; echo "av_positions=$CNN_AV_POSITIONS"; } > "$run/status/config.env"
  log "$name started pid=$(cat "$run/status/train.pid")"
}

launch_sf_phase(){
  local name="$1" run="$2" resume="$3" lr="$4"
  mkdir -p "$run/logs" "$run/status" "$run/checkpoints"
  require_file "$SF_STREAM_CONFIG"; require_file "$SF_VALUE_CACHE/meta.json"; require_file "$AV_CACHE"; require_file "$resume"; require_file training/train_squareformer_v2_torch.py
  write_state "launching_${name}"
  log "launching $name run=$run resume=$resume lr=$lr"
  nohup bash -lc "cd '$ROOT' && '$PY' training/train_squareformer_v2_torch.py \
    --stream-config '$SF_STREAM_CONFIG' \
    --value-cache '$SF_VALUE_CACHE' \
    --av-cache '$AV_CACHE' \
    --resume '$resume' \
    --out '$run/model.pt' \
    --meta-out '$run/model.meta.json' \
    --checkpoint-dir '$run/checkpoints' \
    --layers 6 --d-model 128 --heads 4 --d-ff 256 --history-plies 2 --relation-bias \
    --input-mode onehot --grad-accum-steps 1 \
    --policy-rows '$SF_POLICY_ROWS' --value-rows '$SF_VALUE_ROWS' --av-positions '$SF_AV_POSITIONS' \
    --max-value-rows 0 --max-av-positions 0 \
    --max-dev-rows 50000 --max-av-dev-positions 10000 \
    --batch-size '$SF_BATCH_SIZE' --av-batch-size '$SF_AV_BATCH_SIZE' --epochs 1 \
    --lr '$lr' --weight-decay 1e-4 \
    --device cuda --amp --amp-dtype bf16 --progress-every 500 \
    --checkpoint-every-steps 10000 --eval-every-steps 25000 \
    --early-stop-patience 4 --early-stop-metric composite" \
    > "$run/logs/train.log" 2>&1 &
  echo $! > "$run/status/train.pid"
  date -Is > "$run/status/train.started"
  { echo "arch=squareformer_v2"; echo "phase=$name"; echo "resume=$resume"; echo "lr=$lr"; echo "policy_rows=$SF_POLICY_ROWS"; echo "value_rows=$SF_VALUE_ROWS"; echo "av_positions=$SF_AV_POSITIONS"; } > "$run/status/config.env"
  log "$name started pid=$(cat "$run/status/train.pid")"
}

main(){
  cd "$ROOT"
  wait_run_done "$WAIT_RUN" "cnn80_phase3"

  local c2_resume; c2_resume=$(best_resume_for "$CNN64_PHASE1")
  launch_cnn64_phase "cnn64_phase2_low_lr" "$CNN64_PHASE2" "$c2_resume" "1e-5"
  wait_run_done "$CNN64_PHASE2" "cnn64_phase2"

  local c3_resume; c3_resume=$(best_resume_for "$CNN64_PHASE2")
  launch_cnn64_phase "cnn64_phase3_tiny_lr" "$CNN64_PHASE3" "$c3_resume" "5e-6"
  wait_run_done "$CNN64_PHASE3" "cnn64_phase3"

  local s2_resume; s2_resume=$(best_resume_for "$SF_SOURCE")
  launch_sf_phase "squareformer_phase2_low_lr" "$SF_PHASE2" "$s2_resume" "1e-4"
  wait_run_done "$SF_PHASE2" "squareformer_phase2"

  local s3_resume; s3_resume=$(best_resume_for "$SF_PHASE2")
  launch_sf_phase "squareformer_phase3_tiny_lr" "$SF_PHASE3" "$s3_resume" "5e-5"
  write_state "squareformer_phase3_launched"
}
main "$@"
