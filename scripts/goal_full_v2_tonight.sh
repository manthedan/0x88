#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(pwd)}
PY=${PY:-.venv-onnx/bin/python}
ART=${ART:-artifacts/goal_full_v2_tonight}
LOGDIR="$ART/logs"
STATUS="$ART/status"
mkdir -p "$LOGDIR" "$STATUS"

STREAM_CONFIG=${STREAM_CONFIG:-data/training_streams/squareformer_v2_public_lichess1m_cached_chessbench32shards_direct_cached_v1.json}
VALUE_CACHE=${VALUE_CACHE:-data/public_teacher_overlays/lichess_position_eval_1m_v1/cache_compact_v1}
AV_COLLECTION=${AV_COLLECTION:-data/public_teacher_overlays/chessbench_full_policy_value_direct_top8_32shards_v1/collection_manifest.json}
OUT_RUN=${OUT_RUN:-artifacts/squareformer_v2_full_tonight_explicit_budget_overfit}
POLICY_ROWS=${POLICY_ROWS:-100000000}
VALUE_ROWS=${VALUE_ROWS:-5000000}
AV_POSITIONS=${AV_POSITIONS:-16286511}
EPOCHS=${EPOCHS:-4}
CHECKPOINT_EVERY_STEPS=${CHECKPOINT_EVERY_STEPS:-10000}
EVAL_EVERY_STEPS=${EVAL_EVERY_STEPS:-25000}
EARLY_STOP_PATIENCE=${EARLY_STOP_PATIENCE:-4}
INPUT_MODE=${INPUT_MODE:-embedding}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-1}

log(){ echo "[$(date -Is)] $*" | tee -a "$LOGDIR/goal.log"; }
metric(){ echo "METRIC $1=$2" | tee -a "$LOGDIR/goal.log"; }

require_file(){ if [[ ! -f "$1" ]]; then log "missing required file: $1"; exit 2; fi; }
require_dir(){ if [[ ! -d "$1" ]]; then log "missing required dir: $1"; exit 2; fi; }

write_status(){ printf '%s\n' "$*" > "$STATUS/state.txt"; }

train_cmd(){
  local max_rows=$1 batch=$2 av_batch=$3 run_dir=$4 progress=${5:-500}
  mkdir -p "$run_dir/logs" "$run_dir/checkpoints" "$run_dir/status"
  "$PY" training/train_squareformer_v2_torch.py \
    --stream-config "$STREAM_CONFIG" \
    --value-cache "$VALUE_CACHE" \
    --av-cache "$AV_COLLECTION" \
    --out "$run_dir/model.pt" \
    --meta-out "$run_dir/model.meta.json" \
    --checkpoint-dir "$run_dir/checkpoints" \
    --layers 6 --d-model 128 --heads 4 --d-ff 256 --history-plies 2 --relation-bias \
    --input-mode "$INPUT_MODE" --grad-accum-steps "$GRAD_ACCUM_STEPS" \
    --max-rows "$max_rows" --max-value-rows 0 --max-av-positions 0 \
    --max-dev-rows 20000 --max-av-dev-positions 5000 \
    --batch-size "$batch" --av-batch-size "$av_batch" --epochs 1 \
    --lr 3e-4 --weight-decay 1e-4 \
    --device cuda --amp --amp-dtype bf16 --progress-every "$progress"
}

train_budget_cmd(){
  local policy_rows=$1 value_rows=$2 av_positions=$3 batch=$4 av_batch=$5 epochs=$6 run_dir=$7 progress=${8:-500}
  mkdir -p "$run_dir/logs" "$run_dir/checkpoints" "$run_dir/status"
  "$PY" training/train_squareformer_v2_torch.py \
    --stream-config "$STREAM_CONFIG" \
    --value-cache "$VALUE_CACHE" \
    --av-cache "$AV_COLLECTION" \
    --out "$run_dir/model.pt" \
    --meta-out "$run_dir/model.meta.json" \
    --checkpoint-dir "$run_dir/checkpoints" \
    --layers 6 --d-model 128 --heads 4 --d-ff 256 --history-plies 2 --relation-bias \
    --input-mode "$INPUT_MODE" --grad-accum-steps "$GRAD_ACCUM_STEPS" \
    --policy-rows "$policy_rows" --value-rows "$value_rows" --av-positions "$av_positions" \
    --max-value-rows 0 --max-av-positions 0 \
    --max-dev-rows 50000 --max-av-dev-positions 10000 \
    --batch-size "$batch" --av-batch-size "$av_batch" --epochs "$epochs" \
    --lr 3e-4 --weight-decay 1e-4 \
    --device cuda --amp --amp-dtype bf16 --progress-every "$progress" \
    --checkpoint-every-steps "$CHECKPOINT_EVERY_STEPS" \
    --eval-every-steps "$EVAL_EVERY_STEPS" \
    --early-stop-patience "$EARLY_STOP_PATIENCE" \
    --early-stop-metric composite
}

parse_seconds_per_step(){
  local log=$1
  "$PY" - "$log" <<'PY'
import re,sys
p=sys.argv[1]
steps=[]
for line in open(p,errors='ignore'):
    m=re.search(r'progress epoch=1 step=(\d+)(?:/\d+)? seconds=([0-9.]+)', line)
    if m: steps.append((int(m.group(1)), float(m.group(2))))
if len(steps)>=2:
    s0,t0=steps[0]; s1,t1=steps[-1]
    print((t1-t0)/max(1,s1-s0))
elif steps:
    s,t=steps[-1]; print(t/max(1,s))
else:
    print('nan')
PY
}

launch_bg_budget_train(){
  local policy_rows=$1 value_rows=$2 av_positions=$3 batch=$4 av_batch=$5 epochs=$6 run_dir=$7
  mkdir -p "$run_dir/logs" "$run_dir/checkpoints" "$run_dir/status"
  if [[ -f "$run_dir/status/train.pid" ]] && ps -p "$(cat "$run_dir/status/train.pid")" >/dev/null 2>&1; then
    log "training already running pid=$(cat "$run_dir/status/train.pid") run=$run_dir"
    return 0
  fi
  log "launching explicit-budget V2 train policy_rows=$policy_rows value_rows=$value_rows av_positions=$av_positions epochs=$epochs batch=$batch av_batch=$av_batch run=$run_dir"
  nohup bash -lc "cd '$ROOT' && scripts/goal_full_v2_tonight.sh __run_train_budget '$policy_rows' '$value_rows' '$av_positions' '$batch' '$av_batch' '$epochs' '$run_dir'" \
    > "$run_dir/logs/train.log" 2>&1 &
  echo $! > "$run_dir/status/train.pid"
  date -Is > "$run_dir/status/train.started"
  echo "$batch" > "$run_dir/status/batch_size"
  echo "$av_batch" > "$run_dir/status/av_batch_size"
  echo "$policy_rows" > "$run_dir/status/policy_rows"
  echo "$value_rows" > "$run_dir/status/value_rows"
  echo "$av_positions" > "$run_dir/status/av_positions"
  echo "$epochs" > "$run_dir/status/epochs"
  log "started pid=$(cat "$run_dir/status/train.pid")"
}

monitor_train(){
  local run_dir=$1
  local pid_file="$run_dir/status/train.pid"
  [[ -f "$pid_file" ]] || return 1
  local pid; pid=$(cat "$pid_file")
  if ps -p "$pid" >/dev/null 2>&1; then
    local last
    last=$(tail -n 1 "$run_dir/logs/train.log" 2>/dev/null || true)
    log "running pid=$pid last=${last:0:240}"
    return 0
  fi
  if grep -q '^METRIC dev_policy_ce=' "$run_dir/logs/train.log" 2>/dev/null; then
    date -Is > "$run_dir/status/train.done"
    log "train completed run=$run_dir"
    grep '^METRIC' "$run_dir/logs/train.log" | tail -7 | tee -a "$LOGDIR/goal.log"
    return 2
  fi
  date -Is > "$run_dir/status/train.failed"
  log "train process exited without final metrics; inspect $run_dir/logs/train.log"
  return 3
}

main_loop(){
  cd "$ROOT"
  require_file "$STREAM_CONFIG"
  require_dir "$VALUE_CACHE"
  require_file "$AV_COLLECTION"
  require_file "training/train_squareformer_v2_torch.py"
  log "goal: get full V2 running tonight with all tensorized streams"
  write_status "preflight"

  # Syntax smoke for all tensor-cache scripts.
  "$PY" -m py_compile training/train_squareformer_v2_torch.py training/build_position_eval_cache.py training/build_action_value_cache.py training/build_chessbench_av_cache.py scripts/build_chessbench_av_caches_parallel.py scripts/write_v2_public_stream_config.py
  log "preflight ok"

  # Throughput probe is intentionally short; goal is to avoid launching a multi-hour run with a bad batch size.
  write_status "throughput_probe"
  local best_batch=1024 best_av=512 best_sps=999999
  for spec in "512 256" "1024 512" "2048 512" "4096 1024"; do
    local batch av run log sps
    read -r batch av <<<"$spec"
    run="$ART/probe_b${batch}_av${av}"
    log="$run/logs/train.log"
    if [[ ! -f "$run/status/done" ]]; then
      rm -rf "$run"; mkdir -p "$run/logs" "$run/status"
      log "probe batch=$batch av_batch=$av"
      if timeout 180 bash -lc "cd '$ROOT' && scripts/goal_full_v2_tonight.sh __run_train 262144 '$batch' '$av' '$run' 100" > "$log" 2>&1; then
        date -Is > "$run/status/done"
      else
        date -Is > "$run/status/failed"
        log "probe failed batch=$batch av_batch=$av"
        continue
      fi
    fi
    sps=$(parse_seconds_per_step "$log")
    log "probe_result batch=$batch av_batch=$av seconds_per_step=$sps"
    if "$PY" - "$sps" "$best_sps" <<'PY'
import math,sys
x=float(sys.argv[1]); b=float(sys.argv[2])
sys.exit(0 if math.isfinite(x) and x < b else 1)
PY
    then
      best_sps=$sps; best_batch=$batch; best_av=$av
    fi
  done
  metric best_seconds_per_step "$best_sps"
  metric best_batch_size "$best_batch"
  metric best_av_batch_size "$best_av"

  write_status "launch_full"
  launch_bg_budget_train "$POLICY_ROWS" "$VALUE_ROWS" "$AV_POSITIONS" "$best_batch" "$best_av" "$EPOCHS" "$OUT_RUN"

  write_status "monitoring"
  while true; do
    monitor_train "$OUT_RUN" || rc=$?; rc=${rc:-0}
    if [[ "$rc" == "2" ]]; then write_status "done"; exit 0; fi
    if [[ "$rc" == "3" ]]; then write_status "failed"; exit 3; fi
    rc=0
    sleep 60
  done
}

case "${1:-}" in
  __run_train)
    shift
    train_cmd "$@"
    ;;
  __run_train_budget)
    shift
    train_budget_cmd "$@"
    ;;
  *)
    main_loop
    ;;
esac
