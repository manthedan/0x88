#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-/home/ddbb/projects/tiny_leela}
cd "$ROOT"
MARKER=${MARKER:-artifacts/pi_queue_doctor/user_chill.marker}
LOG=${LOG:-artifacts/pi_queue_doctor/chill.log}
PID_LIST=${PID_LIST:-artifacts/pi_queue_doctor/chill_paused_pids.txt}
mkdir -p "$(dirname "$MARKER")"
reason=${1:-manual chill requested}
now=$(date -Is)
printf '%s %s\n' "$now" "$reason" > "$MARKER"
: > "$PID_LIST"
log(){ printf '%s %s\n' "$(date -Is)" "$*" | tee -a "$LOG"; }
log "CHILL marker written: $MARKER reason=$reason"

pause_pid_file(){
  local file=$1 label=$2
  [[ -s "$file" ]] || return 0
  local pid; pid=$(tr -dc '0-9' < "$file" || true)
  [[ -n "$pid" ]] || return 0
  if ! kill -0 "$pid" 2>/dev/null; then
    log "skip stale $label pid=$pid file=$file"
    return 0
  fi
  local pgid; pgid=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ' || true)
  if [[ -n "$pgid" ]]; then
    log "SIGSTOP process group $pgid for $label pid=$pid file=$file"
    kill -STOP "-$pgid" 2>/dev/null || kill -STOP "$pid" 2>/dev/null || true
    printf '%s\t%s\t%s\t%s\n' "$label" "$pid" "$pgid" "$file" >> "$PID_LIST"
  else
    log "SIGSTOP pid $pid for $label file=$file"
    kill -STOP "$pid" 2>/dev/null || true
    printf '%s\t%s\t\t%s\n' "$label" "$pid" "$file" >> "$PID_LIST"
  fi
}

# Local CPU/GPU queues only. AWS Batch jobs are not cancelled here.
pause_pid_file artifacts/top3_100m_overnight_20260509/rebuild_sidecar_parallel_then_train_mf80.pid top3-mf80-sidecar-chain
pause_pid_file artifacts/top3_100m_overnight_20260509/sidecar.pid top3-sidecar
pause_pid_file artifacts/top3_100m_overnight_20260509/pid top3-main-queue
pause_pid_file artifacts/cloud_h8_dataset_100m/repair_add_2023_extra_submit.pid h8-repair-local-upload
pause_pid_file artifacts/cloud_h8_dataset_100m/overnight_gate/pid h8-cloud-gate

log "Paused pid list: $PID_LIST"
log "Hourly checker will no-op while $MARKER exists. Disable the scheduled prompt too if you want zero chat wakeups."
