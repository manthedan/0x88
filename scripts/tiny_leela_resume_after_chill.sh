#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-/home/ddbb/projects/tiny_leela}
cd "$ROOT"
MARKER=${MARKER:-artifacts/pi_queue_doctor/user_chill.marker}
LOG=${LOG:-artifacts/pi_queue_doctor/chill.log}
PID_LIST=${PID_LIST:-artifacts/pi_queue_doctor/chill_paused_pids.txt}
log(){ printf '%s %s\n' "$(date -Is)" "$*" | tee -a "$LOG"; }

if [[ -s "$PID_LIST" ]]; then
  while IFS=$'\t' read -r label pid pgid file; do
    [[ -n "${pid:-}" ]] || continue
    if kill -0 "$pid" 2>/dev/null; then
      if [[ -n "${pgid:-}" ]]; then
        log "SIGCONT process group $pgid for $label pid=$pid"
        kill -CONT "-$pgid" 2>/dev/null || kill -CONT "$pid" 2>/dev/null || true
      else
        log "SIGCONT pid $pid for $label"
        kill -CONT "$pid" 2>/dev/null || true
      fi
    else
      log "skip dead $label pid=$pid"
    fi
  done < "$PID_LIST"
else
  log "no paused pid list found at $PID_LIST"
fi
rm -f "$MARKER"
log "Removed chill marker: $MARKER"
log "If the scheduled prompt was disabled, re-enable it separately."
