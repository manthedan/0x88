#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/ddbb/projects/tiny_leela}"
PI_BIN="${PI_BIN:-/home/ddbb/.local/bin/pi}"
BASE="$ROOT/artifacts/pi_queue_doctor"
PROMPT="$BASE/hourly_prompt.md"
LOCK="$BASE/lock"
LOG_DIR="$BASE/logs"
SESSION_DIR="$BASE/sessions"
REPORT_DIR="$BASE/reports"
TIMEOUT="${PI_DOCTOR_TIMEOUT:-45m}"

mkdir -p "$BASE" "$LOG_DIR" "$SESSION_DIR" "$REPORT_DIR"
cd "$ROOT"

# Optional non-interactive secrets/config, e.g. ANTHROPIC_API_KEY/OPENAI_API_KEY/PI_MODEL.
# Keep this file out of git.
if [[ -f "$HOME/.pi/agent/cron.env" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$HOME/.pi/agent/cron.env"
  set +a
fi

exec 9>"$LOCK"
if ! flock -n 9; then
  echo "$(date -Is) hourly_pi_queue_doctor already running; skip" >> "$LOG_DIR/skips.log"
  exit 0
fi

export PATH="$HOME/.local/bin:$PATH"
export AWS_PROFILE="${AWS_PROFILE:-tiny-leela}"
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-west-2}"
export GIT_TERMINAL_PROMPT=0
export GIT_SSH_COMMAND="${GIT_SSH_COMMAND:-ssh -o BatchMode=yes -o ConnectTimeout=5}"

if [[ ! -x "$PI_BIN" ]]; then
  echo "$(date -Is) pi binary not executable: $PI_BIN" >> "$LOG_DIR/errors.log"
  exit 1
fi
if [[ ! -s "$PROMPT" ]]; then
  echo "$(date -Is) missing prompt: $PROMPT" >> "$LOG_DIR/errors.log"
  exit 1
fi

stamp="$(date -u +%Y%m%dT%H%M%SZ)"
log="$LOG_DIR/$stamp.log"
model_args=()
if [[ -n "${PI_MODEL:-}" ]]; then
  model_args+=(--model "$PI_MODEL")
fi

echo "$(date -Is) START hourly pi queue doctor" >> "$log"
set +e
timeout "$TIMEOUT" "$PI_BIN" \
  -p \
  --session-dir "$SESSION_DIR" \
  --tools read,bash,edit,write \
  "${model_args[@]}" \
  "@$PROMPT" \
  "Run the hourly queue doctor now. Inspect current statuses, apply only allowed bounded fixes, and write the required report." \
  >> "$log" 2>&1
rc=$?
set -e
if [[ "$rc" -eq 124 ]]; then
  echo "$(date -Is) TIMEOUT after $TIMEOUT" >> "$log"
elif [[ "$rc" -ne 0 ]]; then
  echo "$(date -Is) FAILED rc=$rc" >> "$log"
else
  echo "$(date -Is) DONE" >> "$log"
fi
exit "$rc"
