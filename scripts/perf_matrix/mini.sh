#!/bin/bash
# Run the perf matrix against the Mac mini's Safari while builds and models stay on this laptop.
#
#   scripts/perf_matrix/mini.sh up [dist-dir] [laptop-port] [mini-port]   # server + reverse tunnel + sync scripts
#   scripts/perf_matrix/mini.sh run <name> [run_matrix args...]            # runs ON the mini (~/perf-matrix/out/<name>), under its lock
#   scripts/perf_matrix/mini.sh run-routes <name> [perf:routes args...]     # route gate, same shared mutex
#   scripts/perf_matrix/mini.sh fetch <name> [local-dir]                   # copy results back
#   scripts/perf_matrix/mini.sh down                                       # stop the server + tunnel started by `up`
#   scripts/perf_matrix/mini.sh status
#
# The mini's Safari loads http://127.0.0.1:<mini-port> (a secure, cross-origin-isolated context) through
# `ssh -R`; pidfp, the per-WebContent sampling and the proxy run on the mini. The mini's safaridriver
# (127.0.0.1:4446) is started and stopped by its owner, never by this script. One Safari session at a time
# per machine: run_matrix --lock-dir ~/perf-matrix/safari.lock (the laptop benchlock is not involved).
set -euo pipefail
HOST=${PERF_MINI_HOST:-mac-mini}
DRIVER=${PERF_MINI_DRIVER:-http://127.0.0.1:4446}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
STATE=${PERF_MINI_STATE:-"$REPO/.local-dev-artifacts/perf-matrix/mini-state"}
WORKSPACE=${PERF_MINI_WORKSPACE:-perf-mobile-soak-phase3}
[[ "$WORKSPACE" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'invalid isolated mini workspace'; exit 1; }
mkdir -p "$STATE"
cmd=${1:-status}
shift || true

process_identity() { LC_ALL=C ps -p "$1" -o lstart= -o command= 2>/dev/null; }
recorded_pid_live() {
  [ -f "$1" ] || return 1
  local pid
  pid=$(<"$1")
  [[ "$pid" =~ ^[1-9][0-9]*$ ]] && kill -0 "$pid" 2>/dev/null
}
alive() {
  recorded_pid_live "$1" && [ -f "$1.identity" ] || return 1
  local actual expected
  actual=$(process_identity "$(<"$1")") || return 1
  expected=$(<"$1.identity")
  [ -n "$expected" ] && [ "$actual" = "$expected" ]
}
require_owned_or_dead() {
  if recorded_pid_live "$1" && ! alive "$1"; then
    echo "refusing unverified live PID in $1 (missing/mismatched start time or command); owner inspection required" >&2
    return 1
  fi
}
record_helper() {
  local file=$1 pid=$2 expected=$3 identity attempt
  printf '%s\n' "$pid" >"$file"
  # Wait for nohup/ssh to exec before recording an identity; never trust the fork's shell command.
  for attempt in {1..20}; do
    identity=$(process_identity "$pid") || return 1
    if [[ "$identity" == *"$expected"* ]]; then
      printf '%s\n' "$identity" >"$file.identity"
      return 0
    fi
    sleep 0.1
  done
  echo "could not verify started helper $pid; leaving its PID for owner inspection" >&2
  return 1
}

case "$cmd" in
up)
  DIST=${1:-$REPO/dist-client}
  LPORT=${2:-8781}
  MPORT=${3:-18781}
  [ -d "$DIST" ] || { echo "no build at $DIST (BUILD_SCOPE=product npm run build:client)"; exit 1; }
  require_owned_or_dead "$STATE/server.pid"
  require_owned_or_dead "$STATE/tunnel.pid"
  if ! alive "$STATE/server.pid"; then
    PORT=$LPORT nohup node "$REPO/scripts/serve_isolated_static.mjs" "$DIST" >"$STATE/server.log" 2>&1 &
    record_helper "$STATE/server.pid" "$!" "$REPO/scripts/serve_isolated_static.mjs $DIST"
  fi
  if ! alive "$STATE/tunnel.pid"; then
    ssh -N -o BatchMode=yes -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=4 \
      -R "127.0.0.1:$MPORT:127.0.0.1:$LPORT" "$HOST" >"$STATE/tunnel.log" 2>&1 &
    record_helper "$STATE/tunnel.pid" "$!" "-R 127.0.0.1:$MPORT:127.0.0.1:$LPORT $HOST"
  fi
  echo "$MPORT" >"$STATE/mini-port"
  sleep 2
  alive "$STATE/tunnel.pid" || { echo "tunnel failed: $(cat "$STATE/tunnel.log")"; exit 1; }
  ssh -o BatchMode=yes "$HOST" "mkdir -p ~/$WORKSPACE/scripts/perf_matrix ~/$WORKSPACE/out"
  # Isolate this branch's scripts from agents using ~/perf-matrix; share only its mutex.
  rsync -a --delete "$REPO/scripts/perf_matrix/" "$HOST:$WORKSPACE/scripts/perf_matrix/"
  ssh -o BatchMode=yes "$HOST" "mkdir -p ~/$WORKSPACE/scripts/perf_routes ~/$WORKSPACE/scripts/lib"
  rsync -a --delete "$REPO/scripts/perf_routes/" "$HOST:$WORKSPACE/scripts/perf_routes/"
  rsync -a "$REPO/scripts/lib/perf_canary_stats.mjs" "$HOST:$WORKSPACE/scripts/lib/"
  code=$(ssh -o BatchMode=yes "$HOST" "curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$MPORT/lab/mobile-soak/")
  echo "server pid $(cat "$STATE/server.pid") (laptop :$LPORT), tunnel pid $(cat "$STATE/tunnel.pid") (mini :$MPORT), page HTTP $code"
  ;;
run)
  NAME=$1
  shift
  MPORT=$(cat "$STATE/mini-port")
  alive "$STATE/tunnel.pid" || { echo "tunnel is down: run '$0 up' first"; exit 1; }
  args=$(printf '%q ' "$@")
  ssh -o BatchMode=yes "$HOST" "cd ~/$WORKSPACE && export PATH=/opt/homebrew/bin:\$PATH && \
    echo \"mini load: \$(sysctl -n vm.loadavg)\" && \
    node scripts/perf_matrix/run_matrix.mjs --driver $DRIVER --lock-dir \$HOME/perf-matrix/safari.lock \
      --base http://127.0.0.1:$MPORT --out out/$NAME $args"
  ;;
run-routes)
  NAME=$1
  [[ "$NAME" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'invalid route output name'; exit 1; }
  shift
  MPORT=$(cat "$STATE/mini-port")
  alive "$STATE/tunnel.pid" || { echo "tunnel is down: run '$0 up' first"; exit 1; }
  # Explicit URL flags support two independently transported built previews. Self-comparison
  # uses the same upstream bytes; the session proxy still supplies fresh origins per arm.
  defaults=()
  case " $* " in *' --baseline-url '*) ;; *) defaults+=(--baseline-url "http://127.0.0.1:$MPORT");; esac
  case " $* " in *' --candidate-url '*) ;; *) defaults+=(--candidate-url "http://127.0.0.1:$MPORT");; esac
  args=$(printf '%q ' "${defaults[@]}" "$@")
  ssh -o BatchMode=yes "$HOST" "cd ~/$WORKSPACE && export PATH=/opt/homebrew/bin:\$PATH && \
    echo \"mini load: \$(sysctl -n vm.loadavg)\" && \
    node scripts/perf_routes/run.mjs --driver $DRIVER --lock-dir \$HOME/perf-matrix/safari.lock \
      --out out/$NAME $args"
  ;;
fetch)
  NAME=$1
  DEST=${2:-$REPO/.local-dev-artifacts/perf-matrix/mini/$NAME}
  mkdir -p "$DEST"
  rsync -a "$HOST:$WORKSPACE/out/$NAME/" "$DEST/"
  echo "$DEST"
  ;;
down)
  result=0
  for p in tunnel server; do
    if ! require_owned_or_dead "$STATE/$p.pid"; then result=1; continue; fi
    if alive "$STATE/$p.pid"; then kill "$(<"$STATE/$p.pid")" && echo "stopped $p $(<"$STATE/$p.pid")"; fi
    rm -f "$STATE/$p.pid" "$STATE/$p.pid.identity"
  done
  exit "$result"
  ;;
status)
  for p in tunnel server; do alive "$STATE/$p.pid" && echo "$p running ($(cat "$STATE/$p.pid"))" || echo "$p stopped"; done
  ssh -o BatchMode=yes "$HOST" 'cat ~/perf-matrix/safari.lock/owner 2>/dev/null || echo "mini lock free"; sysctl -n vm.loadavg'
  ;;
*) echo "usage: $0 up|run|run-routes|fetch|down|status"; exit 2 ;;
esac
