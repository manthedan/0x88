#!/bin/bash
# Run the perf matrix against the Mac mini's Safari while builds and models stay on this laptop.
#
#   scripts/perf_matrix/mini.sh up [dist-dir] [laptop-port] [mini-port]   # server + reverse tunnel + sync scripts
#   scripts/perf_matrix/mini.sh run <name> [run_matrix args...]            # runs ON the mini (~/perf-matrix/out/<name>), under its lock
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

alive() { [ -f "$1" ] && kill -0 "$(cat "$1")" 2>/dev/null; }

case "$cmd" in
up)
  DIST=${1:-$REPO/dist-client}
  LPORT=${2:-8781}
  MPORT=${3:-18781}
  [ -d "$DIST" ] || { echo "no build at $DIST (BUILD_SCOPE=product npm run build:client)"; exit 1; }
  if ! alive "$STATE/server.pid"; then
    PORT=$LPORT nohup node "$REPO/scripts/serve_isolated_static.mjs" "$DIST" >"$STATE/server.log" 2>&1 &
    echo $! >"$STATE/server.pid"
  fi
  if ! alive "$STATE/tunnel.pid"; then
    ssh -N -o BatchMode=yes -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=4 \
      -R "127.0.0.1:$MPORT:127.0.0.1:$LPORT" "$HOST" >"$STATE/tunnel.log" 2>&1 &
    echo $! >"$STATE/tunnel.pid"
  fi
  echo "$MPORT" >"$STATE/mini-port"
  sleep 2
  alive "$STATE/tunnel.pid" || { echo "tunnel failed: $(cat "$STATE/tunnel.log")"; exit 1; }
  ssh -o BatchMode=yes "$HOST" "mkdir -p ~/$WORKSPACE/scripts/perf_matrix ~/$WORKSPACE/out"
  # Isolate this branch's scripts from agents using ~/perf-matrix; share only its mutex.
  rsync -a --delete "$REPO/scripts/perf_matrix/" "$HOST:$WORKSPACE/scripts/perf_matrix/"
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
fetch)
  NAME=$1
  DEST=${2:-$REPO/.local-dev-artifacts/perf-matrix/mini/$NAME}
  mkdir -p "$DEST"
  rsync -a "$HOST:$WORKSPACE/out/$NAME/" "$DEST/"
  echo "$DEST"
  ;;
down)
  for p in tunnel server; do
    if alive "$STATE/$p.pid"; then kill "$(cat "$STATE/$p.pid")" && echo "stopped $p $(cat "$STATE/$p.pid")"; fi
    rm -f "$STATE/$p.pid"
  done
  ;;
status)
  for p in tunnel server; do alive "$STATE/$p.pid" && echo "$p running ($(cat "$STATE/$p.pid"))" || echo "$p stopped"; done
  ssh -o BatchMode=yes "$HOST" 'cat ~/perf-matrix/safari.lock/owner 2>/dev/null || echo "mini lock free"; sysctl -n vm.loadavg'
  ;;
*) echo "usage: $0 up|run|fetch|down|status"; exit 2 ;;
esac
