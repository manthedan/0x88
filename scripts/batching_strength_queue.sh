#!/bin/zsh
# Run batching-strength matches under SP/benchlock in <=16 minute chunks.
# Sleeps 20s after each chunk with the lock released.
# Usage: scripts/batching_strength_queue.sh [match-id ...]
set -u
cd "$(dirname "$0")/.."
SP=/private/tmp/claude-501/-Users-macthedan-projects-lc0-browser/701135f9-2bed-49a5-8cf9-e17058c5574a/scratchpad
export AGENT_BROWSER_EXECUTABLE_PATH="${AGENT_BROWSER_EXECUTABLE_PATH:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"
export AGENT_BROWSER_ARGS="${AGENT_BROWSER_ARGS:---enable-unsafe-webgpu}"
export AGENT_BROWSER_SESSION="${AGENT_BROWSER_SESSION:-batching-strength}"
unset AGENT_BROWSER_WEBGPU

matches=("$@")
if (( ${#matches} == 0 )); then
  matches=(
    lqo-v32 b2-v8 b2-v32 b1-v8 b1-v32
    lqo-v16 lqo-v64
    b2-v100 b1-v100
    lqo-v256
    b2-v400 b1-v400
    lqo-time b2-time
  )
fi

for match in "${matches[@]}"; do
  stall=0
  prev=""
  restart=(--restart-browser)
  echo "MATCH_START $match $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  while true; do
    "$SP/benchlock" node --experimental-strip-types scripts/batching_strength_match.mjs --match "$match" --minutes 16 "${restart[@]}"
    code=$?
    restart=()
    status="artifacts/batching-strength/$match/chunk-status.json"
    if [[ -f "$status" ]]; then
      line=$(node -e 'const fs=require("node:fs"); const s=JSON.parse(fs.readFileSync(process.argv[1],"utf8")); process.stdout.write(`games=${s.gamesDone}/${s.gamesTarget} ply=${s.ply ?? "-"} phase=${s.phase} load=${s.loadavg ?? "-"}`);' "$status")
      key=$(node -e 'const fs=require("node:fs"); const s=JSON.parse(fs.readFileSync(process.argv[1],"utf8")); process.stdout.write(`${s.gamesDone} ${s.ply ?? "-"} ${s.phase}`);' "$status")
      echo "CHUNK match=$match exit=$code $line"
    else
      echo "CHUNK match=$match exit=$code status=missing"
      key="missing"
    fi
    if (( code == 2 )); then
      echo "FATAL $match"
      exit 2
    fi
    if (( code == 3 )); then
      echo "DONE $match"
      break
    fi
    phase=${key##* }
    if [[ "$phase" == "loading" || "$phase" == "probing" ]]; then
      stall=0
    elif [[ "$key" == "$prev" ]]; then
      stall=$((stall + 1))
    else
      stall=0
    fi
    prev="$key"
    if (( stall >= 2 )); then
      echo "STALL $match no game or ply progress for two chunks"
      exit 1
    fi
    sleep 20
  done
done
echo "QUEUE_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
