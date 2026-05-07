#!/usr/bin/env bash
set -euo pipefail

NAME=${NAME:-chessformer_v1_100m_e3}
MODEL=${MODEL:-public/models/chessformer_v1_100m_e3_single.onnx}
META=${META:-public/models/chessformer_v1_100m_e3_single.meta.json}
OUT_DIR=${OUT_DIR:-artifacts/distributed_arena/${NAME}_$(date +%Y%m%d_%H%M%S)}
OPENINGS=${OPENINGS:-eval/opening_suite_uho_lite_v1.fen}
VISITS_LIST=${VISITS_LIST:-1,32,64,128,192,256,384,512}
PAIRS=${PAIRS:-3}
OPENING_SHARD_SIZE=${OPENING_SHARD_SIZE:-3}
MAX_PLIES=${MAX_PLIES:-100}
CPUCT=${CPUCT:-1.5}
JOBS=${JOBS:-3}
STOCKFISH_NODES=${STOCKFISH_NODES:-32}
MAIA_NODES=${MAIA_NODES:-32}
mkdir -p "$OUT_DIR/jobs" "$OUT_DIR/logs" "$OUT_DIR/merged"

mapfile -t OPENING_LINES < <(grep -vE '^\s*(#|$)' "$OPENINGS")
TOTAL_OPENINGS=${#OPENING_LINES[@]}
cat > "$OUT_DIR/manifest.json" <<EOF
{"name":"$NAME","model":"$MODEL","meta":"$META","openings":"$OPENINGS","total_openings":$TOTAL_OPENINGS,"visits_list":"$VISITS_LIST","pairs":$PAIRS,"opening_shard_size":$OPENING_SHARD_SIZE,"max_plies":$MAX_PLIES,"cpuct":$CPUCT,"jobs":$JOBS,"created_utc":"$(date -u +%Y-%m-%dT%H:%M:%SZ)","git_commit":"$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"}
EOF

run_one() {
  local tag="$1" visits="$2" start="$3" count="$4" anchor_kind="$5"
  local out="$OUT_DIR/jobs/${tag}.json"
  local log="$OUT_DIR/logs/${tag}.log"
  local args=(node --experimental-strip-types eval/uci_anchor_arena.mjs --candidate="${NAME}_${tag}:${MODEL}:${META}" --openings-file="$OPENINGS" --opening-start="$start" --opening-count="$count" --pairs="$PAIRS" --visits="$visits" --cpuct="$CPUCT" --max-plies="$MAX_PLIES" --stockfish-nodes="$STOCKFISH_NODES" --out "$out")
  case "$anchor_kind" in
    sf1320) args+=(--stockfish-levels=1320 --uci-anchors=) ;;
    sf1600) args+=(--stockfish-levels=1600 --uci-anchors=) ;;
    maia1100) args+=(--include-stockfish=false --stockfish-levels= --uci-anchors="maia1100|.local_engines/maia/lc0-maia-1100.sh|${MAIA_NODES}") ;;
    *) echo "bad anchor $anchor_kind" >&2; return 2 ;;
  esac
  echo "[distributed] start $tag" >&2
  "${args[@]}" > "$log" 2>&1 && touch "${out}.done"
}

pids=()
launch() { while [[ ${#pids[@]} -ge $JOBS ]]; do wait "${pids[0]}"; pids=("${pids[@]:1}"); done; run_one "$@" & pids+=("$!"); }
IFS=',' read -ra VISITS <<< "$VISITS_LIST"
for v in "${VISITS[@]}"; do
  for start in $(seq 0 $OPENING_SHARD_SIZE $((TOTAL_OPENINGS-1))); do
    count=$OPENING_SHARD_SIZE
    if (( start + count > TOTAL_OPENINGS )); then count=$((TOTAL_OPENINGS-start)); fi
    for anchor in sf1320 sf1600 maia1100; do
      launch "v${v}_${anchor}_o${start}_${count}" "$v" "$start" "$count" "$anchor"
    done
  done
done
for p in "${pids[@]}"; do wait "$p"; done

for v in "${VISITS[@]}"; do
  node eval/merge_uci_anchor_arena.mjs --inputs "$OUT_DIR/jobs/v${v}_*.json" --out "$OUT_DIR/merged/v${v}.json" --allow-mixed=true | tee "$OUT_DIR/merged/v${v}.metrics.txt"
done

echo "[distributed] wrote $OUT_DIR"
