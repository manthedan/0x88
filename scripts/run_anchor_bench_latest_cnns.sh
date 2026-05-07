#!/usr/bin/env bash
set -euo pipefail
mkdir -p artifacts/anchor_arena/latest_cnns
COMMON=(--openings-file=eval/opening_suite_uho_lite_v1.fen --pairs=10 --visits=32 --stockfish-levels=1320,1600 --stockfish-nodes=32 --max-plies=100)
MAIA='maia1100|.local_engines/maia/lc0-maia-1100.sh|32,maia1500|.local_engines/maia/lc0-maia-1500.sh|32,maia1900|.local_engines/maia/lc0-maia-1900.sh|32'
run_one() {
  local name="$1" onnx="$2" meta="$3"
  echo "[$(date -Is)] START $name"
  node --experimental-strip-types eval/uci_anchor_arena.mjs \
    --candidate="$name:$onnx:$meta" \
    "${COMMON[@]}" \
    --uci-anchors="$MAIA" \
    --out="artifacts/anchor_arena/latest_cnns/${name}_stockfish_maia_uho_v1.json" \
    > "artifacts/anchor_arena/latest_cnns/${name}_stockfish_maia_uho_v1.log" 2>&1
  echo "[$(date -Is)] DONE $name"
  tail -30 "artifacts/anchor_arena/latest_cnns/${name}_stockfish_maia_uho_v1.log"
}
run_one 32x4 artifacts/residual_32x4_history2.onnx artifacts/residual_32x4_history2.meta.json
run_one 48x5 artifacts/arena_10m_guarded/48x5_e9.onnx artifacts/arena_10m_guarded/48x5_e9.meta.json
run_one 64x6 artifacts/arena_10m_guarded/64x6_e12_ema.onnx artifacts/arena_10m_guarded/64x6_e12_ema.meta.json
