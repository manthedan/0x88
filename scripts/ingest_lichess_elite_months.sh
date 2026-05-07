#!/usr/bin/env bash
set -euo pipefail
RAW_DIR=${RAW_DIR:-data/lichess_elite_raw}
OUT_DIR=${OUT_DIR:-data/lichess_elite_training}
MAX_GAMES=${MAX_GAMES:-250000}
MAX_POSITIONS=${MAX_POSITIONS:-1200000}
SKIP_PLIES=${SKIP_PLIES:-10}
MAX_PLIES_PER_GAME=${MAX_PLIES_PER_GAME:-90}
MIN_ESTIMATED_SECONDS=${MIN_ESTIMATED_SECONDS:-180}
INCLUDE_BULLET=${INCLUDE_BULLET:-0}
mkdir -p "$OUT_DIR"
if [ "$#" -eq 0 ]; then
  echo "Usage: $0 YYYY-MM [YYYY-MM ...]" >&2
  echo "Env: RAW_DIR OUT_DIR MAX_GAMES MAX_POSITIONS SKIP_PLIES MAX_PLIES_PER_GAME MIN_ESTIMATED_SECONDS INCLUDE_BULLET" >&2
  exit 2
fi
for ym in "$@"; do
  zip="$RAW_DIR/lichess_elite_${ym}.zip"
  out="$OUT_DIR/lichess_elite_${ym}.jsonl"
  if [ ! -s "$zip" ]; then
    echo "missing Elite zip: $zip" >&2
    exit 1
  fi
  if [ -s "$out" ]; then
    echo "exists: $out"
    continue
  fi
  echo "ingest elite: $zip -> $out"
  node --experimental-strip-types scripts/lichess_pgn_to_training.mjs \
    --pgn "$zip" \
    --out "$out" \
    --teacher "lichess_elite_${ym}" \
    --no-elo-filter \
    --max-games "$MAX_GAMES" \
    --max-positions "$MAX_POSITIONS" \
    --skip-plies "$SKIP_PLIES" \
    --max-plies-per-game "$MAX_PLIES_PER_GAME" \
    --min-estimated-seconds "$MIN_ESTIMATED_SECONDS" \
    $( [ "$INCLUDE_BULLET" = "1" ] && printf '%s' '--include-bullet' )
done
