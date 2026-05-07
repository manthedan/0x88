#!/usr/bin/env bash
set -euo pipefail
OUT_DIR=${OUT_DIR:-data/lichess_raw}
mkdir -p "$OUT_DIR"
if [ "$#" -eq 0 ]; then
  echo "Usage: $0 YYYY-MM [YYYY-MM ...]" >&2
  echo "Example: OUT_DIR=data/lichess_raw $0 2026-01 2026-02" >&2
  exit 2
fi
for ym in "$@"; do
  url="https://database.lichess.org/standard/lichess_db_standard_rated_${ym}.pgn.zst"
  out="$OUT_DIR/lichess_db_standard_rated_${ym}.pgn.zst"
  if [ -s "$out" ]; then
    echo "exists: $out"
    continue
  fi
  tmp="$out.part"
  echo "download: $url -> $out"
  if command -v aria2c >/dev/null 2>&1; then
    aria2c -x 8 -s 8 -c -o "$(basename "$tmp")" -d "$(dirname "$tmp")" "$url"
  else
    curl -L --continue-at - --fail --output "$tmp" "$url"
  fi
  mv "$tmp" "$out"
  echo "wrote: $out"
done
