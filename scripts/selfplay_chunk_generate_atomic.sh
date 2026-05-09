#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
OUT=""
ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --out)
      OUT="$2"; ARGS+=("$1" "$2"); shift 2;;
    --out=*)
      OUT="${1#--out=}"; ARGS+=("$1"); shift;;
    *) ARGS+=("$1"); shift;;
  esac
done
if [[ -z "$OUT" ]]; then
  echo "usage: $0 [selfplay_generate args...] --out PATH" >&2
  exit 2
fi
mkdir -p "$(dirname "$OUT")"
TMP="${OUT}.tmp.$$"
FINAL_ARGS=()
for ((i=0; i<${#ARGS[@]}; i++)); do
  if [[ "${ARGS[$i]}" == "--out" ]]; then
    FINAL_ARGS+=("--out" "$TMP"); i=$((i+1))
  elif [[ "${ARGS[$i]}" == --out=* ]]; then
    FINAL_ARGS+=("--out=$TMP")
  else
    FINAL_ARGS+=("${ARGS[$i]}")
  fi
done
cleanup(){ rm -f "$TMP"; }
trap cleanup EXIT
node --experimental-strip-types scripts/selfplay_generate.mjs "${FINAL_ARGS[@]}"
.venv-onnx/bin/python scripts/selfplay_chunk_validate.py "$TMP"
mv -f "$TMP" "$OUT"
trap - EXIT
printf 'METRIC selfplay_atomic_chunk_bytes=%s\n' "$(wc -c < "$OUT")"
printf 'METRIC selfplay_atomic_chunk_path=%s\n' "$OUT"
