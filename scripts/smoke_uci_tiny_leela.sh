#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="/home/ddbb/projects/tiny_leela"
cd "$ROOT"
MODEL="${MODEL:-artifacts/supervised_1m_v1_48x5_smoke.onnx}"
META="${META:-artifacts/supervised_1m_v1_48x5_smoke.meta.json}"
OUT="${OUT:-artifacts/analysis/uci_tiny_leela_smoke.txt}"
mkdir -p "$(dirname "$OUT")"
{
  printf 'uci\n'
  sleep 0.2
  printf 'isready\n'
  sleep 1
  printf 'ucinewgame\nposition startpos\ngo nodes 1\n'
  sleep 2
  printf 'quit\n'
} | ORT_INTRA_OP_NUM_THREADS=1 node --experimental-strip-types scripts/uci_tiny_leela.mjs \
  --model "$MODEL" --meta "$META" --visits 1 --batch-size 1 > "$OUT"
grep -q '^uciok$' "$OUT"
grep -q '^readyok$' "$OUT"
grep -q '^bestmove [a-h][1-8][a-h][1-8]' "$OUT"
echo "wrote $OUT"
