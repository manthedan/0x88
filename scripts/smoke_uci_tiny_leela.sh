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
  printf 'setoption name Threads value 1\n'
  printf 'setoption name Hash value 16\n'
  printf 'isready\n'
  sleep 1
  printf 'ucinewgame\n'
  printf 'position startpos\n'
  printf 'go nodes 1\n'
  sleep 1
  printf 'position fen rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1 moves e7e5 g1f3\n'
  printf 'go movetime 20\n'
  sleep 1
  printf 'position startpos moves e2e4 e7e5\n'
  printf 'go nodes 2\n'
  sleep 0.1
  printf 'stop\n'
  sleep 1
  printf 'quit\n'
} | ORT_INTRA_OP_NUM_THREADS=1 node --experimental-strip-types scripts/uci_tiny_leela.mjs \
  --model "$MODEL" --meta "$META" --visits 1 --batch-size 1 > "$OUT"
grep -q '^uciok$' "$OUT"
grep -q '^readyok$' "$OUT"
[[ "$(grep -c '^bestmove \([a-h][1-8][a-h][1-8][qrbn]\?\|0000\)' "$OUT")" -ge 3 ]]
echo "wrote $OUT"
