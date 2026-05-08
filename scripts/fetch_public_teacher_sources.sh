#!/usr/bin/env bash
set -euo pipefail

# Helper for acquiring public teacher sources. Large downloads are deliberately
# opt-in: run individual commands or set RUN_DOWNLOADS=1 after checking space.
ROOT=${ROOT:-data/public_teacher_raw}
mkdir -p "$ROOT"/{lichess_eval,lichess_puzzles,tcec,fishtest,chessbench,lc0}

cat <<'MSG'
Public teacher source acquisition helper

This script prints safe download commands by default. Set RUN_DOWNLOADS=1 to run
small/direct downloads. Hugging Face/TCEC/lc0 bulk data usually needs manual
selection of shards first.
MSG

run_or_print() {
  if [[ "${RUN_DOWNLOADS:-0}" == "1" ]]; then
    echo "+ $*"; eval "$@"
  else
    echo "$*"
  fi
}

echo
cat <<'MSG'
# Lichess public position evaluations (large JSONL.zst, CC0):
MSG
run_or_print "wget -c -O '$ROOT/lichess_eval/lichess_db_eval.jsonl.zst' 'https://database.lichess.org/lichess_db_eval.jsonl.zst'"

echo
cat <<'MSG'
# Lichess puzzles (CSV.zst):
MSG
run_or_print "wget -c -O '$ROOT/lichess_puzzles/lichess_db_puzzle.csv.zst' 'https://database.lichess.org/lichess_db_puzzle.csv.zst'"

echo
cat <<'MSG'
# TCEC full PGNs:
#   Visit https://github.com/TCEC-Chess/tcecgames/releases
#   Download TCEC-events-full.7z or selected full-format event archives into:
MSG
echo "#   $ROOT/tcec/"

echo
cat <<'MSG'
# Fishtest PGNs:
#   Visit https://huggingface.co/datasets/official-stockfish/fishtest_pgns
#   Use huggingface-cli / hf_transfer to download selected shards into:
MSG
echo "#   $ROOT/fishtest/"

echo
cat <<'MSG'
# ChessBench / Searchless Chess action-value data:
#   Start with a small processed/exported JSONL sample if possible.
#   Put JSONL/JSONL.zst exports into:
MSG
echo "#   $ROOT/chessbench/"

echo
cat <<'MSG'
# lc0 chunks:
#   See https://storage.lczero.org/files/training_data/
#   Download only selected chunks initially into:
MSG
echo "#   $ROOT/lc0/"
