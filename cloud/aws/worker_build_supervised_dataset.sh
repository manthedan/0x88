#!/usr/bin/env bash
set -euo pipefail

: "${INPUT_S3_URIS:?set INPUT_S3_URIS to space-separated s3://... raw JSONL inputs}"
: "${OUTPUT_S3_PREFIX:?set OUTPUT_S3_PREFIX to s3://bucket/prefix for dataset output}"
DATASET_NAME="${DATASET_NAME:-supervised_10m_elite_tcec_h8_v1}"
MAX_ROWS="${MAX_ROWS:-10000000}"
DEV_ROWS="${DEV_ROWS:-500000}"
ROWS_PER_SHARD="${ROWS_PER_SHARD:-250000}"
HISTORY_PLIES="${HISTORY_PLIES:-8}"
SKIP_PLIES="${SKIP_PLIES:-10}"
SEED="${SEED:-7}"
MAX_ROWS_PER_GAME="${MAX_ROWS_PER_GAME:-64}"
MAX_ROWS_PER_OPENING="${MAX_ROWS_PER_OPENING:-10000}"
MAX_ROWS_PER_SOURCE="${MAX_ROWS_PER_SOURCE:-800000}"
SOURCE_CAPS="${SOURCE_CAPS:-lichess_training_3m_2200elo_2026-01=2000000 tcec_training_500k=500000}"
WORK_ROOT="${WORK_ROOT:-/work/run}"
RAW_DIR="$WORK_ROOT/raw"
OUT_DIR="$WORK_ROOT/out/$DATASET_NAME"
mkdir -p "$RAW_DIR" "$OUT_DIR"

echo "START dataset worker $(date -Is)"
echo "DATASET_NAME=$DATASET_NAME HISTORY_PLIES=$HISTORY_PLIES MAX_ROWS=$MAX_ROWS DEV_ROWS=$DEV_ROWS OUTPUT=$OUTPUT_S3_PREFIX"
aws sts get-caller-identity || true

declare -a INPUTS=()
for uri in $INPUT_S3_URIS; do
  base="$(basename "$uri")"
  dst="$RAW_DIR/$base"
  echo "DOWNLOAD $uri -> $dst"
  aws s3 cp "$uri" "$dst"
  INPUTS+=("$dst")
done

declare -a CAP_ARGS=()
for cap in $SOURCE_CAPS; do
  CAP_ARGS+=(--source-cap "$cap")
done

echo "RUN build_supervised_dataset_shards inputs=${#INPUTS[@]}"
/usr/local/bin/python /work/scripts/build_supervised_dataset_shards.py \
  --input "${INPUTS[@]}" \
  --out-dir "$OUT_DIR" \
  --name "$DATASET_NAME" \
  --max-rows "$MAX_ROWS" \
  --dev-rows "$DEV_ROWS" \
  --rows-per-shard "$ROWS_PER_SHARD" \
  --max-rows-per-game "$MAX_ROWS_PER_GAME" \
  --max-rows-per-opening "$MAX_ROWS_PER_OPENING" \
  --max-rows-per-source "$MAX_ROWS_PER_SOURCE" \
  "${CAP_ARGS[@]}" \
  --skip-plies "$SKIP_PLIES" \
  --history-plies "$HISTORY_PLIES" \
  --seed "$SEED" \
  --zst

/usr/local/bin/python - <<'PY' "$OUT_DIR" "$HISTORY_PLIES" "$MAX_ROWS" "$DEV_ROWS"
import json, sys
from pathlib import Path
root=Path(sys.argv[1])
expect_history=int(sys.argv[2])
expect_train=int(sys.argv[3])
expect_dev=int(sys.argv[4])
m=json.loads((root/'manifest.json').read_text())
assert m['history_plies'] == expect_history, m['history_plies']
assert m['total_train_rows'] == expect_train, m['total_train_rows']
assert m['total_dev_rows'] == expect_dev, m['total_dev_rows']
missing=[p for p in m['train_shards'] if not (root/p).exists()]
assert not missing, missing[:3]
assert (root/m['dev']).exists(), m['dev']
print(json.dumps({'ok': True, 'history_plies': m['history_plies'], 'train_shards': len(m['train_shards']), 'train_rows': m['total_train_rows'], 'dev_rows': m['total_dev_rows']}))
PY

echo "UPLOAD dataset -> $OUTPUT_S3_PREFIX"
aws s3 sync "$OUT_DIR" "$OUTPUT_S3_PREFIX" \
  --exclude '*' \
  --include 'manifest.json' \
  --include 'reports/*' \
  --include 'train/*.jsonl.zst' \
  --include 'dev/*.jsonl.zst'

echo "DONE dataset worker $(date -Is)"
