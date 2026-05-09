#!/usr/bin/env bash
set -euo pipefail

stamp(){ date -Is; }
log(){ echo "$(stamp) $*"; }
fail(){ rc=$?; log "FAILED rc=$rc line=${BASH_LINENO[0]:-unknown}"; exit "$rc"; }
trap fail ERR

: "${SHARD_LIST_S3:?set SHARD_LIST_S3 to s3://.../train_shards.s3.txt or dev_shards.s3.txt}"
: "${OUTPUT_S3_PREFIX:?set OUTPUT_S3_PREFIX to s3://bucket/prefix/cache_squareformer_hN}"
HISTORY_PLIES="${HISTORY_PLIES:-7}"
SPLIT="${SPLIT:-train}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ -n "${SHARD_INDEX:-}" ]]; then
  IDX="$SHARD_INDEX"
elif [[ -n "${AWS_BATCH_JOB_ARRAY_INDEX:-}" ]]; then
  IDX="$AWS_BATCH_JOB_ARRAY_INDEX"
else
  IDX=0
fi
PADDED=$(printf '%04d' "$IDX")
WORK="${WORK_DIR:-/tmp/tiny_leela_cache_worker}"
INPUT_DIR="$WORK/input"
OUT_DIR="$WORK/out"
mkdir -p "$INPUT_DIR" "$OUT_DIR"

log "START squareformer cache worker split=$SPLIT index=$IDX padded=$PADDED history=$HISTORY_PLIES"
log "SHARD_LIST_S3=$SHARD_LIST_S3"
log "OUTPUT_S3_PREFIX=$OUTPUT_S3_PREFIX"

aws s3 cp "$SHARD_LIST_S3" "$WORK/shards.s3.txt"
INPUT_S3=$(awk -v n=$((IDX+1)) 'NR==n {print; exit}' "$WORK/shards.s3.txt")
if [[ -z "$INPUT_S3" ]]; then
  echo "No shard URI at 0-based index $IDX in $SHARD_LIST_S3" >&2
  exit 2
fi

INPUT_BASE=$(basename "$INPUT_S3")
log "DOWNLOAD input=$INPUT_S3"
aws s3 cp "$INPUT_S3" "$INPUT_DIR/$INPUT_BASE"

CACHE_DIR="$OUT_DIR/$SPLIT/shard_$PADDED"
mkdir -p "$CACHE_DIR"
log "BUILD cache dir=$CACHE_DIR"
"$PYTHON_BIN" training/build_squareformer_token_cache.py \
  --input "$INPUT_DIR/$INPUT_BASE" \
  --out "$CACHE_DIR" \
  --history-plies "$HISTORY_PLIES"

log "PACKAGE cache archive"
tar -C "$CACHE_DIR" -I 'zstd -T0 -19' -cf "$WORK/cache.tar.zst" .
sha256sum "$WORK/cache.tar.zst" > "$WORK/cache.tar.zst.sha256"

ROWS=$("$PYTHON_BIN" - <<PY
import json
print(json.load(open('$CACHE_DIR/meta.json'))['rows'])
PY
)
BYTES=$(stat -c%s "$WORK/cache.tar.zst")
cat > "$WORK/worker_manifest.json" <<JSON
{
  "schema": "tiny_leela.squareformer_cache_worker.v1",
  "created_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "split": "$SPLIT",
  "shard_index": $IDX,
  "shard_name": "shard_$PADDED",
  "history_plies": $HISTORY_PLIES,
  "input_s3": "$INPUT_S3",
  "output_s3_prefix": "$OUTPUT_S3_PREFIX/$SPLIT/shard_$PADDED",
  "rows": $ROWS,
  "archive_bytes": $BYTES,
  "archive_sha256": "$(awk '{print $1}' "$WORK/cache.tar.zst.sha256")"
}
JSON

DEST="$OUTPUT_S3_PREFIX/$SPLIT/shard_$PADDED"
log "UPLOAD dest=$DEST"
aws s3 cp "$CACHE_DIR/meta.json" "$DEST/meta.json"
aws s3 cp "$WORK/cache.tar.zst" "$DEST/cache.tar.zst"
aws s3 cp "$WORK/cache.tar.zst.sha256" "$DEST/cache.tar.zst.sha256"
aws s3 cp "$WORK/worker_manifest.json" "$DEST/worker_manifest.json"
log "DONE split=$SPLIT shard=shard_$PADDED rows=$ROWS archive_bytes=$BYTES"
