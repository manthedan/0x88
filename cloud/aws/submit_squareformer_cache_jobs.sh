#!/usr/bin/env bash
set -euo pipefail

usage(){ cat <<'USAGE'
Submit AWS Batch array jobs to build SquareFormer token caches shard-by-shard.

Required env/args:
  --dataset-dir PATH          local dataset dir with manifest.json, train/, dev/
  --bucket-uri s3://bucket/prefix
  --history N                 history plies to encode, e.g. 7 or 8
  --job-queue NAME_OR_ARN
  --job-definition NAME_OR_ARN

Optional:
  --upload-dataset            aws s3 sync dataset manifest/train/dev first
  --region us-west-2          default env AWS_REGION/AWS_DEFAULT_REGION/us-west-2
  --name NAME                 job name prefix
  --cache-name NAME           default cache_squareformer_hN

Example:
  cloud/aws/submit_squareformer_cache_jobs.sh \
    --dataset-dir data/datasets/supervised_10m_elite_tcec_h8_v1 \
    --bucket-uri s3://tiny-leela-distributed-ddbb/h7h8_10m \
    --history 7 \
    --job-queue tiny-leela-cache-job-queue \
    --job-definition tiny-leela-cache-squareformer \
    --upload-dataset
USAGE
}

DATASET_DIR=""; BUCKET_URI=""; HISTORY=""; JOB_QUEUE=""; JOB_DEFINITION=""; REGION="${AWS_REGION:-${AWS_DEFAULT_REGION:-us-west-2}}"; UPLOAD_DATASET=0; NAME=""; CACHE_NAME=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-dir) DATASET_DIR="$2"; shift 2;;
    --bucket-uri) BUCKET_URI="$2"; shift 2;;
    --history) HISTORY="$2"; shift 2;;
    --job-queue) JOB_QUEUE="$2"; shift 2;;
    --job-definition) JOB_DEFINITION="$2"; shift 2;;
    --region) REGION="$2"; shift 2;;
    --upload-dataset) UPLOAD_DATASET=1; shift;;
    --name) NAME="$2"; shift 2;;
    --cache-name) CACHE_NAME="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "unknown arg: $1" >&2; usage; exit 2;;
  esac
done
[[ -n "$DATASET_DIR" && -n "$BUCKET_URI" && -n "$HISTORY" && -n "$JOB_QUEUE" && -n "$JOB_DEFINITION" ]] || { usage >&2; exit 2; }
[[ -s "$DATASET_DIR/manifest.json" ]] || { echo "missing $DATASET_DIR/manifest.json" >&2; exit 2; }
command -v aws >/dev/null || { echo "aws CLI not found" >&2; exit 2; }

DATASET_NAME=$(python3 - <<PY
import json
print(json.load(open('$DATASET_DIR/manifest.json')).get('name') or '$DATASET_DIR'.rstrip('/').split('/')[-1])
PY
)
CACHE_NAME="${CACHE_NAME:-cache_squareformer_h${HISTORY}}"
NAME="${NAME:-sqf-h${HISTORY}-${DATASET_NAME}}"
DATASET_S3_PREFIX="$BUCKET_URI/datasets/$DATASET_NAME"
JOB_S3_PREFIX="$BUCKET_URI/jobs/$NAME"
OUTPUT_S3_PREFIX="$BUCKET_URI/caches/$DATASET_NAME/$CACHE_NAME"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

if [[ "$UPLOAD_DATASET" == "1" ]]; then
  echo "SYNC dataset -> $DATASET_S3_PREFIX"
  aws s3 sync "$DATASET_DIR" "$DATASET_S3_PREFIX" \
    --exclude '*' \
    --include 'manifest.json' \
    --include 'train/*.jsonl' \
    --include 'train/*.jsonl.zst' \
    --include 'dev/*.jsonl' \
    --include 'dev/*.jsonl.zst' \
    --region "$REGION"
fi

python3 cloud/aws/make_squareformer_shard_lists.py \
  --dataset-manifest "$DATASET_DIR/manifest.json" \
  --dataset-s3-prefix "$DATASET_S3_PREFIX" \
  --out-dir "$TMP/lists"
aws s3 cp "$TMP/lists/train_shards.s3.txt" "$JOB_S3_PREFIX/train_shards.s3.txt" --region "$REGION"
aws s3 cp "$TMP/lists/dev_shards.s3.txt" "$JOB_S3_PREFIX/dev_shards.s3.txt" --region "$REGION"
aws s3 cp "$TMP/lists/shard_lists.summary.json" "$JOB_S3_PREFIX/shard_lists.summary.json" --region "$REGION"

TRAIN_N=$(wc -l < "$TMP/lists/train_shards.s3.txt" | tr -d ' ')
cat > "$TMP/train_overrides.json" <<JSON
{
  "environment": [
    {"name":"SHARD_LIST_S3","value":"$JOB_S3_PREFIX/train_shards.s3.txt"},
    {"name":"OUTPUT_S3_PREFIX","value":"$OUTPUT_S3_PREFIX"},
    {"name":"HISTORY_PLIES","value":"$HISTORY"},
    {"name":"SPLIT","value":"train"}
  ]
}
JSON
cat > "$TMP/dev_overrides.json" <<JSON
{
  "environment": [
    {"name":"SHARD_LIST_S3","value":"$JOB_S3_PREFIX/dev_shards.s3.txt"},
    {"name":"OUTPUT_S3_PREFIX","value":"$OUTPUT_S3_PREFIX"},
    {"name":"HISTORY_PLIES","value":"$HISTORY"},
    {"name":"SPLIT","value":"dev"},
    {"name":"SHARD_INDEX","value":"0"}
  ]
}
JSON

echo "SUBMIT train array size=$TRAIN_N output=$OUTPUT_S3_PREFIX"
aws batch submit-job \
  --region "$REGION" \
  --job-name "$NAME-train" \
  --job-queue "$JOB_QUEUE" \
  --job-definition "$JOB_DEFINITION" \
  --array-properties "size=$TRAIN_N" \
  --container-overrides "file://$TMP/train_overrides.json"

echo "SUBMIT dev job"
aws batch submit-job \
  --region "$REGION" \
  --job-name "$NAME-dev" \
  --job-queue "$JOB_QUEUE" \
  --job-definition "$JOB_DEFINITION" \
  --container-overrides "file://$TMP/dev_overrides.json"

echo "OUTPUT_S3_PREFIX=$OUTPUT_S3_PREFIX"
echo "JOB_S3_PREFIX=$JOB_S3_PREFIX"
