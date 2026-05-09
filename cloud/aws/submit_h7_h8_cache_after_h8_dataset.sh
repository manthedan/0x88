#!/usr/bin/env bash
set -euo pipefail

usage(){ cat <<'USAGE'
Gate and submit SquareFormer h7/h8 AWS Batch cache jobs after the cloud h8 dataset exists.

Default mode is a read-only dry-run: it checks/downloads the S3 manifest, validates it,
and prints the jobs that would be submitted. Use --prepare to upload shard-list files
without submitting jobs, or --submit to upload shard lists and submit Batch jobs.

Defaults match the first tiny-leela AWS deployment:
  dataset:        s3://tiny-leela-distributed-ddbb/h8_dataset_10m/datasets/supervised_10m_elite_tcec_h8_v1
  output caches:  s3://tiny-leela-distributed-ddbb/h8_dataset_10m/caches/supervised_10m_elite_tcec_h8_v1/cache_squareformer_h{7,8}
  job queue:      tiny-leela-cache-queue
  job definition: tiny-leela-cache-squareformer-cache

Usage:
  AWS_PROFILE=tiny-leela AWS_DEFAULT_REGION=us-west-2 \
  cloud/aws/submit_h7_h8_cache_after_h8_dataset.sh --dry-run

  AWS_PROFILE=tiny-leela AWS_DEFAULT_REGION=us-west-2 \
  cloud/aws/submit_h7_h8_cache_after_h8_dataset.sh --submit

Options:
  --dataset-s3-prefix S3   Full S3 prefix containing manifest.json, train/, dev/
  --bucket-uri S3          Base S3 prefix for jobs/caches, default s3://tiny-leela-distributed-ddbb/h8_dataset_10m
  --histories CSV          Histories to submit, default 7,8
  --job-queue NAME         Batch job queue, default tiny-leela-cache-queue
  --job-definition NAME    Batch job definition, default tiny-leela-cache-squareformer-cache
  --region REGION          AWS region, default env AWS_REGION/AWS_DEFAULT_REGION/us-west-2
  --expect-train N         Expected total_train_rows, default 10000000
  --expect-dev N           Expected total_dev_rows, default 500000
  --allow-partial          Warn instead of fail on expected row-count mismatch
  --dry-run                Check and print only, no writes/submits. Default.
  --prepare                Upload shard-list files only, no Batch submit.
  --submit                 Upload shard-list files and submit Batch jobs.
USAGE
}

REGION="${AWS_REGION:-${AWS_DEFAULT_REGION:-us-west-2}}"
BUCKET_URI="s3://tiny-leela-distributed-ddbb/h8_dataset_10m"
DATASET_NAME="supervised_10m_elite_tcec_h8_v1"
DATASET_S3_PREFIX=""
HISTORIES="7,8"
JOB_QUEUE="tiny-leela-cache-queue"
JOB_DEFINITION="tiny-leela-cache-squareformer-cache"
EXPECT_TRAIN="10000000"
EXPECT_DEV="500000"
ALLOW_PARTIAL=0
MODE="dry-run"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-s3-prefix) DATASET_S3_PREFIX="$2"; shift 2;;
    --bucket-uri) BUCKET_URI="$2"; shift 2;;
    --histories) HISTORIES="$2"; shift 2;;
    --job-queue) JOB_QUEUE="$2"; shift 2;;
    --job-definition) JOB_DEFINITION="$2"; shift 2;;
    --region) REGION="$2"; shift 2;;
    --expect-train) EXPECT_TRAIN="$2"; shift 2;;
    --expect-dev) EXPECT_DEV="$2"; shift 2;;
    --allow-partial) ALLOW_PARTIAL=1; shift;;
    --dry-run) MODE="dry-run"; shift;;
    --prepare) MODE="prepare"; shift;;
    --submit) MODE="submit"; shift;;
    -h|--help) usage; exit 0;;
    *) echo "unknown arg: $1" >&2; usage >&2; exit 2;;
  esac
done

command -v aws >/dev/null || { echo "aws CLI not found" >&2; exit 2; }
[[ -n "$DATASET_S3_PREFIX" ]] || DATASET_S3_PREFIX="$BUCKET_URI/datasets/$DATASET_NAME"
DATASET_NAME="$(basename "$DATASET_S3_PREFIX")"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

MANIFEST="$TMP/manifest.json"
echo "CHECK manifest: $DATASET_S3_PREFIX/manifest.json"
aws s3 cp "$DATASET_S3_PREFIX/manifest.json" "$MANIFEST" --region "$REGION" >/dev/null

readarray -t META < <(python3 - "$MANIFEST" "$EXPECT_TRAIN" "$EXPECT_DEV" "$ALLOW_PARTIAL" <<'PY'
import json, sys
p, expect_train, expect_dev, allow_partial = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), bool(int(sys.argv[4]))
m = json.load(open(p))
name = m.get('name') or 'unknown'
h = int(m.get('history_plies') or 0)
train = int(m.get('total_train_rows') or 0)
dev = int(m.get('total_dev_rows') or 0)
train_shards = len(m.get('train_shards') or [])
dev_path = m.get('dev')
print(name)
print(h)
print(train)
print(dev)
print(train_shards)
print(dev_path or '')
errors=[]
if h < 8: errors.append(f'history_plies={h}, expected at least 8')
if train != expect_train: errors.append(f'total_train_rows={train}, expected {expect_train}')
if dev != expect_dev: errors.append(f'total_dev_rows={dev}, expected {expect_dev}')
if not train_shards: errors.append('no train_shards in manifest')
if not dev_path: errors.append('no dev shard in manifest')
if errors:
    msg='; '.join(errors)
    if allow_partial:
        print('WARN '+msg, file=sys.stderr)
    else:
        raise SystemExit(msg)
PY
)
MANIFEST_NAME="${META[0]}"
DATASET_HISTORY="${META[1]}"
TOTAL_TRAIN="${META[2]}"
TOTAL_DEV="${META[3]}"
TRAIN_SHARDS="${META[4]}"
DEV_PATH="${META[5]}"

cat <<EOF
DATASET OK
  name: $MANIFEST_NAME
  s3: $DATASET_S3_PREFIX
  history_plies: $DATASET_HISTORY
  train rows/shards: $TOTAL_TRAIN / $TRAIN_SHARDS
  dev rows/path: $TOTAL_DEV / $DEV_PATH
  mode: $MODE
EOF

IFS=',' read -r -a HISTORY_LIST <<< "$HISTORIES"
for HISTORY in "${HISTORY_LIST[@]}"; do
  HISTORY="$(echo "$HISTORY" | tr -d '[:space:]')"
  [[ "$HISTORY" =~ ^[0-9]+$ ]] || { echo "bad history: $HISTORY" >&2; exit 2; }
  if (( HISTORY > DATASET_HISTORY )); then
    echo "history $HISTORY cannot be derived from dataset history_plies=$DATASET_HISTORY" >&2
    exit 2
  fi

  NAME="sqf-h${HISTORY}-${MANIFEST_NAME}"
  CACHE_NAME="cache_squareformer_h${HISTORY}"
  JOB_S3_PREFIX="$BUCKET_URI/jobs/$NAME"
  OUTPUT_S3_PREFIX="$BUCKET_URI/caches/$MANIFEST_NAME/$CACHE_NAME"
  LIST_DIR="$TMP/lists_h$HISTORY"

  python3 cloud/aws/make_squareformer_shard_lists.py \
    --dataset-manifest "$MANIFEST" \
    --dataset-s3-prefix "$DATASET_S3_PREFIX" \
    --out-dir "$LIST_DIR" >/dev/null
  TRAIN_N=$(wc -l < "$LIST_DIR/train_shards.s3.txt" | tr -d ' ')

  cat > "$TMP/train_h${HISTORY}_overrides.json" <<JSON
{
  "environment": [
    {"name":"SHARD_LIST_S3","value":"$JOB_S3_PREFIX/train_shards.s3.txt"},
    {"name":"OUTPUT_S3_PREFIX","value":"$OUTPUT_S3_PREFIX"},
    {"name":"HISTORY_PLIES","value":"$HISTORY"},
    {"name":"SPLIT","value":"train"}
  ]
}
JSON
  cat > "$TMP/dev_h${HISTORY}_overrides.json" <<JSON
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

  echo
  echo "PLAN h$HISTORY"
  echo "  job name: $NAME"
  echo "  train array size: $TRAIN_N"
  echo "  job lists: $JOB_S3_PREFIX"
  echo "  output: $OUTPUT_S3_PREFIX"

  if [[ "$MODE" == "dry-run" ]]; then
    echo "  dry-run: not uploading shard lists or submitting jobs"
    continue
  fi

  echo "  upload shard lists"
  aws s3 cp "$LIST_DIR/train_shards.s3.txt" "$JOB_S3_PREFIX/train_shards.s3.txt" --region "$REGION"
  aws s3 cp "$LIST_DIR/dev_shards.s3.txt" "$JOB_S3_PREFIX/dev_shards.s3.txt" --region "$REGION"
  aws s3 cp "$LIST_DIR/shard_lists.summary.json" "$JOB_S3_PREFIX/shard_lists.summary.json" --region "$REGION"

  if [[ "$MODE" == "prepare" ]]; then
    echo "  prepare: shard lists uploaded; not submitting jobs"
    continue
  fi

  echo "  submit train array"
  aws batch submit-job \
    --region "$REGION" \
    --job-name "$NAME-train" \
    --job-queue "$JOB_QUEUE" \
    --job-definition "$JOB_DEFINITION" \
    --array-properties "size=$TRAIN_N" \
    --container-overrides "file://$TMP/train_h${HISTORY}_overrides.json"

  echo "  submit dev job"
  aws batch submit-job \
    --region "$REGION" \
    --job-name "$NAME-dev" \
    --job-queue "$JOB_QUEUE" \
    --job-definition "$JOB_DEFINITION" \
    --container-overrides "file://$TMP/dev_h${HISTORY}_overrides.json"
done
