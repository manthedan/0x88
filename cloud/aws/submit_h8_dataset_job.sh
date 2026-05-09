#!/usr/bin/env bash
set -euo pipefail

usage(){ cat <<'USAGE'
Build the 10M h8 supervised dataset in AWS Batch.

Required:
  --bucket-uri s3://bucket/prefix       e.g. s3://tiny-leela-distributed-ddbb/h8_dataset_10m
  --job-queue NAME_OR_ARN              e.g. tiny-leela-cache-queue

Optional:
  --base-dataset-dir PATH              default data/datasets/supervised_10m_elite_tcec_v1
  --dataset-name NAME                  default supervised_10m_elite_tcec_h8_v1
  --region us-west-2
  --upload-inputs                      upload raw reproducibility inputs first
  --compress-inputs                    zstd-compress local raw JSONL inputs before upload
  --compressed-dir PATH                default artifacts/cloud_h8_dataset_10m/raw_zst
  --parallel-uploads N                 parallel aws s3 cp processes; default 1
  --job-definition NAME                default tiny-leela-cache-h8-dataset-builder
  --image URI                          default ECR tiny-leela-cache-worker:dataset-latest
  --vcpus N                            default 16
  --memory MIB                         default 65536
  --job-name NAME                      default h8-dataset-10m
  --max-rows N                         default 10000000
  --dev-rows N                         default 500000
  --rows-per-shard N                   default 250000
  --max-rows-per-game N                default 64
  --max-rows-per-opening N             default 10000
  --max-rows-per-source N              default 800000; use 0 to disable
  --source-caps STRING                 default 10M-specific caps; use '' to disable
  --skip-plies N                       default 10
  --seed N                             default 7

Example:
  cloud/aws/submit_h8_dataset_job.sh \
    --bucket-uri s3://tiny-leela-distributed-ddbb/h8_dataset_10m \
    --job-queue tiny-leela-cache-queue \
    --compress-inputs \
    --upload-inputs \
    --parallel-uploads 4
USAGE
}

BUCKET_URI=""; JOB_QUEUE=""; BASE_DATASET_DIR="data/datasets/supervised_10m_elite_tcec_v1"; DATASET_NAME="supervised_10m_elite_tcec_h8_v1"; REGION="${AWS_REGION:-${AWS_DEFAULT_REGION:-us-west-2}}"; UPLOAD_INPUTS=0; COMPRESS_INPUTS=0; COMPRESSED_DIR="artifacts/cloud_h8_dataset_10m/raw_zst"; PARALLEL_UPLOADS=1; JOB_DEFINITION="tiny-leela-cache-h8-dataset-builder"; IMAGE=""; VCPUS=16; MEMORY=65536; JOB_NAME="h8-dataset-10m"; MAX_ROWS=10000000; DEV_ROWS=500000; ROWS_PER_SHARD=250000; MAX_ROWS_PER_GAME=64; MAX_ROWS_PER_OPENING=10000; MAX_ROWS_PER_SOURCE=800000; SOURCE_CAPS="lichess_training_3m_2200elo_2026-01=2000000 tcec_training_500k=500000"; SKIP_PLIES=10; SEED=7
while [[ $# -gt 0 ]]; do
  case "$1" in
    --bucket-uri) BUCKET_URI="$2"; shift 2;;
    --job-queue) JOB_QUEUE="$2"; shift 2;;
    --base-dataset-dir) BASE_DATASET_DIR="$2"; shift 2;;
    --dataset-name) DATASET_NAME="$2"; shift 2;;
    --region) REGION="$2"; shift 2;;
    --upload-inputs) UPLOAD_INPUTS=1; shift;;
    --compress-inputs) COMPRESS_INPUTS=1; shift;;
    --compressed-dir) COMPRESSED_DIR="$2"; shift 2;;
    --parallel-uploads) PARALLEL_UPLOADS="$2"; shift 2;;
    --job-definition) JOB_DEFINITION="$2"; shift 2;;
    --image) IMAGE="$2"; shift 2;;
    --vcpus) VCPUS="$2"; shift 2;;
    --memory) MEMORY="$2"; shift 2;;
    --job-name) JOB_NAME="$2"; shift 2;;
    --max-rows) MAX_ROWS="$2"; shift 2;;
    --dev-rows) DEV_ROWS="$2"; shift 2;;
    --rows-per-shard) ROWS_PER_SHARD="$2"; shift 2;;
    --max-rows-per-game) MAX_ROWS_PER_GAME="$2"; shift 2;;
    --max-rows-per-opening) MAX_ROWS_PER_OPENING="$2"; shift 2;;
    --max-rows-per-source) MAX_ROWS_PER_SOURCE="$2"; shift 2;;
    --source-caps) SOURCE_CAPS="$2"; shift 2;;
    --skip-plies) SKIP_PLIES="$2"; shift 2;;
    --seed) SEED="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "unknown arg: $1" >&2; usage; exit 2;;
  esac
done
[[ -n "$BUCKET_URI" && -n "$JOB_QUEUE" ]] || { usage >&2; exit 2; }
[[ -s "$BASE_DATASET_DIR/manifest.json" ]] || { echo "missing $BASE_DATASET_DIR/manifest.json" >&2; exit 2; }
command -v aws >/dev/null || { echo "aws CLI not found" >&2; exit 2; }

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
IMAGE="${IMAGE:-${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/tiny-leela-cache-worker:dataset-latest}"
JOB_ROLE_ARN=$(aws batch describe-job-definitions --region "$REGION" --job-definition-name tiny-leela-cache-squareformer-cache --status ACTIVE --query 'jobDefinitions[0].containerProperties.jobRoleArn' --output text)
[[ "$JOB_ROLE_ARN" != "None" && -n "$JOB_ROLE_ARN" ]] || { echo "could not discover Batch job role" >&2; exit 1; }

mapfile -t INPUTS < <(python3 - "$BASE_DATASET_DIR" <<'PY'
import json,sys
from pathlib import Path
m=json.loads((Path(sys.argv[1])/'manifest.json').read_text())
for p in m['reproducibility']['inputs']:
    print(p)
PY
)

RAW_PREFIX="$BUCKET_URI/raw_zst"
DATASET_PREFIX="$BUCKET_URI/datasets/$DATASET_NAME"
declare -a INPUT_S3=()
declare -a UPLOAD_SRC=()
declare -a UPLOAD_DST=()
for p0 in "${INPUTS[@]}"; do
  [[ -s "$p0" ]] || { echo "missing raw input: $p0" >&2; exit 2; }
  p="$p0"
  if [[ "$COMPRESS_INPUTS" == "1" && "$p0" != *.zst ]]; then
    mkdir -p "$COMPRESSED_DIR"
    p="$COMPRESSED_DIR/$(basename "$p0").zst"
    if [[ ! -s "$p" || "$p" -ot "$p0" ]]; then
      echo "COMPRESS $p0 -> $p"
      zstd -T0 -3 -f "$p0" -o "$p"
    else
      echo "REUSE compressed $p"
    fi
  fi
  uri="$RAW_PREFIX/$(basename "$p")"
  INPUT_S3+=("$uri")
  if [[ "$UPLOAD_INPUTS" == "1" ]]; then
    UPLOAD_SRC+=("$p")
    UPLOAD_DST+=("$uri")
  fi
done

if [[ "$UPLOAD_INPUTS" == "1" ]]; then
  [[ "$PARALLEL_UPLOADS" =~ ^[0-9]+$ && "$PARALLEL_UPLOADS" -ge 1 ]] || { echo "--parallel-uploads must be >= 1" >&2; exit 2; }
  echo "UPLOAD ${#UPLOAD_SRC[@]} files with parallelism=$PARALLEL_UPLOADS"
  active=0; failed=0
  for i in "${!UPLOAD_SRC[@]}"; do
    (
      echo "UPLOAD ${UPLOAD_SRC[$i]} -> ${UPLOAD_DST[$i]}"
      aws s3 cp "${UPLOAD_SRC[$i]}" "${UPLOAD_DST[$i]}" --region "$REGION" --only-show-errors
    ) &
    active=$((active+1))
    if [[ "$active" -ge "$PARALLEL_UPLOADS" ]]; then
      if ! wait -n; then failed=1; fi
      active=$((active-1))
    fi
  done
  while [[ "$active" -gt 0 ]]; do
    if ! wait -n; then failed=1; fi
    active=$((active-1))
  done
  [[ "$failed" == "0" ]] || { echo "one or more uploads failed" >&2; exit 1; }
fi
INPUT_JOINED="${INPUT_S3[*]}"

TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
cat > "$TMP/container.json" <<JSON
{
  "image": "$IMAGE",
  "vcpus": $VCPUS,
  "memory": $MEMORY,
  "jobRoleArn": "$JOB_ROLE_ARN",
  "environment": [
    {"name":"AWS_DEFAULT_REGION","value":"$REGION"},
    {"name":"INPUT_S3_URIS","value":"$INPUT_JOINED"},
    {"name":"OUTPUT_S3_PREFIX","value":"$DATASET_PREFIX"},
    {"name":"DATASET_NAME","value":"$DATASET_NAME"},
    {"name":"HISTORY_PLIES","value":"8"},
    {"name":"MAX_ROWS","value":"$MAX_ROWS"},
    {"name":"DEV_ROWS","value":"$DEV_ROWS"},
    {"name":"ROWS_PER_SHARD","value":"$ROWS_PER_SHARD"},
    {"name":"MAX_ROWS_PER_GAME","value":"$MAX_ROWS_PER_GAME"},
    {"name":"MAX_ROWS_PER_OPENING","value":"$MAX_ROWS_PER_OPENING"},
    {"name":"MAX_ROWS_PER_SOURCE","value":"$MAX_ROWS_PER_SOURCE"},
    {"name":"SOURCE_CAPS","value":"$SOURCE_CAPS"},
    {"name":"SKIP_PLIES","value":"$SKIP_PLIES"},
    {"name":"SEED","value":"$SEED"}
  ],
  "logConfiguration": {"logDriver":"awslogs"}
}
JSON

echo "REGISTER job definition $JOB_DEFINITION image=$IMAGE vcpus=$VCPUS memory=$MEMORY"
aws batch register-job-definition \
  --region "$REGION" \
  --job-definition-name "$JOB_DEFINITION" \
  --type container \
  --container-properties "file://$TMP/container.json" >/dev/null

echo "SUBMIT h8 dataset job output=$DATASET_PREFIX"
aws batch submit-job \
  --region "$REGION" \
  --job-name "$JOB_NAME" \
  --job-queue "$JOB_QUEUE" \
  --job-definition "$JOB_DEFINITION"

echo "DATASET_S3_PREFIX=$DATASET_PREFIX"
