#!/usr/bin/env bash
set -euo pipefail

usage(){ cat <<'USAGE'
Prepare or submit the true 100M history-8 supervised dataset build in AWS Batch.

Default is --dry-run: validate local 100M manifest/raw inputs and print the exact
submit command. Use --submit to actually register/submit the dataset job. Use
--upload-inputs and --compress-inputs when you are ready to upload the 100M raw
input set (large: local uncompressed inputs are about tens of GiB).

Usage:
  AWS_PROFILE=tiny-leela AWS_DEFAULT_REGION=us-west-2 \
  cloud/aws/submit_100m_h8_dataset_job.sh --dry-run

  AWS_PROFILE=tiny-leela AWS_DEFAULT_REGION=us-west-2 \
  cloud/aws/submit_100m_h8_dataset_job.sh --submit --compress-inputs --upload-inputs --parallel-uploads 4

Options:
  --bucket-uri S3          default s3://tiny-leela-distributed-ddbb/h8_dataset_100m
  --job-queue NAME         default tiny-leela-cache-queue
  --job-definition NAME    default tiny-leela-cache-h8-dataset-builder-100m
  --base-dataset-dir PATH  default data/datasets/supervised_100m_elite_tcec_v1
  --dataset-name NAME      default supervised_100m_elite_tcec_h8_v1
  --compressed-dir PATH    default artifacts/cloud_h8_dataset_100m/raw_zst
  --parallel-uploads N     default 4 when uploading
  --vcpus N                default 16
  --memory MIB             default 65536
  --dry-run                default
  --submit                 submit job
  --compress-inputs        zstd raw JSONL before upload
  --upload-inputs          upload raw/compressed inputs to S3 first
USAGE
}

MODE="dry-run"
BUCKET_URI="s3://tiny-leela-distributed-ddbb/h8_dataset_100m"
JOB_QUEUE="tiny-leela-cache-queue"
JOB_DEFINITION="tiny-leela-cache-h8-dataset-builder-100m"
BASE_DATASET_DIR="data/datasets/supervised_100m_elite_tcec_v1"
DATASET_NAME="supervised_100m_elite_tcec_h8_v1"
COMPRESSED_DIR="artifacts/cloud_h8_dataset_100m/raw_zst"
PARALLEL_UPLOADS=4
VCPUS=16
MEMORY=65536
COMPRESS_INPUTS=0
UPLOAD_INPUTS=0
REGION="${AWS_REGION:-${AWS_DEFAULT_REGION:-us-west-2}}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bucket-uri) BUCKET_URI="$2"; shift 2;;
    --job-queue) JOB_QUEUE="$2"; shift 2;;
    --job-definition) JOB_DEFINITION="$2"; shift 2;;
    --base-dataset-dir) BASE_DATASET_DIR="$2"; shift 2;;
    --dataset-name) DATASET_NAME="$2"; shift 2;;
    --compressed-dir) COMPRESSED_DIR="$2"; shift 2;;
    --parallel-uploads) PARALLEL_UPLOADS="$2"; shift 2;;
    --vcpus) VCPUS="$2"; shift 2;;
    --memory) MEMORY="$2"; shift 2;;
    --region) REGION="$2"; shift 2;;
    --compress-inputs) COMPRESS_INPUTS=1; shift;;
    --upload-inputs) UPLOAD_INPUTS=1; shift;;
    --dry-run) MODE="dry-run"; shift;;
    --submit) MODE="submit"; shift;;
    -h|--help) usage; exit 0;;
    *) echo "unknown arg: $1" >&2; usage >&2; exit 2;;
  esac
done

[[ -s "$BASE_DATASET_DIR/manifest.json" ]] || { echo "missing $BASE_DATASET_DIR/manifest.json" >&2; exit 2; }

readarray -t INFO < <(python3 - "$BASE_DATASET_DIR" <<'PY'
import json, sys
from pathlib import Path
root=Path(sys.argv[1])
m=json.loads((root/'manifest.json').read_text())
inputs=m.get('reproducibility',{}).get('inputs') or []
missing=[p for p in inputs if not Path(p).exists()]
size=sum(Path(p).stat().st_size for p in inputs if Path(p).exists())
print(m.get('name'))
print(m.get('history_plies'))
print(m.get('total_train_rows'))
print(m.get('total_dev_rows'))
print(len(inputs))
print(size)
if missing:
    raise SystemExit('missing raw inputs: '+', '.join(missing[:5]))
PY
)

cat <<EOF
100M H8 DATASET PLAN
  base dataset: $BASE_DATASET_DIR
  base name/history: ${INFO[0]} / h${INFO[1]}
  raw inputs: ${INFO[4]} files, ${INFO[5]} bytes local
  output dataset: $BUCKET_URI/datasets/$DATASET_NAME
  rows: train=100000000 dev=1000000 rows_per_shard=1000000
  mode: $MODE

PRE-SUBMIT GUARDRAIL
  Before submit, require a passing capacity preflight:
    ./scripts/tlops cloud preflight-h8-dataset \
      --base-dataset-dir $BASE_DATASET_DIR \
      --max-rows 100000000 \
      --dev-rows 1000000 \
      --compress-inputs \
      --estimate-capacity
EOF

cmd=(cloud/aws/submit_h8_dataset_job.sh
  --bucket-uri "$BUCKET_URI"
  --job-queue "$JOB_QUEUE"
  --base-dataset-dir "$BASE_DATASET_DIR"
  --dataset-name "$DATASET_NAME"
  --compressed-dir "$COMPRESSED_DIR"
  --parallel-uploads "$PARALLEL_UPLOADS"
  --job-definition "$JOB_DEFINITION"
  --job-name "h8-dataset-100m"
  --vcpus "$VCPUS"
  --memory "$MEMORY"
  --region "$REGION"
  --max-rows 100000000
  --dev-rows 1000000
  --rows-per-shard 1000000
  --max-rows-per-game 64
  --max-rows-per-opening 100000
  --max-rows-per-source 0
  --source-caps ""
  --skip-plies 10
  --seed 7)
if [[ "$COMPRESS_INPUTS" == "1" ]]; then cmd+=(--compress-inputs); fi
if [[ "$UPLOAD_INPUTS" == "1" ]]; then cmd+=(--upload-inputs); fi

printf 'COMMAND:'
printf ' %q' "${cmd[@]}"
printf '\n'

if [[ "$MODE" == "dry-run" ]]; then
  echo "dry-run: not submitting."
  exit 0
fi

"${cmd[@]}"
