#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  cat >&2 <<'USAGE'
Usage:
  AWS_PROFILE=tiny-leela AWS_DEFAULT_REGION=us-west-2 \
  cloud/aws/watch_100m_h8_dataset_job.sh JOB_ID
USAGE
  exit 2
fi

OUT="${OUT:-artifacts/cloud_h8_dataset_100m/live_status.json}" \
LOG_TAIL="${LOG_TAIL:-artifacts/cloud_h8_dataset_100m/live_cloudwatch_tail.log}" \
DATASET_PREFIX="${DATASET_PREFIX:-s3://tiny-leela-distributed-ddbb/h8_dataset_100m/datasets/supervised_100m_elite_tcec_h8_v1}" \
cloud/aws/watch_h8_dataset_job.sh "$1"
