#!/usr/bin/env bash
set -euo pipefail

# Thin, safe wrapper around the generic h7/h8 cache submitter for the 100M h8 dataset.
# Default mode remains dry-run in submit_h7_h8_cache_after_h8_dataset.sh.

exec cloud/aws/submit_h7_h8_cache_after_h8_dataset.sh \
  --bucket-uri "${BUCKET_URI:-s3://tiny-leela-distributed-ddbb/h8_dataset_100m}" \
  --dataset-s3-prefix "${DATASET_S3_PREFIX:-s3://tiny-leela-distributed-ddbb/h8_dataset_100m/datasets/supervised_100m_elite_tcec_h8_v1}" \
  --histories "${HISTORIES:-7,8}" \
  --job-queue "${JOB_QUEUE:-tiny-leela-cache-queue}" \
  --job-definition "${JOB_DEFINITION:-tiny-leela-cache-squareformer-cache}" \
  --region "${AWS_REGION:-${AWS_DEFAULT_REGION:-us-west-2}}" \
  --expect-train "${EXPECT_TRAIN:-100000000}" \
  --expect-dev "${EXPECT_DEV:-1000000}" \
  "$@"
