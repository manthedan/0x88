#!/usr/bin/env bash
set -euo pipefail

exec cloud/aws/watch_squareformer_cache_jobs.sh \
  --histories "${HISTORIES:-7,8}" \
  --dataset-name "${DATASET_NAME:-supervised_100m_elite_tcec_h8_v1}" \
  --bucket-uri "${BUCKET_URI:-s3://tiny-leela-distributed-ddbb/h8_dataset_100m}" \
  --job-queue "${JOB_QUEUE:-tiny-leela-cache-queue}" \
  --train-shards "${TRAIN_SHARDS:-100}" \
  --expect-train "${EXPECT_TRAIN:-100000000}" \
  --expect-dev "${EXPECT_DEV:-1000000}" \
  --out-dir "${OUT_DIR:-artifacts/cloud_h8_dataset_100m/cache_jobs}" \
  "$@"
