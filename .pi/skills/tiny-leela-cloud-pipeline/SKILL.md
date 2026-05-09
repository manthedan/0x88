---
name: tiny-leela-cloud-pipeline
description: AWS/S3/Bash orchestration for Tiny Leela h8 supervised datasets and SquareFormer h7/h8 caches. Use when submitting, monitoring, validating, or finalizing 10M/100M h8 dataset and cache jobs.
---

# Tiny Leela Cloud Pipeline

Use this skill for the cloud dataset/cache path.

## Canonical pipeline

```text
raw 100M game rows
  -> supervised_100m_elite_tcec_h8_v1
  -> cache_squareformer_h7 + cache_squareformer_h8
  -> BT4/SquareFormer training
```

Critical rule: existing h2 datasets/caches are not valid substitutes for true h7/h8 BT4 training. Build one h8 dataset, then derive h7 and h8 caches from it.

## AWS defaults

```bash
export AWS_PROFILE=tiny-leela
export AWS_DEFAULT_REGION=us-west-2
```

Known resources:

```text
account: 129475212118
bucket: s3://tiny-leela-distributed-ddbb
job queue: tiny-leela-cache-queue
job definition: tiny-leela-cache-squareformer-cache
ECR repo: tiny-leela-cache-worker
budget guardrail: $100
```

Prefer `./scripts/tlops` wrappers for inspection:

```bash
./scripts/tlops cloud jobs --match 100m --statuses RUNNING RUNNABLE SUBMITTED STARTING
./scripts/tlops cloud describe JOB_ID
./scripts/tlops cloud progress JOB_ID --lines 10
```

## h8 dataset preflight and checks

Before submitting a 100M h8 dataset job, run the repo guardrail preflight. For a submit-ready 100M plan, capacity estimation is required unless the user explicitly accepts the risk:

```bash
./scripts/tlops cloud preflight-h8-dataset \
  --base-dataset-dir data/datasets/supervised_100m_elite_tcec_v1 \
  --max-rows 100000000 \
  --dev-rows 1000000 \
  --compress-inputs \
  --estimate-capacity
```

The preflight checks local raw inputs, compression plan, h8 target contract, and optionally streams the raw inputs through the same selection caps as the cloud worker. Do not submit if it reports capacity below the requested safety margin.

Before submitting caches, validate that the dataset manifest exists and is true h8:

```bash
./scripts/tlops cloud validate-h8-manifest \
  --s3-prefix s3://tiny-leela-distributed-ddbb/h8_dataset_100m/datasets/supervised_100m_elite_tcec_h8_v1 \
  --expect-history 8 \
  --expect-train 100000000 \
  --expect-dev 1000000 \
  --expect-train-shards 100
```

Expected for true 100M h8:

```text
history_plies: 8
total_train_rows: 100000000
total_dev_rows: 1000000
train shards: 100
```

## Cache fanout after dataset success

Only after the h8 dataset job succeeds and manifest validates:

```bash
./scripts/tlops cloud submit-cache-h7-h8 \
  --bucket-uri s3://tiny-leela-distributed-ddbb/h8_dataset_100m \
  --dataset-s3-prefix s3://tiny-leela-distributed-ddbb/h8_dataset_100m/datasets/supervised_100m_elite_tcec_h8_v1 \
  --expect-train 100000000 \
  --expect-dev 1000000 \
  --submit
```

Then monitor/finalize with the existing cloud watcher if needed:

```bash
AWS_PROFILE=tiny-leela AWS_DEFAULT_REGION=us-west-2 \
cloud/aws/watch_100m_squareformer_cache_jobs.sh --wait --finalize-on-success
```

## Worker/image notes

- Dataset worker image tag: `dataset-latest`.
- ECR `DescribeImages` may be denied for this operator; do not treat that as proof the image is missing.
- The 10M h8 dataset/cache path is the successful pilot; prefer generalizing from it rather than inventing a new path.

## Status, repair, and failure handling

Use the DAG status helper before deciding what action is allowed:

```bash
./scripts/tlops cloud status-h8-100m
```

If a dataset job fails from row shortfall, generate a repair plan before changing inputs or filters:

```bash
./scripts/tlops cloud repair-plan-h8-dataset --actual-train ACTUAL_ROWS
```

Failure taxonomy and policy:

- `RUNNABLE`: inspect compute environment capacity and queue order before resubmitting.
- Input compression issue: use `.jsonl.zst` via `--compress-inputs`; do not upload huge raw JSONL for 100M by accident.
- Row shortfall: add more high-quality elite/TCEC months first, then rerun `preflight-h8-dataset --estimate-capacity`; only loosen `--max-rows-per-game`, `--skip-plies`, or caps with explicit user approval.
- Manifest missing/invalid: do not submit cache fanout.
- Worker/image mismatch: record image/tag, git SHA, and full command in logs/registry before submit.
- If logs are noisy, filter for semantic messages such as `progress`, `ERROR`, `manifest`, `validation`, or `uploaded`.
- Never start 100M BT4 training until both h7/h8 cache manifests validate.
