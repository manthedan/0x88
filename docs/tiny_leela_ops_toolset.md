# Tiny Leela ops toolset

`tiny_leela_ops` is a lightweight Python CLI for replacing ad-hoc orchestration bash with durable, restartable operations.

Use the project Python:

```bash
.venv-onnx/bin/python -m tiny_leela_ops --help
# or
./scripts/tlops --help
```

## 1. Run registry

Append-only JSONL registry:

```text
artifacts/ops/runs.jsonl
```

Examples:

```bash
./scripts/tlops run record --kind h8_dataset --name h8-dataset-100m --status running \
  --attr aws_job_id=JOB_ID --attr s3_prefix=s3://...
./scripts/tlops run list --active
./scripts/tlops run show RUN_ID
./scripts/tlops run update RUN_ID --status succeeded
```

## 2. CLI wrapper

Top-level groups:

```text
run       registry records
phase     idempotent phase markers
cloud     AWS Batch job controller helpers
artifact  inventory and cold-storage moves
```

## 3. Idempotent phase state

For any run directory, `phase mark` writes structured marker files under `.tlops_state/`:

```bash
./scripts/tlops phase mark artifacts/cloud_h8_dataset_100m upload_inputs --state done
./scripts/tlops phase status artifacts/cloud_h8_dataset_100m
```

## 4. Cloud job controller

List/describe/watch Batch jobs with consistent AWS profile/region defaults:

```bash
./scripts/tlops cloud jobs --match 100m --statuses RUNNABLE RUNNING
./scripts/tlops cloud describe JOB_ID --logs --log-lines 80
./scripts/tlops cloud progress JOB_ID --lines 10
./scripts/tlops cloud watch JOB_ID --interval 60 --record-run-id RUN_ID
```

Adopt existing bash-submitted jobs into the registry:

```bash
./scripts/tlops cloud adopt-log artifacts/cloud_h8_dataset_100m/submit_latest.log \
  --run-id cloud-h8-dataset-100m-0cc7c352 --name h8-dataset-100m
```

`cloud submit-h8` is also available as a registry-aware wrapper around `cloud/aws/submit_h8_dataset_job.sh`.

After a dataset validates, use the registry-aware h7/h8 cache wrapper:

```bash
./scripts/tlops cloud submit-cache-h7-h8 \
  --bucket-uri s3://tiny-leela-distributed-ddbb/h8_dataset_100m \
  --dataset-s3-prefix s3://tiny-leela-distributed-ddbb/h8_dataset_100m/datasets/supervised_100m_elite_tcec_h8_v1 \
  --expect-train 100000000 --expect-dev 1000000 --submit
```

## 5. Artifact / cold-storage manager

Dry-run inventory:

```bash
./scripts/tlops artifact inventory artifacts data/datasets --limit 30
```

Cold-store dry-run, preserving relative paths under the external drive:

```bash
./scripts/tlops artifact cold-store --to /mnt/backup_plus/tiny_leela_cold_storage \
  --older-than-days 14 --min-size 1GB
```

Actually move only after reviewing the manifest:

```bash
./scripts/tlops artifact cold-store --to /mnt/backup_plus/tiny_leela_cold_storage \
  --older-than-days 14 --min-size 1GB --execute
```

Active path protection currently includes:

- live PID directories under `artifacts/**.pid`
- active run paths in `artifacts/ops/runs.jsonl`
- current 100M h8 cloud staging
- local 10M h7/h8 BT4 cache/training paths


## 6. Repo-local agent skills

Repo-local skills live under `.pi/skills/` and encode durable workflow policy, not live run state. Current skills:

```text
tiny-leela-ops              run registry, phase markers, tlops workflow
tiny-leela-cloud-pipeline   AWS/S3 h8 dataset and h7/h8 cache path
tiny-leela-gpu-priority     local GPU scheduling and BT4 priority rules
tiny-leela-artifact-hygiene cold storage, git hygiene, generated-file safety
tiny-leela-eval-promotion   arena/promotion defaults and caution checks
```

If a skill starts being stale or unhelpful, delete the corresponding `.pi/skills/<name>/` directory rather than preserving it as documentation.
