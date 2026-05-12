---
id: decision.aws_batch_selfplay_parallelism
type: decision
title: Decision - AWS Batch self-play parallelism policy
status: active
created: 2026-05-11
updated: 2026-05-11
priority: high
depends_on:
  - [[Concept - Search-improved self-play]]
  - [[Finding - Self-play needs search-improved targets]]
agent_summary: >
  Scale SUP-SP self-play on AWS primarily with Batch array shards, and only use
  multiple workers inside one container to fill explicitly requested vCPUs. Use
  one ORT thread per worker, globally unique shard IDs, per-worker chunks, and
  optimize raw searched positions per vCPU-hour.
---

# Decision - AWS Batch self-play parallelism policy

For supervised-bootstrap self-play (SUP-SP), AWS scaling should use **Batch array/job-level sharding as the primary parallelism primitive**. Sub-machine parallelism inside one Batch container is allowed, but only to fill vCPUs that the job explicitly requested.

## Policy

Primary parallelism:

- Submit many independent AWS Batch array children / shards.
- Each shard writes its own compressed self-play chunk and manifest.
- Merge only after per-shard validation succeeds.

Secondary parallelism inside a container:

- Use only when the Batch job requests multiple vCPUs.
- Prefer one self-play process per 1-2 vCPUs.
- Keep neural inference thread counts at one per worker:
  - `ORT_NUM_THREADS=1`
  - `ORT_INTRA_OP_NUM_THREADS=1`
  - `OMP_NUM_THREADS=1`

Good starting matrix:

- 1 vCPU job: 1 self-play worker.
- 2 vCPU job: 2 workers, each with one ORT thread.
- 4 vCPU job: benchmark 3-4 workers; accept only if raw positions per vCPU-hour improves.
- 8+ vCPU job: avoid until measured; larger containers increase memory/cache contention and retry blast radius.

## Shard identity and chunk safety

Every worker must receive a globally unique `--shard-id`, for example:

```text
batch${AWS_BATCH_JOB_ARRAY_INDEX}_w${WORKER_INDEX}
```

Game IDs must include the shard identity before chunks are concatenated. Otherwise independent workers can produce repeated IDs such as `g000000`, causing collisions in row keys like `(game_id, ply)` and downstream annotation/training joins.

Validated chunk flow:

1. Write one compressed chunk per worker.
2. Validate each chunk independently.
3. Build per-worker manifests.
4. Merge or manifest-join chunks only after validation.
5. Count scale by raw searched positions, not top-k expanded training rows.

## Optimization metric

Optimize:

```text
raw searched positions per vCPU-hour
```

Wall-clock time alone is not enough, because oversubscribing a container can look faster while wasting vCPU allocation or increasing failure risk. For SUP-SP, classic PUCT remains the default generation policy; Gumbel-root remains a separately labeled ablation lane.
