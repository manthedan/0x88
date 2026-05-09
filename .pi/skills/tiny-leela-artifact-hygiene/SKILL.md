---
name: tiny-leela-artifact-hygiene
description: Safe handling of Tiny Leela artifacts, datasets, cold storage, git status, and generated files. Use before moving, deleting, archiving, committing, or summarizing artifact/data directories.
---

# Tiny Leela Artifact Hygiene

Use this skill before modifying artifacts or datasets.

## Non-negotiable rules

Do not commit generated outputs under:

```text
data/*
artifacts/
```

Repo-local skills and source/docs changes may be committed if useful, but generated registries, logs, caches, model checkpoints, and datasets should stay out of git.

## Prefer dry-runs

Before moving anything:

```bash
./scripts/tlops artifact inventory artifacts data/datasets --limit 30
./scripts/tlops artifact cold-store \
  --to /mnt/backup_plus/tiny_leela_cold_storage \
  --older-than-days 14 \
  --min-size 1GB \
  artifacts data/datasets
```

The cold-store command defaults to dry-run unless explicitly configured otherwise. Keep it that way unless the user approves actual moves.

## Active paths to protect

Treat these as active or high-risk unless proven otherwise:

```text
artifacts/cloud_h8_dataset_100m/
artifacts/lc0_lite_squareformer/h7_h8_10m/
data/datasets/supervised_10m_elite_tcec_h8_v1/
data/datasets/supervised_100m_elite_tcec_v1/
```

Also protect any path referenced by:

```text
*.pid files
active registry runs
currently running AWS/local jobs
recent status/log files
```

## Git status workflow

When reviewing repository changes:

```bash
git status --short
```

Classify changes as:

```text
source/docs/scripts: candidates to keep
.pi/skills: repo-local agent workflow, candidates to keep if useful
artifacts/data/logs/checkpoints: generated, do not commit
/tmp or local engine wrappers: usually do not commit unless explicitly intended
```

## Deletion/cold-storage checklist

Before actual move/delete:

1. Confirm no live PID references the path.
2. Confirm no active `tlops` registry run references the path.
3. Confirm not part of the current h8/h7/h8/BT4 pipeline.
4. Prefer move to cold storage over delete.
5. Produce a short manifest of moved paths and byte counts.

## If uncertain

Do not move it. Report why it appears risky and ask the user.
