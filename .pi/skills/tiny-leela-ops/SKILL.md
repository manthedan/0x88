---
name: tiny-leela-ops
description: Tiny Leela run orchestration workflow. Use when checking, adopting, launching, watching, or summarizing local/AWS runs, pid/log/status files, tlops registry entries, or durable phase state in this repo.
---

# Tiny Leela Ops

Use this skill for process orchestration in the Tiny Leela repo.

## First principles

- Prefer the repo-local ops CLI over ad-hoc shell glue:
  ```bash
  ./scripts/tlops --help
  ```
- Use the repo Python unless there is a strong reason not to:
  ```bash
  .venv-onnx/bin/python
  ```
- Keep live state in durable files, not in chat memory:
  - `artifacts/ops/runs.jsonl`
  - `artifacts/ops/runs.sqlite` if present
  - `status.txt`
  - `*.pid`
  - `.tlops_state/*.done|*.failed|*.cancelled`
- Do not hand-edit generated state unless repairing a clearly broken entry. Prefer appending registry events through `tlops`.

## Standard inspection sequence

From repo root:

```bash
./scripts/tlops run list --active
./scripts/tlops cloud jobs --match 100m --statuses RUNNING RUNNABLE SUBMITTED STARTING
./scripts/tlops cloud status-h8-100m
./scripts/tlops artifact inventory artifacts data/datasets --limit 12
```

For a known AWS Batch job:

```bash
./scripts/tlops cloud describe JOB_ID
./scripts/tlops cloud progress JOB_ID --lines 10
```

Before submitting or unblocking cloud dataset/cache phases:

```bash
./scripts/tlops cloud preflight-h8-dataset --compress-inputs --estimate-capacity
./scripts/tlops cloud validate-h8-manifest --s3-prefix S3_DATASET_PREFIX
./scripts/tlops cloud repair-plan-h8-dataset --actual-train ACTUAL_ROWS
```

For a known run:

```bash
./scripts/tlops run show RUN_ID
```

## Native Rust eval/search jobs

For bounded native ONNX arena or parity work, prefer the Rust tools once local correctness gates pass:

```bash
npm run rust:arena -- --candidate-onnx C.onnx --candidate-meta C.meta.json --baseline-onnx B.onnx --baseline-meta B.meta.json
npm run compare:rust-ts-board -- --meta MODEL.meta.json
npm run compare:rust-ts-onnx-eval -- --model MODEL.onnx --meta MODEL.meta.json --generated 4
npm run bench:rust-ts-board
```

`eval/search_mode_arena.mjs --backend rust` is currently a launcher for exactly two classic-PUCT players. Record whether an eval result came from TS/ORT Web, Rust/native ORT, browser WebGPU, Mac mini native CPU, local CUDA, or AWS Batch; runtime backend is part of the artifact provenance.

## When adopting an already-running process

1. Identify the durable identity:
   - AWS job ID, or
   - PID file + log file + output directory.
2. Add/adopt into registry with a stable run id.
3. Record local paths and remote prefixes in attrs.
4. Mark phase state only for actions confirmed from logs/S3/filesystem.

## Status conventions

Use these statuses unless a script already defines a more specific terminal marker:

```text
planned | running | submitted | succeeded | failed | cancelled | paused
```

For split workflows, prefer phase events:

```text
compress_inputs.done
upload_inputs.done
batch_submitted.done
validated.done
cache_submitted.done
```

## Safety rules

- Before starting a long process, ensure it has:
  - log path
  - pid path or AWS job id
  - status path or registry run id
  - restart/resume story
- Do not resume lower-priority GPU jobs without using `tiny-leela-gpu-priority`.
- Do not move/delete artifact paths without using `tiny-leela-artifact-hygiene`.
- If `tlops` is missing a needed command, add a small focused subcommand instead of creating another one-off bash blob.
