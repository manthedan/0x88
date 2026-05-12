---
created: 2026-05-11
updated: 2026-05-11
project: tiny-neural-chess
id: agent_context.execution_policy
type: ops_context
title: Execution policy
status: active
priority: high
depends_on:
  - [[ADR-0002 Workload execution boundaries]]
agent_summary: >
  Expensive local/offline work must use the workload-appropriate native/Rust/GPU path when it exists. Browser/TypeScript paths are for UI, browser parity, and small probes, not bulk cache generation, arenas, self-play, or long-running evaluation.
---

# Execution policy

Agents must choose the execution path by workload, not by whichever implementation already works.

## Runtime ownership

- Browser/TypeScript path: UI, browser compatibility, WebGPU/WASM parity, small probes.
- Python path: PyTorch training/export, orchestration, glue scripts, analysis, and cases where the native path does not exist yet.
- Rust/native path: deterministic chess semantics, cache generation, self-play, arenas, long-running search/eval, and data preprocessing.
- GPU/native ORT path: large batched inference/eval when available.

Do not use browser or TypeScript runtime paths for expensive local/offline work unless the task is explicitly browser parity testing.

## Long-run preflight

Before launching a job expected to take more than a few minutes, estimate or state:

- expected input size
- expected wall time
- available parallelism
- whether Rust/native/GPU support exists
- whether the job is resumable
- whether the selected path is canonical for that workload

If a faster or more canonical path exists, use it first. If the missing optimized path is small enough to implement before the run, implement it before launching the expensive job.

## GPU job queue

Use `scripts/gpu_queue.py` / `npm run gpuq -- ...` for new expensive local GPU jobs that can be expressed as queued commands with dependency markers.

- Queue training/export/eval jobs instead of embedding long polling loops in launcher scripts.
- Use filesystem markers or validated manifests as dependencies, especially for cloud-produced self-play/reanalysis shards.
- Keep cloud workers stateless: they produce chunks/manifests/markers; the trainer machine serializes GPU work through the queue.

## Mid-run upgrade rule

If a better execution path is discovered while a long job is running, compare remaining runtime with restart/build cost. Prefer stopping and restarting when the optimized path materially reduces wall time or reduces semantic drift risk.

Continue the current run only when preserving continuity is more valuable than the expected savings, and state that tradeoff explicitly.
