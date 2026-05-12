---
created: 2026-05-09
updated: 2026-05-12
project: tiny-neural-chess
id: agent_context.ops_constraints
type: ops_context
title: Operations constraints
status: active
priority: high
agent_summary: >
  Current operational guardrails: broad 10M architecture ablations are frozen, QAT is deployment polish unless explicitly reopened, BT4/SquareFormer and MF80 LC0 promotion should use the selected winners, generated artifacts are not committed, classic PUCT remains default for deterministic eval, and expensive offline work must use Rust/native/GPU paths when available.
---

# Operations constraints

- Do not resume Tactical MoveFormer without explicit user approval.
- Do not start new QAT work unless explicitly revisited. Quantization is final-deployment polish for now; accept quantized exports only with effectively zero quality loss.
- Do not launch new broad 10M architecture ablations. Current promotion winners are `bt4_h2_flip_av_relbank_d256_l8` and `mf80_av_top48_10m_flipped_moverel_gate`; reopen only for explicit criteria in [[Decision - LC0 architecture funnel and deployability frontier]].
- BT4/SquareFormer 100M supervised/cache work waits for validated h7/h8 SquareFormer cache manifests; LC0-distillation work should follow the LC0 adapter proof and promoted-winner funnel.
- Classic PUCT is default for deterministic eval; Gumbel-root is experimental/self-play only.
- Do not commit generated outputs under `data/*`, `artifacts/`, `public/models/*.onnx`, `public/models/*.json`, or `dist-client/`.
- Use `.venv-onnx/bin/python` for repo Python tasks.
- Use workload-appropriate execution paths: browser/TypeScript is for UI, browser parity, and small probes; Rust/native owns deterministic cache generation, self-play, arenas, long-running search/eval, and data preprocessing; Python owns PyTorch training/export/orchestration.
- For AWS SUP-SP self-play, use Batch array/job-level sharding as the primary scaling primitive; only run multiple workers inside one container to fill explicitly requested vCPUs, with one ORT thread per worker, globally unique shard IDs, per-worker chunks, and raw searched positions per vCPU-hour as the efficiency metric. See [[Decision - AWS Batch self-play parallelism policy]].
- Use `scripts/gpu_queue.py` / `npm run gpuq -- ...` for new expensive local GPU jobs that need serialization or artifact dependencies. Cloud workers should produce validated chunks/manifests/markers; local trainer/export/eval jobs consume those markers through the queue.
- New training/eval artifacts should use `board_normalization=stm_white_rankflip_v1` by default. Raw datasets/self-play chunks stay unmodified; cache/train/export/inference artifacts must declare normalization and reject mixed normalized/un-normalized inputs. See `docs/board_normalization_standard.md`.
- Before launching jobs expected to take more than a few minutes, state the long-run preflight: input size, expected wall time, parallelism, Rust/native/GPU availability, resumability, and whether the selected path is canonical.
- Local storage topology for LC0-scale work: root `/` is the active repo/working tree, `/mnt/backup_plus` is cold storage/archive, and `/dev/sda1` is mounted at `/mnt/data` with substantial free SSD space. Prefer `/mnt/data/tiny_leela_lc0/` for large active LC0 raw samples, normalized intermediates, and cache outputs when root disk pressure matters; use symlinks/manifests back into repo paths as needed.
- If a materially faster or more canonical execution path is discovered mid-run, compare restart cost with remaining runtime and prefer stopping/restarting when savings or correctness risk justify it.
