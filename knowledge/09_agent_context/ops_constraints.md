---
created: 2026-05-09
updated: 2026-05-09
project: tiny-neural-chess
id: agent_context.ops_constraints
type: ops_context
title: Operations constraints
status: active
priority: high
agent_summary: >
  Current operational guardrails: QAT and Tactical MoveFormer are paused, BT4 waits for validated h7/h8 100M caches, generated artifacts are not committed, and classic PUCT remains default for deterministic eval.
---

# Operations constraints

- Do not resume Tactical MoveFormer without explicit user approval.
- Do not start new QAT work unless explicitly revisited.
- BT4/SquareFormer 100M training waits for validated h7/h8 SquareFormer cache manifests.
- Classic PUCT is default for deterministic eval; Gumbel-root is experimental/self-play only.
- Do not commit generated outputs under `data/*`, `artifacts/`, `public/models/*.onnx`, `public/models/*.json`, or `dist-client/`.
- Use `.venv-onnx/bin/python` for repo Python tasks.
