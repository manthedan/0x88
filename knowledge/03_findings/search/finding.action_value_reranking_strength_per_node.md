---
created: 2026-05-09
updated: 2026-05-09
project: tiny-neural-chess
id: finding.action_value_reranking_strength_per_node
type: finding
title: Finding - Action-value reranking is central to strength per node
status: active
confidence: medium
evidence_level: experiment_supported
priority: high
supports:
  - [[Design - SquareFormer-AV-PUCT]]
depends_on:
  - [[Concept - Action-value head]]
  - [[Concept - PUCT]]
risks:
  - [[Risk - Move-map mismatch]]
agent_summary: >
  Action-value and related aux heads are high-leverage because they can improve move choice and PUCT quality without full-width expensive search.
---

# Finding - Action-value reranking is central to strength per node

Tiny models often have plausible policy but weak consequence modeling. Candidate action-values can rerank high-policy moves, expose regret, and tune aux-PUCT behavior.

Use this finding when designing SquareFormer, MoveFormer, or CNN-AV experiments.
