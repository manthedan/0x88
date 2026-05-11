---
created: 2026-05-09
updated: 2026-05-09
project: tiny-neural-chess
id: agent_context.current_architecture
type: agent_context
title: Current architecture
status: active
depends_on:
  - [[Design - SquareFormer-AV-PUCT]]
agent_summary: >
  Current preferred architecture is SquareFormer-AV-PUCT: square-token model, WDL/value heads, action-value top-k reranking, uncertainty, and conditional PUCT.
---

# Current architecture

Current preferred direction: [[Design - SquareFormer-AV-PUCT]].

Core components:

- 64 square tokens
- chess-aware geometry / relation bias
- from-to policy head
- WDL/value output
- action-value top-k reranking
- uncertainty / regret diagnostics
- conditional compact PUCT
- multi-teacher and on-policy distillation later
