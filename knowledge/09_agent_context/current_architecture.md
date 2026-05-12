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
  Current preferred architecture portfolio is MF80 plus promoted SquareFormer/BT4. The BT4/SquareFormer provisional 10M winner is bt4_h2_flip_av_relbank_d256_l8; the MF80 provisional 10M winner is mf80_av_top48_10m_flipped_moverel_gate. SquareFormer-AV-PUCT remains the compact-transformer design direction.
---

# Current architecture

Current preferred portfolio:

```text
MF80 lane:
  mf80_av_top48_10m_flipped_moverel_gate

BT4/SquareFormer lane:
  bt4_h2_flip_av_relbank_d256_l8
```

Current compact-transformer design direction: [[Design - SquareFormer-AV-PUCT]].

Core components:

- 64 square tokens
- chess-aware geometry / relation bias
- from-to policy head
- WDL/value output
- action-value top-k reranking
- uncertainty / regret diagnostics
- conditional compact PUCT
- multi-teacher and on-policy distillation later
