---
created: 2026-05-09
updated: 2026-05-09
project: tiny-neural-chess
id: design.h7_h8_bt4_training_pipeline
type: design
title: Design - 100M h7-h8 BT4 training pipeline
status: active
confidence: high
priority: high
depends_on:
  - [[Concept - SquareFormer]]
risks:
  - [[Risk - Move-map mismatch]]
agent_summary: >
  BT4/SquareFormer 100M training uses true h8 supervised data, then h7/h8 SquareFormer caches, then local training after manifests validate.
---

# Design - 100M h7-h8 BT4 training pipeline

Canonical pipeline:

```text
raw elite/TCEC rows -> supervised_100m_elite_tcec_h8_v1 -> cache_squareformer_h7 + cache_squareformer_h8 -> BT4 training
```

Do not substitute old h2 caches for true h7/h8 BT4 training.
