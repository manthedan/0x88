---
created: 2026-05-09
updated: 2026-05-09
project: tiny-neural-chess
id: design.squareformer_av_puct
type: design
title: Design - SquareFormer-AV-PUCT
status: active
confidence: medium
priority: high
depends_on:
  - [[Concept - SquareFormer]]
  - [[Concept - Action-value head]]
  - [[Concept - PUCT]]
supported_by:
  - [[Finding - Action-value reranking is central to strength per node]]
  - [[Finding - Chess-specific geometry is high ROI for tiny models]]
risks:
  - [[Risk - Move-map mismatch]]
agent_summary: >
  Current preferred lightweight architecture: SquareFormer policy/WDL/AV outputs with action-value reranking and calibrated conditional PUCT.
---

# Design - SquareFormer-AV-PUCT

SquareFormer-AV-PUCT combines square-token transformer inference with policy, WDL/value, action-value reranking, uncertainty/regret diagnostics, and compact PUCT.

## Current status

Active design direction. BT4/SquareFormer 100M work remains gated on validated h7/h8 cache manifests.
