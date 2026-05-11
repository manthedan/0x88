---
created: 2026-05-09
updated: 2026-05-09
project: tiny-neural-chess
id: finding.chess_specific_geometry_high_roi
type: finding
title: Finding - Chess-specific geometry is high ROI for tiny models
status: active
confidence: medium
evidence_level: paper_supported
priority: high
supports:
  - [[Design - SquareFormer-AV-PUCT]]
depends_on:
  - [[Concept - SquareFormer]]
agent_summary: >
  For tiny chess transformers, chess-specific relation bias and square geometry should be tested before generic transformer tricks.
---

# Finding - Chess-specific geometry is high ROI for tiny models

Chess-specific square relations, geometry, and legality structure are likely more valuable for Tiny Leela than generic transformer embellishments at the same parameter budget.
