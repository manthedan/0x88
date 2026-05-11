---
created: 2026-05-09
updated: 2026-05-09
project: tiny-neural-chess
id: finding.self_play_needs_search_improved_targets
type: finding
title: Finding - Self-play needs search-improved targets
status: active
confidence: high
evidence_level: paper_supported
priority: high
depends_on:
  - [[Concept - Search-improved self-play]]
agent_summary: >
  Self-play improvement requires search-improved visit/value targets; do not train policy on one-hot sampled Gumbel winners.
---

# Finding - Self-play needs search-improved targets

For AlphaZero/lc0-style improvement, train on post-search distributions and value/WDL/Q targets. Sampled moves are game actions, not automatically good supervised policy labels.
