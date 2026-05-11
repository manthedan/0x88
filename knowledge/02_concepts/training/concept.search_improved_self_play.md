---
created: 2026-05-09
updated: 2026-05-09
project: tiny-neural-chess
id: concept.search_improved_self_play
type: concept
title: Concept - Search-improved self-play
status: active
topics:
  - self-play
  - training
  - search
agent_summary: >
  Self-play should train on post-search visit/WDL/Q targets, not raw sampled winners or static imitation alone.
---

# Concept - Search-improved self-play

Search-improved self-play uses model-guided search to generate better training targets. For Tiny Leela, policy targets should come from post-search distributions, with WDL/Q targets and resign calibration.

Finding: [[Finding - Self-play needs search-improved targets]].
