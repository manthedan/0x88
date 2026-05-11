---
created: 2026-05-09
updated: 2026-05-09
project: tiny-neural-chess
id: concept.action_value_head
type: concept
title: Concept - Action-value head
status: active
topics:
  - training
  - search
  - reranking
agent_summary: >
  The action-value head predicts move-conditioned value for top candidates and supports reranking, regret diagnostics, and aux-PUCT experiments.
---

# Concept - Action-value head

An action-value head estimates the value of candidate moves, not only the value of the current position. It is central to reranking high-policy moves and detecting catastrophic regrets.

Supported finding: [[Finding - Action-value reranking is central to strength per node]].
