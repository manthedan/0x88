---
created: 2026-05-09
updated: 2026-05-09
project: tiny-neural-chess
id: concept.puct
type: concept
title: Concept - PUCT
status: active
topics:
  - search
  - puct
  - evaluation
agent_summary: >
  PUCT combines neural priors and value estimates for search; deterministic eval defaults to classic PUCT while aux/Gumbel variants remain experimental.
---

# Concept - PUCT

PUCT is the search policy used to balance policy prior, exploration, and value.

Tiny Leela currently treats classic PUCT as the deterministic eval default. Aux-PUCT and Gumbel-root variants require explicit calibration and separate evidence.
