---
created: 2026-05-09
updated: 2026-05-12
project: tiny-neural-chess
id: agent_context.open_questions
type: agent_context
title: Open questions
status: active
agent_summary: Curated active research questions for the Tiny Leela knowledge graph.
---

# Open questions

- Does h7/h8 history improve strength per byte enough for BT4/SquareFormer?
- Do the promoted 10M winners (`bt4_h2_flip_av_relbank_d256_l8` and `mf80_av_top48_10m_flipped_moverel_gate`) pass LC0 adapter sanity, 10M LC0 pilot, and 100M+ scaling gates?
- Which LC0 public run/test directories and format versions provide the best search-distillation source for Tiny Leela's first MF80 retrain?
- What is the safest LC0 1858-policy to Tiny Leela MoveId adapter contract under side-to-move rank-flip normalization?
- Which aux-PUCT weights transfer across visit counts and anchor types?
- Does action-value reranking reduce catastrophic regret without harming tactical sacrifices?
- When does PUCT become value-useful for each architecture lane?
- What minimal LC0 adapter correctness suite is sufficient before scaling beyond toy distillation samples?
- What minimal self-play correctness suite is sufficient before larger generation runs?
- Which model/backend/precision/batch/thread/legal-bucket combinations dominate the strength-vs-latency-vs-params-vs-FLOPs-vs-bytes Pareto frontier?
- Which historical/anchor pool and fixed tactical/failure suite should define the standard candidate frontier card?
- Which cache-generation hot paths should move first from Python loops to Rust workers while preserving NumPy memmap trainer compatibility?
- Which browser boundary is fastest and simplest: TypeScript PUCT, Rust/WASM PUCT with batched ORT Web callback, or a deeper custom runtime integration?
