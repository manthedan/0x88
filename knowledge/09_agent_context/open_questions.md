---
created: 2026-05-09
updated: 2026-05-11
project: tiny-neural-chess
id: agent_context.open_questions
type: agent_context
title: Open questions
status: active
agent_summary: Curated active research questions for the Tiny Leela knowledge graph.
---

# Open questions

- Does h7/h8 history improve strength per byte enough for BT4/SquareFormer?
- Which aux-PUCT weights transfer across visit counts and anchor types?
- Does action-value reranking reduce catastrophic regret without harming tactical sacrifices?
- When does PUCT become value-useful for each architecture lane?
- What minimal self-play correctness suite is sufficient before larger generation runs?
- Which model/backend/precision/batch/thread/legal-bucket combinations dominate the strength-vs-latency-vs-bytes Pareto frontier?
- Which cache-generation hot paths should move first from Python loops to Rust workers while preserving NumPy memmap trainer compatibility?
- Which browser boundary is fastest and simplest: TypeScript PUCT, Rust/WASM PUCT with batched ORT Web callback, or a deeper custom runtime integration?
