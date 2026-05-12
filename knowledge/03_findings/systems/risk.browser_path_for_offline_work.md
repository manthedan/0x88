---
created: 2026-05-11
updated: 2026-05-11
project: tiny-neural-chess
id: risk.browser_path_for_offline_work
type: risk
title: Risk - Browser path used for expensive offline work
status: active
priority: high
agent_summary: >
  Agents may reuse browser/TypeScript code for expensive local jobs even when Rust/native/GPU paths exist, causing large avoidable slowdowns and sometimes semantic drift.
---

# Risk - Browser path used for expensive offline work

Browser-compatible code is necessary for deployment, but it is usually the wrong backend for bulk local computation.

Failure modes:

- cache generation or arenas run through browser/TypeScript paths instead of Rust/native workers
- long jobs start before checking GPU/native acceleration, batching, parallelism, or resumability
- agents continue a slow run even after a much faster path is identified
- deterministic preprocessing semantics drift because the optimized/canonical path was not implemented first

Mitigation: follow [[ADR-0002 Workload execution boundaries]] and the active [[Execution policy]].
