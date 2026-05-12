---
created: 2026-05-11
updated: 2026-05-11
project: tiny-neural-chess
id: decision.workload_execution_boundaries
type: decision
title: ADR-0002 Workload execution boundaries
status: active
confidence: high
priority: high
supports:
  - [[Design - Runtime target matrix and workflow delegation]]
risks:
  - [[Risk - Browser path used for expensive offline work]]
agent_summary: >
  Choose execution backends by workload: browser/TypeScript for UI and browser parity, Rust/native for deterministic long-running chess/cache/search work, Python for training/export/orchestration, and GPU/native ORT for large batched inference.
---

# ADR-0002 Workload execution boundaries

## Decision

Tiny Leela will maintain multiple implementations when needed, but agents and scripts must select the implementation by workload class.

Runtime ownership:

- Browser/TypeScript: UI, browser compatibility, WebGPU/WASM parity, and small probes.
- Rust/native: deterministic chess semantics, cache generation, self-play, arenas, long-running search/eval, and data preprocessing.
- Python: PyTorch training/export, orchestration, analysis, and glue where native paths do not exist yet.
- GPU/native ORT: large batched inference/eval when available.

## Rationale

The project intentionally targets browser deployment and local/native heavy computation. A browser-compatible path proves deployability; it does not make that path appropriate for expensive offline runs. Long-running jobs should pay a small upfront engineering cost to use Rust/native/GPU paths when those paths exist or are straightforward to add.

## Consequences

- Before launching expensive jobs, agents must perform a long-run preflight: input size, expected wall time, parallelism, Rust/native/GPU availability, resumability, and whether the chosen path is canonical.
- If the optimized path is missing but cheap to build relative to the run, build it first.
- If a better path is discovered mid-run, compare restart cost against remaining runtime and drift risk; stop and restart when the savings or correctness benefits are material.
- Browser/TypeScript execution for bulk offline work is allowed only for explicit browser parity testing or when no native path exists and the expected runtime is small.
