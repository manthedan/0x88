---
created: 2026-05-11
updated: 2026-05-11
project: tiny-neural-chess
id: design.cross_language_contracts_and_drift_control
type: design
title: Design - Cross-language contracts and drift control
status: active
confidence: high
priority: high
depends_on:
  - [[Design - Runtime target matrix and workflow delegation]]
  - [[Design - Agentic engine maintenance]]
related:
  - [[Design - Inference optimization]]
  - [[Roadmap - Current Tiny Leela portfolio]]
risks:
  - [[Risk - Move-map mismatch]]
agent_summary: >
  Tiny Leela may need Rust, TypeScript, and Python implementations during migration and across targets, but they must be governed by versioned contracts, shared fixtures, differential tests, and promotion gates so search, cache, encoding, self-play, and runtime behavior cannot silently drift.
---

# Design - Cross-language contracts and drift control

Tiny Leela will temporarily support overlapping Rust, TypeScript, and Python implementations. That is acceptable only if they are not independent sources of truth. Shared semantics must be defined by versioned contracts and tested with shared fixtures.

## Canonical ownership by phase

During private research and infrastructure optimization, the engine-canonical path is likely to be local/native first:

```text
Rust/native engine -> local CPU/GPU, Mac mini, AWS workers
```

This path is where we can optimize fastest, run large eval/self-play/cache jobs, and learn which model/runtime decisions matter.

When models are strong enough to share publicly, the product-canonical path for playable deployment may become browser-first:

```text
browser-optimized engine -> Rust/WASM search + ORT WebGPU/WASM evaluator + TypeScript shell
```

This is not a contradiction. Canonical ownership is target- and phase-aware. The stable source of truth must be the contract layer, not whichever implementation is fastest this month.

## Contract-first rule

Every cross-language boundary must name a contract version:

```text
policy_map_v1
move_action_id_v1
board_encoding_vN
squareformer_token_cache_v1
puct_trace_v1
selfplay_chunk_v1
failure_packet_v1
export_target_card_v1
cache_manifest_v1
```

A second implementation is allowed only when it includes:

1. contract version declaration,
2. shared fixture coverage,
3. differential/parity test,
4. owner/canonical phase note,
5. sunset or fallback rationale.

## Drift controls

Required controls:

- Shared fixtures under `tests/fixtures/contracts/` for edge positions, move encodings, cache rows, and PUCT traces.
- JSON schemas under `contracts/schemas/` for runtime/export/failure/self-play/cache contracts.
- Differential tests comparing TS/Python by default and Rust optionally or in scheduled gates.
- Trace-level PUCT parity for deterministic mock/student evaluators.
- Backend parity gates for ONNX/WASM/WebGPU/native CPU/CUDA/CoreML candidates.
- Cache validation gates for dtype, shape, row count, hash, policy map, legal slot, and provenance.
- Failure packets for every serious backend/cache/search drift event.

## PUCT parity level

PUCT drift cannot be controlled by checking only the final move. Tests should include:

```text
root legal priors
selected action per simulation for small deterministic traces
Q/value backup signs
visit counts
root policy distribution
batch eval accounting
```

For neural backends, exact trace parity may not be stable across execution providers; use tolerance metrics:

```text
policy KL / L1
top-k overlap
WDL max abs error
action-value/regret drift
final move agreement on deterministic seeds
fixed-time strength regression
```

## Cache-builder parity level

Cache builders are high-risk because Python, Rust, and trainers can silently disagree while producing valid-looking arrays. Each cache format needs:

```text
meta.json schema
cache_manifest_v1
array dtype/shape contract
sample-row fixture checks
policy/action-id parity
legal-slot parity
value-perspective checks
producer language + git/version metadata
```

Preferred architecture:

```text
Python manifest wrapper
  -> Rust shard worker writes memmap-compatible arrays
  -> Python validates manifest and shapes
  -> PyTorch reads unchanged memmaps
```

Python should not retain hidden chess semantics once Rust workers become canonical for a cache type.

## Promotion gate impact

A model/runtime/cache artifact cannot be promoted if it lacks the relevant contract evidence:

```text
encoding parity
move-map parity
PUCT invariant/trace suite
backend output drift report
cache schema validation
self-play chunk validation
export target card
failure packet replay command for known issues
```

This turns implementation drift into a visible gate failure rather than operational chaos.
