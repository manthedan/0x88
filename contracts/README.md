# Tiny Leela cross-language contracts

These files are the maintenance boundary between Rust, TypeScript, and Python.

The rule is:

```text
implementations do not define behavior by copying each other;
implementations conform to versioned contracts and shared fixtures.
```

## Canonical ownership by phase

During private research and infrastructure optimization:

```text
Rust: canonical deterministic engine/data hot paths for local CPU/GPU, Mac, and AWS workers
Python/PyTorch: canonical learning, checkpointing, export, quantization, and high-level orchestration
TypeScript: canonical browser UI/runtime glue and browser-compatibility adapter
```

When a model/runtime is ready for public browser distribution, the browser-optimized path may become the product-canonical engine path for that target. The contract remains the source of truth so native and browser paths can intentionally diverge in performance strategy without drifting in semantics.

## Contract families

- `policy_map_v1`: fixed policy move order and policy indices.
- `move_action_id_v1`: engine-wide action ID encoding for ordinary and promotion moves.
- `board_encoding_v*`: model input planes/tokens and value perspective.
- `puct_trace_v1`: deterministic search trace for implementation parity.
- `cache_manifest_v1`: training-cache file layout, dtype, shape, and provenance.
- `selfplay_chunk_v1`: immutable self-play JSONL row/provenance contract.
- `selfplay_annotation_v1`: sidecar annotation rows for Stockfish labels, diagnostics, and cache-prep metadata.
- `failure_packet_v1`: structured repro packet for tactical/backend/cache failures.
- `export_target_card_v1`: ONNX/runtime target card for deployment and promotion gates.

## Maintenance rules

1. Every second implementation must name the contract version it implements.
2. Every contract change requires fixture updates and at least one differential/parity test.
3. Rust/TS/Python may disagree on performance strategy, but not on action IDs, legal moves, value perspective, cache shapes, or row provenance.
4. Python orchestration must not hide chess semantics that are absent from Rust/TS tests.
5. Browser Rust/WASM must cross into ORT Web through batched evaluator calls, not per-move/per-visit calls.
6. Native Rust should use native ORT bindings for ONNX inference rather than ORT Web/WASM.

## Validation entry points

Fast default test suite:

```bash
npm test
```

Optional Rust/TS search parity:

```bash
npm run compare:rust-ts-search
npm run trace:puct
npm run rust:test
```

Contract fixtures live in:

```text
tests/fixtures/contracts/
```
