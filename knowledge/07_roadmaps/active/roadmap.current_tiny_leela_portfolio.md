---
created: 2026-05-09
updated: 2026-05-11
project: tiny-neural-chess
id: roadmap.current_tiny_leela_portfolio
type: roadmap
title: Roadmap - Current Tiny Leela portfolio
status: active
priority: high
depends_on:
  - [[Design - SquareFormer-AV-PUCT]]
  - [[Design - 100M h7-h8 BT4 training pipeline]]
  - [[Design - Agentic engine maintenance]]
  - [[Design - Inference optimization]]
  - [[Design - Runtime target matrix and workflow delegation]]
  - [[Design - Cross-language contracts and drift control]]
risks:
  - [[Risk - Deprecated roadmap retrieved as active context]]
agent_summary: >
  Current roadmap keeps CNN, Tactical MoveFormer, Tiny BT4/SquareFormer, inference-optimization, and runtime/workflow delegation lanes separate, with 100M MF80/BT4/CNN96 as the near-term portfolio, agentic maintenance around the learner, and latency/throughput/bytes as first-class promotion metrics.
---

# Roadmap - Current Tiny Leela portfolio

Near-term model portfolio:

- CNN96 100M evaluation, release gates, PUCT tuning, and SUP-SP seed usage.
- Tactical MoveFormer / MF80 100M sidecar training, guarded continuation, and eval once explicitly resumed.
- Tiny BT4 / SquareFormer h7/h8 100M training, parity checks, and promotion gates after cache/model milestones.

Self-play and training-improvement foundation:

- Keep clean Gumbel-Zero separate from supervised-bootstrap SUP-SP.
- Build self-play correctness before larger generation: schema, lane provenance, seed/shard uniqueness, compressed cloud outputs, and validator/reporter coverage.
- Train on search-improved policy/WDL/Q/action-value/regret targets, not raw sampled moves.
- Calibrate classic PUCT first; keep Gumbel-root and aux/AV-PUCT experimental until gated.

Agentic engine maintenance lane:

- Treat heuristic/coding-agent learning as infrastructure around the neural model, not as a replacement for neural training.
- Capture every important failure as a structured packet: FEN, selected move, legal moves, policy/WDL/action-values, search stats, teacher best move/eval delta, backend, model id, seed, and reproduction command.
- Convert failures into invariant/metamorphic regression suites instead of brittle hard-coded chess rules.
- Route failures to the right oracle: Stockfish/tables for tactics and endgames, parity checkers for backend drift, schema/provenance validators for self-play shards.
- Mine repeated failures into hard negatives, teacher labels, replay-buffer sampling rules, or auxiliary/action-value/regret targets.
- Periodically compress useful heuristics back into model training, tests, gates, and knowledge-graph notes.

Immediate agentic-maintenance targets:

- Failure-packet schema and replay command contract.
- Queen-blunder / catastrophic-regret memory and teacher labeling.
- Move-map, promotion, castling, en-passant, value-perspective, PUCT-backup, and backend-parity regression cases.
- Quantized-vs-FP drift checks before deployment.
- Cloud self-play shard health checks: compression, manifests, provenance, seed uniqueness, duplicate shard detection, and validation summaries.

Inference optimization and deployment performance lane:

- Treat [[Design - Inference optimization]] and [[Design - Runtime target matrix and workflow delegation]] as first-class lanes, not cleanup after model work.
- Support five practical runtime targets explicitly: WebGPU browsers, all other browsers/WASM, local CUDA, M-chip Mac mini, and AWS Batch workers.
- Benchmark deploy candidates across ORT Web WASM, ORT WebGPU, native ORT CPU/Rust `ort`, local CUDA ORT/TensorRT candidates, Mac-native CPU/CoreML candidates, and quantized variants.
- Maintain matrix coverage for batch size, ORT threads, legal bucket size, precision, cold/warm startup, evals/sec inside PUCT, and fixed-time strength.
- Feed latency, bytes, cache behavior, target-specific export cards, and quantized-vs-FP drift into promotion gates before deployment.
- Prefer Mac mini for bounded CPU inference benchmarking and PUCT throughput sweeps after local correctness gates pass.

Workflow delegation lane:

- Move deterministic hot paths toward Rust: movegen, policy/action IDs, PUCT/Gumbel search, self-play game loops, and cache builders.
- Keep TypeScript focused on browser UI, runtime glue, ORT Web WASM/WebGPU setup, and browser parity harnesses.
- Keep Python/PyTorch focused on training, neural teacher/export/quantization work, NumPy memmap compatibility, and cloud/job orchestration.
- Preserve existing cache file formats so Rust workers can accelerate generation without forcing trainer rewrites.
- Require [[Design - Cross-language contracts and drift control]] for every parallel Rust/TS/Python implementation: contracts, fixtures, differential tests, and explicit promotion gates.

Research foundation:

- Maintain parity tests and UCI/OpenBench readiness.
- Calibrate aux-PUCT and visit curves.
- Keep old roadmap prose as source material only unless represented in this active roadmap or linked active design/decision notes.
