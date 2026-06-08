---
created: 2026-05-09
updated: 2026-05-12
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
  - [[Design - Candidate frontier cards]]
  - [[Design - Cross-language contracts and drift control]]
  - [[Design - LC0 search-distillation pipeline]]
  - [[Decision - AWS Batch self-play parallelism policy]]
  - [[Decision - LC0 architecture funnel and deployability frontier]]
risks:
  - [[Risk - Deprecated roadmap retrieved as active context]]
agent_summary: >
  Current roadmap keeps LC0 search-distillation as the primary supervised strength lane and Gumbel-Zero self-play as the separate clean research/control lane. The 10M ablation matrix is frozen for now: one MF80 winner and one BT4/SquareFormer winner advance to LC0 sanity, 10M pilot, and 100M+ scaling if eval/correctness gates pass. Promotion uses a multi-axis frontier: strength, latency, params, FLOPs, bytes, blunders, and calibration.
---

# Roadmap - Current Tiny Leela portfolio

Near-term model portfolio:

- MF80 is the current product-default/original-project lane. Provisional 10M winner for promotion: `mf80_av_top48_10m_flipped_moverel_gate`.
- BT4/SquareFormer is the compact-transformer strength/runtime lane. Provisional 10M winner for promotion: `bt4_h2_flip_av_relbank_d256_l8`.
- CNN96 remains an incumbent/control baseline and SUP-SP seed, but current blunder diagnostics make it less attractive as the final product default.
- The broad 10M architecture ablation matrix is frozen for now. Do not launch new variants unless [[Decision - LC0 architecture funnel and deployability frontier]] reopen criteria are met.

Primary training lanes:

- Make [[Design - LC0 search-distillation pipeline]] the main supervised-strength lane: public LC0 search-generated chunks -> policy/WDL/Q targets -> the promoted MF80 and BT4/SquareFormer winners. Use a tiny/fast adapter proof first, then 10k-100k LC0 sanity, 10M LC0 pilots on both winners, and 100M+ for both if no blocker appears.
- Keep clean Gumbel-Zero self-play separate as the zero/research lane, not expected to beat LC0 distillation initially but valuable as an independent bootstrap/control path.
- Use ChessBench/Stockfish-style data later as an action-value/ranking/regret teacher, not as an equal replacement for LC0 policy distributions.

Self-play and training-improvement foundation:

- Keep clean Gumbel-Zero separate from LC0-distillation, ChessBench AV training, and supervised-bootstrap SUP-SP.
- Build self-play correctness before larger generation: schema, lane provenance, seed/shard uniqueness, compressed cloud outputs, and validator/reporter coverage.
- Scale AWS SUP-SP generation with Batch array/job-level sharding first; use in-container worker fanout only to fill requested vCPUs, one ORT thread per worker, unique shard-prefixed game IDs, per-worker chunks/manifests, and raw searched positions per vCPU-hour as the efficiency metric.
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
- Feed latency, params, FLOPs/MACs, bytes, cache behavior, target-specific export cards, and quantized-vs-FP drift into promotion gates before deployment.
- Make [[Design - Candidate frontier cards]] the standard eval output for serious candidates: searchless proxy, calibrated protocol Elo at v16/v64/v128, fixed-time strength, evals/sec, params, FLOPs/MACs, bytes, arena/fixed-suite blunder rates, and value calibration.
- Use `docs/browser_runtime_configuration_and_benchmark_schema.md` as the shared runtime artifact contract for LC0, Tiny Leela/SquareFormer, larger LC0 packs, and future UCI/piece-odds variants so runtime recipes and model-quality frontier cards remain comparable but distinct.
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
- Treat [[Experiment - BT4 architecture roadmap next ablations]] as historical/source context for the completed 10M ablation push. Do not restart it without a specific reopen trigger from [[Decision - LC0 architecture funnel and deployability frontier]].
- Keep old roadmap prose as source material only unless represented in this active roadmap or linked active design/decision notes.
