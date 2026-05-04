# Tiny Leela Dovetail Research Program

## Objective
Build a tiny in-browser Leela-like chess engine: policy/value neural evaluation plus PUCT search, deployable through ONNX Runtime Web with WASM baseline and WebGPU optional acceleration.

## Research-paper takeaways
- Do **not** port full lc0 first; build a small AlphaZero/lc0-inspired engine with a fixed feature encoder, fixed legal move indexing, policy/WDL heads, and PUCT.
- First product target: classic dense convolutional student, not modern lc0 transformers.
- Recommended model lanes: micro 16x2/24x3, balanced 48x5/64x6, desktop 112x9 or tiny transformer later.
- Distillation from teacher/search targets is the most important compression/training technique.
- Browser deployment should prefer ONNX Runtime Web, with WASM/SIMD fallback and WebGPU/FP16 as an acceleration lane.
- Highest risk: silent feature/move-encoding mismatches between training, export, and inference.

## Current readiness metric
`tiny_leela_score` (higher is better), emitted by `./dovetail.sh`.

This is a Phase A/B lab-readiness score, not a chess research-performance metric. It rewards verified implementation and benchmark-definition milestones: TypeScript package, chess substrate, fixed move codec, feature encoder, PUCT/evaluator interfaces, docs, and frozen Phase B metric specs.

Phase C research must optimize fixed engine metrics from `eval/benchmark_spec.json`, such as `policy_top1_acc`, `wdl_cross_entropy`, `fixed_playout_suite_score`, latency, and model size. If the benchmark dataset or workload changes, create a new benchmark id rather than comparing directly to this bootstrap segment.

## Benchmark command
```bash
./dovetail.sh
```

## Correctness checks
```bash
./dovetail.checks.sh
```

## Foundation vs research lanes

Before dovetailing heavily, L000 owns the shared minimum engine substrate. These are not differentiated research ideas; they are blocking pre-work used by every lane:

- TypeScript/test scaffold.
- Board state and FEN parsing.
- Deterministic move representation and action id mapping.
- Baseline move generation and move application.
- Feature encoder skeleton.
- Backend-neutral evaluator interface.
- Search entry point consuming only `BoardState` + `Evaluator`.
- Move encoding and browser runtime docs.

Loaded ideas that were actually foundation tasks are marked `foundation_milestone` in `dovetail.ideas.jsonl`. True research-lane ideas are explicitly titled `True research lane: ...` and should compare structurally different approaches after the substrate is stable.

## Dovetail lanes to maintain after substrate
- L0 foundation/product scaffold: deterministic tests, docs, CI-friendly benchmark, shared interfaces.
- Search lane: PUCT variants, FPU, batching, uncertainty-driven playout allocation.
- Architecture lane: 16x2/24x3/48x5/64x6 conv students and later tiny transformers.
- Browser runtime lane: Web Worker, ORT WASM/WebGPU fallback, cache/progressive loading.
- Training/data lane: teacher-label format, distillation losses, export/compression scripts.
- Evaluation lane: node-free metrics, tactical tests, self-play/gauntlet harness, latency/size tracking.

## Safety and reproducibility rules
- Never improve the benchmark by weakening chess correctness or evaluation workload.
- Keep move indexing and feature encoding backwards-compatible once fixtures exist.
- Add tests for every encoding/search rule change.
- If the primary metric changes from bootstrap score to Elo/loss/latency, initialize a new benchmark segment.
