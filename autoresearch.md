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

## Primary metric
`tiny_leela_score` (higher is better), emitted by `./autoresearch.sh`.

The score is a deterministic project fitness proxy until real playing-strength gauntlets exist. It rewards verified implementation milestones: TypeScript package, chess rules/tests, fixed move codec, feature encoder, PUCT search, neural inference interface, browser worker/runtime, model export/compression docs/scripts, and evaluation harness.

When Elo/gauntlet infrastructure exists, create a new benchmark id and metric segment rather than comparing directly to this bootstrap score.

## Benchmark command
```bash
./autoresearch.sh
```

## Correctness checks
```bash
./autoresearch.checks.sh
```

## Dovetail lanes to maintain
- L0 baseline/product scaffold: package structure, deterministic tests, docs, CI-friendly benchmark.
- Architecture/runtime: feature encoder, move indexing, tiny conv model spec, ONNX interface.
- Search/engine: legal moves, PUCT, policy masking, time controls.
- Browser deployment: Web Worker, ORT Web WASM/WebGPU fallback, cache/progressive loading.
- Training/data: teacher-label format, distillation loss, synthetic fixtures, export/compression scripts.
- Evaluation: node-free metrics, tactical tests, self-play/gauntlet harness, latency/size tracking.

## Safety and reproducibility rules
- Never improve the benchmark by weakening chess correctness or evaluation workload.
- Keep move indexing and feature encoding backwards-compatible once fixtures exist.
- Add tests for every encoding/search rule change.
- If the primary metric changes from bootstrap score to Elo/loss/latency, initialize a new benchmark segment.
