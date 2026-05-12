---
created: 2026-05-10
updated: 2026-05-11
project: tiny-neural-chess
id: design.inference_optimization
type: design
title: Design - Inference optimization
status: active
confidence: high
priority: high
depends_on:
  - [[Design - SquareFormer-AV-PUCT]]
  - [[Design - Agentic engine maintenance]]
related:
  - [[Roadmap - Current Tiny Leela portfolio]]
  - [[Design - Runtime target matrix and workflow delegation]]
  - [[Design - Candidate frontier cards]]
risks:
  - [[Risk - Move-map mismatch]]
agent_summary: >
  Inference performance is a first-class roadmap lane, not a cleanup task: Tiny Leela should benchmark params/FLOPs/bytes/latency and parity-check serious candidates, while reserving adaptive browser export selection and quantization polish for final deployable models.
---

# Design - Inference optimization

Inference optimization is a core Tiny Leela lane because engine strength at fixed hardware is a product of both model quality and nodes/positions evaluated per second. A smaller or faster model can dominate a stronger offline model if it enables more visits, lower latency, or better browser deployment. However, deployability is multi-axis: a hard MB cap is less useful than tracking params, FLOPs/MACs, bytes, memory, latency, and strength per wall-clock move.

## Scope

Optimize and measure the whole inference path:

- ONNX export shape choices: dynamic vs fixed legal buckets, batch axes, external data, simplified graphs.
- Runtime backends: use [[Design - Runtime target matrix and workflow delegation]] as the platform map: browser WebGPU, browser WASM fallback, native CPU/Mac, native CUDA, and AWS Batch worker paths.
- Model variants: FP32, FP16 where supported, dynamic/static INT8, and later INT4 only if runtime support is reliable.
- Search integration: batch size, ORT thread count, evaluator cache size, legal-move bucket size, transposition/cache behavior, and visit-count/time-control tradeoffs.
- Deployment packaging: ONNX byte size, external-data handling, bundle size, load time, warmup time, memory, and first-move latency.

## Required benchmark matrix

Maintain a repeatable matrix for every serious deploy candidate:

```text
backend: browser WebGPU, browser WASM, native CPU, Mac mini native CPU/CoreML candidate, local CUDA, AWS CPU/GPU worker
model: CNN96, MF80, BT4/SquareFormer, quantized variants
batch_size: 1, 2, 4, 8, 16 where applicable
ORT threads: 1, 2, 4, 6 where applicable
legal bucket: k64, k128, dynamic where applicable
precision: FP32, dynamic/static INT8, candidate FP16/INT4
protocol: policy-only latency, PUCT evals/sec, fixed-time strength
```

Report both microbenchmarks and engine-level metrics:

- eval latency p50/p95/p99
- positions/sec and evals/sec inside PUCT
- cache hit rate
- memory RSS/browser heap
- ONNX/bundle bytes
- warm-start and cold-start time
- strength at fixed visits and fixed time

## Optimization backlog

High-priority work:

1. Build a standard inference benchmark runner that emits JSON/TSV and updates the model manifest/frontier card with params, FLOPs/MACs, bytes, eval latency, and PUCT throughput.
2. Add final-deployment export target sections for browser-WASM, browser-WebGPU, native-CPU, local-CUDA, Mac-mini/CoreML-candidate, and AWS-worker targets when a model becomes a deployable finalist.
3. Compare `onnxruntime-web` WASM/WebGPU vs native ORT CPU via Rust `ort`/Node/Python on Mac mini and local CPU.
4. Benchmark batch sizes and ORT thread counts for CNN96, MF80 k64/k128, and BT4/SquareFormer.
5. Quantization parity gates for deployment candidates only: policy KL, top-k agreement, WDL drift, AV/regret drift, and small arena parity. Quantization should be accepted only with effectively zero quality loss.
6. Decide per-architecture legal bucket defaults: k64 vs k128 vs dynamic.
7. Optimize evaluator cache sizing and keying under PUCT/Gumbel search.
8. Track fixed-time playing strength, not only fixed-visit Elo.
9. Investigate browser-specific packaging: model split/merge, streaming load, warmup, workers, and memory ceilings.
10. Add native offload benchmark packets to the Mac-mini workflow after each model reaches a training/eval limit.
11. Feed results into promotion gates so a model cannot be promoted without strength/runtime/size/complexity evidence. Final deployable candidates additionally need browser WebGPU/WASM detection, lazy export selection, and adaptive visit benchmarking.

## Relationship to evaluation and training

Inference metrics are promotion metrics. A model should be judged on the Pareto frontier of strength, params, FLOPs/MACs, bytes, latency, memory, and search throughput, not only supervised loss or fixed-visit arena score. Serious candidates should emit [[Design - Candidate frontier cards]] covering searchless proxy strength, calibrated historical/anchor protocol Elo at v16/v64/v128, fixed-time strength when available, evals/sec, params, FLOPs/MACs, bytes, blunder rates, and value calibration.

This lane also informs training choices. If MF80 or BT4 is slower than CNN96 but not stronger enough at fixed time, the next training iteration should either improve strength per eval, reduce export/runtime overhead, or adjust search/bucket settings. Quantization and backend drift failures should produce agentic-maintenance failure packets and regression tests.
