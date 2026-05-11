---
created: 2026-05-11
updated: 2026-05-11
project: tiny-neural-chess
id: design.runtime_target_matrix_and_workflow_delegation
type: design
title: Design - Runtime target matrix and workflow delegation
status: active
confidence: high
priority: high
depends_on:
  - [[Design - Inference optimization]]
  - [[Design - Agentic engine maintenance]]
related:
  - [[Roadmap - Current Tiny Leela portfolio]]
  - [[Design - Cross-language contracts and drift control]]
risks:
  - [[Risk - Move-map mismatch]]
agent_summary: >
  Tiny Leela should treat ONNX as a portable model artifact, not as one universal runtime. Browser WebGPU, browser WASM fallback, local CUDA, M-chip Mac mini, and AWS Batch workers need different execution providers, while Rust, TypeScript, and Python should be delegated by boundary: Rust for chess/search/cache hot paths, TypeScript for UI/browser glue, and Python/PyTorch for learning/export/orchestration.
---

# Design - Runtime target matrix and workflow delegation

Tiny Leela has five practical inference/deployment targets. A single `onnxruntime-web + TypeScript` path is useful for compatibility, but it is not the optimal stack for every target. ONNX should be the shared model artifact; each platform should choose the fastest reliable runtime, execution provider, and host language boundary.

## Platform targets

### 1. WebGPU-compatible browsers

Primary goal: fast in-browser play on modern Chromium/Safari/Firefox builds that expose WebGPU and support the required ONNX operators.

Preferred stack:

```text
TypeScript UI / browser shell
  -> Rust/WASM search engine when available
    -> batched evaluator callback
      -> onnxruntime-web WebGPU execution provider
         fallback: onnxruntime-web WASM
```

Target configuration:

```text
runtime package: onnxruntime-web/webgpu
executionProviders: ['webgpu', 'wasm']
model artifacts: ONNX FP32 baseline, candidate FP16 if parity/ops pass, k64/k128 fixed legal buckets for MF/BT4
packaging: browser-friendly single-file ONNX when possible; avoid large external-data graphs for public deploy unless streaming is explicitly implemented
benchmark gates: cold load, warmup, p50/p95 eval latency, PUCT evals/sec, browser memory, parity vs WASM/native
```

Boundary note: Rust/WASM search cannot directly call the ORT Web WASM/WebGPU internals without JavaScript glue. The practical boundary is a **batched evaluator callback** so Rust crosses into JS once per selected leaf batch, not once per move or visit.

### 2. All other browsers

Primary goal: maximum compatibility.

Preferred stack:

```text
TypeScript UI / browser shell
  -> TypeScript PUCT or Rust/WASM PUCT
    -> onnxruntime-web WASM evaluator
```

Target configuration:

```text
runtime package: onnxruntime-web or onnxruntime-web/wasm
executionProviders: ['wasm'] or implicit WASM
model artifacts: ONNX FP32 and carefully gated dynamic/static INT8 if ORT Web support and drift are acceptable
packaging: single-file ONNX preferred; small metadata JSON; cache via IndexedDB only after explicit load/cache tests
benchmark gates: WASM SIMD/thread availability, first-move latency, p95 latency on weaker devices, memory ceiling, parity vs canonical native/Python output
```

This is the one target where the current `onnxruntime-web + TypeScript` mental model is appropriate as a baseline.

### 3. Local NVIDIA GPU + CUDA machine

Primary goal: high-throughput inference for batched eval, self-play, benchmarks, and training-adjacent checks while preserving GPU priority for training.

Preferred stack:

```text
Rust native engine or Python benchmark harness
  -> native ONNX Runtime CUDA Execution Provider
     optional later: TensorRT Execution Provider
```

Target configuration:

```text
runtime: native ONNX Runtime, not ORT Web
host bindings: Rust `ort` crate for engine/search integration; Python `onnxruntime-gpu` for benchmark/export parity jobs
executionProviders: CUDAExecutionProvider, CPUExecutionProvider fallback; TensorRT only after correctness and build reproducibility are solved
model artifacts: ONNX FP32, FP16 candidates, static INT8/TensorRT candidates only after drift gates
benchmark gates: batch throughput, fixed-time search strength, GPU memory, launch overhead, parity vs FP32 CPU, non-interference with training jobs
```

Local CUDA should be used for bounded benchmark/profiling or batch self-play only when it does not compete with active training.

### 4. M-chip Mac mini

Primary goal: reliable CPU/offload evaluation, fixed-time PUCT sweeps, Stockfish anchors, and native inference benchmarking.

Preferred initial stack:

```text
Rust native engine
  -> native ONNX Runtime CPU via `ort`
```

Candidate accelerated stack:

```text
Rust/native harness
  -> ONNX Runtime CoreML Execution Provider or converted CoreML model
```

Target configuration:

```text
runtime: native ONNX Runtime CPU first; CoreML only if it wins benchmark/parity gates
host bindings: Rust `ort` preferred for search integration; Python/Node native ORT acceptable for benchmark comparison
executionProviders: CPUExecutionProvider, optional CoreMLExecutionProvider
model artifacts: ONNX FP32, dynamic INT8 if supported and strong, possible CoreML-converted artifact tracked separately from ONNX
benchmark gates: p50/p95 eval latency, PUCT evals/sec, ORT thread count, RSS, fixed-time anchors, parity vs canonical FP32
```

The Mac mini should not default to ORT Web WASM for serious eval once native ORT is available.

### 5. AWS Batch cloud workers

Primary goal: scalable dataset/cache/self-play workers with predictable cost and reproducible artifacts.

Preferred CPU-worker stack:

```text
Rust native worker
  -> native ONNX Runtime CPU when neural inference is required
  -> direct Rust chess/cache logic for deterministic cache generation
```

Preferred GPU-worker stack, if used later:

```text
Rust or Python worker
  -> native ONNX Runtime CUDA or TensorRT
```

Target configuration:

```text
runtime: native ORT or direct Rust logic; avoid ORT Web unless compatibility is more important than speed
host bindings: Rust for long-running deterministic workers; Python for PyTorch/teacher/export jobs and AWS orchestration glue
artifacts: compressed .jsonl.zst for self-play/output streams; binary memmap-compatible cache shards for training caches
benchmark gates: rows/sec or games/sec, cost per million rows/games, shard validation, reproducibility, compression, manifest integrity
```

## ONNX export target configurations

Every serious model should have an explicit export target card, not only `model.onnx`.

Canonical export dimensions:

```text
architecture: CNN / MF80 / BT4-SquareFormer
input encoding: board planes / compact square tokens / candidate legal IDs
legal bucket: dynamic, k64, k128, or model-specific fixed K
precision: FP32, FP16 candidate, dynamic INT8, static INT8 candidate
runtime target: browser-wasm, browser-webgpu, native-cpu, native-cuda, native-coreml, aws-cpu
packaging: single ONNX, external-data ONNX, simplified ONNX, quantized ONNX, converted CoreML/TensorRT derivative
```

Minimum artifact set per promoted candidate:

```text
model.fp32.onnx                 canonical reference export
model.fp32.meta.json            metadata: policy map, input encoding, history, K, heads, hashes
model.fp32.onnxsim.onnx         simplified candidate if it passes parity
model.int8.onnx                 quantized candidate only after drift gate
export_target_card.json         intended runtimes, EPs, precision, known unsupported EPs, benchmark results
```

Recommended default targets:

| Target | Runtime | Execution provider | Export preference |
|---|---|---|---|
| WebGPU browser | ORT WebGPU | `webgpu`, fallback `wasm` | ONNX FP32 first; FP16 only after parity; fixed K for candidate heads |
| Other browsers | ORT Web WASM | `wasm` | smallest reliable single-file ONNX; FP32 or gated INT8 |
| Local CUDA | native ORT | CUDA, optional TensorRT | FP32 reference, FP16/INT8 candidates for throughput |
| M-chip Mac mini | native ORT / CoreML | CPU, optional CoreML | FP32 native baseline, INT8/CoreML only if faster and correct |
| AWS CPU Batch | native ORT or direct Rust | CPU | FP32/INT8 native ORT for inference workers; no ORT needed for pure deterministic caches |

## Drift-control dependency

All runtime and workflow delegation must follow [[Design - Cross-language contracts and drift control]]. Parallel implementations are allowed only as migration paths or target-specific adapters, and they must conform to shared contracts/fixtures before they are used for promotion-critical work.

## Language/workflow delegation

Use language boundaries deliberately:

```text
Rust: deterministic chess/search/cache compiler hot paths
TypeScript: browser UI, browser runtime glue, existing eval harnesses during transition
Python: PyTorch training, ONNX export/quantization, NumPy memmap compatibility, cloud orchestration
```

### PUCT / Gumbel / search

Preferred ownership:

```text
Rust owns:
  board state, movegen, make/unmake or immutable child generation, policy action IDs, tree, PUCT/Gumbel selection, backup, search cache, batched leaf selection

TypeScript owns:
  browser UI events, model download/cache UI, WebGPU/WASM ORT session setup, callback glue, compatibility fallbacks

Python owns:
  offline analysis scripts, training-time search-improved label experiments, teacher/Stockfish orchestration where Python ecosystem is simpler
```

Browser practical architecture:

```text
TypeScript shell
  -> persistent Rust/WASM Engine object
    -> Rust selects N leaves
      -> JS/TS runs one ORT Web batch
        -> Rust receives flat policy/value/action-value arrays and backs up
```

Native practical architecture:

```text
Rust engine
  -> Rust `ort` evaluator with persistent native ORT session
    -> CPU/CUDA/CoreML EP depending target
```

Do not optimize by crossing Rust/JS for every visit or legal move. Optimize by moving the search tree and movegen entirely into Rust and crossing the runtime boundary only for batched neural evaluations.

### Training cache generation

Preferred ownership:

```text
Rust owns:
  JSONL/ZST row streaming, FEN parsing, movegen, move_to_action_id, policy target lookup, board plane encoding, compact square token encoding, legal candidate arrays, masks, slots, raw binary writes

Python owns:
  dataset manifest orchestration, shard partitioning, validation summaries, compatibility with np.memmap, trainer configuration, AWS submit/watch/finalize glue

NumPy owns:
  memmap file compatibility, bulk reshape/cast/one-hot expansion when it is genuinely vectorized

PyTorch owns:
  neural teacher forward passes, training, checkpointing, ONNX export
```

Recommended cache-builder shape:

```text
Python manifest wrapper
  -> launches Rust worker per shard
    -> Rust writes np.memmap-compatible binary arrays + meta.json
  -> Python validates rows/shapes/hashes and writes cache_manifest.json
  -> PyTorch trainer reads arrays through np.memmap unchanged
```

Rust should be prioritized for:

- SquareFormer compact token caches.
- Residual/CNN `x.int8` plane caches.
- MoveFormer legal sidecar caches: legal IDs, legal slots, masks, from/to/promotion features.
- Action-value overlay joins when the join is row-wise/key-wise and not neural.
- Self-play shard validators where legality/provenance/duplicates dominate runtime.

Keep Python/NumPy for:

- Already-vectorized compact-token to expanded-feature transforms until Rust proves faster.
- One-off validation/reporting where runtime is not material.
- Interfacing with existing trainers and memmaps.

Keep PyTorch for:

- Training and eval losses.
- Model export and quantization preparation.
- Teacher/student neural forward caches.
- GPU tensor workloads.

### Self-play and cloud workers

Preferred ownership:

```text
Rust owns:
  high-throughput game loop, legal move generation, PUCT/Gumbel, repetition/game-end detection, shard row production, deterministic validation

TypeScript owns:
  browser demos and compatibility probes, existing Node eval harnesses until native Rust replacements exist

Python/Bash owns:
  AWS Batch submission, S3 sync, manifest finalization, cost/status reports, training ingestion adapters
```

Cloud outputs should remain compressed `.jsonl.zst` for stream artifacts, while training caches should prefer binary arrays with explicit `meta.json` and `cache_manifest.json`.

### Evaluation and release gates

Preferred ownership:

```text
Rust/native:
  fixed-time engine matches, native PUCT throughput, local/AWS/Mac offload where speed matters

TypeScript/Node:
  browser-equivalent parity, WebGPU/WASM parity, existing UCI/arena harnesses during transition

Python:
  result aggregation, Stockfish label generation, statistical summaries, promotion packets
```

Promotion packets should report both model quality and runtime facts:

```text
loss / Elo / anchor score
latency p50/p95
PUCT evals/sec
fixed-time strength
bytes and load time
backend parity drift
cache hit rate
```

## Implementation order

1. Add export target cards for CNN96, MF80 k64/k128, and BT4/SquareFormer candidates.
2. Add a benchmark runner that compares ORT Web WASM, ORT WebGPU, native ORT CPU, and native ORT CUDA where available.
3. Prototype Rust native `ort` evaluator for CNN96 FP32 and compare against TS ORT Web parity/latency.
4. Move cache-builder hot paths to Rust while preserving existing memmap file names and trainer reads.
5. Add browser Rust/WASM search with batched ORT Web callback after native Rust search/evaluator contracts are stable.
6. Promote runtime target cards into release gates so models cannot ship without target-specific latency/bytes/parity evidence.
