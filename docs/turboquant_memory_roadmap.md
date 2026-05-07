# TurboQuant / Compressed Memory Roadmap

Goal: use TurboQuant-inspired ideas where they actually fit tiny Leela: compressed embeddings, semantic transposition memory, compact teacher-feature storage, and packed browser search data structures.

This is **not** a v1 model-forward quantization plan. A 64-square encoder transformer has no long-lived KV cache, so TurboQuant-style KV compression is unlikely to help first-pass inference.

## Core decision

```text
Use FP16 / INT8 / QAT for model weights and inference.
Use TurboQuant-like vector compression for stored embeddings and search memory.
```

## Why not use TurboQuant in v1 inference?

SquareFormer-style chess models are fixed-context encoder models:

```text
64 square tokens -> transformer encoder -> policy/WDL
```

There is no LLM-style persistent KV cache over thousands of tokens. Attention tensors are small and temporary. Adding quantize/dequantize work inside a 64-token forward pass is likely slower and riskier than just using FP16 WebGPU.

## Phase 0: baseline deployment compression

Do this before any TurboQuant-like work.

- [ ] Export FP16 ONNX for WebGPU.
- [ ] Measure browser policy-only eval/sec.
- [ ] Measure PUCT eval/sec at 32/64/128 visits.
- [ ] Try INT8 post-training quantization for WASM/CPU path.
- [ ] If INT8 PTQ hurts too much, add INT8 QAT.
- [ ] Store policies as top-k move lists, not full dense 1862-float arrays.
- [ ] Store MCTS nodes in packed typed arrays, not JS object graphs.

Promotion rule:

```text
Only optimize memory representation after latency/memory profiling shows a real bottleneck.
```

## Phase 1: compact policy and MCTS storage

Before semantic embeddings, make ordinary search memory efficient.

### Top-k policy storage

Instead of storing dense policy vectors:

```text
move_ids: uint16[k]
probs:    uint8 or uint16[k]
```

Suggested first format:

```text
k = 8 or 16
move_id: uint16
prob: uint16, normalized to sum over stored top-k
```

Track dropped probability mass:

```text
policy_tail_mass = 1 - sum(top_k_probs)
```

### MCTS typed-array layout

Store per-node fields in parallel arrays:

```text
parent:      int32
first_child: int32
child_count: uint16
move_id:     uint16
visits:      uint32
prior:       float32 or uint16 quantized
q:           float32 or int16 quantized
wdl:         optional packed int16 triplet
zobrist:     uint64 split into two uint32
```

This is likely more useful for browser performance than low-bit attention compression.

## Phase 2: exact transposition table with optional embeddings

Add a conventional neural transposition table first.

Key:

```text
zobrist hash + side/castling/ep state
```

Stored value:

```text
WDL/Q
policy top-k
net_id
search config
age/generation
```

Only after this works, optionally store a trunk embedding:

```text
embedding_fp16[128..256]
```

Use embeddings only for diagnostics at first:

- nearest previously seen positions;
- duplicate/opening cluster analysis;
- whether embedding similarity correlates with policy/value similarity.

## Phase 3: semantic transposition memory

Hypothesis:

```text
Similar neural embeddings can provide useful policy/value hints even when exact Zobrist positions differ.
```

This is risky for strength, so gate carefully.

### Initial use: hints, not replacements

For a new position:

```text
1. compute embedding
2. query approximate nearest neighbors
3. retrieve neighbor policy/value
4. blend only if similarity is high and uncertainty is low
```

Example blend:

```text
policy = (1 - alpha) * net_policy + alpha * memory_policy
value  = (1 - beta)  * net_value  + beta  * memory_value
```

Start with very small alpha/beta and disable in tactical/check positions.

### Guards

- never use semantic reuse in tablebase positions unless exact;
- disable when side-to-move king is in check;
- disable when material differs too much;
- require high cosine similarity;
- require same phase bucket;
- measure tactical regression.

## Phase 4: TurboQuant-like embedding compression

If fp16 embeddings become useful but too large, compress them.

Target vectors:

```text
trunk embedding: 128-256 dims
policy embedding: optional
value embedding: optional
```

Compression principles from TurboQuant:

- preserve inner products/cosine similarity;
- use random rotation before scalar quantization;
- prefer 4-bit values before 2-bit values for quality-sensitive vectors;
- measure retrieval quality, not just bytes.

Suggested first implementation:

```text
1. collect N fp16 embeddings from dev/search positions
2. apply fixed random orthogonal rotation
3. quantize per-vector or per-block to int4/int8
4. pack codes into Uint8Array
5. compare nearest-neighbor recall vs fp16 embeddings
```

Metrics:

```text
embedding_mse
cosine_error
top1_neighbor_recall
top10_neighbor_recall
policy_kl_of_retrieved_neighbors
value_error_of_retrieved_neighbors
browser lookup latency
memory bytes per entry
```

## Phase 5: compressed teacher-feature datasets

If teacher distillation stores dense embeddings or uncertainty features for many positions, use compressed vectors to reduce disk.

Potential row fields:

```json
{
  "fen": "...",
  "policy_topk": [[123, 0.31], [456, 0.17]],
  "wdl": [0.2, 0.5, 0.3],
  "teacher_embedding_q4": "base64-packed-codes",
  "quant_meta": "rotation_seed/block_scale_format"
}
```

Use cases:

- teacher-student representation distillation;
- uncertainty/disagreement heads;
- nearest-neighbor data mining;
- deduplication by semantic similarity.

Do not add this until policy/WDL distillation is already working.

## Phase 6: custom kernels only if justified

Do not port Triton TurboQuant kernels to browser directly. They target vLLM/KV-cache workloads.

Only consider custom WGSL/WebGPU kernels if profiling shows:

```text
compressed vector search is useful,
and dequantization/search dominates runtime,
and standard JS/WASM/WebGPU implementation is too slow.
```

## Evaluation gates

A compressed-memory feature must pass:

- no tactical-suite regression;
- no major policy-only arena regression;
- fixed-opening and varied-opening arena improvement or speedup;
- browser memory reduction is measurable;
- deterministic reproducible reports.

Promotion examples:

```text
keep if semantic memory improves PUCT Elo at same visits
keep if compressed memory preserves strength while reducing browser memory >2x
discard if nearest-neighbor reuse causes tactical blunders or opening collapse
```

## Immediate checklist

- [ ] Add packed top-k policy representation benchmark.
- [ ] Add typed-array MCTS node layout benchmark.
- [ ] Expose optional trunk embedding from CNN/SquareFormer evaluators.
- [ ] Build exact transposition table with policy/WDL reuse.
- [ ] Collect embedding similarity diagnostics.
- [ ] Prototype int8/int4 rotated embedding compression offline.
- [ ] Test semantic memory only as a disabled experimental flag.

## Bottom line

TurboQuant is useful here as a **compressed vector memory idea**, not as the first model quantizer.

The likely winning path is:

```text
FP16/INT8 model inference
+ packed top-k policy
+ compact MCTS arrays
+ exact transposition table
+ optional compressed semantic embedding memory later
```
