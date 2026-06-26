# Centipawn model efficiency metrics

We optimize Centipawn as a Pareto problem, not as one absolute Elo number. A model is only unambiguously better when it is stronger, smaller/simpler, and faster, or at least no worse on every axis. Do not reduce "tiny" to a hard MB cap: params, FLOPs/MACs, ONNX/export bytes, memory, latency, and strength per real wall-clock move are separate axes.

For current engine-family and browser-productization context, see [`engine_catalog.md`](engine_catalog.md). For cross-engine runtime recipes, benchmark artifacts, and footprint object shapes, use [`browser_runtime_configuration_and_benchmark_schema.md`](browser_runtime_configuration_and_benchmark_schema.md).

## Official comparison axes

Every serious model comparison should report:

```text
model name
architecture
training data / phase
params
params
estimated FLOPs/MACs
FP32 bytes
ONNX bundle bytes
quantized bytes when available
batch=1 eval latency
batch=16 eval latency
policy-only strength
classic PUCT strength
AV-PUCT strength, if calibrated
offline policy/AV metrics
```

Keep protocol names explicit.  Do not mix these as one Elo pool without saying so:

```text
policy-only
classic PUCT @ visits=N
AV-PUCT @ visits=N, avWeight=W
fixed-time @ T ms/move
```

## Primary metric: Pareto frontier

A candidate is dominated if another candidate in the same protocol is:

```text
>= Elo / quality
<= latency or stronger at the same wall-clock budget
<= params / FLOPs / bytes / memory pressure where those axes matter
```

with at least one strict improvement. Follow-up runs should focus on non-dominated candidates, but interpret dominance per target: CNNs may have more params/bytes while transformers may have higher FLOPs.

## Resource efficiency metrics

Raw `Elo / byte` is misleading because Elo has an arbitrary zero point.  Use *incremental Elo per resource doubling* instead:

```text
EloPerByteDoubling = ΔElo / log2(model_bytes / baseline_bytes)
EloPerParamDoubling = ΔElo / log2(params / baseline_params)
EloPerLatencyDoubling = ΔElo / log2(eval_ms / baseline_eval_ms)
```

Interpretation:

```text
+100 Elo/byte-doubling = doubling model size bought 100 Elo
  +0 Elo/byte-doubling = bigger model did not improve in this protocol
 negative value        = bigger model was worse
```

Use a relevant baseline, e.g. the smallest model in an arena, current browser model, or current champion.

## Offline screening metrics

Offline metrics are not final strength claims, but they decide what deserves GPU/search time.

Policy targets:

```text
policy CE
played-move top1/top4/top8
legal-move policy mass sanity
```

Teacher/action-value targets:

```text
AV MSE
AV top1 among candidates
Kendall tau or Spearman rho over candidate rankings
NDCG@k over candidate values
regret calibration
```

Tau/ranking metrics matter because search mostly needs the model to order candidate moves well, not merely match one played move.

## Quantization reporting

Quantization changes bytes and latency, not parameter count. Treat PTQ/QAT as deployment polish for now, not an architecture-selection gate. Report all deployed variants independently:

```text
FP32
FP16
INT8 dynamic/static
INT4, only if runtime support is reliable
```

Approximate storage:

```text
FP32 = 4 bytes/param
FP16 = 2 bytes/param
INT8 = 1 byte/param + scales/zero-points
INT4 = 0.5 byte/param + packing/scales
```

Quantized models should replace FP32 only with effectively zero quality loss. They must pass:

```text
policy KL vs FP32
top-k agreement vs FP32
WDL/value drift
AV drift, if AV head exists
fixed-search best-move/PV sanity
small arena parity
```

For transformer/LC0 browser work, start quantization with the largest dense, least softmax-sensitive subgraphs first: FFN dense layers, policy/value dense heads, and projections. Defer full int8 attention/layernorm paths until f16/custom-runtime parity is stable.

## Standard analysis tool

Use:

```bash
.venv-onnx/bin/python eval/model_efficiency_report.py \
  artifacts/search_mode_arena/example.json \
  --out artifacts/analysis/example_efficiency.md \
  --json-out artifacts/analysis/example_efficiency.json
```

The report computes:

```text
params from ONNX initializers
ONNX bundle bytes, including .onnx.data sidecars
Elo per byte/param doubling versus baseline
per-arena size/Elo Pareto frontier
```

If `--baseline NAME` is omitted, the smallest model in each arena is used as the resource baseline.

## Release gate recommendation

For any model we might promote to browser/default search, produce an analysis packet with:

```text
1. training metrics
2. model_efficiency_report output
3. fixed-visit arena: policy, PUCT, AV-PUCT if applicable
4. fixed-time arena: browser-relevant ms/move
5. quantization parity report, if quantized
6. browser WebGPU/WASM adaptive export/visit benchmark for final deployable models only
7. final Pareto decision: keep / reject / promote / needs larger run
```
