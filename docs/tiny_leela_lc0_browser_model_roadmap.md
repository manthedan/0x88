# Tiny Leela -> Larger LC0 Browser Model Roadmap

## Goal

Use the current LC0 custom WebGPU/TVM work as the first reusable browser inference case study, then apply the same pattern to:

1. **Tiny Leela transformer models** — small enough for rapid train/export/pack/runtime iteration.
2. **Larger LC0 transformer models** — strong enough to test whether browser-native custom inference can make higher-quality nets practical without unacceptable download, memory, or latency costs.

The target is not merely lower eval latency. The target is a Pareto improvement:

```text
same or better playing/search quality
+ lower browser latency at realistic move budgets
+ lower download/cache footprint
+ bounded memory/GPU pressure
+ reproducible parity and strength gates
```

Stable native LC0/BLAS and f32 ONNX remain the correctness anchors. Custom WGSL/TVM, f16, int8, low-rank, and compressed-pack lanes must remain opt-in until their quality gates pass.

---

## Model-family ladder

### Family A — Tiny Leela transformer ladder

Purpose: fast architecture/compression iteration with a model class close enough to modern LC0 to transfer lessons.

| Tier | Candidate | Shape target | Params target | Browser target | Main question |
|---|---|---:|---:|---|---|
| A0 | TinyBT-static-S | 64 tokens, 4-6 layers, d=96-128, 4 heads, static chess bias | ~1-3M | WASM baseline + WebGPU custom | Can square-token transformers beat small CNNs at the same browser budget? |
| A1 | TinyBT-static-M | 6-8 layers, d=128-192, 4-6 heads | ~3-6M | WebGPU preferred | How much quality/latency do we buy with width? |
| A2 | TinyBT-smolgen-lite | A1 + lightweight dynamic attention bias | ~4-8M | WebGPU custom | Does LC0-style dynamic bias help small models enough to justify kernels? |
| A3 | TinyBT-AV-PUCT | A1/A2 + action-value/ranking head | +small heads | WebGPU/WASM hybrid | Can we reduce search nodes by predicting move values/rankings? |
| A4 | TinyBT-compressed | best A1-A3 with f16/int8/low-rank variants | same logical params | WebGPU f16 / WASM int8 | Can compression preserve strength while improving browser UX? |

Do not start with a literal shrink of BT4. Copy the useful priors — square tokens, chess relation bias, policy/WDL/moves-left/search compatibility — while keeping the first models simple and measurable.

### Family B — LC0-web 256x10/current-pack ladder

Purpose: productize the already-working custom path and make the runtime abstraction reusable.

| Tier | Candidate | Current role | Runtime target | Main question |
|---|---|---|---|---|
| B0 | ORT ONNX f32/f16 | baseline/control | ORT WebGPU/WASM | What is the browser baseline? |
| B1 | WGSL encoder + ORT heads | stable custom subgraph | custom WGSL + ORT heads | What does custom encoder buy without custom heads? |
| B2 | WGSL encoder + WGSL heads | opt-in custom runtime | custom WGSL heads | Can we avoid ORT and reduce readback while preserving parity? |
| B3 | WASM input + mixed-TVM FFN | current best custom lane | WASM input + hand/TVM WGSL | Does mixed generated/hand code beat ORT end-to-end? |
| B4 | compressed LC0-web | future deployment lane | f16/low-rank/int8 by subgraph | Which compression preserves LC0 behavior? |

This family is the bridge between TinyBT and larger LC0: it gives real LC0 semantics, existing pack tooling, native/f32 parity fixtures, and known WebGPU readback bottlenecks.

### Family C — Larger LC0 transformer ladder

Purpose: test whether stronger LC0 nets can become browser-practical with custom runtime and compression.

| Tier | Candidate | Scale | Browser target | Main question |
|---|---|---:|---|---|
| C0 | LC0 small transformer control | e.g. 256x10-class if available/exportable | ORT baseline + pack metrics | What is the smallest official-ish transformer we can load? |
| C1 | compressed small transformer | f16 pack + custom hot kernels | desktop WebGPU | Can custom kernels beat ORT enough to matter? |
| C2 | medium transformer slice | larger width/depth, maybe partial/layerwise probes first | WebGPU custom diagnostics | Where do memory/upload/readback limits appear? |
| C3 | distilled larger-to-smaller model | student trained from larger LC0 teacher | WebGPU/WASM fallback | Can we keep most strength at browser scale? |
| C4 | quantized/decomposed transformer | FFN int8/low-rank first, heads higher precision | custom runtime | Which subgraphs compress safely? |

Do not begin C with full product integration. Begin with manifest + pack metrics, ORT loadability, one-block/subgraph parity, then full search only after memory and parity look safe.

---

## Shared manifest extensions

The existing model manifest should grow deployment/runtime fields that work for TinyBT and LC0 packs. Suggested fields:

```json
{
  "model_id": "tinybt_static_s_v1",
  "family": "tiny-leela-transformer",
  "architecture": {
    "tokens": 64,
    "input_features": "square-token-history-v1",
    "layers": 6,
    "channels": 128,
    "heads": 4,
    "ffn_channels": 256,
    "attention_bias": "static-chess-relation",
    "heads_exported": ["policy", "wdl", "moves_left"]
  },
  "exports": {
    "onnx_f32": { "path": "...", "bytes": 0, "params": 0 },
    "onnx_f16": { "path": "...", "bytes": 0 },
    "lc0web_pack_f16": { "path": "...", "shards": 0, "bytes": 0 },
    "int8_ptq": { "path": "...", "bytes": 0, "calibration": "..." }
  },
  "browser_runtime": {
    "ort_webgpu_loadable": null,
    "ort_wasm_loadable": null,
    "custom_wgsl_supported": null,
    "preferred_runtime_preset": null,
    "estimated_gpu_bytes": 0,
    "peak_js_heap_bytes": 0,
    "cold_load_ms": null,
    "warm_eval_ms": null
  },
  "quality_gates": {
    "policy_kl_vs_teacher": null,
    "policy_top1": null,
    "policy_top4": null,
    "wdl_mae": null,
    "best_move_agreement": null,
    "fixed_visit_arena": null,
    "fixed_time_arena": null
  }
}
```

Manifest rules:

- Track **actual exported artifacts**, not only theoretical parameter counts.
- Separate FP32/f16/int8 artifacts; quantized size estimates are not release evidence.
- Track runtime support per browser/backend because ORT WebGPU operator coverage and custom kernels differ by model.
- Keep strength protocol-relative: policy-only, fixed visits, fixed time, AV-PUCT, and arena results are separate.

---

## Common parity and strength gates

### Gate 0 — artifact sanity

Required before browser work:

```text
ONNX loads in Python/ORT
initializer count and param count match manifest
input/output names match schema
sidecar/shard bytes match checksums
small fixed-batch inference emits finite outputs
```

### Gate 1 — offline neural parity

For custom runtime or compression variants, compare against the f32 teacher/export:

```text
policy max abs / RMS error
policy KL / CE delta
policy top1/top4/top8 overlap
legal policy mass sanity
WDL/value max abs and MAE
moves-left error, if exported
candidate action-value ranking tau/NDCG, if exported
```

Recommended pass bands should be family-specific. For LC0 custom f16/WGSL subgraphs, use strict numerical drift thresholds. For int8/low-rank/distilled TinyBT students, use quality thresholds versus teacher/search targets rather than exact equality.

### Gate 2 — browser runtime parity

For each deployable browser artifact:

```text
ORT WebGPU vs ORT WASM smoke, if supported
custom WGSL/TVM vs ORT/f32 fixture smoke
best-move agreement on representative FEN/history fixtures
nonzero/nonuniform policy/WDL guards
browser lifecycle cleanup/leak smoke
```

### Gate 3 — search agreement

For engines that use PUCT/search:

```text
fixed visits: 16/32/64/128 over representative fixtures
fixed time: 100/250/500/1000 ms over representative fixtures
best-move agreement vs baseline and teacher
root top-N overlap
PV sanity and value sign sanity
mismatch review for marginal/tactical positions
```

Do not use speculative scheduler modes such as `batchPipelineDepth>1` as parity evidence. They can be speed experiments, but they change search semantics.

### Gate 4 — strength preservation

Promotion candidates need at least one strength signal:

```text
policy-only arena vs parent/baseline
classic PUCT fixed-visit arena
fixed-time arena at browser-relevant budgets
AV-PUCT/rerank arena, if action-value head exists
```

A compressed model can be faster and still rejected if it loses too much strength at equal wall-clock budget.

---

## Runtime and compression experiment matrix

### Baseline matrix for every family

| Axis | Values |
|---|---|
| Runtime | ORT WASM, ORT WebGPU, custom WGSL/TVM |
| Precision | f32 reference, f16 deploy, int8 PTQ/QAT candidates |
| Input path | JS, WASM SIMD, WGSL projection if useful |
| Kernels | hand WGSL, TVM-generated WGSL, mixed hand+TVM |
| Readback | CPU-visible outputs, compact legal outputs, no-readback attribution |
| Search | policy-only, fixed-visit PUCT, fixed-time PUCT, AV/rerank where applicable |

### Compression order

Use this order to avoid mixing too many variables:

1. **f32/f16 export parity** — establish exact-ish model behavior.
2. **custom runtime parity** — prove the runtime can reproduce the model.
3. **f16 browser speed** — first deployment win for WebGPU.
4. **low-rank FFN/head candidates** — preserve dense kernels and reduce bytes/MACs.
5. **int8 FFN/projection candidates** — start with the largest dense subgraphs.
6. **QAT/distillation repair** — only after PTQ shows the quality failure mode.
7. **full search/arena** — only for candidates that pass neural/browser gates.

### Int8 scope

Int8 should be model-quality work, not just a kernel benchmark.

Start with:

```text
FFN dense layers
policy/value dense heads
input/output projections
```

Defer:

```text
softmax-sensitive attention score path
layernorm-heavy sections without clear scale handling
full transformer int8 end-to-end
```

For WebGPU, the important question is whether custom int8/i32 accumulation plus dequant beats f16 dense kernels end-to-end. For WASM, int8 is more likely to help sooner via bandwidth/cache/SIMD.

---

## Reusable custom-runtime architecture

The LC0 custom path should become a reusable engine pattern:

```text
ModelFamilyAdapter
  - feature schema
  - policy index/move mapping
  - output interpretation
  - parity fixtures

ModelPack
  - manifest
  - tensor shards
  - checksums
  - precision/compression metadata
  - static shape table

KernelRegistry
  - op/stage name
  - input/output shapes
  - hand WGSL implementation, if any
  - generated TVM WGSL implementation, if any
  - fallback ORT/WASM path
  - golden-output tests

RuntimePlan
  - ordered stages
  - buffer lifetimes
  - readback policy
  - command submission boundaries
  - attribution counters

BenchmarkAdapter
  - policy-only eval
  - fixed-visit search
  - fixed-time search
  - memory/load metrics
  - artifact JSON summary
```

LC0-specific pieces should stay behind adapters. Kernel generation, pack loading, parity harnessing, and benchmark summaries should be model-family agnostic where possible.

---

## TinyBT execution plan

### Step 1 — inventory current Tiny Leela assets

Produce a table of current Tiny Leela transformer/CNN candidates:

```text
model_id
architecture
params
ONNX bytes
training source
policy/WDL metrics
arena references
browser loadability
```

Use `npm run analysis:model-manifest` as the source of truth where possible.

### Step 2 — choose the first TinyBT target

Pick one first transformer candidate:

```text
TinyBT-static-S
L=6, d=128, heads=4, ffn=256
policy + WDL, moves-left optional
static chess-relation bias
```

The first success criterion is not beating LC0. It is beating or matching the current small CNN frontier at comparable browser latency/bytes.

### Step 3 — export and baseline

For the chosen candidate:

```text
export f32 ONNX
run offline fixture inference
run ORT WASM browser smoke
run ORT WebGPU browser smoke if supported
record pack/load/memory metrics
```

### Step 4 — custom runtime slice

Do not implement the entire transformer first. Start with subgraph slices:

```text
input embedding/projection
QKV projection
attention score + bias
softmax + value
FFN
policy/WDL heads
```

Each slice needs a golden-output check and a browser timing artifact before being integrated.

### Step 5 — compression candidates

Once f16 custom or ORT WebGPU baseline is stable:

```text
FFN low-rank
FFN int8 PTQ
policy/value head int8 PTQ
QAT repair if PTQ drift is too high
```

Only full-search candidates that pass neural drift gates should consume arena time.

---

## Larger LC0 execution plan

### Step 1 — model loadability and pack-size survey

For each candidate larger LC0 net:

```text
source format
ONNX export status
f32/f16 bytes
sidecar/shard count
expected GPU memory
ORT WebGPU loadability
ORT WASM fallback feasibility
```

Reject or defer models that cannot load within practical browser memory before doing custom kernels.

### Step 2 — layer/subgraph probes

Before full inference:

```text
single encoder block parity
attention subgraph parity
FFN subgraph parity
head parity
cold upload/load timing
peak memory estimate
```

This prevents wasting time on a full custom path that is impossible under browser memory limits.

### Step 3 — transfer LC0-web runtime lessons

Reuse from the current custom LC0 path:

```text
WASM input encoder if compatible
hand/TVM mixed FFN kernels
WGSL policy/WDL heads
readback attribution
fixed-FEN search matrix
browser harness cleanup
stable-default guards
```

### Step 4 — compression/distillation

Try model-quality-preserving reductions before exotic kernels:

```text
teacher-to-smaller-student distillation
FFN low-rank with dense kernels
f16 pack and custom f16 runtime
per-channel int8 FFN/projections
keep heads or final logits higher precision if policy drift is high
```

---

## Autoresearch vs curated branch responsibilities

Autoresearch should explore noisy performance spaces:

```text
kernel variants
batch sizes
readback strategies
scheduler variants
browser harness cleanup policy
int8/low-rank speed screens
```

The curated branch should do durable work:

```text
manifest/schema updates
parity gates
model-family adapters
custom-runtime abstractions
benchmark/comparison scripts
architecture docs
promotion criteria
```

Do not let autoresearch change stable defaults. Pull back only opt-in, validated candidates with artifact paths and parity evidence.

---

## Immediate next actions

1. Add manifest fields for `browser_runtime`, `architecture`, and `quality_gates` in the manifest builder/overrides flow.
2. Add a model-family comparison report that groups candidates by TinyBT, LC0-web, and larger LC0.
3. Add a custom-vs-ORT parity packet format shared by TinyBT and LC0.
4. Pick the first TinyBT-static-S candidate and run the offline/ORT browser baseline.
5. Keep current LC0 custom runtime productization focused on the opt-in `WASM input + mixed-TVM FFN + WGSL heads + JS legal + b4 depth1` lane until autoresearch contradicts it with stronger evidence.

---

## Non-goals for the next iteration

- Do not directly port a full large LC0 transformer before memory/loadability probes.
- Do not promote int8 based on kernel speed without policy/WDL/search quality gates.
- Do not optimize `batchPipelineDepth>1` as if it were parity-preserving search.
- Do not collapse all strength into one Elo number; keep protocol-specific results.
- Do not make custom WGSL/TVM the default until stable defaults and parity ladders pass.
