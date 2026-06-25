# Neural browser runtimes: ONNX WebGPU, TVMJS, WGSL, WebNN, and QDQ

The LC0/Maia side of 0x88 is not a CPU UCI engine port. It is a browser neural
inference stack with several runtime lanes. They all consume chess positions and
produce policy/value outputs, but they have different maturity levels and
different artifact formats.

## Runtime matrix

| Runtime lane | Role | Status | Artifact shape |
| --- | --- | --- | --- |
| ORT ONNX WebGPU | Stable browser neural baseline for LC0-family models | Default where WebGPU is available; ORT WASM remains fallback/control | `.onnx` model plus optional external data/meta |
| ONNX QDQ | Download-size reduction for the ORT lane | Promoted per model only after fixture/search drift gates | `.qdq8.onnx` with int8 weights and in-graph `DequantizeLinear` |
| Custom WGSL / hybrid | Hand-written WebGPU kernels for hot LC0 subgraphs | Explicit runtime lane, benchmark/parity gated | lc0web tensor pack + WGSL kernels, sometimes ORT heads |
| TVMJS WebGPU | Whole-model compiler lane: ONNX → TVM Relax → browser WebGPU runtime | Research/opt-in; not a default | TVMJS wasm module, runtime JS, params/tensor cache, manifest |
| ORT WebNN EP | Future non-WebGPU hardware lane through WebNN/CoreML/ANE/NPU | Flag-only probe; not shippable until WebNN is unflagged and model gates pass | ONNX model, often fixed-shape and QDQ/block-DQL adjusted |

## ORT ONNX WebGPU: the stable baseline

ONNX Runtime Web with the WebGPU execution provider is the stable neural runtime
baseline. It is the comparison target for TVMJS, custom WGSL, WebNN, and QDQ
experiments because it gives us:

- a maintained browser runtime with WebGPU and WASM execution providers;
- straightforward model publication (`.onnx` plus manifest/provenance);
- consistent smoke paths for fixture parity and search move comparisons;
- an emergency WASM fallback/control path when WebGPU is absent.

Policy: new neural runtime lanes should prove themselves against ORT WebGPU
before product promotion. If a custom lane is faster but disagrees on fixture
best moves, WDL, or search moves, it stays research-only until the drift is
explained and accepted.

## ONNX QDQ: smaller downloads, same ORT lane

QDQ artifacts are ordinary ONNX graphs with quantized weight initializers and
explicit `QuantizeLinear`/`DequantizeLinear`-style graph nodes. In this repo the
main deployment pattern is weight-only int8 QDQ:

- large MatMul B tensors are stored as int8 with per-output-channel scales;
- compute still runs in the runtime's normal float path after dequantization;
- file size drops by roughly 2× for large LC0/Maia artifacts;
- per-eval latency can be slightly worse because dequantization is now in the
  graph, so QDQ is primarily a transfer/cache win unless a backend accelerates
  the quantized form directly.

Promotion rule: judge QDQ on real chess fixture/search gates, not random-input
node comparisons. Out-of-distribution random tensors can exaggerate int8 error
and are not representative of LC0/Maia activations.

## Custom WGSL / hybrid runtime

The WGSL lane exists because generic ONNX runtimes cannot always express the
best browser schedule for LC0-style networks. The current custom path uses
hand-written WebGPU kernels for selected subgraphs such as encoder blocks,
attention pieces, mapped policy heads, legal-prior/readback experiments, and
batch-slot management.

Two important product distinctions:

- **Hybrid WGSL + ORT heads** can keep the custom encoder on GPU while using ORT
  for smaller head subgraphs. This reduces implementation risk and gives a
  stable comparison anchor.
- **Full WGSL heads** reduce ORT dependence/readback overhead further, but are
  more parity-sensitive and stay explicitly gated.

Every WGSL kernel needs CPU/ORT reference checks, fixed-suite drift gates, and
search-level validation. Microbenchmarks alone are not enough: queue submission,
GPU readback, batch fill, and tree-search interaction often dominate the final
visits/second result.

## TVMJS WebGPU

TVMJS is the whole-model compiler lane. The pipeline imports fixed-batch f16
ONNX exports into TVM Relax, builds browser-loadable TVMJS wasm/runtime assets,
requests WebGPU features such as `shader-f16`, prebuilds pipelines, and runs the
Relax VM in the browser.

Why it matters:

- it tests whether a compiler can beat hand-maintained WGSL and generic ORT for
  whole LC0 models;
- it gives a second implementation to triangulate ORT/WebGPU correctness;
- detached parameter/tensor-cache work can reduce duplicated per-batch artifact
  weight.

Why it is not the default:

- generated runtime artifacts are large and require their own hosting/cache
  policy;
- compiler provenance and per-build parity gates are mandatory;
- cold-start phases (wasm fetch/verify, instantiate, WebGPU init, pipeline
  prebuild, VM creation, parameter upload) must be budgeted separately from warm
  eval timing;
- cross-device evidence, especially non-Apple GPUs, still matters before any
  default promotion.

TVMJS may be exposed as an explicit research/runtime selector, but ORT ONNX
WebGPU remains the rollback/default lane unless a release owner accepts the
promotion evidence.

## ORT WebNN EP: promising, but still flag-gated

The WebNN execution provider is the future hardware-accelerator lane: Chrome can
route supported ONNX graphs through WebNN to platform backends such as CoreML on
Apple Silicon. In current probes it requires launching Chrome with the WebNN
feature flag (`WebMachineLearningNeuralNetwork`), so it is not a normal user
feature yet.

Current lessons:

- fixed shapes and int32-friendly graph cleanup are often required;
- f16 correctness can be backend/model-size sensitive, especially for larger
  LC0 attention/smolgen graphs;
- f32 or carefully shaped QDQ/block-`DequantizeLinear` artifacts can restore
  correctness while still giving useful acceleration;
- one-time platform compile can take seconds, so cold-start and cache behavior
  must be measured separately from warm evals.

Policy: do not ship WebNN as a default or promise it in product copy until the
browser feature is unflagged for the target audience and each model passes the
same fixture/search/parity gates as WebGPU. Keep WebNN probes as a forward-looking
lane for non-WebGPU coverage and NPU/ANE experiments.

## Promotion checklist

A neural runtime lane can be considered for product use only when it has:

1. pinned model/runtime artifacts with manifests and provenance;
2. browser feature detection and clear fallback behavior;
3. eval-level drift gates against native/ORT fixtures;
4. search-level best-move/PV gates at representative visit budgets;
5. cold-start, warm-eval, memory/transfer, and cache measurements;
6. release-hosting policy for generated artifacts and derived models;
7. a rollback path to ORT ONNX WebGPU or ORT WASM.

Related docs: [`lc0_tvmjs_research_runbook.md`](lc0_tvmjs_research_runbook.md),
[`lc0web_custom_inference_checkpoint.md`](lc0web_custom_inference_checkpoint.md),
[`lc0_t3_qdq_webnn_2026-06-10.md`](lc0_t3_qdq_webnn_2026-06-10.md), and
[`onnx_deploy_workflow.md`](onnx_deploy_workflow.md).
