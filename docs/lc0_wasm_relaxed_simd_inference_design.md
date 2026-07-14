# LC0 WASM Relaxed SIMD inference design

Status: proposed  
Target: browser CPU inference for LC0-family ONNX models  
Baseline: ONNX Runtime Web 1.27 WASM with fixed-width SIMD  
Primary question: can a Relaxed SIMD runtime plus a genuinely integer LC0 graph materially improve non-WebGPU inference without unacceptable model drift or compatibility loss?

## Decision summary

Build this as an isolated research lane with three independently measured changes:

1. a matched ONNX Runtime Web WASM build that emits Relaxed SIMD;
2. a true integer LC0 model whose large matrix multiplications remain quantized during execution;
3. runtime feature detection and fallback from Relaxed SIMD to the current fixed-SIMD ORT WASM artifact.

Do not treat the existing weight-only QDQ model as an integer-inference candidate. It stores large weights as int8, then executes `DequantizeLinear` before the original floating-point `MatMul`; its benefit is artifact size, not integer kernels. It is a required control because it represents the current shipped fallback.

The first promotion decision must separate:

- the effect of recompiling ORT with Relaxed SIMD;
- the effect of changing the graph from floating point to integer execution;
- the effect of ORT thread count;
- artifact size, startup, drift, and search throughput.

The expected high-upside cell is true int8 MatMul on Relaxed SIMD, where
`i32x4.relaxed_dot_i8x16_i7x16_add_s` can map to native integer-dot
instructions. Relaxed SIMD alone is not assumed to produce a meaningful win
for the current f16/QDQ graph.

## Context

The stable LC0 browser path uses `Lc0OnnxEvaluator` and
`src/nn/ortRuntime.ts`. ORT selects WebGPU when available and falls back to
WASM when WebGPU is absent or session initialization fails. The WASM path
already has:

- fixed-width SIMD through the standard `onnxruntime-web` artifact;
- optional pthread execution through `ortThreads=auto` or `ortThreads=N`;
- cross-origin-isolation diagnostics;
- graph optimization level `all`;
- stable LC0 input, output, search, and drift harnesses.

The default small-net model is:

```text
t1-256x10-distilled-swa-2432500.batch1.f16.qdq8.onnx
```

That model cuts the large-weight download approximately in half (40.4 to 20.6
MB). The measured record in `docs/lc0_t3_qdq_webnn_2026-06-10.md` shows it
approximately 15-20% slower than the plain f16 graph in the recorded browser
gate. The "approximately 2x slower on WASM" figure appears only in the comment
in `scripts/lc0_prepare_model_assets.mjs`; no benchmark artifact, thread count,
browser, or host backs it. Treat it as an unreproduced estimate that Phase 0
must measure, and treat per-evaluation dequantization as the hypothesized
rather than profiled cause.

The CPU engine work provides a relevant but not directly transferable
precedent. Reckless, Viridithas, and Berserk obtained substantial fixed-SIMD
gains and additional Relaxed SIMD gains after their packed integer dot-product
paths were identified, range-proven, implemented with explicit relaxed-dot
intrinsics, opcode-inspected, and exact-parity gated. The important lesson is
not that `-mrelaxed-simd` automatically makes numerical code faster. The lesson
is that Relaxed SIMD pays when a proven hot kernel maps to a relaxed operation
that the browser can lower efficiently.

## Current-state evidence

The installed runtime is `onnxruntime-web@1.27.0`. Because the application
imports `onnxruntime-web/webgpu`, the current glue resolves
`ort-wasm-simd-threaded.asyncify.mjs` and
`ort-wasm-simd-threaded.asyncify.wasm` even for a WASM-only session. That
deployed Asyncify artifact was inspected locally:

```text
bytes:              24,254,953
SIMD opcode count:      92,159
Relaxed SIMD count:          0
```

The package also includes plain and JSPI flavors, including a 13,479,978-byte
`ort-wasm-simd-threaded.wasm`, but the current application does not select that
artifact. Asyncify/plain/JSPI flavor must therefore be an explicit experimental
dimension. A custom relaxed build without Asyncify cannot be compared directly
to the deployed Asyncify control.

ORT 1.27 exposes:

```ts
ort.env.wasm.simd = 'relaxed';
```

That setting performs Relaxed SIMD capability validation. It does not rewrite
or replace the loaded WASM module. A Relaxed SIMD-capable runtime artifact must
be built and selected explicitly through `wasmPaths` or `wasmBinary`.

The current quantizer, `scripts/lc0_quantize_onnx_weights_qdq.py`, intentionally
produces storage-only quantization:

```text
int8 weight initializer
  -> DequantizeLinear
  -> f16/f32 tensor
  -> original MatMul
```

Therefore:

- the current graph cannot exercise relaxed int8 dot-product kernels for its
  large MatMuls;
- recompiling ORT may expose relaxed floating-point operations, but this is a
  different and likely smaller opportunity;
- a true integer graph is required to test the same class of optimization that
  succeeded in the NNUE engines.

## Goals

### Functional goals

1. Produce matched fixed-SIMD and Relaxed SIMD ORT WASM artifacts from one
   pinned ONNX Runtime source revision.
2. Produce at least one true integer t1 LC0 candidate whose large MatMuls do
   not dequantize their weights before floating-point execution.
3. Preserve the existing LC0 input contract, policy map, WDL/MLH outputs, and
   evaluator API.
4. Add explicit runtime identity to diagnostics and benchmark artifacts.
5. Feature-detect Relaxed SIMD and fall back to the existing fixed-SIMD WASM
   path when unsupported or when relaxed runtime initialization fails.
6. Measure net-only latency, fixed-visit search, fixed-time search, startup,
   memory, download size, and numerical drift.

### Performance goals

Promotion requires all of the following on at least one representative
non-WebGPU browser class:

- at least 15% lower warm median evaluation latency than the best matched
  fixed-SIMD control at the same thread count;
- at least 10% higher fixed-time LC0 visits per second or completed visits;
- no material regression in p95 evaluation latency;
- no material increase in total first-use bytes unless the speed/strength
  tradeoff is explicitly accepted as a separate runtime configuration.

These are research gates, not predictions. If Relaxed SIMD is neutral but true
int8 is faster, integer inference may still be promoted on fixed SIMD. If true
int8 is neutral but a custom ORT build is faster for unrelated reasons, it
must not be attributed to Relaxed SIMD without a matched compiler control.

### Correctness goals

- ONNX model validation succeeds.
- All existing LC0 fixture inputs evaluate successfully.
- No NaN or infinity appears in policy, WDL, or MLH output.
- Legal-policy top-1 agreement, policy divergence, WDL drift, and search
  outcomes stay within explicit gates.
- Fixed-SIMD and Relaxed SIMD executions of the same integer graph are compared
  separately from floating-point-to-integer model drift.

## Non-goals

- Replacing WebGPU or TVMJS on capable devices.
- Creating a custom end-to-end CPU inference runtime before ORT is measured.
- Promoting WebNN as the universal fallback.
- Assuming exact floating-point parity from relaxed fused operations.
- Quantizing every operator in the LC0 graph in the first experiment.
- Optimizing JS input encoding, legal-move generation, or PUCT before backend
  timing shows they are material in the CPU lane.
- Shipping a Relaxed SIMD-only build without a fixed-SIMD fallback.

## Hypotheses

### H1: a Relaxed SIMD rebuild alone provides limited benefit

The current LC0 WASM graph executes floating-point MatMuls. A compiler may emit
relaxed fused multiply-add operations, but there is no guarantee that the
existing ORT/MLAS kernels are structured to benefit or that the target browser
lowers them better than fixed SIMD. The CPU engine experience showed that
enabling the target feature without changing the hot integer kernel was noisy
or slower.

### H2: true int8 MatMul is the high-upside path

LC0 transformer-style models spend substantial work in large matrix
multiplications. A graph that keeps activations and weights in supported
integer forms can potentially route those operations through ORT/MLAS kernels
that use relaxed 8-bit dot products. This can reduce:

- weight bandwidth;
- arithmetic instruction count;
- widening/shuffle overhead required by fixed SIMD;
- decoded model memory.

The actual benefit depends on ORT operator selection, quantization boundaries,
matrix shapes, browser lowering, and the cost of quantize/dequantize nodes.

### H3: thread count and model export can dominate the SIMD delta

The default thread count is context-dependent in `src/nn/ortRuntime.ts`.
Browser main-thread sessions, Node sessions, and production-built workers
default to one thread. The production-worker default is a workaround for a
bundled pthread-glue boot deadlock, not a performance policy. Development
workers leave ORT thread settings unset and can inherit ORT's multi-threaded
default in a cross-origin-isolated context. Explicit `ortThreads=N` is clamped
to one only on a non-isolated main thread, while `ortThreads=auto` resolves to
`clamp(hardwareConcurrency - 1, 2, 4)` when threading is available. Every
benchmark cell must pin a thread count and record the effective value.

Model dtype also matters: f16 reduces bytes, but browser CPU execution does not
provide the same broadly useful native f16 vector path as WebGPU. The current
evaluator converts input planes and outputs per element when the graph exposes
f16 boundaries. A plain f32 graph may outperform f16 on CPU despite its larger
download. Every Relaxed SIMD result must therefore be reported at matched
thread counts and against both f32 and current deployed controls.

## Proposed architecture

### Runtime artifact ladder

Add an ORT WASM artifact family with explicit identities:

```text
ort-wasm-fixed-simd-threaded
ort-wasm-relaxed-simd-threaded
```

Keep the upstream fixed-SIMD artifact as the stable fallback during research.
The Relaxed SIMD artifact must be built from the same ORT revision and with the
same operator-reduction, threading, asyncify/JSPI, optimization, and exception
settings as its matched fixed-SIMD control.

Before building a custom runtime, benchmark the upstream plain and JSPI
artifacts as explicit CPU-lane candidates. If either can replace Asyncify
safely and improve CPU inference, record that as a separate runtime-flavor win,
not a Relaxed SIMD win.

Selection order for an explicit CPU-optimized runtime:

```text
Relaxed SIMD supported and relaxed artifact available
  -> relaxed artifact
otherwise
  -> fixed-SIMD artifact
```

Selection must happen before the first ORT session is created because ORT
environment flags and WASM paths are process-global. In worker-owned
evaluators, selection belongs in the worker initialization message, not in a
later evaluator method.

Do not silently select the experimental artifact for the existing `auto`
runtime until promotion gates pass. Initial access should be an explicit query
or runtime configuration.

### Feature detection

Extend the existing probes in `src/lc0/wasmFeatures.ts`. Its current Relaxed
SIMD probe validates `f32x4.relaxed_madd`; the LC0 integer lane also needs a
probe containing:

```text
i32x4.relaxed_dot_i8x16_i7x16_add_s
```

The integer-dot probe matches ORT's own `simd='relaxed'` capability test and
the operation of interest. Keep the existing broad relaxed-madd probe available
for CPU-engine diagnostics, but do not create a second disconnected feature
module. Record:

- whether fixed SIMD validates;
- whether Relaxed SIMD validates;
- selected artifact;
- requested and actual thread count;
- cross-origin isolation and `SharedArrayBuffer`;
- initialization failure and fallback reason.

Feature detection is a compatibility gate, not a speed gate. Browsers may
support Relaxed SIMD but lower the relevant instructions poorly.

### Model artifact ladder

The product-control matrix should use one model family and fixed batch 1:

1. `t1.batch1.f32.onnx`
2. `t1.batch1.f16.onnx`
3. `t1.batch1.f16.qdq8.onnx` (current storage-only QDQ control)
4. `t1.batch1.int8-exec.f32io.onnx` (new true integer candidate)

Add a kernel-attribution subset with matched f32 graph boundaries:

1. plain f32 MatMul;
2. f32 storage-only QDQ;
3. true integer execution with f32 input and outputs.

The integer candidate must expose f32 `/input/planes`, policy, WDL, and MLH
boundaries. Quantize/dequantize operations belong inside the graph. This avoids
charging the candidate for the evaluator's per-element JS f32-to-f16 input and
f16-to-JS-number output conversions, and it lets those conversion costs remain
visible as a separate product-control result for existing f16 artifacts.

Batch 1 isolates runtime/kernel effects and matches the stable interactive
fallback. A later phase may add fixed batch 4 and 8 if CPU search fills those
batches often enough to improve end-to-end throughput.

### True integer graph

The first candidate should target only large B-side-initializer MatMuls. Use an
ORT-supported integer form selected by an implementation spike, likely:

- `MatMulInteger` plus explicit scaling and bias handling; or
- `QLinearMatMul` where its input/output quantization contract maps cleanly.

The graph transformation must:

1. identify large MatMul weights using the current threshold as an initial
   control;
2. choose per-tensor or per-channel symmetric int8 weights according to actual
   ORT WASM kernel support;
3. insert activation quantization at boundaries that amortize conversion
   rather than around every small MatMul;
4. keep accumulators in int32;
5. rescale to floating point only where required by unsupported operators or
   output contracts;
6. preserve residual, normalization, softmax, policy, WDL, and MLH semantics;
7. emit a transformation report containing operator counts, quantization
   parameters, bytes, and unsupported fallbacks.

Do not assume per-channel quantization is faster merely because it has lower
weight error. Kernel availability and scale-broadcast overhead must be
confirmed. Start with the simplest representation that reaches the relaxed
integer-dot kernel, then compare accuracy refinements.

### Calibration and activation ranges

`i32x4.relaxed_dot_i8x16_i7x16_add_s(a, b, c)` multiplies a signed-int8
operand `a` against an operand `b` whose lanes must have the most significant
bit clear, meaning values in `[0, 127]`. When a `b` lane has the top bit set,
the result is implementation-defined: x86 and ARM may interpret the lane
differently and produce architecture-dependent results. This is not ordinary
quantization drift.

Whichever tensor ORT routes to the i7 slot, typically activations with
full-range signed-int8 weights in the i8 slot, must therefore quantize into
`[0, 127]`. Symmetric signed-int8 activations do not satisfy this contract, and
full-range uint8 activations satisfy it only if values above 127 are prevented
or the kernel splits the range. The operator spike must confirm which
`MatMulInteger`/`QLinearMatMul` input layout ORT's WASM MLAS maps onto the
relaxed instruction. Unlike the NNUE engines, LC0's activation range is not
statically proven, so the model transformation must not claim exact arithmetic
parity.

Use representative LC0 inputs from:

- the existing FEN-only fixtures;
- explicit-history fixtures;
- opening, middlegame, tactical, and endgame positions;
- positions sampled from actual LC0 search leaves, not random tensors.

Collect per-MatMul activation distributions and saturation rates. The
calibration report should include at least:

```text
tensor name
shape
sample count
min/max
p99.9 absolute value
chosen scale
clipped fraction
zero-point
```

The integer graph should initially use a representation that satisfies the
measured kernel's operand contract, plus a held-out fixture set. Do not default
to symmetric activation scales if activations occupy the i7 slot.
Quantization-aware training is a later option only if post-training
quantization is fast but misses quality gates.

### Runtime configuration identity

Extend the existing runtime-configuration schema with fields equivalent to:

```json
{
  "runtimeBackend": "ort-wasm",
  "runtimeConfigId": "lc0-ort-wasm-relaxed-int8-b1",
  "modelArtifact": {
    "kind": "onnx",
    "precision": "int8-exec",
    "layout": "qlinear-or-matmulinteger"
  },
  "wasm": {
    "simd": "relaxed",
    "artifactId": "ort-wasm-relaxed-simd-threaded",
    "artifactSha256": "...",
    "threads": 1
  },
  "fallback": {
    "enabled": true,
    "fallbackRuntimeBackend": "ort-wasm",
    "fallbackRuntimeConfigId": "lc0-ort-wasm-fixed-simd"
  }
}
```

Record model and runtime artifacts separately. A faster custom ORT build must
not be confused with a faster model export.

## Build design

### Source pinning

Add a lock or manifest containing:

- ONNX Runtime repository URL;
- exact commit SHA and release tag;
- Emscripten SDK version;
- CMake/build arguments;
- operator-reduction configuration;
- expected output names and SHA-256 hashes.

The build must generate both fixed and relaxed artifacts in one invocation or
from one immutable build description. The only intentional code-generation
difference should be the Relaxed SIMD target.

### Compiler configuration

The spike must establish the exact upstream ORT build flag that enables
Relaxed SIMD and verify that it reaches MLAS. At minimum:

```text
fixed control:   -msimd128
relaxed variant: -msimd128 -mrelaxed-simd
```

Do not rely on flags alone. Inspect both artifacts and report opcode counts for:

- fixed SIMD instructions;
- `f32x4.relaxed_madd`/`nmadd`;
- relaxed min/max/select operations;
- `i16x8.relaxed_dot_i8x16_i7x16_s`;
- `i32x4.relaxed_dot_i8x16_i7x16_add_s`.

The relaxed artifact fails the build gate if no relaxed opcode is present. The
integer model/runtime pair fails the kernel gate if benchmark execution does
not reach relaxed integer-dot code, even if such code exists elsewhere in the
binary.

### Operator reduction

The first build may use the full ORT Web artifact to reduce variables. Once
correctness is established, generate a reduced-operator build from all model
controls and the integer candidate. Artifact-size comparisons must distinguish:

- full fixed vs full relaxed runtime;
- reduced fixed vs reduced relaxed runtime;
- model bytes;
- total cold-start transferred bytes.

### Artifact serving

Stage runtime files under a versioned immutable path rather than replacing
`/ort/`:

```text
/runtimes/ort-wasm/lc0-cpu/v1/fixed/
/runtimes/ort-wasm/lc0-cpu/v1/relaxed/
```

Each directory should contain a manifest with identity, byte size, hashes,
source revision, build flags, required features, and matching JS glue. Runtime
and glue files must stay paired.

## Benchmark design

### Factorial controls

The minimum matrix is:

| Dimension | Values |
| --- | --- |
| ORT runtime | upstream fixed, matched custom fixed, matched custom relaxed |
| Model | f32, f16, current QDQ, true int8 |
| Threads | 1, 2, 4 where supported |
| Glue/runtime flavor | Asyncify, plain, JSPI where supported |
| Isolation | non-isolated thread-1, isolated matched-thread runs |
| Host/browser | Apple Silicon Chromium, x86_64 Chromium, Firefox, Safari/WebKit |
| Protocol | net-only, fixed-visit search, fixed-time search |

The matched custom fixed build is mandatory. Comparing only upstream fixed
against custom relaxed cannot isolate Relaxed SIMD from compiler revision,
operator selection, or build-option changes.

Chromium with `ortEp=wasm` is a useful controlled proxy, not sufficient proof
for the population that genuinely lacks WebGPU. At least one Firefox run and
one Safari/WebKit run must exercise capability detection, fixed-SIMD int8
value, and fallback behavior before product promotion. A "representative
non-WebGPU browser class" gate cannot be satisfied only by forcing WASM in
Chromium.

### Warm and cold measurements

Report separately:

- model fetch/decode;
- ORT runtime fetch/compile;
- session creation;
- first evaluation;
- warm evaluation median/p95/p99;
- memory after session creation and after warmup;
- evaluator/search throughput.

Persistent warm measurements determine steady-state speed. Cold metrics protect
against a large runtime or compile penalty that makes an otherwise faster
fallback unsuitable for interactive use.

### Net-only metrics

For each fixture and repeated evaluation:

```text
input build ms
session.run ms
output materialization ms
postprocess ms
total eval ms
positions/s
thread count
physical batch size
```

Run enough warm repetitions to compute robust medians and p95. Alternate or
randomize matched variants to reduce thermal and background bias. Keep browser
cleanup and server headers identical.

Every cell must pass an explicit `ortThreads=N` value and record both requested
and effective thread count after ORT initialization. Do not compare development
worker defaults with production-worker defaults.

### Search metrics

Use both:

- fixed-visit search to verify semantic and scheduling comparability;
- fixed-time search to measure the product benefit of faster inference.

Report:

```text
completed visits
visits/s
neural evaluations
positions/eval
evaluation cache hits
best move
root policy/visit distribution
elapsed time
```

Batch and cache settings must be fixed within each comparison.

## Drift and parity gates

Use the existing real-fixture principle. Random tensors may expose numerical
differences but are not a promotion oracle for quantized LC0.

### Layer 1: runtime parity

Compare fixed-SIMD and Relaxed SIMD using the same floating-point model, then
the same integer model. This isolates relaxed execution semantics.

For true integer MatMul, int32 accumulation should be deterministic for a fixed
quantized graph, but surrounding floating-point operations may still differ if
relaxed fused operations are emitted.

Runtime-parity gates for fixed-SIMD vs Relaxed SIMD using the same graph:

```text
all fixtures finite
best move agreement = 100%, excluding explicitly classified numerical ties
max absolute WDL component difference <= 0.003
fixed-visit search best move agreement = 100%, excluding classified ties
```

### Layer 2: model drift

Compare each integer model to its floating-point source:

- policy top-1 agreement;
- policy top-k overlap;
- maximum and RMS policy-logit difference;
- policy KL or Jensen-Shannon divergence;
- WDL maximum absolute error;
- q-value error;
- MLH error;
- legal-prior top move and probability drift.

Model-drift gates for the integer candidate must be derived on the same fixture
suite from the deployed QDQ control before Phase 2 implementation is accepted.
The candidate may not use looser per-metric thresholds than the deployed QDQ
evidence without an explicit owner decision. Existing evidence includes 8/8
evaluation and 8/8 search fixture agreement for t1 QDQ with reported maximum
drift 0.021, and stricter t3 evidence with maximum prior difference 0.0044.
Record exact metric names rather than treating these historical values as
interchangeable. Passing net-only gates is necessary but not sufficient.

### Layer 3: search drift

On the fixed fixture suite:

- compare best move;
- compare top root moves and visit shares;
- compare q and WDL;
- compare completed visits under fixed time;
- run a small matched arena only after net and fixed-search gates pass.

Search disagreement is not automatically a failure when the quantized model is
within accepted numerical tolerance, but concentrated tactical regressions are.

## Rollout plan

### Phase 0: reproducible baseline

1. Reuse `ortEp=wasm`, `collectOrtRuntimeDiagnostics()`, and the existing
   browser benchmark machinery; extend their artifact fields rather than
   creating a parallel CPU runtime path.
2. Record the actually selected Asyncify artifact identity and opcode counts.
3. Benchmark the upstream Asyncify, plain, and JSPI flavors with matched model,
   thread, and glue settings.
4. Benchmark f32, f16, and current QDQ at an explicit one thread.
5. Repeat in an isolated context at explicit two and four threads.
6. Measure the previously unreproduced QDQ WASM slowdown estimate.

Exit condition: stable baseline artifacts on Apple Silicon Chromium and one
x86_64 Chromium host.

### Phase 1: matched Relaxed SIMD ORT build

1. Pin ORT and Emscripten revisions.
2. Build custom fixed and relaxed artifacts.
3. Stage versioned manifests.
4. Add explicit runtime selection and fallback.
5. Inspect opcodes.
6. Benchmark the unchanged f32/f16/QDQ models.

Exit condition: either a reproducible runtime-only win is demonstrated or H1
is closed as neutral/negative.

### Phase 2: true integer t1 graph

1. Profile operator and MatMul shapes.
2. Implement the simplest ORT-supported integer transformation.
3. Calibrate on real positions and report saturation.
4. Validate ORT fixed-SIMD execution.
5. Validate ORT Relaxed SIMD execution.
6. Run the full factorial benchmark and drift gates.

Exit condition: determine whether true int8 is both materially faster and
quality-safe, and whether Relaxed SIMD adds value beyond fixed SIMD.

### Phase 3: search productization

If Phase 2 passes:

1. add adaptive backend/model selection;
2. keep the current stable model as fallback;
3. add runtime-audit events and user-visible diagnostics;
4. run fixed-time arena comparisons;
5. extend to t3 only after t1 provides a positive CPU Pareto result.

### Phase 4: optional deeper work

Only after the ORT lane is measured:

- quantization-aware training;
- custom MLAS microkernel changes;
- XNNPACK-backed WASM experiments;
- mixed int8/f32 graph partitioning;
- fixed-batch CPU search tuning;
- reduced-operator runtime builds.

## Runtime fallback policy

Initialization behavior:

```text
request relaxed CPU runtime
  -> relaxed feature probe
  -> fetch/verify relaxed manifest and artifacts
  -> initialize ORT
  -> create session
on any failure
  -> record fallback reason
  -> initialize fixed-SIMD ORT in a fresh worker
```

Because ORT initialization is global and failed initialization can poison a
worker runtime, fallback should use a new worker rather than attempting to
mutate WASM paths after initialization.

Evaluation-time failures should permanently demote that evaluator instance to
the fixed-SIMD runtime, following the runtime-fallback pattern already used by
the Tiny TVMJS lane. Do not retry the failing relaxed runtime for every move.

## Risks and mitigations

### Relaxed feature support is not performance support

Mitigation: maintain per-architecture evidence and never promote from feature
validation alone.

### Current quantization does not reach integer kernels

Mitigation: inspect the transformed graph and profile actual operator/kernel
selection. Keep current QDQ only as a storage-oriented control.

### The relaxed i8/i7 dot contract may not fit activations

Mitigation: calibrate, report clipping, and use a graph/kernel representation
whose operand order and ranges are explicit. The i7-side lanes must be in
`[0, 127]`; out-of-range behavior can diverge across architectures. Do not copy
the NNUE exactness argument to LC0.

### Floating-point relaxed operations can alter search

Mitigation: isolate runtime parity from model drift and run fixed-search gates.
If integer-dot gains are valuable but relaxed float ops cause drift, build or
patch the runtime so only the proven integer kernels use Relaxed SIMD.

### Threading hides or reverses the single-thread result

Mitigation: test all runtime/model comparisons at matched thread counts and
report scaling efficiency.

### Build differences invalidate attribution

Mitigation: require a custom fixed-SIMD control from the same source, toolchain,
and build configuration.

### Cold-start cost outweighs warm speed

Mitigation: include runtime/model bytes, compile, session creation, and first
evaluation in promotion evidence.

### Quantized operator coverage is incomplete

Mitigation: begin with a partial graph and record every fallback boundary.
Reject transformations whose conversion overhead consumes the kernel win.

### Browser regressions

Mitigation: keep fixed-SIMD fallback, immutable versioned artifacts, and a
versioned manifest field that can disable relaxed selection before default
promotion. The research phase does not assume a pre-existing remote
configuration service.

## Rejected shortcuts

### Set `ort.env.wasm.simd = 'relaxed'` and benchmark

Rejected because the flag only checks feature availability. The current binary
contains no relaxed instructions.

### Recompile ORT and use the existing QDQ model as the int8 test

Rejected because the current graph dequantizes weights before floating-point
MatMul.

### Compare upstream fixed ORT against a custom relaxed build

Rejected because the comparison confounds source revision, compiler, build
flags, and operator selection.

### Promote from a microkernel benchmark

Rejected because LC0 product value depends on graph execution, quantization
boundaries, output drift, and search throughput.

### Replace ORT with custom CPU kernels immediately

Rejected because ORT already supplies graph execution, operator coverage,
threading, and fallback. Custom kernels are justified only after profiling
shows a specific ORT kernel is the limiting factor.

## Deliverables

1. ORT source/build lock and reproducible fixed/relaxed build command.
2. Versioned fixed and relaxed runtime manifests.
3. Extended `src/lc0/wasmFeatures.ts` probes and explicit runtime selector.
4. Extended SIMD opcode inspector suitable for ORT artifacts.
5. True integer t1 model transformer and calibration report.
6. CPU inference benchmark runner using the standard artifact schema.
7. Net-only, fixed-visit, and fixed-time benchmark artifacts.
8. Drift and promotion assessment.
9. A final recommendation choosing one of:
   - retain upstream fixed-SIMD ORT;
   - promote a custom fixed-SIMD integer lane;
   - promote Relaxed SIMD for selected devices;
   - discontinue the experiment.

## Open questions

1. Which ORT integer operator form reaches the best WASM MLAS kernel in the
   pinned release?
2. Does upstream Relaxed SIMD support apply only to quantized kernels, or does
   the selected build also introduce relaxed floating-point operations?
3. Can the runtime build restrict relaxed code generation to integer kernels
   if floating-point drift or performance is unfavorable?
4. Does per-channel quantization reach an optimized kernel, or should the first
   candidate use per-tensor scales?
5. Where should activation quantization boundaries sit to avoid repeated
   conversion around residual and normalization blocks?
6. Does CPU search benefit from fixed batch 4/8 after the batch-1 runtime is
   optimized?
7. Is a reduced ORT operator build necessary for cold-start viability?

## Promotion rule

Relaxed SIMD is promoted only when the complete runtime configuration is a
better browser CPU Pareto point than the best fixed-SIMD alternative:

```text
speed + fixed-time search value + correctness + bytes + startup + compatibility
```

The project should promote true integer inference without Relaxed SIMD if that
is the winning configuration. Likewise, it should retain floating-point ORT if
quantization or relaxed execution fails the quality or product gates. The
research goal is a faster non-WebGPU LC0 path, not Relaxed SIMD adoption for
its own sake.
