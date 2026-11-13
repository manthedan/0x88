# Big-net fused attention (November 2026)

Feasibility study for candidate #2 of
[`kernel_hillclimb_roofline_2026-11.md`](kernel_hillclimb_roofline_2026-11.md) §5:
one fused WGSL kernel for the 64-token attention core of the big Lc0 nets
(BT4-it332 first, then t3), measured against what ORT 1.27's WebGPU EP runs
today, plus a delivery plan. Lab only: nothing on the product path changed.

**Verdict.**

- **The kernel works.** One f16-storage, f32-accumulate dispatch per layer
  matches the ORT WASM reference to f16 rounding. Max-abs is 2.4e-4 to
  1.95e-3, the same as WASM against a float64 reference. Speed:
  - **On the RTX 3090** (clean, `timingValid`) it is **2.4-2.5× faster** than
    ORT's shipped ops for the same subgraph. It is also 1.3-1.5× faster than
    ORT's own fused path (MultiHeadAttention → FlashAttention), which is less
    accurate.
  - **On the M4** it is 3.6-6.6× faster than FlashAttention and 7-16× faster
    than the shipped ops, in same-page standalone timings (§4).
- **It is also more accurate than what ships.** At BT4 layer 14 the shipped
  WebGPU attention is off by up to 2.7 (M4) / 2.0 (3090) on outputs whose
  magnitude reaches 5.9. The f32 kernel is off by 0.002.
- **Projected gain (not measured end to end).** About **−10% (BT4) / −12%
  (t3) of GPU busy per invoke on the M4**, and about **−6% (BT4) / −8% (t3)
  on the 3090** (§5).
  These come from the paired full-model runs plus same-page kernel
  ratios. The M4 is GPU-bound, so per-move time drops by about the same share.
- **The cheap route does not deliver it.** A graph rewrite to the
  `com.microsoft` MultiHeadAttention contrib op (route a) has a measured
  paired gain:

  | | M4 | 3090 |
  |---|---|---|
  | BT4 | −6.1% GPU busy | −4.1% GPU busy |
  | t3 | −7.6% GPU busy | −4.0% GPU busy |

  - Its attention is still f16, so it is no more accurate than shipped.
  - The gate result is a coin flip on near-tie fixtures:
    - BT4 fails on the M4 and passes on the 3090.
    - t3 passes on the M4 and fails on the 3090, on a different row whose
      reference gap is 0.0103 against a 0.01 epsilon.
  - Wrapping the op in f32 Casts fixes the numerics but costs +5-7% on the M4.
  - The f32 version, with the same numerics as the fused kernel, then fails
    that same BT4 fixture on the 3090. The tie policy needs an owner
    decision before any big-net numerics change ships (§3).
- **Recommendation: route (b)** (§7), a small ORT patch on top of the route (a)
  graph. The MHA rewrite gives ORT one node per layer. The patch adds a
  64-token short-sequence branch to MultiHeadAttention's WebGPU kernel that
  runs this kernel.
- **A per-layer hybrid split (route c) is ruled out by measurement.** Each
  extra `session.run` costs 0.15-0.42 ms of CPU, which is as much as or more
  than the per-layer gain.

Setup:

- **Apple M4** (10-core GPU, `apple / metal-3`), headless Chrome 157.
  - It ran the production `dist-client` plus the lab overlay, served by
    `scripts/serve_isolated_static.mjs`. Every M4 timing ran inside
    `notes/kernel-hillclimb/benchlock.sh`.
  - The Mac was shared. For the first half of the session the load average
    was 300-490, from other projects and other agents' GPU work. For the
    second half it was 13-30; every M4 number in the tables comes from the
    calmer half unless marked. Even then, standalone subgraph timings swing,
    so M4 results are quoted as paired or same-page ratios with spreads.
- **The Mac mini runner was tried and failed.** Its tailnet link dropped the
  rsync of the ~700 MB payload on four submissions, so it produced no data.
- **RTX 3090** (Yukon, Chrome for Testing 151, Vulkan, runner default flags
  including `vulkan_enable_f16_on_nvidia`), through
  `scripts/kernel_hillclimb/remote/run.mjs` and gpu-queue. Every Yukon
  number quoted is `runner.timingValid: true`.

## 1. The attention block as ORT runs it

All three big nets share lc0's exported attention core; only the head count
and model width differ. Per encoder layer (15 layers in each net):

| net | model file | d_model | heads × depth | batch | dtype |
|---|---|---|---|---|---|
| BT4-it332 | `BT4-1024x15x32h-swa-6147500-policytune-332.batch8.f16.qdq8.onnx` | 1024 | 32 × 32 | 8 (fixed in the export) | f16 activations, int8 weight-only QDQ folded to f16 at load |
| t3 | `t3-512x15x16h-distill-swa-2767500.batch8.f16.onnx` | 512 | 16 × 32 | 8 | f16 |
| LQO | `lqo_v2.f16.qdq8.onnx` | 768 | 24 × 32 | 8 | f16, QDQ8 |

The graph, per layer (BT4 shapes; `/encoderN/mha/...` names):

```
ln2 [512,1024] ─ Gemm Q (MatMul+Add fused by ORT) ─ Reshape [8,64,32,32] ─ Transpose 0213 ─ Q [8,32,64,32]
               ─ Gemm K                          ─ Reshape               ─ Transpose 0231 ─ Kᵀ [8,32,32,64]
               ─ Gemm V                          ─ Reshape               ─ Transpose 0213 ─ V [8,32,64,32]
QK/matmul  MatMul(Q, Kᵀ)            [8,32,64,64]
QK/scale   Mul(·, 0.17676 f16)      [8,32,64,64]
smolgen_weights  Add(·, smolgen/out/reshape [8,32,64,64])   ← per-position, per-head bias from the smolgen MLP
QK/softmax Softmax(axis=3)          [8,32,64,64]
QKV/matmul MatMul(P, V)             [8,32,64,32]
out/transpose 0213 → out/reshape [512,1024] → Gemm out/dense
```

So one fused kernel has to read Q, K, V (each `[B·64, H·d]`, the Gemm output
layout, no transposes) and the smolgen bias `[B, H, 64, 64]`, and write O in
the same `[B·64, H·d]` layout. At BT4 b8 that is 4 MB of Q/K/V/O plus 2 MB of
bias per layer, 134 MFLOP of QKᵀ and P·V. The scale is applied after the
QKᵀ dot in the graph; the raw QKᵀ reaches ~49,000 in BT4 layer 14 (|score|
8,712 after scale and bias, from the captured activations), which is within
f16 range but is a reason not to accumulate unscaled dots in f16.

What ORT 1.27 WebGPU dispatches for this today (one layer): 4 Transposes,
2 MatMuls (QKᵀ, P·V), the scale Mul, the bias Add and the Softmax: 9 dispatches,
each a full round trip of a `[8,32,64,64]` or `[8,32,64,32]` f16 tensor
through memory.

## 2. The fused kernel

`lab/kernel-hillclimb/fused-attn-kernels.mjs` generates one WGSL compute
shader per (heads, accumulation, layout). One dispatch per encoder layer
replaces the four Transposes, both MatMuls, the scale Mul, the bias Add and
the Softmax:

```
O[b·64+i, h·d+c] = Σ_j softmax_j(scale · Q_i·K_j + bias[b,h,i,j]) · V[b·64+j, h·d+c]
```

- **IO in the Gemm layout.** Q, K, V and O are read and written as
  `[B·64, H·d]` f16 (`vec4<f16>` loads). The bias is the smolgen output
  `[B, H, 64, 64]`. Nothing is transposed.
- **One workgroup per (batch, head)**: 256 workgroups for BT4 b8, 128 for t3.
  The workgroup stages that head's 64 K rows and 64 V rows in workgroup memory
  (2 × 4 KiB at d = 32).
- **Scores stay in registers.** Each query row's 64 scores are an `f32` array
  in the thread. The kernel does max, `exp` (f32), sum and P·V without writing
  P anywhere.
- **Two accumulation modes.**
  - `acc=f32`: Q·K, the softmax sum and P·V accumulate in f32. The scale is
    applied after the dot, as the ONNX graph does it. This is the numerically
    safe mode.
  - `acc=f16`: dots and P·V accumulate in f16, with Q pre-scaled in f16. This
    is ORT FlashAttention's order, and it exists to measure what f16 costs in
    parity.
- **Two layouts.**
  - `rows=1`: 64 threads, one query row per thread.
  - `rows=2`: 128 threads, two threads per row, each owning 32 keys. The pair
    merges (max, sum, partial O) through workgroup memory. With the reduction
    scratch this needs more than WebGPU's default 16 KiB of workgroup storage,
    so the lab device requests the adapter limit (32 KiB on the M4 and the 3090).

Traffic per dispatch is Q + K + V + O + bias. At BT4 b8 that is 4 × 1 MiB +
2 MiB = 6 MiB, and 134 MFLOP. At the M4's measured ~85 GB/s that sets a floor
of about 74 µs per layer.

The lab page (`lab/kernel-hillclimb/fused-attn.html?net=bt4|t3`) runs four
steps:

1. **ORT WebGPU on the same subgraph.** The page runs two sub-models on ORT
   WebGPU with the product session factory: `attn_orig.onnx`, the exported
   ops, and `attn_mha.onnx` / `attn_mha32.onnx`, the MultiHeadAttention
   rewrites. It reports per-dispatch GPU time in split mode and busy time in
   pass mode.
2. **ORT WASM reference.** It runs `attn_orig.onnx` on ORT WASM, and also
   compares against a float64 numpy reference.
3. **Our kernels.** It runs each kernel variant on its own device and records
   max-abs against WASM and f64 per layer, plus GPU timestamps for 40
   dispatches.
4. **`handoff=1`.** It runs chains of 15 ORT `session.run` calls with
   GPU-resident IO (`Tensor.fromGpuBuffer`, `preferredOutputLocation:
   'gpu-buffer'`), with and without one of our dispatches on ORT's device
   consuming each output. This is the boundary cost of a per-layer hybrid
   split (§6c).

The activations are real, not random.
`scripts/kernel_hillclimb/fused_attn_data.py` taps
`/encoderL/mha/{Q,K,V}/b`, the smolgen bias and `/encoderL/mha/out/reshape`
in the shipped model on ORT CPU. It does this for the first 8 fixture
positions (one physical batch) and layers 0, 7 and 14. It then slices
the attention subgraph out of the model at exactly those tensors.

## 3. Parity of the subgraph

Max-abs against the ORT WASM run of the exported subgraph, at BT4 layers 0, 7
and 14. The rows are taken from the RTX 3090 run (`yukon-fused-attn-bt4.json`).
The M4 numbers are the same to the digit for the f32 rows. The f16 kernel
row shows the range over both devices. For scale: WASM itself is 9.7e-4 / 4.8e-4 / 1.9e-3 from a float64
reference, and the layer outputs reach |2.8| / |1.9| / |5.9|.

| path | L0 | L7 | L14 |
|---|---|---|---|
| **fused kernel, f32 accumulate** (`f32r1`, `f32r2`) | 9.8e-4 | 2.4e-4 | **1.95e-3** |
| fused kernel, f16 accumulate (`f16r1`, `f16r2`) | 3.2-4.6e-2 | 3.9-4.4e-3 | 2.07-2.90 |
| ORT WebGPU, exported ops (**shipped**) | 6.6e-2 (3090) / 7.6e-2 (M4) | 1.1e-2 | **1.97 (3090) / 2.71 (M4)** |
| ORT WebGPU, MHA → FlashAttention (f16) | 4.2e-2 / 5.7e-2 | 5.9e-3 / 5.1e-3 | 2.85 / 1.83 |
| ORT WebGPU, MHA with f32 Casts (`mha32`) | 9.8e-4 | 2.4e-4 | 1.95e-3 |

t3 is benign everywhere: scores stay below |18|, f16 paths are at 0.5-1.4e-2,
and f32 paths at 2.4e-4 to 9.8e-4.

**BT4 layer 14 is where f16 breaks down.** Its raw Q·Kᵀ dots reach ~49,000,
and after scale and bias the scores reach |8,712|. In f16 the softmax inputs
then carry absolute errors of several units, so the shipped path's attention
output there is off by up to 2-2.7 on a 5.9-magnitude tensor. **Any f16
accumulation, ORT's or ours, inherits this.** Only f32 accumulation fixes it.
f32 costs little: f32r2 is 8% slower than f16r2 on the 3090, and on the M4
f16 was not reliably faster at all.

### Page gate

The page's `ok` is a parity gate on the f32-accumulate variants only. Every
output element must satisfy

|kernel − WASM| ≤ 2⁻¹² + 2⁻⁹·|WASM| + |WASM − f64|,

with no non-finite outputs.

- **The relative term** is at least two f16 ULPs at any magnitude. Both sides
  store f16, and the largest f32 differences above are one ULP of the
  reference (2.4e-4 at |0.26|, 9.8e-4 at |1.34|, 1.95e-3 at |2.45|).
- **The absolute term** covers outputs near zero.
- **The last term is the reference's own error** against the float64 result
  on the same inputs. ORT WASM is not exact either: at BT4 layer 14 one
  output is 5.26e-4 in WASM, while float64 and our kernel both give 9.37e-4
  (cancellation). A tolerance on WASM alone failed our kernel on that
  element.

On the M4, every element of both nets sits at 0.33 of its allowance or less,
about 3× headroom. The one exception is that BT4 layer 14 element, at 0.63,
where our kernel is within 7e-7 of float64. A zeroed output fails on every
element with |ref| above about 2.5e-4, and a 1% rescale on every element
above about 0.03. The f16-accumulate variants and the ORT WebGPU sessions are
reported but never gate; the result JSON lists them under `gate.diagnostic`.
A failed gate publishes `ok: false` with `gate.problems`, and `bench.mjs`
exits 1. `debugZeroOutput=1` clears the output buffer instead of running the
kernel for the parity readback. It is a negative check, and the gate must
fail on it.

### End-to-end effect (splice)

The fused kernel cannot run inside an ORT session without route (b) or (c).
The `mha32` rewrite (f32 attention inside ORT) is the closest available
proxy, but it is not the kernel's numerics. Its subgraph max-abs against WASM
matches the f32 kernel's at all three layers on both devices, yet the outputs
differ element by element. ORT's FlashAttention scales Q before the dot product
and normalizes the probabilities before P·V; the fused kernel scales after the
dot product and normalizes after accumulating V. At BT4 layer 14, element
374874, mha32 gives 1.128e-3 and the kernel 9.365e-4 (float64: 9.372e-4).
Since the rows below show that small numerical changes can flip the
full-model gate, the `mha32` results are an approximation of the end-to-end
effect of splicing the kernel in. **The kernel's own full-model parity is
unverified** until it runs inside a session (route (b) or (c)).

| BT4 full model, ORT WASM parity gate (10 fixtures) | device | legal-prob max | policy max-abs | WDL | best move | gate |
|---|---|---|---|---|---|---|
| shipped (exported ops) | M4 | 0.0130 | 0.294 | 0.0029 | 10/10 | pass |
| shipped (exported ops) | 3090 | 0.0147 | 0.211 | 0.0031 | 8/10 (2 tolerated ties) | pass |
| MHA f16 (route a) | M4 | 0.0186 | 0.307 | 0.0044 | 9/10 | **fail**: row 6 top-3 order a7a8q > e1d2 reversed |
| MHA f16 (route a) | 3090 | 0.0109 | 0.240 | 0.0026 | 8/10 (2 tolerated ties) | pass |
| **mha32 (f32 attention, proxy for the kernel)** | M4 | 0.0127 | 0.311 | 0.0026 | 9/10 (1 tolerated tie) | pass |
| **mha32 (f32 attention, proxy for the kernel)** | 3090 | 0.0110 | 0.166 | 0.0028 | 9/10 | **fail**: the same row 6 reversal |

Row 6 (`4k3/P6p/8/8/8/8/7P/4K3 w - - 0 1`) has a flat policy:
- a7a8q ≈ 0.146, then king moves ≈ 0.13.
- The reference gap between a7a8q and e1d2 is 0.012-0.014, just above the
  0.01 tie epsilon.

Whether it flips depends on noise elsewhere in the network, mostly the f16
Gemms, not on attention: the more accurate attention fails on the 3090 and the
less accurate one passes there.

t3 has the same exposure. Every variant passes on the M4 (legal-prob max
0.0035-0.0072). On the 3090, the MHA f16 rewrite fails on row 3: a8a1 vs
h8h1, with a reference gap of 0.0103 against the 0.01 epsilon. Its
legal-prob max there is only 0.0071.

**So the gate has no slack on these two rows for any change to the WebGPU
path.** Both reference gaps are 0.010-0.014, against a 0.01 tie epsilon.
Before shipping anything that touches big-net numerics, the owner has to
choose one of these:
1. widen the tie epsilon to ~0.015 for the big nets;
2. replace those fixtures;
3. judge the big nets by the search-parity protocol instead.

## 4. Timing

### Subgraph, same page (GPU timestamps)

Per encoder layer, b8. For ORT, "busy" is pass-mode GPU busy (the runtime's
own passes); split mode, with one timestamped pass per dispatch, is shown where
it differs. For the fused kernel it is the median per-dispatch GPU time over
40-60 dispatches, with [min-max].

| µs per layer | ORT exported ops | ORT MHA (FlashAttention f16) | ORT mha32 | fused f32r1 | fused f32r2 | fused f16r2 |
|---|---|---|---|---|---|---|
| **BT4, RTX 3090** | 103.4 (split 143.9) | 54.8 | 95.7 | 51.2 [49-52] | **42.0 [41-44]** | 38.9 [37-41] |
| **t3, RTX 3090** | 62.5 (split 101.4) | 38.9 | 67.6 | 31.7 [31-36] | **25.6 [25-27]** | 23.6 [24-25] |
| BT4, M4 (3 loads, load ~13) | 928-2,081 | 830-839 | 2,737-3,833 | **128-141** | 155-220 | 125-143 |
| t3, M4 (3 loads, load ~13) | 1,048-1,053 | 441-445 | 1,515-1,528 | **97-122** | 146-266 | 67-177 |

- **On the 3090, f32r2 is 2.46× (BT4) / 2.44× (t3) faster than ORT's shipped
  ops.** It is also 1.30× / 1.52× faster than ORT's own FlashAttention, which
  is less accurate (§3). BT4 f32r2 moves 6 MiB in 42 µs, about 150 GB/s.
  That is far below the 3090's ~900 GB/s. The kernel is latency-bound: there is
  one (batch, head) per workgroup and a serial 64-key loop per thread. There
  is headroom left, for example 4 threads per row or a K-tile loop.
- **On the M4, rows=1 is best.** The two-thread split loses on Apple. f32r1
  takes 128-141 µs, against 830-840 for FlashAttention and ~0.9-2.1 ms for the
  exported ops.
- **M4 standalone subgraph times are inflated.** These one-layer sessions are
  short GPU bursts, and the M4's clocks do not ramp the way they do in a full
  invoke. In-model, FlashAttention takes 200 µs per layer (3.00 ms / 15 in the
  BT4 MHA profile), not 830. The same-page **ratios** are what carry over:
  - fused f32r1 / (FlashAttention + CopyKVCache) is 0.18-0.22 on BT4 and
    0.22-0.27 on t3;
  - fused / exported ops is 0.06-0.15 (BT4) and 0.09-0.12 (t3).
- **The M4 times swing.** The first pass of this study ran at load 300-490 and
  got per-layer ORT times within 0-30% of these, so the swing is real. Treat
  the M4 rows as ratios with the spreads shown, not as absolute ceilings.

### Full model, paired ABAB (route a, measured)

`ort-*-mha-ab` / `ort-*-mha32-ab` candidates (`scripts/kernel_hillclimb/candidates.mjs`):
the rewritten model (A) against the shipped one (B), interleaved in one page
session. The ratios are A/B, median with [p10-p90].

| run | wall ratio | GPU-busy ratio | B busy per invoke | gate |
|---|---|---|---|---|
| BT4 MHA, M4 (load ~20) | **0.943 [0.902-0.978]** | **0.939 [0.904-0.970]** | 82.8 ms | fail (row 6) |
| BT4 MHA, M4 (load ~400, earlier) | 0.955 [0.933-1.023] | 0.963 [0.927-0.994] | (contended) | fail (row 6) |
| BT4 MHA, 3090 | 0.902 [0.785-1.042] | **0.959 [0.958-0.960]** | 15.0 ms | pass |
| BT4 mha32, M4 | 1.052 [1.017-1.084] | 1.061 [1.036-1.077] | 78.9 ms | pass |
| BT4 mha32, 3090 | 1.011 [0.934-1.059] | 0.990 [0.989-0.991] | 15.0 ms | fail (row 6) |
| t3 MHA, M4 (load ~15) | **0.935 [0.880-0.962]** | **0.924 [0.907-0.990]** | 30.9 ms | pass |
| t3 MHA, M4 (load ~400, earlier) | 0.922 [0.791-1.035] | 0.943 [0.857-1.044] | (contended) | pass |
| t3 mha32, M4 | 1.065 [1.026-1.094] | 1.071 [1.038-1.087] | 30.4 ms | pass |
| t3 MHA, 3090 | 0.893 [0.734-0.996] | **0.960 [0.960-0.961]** | 7.9 ms | **fail**: row 3 best move a8a1 → h8h1, reference gap 0.0103 |

- **The MHA rewrite (f16) is a real gain:** −6 to −8% of GPU busy on the M4
  and −4% on the 3090, with tight paired spreads on the 3090.
- **mha32 is a loss on the M4 (+5-7%) and a wash on the 3090.**
  - The f32 FlashAttention is ~3× slower than the f16 one on Apple, in-model
    (BT4 8.76 vs 3.00 ms per invoke).
  - The 75 Casts per invoke cost another 0.6-1.3 ms.

  So route (a) can buy accuracy or speed, not both.

### Hybrid-split boundary cost (`handoff=1`)

The probe runs a chain of 15 runs of the 2-dispatch MHA sub-model with
GPU-resident IO on ORT's device:

| | M4 (3 loads, BT4 / t3) | RTX 3090 host (BT4 / t3) |
|---|---|---|
| `session.run` CPU time per call | 0.15-0.27 ms | 0.42 / 0.39 ms |
| wall per run, ORT only | 0.28-0.37 / 0.33 ms | 0.45 / 0.42 ms |
| wall per run, + one fused dispatch on `out.gpuBuffer` | 0.37-0.49 / 0.37-0.51 ms | 0.60 / 0.55 ms |

- **Sharing the device works.** `ort.env.webgpu.device` is ORT's device;
  `Tensor.fromGpuBuffer` takes our buffers with no copies, and our pipeline
  consumes ORT's output buffer.
- **The cost is per-call, not per-byte.** Every split point adds a
  `session.run` of ~0.15-0.4 ms CPU, plus ~0.1 ms of submit/sync for the
  custom dispatch. That cost lands on the CPU timeline, which on the 3090 is
  already longer than the GPU timeline (shipped BT4: 19.5 ms wall vs 15.0 ms
  busy per invoke).

## 5. Projected gain per invoke and per move

*Projected, not measured.* The fused kernel has not run inside a full invoke.
Each projection takes the shipped invoke's GPU busy (B arm of the paired
runs) and does two things to it:

1. subtracts the in-model attention time;
2. adds the kernel's time.

The in-model attention time is the MHA arm's in-model FlashAttention +
CopyKVCache, plus the paired saving the MHA arm already measured. The kernel's
time is scaled from the same-page ratio in §4. On the 3090 the standalone and
in-model per-layer times agree (FlashAttention 49.5 µs standalone vs 47-50 µs in-model), so the
standalone kernel time is used directly.

| | BT4 M4 | t3 M4 | BT4 3090 | t3 3090 |
|---|---|---|---|---|
| shipped GPU busy per b8 invoke (B arm) | 82.8 ms | 30.9 ms | 15.0 ms | 7.9 ms |
| measured, MHA rewrite (route a), busy ratio | 0.939 | 0.924 | 0.959 | 0.960 |
| in-model FlashAttention + CopyKVCache (A arm profile) | 3.52 ms (4.3%) | 1.89 ms (6.1%) | 0.88 ms (5.9%) | 0.69 ms (8.7%) |
| implied in-model attention, shipped ops | ~9.6 ms (11.6%) | ~4.1 ms (13%) | ~1.5 ms (10%) | ~1.0 ms (13%) |
| fused kernel, 15 layers | 0.63-0.77 ms (0.18-0.22 × FA+copy) | 0.42-0.51 ms (0.22-0.27 ×) | 0.63 ms (f32r2, 15 × 42 µs) | 0.38 ms (15 × 25.6 µs) |
| **projected busy ratio vs shipped** | **0.893-0.905 (≈ −10%)** | **0.878-0.884 (≈ −12%)** | **0.941-0.943 (≈ −6%)** | **0.921 (≈ −8%)** |
| floor (= measured MHA) / ceiling (attention free) | 0.939 / 0.884-0.896 | 0.924 / 0.863-0.869 | 0.959 / 0.899-0.900 | 0.960 / 0.873 |

Projected ratio = MHA ratio − (FA + copy − kernel) / B. The range comes from
using either the median per-round ratio or the ratio of per-position medians
as the base.

- **Bounds.** The kernel beat FlashAttention in every measurement on both
  devices. So the measured MHA gain is a floor. Deleting attention outright
  is the ceiling.
- **Per move.** Big-net search here is NN-bound at physical batch 8, and on
  the M4 the invoke wall is within 5% of GPU busy (BT4: 80.5 ms wall,
  76.4 ms busy). Per-move time at a fixed visit budget therefore scales
  with invoke time: **~10% (BT4) to ~12% (t3) less time per move on Apple**.
  On the 3090 the wall is longer than the GPU busy time (shipped BT4: 19.5 ms
  wall vs 15.0 ms busy per invoke). The ~0.9 ms of saved GPU time is at most
  ~4.6% of wall, and only as much of it as sits on the critical path will
  show.
- **Per-move example.** A BT4 move at 800 visits is about 100 b8 invokes,
  ~8.3 s on the M4. The projection takes about 0.9 s off that.
- **Not measured.** Two items are outside this study:
  - **LQO.** It is 24 heads × 32, the rewrite (`lqo.qdq8.mha.onnx`) is
    generated, and the candidate `ort-lqo-mha-ab` exists.
  - **The WASM fallback.** Any delivery serves the original model to WASM,
    as split-K does.

## 6. Delivery options

The kernel cannot be dropped into ORT's WebGPU EP as it is: ORT compiles its own
WGSL in C++. Three routes, cheapest first.

### (a) Graph rewrite that steers ORT (MultiHeadAttention / GroupQueryAttention)

**Status: built and measured. It fails the BT4 gate and gains less than the
kernel.**

- **Tooling exists.** `scripts/kernel_hillclimb/mha_rewrite.py` replaces each
  layer's 13 nodes (3 Reshapes, 4 Transposes, 2 MatMuls, Mul, Add, Softmax,
  output Reshape) with Reshape → `com.microsoft.MultiHeadAttention(q, k, v, '',
  '', attention_bias=smolgen, num_heads=H, scale)` → Reshape.
  - It runs `onnx.checker`, and its `--check` mode compares the result with
    the original on ORT CPU.
  - `--compute f32` wraps the op in Casts.
  - It leaves the Q/K/V bias Adds outside MHA. ORT already fuses them into the
    Gemms, and MHA's `bias` input would add TransferBSDToBNSH dispatches.
- **What ORT does with it.** ORT 1.27 routes it to
  `contrib_ops/webgpu/bert/flash_attention.cc`, since head_size % 4 == 0 and
  QKV is not packed. That is 2 dispatches per layer instead of 9: CopyKVCache
  (a copy of K and V into BNSH "present" layout) plus FlashAttention.
- **The FlashAttention kernel is built for LLM decode/prefill, not for 64
  tokens.**
  - Its workgroup is 128 threads on Apple, so half are idle at 64 queries,
    and 64 on NVIDIA.
  - It computes Q·K and P·V in the input type (f16), with Q pre-scaled.
  - It still pays the K/V copy.
- **GroupQueryAttention does not help.** It takes the same FlashAttention
  path with extra rotary and KV-cache plumbing, and needs `seqlens_k` /
  `total_sequence_length` inputs. Its `attention_bias` support is a subset of
  MHA's, and it is built for causal LLM use. It was not tried further.
- **Effort to ship: about 1-2 days**, if the gate passed. The model artifacts
  are already deterministic. The release would follow the split-K precedent
  ([`lc0_small_splitk_2026-11.md`](lc0_small_splitk_2026-11.md) §9): a
  manifest entry, an R2 publish (another ~190 MB for BT4 QDQ8), WebGPU-only
  selection, and the WASM fallback loading the original URL.
- **Risks.**
  - **Parity.** BT4 fails the gate on the M4 today (§3).
  - **Accuracy.** FlashAttention's f16 accumulation leaves the BT4 layer-14
    error at ~1.8-2.9, no better than shipped.
  - **ORT upgrades.** ORT's routing and kernel choice can change under us on
    an ORT upgrade (FlashAttention vs the unfused path; the `max_k_step` and
    workgroup choices are vendor tables in C++).

### (b) Patched ORT-web build

**Recommended.** Keep the route (a) graph and patch ORT so that this one
shape runs the fused kernel.

- **The patch.** In `contrib_ops/webgpu/bert/multihead_attention.cc`, before
  the `CanApplyFlashAttention` branch (line ~117 at v1.27.0), add a
  short-sequence branch. It is taken when all of these hold:
  - no past/present KV;
  - `sequence_length == kv_sequence_length ≤ 64`;
  - `head_size == v_head_size`, with `head_size % 4 == 0` and ≤ 64;
  - Q/K/V in BSD layout;
  - the `attention_bias` shape is `[B, H, S, S]`;
  - the dtype is f16.

  The branch dispatches one program per (batch, head), which is this
  kernel ported to ORT's `ShaderHelper` WGSL template. It reads BSD Q/K/V
  directly (no CopyKVCache) and accumulates in f32. Pick `rows` per vendor:
  `rows=1` on Apple, `rows=2` on NVIDIA. Rows=2 needs the larger
  workgroup-storage limit; ORT requests the adapter's limits at device
  creation, but that has to be checked in the build. That is ~200-300 lines
  of C++ plus a `.wgsl.template`. The numerics are already proven by the lab
  kernel.
- **Build.** `.local_ort/onnxruntime` is checked out at v1.27.0 (8f0278c), and
  `scripts/build_lc0_ort_wasm_relaxed_simd.mjs` already drives `build.sh
  --build_wasm --use_webgpu --enable_wasm_threads` with emsdk 4.0.23. It has
  produced a working asyncify WebGPU build (`.local_ort/build-fixed`, July).
  Production needs both GPU pairs, JSPI and asyncify. The JSPI build is one
  more flag set (`--enable_wasm_jspi`), not yet exercised here.
- **Effort.**
  - About 3-5 days to a gated lab build: patch, two builds, staging into
    `public/ort`, the parity gate on t3, BT4 and LQO, and a re-gate of every
    other ORT net (Lc0 Small, Maia3), because the runtime is shared.
  - Then a release that ships ~40 MB of new ORT binaries.
- **Ongoing cost.** Every ORT upgrade becomes rebase + rebuild + re-gate, and
  the `/ort/` assets stop being the npm package's bytes. That breaks
  `check_ort_wasm_dedup` / provenance assumptions, which would need updating.
- **Risks.**
  - Shipping a forked runtime to all ORT users, including the default Lc0
    Small engine, for a gain only on big nets.
  - The mitigation would be a second ORT copy for the big-net worker only.
    That doubles the download for big-net users and needs care, because
    `ort.env` is per worker.

### (c) Custom runtime or hybrid split

- **Per-layer hybrid split: ruled out by measurement.** Splitting each BT4
  invoke at the 15 attention boundaries means 16 ORT sub-sessions plus 15
  custom dispatches per invoke. The handoff probe (§4) runs ORT with
  GPU-resident IO on ORT's own device (`ort.env.webgpu.device`,
  `Tensor.fromGpuBuffer`, `preferredOutputLocation: 'gpu-buffer'`). It shows
  that each `session.run` costs **0.13-0.19 ms of CPU on the M4 and 0.39-0.42
  ms on the Yukon host**, even for a 2-dispatch graph. Inserting one of our
  dispatches between runs adds 0.07-0.15 ms of wall per boundary on top of the
  kernel itself. For BT4 that is 15 × (~0.2-0.6 ms) = 3-9 ms per invoke of new
  CPU/sync cost, against a GPU saving of ~0.9 ms (3090) to ~13-29 ms (M4).
  - On the 3090 it is a clear loss.
  - On the M4 it might net a little, but only because ORT's attention is
    pathologically slow there.
  - It adds 16× the session-management surface.
- **Whole custom runtime: weeks, and out of scope for one kernel.** The repo has
  two: the lc0web WGSL hybrid evaluator (`src/lc0/hybridEvaluator.ts`, small
  nets) and TVMJS (BT4 runs there at ≈ ORT speed). Moving BT4 off ORT for one
  kernel would mean owning every GEMM too. That is candidate #3's question,
  answered below.

### What candidate #3 (large f16 GEMM retile) would take outside ORT's kernels

The GEMMs are 80-86% of big-net kernel time, so there is no hybrid split
around them: the split would be the whole graph. The options are:

1. **Patch ORT's Gemm/MatMul program** (`core/providers/webgpu/math/gemm_packed.cc`,
   `matmul_packed.cc`). These are the same patched build and the same ongoing
   cost as route (b). The change is a 128×128 (or 64×128) tile path for M ≥ 256
   with f16 storage.
   - **f16 accumulation** is where the ~1.7× ceiling is (roofline §2). It
     needs its own numeric gate: BT4 L14 attention already shows how far f16
     accumulation drifts at these magnitudes.
   - **With f32 accumulation** the untuned tile is only at ORT's level
     (1.29-1.40 vs 1.24-1.28 TFLOP/s).

   So #3 is worth building only if f16 accumulation passes the gate, and this
   study's BT4 near-tie shows the gate has little slack.
2. **A graph rewrite to MatMulNBits** (8-bit, `contrib_ops/webgpu/quantization`).
   The weights are int8 QDQ already. ORT 1.27's WebGPU MatMulNBits has
   wide-tile, DP4A and subgroup-matrix programs, and it would keep weights at
   1 byte. *Not measured.* The DP4A path quantizes activations (a numerics
   change), and the subgroup-matrix path is behind a Chrome flag. It is worth
   one rewrite-and-gate experiment before anyone patches ORT's Gemm.
3. **TVMJS schedules.** This is the existing custom runtime with BT4
   onboarded. The GEMM tile is a schedule change there, not a runtime
   project. It is the cheapest place to prove the f16-accumulate numerics
   end to end, and the obvious host if #2 and #3 are ever delivered together
   outside ORT.

## 7. Recommendation

1. **Do not ship route (a) on its own yet.** It is a real −6 to −8% on the M4
   and −4% on the 3090. But its gate result is a coin flip on near-tie
   fixtures:
   - BT4 row 6 fails on the M4;
   - t3 row 3 fails on the 3090, with a reference gap of 0.0103.

   Its attention numerics are also no better than shipped. If the tie policy
   is relaxed, shipping it follows the split-K release recipe:
   - per-model, WebGPU-only selection;
   - the WASM fallback on the original URL;
   - an R2 publish of ~190 MB (BT4) / ~170 MB (t3) per rewritten model.
2. **Resolve the gate's tie policy first.** Any change to the big nets'
   WebGPU numerics can flip one of these rows, including the more accurate
   one (§3).
3. **Then build route (b): the route (a) graph plus a 64-token branch in
   ORT's MultiHeadAttention.**
   - It delivers the projected ~10-12% (Apple) / ~6% (3090) of GPU time on
     BT4/t3/LQO, with attention more accurate than shipped.
   - Budget 3-5 engineering days to a gated lab build, then a runtime release.
   - Make the call together with the plan for candidate #3. A patched ORT
     is the same commitment for both, and #3 is where the larger gain is.
4. **Rule out route (c) for attention.** Per-call ORT overhead makes the
   per-layer hybrid a net loss on the 3090 and marginal on the M4.

**Verified vs projected.**

- **Verified:**
  - the subgraph parity (M4 and 3090);
  - the subgraph timings (3090 `timingValid`; M4 under benchlock with the
    spreads shown);
  - the route (a) paired full-model ratios and gate results;
  - the handoff costs.
- **Projected:** the end-to-end gain of the fused kernel (§5). The
  end-to-end numerics: `mha32` is only an approximate proxy (different
  scaling and normalization order), so the kernel's full-model parity is
  unverified.
- **Not built:** the ORT patch.

## Reproduce

```sh
# fixture planes (browser encoder), then activations + sub-models (onnx venv, isolated)
node --experimental-strip-types scripts/kernel_hillclimb/dump_planes.mjs \
  --fens fixtures/lc0/fen_only.json --out artifacts/kernel-hillclimb/planes
.venv-onnx/bin/python -I scripts/kernel_hillclimb/mha_rewrite.py $M/BT4-…batch8.f16.qdq8.onnx \
  artifacts/kernel-hillclimb/mha/bt4.b8.qdq8.mha.onnx --check
.venv-onnx/bin/python -I scripts/kernel_hillclimb/mha_rewrite.py $M/BT4-…batch8.f16.qdq8.onnx \
  artifacts/kernel-hillclimb/mha/bt4.b8.qdq8.mha32.onnx --compute f32
.venv-onnx/bin/python -I scripts/kernel_hillclimb/fused_attn_data.py --net bt4 --model $M/BT4-…qdq8.onnx \
  --mha-model artifacts/kernel-hillclimb/mha/bt4.b8.qdq8.mha.onnx \
  --mha32-model artifacts/kernel-hillclimb/mha/bt4.b8.qdq8.mha32.onnx \
  --planes artifacts/kernel-hillclimb/planes --out artifacts/kernel-hillclimb/fused-attn
# (same for t3 with t3-512x15x16h-distill-swa-2767500.batch8.f16.onnx)

BUILD_SCOPE=product npm run build:client && node scripts/kernel_hillclimb/build_lab.mjs
node scripts/serve_isolated_static.mjs dist-client 5291

# M4, always under benchlock
notes/kernel-hillclimb/benchlock.sh fa -- node scripts/kernel_hillclimb/bench.mjs --page fused-attn \
  --query 'net=bt4&handoff=1&iters=60' --repeat 3 --base-url http://127.0.0.1:5291 \
  --browser-args=--enable-unsafe-webgpu,--enable-webgpu-developer-features
# negative check: must print ok:false and exit 1
notes/kernel-hillclimb/benchlock.sh fa -- node scripts/kernel_hillclimb/bench.mjs --page fused-attn \
  --query 'net=bt4&iters=20&debugZeroOutput=1' --base-url http://127.0.0.1:5291 \
  --browser-args=--enable-unsafe-webgpu,--enable-webgpu-developer-features
notes/kernel-hillclimb/benchlock.sh fa -- node scripts/kernel_hillclimb/bench.mjs \
  --candidate ort-bt4-mha-ab --gate --base-url http://127.0.0.1:5291 …   # also -mha32-ab, t3

# RTX 3090 (gpu-queue)
node scripts/kernel_hillclimb/remote/run.mjs --host yukon --dist dist-client --with-models \
  --include 'lab/**,assets/**,ort/ort-wasm-simd-threaded.*,models/fused-attn/**' \
  --page 'lab/kernel-hillclimb/fused-attn.html?net=bt4&handoff=1'
```

Results: `artifacts/kernel-hillclimb/results/{m4,yukon}-fused-attn-*.json`,
`m4-ort-*-mha*-ab*.json`, `yukon-*-mha*-ab.json` (artifacts are gitignored;
this document is the record).

