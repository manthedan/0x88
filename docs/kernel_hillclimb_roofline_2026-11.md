# Kernel hill-climb, phase 0: roofline of the shipped GPU paths (November 2026)

Where every shipped WebGPU path stands against the Apple M4 peak, measured
from the production build. Also: the harness that later kernel candidates are
judged by, and the ranked list of what is worth hand-writing. No optimized
kernels were written in this phase.

All numbers come from an Apple M4 (10-core GPU, `apple / metal-3`, subgroup
size 32/32), headless Chrome 157, and the production `dist-client` with the lab
overlay, served by `scripts/serve_isolated_static.mjs` (COOP/COEP, port 5241).
Every timing ran inside `notes/kernel-hillclimb/benchlock.sh`. The machine is
shared: the load average was 7-15 for the whole session (other agents'
CPU work, which benchlock cannot serialize), so each number is a median of 3
page loads × 30 invokes, and the uncertainty is stated wherever it matters.
The harness and its result contract are documented in
[`scripts/kernel_hillclimb/README.md`](../scripts/kernel_hillclimb/README.md).

## 1. What users actually run

These are the GPU paths in the v0 deploy profile (`VITE_BROWSER_CHESS_DEPLOY_PROFILE=v0`).
In v0, `normalizeLc0Runtime` forces `onnx`, so every Lc0 net goes through ORT-web
1.27 using its native C++ WebGPU EP (emdawnwebgpu). For QDQ models, the dequant
fold (`src/nn/onnxDequantFold.ts`, default on) turns the int8 weights into f16
initializers at load time, so no `DequantizeLinear` runs per invoke.

| engine | runtime | model, precision | physical batch | where |
|---|---|---|---|---|
| Lc0 Small (t1-256x10) | ORT WebGPU | `batch1.f16.qdq8`, f16 compute, folded weights | **1** (search `batchSize` 1) | Play / Analysis / Arena default |
| Lc0 t3-512x15x16h | ORT WebGPU | `batch8.f16` | 8 | Analysis / Arena (not a v0 Play option) |
| Lc0 BT4-1024x15x32h-it332 | ORT WebGPU | `batch8.f16.qdq8`, f16 | 8 | Play / Analysis / Arena |
| Leela Queen Odds (lqo_v2) | ORT WebGPU | `f16.qdq8`, f16 | 8 | Play |
| Maia3 | ORT WebGPU | `maia3_simplified.qdq8`, f32 graph with folded weights | 1 per move; 11 for the Elo-grid `evaluateConditions` | Play human opponents, Elo inference |
| Centipawn (bt4_soap_rem_c19000) | TVMJS WebGPU | `f32/v2-shape-k16` | 16 | Centipawn engine default (ORT fallback) |

Not shipped on the GPU in v0: the Lc0 TVMJS lanes (`lc0-tvmjs-webgpu` t1/t3/BT4,
hidden by the forced `onnx`), the hybrid custom-WGSL heads runtime, and Centipawn
`v3-subgroups` (staged, not promoted). Those were measured as research lanes only.

## 2. M4 peaks in Chrome WebGPU

Source: `lab/kernel-hillclimb/peak.html`. Three dev-flag page loads plus two with
default flags (`results/peak-dev*.json`, `peak-defaultflags.json`). Values are
medians, with the min-max across loads in brackets.

| measure | value |
|---|---|
| f32 FMA (8 independent vec4 chains) | **3.22 TFLOP/s** [3.21-3.24]; best single trial 3.47 |
| f16 FMA | **2.72 TFLOP/s** [2.45-2.83]. On M4, f16 brings no ALU-rate gain, only half the bytes |
| read bandwidth (256 MiB, vec4) | **93 GB/s** [73-100] |
| copy, kernel / `copyBufferToBuffer` | 85 GB/s [80-92] / ~83-90 GB/s (wall) |
| ridge point (FMA / read BW) | ~35 FLOP/byte |
| empty dispatch, GPU time | 0.80-0.86 µs (same pass or own pass) |
| own-pass dispatch, wall | 3.3-6.2 µs |
| CPU encode | 0.6-0.8 µs per pass+dispatch |
| submit round trip / dispatch+copy+`mapAsync` readback | 0.23-0.30 ms / 0.28-0.33 ms |
| naive tiled WGSL GEMM, 2048³ | 1.4-1.7 TFLOP/s (f16 storage + f16 accumulate is fastest) |
| `chromium-experimental-subgroup-matrix` GEMM, 2048³ | **2.90 TFLOP/s** f16, 2.42 f32 (8×8×8 configs; 4×4 tiles per subgroup) |

Adapter features with default flags: `shader-f16`, `subgroups`,
`subgroup-size-control`, `timestamp-query`. The adapter `description` is empty
and timestamps are quantized to 65.536 µs.

The `chromium-experimental-subgroup-matrix` feature (f32→f32 and f16→f16, 8×8×8)
only appears with `--enable-unsafe-webgpu`. It is **not available to users**.
Unquantized timestamps need `--enable-webgpu-developer-features`.

GEMM at real model shapes. Each cell is the best kernel per page load, then the
max across loads. "naive" means my 64×64-128×128 register-blocked tile, with no
tuning:

| shape (M×N×K) | naive f32 | naive f16/f16-acc | subgroup-matrix f16 | shipped kernel achieves |
|---|---|---|---|---|
| t1 b1 64×256×256 | 0.32 | 0.47 | 0.85 | ORT 0.22 |
| t1 b16 1024×256×256 | 1.21 | 2.00 | 2.27 | - |
| t1 b16 1024×1024×256 | 1.38 | 2.17 | 2.65 | - |
| Centipawn 1024×384×128 (f32 path) | 0.87 | 1.41 | 1.94 (f32 1.94) | TVMJS 0.27-0.29 (f32) |
| BT4 b8 512×1024×1024 | 0.94 | 1.46 | 2.17 | ORT 1.24 |
| BT4 b8 512×1536×1024 | 1.08 | 1.75 | 1.95 | ORT 1.28 |
| BT4 b8 512×1024×1536 | 0.89 | 1.47 | 1.74 (f32) | ORT 1.24 |

## 3. Roofline per shipped path

Method:

- `lab/kernel-hillclimb/gpu-profiler.mjs` patches `requestDevice` before the
  runtime boots, so it works for ORT's C++ EP and TVMJS alike.
  - **Split mode** gives one timestamped pass per dispatch. This is the
    attribution basis.
  - **Pass mode** times the runtime's own passes without splitting. This is the
    true busy/idle structure.
- FLOPs and bytes come from `scripts/kernel_hillclimb/onnx_cost.py`, which
  computes per-node cost at the real batch.
  - ORT pipelines are named only by op type, so `roofline.mjs` attaches each
    matmul dispatch to its ONNX node by bound bytes.
  - TVMJS kernels take their MACs from the shape-k16 probe JSON.
- Bytes are the sizes of the bound buffers. That is an upper bound on
  compulsory traffic, and there are no DRAM counters.
- "Bound" labels per kernel group:
  - `launch`: median dispatch under 8 µs.
  - `compute` / `bandwidth`: at least 50% of the attainable roof.
  - `underused`: neither roof is reached, so the limit is occupancy or kernel
    quality.

Full output: `node scripts/kernel_hillclimb/roofline.mjs --md ...` (the artifacts
are local and gitignored).

### Per invoke

| path | b | wall ms | GPU kernel ms | GPU share of wall¹ | non-kernel ms¹ | dispatches / passes / submits | GFLOP | TFLOP/s at wall | % FMA peak | matmul TFLOP/s | kernel time by bound² |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Lc0 Small** (ORT) | 1 | **9.9** [8.9-10.1] | 8.5 | 77% | 2.3 | 356 / 23 / 26 | 1.29 | 0.13 | **4.0%** | 0.17 | underused 86%, launch 10% |
| Lc0 Small at b8 (what-if) | 8 | 16.1 [14.2-16.3] | 13.6 | 86% | 2.1 | 356 / 23 / 26 | 10.3 | 0.64 | 20% | 0.93 | underused 91% |
| **t3** (ORT) | 8 | **47.9** [43.8-51.1] | 60.2³ | 92% | 4.9 | 537 / 34 / 37 | 40.9 | 0.85 | 26% | 0.82 | underused 92%, bw 8% |
| **BT4** (ORT) | 8 | **166.5** [162.6-172.4] | 138.2 | 95% | 6.3 | 537 / 34 / 37 | 132.1 | 0.79 | 25% | 1.11 | underused 94%, bw 4% |
| **LQO** (ORT) | 8 | **84.8** [83.9-89.5] | 75.6 | 95% | 4.0 | 527 / 33 / 36 | 71.2 | 0.84 | 26% | 1.11 | underused ~85% |
| **Maia3** move (ORT) | 1 | **11.5** [8.6-12.9] | 8.4 | **60%** | **4.9** (2.9 idle inside GPU envelope) | 817 / 53 / 59 | 2.48 | 0.22 | 6.7% | 0.42 | underused 75%, **launch 25%** |
| **Maia3** Elo grid (ORT) | 11 | **53.0** [52.5-53.6] | 60.6³ | 91% | 5.2 | 817 / 53 / 59 | 27.3 | 0.52 | 16% | 0.57 | underused 88% |
| **Centipawn** (TVMJS v2) | 16 | **11.1** [10.2-17.3] | 9.6 | 86% | 1.5 | 161 / 161 / 3 | 2.03 | 0.18 | **5.7%** | 0.33 | underused 79%, bw 21% |
| Centipawn v3-subgroups | 16 | 10.1 | 9.3 | 92% | 0.8 | 161 / 161 / 3 | 2.03 | 0.20 | 6.2% | 0.32 | - |
| TVMJS Lc0 t1 b8 / b16 | 8/16 | 22.8 / 24.9 | 16.1 / 17.0 | 71% / 69% | 6.7 / 7.8 | 229 | 10.3 / 20.7 | 0.45 / 0.83 | 14% / 26% | 0.72 / 1.37 | underused |
| TVMJS Lc0 BT4 b8 | 8 | 198.7 | 155.4 | 78% | 43.3 | 341 | 132.1 | 0.67 | 21% | 0.90 | underused |

¹ ORT: pass-mode busy and wall over the same invokes. TVMJS puts one dispatch in each pass, and per-pass timestamps add about 25-50 µs of wall each on Apple, so TVMJS uses the unprofiled wall minus the split kernel sum.
² Shares of the kernel sum.
³ The profiled phase ran slower than the unprofiled timing loop that preceded it, by up to 25%. The cause is GPU clock state or contention on the shared machine. Read the kernel columns as shares, not as additive with wall.

### Per search (extrapolated)

Time per 1,000 NN evaluations (= wall × 1000 / batch), assuming full batches,
no cache hits, and no CPU tree time:

| path | ms per 1k evals | evals/s |
|---|---|---|
| Lc0 Small | 9,940 | ~100 |
| t3 | 5,990 | ~167 |
| BT4 | 20,810 | ~48 |
| LQO | 10,600 | ~94 |
| Centipawn | 690 | ~1,440 |

Maia3 costs 11.5 ms per move, plus 53 ms per Elo-grid call.

### Top kernel groups (shipped paths)

**Lc0 Small, b1** (kernel sum 8.45 ms)

| kernel | grid | n | µs | ms/inv | M×N×K | GFLOP/s | GB/s | %FMA | %BW | AI | bound |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Gemm (smolgen compress) | 8x1x1 | 10 | 232 | 2.32 | 1×256×2048 | 5 | 4.5 | 0.1 | 5 | 1 | underused |
| Gemm (q/k/v/out) | 8x2x1 | 43 | 38 | 1.63 | 64×256×256 | 221 | 5.2 | 6.8 | 6 | 43 | underused |
| Gemm (FFN down) | 8x2x1 | 10 | 134 | 1.34 | 64×256×1024 | 251 | 5.1 | 7.8 | 6 | 49 | underused |
| Gemm (FFN up) | 32x2x1 | 10 | 42 | 0.42 | 64×1024×256 | 803 | 16.5 | 24.9 | 18 | 49 | underused |
| Gemm (smolgen expand) | 64x1x1 | 10 | 41 | 0.41 | 1×2048×256 | 26 | 25.8 | 0.8 | 28 | 1 | underused |
| MatMul (smolgen per-head) | 1x2x1 | 10 | 36 | 0.36 | 64×32×256 | 29 | 1.5 | 0.9 | 2 | 20 | underused |
| MatMul (smolgen weight gen) | 128x1x1 | 10 | 36 | 0.36 | 8×4096×256 | 471 | 60.8 | 14.6 | 65 | 8 | bandwidth |
| Gemm (head) | 4x1x1 | 1 | 233 | 0.23 | 1×128×2048 | 2 | 2.3 | 0.1 | 2 | 1 | underused |

FFN up and FFN down have the same FLOPs. Down runs on 16 workgroups at 134 µs;
up runs on 64 workgroups at 42 µs. ORT's Gemm does not split K, so at M ≤ 64 the
grid starves a 10-core GPU. The M=1 GEMVs run on 4-8 workgroups and reach
2-5 GB/s.

**BT4, b8** (kernel sum 138 ms). t3 and LQO show the same pattern at d=512 / 768.

| kernel | grid | n | µs | ms/inv | M×N×K | GFLOP/s | GB/s | %FMA | AI | bound |
|---|---|---|---|---|---|---|---|---|---|---|
| Gemm (q/k/v/out) | 32x16x1 | 63 | 870 | 54.8 | 512×1024×1024 | 1235 | 4.8 | 38.3 | 256 | underused |
| Gemm (FFN down) | 32x16x1 | 16 | 1294 | 20.7 | 512×1024×1536 | 1244 | 4.9 | 38.6 | 256 | underused |
| Gemm (FFN up) | 48x16x1 | 16 | 1259 | 20.1 | 512×1536×1024 | 1279 | 5.0 | 39.7 | 256 | underused |
| MatMul (smolgen weight gen) | 128x8x1 | 15 | 438 | 6.6 | 32×4096×256 ×8 | 1227 | 9.9 | 38.0 | 124 | underused |
| Softmax | 16384x1x1 | 15 | 362 | 5.4 | - | - | 11.6 | - | - | underused |
| Transpose | 8192x1x1 | 61 | 83 | 5.1 | - | - | 25.2 | - | - | underused |
| Gemm (smolgen compress) | 8x1x1 | 15 | 333 | 5.0 | 8×256×2048 | 25 | 3.3 | 0.8 | 8 | underused |
| Gemm (smolgen expand) | 256x1x1 | 15 | 201 | 3.0 | 8×8192×256 | 167 | 21.7 | 5.2 | 8 | underused |

**Maia3** (kernel sum: 8.35 ms at b1, 60.6 ms at b11)

| kernel | b | grid | n | µs | ms/inv | op | GFLOP/s | GB/s | bound |
|---|---|---|---|---|---|---|---|---|---|
| Einsum `bhi,oi->bho` (attention-bias projection) | 11 | 11264x1x1 | 8 | 2402 | **19.2** | 176×4096×128 | 77 | 1.3 | underused |
| Einsum | 1 | 1024x1x1 | 8 | 148 | 1.18 | 16×4096×128 | 114 | 8.0 | underused |
| Gemm | 1 | 4x1x1 | 8 | 185 | 1.48 | 1×128×2048 | 3 | 2.9 | underused |
| Add (shape arithmetic) | 1 | 1x1x1 | 256 | 2 | 0.57 | - | - | - | launch |
| Gather / Concat / Transpose (shape + layout) | 1 | 1x1x1-512x1x1 | 148 | 2-6 | 0.58 | - | - | - | launch |

**Centipawn TVMJS v2, b16** (kernel sum 9.6 ms; f32)

| kernel | grid | n | µs | ms/inv | GFLOP/s | GB/s | %FMA | bound |
|---|---|---|---|---|---|---|---|---|
| softmax | 8192x1x1 | 6 | 298 | 1.79 | - | 21.1 | - | underused |
| fused_matmul3_add4 | 32x12x1 | 5 | 353 | 1.76 | 285 | 8.0 | 8.9 | underused |
| fused_matmul8_add2_add1 | 32x4x1 | 6 | 252 | 1.51 | 266 | 13.0 | 8.3 | underused |
| fused_matmul7_…_erf_… (FFN + GELU) | 32x8x1 | 6 | 246 | 1.48 | 272 | 11.2 | 8.4 | underused |
| matmul5 | 2x128x1 | 6 | 146 | 0.88 | 115 | 32.3 | 3.6 | underused |
| fused_matmul4_add5 | 2x256x1 | 6 | 119 | 0.71 | 141 | 75.0 | 4.4 | bandwidth |
| layer_norm1 | 1024x1x1 | 12 | 52 | 0.62 | - | 20.4 | - | underused |

## 4. Verdict per path

The common finding: **no shipped kernel sits on a roof.** No path reaches the
compute roof or the bandwidth roof. Time is dominated by kernels that are
occupancy- or quality-limited ("underused"), plus launch-bound shape arithmetic
in Maia3. The GPU is busy for 77-95% of each ORT invoke, so the overall
classification is:

| path | classification | headroom |
|---|---|---|
| Lc0 Small b1 | **Kernel quality (small-M GEMM occupancy)**, then CPU/sync. GPU 77% of wall: 86% of that is underused GEMMs, 10% launch-bound; non-kernel 23% (2.3 ms submit/readback/JS) | Large. Weights are 40-59 MB per invoke, so the bandwidth floor is ~0.6 ms against 7.4 ms of matmul time today |
| t3 / BT4 / LQO b8 | **Compute-side kernel quality.** The GEMMs reach 35-40% of FMA peak, at AI 170-256 (well past the ridge). Softmax, transpose and the smolgen GEMV take another 10-15% | Moderate: 1.3-1.6× on GEMMs with plain WGSL; ~2× only with subgroup-matrix (flag-only) |
| Maia3 b1 | **Launch/sync-bound + small-M.** 817 dispatches; 25% of kernel time is launch-bound shape ops; 40% of wall is non-kernel, including 0.8-2.9 ms GPU idle inside the envelope | Large, and mostly reachable with graph fixes |
| Maia3 grid b11 | **One pathological kernel.** The Einsum takes 32% of kernel time at 77 GFLOP/s | ~19 ms per call, from a graph rewrite |
| Centipawn (TVMJS) | **Kernel quality (f32 matmul schedule) + softmax.** Matmuls reach 8-9% of FMA; 1,440 evals/s already | ~2× GPU time. Delivered through TVM schedule work, not ORT |
| TVMJS Lc0 lanes | Slower than ORT today (t1 b8 22.8 vs 16.1 ms; BT4 199 vs 167 ms) and carry 7-43 ms of CPU decode | Not a reason to switch runtimes |

Centipawn v3-subgroups cuts the kernel sum by 3% (9.31 vs 9.59 ms), matching the
4-6% busy saving in `docs/centipawn_tvmjs_subgroups_2026-10.md`. Its wall
difference is inside the noise.

## 5. Ranked kernel candidates

Constraints on delivery:

- ORT's WebGPU EP compiles its own WGSL in C++. A hand-written kernel cannot be
  dropped into the shipped Lc0/Maia3 path without one of:
  1. an ONNX graph rewrite that steers ORT to a better kernel;
  2. a patched ORT build;
  3. a custom runtime for the op (the hybrid-heads pattern).
- Option 1 costs least and is listed first wherever it applies.
- The ceilings below are measured device numbers from §2, not projections from
  vendor specs.

| # | candidate | targets (ms now) | expected ceiling | gate |
|---|---|---|---|---|
| 1 | **Small-M GEMM / GEMV with split-K and wide grids** (M ≤ 64 token rows, and M = batch for smolgen compress/expand and heads; f16 weights read as vec4/vec8; ≥ 128 workgroups) | Lc0 Small b1: all Gemm/MatMul, 7.4 ms/invoke. Smolgen compress in t3/BT4/LQO: 4.5-5.1 ms each. Maia3 1×128×2048 Gemm: 1.5 ms (b1) / 2.5 ms (b11) | Weight streaming at ~85 GB/s, plus about 1 µs per dispatch. Lc0 Small matmul time 7.4 → ~1.0-1.5 ms, so the invoke goes 9.9 → **~3.5-4 ms (2.5-2.8×)**. Each smolgen GEMV 300 µs → ~15 µs (−5 ms BT4/t3, −4.5 ms LQO) | `ort-lc0-small --gate` (WASM parity), then a search A/B. Try first: an ONNX split-K rewrite (Reshape → batched MatMul → ReduceSum), which needs no custom kernel |
| 2 | **Fused 64-token attention** (QKᵀ + smolgen bias add + softmax + ·V, scores in workgroup memory; removes the Transpose round trips) | BT4: Softmax 5.4 + Transpose 5.1 + QK/AV MatMuls ~2.8 + bias Add ~2.1 ≈ 15 ms (11%). t3 ≈ 8 ms (13%). LQO ≈ 10 ms | Q, K, V and bias are read once at ~85 GB/s, ~1-1.5 ms per BT4 invoke. **−10-12% invoke** on t3/BT4/LQO | WASM parity, plus `tieEpsilon` search parity (f16 softmax order changes) |
| 3 | **Large f16 GEMM retile** (512×{768,1024,1536}×K; 128×128 tiles, f16 storage, f16 or f32 accumulate decided by parity) | 80-86% of t3/BT4/LQO kernel time; 1.13-1.28 TFLOP/s today | Plain WGSL: my untuned tile already reaches 1.46-1.75 at these shapes; ~1.8-2.0 is plausible. **1.3-1.6× on GEMMs → BT4 ~166 → ~115-125 ms.** Subgroup-matrix would reach 1.95-2.2 (≈ 1.7×) but is flag-only | WASM parity at BT4's policy threshold; tie-epsilon search parity. f16 accumulate needs its own numeric gate. Highest engineering cost: replacing ORT's Gemm means a patched ORT or a custom runtime |
| 4 | Centipawn f32 matmul + softmax schedule (TVM dlight config or hand WGSL) | 6.0 ms matmul + 1.8 ms softmax of 9.6 | The naive f32 tile gets 0.87 TFLOP/s at the Centipawn qkv shape against 0.27-0.29 shipped. GPU 9.6 → ~4.5 ms per b16 invoke | `tvmjs-centipawn --gate` (≤ 1e-3), plus the search-parity protocol in the subgroups doc |

Is hand-written WGSL worth it?

- **Yes, for #1.** It has the largest relative headroom (10-50× per kernel), it
  sits on the default engine's hot path, and the ceiling is a bandwidth number
  this page measured.
- **Conditionally, for #2-#3.** The gains are 10-40% on the big nets. They need
  ORT surgery or a custom runtime, and the biggest step (subgroup-matrix) is not
  shippable today.

## 6. Non-kernel levers

Ranked by expected effect per unit of work:

1. **Batch the Lc0 Small search.** The physical batch is 1 today. A b8 invoke
   costs 16.1 ms against 9.9 ms at b1: 2.0 vs 9.9 ms per position, **4.9× NN
   throughput** with no kernel work. The cost is search-side: collisions and
   virtual loss at low visit counts, which needs a strength A/B. This beats every
   kernel candidate for the default engine. It also compounds with #1, because
   at b8 the GEMMs leave the M ≤ 64 regime.
2. **Maia3 graph fixes.**
   - Rewrite `Einsum bhi,oi->bho` as a MatMul against a pre-transposed weight:
     about −19 ms per grid call (53 → ~35 ms) and −1 ms per move.
   - Export with static shapes (b1 and b11). That folds the
     Shape/Gather/Unsqueeze/Concat arithmetic: 256 single-workgroup `Add`, 64
     `Gather` and 27 `Concat` per invoke, 2.1 ms launch-bound, plus the
     0.8-2.9 ms GPU idle inside the envelope.
   - Expected Maia3 move: 11.5 → ~6-7 ms.
3. **ORT per-invoke CPU/sync: 2.1-6.3 ms per invoke** (23% of Lc0 Small b1).
   - ORT submits 26-59 times per invoke, about one per 16 dispatches.
   - Options: GPU-resident IO binding, preallocated output tensors, and fewer
     readbacks per search step.
   - Earlier pipelining A/Bs were net negative on GPU-bound invokes. Retry only
     after #1 shrinks the GPU share.
4. **Dispatch count.** 356-817 dispatches per invoke at ~1 µs each is only
   0.3-0.9 ms on the GPU. It is a ceiling item after #1 and #2, not a lead.

## 7. Not measurable here, and why

- **DRAM traffic.** WebGPU has no hardware counters. The GB/s figures use bound
  buffer sizes, which is an upper bound on compulsory traffic. TVMJS binds whole
  buffers, so some of its rows exceed the measured peak.
- **Kernel identity inside ORT.** Pipelines are named only by op type, so matmul
  FLOPs are attached by bound-byte matching. When sizes collide, rows can swap
  labels: LQO has several 768-wide shapes, so its per-kernel table is omitted
  and only its totals are reported.
- **Overlap inside a pass.** Split mode serializes dispatches, so split sums are
  upper bounds. Pass mode has the true structure, but it covers 23-53 passes, not
  individual kernels.
- **A quiet machine.** Load was 7-15 throughout and GPU contention cannot be
  locked out. Wall repeats spread by up to ±15%, and the profiled phases of t3
  and Maia3-grid ran ~20% slower than their timing loops. Promotion decisions
  need paired A/B runs (`compareMetric`), not cross-session comparisons with
  this doc.
- **User devices.** Default Chrome quantizes timestamps to 65.5 µs. Wall timing
  works anywhere; per-kernel rows need the developer flag. Subgroup-matrix and
  the adapter description need `--enable-unsafe-webgpu`. Non-Apple and remote
  devices were not run in this phase, though the pages follow the remote
  contract.
- **Full-search cost.** Per-search figures are extrapolated from invokes. They
  leave out tree work, cache hits and Arena/Analysis scheduling.

## Reproduce

```sh
BUILD_SCOPE=product npm run build:client && node scripts/kernel_hillclimb/build_lab.mjs
node scripts/serve_isolated_static.mjs dist-client 5241 &
LOCK=/Users/macthedan/projects/lc0_browser/notes/kernel-hillclimb/benchlock.sh
FLAGS=--browser-args=--enable-unsafe-webgpu,--enable-webgpu-developer-features
$LOCK peak -- node scripts/kernel_hillclimb/bench.mjs --candidate peak --repeat 3 $FLAGS --out artifacts/kernel-hillclimb/results/peak-dev.json
for c in ort-lc0-small ort-lc0-small-b8 ort-t3 ort-bt4 ort-lqo ort-maia3 ort-maia3-grid tvmjs-centipawn; do
  $LOCK $c -- node scripts/kernel_hillclimb/bench.mjs --candidate $c --gate --repeat 3 $FLAGS --out artifacts/kernel-hillclimb/results/$c.json
done
# costs: .venv-onnx/bin/python -I scripts/kernel_hillclimb/onnx_cost.py <model> --batch <b> --out artifacts/kernel-hillclimb/cost/<name>.json
node scripts/kernel_hillclimb/roofline.mjs --peak artifacts/kernel-hillclimb/results/peak-dev.json --md artifacts/kernel-hillclimb/roofline.md
```

All gates in this run passed:

- ORT WebGPU vs WASM is within the preset `parityMaxAbs`. The baseline max-abs
  for the policy logits was 0.025 (t1 b1), 0.06 (t3), 0.29 (BT4) and 1.05 (LQO).
- TVMJS Centipawn vs ORT: 7e-7 policy, 1e-6 WDL.
- TVMJS Lc0 t1: 8/8 and 16/16 best moves.
- TVMJS Lc0 BT4: 7/8 best moves, with 1 flip tolerated as a tie (≤ 0.01).
