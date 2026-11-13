# Lc0 Small split-K on NVIDIA (November 2026)

Follow-up to [`lc0_small_splitk_2026-11.md`](lc0_small_splitk_2026-11.md).
That rewrite (97 MatMuls) is ~1.5× faster per invoke on the Apple M4 but
1.46× slower on the RTX 3090. This record asks two questions:

1. Does **any** split-K-style rewrite of Lc0 Small beat the shipped plain
   model on NVIDIA?
2. Why does the full rewrite lose there?

**Verdict: a narrow rewrite wins a little on the 3090, but not by enough to
add an NVIDIA rule. Apple-only selection stays.**

- **Why the full rewrite loses (§1, §2).** On the 3090, split-K saves GPU time
  too: −1.2 ms (36%) once the profiler's per-dispatch floor is removed. But
  this host is CPU-bound. The shipped model spends ~16 ms of wall per invoke
  for ~5.4 ms of GPU work, and every added dispatch costs ~8-16 µs of wall
  in CPU encode/submit. On the M4 the same dispatch costs ~3.7 µs. Split-K
  adds 239 dispatches, so on the 3090 it costs more than it saves.
- **What wins (§3).** Split only the 12 M = 1 GEMVs (smolgen dense1 ×10,
  value dense1, MLH dense1), sum the partials with a ones-MatMul that fuses
  the bias, and cap the split at S ≤ 8. That adds just 12 dispatches.
  `gemvmm8` passes the gate on both GPUs. Its wall ratio vs shipped is
  **0.877 [p10-p90 0.752-1.048] and 0.928 [0.727-1.065]** in two
  timing-valid paired 3090 jobs, and 0.750 / 0.778 on the M4 (two
  contended runs; `splitk-all` stays the better Apple model at ~0.56-0.60).
  Uncapped, the same rewrite (`gemvmm`) is 0.86-0.96 over five jobs but
  fails the legal-move gate on the 3090 (0.0165 > 0.015).
- **Why there is no rule yet (§4).** The gain is ~7-12% and p90 is above 1.0.
  The gate margin is smaller than the shipped model's (0.0118 vs 0.0061
  against 0.015). It is measured only on the Vulkan proxy, not on Windows
  D3D12 where NVIDIA WebGPU users actually are. §4 proposes a rule and the
  measurement that should gate it. Nothing in `src/` changed.

## Scope and platforms

- **NVIDIA host:** Yukon, RTX 3090 (Ampere), Linux, Chrome for Testing 151,
  Vulkan backend, driver 580. Every run went through the shared gpu-queue with
  the runner's Yukon flags, including
  `--enable-dawn-features=vulkan_enable_f16_on_nvidia`
  ([`remote/README.md`](../scripts/kernel_hillclimb/remote/README.md)).
- **Real NVIDIA users are not on this configuration.** Stock Linux Chrome on
  NVIDIA/Vulkan exposes no `shader-f16`, and since main 46d53f82 the app
  runs Lc0 Small on ORT WASM there. The NVIDIA users who do get WebGPU are
  on Windows, where Chrome uses D3D12 and exposes f16. We have no Windows GPU
  machine. The 3090 on Vulkan is a proxy for that platform, and every NVIDIA
  number below has that caveat. Dispatch and submit costs in particular are
  backend- and CPU-dependent (see §2).
- **Apple host:** the local M4 (10-core GPU), Chrome 157, under
  `notes/kernel-hillclimb/benchlock.sh` via `run.mjs --host local`.

## 1. Per-kernel win/loss of the full rewrite on the 3090

Source: the two solo profiles from the first 3090 session,
`lc0small-shipped-default.json` and `lc0small-splitk-solo.json` (5 profiled
invokes each). Both are `timingValid: false` (GPU at 100% when the job
started), so they are used for attribution only. The rewritten MatMul groups
were matched dispatch by dispatch between the two sequences: 31 dispatches
per encoder layer in the shipped model, 53 in split-K, plus the embedding and
the heads. A Reshape/Transpose shared by several groups is split between
them.

**The dispatch-mode floor.** The profiler puts every dispatch in its own
timestamped pass. On the 3090 that adds about 4.4 µs to every dispatch: no
dispatch measures under 6 µs, even a 1×1×1 Add. Subtracting 4.37 µs per
dispatch from the dispatch-mode totals reproduces the pass-mode busy time,
which times ORT's own passes unsplit:

| | dispatches | dispatch-mode sum | minus 4.37 µs × dispatches | pass-mode busy |
|---|---|---|---|---|
| shipped | 356 | 4.93 ms | 3.37 ms | 3.40 ms |
| split-K | 595 | 4.77 ms | 2.17 ms | 2.18 ms |

So the raw per-kernel numbers overstate every small dispatch, which is most
of what split-K adds. The table gives both columns, ranked by the corrected
net change. Times are µs per invoke, summed over all layers.

| group (shape M×K→N) | dispatches shipped → split | raw µs shipped → split (net) | corrected µs shipped → split (net) | corrected µs saved per added dispatch |
|---|---|---|---|---|
| smolgen dense1 1×2048→256 (×10) | 10 → 40 | 922 → 320 (−602) | 878 → 145 (**−733**) | 24.4 |
| FFN down 64×1024→256 (×10) | 10 → 40 | 501 → 359 (−141) | 457 → 185 (**−272**) | 9.1 |
| value dense1 1×2048→128 | 1 → 4 | 92 → 29 (−64) | 88 → 11 (**−77**) | 25.5 |
| Q/K/V 64×256→256 (×30) | 30 → 97.5 | 550 → 777 (+227) | 419 → 350 (−68) | 1.0 |
| smolgen compress 64×256→32 (×10) | 10 → 22.5 | 181 → 182 (+1) | 138 → 84 (−54) | 4.3 |
| smolgen dense2 1×256→2048 (×10) | 10 → 30 | 183 → 245 (+62) | 140 → 114 (−26) | 1.3 |
| mlh dense1 1×512→128 | 1 → 4 | 28 → 29 (+1) | 23 → 11 (−12) | 4.0 |
| policy dense1/Q/K 64×256→256 | 3 → 10.5 | 53 → 82 (+29) | 40 → 36 (−4) | 0.5 |
| value embed 64×256→32 | 1 → 3.5 | 18 → 26 (+7) | 14 → 10 (−4) | 1.5 |
| input embed 64×176→256 | 1 → 4 | 15 → 29 (+13) | 11 → 11 (0) | — |
| attention out 64×256→256 (×10) | 10 → 40 | 183 → 319 (+135) | 140 → 144 (+4) | loss |
| FFN up 64×256→1024 (×10) | 10 → 40 | 186 → 352 (+166) | 143 → 178 (**+35**) | loss |
| **all rewritten groups** | 97 → 336 | 2913 → 2747 (−167) | 2489 → 1278 (**−1211**) | 5.1 |

The unchanged ops cost the same in both models (2017 vs 2024 µs raw over 259
dispatches).

On GPU time alone, split-K **wins on the 3090 too**: −1.2 ms per invoke, 36%
of the shipped busy time. That is less than on the M4 (−49%) but in the same
direction, and the per-group ranking is the same: the M = 1 GEMVs (smolgen
dense1, value dense1) and FFN down carry most of it, FFN up is a loss, and
attention-out is a wash. Only three groups save more than 5 µs of GPU time
per dispatch they add.

## 2. Why the full rewrite loses on the 3090: dispatches, not kernels

GPU time is not what limits the 3090. In four of the five paired jobs the
shipped model spends ~16 ms of wall time per invoke (11.8 ms in the fifth) but only ~5.4 ms of GPU busy time
(pass mode), and about 8 ms of each invoke falls outside the GPU envelope
(first pass start to last pass end) altogether. On the M4 the same model is
GPU-bound: busy ≈ envelope ≈ 85% of wall, and only ~1.3 ms falls outside the
envelope.

| per invoke, shipped model | M4 (m4-multi-1) | 3090 (yukon-multi-1) | 3090 (yukon-multi-2) |
|---|---|---|---|
| wall median | 9.78 ms | 15.84 ms | 16.04 ms |
| GPU busy (pass mode) | 8.35 ms | 5.45 ms | 5.34 ms |
| GPU envelope | 8.46 ms | 7.93 ms | 7.79 ms |
| idle gaps inside the envelope | 0.09 ms | 2.48 ms | 2.45 ms |
| wall outside the envelope | 1.32 ms | 7.91 ms | 8.25 ms |

The split-K rewrite adds 239 dispatches (356 → 595) and 15 submits (26 → 41)
to save GPU time. On the M4 the saved GPU time is the critical path. On the
3090 it mostly is not, and the added dispatches each cost CPU time to encode,
validate and submit.

**Dispatch probe.** To price an extra dispatch with no kernel change at all,
`splitk_rewrite.py --dispatch-probe N` builds a control model. It splits
nothing. After each of the 97 MatMuls the rewrite would touch, it appends N
pairs of `Neg` (after the bias Add, so ORT's MatMul+Add→Gemm fusion is
unchanged). `Neg(Neg(x)) = x` exactly, so the outputs are bit-identical on
the GPU, and every `Neg` is one cheap elementwise dispatch. `probe1` adds 194
dispatches (`Neg` pairs after 97 MatMuls), close to split-K's +239. `probe2`
adds 388.

| 3090, paired vs shipped | added dispatches | wall ratio median [p10-p90] | MAD | busy ratio | wall per added dispatch |
|---|---|---|---|---|---|
| probe1, yukon-multi-1 | +194 | 1.217 [0.861-1.481] | 0.163 | 1.130 | ~15.6 µs |
| probe1, yukon-multi-2 | +194 | 1.110 [1.008-1.227] | 0.071 | 1.131 | ~8.3 µs |
| probe2, yukon-multi-2 | +388 | 1.270 [1.134-1.401] | 0.061 | 1.252 | ~9.8 µs |
| split-K, yukon-multi-1 | +239 | 1.252 [1.126-1.413] | 0.106 | 0.638 | |
| split-K, yukon-multi-2 | +239 | 1.121 [0.948-1.296] | 0.130 | 0.643 | |

(Wall per added dispatch is the difference of the arms' wall medians divided
by the added dispatches.)

On the M4, probe1 costs ~3.7 µs of wall per added dispatch (wall ratio 1.074
[0.904-1.105], MAD 0.026, m4-multi-1, a contended run), of which ~2 µs is
GPU time on the `Neg` itself (busy ratio 1.049). On the 3090 one extra
dispatch costs 8-16 µs of wall, while the GPU spends only ~3.7 µs on it (busy +0.71 ms for probe1, +1.39 ms for probe2).
That is CPU time on the encode/submit path (ORT's WebGPU EP plus Dawn's Vulkan
backend), not GPU time.

That accounts for the full rewrite. At ~10 µs each, its 239 extra dispatches
cost ~2.4 ms of wall. Its −1.9 ms of GPU busy time mostly does not shorten
the wall: the envelope grows (7.8 → 9.8 ms in yukon-multi-2) and the GPU
idle gap inside it grows from 2.5 ms to 6.3 ms, because the GPU now waits on
the CPU. The observed wall cost is +1.9 to +3.4 ms (ratio 1.12-1.25 across
the two jobs, and 1.46 in the first session's run, which used a different
round count).

The M4 shows the same behaviour once its CPU is saturated. In a run taken at
load average 145 (m4-multi-2, `timingValid: false`, reported only as an
observation), split-K's busy ratio stayed at 0.52 but its wall ratio rose
to 0.96 [0.79-1.52], and its envelope gap went from 0.3 ms to 4.9 ms. Split-K
pays off only when the GPU, not the CPU, is the bottleneck.

**Caveat for the per-arm dispatch profiles.** The `abProfile` dispatch-mode
profiles taken after the rounds ran at visibly different GPU clocks:
identical elementwise kernels (Add, Mul, LayerNorm) measure up to 1.35×
apart between arms in the same job. A GPU that is ~65% idle can drop its
clocks. So §1 uses only the solo profiles, and the variant comparison below
uses the round-by-round pass-mode busy ratios, which are paired.

## 3. Targeted variants

§1 says which groups save GPU time per added dispatch, and §2 says that on the
3090 an added dispatch costs CPU time on the critical path. So the
candidates split only the few groups with a high saving per dispatch, the
M = 1 GEMVs (smolgen dense1 ×10, value dense1, MLH dense1). Some also add FFN
down. Each one comes from `splitk_rewrite.py` options alone (recipes under
Reproduce):

| variant | MatMuls split (S) | reduce | dispatches |
|---|---|---|---|
| `splitk-all` (released Apple model) | all 97 (2-32) | ReduceSum | 595 |
| `d1mm` | smolgen dense1 ×10 (16) | ones-MatMul, bias fused into Gemm | 366 |
| `d1rs` | smolgen dense1 ×10 (16) | ReduceSum | 386 |
| `gemvmm` | the 12 GEMVs (16-32) | ones-MatMul | 368 |
| `gemvmm8` | the 12 GEMVs (8) | ones-MatMul | 368 |
| `gemvmm4` | the 12 GEMVs (4) | ones-MatMul | 368 |
| `gemvrs` | the 12 GEMVs (16-32) | ReduceSum | 392 |
| `gemvffn2` | the 12 GEMVs + FFN down ×10 | ones-MatMul (GEMVs), ReduceSum (FFN) | 398 |
| `probe1` / `probe2` | none (control, §2) | — | 550 / 744 |

### Gate

This is the same `parity=wasm` gate as the parent record, with no tolerance
changed: the shipped model on ORT WASM is the reference, and the limits are
policy 0.2, WDL 0.006, MLH 2, legal-move prob max-abs 0.015, tie-epsilon
0.01, top-3. In the multi-arm page every non-shipped arm is gated, and any
failure sets `result.ok: false`. On the 3090, the bit-exact probes measure
how far the **shipped** model's WebGPU output is from WASM: legal-prob
0.0061.

| variant | 3090 legal-prob max-abs | 3090 gate | M4 legal-prob max-abs | M4 gate |
|---|---|---|---|---|
| shipped (via `probe1`/`probe2`) | 0.0061 | pass | 0.0062 | pass |
| `splitk-all` | 0.0093 | pass | 0.0030 | pass |
| `d1mm` | **0.0165** | **fail** | 0.0024 | pass |
| `d1rs` | 0.0054 | pass | 0.0127 | pass |
| `gemvmm` | **0.0165** | **fail** (5 of 5 jobs) | 0.0024 | pass |
| `gemvmm8` | 0.0118 | pass | 0.0094 | pass |
| `gemvmm4` | 0.0113 | pass | 0.0080 | pass |
| `gemvrs` | 0.0054 | pass | 0.0127 | pass |
| `gemvffn2` | 0.0086 | pass | 0.0089 | pass |

Best move matched 10/10 everywhere. The `gemvmm` failure is deterministic and
happens only on NVIDIA. On the M4 the same model is the closest of all to
WASM. The reduce form moves the result in opposite directions on the two
GPUs:

- `gemvmm` sums the S partial products with a ones[1,S] MatMul, a Gemm with
  the bias fused. It is the closest variant to WASM on the M4 (0.0024) and
  the furthest on the 3090 (0.0165).
- ReduceSum (`gemvrs`, `d1rs`) is closer to WASM than the shipped model on
  the 3090 (0.0054) but further on the M4 (0.0127).

So where the f16 rounding falls depends on the platform, and a margin
measured on one GPU does not carry over to another. A smaller S (`gemvmm8`,
`gemvmm4`) keeps the cheap fused reduce and passes on both GPUs. On the 3090
it has less margin than the shipped model (0.011-0.012 against the 0.015
limit, vs 0.0061).

### Paired timing on the 3090

Every row is a paired multi-arm job that the runner marked
`timingValid: true` (`contended: false`, GPU idle at start). Each job still
saw ~14.4-15.7 foreign CPU cores on average from other tenants; the rotation
puts that load on every arm. Each job used 12 or 20 rounds of 3 invokes per
arm, with the arm order rotated every round, against the shipped model (B).
Ratios are arm / shipped, per round, median [p10-p90], with MAD.

| variant | job | wall ratio [p10-p90] | MAD | GPU-busy ratio | gate |
|---|---|---|---|---|---|
| `splitk-all` | yukon-multi-1 | 1.252 [1.126-1.413] | 0.106 | 0.638 | pass |
| `splitk-all` | yukon-multi-2 | 1.121 [0.948-1.296] | 0.130 | 0.643 | pass |
| `d1mm` | yukon-multi-1 | 1.038 [0.960-1.342] | 0.079 | 0.772 | fail |
| `d1rs` | yukon-multi-3 | 0.988 [0.800-1.131] | 0.082 | 0.785 | pass |
| `gemvrs` | yukon-multi-3 | 1.025 [0.750-1.105] | 0.056 | 0.759 | pass |
| `gemvffn2` | yukon-multi-1 | 1.075 [0.893-1.297] | 0.088 | 0.674 | pass |
| `gemvffn2` | yukon-multi-2 | 1.018 [0.897-1.079] | 0.056 | 0.673 | pass |
| `gemvmm` | yukon-multi-1 | 0.961 [0.707-1.262] | 0.204 | 0.745 | fail |
| `gemvmm` | yukon-multi-2 | 0.937 [0.721-1.025] | 0.070 | 0.746 | fail |
| `gemvmm` | yukon-multi-3 | 0.916 [0.659-0.995] | 0.077 | 0.746 | fail |
| `gemvmm` | yukon-multi-4 | 0.858 [0.707-0.994] | 0.093 | 0.741 | fail |
| `gemvmm` | yukon-multi-5 | 0.929 [0.771-1.341] | 0.132 | 0.744 | fail |
| `gemvmm8` | yukon-multi-4 | **0.877** [0.752-1.048] | 0.082 | 0.751 | pass |
| `gemvmm8` | yukon-multi-5 | **0.928** [0.727-1.065] | 0.058 | 0.755 | pass |
| `gemvmm4` | yukon-multi-4 | 0.897 [0.729-0.996] | 0.077 | 0.768 | pass |
| `gemvmm4` | yukon-multi-5 | 0.963 [0.799-1.060] | 0.072 | 0.769 | pass |

The shipped arm's wall median was 15.8-16.7 ms in every job except
yukon-multi-4 (11.8 ms, MAD 1.25). The ratios are paired per round, so a
host that is faster across the whole job does not move them, but it is a
reminder that this host's absolute times drift.

What the timing shows:

- **Only the variants that split all 12 GEMVs with the ones-MatMul reduce
  are faster on wall time**, by ~4-14% per job (median of medians ≈ 0.93).
  Each adds only 12 dispatches, one per GEMV. Splitting only the smolgen
  GEMVs (`d1mm`, 1.038) saves too little GPU time to show. Their p90 is near or above 1.0 in most jobs, so a single round
  often goes the other way.
- **The same GEMVs with a ReduceSum reduce are not faster** (`gemvrs` 1.025,
  `d1rs` 0.988), although their GPU-busy ratio is about the same (0.76-0.79
vs 0.74-0.77). They
  add 30-36 dispatches (a Transpose, a ReduceSum and the bias Add, which the
  ones-MatMul form folds into a Gemm). In yukon-multi-3, `gemvrs` and
  `gemvmm` differ by 24 dispatches and ~1.7 ms of wall. That is more than
  the ~10 µs per `Neg` dispatch from §2 predicts, so a Transpose or
  ReduceSum dispatch probably costs more CPU time than a `Neg`. One job
  only; not isolated further.
- **Adding FFN down to the GEMVs gives the GPU time back to the CPU**:
  `gemvffn2` has the lowest busy ratio of the targeted variants (0.67) and a
  wall ratio of 1.02-1.08.
- The full rewrite is 1.12-1.25 here and 1.46 in the parent record's run.

### Apple M4

No M4 run in this record is timing-valid. Other agents' GPU and CPU load
(foreign GPU peaks of 11-20%, load average 5-19) tripped the runner's contention
check every time, so these ratios are corroboration only. The parent
record's three timing-valid repeats remain the M4 reference for `splitk-all`
(wall 0.636-0.678). Arms are paired per round as on the 3090, with 12 rounds
of 3 invokes per arm.

| variant | job | wall ratio [p10-p90] | MAD | GPU-busy ratio |
|---|---|---|---|---|
| `splitk-all` | m4-multi-1 | 0.600 [0.524-0.631] | 0.030 | 0.525 |
| `splitk-all` | m4-multi-3 | 0.560 [0.521-0.605] | 0.016 | 0.509 |
| `splitk-all` | m4-multi-4 | 0.590 [0.552-0.658] | 0.038 | 0.528 |
| `d1mm` | m4-multi-1 | 0.780 [0.745-0.911] | 0.041 | 0.748 |
| `d1mm8` | m4-multi-1 | 0.794 [0.729-0.858] | 0.052 | 0.784 |
| `d1rs` | m4-multi-3 | 0.743 [0.711-0.812] | 0.027 | 0.748 |
| `gemvmm` | m4-multi-1 | 0.731 [0.660-0.814] | 0.056 | 0.756 |
| `gemvmm` | m4-multi-3 | 0.738 [0.682-0.807] | 0.028 | 0.699 |
| `gemvmm8` | m4-multi-3 | **0.750** [0.719-0.800] | 0.025 | 0.703 |
| `gemvmm8` | m4-multi-4 | **0.778** [0.738-0.825] | 0.031 | 0.732 |
| `gemvmm4` | m4-multi-3 | 0.783 [0.750-0.910] | 0.026 | 0.754 |
| `gemvrs` | m4-multi-3 | 0.738 [0.698-0.917] | 0.007 | 0.695 |
| `gemvffn2` | m4-multi-1 | 0.664 [0.591-0.721] | 0.042 | 0.655 |
| `probe1` | m4-multi-1 | 1.074 [0.904-1.105] | 0.026 | 1.049 |

On the M4 every variant's wall ratio tracks its busy ratio, as a GPU-bound
device's should. The ReduceSum and ones-MatMul forms of the same GEMV split
time the same here (`gemvrs` 0.738 vs `gemvmm` 0.738), unlike on the 3090.
`splitk-all` stays clearly the best Apple model: `gemvmm8` is ~22-25% faster
than shipped on the M4, while `splitk-all` is ~40% faster. So nothing here
changes the Apple selection. A fourth run, m4-multi-2 at load average 145,
is used only in §2.

## 4. Selection rule: proposal only

No rule changed. `lc0SplitKAdapterEligible` in
`src/lc0/lc0SmallModelSelection.ts` still returns true for `apple` only, so
NVIDIA WebGPU sessions keep the plain QDQ8 export. That is still correct for
`splitk-all`, which loses on the 3090 in every job.

If `gemvmm8` is to ship for NVIDIA, the rule should mirror the Apple one:

- Add a second variant file, `…batch1.f16.qdq8.gemvsplit8.onnx` (the
  `gemvmm8` recipe), and a `'gemvsplit'` member of `Lc0ModelVariant`.
- `selectLc0OrtModelUrl` maps the shipped QDQ8 URL to it only when
  `providers[0] === 'webgpu'` and the adapter vendor is `nvidia`. Other
  adapters, WASM, other URLs and `?model=` overrides stay unchanged, and a
  failed load or WebGPU failure falls back to the original, as for split-K.
- Tests should mirror `tests/lc0_small_model_selection.test.mjs`: vendor
  case and whitespace, a WASM provider keeps the original, an override skips
  selection, the URL mapping keeps the query, and Apple still gets split-K.

What should gate it, none of which exists yet:

1. **Windows D3D12 on NVIDIA**, stock Chrome flags: at least three paired,
   timing-valid jobs with `gemvmm8` against shipped. Promote only if the
   wall-ratio median is ≤ 0.95 in every job and p90 < 1.0 when pooled.
2. **The gate on that platform**, with legal-prob max-abs ≤ 0.012. That
   keeps the margin `gemvmm8` has on the 3090, since the uncapped variant
   shows how much f16 summation order moves the result on NVIDIA.
3. **Cost**: a third ~20 MB model in the publish set and in the cache
   identity. A ~1-1.5 ms gain on a ~16 ms invoke may not be worth that.

The bigger NVIDIA lever is elsewhere. About half of each 3090 invoke (~8 ms)
falls outside the GPU envelope, and the in-envelope gaps add ~2.5 ms more.
Removing dispatches or per-invoke CPU work would help NVIDIA more than any
change in GPU parallelism.

## What is not verified

- **Windows/D3D12 NVIDIA**, the configuration real NVIDIA WebGPU users run.
  The CPU cost per dispatch, the f16 numerics and the gate margin may all
  differ there.
- **Other NVIDIA GPUs and drivers.** Only one RTX 3090 on Linux driver 580
  was measured, with ~14-16 foreign CPU cores busy on the host in every job.
- **Stock Linux Chrome on NVIDIA.** It has no `shader-f16` and runs
  Lc0 Small on WASM, so none of this applies there.
- **Why a Transpose/ReduceSum dispatch costs more wall than a `Neg`
  dispatch** (§3). That is inferred from one job and not isolated.
- **GPU clocks.** The 3090 is ~65% idle per invoke and may downclock. The
  paired busy ratios are robust to that, but absolute per-kernel times
  across arms are not (§2).

## Reproduce

The variant models are gitignored. `splitk_rewrite.py` writes them to
`artifacts/kernel-hillclimb/splitk/`, and `scripts/kernel_hillclimb/build_lab.mjs`
stages every `.onnx` there under `/models/lc0-splitk/` in the lab build.

```sh
PY=".venv-onnx/bin/python -I"   # the ONNX venv at the checkout root
SRC=public/models/lc0/t1-256x10-distilled-swa-2432500.batch1.f16.qdq8.onnx
OUT=artifacts/kernel-hillclimb/splitk/t1.b1.qdq8
$PY scripts/kernel_hillclimb/splitk_rewrite.py $SRC $OUT.splitk-all.onnx --check --summary $OUT.splitk-all.json
# the other variants add (see the table in §3):
#   d1mm      --include 'smolgen/dense1/w$' --gemv-reduce matmul
#   d1mm8     d1mm + --max-s 8
#   d1rs      --include 'smolgen/dense1/w$'
#   gemvmm    --include 'smolgen/dense1/w$|value/dense1/|mlh/dense1/' --gemv-reduce matmul
#   gemvmm8   gemvmm + --max-s 8
#   gemvmm4   gemvmm + --max-s 4
#   gemvrs    --include 'smolgen/dense1/w$|value/dense1/|mlh/dense1/'
#   gemvffn2  --include 'smolgen/dense1/w$|value/dense1/|mlh/dense1/|ffn/dense2/w$' --gemv-reduce matmul
#   probe1    --dispatch-probe 1    (control: no split, +2 exact Neg dispatches per MatMul)
#   probe2    --dispatch-probe 2

BUILD_SCOPE=product npm run build:client && node scripts/kernel_hillclimb/build_lab.mjs

# one page runs every arm: A = model=, B = the shipped lc0-small model,
# abExtra= adds up to 8 more arms; the arm order rotates every round.
M=/models/lc0-splitk/t1.b1.qdq8
Q="preset=lc0-small&model=$M.gemvffn2.onnx&ab=lc0-small&abExtra=$M.splitk-all.onnx,$M.probe1.onnx,$M.probe2.onnx,$M.gemvmm.onnx"
Q="$Q&parity=wasm&profileInvokes=8&abProfile=8&iters=10&abRounds=20"
node scripts/kernel_hillclimb/remote/run.mjs --host yukon --dist dist-client --with-models \
  --include 'lab/**,assets/**,fixtures/lc0/**,ort/ort-wasm-simd-threaded.*,models/lc0/t1-256x10-distilled-swa-2432500.batch1.f16.qdq8.onnx,models/lc0-splitk/<the arms>.onnx' \
  --page "lab/kernel-hillclimb/ort-profile.html?$Q" --timeout 600 --estimate-min 6 \
  --out artifacts/kernel-hillclimb/results/nvidia/yukon-multi-2.json
# M4: the same page through benchlock, profileInvokes=0
<workspace>/notes/kernel-hillclimb/benchlock.sh node scripts/kernel_hillclimb/remote/run.mjs --host local \
  --dist dist-client --page "lab/kernel-hillclimb/ort-profile.html?…&profileInvokes=0" --timeout 600 --out …
```

`ort-profile.mjs` gates every `abExtra` arm with `parity=wasm` against the
shipped model on ORT WASM, records each arm's pass-mode busy, envelope, gap,
dispatches and submits every round, and with `abProfile=N` adds an N-invoke
dispatch-mode kernel profile (with a per-position `sequenceMeanUs`) for B and
each extra arm after the rounds. In dispatch mode every dispatch takes one
of the profiler's 16,384 timestamp slots, so `abProfile` is capped per arm
to the invokes that fit (`requestedInvokes`, `capped`). `result.ok` is false
when any arm fails its gate, or when an arm's `abProfile` profile or a
per-round pass-mode profile dropped timestamps (`ab.incompleteProfiles` names
the arm). The two jobs here that used `abProfile=8` (yukon-multi-2 and -3)
needed at most 5,952 slots per arm, and every profile in them is complete.
