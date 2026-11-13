# Lc0 Small split-K graph rewrite (November 2026)

Can an ONNX graph rewrite steer ORT 1.27's WebGPU EP to faster small-M matmuls
for Lc0 Small (`t1-256x10-distilled-swa-2432500.batch1.f16.qdq8.onnx`) at
physical batch 1, without a custom kernel? This is option 1 of candidate #1 in
[`kernel_hillclimb_roofline_2026-11.md`](kernel_hillclimb_roofline_2026-11.md) §5.

**Verdict: yes on the Apple M4, no on the RTX 3090.** A split-K rewrite of 97
MatMuls passes the existing ORT WASM parity gate unchanged. On the Apple M4 it
is **~1.5× faster per invoke on wall time and ~1.95× on GPU busy time**, in
paired ABAB runs. On the RTX 3090 (Chrome/Vulkan) a paired run makes it
**~1.46× slower on wall time** (§5). Since release
2026-11-12.kernel-releases.1 it is served to Apple WebGPU sessions only (§9). The same rewrite for the smolgen compress GEMV in t3/BT4/LQO
gives 4-8% per invoke (§7).

Setup matches the roofline doc: Apple M4 (10-core GPU, `apple / metal-3`),
headless Chrome 157, the production `dist-client` plus the lab overlay, served
by `scripts/serve_isolated_static.mjs`. Every M4 timing ran inside
`notes/kernel-hillclimb/benchlock.sh`. The load average was 3-30 during the
session, and other agents also used the GPU outside benchlock. Absolute times
therefore swing, and only paired ratios and GPU-timestamp numbers are quoted as
results.

## 1. Why ORT is slow here

These are the paths in ORT 1.27's WebGPU EP:

- **MatMul** (`webgpu/math/matmul.cc`):
  - Each workgroup is 8×8 threads. Each thread computes 4 output columns. It
    computes 4 rows, or 1 row when M ≤ 8.
  - The output tile is therefore 32×32, or 32×8 when M ≤ 8.
  - The dispatch is `(ceil(N/32), ceil(M/tileM), batch)`. **The batch dimension
    goes on grid z.**
  - When B has batch 1, A's batch is folded into M.
- **Gemm** (`gemm_packed.cc`) always uses a 32×32 tile, even when M = 1. ORT's
  MatMulAddFusion turns every MatMul + bias Add with a constant weight into a
  Gemm, so most Lc0 Small layers run as Gemm.
- **Built-in split-K** exists (`SplitKConfig` in `webgpu_utils.cc`), but it is
  enabled only for Intel adapters. It never runs on Apple or NVIDIA.

At batch 1, Lc0 Small has M = 64 token rows (or M = 1 for the smolgen and head
GEMVs) and N = 256. Most layers therefore launch 16 workgroups, and the
smolgen `dense1` GEMV (1×2048→256) launches 8. Each of those workgroups walks
the full K. The M4 has 10 GPU cores, so 8-16 workgroups leave most of the
device idle. That is why the roofline doc measured 7.4 ms/invoke in Gemm/MatMul
at a few percent of bandwidth.

None of the ORT shapes has a better tile for this case. Transposing the problem
(Yᵀ = WᵀXᵀ) gives the same tile count. The only parallelism lever reachable
from the graph is batch on grid z, which is split-K.

## 2. The rewrite

Script: [`scripts/kernel_hillclimb/splitk_rewrite.py`](../scripts/kernel_hillclimb/splitk_rewrite.py).

For each eligible `Y = X·W` (X is [M,K], W is a constant [K,N] behind a
DequantizeLinear, or a plain initializer), choose a split S and rewrite:

```text
X [M,K] -> Reshape [M,S,K/S] -> Transpose(1,0,2) [S,M,K/S]   (M = 1: one Reshape [S,1,K/S])
W int8 [K,N] -> same bytes re-dimensioned as [S,K/S,N]; DequantizeLinear axis 1 -> 2
MatMul -> [S,M,N]  (batched: ORT puts S on grid z)
ReduceSum(axes=0, keepdims=0) -> [M,N]
```

The downstream bias Add is left alone. ORT fuses it into a Gemm only when the
MatMul has a 2-D constant B, so after the rewrite the bias Add stays a separate
elementwise op.

How S is chosen (`choose_splits`):

- S is the smallest value where:
  - S divides K;
  - K/S is a multiple of 4, which keeps the vec4 path;
  - K/S ≥ 32 (`--min-inner`);
  - the base workgroup count × S ≥ 128 (`--target-wg`).
- The base workgroup count is `ceil(N/32)·ceil(M/(M ≤ 8 ? 8 : 32))`.
- S is capped at 64.

When one weight feeds several MatMuls, the weight is split once and every
consumer must accept the same S. One example is the smolgen weight-gen DQ shared
by all 10 encoders. One Reshape/Transpose is emitted per (input, S) pair, so
the Q/K/V projections of a layer share a single transpose of their input.

What it rewrote in Lc0 Small (97 MatMuls, default options):

| shape M×K→N | count | S | workgroups before → after |
|---|---|---|---|
| 64×256→256 (Q, K, V, attention out, policy dense1/Q/K) | 53 | 8 | 16 → 128 |
| 64×1024→256 (FFN down) | 10 | 8 | 16 → 128 |
| 64×256→1024 (FFN up) | 10 | 2 | 64 → 128 |
| 64×256→32 (smolgen per-head compress, value embed) | 11 | 8 | 2 → 16 |
| 1×2048→256 (smolgen dense1) | 10 | 16 | 8 → 128 |
| 1×256→2048 (smolgen dense2) | 10 | 2 | 64 → 128 |
| 64×176→256 (input embedding) | 1 | 4 | 16 → 64 |
| 1×2048→128 (value dense1) | 1 | 32 | 4 → 128 |
| 1×512→128 (mlh dense1) | 1 | 16 | 4 → 64 |

Not rewritten:

- the smolgen weight-gen MatMul (already 128 workgroups);
- activation × activation MatMuls (attention QKᵀ/AV, policy QKᵀ);
- weights under 4096 elements (the tiny head Gemms).

The script's `--check` mode runs the original and the rewrite on ORT CPU over
random planes. For the f16 model it reports max-abs policy 0.090, WDL 7.8e-5
and MLH 0.029, which is the f16 summation-order noise expected from
re-associating K.

### Dequant fold and size

The rewrite keeps every DequantizeLinear and only changes its axis (1 → 2) and
its int8 initializer's dims. `src/nn/onnxDequantFold.ts` handles per-axis DQ of
any rank, so the in-browser fold still turns all 98 QDQ nodes into f16
initializers. Every run checked `lastOrtDequantFoldSummary()`:
`foldedNodes: 98, skippedNodes: 0`, the same as the shipped model. This rules
out a silent fallback to the unfolded model in `createOrtSession`.

| | download | after fold |
|---|---|---|
| shipped | 20,558,031 B | 40,492,866 B |
| split-K | 20,580,952 B (+22.9 KB, shape constants and the new nodes) | same weights, same f16 size |

## 3. Kernel path before and after

The per-dispatch numbers come from quiet interleaved profiles (`il-*.json`,
`profileInvokes` with GPU timestamps, M4). They are µs per dispatch.

| layer class | shipped | split-K |
|---|---|---|
| smolgen dense1 1×2048→256 | Gemm, 8 wg, **212.8** | MatMul 8×1×16 (128 wg) 23.9 + Transpose 3.8 + ReduceSum 5.0 |
| value dense1 1×2048→128 | Gemm 4×1×1, **213** | MatMul 4×1×32 18.8 + ReduceSum |
| Q/K/V/out/policy 64×256→256 (×43 timed) | Gemm 8×2×1, **35.4** | MatMul 8×2×8 14.1 + ReduceSum 7.5 + shared Transpose ~5.2 |
| FFN down 64×1024→256 | Gemm, **121.5** | MatMul 35.4 + ReduceSum 7.5 + Transpose 9.0 |
| FFN up 64×256→1024 | Gemm, 38.1 | MatMul 33.7 + ReduceSum 8.6 + Transpose 5.2 (**net loss**) |
| smolgen dense2 1×256→2048 | Gemm, 38.6 | MatMul 22.3 + ~8 reduce/reshape |
| smolgen compress 64×256→32 | MatMul 1×2×1, **33.2** | MatMul 9.1 + ReduceSum 4.7 |
| smolgen weight-gen | 128×1×1, 32-38 | unchanged |

Whole invoke, interleaved quiet profiles:

| | run ms | kernel ms | pass-mode GPU busy ms | dispatches |
|---|---|---|---|---|
| shipped | 9.07-9.33 | 7.77-8.34 | 6.81-7.60 | 356 |
| split-K | 5.48-5.85 | 5.23-5.31 | 4.42-4.74 | 595 |

After the rewrite, the large token-row MatMuls run at about 1.15 TFLOP/s, which
is ORT's own GEMM ceiling at these shapes (roofline doc §2). Each of the ~240
extra Transpose/ReduceSum dispatches costs 4-8 µs. Most of the remaining gap
to the roofline ceiling is that dispatch overhead plus ORT's tile.

Still unhandled:

- policy QKᵀ MatMul 2×2×1, ~36 µs (activation × activation);
- mlh embed Gemm 1×2×1, ~33 µs;
- the 1×1×1 head Gemms, 23-25 µs each.

## 4. Gate

The gate is the existing `ort-profile` `parity=wasm` gate, with no tolerance
changed. The reference is the **original shipped model** on ORT WASM. The
candidate runs on WebGPU. The gate covers the full 10-fixture corpus with the
`lc0-small` preset's `parityMaxAbs` (policy 0.2, WDL 0.006, MLH 2) and
`legalPolicy` (max prob-abs 0.015, tie-epsilon 0.01, top-3).

| | split-K on WebGPU | shipped on WebGPU (baseline) |
|---|---|---|
| policy logit max-abs | 0.042 | 0.0996 |
| WDL max-abs | 0.00244 | 0.0027 |
| MLH max-abs | 0.125 | 0.966 |
| legal-move prob max-abs | 0.003 | 0.006 |
| best move | 10/10, no tie flips | 10/10 |

On every output, the candidate is closer to the WASM reference than the shipped
WebGPU path is. The f32 ReduceSum over S partial sums adds less rounding than
ORT's long f16 K walk.

## 5. Paired speed

### Apple M4

Candidate `ort-lc0-small-splitk-ab`: the split-K model is arm A and the shipped
model is arm B. Each round runs 3 invokes of each arm in alternating order in
one page session, for 30 rounds. Each repeat is a fresh page load. Ratios are
A/B, so lower means split-K is faster. Load average was ~5.

| repeat | wall ratio median [p10-p90] | MAD | GPU-busy ratio median [p10-p90] |
|---|---|---|---|
| 1 | 0.636 [0.577-0.742] | 0.099 | 0.514 [0.474-0.570] |
| 2 | 0.652 [0.608-0.743] | 0.042 | 0.505 [0.467-0.544] |
| 3 | 0.678 [0.521-1.150] | 0.223 | 0.513 [0.445-0.638] |

The wall ratio is ~0.65, or **1.53× faster per invoke**. The GPU-busy ratio is
~0.51, or **1.95× less GPU time**. The wall gain is smaller because the extra
dispatches cost CPU-side encode time, and because the shared machine's CPU load
inflates submit latency. The gate passed on all three repeats.

### RTX 3090 (Yukon, Chrome/Vulkan)

**Split-K is ~1.46× slower per position here.** Every run below used the
runner's Yukon flags, which are not standard Chrome. They force Vulkan/ANGLE
and add `--enable-dawn-features=vulkan_enable_f16_on_nvidia`. Without that
flag, NVIDIA/Vulkan does not expose `shader-f16`, so ordinary Chrome users on
this GPU do not get this configuration (see
`scripts/kernel_hillclimb/remote/README.md`). In that default
configuration the f16 nets cannot run on WebGPU at all. Since main 46d53f82
the app sends them to WASM there.

**Paired run** (`abRounds=10`, block 3, `parity=wasm`, no profiled invokes;
`lc0small-splitk-paired.json`). The runner marked it `timingValid: true`
(GPU idle at start), although other processes still used ~14.6 CPU cores on
average. Rounds interleave the two models, so that load hits both sides:

| per position | split-K | shipped | split-K / shipped |
|---|---|---|---|
| wall median (MAD) | 22.41 ms (4.54) | 15.50 ms (3.47) | **1.46** (MAD 0.11, p10-p90 1.22-1.75) |
| GPU busy median (MAD) | 3.59 ms (1.41) | 5.71 ms (2.30) | (both noisy) |

The ratio is stable even though the absolute times are noisy: every round
had split-K slower (min ratio 1.16). The gate passed. The GPU-busy medians
have a MAD near 40% of the median in this mode, so they are not a result.

**Earlier unpaired solo runs** (30 timed iterations plus 5 profiled invokes;
`lc0small-shipped-default.json`, `lc0small-splitk-solo.json`). Both were
marked `timingValid: false` (GPU at 100% at job start, ~14-15 foreign CPU
cores) and are kept only for the per-kernel profile:

| model | wall median | GPU busy / invoke | dispatches / invoke | gate |
|---|---|---|---|---|
| shipped `batch1.f16.qdq8` | 9.86 ms (MAD 1.06) | 4.95 ms | 356 | pass |
| split-K (`splitk-all`) | 15.07 ms | 4.78 ms | 595 | pass (legal-prob 0.009) |

In that profile, net GPU busy time drops only ~3% across all kernels. The net
figure hides large per-kernel changes in both directions. For example, the
first smolgen dense1 goes from a 92.16 µs Gemm at `8x1x1` to a 9.22 µs MatMul
at `8x1x16`. The added reshape/transpose/reduce/bias work and the 239 extra
dispatches per invoke offset these gains. One explanation for the wall gap is
the extra CPU-side encode and submit cost of those dispatches. That was a
hypothesis here. The follow-up
[`lc0_small_splitk_nvidia_2026-11.md`](lc0_small_splitk_nvidia_2026-11.md)
measured it. Once the profiler's per-dispatch floor is removed, split-K saves
~36% of GPU time on the 3090 as well. Each extra dispatch, though, costs
8-16 µs of CPU-side wall time on this host, and the 3090 run is CPU-bound.

The first attempt, a 30-round paired A/B with default iterations, crashed
Chrome's GPU process (`SEGV_ACCERR` about 30 s in), restarted it, and then hit
the runner's 990 s deadline with no result. Neither model reproduced it alone,
and the shorter 10-round paired run above finished without a crash.

### WASM EP (the fallback path)

The same model on ORT WASM, paired 10 rounds on the M4, gives **wall ratio
1.21** (split-K is *slower*, p10-p90 1.17-1.21). On the CPU the
reshape/transpose/reduce work is pure overhead, and the shipped MatMul is
already as parallel as it can be. A release must therefore serve the split-K
model only to WebGPU sessions (see §8).

## 6. Variants and negative results

Each variant was paired against another variant or against the shipped model
with the same harness. The ratios are GPU-busy A/B. The gate column is the §4
gate with the variant as the gated model, either as `model=` in a gated run
against the shipped model, or (lt64, lt64w0, mm) re-run after the
`abModel=` fix below.

| variant | vs | busy ratio | gate | note |
|---|---|---|---|---|
| `mm`: M = 1 reduce as a ones-vector MatMul | all | 1.003 | pass (legal 0.008) | noise |
| `256`: target 256 workgroups, plus the rank-3 attention-input MatMuls (107 split) | mm | 0.998 | pass | noise |
| `lt64`: skip layers already at ≥ 64 workgroups (FFN up, smolgen dense2) | all | 1.020 | **fail**: legal prob 0.0165 > 0.015 | noise, even though FFN up is a per-dispatch loss |
| `lt64w0`: lt64 plus weights under 4096 elements | lt64 | 1.020 | **fail**: legal prob 0.0165 > 0.015 | noise |
| `gemv`: GEMVs only (M = 1) | mm | 1.386 | pass | **worse**; the token-row GEMMs carry most of the gain |
| `mmr`: token-row reduce as a ones MatMul | shipped | 0.684 | pass | the reduce MatMul costs 17-20 µs, and 96 µs at N = 65536 |
| bias fused back via Tile + Gemm | shipped | 0.837 | pass | ORT does not constant-fold the Tile; the fused Gemm costs 35-172 µs |
| `256i16`: target 256, min inner K 16 | shipped | 0.584 | pass | worse than the default (~0.51) |

The default options are the simplest variant that passes the gate and is
within noise of the best, so they are the candidate.

**Harness fix (review).** In variant-vs-variant pairs, the `abModel=`
override ran in arm B. Before the fix, `parity=wasm` checked only arm A, so
arm B was not gated. Now:

- `ort-profile` gates an overridden arm B against its preset's original model
  (`ab.b.parity`), and that result counts toward `ok`.
- An `abModel=` run without `parity=wasm` is never `ok`.

Every parity number in this doc comes from a run where the reported model was
arm A (`model=`): §4, the sweep rows above, §5 (`ort-lc0-small-splitk-ab`)
and §7. The variant pairs carried no parity claim. lt64 and lt64w0 were gated
for the first time with the fix, and both fail.

The fixed gate was checked both ways. A split-K copy with one FFN dequant
scale multiplied by 1.5 fails as `abModel=` (ok false, arm A passes). It also
fails as `model=`. The real candidate (`model=`) plus `mm` (`abModel=`) passes
with ok true.

## 7. Stretch: smolgen compress GEMVs in t3, BT4 and LQO

These nets run at physical batch 8 with a dynamic first dimension. The rewrite
used `--include 'smolgen/dense1/w$' --assume-m 8`, which splits 15 MatMuls per
net with S = 16 (8 → 128 workgroups). The dense1 MatMul drops from 300+ µs to
32-36 µs per dispatch. Each net was paired 10 rounds against its shipped model
under its own preset gate.

| net | wall ratio [p10-p90] | busy ratio | policy max-abs | WDL max-abs | best move | fold |
|---|---|---|---|---|---|---|
| t3 (f16, b8) | 0.916 [0.839-0.938] | 0.921 | 0.056 | 0.0044 | 10/10 | n/a (no QDQ) |
| BT4 (QDQ8, b8) | 0.958 [0.887-0.996] | 0.989 | 0.302 | 0.0024 | 9/10 (one tie, reference gap 0.00028) | 147 nodes |
| LQO (QDQ8) | 0.947 [0.901-0.987] | 0.954 | 0.898 | 0.0198 | 9/10 (the allowlisted flip, gap 0.045) | 144 nodes |

All three pass their existing gates. The gain is 4-8%, smaller than the
roofline doc's estimate (−4.5 to −5 ms per invoke). At b8 these nets are
dominated by the large GEMMs (§5 candidate #3), and the saved dense1 time is
partly offset by the extra dispatches. The follow-up, with regenerated
artifacts, three-repeat paired runs, the RTX 3090, and a paired search A/B, is
[`bignet_smolgen_splitk_2026-11.md`](bignet_smolgen_splitk_2026-11.md):
the t3 rewrite is wired for Apple WebGPU sessions (0.90-0.94× per move, pending the owner's R2 publish); BT4 and LQO keep their shipped models.

## 8. Is a custom kernel still needed?

Not to get the first 1.5-2×; the graph rewrite gets it. Option 3 (a custom
runtime for the op) still has headroom:

- The rewritten token-row MatMuls sit at ORT's ~1.15 TFLOP/s tile ceiling, and
  the roofline doc measured ~2.2 TFLOP/s for an f16-accumulate tile at similar
  shapes.
- The rewrite adds ~240 dispatches (356 → 595), 4-8 µs each. That is roughly
  1.2-1.5 ms of the remaining 4.4-4.7 ms GPU busy. A native split-K kernel
  with a fused reduce + bias would remove them.
- The roofline doc's ~3.5-4 ms invoke target is still ~1.5 ms below the
  split-K model's 5.5 ms run time.

So a custom kernel is now worth roughly another 1.3-1.5×, not the 2.5-2.8× the
roofline doc estimated. Its cost (ORT surgery or a hybrid runtime for every
matmul) should be weighed against that smaller gain. The cheap next steps stay
in-graph: drop the FFN-up split, and look at the remaining head Gemms.

## 9. Release steps

**Status: released** in 2026-11-12.kernel-releases.1 (main 60a36d58), with
steps 1–3, 5 and 6 done. The live site was checked on the M4: Lc0 Small
fetched and cached the split-K artifact and analysed. The step 4 items listed
under "Not yet run" below are still open.

- **Artifact.** `t1-256x10-distilled-swa-2432500.batch1.f16.qdq8.splitk.onnx`
  is 20,580,952 bytes, sha256
  `3a5ccb636ebc596a66021b90a39a3d5b708fbd280048b07ff2d7855d240b37a1`. It was
  made by step 1 with default options (97 MatMuls rewritten) and is
  byte-identical to the gated `splitk-all` candidate. It is listed in
  `manifest.json` and both `r2-*-present` manifests, with a matching
  artifact-index update.
- **Selection rule.** `src/lc0/lc0SmallModelSelection.ts` applies it in one
  place. The model is split-K only when the first resolved ORT provider is
  `webgpu` AND `GPUAdapterInfo.vendor` is `apple`. Everything else gets the
  plain QDQ8 export, including NVIDIA, AMD, Intel and WASM. NVIDIA is
  excluded because it was not shown to help (§5).
- **Who opts in.** The callers that set `deviceSelection` are the
  search-worker init, the Analysis, Play and Arena main-thread loaders, and
  Critical Moments, but only when no `?model=` override is given. An
  explicit `?model=` URL, even one naming the default file, loads exactly the
  bytes it names. So do the lab tools
  `policyOnlyBrowser.ts` and `batchingStrengthMatch.ts`.
- **Fallbacks.** If the split-K load fails, for example because the CDN does
  not have it yet, the original model is served. The split-K session requests
  `['webgpu']` only. With `wasm` listed too, ORT would silently build a WASM
  session from the split-K bytes when its WebGPU backend fails to start. If
  the WebGPU attempts fail, `createOrtSession`'s WASM attempt loads the
  ORIGINAL URL through `wasmFallbackModel`. Unreadable adapter metadata
  leaves the vendor unknown, so the original model is served and WebGPU is
  kept.
- **Caches.** Each model's bytes are cached and sha256-verified under its own
  URL. The worker reports `servedModelUrl`, and that URL is part of the
  Analysis evaluation-cache key, the audit `modelUrl`, and the Critical
  Moments cell key. A Critical Moments report starts its Lc0 worker before
  accepting any cached Lc0 result, so the key names that worker's model.
- **Gate run.** `ort-lc0-small-splitk --gate --repeat 3` passed 3/3 on Apple
  M4 (Metal 3), all 98 fold nodes folded. The max-abs differences were
  policy 0.042, WDL 0.0024 and MLH 0.125, against limits of 0.2, 0.006 and 2.
  Best move matched in 10/10.
- **Built-preview smokes.**
  - Play and Analysis loaded only the split-K URL on WebGPU.
  - Play with `?ep=wasm` loaded only the original.
  - Non-Apple selection is covered by `tests/lc0_small_model_selection.test.mjs`.
- **Not yet run.** The non-Apple GPU browser gate, an Arena smoke and the
  paired search A/B.

1. **Generate the artifact** from the shipped model, deterministically:

   ```sh
   .venv-onnx/bin/python -I scripts/kernel_hillclimb/splitk_rewrite.py \
     ../models/lc0-bestnets/onnx/t1-256x10-distilled-swa-2432500.batch1.f16.qdq8.onnx \
     ../models/lc0-bestnets/onnx/t1-256x10-distilled-swa-2432500.batch1.f16.qdq8.splitk.onnx \
     --check --summary <out>.json
   ```

2. **Add a manifest entry** in `public/models/lc0/manifest.json` with `file`,
   `url`, `mode: symlink`, `source`, `bytes` and `sha256`, next to the
   `batch1.f16.qdq8` entry. Then stage it with
   `npm run lc0:prepare-model-assets`. Add it to
   `public/models/lc0/r2-v0-present.manifest.json` (and
   `r2-stable-present.manifest.json`), then run `npm run artifact-index`
   (`tests/artifact_index.test.mjs` checks it).
3. **Select the model by execution provider.** The split-K model is 1.21×
   slower on WASM and 1.46× slower on the RTX 3090 WebGPU (§5). Do not just
   swap the URL. The gain is device-specific, so a release also needs a
   device condition (for example Apple adapters only), or a short on-device
   timing probe, measured on more GPUs first:
   - Map the default Lc0 Small URL to the split-K URL only when the resolved
     ORT providers include `webgpu`. The resolution happens in
     `createOrtSession` (`src/nn/ortRuntime.ts`), or in a helper the callers
     use.
   - The callers that hard-code the batch-1 QDQ8 URL are:
     - `src/lc0/analysisBrowser.ts` (`DEFAULT_MODEL_URL`)
     - `src/lc0/arenaBrowser.ts`
     - `src/lc0/playBrowser.ts`
     - `src/lc0/criticalMomentsEngines.ts` (`CRITICAL_LC0_MODEL`)
     - `src/lc0/policyOnlyBrowser.ts`
     - `src/lc0/batchingStrengthMatch.ts` (`SMALL_BATCH1_URL`)
   - The WebGPU → WASM fallback after a session failure would then load the
     split-K model on WASM. That is correct, but 1.21× slower. Either accept
     this or make the fallback attempt use the original URL.
   - Check that the model-storage and evaluation caches
     ([`analysis_continuity_2026-10.md`](analysis_continuity_2026-10.md)) key on
     the served URL or hash, so cached evaluations from the old model are not
     mixed in.
4. **Gate in the built product.** Run:
   - `ort-lc0-small-splitk --gate` on the M4;
   - the same on at least one non-Apple GPU;
   - a built-preview WebGPU smoke of Analysis, Play and Arena, since dev
     masks worker issues;
   - a paired search A/B at fixed visits (the roofline doc's "then a search
     A/B"). Outputs differ from the shipped model at the f16 reordering level,
     so best-move parity is expected rather than bit identity.
5. **Publish and release** through the existing artifact flow:
   - `npm run deploy:write-release-manifests`, which stages a new manifest at
     `.local-dev-artifacts/artifact-releases/releases/<release-id>.json` by
     default (override with `--out-dir` only for a managed alternative root);
   - publish the hashed artifact to R2;
   - then `npm run release:netlify-r2`.

   As always, publish to R2 before deploying the app shell.
6. Update [`hosted_artifacts.md`](hosted_artifacts.md) for the Lc0 Small row:
   - WebGPU sessions on the devices selected in step 3 get the split-K export.
   - Other WebGPU devices and WASM keep the plain QDQ8 export.

## Reproduce

The candidate models are gitignored. They are written to
`artifacts/kernel-hillclimb/splitk/` (with a `.json` summary each), and
`scripts/kernel_hillclimb/build_lab.mjs` stages them under
`/models/lc0-splitk/` in the lab build.

```sh
PY=".venv-onnx/bin/python -I"   # the ONNX venv at the checkout root
SRC=public/models/lc0/t1-256x10-distilled-swa-2432500.batch1.f16.qdq8.onnx
OUT=artifacts/kernel-hillclimb/splitk
$PY scripts/kernel_hillclimb/splitk_rewrite.py $SRC $OUT/t1.b1.qdq8.splitk-all.onnx --check --summary $OUT/t1.b1.qdq8.splitk-all.json
# variants: --gemv-reduce matmul (mm); mm + --target-wg 256 --rank3 (256); --max-base-wg 64 (lt64);
#   lt64 + --min-weight 0 (lt64w0); --rows gemv (gemv); --gemv-reduce matmul --reduce matmul (mmr)
# stretch: --include 'smolgen/dense1/w$' --assume-m 8 on the t3 / BT4 / LQO exports

BUILD_SCOPE=product npm run build:client && node scripts/kernel_hillclimb/build_lab.mjs
node scripts/serve_isolated_static.mjs dist-client 5241 &
LOCK=<workspace>/notes/kernel-hillclimb/benchlock.sh
FLAGS=--browser-args=--enable-unsafe-webgpu,--enable-webgpu-developer-features
$LOCK splitk -- node scripts/kernel_hillclimb/bench.mjs --candidate ort-lc0-small-splitk --gate --repeat 3 $FLAGS \
  --out artifacts/kernel-hillclimb/results/splitk.json
$LOCK splitk-ab -- node scripts/kernel_hillclimb/bench.mjs --candidate ort-lc0-small-splitk-ab --gate --repeat 3 $FLAGS \
  --out artifacts/kernel-hillclimb/results/splitk-ab.json
```

The `ab=` pairing in `lab/kernel-hillclimb/ort-profile.mjs` takes an
`abModel=<url>` override. Arm B then runs that model with the B preset's batch,
which is how variant-vs-variant pairs and the stretch nets (`preset=t3&model=…&ab=t3`)
were measured. Add `ep=wasm` for the WASM pairing.
