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

This doc was revised after a Codex review of the harness (9 findings, all
accepted). The corrections, and which conclusions they moved, are listed in
[Changes after review](#changes-after-review). The wall times in §3 are still
the phase-0 runs (load 7-15). A review sweep ran at load 14-31 and is used only
for GPU-timestamp checks and paired ratios.

## Changes after review

What was wrong, and what it changed:

| finding | fix | effect on numbers |
|---|---|---|
| ORT parity compared the candidate model with itself on WASM | `parity=wasm` now runs the preset's original model (or `referenceModel=`) as the reference; no preset and no reference = fail | Shipped presets: no change (the candidate is the reference, so max-abs is unchanged). The gate now rejects broken weights: with `model=` a t1 copy whose `encoder5` Q scale is ×1.1 fails on policy (0.115 > 0.06), and one whose `encoder5` FFN-down scale is ×1.5 fails on all three outputs. Judged against itself (the old behaviour), the ×1.1 copy passed |
| TVMJS Lc0 gate ignored WDL and measured tie gaps in the candidate's own priors | `compareLc0` (now `lab/kernel-hillclimb/gates.mjs`) requires the same legal-move set, finite values, \|prior diff\| ≤ 0.03 (t1) / 0.05 (t3, BT4), \|WDL diff\| ≤ 0.01, and tie gaps from the reference | All lanes still pass: t1 b8/b16 prior 0.0063, WDL 0.0022, 8/8 and 16/16; BT4 prior 0.012, WDL 0.0024, 7/8 with one tie whose reference gap is 0.00055 |
| Plain WGSL GEMM had a cross-invocation race (4 invocations wrote components of one workgroup `vec4`) and no output check | Race-free A-tile load (each invocation writes whole vectors); every GEMM and subgroup-matrix row is checked against a float64 CPU reference on 96 sampled outputs and excluded from the peaks if it fails | Every row passes (max relative error 1e-7 at f32 accumulation, ≤ 1.1e-3 at f16). The plain GEMM ceiling rises **1.99 → 2.59 TFLOP/s** (the race-free load is also faster). At the BT4 shapes, the untuned f16/f16-accumulate tile now reaches **2.16-2.23** (was 1.46-1.75) |
| FMA kernel used splat constants and only invocation 0 stored its result | Per-lane multiplier, addend and seeds; every invocation stores | f32 FMA **3.22 → 3.44 TFLOP/s** (+7%), f16 **2.72 → 3.28** (+21%). Ridge 35 → 37 FLOP/B |
| Byte matcher was greedy on bound bytes: BT4's 512×1024×1024 projection and its 256-batch 64×64×32 attention matmuls both bind 4,194,304 B, which gave some dispatches ~12 TFLOP/s on a 3.2 TFLOP/s roof | `matchMatmuls` filters by grid shape (ORT tiles C at 32×32, z = batch), then by node count, and leaves a class unassigned if its candidates still differ in FLOPs | Path totals are unchanged: matmul TFLOP/s and % of peak use graph FLOPs, not the matcher. The per-kernel "compute"-bound time disappears: BT4 1.33 ms, LQO 7.95 ms and Maia3-grid 0.95 ms were mis-attributions, and **no shipped kernel is compute-bound**. LQO's per-kernel table is now valid (it was omitted). The Maia3-grid MatMuls were under-credited: 1.18-1.26 TFLOP/s, not 0.42-0.61. The matcher now attaches 99.9-100% of each graph's matmul FLOPs; the only unassigned dispatches are sub-20-KFLOP head GEMVs |
| `profileInvokes` up to 200 could overflow the 16384 timestamp slots, with the run still `ok` | Dropped dispatches make the profile incomplete and the run `ok: false` | No phase-0 run had dropped slots |
| Bandwidth test bound 256 MiB unconditionally; WebGPU errors were only logged | Size clamped to `maxStorageBufferBindingSize`; error scopes around timed work; any uncaptured error fails the page | M4 limit allows 256 MiB: no change (read 93 GB/s) |
| Roofline used failed and quantized repeats, and a zero median gave infinite throughput | Only `ok`, unquantized, complete repeats are used (rejections are listed); zero durations are labelled `unresolved` | No phase-0 repeat is rejected |
| Peaks threw without an f16 FMA figure | Missing peak figures are skipped; failed peak runs are ignored | None on M4 |

Did the conclusions move?

- **% of peak per path**: every path drops by the FMA-peak change (×0.94),
  with nothing else moving. Lc0 Small 4.0 → 3.8%, t3 26.4 → 24.7%, BT4
  24.6 → 23.0%, LQO 26.0 → 24.4%, Maia3 6.7 → 6.2%, Maia3 grid 16.0 → 15.0%,
  Centipawn 5.7 → 5.3%. The big-net GEMMs sit at 36-37% of FMA peak (was 38-40%)
  and at ~48% of the corrected plain-WGSL GEMM ceiling (was ~62%).
- **Top-3 ranking: unchanged.** #1 (small-M GEMM) and #2 (fused attention)
  rest on bandwidth and launch numbers, which did not move. #3 (large GEMM
  retile) gets a higher ceiling but keeps its place, because it still needs
  ORT surgery. The corrected tile shows the gain depends on f16 accumulation:
  about 1.7× on the BT4 GEMMs with f16 accumulate (2.2 vs 1.24-1.28 TFLOP/s),
  only about 1.05× with f32 accumulate (1.3-1.4). Subgroup-matrix no longer
  beats plain f16 WGSL at these shapes.
- **Lc0 Small b1 vs b8: unchanged.** A new paired run (ABAB-interleaved in one
  page session, 20 rounds × 3 invokes per arm, 3 page loads at load 15-20)
  gives b1/b8 per-position wall **5.4×** [5.1-5.5] and per-position GPU busy
  **4.6×** [4.58-4.66]. The phase-0 unpaired figure was 4.9×.

The review sweep at load 14-31 is contended. Its per-kernel GPU times confirm
the large-net tables within 2-4% (BT4 q/k/v Gemm 852 vs 870 µs, FFN 1218 vs
1294 µs). The small-kernel paths ran 20-30% slower on the GPU as well (Lc0
Small smolgen compress 298 vs 232 µs; GPU busy per invoke 9.0-13.6 ms against
phase 0's ~7.6), so GPU timestamps on this shared machine are not immune to
contention. Its absolute walls (Lc0 Small 11.0-13.4 ms, BT4 125-156 ms, LQO
123-152 ms, all at load 14-31) are not used anywhere in this doc. Clean wall
times will come from the queued remote runs.

### Second review round

| finding | fix | effect on numbers |
|---|---|---|
| GEMM variants shared one C buffer, so a variant whose grid missed tiles could inherit the previous variant's correct values and pass | C is filled with NaN before every plain and subgroup-matrix variant, and the check scans all of C for unwritten (non-finite) elements before sampling. `peak.html?undersizeGrid=1` drops one workgroup column as a self-test | Self-test: all 18 undersized rows fail. Real runs: 0 of 936 rows invalid over 6 page loads (load 4-5), so no earlier row was passing on inherited values. Ceilings unchanged within run-to-run spread: plain 2.61 (was 2.59), subgroup-matrix 2.78 at t1 b16 FFN1 (2.77 before). BT4-shape f16/f16-accumulate is 2.23-2.61 at best and 1.39-2.03 in the slower batch of three loads, so treat 2.2 as typical, not a floor |
| A failed `--command-gate` set the report's `ok` to false, but roofline still used its repeats | `roofline.mjs` rejects the whole report and lists it under "Rejected" in the markdown | No committed run has a failed command gate |
| Centipawn `v3-subgroups` silently falls back to `v2-shape-k16` without fixed 32-wide subgroups, and the run was recorded as v3 | `candidate.manifest` is the manifest that ran, `requestedManifest` is what was asked for, and a fallback makes the run `ok: false` with an error | On the M4, v3-subgroups really runs v3 (no fallback) |
| Parity reused the timing feeds: Lc0 Small b1 checked only the start position, and b8 lanes skipped the last fixtures, including the only nonzero rule50 position | Every ORT and TVMJS lane checks all 10 fixtures, in physical-batch chunks (last chunk padded) separate from the timed workload | See below |

Full-corpus parity moved two b1 baselines past their limits. The old limits
were 2× a start-position-only baseline, so they were recalibrated to about 2×
the full-corpus baseline:

| lane | full-corpus max-abs | old limit | new limit |
|---|---|---|---|
| Lc0 Small b1 policy / MLH | 0.0996 / 0.966 (start position only: 0.025 / -) | 0.06 / 0.3 | 0.2 / 2 |
| Maia3 b1 move / value logits | 0.641 / 0.031 | 0.5 / 0.01 | 1.3 / 0.06 |

The rest stay inside their existing limits: t1 b8 policy 0.085, t3 0.063,
BT4 0.29, LQO 1.30 (limit 2.0; it was 1.05 on 8 positions), Maia3 grid
0.44, WDL ≤ 0.009 on every Lc0 path. TVMJS: t1 b8/b16 10/10 best moves, prior
0.0063, WDL 0.0022; BT4 9/10 with the same tolerated tie (gap 0.00055);
Centipawn v2 and v3 ≤ 1.4e-6.

The looser b1 limits still reject the broken candidates. A t1 copy that
zeroes the rule50 plane (input plane 109) fails at b1 and b8 on policy (0.53).
Its MLH diff (1.89) is inside the b1 noise limit. The ×1.1 Q-scale copy fails
on policy (0.387) and WDL (0.0071), and the ×1.5 FFN copy fails on all three
outputs. On the old timing feeds, the rule50 copy passed: zeroing the plane
does nothing to the start position. A unit test reproduces this with a fake
model.

### Third review round

| finding | fix | effect on numbers |
|---|---|---|
| ORT parity checked raw logits only: swapping the two leading legal-move logits ([1, 0] → [0, 1]) passes a max-abs of 1 while reversing a 73/27 split | Kept the raw max-abs as a drift tripwire and added `legalPolicyParity` for Lc0 and Maia3: softmax over legal moves only, max \|prob diff\|, same best move unless the reference gap is ≤ tie-epsilon, and top-3 order kept for pairs whose reference gap exceeds it. Limits are ~2× the full-corpus baseline | Every shipped lane passes (table below). A unit test swaps the top-2 legal logits: the raw tripwire passes it, the new gate fails it |
| Maia3 parity fed the same ratings to `elo_self` and `elo_oppo`, and at b1 every fixture ran at 1100/1100 | Each fixture runs under 4 conditions (1100/1900, 1900/1100, 1500/1500, 2300/800): 40 rows | The Maia3 value-logit baseline rises to 0.078, so its limit goes 0.06 → 0.15. A Maia3 copy that swaps the two Elo inputs fails on both outputs (move 5.3, value 3.3) and on the legal policy (prob diff 0.27; 8 best-move flips and 10 top-3 order reversals across 11 of 40 rows). It is also a unit test |
| With `ortGpuOutputs=1` or `ortPreferredOutputLocation=gpu-buffer`, `session.run` returns after ORT's non-waiting Flush and the outputs are never disposed | Rejected: the harness measures the shipped CPU-output path, so a run whose outputs stay on the GPU disposes them and fails | None (no recorded run used these overrides) |
| One `max(f32, f16)` FMA roof was used for every kernel | Separate f32 and f16 roofs and ridges. Each path declares its arithmetic precision (Lc0 and Maia3 ORT and TVMJS Lc0 paths: f16; Centipawn TVMJS: f32), which picks the roof for utilization, % of peak and classification | M4: f32 3.44, f16 3.28 TFLOP/s. **No kernel changes class.** % of FMA peak for the f16 paths rises ×1.05 (e.g. BT4 23.0 → 24.1%, big-net GEMMs 36-37 → 38-39%). Centipawn is unchanged. On a device with a wider f16/f32 gap (3090 probe: 36.7 / 28.4) the split matters |

Legal-move policy over the full corpus (max \|legal prob diff\| / limit; best
moves; flips tolerated as reference near-ties):

| lane | prob diff | limit | best move | tolerated flips (reference gap) |
|---|---|---|---|---|
| Lc0 Small b1 | 0.0062 | 0.015 | 10/10 | - |
| Lc0 Small b8 | 0.018 | 0.04 | 10/10 | - |
| t3 b8 | 0.0072 | 0.015 | 10/10 | - |
| BT4 b8 | 0.013 | 0.03 | 10/10 | - |
| LQO b8 | 0.039 | 0.08 | 8/10 | allowlisted: Rxh8 → Rxa8 (0.070), h1=Q → a1=Q (0.045) |
| Maia3 b1 / grid | 0.026 / 0.024 | 0.05 | 38/40 | the same h1=Q / a1=Q pair (0.013) |

Tie-epsilon is 0.01 for the Lc0 nets, LQO included, and 0.02 for Maia3. LQO's
WebGPU drift (prob diff up to 0.039) flips both near-mirror move pairs in
fixtures 2 and 7. Those two position/move pairs are allowlisted explicitly,
for the best-move and top-3 checks, up to a reference gap of 0.1 (fourth
review round; a blanket 0.08 epsilon would have passed any new 0.535/0.465
swap). A swap of any other pair, or in any other position, fails. The flips
are drift between WebGPU f16 and the WASM reference, not a proven defect. A
native check of LQO on those two positions would settle it.

Fourth review round, second fix: page identity now comes from the adapter
that created the runtime's device (recorded in the profiler's
`requestDevice` wrapper), not from a fresh high-performance
`requestAdapter()`, which can pick the other GPU on a dual-GPU laptop.
The broken t1 copies still fail: the ×1.1 Q-scale copy on legal prob (0.064),
the ×1.5 FFN copy on legal prob (0.13), and the b1 rule50 copy on legal prob
(0.024) as well as raw policy.

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

Source: `lab/kernel-hillclimb/peak.html`, re-measured after the review (FMA
and GEMM kernels changed): three dev-flag page loads at load 7-16
(`results/peak-dev.json`) plus one with default flags at load 16
(`peak-defaultflags.json`). Values are medians, with the min-max across loads
in brackets. Overhead rows are from phase 0, since those kernels did not change.

| measure | value |
|---|---|
| f32 FMA (8 independent vec4 chains, per-lane constants, every invocation stores) | **3.44 TFLOP/s** [3.32-3.46] (pre-review kernel: 3.22) |
| f16 FMA | **3.28 TFLOP/s** [3.11-3.51] (pre-review: 2.72). On M4, f16 brings at most a small ALU-rate gain; the main gain is half the bytes |
| read bandwidth (256 MiB, vec4) | **93 GB/s** [75-99] |
| copy, kernel / `copyBufferToBuffer` | 88 GB/s [76-89] / ~83-90 GB/s (wall) |
| ridge point (FMA / read BW) | ~37 FLOP/byte |
| empty dispatch, GPU time | 0.80-0.86 µs (same pass or own pass) |
| own-pass dispatch, wall | 3.3-6.2 µs |
| CPU encode | 0.6-0.8 µs per pass+dispatch |
| submit round trip / dispatch+copy+`mapAsync` readback | 0.23-0.30 ms / 0.28-0.33 ms |
| naive tiled WGSL GEMM, 2048³ (output-checked) | 1.6-2.2 TFLOP/s (f16 storage + f16 accumulate is fastest); best over shapes **2.59** (1024×1024×256) |
| `chromium-experimental-subgroup-matrix` GEMM, 2048³ (output-checked) | **2.60 TFLOP/s** f16 and f32 (8×8×8 configs; 4×4 tiles per subgroup). Phase 0 read 2.90 with the same kernel, so ±10% is the run-to-run spread under load |

Adapter features with default flags: `shader-f16`, `subgroups`,
`subgroup-size-control`, `timestamp-query`. The adapter `description` is empty
and timestamps are quantized to 65.536 µs.

The `chromium-experimental-subgroup-matrix` feature (f32→f32 and f16→f16, 8×8×8)
only appears with `--enable-unsafe-webgpu`. It is **not available to users**.
Unquantized timestamps need `--enable-webgpu-developer-features`.

GEMM at real model shapes, in TFLOP/s. Each cell is the max across three page
loads, and every cell passed the output check. "naive" means my 64×64-128×128
register-blocked tile (race-free since the review), with no tuning:

| shape (M×N×K) | naive f32 | naive f16/f16-acc | naive f16/f32-acc | subgroup-matrix f16 / f32 | shipped kernel achieves |
|---|---|---|---|---|---|
| t1 b1 64×256×256 | 0.37 | 0.52 | 0.39 | 0.86 / 0.66 | ORT 0.22 |
| t1 b16 1024×256×256 | 1.40 | 2.23 | 1.46 | 2.46 / 1.99 | - |
| t1 b16 1024×1024×256 | 1.67 | 2.60 | 1.53 | 2.77 / 2.01 | - |
| Centipawn 1024×384×128 (f32 path) | 1.31 | 2.03 | 1.29 | 2.48 / 1.96 | TVMJS 0.27-0.29 (f32) |
| BT4 b8 512×1024×1024 | 1.38 | 2.23 | 1.29 | 2.17 / 2.40 | ORT 1.24 |
| BT4 b8 512×1536×1024 | 1.31 | 2.16 | 1.31 | 1.19 / 2.42 | ORT 1.28 |
| BT4 b8 512×1024×1536 | 1.35 | 2.22 | 1.40 | 1.64 / 1.75 | ORT 1.24 |

ORT's shipped Gemm runs f16 storage at about 1.24-1.28 TFLOP/s. That is close
to my naive f16/f32-accumulate tile; only f16 accumulation gives the ~1.7×.

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
    matmul dispatch to its ONNX node by grid shape, bound bytes and node count.
    It leaves the dispatch unassigned when that is ambiguous (see Changes
    after review).
  - TVMJS kernels take their MACs from the shape-k16 probe JSON.
- Bytes are the sizes of the bound buffers. That is an upper bound on
  compulsory traffic, and there are no DRAM counters.
- "Bound" labels per kernel group:
  - `launch`: median dispatch under 8 µs.
  - `compute` / `bandwidth`: at least 50% of the attainable roof.
  - `underused`: neither roof is reached, so the limit is occupancy or kernel
    quality.

Full output: `node scripts/kernel_hillclimb/roofline.mjs --md ...` (the artifacts
are local and gitignored). The tables below are the phase-0 measurements
(load 7-15) re-analysed with the corrected matcher and the post-review peaks.

### Per invoke

| path | b | wall ms | GPU kernel ms | GPU share of wall¹ | non-kernel ms¹ | dispatches / passes / submits | GFLOP | TFLOP/s at wall | % FMA peak | matmul TFLOP/s | kernel time by bound² |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Lc0 Small** (ORT) | 1 | **9.9** [8.9-10.1] | 8.5 | 77% | 2.3 | 356 / 23 / 26 | 1.29 | 0.13 | **3.8%** | 0.17 | underused 86%, launch 10% |
| Lc0 Small at b8 (what-if) | 8 | 16.1 [14.2-16.3] | 13.6 | 86% | 2.1 | 356 / 23 / 26 | 10.3 | 0.64 | 19% | 0.93 | underused 91% |
| **t3** (ORT) | 8 | **47.9** [43.8-51.1] | 60.2³ | 92% | 4.9 | 537 / 34 / 37 | 40.9 | 0.85 | 25% | 0.82 | underused 92%, bw 8% |
| **BT4** (ORT) | 8 | **166.5** [162.6-172.4] | 138.2 | 95% | 6.3 | 537 / 34 / 37 | 132.1 | 0.79 | 23% | 1.11 | underused 94%, bw 5% |
| **LQO** (ORT) | 8 | **84.8** [83.9-89.5] | 75.6 | 95% | 4.0 | 527 / 33 / 36 | 71.2 | 0.84 | 24% | 1.11 | underused 93%, bw 7% |
| **Maia3** move (ORT) | 1 | **11.5** [8.6-12.9] | 8.4 | **60%** | **4.9** (2.9 idle inside GPU envelope) | 817 / 53 / 59 | 2.48 | 0.22 | 6.2% | 0.42 | underused 75%, **launch 25%** |
| **Maia3** Elo grid (ORT) | 11 | **53.0** [52.5-53.6] | 60.6³ | 91% | 5.2 | 817 / 53 / 59 | 27.3 | 0.52 | 15% | 0.57 | underused 88%, bw 8% |
| **Centipawn** (TVMJS v2) | 16 | **11.1** [10.2-17.3] | 9.6 | 86% | 1.5 | 161 / 161 / 3 | 2.03 | 0.18 | **5.3%** | 0.33 | underused 79%, bw 21% |
| Centipawn v3-subgroups | 16 | 10.1 | 9.3 | 92% | 0.8 | 161 / 161 / 3 | 2.03 | 0.20 | 5.8% | 0.32 | - |
| TVMJS Lc0 t1 b8 / b16 | 8/16 | 22.8 / 24.9 | 16.1 / 17.0 | 71% / 69% | 6.7 / 7.8 | 229 | 10.3 / 20.7 | 0.45 / 0.83 | 13% / 24% | 0.72 / 1.37 | underused |
| TVMJS Lc0 BT4 b8 | 8 | 198.7 | 155.4 | 78% | 43.3 | 341 | 132.1 | 0.67 | 19% | 0.90 | underused |

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
| Gemm (q/k/v/out) | 8x2x1 | 43 | 38 | 1.63 | 64×256×256 | 221 | 5.2 | 6.4 | 6 | 43 | underused |
| Gemm (FFN down) | 8x2x1 | 10 | 134 | 1.34 | 64×256×1024 | 251 | 5.1 | 7.3 | 6 | 49 | underused |
| Gemm (FFN up) | 32x2x1 | 10 | 42 | 0.42 | 64×1024×256 | 803 | 16.5 | 23.3 | 18 | 49 | underused |
| Gemm (smolgen expand) | 64x1x1 | 10 | 41 | 0.41 | 1×2048×256 | 26 | 25.8 | 0.7 | 28 | 1 | underused |
| MatMul (smolgen per-head) | 1x2x1 | 10 | 36 | 0.36 | 64×32×256 | 29 | 1.5 | 0.8 | 2 | 20 | underused |
| MatMul (smolgen weight gen) | 128x1x1 | 10 | 36 | 0.36 | 8×4096×256 | 471 | 60.8 | 13.7 | 65 | 8 | bandwidth |
| Gemm (head) | 4x1x1 | 1 | 233 | 0.23 | 1×128×2048 | 2 | 2.3 | 0.1 | 2 | 1 | underused |

FFN up and FFN down have the same FLOPs. Down runs on 16 workgroups at 134 µs;
up runs on 64 workgroups at 42 µs. ORT's Gemm does not split K, so at M ≤ 64 the
grid starves a 10-core GPU. The M=1 GEMVs run on 4-8 workgroups and reach
2-5 GB/s.

**BT4, b8** (kernel sum 138 ms). t3 and LQO show the same pattern at d=512 / 768
(LQO, now attributable: q/k/v/out 512×768×768 at 1.21 TFLOP/s, FFN 1.22-1.23,
smolgen weight gen 1.20, all 35-36% of FMA peak).

| kernel | grid | n | µs | ms/inv | M×N×K | GFLOP/s | GB/s | %FMA | AI | bound |
|---|---|---|---|---|---|---|---|---|---|---|
| Gemm (q/k/v/out) | 32x16x1 | 63 | 870 | 54.8 | 512×1024×1024 | 1235 | 4.8 | 35.8 | 256 | underused |
| Gemm (FFN down) | 32x16x1 | 16 | 1294 | 20.7 | 512×1024×1536 | 1244 | 4.9 | 36.1 | 256 | underused |
| Gemm (FFN up) | 48x16x1 | 16 | 1259 | 20.1 | 512×1536×1024 | 1279 | 5.0 | 37.1 | 256 | underused |
| MatMul (smolgen weight gen) | 128x8x1 | 15 | 438 | 6.6 | 32×4096×256 ×8 | 1227 | 9.9 | 35.6 | 124 | underused |
| Softmax | 16384x1x1 | 15 | 362 | 5.4 | - | - | 11.6 | - | - | underused |
| Transpose | 8192x1x1 | 61 | 83 | 5.1 | - | - | 25.2 | - | - | underused |
| Gemm (smolgen compress) | 8x1x1 | 15 | 333 | 5.0 | 8×256×2048 | 25 | 3.3 | 0.7 | 8 | underused |
| Gemm (smolgen expand) | 256x1x1 | 15 | 201 | 3.0 | 8×8192×256 | 167 | 21.7 | 4.9 | 8 | underused |

**Maia3** (kernel sum: 8.35 ms at b1, 60.6 ms at b11)

| kernel | b | grid | n | µs | ms/inv | op | GFLOP/s | GB/s | bound |
|---|---|---|---|---|---|---|---|---|---|
| Einsum `bhi,oi->bho` (attention-bias projection) | 11 | 11264x1x1 | 8 | 2402 | **19.2** | 176×4096×128 | 77 | 1.3 | underused |
| MatMul (FFN up / down, q/k/v) | 11 | 16-48x22x1 | 32 | 600-880 | 19.4 | 704×{512,1024,1536}×{512,1024} | 1180-1260 | 6.7-8.3 | underused |
| Einsum | 1 | 1024x1x1 | 8 | 148 | 1.18 | 16×4096×128 | 114 | 8.0 | underused |
| Gemm | 1 | 4x1x1 | 8 | 185 | 1.48 | 1×128×2048 | 3 | 2.9 | underused |
| Add (shape arithmetic) | 1 | 1x1x1 | 256 | 2 | 0.57 | - | - | - | launch |
| Gather / Concat / Transpose (shape + layout) | 1 | 1x1x1-512x1x1 | 148 | 2-6 | 0.58 | - | - | - | launch |

**Centipawn TVMJS v2, b16** (kernel sum 9.6 ms; f32)

| kernel | grid | n | µs | ms/inv | GFLOP/s | GB/s | %FMA | bound |
|---|---|---|---|---|---|---|---|---|
| softmax | 8192x1x1 | 6 | 298 | 1.79 | - | 21.1 | - | underused |
| fused_matmul3_add4 | 32x12x1 | 5 | 353 | 1.76 | 285 | 8.0 | 8.3 | underused |
| fused_matmul8_add2_add1 | 32x4x1 | 6 | 252 | 1.51 | 266 | 13.0 | 7.7 | underused |
| fused_matmul7_…_erf_… (FFN + GELU) | 32x8x1 | 6 | 246 | 1.48 | 272 | 11.2 | 7.9 | underused |
| matmul5 | 2x128x1 | 6 | 146 | 0.88 | 115 | 32.3 | 3.3 | underused |
| fused_matmul4_add5 | 2x256x1 | 6 | 119 | 0.71 | 141 | 75.0 | 4.1 | bandwidth |
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
| t3 / BT4 / LQO b8 | **Compute-side kernel quality.** The GEMMs reach 35-37% of FMA peak, at AI 170-256 (well past the ridge). Softmax, transpose and the smolgen GEMV take another 10-15% | Moderate: ~1.7× on GEMMs with plain WGSL and f16 accumulation (output-checked untuned tile 2.16-2.23 vs ORT 1.24-1.28); ~1.05× if f32 accumulation is required. Subgroup-matrix adds nothing at these shapes |
| Maia3 b1 | **Launch/sync-bound + small-M.** 817 dispatches; 25% of kernel time is launch-bound shape ops; 40% of wall is non-kernel, including 0.8-2.9 ms GPU idle inside the envelope | Large, and mostly reachable with graph fixes |
| Maia3 grid b11 | **One pathological kernel.** The Einsum takes 32% of kernel time at 77 GFLOP/s; the grid's MatMuls run at 1.18-1.26 TFLOP/s like the big Lc0 GEMMs | ~19 ms per call, from a graph rewrite |
| Centipawn (TVMJS) | **Kernel quality (f32 matmul schedule) + softmax.** Matmuls reach 7-8% of FMA; 1,440 evals/s already | ~2× GPU time. Delivered through TVM schedule work, not ORT |
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
| 3 | **Large f16 GEMM retile** (512×{768,1024,1536}×K; 128×128 tiles, f16 storage, f16 or f32 accumulate decided by parity) | 80-86% of t3/BT4/LQO kernel time; 1.13-1.28 TFLOP/s today | Plain WGSL with f16 accumulation: my untuned, output-checked tile already reaches 2.16-2.23 at these shapes. **~1.7× on GEMMs → BT4 ~166 → ~115 ms**, if f16 accumulation passes parity. With f32 accumulation the same tile reaches only 1.29-1.40, about ORT's level. Subgroup-matrix (flag-only) is no better at these shapes | WASM parity at BT4's policy threshold; tie-epsilon search parity. f16 accumulate needs its own numeric gate. Highest engineering cost: replacing ORT's Gemm means a patched ORT or a custom runtime |
| 4 | Centipawn f32 matmul + softmax schedule (TVM dlight config or hand WGSL) | 6.0 ms matmul + 1.8 ms softmax of 9.6 | The naive f32 tile gets 1.31 TFLOP/s at the Centipawn qkv shape against 0.27-0.29 shipped. GPU 9.6 → ~4.5 ms per b16 invoke | `tvmjs-centipawn --gate` (≤ 1e-3), plus the search-parity protocol in the subgroups doc |

Is hand-written WGSL worth it?

- **Yes, for #1.** It has the largest relative headroom (10-50× per kernel), it
  sits on the default engine's hot path, and the ceiling is a bandwidth number
  this page measured.
- **Conditionally, for #2-#3.** The gains are 10-40% on the big nets, and #3's
  gain exists only with f16 accumulation. They need
  ORT surgery or a custom runtime, and the biggest step (subgroup-matrix) is not
  shippable today.

## 6. Non-kernel levers

Ranked by expected effect per unit of work:

1. **Batch the Lc0 Small search.** The physical batch is 1 today. A b8 invoke
   costs 16.1 ms against 9.9 ms at b1: 2.0 vs 9.9 ms per position, **4.9× NN
   throughput** with no kernel work. The paired re-check (ABAB in one page
   session, candidate `ort-lc0-small-ab-b8`) gives 5.4× per-position wall
   [5.1-5.5] and 4.6× per-position GPU busy. The cost is search-side: collisions and
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
  FLOPs are attached by matching grid shape, bound bytes and node count.
  Bound bytes are buffer sizes that ORT's allocator rounds up into buckets (up to
  ~50% over the tensor bytes on Maia3-grid), so they also inflate GB/s. Classes
  that stay ambiguous are left without FLOPs. On every path here these are only
  head GEMVs under 20 KFLOP.
- **Overlap inside a pass.** Split mode serializes dispatches, so split sums are
  upper bounds. Pass mode has the true structure, but it covers 23-53 passes, not
  individual kernels.
- **A quiet machine.** Load was 7-15 in phase 0 and 14-31 in the review
  sweep, and GPU contention cannot be locked out. GPU timestamps are affected
  too (small-kernel paths ran 20-30% slower on the GPU in the review sweep). Wall repeats spread by up to ±15%, and the profiled phases of t3
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
for c in ort-lc0-small ort-lc0-small-b8 ort-lc0-small-ab-b8 ort-t3 ort-bt4 ort-lqo ort-maia3 ort-maia3-grid tvmjs-centipawn; do
  $LOCK $c -- node scripts/kernel_hillclimb/bench.mjs --candidate $c --gate --repeat 3 $FLAGS --out artifacts/kernel-hillclimb/results/$c.json
done
# costs: .venv-onnx/bin/python -I scripts/kernel_hillclimb/onnx_cost.py <model> --batch <b> --out artifacts/kernel-hillclimb/cost/<name>.json
node scripts/kernel_hillclimb/roofline.mjs --peak artifacts/kernel-hillclimb/results/peak-dev.json --md artifacts/kernel-hillclimb/roofline.md
```

All gates passed, in phase 0 and again in the review sweep under the
corrected gates:

- ORT WebGPU vs the reference model on ORT WASM is within the preset
  `parityMaxAbs` over all 10 fixtures (second review round; the b1 limits were
  recalibrated). The policy-logit max-abs was 0.0996 (t1 b1), 0.085 (t1 b8),
  0.063 (t3), 0.29 (BT4) and 1.30 (LQO). WDL was ≤ 0.009 on every Lc0 path.
  The legal-move policy gate passes on every lane (third review round, table
  above), with Maia3 over 10 fixtures × 4 Elo conditions.
- TVMJS Centipawn vs ORT: ≤ 1.1e-6 policy and ≤ 1.4e-6 WDL, same move sets.
- TVMJS Lc0 t1: 10/10 best moves at b8 and b16, prior 0.0063, WDL 0.0022.
- TVMJS Lc0 BT4: 9/10 best moves, with 1 flip whose reference prior gap is
  0.00055 (≤ 0.01). Prior 0.012, WDL 0.0024.
- The gates reject deliberately broken candidates. In the browser, scaled-weight
  and rule50-zeroed t1 copies and an Elo-swapped Maia3 copy fail ORT parity (above). In `tests/kernel_hillclimb.test.mjs`, unit
  tests fail scaled or NaN outputs, NaN or shifted WDL, missing moves,
  flattened policies, dropped profiles and ambiguous byte matches.
