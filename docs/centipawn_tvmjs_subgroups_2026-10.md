# Centipawn TVMJS subgroup rebuild — October 2026

## Decision

**Stage `f32/v3-subgroups` as experimental; do not promote.** The compiler-only change passes numerical/search/lifecycle checks on this Apple M4 and lowers complete-invoke GPU busy time by approximately 4–6%. Search wall improvement is not sufficiently repeatable across fixtures: the longer-search AB run regresses in aggregate, and some matched fixture distributions regress. The required non-Apple/topology evidence is also absent. The registry, stable channel, release manifests, and automatic runtime selection remain on **v2-shape-k16**, with the existing ORT fallback.

This is not a native subgroupAdd/Max implementation. The four changed reduction bodies use **subgroupShuffleDown**, plus workgroup memory for cross-subgroup partials. No hand-written matmul or shader rewrite was made.

## Identity and exact recipe

All measurements: Apple M4, adapter `apple / metal-3`, fixed subgroup min/max **32/32**, headless Chrome **149.0.0.0**, same-origin cross-origin-isolated production build on port **5195**, physical f32 batch **16**, search leaf batch **16**.

| Input/artifact | SHA-256 |
|---|---|
| Production ONNX, 4,464,375 bytes | `de2dd68518f6e2ed162da5ee27c67a5a4795567d6e39972e3705824572b7660e` |
| Fixed-b16 simplified ONNX, 16,246,184 bytes | `c4adcf876904baf3381727720391ffba2fa2981e0395334e81af9424815b0506` |
| Promoted v2 model wasm, 20,192,787 bytes | `fd3d8635c8368338c3a0d1b57a725a70b4f8a74f8d130ee831848c641cba75b5` |
| Experimental v3 model wasm, 20,194,291 bytes | `8bd117093f1914d60a5e7b89dbcb768c5a91130c346246fa7e9ebd763213b6c9` |
| Rebuilt subgroup-disabled control wasm | `6160beeb253dfa9165acaf7d1554798fd46ac7016bd9eebfff62f8c1c2b07373` |
| Unchanged TVMJS JS bundle | `e5225b46a98cc6c5ab133233a67f830e856ac0a23b01022768e7f94ed58a8111` |
| Unchanged TVMJS runtime wasm | `f5d0b00f63ee878a2144f576ab41825a7c3d13a79621782dcc98b14f4d1c58e5` |

Compiler: **`94398e51fbb8bf9022da85d83e25f914ef033110`**, existing `build-tvmjs` library, plus v2's exact storage-rewrite patch, SHA-256 `33b73d6cfa2a198a792e8642332c51ba236a8547c1e8c854a6d8301c7b31098f`. The shared compiler checkout was **read-only**. No compiler/runtime rebuild, precision change, parameter detachment, fusion-depth change, or schedule configuration change was introduced. Emscripten is **5.0.7-git**, as in v2.

Durable recipe: [`scripts/build_centipawn_tvmjs_subgroups.sh`](../scripts/build_centipawn_tvmjs_subgroups.sh), using the existing [`lc0_tvm_whole_onnx_probe.py`](../scripts/lc0_tvm_whole_onnx_probe.py) and staging script. It checks the compiler commit/patch and original/fixed ONNX hashes. The fixed graph is reconstructed with **onnx 1.21.0 / onnxsim v0.6.4**, `overwrite_input_shapes={tokens:[16,64,24], attack_summary:[16,64,28]}`: exactly **342 nodes / 139 initializers**, matching v2's probe.

Preserved TVM flags:

- wasm32 LLVM host `{"kind":"llvm","mtriple":"wasm32-unknown-unknown-wasm"}`;
- `--dtype '{"tokens":"int32"}'`, initializer int64→int32 conversion;
- trusted nonnegative constant/runtime gather indices, sanitized ONNX names;
- `--max-fuse-depth 6 --dlight --export-tvmjs-wasm`, embedded f32 parameters;
- shape selector `static_large`, `min_macs=8000000`, `j_multiple=32`, `k_multiple=16`, override **micro_size_k=16**. All other selected/default Matmul.Config fields match v2.

The sole semantic build input difference is target **`{"kind":"webgpu","supports_subgroups":true}`** rather than the subgroup-disabled target. TVM canonicalizes thread_warp_size from **1 to 32**. `--capture-module-sources` is an additional **diagnostic** flag for both rebuilt legs, not a schedule/precision change. Captured sources remain local; staged probe metadata retains their sizes/hashes, not machine paths.

The control is not byte-identical to the historical v2 wasm: the compiler emits unused CSE `let` declarations in nondeterministic map order. Inspection of the shader strings **embedded in all three wasm files**, removing only these unused declarations (and rejecting any live CSE reference), proves all **46 control shader arithmetic bodies reproduce v2**. Between control and v3, captured host LLVM is byte-identical. All non-reduction shader bodies reproduce v2 after the same normalization. Thus a changed wasm hash is not evidence of a changed matmul schedule.

## Emitted WGSL

[`wgsl-inspection.json`](../lab/centipawn-tvmjs-subgroups-results/wgsl-inspection.json) records all 46 kernels, hashes, workgroup geometry, and normalized comparisons. The four actual changed bodies are retained as inspectable [WGSL](../lab/centipawn-tvmjs-subgroups-results/wgsl/). The inspection script verifies their bytes occur in the staged wasm; tests repeat that check.

| Kernel | Meaning | Workgroup | ShuffleDown sites | Dispatches/invoke |
|---|---|---|---:|---:|
| `softmax_kernel` | attention rows, width 64 | 64×1×1 | 12 | 6 |
| `layer_norm_kernel` | attack-summary normalization, width 28 | 64×1×1 | 12 | 1 |
| `layer_norm1_kernel` | encoder normalization, width 128 | 64×1×1 | 12 | 12 |
| `fused_matmul12_add9_kernel` | small GEMV/value-head reduction | 64×1×1 | 6 | 1 |

All 46 shaders acquire `enable subgroups;`; **only these four execute subgroup operations**. There is no subgroupAdd/Max, subgroup f16, or subgroup matrix operation. The dominant tiled matmuls keep their schedules/arithmetic; the automatic GEMV reduction change above is part of flipping the target flag, not a manual matmul rewrite. Partial input masks initialize neutral values before the collectives; the two 32-lane groups stage partials through workgroup memory.

## Runtime safety and fallback

`SquareformerTvmjsWebgpuEvaluator.create` permits the `subgroups` request only for a manifest declaring **version v3-subgroups, requiredFeatures containing subgroups, subgroupSize 32**. Mismatched contracts reject. Before requesting a device or fetching the v3 model, it requires feature availability and **adapter.info.subgroupMinSize == subgroupMaxSize == 32**; absent or variable metadata selects the existing v2 manifest. V2 never requests `subgroups`. Fallback resolves the known sibling v2 manifest relative to v3, preserving configured CDN/same-origin path prefixes; the first structured review identified and corrected the original root-relative URL bug, with regression coverage. The selected manifest URL is exposed, and the browser factory's result/audit uses the **selected** artifact/config identity, not an incorrectly labelled v3 request after a v2 fallback.

The final production-browser [routing probe](../lab/centipawn-tvmjs-subgroups-results/routing.json) verifies candidate→v3 with `['subgroups']`, simulated variable size32–64→v2 with no features, and default auto→v2 with no features, including correct selected config IDs and a real evaluation/destruction in each case. This simulates adapter metadata, not a second physical adapter.

Existing device limits, SRI/wasm hash verification, error/device-loss handling, deferred destruction, and the outer ORT fallback are preserved. No runtime default or release/channel change was made. The new wasm files use the **existing Centipawn Git LFS rule** in `.gitattributes`; no new large binary mechanism was added. Two narrowly scoped whitespace attributes preserve the exact inherited vendor JS CRLF/trailing spaces and patch context blank lines without changing their integrity hashes.

A fixed subgroup size does **not** establish portable local-invocation/lane ordering for this compiler's index-based partials. Model parity demonstrates this topology works on this tested M4/Chrome path, not on every size-32 device. This is another reason not to promote without a non-Apple datapoint and actual-pipeline topology validation.

## Parity and lifecycle

The existing Centipawn whole-output parity page was **production compiled**, with feature/size checks added for the candidate. Synthetic seeded b16 input, ORT strict WebGPU, all finite/expected lengths, original **max absolute drift <= 0.001** gate:

| Raw output | v2 vs ORT max abs | v3 vs ORT max abs |
|---|---:|---:|
| policy, 327,680 floats | 5.15e-5 | 5.15e-5 |
| WDL, 48 floats | 2.67e-5 | 3.05e-5 |
| q, 16 floats | 1.43e-6 | 9.54e-7 |
| hidden, 131,072 floats | 1.93e-5 | 1.72e-5 |

Retained: [`raw-parity-v2.json`](../lab/centipawn-tvmjs-subgroups-results/raw-parity-v2.json), [`raw-parity-v3.json`](../lab/centipawn-tvmjs-subgroups-results/raw-parity-v3.json). Their one-sample timings are **not** used as performance evidence.

The production evaluator additionally checks all legal-policy probabilities and WDL probabilities on all **10 FEN-only fixtures**, padded to physical b16:

| Comparison | max legal-policy abs | max WDL abs |
|---|---:|---:|
| v3 vs v2 | 5.81e-7 | 2.41e-7 |
| v2 vs ORT | 7.05e-7 | 1.12e-6 |
| v3 vs ORT | 1.08e-6 | 1.36e-6 |

Search uses the production **`searchRoot`** path, fixed requested visits, leaf batch16, temperature0, PV depth8, no reused tree. V2/v3 **best move and entire extracted PV match on every retained repetition**: 4 fixtures × 7 repeats × 2 runs at 128 visits; 4 × 11 × 2 at 256. ORT independently matches the best move/PV once per fixture per run. The remaining six fixtures also pass v2/v3/ORT best-move/PV parity at 256 visits: en passant for both sides, promotions for both sides, check evasion, nonzero rule50. All ten are therefore covered at 256 visits. [Edge parity evidence](../lab/centipawn-tvmjs-subgroups-results/edge-parity.json) is not pooled into benchmark results.

Lifecycle: evaluation, destruction during an in-flight evaluation, rejecting new work after destruction, recreate/evaluate/destroy, for both variants. The [final hardened smoke](../lab/centipawn-tvmjs-subgroups-results/final-smoke.json) additionally asserts the evaluation has not settled at destruction and awaits each device's `lost.reason == 'destroyed'`. All completed browser runs report lifecycle PASS and **zero GPU errors/device loss** (intentional device destruction excluded). Focused tests cover missing feature, absent/variable subgroup size, descriptor feature requests, invalid manifest contracts, and pipeline-init cleanup. No tolerances were loosened. This is FEN-only/search parity, not exhaustive history/repetition/near-tie portability validation.

## Locked measurement method

All GPU/search/parity traffic was inside the supplied machine-wide **benchlock**. A locked coordinator ran `node scripts/serve_isolated_static.mjs dist-client 5195` and held the mutex across the native `agent_browser` calls, until the browser closed and the coordinator's release marker was written. Only port 5195 and unique sessions **centipawn-subgroups-5195** / **centipawn-subgroups-5195-final** were used. No direct browser-driving bash command or dev server was used. The static server was stopped after each locked window.

Build: `BUILD_SCOPE=product npm run build:client`, then `node scripts/build_centipawn_subgroups_lab.mjs` production-compiles the two diagnostic HTML entries and overlays only their compiled lab assets/fixtures into `dist-client`. The lab imports the actual production evaluators/search, not a modified search implementation. No file-changing build/test jobs ran concurrently with an active timed run in this worktree.

Each main run:

- **31 adjacent A/B invoke pairs**, alternating AB/BA; repeated runs reverse the starting order. Five untimed evaluator warmups per variant.
- Complete b16 evaluator invoke, including input encoding/upload, model, policy/WDL readback, legal mapping. GPU measurements exclude query allocation/resolution/readback and output mapping from the GPU interval; wall time excludes query allocation/resolution but includes normal evaluator work.
- Actual GPU compute-pass **timestamp queries**: **161 passes/invoke**. `busyMs` sums end−start for every pass; `envelopeMs` is first start→last end, including intervening GPU gaps. These are not CPU-submit duration and not a wall-clock substitute.
- Timestamp-query is requested **only by lab instrumentation**, equally in both legs. Search leaves timestamp writes/resolution disabled. Instrumentation does not rewrite WGSL; extra API wrappers remain on both devices.
- An initial pilot rejected zero-duration individual passes and was discarded; the retained runs explicitly account for those quantized zeros. Chrome timestamps visibly quantize at **65.536 µs**. Individual tiny passes can be zero; zero counts are retained, negative or nonpositive *whole-invoke* intervals reject. This limits per-kernel conclusions, especially GEMV and the single width-28 layernorm. Do not interpret zero as zero kernel cost. Query quantization was not disabled.
- Search starts with a fresh root, one untimed warm search per fixture/variant, then **7 or 11 paired repetitions**. Aggregate search rows mix four fixtures, so paired and per-fixture results are reported rather than treating a pooled median ratio as the promotion statistic.

Raw samples, shader identities, model/manifest hashes, parity checks and search results: [`lab/centipawn-tvmjs-subgroups-results/`](../lab/centipawn-tvmjs-subgroups-results/). `summary.json` is regenerated by `node scripts/summarize_centipawn_subgroups.mjs` and recomputed in tests. Run 1 predates the starting-order/bundle-response hash fields; runs 1–2 predate per-kernel attribution. Runs 1–2 used the same v3 binary but the pre-sanitization probe/manifest metadata; only machine-path metadata was removed before runs 3–4. Later `executedBundle` hashes are the top-level static JS response fetched after completion, not a complete browser-source attestation of every imported module. Static files were not modified during a run.

## Results

**Milliseconds, median [p10–p90].** No pooling across visit budgets/runs.

| Run / order / visits | v2 GPU busy | v3 GPU busy | v2 invoke wall | v3 invoke wall |
|---|---:|---:|---:|---:|
| 1 / AB / 128 | 13.304 [6.291–14.418] | 12.517 [6.881–13.631] | 17.965 [11.245–19.575] | 17.215 [11.470–18.600] |
| 2 / BA / 128 | 13.304 [6.095–14.025] | 12.648 [5.767–13.238] | 17.920 [10.780–19.210] | 17.120 [10.095–18.160] |
| 3 / AB / 256 | 13.238 [5.571–15.139] | 12.714 [5.571–14.483] | 18.860 [10.660–21.460] | 18.345 [10.440–20.550] |
| 4 / BA / 256 | 13.304 [5.439–14.221] | 12.517 [5.112–13.369] | 17.505 [9.505–18.690] | 16.770 [9.840–17.845] |

GPU envelope medians v2/v3: **14.090/13.435, 14.680/13.894, 14.418/14.221, 14.156/13.369 ms**. GPU busy median savings: **5.9%, 4.9%, 4.0%, 5.9%**.

| Run | Search samples/variant | v2 search wall | v3 search wall | Paired saving %, median [p10–p90] |
|---|---:|---:|---:|---:|
| 1, 128 visits | 28 | 117.232 [81.880–148.115] | 103.548 [78.345–148.700] | 2.32 [−15.50–21.84] |
| 2, 128 visits | 28 | 146.747 [85.280–151.115] | 142.538 [77.540–147.100] | 3.41 [−14.89–9.08] |
| 3, 256 visits | 44 | 140.740 [113.695–202.515] | **143.048 [111.655–224.420]** | 0.64 [−23.78–15.72] |
| 4, 256 visits | 44 | 219.118 [175.630–287.590] | 210.995 [165.730–279.130] | 3.12 [−41.76–30.71] |

Positive paired saving is `100*(1-v3_wall/v2_wall)`. Per-fixture **median paired** saving percentages:

| Fixture | Run 1 | Run 2 | Run 3 | Run 4 |
|---|---:|---:|---:|---:|
| startpos | −1.41 | 3.24 | −7.10 | 3.23 |
| black-to-move-after-e4 | 8.18 | 3.51 | 7.41 | 10.14 |
| castling-rights | 5.69 | 4.48 | 0.15 | 15.86 |
| black-castling-rights | 0.51 | 2.97 | −0.26 | **−16.96** |

The wide within-run and between-run spreads remain despite interleaving/mutex; this is **not a quiet-machine latency guarantee**. In particular, the pooled median's large run-1 apparent win and run-4 apparent win must not replace the paired statistic or conceal fixture regressions. A repeatable evaluator gain is plausible; a repeatable complete-search promotion win is **not established**.

Changed-kernel attribution, median **sum over dispatches per complete invoke**, run 3 / run 4, v2→v3:

- attention softmax (6 dispatches): **1.507→1.180 / 1.704→1.245 ms**;
- width128 layernorm (12): **0.590→0.393 / 0.590→0.393 ms**;
- width28 layernorm (1): **0.066→0.066 / 0.066→0.000 ms**, below reliable resolution;
- small GEMV (1): **0→0 / 0→0 ms**, unresolvable, **no speedup claim**.

Most GPU time remains in unchanged matmul/projection work. The spike's native reduction speedups cannot be substituted for these compiler-emitted shuffle measurements.

## Reproduce / remaining gates

```sh
# Point these at existing read-only installations; outputs stay in this worktree.
TVM_SRC=/path/to/tvm-webgpu-src TVM_ENV=/path/to/tvm-mlc-py313 \
ONNX_PYTHON=/path/to/onnxsim/python bash scripts/build_centipawn_tvmjs_subgroups.sh
BUILD_SCOPE=product npm run build:client
node scripts/build_centipawn_subgroups_lab.mjs
# Under the supplied benchlock, serve dist-client on assigned port 5195.
# Native agent_browser, unique session: open then click Run locked gates.
# /lab/centipawn-tvmjs-subgroups.html?order=ab&visits=256&searchRepeats=11
# Repeat with order=ba; default repeats=31. Save window.subgroupsResult.
# Remaining edge fixtures: searchFixtureOffset=4&searchFixtureCount=6.
# Existing all-output gate: /lab/centipawn-tvmjs-webgpu-bench.html
#   ?manifest=<v2-or-v3-manifest>&warmup=2&repeats=1&autorun=1
node scripts/summarize_centipawn_subgroups.mjs
node --experimental-strip-types --test tests/centipawn_tvmjs_evaluator.test.mjs \
  tests/centipawn_tvmjs_subgroups.test.mjs tests/centipawn_subgroups_evidence.test.mjs
npm run typecheck
npm test
git diff --check
```

Closeout gates: typecheck **446 files, 0 errors/0 warnings**; focused tests **32/32**; full `npm test` **790 tests, 787 passed, 3 existing skips, 0 failures**; production builds, shell/Python inspection checks and `git diff --check` pass. Browser sessions and port5195 servers are closed. Structured branch review is reported in the final task closeout, not treated as a performance gate.

Before reconsidering promotion: longer/stable matched search distributions (including near ties/history/reuse), actual shader lane-topology evidence, and at least one non-Apple size32 adapter. No subgroup size32 routing claim is made for untested hardware. **The default stays unchanged.**
