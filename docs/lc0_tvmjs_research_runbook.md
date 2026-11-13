# LC0 TVMJS/WebGPU research runbook

Status: **PROMOTED to visible non-default option 2026-06-10** (release-owner
decision, self-hosted scope). The whole-model TVMJS/WebGPU runtime is now
listed in non-v0 Arena/Analysis LC0 runtime selectors. The v0 deploy profile
forces ONNX, removes non-ONNX options and ignores TVMJS query opt-ins
(`src/lc0/browserLc0Runtime.ts`). ORT ONNX/WebGPU remains the default/fallback.
LC0 whole-model TVMJS stays page-level, outside the shared NN registry and
evaluator (`src/nn/runtimeRegistry.ts`, `src/nn/browserRuntimeEvaluator.ts`).
This statement is LC0-specific; other families are out of scope here. Accepted caveats at promotion: no non-Apple GPU datapoint yet and
no public hosting/cache policy (artifacts remain local/gitignored). Evidence
basis: t1 94/94 fixed-suite search matches vs ORT f16 + Stockfish cp deltas
0/0/0; t3 native 8/8 strict; BT4-it332 gated with tie tolerance.
`scripts/check_lc0_tvmjs_research_only.mjs` (v2) enforces the new policy.

## Current operational policy (source-checked 2026-11-13)

Use this runbook for LC0 export, staging and evidence commands. Self-hosted
non-v0 selection uses `lc0Runtime=whole-onnx-webgpu` (aliases include
`tvmjs-webgpu`). It is visible but non-default, not a public artifact release or
a default-runtime promotion. Generated bundles stay local/ignored unless a
release owner approves a specific publication. Public-release/default blockers
are exact bundle provenance, host/cache canaries, agreed drift/search gates,
startup/footprint budgets and broader device coverage. Do not re-open completed
BT4 onboarding: see [`lc0_tvmjs_bt4it332_onboarding_2026-06-09.md`](./lc0_tvmjs_bt4it332_onboarding_2026-06-09.md).

The following readiness snapshot and experiment ledger predate that promotion.
“Blocked” and “next” in dated records describe their stage, not today's selector.

## Pre-promotion execution snapshot (historical)

The whole-model LC0 f16 ONNX → TVM Relax → TVMJS/WebGPU path is mechanically real:

- durable source-built TVM/TVMJS toolchain exists;
- b1/b4/b8 f16 WebGPU model wasm artifacts can be exported;
- browser loading verifies manifest bytes and SHA-256;
- browser execution requests `shader-f16`, instantiates the TVM runtime asynchronously, initializes WebGPU, prebuilds pipelines, and runs the Relax VM;
- smoke evidence compares real fixture eval/search rows against ORT f16 WebGPU.

At this pre-2026-06-10 stage, it was not promotion-ready: evidence was
smoke-derived and artifacts local/ignored. The later self-hosted non-default
promotion accepted explicit caveats. It did not authorize public publication
or a default change.

## Historical readiness snapshot (before 2026-06-10)

| Area | Current state | Promotion implication |
| --- | --- | --- |
| Default runtime safety | ORT ONNX/WebGPU remains stable default; TVMJS is absent from runtime registry/default UI. | Safe to continue research without changing product behavior. |
| Browser execution | Whole-model f16 TVMJS/WebGPU loads, verifies wasm SHA/bytes, requests `shader-f16`, prebuilds pipelines, and runs Relax VM. | Mechanically close to opt-in, not default. |
| ORT f16 comparison | Smoke/search rows currently match ORT f16 WebGPU in local evidence. | Promising, but still smoke-derived. |
| Hybrid comparison | Same-server and strict same-FEN TVMJS-vs-hybrid matrix exists. | First direct evidence exists; expand beyond n=2 smoke rows. |
| Startup timing | TVMJS smoke records bundle load, wasm fetch/verify, instantiate, WebGPU device, pipeline prebuild, VM creation, and input upload. | Good enough for first amortization analysis. |
| GPU footprint | TVMJS smoke and hybrid policy-only worker now both report `GPUDevice.createBuffer` allocation-request counts/bytes; ORT has opt-in prototype-level WebGPU API diagnostics for `createBuffer` during ORT evals. | Useful for relative startup/run allocation telemetry, but still not live GPU residency or a promotion-grade memory budget. |
| Fixed-suite evidence | Existing reports are fixed-suite-style but smoke-harness based. | Still blocked from promotion. |
| Artifact release | Generated TVMJS wasm/runtime artifacts remain ignored/local. | Publication/hosting/cache policy required. |

## Original promotion blockers (historical; public/default gates remain separate)

- Expand direct same-session full-model TVMJS vs custom hybrid TVM/WGSL comparison beyond smoke-sized rows.
- Production-style fixed-suite integration, not only `lab/lc0-tvmjs-webgpu-smoke.html`.
- Repeated throughput and startup/pipeline compile amortization evidence.
- Broader cross-runtime GPU allocation/footprint instrumentation and interpretation, including ORT WebGPU API diagnostics limits.
- Release/hosting/cache policy for generated model/runtime wasm artifacts.
- f16 drift/tolerance policy against native/ORT baselines.

## Durable local paths

From the workspace root (the parent of this checkout, `0x88/` by default):

```bash
export ROOT=<path-to-workspace-root>  # contains this checkout, .deps/, .envs/
export WEB=$ROOT/0x88                  # this checkout or an up-to-date worktree of it
export TVM_SRC=$ROOT/.deps/tvm-webgpu-src
export TVM_ENV=$ROOT/.envs/tvm-mlc-py313
export TVM_BUILD_DIR=build-tvmjs
export TVM_LIBRARY_PATH=$TVM_SRC/$TVM_BUILD_DIR/lib
export DYLD_LIBRARY_PATH=$TVM_LIBRARY_PATH:${DYLD_LIBRARY_PATH:-}
export PYTHONPATH=$TVM_SRC/python:${PYTHONPATH:-}
```

Do not use `/tmp` for durable TVM source/build/env artifacts.

## Toolchain check

```bash
cd "$WEB"
npm run lc0:check-tvmjs-webgpu-toolchain
```

Expected: JSON with `ok: true` and `TVM_BUILD_DIR` resolving to `build-tvmjs`.

If the toolchain needs rebuilding:

```bash
cd "$WEB"
npm run lc0:build-tvmjs-webgpu-toolchain
npm run lc0:check-tvmjs-webgpu-toolchain
```

## Export whole-model TVMJS/WebGPU wasm

For the t1 f16 b1/b4/b8 research exports (not today's QDQ8 product model):

```bash
cd "$WEB"
CAST_INT64_INITIALIZERS_TO_INT32=1 \
TRUST_NONNEGATIVE_GATHER_INDICES=1 \
SANITIZE_ONNX_NAMES=1 \
EXPORT_TVMJS_WASM=1 \
TVM_BUILD_DIR=build-tvmjs \
TVM_HOST_TARGET='{"kind":"llvm","mtriple":"wasm32-unknown-unknown-wasm"}' \
./scripts/run_lc0_tvm_whole_onnx_probe.sh
```

Required mitigations:

- `CAST_INT64_INITIALIZERS_TO_INT32=1`: WebGPU codegen rejects i64 constants.
- `TRUST_NONNEGATIVE_GATHER_INDICES=1`: LC0 policy gather indices are constant and nonnegative; avoids dead signed-index i64 paths.
- `SANITIZE_ONNX_NAMES=1`: makes generated C wrapper symbols valid.
- `EXPORT_TVMJS_WASM=1` plus wasm32 host target: emits browser-loadable TVMJS wasm.

## Stage browser artifacts

```bash
cd "$WEB"
npm run lc0:stage-tvmjs-webgpu
npm run lc0:tvmjs-webgpu-local-artifacts-check
```

Staged artifacts live under:

```text
public/runtimes/lc0-tvmjs-webgpu/<model-id>/f16/v1/
```

They remain ignored/local until release policy changes.

## Browser smoke and evidence gate

Small loader/invoke smoke:

```bash
cd "$WEB"
npm run lc0:tvmjs-webgpu-smoke -- \
  --batch 8 \
  --no-fixtures \
  --out artifacts/tvm/lc0_tvmjs_webgpu_smoke_b8_loader.json
```

Fixture/search comparison against ORT f16 WebGPU:

```bash
cd "$WEB"
npm run lc0:tvmjs-webgpu-smoke -- \
  --batch 8 \
  --ort-compare f16 \
  --ort-ep webgpu \
  --search-visits 32 \
  --search-fixtures 8 \
  --search-repeats 1 \
  --out artifacts/tvm/lc0_tvmjs_webgpu_smoke_b8_v32_ortf16.json
```

Aggregate the current local research evidence from an existing smoke artifact:

```bash
cd "$WEB"
npm run lc0:tvmjs-webgpu-summary -- \
  --in artifacts/tvm/lc0_tvmjs_webgpu_smoke_b8_v16_ortf16.json \
  --out artifacts/tvm/lc0_tvmjs_webgpu_fixed_suite_b8_v16_ortf16_report.json
```

Research-only fixed-suite bridge, using the same fixed FEN input style as `lc0_browser_runtime_fixed_suite.mjs` but without registry/default promotion of LC0 whole-model TVMJS. Non-v0 page-level selection is allowed.
The UHO-lite opening suites (`eval/opening_suite_uho_lite_v1.fen` and `_v2`)
are no longer in this repository; they were removed in commit `004504e`
(2026-06-11). Recover one with
`git show 004504e^:eval/opening_suite_uho_lite_v2.fen > <file>` (v2 corrects an
impossible row 9 in v1) and pass that file to `--fens`.

```bash
npm run lc0:tvmjs-webgpu-fixed-suite -- \
  --fens ../leelaweb-arena-diagnostics/eval/opening_suite_uho_lite_v1.fen \
  --max-positions 2 \
  --batch 8 \
  --visits 16 \
  --repeats 1 \
  --stockfish-score-depth 3 \
  --out artifacts/tvm/lc0_tvmjs_webgpu_fixed_suite_bridge_uho2_v16_r1_sfdepth3.json
```

The bridge emits normalized suite FENs, a child TVMJS smoke artifact, a `*_report.json` fixed-suite-style report discoverable by the evidence summarizer, and an aggregate `lc0_browser.tvmjs_webgpu_fixed_suite_research_bridge.v1` JSON with `researchOnly: true` and `noStableRuntimePromotion: true`.

Then refresh/check aggregate evidence:

```bash
npm run lc0:tvmjs-webgpu-evidence-summary
npm run lc0:tvmjs-webgpu-research-gate
```

The combined research gate verifies evidence, local artifacts, and research-only isolation. It is not a promotion gate.

## Same-session TVMJS vs hybrid comparison target

The next required research artifact should run these in one browser/dev-server session against the same FEN rows:

1. TVMJS full-model WebGPU via `lab/lc0-tvmjs-webgpu-smoke.html`.
2. Custom hybrid piecemeal TVM/WGSL via `lc0-policy-only.html` / hybrid search fixture harness.
3. Optional ORT f16 WebGPU baseline for triangulation.

The minimum first-pass matrix:

```bash
# TVMJS full-model, same FEN file and search budget
npm run lc0:tvmjs-webgpu-smoke -- \
  --batch 8 \
  --fens ../leelaweb-arena-diagnostics/eval/opening_suite_uho_lite_v1.fen \
  --fixture-count 4 \
  --ort-compare f16 \
  --ort-ep webgpu \
  --search-visits 16 \
  --search-fixtures 4 \
  --search-repeats 1 \
  --out artifacts/tvm/lc0_tvmjs_vs_hybrid_tvmjs_uho4_v16.json

# Hybrid custom WGSL/piecemeal path, same rough search budget
npm run lc0:browser-hybrid-search-fixture-parity -- \
  --preset lc0-webgpu-research-b4 \
  --head-backend wgsl \
  --input-backend wasm \
  --encoder-kernel mixed-tvm-ffn-smolgen-project \
  --legal-priors-backend js \
  --batch 4 \
  --visits 32 \
  --fixture-limit 4 \
  --repeats 1 \
  --allow-mismatches \
  --out artifacts/tvm/lc0_tvmjs_vs_hybrid_hybrid_uho4_v32.json
```

Preferred first-pass wrapper:

```bash
cd "$WEB"
npm run lc0:tvmjs-vs-hybrid-matrix -- \
  --batch 8 \
  --hybrid-batch 4 \
  --fixtures 4 \
  --visits 32 \
  --repeats 1 \
  --out artifacts/tvm/lc0_tvmjs_vs_hybrid_b8_hb4_v32_n4_r1.json
```

Scored strict same-FEN UHO-lite wrapper:

```bash
cd "$WEB"
npm run lc0:tvmjs-vs-hybrid-matrix -- \
  --batch 8 \
  --hybrid-batch 4 \
  --fixtures 4 \
  --visits 16 \
  --repeats 2 \
  --stockfish-score-depth 3 \
  --fens ../leelaweb-arena-diagnostics/eval/opening_suite_uho_lite_v1.fen \
  --out artifacts/tvm/lc0_tvmjs_vs_hybrid_uho_b8_hb4_v16_n4_r2_sfdepth3.json
```

The wrapper writes an aggregate `lc0_browser.tvmjs_vs_hybrid_matrix.v1` JSON artifact and child artifacts for each lane. With `--fens`, both lanes run the same FEN rows. The TVMJS child artifact and aggregate include:

- `startupTimings`: manifest fetch, TVMJS bundle load, WebGPU adapter/device acquisition, wasm fetch/verification, async wasm instantiate, TVM WebGPU init, `systemLib`, WebGPU pipeline prebuild, VM creation, input tensor allocation, and input upload.
- `gpuBufferAllocation`: a `GPUDevice.createBuffer` monkeypatch snapshot with buffer count, total bytes, max buffer bytes, mapped-at-creation count, and usage-category totals. This is a prototype allocation footprint, not a browser/GPU-resident memory guarantee.

First local wrapper artifacts:

```text
artifacts/tvm/lc0_tvmjs_vs_hybrid_b8_hb4_v32_n2_r1.json
artifacts/tvm/lc0_tvmjs_vs_hybrid_uho_b8_hb4_v16_n2_r1.json
```

Summary from the built-in-fixture smoke matrix:

- TVMJS b8 vs ORT f16 search move match: `2/2`.
- TVMJS b8 search mean: `100.72 ms`; ORT f16 same smoke mean: `110.74 ms`.
- Hybrid b4 WGSL-heads/native-search-fixture match: `2/2`.
- Hybrid b4 search mean: `329.65 ms`; mean backend search elapsed: `329.31 ms`.
- Useful early evidence, but not strict row identity unless `--fens` is used.

Summary from the strict same-FEN UHO-lite smoke matrix:

- Artifact: `artifacts/tvm/lc0_tvmjs_vs_hybrid_uho_b8_hb4_v16_n2_r1.json`.
- Same FEN rows: `2`.
- TVMJS-vs-hybrid search move match: `2/2`.
- TVMJS b8 mean search timing: `71.48 ms`.
- Hybrid b4 WGSL-heads mean search timing: `224.76 ms`.
- TVMJS-vs-ORT f16 move match within the TVMJS lane: `2/2`.
- Hybrid arbitrary-FEN mode has no native best-move oracle; compare row moves, timings, and hybrid depth-baseline stability.

Summary from the scored strict same-FEN UHO-lite matrix:

- Artifact: `artifacts/tvm/lc0_tvmjs_vs_hybrid_uho_b8_hb4_v16_n4_r2_sfdepth3.json`.
- Same FEN/repeat rows: `8`.
- TVMJS-vs-hybrid search move match: `6/8`.
- The two mismatches are the same FEN across both repeats: TVMJS chose `d7d6`; hybrid chose `e5d4`.
- TVMJS b8 mean search timing: `48.67 ms`.
- Hybrid b4 WGSL-heads mean search timing: `133.27 ms`.
- TVMJS-vs-ORT f16 Stockfish-scored deltas: `6` scored rows, cp delta min/max/mean `0/0/0`.
- Interpretation: TVMJS remains aligned with ORT f16 on the scored rows, while the custom hybrid piecemeal runtime can choose a different search move on at least one UHO-lite position at 16 visits. This supports keeping hybrid and TVMJS as distinct research/product lanes rather than treating them as interchangeable.

Current production-style fixed-suite bridge evidence:

- Main repeat artifact: `artifacts/tvm/lc0_tvmjs_webgpu_fixed_suite_bridge_uho16_v16_r2_sfdepth3.json`.
  - Status: `ok: true`, `researchOnly: true`, `noStableRuntimePromotion: true`.
  - Fixed FEN rows: `16`; search rows: `32` (`repeats=2`, visits `16`).
  - TVMJS-vs-ORT f16 search move match: `32/32`.
  - TVMJS mean/median/max search timing: `48.20 / 47.28 / 59.45 ms`.
  - Stockfish TVMJS-minus-ORT f16 cp deltas: `30` scored rows, min/max/mean `0/0/0`.
- Visit-depth artifact: `artifacts/tvm/lc0_tvmjs_webgpu_fixed_suite_bridge_uho8_v32_r1_sfdepth3.json`.
  - Fixed FEN rows: `8`; search rows: `8` (visits `32`).
  - TVMJS-vs-ORT f16 search move match: `8/8`.
  - TVMJS mean/median/max search timing: `84.36 / 79.48 / 108.72 ms`.
  - Stockfish TVMJS-minus-ORT f16 cp deltas: `7` scored rows, min/max/mean `0/0/0`.
- Earlier smoke artifacts: `artifacts/tvm/lc0_tvmjs_webgpu_fixed_suite_bridge_uho8_v16_r1_sfdepth3.json` and `artifacts/tvm/lc0_tvmjs_webgpu_fixed_suite_bridge_uho2_v16_r1_sfdepth3.json`.
- The bridge now covers more rows, repeats, and a higher visit budget, but the evidence remains non-promotional until broader suites and release/hosting policy are accepted.

Startup/footprint sample from `artifacts/tvm/lc0_tvmjs_vs_hybrid_uho_b8_hb4_v16_n2_r1_startup.json`:

- TVMJS bundle load: `3.505 ms`.
- wasm fetch+verify: `126.82 ms`.
- TVMJS instantiate: `42.47 ms`.
- WebGPU pipeline prebuild: `101.635 ms`.
- VM creation: `27.78 ms`.

Startup/amortization sidecar:

```bash
npm run lc0:tvmjs-startup-amortization -- \
  --in artifacts/tvm/lc0_tvmjs_vs_hybrid_uho_b8_hb4_v16_n1_r1_hybrid_alloc.json \
  --out artifacts/tvm/lc0_tvmjs_startup_amortization_uho_b8_hb4_v16_n1_r1.json
```

Current one-row sidecar findings:

- TVMJS known cold-start phase sum: `434.615 ms`; TVMJS search mean: `78.205 ms`; one-row amortized mean: `512.82 ms`.
- ORT f16 search mean in the same smoke comparison: `89.905 ms`; ORT session/device startup is not separated in this matrix artifact.
- Hybrid worker init: `68.16 ms`; hybrid search mean: `404.665 ms`; one-row amortized mean: `472.825 ms`.
- Caveat: this sidecar sums observed phase timings for research triage; it is not proof of a strictly serialized critical path or a production startup SLA.

Timing-breakdown sidecar for smoke/matrix artifacts:

```bash
npm run lc0:tvmjs-timing-breakdown -- \
  --in artifacts/tvm/lc0_tvmjs_vs_hybrid_uho_b8_hb4_v16_n4_r2_sfdepth3.json \
  --in artifacts/tvm/lc0_tvmjs_vs_hybrid_b8_hb4_v16_n4_r2.tvmjs.json \
  --out artifacts/tvm/lc0_tvmjs_timing_breakdown_uho_b8_hb4_v16_n4_r2.json
```

Current timing-breakdown sidecar summary:

- Artifact: `artifacts/tvm/lc0_tvmjs_timing_breakdown_uho_b8_hb4_v16_n4_r2.json`.
- Inputs: matrix aggregate plus TVMJS child smoke for the scored strict same-FEN UHO-lite run.
- TVMJS search mean: `48.67 ms`; top-level invoke: `5.145 ms`; search-to-single-invoke ratio: `9.46`.
- Known startup phase sum for this artifact pair: `373.205 ms`.
- Caveat: older smoke artifacts do not contain per-evaluation phase buckets. Newly generated smoke artifacts include `evalTiming` and `searchParity.tvmEvalTiming` buckets for encode, f16 conversion, tensor allocation, upload, `set_input`, VM invoke, output handle fetch, readback/sync, decode, and legal-prior filtering.
- New one-fixture timing proof: `artifacts/tvm/lc0_tvmjs_webgpu_smoke_b8_fixture1_timing.json` and `artifacts/tvm/lc0_tvmjs_timing_breakdown_b8_fixture1.json` show fixture parity `1/1`, top-level invoke `5.435 ms`, known startup phase sum `364.845 ms`, per-eval batch eval `14.545 ms`, VM invoke `2.675 ms`, and output readback/sync `10.325 ms`.

### Visit-loop attribution (per-search wall-time reconciliation)

Search-parity rows now carry `tvmVisitLoop` (per-search sums of the evaluator phase
buckets reconciled against that search's wall time, plus batch-fill stats) and
`tvmSearchStats` (visit/batch counters from the PUCT searcher). The aggregate is
`searchParity.visitLoopBreakdown` with per-phase totals and `shareOfSearchWall`.
Phase semantics: `tvmInvokeMs` is CPU-side VM submit; GPU completion is awaited
inside `outputReadbackMs`, so the readback share includes the GPU compute wait.

```bash
node scripts/lc0_tvmjs_webgpu_smoke.mjs --batch 8 --fixture-count 8 \
  --search-visits 16 --search-fixtures 4 --search-repeats 2 \
  --out artifacts/tvm/lc0_tvmjs_visit_loop_profile_b8_v16.json
```

Current visit-loop attribution result (`artifacts/tvm/lc0_tvmjs_visit_loop_profile_b8_v16.json`,
8 search rows at v16/b8, search parity 8/8, native fixture parity 8/8):

- Search wall mean `52.88 ms`; in-evaluator mean `50.98 ms` (`96.4%`); JS search overhead (selection, PUCT, backup, context building) mean `1.90 ms` (`3.6%`).
- Share of search wall: output readback/GPU-wait `86.6%`, VM submit `4.6%`, encode `1.5%`, f16 input convert `1.1%`, legal priors `1.1%`, output decode `0.9%`; tensor alloc, upload, `set_input`, and handle fetch each `<0.3%`.
- Eval batches per search: `3`; mean logical batch size `4.83/8` (`60.4%` fill); per-search leaf-collision retries ranged `11–69`.
- Reading: the visit loop is serialized on per-batch GPU completion (about `14 ms` per batch, of which about `2.4 ms` is CPU submit). JS search optimization is not a lever at ~`2 ms` per search.

### Pipelined evaluateBatchSequence A/B (negative result, 2026-06-09)

The TVMJS provider now implements `evaluateBatchSequence` (submit every physical
batch, then ONE shared `device.sync()`, then decode), and the search smoke
accepts `--search-pipeline-depth N` (page param `searchPipelineDepth`, applied
to both searchers so the search shape stays matched). A/B at v16/b8,
4 fixtures × 2 repeats, parity 8/8 on both legs:

- Depth 1 control (`artifacts/tvm/lc0_tvmjs_visit_loop_profile_b8_v16_pd1.json`): TVMJS search mean `54.30 ms`, batches/search `3`, fill `60.4%`.
- Depth 2 (`artifacts/tvm/lc0_tvmjs_visit_loop_profile_b8_v16_pd2.json`): TVMJS search mean `68.81 ms` (**+27%**), batches/search `4.25`, fill `52.2%`.
- Mechanism check: per-batch GPU-wait share was `15.58 ms` with a dedicated sync per batch versus `13.85 ms` as half of a 2-batch shared sync — a 2-batch flush syncs in roughly 2× a single batch. The "readback wait" is therefore **GPU compute (~13–15 ms per padded b8 invoke), not sync/readback overhead**, and pipelining cannot hide it. The depth-2 scheduler also fragments the small visit budget into more, worse-filled physical batches, each costing a full padded-b8 GPU pass.
- Revised lever order: (1) per-batch GPU compute — kernel/schedule tuning (dlight rules, per-kernel timestamp-query attribution); (2) physical invoke count — batch fill (~40% of GPU work is padding at 60% fill) and visit budgets that fill b8; (3) pipelining only pays once per-batch GPU time shrinks or CPU encode grows. Keep `searchPipelineDepth=1` as the default; the pipelined path stays as an opt-in research knob.

### Per-kernel GPU attribution via timestamp-query (2026-06-09)

The smoke page now installs a GPU kernel profiler when the adapter offers
`timestamp-query` (optional feature; `kernelProfile=0` disables). It wraps
compute-pipeline creation to learn TVM kernel entry-point names, injects
`timestampWrites` into every compute pass during one dedicated warm b8 invoke,
and attaches the aggregate as `result.gpuKernelProfile`.

Current b8 result (`artifacts/tvm/lc0_tvmjs_kernel_profile_b8.json`, Apple
Silicon Chrome, one warm invoke):

- **229 compute passes** per b8 invoke; per-pass GPU busy time sums to **9.24 ms** versus the ~`14 ms` end-to-end GPU wait, so roughly a third of the GPU wall is inter-pass gaps/submit/readback overhead rather than kernel execution.
- No dominant kernel. Top entries: `fused_matmul4_add2_tir_sigmoid_multiply2` `1.70 ms` (18.4%, ×10), `fused_matmul9_add_multiply5_add5` `1.51 ms` (16.3%, ×10), `fused_matmul8_add6_relu_multiply6` `1.38 ms` (14.9%, ×10), `fused_matmul1_add` `1.25 ms` (13.5%, ×32), `matmul6` `0.66 ms` (×10), `softmax` `0.52 ms` (×10). The top 4 matmul families are ~63% of kernel time.
- Reshape/transpose/layer-norm kernels are individually tiny (~`0.007 ms`) but cost a full pass each (~50 of the 229 passes), so their cost is mostly pass overhead, not arithmetic.
- Tuning implications: (a) matmul schedules are `3–10×` off a naive roofline estimate, spread across ~6 fused matmul variants — dlight-style schedule rules would need to lift the whole family, not one hot spot; (b) pass-count reduction (more aggressive fusion of elementwise/reshape chains, or pass/encoder batching in the tvmjs runtime) attacks the ~`5 ms` non-kernel GPU wall and the long tail of tiny passes.
- Caveats: single warm invoke, one device; Chrome may quantize timestamps without developer features (observed values are not 100 µs multiples here, so quantization did not flatten this run).

### Dlight default-schedule rebuild (marginal) and pass coalescing (nil), 2026-06-09

Two follow-up experiments on the same v16/b8 protocol, both parity-clean (native
8/8 with identical max top-prior diff, search 8/8):

- **Dlight rebuild**: `scripts/lc0_tvm_whole_onnx_probe.py --dlight` (env `DLIGHT=1`) applies `tvm.s_tir.dlight` GPU rules (Matmul/GEMV/Reduction/GeneralReduction/Fallback) after the zero pipeline instead of relying on naive `DefaultGPUSchedule`. Staged separately under `.../f16/v2-dlight` (stage script takes `--artifacts=`/`--out=` in `--name=value` form only); the smoke driver gained `--manifest`. Result (`artifacts/tvm/lc0_tvmjs_dlight_profile_b8_v16.json`): kernel time `9.24 → 8.52 ms` (−8%), search wall `56.9 ms` vs `54.3 ms` control — no end-to-end gain. Same 229 passes; top kernel ranking unchanged.
- **Compute-pass coalescing**: the tvmjs runtime opens one compute pass per dispatch; the smoke can now coalesce all dispatches on the pending encoder into one shared pass (`--pass-coalesce`, page param `passCoalesce=1`, deferring `pass.end()` to `encoder.finish()`; legal because WebGPU orders dispatches within a pass and the runtime flushes before copies/readback). Result (`artifacts/tvm/lc0_tvmjs_passcoalesce_profile_b8_v16.json`): 5928 dispatches coalesced into 26 passes, search wall `53.0 ms` vs `54.3 ms` control — within noise. Pass boundaries are not the missing GPU wall either.

Campaign conclusion: per-b8-batch GPU wall (~`14 ms`) decomposes into ~`9 ms`
genuine kernel arithmetic (dlight defaults barely move it) plus ~`2–3 ms` sync/
readback machinery and small residuals. Eliminated as levers: JS search
overhead (~2 ms), pipelining (net negative), pass-boundary overhead (nil),
default dlight rules (−8% kernel, nil end-to-end). Remaining levers, in order:
real per-shape schedule tuning (metaschedule or hand-written dlight rules for
the 4 fused matmul families at 512×256-class shapes), larger physical batches
(b16/b32) for visit budgets that can fill them, and evaluation-count reduction
(cache/tree reuse) on the search side.

### Dlight Matmul rule attribution and tile-config sweep (2026-06-09)

Attribution fix first: the initial "dlight only scheduled 1 function" reading
was an accounting bug — these PrimFuncs carry no `global_symbol` attr after the
zero pipeline, so every rule hit was recorded under one key. With naming fixed
(probe resolves names via the module's GlobalVars), the true attribution for
the b16 module is **Matmul 19, GEMV 1, GeneralReduction 5, Fallback 14** — the
Matmul rule does fire on every matmul kernel. The −8%-kernel/nil-end-to-end
dlight result therefore means the default dlight schedules perform like the
naive schedules here, not that dlight failed to apply.

Tile-config sweep (`scripts/sweep_dlight_matmul_configs.sh`, env
`DLIGHT_MATMUL_CONFIG` / probe `--dlight-matmul-config`, one config per build,
8 profiled invokes per measurement via `--kernel-profile-invokes`, b16, all
legs native parity 8/8; results in
`artifacts/tvm/dlight_matmul_config_sweep.jsonl`):

| Config | GPU ms/invoke | vs baseline |
| --- | ---: | --- |
| baseline (webgpu default: 8×8 threads, 4×4 micro, k8, vec1, shared) | `13.78` | — |
| `vector_size=2` | build failed | TIR vectorize legality: one config applies to ALL matmuls incl. WDL `N=3` / policy `N=1858` heads |
| `micro_size_k=16` | `13.10` | −5% (≈ noise margin) |
| `block_size_y=16` | `20.19` | +47% |
| `storage_align=true` | `14.84` | +8% |
| `inner_x=true` | `13.90` | ~0 |
| `use_shared=false` | `25.45` | +85% |
| `micro_size_x/y=8` | `18.23` | +32% |

Conclusion: dlight's default webgpu Matmul config is a local optimum for these
512×256-class f16 shapes on Apple Silicon; every single-knob change is neutral
or worse, and the one promising lever (vectorized loads) requires per-function
config dispatch to stay legal on the head matmuls, for a bounded payoff
(matmuls are 63% of ~`13.8 ms` at b16; a generous vec2 win on the big matmuls
projects to low-single-digit percent of search wall). Per-shape schedule
tuning is parked as low-ROI on this device; revisit only with
metaschedule-grade search or evidence from a different GPU class.

### Batch-scaling sweep: b16/b32 exports (2026-06-09, the lever that worked)

New fixed-batch f16 ONNX exports via native lc0
(`lc0 leela2onnx --onnx-batch-size={16,32} --onnx-data-type=f16`, sha256
sidecars in `../models/lc0-bestnets/onnx/`, exposed through
`scripts/lc0_prepare_model_assets.mjs` symlinks), built into TVMJS wasms with
the standard mitigations and staged into the v1 runtime
(`stage_lc0_tvmjs_webgpu_artifacts.mjs --batches=1,4,8,16,32`). The smoke
page/driver accept batches 16 and 32.

Search-wall sweep (4 fixtures × 2 repeats, all legs native parity 8/8 and
TVMJS-vs-ORT search move match 8/8; artifacts
`artifacts/tvm/lc0_tvmjs_b{8,16,32}_v{16,32,64}.json`):

| Visits | b8 mean | b16 mean | b32 mean | Best |
| --- | ---: | ---: | ---: | --- |
| 16 | `54.30 ms` | `39.95 ms` | — | **b16, −26%** |
| 32 | `79.60 ms` | `61.74 ms` | `63.86 ms` | **b16, −22%** |
| 64 | — | `99.63 ms` | `94.57 ms` | b32, −5% vs b16 |

Mechanism: per-batch eval cost is strongly sublinear in physical batch size
(`15.1 ms` b8 → `19.4 ms` b16 → `30.3 ms` b32; per-position at full fill
`1.89 → 1.21 → 0.95 ms`), so fewer larger batches beat better-filled smaller
ones even at `36–47%` fill. b16 is the recommended physical batch for the
practical 16–32-visit browser budgets; the b16/b32 crossover begins around
64 visits. ORT WebGPU f16 shows the same direction (its b16 model also beat
its b8 model in the same sessions), so this is a batching effect, not a
TVMJS-specific one. Footprint caveat: each extra fixed-batch wasm adds ~44 MB
to the staged runtime; the separated-params/tensor-cache plan is what makes a
multi-batch runtime distributable.

Strict same-FEN visit sweep samples now cover visits `16`, `32`, and `64`:

- `artifacts/tvm/lc0_tvmjs_vs_hybrid_uho_b8_hb4_v16_n1_r1_hybrid_alloc.json`: rows `1`, TVMJS-vs-hybrid `1/1`, TVMJS mean `78.205 ms`, hybrid mean `404.665 ms`.
- `artifacts/tvm/lc0_tvmjs_vs_hybrid_uho_b8_hb4_v32_n2_r1.json`: rows `2`, TVMJS-vs-hybrid `2/2`, TVMJS mean `97.54 ms`, hybrid mean `299.795 ms`.
- `artifacts/tvm/lc0_tvmjs_vs_hybrid_uho_b8_hb4_v64_n2_r1.json`: rows `2`, TVMJS-vs-hybrid `2/2`, TVMJS mean `180.55 ms`, hybrid mean `457.143 ms`.

## GPU allocation instrumentation status

Current prototypes:

- `lab/lc0-tvmjs-webgpu-smoke.html` wraps the acquired TVMJS `GPUDevice.createBuffer` before TVM initialization and model execution. Results report total allocation calls/bytes, max buffer size, mapped-at-creation count, and categories decoded from `GPUBufferUsage` flags.
- The hybrid policy-only worker now installs a worker-side `navigator.gpu.requestAdapter`/`requestDevice`/`GPUDevice.createBuffer` patch before initializing the custom WGSL evaluator. Hybrid search fixture artifacts include `gpuBufferAllocation` on each row and at the aggregate top level.
- `scripts/lc0_tvmjs_vs_hybrid_matrix.mjs` forwards the latest hybrid allocation snapshot into `summary.hybrid.gpuBufferAllocation`, next to `summary.tvmjs.gpuBufferAllocation`, for strict same-FEN comparisons.
- ORT already has opt-in WebGPU API instrumentation in `src/nn/ortRuntime.ts` when ORT diagnostics/API tracing are requested; `src/lc0/onnxEvaluator.ts` records per-eval deltas such as `webgpuCreateBufferCount` and `webgpuCreateBufferBytes`. This is patchable only when browser WebGPU prototypes are visible before ORT creates/uses the device.

Sample strict same-FEN allocation artifact:

```bash
npm run lc0:tvmjs-vs-hybrid-matrix -- \
  --batch 8 \
  --hybrid-batch 4 \
  --fixtures 1 \
  --visits 16 \
  --repeats 1 \
  --fens ../leelaweb-arena-diagnostics/eval/opening_suite_uho_lite_v1.fen \
  --out artifacts/tvm/lc0_tvmjs_vs_hybrid_uho_b8_hb4_v16_n1_r1_hybrid_alloc.json
```

Observed in that one-row telemetry artifact:

- TVMJS: `538` buffer creation requests, `45,201,396` requested bytes, max buffer `2,097,152` bytes.
- Hybrid worker: `548` buffer creation requests, `56,901,112` requested bytes, max buffer `2,097,152` bytes.
- Startup/init fields now appear side by side in the matrix summary: TVMJS startup timings include wasm fetch/verify `153.01 ms`, instantiate `43.01 ms`, pipeline prebuild `122.03 ms`, VM creation `25.55 ms`; hybrid reports worker init `68.16 ms` for the same run.
- Head-to-head row: TVMJS and hybrid both chose `c6c5`; TVMJS search `78.20 ms`, hybrid search `404.67 ms`.

Known limitations:

- It is allocation-request telemetry, not a definitive GPU memory residency measurement.
- It does not observe allocations hidden behind an unpatched device/prototype or native browser internals.
- It does not decrement on destruction; use it for startup/run footprint comparison, not live-memory accounting.
- ORT WebGPU telemetry should be treated as direct only when the diagnostic summary reports the API patch installed and nonzero per-eval deltas; otherwise record ORT allocation footprint as indirect/unknown and rely on model/network footprint plus timing.

Next footprint work:

1. Run the TVMJS-vs-hybrid allocation matrix across larger UHO-lite visit/repeat rows.
2. Add an ORT-focused fixed-suite/readback diagnostic artifact that asserts whether `webgpuCreateBufferCount`/bytes are observable in the target browser.
3. Add release-host smoke or static-serving check for gzip/Brotli `Content-Encoding` behavior before publication.
4. Expand artifact/network footprint sidecars across future model families and release candidates.

Current bundle-footprint sidecar:

```bash
npm run lc0:tvmjs-webgpu-bundle-footprint -- \
  --manifest public/runtimes/lc0-tvmjs-webgpu/t1-256x10-distilled-swa-2432500/f16/v1/manifest.json \
  --out artifacts/tvm/lc0_tvmjs_webgpu_bundle_footprint_t1_f16_v1.json
```

Current default-family footprint sidecar summary:

- Artifact: `artifacts/tvm/lc0_tvmjs_webgpu_bundle_footprint_t1_f16_v1.json`.
- Files: `8`.
- Raw bytes: `139,358,480`.
- gzip level-9 bytes: `113,885,414` (`0.8172` ratio).
- Brotli quality-11 bytes: `107,101,046` (`0.7685` ratio).
- Caveat: these are Node zlib estimates and do not prove deployed `Content-Encoding` behavior.

### Detached-params export now works (2026-06-09)

`scripts/lc0_tvm_whole_onnx_probe.py --detach-params` (env `DETACH_PARAMS=1`)
imports with `keep_params_in_input`, keeps initializers ≤4096 elements as
inline constants (shape-feeding tensors must keep constant values — a probe-side
patch of `_parse_graph_initializers`, same style as the Gather patch), detaches
the 100 large weight tensors via `relax.frontend.detach_params`, and dumps them
as a raw-encoded tensor-cache (`param_0..param_99`). The smoke page detects
`parameterStrategy.current === 'detached-tensor-cache'` in the staged manifest,
loads weights via `tvm.fetchTensorCache` (which also gives Cache Storage
caching of shards for repeat loads), and spreads them into
`set_input('main', input, ...params)`.

b16 result (`f16/v3-detached`, `artifacts/tvm/lc0_tvmjs_detached_b16_v16.json`):

- Model wasm: `44.6 MB → 4.5 MB`; weights: `38 MB` tensor-cache sidecar (`40.2 MB` of params).
- Native fixture parity `8/8` with bit-identical max top-prior diff to the embedded build; search parity `8/8`; search wall `41.6 ms` mean at v16/b16 vs `40.0 ms` embedded — within noise.
- Startup shift: wasm fetch+verify `104 → 21 ms`, instantiate `41 → 27 ms`, tensor-cache fetch+GPU upload `238 ms` (first load; artifact-cached afterwards).
- Why it matters: a multi-batch runtime (b1+b16+…) can share ONE weight sidecar instead of embedding `~44 MB` per batch wasm — e.g. five batches drop from `~223 MB` to `~60 MB` staged.
- Repeat-load cache behavior and compressed sizes are now measured (2026-06-09):
  - Compressed footprint (`artifacts/tvm/lc0_tvmjs_v3_detached_footprint.json`, via `lc0:tvmjs-webgpu-bundle-footprint` against the `v3-detached` manifest): b1+b16 wasms + runtime + shared tensor-cache = `54.66 MB` raw, `39.71 MB` gzip, **`36.78 MB` Brotli** — two batch variants ship at roughly the compressed cost of one embedded model wasm.
  - Repeat-load (`artifacts/tvm/lc0_tvmjs_detached_repeat_load.json`): `fetchTensorCache` persists all shards into a Cache Storage bucket (`3` entries, `40.2 MB`); reading them back from the bucket — the warm path a second visit takes instead of the network — costs **`21 ms`** versus `235 ms` cold fetch+GPU-upload. Note: the runtime tensor cache rejects same-instance re-insertion, so a second `fetchTensorCache` is not a valid repeat-load probe; the smoke measures the Cache Storage read directly (`tensorCacheWarmReadMs`/`tensorCacheCachedEntries`/`tensorCacheCachedBytes` in `startupTimings`).
- Remaining before any release decision: non-Apple GPU coverage and a written hosting/cache-header policy (the compression sidecar does not prove production `Content-Encoding`).

Cross-batch sharing is now **proven** (2026-06-09):

- b1 and b16 `--detach-params` builds dump **byte-identical** tensor-caches (matching sha256 for both `params_shard_*.bin` and `tensor-cache.json`).
- `f16/v3-detached` now stages b1 + b16 wasms (`4.5 MB` each) with ONE shared sidecar (`55 MB` total staged versus `~89 MB` embedded for the same two batches).
- End-to-end: the b1 wasm running against the b16-dumped sidecar passed native best-move parity `8/8` (`artifacts/tvm/lc0_tvmjs_detached_b1_shared_cache.json`; max top-prior abs diff `0.0059`). At this stage b1 search parity was blocked by unaligned WDL/MLH readback. The 2026-06-10 runtime fix closed it; see the BT4 onboarding record's “Small debts cleared” and `docs/upstream/tvm-webgpu-unaligned-readback.patch`. These original evaluator measurements are unchanged.

Fixed-suite bridge evidence now covers the new configs (the bridge accepts
`--batch 16/32` and `--manifest`):

- `artifacts/tvm/lc0_tvmjs_webgpu_fixed_suite_bridge_uho16_b16_v16_sfdepth3.json`: embedded b16, 16 UHO positions, v16 — search move match `16/16` vs ORT f16, Stockfish depth-3 cp delta `0/0/0` over 15 comparable rows, TVMJS mean `39.07 ms` vs ORT `45.41 ms`.
- `artifacts/tvm/lc0_tvmjs_webgpu_fixed_suite_bridge_uho16_b16detached_v16_sfdepth3.json`: detached b16 (`v3-detached` manifest), same protocol — `16/16`, cp delta `0/0/0`, TVMJS mean `38.68 ms` vs ORT `46.18 ms`.
- Both `*_report.json` artifacts are discovered by `lc0:tvmjs-webgpu-evidence-summary` (6 fixed-suite-style reports total) and the combined research gate passes. Note: the summary's `searchRows` counter only scans `lc0_tvmjs_webgpu_search_smoke_*` filenames, so bridge rows are represented through the report list rather than that counter.

Reproduce:

```bash
CAST_INT64_INITIALIZERS_TO_INT32=1 TRUST_NONNEGATIVE_GATHER_INDICES=1 \
SANITIZE_ONNX_NAMES=1 EXPORT_TVMJS_WASM=1 DETACH_PARAMS=1 \
TVM_BUILD_DIR=build-tvmjs \
TVM_HOST_TARGET='{"kind":"llvm","mtriple":"wasm32-unknown-unknown-wasm"}' \
LC0_TVMJS_BATCHES=16 OUT_DIR=$PWD/artifacts/tvm-detached \
./scripts/run_lc0_tvm_whole_onnx_probe.sh

node scripts/stage_lc0_tvmjs_webgpu_artifacts.mjs \
  --artifacts=artifacts/tvm-detached --batches=16 \
  --tensor-cache-dir=artifacts/tvm-detached/t1-256x10-distilled-swa-2432500.batch16.f16.webgpu.tvmjs-wasm.probe.tensor-cache \
  --out=public/runtimes/lc0-tvmjs-webgpu/t1-256x10-distilled-swa-2432500/f16/v3-detached \
  --params=detached

node scripts/lc0_tvmjs_webgpu_smoke.mjs --batch 16 \
  --manifest /runtimes/lc0-tvmjs-webgpu/t1-256x10-distilled-swa-2432500/f16/v3-detached/manifest.json \
  --fixture-count 8 --search-visits 16 --search-fixtures 4 --search-repeats 2 \
  --out artifacts/tvm/lc0_tvmjs_detached_b16_v16.json
```

### Tensor-cache / separated-params planning probe

Use the weight-cache planning sidecar before any release-candidate staging decision:

```bash
npm run lc0:tvmjs-weight-cache-plan -- \
  --manifest public/runtimes/lc0-tvmjs-webgpu/t1-256x10-distilled-swa-2432500/f16/v1/manifest.json \
  --tvm-src ../.deps/tvm-webgpu-src \
  --out artifacts/tvm/lc0_tvmjs_weight_cache_plan_current.json
```

Current default-family planning sidecar summary:

- Artifact: `artifacts/tvm/lc0_tvmjs_weight_cache_plan_current.json`.
- Local TVM API support: `tvm.contrib.tvmjs.dump_tensor_cache`, `tvm.contrib.tvmjs.load_tensor_cache`, browser `tvm.fetchTensorCache`, artifact cache, and Relax `detach_params` are present.
- Current parameter strategy: `embedded-in-per-batch-wasm`.
- Runtime + model wasm bytes covered by the staged manifest: `139,228,830`.
- Model wasm share: `95.93%`.
- Duplicate batch-wasm upper bound: `88,919,507` bytes (`63.87%` of runtime+model bytes).
- Caveat: this is an upper bound, not a guaranteed savings figure; exact savings require a real detached-param export because TVM code and metadata also differ by batch.

Separated-params release-candidate recipe:

1. Detach Relax parameters when possible before VM build, or otherwise emit f16 params through `tvm.contrib.tvmjs.dump_tensor_cache(..., encode_format="raw")`.
2. Stage `tensor-cache.json` plus `params_shard*` files under the immutable model-family/dtype/version path. The staging script has a research-only sidecar path:

   ```bash
   npm run lc0:stage-tvmjs-webgpu -- \
     --tensor-cache-dir=artifacts/tvm/<model-id>.f16.tensor-cache
   npm run lc0:tvmjs-webgpu-local-artifacts-check -- --no-evidence
   ```

3. Manifest entries must include bytes/SHA-256 for every tensor-cache shard and record whether params are embedded, staged as a sidecar, or truly detached before build. Current `--tensor-cache-dir` staging records `embedded-wasm-plus-staged-tensor-cache`; this is not yet proof of duplicate-weight removal.
4. Load shared params in the browser through TVMJS `fetchTensorCache` before VM invocation; keep `shader-f16` gating and ORT fallback behavior unchanged. The research smoke has an opt-in fetch path:

   ```bash
   npm run lc0:tvmjs-webgpu-smoke -- \
     --batch 8 \
     --no-fixtures \
     --tensor-cache \
     --out artifacts/tvm/lc0_tvmjs_webgpu_tensor_cache_fetch_smoke.json
   ```

   This validates browser fetch/cache plumbing and records `startupTimings.tensorCacheFetchMs`, but it does not prove duplicate weights were removed unless the export used detached params.
5. Compare embedded vs tensor-cache on cold start, repeat-load cache hit behavior, raw/gzip/Brotli footprint, search parity, and Stockfish-scored deltas before publication.

Local static `Content-Encoding` smoke:

```bash
npm run lc0:tvmjs-static-content-encoding-smoke -- \
  --file public/runtimes/lc0-tvmjs-webgpu/t1-256x10-distilled-swa-2432500/f16/v1/tvmjs_runtime.wasm \
  --out artifacts/tvm/lc0_tvmjs_static_content_encoding_smoke_runtime_wasm.json
```

Current local static smoke summary:

- Artifact: `artifacts/tvm/lc0_tvmjs_static_content_encoding_smoke_runtime_wasm.json`.
- Status: `ok: true`.
- For `Accept-Encoding: br, gzip`, response header `Content-Encoding: br`, `Content-Type: application/wasm`, bytes `673,133`.
- For `Accept-Encoding: gzip`, response header `Content-Encoding: gzip`, `Content-Type: application/wasm`, bytes `1,200,884`.
- For `Accept-Encoding: identity`, no `Content-Encoding`, raw bytes `5,354,374`.
- Caveat: this proves the local static-serving semantics for temporary sidecars only. Production still depends on host/CDN metadata and rewrites for the exact published path.

## Recipes for additional net families

Use a stable `<family>/<precision>/<version>` layout and commit only manifests/scripts/docs unless release policy says otherwise.

Recommended identifiers:

- Large BT4: `bt4-large/<net-id>/f16/v1`
- BT4 Human Sparring Nets: `bt4-human-sparring/<net-id>/f16/v1`
- BT4 Odds Bots: `bt4-odds/<net-id>/f16/v1`

Per net family:

1. Place or generate ONNX exports for all intended browser batch sizes, ideally b1/b4/b8.
2. Confirm input/output names and shapes match the LC0 classical 112-plane path or document adapter differences.
3. Run the whole-model TVMJS export with the required mitigations.
4. Stage artifacts to a family-specific manifest path.
5. Run local artifact check and loader smoke.
6. Run fixture eval/search against ORT f16 WebGPU.
7. Run same-session TVMJS vs hybrid comparison if a hybrid bundle exists for that model family.
8. Update evidence summary and promotion-readiness notes.

Template environment for a new net:

```bash
export LC0_WEB_REPO="$WEB"
export LC0_TVMJS_MODEL_FAMILY='bt4-large/<net-id>'
export LC0_TVMJS_DTYPE='f16'
export LC0_TVMJS_VERSION='v1'
export LC0_TVMJS_BATCHES='1,4,8'
# Default model template is: {family}.batch{batch}.{dtype}.onnx
# Override when source ONNX names differ, e.g.:
# export LC0_TVMJS_MODEL_TEMPLATE='{family}.b{batch}.{dtype}.onnx'
```

Check resolved export inputs before compiling:

```bash
cd "$WEB"
DRY_RUN=1 ./scripts/run_lc0_tvm_whole_onnx_probe.sh
```

Export with the same TVMJS mitigations:

```bash
cd "$WEB"
CAST_INT64_INITIALIZERS_TO_INT32=1 \
TRUST_NONNEGATIVE_GATHER_INDICES=1 \
SANITIZE_ONNX_NAMES=1 \
EXPORT_TVMJS_WASM=1 \
TVM_BUILD_DIR=build-tvmjs \
TVM_HOST_TARGET='{"kind":"llvm","mtriple":"wasm32-unknown-unknown-wasm"}' \
./scripts/run_lc0_tvm_whole_onnx_probe.sh
```

Stage a family-specific bundle:

```bash
cd "$WEB"
npm run lc0:stage-tvmjs-webgpu -- \
  --model-family="$LC0_TVMJS_MODEL_FAMILY" \
  --dtype="$LC0_TVMJS_DTYPE" \
  --version="$LC0_TVMJS_VERSION" \
  --batches="$LC0_TVMJS_BATCHES"
```

The staging script also accepts `--stem-template='{modelFamily}.batch{batch}.{dtype}.webgpu.tvmjs-wasm.probe'` when exported artifact stems differ from the default.

The export and staging scripts are parameterized for model family, dtype, batch list, naming templates, and destination path. Check the staged manifest/files for the exact family before running or publishing evidence:

```bash
npm run lc0:tvmjs-webgpu-local-artifacts-check -- \
  --manifest "public/runtimes/lc0-tvmjs-webgpu/$LC0_TVMJS_MODEL_FAMILY/$LC0_TVMJS_DTYPE/$LC0_TVMJS_VERSION/manifest.json" \
  --no-evidence \
  --expected-model-family "$LC0_TVMJS_MODEL_FAMILY" \
  --expected-dtype "$LC0_TVMJS_DTYPE" \
  --expected-version "$LC0_TVMJS_VERSION" \
  --expected-batches "$LC0_TVMJS_BATCHES"
```

Once family-specific evidence exists, drop `--no-evidence` and pass `--evidence <family-summary.json>` plus appropriate thresholds:

```bash
npm run lc0:tvmjs-webgpu-local-artifacts-check -- \
  --manifest "public/runtimes/lc0-tvmjs-webgpu/$LC0_TVMJS_MODEL_FAMILY/$LC0_TVMJS_DTYPE/$LC0_TVMJS_VERSION/manifest.json" \
  --evidence artifacts/tvm/<family>_tvmjs_webgpu_search_smoke_summary.json \
  --expected-model-family "$LC0_TVMJS_MODEL_FAMILY" \
  --expected-dtype "$LC0_TVMJS_DTYPE" \
  --expected-version "$LC0_TVMJS_VERSION" \
  --expected-batches "$LC0_TVMJS_BATCHES" \
  --min-search-rows 32 \
  --min-fixed-suite-reports 1 \
  --min-stockfish-scored-runs 1 \
  --require-all-matches
```

## Generated artifact release/hosting/cache policy

Current policy: generated TVMJS wasm/runtime artifacts are local research artifacts and must stay ignored/unpublished until a release owner explicitly promotes a bundle. A promotable release needs all of the following:

1. A manifest-specific local artifact check for the exact model family/dtype/version/batches.
2. Evidence summary covering loader, evaluator parity, search parity, fixed-suite bridge rows, and same-session hybrid/ORT comparisons where applicable.
3. Raw size plus gzip/Brotli sidecar size audit for `tvmjs.bundle.js`, `tvmjs_runtime.wasm`, each model wasm, probe metadata, and tensor-cache shards if separated params are used.
4. Host/CDN configuration that serves compressed sidecars with correct `Content-Encoding`; do not assume safe Brotli/gzip serving from file extension alone.
5. Immutable versioned paths (`.../<model-family>/<dtype>/<version>/`) and long-lived cache headers only for content-hashed/manifest-pinned files. Manifests should use a shorter revalidation policy unless their path is immutable and release-tagged.
6. Rollback plan: keep ORT ONNX/WebGPU as the stable default, and require opt-in TVMJS selection until promotion evidence and artifact policy are accepted.
7. Compiler provenance in the staged manifest: TVM commit/dirty status, TVM build dir, Emscripten version, required WebGPU features, and parameter strategy.

Suggested numeric promotion gates to ratify before stable exposure:

- Correctness: TVMJS-vs-ORT f16 best-move/search-move parity is `100%` on the agreed fixed suite, with an explicit f16 numeric drift tolerance for policy top-prior, WDL/Q, and MLH.
- Search quality: Stockfish-scored TVMJS-minus-ORT move deltas are within the accepted cp/mate threshold on the fixed suite; all nonzero deltas are triaged.
- Performance: TVMJS visits/s or mean search latency beats ORT f16 WebGPU by the agreed margin on the reference device and does not regress beyond tolerance on secondary devices.
- Startup: cold start and repeat-load cached start fit the accepted budget, including TVMJS wasm fetch/verify, instantiate, WebGPU pipeline prebuild, VM creation, tensor-cache load if used, and first useful search.
- Footprint: raw/gzip/Brotli artifact footprint fits the accepted size budget; duplicated batch-specific weight storage is either removed or explicitly accepted.
- Coverage: pass on at least the agreed Chrome/WebGPU device matrix, including a non-Apple GPU, plus documented fallback behavior when `shader-f16` or WebGPU is unavailable.
- Isolation: LC0 whole-model TVMJS stays outside shared registry/default flow. Visible non-v0 page-level selection is approved; v0 still forces ONNX.

Do not add TVMJS to `src/nn/runtimeRegistry.ts`, `src/nn/browserRuntimeEvaluator.ts`, or the stable arena runtime UI as part of artifact publication alone.

## Historical branch closeout checklist (`tvmjs-hybrid-research-runbooks`)

This records the pre-promotion branch workflow, not current Git instructions.
Its evidence does not establish a public release. At that stage:

1. Confirm the worktree is clean except the Ralph task file in the parent coordination repo, if that repo tracks it separately.
2. Confirm `git log --oneline main..tvmjs-hybrid-research-runbooks` contains only research-lane commits and no generated TVMJS wasm/runtime artifacts.
3. Re-run:

   ```bash
   npm run typecheck
   npm run lc0:tvmjs-webgpu-research-gate
   npm run lc0:tvmjs-research-only-check
   git diff --check
   ```

4. Run the local review gate with the same validation bundle:

   ```bash
   autoreview --mode local --parallel-tests "npm run typecheck && npm run lc0:tvmjs-webgpu-research-gate"
   ```

5. Push the branch only after the checks pass:

   ```bash
   git push -u origin tvmjs-hybrid-research-runbooks
   ```

6. In current merge notes, distinguish approved non-default/non-v0 LC0 page selection from shared-registry/default promotion. Do not claim LC0 TVMJS is absent from all UI.
7. Mention ignored/local artifact policy: generated TVMJS wasm/runtime files and evidence artifacts are not release payloads unless a release owner separately accepts the hosting/cache policy.
8. Include the current evidence highlights: fixed-suite bridge `94/94` TVMJS-vs-ORT f16 search move matches, same-session TVMJS-vs-hybrid visit samples at `16/32/64`, GPU allocation-request telemetry, startup/amortization sidecar, bundle footprint sidecar, and local static `Content-Encoding` smoke.
9. A research/runbook merge alone does not authorize public/default promotion. Broader fixed-suite evidence, release-host verification, drift policy and artifact-publication approval remain separate gates.

## Evidence naming convention

Use names that encode lane, model, suite, batch, visits, repeats, and baseline:

```text
artifacts/tvm/<model-id>_<suite>_b<batch>_v<visits>_r<repeats>_tvmjs_vs_ortf16.json
artifacts/tvm/<model-id>_<suite>_b<batch>_v<visits>_r<repeats>_tvmjs_vs_hybrid.json
```

Keep generated artifacts ignored unless release policy changes.

## Original whole-ONNX probe results (historical, 2026-06)

The original probe used `scripts/lc0_tvm_whole_onnx_probe.py` and
`scripts/run_lc0_tvm_whole_onnx_probe.sh`. The three import mitigations and
browser-export commands are consolidated above. Optional
`--capture-module-sources` / `CAPTURE_MODULE_SOURCES=1` captures WGSL/C next to
the ignored JSON. Use the installed `apache-tvm-ffi` wheel, not an extra
`3rdparty/tvm-ffi/python` entry on `PYTHONPATH`.

### Native-export limitation and toolchain outcome

The first `.tvm_export` was a macOS Mach-O shared library embedding WebGPU.
Python could compile/export it but `tvm.device("webgpu", 0).exist == false`;
loading a copied `.dylib` failed on `ffi.Module.load_from_bytes.webgpu`.
That milestone was compile/source/export success, not browser parity/perf.
The browser path instead used WebGPU target + wasm32 LLVM host and
`tvm.contrib.tvmjs.create_tvmjs_wasm`. Native export is skipped in that mode:
the platform C++ linker cannot link wasm32 host objects.

The local toolchain used emcc 5.0.7, web runtime bitcode under
`.deps/tvm-webgpu-src/web/dist/wasm/`, and an LLVM-enabled `build-tvmjs` with
`TVM_LIBRARY_PATH` selecting its `lib/`. The wrapper assumes a `leelaweb`
checkout; set `LC0_WEB_REPO` when the checkout has another name. The maintained
build/check commands above replace the duplicated bootstrap recipes.

Original ignored `artifacts/tvm/` native/source probe sizes:

| batch | model bytes | export bytes | WGSL bytes | C bytes | kernels | Relax i64 tokens | `shape_to_tensor` tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 40,491,226 | 41,864,688 | 732,713 | 1,053,493 | 39 | 0 | 0 |
| 4 | 40,491,226 | 42,145,392 | 1,013,578 | 1,060,542 | 39 | 0 | 0 |
| 8 | 40,491,226 | 44,258,928 | 1,018,009 | 1,060,560 | 39 | 0 | 0 |

Stage timings: ONNX load `0.015–0.027s`, Relax import `0.045–0.080s`, WebGPU
build `0.54–0.61s`, source capture `~0.004s`, native export `22–24s`.
Browser model WASMs named
`artifacts/tvm/t1-256x10-distilled-swa-2432500.batch{1,4,8}.f16.webgpu.tvmjs-wasm.probe.tvmjs.wasm`
were respectively `44,308,011`, `44,611,496`, and `44,644,785` bytes.

### Loader, invocation and fixture results

`lab/lc0-tvmjs-webgpu-smoke.html` reported `SMOKE_OK` with
`vm_load_executable` for b1/b4/b8: manifest fetch, SHA/size check,
`shader-f16` device, asynchronous `tvmjs.instantiate` (sync compile is blocked
for >8 MB WASM), `initWebGPU`, `systemLib`, and pipeline prebuild.
Zero-input VM invocation uploaded `[batch,112,8,8]` f16, called
`set_input` / `invoke_stateful`, retrieved outputs and synced/read back:

| Batch | `invoke_stateful` smoke timing | Outputs |
| --- | ---: | --- |
| 1 | about `4.0 ms` | `[1,1858]`, `[1,3]`, `[1,1]` f16 |
| 4 | about `4.1–5.5 ms` | `[4,1858]`, `[4,3]`, `[4,1]` f16 |
| 8 | about `5.4–5.9 ms` | `[8,1858]`, `[8,3]`, `[8,1]` f16 |

Fixture mode used `encodeLc0Classical112(..., {historyFill: "fen_only"})`,
f16 conversion, legal-prior decoding, and native BLAS rows from
`fixtures/lc0/native_fen_only_blas.jsonl`. The original b8 smoke matched native
bestmove `8/8`: max top-prior diff `0.004802150612562506`, Q diff
`0.004474531249999997`, MLH diff `0.2749999999999986`; observed invoke about
`4.2 ms`.

Original same-page ORT comparisons:

| Artifact | ORT compare | Best move | Max top-prior abs diff | Max Q abs diff | Max MLH abs diff |
| --- | --- | ---: | ---: | ---: | ---: |
| batch8 vs ORT f16 WebGPU | f16 | `8/8` | `0.009859013926522886` | `0.003173828125` | `0.25` |
| batch1 vs ORT f16 WebGPU | f16 | `1/1` | `0.0006974565401345473` | n/a | n/a |
| batch1 vs ORT f32 WebGPU | f32 | `1/1` | `0.0004851487340732652` | n/a | n/a |

At that stage b1 WDL/MLH reads were skipped because their byte lengths were
not 4-byte aligned. This is fixed, not a current limitation: the 2026-06-10
runtime patch pads copies and returns only requested bytes; b1 WDL is 6 B and
MLH 2 B. See [`lc0_tvmjs_bt4it332_onboarding_2026-06-09.md`](./lc0_tvmjs_bt4it332_onboarding_2026-06-09.md)
and [`upstream/PR_tvm_webgpu_readback.md`](./upstream/PR_tvm_webgpu_readback.md).
Do not reinterpret the historical n/a cells as new measurements.

All-10-fixture coverage report:
`artifacts/tvm/lc0_tvmjs_webgpu_fixture_coverage_summary.json`. Native and ORT
f16 bestmove both `10/10`; maximum native prior/Q/MLH differences remained
`0.004802150612562506` / `0.004474531249999997` / `0.2749999999999986`;
ORT f16 differences `0.009859013926522886` / `0.003173828125` / `0.25`.

### Original search-smoke aggregate (v9)

`lc0_browser.tvmjs_webgpu_search_smoke_summary.v9` covered 94 rows at visits
4/16/32, all `94/94` TVMJS-vs-ORT f16 move matches. All-fixture 16- and
32-visit runs each had native eval, ORT eval and search `10/10`. UHO-lite
8-position v16 had ORT eval/search `8/8`. The 16-position repeat-2 run had
ORT eval `16/16`, search `32/32`, TVMJS mean/median `43.356/40.185 ms` versus
ORT `50.739/47.087 ms`. Stockfish-depth-3 runs with 4/8/16 positions had
4/8/16 search/scored rows and 3/7/15 comparable cp deltas, all mean/min/max
`0/0/0`. Maximum search wall was `82.760 ms` TVMJS / `93.840 ms` ORT.

Original local evidence paths (not committed payloads):

```text
artifacts/tvm/lc0_tvmjs_webgpu_smoke_batch8.json
artifacts/tvm/lc0_tvmjs_webgpu_smoke_batch8_ortf16_webgpu.json
artifacts/tvm/lc0_tvmjs_webgpu_smoke_batch1_ortboth_webgpu.json
artifacts/tvm/lc0_tvmjs_webgpu_search_smoke_v4_batch8_first2.json
artifacts/tvm/lc0_tvmjs_webgpu_search_smoke_v16_batch8_first2.json
artifacts/tvm/lc0_tvmjs_webgpu_search_smoke_v16_batch8_promotion2.json
artifacts/tvm/lc0_tvmjs_webgpu_search_smoke_v16_batch8_all10.json
artifacts/tvm/lc0_tvmjs_webgpu_search_smoke_v32_batch8_all10.json
artifacts/tvm/lc0_tvmjs_webgpu_search_smoke_v16_batch8_uho_lite8.json
artifacts/tvm/lc0_tvmjs_webgpu_search_smoke_v16_batch8_uho_lite16_repeat2.json
artifacts/tvm/lc0_tvmjs_webgpu_search_smoke_v16_batch8_uho_lite4_sfdepth3.json
artifacts/tvm/lc0_tvmjs_webgpu_search_smoke_v16_batch8_uho_lite8_sfdepth3.json
artifacts/tvm/lc0_tvmjs_webgpu_search_smoke_v16_batch8_uho_lite16_sfdepth3.json
artifacts/tvm/lc0_tvmjs_webgpu_search_smoke_summary.json
artifacts/tvm/lc0_tvmjs_webgpu_fixed_suite_uho_lite16_repeat2_report.json
artifacts/tvm/lc0_tvmjs_webgpu_fixed_suite_uho_lite4_sfdepth3_report.json
artifacts/tvm/lc0_tvmjs_webgpu_fixed_suite_uho_lite8_sfdepth3_report.json
artifacts/tvm/lc0_tvmjs_webgpu_fixed_suite_uho_lite16_sfdepth3_report.json
```

The source corpus included the
10 fixture rows and first 16 UHO-lite v1 rows. That v1 suite later proved to
have a corrupt row; use the recovery/correction note above, not an assumed
current sibling checkout. These are warm, short smoke slices, not a native
visit-distribution or universal throughput claim. Report v1 was unscored;
Stockfish-scored v2 explicitly remained smoke-derived rather than output of
`lc0_browser_runtime_fixed_suite.mjs`. The original local evidence gate required
all matches, at least 94 rows, 3 scored runs and 4 fixed-suite-style reports.
The later self-hosted promotion is separate from those thresholds.

## Compact t1 campaign outcome (historical, 2026-06-09)

Scope: M-chip Mac WebGPU/WASM, f16 t1-256x10-distilled. This is not a current
QDQ/graph-capture benchmark or a cross-GPU conclusion.

| Question | Dated verdict |
| --- | --- |
| Visit loop v16/b8 | 86% GPU wait, JS search ≈2 ms (non-lever) |
| GPU wait | ~9 ms real arithmetic, 229 passes, top four matmul families ≈63% |
| Pipeline depth 2 | +27% slower; fragments batch fill |
| Pass coalescing 5928→26 | No resolved gain |
| Dlight default | Kernel −8%, end-to-end nil; Matmul applied to all 19 kernels |
| Eight tile configs | Default local optimum; vec>1 illegal on N=3/1858 heads; tuning parked on this device |
| Batch scaling | b8/b16/b32 costs 15.1/19.4/30.3 ms; b16 cuts v16/v32 search 22–26%; b32 wins from ~v64 |

Bottom line: b16/v16 ≈39–40 ms versus ORT f16 ≈45–51 ms, `16/16` UHO move
match and cp delta `0/0/0`. Detached params cut model WASM 44.6→4.5 MB plus
~38 MB raw-f16 cache; ≤4096-element initializers stayed inline to preserve
shape constants. b1/b16 caches were byte-identical. The shared bundle was
54.7 MB raw / 36.8 MB Brotli; warm 40.2 MB cache read was 21 ms versus
235 ms cold fetch/upload. The detailed records above retain controls/caveats.

Reusable infrastructure included visit-loop/timestamp attribution,
`--kernel-profile-invokes`, pass coalescing, `evaluateBatchSequence`,
`--detach-params`, Dlight configs, fuse depth, fixed batch, JSON input dtype,
gather mitigations, scheduler crash fallback, and parameterized manifests /
model families. Keep pipeline depth 1. At the campaign date, non-Apple evidence
and public hosting/cache policy were deferred; those caveats were accepted only
for later self-hosted non-default selection.

The old “next target BT4” export/stage/checklist is completed, not an open task.
The policytune-332 results (including int8/int4, tie tolerance, detached sharing
and the readback fix) live in
[`lc0_tvmjs_bt4it332_onboarding_2026-06-09.md`](./lc0_tvmjs_bt4it332_onboarding_2026-06-09.md).

# Centipawn (deferred)

The following Tiny Leela probe subsection and campaign discipline bullet are
retained verbatim. They are historical records, not a current Centipawn audit.

## Tiny Leela (squareformer_v2) TVMJS probe (2026-06-09)

First attempt at the project's own model family
(`public/models/bt4_anneal_muon_best.onnx`, squareformer_v2 6×128, uint8
compact-token inputs, PyTorch export). Status: **compiles, loads, and runs on
WebGPU; numerics are wrong; blocked on upstream TVM frontend bugs.**

What it took to compile (new probe capabilities, all opt-in):

1. onnxsim with `overwrite_input_shapes` (batch fixed to 16) — folds all
   `Shape`/`Gather`/`Concat` dim arithmetic; the raw export fails Relax import.
2. `--dtype '{"tokens":"int32"}'` (JSON per-input dtype) — i64 graph input.
3. `--trust-runtime-gather-indices` — the runtime embedding gathers otherwise
   emit i64 `shape_to_tensor` wrap logic; this model clamps indices in-graph.
4. Gather patch now skips Shape-typed data (PyTorch `Shape→Gather` pattern).
5. `--max-fuse-depth 6` (`relax.FuseOps.max_depth`) — the 21-term embedding
   sum otherwise fuses into one kernel needing 12 storage buffers vs the
   device's 10/stage limit. Relax FuseOps ignores target `max_function_args`
   (hardcoded 0 at `fuse_ops.cc:1053`; webgpu target kind doesn't declare it) —
   upstream gap. `--no-fuse-ops` also exists but unfused bool intermediates
   become `array<i8>` WGSL buffers, which WGSL rejects — second upstream gap.
6. Dlight rule crash fall-through (rfactor `bind` fails on the head-sum
   reduction with both the default scheduler and `dl.gpu.Reduction`); crashed
   functions now go straight to `Fallback`.

Browser result (`tiny-tvmjs-webgpu-smoke.html`, b16, synthetic inputs vs ORT
on the same fixed-batch ONNX): pipelines build (needs adapter
`maxStorageBuffersPerShaderStage` in requiredLimits), invoke+readback of all
four outputs ≈ `29.6 ms`, all outputs finite — but `maxAbsDiff` vs ORT is
~`36` on policy/`8` on wdl. TVM is internally consistent (CPU llvm build
produces the same wrong values as WebGPU), so this is a deterministic
op-conversion divergence in the Relax ONNX frontend (suspects:
`LayerNormalization`/`Gemm`/`BitwiseAnd`/`Slice`/Clip-isnan paths — ops the
LC0 lane never exercised). Separately, **unseeded imports are
nondeterministic**: identical scripts produce all-NaN outputs on some runs and
finite ones on others; pinning `PYTHONHASHSEED` makes builds deterministic.
Both issues are upstream-class TVM bugs.

### Numerics bisection result (2026-06-09, same day)

The divergence was bisected to a **TVM compiler-pass bug in constant scalar
Gather-index folding**, not the frontend and not fusion:

- Coarse probes: the `/ReduceSum` piece-embedding sum is exact; the first wrong
  tensor is in the embedding stem (`/Clip_2`, the rank-column clamp), with
  integer-exact diffs.
- Empirically, TVM's "rank" path computes `clamp(tokens[:,:,15], 0, 7)` — it
  reads token column **15** (square) instead of column **12** (rank). The
  imported Relax IR is verified correct (the scalar index wrap chains carry
  `R.const 8/9/10/11/12` exactly as the ONNX specifies), so
  LegalizeOps/FoldConstant/build mis-folds the `shape_to_tensor`/`take`/`where`
  scalar-index chains. Reproduces identically with fusion on, with a forced
  fusion boundary, and with FuseOps disabled (per-op kernels).
- Deterministic repro: `artifacts/tvm-tiny/cut_add4_boundary.onnx` (45 nodes,
  extracted via `onnx.utils.extract_model` to `/Add_4_output_0` +
  `/Clip_2_output_0`) with `artifacts/tvm-tiny/repro_inputs.npz`; tiny cuts of
  the same ops in isolation (2–3 nodes) are exact, so context (the constant
  pool of the surrounding graph) is required.
- Separately, larger programs built from the **byte-identical imported module**
  show run-to-run output variance (one process produced four different outputs
  from one build; others alternate clean/all-NaN per build) — an
  uninitialized-memory-class bug in build/runtime. WebGPU's zero-initialized
  buffers make the browser behavior deterministic (it matches the zero-heap CPU
  outputs), which is why the browser smoke looked stably wrong.

Practical consequences: the LC0 lane is structurally unaffected (no runtime
scalar-index gather chains survive its import thanks to the gather patches, and
its parity is gated against native fixtures per build), but per-build parity
gating remains mandatory for ANY TVM export. The Tiny lane is blocked on the
upstream pass bug; the repro above is the attachment for a TVM issue. A
frontend-level workaround sketch (emit `R.const` scalar indices directly
instead of wrap chains for in-range constant Gather indices) is already what
`--trust-nonnegative-gather-indices` does — but the full patched model still
diverges, so at least one more mis-folded pattern exists beyond the gather
chains; finding it follows the same extract-and-cut method.

- Mandatory discipline (reinforced by the Tiny lane bisection, commit
  `e1ae54c`): **per-build parity gating against native fixtures for every TVM
  artifact** — the TVM toolchain has demonstrated compiler-level correctness
  bugs and build nondeterminism on other model shapes.
