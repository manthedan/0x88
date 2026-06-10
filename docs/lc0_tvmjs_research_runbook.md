# LC0 TVMJS/WebGPU research runbook

Status: research-only. ORT ONNX/WebGPU remains the stable default. Do not add TVMJS to the runtime registry or default UI until promotion-grade evidence and artifact release policy exist.

## What is close to promotion

The whole-model LC0 f16 ONNX → TVM Relax → TVMJS/WebGPU path is mechanically real:

- durable source-built TVM/TVMJS toolchain exists;
- b1/b4/b8 f16 WebGPU model wasm artifacts can be exported;
- browser loading verifies manifest bytes and SHA-256;
- browser execution requests `shader-f16`, instantiates the TVM runtime asynchronously, initializes WebGPU, prebuilds pipelines, and runs the Relax VM;
- smoke evidence compares real fixture eval/search rows against ORT f16 WebGPU.

It is **not promotion-ready** yet because the evidence is still smoke-derived and generated artifacts are local/ignored. Promotion remains blocked by the checklist below.

## Promotion readiness snapshot

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

## Promotion blockers

- Expand direct same-session full-model TVMJS vs custom hybrid TVM/WGSL comparison beyond smoke-sized rows.
- Production-style fixed-suite integration, not only `lc0-tvmjs-webgpu-smoke.html`.
- Repeated throughput and startup/pipeline compile amortization evidence.
- Broader cross-runtime GPU allocation/footprint instrumentation and interpretation, including ORT WebGPU API diagnostics limits.
- Release/hosting/cache policy for generated model/runtime wasm artifacts.
- f16 drift/tolerance policy against native/ORT baselines.

## Durable local paths

From project root `/Users/macthedan/projects/lc0_browser`:

```bash
export ROOT=/Users/macthedan/projects/lc0_browser
export WEB=$ROOT/leelaweb-lc0-top-roi   # or another up-to-date leelaweb worktree
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

For the current default LC0 f16 b1/b4/b8 ONNX artifacts:

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

Research-only fixed-suite bridge, using the same fixed FEN input style as `lc0_browser_runtime_fixed_suite.mjs` but without adding TVMJS to the stable runtime registry/UI:

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

1. TVMJS full-model WebGPU via `lc0-tvmjs-webgpu-smoke.html`.
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

Strict same-FEN visit sweep samples now cover visits `16`, `32`, and `64`:

- `artifacts/tvm/lc0_tvmjs_vs_hybrid_uho_b8_hb4_v16_n1_r1_hybrid_alloc.json`: rows `1`, TVMJS-vs-hybrid `1/1`, TVMJS mean `78.205 ms`, hybrid mean `404.665 ms`.
- `artifacts/tvm/lc0_tvmjs_vs_hybrid_uho_b8_hb4_v32_n2_r1.json`: rows `2`, TVMJS-vs-hybrid `2/2`, TVMJS mean `97.54 ms`, hybrid mean `299.795 ms`.
- `artifacts/tvm/lc0_tvmjs_vs_hybrid_uho_b8_hb4_v64_n2_r1.json`: rows `2`, TVMJS-vs-hybrid `2/2`, TVMJS mean `180.55 ms`, hybrid mean `457.143 ms`.

## GPU allocation instrumentation status

Current prototypes:

- `lc0-tvmjs-webgpu-smoke.html` wraps the acquired TVMJS `GPUDevice.createBuffer` before TVM initialization and model execution. Results report total allocation calls/bytes, max buffer size, mapped-at-creation count, and categories decoded from `GPUBufferUsage` flags.
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
- Isolation: TVMJS remains absent from stable runtime registry/UI/default arena flow until release owner approval explicitly changes that policy.

Do not add TVMJS to `src/nn/runtimeRegistry.ts`, `src/nn/browserRuntimeEvaluator.ts`, or the stable arena runtime UI as part of artifact publication alone.

## Branch push/merge checklist for `tvmjs-hybrid-research-runbooks`

Before pushing or merging this research branch:

1. Confirm the worktree is clean except the Ralph task file in the parent coordination repo, if that repo tracks it separately.
2. Confirm `git log --oneline main..tvmjs-hybrid-research-runbooks` contains only research-lane commits and no generated TVMJS wasm/runtime artifacts.
3. Re-run:

   ```bash
   npm run typecheck
   npm run lc0:tvmjs-webgpu-research-gate
   npm run lc0:tvmjs-research-only-check
   git diff --check
   ```

4. Run autoreview with the same validation bundle:

   ```bash
   /Users/macthedan/.pi/agent/skills/autoreview/scripts/autoreview --mode local --parallel-tests "npm run typecheck && npm run lc0:tvmjs-webgpu-research-gate"
   ```

5. Push the branch only after the checks pass:

   ```bash
   git push -u origin tvmjs-hybrid-research-runbooks
   ```

6. In the PR/merge notes, state explicitly that TVMJS remains research-only and is not added to `src/nn/runtimeRegistry.ts`, `src/nn/browserRuntimeEvaluator.ts`, stable runtime UI, or default arena flow.
7. Mention ignored/local artifact policy: generated TVMJS wasm/runtime files and evidence artifacts are not release payloads unless a release owner separately accepts the hosting/cache policy.
8. Include the current evidence highlights: fixed-suite bridge `94/94` TVMJS-vs-ORT f16 search move matches, same-session TVMJS-vs-hybrid visit samples at `16/32/64`, GPU allocation-request telemetry, startup/amortization sidecar, bundle footprint sidecar, and local static `Content-Encoding` smoke.
9. Treat merge as a research/runbook merge only; TVMJS promotion remains blocked by broader production fixed-suite evidence, release-host verification, f16 drift/tolerance policy, and accepted artifact publication policy.

## Evidence naming convention

Use names that encode lane, model, suite, batch, visits, repeats, and baseline:

```text
artifacts/tvm/<model-id>_<suite>_b<batch>_v<visits>_r<repeats>_tvmjs_vs_ortf16.json
artifacts/tvm/<model-id>_<suite>_b<batch>_v<visits>_r<repeats>_tvmjs_vs_hybrid.json
```

Keep generated artifacts ignored unless release policy changes.
