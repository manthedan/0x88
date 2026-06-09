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

## Promotion blockers

- Direct same-session full-model TVMJS vs custom hybrid TVM/WGSL comparison.
- Production-style fixed-suite integration, not only `lc0-tvmjs-webgpu-smoke.html`.
- Repeated throughput and startup/pipeline compile amortization evidence.
- GPU allocation/footprint instrumentation.
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

Aggregate the current local research evidence:

```bash
cd "$WEB"
npm run lc0:tvmjs-webgpu-summary -- \
  --artifact artifacts/tvm/lc0_tvmjs_webgpu_smoke_b8_v16_ortf16.json \
  --out artifacts/tvm/lc0_tvmjs_webgpu_fixed_suite_b8_v16_ortf16_report.json
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

The wrapper writes an aggregate `lc0_browser.tvmjs_vs_hybrid_matrix.v1` JSON artifact and child artifacts for each lane. With `--fens`, both lanes run the same FEN rows. The TVMJS child artifact and aggregate include `startupTimings` for manifest fetch, TVMJS bundle load, WebGPU adapter/device acquisition, wasm fetch/verification, async wasm instantiate, TVM WebGPU init, `systemLib`, WebGPU pipeline prebuild, VM creation, input tensor allocation, and input upload.

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
export LC0_TVMJS_MODEL_ID='<family>/<net-id>'
export LC0_TVMJS_PRECISION='f16'
export LC0_TVMJS_VERSION='v1'
# TODO: wire these env vars into staging/export scripts before relying on them.
```

Current scripts are still default-model oriented. Before repeating this at scale, parameterize the export/staging/check scripts so model id, ONNX paths, precision, batch list, and destination manifest path are explicit CLI/env inputs.

## Evidence naming convention

Use names that encode lane, model, suite, batch, visits, repeats, and baseline:

```text
artifacts/tvm/<model-id>_<suite>_b<batch>_v<visits>_r<repeats>_tvmjs_vs_ortf16.json
artifacts/tvm/<model-id>_<suite>_b<batch>_v<visits>_r<repeats>_tvmjs_vs_hybrid.json
```

Keep generated artifacts ignored unless release policy changes.
