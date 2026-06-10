# LC0 whole-model ONNX → TVM Relax → WebGPU probe

Status: diagnostic/research artifact path only. This does **not** change the default browser runtime; ORT ONNX/WebGPU remains the stable default.

## Durable toolchain

From project root `/Users/macthedan/projects/lc0_browser`:

```bash
export TVM_SRC="$PWD/.deps/tvm-webgpu-src"
export TVM_ENV="$PWD/.envs/tvm-mlc-py313"
export TVM_LIBRARY_PATH="$TVM_SRC/build/lib"
export DYLD_LIBRARY_PATH="$TVM_SRC/build/lib:${DYLD_LIBRARY_PATH:-}"
export PYTHONPATH="$TVM_SRC/python:${PYTHONPATH:-}"
```

Do not use `/tmp` for this toolchain state. Do not add `3rdparty/tvm-ffi/python` to `PYTHONPATH`; use the installed `apache-tvm-ffi` wheel plus the source-built TVM libraries.

## Scripts

- `scripts/lc0_tvm_whole_onnx_probe.py`
- `scripts/run_lc0_tvm_whole_onnx_probe.sh`

Successful b1/b4/b8 WebGPU codegen currently requires all three opt-in mitigations:

1. `--cast-int64-initializers-to-int32`
   - LC0 ONNX exports shape/split/reshape constants as `INT64`; TVM WebGPU codegen rejects i64.
2. `--trust-nonnegative-gather-indices`
   - TVM ONNX `Gather` lowering emits signed-index wrap logic for constant policy mapping indices, leaving dead `shape_to_tensor`/i64 paths. LC0 policy mapping indices are constant and nonnegative, so the diagnostic patch emits direct `relax.op.take`.
3. `--sanitize-onnx-names`
   - ONNX names such as `/input/planes` otherwise leak into generated C wrappers as invalid identifiers.

Optional source capture:

- `--capture-module-sources` writes generated WGSL/C next to the JSON artifact under ignored `artifacts/tvm/`.

## Reproduce b1/b4/b8 codegen

From `leelaweb/`:

```bash
CAST_INT64_INITIALIZERS_TO_INT32=1 \
TRUST_NONNEGATIVE_GATHER_INDICES=1 \
SANITIZE_ONNX_NAMES=1 \
CAPTURE_MODULE_SOURCES=1 \
./scripts/run_lc0_tvm_whole_onnx_probe.sh
```

## Current artifact summary

Generated under ignored `artifacts/tvm/`:

| batch | model bytes | export bytes | WGSL bytes | C bytes | kernels | Relax i64 tokens | `shape_to_tensor` tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 40,491,226 | 41,864,688 | 732,713 | 1,053,493 | 39 | 0 | 0 |
| 4 | 40,491,226 | 42,145,392 | 1,013,578 | 1,060,542 | 39 | 0 | 0 |
| 8 | 40,491,226 | 44,258,928 | 1,018,009 | 1,060,560 | 39 | 0 | 0 |

Stage timings observed on the durable local build were approximately:

- ONNX load: `0.015–0.027s`
- Relax import: `0.045–0.080s`
- Relax build target `webgpu`: `0.54–0.61s`
- source capture: `~0.004s`
- native export: `22–24s`

## Runtime limitation

The generated `.tvm_export` is a macOS Mach-O shared library embedding a WebGPU module. Local Python TVM can compile and export it, but cannot run it here:

- `tvm.device("webgpu", 0).exist == false`
- loading a copied `.dylib` fails with missing `ffi.Module.load_from_bytes.webgpu`

So the current milestone is **compile/source/export success**, not browser parity/perf.

## TVMJS/WebGPU browser export path

This TVM checkout includes a `web/` runtime. Its tests use this pattern for browser/WebGPU artifacts:

```python
target = tvm.target.Target(
    "webgpu",
    host={"kind": "llvm", "mtriple": "wasm32-unknown-unknown-wasm"},
)
executable.export_library(wasm_path, fcompile=tvm.contrib.tvmjs.create_tvmjs_wasm)
```

Check readiness first:

```bash
cd leelaweb
ROOT=/Users/macthedan/projects/lc0_browser
export TVM_SRC="$ROOT/.deps/tvm-webgpu-src"
export TVM_ENV="$ROOT/.envs/tvm-mlc-py313"
export TVM_LIBRARY_PATH="$TVM_SRC/build/lib"
export DYLD_LIBRARY_PATH="$TVM_SRC/build/lib:$DYLD_LIBRARY_PATH"
export PYTHONPATH="$TVM_SRC/python:$PYTHONPATH"
"$TVM_ENV/bin/python" scripts/check_tvmjs_webgpu_toolchain.py
```

The probe script now has opt-in support for that shape:

```bash
"$TVM_ENV/bin/python" scripts/lc0_tvm_whole_onnx_probe.py \
  --model public/models/lc0/t1-256x10-distilled-swa-2432500.batch1.f16.onnx \
  --target webgpu \
  --host-target '{"kind":"llvm","mtriple":"wasm32-unknown-unknown-wasm"}' \
  --out artifacts/tvm/t1-256x10-distilled-swa-2432500.batch1.f16.webgpu.tvmjs-wasm-attempt.probe.json \
  --cast-int64-initializers-to-int32 \
  --trust-nonnegative-gather-indices \
  --sanitize-onnx-names \
  --export-tvmjs-wasm
```

Current status: the local TVMJS/WebGPU export toolchain has been made available separately from the original native/source probe build. Rebuild/check it with:

```bash
npm run lc0:build-tvmjs-webgpu-toolchain
npm run lc0:check-tvmjs-webgpu-toolchain
```

Details:

- Emscripten installed via Homebrew: `emcc 5.0.7`
- TVM web runtime bitcode built under `.deps/tvm-webgpu-src/web/dist/wasm/`
- LLVM-enabled TVM build created under `.deps/tvm-webgpu-src/build-tvmjs/`
- `scripts/check_tvmjs_webgpu_toolchain.py` passes when `TVM_LIBRARY_PATH` points at `build-tvmjs/lib`

Use this wrapper command for the browser-loadable TVMJS wasm artifacts:

```bash
cd /Users/macthedan/projects/lc0_browser/leelaweb
CAST_INT64_INITIALIZERS_TO_INT32=1 \
TRUST_NONNEGATIVE_GATHER_INDICES=1 \
SANITIZE_ONNX_NAMES=1 \
EXPORT_TVMJS_WASM=1 \
TVM_BUILD_DIR=build-tvmjs \
TVM_HOST_TARGET='{"kind":"llvm","mtriple":"wasm32-unknown-unknown-wasm"}' \
./scripts/run_lc0_tvm_whole_onnx_probe.sh
```

This writes `.tvmjs-wasm.probe.json` and `.tvmjs-wasm.probe.tvmjs.wasm` artifacts. Native `.tvm_export` is intentionally skipped in this mode because wasm32 host objects cannot be linked by the platform C++ linker.

Current generated whole-model f16 TVMJS/WebGPU wasm artifacts:

| Batch | Artifact | Bytes |
| --- | --- | ---: |
| 1 | `artifacts/tvm/t1-256x10-distilled-swa-2432500.batch1.f16.webgpu.tvmjs-wasm.probe.tvmjs.wasm` | `44,308,011` |
| 4 | `artifacts/tvm/t1-256x10-distilled-swa-2432500.batch4.f16.webgpu.tvmjs-wasm.probe.tvmjs.wasm` | `44,611,496` |
| 8 | `artifacts/tvm/t1-256x10-distilled-swa-2432500.batch8.f16.webgpu.tvmjs-wasm.probe.tvmjs.wasm` | `44,644,785` |

Browser loader smoke now passes for batches 1/4/8 using:

```text
lc0-tvmjs-webgpu-smoke.html
```

The smoke verifies:

- manifest fetch
- SHA-256/size verification for the selected wasm artifact
- WebGPU adapter/device acquisition with `shader-f16`
- async WebAssembly instantiation via `tvmjs.instantiate` (required because sync compile/instantiate is blocked for >8 MB wasm)
- `tvm.initWebGPU(device)`
- `tvm.systemLib()`
- `tvm.asyncLoadWebGPUPipelines(systemLib)`

Observed loader result for b1/b4/b8: `SMOKE_OK`, with `vm_load_executable` present.

The smoke page also has an opt-in `Invoke zero-input VM` checkbox. With it enabled, b1/b4/b8 now perform a real Relax VM invocation:

- create `float16` WebGPU input tensor shaped `[batch,112,8,8]`
- upload zero bytes with `copyFromRawBytes`
- create VM via `tvm.createVirtualMachine(tvm.webgpu())`
- call `set_input("main", input)`
- call `invoke_stateful("main")`
- retrieve three outputs via `get_output("main", i)`
- copy aligned f16 outputs back to CPU after `await out.device.sync()`

Observed zero-input invocation smoke results:

| Batch | `invoke_stateful` smoke timing | Outputs |
| --- | ---: | --- |
| 1 | about `4.0 ms` | `[1,1858]`, `[1,3]`, `[1,1]` f16 |
| 4 | about `4.1–5.5 ms` | `[4,1858]`, `[4,3]`, `[4,1]` f16 |
| 8 | about `5.4–5.9 ms` | `[8,1858]`, `[8,3]`, `[8,1]` f16 |

This is **runtime invocation evidence**, but still not full promotion evidence.

Real-fixture smoke now exists via:

```bash
npm run lc0:tvmjs-webgpu-smoke -- \
  --base-url http://127.0.0.1:5173 \
  --batch 8 \
  --out artifacts/tvm/lc0_tvmjs_webgpu_smoke_batch8.json
```

It opens `lc0-tvmjs-webgpu-smoke.html?batch=8&invoke=1&fixtures=1&autorun=1`, encodes `fixtures/lc0/fen_only.json` with the existing `encodeLc0Classical112(..., {historyFill: "fen_only"})`, converts planes to f16 bytes, invokes the TVMJS VM, decodes f16 policy/WDL/MLH, computes legal priors, and compares against `fixtures/lc0/native_fen_only_blas.jsonl`.

Current batch-8 fixture smoke result against native BLAS fixture rows:

- native best-move match: `8/8`
- max native top-prior abs diff over native top-prior rows: `0.004802150612562506`
- max Q abs diff vs native BLAS fixture rows: `0.004474531249999997`
- max MLH abs diff vs native BLAS fixture rows: `0.2749999999999986`
- observed VM `invoke_stateful` time in this smoke: about `4.2 ms`

Same-page ORT comparison is available with `--ort-compare f16|f32|both --ort-ep webgpu|wasm|webgpu,wasm`.

Current browser ORT comparison artifacts:

```bash
npm run lc0:tvmjs-webgpu-smoke -- \
  --base-url http://127.0.0.1:5173 \
  --batch 8 \
  --ort-compare f16 \
  --ort-ep webgpu \
  --out artifacts/tvm/lc0_tvmjs_webgpu_smoke_batch8_ortf16_webgpu.json

npm run lc0:tvmjs-webgpu-smoke -- \
  --base-url http://127.0.0.1:5173 \
  --batch 1 \
  --ort-compare both \
  --ort-ep webgpu \
  --out artifacts/tvm/lc0_tvmjs_webgpu_smoke_batch1_ortboth_webgpu.json
```

Observed results:

| Artifact | ORT compare | Best move | Max top-prior abs diff | Max Q abs diff | Max MLH abs diff |
| --- | --- | ---: | ---: | ---: | ---: |
| batch8 vs ORT f16 WebGPU | f16 | `8/8` | `0.009859013926522886` | `0.003173828125` | `0.25` |
| batch1 vs ORT f16 WebGPU | f16 | `1/1` | `0.0006974565401345473` | n/a | n/a |
| batch1 vs ORT f32 WebGPU | f32 | `1/1` | `0.0004851487340732652` | n/a | n/a |

For batch1, TVMJS f16 WDL/MLH output byte lengths are not 4-byte aligned, so the current WebGPU readback helper skips those small outputs; policy/top-prior and best-move comparison still runs. Batch8 provides aligned WDL/MLH readback.

Full current `fixtures/lc0/fen_only.json` coverage is captured by:

```text
artifacts/tvm/lc0_tvmjs_webgpu_fixture_coverage_summary.json
```

Aggregate result over all 10 fixtures:

- native best-move match: `10/10`
- ORT f16 WebGPU best-move match: `10/10`
- max native top-prior abs diff: `0.004802150612562506`
- max native Q abs diff: `0.004474531249999997`
- max native MLH abs diff: `0.2749999999999986`
- max ORT f16 WebGPU top-prior abs diff: `0.009859013926522886`
- max ORT f16 WebGPU Q abs diff: `0.003173828125`
- max ORT f16 WebGPU MLH abs diff: `0.25`

Current decision: **TVMJS remains research-only**. It is now a credible evaluator-level runtime candidate, but it is not an opt-in runtime configuration for users until fixed-suite search parity/perf and broader regression coverage exist. The stable/default path remains ORT ONNX/WebGPU.

## Search parity smoke

`lc0-tvmjs-webgpu-smoke.html` now includes a narrow TVMJS-backed `Lc0EvaluationProvider` and can run existing `Lc0PuctSearcher` against both TVMJS and ORT f16 WebGPU in the same browser page. The automated smoke accepts:

```bash
--search-visits N
--search-fixtures N
--search-repeats N
--stockfish-score-depth N
--stockfish-score-ms N
--fens path/to/suite.fen
```

`--fens` reads newline-separated FENs, bypasses `fixtures/lc0/fen_only.json`, passes the selected slice to the page as `fixedSuiteFens`, and keeps the run isolated from stable runtime defaults.

Current smoke/report artifacts:

```text
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

A smoke artifact can be converted into a fixed-suite-style research report with explicit caveats:

```bash
npm run lc0:tvmjs-webgpu-summary -- \
  --in artifacts/tvm/lc0_tvmjs_webgpu_search_smoke_v16_batch8_uho_lite16_repeat2.json \
  --out artifacts/tvm/lc0_tvmjs_webgpu_fixed_suite_uho_lite16_repeat2_report.json
```

The unscored report schema is `lc0_browser.tvmjs_webgpu_fixed_suite_report.v1`. When `--stockfish-score-depth` or `--stockfish-score-ms` is used, the report schema is `lc0_browser.tvmjs_webgpu_fixed_suite_report.v2` and includes per-row post-search Stockfish scoring for the TVMJS and ORT moves. Both schemas deliberately report that they are smoke-derived and not `lc0_browser_runtime_fixed_suite.mjs` output.

The aggregate can be regenerated from local smoke/report artifacts with:

```bash
npm run lc0:tvmjs-webgpu-evidence-summary -- \
  --out artifacts/tvm/lc0_tvmjs_webgpu_search_smoke_summary.json
```

The current local evidence gate is:

```bash
npm run lc0:tvmjs-webgpu-evidence-check
```

It requires all discovered search rows to match TVMJS-vs-ORT f16 moves, at least `94` search rows, at least `3` Stockfish-scored runs, and at least `4` fixed-suite-style report artifacts. These thresholds describe the current local research evidence only; they are not promotion criteria.

Local generated artifact availability can be checked with:

```bash
npm run lc0:tvmjs-webgpu-local-artifacts-check
```

This validates the staged runtime manifest, its model byte/SHA metadata, and the current evidence summary. It does not imply those generated artifacts should be committed.

Current aggregate schema: `lc0_browser.tvmjs_webgpu_search_smoke_summary.v9`.

Aggregate search smoke result:

- rows: `94`
- visits covered: `4`, `16`, `32`
- TVMJS-vs-ORT f16 WebGPU move match: `94/94`
- all-current-fixture 16-visit run: evaluator native match `10/10`, ORT f16 evaluator match `10/10`, search move match `10/10`
- all-current-fixture 32-visit run: evaluator native match `10/10`, ORT f16 evaluator match `10/10`, search move match `10/10`
- arbitrary-FEN UHO-lite 8-position 16-visit run: ORT f16 evaluator match `8/8`, search move match `8/8`
- arbitrary-FEN UHO-lite 16-position 16-visit repeat-2 run: ORT f16 evaluator match `16/16`, search move match `32/32`, TVMJS mean/median search wall time `43.356/40.185 ms`, ORT f16 WebGPU mean/median search wall time `50.739/47.087 ms`
- Stockfish-scored UHO-lite 4-position 16-visit run at Stockfish depth `3`: search move match `4/4`, scored rows `4`, TVMJS-minus-ORT LC0-perspective cp delta count `3`, mean/min/max `0/0/0`
- Stockfish-scored UHO-lite 8-position 16-visit run at Stockfish depth `3`: search move match `8/8`, scored rows `8`, TVMJS-minus-ORT LC0-perspective cp delta count `7`, mean/min/max `0/0/0`
- Stockfish-scored UHO-lite 16-position 16-visit run at Stockfish depth `3`: search move match `16/16`, scored rows `16`, TVMJS-minus-ORT LC0-perspective cp delta count `15`, mean/min/max `0/0/0`
- max observed TVMJS search wall time in these short smokes: `82.760 ms`
- max observed ORT f16 WebGPU search wall time in these short smokes: `93.840 ms`

Covered positions include the current 10 `fixtures/lc0/fen_only.json` rows, including start, black-after-e4, rook moves, and white/black promotion-near fixtures, plus the first 16 rows from `../leelaweb-arena-diagnostics/eval/opening_suite_uho_lite_v1.fen`. The smoke page now chunks fixture evaluation through the TVMJS evaluation provider so `fixtureCount` can exceed the physical model batch size. This is still a **search smoke**, not promotion evidence: it uses small fixed-suite slices, warm browser state, and does not replace the matched fixed-suite throughput + drift/parity protocol.

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

## Local artifact/release policy

Source-of-truth files for this research lane are the scripts, smoke page, and docs. The following are generated local artifacts and remain ignored unless a separate release decision is made:

- `artifacts/tvm/` probe/evidence/report JSON, captured source, and wasm outputs
- `public/runtimes/lc0-tvmjs-webgpu/` staged browser runtime artifacts
- `.deps/tvm-webgpu-src/` and `.envs/tvm-mlc-py313/` durable local toolchain state

For handoff/review, commit the reproducible commands and schemas, not the large wasm/runtime outputs. To recreate local state, rebuild/check the TVMJS toolchain, rerun export/staging, rerun smoke/evidence commands, then run the combined research gate:

```bash
npm run lc0:tvmjs-webgpu-research-gate
```

The combined gate runs the lower-level checks:

```bash
npm run lc0:tvmjs-webgpu-evidence-check
npm run lc0:tvmjs-webgpu-local-artifacts-check
npm run lc0:tvmjs-research-only-check
```

Do not promote or publish the generated TVMJS runtime until a release policy defines hosting, cache headers, artifact checksums, browser feature requirements, and non-smoke fixed-suite/perf acceptance criteria.

The research-only guard checks that TVMJS is not present in the stable runtime registry, stable browser runtime evaluator, arena LC0 backend selector, or `src/**` runtime code:

```bash
npm run lc0:tvmjs-research-only-check
```

Current passing schema: `lc0_browser.tvmjs_research_only_check.v1`.

## Next integration choices

1. **TVMJS/WebGPU runtime path**: build the missing LLVM/Emscripten/web-runtime pieces, produce a `.wasm`, then use `tvmjs.Instance`, `initWebGPU(device)`, `asyncLoadWebGPUPipelines(mod)`, and `systemLib()/getFunction(...)` in a browser harness.
2. **Custom scheduler path**: use captured WGSL plus generated C wrapper metadata to reconstruct dispatch order, buffer bindings, and constants in TypeScript/WebGPU. This is more work but may fit the existing custom hybrid runtime architecture.

Before any promotion/default change, require matched fixed-suite throughput and strict drift/parity checks. ONNX f16 WebGPU remains opt-in/research because ORT WebGPU f16 currently fails strict native-BLAS prior parity.

## Visit-loop performance campaign (2026-06-09)

The per-search bottleneck question ("where do ~50 ms of a 16-visit TVMJS search
go?") was answered systematically; full details, commands, and artifact paths
live in `docs/lc0_tvmjs_research_runbook.md` under "Visit-loop attribution",
"Pipelined evaluateBatchSequence A/B", "Per-kernel GPU attribution", and
"Dlight default-schedule rebuild / pass coalescing". Summary of measured
verdicts at v16/b8 (all legs parity-clean):

| Lever | Result |
| --- | --- |
| JS search overhead | ~`2 ms`/search (3.6%) — non-lever |
| Pipelined submit + shared sync (`batchPipelineDepth=2`) | **+27% slower** — GPU-compute-bound, fragments batch fill |
| Pass-boundary overhead (coalesce 5928 dispatches → 26 passes) | nil |
| Dlight default GPU schedules (`--dlight` probe rebuild, `f16/v2-dlight`) | kernel time −8%, end-to-end nil |
| Per-kernel attribution | 229 passes, `9.2 ms` kernel time per b8 invoke, no dominant kernel (top 4 fused matmul families ≈63%) |

Remaining levers, in order: real per-shape schedule tuning of the fused matmul
families, larger physical batches (b16+) for visit budgets that fill them, and
search-side evaluation-count reduction (cache/tree reuse).
