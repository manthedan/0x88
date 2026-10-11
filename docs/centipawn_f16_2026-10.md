# Centipawn mixed-f16 — October 2026

## Decision: reject at the numerical gate; stop

One physical **batch-16** mixed-f16 TVMJS module was built and executed against the promoted **f32/v2-shape-k16** module. It compiles, loads, and produces finite outputs, but **fails the existing absolute-error tolerance of 0.001**. Even the product-facing WDL probabilities fail: maximum drift **0.002876815** on `en-passant-available`. Legal-policy top-1 agreement is **10/10**, not sufficient to override this failure.

No performance benchmark, schedule iteration, runtime dtype relaxation, shader-f16 fallback implementation, default selection change, or release/publication follows this failed gate. **Production remains f32 v2.** This is a numerical rejection, not a finding that f16 performance is flat or slower.

Committed evidence: [`centipawn_f16_2026-10.evidence.json`](centipawn_f16_2026-10.evidence.json). The wasm, ONNX, generated sources, diagnostic harness, browser capture, and screenshot remain **local and ignored**, under `artifacts/centipawn-f16/`. Only documentation/evidence is committed.

## Build identity and precision contract

Existing pipeline: [`lc0_tvm_whole_onnx_probe.py`](../scripts/lc0_tvm_whole_onnx_probe.py), [`build_centipawn_tvmjs_subgroups.sh`](../scripts/build_centipawn_tvmjs_subgroups.sh), [`lc0_tvmjs_research_runbook.md`](lc0_tvmjs_research_runbook.md), and the promoted [v2 manifest](../public/runtimes/centipawn-tvmjs-webgpu/bt4-soap-rem-c19000-final/f32/v2-shape-k16/manifest.json). The Lc0 f16 exports establish toolchain feasibility, not Centipawn numerical equivalence.

- Read-only compiler checkout: `94398e51fbb8bf9022da85d83e25f914ef033110`, `build-tvmjs/lib`, Emscripten **5.0.7-git**.
- V2 storage-rewrite patch verified byte-for-byte: SHA-256 `33b73d6cfa2a198a792e8642332c51ba236a8547c1e8c854a6d8301c7b31098f`.
- tvm-ffi submodule fix **`afea100`**: [`PR_tvm_ffi_structural_equal.md`](upstream/PR_tvm_ffi_structural_equal.md). The existing `docs/upstream/repro_constant_pool_dedup.py` passes with this loaded compiler/library environment before export. No shared compiler, library, or Python environment was modified.
- Original ONNX: **4,464,375 bytes**, SHA-256 `de2dd68518f6e2ed162da5ee27c67a5a4795567d6e39972e3705824572b7660e`.
- Fixed-b16 f32 ONNX: **16,246,184 bytes**, SHA-256 `c4adcf876904baf3381727720391ffba2fa2981e0395334e81af9424815b0506`; onnx **1.21.0**, onnxsim **v0.6.4**, **342 nodes / 139 initializers**, exactly the v2 graph.

The sole intended arithmetic change is TVM Relax `ConvertToDataflow → ToMixedPrecision(out_dtype='float32') → FoldConstant` immediately after ONNX import. Matmul weights/operands and safe following activations become f16; accumulation remains f32. **LayerNorm and Softmax are explicitly registered as `kNever` (policy 2)** for mixed precision, because this checkout's LayerNorm defaults to `kFollow`. This changes only in-process op attributes. Integer tokens stay int32; attack-summary input and all four returned tensors retain the f32 ABI. The output ABI is asserted before compilation and again in the browser.

Captured Relax IR shows f32 LayerNorm, Softmax, sum/mean, and matmul `out_dtype`. Captured WGSL has **51 kernels**, **32 enabling f16**, no subgroup instructions, and f32 matmul/GEMV accumulators and LayerNorm/Softmax reduction buffers. Normalization/softmax results may then be cast to f16 for the next matmul; the reduction arithmetic itself is f32. Inspection details/hashes are retained in the evidence JSON and local `wgsl-inspection.json`.

**V2's schedule recipe is preserved**, not retuned: subgroup-disabled WebGPU target, wasm32 LLVM host, max-fuse-depth 6, dlight's existing rule order, trusted gather indices, initializer i64→i32, and name sanitization. Shape selector remains `static_large / min_macs=8000000 / j_multiple=32 / k_multiple=16`, with `micro_size_k=16`. All recorded Matmul.Config fields equal v2: block 8×8, vthread 1×1, micro 4×4×16, vector_size 1, unroll 256, shared storage true, storage_align/inner_x false. Precision casts naturally change automatic fusion/kernel identities (v2 has 46 kernels); there was no manual policy-head fusion, tile change, new batch, parameter detachment, or subgroup rebuild.

| Model wasm | Raw bytes | SHA-256 |
|---|---:|---|
| Promoted f32 v2 | 20,192,787 | `fd3d8635c8368338c3a0d1b57a725a70b4f8a74f8d130ee831848c641cba75b5` |
| One mixed-f16 candidate | 12,095,744 | `f7054858eec28f6b3131fbbcea53c265c814426a64344a48156959a63e899bb7` |

These are file sizes, **not transfer/startup/performance measurements**. Brotli compression was not run after the numerical failure. The unchanged v2 TVMJS JS bundle and runtime wasm hashes are `e5225b46a98cc6c5ab133233a67f830e856ac0a23b01022768e7f94ed58a8111` and `f5d0b00f63ee878a2144f576ab41825a7c3d13a79621782dcc98b14f4d1c58e5`.

## Exact build commands

All commands executed from the assigned worktree on `perf/centipawn-f16`:

```sh
cd /Users/macthedan/projects/lc0_browser/leelaweb/.claude/worktrees/centipawn-f16
bash artifacts/centipawn-f16/build.sh > artifacts/centipawn-f16/build.log 2>&1
```

That local script (hash recorded in evidence) uses these environment values:

```sh
export TVM_SRC=/Users/macthedan/projects/lc0_browser/.deps/tvm-webgpu-src
export TVM_ENV=/Users/macthedan/projects/lc0_browser/.envs/tvm-mlc-py313
export ONNX_PYTHON=/Users/macthedan/projects/lc0_browser/leelaweb/.venv-onnx/bin/python
export PYTHONDONTWRITEBYTECODE=1
export PATH="/opt/homebrew/bin:/opt/homebrew/opt/llvm/bin:$TVM_ENV/bin:$PATH"
export TVM_LIBRARY_PATH="$TVM_SRC/build-tvmjs/lib"
export DYLD_LIBRARY_PATH="$TVM_LIBRARY_PATH:/opt/homebrew/opt/llvm/lib:${DYLD_LIBRARY_PATH:-}"
export PYTHONPATH="$TVM_SRC/python"
```

It verifies the compiler commit/patch, reconstructs the exact fixed graph as in the existing subgroup build script using:

```python
model, valid = onnxsim.simplify(
    onnx.load('public/models/bt4_soap_rem_c19000_final.onnx'),
    overwrite_input_shapes={'tokens': [16, 64, 24], 'attack_summary': [16, 64, 28]},
)
assert valid and len(model.graph.node) == 342 and len(model.graph.initializer) == 139
onnx.save(model, 'artifacts/centipawn-f16/bt4-soap-rem-c19000-final.batch16.f32.onnx')
# Assert the original/fixed hashes above before proceeding.
```

The local `artifacts/centipawn-f16/build_mixed.py` wrapper contains the following precision injection (the existing probe still owns import mutations, scheduling, diagnostics, and export):

```python
from pathlib import Path
import sys
import tvm
from tvm import relax
import tvm.relax.frontend.onnx as frontend

sys.path.insert(0, str(Path('scripts').resolve()))
import lc0_tvm_whole_onnx_probe as probe

for name in ('relax.nn.layer_norm', 'relax.nn.softmax'):
    tvm.ir.register_op_attr(name, 'TMixedPrecisionPolicy', 2, level=100)
original = frontend.from_onnx

def mixed_import(*args, **kwargs):
    mod = original(*args, **kwargs)
    mod = relax.transform.ConvertToDataflow()(mod)
    mod = relax.transform.ToMixedPrecision(out_dtype='float32')(mod)
    mod = relax.transform.FoldConstant()(mod)
    Path('artifacts/centipawn-f16/mixed.relax.py').write_text(mod.script())
    main = mod['main']
    assert [str(p.struct_info.dtype) for p in main.params] == ['int32', 'float32']
    assert all(str(s.dtype) == 'float32' for s in main.ret_struct_info.fields)
    return mod

frontend.from_onnx = mixed_import
raise SystemExit(probe.main())
```

The actual single export command:

```sh
"$TVM_ENV/bin/python" docs/upstream/repro_constant_pool_dedup.py
CONFIG='{"selector":{"kind":"static_large","min_macs":8000000,"j_multiple":32,"k_multiple":16},"config":{"micro_size_k":16}}'
"$TVM_ENV/bin/python" artifacts/centipawn-f16/build_mixed.py \
  --model artifacts/centipawn-f16/bt4-soap-rem-c19000-final.batch16.f32.onnx \
  --out artifacts/centipawn-f16/bt4-soap-rem-c19000-final.batch16.f16.webgpu.tvmjs-wasm.probe.json \
  --target '{"kind":"webgpu","supports_subgroups":false}' \
  --host-target '{"kind":"llvm","mtriple":"wasm32-unknown-unknown-wasm"}' \
  --dtype '{"tokens":"int32"}' --cast-int64-initializers-to-int32 \
  --trust-nonnegative-gather-indices --trust-runtime-gather-indices \
  --sanitize-onnx-names --max-fuse-depth 6 --dlight \
  --dlight-matmul-config "$CONFIG" --export-tvmjs-wasm --capture-module-sources
```

## Built-app numerical gate

Apple M4 MacBook Air, 24 GB, macOS Darwin 25.4.0; Chromium **149.0.0.0**, WebGPU adapter **apple / metal-3**, shader-f16 available. The browser requests shader-f16 only for the candidate, no subgroups or timestamp-query in either leg. All GPU execution was inside the supplied machine-wide **benchlock**, on the COOP/COEP built-app server **port 5184**. Builds/tests did not run while this worktree held the mutex. No latency inference is made from this single numerical run or machine load.

The production evaluator correctly rejects f16 today, so **no production validation was bypassed or edited**. Local `artifacts/centipawn-f16/parity.mjs` directly runs each TVM VM with the same production encoders and legal-policy/WDL transforms. It verifies each model's bytes/SHA, requests the manifest's features, checks all output shapes/dtypes and finite values, and destroys both runtimes/devices in `finally`. This is a built diagnostic overlay, not an f16 Play integration.

The current gate is implemented in [`lab/centipawn-tvmjs-webgpu-bench.html`](../lab/centipawn-tvmjs-webgpu-bench.html): all four raw outputs finite and **max absolute drift ≤ 0.001**. [`lab/centipawn-tvmjs-subgroups.mjs`](../lab/centipawn-tvmjs-subgroups.mjs) independently uses the same tolerance on legal policy and WDL probabilities for `fixtures/lc0/fen_only.json`. Both tolerances were preserved, applied **candidate vs f32 v2**, not loosened for f16.

Inputs: the existing seeded synthetic b16 generator (seed 7), then all **10 FEN-only fixtures**, stm-white normalized with empty history, using `squareformerCompactInput` and `threatgraphSquareSummaryV1`. Padding to physical b16 repeats the last encoded row, as production does. Raw fixture metrics include six padded duplicates; probability aggregates/top-1 include **only the 10 unique fixtures**. Per-fixture UCI strings in evidence are on the normalized board; top-1 equality is unchanged by inverse normalization.

| Output/comparison | Max absolute delta | Mean absolute delta | Gate |
|---|---:|---:|---|
| Synthetic raw policy (327,680) | 0.054327011 | 0.010789115 | FAIL |
| Synthetic raw WDL (48) | 0.011913300 | 0.004017055 | FAIL |
| Synthetic raw q (16) | 0.002429485 | 0.000741571 | FAIL |
| Synthetic raw hidden (131,072) | 0.021570921 | 0.003147563 | FAIL |
| Fixture raw policy (327,680) | 0.080959320 | 0.010181610 | FAIL |
| Fixture raw WDL (48) | 0.017322540 | 0.007331098 | FAIL |
| Fixture raw q (16) | 0.004000366 | 0.000928522 | FAIL |
| Fixture raw hidden (131,072) | 0.032911658 | 0.003391294 | FAIL |
| Unique-fixture legal policy (174 probabilities) | 0.000957503 | 0.000077243 | PASS |
| Unique-fixture WDL (30 probabilities) | **0.002876815** | 0.000240597 | **FAIL** |

Legal-policy top-1: **10/10 (100%)**. Unfiltered full-policy argmax: **16/16** for synthetic and fixture batches (the latter includes padding). All compared values are finite; **zero uncaptured GPU errors**. Neither policy agreement nor smaller embedded weights rescues the failed WDL/raw-output gate. No ORT third leg or search equivalence run was needed to establish this direct f32-v2 rejection.

Build/staging/browser commands:

```sh
BUILD_SCOPE=product npm run build:client
node scripts/stage_lc0_tvmjs_webgpu_artifacts.mjs \
  --model-family=bt4-soap-rem-c19000-final --dtype=f16 --version=local-one-b16 --batches=16 \
  --artifacts=artifacts/centipawn-f16 \
  --tvm-src=/Users/macthedan/projects/lc0_browser/.deps/tvm-webgpu-src \
  --tvmjs-bundle=public/runtimes/centipawn-tvmjs-webgpu/bt4-soap-rem-c19000-final/f32/v2-shape-k16/tvmjs.bundle.js \
  --tvmjs-runtime-wasm=public/runtimes/centipawn-tvmjs-webgpu/bt4-soap-rem-c19000-final/f32/v2-shape-k16/tvmjs_runtime.wasm \
  --out=dist-client/runtimes/centipawn-tvmjs-webgpu/bt4-soap-rem-c19000-final/f16/local-one-b16
```

The local HTML/MJS was Vite production-compiled with `configFile:false`, the existing `onnxruntime-web-use-extern-wasm/module/browser/production` conditions, input `artifacts/centipawn-f16/parity.html`, output `artifacts/centipawn-f16/lab-dist`. Compiled `artifacts/` and `assets/` were overlaid into `dist-client`, plus the fixture JSON. Under benchlock, the bounded local coordinator `artifacts/centipawn-f16/locked_browser.sh` served `PORT=5184 npm run web:isolated:static`. Native agent-browser session `centipawn-f16-5184`: open `http://localhost:5184/artifacts/centipawn-f16/parity.html`, snapshot, click **Run numerical gate**, wait for `window.f16Result`, extract the full result, screenshot, close. The coordinator then received its release marker and stopped its recorded server PID. Browser sessions and port 5184 server are closed.

Local screenshot, saved/verified and read: **`artifacts/centipawn-f16/parity-failed.png`**. Full result: **`artifacts/centipawn-f16/parity.json`**. Source/compiled harness, result, shader inspection, and artifact hashes are recorded in the committed evidence JSON. Candidate staged only in ignored `dist-client`; no new file under `public/`, release manifests, or channel aliases. Existing Centipawn public wasm uses Git LFS (`.gitattributes`), but this rejected artifact was **not added to LFS/git or published**.

## Stopped gates and verification

Per the numerical-first stop rule, **not run**: Brotli size, ABAB session/init timing, kernel timestamp timing, invoke+readback timing, Play-budget search wall, or built-app f16 Play smoke. There are no performance medians/spreads to report. The subgroup study's noisy/reversing search-wall distributions ([October rebuild](centipawn_tvmjs_subgroups_2026-10.md)) remain the search promotion kill criterion, but were never reached here. Do not reinterpret this result as a measured f16 search regression or reopen a tile/precision sweep.

- `BUILD_SCOPE=product npm run build:client`: PASS; local diagnostic Vite production build: PASS.
- `npm test`: **1005 total, 1002 passed, 3 existing skips, 0 failures**; included `npm run typecheck`: **451 files, 0 errors / 0 warnings**.
- Focused `node --experimental-strip-types --test tests/centipawn_tvmjs_evaluator.test.mjs tests/centipawn_release.test.mjs`: **14/14 passed**.
- No application logic was changed, so no new behavioral unit test is claimed. The retained numerical failure is the experiment's verification, not a green parity assertion.

Structured branch review is recorded in the task closeout; it is a documentation review, not a performance or numerical promotion gate.
