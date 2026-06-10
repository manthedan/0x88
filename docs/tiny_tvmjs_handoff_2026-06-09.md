# Handoff: Tiny Leela TVMJS follow-ups (2026-06-09)

> **RESOLVED 2026-06-10 — the lane is unblocked.** Bugs A and B were ONE bug,
> found and fixed in `3rdparty/tvm-ffi` (commit `afea100` in the tvm-ffi
> checkout, bumped by `94398e51f` in `.deps/tvm-webgpu-src`):
> `ffi::StructuralEqual::operator()` passed `skip_tensor_content=true` while
> `ffi::StructuralHash::operator()` hashes tensor content — an inconsistent
> (hash, equal) functor pair, used by structural containers and direct
> equality checks throughout the relax build path. Same-shape/dtype tensors
> with different data compare "equal", so constant dedup somewhere in that
> path merged distinct scalar constants: the Tiny rank clip executed
> `min(max(x, 7), 7)` (const 0 resolved to a const-7 tensor) → constant 7s;
> the earlier "reads column 15" reading was an artifact of periodic test
> data. The merge only appears in large-enough modules (context-dependence)
> and varies with process container layout (Bug B's "byte-identical modules,
> varying outputs", incl. NaN when float consts collided). Note: a minimal
> synthetic (512 distinct scalars through the VM const pool) does NOT
> trigger on unpatched builds — libc++ maps pre-compare stored hashes before
> key_eq, masking the broken equality there; the exact interior equality
> path that bites in the 45-node cut was not pinpointed, but the revert
> cycle proves causality end-to-end. The bisection method that localized it: companion-subgraph
> binary search via `onnx.utils.extract_model` against a numpy oracle, then
> instrumenting the surviving suspect with extra ONNX outputs (raw gather was
> CORRECT, clip of those values was wrong → wrong const operand), then direct
> kernel calls (compiled kernels correct) → VM bytecode constant args → pool
> dedup code.
> Validation: 45-node cut now bit-exact vs ORT (`maxdiff 0.0`); full
> `bt4_anneal_muon_best.batch16.sim.onnx` matches ORT CPU at ≤4e-5 across
> policy/wdl/q/hidden and is stable across build×2/run×2; rebuilt
> WebGPU wasm matches ORT in-browser at ≤4.6e-5 all outputs
> (`tiny-tvmjs-webgpu-smoke.html`, SMOKE_OK, ~39 ms/b16 invoke incl.
> readback). BT4-it332 rebuilt with the fixed compiler shows identical
> healthy parity (no regression). Caveat: the probe's dlight/no-fuse builds
> must use `relax_pipeline="default"` (contains the VM-lowering passes and no
> FuseOps); `relax_pipeline=None` fails VMCodeGen on raw `call_tir`.
> Items below kept for historical context; Item 1 is now an upstream PR
> opportunity (tvm-ffi fix + repro) rather than a bug report.

Status snapshot for a fresh session. Two independent work items, both unblocked
and well-scoped. Full investigation narrative: "Tiny Leela (squareformer_v2)
TVMJS probe" + "Numerics bisection result" sections in
`docs/lc0_tvm_whole_onnx_webgpu_probe.md`. Branch context: all work referenced
here is on `profile/tvmjs-visit-loop` through commit `e1ae54c`.

## Where the Tiny TVMJS lane stands

- The shipped squareformer_v2 (`public/models/bt4_anneal_muon_best.onnx`)
  compiles to a 19.9 MB TVMJS/WebGPU wasm and invokes b16 in ~29.6 ms
  (incl. readback of all four outputs) in `tiny-tvmjs-webgpu-smoke.html`.
- Required recipe (all already implemented):
  1. onnxsim prefold with `overwrite_input_shapes` (fixed batch 16) — raw
     export fails Relax import on Shape/Concat dim arithmetic.
  2. Probe flags: `--dtype '{"tokens":"int32"}' --cast-int64-initializers-to-int32
     --trust-nonnegative-gather-indices --trust-runtime-gather-indices
     --sanitize-onnx-names --max-fuse-depth 6 --dlight --export-tvmjs-wasm`.
  3. Browser must request adapter `maxStorageBuffersPerShaderStage` in
     `requiredLimits` (default 8 < the 12 a fused embedding-sum kernel wants;
     `--max-fuse-depth 6` keeps fused kernels under the device's 10).
- **Numerics are wrong and the lane is blocked on upstream TVM bugs** (below).
  No parity, no integration. Do not promote anything from this lane.

## Item 1: file the TVM issue(s)

Two distinct upstream bugs, both characterized with evidence:

**Bug A — compiler pass mis-folds constant scalar Gather-index chains.**
- Symptom: the rank-column gather `clamp(tokens[:,:,12], 0, 7)` executes as
  `clamp(tokens[:,:,15], 0, 7)` (verified by matching TVM output against
  every candidate column).
- The imported Relax IR is verified CORRECT (scalar wrap chains carry
  `R.const 8/9/10/11/12` per the ONNX). The corruption happens in
  LegalizeOps/FoldConstant/build of the `shape_to_tensor`/`take`/`where`
  scalar-index chains.
- Reproduces with FuseOps on, with a forced fusion boundary, and with FuseOps
  disabled — fusion-independent, deterministic in the 45-node cut.
- Repro artifacts: `artifacts/tvm-tiny/cut_add4_boundary.onnx` (45 nodes, two
  outputs `/Add_4_output_0` + `/Clip_2_output_0`) and
  `artifacts/tvm-tiny/repro_inputs.npz`. Tiny cuts of the same ops in
  isolation are exact — keep the full 45-node context in the issue.
- Repro script shape (llvm target, no WebGPU needed):
  `from_onnx(model) → relax.build(llvm) → VirtualMachine → compare vs
  onnxruntime CPUExecutionProvider on repro_inputs`.
- TVM checkout: `.deps/tvm-webgpu-src` @ `15b1d9839` ("[Web] Bump tvmjs version
  to 0.25.0-dev1 (#19687)").

**Bug B — nondeterministic outputs from byte-identical modules.**
- `from_onnx` is deterministic (identical `mod.script()` sha across runs), but
  building+running the same module yields run-to-run-varying outputs: one
  process produced four different outputs from one build; others alternate
  clean/all-NaN per build. Uninitialized-memory-class.
- WebGPU's zero-initialized buffers mask this as stable-wrong in browsers
  (browser outputs match zero-heap CPU runs). `PYTHONHASHSEED` is NOT a
  reliable pin (NaN observed under seed 0).
- Repro: full `artifacts/tvm-tiny/bt4_anneal_muon_best.batch16.sim.onnx`,
  build×2/run×2 across several processes; `artifacts/tvm-tiny/stem_only.onnx`
  (62 nodes) is stable and may not show this — the variance needs the larger
  program.
- Bug A and B may share a root cause (a pass reading unstable container
  order / uninitialized state); report together with cross-reference.

Also worth including: the two WebGPU-specific gaps found en route (not
correctness bugs, but real): relax FuseOps ignores target `max_function_args`
(`src/relax/transform/fuse_ops.cc:1053` hardcodes 0; webgpu target kind never
declares the attr, unlike Metal's 31), and unfused bool intermediates lower to
`array<i8>` WGSL storage buffers, which WGSL rejects.

## Item 2: export-wrapper cleanups for the next training/export cycle

File: `training/export_squareformer_v2_av_onnx.py`. None of these change what
the network learns; all shrink the deployment graph surface (helps ORT WebGPU
and any future runtime, not just TVM):

1. **Fixed-batch deploy exports** (like leela2onnx): kills the onnxsim
   prefold requirement and all Shape-op dim arithmetic at the source.
2. **Cast `tokens` to int32 at model entry** (`x = x.int()` in the wrapper):
   removes the i64 input and the per-input dtype override.
3. **Fold square-static embeddings**: `rank_emb + file_emb + square_emb (+ pos)`
   are all functions of the square index alone — precompute one constant
   `[64, d]` table. Removes gathers/adds from the 21-term stem chain (the
   chain that hits WebGPU's storage-buffer limit when fused).
4. **History sum as matmul**: the `[b,64,8,d] → ReduceSum → [b,64,d]` ply sum
   crashes both TVM schedulers (rfactor bind). Express as matmul-with-ones or
   a learned `[8]` weighting — strictly more expressive, compiles everywhere.
5. **Drop or int-domain the in-graph `clamp()` calls** for deploy exports —
   the JS feature encoder constructs the tokens and already guarantees ranges;
   the clamps generate the isnan/where/bool-i8 patterns.
6. **Deploy-export output trim**: `policy + wdl` only (q is derivable; hidden
   and action_values are diagnostics) — halves GPU readback traffic.
7. **Debug-export flag** that appends intermediate tensors (stem end, each
   layer residual) to `output_names` — turns any future numerics bisection
   into a 10-minute diff. The `onnx.utils.extract_model` single-output-cut
   method worked well; note that adding many probe outputs changes fusion and
   can shift compiler bugs around.

## Gotchas the next session should know

- Run TVM toolchain via the documented env (`.deps/tvm-webgpu-src`,
  `.envs/tvm-mlc-py313`, `TVM_BUILD_DIR=build-tvmjs`); `.venv-onnx` has
  onnxruntime/onnxsim; the TVM env does NOT have onnxruntime (compare in two
  steps via /tmp npz files).
- The dlight rule-crash fall-through in the probe sends crashed functions to
  `Fallback` (commit `e1ae54c` lineage) — keep that; sibling reduction rules
  share assumptions with the crashing one.
- `agent-browser` + a vite on a dedicated port is the browser-verification
  loop; `tiny-tvmjs-webgpu-smoke.html` staging lives under gitignored
  `public/runtimes/tiny-tvmjs-webgpu/` (copy wasm + bundle + runtime wasm).
