# LC0 TVMJS campaign summary (2026-06-09) and BT4 plan

Everything the LC0 TVMJS/WebGPU lane established this cycle, written as the
launchpad for the next target: **BT4-1024x15x32h** (the largest LC0 net we
hold). Detailed sections, commands, and artifact paths live in
`docs/lc0_tvmjs_research_runbook.md` and
`docs/lc0_tvm_whole_onnx_webgpu_probe.md`; this doc is the map.

Scope reminder: M-chip Mac WebGPU/WASM only for now; other GPUs/browsers
deliberately deferred.

## What the campaign established (t1-256x10-distilled, f16)

### Performance — measured, every leg parity-gated

| Question | Verdict |
| --- | --- |
| Visit-loop breakdown (v16/b8) | 86% of search wall is per-batch GPU wait; JS search ≈ 2 ms (non-lever) |
| Is the GPU wait sync overhead? | No — ~9 ms real kernel arithmetic per b8 invoke (229 passes, no dominant kernel; top 4 matmul families ≈63%) |
| Pipelined submit (`batchPipelineDepth=2`) | **+27% slower** — fragments batch fill; GPU-compute-bound |
| Pass-boundary overhead (coalesce 5928→26 passes) | nil |
| Dlight default schedules | kernel −8%, end-to-end nil; attribution verified (Matmul fires on all 19 matmul kernels) |
| Dlight Matmul tile sweep (8 configs) | default is a local optimum; vec>1 fails TIR legality on N=3/1858 heads; **tuning parked as low-ROI on this device** |
| **Batch scaling (the lever that worked)** | per-batch cost sublinear (15.1/19.4/30.3 ms for b8/b16/b32) → **b16 cuts search wall 22–26%** at v16/v32; b32 wins only from ~v64 |

Bottom line: TVMJS at b16/v16 ≈ **39–40 ms/search vs ORT WebGPU f16 ≈ 45–51 ms**,
16/16 move match on arbitrary UHO positions with Stockfish cp delta uniformly 0.

### Distribution — solved

- **Detached params** (`--detach-params`): model wasm 44.6 → 4.5 MB + a 38 MB
  raw-f16 tensor-cache sidecar (`param_0..param_N`). Parity bit-identical,
  perf unchanged. Key trick: initializers ≤4096 elements stay inline constants
  (shape-feeding tensors need constant values) — `keep_params_in_input` alone
  breaks the import.
- **Cross-batch sharing proven**: b1 and b16 builds dump byte-identical
  tensor-caches; one sidecar serves both wasms (staged `f16/v3-detached`,
  55 MB total vs ~89 MB embedded). Browser loads via `tvm.fetchTensorCache` +
  `tensorCacheGet('param_i')` spread into `set_input`.
- **Footprint**: b1+b16+runtime+sidecar = 54.7 MB raw / **36.8 MB Brotli**.
- **Repeat-load**: shards persist in Cache Storage; warm read-back of 40.2 MB
  = **21 ms** vs 235 ms cold fetch+upload.

### Evidence/infra built this cycle (reusable for BT4)

- Per-search visit-loop attribution + `tvmSearchStats` in the smoke page;
  GPU kernel profiler (timestamp-query, `--kernel-profile-invokes N`);
  pass coalescer (research knob); pipelined `evaluateBatchSequence`
  (`--search-pipeline-depth`, keep 1).
- Probe: `--detach-params`, `--dlight(-matmul-config)`, `--max-fuse-depth`,
  `--fix-batch-dim`, JSON `--dtype`, gather patches, rule-crash fall-through.
- Driver: `--manifest`, `--batch 1/4/8/16/32`; stage script:
  `--batches=`, `--tensor-cache-dir=`, `--params=detached`.
- Fixed-suite bridge accepts `--batch 16/32` + `--manifest`; evidence
  summarizer + research gate all green.
- Mandatory discipline (reinforced by the Tiny lane bisection, commit
  `e1ae54c`): **per-build parity gating against native fixtures for every TVM
  artifact** — the TVM toolchain has demonstrated compiler-level correctness
  bugs and build nondeterminism on other model shapes.

### Promotion state

All local checklist items pass (parity, perf, artifact layout, footprint,
repeat-load). Remaining: non-Apple GPU datapoint (deferred by decision) and a
written hosting/cache-header release policy. TVMJS remains research-only
behind `lc0:tvmjs-research-only-check`.

## Next target: BT4-1024x15x32h-swa-6147500

Why: top-tier playing strength in the browser is the headline product win,
and the entire t1 toolchain transfers. BT4 ≈ 40× the compute of t1-256x10
per eval (1024 d_model × 15 layers × 32 heads vs 256×10).

### Assets already on disk

- `../models/lc0-bestnets/BT4-1024x15x32h-swa-6147500.pb.gz` (336 MB, + sha).
- `../models/lc0-bestnets/onnx/BT4-1024x15x32h-swa-6147500.batch1.f16.onnx`
  (370 MB, + sha) — exposed via `scripts/lc0_prepare_model_assets.mjs`.
- Native lc0 0.32 binary + a BT4 smoke baseline
  (`native/lc0-smoke-startpos-bt4-1024-nodes1.txt`).
- An ORT-path BT4 engine already exists (`src/lc0/bt4Engine.ts`,
  WebGPU-gated, lazy-loaded) — useful as the ORT comparison side.

### Plan, in order

1. **Fixed-batch exports**: `lc0 leela2onnx --onnx-batch-size={1,4,8}
   --onnx-data-type=f16` from the BT4 .pb.gz (+ sha sidecars, add to
   `lc0_prepare_model_assets.mjs`). Start small on batch: per-batch GPU cost
   will be ~40× t1's (extrapolate ≈ 25–60 ms/invoke at b4–b8 untuned);
   the b16-style sweet-spot sweep must be redone — expect it LOWER (b4/b8)
   because each batch is so much heavier.
2. **Detached params is mandatory, not optional**: a 370 MB embedded wasm
   likely exceeds practical instantiate limits and is undeployable anyway.
   `--detach-params` + tensor-cache (32 MB shards; `fetchTensorCache` is the
   same machinery WebLLM uses for multi-GB models). Expect ~350 MB sidecar +
   small per-batch wasms. Check `dump_tensor_cache` peak memory during export.
3. **Compile probe**: same mitigations as t1 (`CAST_INT64... TRUST_NONNEGATIVE...
   SANITIZE... EXPORT_TVMJS_WASM=1 TVM_BUILD_DIR=build-tvmjs`), with
   `LC0_TVMJS_MODEL_FAMILY=BT4-1024x15x32h-swa-6147500 LC0_TVMJS_BATCHES=...`.
   BT4 is an attention net from the same exporter, so the op surface should
   match t1; watch for smolgen-variant ops and `--max-fuse-depth` needs.
4. **Native fixture baseline for BT4**: regenerate the equivalents of
   `fixtures/lc0/native_fen_only_blas.jsonl` with the native lc0 + BT4 net
   (the existing generation scripts are t1-pointed; parametrize the model).
   Without this, there is no parity gate — do not skip.
5. **De-hardcode the smoke page/driver model family** (3 t1 references in
   `lc0-tvmjs-webgpu-smoke.html`: default manifest + the ORT comparison model
   path template + fixture baseline path). Make family/manifest/ORT-model
   page params; the driver already passes `--manifest`.
6. **Memory/startup telemetry first runs**: GPU buffer allocation probe and
   startup timings already report; BT4 will stress pipeline prebuild
   (WGSL sizes scale with d_model) and tensor-cache upload (~350 MB → expect
   ~2 s upload at the measured ~170 MB/s; artifact-cached repeats much less).
   Cold-start UX likely needs the progressive story (play on t1, swap to BT4).
7. **Search budget reality check**: at ~30–60 ms/invoke, 16 visits ≈ 0.2–0.5 s
   per move at b4–b8 — fine for analysis/casual play. Run the v16 bridge with
   Stockfish scoring vs the ORT BT4 path for the first quality/perf evidence.
8. Footprint sidecar + repeat-load measurement for the BT4 staging (same
   commands, new manifest), then evidence-summary refresh + research gate.

### Risks specific to BT4

- WebGPU limits: 1024-wide matmuls per-kernel buffer sizes are fine, but
  `maxStorageBufferBindingSize`/`maxBufferSize` should be requested from the
  adapter (the Tiny lane's `requiredLimits` pattern; the LC0 page currently
  requests only `shader-f16`).
- Single-queue GPU contention: a BT4 invoke occupies the GPU for tens of ms —
  UI jank if the page renders on the same device; consider worker isolation
  (the hybrid lane's worker patterns apply).
- Export time: leela2onnx on 336 MB net + 3 batch variants + tensor-cache dump
  — expect minutes, not seconds; disk: ~1.5 GB of artifacts per full set.
- Keep `lc0:tvmjs-research-only-check` green throughout; BT4 staging goes
  under the same gitignored `public/runtimes/lc0-tvmjs-webgpu/<family>/...`
  layout with manifest sha pinning.
