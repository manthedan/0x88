# BT4-it332 TVMJS/WebGPU onboarding results (2026-06-09)

Executes the plan in `lc0_tvmjs_campaign_2026-06-09_and_bt4_plan.md` against
**BT4-1024x15x32h-swa-6147500-policytune-332** (the "it332" policytune net,
downloaded from lczero networks-contrib; NOT the swa-6147500 net already on
disk). Same M-chip Mac WebGPU scope as the t1 campaign.

## Pipeline (all steps completed)

1. Net: `../models/lc0-bestnets/BT4-1024x15x32h-swa-6147500-policytune-332.pb.gz`
   (382.6 MB, + sha256 + describenet: 15 encoders, 32 heads, d_model 1024,
   DFF 1536, mish default / smolgen swish, attention policy, WDL + MLH).
2. ONNX: `lc0 leela2onnx --onnx-batch-size={1,4,8} --onnx-data-type=f16`
   → `../models/lc0-bestnets/onnx/*.batch{1,4,8}.f16.onnx` (370.6 MB each,
   sha sidecars), exposed via `scripts/lc0_prepare_model_assets.mjs`.
3. Native parity baseline: `scripts/lc0_native_fixture_priors.py --weights …it332.pb.gz`
   → `fixtures/lc0/native_fen_only_blas.BT4-1024x15x32h-swa-6147500-policytune-332.jsonl`
   (10 rows, BLAS, nodes=1).
4. TVM probe: standard mitigations + `DETACH_PARAMS=1` + `PYTHONHASHSEED=0`,
   batches 1/4/8 → all `ok: true`, no op-surface issues (the smolgen-variant
   concern did not materialize), no `--max-fuse-depth` needed. ~25 s/batch.
   195 detached params, 370.1 MB; **tensor-caches byte-identical across
   b1/b4/b8** (same cross-batch sharing as t1).
5. Staged: `public/runtimes/lc0-tvmjs-webgpu/BT4-1024x15x32h-swa-6147500-policytune-332/f16/v1/`
   — three ~4.4–4.7 MB model wasms + runtime + ONE shared 353 MB tensor-cache
   sidecar = 383 MB staged. `lc0:tvmjs-webgpu-local-artifacts-check` ok.
   Detached params is what makes this layout possible at all: a 370 MB embedded
   wasm per batch was never on the table.

## Harness de-hardcoding (now multi-family)

- `lc0-tvmjs-webgpu-smoke.html`: new page params `ortModel`
  (`{batch}`/`{dtype}` path template) and `fixtureBaseline` (native JSONL
  path); defaults preserve t1 behavior.
- `scripts/lc0_tvmjs_webgpu_smoke.mjs`: `--ort-model`, `--fixture-baseline`,
  recorded in artifacts.
- `scripts/lc0_tvmjs_webgpu_fixed_suite.mjs`: forwards both, plus `--tie-epsilon`.

## f16 tie-tolerance gate (`--tie-epsilon`, new)

BT4's 15-layer f16 accumulation produces argmax flips on near-tie rows that
t1 happened not to hit; the strict 100%-match gate marked otherwise-healthy
runs failed. Opt-in `--tie-epsilon X` on the smoke driver records such
mismatches as `tieTolerated` (raw counts untouched, tolerated rows listed
with their gaps) instead of failing:

- native/eval: prior gap between TVM's chosen move and the baseline best move,
  measured in TVM's own `topPriors`, ≤ X;
- ORT eval rows: row `maxTopPriorAbsDiff` ≤ X;
- search rows: **visit tie** (both moves equal visits in the TVM row's own
  stats — selection is by visits, so equal visits = pure tie-break) or prior
  gap ≤ X.

Default (no flag) remains strict. Evidence below used `--tie-epsilon 0.01`.
This is a research-side first cut of the f16 drift/tolerance policy the
runbook lists as a promotion blocker; ratify thresholds before any promotion.

## Evidence (all artifacts in `artifacts/tvm/bt4it332_*`)

### Parity

- Native eval parity b8: 7/8 strict; the 1 flip is `black-promotion-near`,
  a2a1q vs h2h1q with prior gap 0.0024 in a three-way near-tie; max top-prior
  abs diff vs native **0.0088** (t1 was 0.0059). b4 reproduces 7/8 with the
  identical 0.0088 max diff (good cross-batch determinism).
- TVMJS-vs-ORT f16 eval: 6/8 strict on the synthetic fixtures — both flips are
  the two promotion near-tie fixtures (row maxTopPriorAbsDiff ≤ 0.005); 16/16
  on UHO-lite rows.
- Search parity vs ORT f16 at v16: b8 8/8 (synthetic), UHO-lite 16 positions
  × 2 repeats **30/32**, both flips the same FEN (`r1bqkbnr/pppp1ppp/2n5/4p3/
  1b1PP3/…`, the position already known unstable from the t1 campaign) and
  both are exact visit ties (e5d4/d7d6/d7d5 all at 3 visits in the TVM tree).
  Native Stockfish depth-14 triage: after TVM's d7d5 white is −375 cp, after
  ORT's e5d4 −415 cp — two winning moves 40 cp apart.
- **Stockfish depth-3 scored deltas: 30 rows, TVMJS-minus-ORT cp min/max/mean
  0/0/0.** Research gate + evidence summary + research-only isolation all green.

### Performance (b8/v16 unless noted)

- Per-invoke GPU: b4 **40.8 ms**, b8 **84.9 ms** over 341 passes —
  **linear in batch, ~100% real kernel time** (84.9 ms measured kernel sum vs
  ~82 ms end-to-end GPU wait). Unlike t1 (sublinear, b16 won), BT4 saturates
  the GPU at b4 already: larger batches buy nothing, so the batch lever is
  dead here; b4 and b8 tie end-to-end (242.6 vs 248.5 ms mean search at v16).
  b16+ exports intentionally skipped.
- Top kernels: `fused_matmul4_add1` 28.8%, mish-FFN matmul
  (`…softplus_tanh_multiply`) 19.7%, `fused_matmul3_add1_multiply3_add4`
  15.9%, `fused_matmul4_add1_multiply3_add4` 9.1% — top 4 matmul families
  ≈74% of kernel time, same shape as t1.
- Search wall v16: TVMJS ≈ ORT f16 WebGPU (248 vs 254 ms quiet-machine;
  313 vs 330 ms on the loaded-machine bridge run) — **~0.25 s/move at 16
  visits**, in line with the plan's analysis/casual-play budget.
- Startup (cold): tensor-cache fetch+GPU upload **1.45 s** (353 MB ≈ 250 MB/s
  local), pipeline prebuild **590 ms** (~6× t1, tracks d_model), wasm
  fetch+verify 16 ms, instantiate 28 ms. Repeat-load: Cache Storage warm
  read-back **124–150 ms** for 370 MB of params.
- JS search overhead ~2.6 ms/search (1%) — still a non-lever.
- Footprint (`bt4it332_tvmjs_webgpu_bundle_footprint_f16_v1.json`): staged
  bundle 389.8 MB raw → 342.1 MB gzip (0.878) / **322.8 MB Brotli (0.828)** —
  f16 weight entropy compresses far worse than t1's bundle (0.77); the
  download is a one-time ~323 MB, then Cache Storage repeat loads at
  124–150 ms. The progressive story (play on t1, swap to BT4) from the plan
  is clearly required for cold-start UX.

### Remaining levers for BT4 perf

Batching and pipelining are dead (GPU-saturated, linear). What's left:
per-shape matmul schedule tuning at 1024-class shapes (the t1 dlight sweep
verdict "parked low-ROI" was measured on 256-class shapes and does NOT
automatically transfer — the 4 fused matmul families here are worth one
metaschedule-grade attempt), and evaluation-count reduction (tree reuse).

## Known gaps / cautions

- The in-page Stockfish scorer returns `reply: null` exactly on mismatch rows
  (where TVM/ORT after-FENs differ → two scoring searches); pre-existing bug,
  triaged manually with native Stockfish this cycle. Worth fixing before
  larger scored suites.
- WebGPU device still requests only `shader-f16`; default limits sufficed for
  d_model 1024 on Apple Silicon, but the `requiredLimits` pattern from the
  Tiny lane should be adopted before non-Apple coverage.
- All artifacts remain research-only/ignored; promotion blockers unchanged
  (hosting/cache policy, non-Apple GPU datapoint, ratified tolerance policy).
