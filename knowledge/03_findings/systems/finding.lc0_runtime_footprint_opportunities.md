---
created: 2026-06-05
updated: 2026-06-05
project: tiny-neural-chess
id: finding.lc0_runtime_footprint_opportunities
type: finding
title: Finding - LC0 Runtime Footprint opportunities
status: active
confidence: medium
priority: high
supports:
  - [[Design - Inference optimization]]
agent_summary: >
  The current LC0 WebGPU opt-in has concrete Pack Footprint, Execution Footprint, and cache/reuse opportunities. Pack compression is low-risk but modest; smolgen weights and per-slot/per-layer GPU scratch buffers dominate likely runtime footprint; evaluator cache and tree reuse are useful for repeated-position analysis but require suite-level hit-rate attribution before promotion.
---

# Finding - LC0 Runtime Footprint opportunities

## Scope

This is an opportunity assessment for the opt-in **LC0 WebGPU Research b4** Runtime Configuration, not a promotion claim and not a default-runtime change.

Runtime/search preset under discussion:

- `hybrid-wgsl-heads`
- WASM input
- `mixed-tvm-ffn-smolgen-project`
- JS legal priors
- Leaf Batch size `4`
- Speculative Search Pipeline depth `1`

## Current Pack Footprint baseline

Measured through the browser pack-load probe after adding the canonical `packFootprint` object:

```json
{
  "declaredTensorBytes": 40413598,
  "loadedTensorBytes": 40413598,
  "totalShardBytes": 40418450,
  "loadedShardBytes": 40418450,
  "tensorCount": 375,
  "loadedTensorCount": 375,
  "shardCount": 3,
  "loadedShardCount": 3,
  "dtypeHistogram": { "f16": 306, "i32": 5, "i64": 64 }
}
```

Probe details:

- artifact: `/tmp/lc0_pack_footprint_probe_20260605.json`
- URL: `/lc0-policy-only.html?packProbe=1&packVerify=0`
- model: `t1-256x10-distilled-swa-2432500.batch8.f16`
- layout: `raw-f16`
- elapsed worker pack load: about `85.9 ms` in the sampled browser run
- round trip: about `139.3 ms`

Interpretation:

- The full current pack is already dense f16; there is no obvious whole-pack unused-weight win for the 10-layer Runtime Configuration.
- Shard padding/overhead is negligible: `totalShardBytes - declaredTensorBytes = 4,852 bytes`.
- Tensor filtering can reduce `loadedTensorBytes`, but current coarse shard layout means selected tensors still fetch whole containing shards unless the pack is reshaped.

## Pack composition

Approximate tensor byte attribution from the manifest:

| Category | Bytes | MB | Share |
|---|---:|---:|---:|
| encoder smolgen layer weights | 21,274,320 | 21.274 | 52.6% |
| encoder FFN weights | 10,511,380 | 10.511 | 26.0% |
| encoder attention weights | 5,263,860 | 5.264 | 13.0% |
| global smolgen constant | 2,097,152 | 2.097 | 5.2% |
| value head | 677,496 | 0.677 | 1.7% |
| policy head | 396,930 | 0.397 | 1.0% |
| input body | 164,416 | 0.164 | 0.4% |

The largest footprint target is smolgen: layer smolgen plus the global smolgen constant is about `23.37 MB`, or `57.8%` of declared tensor bytes.

## Packaging compression opportunity

Local gzip/brotli measurements on the current manifest and three weight shards:

| Asset | Raw bytes | gzip-9 | brotli-11 | brotli ratio |
|---|---:|---:|---:|---:|
| manifest JSON | 296,585 | 26,397 | 17,996 | 6.1% |
| `weights.000.bin` | 16,565,504 | 15,292,856 | 14,542,191 | 87.8% |
| `weights.001.bin` | 15,766,736 | 14,563,552 | 13,853,255 | 87.9% |
| `weights.002.bin` | 8,086,210 | 7,007,779 | 6,642,114 | 82.1% |
| total | 40,715,035 | 36,890,584 | 35,055,556 | 86.1% |

Opportunity:

- **Pack Footprint compression** with HTTP brotli/gzip is low-risk and does not change kernels or Drift Checks.
- Expected whole-pack byte reduction is modest, roughly `9.4%` for gzip-9 or `13.9%` for brotli-11 in this sample.
- The binary f16 shards are already high-entropy enough that ordinary transport compression is not a major breakthrough.

## Execution Footprint opportunities

Execution Footprint still needs explicit GPU buffer accounting before any gate. Code inspection shows several likely attribution buckets.

### 1. Per-layer scratch buffers are likely the largest reusable execution buffers

For each encoder layer slot, the WGSL runtime allocates persistent intermediate buffers for smolgen, QKV, scores, probabilities, attention, FFN hidden, skips, and output. From the fixed shapes in `wgslMatmulAddProbe.ts`, this is roughly `1.20 MB` per layer slot before small uniforms.

At 10 layers:

- about `11.96 MB` per physical evaluation slot for encoder scratch/intermediates;
- Leaf Batch size `4` can require four physical slots for WGSL-head batch evaluation, so slot scratch can become a material part of Execution Footprint;
- Speculative Search Pipeline depth greater than `1` can duplicate additional slot/readback resources, but that remains an opt-in speculative path.

Opportunity:

- Add explicit `executionFootprint.gpuBuffers` first.
- Then investigate lifetime-based scratch reuse across layers and ping-pong output buffers.
- This should be treated as footprint work first; any throughput effect must be measured in fixed-suite search.

### 2. Weights are duplicated between loaded pack bytes and GPU buffers

The current pack loader keeps shard-backed `Uint8Array` tensor views while the runtime also uploads weights to GPU buffers. This is expected, but it means Execution Footprint includes both JS-side pack storage and GPU-side weights unless pack bytes are released after upload.

Opportunity:

- Attribute JS pack bytes separately from GPU weight bytes.
- Consider an opt-in post-upload release path only after verifying that no runtime path needs tensor bytes for later lazy construction, batch slot creation, or diagnostics.

### 3. WGSL-head weights are expanded to f32

WGSL policy/value heads convert f16 head tensors to f32 storage buffers. The head is small compared with the encoder, so this is a lower-priority compression target, but it is a clean accounting item.

Opportunity:

- Account for f16-pack bytes versus f32 execution bytes in `executionFootprint`.
- Do not prioritize head compression unless accounting shows unexpected growth.

### 4. Input-body GPU resources are duplicated per physical slot

With WASM input, the runtime still uses GPU input-body projection. The input-body buffers are small, but they are created per physical slot.

Opportunity:

- Account per-slot duplicated input-body bytes.
- Only optimize if execution accounting shows slot count is a meaningful memory ceiling.

## Cache and reuse opportunities

Small repeated-position probes with the current preset show the cache/reuse mechanisms are real, but the benchmark shape is diagnostic only.

Artifacts:

- `/tmp/lc0_reuse_cache_false_cache0.json`
- `/tmp/lc0_reuse_cache_false_cache2048.json`
- `/tmp/lc0_reuse_cache_true_cache0.json`

Probe summary, each at visits `32`, Leaf Batch `4`, 3 repeated searches of the same FEN:

| Setup | Times ms | Completed visits | Cache hits | Neural misses | Hit rate | Root reused |
|---|---:|---:|---:|---:|---:|---:|
| reset tree, no eval cache | 643.7, 165.1, 158.7 | 96 | 0 | 99 | 0.000 | 0 |
| reset tree, eval cache 2048 | 498.3, 3.6, 2.6 | 96 | 66 | 33 | 0.667 | 0 |
| reuse tree, no eval cache | 484.8, 1.1, 0.8 | 32 | 0 | 33 | 0.000 | 2 |

Interpretation:

- Evaluator cache can eliminate repeated neural evals for identical repeated analysis; in this probe, the second and third reset-tree searches mostly hit cache.
- Tree reuse can be even stronger for repeated analysis of the same root because the root already has the requested visits; this is user-interaction reuse, not fixed-suite promotion evidence.
- These wins do not automatically transfer to a varied fixed-suite or game arena. The next step is hit-rate attribution over representative suites and real move sequences.

## Ranked optimization lanes

1. **Execution Footprint accounting**: GPU buffer bytes by category (`encoderWeights`, `encoderScratch`, `wgslHeadWeights`, `wgslHeadScratch`, `readback`, `upload`, `inputBody`) plus physical slot/layer counts. This is the first implementation lane because it does not change numerics or stable defaults.
2. **Shared immutable smolgen projection weight**: code inspection shows `/const/smolgen_w` is a global tensor but the current hybrid runtime uploads it inside each encoder layer's weight object. If accounting confirms the expected duplicate GPU weight bytes, reuse one immutable GPU buffer across layers before broader compression work. This should be a low-footprint Runtime Configuration improvement with Drift Checks and fixed-suite timing.
3. **Encoder scratch / activation reuse**: after accounting, test lifetime-based reuse or ping-pong buffers for layer intermediates. Keep only if Drift Checks pass and Fixed-suite Search Throughput is neutral/better, or document as low-footprint opt-in.
4. **Cache-hit instrumentation and cache policy matrix**: emit cache entry count, hits/misses, approximate cache bytes, and per-search hit rate in artifacts; evaluate on repeated analysis, move-by-move arena, and fixed-suite protocols separately.
5. **Package compression / serving check**: document whether deployment hosting already serves brotli/gzip for `.bin` and `.json`; if not, enable as Pack Footprint improvement only.
6. **Post-upload JS pack release**: verify which retained tensor views keep shard `ArrayBuffer`s alive, especially head tensors used for lazy batch-slot construction; release or copy only after ownership is clear.
7. **Generated smolgen compression/quantization research**: smolgen dominates Pack Footprint, but runtime compression/quantization must pass Drift Checks and throughput gates; do this only after accounting and generated-kernel plans are clear.

## Started implementation

The first accounting pass now emits `executionFootprint` from the hybrid runtime for eval/profile/search artifacts where the worker owns a custom lc0web runtime. The object reports explicit lc0web-owned persistent GPU buffer bytes by category, GPU buffer counts, layer count, physical WGSL batch slot count, deferred ring count, and upload/readback capacities. It intentionally excludes browser/driver overhead, shader/pipeline objects, transient timestamp-query buffers, ORT internals, and browser heap samples.

Verification probe:

- artifact: `/tmp/lc0_execution_footprint_probe.json`
- preset: **LC0 WebGPU Research b4**
- visits: `4`
- Leaf Batch size: `4`
- result: `executionFootprintMB.gpuBufferMB = 116.842`
- physical slots: `4`

Top reported GPU categories:

| Category | MB |
|---|---:|
| `encoderSmolgenWeights` | 42.245 |
| `encoderAttentionScratch` | 26.214 |
| `encoderFfnScratch` | 15.729 |
| `encoderFfnWeights` | 10.522 |
| `wgslHeadWeights` | 7.538 |
| `encoderSmolgenScratch` | 6.308 |
| `encoderAttentionWeights` | 5.274 |

The probe confirmed that `encoderSmolgenWeights` was the largest explicit GPU category. It included repeated uploads of the immutable `/const/smolgen_w` projection tensor through each layer weight object.

Started optimization:

- artifact: `/tmp/lc0_execution_footprint_shared_smolgen_probe.json`
- change: share one immutable `/const/smolgen_w` GPU buffer across encoder layers
- result: `executionFootprintMB.gpuBufferMB = 97.968`
- reduction: about `18.874 MB`, or `16.2%` of explicit lc0web-owned persistent GPU buffer bytes in the b4/depth1/Leaf Batch 4 probe
- `encoderSmolgenWeights` dropped from `42.245 MB` to `21.274 MB`
- new `encoderSharedSmolgenWeight` category reports the single shared `2.097 MB` buffer

This is a Runtime Footprint optimization, not a throughput promotion claim. It still requires Drift Checks and Fixed-suite Search Throughput comparison before being treated as a kept Runtime Configuration improvement.

Follow-up optimization started:

- artifact: `/tmp/lc0_execution_footprint_shared_scratch_probe.json`
- change: share per-slot encoder scratch/intermediate buffers across sequential encoder layers while keeping layer output buffers distinct
- result: `executionFootprintMB.gpuBufferMB = 56.901`
- reduction versus shared-smolgen-only probe: about `41.067 MB`
- reduction versus original accounting probe: about `59.941 MB`, or `51.3%` of explicit lc0web-owned persistent GPU buffer bytes
- `encoderAttentionScratch` dropped from `26.214 MB` to `2.621 MB`
- `encoderSmolgenScratch` dropped from `6.308 MB` to `0.631 MB`
- `encoderFfnScratch` dropped from `15.729 MB` to `1.311 MB`
- distinct `encoderLayerOutputs` are now reported separately at `2.621 MB`

Validation artifacts after scratch sharing:

- Drift Check artifact: `/tmp/lc0_shared_scratch_drift.json`
  - fixtures: `9`
  - f32 best-move matches: `9/9`
  - native best-move matches: `9/9`
  - f32 top-prior max abs diff: `0.0004032383`
  - native top-prior max abs diff: `0.0008249045`
- Search smoke artifact: `/tmp/lc0_shared_scratch_search_b32_v32_rep3.json`
  - one-FEN fixed-visit smoke only, not fixed-suite promotion evidence
  - visits: `32`, Leaf Batch size: `4`, repeats: `3`
  - mean search round trip: `239.9917 ms`
  - completed visits/s: `133.338`
  - execution footprint remained `56.901 MB`

An earlier 16-position fixed-suite smoke attempt with short movetime timed out before producing an artifact. Later 20-position fixed-suite smokes completed after fixing FEN-comment filtering and `--cache 0`, but a full matched promotion-grade fixed-suite throughput comparison against the recovered pre-footprint-control state remains pending.

Cache-footprint instrumentation started:

- implementation: `CachedLc0Evaluator.cacheFootprint()` reports cache entries, max entries, approximate key bytes, approximate evaluation payload bytes, and a note that JS object/map overhead is excluded.
- worker/search artifacts now include `cacheFootprint` and rounded `cacheFootprintKB` alongside `executionFootprint`.
- artifact: `/tmp/lc0_cache_footprint_b4_cache2048_probe.json`
  - preset: **LC0 WebGPU Research b4**
  - eval cache entries setting: `2048`
  - repeated same-FEN searches: `3`, visits `32`, Leaf Batch size `4`
  - cache hits: `66`
  - neural eval misses: `33`
  - hit rate: `0.666667`
  - cache entries retained: `33`
  - approximate cache payload: `46.194 KB`
  - execution footprint: `56.901 MB`

This confirms the cache artifact path can now separate Runtime Footprint GPU buffers from evaluator-cache payload attribution. It remains diagnostic until run over representative fixed-suite and move-sequence protocols.

Fixed-suite/script integration follow-up:

- `scripts/lc0_browser_runtime_fixed_suite.mjs` now filters `#` comments from FEN files and accepts `--cache 0`, so no-cache controls can be run directly from fixture files.
- Arena fixed-suite and arena bench artifacts now include `telemetry.lc0ExecutionFootprint` and `telemetry.lc0CacheFootprint`.
- Smoke artifact: `/tmp/lc0_fixed_suite_shared_scratch_4pos_footprint.json`
  - positions: `4`
  - movetime: `100ms`
  - Stockfish score depth: `1`
  - cache: `0`
  - Leaf Batch size: `4`
  - execution footprint in fixed-suite telemetry: `56.901 MB`
  - cache footprint: `0` entries / `0` approximate bytes
- Summary-footprint artifact: `/tmp/lc0_fixed_suite_shared_scratch_4pos_summary_decimal_footprint.json`
  - fixed-suite CLI summary now reports `executionFootprintMB`, `cacheFootprintKB`, and `cacheEntries`
- 20-position fixed-suite smoke artifacts, same preset and b4/depth1 controls:
  - no-cache: `/tmp/lc0_fixed_suite_shared_scratch_20pos_cache0_mt100_depth1.json`
    - positions: `20`
    - movetime: `100ms`
    - Stockfish score depth: `1`
    - execution footprint: `56.901 MB`
    - cache footprint: `0 KB`, entries `0`
    - evals/s: `190.781`
  - cache 2048: `/tmp/lc0_fixed_suite_shared_scratch_20pos_cache2048_mt100_depth1.json`
    - positions: `20`
    - movetime: `100ms`
    - Stockfish score depth: `1`
    - execution footprint: `56.901 MB`
    - cache footprint: `696.858 KB`, entries `386`
    - cache hits/misses: `0/386` in this varied fixed-suite smoke
    - evals/s: `178.971`
- 20-position fixed-suite timing artifacts using the previous failed-run shape (`movetime 200ms`, Stockfish score `50ms`) now complete:
  - no-cache: `/tmp/lc0_fixed_suite_shared_scratch_20pos_cache0_mt200_sf50.json`
    - execution footprint: `56.901 MB`
    - cache footprint: `0 KB`, entries `0`
    - cache hits/misses: `0/734`
    - evals/s: `170.267`
  - cache 2048: `/tmp/lc0_fixed_suite_shared_scratch_20pos_cache2048_mt200_sf50.json`
    - execution footprint: `56.901 MB`
    - cache footprint: `1508.704 KB`, entries `786`
    - cache hits/misses: `0/786` in this varied fixed-suite smoke
    - evals/s: `185.837`

Fixture parity/search confirmation after shared scratch:

- artifact: `/tmp/lc0_fixture_parity_shared_scratch_v32_b4_16_rep3.json`
- cells: `48` (`16` fixtures × `3` repeats)
- native best-move matches: `48/48`
- depth-baseline best-move matches: `48/48`
- mismatches: `0`
- mean `totalEvalMsPerPosition`: `4.802193`
- median `totalEvalMsPerPosition`: `4.71525`
- mean search round trip: `164.231875 ms`
- total eval calls: `1557`

This is stronger evidence that the Runtime Footprint reductions did not break fixed-search parity. It is still not a full 32-position fixed-suite promotion rerun, but it is a matched fixed-search fixture confirmation with the current preset and b4/depth1 controls.

## Non-goals for this finding

- Do not change stable defaults.
- Do not promote cache/tree-reuse measurements as fixed-suite throughput evidence.
- Do not begin runtime quantization/compression without Drift Checks.
- Do not treat browser heap APIs as primary evidence; use explicit lc0web accounting first.
