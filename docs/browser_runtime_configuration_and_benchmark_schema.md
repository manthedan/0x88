# Browser Runtime Configuration and Benchmark Schema

This document standardizes how LC0, Tiny Leela/SquareFormer, and future browser chess engines describe runtime choices, benchmark recipes, and promotion evidence.

The goal is to make performance work reproducible across:

- LC0 small/distilled browser packs;
- future larger LC0 packs;
- Tiny Leela MF80 / BT4 / SquareFormer models;
- UCI or piece-odds LC0 variants;
- ORT WebGPU/WASM, custom WebGPU, native ORT, and future cloud runtimes.

## Core rule

Separate these concepts:

1. **Model identity** — the neural artifact and training lineage.
2. **Ruleset identity** — standard chess, piece odds, or another UCI-compatible variant.
3. **Runtime Configuration** — how the model is executed.
4. **Benchmark Protocol** — the measurement recipe.
5. **Promotion Evidence** — the gate used to keep, promote, or reject a Runtime Configuration.

Do not encode benchmark protocol details into a runtime preset. A preset may fill runtime/search knobs, but fixtures, repeats, warmup, thresholds, cleanup policy, and promotion gates belong to the protocol and artifact.

## Runtime Configuration schema

Every benchmark artifact that compares inference or search performance should include a `runtimeConfiguration` object.

Recommended shape:

```json
{
  "schema": "browser-engine.runtime-configuration.v1",
  "engineFamily": "lc0",
  "modelId": "t1-256x10-distilled-swa-2432500",
  "modelArtifact": {
    "kind": "lc0web-pack",
    "precision": "f16",
    "layout": "raw-f16",
    "manifestUrl": "/models/lc0/t1-256x10-distilled-swa-2432500.batch8.f16.lc0web/model.lc0web.json"
  },
  "ruleset": {
    "id": "standard-chess",
    "policyMap": "lc0-1858",
    "inputEncoding": "lc0-classical-112"
  },
  "runtimeBackend": "custom-webgpu",
  "runtimeConfigId": "lc0-webgpu-research-b4",
  "headBackend": "wgsl",
  "inputBackend": "wasm",
  "encoderKernel": "mixed-tvm-ffn-smolgen-project",
  "legalPriorsBackend": "js",
  "precision": "f16-storage-f32-accum",
  "leafBatchSize": 4,
  "batchPipelineDepth": 1,
  "evalCacheEntries": 0,
  "reuseTree": false,
  "fallback": {
    "enabled": true,
    "fallbackRuntimeBackend": "ort-webgpu"
  }
}
```

### Required fields

| Field | Meaning |
|---|---|
| `schema` | Schema id, currently `browser-engine.runtime-configuration.v1`. |
| `engineFamily` | `lc0`, `tiny-squareformer`, `tiny-mf80`, `stockfish`, etc. |
| `modelId` | Stable model identity, not a runtime-path name. |
| `modelArtifact.kind` | `onnx`, `lc0web-pack`, `tvm-hybrid-manifest`, `wasm-engine`, etc. |
| `ruleset.id` | `standard-chess`, `piece-odds`, or variant id. |
| `runtimeBackend` | `ort-webgpu`, `ort-wasm`, `custom-webgpu`, `native-ort-cpu`, `native-ort-cuda`, `wasm-engine`, etc. |
| `runtimeConfigId` | Named Runtime Configuration or preset id, when available. |

### Optional but recommended fields

| Field | Meaning |
|---|---|
| `policyMap` | LC0 1858, SquareFormer action ids, variant-specific map. |
| `inputEncoding` | Feature/input contract. |
| `precision` | Effective execution precision, not only artifact dtype. |
| `headBackend` | `ort`, `wgsl`, `native`, etc. |
| `inputBackend` | `js`, `wasm`, `wgsl`, `native`, etc. |
| `encoderKernel` | Kernel family or generated-kernel id. |
| `legalPriorsBackend` | `js`, `wasm`, `gpu`, `native`, etc. |
| `leafBatchSize` | Search leaf batch size. |
| `batchPipelineDepth` | Speculative Search Pipeline depth. `1` is parity-preserving baseline. |
| `evalCacheEntries` | Evaluator cache capacity. |
| `reuseTree` | Whether search tree is reused between searches. |
| `fallback` | Runtime fallback policy and observed fallback result. |

## Benchmark artifact schema

Every benchmark artifact should include enough context to reproduce the result.

Recommended top-level shape:

```json
{
  "schema": "browser-engine.benchmark-artifact.v1",
  "artifactKind": "fixed-suite-search-throughput",
  "createdAt": "2026-06-05T00:00:00.000Z",
  "git": {
    "repo": "lc0_webgpu",
    "sha": "...",
    "dirty": false
  },
  "runtimeConfiguration": {},
  "benchmarkProtocol": {},
  "environment": {},
  "metrics": {},
  "driftCheck": {},
  "packFootprint": {},
  "executionFootprint": {},
  "cacheFootprint": {},
  "promotionAssessment": {}
}
```

### `benchmarkProtocol`

```json
{
  "name": "fixed-suite-search-throughput",
  "suiteId": "opening-suite-uho-lite-v1",
  "positionCount": 32,
  "visits": 32,
  "movetimeMs": null,
  "repeats": 3,
  "warmup": {
    "evalIters": 1,
    "searchIters": 1
  },
  "cleanupPolicy": "fresh-agent-browser-session-per-cell",
  "stopConditions": ["visit-budget"],
  "comparisonBaseline": "matched-runtime-control"
}
```

Keep protocol names explicit. Recommended names:

- `net-only-drift`
- `policy-only-latency`
- `fixed-visit-search`
- `fixed-time-search`
- `fixed-suite-search-throughput`
- `repeated-root-cache-reuse`
- `move-sequence-cache-reuse`
- `pack-load-footprint`
- `execution-footprint-probe`
- `arena-smoke`
- `arena-strength`

### `environment`

```json
{
  "browser": {
    "name": "Chromium",
    "version": "149",
    "userAgent": "...",
    "crossOriginIsolated": true
  },
  "webgpu": {
    "available": true,
    "adapter": "Apple M-series / Metal",
    "isFallbackAdapter": false,
    "features": ["shader-f16", "timestamp-query"]
  },
  "ort": {
    "requestedExecutionProvider": "webgpu",
    "resolvedExecutionProviders": ["webgpu", "wasm"],
    "threads": 1,
    "preferredOutputLocation": null
  },
  "host": {
    "os": "darwin",
    "arch": "arm64",
    "deviceClass": "local-dev"
  }
}
```

### `metrics`

Search/eval metrics should be reported separately.

```json
{
  "eval": {
    "latencyMs": { "mean": 4.86, "median": 4.72, "p95": 5.2 },
    "evalsPerSecond": 205.6,
    "backendTiming": {
      "inputBuildMs": 0.05,
      "gpuSubmitMs": 0.2,
      "readbackMs": 1.7,
      "legalPriorsMs": 0.4
    }
  },
  "search": {
    "requestedVisits": 32,
    "completedVisits": 32,
    "evalsPerSecond": 190.7,
    "visitsPerSecond": 180.2,
    "stopReason": "visit-budget",
    "bestMove": "d2d4",
    "batchHistogram": { "4": 8 },
    "readbackBytes": 7444,
    "readbackMapCount": 1
  },
  "cache": {
    "hits": 66,
    "misses": 33,
    "hitRate": 0.667,
    "entries": 33
  }
}
```

## Footprint schemas

### `packFootprint`

Use for model artifact bytes loaded from browser assets.

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

### `executionFootprint`

Use for persistent execution memory owned by the Runtime Configuration.

```json
{
  "schema": "browser-engine.execution-footprint.v1",
  "gpuBufferBytes": 56901000,
  "gpuBufferCount": 123,
  "categories": {
    "encoderWeights": 37000000,
    "encoderScratch": 12000000,
    "wgslHeadWeights": 7500000,
    "readback": 30000,
    "upload": 120000
  },
  "counts": {
    "layers": 10,
    "physicalSlots": 4,
    "deferredReadbackSlots": 1
  },
  "exclusions": [
    "browser/driver overhead",
    "shader and pipeline objects",
    "transient timestamp query buffers",
    "ORT internals unless explicitly reported",
    "JS object overhead unless explicitly reported"
  ]
}
```

### `cacheFootprint`

Use for evaluator-cache attribution, reported separately from GPU-buffer execution footprint.

```json
{
  "entries": 33,
  "maxEntries": 2048,
  "approxBytes": 46194,
  "approxKeyBytes": 10240,
  "approxEvaluationBytes": 35954,
  "note": "Approximate payload only; excludes JS Map/object overhead and backend resources."
}
```

## Drift and parity schema

Use `driftCheck` for backend/runtime correctness evidence.

```json
{
  "baseline": "native-lc0-blas-and-f32-onnx",
  "fixtureSuite": "native-fen-history-fixtures-v1",
  "fixtureCount": 9,
  "passed": true,
  "bestMoveMatches": 9,
  "bestMoveTotal": 9,
  "maxWdlAbsDiff": 0.0019,
  "maxTopPriorAbsDiff": 0.0162,
  "thresholds": {
    "bestMoveMatchesRequired": 9,
    "maxWdlAbsDiff": 0.003,
    "maxTopPriorAbsDiff": 0.02
  }
}
```

For Tiny Leela/SquareFormer, baselines may be FP32 ONNX, current champion, fixed policy fixtures, or cross-language Rust/TS contracts. State the baseline explicitly.

## Promotion assessment

Artifacts used for decisions should include a machine-readable final assessment.

```json
{
  "decision": "keep-opt-in",
  "reason": "Drift passed and fixed-suite throughput improved under matched runtime controls; stable default unchanged.",
  "defaultPromotion": false,
  "evidence": [
    "drift-check",
    "fixed-suite-search-throughput",
    "project-checks",
    "footprint-reported"
  ],
  "missingEvidence": [
    "full-32-position matched rerun"
  ]
}
```

Recommended decision values:

- `reject`
- `diagnostic-only`
- `keep-opt-in`
- `promote-runtime-configuration`
- `promote-default`
- `needs-more-evidence`

## Standard recipes

### LC0 small / current opt-in custom WebGPU

Runtime preset:

```text
--preset lc0-webgpu-research-b4
```

Meaning:

```text
runtimeBackend: custom-webgpu
runtime: hybrid-wgsl-heads
inputBackend: wasm
encoderKernel: mixed-tvm-ffn-smolgen-project
legalPriorsBackend: js
leafBatchSize: 4
batchPipelineDepth: 1
```

Required artifact set before any stronger claim:

```text
1. net-only drift against f32 ONNX/native LC0 BLAS
2. fixed-search fixture parity at depth 1
3. fixed-suite search throughput under matched controls
4. packFootprint
5. executionFootprint
6. cacheFootprint when cache is enabled
7. project checks / autoreview for code changes
```

Example commands:

```bash
npm run lc0:browser-hybrid-drift -- \
  --preset lc0-webgpu-research-b4 \
  --limit 9 \
  --baseline-mode serial

npm run lc0:browser-hybrid-search-fixture-parity -- \
  --preset lc0-webgpu-research-b4 \
  --visits 32 \
  --batch 4 \
  --batch-pipeline-depths 1 \
  --repeats 3 \
  --out /tmp/lc0_fixture_parity_b4.json

npm run lc0:browser-runtime-fixed-suite -- \
  --preset lc0-webgpu-research-b4 \
  --cache 0 \
  --fens eval/opening_suite_uho_lite_v1.fen \
  --max-positions 32 \
  --out /tmp/lc0_fixed_suite_b4.json
```

### Tiny SquareFormer / custom TVM WebGPU

Runtime Configuration should distinguish model and runtime:

```json
{
  "engineFamily": "tiny-squareformer",
  "modelId": "bt4-anneal-muon-best",
  "runtimeBackend": "custom-webgpu",
  "runtimeConfigId": "squareformer-tvm-webgpu-hybrid-v1",
  "modelArtifact": {
    "kind": "tvm-hybrid-manifest",
    "manifestUrl": "/runtimes/squareformer-tvm-hybrid/bt4-anneal-muon-best/v1/manifest.json"
  },
  "ruleset": {
    "id": "standard-chess",
    "policyMap": "squareformer-action-ids",
    "inputEncoding": "compact-square-tokens"
  }
}
```

Recommended standard evidence:

```text
1. contract/parity fixtures for feature encoding and move ids
2. FP32 ONNX vs custom WebGPU drift
3. policy-only latency
4. PUCT fixed-visit throughput
5. fixed-time arena/strength when serious
6. model efficiency/frontier card
7. pack/execution/cache footprint once runtime is deployable
```

Tiny already has candidate frontier-card requirements. The missing standardization work is to emit LC0-style footprint objects and make cache/broker metrics shape-compatible with LC0 artifacts.

### Future large LC0 model

Treat as a new `modelId` and likely a new `runtimeConfigId`, not as a replacement for LC0 small.

Additional required fields:

```text
modelArtifact.parameterScale
modelArtifact.shardCount
modelArtifact.compressedTransferBytes
executionFootprint.memoryCeilingClass
loadProtocol.streaming|eager
```

Promotion gates should include:

```text
1. load success and memory ceiling on target browsers
2. Pack Footprint and compressed serving report
3. Execution Footprint by category
4. drift against native LC0 baseline for the same network
5. fixed-suite search throughput at selected leaf batches
6. fallback behavior when memory/WebGPU is unavailable
```

### Future UCI / piece-odds LC0 variants

Do not reuse standard-chess artifacts without labeling the ruleset.

Required ruleset fields:

```json
{
  "id": "piece-odds",
  "variant": "queen-odds",
  "policyMap": "lc0-1858-piece-odds-compatible",
  "inputEncoding": "lc0-classical-112-piece-odds",
  "legalMoveContract": "uci-variant-v1"
}
```

Additional gates:

```text
1. legal-move contract fixtures for the variant
2. input-plane parity for odds/variant starting positions
3. policy-map legality and promotion/castling/en-passant edge cases
4. search fixture parity within the variant ruleset
5. no cross-pollution with standard-chess baseline artifacts
```

## Optimization mapping: LC0 small vs Tiny Leela

| Optimization family | LC0 small status | Tiny Leela/SquareFormer status | Standardization action |
|---|---|---|---|
| Runtime registry | LC0 opt-ins are script/runtime controls; stable default ORT | SquareFormer has promoted custom runtime registry entry | Use shared `runtimeConfiguration` schema for both. |
| Generated/custom kernels | LC0 uses WGSL heads and generated/tiled encoder kernels | SquareFormer TVM hybrid uses generated WebGPU kernels | Standardize kernel id and artifact manifest fields, not kernel internals. |
| Physical batching | LC0 WGSL-head path has real physical Leaf Batch b4 | Generic broker coalesces requests; SquareFormer hybrid currently serializes `evaluateBatch` | Use shared batch metrics: logical requests, physical batch histogram, represented positions. |
| Readback attribution | LC0 reports readback bytes/maps/timing | Tiny should report analogous ORT/custom timing where available | Standardize `metrics.eval.backendTiming` and `metrics.search.backendTiming`. |
| Cache/reuse | LC0 reports hits/misses and `cacheFootprint` | Tiny has `CachedEvaluator` and `BrokeredEvaluator` metrics | Add compatible `cacheFootprint` to Tiny artifacts. |
| Pack footprint | LC0 `lc0web` pack reports tensor/shard bytes | Tiny mostly reports ONNX/export bytes today | Add `packFootprint` for ONNX and TVM manifests. |
| Execution footprint | LC0 custom runtime reports explicit GPU buffer categories | Tiny custom runtime should report GPU buffers by block/category | Add `executionFootprint` to custom SquareFormer runtime. |
| Quantization | LC0 simple int8 attempts regressed; drift gates required | Tiny quantization is deployment polish with strict parity gates | Use one quantization drift schema across both. |
| Promotion | LC0 requires drift + fixed-suite throughput | Tiny uses frontier cards and strength/runtime Pareto | Require both runtime artifact and frontier card for serious deployment. |

## Implementation checklist for new engines/models

When onboarding a new engine/model, create or verify:

```text
1. model identity and artifact manifest
2. ruleset identity and legal-move contract
3. Runtime Configuration id
4. fallback Runtime Configuration
5. net/feature parity fixtures
6. policy-map fixtures
7. benchmark protocols to run
8. packFootprint support
9. executionFootprint support, or explicit unsupported note
10. cacheFootprint support when cache exists
11. fixed-suite search throughput artifact
12. drift/parity artifact
13. candidate frontier card if model quality is being judged
14. promotion assessment
```

## Non-goals

- Do not force LC0 and Tiny Leela to share WGSL kernels.
- Do not treat Stage Harness wins as promotion evidence.
- Do not promote `batchPipelineDepth > 1` as parity-preserving fixed-search evidence.
- Do not compare model strength and runtime speed as one unlabelled number.
- Do not let standard-chess artifacts stand in for piece-odds or other UCI variants.
