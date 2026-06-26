# Model manifest

Centipawn model metadata is split across ONNX files, runtime `*.meta.json` files,
dataset manifests, training logs, arena results, and frontier cards. The project-level model
manifest makes those pieces queryable from one place.

For the Centipawn transformer -> larger LC0 browser deployment ladder, see
the current browser-productization docs. Future model cards should define `architecture`, `browser_runtime`, and `quality_gates` fields before a model is treated as browser-productizable.

## Files

- Curated provenance overrides:
  ```text
  eval/model_manifest_overrides.json
  ```
- Manifest builder:
  ```text
  eval/build_model_manifest.py
  ```
- Generated current manifest:
  ```text
  artifacts/analysis/model_manifest.current.json
  artifacts/analysis/model_manifest.current.md
  ```

`artifacts/analysis/*` is generated and may be refreshed at any time. The source
of truth for manual provenance is `eval/model_manifest_overrides.json`.

## Refresh

```bash
npm run analysis:model-manifest
```

or directly:

```bash
.venv-onnx/bin/python eval/build_model_manifest.py \
  --overrides eval/model_manifest_overrides.json \
  --out artifacts/analysis/model_manifest.current.json \
  --md-out artifacts/analysis/model_manifest.current.md
```

## Schema sketch

Each generated model entry contains:

```json
{
  "model_id": "cnn64x6_cand48_phase1",
  "display_name": "CNN-AV 64x6 C=48 phase1",
  "family": "ResidualCNN_AV",
  "status": "completed",
  "artifacts": {
    "onnx": "...",
    "meta": "...",
    "train_log": "..."
  },
  "training": {
    "policy_dataset": "supervised_100m_elite_tcec_v1",
    "configured_policy_rows": 25000000,
    "av_dataset": "chessbench_av_top48_v1",
    "av_positions_available": 16286511,
    "epochs": 1
  },
  "runtime_meta": {
    "architecture": "residual_tower",
    "channels": 64,
    "blocks": 6,
    "input_planes": 38,
    "av_head_exported": true
  },
  "exports": {
    "primary": {
      "params": 8545012,
      "flops_estimate": null,
      "bundle_bytes": 34201144,
      "int8_param_mib_est": 8.15
    }
  },
  "frontier_card": {
    "params": 8545012,
    "flops_macs_estimate": null,
    "fixed_visit_strength": {},
    "fixed_time_strength": {},
    "deployment": null
  },
  "training_metrics": {
    "metrics": {
      "epoch1_dev_policy_ce": 1.838059
    },
    "best": {
      "best_composite_key": "step20000_composite"
    }
  },
  "arena_refs": []
}
```

## What is computed vs manual

Computed by `eval/build_model_manifest.py`:

- artifact existence
- ONNX parameter count
- ONNX bundle bytes, including `.onnx.data` sidecars
- FP16/INT8/INT4 parameter-size estimates
- FLOPs/MACs estimates when an architecture-specific counter is available
- compact runtime metadata from `*.meta.json` with huge move lists removed
- `METRIC key=value` lines from training logs
- dataset row/candidate totals from dataset manifests
- search-mode arena references when arena protocol resources include ONNX paths

Manual in `eval/model_manifest_overrides.json`:

- canonical `model_id`
- display name/family/status
- training source and intended row budget
- whether a run is queued/running/completed
- notes/tags for historical context

## Browser deployment extension

The next manifest iteration should add deployment-specific fields shared by TinyBT and LC0-web packs:

```text
architecture      token/plane schema, layers, channels, heads, FFN size, exported heads
exports           f32/f16/int8 artifacts, pack shards, checksums, byte sizes
browser_runtime   ORT WebGPU/WASM loadability, custom WGSL support, memory/load/latency
quality_gates     policy/WDL drift, top-k agreement, fixed-search and arena references
```

Keep these as evidence fields, not aspirations: unknown browser support should be `null`/missing until a smoke or artifact exists.

## Conventions

- Use explicit candidate-count names, e.g. `cnn64x6_cand48_phase1`.
- Do not claim one global Elo; keep strength protocol-relative.
- Track policy-only, classic PUCT, AV-PUCT, fixed-visit, and fixed-time results separately.
- Quantized size estimates are not release claims until actual quantized artifacts are tested with effectively zero quality loss.
- Do not use a hard MB cap as the promotion rule; use the frontier-card axes: strength, params, FLOPs/MACs, bytes, latency, blunders, and calibration.
- For old artifacts with incomplete metadata, prefer `provenance_confidence: low` or notes instead of guessing.
