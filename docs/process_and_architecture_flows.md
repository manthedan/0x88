# Tiny Leela process and architecture flow atlas

This is the diagram-first map for the project. It is intentionally high level: live run state stays in `artifacts/*`, canonical agent context stays in `knowledge/09_agent_context/`, and this document explains how the major systems fit together.

Use this as the first stop when onboarding someone to model lanes, training loops, cloud dataflow, evaluation, and deployment.

## 1. Portfolio lanes

```mermaid
flowchart TD
  Project["Tiny Leela<br/>browser-deployable neural chess"]

  Project --> CNN["CNN supervised anchors<br/>deployable baseline lane"]
  Project --> MF["Tactical MoveFormer hybrid<br/>tactical / sidecar lane"]
  Project --> BT4["Tiny BT4 / SquareFormer<br/>h7/h8 square-token lane"]
  Project --> SUPSP["Supervised-bootstrap self-play<br/>SUP-SP / Gumbel"]
  Project --> ZERO["Clean Gumbel-Zero<br/>rules-only from scratch"]

  CNN --> CNNOut["ONNX + browser parity<br/>classic PUCT default"]
  MF --> MFOut["paused unless explicitly resumed<br/>MF80 supervised checkpoints"]
  BT4 --> BT4Out["h7/h8 caches -> BT4 train<br/>SquareFormer parity gates"]
  SUPSP --> SUPOut["accepted supervised model<br/>generates labeled self-play"]
  ZERO --> ZeroOut["random/uniform bootstrap<br/>no teacher contamination"]

  classDef lane fill:#eef6ff,stroke:#2563eb,color:#172554;
  classDef guard fill:#fff7ed,stroke:#c2410c,color:#7c2d12;
  classDef output fill:#ecfdf5,stroke:#059669,color:#064e3b;
  class CNN,MF,BT4,SUPSP,ZERO lane;
  class MFOut,ZeroOut guard;
  class CNNOut,BT4Out,SUPOut output;
```

Core separation rule:

- **Supervised / SUP-SP** may use trained models and labeled supervised provenance.
- **Gumbel-Zero** must remain rules-only and random/from-scratch, with separate buffers and manifests.
- **Evaluation/deployment** defaults to classic PUCT unless a protocol card explicitly declares Gumbel or aux/AV search.

## 2. Model architecture breakdowns

### CNN supervised anchors

```mermaid
flowchart LR
  Fen["FEN + history<br/>board tensor"] --> Tower["Residual CNN tower<br/>64/80/96 channels"]
  Tower --> Policy["policy head<br/>move distribution"]
  Tower --> WDL["WDL/value head"]
  Policy --> Search["classic PUCT<br/>legal move masking"]
  WDL --> Search
  Search --> Export["ONNX export<br/>meta sidecar"]
  Export --> Browser["browser / node evaluator<br/>release candidate"]
```

CNN models are the most mature deployment lane: easiest export path, mature ONNX evaluator, browser parity tests, and current best seed for supervised-bootstrap Gumbel self-play.

### Tactical MoveFormer hybrid

```mermaid
flowchart LR
  Board["board tensor"] --> Hybrid["CNN stem / board features"]
  Sidecar["move/token sidecar cache"] --> MoveTokens["move-token features"]
  Hybrid --> Fusion["hybrid fusion trunk"]
  MoveTokens --> Fusion
  Fusion --> Policy["policy"]
  Fusion --> Value["WDL/value"]
  Fusion --> Aux["optional aux diagnostics"]
  Policy --> Eval["arena + PUCT tuning"]
  Value --> Eval
  Aux --> Eval
```

This lane is useful for tactical/move-centric experiments, but is operationally paused unless explicitly resumed. Treat existing MF80 checkpoints as supervised candidates and do not start new Tactical MoveFormer work without approval.

### Tiny BT4 / SquareFormer

```mermaid
flowchart TD
  Rows["h8 supervised rows<br/>history-aware positions"] --> Cache["SquareFormer h7/h8 caches<br/>square tokens + relations"]
  Cache --> Tokens["64 square tokens<br/>piece/state/history features"]
  Tokens --> Bias["chess-aware relation bias<br/>geometry / rays / attacks"]
  Bias --> Trunk["small transformer trunk<br/>BT4-inspired blocks"]
  Trunk --> PHead["from-to policy head"]
  Trunk --> VHead["WDL/value head"]
  Trunk --> AVHead["action-value / regret head<br/>experimental"]
  PHead --> Search["classic PUCT default"]
  VHead --> Search
  AVHead --> AuxSearch["conditional AV/aux PUCT<br/>experimental"]
```

The SquareFormer lane is cache-sensitive: h2 caches are not substitutes for true h7/h8 training. BT4/SquareFormer promotion requires cache manifest validation, model export/parity checks, and eval gates.

## 3. Supervised data and training pipeline

```mermaid
flowchart TD
  Raw["raw elite / TCEC / teacher rows"] --> Build["build supervised dataset<br/>schema + row caps + reports"]
  Build --> Manifest["dataset manifest<br/>train/dev counts + history_plies"]
  Manifest --> CacheChoice{"architecture needs cache?"}

  CacheChoice -->|CNN| TrainCNN["train_board_cnn / residual trainer"]
  CacheChoice -->|MoveFormer| Sidecar["build moveformer sidecar cache"]
  CacheChoice -->|SquareFormer| SqCache["build squareformer h7/h8 cache"]

  Sidecar --> TrainMF["train MoveFormer hybrid"]
  SqCache --> TrainBT4["train SquareFormer / BT4"]

  TrainCNN --> Ckpt["checkpoints + train/dev metrics"]
  TrainMF --> Ckpt
  TrainBT4 --> Ckpt

  Ckpt --> Export["export ONNX + meta"]
  Export --> Parity["ONNX parity + evaluator tests"]
  Parity --> Eval["release gates / arenas"]
  Eval --> Promote{"promote?"}
  Promote -->|yes| Deploy["public/model registry / FE"]
  Promote -->|no| Archive["archive with protocol card"]
```

Operational notes:

- Use `.venv-onnx/bin/python` for repo Python tools.
- Do not commit generated outputs under `data/*`, `artifacts/`, `public/models/*.onnx`, `public/models/*.json`, or `dist-client/`.
- Epoch-level checkpoints mean interrupted partial epochs may need to resume from the previous completed epoch.

## 4. AWS data/cache cloud pipeline

```mermaid
flowchart TD
  Preflight["local preflight<br/>capacity + schema + compression"] --> Upload["upload compact inputs<br/>prefer jsonl.zst"]
  Upload --> Batch["AWS Batch array<br/>unique RUN_ID + shard indexes"]
  Batch --> Shards["immutable shard outputs<br/>shard_000000/..."]
  Shards --> Manifests["per-shard manifests<br/>row counts + sha + provenance"]
  Manifests --> Validate["global validation<br/>manifest checks + row totals"]
  Validate --> Finalize["finalize S3 prefix<br/>cache_manifest / dataset manifest"]
  Finalize --> Local["sync compact outputs locally<br/>only when needed for training"]

  classDef standard fill:#ecfdf5,stroke:#059669,color:#064e3b;
  class Upload,Shards,Manifests,Validate standard;
```

Cloud standards for all large jobs:

- Use compressed shard data by default: `.jsonl.zst` or equivalent.
- Keep outputs under immutable `RUN_ID` prefixes.
- Record `SHARD_INDEX`, seed, source model, model/meta SHA, git/job metadata, and output prefix.
- Do not bake large model artifacts into worker images; fetch by S3 URI and verify SHA256.
- Reuse model S3 artifacts where possible instead of re-uploading the same ONNX every run.
- Sync/download only the outputs needed for validation or training; exclude repeated model files.

## 5. Supervised-bootstrap Gumbel self-play cloud flow

```mermaid
flowchart TD
  Seed["accepted supervised seed<br/>example: cnn96x8_100m/e08"] --> Submit["submit_gumbel_selfplay_jobs.sh<br/>run packet + Batch array"]
  Submit --> ModelRef["model/meta S3 URI<br/>SHA256 provenance"]
  Submit --> Array["Batch shards<br/>per-shard seed = base + index stride"]
  ModelRef --> Worker["gumbel worker<br/>ONNX evaluator"]
  Array --> Worker
  Worker --> Raw["chunk.jsonl.zst<br/>search-improved root targets"]
  Worker --> Adapted["training_expanded.jsonl.zst<br/>supervised_sp schema"]
  Worker --> Logs["validate/report/adapter logs"]
  Raw --> Summary["run summary validation"]
  Adapted --> Summary
  Logs --> Summary
  Summary --> Curate["optional Spark/Glue/DuckDB curation<br/>compact, split, dedupe, metrics"]
  Curate --> Train["future SUP-SP training buffer"]
```

This lane is **not** clean Zero. It is supervised model self-play improvement, so every shard must carry `lane=supervised_sp` and `source_model` provenance.

## 6. Clean Gumbel-Zero loop

```mermaid
flowchart TD
  Init["zero lane evaluator<br/>uniform bootstrap or random init"] --> SelfPlay["Gumbel root self-play<br/>rules-only chess"]
  SelfPlay --> ZeroChunk["zero chunk<br/>lane=zero, source_model empty"]
  ZeroChunk --> Validate["strict validation<br/>legal/FEN/policy/result checks"]
  Validate --> Replay["zero-only replay buffer"]
  Replay --> TrainZero["train zero checkpoint<br/>policy + WDL first"]
  TrainZero --> ExportZero["export zero model"]
  ExportZero --> Arena["gzero_i vs gzero_i-1"]
  Arena --> PromoteZero{"within-lane progress?"}
  PromoteZero -->|yes| Init
  PromoteZero -->|no| Archive["archive checkpoint + metrics"]
```

Zero invariants:

- no supervised initialization,
- no teacher labels,
- no supervised replay contamination,
- checkpoint-vs-checkpoint promotion before cross-lane comparisons.

## 7. Evaluation and promotion flow

```mermaid
flowchart TD
  Candidate["candidate model + meta"] --> Smoke["static smoke<br/>load, legal moves, metadata"]
  Smoke --> Parity["encoding/evaluator parity<br/>move map, mirrored FEN, bytes-vs-path"]
  Parity --> Bucket["bucket eval / policy diagnostics<br/>dev metrics, top-k, WDL"]
  Bucket --> Search["PUCT consistency<br/>root prior parity, no illegal moves"]
  Search --> Quick["quick anchors<br/>Stockfish/Maia/Lite small packets"]
  Quick --> Deep["deeper standard gates<br/>visit curves + stronger anchors"]
  Deep --> Diagnostics["queen/material/blunder/regret diagnostics"]
  Diagnostics --> Decision{"promotion decision"}
  Decision -->|pass| Promote["promote/deploy/register accepted"]
  Decision -->|needs confidence| OpenBench["optional OpenBench<br/>high-sample candidate vs incumbent"]
  Decision -->|fail| Reject["reject/archive<br/>protocol-relative reason"]
  OpenBench --> Promote
```

Promotion claims should always include protocol context: model SHA/path, meta, search config, visits, openings, anchors, illegal counts, WDL, backend, and error bars.

## 8. ONNX/browser deployment flow

```mermaid
flowchart LR
  PT["PyTorch checkpoint"] --> Export["export ONNX + meta"]
  Export --> Check["onnx.checker + metadata audit"]
  Check --> Simplify["optional onnxsim deploy sibling"]
  Simplify --> Parity["semantic parity <= tolerance"]
  Parity --> Bench["latency/search benchmark<br/>WASM/native, batch/thread matrix"]
  Bench --> FE["frontend model registry"]
  FE --> Build["npm typecheck + client build"]
  Build --> Release["deployable release artifact"]
```

Quantization belongs in this flow only after FP32 parity and benchmark baselines are known. PTQ should be debugged and benchmarked before revisiting QAT.

## 9. Where Spark-like dataflow fits

```mermaid
flowchart TD
  BatchOut["many Batch shard outputs<br/>jsonl.zst + manifests"] --> Lake["S3 run prefix"]
  Lake --> Spark["Spark / Glue / EMR Serverless<br/>large-run curation"]
  Spark --> Checks["global validation<br/>dedupe, policy mass, schema, lane contamination"]
  Spark --> Compact["compaction<br/>jsonl.zst/parquet partitions"]
  Spark --> Splits["train/dev/test splits<br/>stratified sampling"]
  Spark --> Metrics["run_summary + dashboards"]
  Compact --> Consumers["training/cache builders"]
  Splits --> Consumers
```

Spark is a good fit for large data curation and validation, not for Gumbel search or model inference itself. For medium runs, DuckDB/Polars may be enough; for hundreds of millions of rows or multi-TB transforms, Spark/Glue/EMR becomes the right janitor layer.

## 10. Documentation backfill checklist

When adding or revising process docs, include:

- one overview diagram,
- exact command entrypoints,
- required inputs and outputs,
- provenance fields and manifests,
- validation gates,
- failure/repair policy,
- lane contamination rules,
- links to current scripts and canonical knowledge notes.

Recommended topic pages to backfill next:

1. `docs/model_architecture_atlas.md` — CNN, MoveFormer, SquareFormer/BT4, heads, encodings.
2. `docs/training_pipeline_atlas.md` — supervised, cache-based, SUP-SP, Zero.
3. `docs/cloud_pipeline_atlas.md` — AWS Batch, S3 layout, compression, Spark curation.
4. `docs/evaluation_promotion_atlas.md` — release gates, anchors, OpenBench, promotion policy.
5. `docs/deployment_runtime_atlas.md` — ONNX export, browser/runtime parity, quantization.
