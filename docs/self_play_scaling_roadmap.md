# Self-Play Scaling Roadmap

Goal: turn the supervised tiny-Leela models into an lc0-style policy-improvement loop while preserving the project goal: strongest lightweight/browser neural chess engine per millisecond, not pure searchlessness.

A major scaling route is **distributed burst compute**: keep `yukon` as the scheduler/trainer/promoter, then launch many stateless container workers for horizontally scalable self-play, Stockfish/lc0/Reckless labeling, reanalysis, and evaluation jobs.

## Core thesis

Self-play is feasible for the SquareFormer/CNN tracks if every promoted model preserves the AlphaZero interface:

```text
position -> legal policy + WDL/value
policy + value -> PUCT/search -> improved policy/value targets -> next network
```

The deployed model may often play search-light, but the training loop should use search/reanalysis to create stronger labels than the raw model can produce.

## Recommended architecture target

Primary target: `SquareFormer-AV-PUCT`.

```text
64 square tokens
+ chess-aware attention
+ from-to legal policy
+ WDL/value bucket heads
+ action-value top-k reranker
+ uncertainty/value-error head
+ optional multi-ply auxiliary heads
+ conditional small PUCT fallback
```

CNNs remain important as strong baselines and possible actors/students, but SquareFormer is the preferred self-play research architecture because its policy/action-value heads map naturally to legal move ranking and search statistics.

## Phase 0: supervised/distilled warm start

Do not start from random.

Inputs:

```text
100M supervised human/elite dataset
Stockfish/lc0 teacher labels on selected positions
puzzle/tactical positions
tablebase/endgame positions
current model failure positions
```

Initial losses:

```text
policy CE/KL
WDL CE
value bucket CE
action-value/ranking loss when labels exist
```

Exit criteria:

- model is legal and coherent in policy-only mode;
- PUCT with the model beats policy-only;
- anchor benchmarks and queen-blunder diagnostics are stable enough to detect regressions.

## Phase 1: shallow search-improved self-play

Actors:

```text
current promoted model
32-node or 64-node PUCT
varied/unbalanced openings
root temperature/noise in early plies
fixed deterministic seeds
```

Store per move:

```json
{
  "run_id": "sp_0001",
  "net_id": "model_id",
  "fen": "...",
  "history_fens": ["..."],
  "legal_moves": ["e2e4", "d2d4"],
  "raw_policy": {"e2e4": 0.21},
  "search_visits": {"e2e4": 37},
  "search_q": {"e2e4": 0.08},
  "root_wdl": [0.38, 0.42, 0.20],
  "played_move": "e2e4",
  "pv": ["e2e4", "c7c5"],
  "policy_entropy": 2.31,
  "uncertainty": 0.42,
  "nodes": 64,
  "temperature": 1.0,
  "opening_id": "uho_lite:0007",
  "seed": 12345
}
```

Training target:

```text
policy_target = normalized search visits
value_target = final game WDL mixed with root WDL
action_value_target = per-move backed-up Q for visited/top-k moves
```

Exit criterion: candidate checkpoint beats previous accepted net in fixed-opening, fixed-node matches.

## Phase 2: action-value self-play distillation

Use PUCT root stats to train an action-value head:

```text
(position, move) -> Q(move) / value bucket / regret bucket
```

Label only a compact candidate set:

```text
PUCT top-k moves
raw-policy top-k moves
checks/captures/promotions
played move
random legal distractors
moves where search overturns policy
```

Runtime goal:

```text
policy top-k -> action-value rerank
```

Success criterion: same Elo with fewer PUCT nodes, or more Elo at same nodes.

## Phase 3: uncertainty-gated conditional search

Train uncertainty from:

```text
policy/search disagreement
value swing during search
shallow-vs-deep disagreement
teacher disagreement
actual regret/blunder labels
failed puzzle/queen-hang positions
```

Runtime ladder:

```text
low uncertainty: policy or action-value rerank
medium uncertainty: top-k child value verification
high uncertainty: 32-128 node PUCT
endgame: tablebase if available
```

Measure:

```text
Elo per millisecond
fallback frequency
fallback benefit vs cost
uncertainty calibration vs actual regret
```

## Phase 4: targeted reanalysis

Mine positions from self-play, anchor losses, and puzzle failures:

```text
high entropy
large search policy shift
material/queen blunders
losses from favorable positions
rare endgames
tactical/check-heavy positions
teacher/model disagreement
```

Reanalyze with:

```text
deeper student PUCT
Stockfish MultiPV
lc0 policy/value
Syzygy tablebase
Maia only for human-style variants
```

Sampling mix for training:

```text
50% recent accepted-net self-play
20% hard mined/reanalyzed positions
10% teacher-labeled static positions
10% tablebase/endgame curriculum
10% openings/puzzles/tactical curated data
```

## Phase 5: actor/student split

Once local self-play works:

```text
actor: larger model and/or deeper PUCT generates high-quality data
student: smaller browser model distills policy, WDL, action-values, uncertainty
```

This lets training use more compute than deployment.

## Distributed burst compute architecture

This project is a strong fit for bursty distributed workers because most expensive data-generation tasks are embarrassingly parallel:

```text
self-play games
Stockfish/lc0/Reckless labeling
candidate move action-value labels
puzzle/tactical reanalysis
anchor gauntlets
hard-position mining
```

Do not distribute training first. Keep training centralized on `yukon`; distribute data generation and labeling.

```text
yukon:
  scheduler
  dataset/job registry
  artifact manifest writer
  trainer
  evaluator/promoter

container workers:
  stateless actors
  pull jobs
  download model/input shards
  run self-play or labeling
  upload immutable result shards
  exit or request another job

object storage / shared storage:
  model artifacts
  job manifests
  input shards
  output shards
  reports
```

### Worker job families

#### Self-play actors

Each worker receives a model, openings, search config, and seed range:

```json
{
  "job_type": "selfplay",
  "run_id": "sp_0007",
  "net_id": "chessformer_v1_100m_e3",
  "model_uri": "r2://tiny-leela/models/chessformer_v1.onnx",
  "opening_shard": "r2://tiny-leela/openings/uho_lite/shard_0004.jsonl.zst",
  "games": 128,
  "search": { "nodes": 64, "cpuct": 1.5 },
  "seed": 123456
}
```

Outputs:

```text
selfplay/<run_id>/results/<job_id>_<worker_id>.jsonl.zst
```

Rows include raw policy, search visits, Q, WDL, PV, played move, result, entropy, uncertainty, and metadata.

#### Teacher labeling workers

Workers label independent position shards with specialist engines:

```json
{
  "job_type": "label",
  "teacher": "stockfish|lc0|reckless|syzygy",
  "input_positions": "r2://tiny-leela/positions/hard/shard_0042.jsonl.zst",
  "multipv": 8,
  "depth": 12,
  "nodes": 50000,
  "output_uri": "r2://tiny-leela/labels/run_001/shard_0042.jsonl.zst"
}
```

Use cases:

```text
Stockfish MultiPV policy/Q labels
lc0 policy/WDL labels
Reckless tactical labels
Syzygy exact endgame labels
candidate-move action-value/regret labels
```

#### Reanalysis workers

Input is mined hard positions:

```text
queen/material blunders
policy/search disagreement
high uncertainty
anchor losses
failed puzzle motifs
rare endgames
```

Workers run deeper or specialist analysis and return rich labels for replay-buffer upweighting.

#### Evaluation workers

Run large candidate-vs-incumbent matches in parallel:

```text
fixed openings
reversed pairs
fixed nodes/search config
WDL/result shard output
```

`yukon` aggregates WDL, Elo estimates, confidence intervals, and pentanomial stats.

### Queue and storage model

Initial robust design:

```text
object store / shared root:
  jobs/pending/
  jobs/claimed/
  jobs/done/
  jobs/failed/
  artifacts/models/
  datasets/selfplay/
  datasets/labels/
  reports/worker-heartbeats/
```

If object-store claim semantics are awkward, use a tiny `yukon` coordinator API:

```text
POST /claim
POST /heartbeat
POST /complete
POST /fail
```

Then R2/S3/local shared disk only stores artifacts and result shards.

Workers should be pull-based:

```text
boot -> claim job -> run -> upload result -> complete -> claim next job or exit
```

This works well for preemptible/spot/burst instances.

### Container image

One flexible worker image is enough initially:

```text
tiny-leela-worker:
  project code
  node + npm deps
  python env
  onnxruntime
  Stockfish
  lc0
  Reckless
  zstd
  optional Syzygy mount
```

Later split CPU and GPU images:

```text
CPU workers: Stockfish/Reckless/self-play without GPU
GPU workers: lc0 labeling and neural self-play
```

### Validation and merge

`yukon` must validate every shard before adding it to a training manifest:

```text
schema valid
net_id/model hash matches job
teacher/search metadata present
legal moves parse
policy/visit distributions normalize
no impossible FEN/move pairs
row counts match manifest
no duplicate job ids unless explicitly deduped
```

Output shards are immutable. Merged datasets are manifests over immutable shards, not rewritten monoliths.

### Fault tolerance rules

- write complete local temp output, then upload final `.jsonl.zst` shard;
- include `job_id`, `worker_id`, `seed`, `net_id`, `teacher_id`, backend, and search config in every row or shard manifest;
- heartbeat during long jobs;
- requeue stale claimed jobs after timeout;
- tolerate duplicate completed jobs by deterministic job ids and validation dedupe;
- record partial/interrupted games instead of silently discarding them;
- never train on worker output until validation passes.

### Cloud/provider fit

Good worker targets:

```text
CPU spot/preemptible: Stockfish/Reckless labeling, many shallow games
GPU spot/burst: lc0 labeling, neural self-play if GPU runtime is needed
local LAN/Tailscale machines: trusted always-on workers
```

Candidate providers:

```text
RunPod/Vast/Lambda for GPU bursts
Hetzner/AWS/GCP/Oracle/Fly for CPU bursts
Kubernetes later if worker count and ops complexity justify it
```

Object storage such as Cloudflare R2 is useful for job/result/model exchange, but training should still read from local validated caches on `yukon`.

### Distributed MVP phases

```text
Phase D0: local protocol
  scheduler + 4-8 local workers on yukon
  job types: selfplay, stockfish_label
  local result directory

Phase D1: container worker
  Dockerfile.worker
  pull job, run, upload local/shared result
  heartbeat and stale-job recovery

Phase D2: object storage backend
  model/input/result URIs
  immutable manifests
  validation/merge on yukon

Phase D3: burst launch
  provider-specific launcher for 10-100+ workers
  separate CPU/GPU job queues
  cost/accounting reports

Phase D4: promotion-integrated loop
  accepted net -> create self-play jobs
  validate/merge results
  train candidate
  distributed evaluation gate
  promote or reject
```

### Near-term distributed checklist

- [ ] Define `Job` schema for self-play, labeling, reanalysis, and eval jobs.
- [ ] Define result shard manifest schema.
- [ ] Implement `scripts/distributed/scheduler.*` on `yukon`.
- [ ] Implement `scripts/distributed/worker.*` with pull/heartbeat/complete/fail.
- [ ] Add Stockfish/Reckless label job type.
- [ ] Add self-play job type using current ONNX evaluator/PUCT.
- [ ] Add shard validator and manifest merger.
- [ ] Add Dockerfile for worker image.
- [ ] Add R2/S3 URI support or coordinator-mediated upload/download.
- [ ] Add provider launcher for burst workers.

## Fault-tolerant rollout infrastructure

Use write-ahead logs. After every move, append a complete JSON object. On restart, resume from last complete move or record the partial game explicitly.

Do not silently discard interrupted games; that biases toward short/easy games.

Required metadata:

```text
net id
artifact hash
search config
opening id
seed
worker id
backend
nodes/time budget
temperature/noise config
partial/discard reason if any
```

## Exploration policy

Use diverse starts rather than always starting from the initial position:

```text
UHO/unbalanced openings
fixed varied opening suites
known tactical motifs
endgame tablebase starts
positions where previous models blundered
high-disagreement positions
```

Root exploration:

```text
early opening: higher temperature/noise
middlegame: moderate exploration
endgame: low exploration/tablebase when possible
```

## Promotion gates

Do not let every checkpoint generate future data.

Quick gate:

```text
400-1000 games
fixed openings
same nodes/search settings
both colors
WDL + Elo CI + pentanomial stats
```

Serious gate:

```text
5000+ games
multiple node budgets
Stockfish/Maia anchor spot checks
puzzle/endgame/queen-blunder diagnostics
browser latency check
```

Promote only if candidate improves primary strength or efficiency without major regressions.

## Scaling axes, in order

1. Better data: hard-position mining, reanalysis, tablebases, replay buffer quality.
2. Better targets: more useful PUCT stats, deeper search for hard positions, action-value labels.
3. More capacity: d128->192->256, layers 6->8->12, richer relation attention.
4. Deployment compression: FP16 WebGPU, INT8/PTQ/QAT, distill actor into browser student.

## Near-term implementation checklist

- [ ] Add self-play WAL schema and writer.
- [ ] Add resume/accounting for interrupted games.
- [ ] Store root visit distributions, Q, WDL, PV, entropy, uncertainty.
- [ ] Add candidate extraction for action-value labels.
- [ ] Add training loader for self-play visit-policy JSONL.
- [ ] Add action-value head to SquareFormer.
- [ ] Add uncertainty/value-error head.
- [ ] Add promotion gate script: candidate vs incumbent, fixed openings, fixed nodes.
- [ ] Add hard-position miner from anchor/blunder logs.
- [ ] Add targeted Stockfish/lc0/tablebase reanalysis queue.

## Non-goals for first self-play loop

- random-network AlphaZero training from scratch;
- massive distributed RL before local promotion gates work;
- always-heavy PUCT as the deployment default;
- throwing away supervised/teacher anchors;
- promoting models without fixed-opening match evidence.
