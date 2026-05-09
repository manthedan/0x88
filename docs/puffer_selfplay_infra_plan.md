# PufferLib-inspired self-play infrastructure plan

Purpose: translate PufferLib/PuffeRL/Ocean/Protein ideas into the Tiny Leela AlphaZero/lc0-style self-play roadmap. PufferLib is not the algorithmic core for chess, but it is a strong systems reference for actor throughput, static memory, correctness environments, and cost-aware sweeps.

## PufferLib + MiniZero ideas worth borrowing

From the PufferLib docs/blog:

- Treat simulator/actor throughput as a first-class RL objective.
- Use static memory: allocate buffers up front, avoid tensor/object churn in hot paths.
- Vectorize many environments/rollouts and batch transfers/inference.
- Use tiny correctness environments to catch RL plumbing bugs early.
- Make sweeps the unit of compute; Protein/CARBS-style tuning models a score/cost Pareto frontier.
- Muon + cosine schedules + segment prioritization are promising, but should be tested against AdamW under project-specific gates.

From MiniZero's AlphaZero/MuZero/Gumbel Zero stack:

- Keep a clean topology: coordinator/server, self-play actors, optimization worker, replay/data storage, explicit model iterations.
- Let each actor maintain many games/MCTS instances and batch NN leaf evaluations across them.
- Separate training self-play config from evaluation/fight config; eval should disable stochastic root noise and use deterministic action choice.
- Use iteration-window replay buffers, with optional prioritized experience replay for surprising/high-error games or positions later.
- Add root exploration modes as explicit config: Dirichlet noise, temperature schedules, and a low-visit Gumbel-root experiment.
- Add resign calibration via a disable-resign fraction so early false resignations are detected rather than baked into data.
- Provide an `env_test`-style correctness mode: random legal game, serialize record, replay from record, assert final state and legal/action invariants.

## Do not copy directly

Do not make PPO the main chess learning algorithm. Tiny Leela should remain AlphaZero/lc0-style:

```text
model policy/value -> PUCT/search -> visit/WDL/Q targets -> supervised update -> promotion gate
```

PufferLib's main value here is actor systems design, not replacing search-improved policy training.

## Current repo gap

Current self-play support is useful for smoke tests but not enough for serious self-play scaling:

```text
scripts/selfplay_generate.mjs
```

It generates games sequentially, writes JSONL rows, and does not yet have:

- many games/searches in flight;
- batched neural inference across leaves;
- preallocated MCTS node/edge pools;
- WAL + resumable chunk protocol;
- compact binary/zstd chunk records;
- explicit model/config hashes on every chunk;
- separate self-play vs eval/fight search configs;
- resign-disable calibration and false-resign accounting;
- optional low-visit Gumbel-root target generation;
- hot model/config reload for long-lived workers;
- actor throughput metrics as first-class outputs;
- self-play chunk validator and gatekeeper integration.

## Target worker: ChessVecWorker

Build a local-first vectorized actor before cloud distribution.

```text
ChessVecWorker
  GameState[N]
  SearchTree[N, max_nodes]
  EvalQueue[batch]
  EvalOutput[batch]
  ChunkBuffer[records]
  WAL per active game
```

MiniZero-style hot loop:

```text
1. Maintain many games/searches in flight.
2. For each active search, CPU-select one or more PUCT/Gumbel leaves.
3. Enqueue leaf positions into a shared inference batch.
4. Run one ONNX/Torch batch eval.
5. Expand leaves and back up WDL/Q with virtual-loss protection when parallel.
6. When root visits complete, select a move using the self-play temperature/noise config.
7. Apply resign logic, with a configured fraction of games forced to ignore resignation.
8. Append a compact record to chunk buffer + WAL.
9. Atomically finalize chunk and manifest.
```

Primary metrics:

```text
games/hour
positions/hour
nodes/sec
evals/sec
mean/p95 eval batch size
CPU search utilization
GPU inference utilization
chunk write MB/sec
usable positions/dollar
```

## Static PUCT memory layout

Prefer struct-of-arrays typed buffers:

```text
node_first_edge: int32[max_nodes]
node_edge_count: uint16[max_nodes]
node_parent_edge: int32[max_nodes]
node_zobrist: uint64[max_nodes]
edge_move: uint16[max_edges]
edge_prior: float32[max_edges]
edge_q_sum: float32[max_edges]
edge_visits: uint32[max_edges]
edge_child: int32[max_edges]
```

Avoid per-node JS/Python objects in the hot path.

## Chunk schema v0

Start with compressed JSONL for debuggability, then migrate the payload to msgpack/npz/arrow-like binary once stable.

Required per-position fields:

```json
{
  "schema": "tiny-leela.selfplay.v0",
  "run_id": "sp_local_0001",
  "net_id": "cnn96x8_100m_e08",
  "net_sha256": "...",
  "model_iteration": 8,
  "config_hash": "...",
  "search_config_id": "puct64_train_noise_v1",
  "game_id": "...",
  "ply": 42,
  "fen": "...",
  "history_fens": ["..."],
  "legal_moves": ["e2e4", "d2d4"],
  "raw_policy": {"e2e4": 0.21},
  "search_visits": {"e2e4": 37},
  "search_q": {"e2e4": 0.08},
  "root_wdl": [0.38, 0.42, 0.20],
  "played_move": "e2e4",
  "pv": ["e2e4", "c7c5"],
  "nodes": 64,
  "temperature": 1.0,
  "root_exploration": {"kind": "dirichlet", "alpha": 0.3, "weight": 0.25},
  "gumbel_root": {"enabled": false, "top_k": 0, "sequential_halving": false},
  "resign_disabled": false,
  "resign_threshold": -0.9,
  "opening_id": "uho_lite:0007",
  "seed": 12345,
  "result": [1, 0, 0]
}
```

## ChessOcean sanity suite

Before any serious self-play run, execute deterministic micro-environments/tests:

```text
legal_move_roundtrip
promotion_roundtrip
en_passant_roundtrip
castling_roundtrip
side_to_move_flip
value_perspective_flip
mate_in_one
stalemate_terminal
queen_can_be_captured
sound_queen_sac
repetition_claim
root_policy_equals_puct_prior
puct_backup_sign
puct_tie_break
onnx_batch_vs_single_parity
browser_node_policy_parity
```

Add a MiniZero-style `env_test` mode:

```text
random legal game to terminal or max plies
serialize FEN/action/result record
replay record from start/opening FEN
assert final FEN/result matches
assert every replayed move was legal at its ply
assert policy action ids roundtrip to UCI at every ply
run with fixed seeds in CI and randomized seeds before long jobs
```

These should fail loudly before expensive jobs create bad data.

## Muon optimizer plan

Treat Muon as an experiment, not a default replacement.

Initial test matrix:

```text
baseline: AdamW all parameters
hybrid:   Muon for 2D matrix weights, AdamW for embeddings/norms/biases/heads
schedule: cosine with warmup for both
models:   small SquareFormer/CNN smoke first, then MF/BT only if stable
```

Metrics:

```text
policy CE/KL
WDL calibration
action-value rank/tau/regret
NaN/instability rate
policy-only arena
PUCT/AV-rerank arena
latency/parameter efficiency
```

Do not use Muon for QAT yet; QAT is paused until quality gates improve.

## Protein/CARBS-style sweep plan

Use a tiny repo-local sweep ledger first; do not depend on PufferLib internals.

Each trial records:

```json
{
  "trial_id": "...",
  "params": {"lr": 0.001, "cpuct": 1.5},
  "score": 0.0,
  "cost": {"wall_seconds": 0, "gpu_hours": 0, "positions": 0},
  "status": "succeeded|failed|oom|rejected",
  "artifacts": ["..."]
}
```

Candidate objectives:

```text
score = quick_gate_score + small_arena_elo - latency_penalty - catastrophic_blunder_penalty
cost  = wall_seconds or gpu_hours + selfplay_worker_hours
```

Use this first for search/data/optimizer sweeps, then architecture sweeps.

## First implementation sprint

1. Add a self-play chunk schema validator.
2. Add `scripts/selfplay_chunk_validate.py` for JSONL/zst chunks.
3. Add a WAL/atomic chunk writer wrapper around `scripts/selfplay_generate.mjs`.
4. Add model/config hash fields plus train/eval search config IDs to chunk metadata.
5. Add local throughput metrics and protocol cards.
6. Add ChessOcean tests to `tests/` for PUCT/value/policy plumbing.
7. Add MiniZero-style random-game serialize/replay `env_test` coverage.
8. Prototype batched inference/search with many games in flight.
9. Add resign-disable accounting and false-resign metrics.
10. Prototype a low-visit Gumbel-root mode behind an experimental flag.
11. Add Muon as an opt-in optimizer in one small training script.
12. Add a simple sweep ledger and Pareto report.

## Promotion discipline

Self-play data should only be generated by accepted models unless explicitly doing candidate diagnostics. Candidate models enter the actor pool only after:

```text
protocol checks pass
chunk schema validates
policy/value gates pass
anchor arena beats current accepted net
no catastrophic tactical/queen regression
```
