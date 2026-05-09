# Maia-inspired roadmap for tiny Leela

> **Status: Superseded / reference.** The supervised-human-policy lesson remains important, but current Maia/lc0 work is tracked through [`lc0_maia_gap_closure_roadmap.md`](lc0_maia_gap_closure_roadmap.md), evaluation anchor docs, and current queue artifacts.

This note summarizes ideas from `CSSLab/maia-chess`, `CSSLab/maia2`, and how they should change the project roadmap.

## Core lesson

Maia is the strongest precedent for this project if self-play is too weak: a Leela-shaped chess net can be useful when trained primarily as a supervised human-move predictor. Search can be disabled or added later.

For tiny Leela, the main near-term objective should be:

```text
Predict strong human moves extremely well first.
Only then tune search/self-play.
```

The current project evidence agrees:

- pure self-play degraded the supervised incumbent,
- Stockfish aux improved WDL/top-k but damaged policy calibration when pushed too hard,
- PUCT did not improve the tiny tactics suite over policy-only.

So the optimal path is supervised scaling + better policy/value targets, not RL-first.

## Repo findings from Maia v1

Useful concrete details from `CSSLab/maia-chess`:

### 1. Nodes=1 is intentional

Maia weights are loaded into lc0 and run with:

```text
go nodes 1
```

This means Maia primarily benchmarks the network policy itself, not search. For us this implies a required eval ladder:

```text
policy-only argmax
policy stochastic / temperature
top-k value rerank
small PUCT
browser strength mode
```

Policy-only should become a first-class benchmark.

### 2. Final Maia v1 model scale

`move_prediction/maia_config.yaml` uses:

```yaml
model:
  filters: 64
  residual_blocks: 6
  se_ratio: 8

training:
  precision: half
  batch_size: 1024
  total_steps: 400000
  shuffle_size: 250000
  lr_values: [0.1, 0.01, 0.001, 0.0001]
  lr_boundaries: [80000, 200000, 360000]
  policy_loss_weight: 1.0
  value_loss_weight: 1.0
```

This validates `64x6` as a serious tiny-Leela baseline size. Our `48x5` is a good incumbent, but the next model-size baseline should include `64x6` on a larger dataset.

### 3. Leela chunk pipeline and sampling

Maia uses Leela training chunks via `pgn-extract` + `trainingdata-tool`. Its `train_maia.py` loads chunk files, shuffles chunks, uses a shuffle buffer, and samples with `SKIP = 32` in the parser.

Takeaway for our 100M plan:

- shard/chunk the dataset,
- train from a manifest/stream, not one giant JSONL,
- use a large shuffle buffer or shuffled shard order,
- avoid materializing too many duplicate files.

### 4. Rating-band data construction

Maia v1 trained separate rating-band models. The replication script selects games by Elo/year and keeps a test split from held-out 2019 data. The extractor can remove bullet and low-time moves.

Takeaway:

- fixed held-out dev/test by month/source matters,
- we should report metrics by rating band,
- avoid bullet/hyperbullet and low-clock noise for the main baseline.

## Repo findings from Maia blunder prediction

The `blunder_prediction` tree has several ideas directly relevant to our Stockfish work.

### 1. Winrate-loss target

Configs include `winrate_loss` and `is_blunder_wr`. This is better than raw centipawn loss for human modeling.

Current project used:

```text
stockfish_q = tanh(cp_best / 400)
```

Better next target:

```text
best_winrate   = cp_to_winrate(cp_best)
played_winrate = cp_to_winrate(cp_played)
winrate_loss   = best_winrate - played_winrate
```

Use winrate loss for:

- blunder/error/inaccuracy classification,
- move-quality buckets,
- sample quality estimates,
- teaching UI.

### 2. Extra scalar inputs

Maia blunder configs include optional scalar inputs:

```yaml
inputs:
  - cp_rel
  - clock_percent
  - move_ply
  - opponent_elo
  - active_elo
```

They inject scalars either as channels or near the top of the model.

For us:

- active/opponent Elo should be kept as evaluation metadata, not model conditioning,
- move ply/game phase can be useful as metadata and maybe as a future non-Elo input,
- clock/time control can be useful for dataset filtering/reporting,
- cp-related inputs should not be required at browser inference unless the mode is analysis-only.

## Repo findings from Maia-2

`CSSLab/maia2` is even more actionable for our next architecture.

### 1. Current-board-only input

Maia-2 uses 18 channels:

```text
12 piece planes
side to move
4 castling-right planes
en-passant plane
```

This is a major simplification over lc0-style 112 planes. It helps browser deployment and puzzle/FEN inference.

For us, create an experimental current-board-only line:

```text
input_planes: 18
history_plies: 0
```

Compare it against current 46-plane history2 representation. If close, it becomes the preferred 100M baseline because storage/input bandwidth drop a lot.

### 2. Elo is useful for evaluation, not architecture

Maia-2 conditions on active/opponent Elo, but this project is not trying to play at many human skill levels. Do **not** feed Elo to the browser model and do **not** train separate Elo models.

Keep Elo only as metadata for:

- dev/test breakdowns,
- source-quality diagnostics,
- dataset balancing/capping,
- checking whether the model overfits one player-strength band.

### 3. Side-info auxiliary head

Maia-2 predicts side information in addition to the move:

```text
legal move mask
moved piece type
captured piece type
check flag
from square
to square
```

The code concatenates:

```text
6 moved-piece labels
6 captured-piece labels
1 check flag
64 from-square labels
64 to-square labels
legal move mask over policy moves
```

This is very useful for tiny models. It provides dense chess-structure supervision without requiring Stockfish.

Recommended auxiliary losses:

```text
policy_ce
wdl/value loss
legal_mask_bce, low weight
from_square_ce
to_square_ce
moved_piece_ce
captured_piece_bce/ce
check_bce
optional winrate_loss/blunder head
```

At inference, these heads can be ignored or used for analysis UI.

### 4. Streaming monthly PGN training

Maia-2 decompresses monthly `.pgn.zst`, chunks it, preprocesses chunks in CPU processes, trains, checkpoints, then deletes decompressed PGN.

This is exactly the right pattern for a 100M dataset on our disk:

```text
raw .pgn.zst
→ chunk/preprocess month
→ train/build shard
→ delete temporary expanded data
```

Do not store 100M as multiple duplicated giant JSONLs if avoidable.

## Revised 100M baseline plan

### Dataset target

```text
100M supervised positions
standard chess only
prefer rapid/classical-like quality first
exclude bullet/hyperbullet
skip first 10 plies
skip low-clock moves where clock exists
balance/cap by month/source/rating bucket
fixed dev: 1M positions
fixed tactical/search suites
```

If rapid-only does not provide enough from available files, use high-quality blitz/standard carefully, but keep time-control metadata.

### Storage format

Use shards from the beginning:

```text
data/datasets/maia_100m_v1/
  manifest.json
  train/shard_000.jsonl.zst or binary
  cache_h2_or_h0/shard_000/
  dev/dev_1m.jsonl.zst
  stockfish_aux/sidecars/...
```

For 100M, current 46-plane cache is about 300GB. Current-board-only 18-plane cache would be roughly 40% of that, so it is highly attractive.

### Stockfish labels

Do not annotate all 100M initially.

Recommended:

```text
5M-10M representative Stockfish labels
16 workers
depth 8
MultiPV 4
```

Use for:

- value aux,
- winrate-loss/blunder aux,
- bucket/source quality estimation,
- tactical validation subsets.

### Model ladder on same 100M baseline

Train/evaluate on the same dataset:

```text
48x5 current-style baseline
64x6 Maia-like baseline
64x6 current-board-only baseline
64x6 rating-conditioned baseline
```

Only then consider larger models.

## Concrete next implementation ideas

1. Add policy-only eval command:

```text
policy top1/top3/top5 by rating bucket
policy CE/perplexity
nodes=1 browser mode
```

2. Add current-board-only feature builder/trainer:

```text
18 input channels
no history requirement
smaller cache
better for arbitrary FENs
```

3. Preserve rating/time metadata for reports only:

```text
active_elo_bucket
opponent_elo_bucket
time_control bucket
move_ply
```

These should not be model inputs for the main Tiny Leela line.

4. Add side-info aux cache fields:

```text
legal mask maybe sparse/top-level
from_square
to_square
moved_piece
captured_piece
check flag
```

5. Convert cp to winrate loss rather than raw `tanh(cp/400)` for blunder/mistake heads.

6. Add stochastic policy play modes:

```text
argmax
sample temperature
top-k/top-p
```

## Recommended priority order

1. Finish evaluating current Stockfish full-aux candidate, but do not promote if CE is bad.
2. Build policy-only evaluation and report by rating/source/phase.
3. Build the 100M sharded supervised dataset.
4. Train a `64x6` Maia-like supervised baseline on 100M.
5. Add current-board-only 18-plane path and compare storage/performance.
6. Add side-info and winrate-loss heads.
7. Revisit PUCT only after policy-only is strong.

## Bottom line

The best roadmap is now:

```text
Maia-style supervised scaling first,
Stockfish as sparse tactical/value annotation,
Elo only for evaluation/reporting,
search/self-play last.
```

That is more realistic, more measurable, and better aligned with the browser toy/training-partner direction than trying to bootstrap AlphaZero-style self-play from a weak engine.
