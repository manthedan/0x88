# tiny_leela Expert Handoff: Current State and Open Questions

> **Status: Historical snapshot.** Useful for provenance, but it predates the current CNN / Tactical-MoveFormer / TinyBT portfolio and several completed queues. For current expert handoff, use [`expert_handoff_2026-05-current.md`](expert_handoff_2026-05-current.md). For general orientation, start with [`README.md`](README.md).

## Project goal

`tiny_leela` is an experimental small Leela-like chess engine intended to run in-browser / lightweight runtimes. The near-term goal is not grandmaster strength, but a compact model/search stack that plays plausible chess, can be evaluated reliably, and can be iteratively improved.

The current model can make some normal-looking moves, but is still very weak in play. Subjectively it may be around a few hundred Elo: better than random legal moves, but not human strength.

## Current architecture direction

The current main model is a trainable board-input CNN student:

- board/state input planes
- small convolutional trunk
- policy head over UCI move vocabulary
- WDL/value head
- Rust and TypeScript inference paths
- PUCT-style search around the evaluator

Current training target is supervised imitation from PGN-derived positions, with WDL/result labels.

The primary training metric tracked in autoresearch has been:

```text
dev_policy_top8 higher is better
```

Secondary metrics include:

```text
dev_policy_top1
dev_policy_top4
dev_policy_ce
dev_wdl_ce
rows
runtime
```

## Important artifacts

Current / recent model artifacts include:

```text
artifacts/student_distill_benchmark.json
artifacts/student_board_cnn_spatial_state_check_500k_32ch_20ep.json
artifacts/student_board_cnn_spatial_state_check_balanced_ft_25ep.json
artifacts/student_board_cnn_spatial_50k_8ep.json
artifacts/checkpoints/board_cnn_spatial_state_check_500k_32ch.pkl
artifacts/checkpoints/board_cnn_spatial_state_check_balanced_ft_25ep.pkl
```

The balanced fine-tune checkpoint was converted from pickle to playable JSON:

```text
artifacts/student_board_cnn_spatial_state_check_balanced_ft_25ep.json
```

Sanity check from the initial position with Rust evaluator produced legal output, e.g. `best_move=d2d4` and `policy_legal_count=20`.

## Training data currently involved

Relevant datasets:

```text
data/lichess_training_400k_2000elo_2016-01.jsonl
data/lichess_training_400k_2000elo.jsonl
data/lichess_training_50k.jsonl
data/tcec_training_100k.jsonl
data/balanced_finetune_300k.jsonl
```

The original mixed data included Lichess and TCEC PGN-derived rows. Rows are JSONL positions with:

- `id`
- `fen`
- one-hot `policy`
- `wdl`
- result/q metadata

Example TCEC row:

```json
{"id":"tcec_pgn_000000_008","fen":"r1bqkb1r/ppp1pppp/2np4/3nP3/3P4/5N2/PPP2PPP/RNBQKB1R w KQkq - 2 5","policy":{"c2c4":1},"wdl":[0,1,0]}
```

## Discovered data issue: opening bias

A major issue was discovered: the head of the TCEC dataset is heavily opening-biased because TCEC uses fixed openings. The first TCEC games include Alekhine's Defence and other repeated fixed lines.

Raw TCEC PGN head contains entries like:

```text
B04 Alekhine's defence modern variation
B04 Alekhine's defence modern, Larsen variation
A58 Benko gambit accepted
A58 Benko gambit accepted
E10 Blumenfeld counter-gambit
E10 Queen's pawn game
E11 Bogo-Indian defence, Nimzovich variation
E11 Bogo-Indian defence, Nimzovich variation
B12 Caro-Kann advance variation
B12 Caro-Kann advance variation
```

Grouping `data/tcec_training_100k.jsonl` by game id showed measurable repetition:

- about 988 games
- top first sampled position: 69 games, around 7.0%
- second most common first sampled position: 37 games, around 3.7%

The model was observed to repeat weird openings during testing, which may be explained by this data bias.

A write-up exists at:

```text
docs/opening-bias-and-finetuning.md
```

## Proposed / partially implemented data fixes

Recommended data fixes:

1. Shuffle by game, not row.
2. Cap rows per game.
3. Skip early TCEC plies, e.g. first 8-16 plies.
4. Deduplicate normalized FENs.
5. Balance or cap by ECO/opening family for TCEC.
6. Split train/dev by game id rather than row index.
7. Report opening concentration before training.

A balanced fine-tuning dataset was created:

```text
data/balanced_finetune_300k.jsonl
```

The balanced fine-tune checkpoint:

```text
artifacts/checkpoints/board_cnn_spatial_state_check_balanced_ft_25ep.pkl
```

was converted to:

```text
artifacts/student_board_cnn_spatial_state_check_balanced_ft_25ep.json
```

## Fine-tuning caveat

The trainer builds a policy move vocabulary from the training data. For safe fine-tuning, the policy head's move vocabulary must remain aligned with the checkpoint.

Current trainer should preserve `ck['moves']` when resuming, or otherwise ensure move vocabulary compatibility. If the move list changes between checkpoint and fine-tune data, the policy head can become misaligned.

This is a key thing to audit.

## Evaluation / Elo work

A CPU-parallel Rust arena framework has been started.

Added:

```text
scripts/elo_arena_parallel.mjs
eval/opening_suite_v1.fen
```

The Node coordinator spawns multiple Rust arena workers:

```text
rust/tiny_leela_core/src/bin/arena.rs
```

The Rust arena supports:

- candidate vs baseline
- paired openings
- `--start-game` sharding for parallelism
- fixed visits
- max plies
- terminal adjudication
- value adjudication
- Stockfish-at-max-ply adjudication

Package command:

```bash
npm run elo:arena -- --candidate=... --baseline=...
```

Example:

```bash
npm run elo:arena -- \
  --candidate=artifacts/student_board_cnn_spatial_state_check_500k_32ch_20ep.json \
  --baseline=artifacts/student_board_cnn_spatial_50k_8ep.json \
  --pairs=20 \
  --workers=8 \
  --visits=16 \
  --max-plies=120 \
  --adjudicate=terminal
```

A write-up exists at:

```text
docs/elo-evaluation-process.md
```

## Preliminary arena results

These are not reliable strength estimates yet; they are mostly framework validation.

### Short low-visit test

Candidate:

```text
artifacts/student_board_cnn_spatial_state_check_500k_32ch_20ep.json
```

Baseline:

```text
artifacts/student_board_cnn_spatial_50k_8ep.json
```

With 100 games, visits=2, max plies=20, value adjudication:

```text
W-D-L: 44-9-47
score: 48.5%
Elo diff: -10.4
CI half-width: ±66 Elo
```

This was considered too noisy and too dependent on weak value adjudication.

### Longer terminal-only test

40 games, visits=16, max plies=120, terminal adjudication:

```text
W-D-L: 4-35-1
score: 53.75%
Elo diff: +26.1
CI half-width: ±38 Elo
avg plies: 116.7
```

This suggests the 500k/20ep model may be stronger, but the result is very draw-heavy and still uncertain.

### Stockfish-at-max-ply test

Stockfish adjudication was added. Local Stockfish path:

```text
./.local_engines/stockfish_pkg/usr/games/stockfish
```

Example 20-game test, visits=8, max plies=60, Stockfish depth=6, draw threshold 50 cp:

```text
W-D-L: 1-19-0
score: 52.5%
Elo diff: +17.4
CI half-width: ±34 Elo
avg plies: 59.75
```

Still draw-heavy. Likely tuning needed:

```text
--stockfish-depth=8 or 10
--stockfish-draw-cp=20
--max-plies=80+
```

## Current concerns

### 1. Model may be too small too early

A key strategic question: did we try to make the model tiny before proving the training/evaluation pipeline?

Current view: probably yes, partially. The current tiny model learns some normal-looking chess behavior, but it is hard to tell whether weaknesses come from:

- architecture capacity
- data quality
- opening bias
- WDL label quality
- one-hot imitation target limitations
- train/dev leakage
- search/evaluator mismatch
- insufficient training

### 2. Need a medium reference model

Instead of jumping to full LCZero, a better next step may be to train a medium reference model:

```text
64-128 channels
4-8 residual blocks
board/state planes
policy head
WDL/value head
clean balanced dataset
game-level train/dev split
```

Purpose:

1. Prove the data/training/search pipeline can learn real chess.
2. Establish a stronger teacher/reference.
3. Later distill/compress into tiny/browser model.

This separates two questions:

```text
Can the pipeline learn chess?
Can the learned model be made tiny enough?
```

Currently those are entangled.

### 3. Supervised one-hot PGN imitation may be insufficient

The current policy target is generally a one-hot human/TCEC move. This may teach plausible moves but not robust tactical strength. Possible improvements:

- use stronger teacher policy distributions
- Stockfish/LC0 teacher labels
- multi-PV targets
- value targets from engine evals
- self-play / expert iteration once a baseline is stable
- distillation from a medium model

### 4. WDL/value head appears weak

Value adjudication was not very trustworthy. Terminal-only games are draw-heavy. Stockfish adjudication is now available and likely needed for evaluation until the value head improves.

### 5. Evaluation still immature

The Rust arena is promising, but more work is needed:

- better opening suite
- result JSONL with per-game final FEN/eval
- Stockfish process reuse instead of spawning per adjudication
- confidence intervals / SPRT
- calibration vs simple baselines
- compare against random/legal/material engines

## Questions for expert advice

1. Should we continue optimizing the tiny CNN, or first train a medium Leela-like reference model?
2. What is the smallest architecture likely to learn nontrivial chess from supervised PGN/teacher data?
3. Should we switch from one-hot PGN imitation to engine policy/value distillation?
4. How should we structure training data to avoid opening memorization while preserving useful opening knowledge?
5. What is a good evaluation ladder for a very weak engine?
6. How should max-ply adjudication be handled for weak self-play engines?
7. Is PUCT appropriate with such weak/noisy policy and value heads, or should early play use simpler search/eval?
8. What is the best compression path: train big then distill, or train tiny directly?
9. How much model capacity is needed before search starts helping rather than amplifying noise?
10. Should the value head be trained from game result, engine eval, or self-play outcomes?

## Suggested next steps

1. Audit the trainer resume path for move vocabulary correctness.
2. Confirm the balanced fine-tuned JSON is the active playable model and test subjectively.
3. Improve the arena to log per-game final FEN and Stockfish eval.
4. Tune Stockfish adjudication thresholds to reduce draw-heavy results.
5. Build a simple baseline ladder: random, material, old tiny models, current tiny models.
6. Create a clean game-level train/dev split and opening-balanced dataset.
7. Train a medium reference model to test whether the pipeline can reach clearly stronger play.
8. If the medium model works, distill/compress into the tiny/browser target.
