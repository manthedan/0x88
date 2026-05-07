# 100M Scaling + Anti-Overfitting Training Plan

Goal: scale from the current `supervised_10m_elite_tcec_v1` setup to a disciplined 100M supervised dataset, then train stronger browser-sized Leela/Maia-class models with overfitting guards, annealing, and promotion gates.

## Current baseline

Current 10M artifacts:

```text
data/datasets/supervised_10m_elite_tcec_v1
train rows: 10,000,000
dev rows:     500,000
cache: data/datasets/supervised_10m_elite_tcec_v1/cache_h2_state
input planes: 46
```

Current promising models:

```text
48x5 10M e6:          ~6.28M params, best first arena result
64x6 10M e9:          ~8.53M params, good CE/top8
80x5 hybrid 10M e9:  ~10.85M params, best 10M dev CE/top8 so far
```

Disk status at plan creation:

```text
free: ~1.5T
10M dataset+cache: ~30G
```

100M is feasible if we avoid uncompressed intermediates and keep generated corpora/artifacts out of git.

## Main principle

Do **not** jump straight to expensive 100M training until the trainer and gates can tell us whether the result is genuinely better.

Execution order:

```text
1. Improve trainer: scheduler, EMA, regularization, richer checkpoints.
2. Freeze fixed dev/gate sets.
3. Run better long training on 10M.
4. Improve arena/gating.
5. Build 100M dataset/cache.
6. Train on 100M until guards trip.
```

## Trainer improvements before long runs

### 1. LR annealing

Add scheduler support to `training/train_residual_aux_multicache_torch.py`:

- `--lr-schedule constant|cosine|onecycle|step`
- `--warmup-steps N`
- `--min-lr X`
- `--total-steps` or infer from epochs × shards

Suggested schedules:

```text
48x5 continuation: 1e-5 -> 1e-6 cosine
64x6/80x5 from scratch: warmup to 1e-4, cosine to 1e-5 or 3e-6
```

Reason: constant LR is okay for short smokes but risky for long 10M/100M runs.

### 2. EMA weights

Add exponential moving average:

- `--ema-decay 0.999` / `0.9995` / `0.9999`
- save raw best and EMA best
- evaluate both at epoch end

Artifacts:

```text
*.best.pt
*.ema.best.pt
*.ckpt.pt
```

Reason: EMA often improves stability and play quality, especially when supervised CE is noisy.

### 3. Weight decay

Expose `--weight-decay`; default should be small but nonzero:

```text
AdamW weight_decay: 1e-4 or 5e-5
```

Try grid:

```text
0
1e-5
5e-5
1e-4
```

Reason: this is the safest first regularizer for residual conv nets.

### 4. Label smoothing / soft policy smoothing

Add optional policy label smoothing:

```text
--policy-label-smoothing 0.0|0.01|0.02|0.05
```

Use very small values only. Human move prediction is inherently multi-modal, but over-smoothing can harm sharp play and PUCT.

Suggested first tests:

```text
0.01 for 80x5 hybrid
0.00 for 48x5 continuation unless overfitting appears
```

### 5. Gradient clipping

Add:

```text
--grad-clip-norm 1.0
```

Reason: cheap safety for long runs and larger models.

### 6. Dropout / stochastic depth

Be cautious. Standard LC0-style conv residual nets usually do not rely on dropout.

Possible experiments:

```text
--policy-dropout 0.05 only before final policy linear
--stochastic-depth 0.02..0.05 for deeper towers
```

Do not enable by default. Test only if train/dev divergence appears.

### 7. Data augmentation by board mirroring

Potentially valuable but requires correct policy remapping.

Candidate augmentation:

- horizontal mirror (`a` ↔ `h`) is chess-legal if castling/en-passant/move policy are remapped correctly.
- color flip is more complex because side-to-move and WDL must be transformed carefully.

Do not add until there is a robust move/FEN/policy transform test suite.

### 8. Source-balanced sampling

For 100M, avoid letting one source/month dominate.

Add or simulate sampling controls:

```text
--source-temperature
--source-weights elite=1.0,lichess2200=0.4,tcec=0.2
--max-rows-per-source/month caps
```

Reason: reduces month/opening/source overfit.

### 9. Phase-aware monitoring

Training can improve average CE while degrading openings/tactics/endgames.

Keep split metrics for:

```text
opening
middlegame
endgame
Elite
broader Lichess 2200+
TCEC
high duplication/opening proxy buckets
```

## Fixed dev/gate sets

Create and freeze:

```text
data/datasets/dev_250k_v1
data/datasets/dev_1m_v1
```

Properties:

- whole-game split
- source labels preserved
- phase labels reportable
- no train-game overlap
- standard chess only
- no bullet/hyperbullet by default
- `--skip-plies 10`
- history planes preserved

Use these for all future 10M/100M comparisons.

## Better evaluation gates

### Supervised gates

Track:

```text
policy CE
policy top1/top4/top8
legal-move-filtered rank metrics
WDL CE
bucketed CE/top-k by source and phase
```

### Play/search gates

Improve ONNX arena before promotion:

- varied opening set, not repeated tiny fixed list
- paired colors
- random but deterministic seed
- larger sample: 100–400 games for serious gates
- PUCT and policy-only modes
- optional Stockfish adjudication

### Tactical/value gates

Keep sparse Stockfish useful as validation:

- tactical suite
- blunder/cp-loss eval subset
- value calibration on partially labeled Stockfish cache

Do **not** Stockfish-label all 100M rows.

## 10M long-training phase

Before 100M, run improved trainer on existing 10M:

```text
48x5 continuation:
  resume incumbent weights only
  cosine 1e-5 -> 1e-6
  EMA on
  weight_decay 5e-5 or 1e-4
  patience on dev CE + top-k guards

64x6:
  from scratch or current e9 continuation
  cosine 1e-4 -> 1e-5
  EMA on
  weight_decay 1e-4

80x5 hybrid:
  continue e9 or restart clean
  cosine 1e-4 -> 1e-5
  EMA on
  compare label smoothing 0 vs 0.01
```

Stop if:

- dev CE worsens repeatedly,
- top1/top4 drops while CE improves,
- source/phase bucket regression is large,
- tactical/search gate regresses.

## 100M dataset build plan

Target:

```text
data/datasets/supervised_100m_elite_tcec_v1
```

Source mix:

- Lichess Elite as primary source
- capped broader Lichess 2200+ for diversity
- capped TCEC for tactical/engine-quality signal

Keep:

```text
--skip-plies 10
--history-plies 2
--state-planes
standard chess only
exclude bullet/hyperbullet where metadata exists
```

Avoid for now:

- current-FEN-only dedupe in the main history-net path
- full Stockfish labeling
- giant uncompressed JSONL intermediates

Suggested rough caps:

```text
Elite: majority, many months/years
Broader Lichess 2200+: 10M–20M rows
TCEC: 2M–5M rows max unless bucketed carefully
```

Storage discipline:

- write `.jsonl.zst` shards directly
- cache shard-by-shard
- keep train/dev manifests reproducible
- record input sizes, script SHA, git commit, argv, seed
- delete failed partial caches explicitly

## 100M training plan

Train until overfitting guards trip, not for a fixed ego epoch count.

Initial candidates:

```text
48x5 continuation from best 10M/old incumbent
64x6 from best 10M/e9 or restart clean
80x5 hybrid from best 10M/e9
possibly 96x6 hybrid if browser size allows
```

Likely best tradeoff candidates:

```text
48x5: strongest practical PUCT so far, small/browser friendly
80x5 hybrid: best CE/top8 so far, may need better search tuning
64x6: middle ground
```

## Architecture and methods roadmaps

Companion roadmaps created from the Chessformer, DeepSeek, Search-Light, self-play, and TurboQuant notes:

```text
docs/transformer_model_roadmap.md
docs/deepseek_methods_roadmap.md
docs/self_play_scaling_roadmap.md
docs/turboquant_memory_roadmap.md
```

How they fit this plan:

- `transformer_model_roadmap.md`: build a 64-square SquareFormer baseline, then extend to action-value/search-light heads before any model replacement.
- `deepseek_methods_roadmap.md`: add specialist-teacher, action-value/regret, uncertainty, and on-policy distillation after the supervised 100M baseline is stable.
- `self_play_scaling_roadmap.md`: define the lc0-style policy-improvement loop, self-play WAL format, promotion gates, targeted reanalysis, and actor/student split.
- `turboquant_memory_roadmap.md`: use TurboQuant-like ideas later for compressed embeddings/search memory, not v1 model inference.

## Additional ideas to queue, not necessarily immediate

- Legal-move-filtered policy CE/rank metrics.
- Policy entropy diagnostics: compare sharpness vs arena strength.
- Value-head calibration plots by game result/source/phase.
- Opening-book stress test: force positions at ply 0/4/8/12/16 and compare move quality.
- Distill an ensemble of 48x5/64x6/80x5 into a smaller browser model.
- Quantization-aware export or ONNX int8 experiments after model selection.
- History-aware dedupe key for Rust deduper before 100M+ dedupe.
- `.zst` support in Rust deduper.
- Progressive resizing/curriculum: 48x5 warm-start -> 64x6/80x5 distillation.

## Immediate next checklist

- [x] Add scheduler support.
  - implemented `--lr-schedule constant|cosine`, `--warmup-steps`, `--min-lr` in `training/train_residual_aux_multicache_torch.py`.
- [x] Add EMA support.
  - implemented `--ema-decay` and `--ema-best-checkpoint`.
- [x] Add weight decay, label smoothing, grad clipping args.
  - implemented `--weight-decay`, `--policy-label-smoothing`, `--grad-clip-norm`.
- [x] Add bucketed dev eval report or fixed dev-set builder.
  - added `eval/onnx_bucket_eval_jsonl.mjs` for source/phase/time-control ONNX eval on `.jsonl`/`.jsonl.zst`.
  - first `80x5 hybrid e12 EMA` 2k-per-bucket report saved under `data/datasets/supervised_10m_elite_tcec_v1/reports/bucket_eval_80x5_hybrid_e12_ema_2k.json`.
- [x] Improve ONNX arena opening diversity.
  - added `--openings-file` to `eval/onnx_round_robin_arena.mjs`.
  - froze `data/gates/varied_openings_10m_dev_v1.fen` with 240 unique dev-derived FENs from plies 10–30.
- [~] Run 10M long-training comparison with guards.
  - `48x5` e6→e9 cosine/EMA/weight-decay run complete: raw epoch 9 CE `2.093941`, top1 `0.403000`, top4 `0.724160`, top8 `0.857520`; EMA epoch 9 CE `2.094248`.
  - `64x6` e9→e12 cosine/EMA/weight-decay run complete: raw epoch 11 best CE `2.073967`, top8 `0.863250`; EMA epoch 10 best CE `2.053688`, top1 `0.396106`, top4 `0.723672`, top8 `0.865060`.
  - `80x5 hybrid` e9→e12 cosine/EMA/label-smoothing run complete: raw epoch 11 best CE `2.051000`, top8 `0.867282`; EMA epoch 10 best CE `2.034430`, top1 `0.403318`, top4 `0.730706`, top8 `0.868300`.
  - varied-opening ONNX arena, visits 64, 48 games: `80x5e12ema` score `0.6875` WDL `16/1/7`; `64x6e12ema` `0.5417`; `48x5e9` `0.4792`; incumbent `0.2917`.
- [x] Add architecture/method companion roadmaps to the main plan.
  - `docs/transformer_model_roadmap.md`
  - `docs/deepseek_methods_roadmap.md`
  - `docs/turboquant_memory_roadmap.md`
- [~] Build and train initial SquareFormer v0/v1 baselines.
  - added `training/train_squareformer_torch.py`.
  - v0 100k-row/10k-dev/3ep: top8 `0.290600`.
  - v1 100k-row/10k-dev/3ep: top8 `0.416800`.
  - built compact SquareFormer h2 cache: `data/datasets/supervised_10m_elite_tcec_v1/cache_squareformer_h2`, train `10,000,000`, dev `500,000`, size `7.1G`.
  - started cached v1 10M/3ep training, PID `1798802`, log `artifacts/squareformer/squareformer_v1_10m_e3.log`.
- [~] If stable, start `supervised_100m_elite_tcec_v1` build.
  - abandoned overnight 10M SquareFormer training per updated priority; killed PID `1798802`.
  - new priority order after SquareFormer smoke results: finish 100M dataset, build CNN cache, build SquareFormer cache, train canonical 100M models: `chessformer v1`, `32x4`, `48x5`, `64x6`, `chessformer v0`.
  - important guard: full SquareFormer cache training must pass `--max-rows 0 --max-dev-rows 0`; trainer defaults are only 100k/20k and are for smoke tests.
  - added durable/resumable overnight orchestrator: `scripts/overnight_100m_pipeline.sh`.
  - started overnight pipeline PID `1800853`; adjusted defaults to rebuild 2024-01..2025-11 Elite months at up to `5,000,000` positions/month, validate zip downloads, then build 100M dataset/caches/models.
