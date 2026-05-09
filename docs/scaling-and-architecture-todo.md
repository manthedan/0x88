# Tiny Leela scaling + architecture TODO

> **Status: Superseded / historical.** This was an early all-in-one TODO. Prefer the current split in [`README.md`](README.md): CNN/100M scaling, Tactical-MoveFormer, and TinyBT/SquareFormer roadmaps. Keep this file for provenance and completed-checklist context.

## 0. Immediate cleanup

- [ ] Add/verify `.gitignore` protects generated data:
  - [x] `data/cache/`
  - [x] `data/datasets/`
  - [x] `data/stockfish_aug/`
  - [x] `artifacts/*.onnx` via `artifacts/`
  - [x] temporary smoke artifacts via `data/datasets/` and `artifacts/`
- [ ] Decide which helper scripts should be committed:
  - `scripts/build_supervised_dataset_shards.py`
  - `scripts/report_dataset_shards.py`
  - `scripts/build_residual_cache_from_dataset.py`
  - `docs/maia-inspired-roadmap.md`
- [ ] Remove or archive smoke artifacts if not needed.

---

## 1. Dataset pipeline

### 1.1 Sharded supervised dataset builder

- [x] Build initial sharded dataset script.
- [x] Build dataset report script.
- [x] Validate on `20k` smoke dataset.
- [x] Add support for `.jsonl.zst` output using `pyzstd` if installed, otherwise system `zstd`.
- [x] Add standard-chess-only validation where source metadata allows.
- [x] Add explicit source/month caps:
  - Lichess month cap via source name cap
  - TCEC cap via source name cap
  - any future source cap via `--max-rows-per-source` / repeated `--source-cap SOURCE=N`
- [x] Add time-control filtering:
  - exclude bullet/hyperbullet by default in Lichess PGN ingest
  - support `--min-estimated-seconds`, `--min-initial-seconds`, and `--include-bullet`
  - preserve `time_control` metadata on generated rows
- [ ] Add low-clock filtering where clock metadata exists.
- [x] Add rating/source/time-control/variant metadata preservation for reports.
- [x] Add duplicate-position diagnostics:
  - exact FEN key
  - opening proxy
  - game id overlap
- [x] Add reproducibility manifest:
  - input files
  - row counts
  - seed
  - caps
  - git commit
  - script version/hash
- [x] Add raw Lichess monthly download/ingest helper scripts:
  - `scripts/download_lichess_months.sh`
  - `scripts/ingest_lichess_months.sh`

### 1.2 Dataset scale ladder

Build these in order:

- [x] `20k` smoke
- [x] `1M` pipeline validation
- [ ] `5M` pipeline validation
  - [x] `supervised_5m_v1` attempt produced `1,976,065` train rows / `98,819` dev rows after skip/dedupe from current 5M source; need more raw input for a true 5M deduped set.
  - [x] `supervised_rawmix_v1` attempt from raw Lichess/TCEC files produced `2,275,670` train rows / `114,779` dev rows after skip/dedupe; still insufficient for true 5M.
- [ ] `10M` first serious scaling dataset
- [ ] `25M` intermediate scaling point
- [ ] `100M` baseline dataset

Target layout:

```text
data/datasets/supervised_100m_v1/
  manifest.json
  train/
    shard_0000.jsonl.zst
    ...
  dev/
    dev_1m.jsonl.zst
  reports/
    dataset_report.json
    scan_report.json
  cache_h2_state/
  cache_h0_state/
  stockfish_aux/
```

---

## 2. Fixed eval/dev/gates

### 2.1 Dev sets

- [ ] Create fixed `dev_250k_v1`.
- [ ] Create fixed `dev_1m_v1`.
- [x] Ensure dev split is by whole game, not random rows.
- [x] Ensure dev excludes train games.
- [x] Report dev metrics by:
  - source
  - rating bucket, evaluation only
  - ply bucket
  - game phase
  - time control if available
  - material bucket proxy

### 2.2 Policy-only eval

- [x] Add policy-only eval command.
- [x] Metrics:
  - policy CE
  - perplexity
  - top1/top4/top8
  - full policy rank proxy
  - selected move legality proxy for generated legal labels
  - Note: true legal-move-filtered rank needs legal masks/FEN-aware evaluator.
- [x] Report by bucket.
- [x] Add policy-only browser mode / `nodes=1` via `?mode=argmax|sample`.

### 2.3 Tactics/search eval

- [x] Expand tactics suite beyond 6 positions:
  - mate in 1
  - mate in 2
  - hanging queen
  - forks
  - pins/skewers
  - recaptures
  - promotions
  - in-check legal-only positions
  - stalemate/checkmate terminal positions
  - Current `eval/puct_sweep_onnx.mjs` expanded to 14 legal-validated positions.
  - Incumbent `48x5` expanded sweep: policy/pass and PUCT best both `8/14 = 0.571429`.
- [ ] Separate:
  - policy-only tactics
  - value-rerank tactics
  - PUCT tactics
- [x] Keep PUCT calibration separate from supervised CE via `eval/puct_sweep_onnx.mjs` metrics/artifacts.

### 2.4 Arena/gates

- [ ] Keep current `48x5` best-dev as incumbent.
- [ ] Add fixed varied-opening gate set.
- [ ] Add paired candidate-vs-incumbent ONNX gate.
- [ ] Require:
  - no bad dev CE regression,
  - no tactical regression,
  - no varied-opening collapse,
  - browser loads.

---

## 3. Cache/training infrastructure

### 3.1 Cache builder

- [x] Existing residual cache builder works on shard files.
- [x] Added dataset-cache helper.
- [x] Add parallel cache building.
- [x] Add `.jsonl.zst` input support to `build_residual_feature_cache.py`.
- [x] Add current-board-only `18`-plane cache mode.
- [x] Add side-info fields to cache:
  - from square
  - to square
  - moved piece
  - captured piece
  - gives check
  - maybe legal mask sparse representation
- [x] Add sparse Stockfish aux fields:
  - `stockfish_q`
  - `winrate_best`
  - `winrate_played`
  - `winrate_loss`
  - `blunder_bucket`
- [x] Add cache manifest checks:
  - all shards same input planes
  - same policy map
  - same history/state settings
  - row counts summarized in manifest validation
  - train/dev cache compatibility
  - side-info fields present if requested

### 3.2 Multishard trainer

- [x] Existing multishard trainer smoke-tested.
- [x] Clean up trainer CLI for non-aux baseline:
  - [x] make `--resume` optional,
  - [x] allow train from scratch,
  - [x] support `--max-steps`,
  - [x] support eval interval,
  - [x] support best-checkpoint saving via `--best-checkpoint`.
- [x] Add proper logging:
  - train loss
  - dev policy CE
  - WDL CE
  - top-k
  - LR
  - examples/sec
- [ ] Add early-stop rules:
  - [x] max dev CE
  - [x] no improvement patience
  - [ ] catastrophic WDL/tactics regression
- [x] Add CUDA path verification on RTX 3090.
- [x] Add mixed precision verification.
- [x] Add resume-from-checkpoint across shards.
  - Smoke resumed `artifacts/checkpoints/logging_smoke.pt` at epoch 2 with optimizer state.

---

## 4. Baseline scaling experiments

Run these before novel architecture work.

### 4.1 Current incumbent architecture

- [ ] Train `48x5 history2 state` on `1M`.
- [ ] Train `48x5 history2 state` on `5M`.
- [ ] Train `48x5 history2 state` on `10M`.
- [ ] Compare against current `3.6M` best:
  - dev CE
  - top-k
  - WDL CE
  - tactics
  - varied-opening arena
  - subjective browser play

### 4.2 Maia-like LC0 baseline

- [ ] Train `64x6 history2 state` on `10M`.
  - [x] Smoke-tested `64x6 history2 state` train-from-scratch path on `supervised_1m_v1` for 200 steps.
  - [x] Ran comparable `64x6` baseline for 500 GPU steps on `supervised_1m_v1`: dev_policy_ce `5.927134`, top1 `0.042350`.
  - [x] Ran longer `64x6` 1M smoke (~8 epochs / ~2k optimizer steps): dev_policy_ce `3.086569`, top1 `0.228530`, top4 `0.480080`, top8 `0.639160`.
- [ ] If good, train `64x6 history2 state` on `25M`.
- [ ] If good, train `64x6 history2 state` on `100M`.

Primary promotion candidate:

```text
64x6 SE ResNet
policy + WDL
history2/state planes
```

---

## 5. Novel architecture ladder

Only compare on same dataset/dev/gates.

### 5.1 Input representation ablation

- [ ] `46-plane history2 + state`
- [ ] `38-plane history2 no state`
- [x] `18-plane current-board-only` cache/training smoke path
- [ ] Optional `state-only current-board` variant

Key question:

```text
Can 18-plane current-board-only match history2 closely enough to save storage/browser cost?
```

### 5.2 Side-info auxiliary heads

Add Maia-2-inspired auxiliary heads:

- [x] from-square CE
- [x] to-square CE
- [x] moved-piece CE
- [x] captured-piece CE/BCE
- [x] gives-check BCE
- [ ] optional legal-mask BCE

Experiment:

```text
64x6 baseline
vs
64x6 + side-info aux

Status: `64x6 + side-info aux` smoke path validated on `supervised_1m_v1` for 500 GPU steps: dev_policy_ce `5.901123`, top1 `0.042290`. Longer 1M smoke did not beat baseline: dev_policy_ce `3.092488`, top1 `0.225960`.
```

### 5.3 Sparse Stockfish winrate/blunder heads

- [x] Convert cp to winrate:
  - `best_winrate`
  - `played_winrate`
  - `winrate_loss`
- [x] Add blunder buckets:
  - good
  - inaccuracy
  - mistake
  - blunder
- [x] Add trainer heads/losses for sparse winrate-loss and blunder-bucket labels.
- [x] Build `supervised_1m_v1_winrate_aug` sparse Stockfish winrate-loss dataset copy: `74,382` labeled / `1,100,000` rows.
- [x] Build `supervised_1m_v1_winrate_aug/cache_h2_state_sideinfo` with side-info + winrate-loss cache fields.
- [x] Train sparse-label smoke on 1M for 500 GPU AMP steps: dev_policy_ce `5.885704`, top1 `0.043110`, top4 `0.111150`, top8 `0.170980`.
- [x] Train sparse-label 1M smoke for ~8 epochs / ~2k steps: dev_policy_ce `3.085480`, top1 `0.224590`, top4 `0.477540`, top8 `0.638050`; tiny CE edge but worse top-k/WDL than baseline.
- [ ] Train with sparse labels only.
- [ ] Try low weights:
  - `0.005`
  - `0.01`
  - `0.03`
- [ ] Try value-head-only/frozen-policy variants.

### 5.4 Tiny global mixer

Hybrid architecture:

```text
SE ResNet trunk
→ 64 square tokens
→ 1-2 lightweight mixer/attention blocks
→ policy/WDL/aux heads
```

Experiments:

- [x] `64x6 SE` training smoke path
- [ ] `64x6 SE + 1 mixer`
- [ ] `64x6 SE + 2 mixer`
- [ ] Compare ONNX/browser speed.

### 5.5 Factorized policy auxiliaries

- [ ] Keep main LC0 policy map.
- [ ] Add auxiliary:
  - from-square
  - to-square
  - promotion type
- [ ] Do not replace main policy unless proven better.
- [ ] Test whether aux improves top-k and legal ranking.

---

## 6. Stockfish augmentation plan

### 6.1 Partial labeling

- [ ] Build representative `5M` Stockfish sample.
- [ ] Optional `10M` sample if useful.
- [ ] Use:
  - depth `8`
  - MultiPV `4`
  - 16 workers
- [ ] Stratify sample by:
  - source
  - ply bucket
  - policy confidence
  - rating bucket for eval/report only
  - opening bucket

### 6.2 Use labels safely

Use Stockfish for:

- [ ] WDL/value calibration
- [x] winrate-loss aux path
- [x] blunder classification path
- [ ] tactical validation
- [ ] source/bucket quality diagnostics

Do **not** use Stockfish as broad policy replacement.

---

## 7. Browser/playability

- [ ] Keep model selector:
  - `32x4`
  - `48x5`
  - `sfaux`
  - future `64x6`
- [x] Add policy-only mode:
  - `?mode=argmax`
  - `?mode=sample`
- [x] Add stochastic policy mode:
  - `?temperature=...`
  - `?topk=...`
  - `?topp=...`
- [ ] Add model metadata display:
  - architecture
  - dataset
  - dev CE
  - top-k
  - WDL CE
- [ ] Add browser performance benchmark:
  - eval ms
  - visits/sec
  - memory
  - load time

---

## 8. Recommended execution order

### Now

- [x] Build `1M` sharded dataset.
- [x] Build cache from it.
- [x] Train/eval `48x5` smoke/short run.
- [x] Add policy-only eval script.

### Next

- [ ] Build true `5M` sharded dataset.
  - [x] Built `supervised_5m_v1` partial validation set: `1,976,065` train / `98,819` dev after skip/dedupe.
- [x] Train `48x5` on partial `supervised_5m_v1` validation set.
- [x] Train `64x6` smoke paths.
- [ ] Compare to current incumbent.

### Then

- [ ] Build `10M`.
- [ ] Run clean architecture ladder:
  - `48x5`
  - `64x6`
  - `18-plane`
  - `64x6 + side-info`

### Finally

- [ ] Build `100M`.
- [ ] Train best baseline architecture.
- [ ] Add sparse Stockfish winrate/blunder aux.
- [ ] Revisit PUCT/search/self-play.

Final guiding rule:

```text
Scale data first enough to trust comparisons,
but only run 100M after 10M proves the pipeline and architecture are improving real gates.
```
