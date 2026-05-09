# 10M → 100M dataset build TODO

> **Status: Historical / partially superseded.** This records the dataset-build plan that led to the current 10M/100M datasets. For live truth, check dataset manifests under `data/datasets/*/manifest.json`, cache manifests, and queue status files under `artifacts/*/status.txt`.

Objective: build a clean `10M` supervised dataset first, using the `5M` milestone as an intermediate validation point, then scale the same pipeline toward `100M`.

## Source policy

- Main source: Lichess Elite monthly PGNs.
- Secondary diversity source: filtered Lichess 2200+ standard/non-bullet rows.
- Engine source: capped TCEC rows only; do not let TCEC dominate policy training.
- Use `--skip-plies 10` for now instead of explicit dedupe in the main path.
- Keep Rust deduper experimental until it supports history-aware keys and `.zst`.

## 0. Download / source staging

- [ ] Finish/resume Lichess Elite downloads:
  - [x] `2025-09`
  - [x] `2025-10`
  - [x] `2025-11`
  - [ ] `2025-08` partial download may exist as `.part`; resume or delete/retry.
- [x] Download enough additional Elite months for true `10M` after filtering:
  - completed: `2025-01` through `2025-07`, plus `2025-09` through `2025-11`.
- [ ] Keep raw zips/PGNs under generated-data paths only:
  - `data/lichess_elite_raw/`
  - `data/lichess_elite_training/`

## 1. Lichess Elite ingestion

- [x] Inspect zip layout for downloaded Elite files.
- [x] Add or adapt an Elite ingest helper script:
  - input: `data/lichess_elite_raw/lichess_elite_YYYY-MM.zip`
  - output: `data/lichess_elite_training/lichess_elite_YYYY-MM.jsonl`
  - preserve `source=lichess_elite_YYYY-MM`
  - preserve rating/time-control/variant metadata if present
  - skip first `10` plies
  - exclude bullet if metadata exists
  - max plies per game ~90
- [x] Smoke ingest one Elite month.
- [x] Report rows/month after filtering.
  - current monthly cap used: `1,200,000` rows/month.

## 2. Build 5M intermediate dataset

- [x] Build `data/datasets/supervised_5m_elite_tcec_v1` as intermediate validation.
- [ ] Target mix:
  - ~3.5M Lichess Elite
  - ~1.0M broader Lichess 2200+ filtered rows
  - ~0.5M capped TCEC
- [ ] Builder settings:
  - `--skip-plies 10`
  - `--history-plies 2`
  - `--dev-rows 250000`
  - `--rows-per-shard 250000`
  - source caps enabled
  - opening/game caps enabled
  - `.jsonl.zst` output
- [ ] Run `scripts/report_dataset_shards.py` and inspect:
  - source counts
  - time-control buckets
  - variant counts
  - phase/material buckets
  - duplicate FEN diagnostics
  - opening proxy counts

## 3. Build true 10M dataset

- [x] Build `data/datasets/supervised_10m_elite_tcec_v1`.
- [ ] Target mix:
  - ~7M Lichess Elite
  - ~2M broader Lichess 2200+ filtered rows
  - ~1M capped TCEC
- [ ] Builder settings:
  - `--skip-plies 10`
  - `--history-plies 2`
  - `--dev-rows 500000` or fixed `dev_250k_v1` if ready
  - `--rows-per-shard 250000`
  - `.jsonl.zst`
  - source caps and opening/game caps
- [x] Run dataset report and save under dataset `reports/`.
- [ ] Create/freeze fixed dev set from the 10M source pool if not already done:
  - `dev_250k_v1`
  - later `dev_1m_v1`

## 4. Cache and eval 10M

- [x] Build `cache_h2_state` for `supervised_10m_elite_tcec_v1`:
  - history2
  - state planes
  - 46 input planes
  - cache manifest validation must pass
- [x] Run policy eval on incumbent `48x5` against fixed dev.
- [x] Train/eval `48x5 history2 state` on 10M.
  - first 200-step continuation smoke complete.
  - real epoch-1 continuation complete with `--resume-model-only`, LR reset to `1e-5`: dev policy CE `2.276072`, top1 `0.380848`, top4 `0.696280`, top8 `0.833784`.
  - longer epoch-6 continuation complete at LR `1e-5`: dev policy CE `2.115546`, top1 `0.399874`, top4 `0.720214`, top8 `0.854100`.
- [x] Train/eval `64x6 history2 state` on 10M.
  - first 200-step from-scratch smoke complete.
  - clean 3-epoch from-scratch baseline complete at LR `1e-4`: best/final dev policy CE `2.263632`, top1 `0.349424`, top4 `0.666376`, top8 `0.819770`.
  - continued through epoch 9 at LR `1e-4`: dev policy CE `2.106797`, top1 `0.383906`, top4 `0.711602`, top8 `0.856588`.
- [ ] Compare to current trusted `48x5` 2026mix best-dev incumbent:
  - dev policy CE
  - top1/top4/top8
  - WDL CE
  - tactics/PUCT sweep
  - browser load/playability

## 5. Gate before 100M

- [ ] Add/freeze varied-opening gate set.
- [ ] Add paired candidate-vs-incumbent ONNX gate.
- [ ] Require before scaling to 100M:
  - no bad dev CE regression
  - no tactical regression
  - no varied-opening collapse
  - browser loads
  - 10M improves or gives clear architecture signal

## 6. 100M planning

- [ ] Estimate rows/month from Elite ingestion.
- [ ] Estimate disk use for compressed shards and one cache representation.
- [ ] Avoid duplicate JSONL/cache intermediates.
- [ ] Decide whether 100M needs Rust streaming/sharded preprocessing.
- [ ] Build `supervised_100m_elite_tcec_v1` only after 10M pipeline/gates are clean.

## Immediate next command ideas

```bash
# Resume/download more Elite months
mkdir -p data/lichess_elite_raw
curl -L -C - --fail -o data/lichess_elite_raw/lichess_elite_2025-08.zip.part \
  https://database.nikonoel.fr/lichess_elite_2025-08.zip

# Inspect a completed zip
unzip -l data/lichess_elite_raw/lichess_elite_2025-09.zip | head
```
