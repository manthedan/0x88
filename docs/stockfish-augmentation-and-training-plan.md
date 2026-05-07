# Stockfish augmentation and training plan

This document captures the current lessons from Stockfish labeling, cp-loss weighting, auxiliary value training, PUCT calibration, and the recommended path for improving the tiny Leela-like engine without poisoning the supervised incumbent.

## Current trusted incumbent

Keep the supervised 48x5 2026mix best-dev model as the anchor:

```text
artifacts/residual_48x5_history2_2026mix_3633k_best.onnx
artifacts/residual_48x5_history2_2026mix_3633k_best.meta.json
artifacts/checkpoints/residual_48x5_history2_2026mix_3633k_e100.best.pt
artifacts/selfplay_best.onnx
artifacts/selfplay_best.meta.json
```

Best known supervised dev metrics:

```text
dev_policy_ce:   2.827080
dev_wdl_ce:      0.865841
dev_policy_top1: 0.362800
dev_policy_top4: 0.683871
dev_policy_top8: 0.822929
```

Do not replace this incumbent unless a candidate passes dev, tactics, and varied-opening gate checks.

## Why Stockfish augmentation

Pure self-play/search distillation from the current engine was too weak and degraded supervised strength. Stockfish is useful as a tactical/quality signal, but should not broadly replace the human/LC0-like policy target.

Preferred use:

- keep human/supervised policy target,
- keep game-result WDL target,
- add Stockfish tactical value signal,
- optionally use mild cp-loss quality weighting later.

Avoid broad Stockfish policy replacement unless heavily mixed and gated.

## Labeling format

The Stockfish cp-loss labeler writes one row per supervised position:

```json
{
  "id": "...",
  "fen": "...",
  "played": "e2e4",
  "best": "d2d4",
  "cp_best": 34.0,
  "cp_played": 12.0,
  "cp_loss": 22.0,
  "depth": 8,
  "multipv": 4
}
```

Derived auxiliary target:

```text
stockfish_q = tanh(cp_best / 400)
model_q     = P(win) - P(loss)
aux_loss    = mse(model_q, stockfish_q)
```

Recommended initial full-dataset aux loss:

```text
loss = policy_ce + wdl_ce + 0.05 * aux_loss
```

The 250k test used `0.1`; full run uses safer `0.05`.

## Labeling performance

Stockfish is CPU-bound. GPU does not help Stockfish. Parallelism is via multiple Stockfish processes over row shards.

Measured:

```text
250k, depth 8, MultiPV 4, 8 workers:
  250,000 rows in 471 sec
  ~530 rows/sec

full 3,633,118, depth 8, MultiPV 4, 16 workers:
  3,633,118 rows in 4,132 sec
  ~879 rows/sec
  ~68.9 min
```

16 workers works on the 24-core machine and is the current preferred full-labeling setting.

## Experiments run

### 1. Harsh cp-loss weighting on 250k

Weights:

```text
cp_loss <= 25:   1.0
cp_loss <= 75:   0.5
cp_loss <= 150:  0.2
cp_loss > 150:   0.05
```

Result after 1 epoch:

```text
dev_policy_ce:   2.858133  worse than incumbent by +0.031053
dev_wdl_ce:      0.864054  better by -0.001787
dev_policy_top1: 0.368541  better by +0.005741
dev_policy_top4: 0.687937  better by +0.004066
dev_policy_top8: 0.825822  better by +0.002893
```

Conclusion: useful signal but too aggressive/distribution-shifty. Do not use harsh cp-loss weighting as the primary path.

### 2. Stockfish auxiliary value on 250k

No policy downweighting. Added only Stockfish q auxiliary target.

Settings:

```text
rows:         250k
depth:        8
multipv:      4
aux_q_weight: 0.1
lr:           1e-5
epochs:       1
```

Result:

```text
dev_policy_ce:   2.832686  worse by +0.005606
dev_wdl_ce:      0.863304  better by -0.002537
dev_policy_top1: 0.365961  better by +0.003161
dev_policy_top4: 0.686151  better by +0.002280
dev_policy_top8: 0.824636  better by +0.001707
```

Conclusion: much safer than cp-loss weighting. Policy CE slightly regressed, but WDL and top-k improved. Promising enough to scale with lower aux weight.

### 3. Full Stockfish auxiliary value run

Current/full test:

```text
rows:         3,633,118
depth:        8
multipv:      4
workers:      16
aux_q_weight: 0.05
lr:           1e-5
epochs:       1
brake:        max_dev_policy_ce=2.85
```

Pipeline:

```text
scripts/run_stockfish_full_aux.sh
```

Outputs:

```text
data/stockfish_aug/cp_loss_2026mix_train_full_d8_mpv4.jsonl
data/stockfish_aug/balanced_history_train_2026mix_full_sfaux_d8_mpv4.jsonl
data/cache/residual_2026mix_full_h2_sfaux_d8_mpv4
artifacts/residual_48x5_history2_2026mix_sfauxfull_d8_mpv4.onnx
artifacts/residual_48x5_history2_2026mix_sfauxfull_d8_mpv4.meta.json
artifacts/checkpoints/residual_48x5_history2_2026mix_sfauxfull_d8_mpv4.pt
```

## PUCT calibration lesson

A small PUCT sweep over the incumbent showed no change on the current 6-position tactics suite:

```text
policy-only: 3/6
PUCT:        3/6
visits:      1,4,8,16,32,64
cpuct:       0.5,0.75,1,1.25,1.5,2,2.5,3
```

Conclusion:

- current tactics suite is too small/coarse for calibration,
- PUCT did not obviously rescue tactics over policy-only,
- improve the tactical/search eval harness before trusting search-distillation/self-play,
- do not infer much from startpos-only gates.

## Recommended optimal path from here

### Candidate training path

1. Use full Stockfish aux labels over the supervised 2026mix training set.
2. Finetune from incumbent best checkpoint only.
3. Do not replace human policy.
4. Use a small aux weight first:

```text
aux_q_weight = 0.03 to 0.05
lr           = 1e-5
epochs       = 1
max_dev_policy_ce = 2.85 hard brake
```

5. If full aux improves or nearly matches policy CE while improving WDL/top-k, test:

```text
aux_q_weight = 0.03
aux_q_weight = 0.05
aux_q_weight = 0.075
```

6. Only later try mild cp-loss weighting:

```text
cp_loss <= 25:   1.0
cp_loss <= 75:   0.8
cp_loss <= 150:  0.5
cp_loss > 150:   0.2
```

Never return to harsh weighting unless a specific gate justifies it.

### Evaluation gates

A candidate is not promotion-worthy based on training loss alone. Require:

```text
1. dev_policy_ce near or better than 2.827080
2. dev_wdl_ce non-regression or improvement
3. dev top1/top4/top8 non-regression or improvement
4. tactics suite non-regression
5. varied-opening paired arena gate, not startpos-only
```

If metrics trade off:

- small CE regression with WDL/top-k improvement can be inspected,
- large CE regression is discard,
- final decision requires tactics/gate.

### Search/calibration path

Before more self-play training:

1. Expand tactics suite substantially:
   - mate-in-1,
   - mate-in-2 if feasible,
   - hanging queen,
   - recapture,
   - fork,
   - promotion,
   - legal-only in-check sanity,
   - stalemate/checkmate terminal cases.
2. Re-run policy-only vs PUCT sweeps.
3. Sweep:

```text
cpuct: 0.5, 0.75, 1.0, 1.25, 1.5, 2.0
visits: 8, 16, 32, 64, 128
```

4. Consider prior temperature transform if top-k improves but CE worsens:

```text
p' ∝ p^alpha
alpha < 1 flattens priors
alpha > 1 sharpens priors
```

## Parallel/sharded infrastructure

For future iterations, avoid one giant sequential cache pipeline. Use shards.

Added tools:

```text
scripts/apply_stockfish_aux_shard.py
scripts/build_aux_cache_shards.sh
training/train_residual_aux_multicache_torch.py
```

Shard build example:

```bash
SHARDS=16 JOBS=8 scripts/build_aux_cache_shards.sh
```

Produces:

```text
data/stockfish_aug/shards_full_sfaux_d8_mpv4/shard_000.jsonl
...
data/cache/residual_2026mix_full_h2_sfaux_d8_mpv4_shards/shard_000/
...
data/cache/residual_2026mix_full_h2_sfaux_d8_mpv4_shards/manifest.json
```

Train directly from shards:

```bash
.venv-onnx/bin/python training/train_residual_aux_multicache_torch.py \
  --manifest data/cache/residual_2026mix_full_h2_sfaux_d8_mpv4_shards/manifest.json \
  --dev-cache data/cache/residual_2026mix_dev_216k_h2 \
  --resume artifacts/checkpoints/residual_48x5_history2_2026mix_3633k_e100.best.pt \
  --out artifacts/residual_48x5_history2_2026mix_sfauxfull_sharded.pt \
  --onnx-out artifacts/residual_48x5_history2_2026mix_sfauxfull_sharded.onnx \
  --meta-out artifacts/residual_48x5_history2_2026mix_sfauxfull_sharded.meta.json \
  --checkpoint artifacts/checkpoints/residual_48x5_history2_2026mix_sfauxfull_sharded.pt \
  --epochs 1 \
  --lr 0.00001 \
  --aux-q-weight 0.05 \
  --max-dev-policy-ce 2.85 \
  --device cuda
```

Note: the current full run uses the original sequential cache path; the sharded path is for future runs or reruns.

## Operational cautions

- Do not commit generated labels/caches/artifacts/logs.
- Keep `artifacts/selfplay_best.onnx` pinned to incumbent unless a candidate truly passes gates.
- Avoid autoresearch `log_experiment(status=crash/discard)` while important untracked helper scripts are not committed/stashed, because it may revert untracked work.
- Do not trust deterministic start-position arena promotions.
- Do not resume self-play promotion until search/eval harness is stronger.
