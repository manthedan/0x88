# Autoresearch: trainable board CNN for tiny Leela

## Objective
Find a trainable board-input neural evaluator that improves supervised chess policy quality over the current frozen-feature linear/MLP students. The current live model is legal but plays horribly; frozen random conv features are the suspected bottleneck.

## Metrics
- **Primary**: `dev_policy_top8` (unitless, higher is better) — top-8 target move recall on held-out PGN positions.
- **Secondary**: `dev_policy_top1`, `dev_policy_top4`, `dev_policy_ce`, `dev_wdl_ce`, `board_cnn_rows`, runtime.

## How to Run
`./autoresearch.sh` — trains a small board CNN smoke workload and emits `METRIC` lines.

## Files in Scope
- `training/train_board_cnn.py`: tinygrad trainable board CNN prototype; main target.
- `training/train_feature_mlp.py`: frozen-feature MLP reference; only edit for metric comparison if needed.
- `src/nn/studentEvaluator.ts`, `rust/tiny_leela_core/src/lib.rs`: inference wiring only after a model clearly beats frozen baselines.
- `autoresearch.sh`: benchmark harness.

## Off Limits
- Do not commit generated data/model artifacts under `data/*training*.jsonl`, `data/tcec_raw/`, or `artifacts/`.
- Do not use Dovetail for this overnight loop.
- Do not replace the live web model unless a trained model clearly improves policy metrics and inference support exists.

## Constraints
- Use `.venv-tinygrad` and local CUDA path.
- Keep each experiment reasonably short: prefer 10k-50k rows, 3-10 epochs until the trainer is stable.
- Always output `METRIC dev_policy_top8=...`.
- Passing code should satisfy `python3 -m py_compile training/train_board_cnn.py`.

## What's Been Tried
- Frozen conv + linear heads on 500k mixed Lichess/TCEC: `top1=0.0238`, `top4=0.0587`, `top8=0.0874`.
- Frozen conv + MLP 512, Adam, 80 epochs: `top1=0.0238`, `top4=0.0616`, `top8=0.10205`; live play still bad.
- Initial `training/train_board_cnn.py` prototype exists but first run was aborted before results; needs stabilization.
