# Protein/CARBS ideas for Tiny Leela hyperparameter tuning

PufferLib's Protein is a pragmatic CARBS descendant: model both score and cost, then spend new trials on the useful score/cost frontier instead of blindly maximizing score at any cost. We can use the same idea without depending on PufferLib internals.

## Fit to this repo

Good targets:

- PUCT calibration: `cpuct`, `fpu`, temperature, Dirichlet noise, AV weight, rank/regret blending.
- Muon/AdamW training knobs: LR, warmup/cosine, weight decay, Muon momentum/NS steps.
- Self-play actors: visits, batch size, number of games in flight, chunk size, resign/adjudication thresholds.
- Evaluation budgets: visits vs games vs opening diversity.

Bad targets:

- Expensive unconstrained architecture search before stable self-play chunks exist.
- QAT sweeps while QAT quality gates are paused.

## Ledger schema

Use append-only JSONL:

```json
{"trial_id":"puct-0001","params":{"cpuct":1.4,"av_weight":0.10},"score":0.73,"cost":{"wall_seconds":912,"gpu_hours":0.0,"games":96},"status":"succeeded","artifacts":["artifacts/search_mode_arena/..."]}
```

Higher `score` is better; lower `cost` is better. For PUCT calibration, a useful starting score is:

```text
score = winrate_vs_anchor
      + 0.25 * draw_safe_rate
      - 0.50 * catastrophic_blunder_rate
      - 0.05 * log1p(latency_ms)
```

Keep raw metrics in each row so the scoring formula can be changed later.

## Implemented aux-PUCT tuning upgrade

`eval/bayesian_aux_puct_tune.mjs` now uses a Protein/CARBS-style loop:

- append-only `ledger.jsonl` per visit count, including params, score, wall-clock cost, games, and artifacts
- actual surrogate-guided search once `--initial-budget` is exhausted
- cost-aware acquisition via `--cost-aware 1`
- conditional aux weight dimensions through `--dims av,rank,regret`
- optional `cpuct` and `fpu` tuning with `--cpuct-values` and `--fpu-values`
- adaptive refinement from a previous `best_by_visit.tsv` via `--prior-best-tsv`
- confirmation reruns for near-frontier candidates via `--confirm-top-k` and `--confirm-games`
- failures are recorded in the ledger before the run exits

The Mac-mini wrapper defaults to a larger tuning run plus confirmation:

```bash
ITERATIONS=20 \
CONFIRM_TOP_K=3 \
CONFIRM_GAMES=24 \
./scripts/remote_cpu_offload_puct_bayes_by_visit.sh
```

## Pareto reporting

A small local report script is available:

```bash
.venv-onnx/bin/python eval/pareto_sweep_report.py artifacts/sweeps/puct_calibration/ledger.jsonl
```

This lists nondominated trials: no other trial has both higher/equal score and lower/equal cost.

## Protein/CARBS policies to borrow

1. **Cost-aware acquisition**: do not only pick highest expected score; pick trials likely to improve the frontier per unit cost.
2. **Early low-budget probes**: run small visit/game counts first, then promote promising regions to larger arenas.
3. **Noise robustness**: repeat near-frontier trials with different seeds/openings before trusting a lucky point.
4. **Conditional spaces**: only sample AV/rank/regret weights when the model exports those heads.
5. **Failure as data**: OOM, timeout, invalid protocol, or tactical regression should remain in the ledger with `status != succeeded`.
6. **Promotion bands**: a trial can be cheap-and-good enough for analysis without being deployment-grade.

## PUCT calibration search space v0

```json
{
  "cpuct": [0.8, 1.0, 1.25, 1.5, 1.8, 2.2],
  "fpu": [-0.20, -0.10, 0.0, 0.10],
  "av_weight": [0.0, 0.05, 0.10, 0.15, 0.25],
  "rank_weight": [0.0, 0.05, 0.10],
  "regret_weight": [0.0, 0.05, 0.10],
  "visits": [16, 32, 64, 128]
}
```

Start with a Latin/random sample, report the frontier, then allocate follow-ups near nondominated points.
