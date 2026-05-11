# Gumbel-Zero self-play track

This is a parallel research lane for **rules-only chess self-play from zero**. It is intentionally separate from the supervised-to-self-play lane.

## Invariants

- No supervised initialization for this lane.
- No Stockfish/Maia/lc0 teacher labels in the zero buffer.
- No mixing with supervised model self-play until an explicit league/cross-play phase.
- Promotion is checkpoint-vs-checkpoint first, then cross-lane comparison.
- Classic PUCT remains the default deploy/eval baseline; Gumbel-Zero is experimental.

## Initial loop

One move in the bootstrap generator:

1. evaluate board with the current zero-lane evaluator (`uniform` for bootstrap smoke, later a zero-initialized exported model),
2. choose a root candidate set with `log(prior) + Gumbel`,
3. force at least one evaluation for each candidate where the budget allows,
4. spend remaining visits with a low-node Gumbel/PUCT root policy,
5. derive a Q/prior/Gumbel-improved target over the root candidates,
6. play the selected move and store candidate Q/regret diagnostics.

The generated JSONL uses `schema=tiny_leela_gumbel_zero_selfplay_v1` and remains valid for the existing `scripts/selfplay_chunk_validate.py` because `policy` and terminal `result` keep the standard shape.

## Bootstrap smoke command only

Do not launch a long run without review. The small local smoke is:

```bash
npm run selfplay:gumbel-zero -- \
  --evaluator uniform \
  --games 2 \
  --max-plies 16 \
  --visits 8 \
  --candidate-count 8 \
  --out /tmp/gumbel_zero_smoke.jsonl
.venv-onnx/bin/python scripts/selfplay_chunk_validate.py /tmp/gumbel_zero_smoke.jsonl
```

## Phase 1 proposal for review

```text
evaluator: uniform bootstrap, then zero-initialized SquareFormer export
simulations: 16
candidate_count: 16
min_candidate_visits: 1
estimated_q: pessimistic
move_selection: argmax initially; sample-target as an ablation
max_plies: 160
replay: zero_selfplay only
losses: policy + WDL first, weak AV/regret until model is non-random
```

Suggested first reviewed launch size:

```text
1k-5k games, CPU workers only, no GPU training until the chunk schema and metrics are approved.
```

## Required diagnostics

Track per chunk:

- legal game completion rate,
- decisive/checkmate/max-ply rates,
- selected-outside-policy-top1 rate,
- policy entropy,
- candidate Q spread,
- selected regret,
- candidate visits and root score distribution,
- validation pass/fail.

## Later comparison plan

Keep buffers separate:

```text
data/selfplay/gumbel_zero/
data/selfplay/supervised_sp/
data/selfplay/crossplay_league/
```

Evaluation order:

1. `gzero_i` vs `gzero_{i-1}` at fixed nodes,
2. Gumbel-search and classic-PUCT searches against random/legal and weak supervised anchors,
3. cross-lane games against CNN/MF/BT4 only after within-lane progress is real,
4. optional league/cross-play data generation as a separate, labeled phase.
