# Small-BT4 Progression for Tiny Leela

## Core claim

The right small analogue of lc0 BT4 is **not** “BT4 with fewer layers.” It is:

```text
BT4/Chessformer board representation
+ cheapest chess-topology prior that works
+ policy/value interface compatible with PUCT
+ action-value/ranking heads once the trunk is competent
+ conditional search rather than always-on heavy search
```

Organize the project around this ordering:

```text
TinyBT-static  = architectural baseline
TinyBT-AV      = strength baseline
TinyBT-smolgen = efficiency/representation experiment
MiniBT         = scaling probe
PUCT/self-play = after value/action-value are search-useful
```

A 2026-05 external DeepResearch architecture pass did not change this ordering. The useful accepted ideas are summarized in `docs/deepresearch_architecture_triage_2026-05.md`: search-light framing, geometry/history bias before exotic attention, a legal-move query decoder as the best next head/interface bet, and explicit AV/regret metrics.

A 2026-05 Unsloth/GRPO economics pass also did **not** change the architecture ladder. Its useful accepted idea is training-loop design: treat a chess position as the prompt, candidate moves as generations, and teacher/search Q or regret as reward. Implement this as chunked candidate AV/regret/ranking distillation, not as a generic LLM RL framework. See `docs/unsloth_rl_economics_triage_2026-05.md`.

## What to keep from BT4

```text
Keep:
  64 square tokens
  static chess-topology attention bias
  policy + WDL/value interface
  PUCT compatibility
  richer history/state once parity is clean

Shrink:
  smolgen dynamic attention -> template-gated or low-rank smolgen-lite
  global/new embedding -> cheap summary conditioning
  moves-left -> low-weight auxiliary head

Delay:
  side-to-move token-order flip until move-map parity is boringly perfect
  medium/large d256+ scaling until tiny recipe shows signal
  self-play until value/AV improve search

Drop as default tiny assumptions:
  d_model=1024
  32 heads
  15 layers
  4x FFN expansion
  multi-GB GPU runtime
```

References:

```text
https://lczero.org/play/networks/bestnets/
https://lczero.org/blog/2024/02/transformer-progress/
```

## Model ladder

### TinyBT-static

Purpose: first serious BT-style baseline.

```text
input:      64 square tokens, h2 then h7/h8
body:       4-6 layers, d_model 96-128, 4-8 heads, FFN 1x-2x
attention:  static chess-relation bias
heads:      from-to policy + WDL/q + optional value bucket
question:   can tiny chess-aware square attention beat CNN/MoveFormer at same size/latency?
```

### TinyBT-AV

Purpose: practical strength baseline.

```text
trunk:      TinyBT-static winner
heads:      policy + WDL/q + legal-move AV/rank/regret
training:   teacher top-k + student top-k + checks/captures/promotions + hard negatives
question:   can action-value/ranking reduce blunders and improve strength per node?
```

For tiny engines, build this before full smolgen. Action-value/reranking likely buys more practical strength per parameter once static topology works.

### TinyBT-smolgen

Purpose: dynamic topology / representation efficiency test.

```text
trunk:      TinyBT-static or TinyBT-AV winner
attention:  static relation bias + smolgen-lite
variants:   template-gated relation coefficients, then low-rank dynamic bias
question:   does dynamic attention beat equivalent params, AV improvements, or d_model scaling?
```

Do not start with full BT4 smolgen.

### MiniBT

Purpose: scaling probe, not pure WASM-first.

```text
layers:     8
hidden:     d_model 192-256
heads:      8
ffn:        1x-2x
heads:      policy + WDL/q/value bucket + AV + uncertainty
question:   does the architecture have a healthy scaling curve for WebGPU/desktop ONNX?
```

### Later hybrids

```text
CNN + TinyBT:
  small CNN stem -> 64 square embeddings -> 2-4 attention blocks
  tests whether local convolution improves tiny transformer sample efficiency

Shared-weight/recurrent TinyBT:
  3-4 unique blocks repeated 2-3 times
  tests iterative calculation depth without proportional parameter growth
```

## Best smolgen-lite first design

Use dynamic relation-template gates before low-rank free bias.

```text
fixed templates:
  T_r[i,j] for relation r

summary:
  g = MLP(mean_pool(tokens) + side/rule summary)

coefficients:
  alpha[layer, head, relation] = Linear(g)

attention:
  score[l,h,i,j] = QK/sqrt(d)
                 + static_bias[l,h,i,j]
                 + sum_r alpha[l,h,r] * T_r[i,j]
```

Why first:

```text
tiny parameter cost
interpretable
export-friendly
harder to overfit than full generated 64x64 matrices
preserves chess topology explicitly
```

Try low-rank dynamic bias only after template gates show signal:

```text
score += U[layer,head,i,k] * V[layer,head,j,k]
rank k = 2 or 4 initially
```

## Starting relation templates

Use roughly 16-32 classes first:

```text
same_square
same_rank
same_file
same_diagonal
same_antidiagonal
knight_move
king_move
white_pawn_attack
black_pawn_attack
rook_ray_distance_1
rook_ray_distance_2_3
rook_ray_distance_4_plus
bishop_ray_distance_1
bishop_ray_distance_2_3
bishop_ray_distance_4_plus
same_color_complex
center_to_center
center_to_edge
edge_to_edge
corner_relation
promotion_rank_relation
king_zone_relation_if_known
```

Add occupancy-conditioned behavior through smolgen-lite, not by exploding static relation classes.

## Experiment stages

```text
Stage A — lock TinyBT-static:
  h2 control
  static chess relation bias
  policy + WDL/value bucket
  no smolgen yet

Stage B — prove ranking:
  add candidate-only action-value/regret/ranking heads
  train on teacher top-k + student top-k + played move + checks/captures/promotions + hard negatives/random legal distractors
  score AV/regret/reply candidates in chunks instead of dense all-move auxiliary tensors
  evaluate AV-top8, pairwise ranking, regret@top1, queen/material blunders

Stage C — test dynamic topology:
  add smolgen-lite template gates
  compare against more d_model and against AV improvements

Stage D — improve state:
  h7/h8 history
  moves-left/horizon auxiliary
  side-to-move flip only after strict parity tests

Stage E — scale:
  MiniBT d192/d256
  WebGPU-first
  compare against best CNN/MoveFormer baseline

Stage F — engine integration:
  uncertainty head
  conditional PUCT
  self-play/search-improved labels
  on-policy hard-position mining

Stage G — deployment:
  FP16 WebGPU
  INT8/PTQ/QAT CPU/WASM if quality survives
  size/latency/Elo Pareto frontier
```

## Decision rules

```text
Static bias:
  keep if it improves policy/search with negligible latency.

AV/ranking:
  keep if chunked candidate losses improve regret, pairwise ranking, AV-rerank Elo, and queen/material diagnostics.

Smolgen-lite:
  keep only if it improves the strength/latency frontier versus more d_model or AV.

h7/h8:
  keep if rule/repetition/tactical/endgame splits improve without major latency hit.

Moves-left:
  keep only if WDL calibration, PUCT, or endgame/horizon tests improve.

Side-to-move flip:
  keep only with perfect policy-map/parity tests and measurable symmetry/Elo gain.

MiniBT:
  run only after the tiny recipe shows signal.
```

## Metrics

Do not pick winners by validation CE alone.

```text
Static quality:
  policy CE/KL, top-k, WDL Brier/calibration, value bucket CE

Move ranking:
  pairwise accuracy, weighted tau, average regret, p95/p99 regret, catastrophic regret rate

Engine quality:
  policy-only Elo, AV-topk rerank, PUCT 32/64/128, conditional-search Elo

Diagnostics:
  queen death/material blunder rate, shallow Stockfish deltas, tactical/open/closed suites

Deployment:
  ONNX bytes, FP16/INT8 size, WebGPU/WASM latency, memory, Elo per byte/ms
```
