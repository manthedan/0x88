The core answer: the right “small BT4” analogue is **not** “BT4 with fewer layers.” It is:

```text
BT4/Chessformer board representation
+ the cheapest chess-topology prior that works
+ a policy/value interface compatible with PUCT
+ action-value/ranking heads once the trunk is competent
+ conditional search rather than always-on heavy search
```

So I would organize the project around this claim:

```text
TinyBT-static is the architectural baseline.
TinyBT-AV is the strength baseline.
TinyBT-smolgen is an efficiency/representation experiment.
MiniBT is the scaling probe.
PUCT/self-play comes after value/action-value are search-useful.
```

Modern Lc0’s top public competition net is still BT4-family: the current best-net page lists `BT4-it332` first, as a `1024 × 15` net, about **365 MB**, using about **4 GB** GPU memory, and describes it as the net currently sent to competitions like TCEC/CCC. ([Leela Chess Zero][1]) The 2024 transformer post remains the key architecture description: encoder-only, 64 square tokens, side-to-move token-order flip, Mish FFNs, DeepNet/Post-LN style blocks, chess-topology attention bias, smolgen dynamic attention logits, policy/WDL/moves-left heads, and a 15-layer 1024-wide BT4 at about **191.3M parameters** and **7.613 GFLOPs per position**. ([Leela Chess Zero][2])

For your project, that means the right target is **BT4’s inductive bias, not BT4’s scale**.

---

# My recommended small-BT analogue

## TinyBT-static: the default first serious target

```text
Input:
  64 square tokens
  h2 or h7/h8 history variant
  rule-state features: side, castling, en passant, repetition, rule-50

Body:
  4–6 transformer layers
  d_model 96–128
  heads 4–8
  head_dim 16–32
  FFN ratio 1×–2×
  static chess-relation attention bias

Heads:
  from-to policy head
  WDL head
  value bucket or q head
  optional value-error/uncertainty head
```

This is the model that should answer:

```text
Can a tiny square-token transformer beat the MoveFormer/CNN baseline
at the same size and latency?
```

The Chessformer notes support this direction strongly: 64 square tokens, chess-aware positional structure, and from-to move heads are the useful tiny-model pieces; the goal is not copying CF-240M or BT4 wholesale. 

## TinyBT-smolgen: only after static bias proves useful

```text
Same as TinyBT-static,
plus low-rank or template-conditioned dynamic attention bias.
```

Do **not** start with full BT4 smolgen. Use a cheaper version that asks:

```text
Can the model dynamically modulate the static chess topology
based on whether the position is open, closed, tactical, king-side loaded, etc.?
```

Lc0’s post says static attention-logit bias made models behave as if about **50% larger** with negligible throughput loss, while smolgen gave another similar “acts larger” effect with about **10% throughput reduction**. That suggests the tiny-model ordering should be: static bias first, dynamic bias second. ([Leela Chess Zero][2])

## MiniBT: the scaling probe

```text
Input:
  64 square tokens, likely h7/h8

Body:
  8 layers
  d_model 192–256
  heads 8
  FFN ratio 1×–2×
  static bias or smolgen-lite

Heads:
  policy
  WDL/q/value bucket
  action-value
  uncertainty
```

This is not the pure WASM-first model. It is the WebGPU/desktop model that tests whether the architecture has a healthy scaling curve.

## TinyBT-AV-PUCT: the engine architecture

```text
TinyBT or MiniBT trunk
  → policy head
  → WDL/value heads
  → action-value top-k reranker
  → uncertainty/tactical heads
  → conditional small PUCT
```

This should be the “best lightweight engine” candidate. It is not pure searchless play; it is search-light, with search used when the model is uncertain or tactical. The uploaded search-light design already converges on this: policy and WDL are necessary, but action-value reranking and uncertainty-gated PUCT are the likely strength-per-millisecond winners. 

---

# What to keep, shrink, delay, or drop from BT4

| BT4 ingredient                       | Tiny analogue      |    Priority | Why                                                                                              |
| ------------------------------------ | ------------------ | ----------: | ------------------------------------------------------------------------------------------------ |
| 64 square tokens                     | Keep               |     Highest | This is the core chess-native transformer representation.                                        |
| Side-to-move flipping                | Test carefully     |      Medium | Useful symmetry, but dangerous for move-map bugs.                                                |
| Static chess-topology attention bias | Keep early         |     Highest | Cheap, strong prior, likely best tiny ROI.                                                       |
| Full smolgen                         | Shrink             |      Medium | Good idea, but full dynamic 64×64/head generation is too expensive for tiny.                     |
| New/global embedding                 | Shrink             |      Medium | Use a cheap global board summary token or per-square conditioning.                               |
| Policy head                          | Keep               |     Highest | Needed for move generation and PUCT.                                                             |
| WDL/value head                       | Keep               |     Highest | Needed for evaluation and search.                                                                |
| Moves-left head                      | Optional auxiliary |  Low/medium | Useful for calibration/horizon, probably not core strength early.                                |
| Full PUCT                            | Conditional        | Medium/high | Search is valuable but should be spent selectively.                                              |
| 15 layers, d=1024                    | Drop               |  Not target | Wrong scale for browser/tiny.                                                                    |
| FFN expansion 4×                     | Drop               |         Low | Lc0 found little benefit from large FFN expansion in chess transformers. ([Leela Chess Zero][2]) |

---

# Answers to your six uncertainties

## 1. Is static chess-relation bias enough, or is smolgen-style dynamic bias critical?

For tiny models, I would assume:

```text
static relation bias is the first-order win;
smolgen-lite is a second-order refinement.
```

Static bias is cheap and directly encodes chess topology:

```text
same file
same rank
same diagonal
knight relation
pawn attack
king neighborhood
same color complex
rook/bishop/queen ray buckets
```

That gives the model chess geometry before it has to learn everything from data. It also adds tiny parameter cost compared with the trunk.

Dynamic bias becomes important when the same geometric relation should mean different things in different positions:

```text
closed center:
  long-range communication should often be suppressed

open file:
  rook/file communication should be amplified

king attack:
  attacker/king-zone paths should be amplified

locked pawn structure:
  diagonals/files may be blocked or strategically irrelevant
```

My recommendation:

```text
E1 static bias is mandatory.
E2 smolgen-lite is justified only if E1 beats control
and dynamic bias improves strength/latency on hard tactical/open/closed suites.
```

Do not judge E2 only by validation CE. Judge it by:

```text
policy-only Elo
AV-rerank Elo
weighted tau / regret
tactical suite
latency overhead
```

## 2. How should smolgen be simplified for a 1M–6M parameter model?

Use **template-gated** or **low-rank** smolgen, not full BT4 smolgen.

### Option A: dynamic relation-template gates

Precompute relation templates:

```text
T_r[i, j] for relation r
```

Examples:

```text
same file
same diagonal
knight move
pawn attack
king zone
rook ray distance
bishop ray distance
```

Then generate coefficients from the position:

```text
g = MLP(mean_pool(tokens))
alpha[layer, head, relation] = Linear(g)

dynamic_bias[layer, head, i, j] =
  static_bias[layer, head, i, j]
  + Σ_r alpha[layer, head, r] * T_r[i, j]
```

This is the best first smolgen-lite. It is tiny, interpretable, and chess-specific.

### Option B: low-rank dynamic bias

Generate:

```text
U[layer, head] ∈ R[64, r]
V[layer, head] ∈ R[64, r]

dynamic_bias = U @ Vᵀ
```

Use small rank:

```text
r = 2, 4, or 8
```

This can model dynamic open/closed patterns without generating a full free 64×64 matrix per head.

### Option C: sparse layer placement

Apply dynamic bias only in:

```text
middle layers
or final 2–3 layers
or every other layer
```

For tiny models, not every layer needs smolgen.

### Avoid initially

```text
full per-layer, per-head 64×64 generated bias
large 2048→256→head→4096 BT4-style pathway
large hidden smolgen MLPs
```

That is too much complexity before you know static bias is saturated.

## 3. Is side-to-move token-order flipping worth the move-map complexity?

Probably yes eventually, but **not until parity tests are boringly perfect**.

The upside:

```text
reduces color/asymmetry burden
matches Lc0/Chessformer-style canonicalization
helps the model share patterns between White and Black
```

The downside:

```text
one orientation bug can destroy move quality
promotion/castling/en-passant bugs become harder to see
from-to policy mapping can silently drift
```

Your own infrastructure notes already flag policy-map hardening, side-to-move/color transforms, backend parity, and policy-root prior parity as major risk areas. 

My recommendation:

```text
Do E6 only after:
  moveToActionId/actionIdToMove parity passes
  mirrored FEN consistency passes
  promotion/castling/en-passant tests pass
  root policy top-k equals PUCT root priors
  browser/Node/Python parity passes
```

A conservative compromise:

```text
Do not flip token order initially.
Instead encode side-to-move and color explicitly.
Use mirrored augmentation in training.
Then test flipping as an ablation.
```

Keep E6 if it improves:

```text
policy KL/top-k
black/white split performance
mirrored-position consistency
engine Elo
```

and causes zero mapping regressions.

## 4. Pure square-token body or CNN stem + attention blocks?

Start pure square-token. Test CNN hybrid later.

Reason:

```text
pure SquareFormer isolates the BT4/Chessformer hypothesis
and keeps move/input parity simpler.
```

The pure square-token model answers:

```text
Does chess-aware attention itself buy strength at tiny scale?
```

A CNN stem answers a different question:

```text
Can local convolutional features improve tiny transformer sample efficiency?
```

That is worth testing, but it should be a later controlled comparison.

I would test three models at equal parameter/latency budgets:

```text
A. pure TinyBT-static
B. small CNN / MoveFormer baseline
C. CNN-stem + 2–4 transformer blocks
```

A good hybrid is:

```text
input planes or square features
→ tiny local conv stem, 1–2 blocks
→ 64 square embeddings
→ 2–4 chess-attention blocks
→ policy/WDL/AV heads
```

The uploaded Chessformer notes already recommend comparing SquareFormer directly against `64×6`-style conv baselines rather than assuming transformer dominance. 

## 5. Are moves-left/horizon heads useful at small scale?

Use them as **auxiliary heads**, not as a core v1 decision factor.

Moves-left can help:

```text
value calibration
drawishness / conversion
search horizon awareness
time-to-resolution features
mate/endgame-like positions
```

But at tiny scale it can also distract capacity from policy/value unless the labels are clean.

My recommendation:

```text
E4 should be cheap and optional:
  add moves-left / horizon as an auxiliary
  measure WDL calibration and PUCT strength
  do not judge only by policy accuracy
```

Keep it if it improves:

```text
WDL calibration
value bucket calibration
PUCT Elo
endgame suite
resignation/draw behavior
```

Drop or downweight it if it hurts:

```text
policy quality
AV ranking
latency
training stability
```

## 6. For strength per byte/node, prioritize raw policy quality or action-value/reranking heads?

Prioritize in this order:

```text
1. raw policy quality
2. WDL/value calibration
3. action-value/reranking
4. conditional PUCT
```

But once policy is “good enough,” action-value is probably the highest ROI.

Why policy first:

```text
policy is needed for:
  policy-only mode
  top-k candidate generation
  PUCT priors
  self-play search targets
```

Why action-value next:

```text
policy says “this move is plausible”
action-value says “this move survives consequences”
```

That is exactly what your queen-suicide issue exposed. The infra gap analysis identifies the missing capability as:

```text
candidate move → opponent reply → consequence
```

and recommends action-value/opponent-reply/blunder-risk style modeling rather than relying only on plain policy imitation. 

So the practical answer:

```text
For TinyBT-static:
  optimize raw policy + WDL first.

For TinyBT-AV-PUCT:
  optimize action-value/regret because it improves strength per node and catches policy blunders.

For self-play:
  policy and WDL remain mandatory because PUCT needs them.
```

---

# Recommended experiment ordering

You listed:

```text
E0: h2 SquareFormer control
E1: h2 + static chess-relation bias
E2: smolgen-lite dynamic attention bias
E3: h7/h8 history cache
E4: moves-left/horizon auxiliary head
E5: medium d256 research model
E6: side-to-move token-order flip
```

I would reorder slightly into a clean sparse-factorial plan.

## Phase 1: establish the representation baseline

```text
E0: h2 SquareFormer control
E1: h2 + static chess-relation bias
```

Decision:

```text
If E1 does not clearly beat E0,
fix relation design before testing smolgen.
```

Metrics:

```text
policy KL
top-k accuracy
weighted tau if available
policy-only Elo
AV/top-k rerank Elo if available
latency
attention-map sanity
```

Expected result:

```text
E1 should win.
If it does not, the relation classes or implementation may be wrong.
```

## Phase 2: test dynamic topology only after static topology wins

```text
E2a: template-gated smolgen-lite
E2b: low-rank dynamic bias
E2c: dynamic bias only in later layers
```

Decision:

```text
Keep dynamic bias only if it improves Elo/latency frontier,
not just validation loss.
```

The Lc0 post suggests smolgen is valuable at large scale, but it costs throughput; in a tiny model, it must prove that it beats simply increasing d_model or adding an action-value head. ([Leela Chess Zero][2])

## Phase 3: test state/history features

```text
E3: h7/h8 history cache
```

Run with the best Phase 1 architecture:

```text
static bias winner
or smolgen-lite winner if clearly better
```

Measure by position class:

```text
openings
tactics
repetition/rule-state positions
en passant/castling edge cases
endgames
```

History may improve rule-awareness and tactical memory, but it also increases encoding complexity. The Chessformer handoff notes that full 112-feature history is closer to lc0/CF, while current-board-only is easier for browser/data work. 

## Phase 4: add auxiliary horizon/value heads

```text
E4: moves-left/horizon auxiliary
```

Do not treat E4 as an architecture replacement. Treat it as:

```text
a training regularizer / calibration test
```

Keep if it improves:

```text
value calibration
PUCT strength
endgame/horizon tests
```

## Phase 5: scale only the best tiny recipe

```text
E5: d256 medium research model
```

Do not scale a weak recipe.

Run E5 only after answering:

```text
static vs control
dynamic vs static
h2 vs h7/h8
aux head yes/no
```

E5 should answer:

```text
Does the approach scale smoothly?
Where is the parameter/latency frontier?
```

## Phase 6: side-to-move flip as a guarded ablation

```text
E6: side-to-move token-order flip
```

Run this with strict parity tests. Treat it as:

```text
potential strength gain
but high bug risk
```

The queen-safety and tactical-verification notes strongly emphasize plumbing-first diagnostics: move map, legal mask, side-to-move/value perspective, search backup, backend parity, and deterministic reproduction should be proven before model changes are trusted. 

---

# Concrete model ladder

Approximate parameter counts below are rough, assuming dense transformer blocks, relation biases, policy/WDL/moves-left-style small heads, and FFN ratio near 1×. They are useful for planning, not exact accounting.

| Model             | Layers | d_model | Heads |  FFN | Rough params | Target                 |
| ----------------- | -----: | ------: | ----: | ---: | -----------: | ---------------------- |
| TinyBT-96x4       |      4 |      96 |     4 |   1× |    ~0.6–0.8M | WASM/debug/instant     |
| TinyBT-128x6      |      6 |     128 |   4–8 |   1× |    ~1.1–1.4M | first serious tiny     |
| TinyBT-128x6-smol |      6 |     128 |   4–8 |   1× |    ~1.3–1.8M | dynamic bias test      |
| TinyBT-192x8      |      8 |     192 |     8 |   1× |    ~2.5–3.5M | balanced WebGPU        |
| MiniBT-256x8      |      8 |     256 |     8 |   1× |        ~4–6M | desktop/WebGPU         |
| MiniBT-256x10     |     10 |     256 |     8 | 1.5× |        ~5–8M | research actor/student |

These are all tiny compared with BT4’s public `1024 × 15` competition net. ([Leela Chess Zero][1])

---

# What I would actually build next

Given your V0/V1 experiments already look promising, I would not jump straight to full smolgen or d256. I would do this:

```text
1. Lock a clean TinyBT-128x6-static baseline.
2. Run E1 static relation bias vs control.
3. Add value bucket head if not already present.
4. Add action-value head on top-k candidate moves.
5. Only then test smolgen-lite.
6. Then scale to d192/d256.
```

The reason is strategic:

```text
Static bias tests representation.
Action-value tests move consequence modeling.
Smolgen tests dynamic attention.
Scaling tests capacity.
```

If you test all at once, you will not know what caused the gain.

---

# Best smolgen-lite design for your plan

I would implement this one first:

```text
Global summary:
  g = MLP(mean_pool(tokens) + side/rule summary)

Relation templates:
  T_r[64, 64] fixed binary/float masks

Per layer/head coefficients:
  α = Linear(g) → [layers, heads, num_relations]

Attention:
  scores[l,h,i,j] =
      QK[l,h,i,j] / sqrt(d)
    + static_bias[l,h,i,j]
    + Σ_r α[l,h,r] * T_r[i,j]
```

Why this is the best first version:

```text
tiny parameter cost
easy to inspect
hard to overfit
easy to export
compatible with WebGPU/ONNX
keeps chess topology explicit
```

Then test low-rank dynamic bias if template-gating helps but seems limited:

```text
scores += U[l,h,i,k] V[l,h,j,k]
```

with:

```text
rank k = 2 or 4 initially
```

---

# Best relation templates

Start with about 16–32 relation classes, not hundreds.

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

You can add occupancy-conditioned dynamics later through smolgen-lite rather than static relation explosion.

---

# Where action-value fits into the small-BT ladder

I would add action-value earlier than full PUCT maturity.

A good TinyBT-AV structure:

```text
trunk output:
  H[64, d]

global:
  g = pool(H)

move embedding:
  h_from = H[from]
  h_to = H[to]
  promo = embedding[promotion_type]
  move_type = embedding[normal/capture/check/promo/castle optional]

action_feature:
  z = MLP([g, h_from, h_to, h_from * h_to, promo, move_type])

outputs:
  av_bucket(move)
  regret(move)
  optional refutation/uncertainty(move)
```

Candidate set during training:

```text
teacher top-k
student top-k
checks/captures/promotions
played move
random legal distractors
known hard negatives
```

This is the clean way to handle queen blunders without baking queen-specific rules into the architecture. The queen-safety doc frames queen failures as tactical-verification/action-value failures rather than rules problems. 

---

# How to judge the experiments

Do not optimize a single metric. Use four classes.

## 1. Static model quality

```text
policy KL/CE
top-1/top-3/top-5 teacher agreement
WDL Brier/calibration
value bucket CE
attention-map sanity
```

## 2. Move ranking quality

```text
Kendall tau over legal moves
weighted tau over top moves
pairwise ranking accuracy
average regret
p95/p99 regret
catastrophic regret rate
```

This is crucial for AV heads.

## 3. Engine quality

```text
policy-only Elo
AV-top8 Elo
child-value-rerank Elo
PUCT32/64/128 Elo
conditional-search Elo
```

## 4. Deployment quality

```text
ONNX FP16 size
INT8 size if applicable
WebGPU p50/p95 latency
WASM latency
evals/sec
memory
fallback rate
Elo per latency/size frontier
```

The right “best model” is the one on the size/latency/strength Pareto frontier, not necessarily the biggest or the highest raw policy accuracy.

---

# What success looks like for each experiment

| Experiment       | Keep it if…                                                   | Drop or defer if…                            |
| ---------------- | ------------------------------------------------------------- | -------------------------------------------- |
| E1 static bias   | improves policy/Elo with negligible latency                   | no gain or attention maps nonsensical        |
| E2 smolgen-lite  | improves hard-position Elo/regret more than equivalent params | only improves loss, hurts latency            |
| E3 h7/h8 history | improves rule/repetition/tactical/endgame splits              | adds complexity with little strength         |
| E4 moves-left    | improves WDL calibration/PUCT/endgame                         | distracts policy or adds no engine gain      |
| E5 d256          | scaling curve stays healthy                                   | smaller AV/search model beats it per ms      |
| E6 side flip     | improves symmetry and Elo with perfect parity                 | any mapping/perspective flakiness appears    |
| AV head          | improves regret and AV-rerank Elo                             | action labels are too noisy or head overfits |
| PUCT             | improves at visits sweep                                      | value is too noisy; fix value/AV first       |

---

# My direct answers as decisions

```text
1. Static bias vs smolgen:
   static first; smolgen-lite second; full smolgen never first.

2. Smolgen simplification:
   dynamic relation-template gates first,
   low-rank dynamic bias second,
   full 64×64 generation only for research models.

3. Side-to-move flip:
   potentially worth it, but guarded by strict parity tests;
   defer until baseline is stable.

4. Pure square-token vs CNN stem:
   pure SquareFormer first;
   CNN hybrid as a controlled later baseline.

5. Moves-left/horizon:
   training auxiliary, not core v1;
   keep only if it improves value/search/endgame calibration.

6. Policy vs action-value:
   policy first because everything depends on it;
   action-value next because it gives the best strength-per-node gain once policy is decent.
```

---

# The final unified roadmap

```text
Stage A — lock TinyBT-static:
  h2 control
  static chess relation bias
  policy + WDL/value bucket
  no smolgen yet

Stage B — prove ranking:
  add action-value/regret head
  train on teacher top-k + student top-k + hard negatives
  evaluate AV-top8 and regret

Stage C — test dynamic topology:
  add smolgen-lite template gates
  compare against simply scaling d_model or adding heads

Stage D — improve state:
  h7/h8 history
  moves-left/horizon auxiliary
  optional side-to-move flip only after parity tests

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
  INT8/QAT CPU/WASM
  size/latency/Elo Pareto frontier
```

The self-play/scaling notes fit naturally at Stage F: keep the AlphaZero interface of policy + value, use PUCT to create improved policy/value targets, store rich search statistics including visits/Q/PVs, and then distill those back into the next network.  The DeepSeek-inspired ideas fit mostly into Stages B, F, and G: multi-teacher/on-policy distillation, multi-ply auxiliary heads, deterministic evaluation, and deployment-aware quantization. 

My strongest recommendation is therefore:

```text
Do not build “TinyBT-smolgen” before “TinyBT-AV”.
```

For tiny engines, **action-value/reranking is likely to buy more practical strength per parameter than dynamic attention** once static chess relation bias is in place. Smolgen-lite is still worth testing, but it should compete against the actual alternatives:

```text
+ more d_model
+ action-value head
+ value bucket head
+ better history features
+ conditional PUCT
```

The “small BT4” analogue is best understood as:

```text
square-token chess topology
+ compact transformer trunk
+ from-to policy
+ calibrated value
+ action-value consequence modeling
+ optional dynamic attention
+ conditional search
```

not simply:

```text
BT4, but smaller.
```

[1]: https://lczero.org/play/networks/bestnets/?utm_source=chatgpt.com "Best Networks | Leela Chess Zero"
[2]: https://lczero.org/blog/2024/02/transformer-progress/?utm_source=chatgpt.com "Transformer Progress | Leela Chess Zero"
