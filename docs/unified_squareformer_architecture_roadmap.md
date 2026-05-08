# Unified SquareFormer Architecture Development Roadmap

## Purpose

This document consolidates the project’s overlapping architecture notes into one development roadmap for a lightweight neural chess engine. It assumes **SquareFormer V0 and V1 have already been implemented and look promising**.

The goal is **not** to build a purely searchless model. The goal is:

```text
strongest lightweight neural chess engine per millisecond
```

That means:

```text
Use the model to amortize as much search as possible.
Use explicit search only where it buys strength.
Keep the engine portable, debuggable, and deployable.
```

The unified direction is:

```text
SquareFormer-AV-PUCT
  = square-token transformer
  + chess-aware attention
  + from-to policy head
  + WDL/value heads
  + action-value reranking
  + tactical/uncertainty heads
  + conditional PUCT
  + multi-teacher/on-policy distillation
```

---

## Guiding Principles

### 1. Do not optimize for “searchless” as an ideology

Searchless play is a useful inference mode, but not the final objective.

The right architecture should support a compute ladder:

```text
Mode 0: policy-only
Mode 1: policy + action-value rerank
Mode 2: top-k child verification
Mode 3: small PUCT, e.g. 32–128 nodes
Mode 4: deeper/offline teacher reanalysis
```

The model should learn to make Modes 0–2 strong enough that Mode 3 is needed only for sharp or uncertain positions.

### 2. Keep the AlphaZero/lc0 interface alive

Every serious model version should expose:

```text
position -> policy over legal moves
position -> WDL / value
```

This keeps the door open to:

```text
PUCT search
self-play policy improvement
search-improved policy targets
lc0-style training loops
```

### 3. Separate model strength from engine maturity

A promising model can still fail because of:

```text
move-map bugs
legal-mask bugs
side-to-move perspective bugs
value sign bugs
weak PUCT calibration
browser/export parity problems
queen/tactical verification gaps
```

Before architecture complexity, keep hardening the plumbing.

### 4. Treat tactics failures as training/verification problems, not rules problems

Do not ban queen sacrifices. Instead, distinguish:

```text
sound queen sacrifice
vs.
queen loss without compensation
```

This requires:

```text
action-value labels
opponent-reply modeling
after-reply value
regret / blunder-risk heads
uncertainty-gated verification search
```

### 5. Use multi-teacher distillation, then on-policy distillation

The strongest lightweight model should learn from a committee:

```text
Stockfish         -> tactics, concrete calculation, material conversion
lc0               -> policy priors, strategic value, Leela-like search
Syzygy/tablebase  -> exact endgame truth
Maia/human data   -> human-likeness, rating/style modes
self-play PUCT    -> behavior under our actual deployment search
puzzles/openings  -> targeted tactical and repertoire competence
```

Then let the student generate positions, mine the hard ones, and re-label them.

---

## Architecture North Star: SquareFormer-AV-PUCT

### Input

Use a chess-native square-token representation:

```text
64 tokens, one per board square
```

Two input tracks should remain available:

```text
Compatibility input:
  112 features per square
  current + history/rule features
  closest to Chessformer/lc0-style representation

Portable input:
  18–32 current-board features per square
  plus castling, en passant, rule-50, repetition bits
  easier to encode and deploy
```

Default recommendation:

```text
V2+ strength track: 112-feature compatibility input
V2+ portable ablation: 18–32 feature current-board input
```

### Trunk

Start from the promising V1 trunk and scale cautiously:

```text
V1-ish:   d_model 128, layers 4–6, heads 4–8
V2:       d_model 128–192, layers 6–8, heads 4–8
V3+:      d_model 192–256, layers 8–10, heads 8
```

Attention should remain chess-aware:

```text
attention_score[h, i, j] =
    Q[h, i] · K[h, j] / sqrt(d_head)
  + relation_bias[h, relation(i, j)]
```

Relation classes:

```text
same square
same rank
same file
same diagonal
same anti-diagonal
knight move
king move
white pawn attack
black pawn attack
rook ray distance bucket
bishop ray distance bucket
queen ray
same color complex
center / edge / corner relation
```

Do not prioritize MoE, GLU, long-context attention, or custom low-bit attention kernels before V3/V4.

### Core heads

Every serious version should have:

```text
policy_head:
  from-square × to-square logits
  promotion logits
  illegal-move masking

wdl_head:
  win / draw / loss probabilities

q_bucket_head:
  categorical value buckets, e.g. 32 or 64 bins
```

### V2+ heads

Add:

```text
action_value_head:
  score candidate legal moves

regret / ranking objective:
  teacher_best_value - candidate_value

uncertainty_head:
  predict when policy/value/action-value is unreliable
```

### V3+ tactical heads

Add selectively:

```text
queen_lost_after_reply
material_delta_after_reply
tactical_refutation_probability
after_reply_wdl / after_reply_value_bucket
sacrifice_soundness
opponent_reply_policy
```

### Training-only auxiliary heads

Useful but optional:

```text
reply move / reply policy
PV first 2–4 plies
policy-after-move
future value trajectory
move metadata: from, to, moved piece, captured piece, check flag
moves-left
legal-move auxiliary
```

These should shape the trunk, not bloat inference.

---

## Version Roadmap

## V0 — Debug SquareFormer

### Status

Already experimented with and promising enough to move on.

### Purpose

V0 should remain a permanent regression/debug baseline.

### Typical shape

```text
64 square tokens
small d_model, e.g. 64
4 transformer layers
policy + WDL heads
learned square embedding or simple relation bias
```

### What V0 proves

```text
tokenization works
legal masking works
from-to policy head works
promotion handling works
ONNX/export works
policy-only inference behaves sanely
```

### Do not keep scaling V0

Only return to V0 for:

```text
ablation
unit tests
fast export checks
attention visualization sanity
```

---

## V1 — Promising SquareFormer Baseline

### Status

Already experimented with and promising. Treat it as the first serious baseline.

### Purpose

V1 establishes whether square-token attention is genuinely useful at tiny scale.

### Expected shape

```text
64 square tokens
112 or compact features
4–6 layers
d_model around 128
chess relation bias
policy + WDL + optional q/value bucket
```

### V1 inference modes

```text
policy-only
top-k child-value rerank
small PUCT experiments
```

### V1 gates before moving to V2

Do not judge V1 only by games. Establish:

```text
policy top-k quality
WDL calibration
value sign/perspective correctness
PUCT visits=1 parity with policy-only
export/backend parity
queen-safety regression baseline
policy-only vs top-k rerank vs PUCT strength
```

### V1 decision

If V1 is promising, **freeze it as Baseline A**.

Then build V2 by changing one major axis:

```text
same trunk family
add action-conditioned heads
```

This prevents losing the signal in architecture churn.

---

## V1.5 — Plumbing and Engine Hardening Gate

Before V2, harden the infrastructure. This is not optional.

### Required tests

```text
moveToActionId / actionIdToMove round trips
promotion and underpromotion mapping
castling mapping
legal mask exactness
root prior equals evaluator top-k
side-to-move canonicalization invariants
WDL perspective tests
PUCT value sign flip tests
terminal mate/stalemate handling
browser/Node/Python backend parity
FP32/FP16/ONNX parity
```

### Required diagnostics

```text
queen-suicide failure capture
fixed queen-safety suite
policy-only vs PUCT visits sweep
value calibration by bucket
teacher regret on selected moves
```

### Exit criteria

```text
No known policy-map or perspective bugs.
PUCT visits=1 matches policy-only.
Value backup tests pass.
Browser/backend outputs match within tolerance.
Queen blunders are classified as model/search failures, not plumbing failures.
```

---

## V2 — SquareFormer-AV: Action-Value Reranker

### Purpose

V2 is the most important next model version.

It should answer:

```text
Can the model rank candidate moves, not merely imitate plausible moves?
```

### Architecture

Keep the V1 trunk as much as possible, then add an action-value path.

Recommended action feature:

```text
h_global = pooled board embedding
h_from   = square embedding[from]
h_to     = square embedding[to]
e_move   = promotion / move-type embedding
optional tactical features

move_feature = MLP([h_global, h_from, h_to, e_move, optional_features])
```

Heads:

```text
policy_logit(move)
action_value_bucket(move)
action_wdl(move), optional
regret_score(move), optional
```

### Candidate set for training

Do not label every legal move initially. Label a useful subset:

```text
teacher top-k moves
student policy top-k moves
played human/self-play move
checks
captures
promotions
queen moves / queen-risk moves
random legal distractors
known tactical blunders
sound sacrifices
```

### Targets

```text
teacher value after candidate move
teacher value after best reply
PUCT Q, if self-play/search data is available
regret = teacher_best_value - teacher_candidate_value
value bucket / WDL bucket
```

### Loss

```text
L =
  1.00 * policy_KL
+ 1.00 * WDL_CE
+ 0.50 * q_bucket_CE
+ 1.00 * action_value_CE_or_MSE
+ 0.25 * pairwise_ranking_loss
```

### V2 inference

```text
1. Run trunk once.
2. Use policy to get top-k legal moves, e.g. k=8 or 16.
3. Rerank with action-value.
4. Play best if no uncertainty/tactical trigger fires.
```

### V2 gates

V2 must beat V1 in at least one default fast mode:

```text
policy + AV rerank > policy-only
policy + AV rerank > top-k child-value rerank
```

Track:

```text
Kendall tau over legal move rankings
pairwise ranking accuracy
teacher regret of selected move
queen blunder rate
sound sacrifice preservation
policy-only Elo
AV-rerank Elo
latency overhead
```

### Decision

If AV rerank is stronger and not too slow, V2 becomes the new default model family.

---

## V3 — Tactical and Uncertainty SquareFormer

### Purpose

V3 makes the model robust. It targets the observed failure mode:

```text
plays well, then misses a one- or two-ply tactical refutation
```

### Add heads

```text
uncertainty / value-error
queen_lost_after_reply
material_delta_after_reply
after_reply_value_bucket
tactical_refutation_probability
sacrifice_soundness
opponent_reply_policy
```

### Multi-ply auxiliary training

DeepSeek-style multi-token prediction maps to chess as multi-ply prediction.

Use:

```text
main policy:
  what should I play now?

reply policy:
  what is the opponent’s best/likely reply?

after-reply value:
  what is the position worth after the opponent reply?

PV-lite:
  first 2–4 teacher/search PV moves, mostly on tactical data

regret:
  how costly is this candidate move?
```

Suggested loss:

```text
L =
  1.00 * policy_KL
+ 1.00 * WDL_CE
+ 0.50 * q_bucket_CE
+ 1.00 * action_value_loss
+ 0.25 * reply_policy_CE
+ 0.25 * after_reply_value_CE
+ 0.25 * queen_loss_BCE
+ 0.25 * pairwise_regret_loss
+ 0.10 * uncertainty_calibration
+ 0.10 * PV_lite_loss_on_tactics
```

### Runtime

V3 introduces conditional verification:

```text
if low uncertainty and no tactical risk:
  policy + AV rerank

if high uncertainty or tactical risk:
  top-k child verification or 32–128 node PUCT

if tablebase available:
  exact tablebase
```

### V3 data

Use a training mix like:

```text
70% normal policy/value/action-value data
10% model-mined blunders
10% sound sacrifices / tactical motifs
5% puzzles / forcing lines
5% tablebase/endgame positions
```

### V3 gates

```text
queen blunder rate decreases sharply
sound queen sacrifice preservation remains high
uncertainty correlates with regret/search error
conditional search improves Elo per millisecond
AV + tactical heads outperform V2 on puzzle/tactical suites
```

---

## V4 — Search-Improved Self-Play and On-Policy Distillation

### Purpose

V4 turns the model from a supervised imitator into part of a policy-improvement loop.

### Loop

```text
current model
  -> policy/AV/PUCT self-play games
  -> store search visits, Q, WDL, PV, result
  -> train next model on search-improved targets
  -> evaluate candidate vs current best
  -> promote only if stronger
```

### Self-play actor

Use compute tiers:

```text
cheap actor:
  policy + AV + 32-node PUCT

hard-position actor:
  128–512-node PUCT

teacher reanalysis:
  Stockfish / lc0 / tablebase on mined hard positions
```

### Store rich records

Every self-play position should store:

```text
FEN / encoded position
legal moves
raw policy
search visit distribution
search Q per candidate
root WDL/value
played move
game result
PV line
policy entropy
uncertainty
nodes / temperature / net id
```

### Training targets

```text
policy_head:
  normalized search visits

WDL/value:
  final game result and/or root WDL

action_value_head:
  search Q for candidate moves

reply/multi-ply heads:
  search PV / reply policy / after-reply value

uncertainty:
  model-search disagreement, value error, regret
```

### Replay buffer mix

```text
50% recent accepted-net self-play
20% hard mined positions
10% teacher reanalysis positions
10% tablebase/endgame curriculum
10% opening/puzzle/tactical curated data
```

### Promotion gates

```text
candidate vs current best
fixed openings
fixed search budgets
both colors
sufficient games
track Elo, regret, queen safety, value calibration, latency
```

### V4 gates

```text
checkpoint N+1 beats checkpoint N at fixed policy/AV mode
checkpoint N+1 beats checkpoint N at small-PUCT mode
no value calibration collapse
no queen/tactical regression
```

---

## V5 — Deployment and Compression Track

This should run in parallel, but becomes mandatory once V2/V3 are strong.

### Targets

```text
FP16 ONNX WebGPU primary
WASM/CPU fallback
INT8 PTQ baseline
INT8 QAT fine-tuning if PTQ hurts strength
single-file or cache-friendly model packaging
```

### Metrics

```text
model file size
load time
evals/sec
p50/p95 move latency
memory
battery/mobile behavior
policy KL drift after quantization
WDL calibration drift
action-value ranking drift
fixed-node Elo drift
```

### TurboQuant-style ideas

Do not apply TurboQuant to the model forward pass early.

Potential later use:

```text
compressed position embeddings
semantic transposition memory
teacher-feature dataset compression
opening/memory retrieval
```

Core deployment compression should remain:

```text
FP16 first
INT8/PTQ second
INT8/QAT third
```

---

## V6+ Advanced Research Lanes

These are not immediate next steps. They become useful after V2/V3 are stable.

## Lane A — Conv-SquareFormer Hybrid

```text
small conv stem or SE-ResNet trunk
-> 64 square embeddings
-> 2–6 attention blocks
-> policy/WDL/AV heads
```

Hypothesis:

```text
conv handles local board motifs
attention handles global piece relations
```

Compare against:

```text
pure SquareFormer
64×6 conv baseline
```

## Lane B — NNUE + SquareFormer Hybrid

```text
fast NNUE-lite path for cheap eval/search
SquareFormer policy/value advisor at root/PV/uncertain nodes
```

Most promising for raw strength if building a more classical engine stack.

Use SquareFormer as:

```text
root move-ordering advisor
selected-node verifier
teacher for NNUE distillation
uncertainty/tacticality scorer
```

## Lane C — Iterative SquareFormer

```text
shared reasoning block repeated K times
adaptive K based on uncertainty
heads after each iteration or final iteration
```

Hypothesis:

```text
internal iterative compute may learn shallow lookahead
```

Not v1/v2. Harder to export and tune.

## Lane D — Memory / Semantic Transposition

```text
position -> embedding
retrieve similar analyzed positions
blend retrieved policy/value as hint
```

Use only as a hint, not an oracle. Chess similarity is dangerous.

TurboQuant-like compression may become useful here.

## Lane E — Personality / Adapter Track

```text
base strength model
+ persona/style embeddings or small adapters
+ style-aware opening books
+ style scoring in move selection
```

Keep separate from the pure strength track.

## Lane F — Optimizer / Residual Experiments

```text
AdamW baseline
Muon experiment for matrix weights
simple 2-stream residual mixing for deeper models
mHC-inspired residuals only if deep models need stability
```

Low priority before V3.

---

## Unified Evaluation Framework

Every architecture version should be evaluated on the same ladder.

### Static prediction

```text
policy CE / KL
policy top-1 / top-3 / top-5
WDL calibration
q bucket accuracy
Brier score
```

### Move ranking

```text
Kendall tau over legal moves
pairwise ranking accuracy
top-k regret vs teacher
action-value bucket accuracy
```

### Tactical robustness

```text
queen blunder rate
sound queen sacrifice preservation
queen-winning tactic accuracy
puzzle accuracy
mate suite
material swing suite
```

### Search behavior

```text
policy-only Elo
state-value rerank Elo
AV-rerank Elo
PUCT 16/32/64/128 Elo
conditional-search Elo
PUCT visit sweep
value usefulness under search
```

### Engine robustness

```text
illegal move count
move-map parity
browser/backend parity
terminal handling
repetition/rule-state handling
export precision drift
```

### Efficiency

```text
model size
load time
evals/sec
latency by inference mode
Elo per millisecond
fallback frequency
memory use
```

### Self-play improvement

```text
candidate vs previous checkpoint
promotion pass/fail
fixed opening gauntlets
policy/search disagreement trend
value calibration trend
failure mining yield
```

---

## Priority Stack

## Immediate priorities

```text
1. Freeze V1 as baseline.
2. Complete V1.5 hardening: policy map, value perspective, PUCT, export parity.
3. Build V2 action-value head on V1 trunk.
4. Add action-value data pipeline and ranking/regret metrics.
5. Maintain queen-safety regression suite.
```

## Next priorities

```text
6. Add tactical/uncertainty heads.
7. Add multi-ply auxiliary targets.
8. Add conditional PUCT fallback.
9. Mine on-policy hard failures and re-label them.
10. Start shallow self-play only after V2/V3 are stable.
```

## Later priorities

```text
11. Actor/student self-play split.
12. INT8/QAT deployment.
13. Conv-SquareFormer and NNUE hybrid experiments.
14. Iterative SquareFormer.
15. Memory/TurboQuant/personality adapters.
```

---

## Recommended Next 8 Milestones

### Milestone 1 — V1 freeze report

Deliver:

```text
V1 architecture card
training data card
policy-only metrics
value calibration
PUCT sweep
known failure list
browser/export parity
```

### Milestone 2 — plumbing gate

Deliver:

```text
move-map test suite
side-to-move/value perspective tests
PUCT backup tests
queen failure capture harness
fixed queen/tactics suite
```

### Milestone 3 — AV labeling pipeline

Deliver:

```text
candidate sampler
teacher action-value labels
regret labels
sparse storage format
Kendall/pairwise ranking eval
```

### Milestone 4 — V2 model

Deliver:

```text
V1 trunk + action-value head
policy + WDL + q bucket + AV losses
AV-rerank inference mode
```

### Milestone 5 — V2 evaluation

Deliver:

```text
policy-only vs AV-rerank vs child-value rerank
queen suite
puzzle/tactics suite
latency
teacher regret
```

### Milestone 6 — tactical heads

Deliver:

```text
queen_lost_after_reply
after_reply_value
opponent reply policy
uncertainty head
sound-sacrifice contrastive data
```

### Milestone 7 — conditional search

Deliver:

```text
uncertainty-gated PUCT
fallback precision/recall
Elo per millisecond
search budget policy
```

### Milestone 8 — shallow self-play pilot

Deliver:

```text
32-node PUCT self-play workers
rich game record format
promotion gate
first N -> N+1 improvement attempt
hard-position reanalysis queue
```

---

## Final Recommendation

The unified roadmap should converge on this architecture family:

```text
SquareFormer-AV-PUCT
```

with this staged plan:

```text
V0/V1:
  prove square-token policy/value works

V1.5:
  harden plumbing and diagnostics

V2:
  add action-value and regret ranking

V3:
  add tactical, uncertainty, and multi-ply auxiliary heads

V4:
  add search-improved self-play and on-policy specialist distillation

V5:
  optimize deployment and quantization

V6+:
  explore hybrids, memory, iterative compute, and personality/adapters
```

The key architectural bet is:

```text
A small square-token transformer can learn strong chess priors,
while action-value and multi-ply auxiliary heads teach it tactical consequences,
and conditional PUCT supplies strength only when the model needs external search.
```

The key engineering bet is:

```text
Better diagnostics and targeted labels will improve the model faster than simply scaling layers.
```

The key training bet is:

```text
multi-teacher on-policy distillation + search-improved self-play
will eventually outperform fixed supervised imitation,
while still keeping the deployed model lightweight.
```
