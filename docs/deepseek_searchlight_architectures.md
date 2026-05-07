# DeepSeek-Inspired + Search-Light Architectures for a Lightweight Neural Chess Engine

## Goal

The guiding objective is **not** to make a purely search-less chess model. The goal is to build the **strongest lightweight neural chess engine possible** under practical constraints: browser/portable deployment, modest training compute, and optional cheap distributed self-play.

The best synthesis is:

```text
Use the neural model to amortize as much search as possible,
then spend explicit search only where it buys strength.
```

In other words:

```text
common/easy positions:
  fast neural policy/action-value inference

sharp/uncertain positions:
  small PUCT, shallow verification, tablebase, or extra model iterations
```

This combines the search-light transformer design direction with DeepSeek-inspired ideas:

- **multi-teacher distillation** rather than pure self-play from scratch;
- **on-policy distillation** from positions the student actually reaches;
- **multi-ply auxiliary objectives**, analogous to multi-token prediction;
- **deployment-aware quantization training**;
- **fault-tolerant rollout infrastructure** for cheap/preemptible self-play workers;
- **conditional compute**, not a fixed inference budget for every move.

---

## Core Synthesis

### Search-light idea

A transformer can reduce explicit PUCT by learning outputs that search would normally provide:

```text
position → policy over legal moves
position → WDL / value
position + candidate move → action value
position → ranking of legal moves
```

This is amortized planning. The model absorbs search work during training, usually from a strong teacher such as Stockfish, lc0, tablebases, or a search-enhanced version of the student.

### DeepSeek-inspired idea

DeepSeek-style ideas are most useful at the **training-system level**, not because we should copy trillion-parameter MoE design. The transferable pattern is:

```text
specialist teachers
  → full-distribution / logits-level distillation
  → unified student
  → on-policy data from student behavior
  → quantization-aware deployment path
```

For chess, this maps cleanly:

```text
Stockfish teacher        → tactics, exact calculation, material conversion
lc0 teacher              → policy priors, strategic/Leela-like value
Syzygy/tablebase teacher → exact endgame truth
Maia teacher             → human-likeness, rating-conditioned style
self-play MCTS teacher   → behavior under our actual deployment search
opening/puzzle teachers  → specialized local competence
```

The final deployed model is small, but its labels come from a committee of much stronger or more specialized systems.

---

## The Main Principle

Do not ask whether the engine is “search-less” or “search-based.” Ask:

```text
Where is the cheapest place to spend computation for this position?
```

A strong lightweight engine should have **tiers of compute**:

```text
Tier 0: instant policy
Tier 1: policy + action-value rerank
Tier 2: one-ply or top-k child-value verification
Tier 3: small PUCT, 32–256 nodes
Tier 4: exact tablebase / deeper search only when available
```

The model’s job is to make Tier 0–2 strong enough that Tier 3 is only needed for the hard cases.

---

# Top Architecture Candidates

## 1. SquareFormer-AV-PUCT: Square-token transformer with action-value reranking and conditional PUCT

### Status

**Most promising default architecture.** This is the best synthesis of Chessformer-style architecture, searchless action-value learning, and DeepSeek-style multi-teacher distillation.

### High-level structure

```text
64 square tokens
  → chess-aware transformer trunk
  → policy head
  → WDL/value heads
  → action-value head
  → uncertainty/error head
  → optional multi-ply auxiliary heads during training
```

### Input

Use a board-native representation:

```text
64 tokens, one per square
piece identity
side-to-move canonicalization
castling rights
en passant
rule-50 / repetition features
optional last N plies or piece-history features
```

Two input tracks are worth testing:

```text
compatibility track:
  112 lc0/Chessformer-like features per square

portable track:
  current-board-only 18–32 features per square
```

The compatibility track is likely stronger. The portable track is easier to deploy, encode, and reason about.

### Trunk

Initial recommended sizes:

```text
micro:     d_model 64,  layers 4, heads 4
small:     d_model 128, layers 6, heads 4 or 8
balanced:  d_model 192, layers 8, heads 8
```

Use chess-aware attention:

```text
attention_score[h, i, j] =
    Q[h, i] · K[h, j] / sqrt(d_head)
  + relation_bias[h, relation(i, j)]
```

Relation classes can include:

```text
same square
same rank
same file
same diagonal
same anti-diagonal
knight move
king move
pawn attack
rook ray distance bucket
bishop ray distance bucket
same color complex
center/edge/corner relation
```

This keeps the model small while giving it chess geometry.

### Heads

Required:

```text
policy_head:
  from-square × to-square logits
  promotion logits
  illegal move mask

wdl_head:
  win/draw/loss probabilities

q_bucket_head:
  categorical value buckets, e.g. 32 or 64 bins
```

Highly recommended:

```text
action_value_head:
  score candidate legal moves using board embedding + move embedding

uncertainty_head:
  predict policy/value/action-value unreliability
```

Optional training-only heads:

```text
reply_move_head
pv_continuation_head
policy_after_move_head
value_after_best_reply_head
moves_left_head
legal-move auxiliary head
move metadata heads: from, to, moved piece, captured piece, check flag
```

### Inference

Default mode:

```text
1. Run model once.
2. Use policy to get top-k legal moves.
3. Use action-value head to rerank top-k.
4. Play best move if uncertainty is low.
```

Fallback mode:

```text
if uncertainty is high
or policy entropy is high
or action-value/policy disagree
or position is tactical:
    run small PUCT using policy + WDL
```

### Why this is strong

It gets most of the benefits of searchless play while keeping a search escape hatch:

```text
policy gives candidate moves
action-value head approximates teacher search
uncertainty decides whether to spend explicit search
PUCT is reserved for high-value positions
```

This should be stronger than pure policy-only play and much cheaper than always running PUCT.

---

## 2. Dual-Path NNUE + SquareFormer Hybrid

### Status

**Most promising if raw playing strength matters more than architectural purity.** This is a Stockfish-inspired route: make the common path extremely cheap, and call the transformer only when it can pay for itself.

### High-level structure

```text
fast path:
  NNUE-lite / sparse MLP evaluator
  cheap material/PSQT branch
  maybe small policy prior

slow path:
  SquareFormer policy/value/action-value advisor
  called at root, PV nodes, or uncertain positions
```

### Why it matters

Stockfish’s strength comes partly from doing huge amounts of cheap search. A transformer is too expensive to evaluate at every alpha-beta leaf, but it can provide powerful guidance:

```text
root move ordering
PV-node policy
candidate verification
uncertainty/tacticality scoring
training labels for NNUE student
```

### Possible inference modes

```text
alpha-beta or small search:
  NNUE-lite evaluates most leaves
  SquareFormer provides root/PV policy and selected value checks
```

or:

```text
PUCT-like search:
  SquareFormer at root and selected nodes
  NNUE-lite as cheap rollout/value approximation
```

### Research questions

```text
Can SquareFormer policy improve move ordering enough to offset its cost?
Can transformer values selectively correct NNUE blind spots?
Can transformer teacher labels distill into an NNUE-lite student?
What is the best gate for calling the transformer?
```

### Why it is risky

Integrating neural sidecars into alpha-beta is difficult. If the expensive model is called too often, speed collapses. If it is called in the wrong places, it can make pruning less stable.

### Best use

This is a second-stage architecture after the SquareFormer-AV baseline works.

---

## 3. Conv-SquareFormer Hybrid

### Status

**Most practical bridge between lc0-style conv nets and transformer models.**

### Structure

```text
board planes or square features
  → small convolutional stem
  → square-token projection
  → 2–6 transformer blocks
  → policy/WDL/action-value heads
```

or:

```text
small SE-ResNet trunk
  → flatten to 64 square embeddings
  → chess-aware attention blocks
  → heads
```

### Why it might work

Convolutions are strong at local board patterns:

```text
pawn structure
local king shelter
piece clusters
nearby tactical shapes
```

Attention is strong at global relations:

```text
diagonals
files
long-range pins
piece coordination
promotion races
```

A hybrid may outperform a pure transformer at the same parameter count, especially on tiny models.

### Research questions

```text
Does a conv stem reduce data requirements?
Does attention still learn useful chess relation heads after conv preprocessing?
Is hybrid faster or slower in WebGPU/ONNX than pure SquareFormer?
```

### Best use

Run as a baseline against pure SquareFormer and 64×6 conv.

---

## 4. Iterative / Recurrent SquareFormer

### Status

**Most interesting research architecture for internalized lookahead.** Not a v1 deployment choice.

### Structure

```text
position tokens
  → shared reasoning block repeated K times
  → heads at each iteration or final iteration
```

Example:

```text
h0 = embed(position)
for t in 1..K:
    h_t = reasoning_block(h_{t-1})
policy_t, value_t = heads(h_t)
```

K can be fixed or adaptive:

```text
easy position: K = 2
hard position: K = 8
```

### Relation to search

This is not PUCT, but it gives the model repeated internal computation. It is the closest searchless analogue to “thinking longer.”

### DeepSeek connection

DeepSeek-inspired residual/connection ideas, such as multi-stream residuals or mHC-like routing, are most relevant here. If we make the model deeper or iterative, signal propagation and stability matter more.

### Multi-ply heads fit naturally

At iteration `t`, supervise the model with progressively deeper targets:

```text
iteration 1:
  immediate policy/value

iteration 2:
  best reply awareness

iteration 3:
  two-ply value or PV continuation

final iteration:
  teacher search policy/action-value
```

### Risks

```text
harder to train
harder to export
latency less predictable
uncertainty/halting is nontrivial
```

### Best use

Research lane after the single-pass SquareFormer plateaus.

---

## 5. History/Memory-Augmented SquareFormer

### Status

**Most novel architecture lane.** Useful if we care about openings, repetition, personalization, and strategic memory.

### Structure

```text
64 board-square tokens
+ recent move tokens
+ compressed history summary tokens
+ optional retrieved memory tokens
```

Possible token layout:

```text
64 board tokens
8 recent-ply tokens
4 game-summary tokens
K retrieved memory tokens
```

### DeepSeek connection

DeepSeek-style compressed context is not needed for a 64-token board, but it becomes relevant if we represent game history or retrieval memory.

Chess adaptation:

```text
recent moves:
  full-resolution tokens

older history:
  compressed summary tokens

opening/memory database:
  retrieved sparse tokens
```

### Uses

```text
threefold repetition awareness
opening plan consistency
long-term strategic plans
user personalization
semantic transposition memory
style/personality continuity
```

### Risks

```text
retrieval mistakes are dangerous in chess
history tokens may add latency without Elo gain
repetition can be handled with explicit rules instead
```

### Best use

Not v1. Add after the base model is strong and we want differentiating features.

---

## 6. Specialist-Adapter Unified Student

### Status

**Most important training architecture.** The deployed model can be any of the above, but the training system treats specialists as teachers and adapters as optional modules.

### Structure

```text
base student trunk
+ optional small adapters / persona embeddings / domain embeddings
+ unified policy/value/action-value heads
```

Specialists:

```text
Stockfish tactical/deep teacher
lc0 strategic teacher
Syzygy endgame teacher
Maia human-style teacher
opening-book teacher
self-play MCTS teacher
```

Two ways to use specialists:

```text
1. Distill all into one base model.
2. Keep tiny adapters for modes/domains/personas.
```

For pure strength, prefer unified distillation. For controllable personalities, keep adapters.

---

# Multi-Token Prediction → Multi-Ply Auxiliary Heads

## The analogy

In language models, multi-token prediction asks the model to predict not only the next token, but several future tokens. The goal is to make the internal representation aware of near-future structure.

In chess, the analogous idea is:

```text
predict not only the next move,
but also replies, continuations, and future values.
```

This is **not** the same as asking the model to play a full line at inference. It is primarily a training signal that encourages the trunk to represent tactical and strategic consequences.

---

## Candidate multi-ply auxiliary heads

### 1. Opponent reply head

```text
input: current position
main target: best/current move
aux target: likely/best opponent reply after chosen move
```

Implementation options:

```text
A. condition reply head on teacher best move
B. condition reply head on sampled candidate move
C. predict reply distribution for top-k candidate moves
```

Most practical:

```text
policy top-k candidate moves
→ for each candidate, predict opponent best-reply policy/value
```

### 2. Two-ply value head

```text
position + candidate move
→ value after opponent best reply
```

This is a stronger action-value target:

```text
Q_1(move): value immediately after move
Q_2(move): value after best/likely reply
```

### 3. Policy-after-move head

```text
position + move
→ policy distribution in child position
```

This teaches the trunk how a move changes the candidate-move landscape.

### 4. Principal-variation head

```text
position → move_1, reply_1, move_2, reply_2, ...
```

Use soft labels from teacher PVs or MCTS most-visited lines. This is useful for tactical puzzles and UI explanations, but it should not dominate training because many valid lines exist.

### 5. Future WDL / value trajectory head

```text
position → value at ply +1, +2, +4, +8 under teacher line
```

This can help the model distinguish:

```text
quiet stable advantage
temporary tactical spike
forced conversion
unstable sacrificial attack
```

### 6. Regret / blunder head

```text
position + candidate move
→ teacher_best_value - teacher_value(candidate)
```

This is extremely useful for move ranking and training personalities. It tells the model not just which move is best, but how costly alternatives are.

### 7. Tactical forcing head

```text
position + move
→ is the line forcing?
→ number of only-move replies?
→ check/capture/threat sequence length?
```

This supports conditional search:

```text
if forcing score high and uncertainty high:
  spend more search
```

---

## Recommended multi-ply objective

Do not start with all heads. Use a compact auxiliary set:

```text
main:
  policy KL / CE
  WDL CE
  value bucket CE
  action-value bucket CE

multi-ply auxiliary:
  reply policy CE for top-k candidate moves
  two-ply action-value bucket CE
  regret/ranking loss
  optional PV first 2 plies for tactical data only
```

Example loss:

```text
L =
  1.00 * KL(policy_teacher || policy_student)
+ 1.00 * CE(WDL_teacher, WDL_student)
+ 0.50 * CE(q_bucket_teacher, q_bucket_student)
+ 1.00 * CE(action_value_bucket, AV_head)
+ 0.25 * CE(reply_policy_teacher, reply_head)
+ 0.25 * CE(two_ply_value_bucket, two_ply_head)
+ 0.25 * pairwise_ranking_loss(candidate_moves)
+ 0.10 * uncertainty_calibration_loss
```

For small models, keep auxiliary weights modest. The goal is representation shaping, not forcing the model to memorize brittle PVs.

---

## How to generate multi-ply labels cheaply

Full labeling of every legal continuation is expensive. Use selective labeling:

For each position:

```text
candidate set =
  teacher top-k moves
  student top-k moves
  human played move, if available
  checks/captures/promotions
  random legal distractors
```

For each candidate:

```text
ask teacher for:
  value after move
  best reply
  value after best reply
  maybe PV first 2–4 plies
```

Teacher choices:

```text
Stockfish shallow/deep for tactics
lc0 for policy/value distribution
Syzygy for endgames
student PUCT for on-policy behavior
```

This gives the model multi-ply supervision without full self-play scale.

---

## Why multi-ply heads help strength

They give a tiny model cheap “lookahead pressure” during training:

```text
single-ply policy:
  learn what move is likely/best

multi-ply auxiliary:
  learn what the opponent can do next
  learn which moves survive best replies
  learn which candidate moves are traps
  learn which positions require explicit search
```

This should improve both:

```text
searchless/top-k rerank play
small-PUCT efficiency
```

A better trunk means fewer PUCT nodes are needed to catch obvious tactical refutations.

---

## Risks of multi-ply heads

```text
PV ambiguity:
  many lines are equally valid

teacher inconsistency:
  Stockfish/lc0 may choose different continuations

label brittleness:
  one PV can change with search depth

overweighting tactics:
  model may become too forcing-line obsessed

latency creep:
  inference heads must be optional or cheap
```

Mitigations:

```text
use soft distributions where possible
use top-k candidates, not single forced labels everywhere
use auxiliary weights below main policy/value weights
apply PV losses mostly to tactical/puzzle subsets
keep training-only heads removable at export
```

---

# DeepSeek-Inspired Research Lanes

## Lane A: Multi-teacher on-policy distillation

### Hypothesis

A small model trained from a committee of specialists will outperform a small model trained from one teacher or raw human games.

### Pipeline

```text
1. Train base student on static teacher dataset.
2. Let student play/search positions under its actual deployment modes.
3. Identify uncertain, mistaken, or high-regret positions.
4. Route positions to specialist teachers.
5. Distill full legal-move distributions and values back into student.
```

### Teacher routing

```text
piece_count <= 7:
  Syzygy/tablebase

tactical/check-heavy/high material swing:
  Stockfish

strategic/quiet/high policy entropy:
  lc0

human/personality mode:
  Maia/human model

positions produced by our own PUCT:
  self-play search teacher
```

### Why it is promising

It focuses scarce labeling compute on the student’s actual weaknesses.

---

## Lane B: Conditional compute and uncertainty calibration

### Hypothesis

The strongest lightweight engine will be adaptive, not fixed-cost.

### Model outputs

```text
policy entropy
value uncertainty
action-value disagreement
teacher-disagreement prediction
blunder-risk prediction
```

### Runtime policy

```text
if confidence high:
  policy/action-value move

if confidence medium:
  top-k child-value verification

if confidence low:
  small PUCT

if exact endgame:
  tablebase
```

### Evaluation

Measure:

```text
Elo per millisecond
Elo per model evaluation
fallback frequency
calibration of uncertainty vs actual regret
```

---

## Lane C: Deployment-aware quantization training

### Hypothesis

Training with deployment precision in mind will preserve more strength than post-training quantization alone.

### Stages

```text
v1:
  train fp16/bf16, export FP16 ONNX for WebGPU

v2:
  INT8 PTQ for CPU/WASM baseline

v3:
  INT8 QAT / fake quantization during fine-tuning

v4:
  experiment with FP8/INT4 only if runtime kernels support it
```

### What to measure

```text
policy KL before/after quantization
WDL calibration drift
action-value ranking drift
fixed-node Elo
browser eval latency
```

---

## Lane D: Fault-tolerant cheap rollout infrastructure

### Hypothesis

Cheap/preemptible self-play is useful if every worker can die without corrupting the data distribution.

### Worker design

After every move, write:

```text
FEN
move
search policy / visit counts
WDL/Q
RNG seed
net ID
worker ID
clock/search config
```

Use a write-ahead-log style format:

```text
resume from last complete move
avoid bias toward short completed games
keep accounting for discarded partials
```

### Why it matters

If interrupted games are discarded, the dataset may bias toward short, simple, or decisive games.

---

## Lane E: Compressed history and semantic memory

### Hypothesis

A lightweight engine can gain practical strength from external memory, especially in openings, repeated positions, and personalized play.

### Memory item

```text
position embedding
compressed code
WDL/value
top-k policy
source/confidence
```

### Uses

```text
opening generalization
semantic transposition hints
personalized user model
style/persona continuity
teacher-data deduplication
```

### Caution

Chess is exact. Semantic similarity can be misleading. Use memory as a hint, not an oracle.

---

## Lane F: Optimizer and residual/connection experiments

### Hypothesis

For deeper transformer variants, optimizer and residual design may matter enough to improve strength per parameter.

### Experiments

```text
AdamW baseline
Muon for matrix weights + AdamW for embeddings/norms/heads
simple 2-stream residual mixing
mHC-inspired constrained mixing only if deeper models need it
```

### Priority

Low for v1. Medium for 8–16 layer models.

---

# Architecture Ranking

## Highest priority

### 1. SquareFormer-AV-PUCT

Best balance of strength, simplicity, and browser feasibility.

```text
square-token transformer
policy + WDL + action-value + uncertainty
conditional PUCT
multi-teacher distillation
```

### 2. Conv baseline and Conv-SquareFormer hybrid

Necessary for comparison and possibly better strength/latency at tiny sizes.

```text
64×6 SE-ResNet baseline
conv stem + attention blocks hybrid
```

### 3. Multi-ply auxiliary training

Not a separate deployed model, but likely high leverage for all models.

```text
reply prediction
two-ply value
regret/ranking
PV-lite for tactics
```

## Medium priority

### 4. NNUE + SquareFormer hybrid

Potentially strongest if we are willing to build a more engine-like search stack.

### 5. Iterative SquareFormer

High research interest; harder deployment.

### 6. History/memory augmented model

Novel and useful, but only after base strength is solid.

## Low priority / not now

```text
large sparse MoE
full long-context attention
custom FP4 kernels
full DeepSeek-style compressed attention for 64 board tokens
large PV-generating sequence model as primary player
```

---

# Recommended Roadmap

## Phase 0: Baseline sanity

```text
64×6 conv baseline
small SquareFormer baseline
policy + WDL only
policy-only and top-k child-value eval
```

Goal:

```text
verify data, move encoding, legal mask, export, eval harness
```

## Phase 1: SquareFormer-AV

```text
add action-value head
train on top-k teacher-labeled candidate moves
track Kendall tau / move ranking / regret
```

Goal:

```text
make top-k reranking stronger than policy-only and child-value rerank
```

## Phase 2: Multi-ply auxiliary heads

```text
reply policy
two-ply value
regret/ranking
PV-lite on tactical subset
```

Goal:

```text
improve tactical robustness and reduce need for PUCT
```

## Phase 3: Conditional search

```text
uncertainty head
calibrated fallback to 32–128 node PUCT
adaptive compute policy
```

Goal:

```text
maximize Elo per millisecond
```

## Phase 4: On-policy distillation loop

```text
student generates positions
teacher labels weaknesses
student fine-tunes
promotion gates via fixed-node matches
```

Goal:

```text
teach the model its own failure distribution
```

## Phase 5: Deployment compression

```text
FP16 WebGPU
INT8 CPU/WASM
QAT fine-tune
measure Elo/latency/file-size tradeoffs
```

Goal:

```text
portable engine with minimal strength loss
```

## Phase 6: Advanced lanes

```text
NNUE hybrid
iterative transformer
memory/history tokens
personality adapters
```

Goal:

```text
push beyond the straightforward model once the base system is strong
```

---

# Recommended Evaluation Suite

## Static prediction

```text
policy KL / CE
policy top-1/top-3/top-5
WDL calibration
value bucket accuracy
```

## Move ranking

```text
Kendall tau over legal moves
pairwise ranking accuracy
top-k regret vs teacher
action-value bucket accuracy
```

## Multi-ply auxiliary quality

```text
reply prediction accuracy
two-ply value accuracy
PV first-move/second-move accuracy on tactical subset
forcing-line detection
```

## Engine strength

```text
policy-only Elo
action-value rerank Elo
small-PUCT Elo
conditional-search Elo
puzzle accuracy
endgame suite accuracy
```

## Efficiency

```text
model size
evals/sec
latency by inference mode
Elo per millisecond
Elo per MB
fallback frequency
```

## Calibration

```text
uncertainty vs actual regret
fallback benefit vs cost
quantization drift
teacher disagreement prediction
```

---

# Final Recommendation

The most promising system is not pure searchless play and not full lc0-style search. It is:

```text
SquareFormer-AV-PUCT:
  64 square tokens
  chess-aware attention
  from-to policy head
  WDL/value bucket heads
  action-value top-k reranker
  multi-ply auxiliary heads during training
  uncertainty head
  conditional small PUCT fallback
  multi-teacher on-policy distillation
  deployment-aware quantization
```

The key idea is:

```text
Train the model to approximate expensive search,
then use explicit search only when the model knows it is unsure.
```

The top research bet is **multi-teacher action-value distillation plus multi-ply auxiliary training**. The top engineering bet is **conditional compute**. The top deployment bet is **FP16 WebGPU first, INT8/QAT later**.

If this works, the engine should be stronger than a pure policy model, faster than always-on PUCT, and much more practical than full-scale self-play RL.
