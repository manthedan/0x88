# Search-Less and Search-Light Transformer Designs for a Tiny Neural Chess Engine

## Purpose

This document summarizes abstract ways a transformer-based chess model can reduce or avoid explicit PUCT/MCTS search, then proposes a best synthesis for a practical “tiny Leela” style project.

The central idea is **amortized planning**:

```text
PUCT / MCTS engine:
  spends compute at move time by expanding a tree.

Search-less transformer:
  spends compute during training by learning from many examples of what search would have found.
```

A transformer does not literally run a tree search inside its attention layers. Instead, it can learn a function that approximates search outputs:

```text
position → good move distribution
position → value / WDL
position + move → value of that move
position → ranking of legal moves
```

The best practical route is not to fully reject search. It is to make search **optional and conditional**:

```text
Easy/confident positions:
  one forward pass or top-k reranking.

Hard/uncertain positions:
  small PUCT, shallow rollout, or exact tablebase.
```

---

## Background: What PUCT Provides

Lc0-style engines use a neural network and PUCT/MCTS together.

The neural network outputs:

```text
policy: which legal moves look promising?
value/WDL: who is likely to win, draw, or lose from this position?
```

PUCT then grows a search tree. At each node, it selects moves using something like:

```text
PUCT_score(move) = Q(move) + U(move)
```

where:

```text
Q(move) = average searched value for that move
U(move) = exploration bonus based on the policy prior and visit counts
```

A simplified exploration term is:

```text
U(move) = c_puct * P(move) * sqrt(N_parent) / (1 + N_child)
```

So PUCT gives the engine an external scratchpad:

```text
1. Use policy to choose promising moves.
2. Explore possible continuations.
3. Evaluate leaf positions with the value network.
4. Back up values through the tree.
5. Play the root move search trusted most, usually the highest-visit move.
```

Search-less transformers try to learn what this process would output, so they can play with far fewer runtime evaluations.

Useful references:

- Lc0 AlphaZero/PUCT overview: <https://lczero.org/dev/lc0/search/alphazero/>
- Chessformer paper: <https://arxiv.org/abs/2409.12272>
- Grandmaster-Level Chess Without Search / amortized planning paper: <https://arxiv.org/abs/2402.04494>

---

## What Attention Contributes

Attention is useful in chess because chess is relational.

A move is rarely good because of a single square. It depends on relationships:

```text
bishop ↔ diagonal target
rook ↔ open file
queen + bishop ↔ battery
knight ↔ fork geometry
king ↔ escape squares
pinned piece ↔ defender that cannot move
passed pawn ↔ blockers and promotion race
sacrifice ↔ opened line and forcing continuation
```

A transformer lets every square communicate with every other square in one layer:

```text
bishop on c1 can attend directly to h6
rook on e1 can attend directly to e-file blockers
king on g8 can attend directly to attackers around g-file/h-file
knight on f7 can attend to king, queen, rook fork targets
```

Abstractly, attention is learned relational message passing:

```text
for each token:
  decide which other tokens matter
  gather information from them
  update the token representation
```

In chess, heads can specialize into roles such as:

```text
diagonal heads
file/rank heads
knight-jump heads
king-zone heads
attacker/defender heads
pin/skewer heads
promotion-race heads
same-color-complex heads
```

Multiple layers can compose these relations:

```text
layer 1: who attacks what?
layer 2: which defenders are pinned or overloaded?
layer 3: which candidate moves create threats?
layer 4: which threats lead to winning values?
```

This is the intuition behind “internalized” or “amortized” search. The model is not explicitly expanding a tree, but it can learn patterns and consequences that search would otherwise discover.

---

## Chessformer-Specific Lessons

Chessformer is a strong architectural reference because it uses a chess-native transformer representation.

### 1. One Token Per Square

Chessformer uses:

```text
64 tokens, one per board square
```

Each token contains board/rule/history features, roughly corresponding to lc0’s old 112-plane input but rearranged:

```text
lc0 convolution input:
  [112, 8, 8]

Chessformer input:
  [64, 112]
```

This makes attention natural: every square can directly interact with every other square.

### 2. Chess-Aware Positional Structure

Chessformer’s major finding is that positional representation inside attention matters a lot.

Simple Euclidean closeness is not enough for chess. A square can be “near” in a chess sense if it is:

```text
on the same diagonal
on the same file
on the same rank
a knight move away
a pawn attack away
in a king zone
part of the same battery or ray
```

For a tiny engine, the most practical approximation is a relation-aware attention bias:

```text
attention_score[h, i, j] =
    Q[h, i] · K[h, j] / sqrt(d_head)
  + relation_bias[h, relation(i, j)]
```

Possible relation classes:

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

This is simpler than full Shaw-style Q/K/V relative vectors but preserves the key idea: chess geometry should shape attention.

### 3. From-To Policy Head

Chessformer’s policy head is also useful. Rather than treating policy as an opaque flat vector, it scores moves in a from-square/to-square form:

```text
ordinary move logit ≈ score[from_square, to_square]
promotion logits handled separately
illegal moves masked
```

This aligns with how a chess engine already represents moves:

```text
move = (from, to, promotion)
```

For a tiny engine, this avoids much of the friction around lc0’s legacy 1858/1862 move-indexing details.

---

# Abstract Search-Less / Search-Light Designs

## Design 1: Policy-Only Transformer

### Interface

```text
position → legal move distribution
```

At inference:

```text
1. Encode position.
2. Run model once.
3. Mask illegal moves.
4. Play argmax(policy), or sample from the policy.
```

### What It Replaces

This replaces the entire PUCT tree with one learned policy estimate.

### Pros

```text
fastest possible neural move selection
one forward pass per move
simple browser deployment
excellent for human-like Maia-style play
works well with opening books and style sampling
```

### Cons

```text
no explicit blunder checking
no candidate verification
rare tactics can be missed
requires very strong policy labels
can be repetitive if always argmax
```

### Best Training Targets

```text
teacher search visit distribution
Stockfish/lc0 best-move distribution
human move distribution for human-style play
Maia-style rating-conditioned moves
```

### Project Role

Useful as:

```text
v1 instant mode
baseline for all other designs
human/personality mode
mobile-safe mode
```

---

## Design 2: State-Value Child Reranker

### Interface

```text
position → policy
child_position → value/WDL
```

At inference:

```text
1. Use policy to select top-k candidate moves.
2. Make each candidate move on a copied board.
3. Evaluate the resulting child position.
4. Choose the move whose child value is best for us.
```

Pseudo-code:

```python
policy, wdl = model(position)
candidates = top_k_legal(policy, k=8)

best_move = None
best_score = -inf

for move in candidates:
    child = position.make(move)
    _, child_wdl = model(child)
    score = -value_for_side_to_move(child_wdl)  # perspective flip
    if score > best_score:
        best_score = score
        best_move = move
```

### What It Replaces

This is not full PUCT. It is a shallow 1-ply verification layer.

### Pros

```text
simple
catches some obvious bad policy moves
uses the same value head needed for search
much cheaper than MCTS if top-k is small
```

### Cons

```text
still shallow
cost is k+1 model evaluations
value head must be well calibrated
can prefer quiet-looking moves over tactical forcing moves
```

### Best Training Targets

```text
WDL/result labels
Stockfish/lc0 value labels
q-value or value-bucket labels
endgame tablebase labels
```

### Project Role

Good for:

```text
fast mode
sanity-checking policy-only moves
browser mode on desktop/WebGPU
```

---

## Design 3: Action-Value Transformer

### Interface

```text
position + candidate move → value of that move
```

or, more efficiently:

```text
position → shared trunk
(position_embedding, move_embedding) → action value
```

At inference:

```text
1. Generate legal moves.
2. Score candidate moves with the action-value head.
3. Play the highest-valued move.
```

### What It Replaces

This replaces search with a learned evaluator of legal moves.

Instead of asking:

```text
What happens if I search this move and its continuations?
```

it asks:

```text
What did training teach me this move is worth?
```

### Pros

```text
strong move-ranking objective
aligns directly with Stockfish/lc0 teacher labels
can learn tactical move quality better than policy-only cloning
works well with top-k candidate filtering
```

### Cons

```text
more expensive if evaluated separately for every legal move
requires teacher action-value labels
harder data pipeline
q/value calibration matters a lot
```

### Best Training Targets

```text
Stockfish value for each candidate move
lc0 search value for candidate moves
teacher WDL after candidate moves
categorical win-rate buckets
pairwise move preferences
```

### Project Role

This is one of the highest-value additions for a search-light tiny engine.

Best practical version:

```text
policy selects top-k moves
shared trunk computes board embedding once
action-value head reranks top-k moves
```

---

## Design 4: All-Moves Q Head

### Interface

```text
position → Q(move) for every legal move
```

This is similar to a policy head, but the output is estimated move value rather than move likelihood.

At inference:

```text
1. Run model once.
2. Mask illegal moves.
3. Pick move with highest Q.
```

### Difference Between Policy and Q

```text
policy(move): how plausible/promising the move looks
Q(move): how good the move is expected to be
```

A move can be low-policy but high-Q if it is tactically surprising.

### Pros

```text
one forward pass
directly ranks legal moves by expected value
avoids separate child evaluations
very browser-friendly if accurate
```

### Cons

```text
harder target than policy
Q errors are dangerous
teacher labels are expensive
may be poorly calibrated on rare moves
```

### Best Training Targets

```text
action-value buckets for legal moves
Stockfish/lc0 q-values
pairwise ranking losses
Kendall-tau optimized ranking objectives
```

### Project Role

Promising for v2/v3 after we have a good data generator. For v1, action-value top-k reranking is safer.

---

## Design 5: Principal-Variation / Line Predictor

### Interface

```text
position → next N moves
```

Example:

```text
model predicts:
  my move
  likely opponent reply
  my response
  likely continuation
```

### What It Replaces

This tries to make the model explicitly output a plan or tactical line instead of only a move.

### Pros

```text
encourages sequence reasoning
useful for tactics and forcing lines
can provide explanations or UI lines
may improve policy training as an auxiliary objective
```

### Cons

```text
many valid continuations exist
sequence labels are brittle
can hallucinate illegal or inferior lines if not constrained
less reliable as the primary move selector
```

### Best Training Targets

```text
teacher principal variations
puzzle solution lines
self-play MCTS most-visited lines
Stockfish PVs
human game continuations for style
```

### Project Role

Best used as an auxiliary training head, not the main inference mechanism:

```text
main output: policy / action-value
auxiliary output: predicted reply / PV tokens
```

---

## Design 6: Recurrent or Iterative Transformer

### Interface

```text
position hidden state
→ repeatedly apply same reasoning block
→ output policy/value
```

At inference:

```text
for t in 1..K:
    hidden = reasoning_block(hidden)
policy, value = heads(hidden)
```

Potentially use variable compute:

```text
easy position: K = 2
hard position: K = 8
```

### What It Replaces

This gives the model an internal iterative computation budget. It is closer in spirit to search than a single fixed forward pass.

### Pros

```text
parameter-efficient if weights are shared
supports adaptive compute
may learn search-like refinement
useful for hard tactical positions
```

### Cons

```text
harder to train
harder to export cleanly to ONNX/WebGPU
latency less predictable
requires good uncertainty or halting logic
```

### Best Training Targets

```text
policy/value at each iteration
teacher search-improved policy
puzzle/tactical data
intermediate consistency losses
```

### Project Role

Research track. Not v1. Potentially valuable if a single-pass SquareFormer plateaus.

---

## Design 7: Transformer + Memory / Retrieval

### Interface

```text
position → embedding
embedding → retrieve similar positions / motifs / openings
position + retrieved memories → move/value
```

A memory item might store:

```text
compressed position embedding
top-k policy moves
WDL/value
source: opening book, self-play, teacher analysis, user games
confidence / visit count
```

### What It Replaces

This does not replace search directly. It gives the model external knowledge without expanding a tree.

### Pros

```text
useful for openings
useful for rare motifs
supports user personalization
can reuse previous analysis
works well with compressed vector storage
```

### Cons

```text
retrieval errors can mislead
requires memory infrastructure
semantic similarity is risky in chess
not a substitute for exact transposition tables
```

### Best Training Targets

```text
contrastive position embeddings
teacher policy/value memory
opening-book outcomes
self-play replay data
human/personality game data
```

### Project Role

Potential v3 feature. Especially interesting for:

```text
semantic opening book
personalized Maia-like player model
compressed neural transposition memory
```

---

## Design 8: Conditional Search-Light Transformer

### Interface

```text
position → policy, value, action-value, uncertainty
```

At inference:

```text
if confidence high:
    play policy/action-value move
else:
    run small PUCT or top-k rollout
```

### What It Replaces

This does not fully avoid search. It avoids wasting search on positions where the model is already confident.

### Pros

```text
best strength/latency tradeoff
adaptive compute
browser-friendly
handles tactical uncertainty better than pure searchless play
compatible with tablebases
```

### Cons

```text
requires uncertainty calibration
more code paths
harder evaluation
must avoid over-trusting bad confidence estimates
```

### Best Training Targets

```text
value error labels
teacher disagreement
policy entropy
search instability
puzzle failure labels
calibration losses
```

### Project Role

This is the recommended product architecture.

---

# Best Synthesis Recommendation

## Recommended Model: Tiny SquareFormer with Action-Value Reranking

The best synthesis is:

```text
Chessformer-style square-token transformer
+ policy head
+ WDL/value heads
+ action-value top-k reranking head
+ uncertainty head
+ optional small PUCT only when uncertainty is high
```

This combines:

```text
Chessformer architecture lesson:
  use 64 square tokens and chess-aware attention.

Searchless-chess training lesson:
  distill action-values / move rankings from strong search.

Lc0 lesson:
  policy and value are complementary.

Browser/product lesson:
  use adaptive compute rather than fixed expensive search.
```

---

## Proposed Architecture

### Input

Start with a simple but chess-native input:

```text
64 square tokens
side-to-move canonicalization
piece identity
castling rights
en passant
rule-50 / repetition features
optional last 7 plies / history
```

Two possible input tracks:

```text
compatibility track:
  Chessformer/lc0-like 112 features per square

portable track:
  current-board-only 18–32 features per square
```

### Trunk

Initial target:

```text
layers: 4–8
d_model: 128–192
heads: 4–8
ffn_ratio: 1x–2x
attention: chess relation bias or Shaw-lite relative attention
activation: GELU or Mish
```

Avoid starting with:

```text
MoE
large FFN expansion
very deep transformer
long-context move-history transformer
custom low-bit attention kernels
```

### Heads

Required:

```text
policy_head:
  from-square × to-square logits
  promotion logits
  illegal-move mask

wdl_head:
  win/draw/loss probabilities

q_bucket_head:
  categorical value buckets, e.g. 32 or 64 win-rate buckets
```

Recommended:

```text
action_value_head:
  score candidate moves from shared board embedding + move embedding

uncertainty_head:
  predict whether policy/value/action-value is unreliable
```

Optional:

```text
moves_left_head
reply_move_head
PV/head for next 2–4 plies
style/persona heads
legal-move auxiliary head
from/to/piece/check auxiliary heads
```

---

## Inference Modes

### Mode 1: Instant Policy

```text
policy = model(position)
move = argmax_legal(policy)
```

Use for:

```text
mobile
beginner/human-like play
fast UI previews
```

### Mode 2: Top-k Action-Value Rerank

```text
policy, value, uncertainty = model(position)
candidates = top_k_legal(policy, k=8 or 16)
q_moves = action_value_head(position_embedding, candidates)
move = argmax(q_moves + policy_bonus + style_bonus)
```

Use for:

```text
default browser play
stronger no-tree mode
personality/style modes
```

### Mode 3: Conditional Small PUCT

```text
if uncertainty < threshold and policy is sharp:
    use top-k action-value rerank
else:
    run 32–128 node PUCT using policy + WDL
```

Use for:

```text
sharp positions
tactical positions
high-entropy policy positions
positions with teacher/model disagreement
```

### Mode 4: Exact Endgame Override

```text
if tablebase available and piece_count <= limit:
    use tablebase
else:
    use model/search
```

Use for:

```text
analysis mode
desktop builds
remote optional lookup
```

---

## Training Strategy

## Stage 1: Policy/WDL Warm Start

Train the model on supervised data:

```text
position → teacher policy
position → WDL/value
```

Possible data sources:

```text
human games, Maia-style
Stockfish best moves
lc0 policy/value outputs
small self-play games
puzzles and tactical positions
```

Initial loss:

```text
L =
  CE_or_KL(policy_target, policy_pred)
+ CE(wdl_target, wdl_pred)
+ CE(value_bucket_target, value_bucket_pred)
```

## Stage 2: Action-Value Distillation

Add candidate move values.

For each position, label a subset of moves:

```text
teacher best move
teacher top-k moves
human played move
checks/captures/promotions
random legal distractors
policy top-k moves from current student
```

Target:

```text
(position, move) → value bucket / WDL / q
```

Loss:

```text
L += action_value_loss
```

Useful ranking loss:

```text
for move_a, move_b:
  if teacher_value(a) > teacher_value(b):
      score(a) should be > score(b)
```

## Stage 3: On-Policy Distillation

Generate positions from the student’s own play:

```text
student plays games
collect positions where it is uncertain or wrong
ask teacher to label those positions/moves
train on them
```

This is cheaper and more targeted than full AlphaZero-style self-play.

## Stage 4: Limited Self-Play Fine-Tuning

Only after supervised learning works:

```text
small PUCT self-play
32–128 nodes/move
save search visit distributions
train policy to imitate improved search
```

This adds Leela-like feedback without requiring massive distributed RL.

---

## Evaluation Plan

Do not use only Elo. Evaluate separate capabilities.

### Static Metrics

```text
policy top-1 accuracy
policy top-3 / top-5 accuracy
policy KL / cross-entropy
WDL calibration
value bucket accuracy
Brier score for WDL
```

### Move Ranking Metrics

```text
Kendall's tau over legal move ranking
pairwise ranking accuracy
action-value MSE / cross-entropy
best-move regret vs teacher
```

Kendall's tau is especially useful for action-value models because the model may not exactly match the teacher’s top move but can still rank legal moves similarly.

### Chess Metrics

```text
policy-only Elo
top-k rerank Elo
small-PUCT Elo
puzzle accuracy
mate/tactical suite accuracy
endgame suite accuracy
```

### Deployment Metrics

```text
model file size
ONNX FP16 size
INT8 size, if applicable
evals/sec in browser
move latency by mode
memory usage
battery/mobile behavior
```

### Confidence Metrics

```text
uncertainty vs actual error
calibration curves
frequency of fallback to PUCT
Elo gain per extra fallback call
```

---

# Recommended Project Roadmap

## v0: Search-Less Baseline

```text
Model:
  64 square tokens
  4 layers
  d_model 64 or 128
  policy + WDL heads

Inference:
  policy-only

Goal:
  verify tokenization, legal masking, from-to policy, ONNX export
```

## v1: Practical Tiny Search-Less Engine

```text
Model:
  4–6 layers
  d_model 128
  chess relation bias
  policy + WDL + value bucket

Inference:
  policy-only
  top-k state-value child rerank

Goal:
  playable browser engine without tree search
```

## v2: Action-Value Reranker

```text
Model:
  6–8 layers
  d_model 128–192
  add action-value head

Inference:
  policy top-8/top-16
  action-value rerank

Goal:
  strong fast mode that avoids PUCT most of the time
```

## v3: Conditional Search-Light Engine

```text
Model:
  add uncertainty/value-error head

Inference:
  if confident: no search
  if uncertain: 32–128 node PUCT

Goal:
  best strength/latency tradeoff
```

## v4: Memory and Personality

```text
Add:
  style/persona conditioning
  compressed position memory
  opening/style books
  optional local user personalization
```

Goal:

```text
not just a weaker engine, but a controllable chess sparring partner
```

---

# Final Recommendation

Do not build a pure “transformer replaces search” engine as the only path.

Build a **search-less-first, search-light-when-needed** engine:

```text
Tiny SquareFormer
  64 square tokens
  chess-aware attention
  from-to policy head
  WDL/value bucket heads
  action-value top-k reranker
  uncertainty head

Default inference:
  no tree search
  policy + action-value reranking

Fallback inference:
  small PUCT only for uncertain positions
```

This gives the best practical balance:

```text
fast enough for browser
stronger than policy-only
more robust than fully searchless
much cheaper than always running PUCT
compatible with future distillation/self-play
```

The guiding principle:

```text
Use attention to understand board relationships.
Use teacher action-values to amortize search into the model.
Use small search only when the model admits uncertainty.
```
