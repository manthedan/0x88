Yes. Chessformer is one of the most relevant papers for your project because it answers a very specific question: **what does a transformer need to know about chessboard geometry to be useful?** The answer is not “make it bigger.” The answer is: **represent the board as 64 square tokens and give attention a chess-aware positional structure.**

## What Chessformer is, in one paragraph

Chessformer is an encoder-only transformer for chess. Instead of feeding a convolutional tower 112 planes, it feeds a transformer **64 tokens**, one token per square. Each token carries piece/history/rules information for that square. The model predicts policy and value targets, and the paper’s core claim is that the **positional representation inside attention** matters enormously: ordinary absolute embeddings or simple relative biases are not enough; richer square-to-square relationship information lets a smaller transformer compete with much more expensive chess models. The authors report that their 6M-parameter policy-only agent outperformed their AlphaZero-policy comparison while using about 8× fewer FLOPs, and their 240M model matched or exceeded prior grandmaster-level transformer agents at much lower compute in their evaluation setup. 

The big caveat: this is not a full search engine paper in the lc0 sense. The main reported agents are **policy-only** or **one-ply value-maximization** agents, not full MCTS engines. That makes it especially interesting for us, because a tiny browser Leela also needs a strong raw policy before search. 

---

## The core architecture

### 1. Input: one token per square

Chessformer uses a fixed sequence of **64 tokens**, one for each board square. The board is flipped according to the side to move, and each token has length 112. That 112-dimensional token includes piece identity at that square for the current and previous seven positions, en passant/castling information, a normalized rule-50-ish counter, and repetition indicators. 

Conceptually:

```text
position
→ 64 square tokens
→ token_i = features for square_i
→ transformer encoder
→ policy/value heads
```

This is very close to lc0’s historical 112-plane input, but rearranged:

```text
lc0 conv:
  [112, 8, 8]

Chessformer:
  [64, 112]
```

That rearrangement is the first big idea. It makes the board natural for attention: every square can directly attend to every other square.

### 2. Body: encoder-only transformer

The body is a stack of encoder layers with fixed context length 64. The paper uses Post-LN normalization, DeepNorm-style initialization/gain, Mish activations, and a few throughput optimizations such as omitting biases in QKV projections and some normalization terms; they report that this bias removal improved training throughput by around 10% without hurting quality in their experiments. 

Their two named model scales are:

| Model   | Layers | Embedding | Heads | FFN depth | Params | Training hardware |
| ------- | -----: | --------: | ----: | --------: | -----: | ----------------- |
| CF-6M   |      8 |       256 |     8 |       256 |    ~6M | single A100       |
| CF-240M |     15 |      1024 |    32 |      4096 |  ~243M | 8 A100s           |

The 240M model was trained for 3.7M steps with batch size 4096 on a huge static dataset; the 6M model was trained with batch size 2048 on a single A100. 

For our project, **CF-6M is the relevant ceiling**, not CF-240M.

---

## The most important idea: chess-aware positional attention

The paper’s central point is that normal transformer positional encodings are the wrong bias for chess.

In text and images, nearby tokens are often more related. In chess, that is often false. A bishop on `c1` may care more about `h6` than `d2`; a rook may care about distant squares on the same file; a knight cares about L-shaped jumps; a queen cares about rays. The paper explicitly argues that Euclidean distance does not capture chessboard topology. 

They compare three position representations:

| Representation              | Policy accuracy | Value accuracy |
| --------------------------- | --------------: | -------------: |
| Absolute position embedding |          57.44% |         89.11% |
| Relative bias               |          58.23% |         89.26% |
| Shaw-style relative vectors |          59.27% |         89.53% |

The Shaw-style representation beat absolute embeddings by **1.83 percentage points** in policy accuracy at the 6M scale. The authors note that, at this scale, doubling model size produced roughly a 1.5% policy-accuracy gain, so a better positional representation was comparable to a major model-size increase. 

That is probably the most important lesson for tiny Leela:

```text
A good chess inductive bias may buy more than simply making the model larger.
```

### What Shaw-style attention does

Vanilla attention computes:

```text
score(i, j) = Q_i · K_j / sqrt(d)
```

Chessformer’s chosen representation modifies the query/key/value interactions with learned square-pair positional vectors. In simplified language:

```text
score(i, j) =
  relation-aware_query(i, j) · relation-aware_key(i, j)
```

and the value path also gets square-pair positional information. The paper says this is computationally expensive at large context lengths, but with only 64 tokens the extra cost is manageable. 

For a tiny browser model, I would not necessarily copy the full implementation first. I would implement a **Shaw-lite chess relation bias**:

```text
attention_score[h, i, j] =
    Q[h, i] · K[h, j] / sqrt(d_head)
  + bias[h, relation(i, j)]
```

Where `relation(i, j)` includes chess relationships:

```text
same square
same rank
same file
same diagonal
same anti-diagonal
knight move
king move
rook ray distance bucket
bishop ray distance bucket
queen ray
white pawn attack
black pawn attack
same color complex
center / edge / corner relation
```

This would be cheaper and easier than full Shaw-style Q/K/V relative vectors, but it captures the same principle.

---

## The policy head is excellent for our project

Chessformer’s policy head is especially attractive. Instead of using an opaque lc0-style flat move vector immediately, it builds a **64×64 origin-destination matrix**:

```text
from_square query vectors
to_square key vectors
policy_logit[from, to] = query[from] · key[to]
```

This matrix represents all ordinary moves as “piece moves from square A to square B.” Promotions are handled with extra additive logits for promotion piece choice. Finally, illegal moves are masked before the policy softmax, which the paper says improved training stability. 

This is a beautiful fit for tiny Leela.

Instead of starting with lc0’s awkward 1858/1862 indexing issue, we could define the policy internally as:

```text
ordinary moves:
  from_square ∈ 64
  to_square   ∈ 64
  → 4096 logits

promotions:
  from_square, to_square, promotion_piece
  → extra logits

then:
  legal_mask(position)
  softmax over legal moves only
```

For browser deployment and search, this is convenient because move generation already gives you `(from, to, promotion)` tuples. You can map the model’s policy logits to legal moves without a fragile legacy lc0 gather table.

I would steal this policy head almost directly.

---

## The value heads are also useful, but we should simplify

Chessformer has multiple value heads. One predicts game result as WDL. Others predict search-derived scalar reward, categorical reward buckets, short-term value, and value error. The paper frames these auxiliary policy/value targets as convergence accelerators. 

For our tiny project, I would use a simpler version:

```text
required:
  policy head
  WDL result head

nice-to-have:
  q scalar head
  value_error / uncertainty head
  short-term value head
```

The **value-error head** is especially interesting for browser search. It can become a search-budget signal:

```text
if value_error is high:
  spend more PUCT nodes

if value_error is low and policy is sharp:
  move quickly
```

That gives us adaptive search, which is very useful on mobile or in-browser.

---

## Chessformer’s training setup is very relevant to our “scrappy” constraints

The paper does **not** train through a fresh online self-play loop. It uses a **static dataset** of self-play games from an older reinforcement-learning run, and the authors say initial experiments showed no quality degradation from using the static dataset. 

That is directly aligned with our setup.

Instead of needing distributed self-play from day one, we can do:

```text
teacher-generated static dataset
→ train Chessformer-tiny supervised
→ evaluate policy/value
→ optionally add small self-play later
```

This is the same “distillation first, RL later” strategy we have been converging on.

A tiny training objective inspired by Chessformer:

```text
L =
  1.0 * CE/KL(policy_target, policy_pred)
+ 1.0 * CE(WDL_target, WDL_pred)
+ 0.5 * MSE(q_target, q_pred)                    # optional
+ 0.1 * CE(soft_policy_target, soft_policy_pred) # optional
+ 0.1 * MSE(value_error_target, error_pred)       # optional
```

The soft-policy head is worth considering because Chessformer trains a second policy target using a high-temperature version of the policy distribution. 

---

## The evaluation methodology is something we should copy

Chessformer evaluates two cheap agents:

```text
policy agent:
  pick the highest-policy legal move
  cost = 1 model eval

value agent:
  evaluate each legal move after making it
  pick move with best value
  cost ≈ 20 model evals
```

They use both Elo-style comparisons and puzzle-solving accuracy. 

For our tiny Leela, that gives a clean evaluation ladder before we even implement full MCTS:

```text
Eval 0: policy top-1 / top-3 accuracy on held-out teacher data
Eval 1: policy-only play
Eval 2: one-ply value rerank
Eval 3: small PUCT, 32–128 playouts
Eval 4: browser latency at each level
```

This is much cheaper than immediately running thousands of PUCT games at every checkpoint.

---

## Attention-map interpretability is not just cosmetic

The paper’s attention maps showed heads specializing in chess-relevant patterns: bishop moves, knight moves, rook moves, king moves, queen attention, same-color-square attention, opponent-piece attention, and “pieces that can move to the querying square.” 

For us, that is a debugging tool.

If our tiny SquareFormer is learning properly, we should expect to see some heads become:

```text
rook-ray heads
bishop-diagonal heads
knight-jump heads
king-neighborhood heads
attacker/defender heads
same-color-complex heads
promotion-rank heads
```

If all heads mostly attend locally or randomly, the model is probably wasting capacity.

A tiny interpretability dashboard could show:

```text
click square e4
→ visualize attention over board
→ choose layer/head
→ compare across checkpoints
```

This could be a fun part of the project and a useful model-debugging tool.

---

## What not to copy

### Do not copy CF-240M

CF-240M is not the portable target. It has ~243M parameters and 12.8B FLOPs per policy evaluation in the paper’s accounting. It is interesting as evidence that the architecture scales, not as a browser model. 

### Do not start with one-ply value-max in browser

The value agent evaluates every legal move, and the paper approximates that as 20× the model FLOPs because there are roughly 20 legal moves on average. 

For a browser model, that is expensive. I would use value reranking only over the top-k policy moves:

```text
policy top-k = 4 to 8 moves
evaluate only those child positions
pick best value-adjusted move
```

That gives most of the benefit at much lower cost.

### Do not copy generic LLM tricks blindly

The paper tried GLU, sparse MoE, and soft MoE variants in the feed-forward layer and found they did not improve performance; GLU slightly degraded quality at constant FLOPs. 

This reinforces an important lesson:

```text
Chess transformers are not just tiny LLMs.
Chess-specific geometry matters more than trendy transformer blocks.
```

---

## How I would apply Chessformer to our tiny-Leela project

I would now split the project into two parallel baselines:

```text
A. Tiny Conv Leela:
   48×5 or 64×6 SE-ResNet

B. Tiny Chessformer / SquareFormer:
   64 square tokens + chess-aware attention
```

The goal is not to assume transformers win. The goal is to test whether Chessformer-style positional attention beats a conv tower at the same browser budget.

## Proposed “SquareFormer-tiny” architecture

### Version 0: micro debug model

```text
Input:
  64 tokens × 32–64 features
  current board only, no history at first

Body:
  4 transformer layers
  d_model = 64
  heads = 4
  d_ff = 64 or 128
  chess relation bias

Heads:
  from-to policy head
  WDL value head
```

Use this to debug:

```text
tokenization
legal masking
promotion handling
ONNX export
browser inference
attention visualization
```

### Version 1: real tiny model

```text
Input:
  64 tokens × 112 features
  current + 7 historical positions, lc0/Chessformer-style

Body:
  6 layers
  d_model = 128
  heads = 4 or 8
  d_ff = 128 or 256
  Mish or GELU
  Shaw-lite relation bias

Heads:
  from-to policy
  WDL
  q scalar
  optional value-error
```

This should land roughly in the **1M–3M parameter** class depending on relation-bias/head choices. It should be much smaller than CF-6M and realistic for WebGPU.

### Version 2: browser-strong small model

```text
Input:
  64 tokens × 112 features

Body:
  8 layers
  d_model = 192 or 256
  heads = 8
  d_ff = d_model, not 4× d_model
  richer Shaw-style relation vectors

Heads:
  from-to policy
  WDL
  q
  uncertainty/error
  soft policy auxiliary during training
```

This is the “try to approach CF-6M but browser-aware” version.

---

## The policy format I would use

Internally:

```text
ordinary_logits: [64, 64]
promotion_logits: [from, to, promotion_piece]
```

At training time:

```text
legal_logits = gather_legal_moves(ordinary_logits, promotion_logits, legal_moves)
loss = cross_entropy(legal_logits, target_policy_over_legal_moves)
```

At inference time:

```text
legal_moves = movegen(position)
policy_probs = model(position, legal_mask)
PUCT uses policy_probs over legal_moves
```

This avoids the lc0 legacy policy-vector complexity while preserving all chess moves cleanly.

---

## Training plan inspired by Chessformer

Chessformer’s most useful training lesson is: **static supervised data can work**. They trained from old self-play data rather than generating online self-play during training. 

For us:

```text
Stage 1: human/teacher policy warm-start
  data: Lichess games, Maia-style, or Stockfish/lc0 teacher labels
  target: played move or teacher search policy

Stage 2: stronger teacher distillation
  data: positions labeled by lc0/Stockfish
  target: policy distribution + WDL/q

Stage 3: limited self-play
  data: your own tiny engine at 32–128 nodes
  target: search-improved policy

Stage 4: browser compression
  FP16 WebGPU
  INT8/WASM fallback
```

I would not use pure human moves only if the goal is strength. I would use either:

```text
teacher search policy:
  best for strength

human move policy:
  best for human-like sparring

mixed:
  best for a fun browser engine
```

---

## Project-specific experiments to run

### Experiment 1: position representation ablation

Copy Chessformer’s ablation idea at tiny scale:

```text
same model, same data, same steps:

A. learned absolute square embedding only
B. 2D relative bias
C. chess relation bias
D. Shaw-lite Q/K/V relative vectors
```

Track:

```text
policy accuracy
policy KL
WDL calibration
puzzle accuracy
browser eval latency
```

If the paper’s lesson holds, relation-aware attention should beat absolute embeddings even at small scale.

### Experiment 2: conv versus SquareFormer

Train:

```text
64×6 SE-ResNet
vs
SquareFormer d=128 L=6
```

Same dataset, same target, same parameter/file-size budget.

Compare:

```text
policy-only Elo
PUCT Elo at 64 nodes
puzzle accuracy
evals/sec
ONNX size
mobile latency
```

This answers the project’s real architectural question.

### Experiment 3: policy-only first, search later

Use Chessformer-style policy agents:

```text
argmax policy
sampled policy
top-k value rerank
PUCT
```

This tells us whether the raw net is good enough to justify search.

### Experiment 4: attention sanity checks

Visualize heads. Look for:

```text
diagonal attention
rank/file attention
knight jumps
attacker/defender attention
king safety zones
promotion-rank attention
```

This gives qualitative feedback that a conv net cannot offer as easily.

---

## What Chessformer changes in our roadmap

Before reading Chessformer, the obvious tiny-Leela plan was:

```text
start with 48×5 or 64×6 conv
maybe try transformer later
```

After Chessformer, I would change that to:

```text
start with 64×6 conv baseline
also build SquareFormer-tiny immediately
make position representation the main research variable
```

The main thing to steal is not the 240M model. It is this design pattern:

```text
64 square tokens
+ chess-aware relative attention
+ origin-destination policy head
+ static supervised training
+ policy-only / value-rerank evaluation
```

That is almost perfectly aligned with a portable tiny Leela.

## Bottom line

Chessformer gives us a credible architecture for a **tiny neural chess model that is not just a smaller lc0 conv net**.

The best project adaptation is:

```text
Tiny SquareFormer:
  64 square tokens
  chess relation attention
  from-to policy head
  WDL/q value heads
  static teacher-distillation dataset
  policy-only eval first
  small PUCT later
```

For a browser target, I would aim first for a **1M–3M parameter SquareFormer**, not CF-6M or CF-240M. The open research question for our project becomes very concrete: **does chess-aware attention beat a 64×6 conv tower at the same latency and file size?**
