Yes — I think your intuition is right: **for small neural chess nets, pure transformers are probably not automatically more efficient than CNNs**. A tiny CNN gets a lot “for free”: locality, translation-ish sharing, cheap dense kernels, and board-grid bias. A tiny transformer has to spend parameters learning things a CNN already knows, unless we give it **better tokens**.

The best direction is probably not “replace CNN with SquareFormer.” It is:

```text
use CNN / NNUE / chess features for cheap local and tactical structure
use attention only where it is unusually valuable:
  long-range relations,
  piece interactions,
  legal move scoring,
  candidate refutations,
  and conditional search guidance.
```

Lc0’s own transformer progress supports this: the win came from **chess-specific architecture**, not generic transformer scaling. They use 64 square tokens, side-to-move flipping, chess-topology attention biases, smolgen-style dynamic attention, small FFN ratios, and attention maps that learn rook/bishop/knight/king relationships. ([Leela Chess Zero][1]) But Stockfish’s NNUE is the opposite lesson: sparse, incrementally updatable, shallow, quantized evaluators are dominant when you need millions of low-latency evals. ([Stockfish][2])

So the architecture question becomes:

```text
What should the tokens be so attention does not waste compute?
```

Below are the most promising tokenization/model ideas I’d explore.

---

## 1. Conv-SquareFormer Hybrid

This is my highest-confidence architecture.

Instead of:

```text
64 raw square tokens → transformer
```

do:

```text
board planes
  → small CNN / SE-ResNet stem
  → 64 enriched square embeddings
  → 2–4 chess-aware attention blocks
  → policy / WDL / action-value heads
```

The CNN handles cheap local structure:

```text
pawn shapes
piece clusters
king shelter
local tactics
piece-square patterns
```

The attention layers handle global structure:

```text
long diagonals
open files
pins
batteries
king attacks
promotion races
piece coordination
```

This is likely stronger per parameter than a pure SquareFormer at tiny sizes, because the transformer receives **already chess-shaped square embeddings**, not raw square features.

A sketch:

```text
Input: [C, 8, 8] board planes

CNN stem:
  Conv 3×3, width 64
  3–6 residual/SE blocks

Tokenization:
  each square gets its CNN feature vector
  → [64, d_model]

Attention:
  2–4 relation-biased attention blocks

Heads:
  from-to policy
  WDL/value bucket
  action-value top-k
  uncertainty
```

This directly addresses your concern that CNNs are more efficient for small nets. The attention becomes a **global relation add-on**, not the whole model.

I would test this against your pure SquareFormer V1 immediately:

```text
same parameter budget
same dataset
same policy/WDL/action-value targets
policy-only Elo
top-k rerank Elo
PUCT Elo
latency
queen-blunder/regret suite
```

The internal roadmap already treats conv hybrids as one of the strongest follow-on lanes after the SquareFormer baseline, and the infra gap analysis says the high-upside path is action/opponent-reply modeling rather than plain policy imitation. 

---

## 2. PieceFormer: active-piece tokens, not 64 square tokens

A chess position has at most 32 pieces. Empty squares matter, but often indirectly.

Instead of tokenizing all 64 squares:

```text
64 square tokens
```

use:

```text
≤32 active piece tokens
+ a few rule/state/global tokens
```

Each piece token contains:

```text
piece type
color
square
side-to-move relative square
material value
mobility
attacked/defended counts
pinned/skewered flags
king-distance features
```

Then attention asks:

```text
Which pieces interact?
```

This is very natural for tactics:

```text
bishop token attends to king token along diagonal
rook token attends to pinned defender
knight token attends to fork targets
queen token attends to overloaded pieces
```

The benefit is that attention cost drops from:

```text
64² = 4096 square-pair relations
```

to roughly:

```text
24²–32² = 576–1024 piece-pair relations
```

in many positions.

The challenge is policy output: moves go to empty squares too. Solve that with a **move decoder**:

```text
piece tokens + square embeddings
  → from-to / legal move policy
```

or with legal move tokens, described next.

This architecture might beat SquareFormer at small size because it does not waste compute on empty squares. But it needs strong positional embeddings and explicit square-state features so empty-square concepts like outposts, escape squares, and promotion squares are not lost.

---

## 3. MoveFormer: legal moves as tokens

This may be the most directly useful for your **action-value** direction.

Tokenize not just the board, but the **legal candidate moves**:

```text
position tokens:
  pieces / squares / CNN features

move tokens:
  one token per legal move
```

A move token contains:

```text
from square
to square
promotion
moving piece
captured piece, if any
is check
is capture
is castle
is promotion
SEE/material delta
attacks created
queen/king exposure features
```

Then:

```text
move tokens cross-attend to board tokens
→ output policy logit and action-value for each legal move
```

Diagram:

```text
board representation
   │
   ├── piece/square tokens
   │
legal moves
   │
   └── move tokens
          │
          ▼
   move-token cross-attention
          │
          ├── policy(move)
          ├── action_value(move)
          ├── regret(move)
          └── tactical_risk(move)
```

This makes the model’s job very explicit:

```text
Do not understand all 4096 from-to pairs.
Only score legal moves.
```

That is ideal for a lightweight engine. The average legal move count is far below 4096, and even in high-mobility positions it is usually manageable.

This also naturally fixes the queen-blunder class without hard-coding queen safety:

```text
candidate move Qh5
→ move token attends to opponent pawn/knight/bishop defenders
→ action-value/regret head learns Qh5 is bad
```

The search-light design already identifies action-value modeling as one of the highest-value additions because it lets the model rank candidate moves directly rather than trusting raw policy. 

A strong version:

```text
CNN stem → square features
active pieces → piece tokens
legal moves → move tokens
move tokens cross-attend to piece/square tokens
heads output P, Q, regret, uncertainty per move
```

This is probably my favorite **novel tokenization** idea for your project.

---

## 4. RayFormer: tokenize sliding-piece rays and attack lines

Chess tactics are often about rays:

```text
bishop diagonal
rook file
queen battery
pin along king line
skewer
discovered attack
back-rank pressure
```

CNNs handle local patterns well but need depth for long rays. Square attention can learn rays, but it may waste capacity discovering what the move generator already knows.

So create explicit **ray tokens**.

Examples:

```text
rook rays:
  from each rook/queen along ranks/files until blocker

bishop rays:
  from each bishop/queen along diagonals until blocker

king rays:
  lines toward king

pin/skewer candidate rays:
  attacker → blocker → king/queen/high-value piece
```

Each ray token contains:

```text
attacker piece
direction
squares on ray
first blocker
second blocker
target piece
king/queen involvement
x-ray potential
```

Then the model attends over:

```text
piece tokens
+ ray tokens
+ king-zone tokens
+ move tokens
```

This gives a small network direct access to long-range tactical geometry. It is a chess-specific inductive bias that could be much more parameter-efficient than hoping attention discovers all ray semantics from scratch.

This also aligns with Lc0’s observation that chess topology is not Euclidean: squares related by rook/bishop/knight moves are “near” in chess terms even if far on the board. ([Leela Chess Zero][1])

Potential downside: ray token generation is more engineering, and if labels are noisy the model may overfit to handcrafted tactical features. But for a small net, this is probably a worthwhile trade.

---

## 5. AttackMapFormer: tokenize attack/defense maps

Another strong idea: expose the **attack graph** directly.

For each square, precompute:

```text
white attacks this square by:
  pawns, knights, bishops, rooks, queen, king

black attacks this square by:
  pawns, knights, bishops, rooks, queen, king

defender counts
least valuable attacker
least valuable defender
king-zone attacks
pinned defenders
```

You can either add these as channels to square tokens, or create separate attack tokens:

```text
square token e4
attack token: white attacks e4
attack token: black attacks e4
king-zone attack token
```

For tiny models, I would not be shy about giving attack maps. Stockfish and NNUE are packed with chess-specific structure; there is no prize for making the tiny model rediscover legal attacks from raw occupancy. The Stockfish NNUE docs emphasize that its power comes from sparse chess features, incremental updates, and low-precision shallow inference rather than generic architecture purity. ([Stockfish][2])

A practical encoding:

```text
64 square tokens, each augmented with:
  attacked_by_white_bitmask_or_counts
  attacked_by_black_bitmask_or_counts
  defended_by_own_count
  attacked_by_enemy_count
  pinned_piece_flag
  king_zone_flag
  SEE-lite bucket
```

This may make a pure SquareFormer much more tactically robust without increasing depth.

---

## 6. NNUE-Token Transformer: sparse feature tokens

This is a bridge between Stockfish and transformers.

Stockfish’s NNUE is built around sparse active features and accumulators. Instead of tokenizing squares, tokenize **active NNUE-like features**:

```text
king-relative piece-square feature
piece-square feature
threat feature
pawn-structure feature
king-zone feature
```

A position becomes a set of sparse feature tokens:

```text
[white king e1 + white knight f3]
[black king g8 + white bishop c4]
[black king g8 + white queen h5]
...
```

Then use:

```text
feature-token pooling / attention
→ value / policy / action-value heads
```

This could be very parameter-efficient because you feed the model high-value chess concepts rather than raw board squares.

A compact version:

```text
active sparse features
  → embedding table
  → DeepSets / Set Transformer pooling
  → small MLP or attention
```

This has three advantages:

```text
1. It inherits NNUE’s feature-efficiency.
2. It can be quantized and CPU-friendly.
3. It may pair well with alpha-beta or cheap search.
```

The risk is policy output. NNUE-like features are naturally value-oriented; for policy/action-value you would combine them with legal move tokens:

```text
sparse feature embedding
+ move token
→ action-value / regret
```

This is a promising research lane if you want “strongest lightweight” more than “pure neural elegance.”

---

## 7. KingZoneFormer: dedicated tactical submodel around kings

Many catastrophic mistakes and many brilliant sacrifices are king-zone phenomena. A small model can waste capacity modeling the whole board equally.

Create specialized tokens around each king:

```text
own king zone tokens
enemy king zone tokens
attacking piece tokens
defending piece tokens
open-line tokens into king zone
escape-square tokens
```

Then produce auxiliary scores:

```text
king_attack_score
mate_threat_score
sacrifice_soundness
defensive_resource_score
```

This can run as either:

```text
a branch inside the main model
```

or:

```text
a cheap tactical verifier called only for attacking moves
```

This is not a replacement for policy/value, but it could significantly improve tactical aggression and avoid unsound sacrifices.

---

## 8. Latent-Token / Perceiver-style Chess Model

Instead of letting all 64 squares attend to each other every layer, use a small set of learned “analysis tokens”:

```text
64 square/piece tokens
  → cross-attend into 8–32 latent tokens
  → latent tokens self-attend
  → heads read from latents and square features
```

The latent tokens can specialize:

```text
material token
king safety token
pawn structure token
tactical motif token
endgame token
initiative token
opening token
```

This might be more efficient than full square attention for small models:

```text
full attention: 64² per layer
latent bottleneck: 64×L + L², with L = 16 or 32
```

It also gives a clean way to add global board understanding to a CNN:

```text
CNN square features
→ 16 latent analysis tokens
→ policy/value/action heads
```

Risk: a bottleneck may throw away details needed for precise tactics. Good candidate for WDL/value and uncertainty, less obviously for exact policy.

---

## 9. Afterstate / Delta Tokenization

Most candidate move evaluation is about a **small board delta**:

```text
piece leaves from-square
piece arrives to-square
maybe captures
maybe promotes
maybe changes castling/en passant
```

Instead of evaluating full child boards for top-k moves, represent a candidate move as **delta tokens**:

```text
from token
to token
captured token
promotion token
rule-state delta token
```

Then:

```text
current board embedding + move delta tokens
→ afterstate value estimate
```

This is more efficient than:

```text
make move → re-encode whole board → run full model
```

A move-conditioned action-value head is a simple version of this. A richer version uses a few attention layers where **move delta tokens cross-attend to board tokens**.

This is likely very useful for action-value/reranking:

```text
policy gives top-8
delta-action head estimates each afterstate
no full child forward pass needed
```

---

## 10. History tokens, but compressed and selective

Full move history can be wasteful. But some history matters:

```text
repetition
rule 50
castling rights
en passant
opening plan
recent tactical sequence
```

Rather than use 7 full history planes everywhere, use:

```text
64 current board tokens
+ 4–8 recent move tokens
+ 1 repetition/rule token
+ 1 castling/en-passant token
+ optional opening-plan token
```

A recent move token:

```text
move number relative to now
from
to
piece
capture/promotion/check
resulting material delta
```

This lets the model know “what just changed,” which may help tactics and repetition, without full 112-feature square history.

Lc0 and Chessformer use current plus previous seven plies/repetition features, but for a tiny model it is worth testing whether **compact recent-move tokens** give better strength-per-byte than full history channels. ([Leela Chess Zero][1])

---

## 11. Motif-token tokenizer: learned chess “BPE”

This is the most speculative but interesting idea.

Train a tokenizer/autoencoder over common board motifs:

```text
king shelter patterns
pawn structures
minor-piece outposts
fianchetto structures
rook on open file
battery motifs
pin motifs
passed-pawn structures
fortress motifs
```

Then a position becomes:

```text
raw square/piece tokens
+ detected motif tokens
```

The motif detector could be:

```text
hand-engineered at first
learned via VQ-VAE later
mined from intermediate CNN activations
```

Examples:

```text
"white kingside fianchetto"
"black isolated queen pawn"
"bishop queen battery on h7"
"rook on seventh rank"
"advanced passed pawn"
"back-rank weakness"
```

This is analogous to subword tokenization in language, but chess-specific. It may help tiny models because a motif token compresses a pattern that would otherwise require many layers to infer.

The danger is brittleness: motifs can be context-dependent. Use them as auxiliary tokens, not the only representation.

---

## 12. Search-state tokens for model-guided PUCT

If you keep small PUCT, feed partial search information back into a tiny network:

```text
root position tokens
+ candidate move tokens
+ current PUCT stats tokens:
    prior P
    visit count N
    Q
    uncertainty
    depth reached
```

Then the model can predict:

```text
which move deserves more search?
which candidate is unstable?
which child should be expanded?
```

This turns the model into a **search controller**, not just a position evaluator.

This is probably not V2, but it is a novel route for “strongest lightweight per millisecond.” It might reduce wasted PUCT nodes dramatically.

---

# My top-ranked research lanes

## Lane 1: CNN stem + move tokens + action-value

This is my top recommendation.

```text
CNN board trunk
  → square/piece embeddings
legal move tokens
  → cross-attend to board embeddings
  → policy + action-value + regret
global pooled board
  → WDL + uncertainty
conditional PUCT if uncertain
```

Why it is promising:

```text
CNN gives cheap local chess structure
move tokens make legal-move ranking explicit
action-value solves policy hallucination
attention handles global interactions only where needed
```

This is probably the best response to “CNNs seem more efficient overall.”

Call it:

```text
MoveFormer-CNN-AV
```

---

## Lane 2: Piece + ray tokens

```text
active piece tokens
+ ray/pin/battery tokens
+ legal move tokens
→ attention/reranker
```

Why it is promising:

```text
lower token count than 64 squares
directly exposes long-range chess tactics
more chess-native than generic attention
```

Call it:

```text
RayPieceFormer
```

This could be especially strong at tactics and queen-safety without hard-coded rules.

---

## Lane 3: NNUE-token value + transformer policy sidecar

```text
NNUE-lite sparse evaluator
+ Square/MoveFormer policy/action-value sidecar
```

Why it is promising:

```text
Stockfish-style speed for value/search
attention only for policy/global/candidate guidance
likely best strength-per-millisecond if engineered well
```

Call it:

```text
NNUE-PolicyFormer Hybrid
```

This is less “Leela-like,” but if the goal is strongest lightweight engine, it deserves serious attention.

---

## Lane 4: Latent analysis tokens over CNN features

```text
CNN square features
→ 16 learned analysis tokens
→ global attention reasoning
→ WDL/uncertainty/action-value
```

Why it is promising:

```text
cheaper than full 64-token attention
forces compact global representation
good for browser inference
```

Call it:

```text
PerceiverChess
```

---

## Lane 5: Compact history + memory tokens

```text
current board tokens
+ recent move tokens
+ retrieved memory/opening tokens
```

Why it is promising:

```text
handles repetition/plans/openings/personalization
could pair with compressed embedding memory later
```

This relates to the TurboQuant notes: TurboQuant-like compression is probably not useful for the model’s temporary 64-token attention, but it may become useful for compressed position embeddings and semantic memory. 

Call it:

```text
MemoryFormer Chess
```

---

# What I would not prioritize

I would **not** make these first-line research directions:

```text
pure FEN text transformer
large decoder-only move-sequence model
large MoE
generic LLM-style 4× FFN blocks
full autoregressive PGN generator as engine
long-context attention over entire game
TurboQuant-like KV cache for 64-token encoder inference
```

Reasons:

```text
too little chess inductive bias
too much latency
bad browser fit
not clearly stronger per byte
```

Lc0’s transformer progress explicitly says many generic transformer tricks, including MoE/GLU-style FFN modifications, did not noticeably improve performance, and that chess transformers did not benefit much from large FFN expansion ratios. ([Leela Chess Zero][1])

---

# Concrete experimental matrix

I would run a focused architecture tournament, not endless variants.

Use the same dataset, same optimizer, same parameter bands, and same evaluation.

## Parameter bands

```text
tiny:   0.5M–1M
small:  1M–3M
medium: 3M–8M
```

## Models

```text
A. 64×6 or 48×5 CNN baseline
B. Pure SquareFormer
C. CNN + Square attention hybrid
D. CNN + legal move-token action-value model
E. Piece/ray token model
F. NNUE-token value + move-token policy model
```

## Evaluation

```text
policy CE/KL
top-k accuracy
action-value ranking / Kendall tau
teacher regret
puzzle accuracy
queen/tactical failure suite
policy-only Elo
top-k rerank Elo
32/64-node PUCT Elo
browser latency
Elo per MB
Elo per millisecond
```

The infrastructure notes already warn that policy-only and PUCT should be measured separately because weak/noisy value can make search worse even when policy is useful. 

---

# The architecture I’d build next

If V0/V1 SquareFormer are promising but CNNs seem more efficient, I’d build this next:

```text
MoveFormer-CNN-AV v2
```

## Input

```text
board planes:
  current pieces
  side to move
  castling
  en passant
  rule state
  optional compact history

engine features:
  attack maps
  pinned pieces
  mobility buckets
  king-zone flags
```

## Trunk

```text
small CNN:
  width 64–96
  4–8 residual/SE blocks

square embeddings:
  [64, d_model]
```

## Legal move tokens

For each legal move:

```text
from square embedding
to square embedding
moving piece
captured piece
promotion
check/capture/castle flags
SEE-lite/material delta
king exposure delta
```

## Move-token cross-attention

```text
move tokens attend to:
  square embeddings
  active piece tokens
  optional ray tokens
```

Use only 1–3 lightweight cross-attention blocks.

## Heads

```text
per-move:
  policy logit
  action-value bucket
  regret
  tactical uncertainty

global:
  WDL
  value bucket
  moves-left / rule-state optional
```

## Inference

```text
1. CNN trunk once.
2. Generate legal move tokens.
3. Score all legal moves or top-k.
4. If confidence high, play best action-value move.
5. If confidence low, run small PUCT.
```

This preserves the best of your current roadmap:

```text
policy + value + action-value + uncertainty + conditional search
```

but changes tokenization from:

```text
all squares are equal tokens
```

to:

```text
local CNN features + legal moves are first-class tokens
```

That is likely more efficient for a small engine.

The existing search-light and DeepSeek-inspired briefs already point toward exactly this kind of system: action-value/ranking, uncertainty-gated search, multi-teacher/on-policy distillation, and multi-ply auxiliary targets. 

---

# Final opinion

Your skepticism is healthy. For a small engine, **pure SquareFormer is probably not the endpoint**. It is an excellent research baseline, but the strongest lightweight architecture is likely one of these hybrids:

```text
1. CNN + legal move tokens + action-value reranker
2. Piece/ray token transformer
3. NNUE-lite value + transformer policy/action sidecar
```

If I had to choose one next experiment, I would choose:

```text
CNN stem + legal move tokens + action-value/regret heads
```

because it attacks the biggest current weakness directly:

```text
not “can the model understand the board?”
but “can it judge the consequences of the moves it is about to play?”
```

That gives you CNN efficiency, transformer relational power, and action-value training in one architecture.

[1]: https://lczero.org/blog/2024/02/transformer-progress/ "Transformer Progress | Leela Chess Zero"
[2]: https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html?utm_source=chatgpt.com "NNUE | Stockfish Docs"
