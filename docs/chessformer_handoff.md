# Chessformer Handoff Notes: GC vs CF and Tiny-Leela Applications

**Context:** These notes summarize the relevant takeaways from *Mastering Chess with a Transformer Model* by Monroe and Chalmers, with a focus on how its ideas can inform a portable/tiny Leela-style chess engine. The paper compares its **CF** models, short for **Chessformer**, against **GC** models from Ruoss et al.'s *Grandmaster-Level Chess Without Search* / *Amortized Planning with Large-Scale Transformers* work.

## Executive summary

The strongest takeaway is not merely “use a transformer.” It is:

> For chess, **domain-specific board representation and attention geometry can substitute for a lot of scale**.

The Chessformer paper shows that a square-token transformer with a strong relative-position representation can match or beat much larger prior searchless transformer agents at far lower inference cost. For our project, this points to a hybrid architecture:

```text
Chessformer-style 64-square-token model
+ from-to policy head
+ WDL/value heads
+ optional action-value distillation from Stockfish/lc0
+ small PUCT or top-k value reranking
```

The best project move is probably **not** to clone the full CF-240M. Instead, build a **CF-lite / SquareFormer** in the 1M–6M parameter range and compare it directly against a `48x5` or `64x6` conv baseline.

---

## What “GC” and “CF” mean

### GC models

In the Chessformer paper, **GC-9M**, **GC-136M**, and **GC-270M** refer to the transformer models from Ruoss et al., *Amortized Planning with Large-Scale Transformers: A Case Study on Chess* / *Grandmaster-Level Chess Without Search*.

Their core recipe:

```text
Lichess games
→ unique board states
→ Stockfish 16 annotations
→ supervised transformer
→ searchless move selection at inference
```

The main GC model type predicts **action-values**:

```text
input:  FEN position + candidate UCI move
output: predicted Stockfish win-probability bucket for that move
policy: evaluate all legal moves and choose the move with highest predicted action value
```

They trained from **10 million Lichess games** annotated by **Stockfish 16**, yielding about **530M board states** and **15.3B action-value estimates**.

### CF models

**CF** means **Chessformer**, the architecture introduced in *Mastering Chess with a Transformer Model*.

Its core recipe:

```text
AlphaZero/lc0-style self-play data
→ static supervised dataset
→ 64 square-token encoder transformer
→ policy + value + auxiliary heads
→ policy-only or one-ply value agent
```

CF uses a chess-native representation:

```text
64 tokens = one token per chessboard square
112 features per square token
side-to-move canonicalization by board flipping
Shaw-style relative positional vectors inside attention
from-square × to-square policy head
WDL/q/short-term value heads
```

---

## Major differences between GC and CF

| Dimension | GC models | CF / Chessformer models | Tiny-Leela implication |
|---|---|---|---|
| Primary paper | Ruoss et al., *Grandmaster-Level Chess Without Search* | Monroe & Chalmers, *Mastering Chess with a Transformer Model* | Use GC for training/eval ideas; use CF for architecture. |
| Core goal | Distill Stockfish search into a searchless transformer. | Build a chess-native transformer architecture that is compute-efficient. | Combine them: CF architecture + GC-style action-value distillation. |
| Training source | Human Lichess positions annotated by Stockfish 16. | Static AlphaZero-style self-play data from an older RL run. | We can use either teacher-labeled data or self-play chunks. |
| Supervision | Stockfish state-values, action-values, and best moves. | Self-play policy/value targets plus auxiliary value/policy targets. | Use policy KL + WDL + q/action-value buckets. |
| Input representation | Flattened FEN text-like tokens; UCI action token for action-value prediction. | 64 square tokens; each token has current/past piece and rule features. | Square tokens are likely better for tiny models. |
| History | FEN has limited history; cannot fully handle threefold repetition. | Includes current + previous 7 positions and repetition indicators. | Include at least repetition/rule-50 features; full 7-ply history optional for v1. |
| Backbone | Decoder-only transformer over fixed FEN/action token sequence. | Encoder-only transformer over 64 board-square tokens. | Encoder-only square transformer is cleaner for browser inference. |
| Position encoding | Learned positional embeddings over token positions. | Domain-specific attention position representation; final model uses Shaw-style Q/K/V relative vectors. | Implement chess-relation bias first, then Shaw-lite if needed. |
| Move head | AV model evaluates `(position, candidate move)`; BC model predicts one of 1968 UCI moves. | Policy head builds a 64×64 from-to matrix plus promotion biases, with illegal moves masked. | Use CF from-to policy head; it maps naturally to legal movegen. |
| Inference cost | Action-value policy requires evaluating all legal moves. | Policy agent uses one model eval; value agent evaluates each legal child. | Browser v1 should be policy-first; rerank only top-k moves. |
| Search | No explicit search at inference. | No full MCTS in the paper; compares policy-only and depth-1 value agents. | Add small PUCT after policy/value are good. |
| Interpretability | Less board-geometric; token attention over FEN/action sequence. | Attention maps show bishop/rook/knight/king patterns, opponent queen, same-color squares, attackers. | Attention visualization is useful model debugging. |
| Best use for us | Dataset construction, action-value labels, value binning, Kendall tau eval. | Architecture, position representation, policy head, auxiliary heads, attention-debugging. | Natural fusion: **CF-lite + GC-style labels**. |

---

## Results comparison from the Chessformer paper

The Chessformer paper compares CF agents against GC models using a shared benchmark setup from Ruoss et al. The important point is **compute efficiency**.

| Agent | Elo | Puzzle accuracy | FLOPs per move/eval policy used in paper |
|---|---:|---:|---:|
| CF-6M-policy | 2105 ±28 | 65.3% | 214M |
| CF-240M-policy | 2347 ±10 | 93.5% | 12.8B |
| CF-240M-value | 2385 ±10 | 97.6% | 256B |
| GC-9M | 2007 ±15 | 85.5% | 14.2B |
| GC-136M | 2224 ±14 | 92.1% | 215B |
| GC-270M | 2299 ±14 | 93.5% | 427B |
| AlphaZero-policy | 1620 ±22 | 61.0% | 1.77B |
| AlphaZero-value | 1853 ±16 | 82.1% | 35.3B |

Key interpretation:

- **CF-240M-policy matches GC-270M puzzle accuracy and exceeds its Elo while using about 30× fewer FLOPs** in the paper’s accounting.
- **CF-6M-policy beats AlphaZero-policy with about 8× fewer FLOPs**.
- GC-9M has much higher puzzle accuracy than CF-6M-policy, but it is also much more expensive because the action-value policy evaluates many candidate moves.

Caveat: these are not perfectly apples-to-apples. GC is trained from Stockfish-supervised action values. CF is trained from self-play policy/value targets. GC action-value inference is effectively a learned one-ply reranker over legal moves. CF-policy is a one-shot policy model.

---

## Key learnings from the Chessformer paper

## 1. Board topology matters more than generic position encodings

A vanilla transformer needs position information because self-attention is permutation-invariant. The paper argues that ordinary Euclidean-distance biases are poorly matched to chess. A bishop may care more about a distant diagonal square than about a nearby orthogonal square; a rook cares about files/ranks; a knight cares about L-shaped offsets.

The authors compare three representations at the CF-6M scale:

| Position representation | Policy loss | Value loss | Policy accuracy | Value accuracy |
|---|---:|---:|---:|---:|
| Absolute position embedding | 0.3460 | 0.5607 | 57.44% | 89.11% |
| 2D relative bias | 0.3321 | 0.5586 | 58.23% | 89.26% |
| Shaw-style relative Q/K/V vectors | 0.3130 | 0.5549 | 59.27% | 89.53% |

The Shaw-style encoding improves policy accuracy by **1.83 percentage points** over learned absolute embeddings. The paper notes that, at this scale, doubling model size gave only about a 1.5-point policy accuracy gain, so representation quality can be comparable to a major size increase.

### Tiny-Leela action item

Implement position representations in this order:

```text
v0: learned square embedding
v1: chess-relation attention bias
v2: Shaw-lite relative Q/K/V vectors
```

A practical relation-bias table could include:

```text
same square
same rank/file
same diagonal/anti-diagonal
knight move
king move
pawn attack relation
rook ray distance bucket
bishop ray distance bucket
same color complex
edge/center/corner relation
```

---

## 2. Use 64 square tokens, not FEN text, for the main model

CF represents the board as:

```text
64 tokens = one token per square
```

Each square token contains:

```text
current piece + previous 7 positions, one-hot over 12 piece types
en passant and castling information
rule-50-ish counter
repetition indicators
```

This is essentially the classic lc0 plane representation rearranged from:

```text
[112, 8, 8]
```

into:

```text
[64, 112]
```

### Tiny-Leela action item

Use square tokens for the transformer track:

```text
SquareFormer input: [batch, 64, features]
```

Start with one of two encodings:

```text
compatibility mode: 112 features, current + 7 historical positions
portable mode: 18–32 current-position features + rule/state bits
```

The 112-feature version is closer to lc0/CF; the smaller current-board version is easier for browser and dataset work.

---

## 3. The from-to policy head is probably the right policy head for our project

The CF policy head maps square-token outputs into a move matrix:

```text
from-square query vectors
×
to-square key vectors
→ 64×64 ordinary move logits
```

Promotions are handled by adding promotion-piece logits for moves from the penultimate rank to the promotion rank. Illegal moves are masked before the final softmax.

### Tiny-Leela action item

Use this instead of immediately copying lc0’s legacy 1858/1862 policy encoding:

```text
ordinary_policy_logits = [64, 64]
promotion_logits = [from, to, promotion_piece]
legal_moves = generated by movegen
policy_over_legal_moves = gather + mask + softmax
```

Benefits:

- easier to debug;
- natural fit for `python-chess` / browser movegen;
- no fragile legacy gather table;
- lets us use top-k reranking directly over legal moves.

---

## 4. Static supervised data can replace online RL for early work

Chessformer trained from a **static dataset of self-play games** produced by an older AlphaZero-like run. The paper states that this avoided online data generation and sped up training, with no observed quality degradation in initial experiments.

### Tiny-Leela action item

Do not build distributed self-play first. Build:

```text
static teacher/self-play dataset
→ supervised training
→ policy/value eval
→ browser inference
→ only then limited self-play fine-tuning
```

Potential data sources:

```text
1. lc0-style self-play chunks
2. Stockfish/lc0 teacher-labeled positions
3. Maia/Lichess human games for human-like variants
4. puzzle/tactical positions for curriculum
5. tablebase-rescored endgames
```

---

## 5. Auxiliary targets are worth adding early

CF trains with policy and value heads, plus auxiliary targets:

```text
main policy
soft policy with temperature 4
WDL game result
q scalar value
q categorical value buckets
q prediction error
short-term value
short-term categorical value
short-term error
```

The paper says these auxiliary policy/value targets were used to accelerate convergence and improve final performance.

### Tiny-Leela action item

For v1, keep the auxiliary set smaller:

```text
required:
  policy KL / CE over legal moves
  WDL CE

recommended:
  q scalar or q bucket head
  value-error / uncertainty head
  soft-policy head

later:
  short-term value head
  moves-left head
  tactical metadata heads
```

A good first loss:

```text
L = 1.0 * KL(policy_teacher || policy_model)
  + 1.0 * CE(WDL_teacher, WDL_model)
  + 0.5 * CE(q_bucket_teacher, q_bucket_model)
  + 0.1 * MSE(value_error_target, value_error_pred)
  + 0.1 * KL(soft_policy_teacher || soft_policy_model)
```

---

## 6. Attention maps are a real debugging tool

CF’s attention maps show recognizable chess patterns:

```text
bishop-move heads
rook-move heads
knight-move heads
king-move heads
opponent queen attention
same-color-square attention
opponent-piece attention
attacker/defender attention
```

The paper notes that some movement-specialized heads remain fairly static across positions, while later heads can depend on accumulated information from early layers.

### Tiny-Leela action item

Build a tiny attention visualizer for SquareFormer:

```text
input: FEN + layer + head + query square
output: 8×8 heatmap of attention weights
```

Use it as a training sanity check:

```text
If no heads learn rank/file/diagonal/knight/king patterns, the model may be wasting capacity.
```

---

## 7. Do not blindly import LLM architecture tricks

The Chessformer appendix reports that several trendy transformer modifications did not help:

```text
GLU feed-forward replacement: slightly degraded quality at constant FLOPs
sparse MoE FFN: did not improve performance
soft MoE FFN: did not improve performance
```

### Tiny-Leela action item

For v1, keep the model boring:

```text
encoder block
Mish or GELU FFN
small FFN ratio, likely 1x–2x rather than 4x
chess-aware attention bias
from-to policy head
WDL/value heads
```

Spend complexity budget on **chess geometry and labels**, not MoE/GLU experiments.

---

## 8. Evaluate policy-only, value-rerank, and search separately

CF evaluates two agent types:

```text
policy agent:
  choose argmax policy move
  one model evaluation

value agent:
  evaluate each legal move after making it
  choose best child value
  about 20 model evaluations on average
```

This is a clean way to separate:

```text
raw policy quality
value quality
inference cost
search benefit
```

### Tiny-Leela action item

Adopt this evaluation ladder:

```text
Eval 0: policy CE/KL/top-k accuracy on held-out teacher labels
Eval 1: policy-only games
Eval 2: top-k value rerank, k = 4/8/16
Eval 3: small PUCT, 32/64/128 nodes
Eval 4: browser latency and model size
```

For browser, avoid evaluating every legal move by default. Use:

```text
policy top-k → value/action-value rerank top-k → optional PUCT
```

---

## Recommended architecture for our project

## SquareFormer-v0: debug model

```yaml
name: squareformer-v0
input:
  tokens: 64
  features: 32_to_112
body:
  layers: 4
  d_model: 64
  heads: 4
  d_ff: 128
  position: learned_square_embedding_plus_relation_bias
heads:
  policy: from_to_64x64_plus_promotions
  value: WDL
training:
  data: 100k_to_500k_positions
  objective: policy_CE + WDL_CE
purpose:
  validate tokenization, move mapping, masking, export, browser inference
```

## SquareFormer-v1: first serious model

```yaml
name: squareformer-v1
input:
  tokens: 64
  features: 112
body:
  layers: 6
  d_model: 128
  heads: 4_or_8
  d_ff: 128_or_256
  activation: Mish_or_GELU
  position: chess_relation_bias
heads:
  policy: from_to_64x64_plus_promotions
  value: WDL
  q_bucket: 32_or_64_bins
  uncertainty: optional
training:
  data: 1M_to_5M_positions
  objective: policy_KL + WDL_CE + q_bucket_CE + optional_soft_policy
export:
  webgpu: ONNX FP16
  wasm: INT8_QAT_later
```

## SquareFormer-v2: browser-strong model

```yaml
name: squareformer-v2
input:
  tokens: 64
  features: 112
body:
  layers: 8
  d_model: 192_or_256
  heads: 8
  d_ff: d_model_or_2x_d_model
  position: Shaw-lite_relative_QKV
heads:
  policy: from_to_64x64_plus_promotions
  value: WDL
  q_bucket: 64_bins
  action_value_topk: optional
  value_error: true
training:
  data: 5M_to_50M_positions
  objective: multi_teacher_policy + WDL + q/action_value + uncertainty
```

---

## Best fusion: CF architecture + GC-style action-value supervision

The GC paper’s most useful idea is **action-value supervision**: label candidate moves with a teacher value, not merely the best move. The CF paper’s most useful idea is **chess-native square-token architecture**.

The combined system:

```text
SquareFormer trunk
  → policy head over legal moves
  → WDL/q value head
  → action-value head for candidate moves
```

Training data per position:

```text
board features
legal moves
teacher policy distribution
teacher WDL/q
candidate move action-values for:
  teacher top-k moves
  model top-k moves
  checks/captures/promotions
  random legal distractors
```

Inference:

```text
1. One model pass gives policy + WDL.
2. Select top-k policy moves.
3. Rerank top-k with action-value head or child WDL.
4. Optional PUCT uses policy prior + WDL leaf values.
```

This is the most practical route for a portable but strong neural chess model.

---

## Concrete implementation tasks

## Data pipeline

- Define square-token feature schema.
- Define legal move representation: `(from, to, promotion)`.
- Build `policy_gather(position, logits_64x64, promotion_logits)`.
- Generate teacher labels:
  - teacher policy distribution;
  - WDL/q;
  - optional top-k action-values.
- Store sparse policy targets, not full dense vectors unless convenient.

## Model

- Implement `SquareFormerBlock`.
- Implement relation-bias attention.
- Implement from-to policy head.
- Implement WDL/q heads.
- Add optional soft-policy and value-error heads.

## Training

- Start with a small static dataset.
- Train v0 to overfit a tiny shard first.
- Then train v1 on 1M–5M positions.
- Compare against `64x6` conv baseline.
- Try stochastic weight averaging near the end.

## Evaluation

- Policy CE/KL.
- Top-1/top-3 teacher move accuracy.
- Kendall tau over legal move rankings if action-values exist.
- WDL calibration.
- Puzzle solving.
- Policy-only games.
- Top-k value-rerank games.
- 32/64/128-node PUCT games.
- Browser eval/sec and latency.

## Browser deployment

- Export ONNX FP16 for WebGPU.
- Keep a WASM/INT8 fallback target.
- Start with policy-only inference in a Web Worker.
- Add top-k reranking before full PUCT.
- Cache model and opening book separately.

---

## Project-level conclusions

1. **Use CF-style 64 square tokens as the main transformer architecture.** GC’s FEN-token model is simpler for large-scale supervised learning but less attractive for a tiny model.
2. **Use chess-aware position encoding early.** A relation-bias table is the best engineering compromise before full Shaw Q/K/V relative vectors.
3. **Use the CF from-to policy head.** It avoids lc0’s awkward legacy policy vector and maps cleanly to legal move generation.
4. **Use static supervised data first.** Both CF and GC show that expensive online RL is not required to get useful models.
5. **Borrow GC’s action-value labels.** They are excellent for top-k reranking and for teaching the model move ordering.
6. **Evaluate multiple inference modes.** Policy-only, top-k rerank, and PUCT answer different questions.
7. **Keep the model small and chess-specific.** Do not burn complexity on MoE/GLU before proving the representation and labels work.

---

## Source notes

- Monroe, D. and Chalmers, P. A. *Mastering Chess with a Transformer Model*. arXiv:2409.12272. https://arxiv.org/abs/2409.12272
- Ruoss, A. et al. *Amortized Planning with Large-Scale Transformers: A Case Study on Chess*. arXiv:2402.04494. https://arxiv.org/abs/2402.04494
- Searchless Chess / ChessBench code and data: https://github.com/google-deepmind/searchless_chess
- Chessformer training code reference in the paper: https://github.com/Ergodice/lczero-training

