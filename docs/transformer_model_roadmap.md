# Transformer Model Roadmap: Tiny 64-Square SquareFormer

Goal: build and evaluate a small Chessformer-inspired transformer against the current browser-sized residual CNNs, especially the guarded `48x5`, `64x6`, and `80x5 hybrid` ONNX models.

The core hypothesis is:

```text
64 square tokens
+ chess-aware attention bias
+ origin/destination policy head
+ supervised/distilled training

may beat a small conv tower at the same browser latency or file-size budget.
```

This roadmap is deliberately incremental. Do not replace the current CNN path until SquareFormer beats it in fixed supervised gates and at least one play/search gate.

## Baselines to beat

Current strongest 10M guarded models:

```text
48x5 e9 guarded
64x6 e12 EMA
80x5 hybrid e12 EMA
```

Known useful comparisons:

```text
Supervised dev:
  policy CE
  top1/top4/top8
  legal-move rank
  bucketed source/phase metrics

Play/search:
  policy-only arena
  top-k value rerank arena
  PUCT 32/64/128 visits
  browser eval latency
  ONNX size
```

The first target is not grandmaster strength. The first target is a fair architecture comparison against the existing CNNs on the same data and deployment constraints.

## New paper-note synthesis

The new Chessformer/Searchless notes strengthen the existing direction:

```text
Chessformer lesson: domain-specific square tokens + chess relation attention can substitute for scale.
Searchless/GC lesson: action-value and move-ranking labels amortize search into the model.
Project synthesis: SquareFormer first, then SquareFormer-AV-PUCT rather than pure searchless play.
```

Important updates:

- keep `v1` boring and chess-specific; avoid MoE/GLU/custom kernels for now;
- move from current `history=2` compact tokens toward a compatibility track with richer lc0/Chessformer-like history features when feasible;
- add action-value, value-bucket, and uncertainty heads after the policy/WDL baseline is stable;
- evaluate four inference modes separately: policy-only, top-k child-value rerank, action-value rerank, and small PUCT;
- add attention visualization as a real model-debugging tool, not just UI polish.

## Architecture target

### SquareFormer-v0: debug model

Purpose: prove tokenization, legal masking, policy mapping, training, export, and inference.

```text
input:       64 square tokens
features:    current board + rules only initially
layers:      4
model dim:   64
heads:       4
ffn dim:     128
params:      small/debug
heads:       from-to policy + WDL
```

Required capabilities:

- train on a tiny shard quickly;
- overfit a small batch;
- export to ONNX;
- run in Node/browser evaluator;
- map legal moves to logits without LC0 policy-index confusion.

### SquareFormer-v1: real tiny baseline

Purpose: compare seriously against small residual CNNs.

```text
input:       64 tokens × lc0-style history/state features
history:     current + 2 previous positions at first, expandable later
layers:      6
model dim:   128
heads:       4 or 8
ffn dim:     128 or 256
activation:  GELU or Mish
position:    chess relation bias
heads:       from-to policy + WDL + optional Q
params:      roughly 1M-3M
```

### SquareFormer-v1.5: action-value/search-light candidate

Purpose: make SquareFormer useful as a search-light engine, not just a supervised policy model.

```text
input:       64 tokens, preferably richer history/features than current h2 compact cache
body:        v1 trunk or slightly wider d192 variant
heads:       policy + WDL + value bucket + action-value top-k + uncertainty
training:    teacher/self-play move-ranking labels for candidate moves
inference:   policy top-k -> action-value rerank -> optional PUCT
```

Candidate action-value labels:

```text
Stockfish/lc0 MultiPV Q/WDL
PUCT root Q from self-play
teacher top-k + student top-k + checks/captures/promotions + random distractors
```

Success criterion: top-k action-value rerank is stronger than policy-only and reduces the PUCT nodes needed to match a baseline.

### SquareFormer-v2: browser-strong candidate

Only build after v1 is competitive.

```text
layers:      8
model dim:   192 or 256
heads:       8
ffn dim:     model_dim or 2× model_dim, not default 4× unless latency allows
position:    richer Shaw-lite relation vectors
heads:       policy + WDL + Q + uncertainty/value-error
params:      browser budget dependent, likely 3M-8M
```

### SquareFormer-v2-token: token-native quality/speed candidate

Purpose: keep the compact-cache speed win while improving chess-specific inductive bias and preparing for search-light play.

```text
input:       compact uint8 square tokens, transferred as uint8 and cast on GPU
body:        v1 trunk initially, then 8×96 / 6×160 / 8×128 ablations
position:    learned chess relation bias or SDPA-compatible relation embeddings
pooling:     learned global/CLS token for WDL and auxiliary heads
heads:       from-to policy + WDL + optional value bucket/action-value/uncertainty
```

Priority changes:

1. **uint8-to-device path**: preserve compact tokens through H2D transfer and cast to integer indices inside the model. Expected speed gain with no quality change.
2. **learned relation embeddings/bias table**: replace or augment the current additive attention-mask relation bias so optimized attention kernels remain available. Expected quality neutral-to-positive; speed depends on kernel path.
3. **custom SDPA attention path**: use explicit Q/K/V projections plus `scaled_dot_product_attention` for profiling and kernel control. Should be mathematically equivalent if relation handling matches.
4. **global token**: add a learned board/global token and use it for WDL/value/uncertainty pooling instead of mean-pooling only. Slight speed cost, likely quality-positive for global position evaluation.
5. **action-value-ready heads**: keep supervised policy/WDL canonical, but make the trunk/head layout ready for candidate-move Q, value buckets, and uncertainty labels.

Quality caveat: relation/global/action-value changes are architecture changes and require 10M gates before replacing the 100M canonical v1. The uint8 transfer path is implementation-only and should be safe.

## Input representation

Use one token per square.

```text
board position -> 64 tokens
square token -> piece/history/rules features for that square
```

Start simple, then match the residual cache input semantics.

### v0 features

- piece on square;
- side to move normalized perspective;
- castling rights broadcast or square-local flags;
- en-passant square flag;
- rule-50 counter as broadcast scalar;
- repetition flag if available.

### v1 features

- current board plus history planes from existing dataset rows;
- side-to-move normalized orientation;
- castling/en-passant/rule counters;
- same source labels and WDL targets as current residual dataset.

Important: keep a single canonical square order and document it. All ONNX/browser code must use the same order.

## Chess-aware positional attention

Implement positional handling in stages.

### Stage A: absolute square embedding

Baseline transformer positional representation:

```text
token_i += learned_square_embedding[i]
```

This is expected to be weaker, but it is the simplest sanity check.

### Stage B: 2D relative bias

Bias attention by rank/file deltas:

```text
score[h,i,j] = q[h,i] · k[h,j] / sqrt(d) + bias[h, dr, df]
```

### Stage C: chess relation bias

Primary target for v1.

Relations should include:

```text
same square
same rank
same file
same diagonal
same anti-diagonal
knight move
king move
pawn attacks by side
rook ray distance bucket
bishop ray distance bucket
same color complex
center/edge/corner relation
```

Use a compact learned bias table:

```text
bias[head, relation_id]
```

If multiple relations apply, either sum all matching relation biases or use a precomputed multi-hot relation matrix.

### Stage D: Shaw-lite relative vectors

Only after Stage C works.

Add learned relation vectors to Q/K/V paths. This is closer to Chessformer but more expensive and harder to export cleanly.

## Policy head

Use a from/to policy head rather than starting with LC0 flat policy indexing.

```text
hidden[64,d]
from_query = linear_from(hidden)   -> [64,d]
to_key     = linear_to(hidden)     -> [64,d]
ordinary_logits[from,to] = dot(from_query[from], to_key[to])
```

Promotion handling:

```text
promotion_logits[from,to,promo_piece]
```

Training/inference path:

```text
legal_moves = movegen(position)
legal_logits = gather logits for each legal move
policy_loss = CE/KL over legal legal_logits only
```

This should reduce mapping bugs and make browser search simpler.

## Value, action-value, and uncertainty heads

Required:

```text
WDL head from pooled board representation
```

Recommended after v1:

```text
value bucket head: 32 or 64 bins
Q scalar head
action-value head: score selected candidate legal moves
uncertainty/value-error head
```

Action-value head purpose:

```text
policy gives plausible moves
action-value predicts which plausible moves survive teacher/search verification
PUCT uses both as better priors/orderings
```

Uncertainty labels can come from policy/search disagreement, value swing during search, teacher disagreement, and actual regret/blunder outcomes.

Pooling options:

- learned CLS token;
- mean-pool over 64 square tokens;
- side-to-move king-square token plus mean-pool.

Start with mean-pool for simplicity.

## Training stages

### Stage 0: infrastructure sanity

- implement tokenizer;
- implement relation matrices;
- implement legal gather loss;
- train on 10k-100k rows;
- verify batch overfit;
- verify ONNX export;
- verify Node inference matches Python logits within tolerance.

### Stage 1: 10M supervised comparison

Train SquareFormer-v1 on the existing 10M dataset/cache or direct JSONL pipeline.

Suggested first run:

```text
model:       d128 L6 H4
schedule:    warmup + cosine
optimizer:   AdamW
weight decay:1e-4
EMA:         0.999
batch:       as large as GPU allows
loss:        policy CE + WDL CE
```

Compare against 10M CNNs on:

- dev CE/top-k;
- bucketed source/phase metrics;
- policy-only arena;
- PUCT 64 varied-opening arena;
- ONNX latency.

### Stage 2: positional ablation

Run identical v1 models with:

```text
A. absolute embedding
B. 2D relative bias
C. chess relation bias
D. Shaw-lite relation vectors, if ready
```

This directly tests the main Chessformer claim in our project.

### Stage 3: policy-head ablation

Compare:

```text
A. from-to legal-gather policy head
B. existing flat policy-vector head
```

Only keep the flat head if it is clearly easier or stronger.

### Stage 4: 100M scaling

If SquareFormer-v1 is competitive on 10M, train it on `supervised_100m_elite_tcec_v1` alongside the best CNN candidate.

Use the same fixed dev/gate sets so the comparison remains fair.

## Evaluation ladder

Use cheap gates before expensive PUCT.

```text
Eval 0: supervised CE/top-k/rank
Eval 1: bucketed source/phase reports
Eval 2: policy-only play from varied openings
Eval 3: top-k child-value rerank, k=4/8/16
Eval 4: top-k action-value rerank, k=8/16
Eval 5: conditional PUCT 32/64/128 visits
Eval 6: browser latency and memory
```

Add move-ranking metrics as soon as action-value labels exist:

```text
Kendall tau over legal/candidate moves
pairwise ranking accuracy
top-k regret vs teacher
action-value bucket CE/MSE
uncertainty calibration vs actual regret
```

Promotion threshold for SquareFormer:

```text
Must beat or match the best CNN on either:
  - policy/search strength at comparable latency, or
  - latency/file size at comparable strength.
```

## Browser/export requirements

- ONNX export works without custom ops.
- Web/client model registry can load SquareFormer metadata.
- Node and browser inference agree on legal move probabilities.
- Latency is measured at fixed visits and policy-only mode.
- Quantized or FP16 export is tested only after FP32/FP16 correctness.

## Debugging and interpretability

Add an optional attention visualizer after v1 works:

```text
select FEN
select layer/head
click source square
show attention heatmap over 64 squares
```

Healthy heads should specialize in patterns like:

```text
rank/file rays
diagonals
knight jumps
king-neighborhood
attacker/defender squares
promotion paths
same-color complexes
```

If all heads are local or random, relation bias or tokenization is likely wrong.

## Near-term architecture experiments after canonical v1

Run these as controlled 10M or 100k/1M smoke ablations before promoting to the 100M canonical lane:

```text
A. v1-token-h2: current compact embeddings baseline
B. v1-token-uint8-h2: same model, uint8 H2D transfer + GPU cast
C. v1-token-global-h2: add global token for WDL pooling
D. v1-token-relation-learned-h2: learned relation table instead of additive MHA mask
E. v1-token-sdpa-h2: explicit SDPA attention path
F. v1.5-av: add value bucket + candidate action-value + uncertainty heads
```

Measure:

```text
training rows/sec
GPU memory
supervised CE/top-k/WDL
bucketed dev eval
policy-only arena
PUCT 32/64/128
browser/ONNX latency
```

Promotion rule: implementation-only speed changes may replace baseline if logits/metrics match within noise. Architecture changes need supervised and play/search evidence.

## Implementation checklist

- [x] Create `training/train_squareformer_torch.py`.
  - initial JSONL/manifest trainer supports `--variant v0|v1`.
- [x] Add square-token encoder from dataset rows/FEN/history.
  - v0 uses current FEN; v1 defaults to 2 history FENs when present.
- [x] Add legal-move gather policy loss.
  - first implementation uses from/to/promotion class target from row policy UCI; full legal-mask KL remains future work.
- [x] Add from-to + promotion policy head.
- [x] Add WDL head.
- [x] Add absolute embedding baseline.
- [x] Add chess relation bias matrix.
- [x] Add ONNX export and metadata format.
+  - smoke ONNX artifacts written under `artifacts/squareformer/`.
- [ ] Add Node ONNX evaluator for SquareFormer.
- [ ] Add policy-only and PUCT arena support.
- [~] Run 10M v1 comparison against `80x5 hybrid e12 EMA`.
  - initial 100k-row/10k-dev smoke comparison complete:
    - v0 relation-bias, 4L d64, 3 epochs: CE `4.776766`, top1 `0.085400`, top4 `0.201800`, top8 `0.290600`.
    - v1 relation-bias, 6L d128, 3 epochs: CE `4.101683`, top1 `0.154700`, top4 `0.312000`, top8 `0.416800`.
  - added `--stream-train` for low-RAM JSONL/ZST training over large manifests.
  - streaming v1 timing probe, 200k rows/1 epoch/batch 384: `99.5s`, top8 `0.357700`.
  - added compact SquareFormer binary token cache builder:
    - `training/build_squareformer_token_cache.py`
    - `scripts/build_squareformer_cache_from_dataset.py`
  - built full 10M h2 cache at `data/datasets/supervised_10m_elite_tcec_v1/cache_squareformer_h2`: train `10,000,000`, dev `500,000`, size `7.1G`, token features `11`.
  - cached trainer path added via `--cache-manifest`; next checkpoint is v1 cached 10M training.
- [ ] Run positional ablation.
- [ ] Decide whether SquareFormer enters 100M training.

## Non-goals for the first pass

- Do not copy CF-240M.
- Do not add long-context compressed attention for a 64-token model.
- Do not implement custom WebGPU kernels before ONNX baseline works.
- Do not replace the current CNN model unless SquareFormer clears gates.
