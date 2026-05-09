# DeepResearch architecture triage, 2026-05

## Verdict

The DeepResearch report did **not** change the active roadmap. Its useful contribution is external confirmation and a sharper ordering of ideas already present in the TinyBT/SquareFormer plans:

```text
search-light SquareFormer
+ cheap chess geometry/history bias
+ legal-candidate action-value decoder
+ explicit regret/risk metrics
+ conditional PUCT only when needed
```

Do not use this report to restart the architecture plan or expand the experiment matrix. Use it to tighten the next TinyBT/SquareFormer deltas after the current 10M/100M h7/h8 BT4 pipeline is stable.

## Accepted useful ideas

### 1. Search-light is the right framing

Keep the objective as:

```text
strongest lightweight engine per millisecond, not searchless purity
```

The desired inference ladder remains:

```text
policy-only
policy + AV/risk rerank
top-k child / candidate verification
small conditional PUCT
larger teacher/search only offline
```

This supports the current project defaults:

- classic PUCT remains the safe default search path;
- AV/aux-PUCT remains opt-in until calibrated;
- conditional search is the product direction for compact models.

### 2. Cheap geometry/history bias before exotic attention

For 64 square tokens, full attention is not the bottleneck. The next trunk improvement should therefore be better chess bias, not generic efficient attention.

Useful concrete direction:

```text
static relation bias
+ side-relative geometry
+ color-complex / ray / knight / pawn / king-zone templates
+ h7/h8 history traces
+ optional smolgen-lite template gates
```

This should be evaluated before attack-graph sparsity, conv-graph hybrids, or recurrent verification.

### 3. Move-query legal-candidate decoder is the best next head/interface bet

After a stable SquareFormer/TinyBT baseline, the best architectural delta is a small decoder over legal candidate moves:

```text
64 square-token trunk
-> shortlist top-k legal moves
-> move tokens: from, to, promo, move type, legality witness
-> 1-2 cross-attention layers over board tokens
-> refined policy, AV, regret, risk / needs-verification
```

Initial candidate sizes:

```text
tiny:  k=8
small: k=12
medium: k=16
```

This is the cleanest way to spend a little extra compute only where it helps: ranking legal moves and identifying moves that require search.

### 4. Track ranking/regret metrics explicitly

Add or emphasize metrics that measure whether the model avoids high-cost candidate mistakes, not just whether it imitates the played move.

Preferred metrics:

```text
AV NDCG@k
candidate-rank Kendall/Spearman tau
regret@1
p95 regret
policy top-k on legal moves
risk calibration / needs-verification calibration
tactical-slice failure rates
```

Critical tactical slices:

```text
queen-loss after reply
hanging pieces
mate-in-N / forced tactics
sound sacrifice vs unsound material loss
pins / skewers / overloaded defenders when labels are available
```

### 5. Start motif supervision narrowly

The report's motif-head idea is useful only if kept small. Do not build a large motif ontology first.

Accepted near-term heads:

```text
material swing / after-reply value
queen-loss or major-piece-loss risk
mate / forced-tactic bucket if labels are reliable
needs-verification / high-regret risk
```

Keep most motif heads training-only unless a small runtime risk vector clearly improves conditional search.

## Immediate action ordering

Do not interrupt current execution:

```text
1. Finish 10M h8/h7 BT4 training.
2. Finish true 100M h8 dataset.
3. Build/validate 100M h7 and h8 SquareFormer caches.
4. Train 100M BT4/SquareFormer baseline.
```

First serious architecture deltas after the stable baseline:

```text
A. Symmetry/history geometric-bias v2.
B. Move-query legal-candidate decoder with k=8/12.
C. AV/regret/risk heads with explicit ranking metrics.
```

Only after those have clean ablations should we revisit graph sparse attention, conv-graph hybrids, recurrent verification, or adapter distillation.

## Evaluation/card additions

Any promotion card for these deltas should include:

```text
params and bytes
p50/p95 latency
policy CE/top-k
WDL/value calibration
AV NDCG@k or rank correlation
regret@1 and p95 regret
tactical-slice summary
policy-only Elo
AV-rerank Elo
classic PUCT Elo at fixed visits
conditional-PUCT Elo if used
```

This is the useful evaluation contribution from the report: make action-value quality and regret visible before spending arena time.
