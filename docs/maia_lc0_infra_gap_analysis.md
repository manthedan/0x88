# Maia / lc0 Infrastructure Gap Analysis

This note explains why a Maia-sized network does not behave like our current tiny Leela models, even when the nominal CNN size looks similar. The short version is that Maia/lc0 is a mature chess engine system, while our current stack is still mostly a supervised policy/value model plus a minimal search wrapper.

## Context

Maia networks are commonly described as lc0-style residual CNNs around the small-network family, roughly comparable in nominal size to a `64x6` residual CNN. That makes it tempting to ask:

> If Maia is roughly 64x6, why does our 64x6 still hang queens?

The answer is that parameter count is only one part of engine strength. Maia benefits from mature infrastructure around representation, labels, value calibration, search, engine integration, and evaluation. Our models are improving quickly, but many of those pieces are still young or missing.

## Key Difference

Our current system is close to:

```text
position -> supervised policy/value net -> optional minimal PUCT
```

Maia/lc0 is closer to:

```text
rich chess representation
+ large-scale curated training
+ mature policy/value network
+ mature MCTS/PUCT
+ hardened move encoding/runtime
+ strong engine/evaluation plumbing
```

So even if two networks have similar block/filter counts, they are not equivalent chess engines.

---

## 1. Input Representation Gap

Lc0-style networks use battle-tested input planes for chess state. These usually include rich information such as:

```text
piece placement history
side to move
castling rights
en passant state
repetition-related history
rule-state information
possibly move counters / no-progress features
```

Our stack has history-2 and state planes, and SquareFormer has square-token inputs, but these representations are still being validated.

Important gaps or risk areas:

```text
repetition awareness
fifty-move / no-progress awareness
castling and en passant edge cases
side-to-move/color transform invariants
history ordering consistency
browser/Node/Python parity
```

A model can only learn from the information provided. Missing or inconsistently encoded state can lead to search/value errors that mature lc0 nets avoid.

---

## 2. Policy Map Hardening Gap

Lc0's policy encoding has been hardened over years. It handles all move types and engine integration details reliably.

We are still hardening:

```text
moveToActionId / actionIdToMove parity
promotion and underpromotion mapping
castling representation
orientation / color perspective
legal move filtering
ONNX adapter parity
SquareFormer adapter parity
browser vs Node parity
```

Recent work found that `PUCT visits=1` matches policy-only for SquareFormer, which is a good sign. But the broader lesson remains: even small policy-map mismatches can make a decent model play nonsensically.

Permanent tests should include:

```text
policy top-k equals root PUCT priors
promotion/castling action map roundtrips
mirrored FEN consistency
browser evaluator vs Node evaluator parity
CNN vs SquareFormer adapter parity
```

---

## 3. Search / PUCT Maturity Gap

Our current PUCT is intentionally minimal. It does:

```text
expand legal moves
normalize legal priors
back up value with perspective flip
select by PUCT score
choose by visit count
```

Lc0 has years of engineering around:

```text
PUCT tuning
FPU / first-play urgency
root Dirichlet noise for self-play
temperature schedules
virtual loss
batched inference
transposition/cache behavior
terminal and draw handling
mate handling
resign logic
time management
move selection tie-breaks
```

We already found one real PUCT bug: temperature-0 selection broke tied visit counts by legal move order rather than using Q/prior. That could make low-visit search look worse than policy-only, especially for weak models with many tied child visits.

Fixed tie-break now:

```text
highest visits
then highest Q
then highest prior
```

This improves low-visit behavior, but our PUCT is still not lc0-level.

---

## 4. Value Calibration Gap

Search only helps if value is useful.

Our current evidence suggests:

```text
policy can become useful before value becomes search-useful
```

For SquareFormer 100k, a clean test showed:

```text
PUCT visits=1 == policy-only
PUCT visits>1 changes many moves
PUCT can appear worse than policy-only
```

That points less toward action-map bugs and more toward:

```text
weak/noisy value head
or value/search calibration issues
```

Maia/lc0 value heads are trained and used in a system where value quality is central. Our supervised value heads are still early and may not be strong enough to guide search.

For us, value needs dedicated evaluation:

```text
value calibration by game/result/eval buckets
value perspective tests
mate/stalemate terminal tests
PUCT visits sweep
policy-only vs PUCT arena comparison
queen-risk positions before/after value search
```

---

## 5. Tactical Verification Gap

Our queen diagnostics found that current models sometimes select queen-blundering moves with high confidence.

Examples from diagnostics showed moves like:

```text
selected move: queen move or queen-exposing move
policy probability: often high
rank: often 1
opponent reply: captures queen
Stockfish eval: large drop
```

This suggests a missing capability:

```text
candidate move -> opponent reply -> consequence
```

Maia/lc0 avoids many of these because:

```text
larger/stronger net has learned many tactical refutations
value head is better calibrated
search checks replies
mature engine handles tactics better
```

Our current supervised policy objective does not explicitly ask:

```text
If I play this candidate move, can opponent take my queen next move?
```

Recommended bridge for our system:

```text
one-ply queen/material verifier
action-value head
blunder-risk head
opponent-reply prediction head
Stockfish/lc0-generated tactical labels
```

This should be diagnostic/verification-first, not a hard rule that queen sacrifices are illegal.

---

## 6. Training Target and Data Gap

Maia was trained on large human game corpora carefully bucketed by rating. It benefits from scale, curation, and alignment to human play.

Our 100M dataset is a major step forward, but it is still newer and less proven. Current labels are mostly supervised move/value style labels. Missing or limited signals include:

```text
action-value labels
opponent best-reply labels
explicit blunder-risk labels
search-improved policy labels
uncertainty labels
material/tactical outcome labels
```

Pure policy imitation can learn many good moves, but it may not punish tactical refutations enough unless the data and model capacity are sufficient.

For unacceptable queen blunders, more epochs may reduce frequency, but relying only on more epochs is risky. We need targeted objectives.

---

## 7. Self-Play / Search-Improved Training Gap

Lc0's ecosystem is search-aware. Even when Maia is human-imitation oriented, it lives inside an lc0-style architecture and runtime.

Our current core training path is mostly:

```text
supervised position -> teacher/human move
```

Missing or early:

```text
self-play generation
search-improved policy targets
policy/value consistency training
action-value training
reply-prediction training
promotion loop
uncertainty-aware selection
```

The DeepSeek/search-light direction is important here. Instead of only predicting the played move, the model should learn things like:

```text
for candidate move a:
  likely opponent reply
  resulting value
  tactical risk
  uncertainty
```

This is especially natural for SquareFormer-style architectures.

---

## 8. Engine Integration Gap

Lc0 is a mature UCI chess engine. It has robust behavior around:

```text
UCI protocol
time controls
nodes/visits controls
pondering/search lifecycle
resignation
repetition/draw claims
threading/batching
hash/cache behavior
```

Our engine/eval integration is improving but still young:

```text
UCI anchor arena still being hardened
Maia/lc0 anchor illegal handling has shown caveats
browser ONNX external-data handling needed fixes
SquareFormer evaluator was recently added
PUCT tests are new
queen diagnostics are new
```

This matters because engine bugs can masquerade as model weakness.

---

## 9. Evaluation Maturity Gap

Lc0/Maia comparisons usually rely on mature engine protocols. We are building our protocol discipline now:

```text
fixed openings
reversed pairs
fixed visits/nodes
anchor pools
Stockfish/Maia/Reckless/lc0 anchors
bucketed supervised eval
queen/material blunder diagnostics
browser parity checks
```

Important rule for our project:

```text
Elo claims must be protocol-relative.
```

Always report:

```text
anchor pool
backend
search budget
openings
games
WDL
error bars
illegal count
caveats
```

---

## 10. Time Management and Deployment Gap

Lc0 is optimized for real play. It allocates time, batches inference, reuses search state, and handles search lifecycle efficiently.

Our current comparisons are mostly fixed-budget:

```text
policy-only
PUCT visits=1/8/16/32/64
simple browser runtime
```

That is correct for model comparison, but it is not yet mature engine play.

Browser deployment also has constraints:

```text
latency
single-file ONNX packaging
WebAssembly/WebGPU behavior
memory limits
deterministic fallback
UI responsiveness
```

A model that is good offline must still be verified in browser runtime.

---

# What We Should Do Next

## Near-term hardening

1. Keep PUCT core tests permanent:

```text
policy identity at visits=1
value perspective flip
terminal mate/stalemate
prior normalization
selection tie-breaks
```

2. Add root-prior parity tests for real models:

```text
evaluator legal top-k == PUCT root prior top-k
```

3. Keep queen fixed-suite as a release gate:

```text
selected queen-risk rate
actual immediate queen capture risk
Stockfish eval drop
white/black split
policy mass on risky moves
```

4. Treat policy-only and PUCT separately:

```text
policy-only strength
value calibration
PUCT strength by visits
```

Do not assume search helps until value proves it.

## Medium-term model improvements

For CNNs:

```text
finish 32x4 / 48x5 / 64x6 / 80x5 100M baselines
measure capacity curve
use best CNN as deployment fallback and benchmark anchor
```

For SquareFormer:

```text
continue v1 if 100M e3 is promising
skip v0 unless needed for ablation
build v2 with action-conditioned heads
```

Recommended SquareFormer v2 heads:

```text
policy head
value head
action-value head
blunder-risk head
optional opponent-reply head
```

Action decoder sketch:

```text
h_from = square embedding[from]
h_to   = square embedding[to]
h_g    = global board embedding
e_move = promotion/move-type embedding

action_feature = MLP([h_g, h_from, h_to, e_move])

policy_logit[action] = policy_mlp(action_feature)
q_value[action]      = q_mlp(action_feature)
risk_logit[action]   = risk_mlp(action_feature)
```

This directly targets queen suicide and other tactical one-ply failures.

## Longer-term engine direction

To approach Maia/lc0 behavior, we need:

```text
better value calibration
search-aware training
opponent-reply modeling
self-play/search-improved policy loop
mature PUCT with FPU/noise/temperature/time management
robust browser/Node parity
stronger UCI engine integration
```

---

# Bottom Line

A Maia-sized network is not enough. Maia's strength and queen-safety come from the whole lc0 ecosystem:

```text
representation
policy map
training data
value calibration
search
engine runtime
evaluation discipline
```

Our current tiny Leela stack is making good progress, but the most important gaps are:

```text
1. value calibration
2. tactical verification
3. action/opponent-reply training
4. mature PUCT/search infrastructure
5. hardened input/policy/runtime parity
```

The CNN baselines will tell us how far compact supervised convolutional models can go. But if SquareFormer v1 shows serious promise, the higher-upside path is to pursue transformer-style action-value and opponent-reply modeling rather than merely scaling plain policy imitation.
