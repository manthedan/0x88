# Queen-Safety and Tactical Verification Design for a Tiny Neural Chess Engine

## Purpose

This document addresses a common failure mode in small neural chess engines:

```text
The model plays strong, coherent chess for many moves, then suddenly hangs or suicides the queen and loses.
```

The goal is **not** to forbid queen sacrifices. Sound queen sacrifices are central to tactical chess and to any romantic or aggressive personality mode. The goal is to distinguish:

```text
sound queen sacrifice     = queen loss with tactical, mating, material, or positional compensation
queen suicide / blunder   = queen loss without compensation, usually due to missing a forcing reply
```

The recommended solution is a layered system:

```text
1. First verify the plumbing.
2. Add deterministic diagnostics and failure capture.
3. Add soft tactical verification at inference time.
4. Add action-value, regret, and multi-ply auxiliary training.
5. Add a tactics specialist teacher and on-policy hard-negative mining.
6. Use uncertainty-gated search only when the model is tactically unsure.
```

This design assumes the broader project may use a conv net, a SquareFormer/Chessformer-style transformer, or a hybrid policy/value/action-value model. It is compatible with policy-only, top-k action-value reranking, and small PUCT search.

---

## Executive Summary

Do **not** implement a hard rule like:

```text
if move loses queen: illegal
```

That will destroy real queen sacrifices.

Instead implement:

```text
if move risks queen loss:
    estimate compensation
    estimate opponent best reply
    estimate after-reply value
    if risk is high and compensation is low:
        downrank or trigger verification search
```

The central runtime score can be:

```text
score(move) =
    action_value(move)
  + α * log_policy(move)
  - β * queen_loss_without_compensation(move)
  - γ * tactical_refutation(move)
  - δ * loss_spike_after_reply(move)
  + η * style_bonus(move)
```

Sound sacrifices survive because their compensation estimate, after-reply value, or tactical continuation remains high.

---

# Part I: Plumbing Diagnostics First

Before retraining, assume the failure may be a bug. Queen blunders are often the first visible symptom of a perspective, move-index, or search-backup bug.

## 1. Build a Failure Capture Harness

Every queen-suicide incident should be reproducible from a single record.

### Required failure record

```json
{
  "engine_version": "tiny-leela-0.3.1",
  "model_id": "sqf_128x6_fp16_0042",
  "backend": "onnxruntime-webgpu",
  "precision": "fp16",
  "search_mode": "policy_av_rerank",
  "rng_seed": 123456,
  "fen_before": "...",
  "selected_move_uci": "d1h5",
  "selected_move_index": 1234,
  "legal_moves_uci": ["e2e4", "d2d4", "d1h5"],
  "raw_policy_logits_top": [["d1h5", 8.91], ["e2e4", 7.12]],
  "masked_policy_probs_top": [["d1h5", 0.42], ["e2e4", 0.18]],
  "wdl_before": [0.41, 0.36, 0.23],
  "q_before": 0.18,
  "action_values_top": [["d1h5", 0.31], ["e2e4", 0.25]],
  "fen_after_selected": "...",
  "teacher_best_reply": "g7g6",
  "teacher_eval_before_cp": 35,
  "teacher_eval_after_move_cp": -620,
  "teacher_eval_after_reply_cp": -780,
  "queen_lost_within_plies": 2,
  "search_stats": {
    "nodes": 64,
    "root_visits": [["d1h5", 22], ["e2e4", 17]],
    "root_q": [["d1h5", 0.29], ["e2e4", 0.22]]
  }
}
```

### Minimum fields

If the full schema is too much, store at least:

```text
FEN before
selected move
legal moves
model top-k policy
model WDL/value
teacher best reply
teacher eval drop
position after selected move
position after teacher reply
backend / model / precision / seed
```

---

## 2. Define the Failure Class Precisely

Use labels that distinguish queen losses from queen sacrifices.

```text
queen_risk:
  The queen is attacked, trapped, or capturable after the selected move.

queen_lost_after_reply:
  The opponent's best or obvious reply wins the queen within N plies.

queen_loss_without_compensation:
  queen_lost_after_reply is true and the position value collapses.

sound_queen_sacrifice:
  queen_lost_after_reply is true but teacher/search/tablebase says the move is good or acceptable.
```

Suggested thresholds:

```text
N plies for queen-loss check: 1 to 3 initially
centipawn collapse: >= 300 cp
WDL loss spike: >= 0.20 increase in loss probability
compensation: mate, forced win, major material recovery, or teacher value near/equal/better than baseline
```

---

## 3. Create a Queen-Safety Regression Suite

Create a small suite before changing anything.

Recommended categories:

| Category | Count | Purpose |
|---|---:|---|
| Pure queen blunders | 50–200 | Ensure the engine avoids obvious queen hangs. |
| Sound queen sacrifices | 50–200 | Ensure the engine does not become cowardly. |
| Queen trades | 50–200 | Ensure exchanges are handled neutrally. |
| Temporary queen en prise | 50–200 | Ensure the engine sees tactical immunity. |
| Queen-winning tactics | 50–200 | Ensure the engine can win the opponent queen. |
| Low-elo puzzle queen sacs | 50–200 | Preserve common tactical motifs. |
| Model-mined failures | grows over time | Track real regressions from self-play. |

For each test position, store:

```text
FEN
candidate moves
expected safe/unsafe label
teacher best move
teacher best reply
teacher eval before/after
sound sacrifice flag
```

---

# Part II: Plumbing Audit Checklist

Run these checks before adding model complexity.

## 4. Move Encoding and Legal Mask Invariants

### Round-trip tests

For every legal move in a test position:

```text
move object → UCI → policy index / from-to representation → UCI → move object
```

Assert:

```text
round_trip_move == original_move
```

Include special cases:

```text
castling: e1g1, e1c1, e8g8, e8c8
promotions: e7e8q, e7e8n, captures with promotion
en passant
underpromotions
checks
mates
queen moves from every file/rank/diagonal
black-to-move positions
```

### Legal mask tests

For every position:

```text
sum(policy_probs[legal_moves]) ≈ 1.0
policy_probs[illegal_moves] == 0 or logits == -inf after masking
all legal moves are represented exactly once
no illegal move appears in top-k after masking
```

### Common bug symptoms

| Symptom | Likely bug |
|---|---|
| Engine chooses impossible-looking move | Legal mask / move map mismatch |
| Engine sacrifices wrong piece | Policy index maps to different UCI move |
| Promotions behave strangely | Promotion indexing bug |
| Castling weirdness | Castling rights encoding or move map bug |
| Works as White but not Black | Side-to-move canonicalization bug |

---

## 5. Board Encoding and Perspective Tests

Small neural engines often fail because the board is encoded from the wrong perspective.

### Required tests

For a position `P` and color-flipped mirror `P'`, verify:

```text
policy transforms correctly under board flip
WDL/value transforms correctly under side-to-move flip
piece planes identify queen color correctly
castling/en-passant/rule-state features survive round-trip
```

If using side-to-move canonicalization:

```text
encode(P, white_to_move)
encode(flip_colors_and_board(P), black_to_move)
```

should have consistent transformed features.

### WDL/value perspective

Choose one convention and enforce it everywhere:

```text
WDL is from side-to-move perspective
```

Then:

```text
Q = W - L
```

After making a move, the side to move flips, so child value must be negated when backing up to the parent:

```text
parent_q_for_move = -child_q
```

For WDL:

```text
child W for opponent = parent L
child L for opponent = parent W
```

---

## 6. Search Backup Tests

If using PUCT or any tree search, test with tiny synthetic trees.

### One-ply value backup

Create a position where one legal move wins the opponent queen and another loses your queen. Hard-code leaf values:

```text
move_good child_q = -0.80 from opponent perspective
move_bad  child_q = +0.80 from opponent perspective
```

After perspective flip:

```text
move_good parent_q = +0.80
move_bad  parent_q = -0.80
```

If this fails, search is backing up the wrong sign.

### Visit-count sanity

At the root, after many iterations:

```text
high-visit move should usually match best searched move
low-visit lucky high-Q move should not dominate final selection
```

Record:

```text
P, N, Q, U, Q+U for each root move
```

### Common PUCT bugs

| Symptom | Likely bug |
|---|---|
| Engine likes moves that are good for opponent | Missing sign flip |
| Engine always follows initial policy | Q not updated or c_puct too high |
| Engine ignores policy entirely | P not initialized or c_puct too low |
| Engine plays high-Q one-visit move | Choosing by Q instead of visits/root policy after unstable search |
| Engine collapses in black positions | WDL perspective bug |

---

## 7. Training Label Alignment Tests

Training targets must use the same move mapping and perspective as inference.

### Policy target checks

For a batch of training examples:

```text
argmax(policy_target) → UCI move
```

Manually inspect positions. Make sure the target move is legal and plausible.

Check:

```text
policy_target mass only on legal moves
policy_target move index maps to same UCI as inference
teacher top move round-trips through the same mapper
```

### Value target checks

For game result labels:

```text
if side to move eventually wins: WDL = [1, 0, 0]
if side to move eventually loses: WDL = [0, 0, 1]
if draw: WDL = [0, 1, 0]
```

Do not accidentally label from White's perspective during training and side-to-move perspective during inference.

---

## 8. Export and Backend Parity Tests

Before trusting browser play, compare outputs across backends.

For a fixed set of 1,000 positions:

```text
PyTorch/JAX output
ONNX CPU output
ONNX WebGPU output
ONNX WASM output
quantized output, if applicable
```

Compare:

```text
max absolute logit difference
policy top-1/top-5 agreement
WDL difference
action-value top-k agreement
```

Suggested tolerances:

```text
FP32 → ONNX FP32: very tight
FP32 → FP16: top-k should mostly agree; WDL close
FP16 → INT8: track degradation; do not assume okay
```

If queen blunders appear only in one backend, suspect:

```text
precision overflow
incorrect mask application
softmax numerical issue
layout/order mismatch
operator export bug
```

---

## 9. Deterministic Reproduction

All failure tests should be deterministic.

Pin:

```text
model checkpoint
backend
precision
random seed
temperature
top-k/top-p settings
opening book seed
search node budget
thread count, if relevant
legal move ordering
```

A non-deterministic queen hang is much harder to debug.

---

# Part III: Immediate Non-Training Fixes

These can reduce catastrophic queen blunders while the model is improved.

## 10. Soft Queen-Safety Verifier

Implement a verifier that produces features, not hard bans.

For each candidate move:

```text
make move
compute opponent attacks
is our queen attacked?
can opponent capture queen?
what is static exchange evaluation of queen capture?
does our move give check/mate/threat?
can we recapture or force mate?
```

Output:

```text
queen_en_prise_after_move: bool
queen_capture_legal_for_opponent: bool
queen_capture_see: material estimate
queen_has_tactical_immunity: bool
move_is_check_or_mate_threat: bool
```

Then:

```text
if queen risk is high and compensation is unknown:
    downrank or trigger deeper verification
```

Do not automatically reject.

---

## 11. Suspicious-Move Trigger

Trigger extra verification when any of these are true:

```text
top move moves queen to attacked square
top move leaves queen undefended and attackable
top move has high policy but low action-value
top move causes large material SEE loss
model uncertainty is high
policy entropy is high
a tactical-refutation head fires
opponent has immediate queen capture/check/mate reply
```

Verification options, from cheapest to strongest:

```text
1. static attack/SEE check
2. top-k child value rerank
3. one-ply opponent best-reply check
4. 32–128 node PUCT
5. external teacher check in analysis/offline mode
```

---

## 12. Top-k Child Value Rerank

If the policy top move is suspicious, evaluate child positions for the top-k moves.

```python
policy, wdl, uncertainty = model(position)
candidates = top_k_legal(policy, k=8)

scores = []
for move in candidates:
    child = position.make(move)
    child_policy, child_wdl, child_unc = model(child)
    score = -wdl_to_q(child_wdl)  # flip perspective
    scores.append((move, score))

play = argmax(scores)
```

This is not full search, but it catches many one-ply disasters.

---

## 13. Conditional Small PUCT

Use small search only when the safety verifier or uncertainty head says the position deserves it.

```text
confident quiet position:
  policy + action-value rerank

suspicious queen/tactical position:
  32–128 node PUCT

endgame with tablebase:
  exact tablebase
```

This keeps the engine lightweight while preventing the worst tactical collapses.

---

# Part IV: Model Improvements

## 14. Add an Action-Value Head

A policy head answers:

```text
Which moves look plausible?
```

An action-value head answers:

```text
How good is this candidate move?
```

For queen blunders, action-value is often the missing capability.

### Suggested architecture

```text
board_embedding = trunk(position)
move_embedding = embed(from_square, to_square, promotion)
input = concat(board_embedding, move_embedding, tactical_features)
action_value = MLP(input) → value bucket or scalar Q
```

Train on candidate moves:

```text
teacher best move
model selected move
policy top-k moves
checks/captures/promotions
random legal moves
known queen blunders
sound queen sacrifices
```

Targets:

```text
Stockfish/lc0 value after move
PUCT Q value
WDL after best reply
value bucket
```

---

## 15. Train Move Regret

The model should learn not only good moves but also how bad bad moves are.

Define:

```text
regret(move) = teacher_value(best_move) - teacher_value(move)
```

Examples:

```text
normal inaccuracy: small regret
queen hang: huge regret
sound queen sacrifice: low regret
```

Useful loss:

```text
L_rank = max(0, margin - score(good_move) + score(bad_move))
```

For queen sacrifices, include contrastive pairs:

```text
score(sound_queen_sac) > score(materialistic_safe_but_worse_move)
score(safe_good_move)  > score(queen_hang_without_compensation)
```

This directly prevents the model from treating all queen losses as equivalent.

---

## 16. Add Queen-Safety and Compensation Heads

Add auxiliary heads that explicitly predict the relevant tactical concepts.

Recommended heads:

```text
queen_en_prise_after_move: binary
queen_lost_after_best_reply: binary
best_reply_captures_queen: binary
material_delta_after_best_reply: bucketed scalar
after_reply_wdl: W/D/L
after_reply_value_bucket: categorical value
sacrifice_soundness: {sound, dubious, losing}
tactical_refutation_probability: scalar
```

The most important head is:

```text
after_reply_value(position, move)
```

because queen blunders usually happen when the model misses the opponent's forcing reply.

---

## 17. Multi-Ply Auxiliary Heads

DeepSeek-style multi-token prediction maps naturally to chess as multi-ply prediction. The main head predicts the current move; auxiliary heads predict the opponent reply and short continuation.

Recommended auxiliary targets:

```text
ply_0_policy:
  search-improved policy at current position

ply_1_reply_policy:
  opponent's best or search-improved reply after selected/top candidate move

ply_1_after_reply_wdl:
  WDL after opponent's best reply

ply_1_best_reply_captures_queen:
  whether the reply wins the queen

ply_2_continuation_policy:
  our best continuation after the opponent reply

pv_moves:
  first 2–4 moves of teacher/search principal variation
```

Suggested loss:

```text
L =
  KL(policy_target, policy_pred)
+ CE(WDL_target, WDL_pred)
+ CE(value_bucket_target, value_bucket_pred)
+ λ_av  * action_value_loss
+ λ_r1  * KL(reply_policy_target, reply_policy_pred)
+ λ_v1  * CE(after_reply_value_bucket)
+ λ_qs  * BCE(queen_lost_after_reply)
+ λ_pv  * CE(PV_move_tokens)
```

Starting weights:

```text
λ_av  = 0.5
λ_r1  = 0.25
λ_v1  = 0.25
λ_qs  = 0.25
λ_pv  = 0.10
```

Tune based on whether the auxiliary heads improve or distract from policy/value performance.

---

## 18. Add an Uncertainty / Value-Error Head

The model should know when not to trust itself.

Labels can come from:

```text
absolute error between model value and teacher value
policy disagreement with search
large value swing after PUCT
teacher disagreement between Stockfish and lc0
failed puzzle positions
queen-risk verifier flags
```

Use at runtime:

```text
if uncertainty high:
    run small PUCT or stronger verification
else:
    use policy + action-value rerank
```

This turns tactical verification into adaptive compute rather than always-on overhead.

---

# Part V: DeepSeek-Style Specialist Distillation

## 19. Tactics Specialist Teacher

Create a tactical teacher focused on queen safety, material swings, checks, mates, and forcing replies.

Teacher sources:

```text
Stockfish shallow/deep searches
lc0 search for Leela-like policy/value
Syzygy tablebases for endgames
puzzle solution lines
self-play failure positions
synthetic queen-trap positions
human queen-sac games for sound sacrifice examples
```

Labels:

```text
teacher top-k policy
teacher value/WDL
candidate move action-values
best opponent reply
reply policy
PV line
queen_lost_after_reply
material delta after reply
sound sacrifice label
tactical refutation probability
```

Training:

```text
student learns full legal-move distribution where possible
student learns top-k action-values
student learns tactical auxiliary heads
```

This is a specialist-to-unified-student distillation pattern: instead of hoping one tiny model learns tactics from noisy data, route tactical positions to a tactical teacher and distill the labels back into the main model.

---

## 20. On-Policy Hard Negative Mining

The fastest improvements will come from the model's own failures.

Loop:

```text
1. Let the current model play or analyze positions.
2. Capture moves where queen/material loss occurs.
3. Reanalyze with tactical teacher.
4. Add labeled failures to the training set.
5. Oversample them until fixed.
6. Keep a regression suite so they never come back.
```

Mining criteria:

```text
selected move is high-policy or high-confidence
queen lost within 1–3 plies under teacher best reply
teacher eval drops by >= 300 cp or WDL loss rises by >= 0.20
no mate or adequate compensation appears
```

Also mine **false positives**:

```text
move loses queen but is sound
teacher says move wins or draws comfortably
model/verifier incorrectly punished it
```

These are essential to preserve queen sacrifices.

Suggested batch mix:

```text
70% normal policy/value data
10% model-mined queen/material blunders
10% sound sacrifices / tactical motifs
5% puzzles / forcing lines
5% tablebase/endgame exact positions
```

Adjust once the failure rate changes.

---

## 21. Sacrifice Taxonomy Dataset

Build a small, high-quality taxonomy dataset.

Categories:

```text
pure queen blunder
temporary queen en prise but tactically immune
queen trade
queen sacrifice for mate
queen sacrifice for material recovery
queen sacrifice for attack/initiative
speculative but playable sacrifice
unsound romantic sacrifice
opponent queen-winning tactic
```

For each example:

```json
{
  "fen": "...",
  "candidate_move": "h5f7",
  "category": "queen_sacrifice_for_mate",
  "soundness": "sound",
  "teacher_eval_before_cp": 30,
  "teacher_eval_after_move_cp": 900,
  "teacher_best_reply": "e8f7",
  "pv": ["h5f7", "e8f7", "f3g5", "f7g8"],
  "queen_lost_after_reply": true,
  "mate_in": 4,
  "material_delta_after_2ply": -9,
  "wdl_after_reply": [0.92, 0.06, 0.02]
}
```

This gives the model a vocabulary for the difference between a blunder and a sacrifice.

---

# Part VI: Runtime Scoring Design

## 22. Candidate Scoring Formula

Use a soft formula rather than hard rules.

```text
score(m) =
    AV(m)
  + α * log(P(m))
  - β * QueenRiskNoComp(m)
  - γ * TacticalRefutation(m)
  - δ * LossSpikeAfterReply(m)
  + η * StyleBonus(m)
```

Where:

```text
AV(m):
  action-value estimate or child-value estimate

P(m):
  policy prior

QueenRiskNoComp(m):
  queen-loss risk multiplied by lack of compensation

TacticalRefutation(m):
  probability opponent has a forcing refutation

LossSpikeAfterReply(m):
  predicted increase in loss probability after opponent best reply

StyleBonus(m):
  personality/style term, e.g. Tal sacrifice bonus
```

Define:

```text
QueenRiskNoComp(m) = queen_lost_after_reply_prob(m)
                     * max(0, material_loss(m) - compensation_estimate(m))
```

Sound queen sacrifices should have:

```text
high compensation_estimate
or high AV
or high WDL after reply
or mate/forcing-line flag
```

so their penalty collapses toward zero.

---

## 23. Inference Pseudocode

```python
def choose_move(position, model, config):
    outputs = model(position)
    policy = mask_legal(outputs.policy, position.legal_moves)
    candidates = top_k(policy, k=config.top_k)

    scored = []
    for move in candidates:
        tactical = compute_tactical_features(position, move)
        av = model.action_value(position, move, tactical)

        queen_risk = estimate_queen_risk(position, move, tactical)
        refutation = outputs.refutation_head.get(move, 0.0)
        loss_spike = estimate_loss_spike(outputs, move)
        style = style_bonus(position, move, config.persona)

        score = (
            av
            + config.alpha_policy * log(policy[move])
            - config.beta_queen * queen_risk
            - config.gamma_refutation * refutation
            - config.delta_loss_spike * loss_spike
            + config.eta_style * style
        )
        scored.append((move, score, tactical))

    best_move, best_score, best_tactical = max(scored, key=lambda x: x[1])

    if should_verify(best_move, outputs, best_tactical, config):
        return run_small_puct_or_child_rerank(position, model, candidates, config)

    return best_move
```

---

## 24. Style Interaction

For personalities like Tal or Morphy, adjust the compensation threshold, not the core safety logic.

```text
Tal mode:
  higher style bonus for sacrifices, checks, king attacks, complexity
  slightly higher risk tolerance
  still punish queen loss with no compensation

Morphy mode:
  bonus for development, open lines, initiative, sound sacrifices
  moderate risk tolerance
  punish speculative unsound queen loss

Safe/Capablanca mode:
  lower risk tolerance
  favor clean conversion and low loss probability
```

Do not disable queen safety for aggressive personas.

---

# Part VII: Evaluation and Release Gates

## 25. Core Metrics

Track these separately from Elo.

```text
queen_blunder_rate:
  fraction of positions where selected move loses queen without compensation

sound_sac_preservation:
  fraction of known sound queen sacrifices ranked top-k or selected

queen_tactic_accuracy:
  fraction of queen-winning tactics solved

regret_vs_teacher:
  teacher_value(best_move) - teacher_value(selected_move)

loss_spike_rate:
  fraction of selected moves causing large predicted/teacher loss spike

tactical_fallback_rate:
  fraction of moves triggering verification

fallback_precision:
  when fallback triggers, was it actually tactically useful?

fallback_recall:
  of true queen blunders, how many were caught by fallback?

latency_overhead:
  average and p95 extra time from verifier/fallback
```

---

## 26. Release Gates

A candidate model should pass:

```text
1. No known plumbing regressions.
2. Export parity within tolerance.
3. Queen blunder rate below threshold on test suite.
4. Sound queen sacrifice preservation above threshold.
5. No major Elo loss at fixed search/inference mode.
6. Latency overhead acceptable.
```

Example thresholds:

```text
queen_blunder_rate on curated suite: < 1%
sound_sac_preservation: > 80% top-3, > 50% selected depending style
policy top-k agreement vs previous model: no catastrophic drop
average latency overhead: < 10–20% in default mode
```

Use stricter gates as the engine matures.

---

# Part VIII: Implementation Roadmap

## Phase 0: Reproduction and Audit

Deliverables:

```text
failure capture schema
queen-safety regression suite
move-map round-trip tests
WDL perspective tests
search backup tests
export parity tests
```

Exit criteria:

```text
every queen-suicide example is reproducible
no known mapping/perspective/search bugs remain
```

---

## Phase 1: Soft Runtime Verifier

Deliverables:

```text
attack map / SEE-based queen-risk features
suspicious-move trigger
top-k child-value rerank or small PUCT fallback
runtime scoring formula
```

Exit criteria:

```text
queen blunders drop sharply without retraining
sound sacrifices still allowed when verified by value/search
```

---

## Phase 2: Action-Value and Regret Training

Deliverables:

```text
action-value head
candidate move labeling pipeline
regret/ranking loss
hard-negative queen blunder set
sound-sacrifice contrastive set
```

Exit criteria:

```text
action-value head ranks known queen blunders below safe alternatives
sound sacrifices do not get universally punished
```

---

## Phase 3: Multi-Ply Tactical Auxiliaries

Deliverables:

```text
opponent reply policy head
after-reply WDL/value head
queen_lost_after_reply head
PV/multi-ply auxiliary targets
```

Exit criteria:

```text
model predicts obvious queen-winning replies
model's uncertainty rises in tactically dangerous positions
```

---

## Phase 4: Specialist Tactics Distillation

Deliverables:

```text
Stockfish/lc0 tactical teacher pipeline
on-policy hard-negative mining
sacrifice taxonomy dataset
training mix with tactical oversampling
```

Exit criteria:

```text
new model improves queen-safety suite
fixed-node Elo does not regress
puzzle/tactics suite improves
```

---

## Phase 5: Adaptive Search-Light Engine

Deliverables:

```text
uncertainty-gated search policy
per-position compute budget selection
metrics for fallback precision/recall
browser latency evaluation
```

Exit criteria:

```text
stronger per millisecond than always-search and never-search baselines
queen blunder rate remains low in live play
```

---

# Appendix A: Minimal Test Positions to Include

Include positions where:

```text
queen is trapped by a simple pawn move
queen can be captured immediately
queen is apparently hanging but cannot be taken due to mate
queen sacrifice forces mate
queen sacrifice wins opponent queen
queen trade is correct
queen retreat is necessary
low-elo puzzle queen sacrifice is correct
model previously failed in self-play
```

Do not rely only on hand-picked puzzles. Include model-mined failures.

---

# Appendix B: Suggested Config

```yaml
queen_safety:
  enabled: true
  top_k_candidates: 8
  trigger_small_puct: true
  puct_nodes_on_trigger: 64

  thresholds:
    queen_loss_plies: 3
    eval_drop_cp: 300
    loss_spike_wdl: 0.20
    uncertainty_trigger: 0.55

  scoring:
    alpha_policy: 0.05
    beta_queen_risk: 0.75
    gamma_refutation: 0.50
    delta_loss_spike: 0.50
    eta_style: 0.10

training:
  auxiliary_heads:
    action_value: true
    queen_lost_after_reply: true
    after_reply_value: true
    reply_policy: true
    sacrifice_soundness: true
    uncertainty: true

  loss_weights:
    policy: 1.0
    wdl: 1.0
    value_bucket: 0.5
    action_value: 0.5
    reply_policy: 0.25
    after_reply_value: 0.25
    queen_lost_after_reply: 0.25
    sacrifice_soundness: 0.10
    uncertainty: 0.10

  batch_mix:
    normal_policy_value: 0.70
    model_mined_blunders: 0.10
    sound_sacrifices: 0.10
    puzzles_tactics: 0.05
    tablebase_endgames: 0.05
```

---

# Appendix C: Common Anti-Patterns

Avoid:

```text
hard-banning queen sacrifices
training only on queen blunders and no sound sacrifices
trusting policy-only top move in tactical positions
using scalar value only when WDL loss spike is available
mixing White-perspective and side-to-move labels
letting quantized/browser backend drift without parity tests
promoting checkpoints without tactical regression tests
```

Prefer:

```text
soft risk penalties
contrastive sound-sac vs blunder examples
action-value reranking
opponent-reply prediction
on-policy hard-negative mining
uncertainty-gated small search
explicit regression suite
```

---

# Final Recommendation

Treat queen suicides as a tactical verification and training-data problem, not as a rules problem.

The strongest design is:

```text
policy/WDL base model
+ action-value top-k reranker
+ queen-risk and compensation heads
+ opponent-reply / after-reply value heads
+ uncertainty-gated small PUCT
+ DeepSeek-style tactical specialist distillation
+ on-policy hard-negative mining
```

This preserves real queen sacrifices while removing the catastrophic cases where the model simply misses an obvious refutation.
