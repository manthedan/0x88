# lc0 / Maia Gap Closure Roadmap

This is the execution roadmap for closing the gap between our tiny Leela stack and mature lc0/Maia-style engines. It builds on `docs/maia_lc0_infra_gap_analysis.md` and current ChessFormer v1 100M e3 findings.

## Current Evidence Snapshot

ChessFormer/SquareFormer v1 100M e3 changed the search story:

```text
supervised dev policy CE: 1.924837
policy top1/top4/top8: 41.6% / 75.4% / 88.6%
queen fixed-suite risk: 57.1%  (better than 10M CNNs, still high)
```

PUCT visits are not monotonic-proof yet, but the first sweep suggests value/search becomes useful only at higher visit counts:

```text
visits=1:   weak anchor play
visits=32:  mixed / sometimes worse
visits=128: improves Stockfish results, still weak vs Maia 1100
visits=192/256: better Maia signal in small samples
visits=512: strong early anchor signal; currently under confirmation
```

Key interpretation:

```text
The value head is not useless, but low-visit PUCT under-explores and/or is too noisy.
512 visits may cross the threshold where tactical/value signal starts dominating policy noise.
```

Do not overclaim until the 512 full anchor run and visit-curve confirmation finish.

---

## Roadmap Principles

1. **Separate model quality from engine quality**
   - policy-only strength
   - value calibration
   - PUCT/search strength
   - browser latency/playability

2. **Treat Maia/lc0 as a system target, not just a net-size target**
   - representation parity
   - action map hardening
   - search calibration
   - tactical verification
   - UCI/runtime robustness

3. **Use cheap calibration probes before expensive arenas**
   - PUCT consistency
   - root-prior parity
   - Stockfish move-delta annotation
   - fixed queen suite

4. **Make Elo claims protocol-relative**
   - anchors, visits/nodes, openings, games, WDL, CI, illegal count, caveats

---

# Phase 0 — Baseline Lockdown

Goal: make the current system reproducible enough that future improvements are meaningful.

## Deliverables

- [x] Permanent PUCT core tests
- [x] SquareFormer browser evaluator
- [x] UHO-lite reversed-pair openings
- [x] Queen fixed-suite gate
- [x] Anchor arena supports SquareFormer
- [x] Round-robin arena supports SquareFormer
- [ ] Bucket eval fully SquareFormer-compatible and documented
- [ ] One command to run a release-gate bundle for any model
- [ ] Store protocol cards beside every benchmark JSON

## Release-gate bundle

For each candidate model:

```bash
npm run build:client
node --experimental-strip-types eval/puct_core_tests.mjs
node --test tests/puct_core.test.mjs
node --experimental-strip-types eval/puct_consistency_check.mjs --visits 1,32,128,512 ...
node --experimental-strip-types eval/queen_plumbing_diagnostic.mjs --positions-json ...
node --experimental-strip-types eval/onnx_bucket_eval_jsonl.mjs ...
node --experimental-strip-types eval/uci_anchor_arena.mjs --visits 1 ...
node --experimental-strip-types eval/uci_anchor_arena.mjs --visits 512 ...
```

---

# Phase 1 — Search Calibration and PUCT Curve

Goal: determine whether current value heads are genuinely search-useful, and at what visit budget.

## Visit Curve

Run the same openings/anchors at:

```text
1, 32, 64, 128, 192, 256, 384, 512, optionally 768/1024
```

Recommended cheap protocol:

```text
pairs=3 or 5
anchors=Stockfish1320, Stockfish1600, Maia1100
max plies=100
```

Promotion protocol:

```text
If a visit setting wins the cheap curve, rerun with pairs=10-20 and full anchors.
```

## Direct Calibration Probe

For positions where PUCT changes the policy top move:

```text
fen
policy_top_move
puct_move at N visits
policy_prior(policy_top)
policy_prior(puct_move)
root_Q(policy_top)
root_Q(puct_move)
visit_count(policy_top)
visit_count(puct_move)
Stockfish eval after policy_top
Stockfish eval after puct_move
```

Metrics:

```text
puct_override_rate
puct_override_stockfish_win_rate
mean_cp_delta_puct_minus_policy
queen-risk delta
material-drop delta
```

This answers the important question:

```text
When PUCT overrides policy, is it usually right?
```

## PUCT Engine Improvements to Test

- [ ] Tune `cpuct` by visit budget
- [ ] Add FPU / first-play urgency
- [ ] Add optional policy softmax temperature at root prior level
- [ ] Add min-prior floor only after legal normalization experiments
- [ ] Add transposition/cache reuse within one search
- [ ] Batch leaf evaluations for browser/Node speed
- [ ] Better terminal/draw/repetition handling

Suggested ablation matrix:

```text
baseline PUCT
baseline + FPU
baseline + tuned cpuct
baseline + FPU + tuned cpuct
```

Gate each by policy-vs-PUCT Stockfish delta and anchor mini-sweep, not by intuition.

---

# Phase 2 — Tactical Verification Bridge

Goal: reduce one-ply/two-ply tactical failures without hard-coding illegal queen-sac rules.

## Queen/Material Risk Labels

Create labels from existing diagnostics and Stockfish annotation:

```text
candidate move
opponent best reply / observed reply
queen captured next ply?
material swing after reply
Stockfish cp drop after selected move
risk bucket: safe / dubious / blunder / catastrophic
```

Primary targets:

```text
risk_logit(action)
action_value(action)
opponent_reply_policy(action -> reply)
```

## Diagnostic-First Guardrail

Before using risk in play, measure:

```text
risk AUC on fixed queen suite
risk calibration by policy probability bucket
false positive queen sacrifices
false negative queen hangs
```

Only after that, test soft integration:

```text
adjusted_logit = policy_logit - lambda * risk_logit
or
PUCT prior *= exp(-lambda * risk)
```

Never start with a hard ban.

---

# Phase 3 — SquareFormer v2 Action Heads

Goal: make search cheaper and more reliable by predicting per-action value/risk directly.

## Architecture Sketch

```text
square tokens -> transformer blocks -> square embeddings + global embedding

for each legal action a=(from,to,promo):
  h_from = square_embedding[from]
  h_to   = square_embedding[to]
  h_g    = global_embedding
  h_a    = move_type/promotion embedding
  z_a    = MLP([h_g, h_from, h_to, h_a])

heads:
  policy_logit[a]
  action_q[a]
  blunder_risk[a]
  optional reply_distribution[a, reply]
```

## Training Losses

Start simple:

```text
L = policy CE
  + w_value * WDL CE
  + w_q * action-value loss
  + w_risk * risk BCE/focal loss
```

Later:

```text
+ w_reply * opponent-reply CE for high-risk/tactical samples
+ consistency loss: root value ~= selected action_q after policy aggregation
```

## Data Sources

- supervised 100M labels
- Stockfish shallow tactical labels for candidate moves
- queen-risk suite positions
- arena self-play blunder positions
- future search-improved labels from current best engine

---

# Phase 4 — lc0/Maia Runtime Parity

Goal: stop losing strength through infrastructure.

## UCI / Anchor Hardening

- [ ] Fix and isolate Maia/lc0 illegal anomalies
- [ ] Log full UCI transcript on illegal/timeout
- [ ] Add per-engine startup readiness tests
- [ ] Add deterministic engine shutdown
- [ ] Validate fixed nodes/visits semantics per anchor

## Runtime Parity

- [ ] Node vs browser evaluator parity on fixed FEN pack
- [ ] CNN vs SquareFormer metadata validation
- [ ] ONNX external-data packaging tests
- [ ] Browser latency benchmark per model/visit budget
- [ ] Search budget wall-clock normalization separate from fixed-visits tests

## Engine Features

- [ ] Search tree reuse between moves
- [ ] Batched inference
- [ ] Time management
- [ ] Draw/repetition handling
- [ ] Resignation disabled for eval, optional for play

---

# Phase 5 — Search-Improved Training Loop

Goal: move from pure imitation toward search-aware chess improvement.

Minimal loop:

```text
1. sample positions from games / self-play / failure suites
2. run current best model with high visits
3. optionally annotate top actions with Stockfish/lc0
4. train on improved policy/action-value/risk labels
5. gate by dev CE, queen suite, visit curve, anchors
```

Promotion criteria:

```text
supervised CE not catastrophically worse
queen fixed-suite improves or stable
policy-only anchor stable
PUCT anchor improves
visit curve does not become more brittle
browser speed acceptable
```

---

# Immediate Next Tasks

1. Finish the 512 full anchor run.
2. Finish the 192/256/384 visit curve.
3. Add a PUCT override annotation script:
   ```text
   eval/puct_override_stockfish_delta.mjs/py
   ```
4. Add protocol-card writing to arena outputs.
5. Run queen diagnostics on 512 games, not just PUCT32 games.
6. Decide whether ChessFormer v1 should continue to e4-e6 based on:
   ```text
   dev CE
   bucket CE
   queen suite
   512 anchors
   visit curve shape
   ```
7. Draft SquareFormer v2 action-head trainer once CNN canonical sweep completes.

---

# Success Criteria

Near-term success:

```text
512 PUCT confirmed stronger than 1/32/128 on full-ish anchors
queen risk below current CNNs and trending down
no Maia/lc0 illegal caveats in trusted runs
```

Medium-term success:

```text
action-risk head predicts queen/material blunders with useful precision
PUCT overrides policy with positive Stockfish cp delta
SquareFormer v2 beats v1 at equal visits or equal wall-clock
```

Long-term success:

```text
tiny engine behaves less like a policy imitator and more like a small lc0-style engine:
  fewer catastrophic blunders
  search helps consistently
  value/action values are calibrated
  browser and Node behavior match
```
