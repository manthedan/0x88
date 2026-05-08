# Auxiliary Head Ablation Roadmap

## Goal

Measure which auxiliary heads/training streams actually improve tiny Leela-style models, separately from architecture changes and runtime/search protocol.

Primary question:

```text
Do AV/rank/regret/risk-style auxiliary streams improve policy-only play, classic PUCT, AV-PUCT, calibration, or tactical robustness enough to justify their bytes/latency/training complexity?
```

## Current pivot

MoveFormer is parked for now. Its K64/no-board sidecar build may finish, but GPU training is intentionally held while we prioritize ChannelFormer/CNN head experiments.

Park marker:

```text
artifacts/moveformer_10m_supervised_mf64x6_e3/PARK_MOVEFORMER
```

Remove that file to let the MoveFormer GPU pipeline continue later.

## Data we already have

### Supervised 10M policy/WDL stream

```text
data/datasets/supervised_10m_elite_tcec_v1/
```

Provides:

```text
board planes/cache
played/teacher policy target
WDL target
sample weights
```

### ChessBench C=48 AV stream

```text
data/public_teacher_overlays/chessbench_full_policy_value_direct_top48_32shards_v1/collection_manifest.json
```

Provides:

```text
compact board tokens
candidate moves
candidate action values
candidate regrets
candidate masks
```

This supports AV, rank, and regret losses immediately.

## Heads / streams to test now

### Always-on baseline heads

1. **Policy**
   - Target: supervised move/policy label.
   - Runtime: yes.

2. **WDL/value**
   - Target: WDL/result label.
   - Runtime: yes, search leaf value.

### Auxiliary candidate heads/losses available now

3. **AV regression**
   - Target: per-candidate teacher value.
   - Loss: SmoothL1/MSE after tanh.
   - Runtime: optional, AV-PUCT.

4. **Rank/listwise**
   - Target: best candidate by teacher value.
   - Loss: cross-entropy over candidate scores.
   - Runtime: usually no separate output; shapes trunk/candidate scorer.

5. **Regret**
   - Target: best-value minus candidate-value regret.
   - Loss: SmoothL1 on candidate gaps.
   - Runtime: usually no separate output; shapes calibration/order.

### Later streams requiring more labels/tooling

6. **Risk/uncertainty**
   - Needs: self-play search stats, deeper-search instability, or Stockfish spot labels.

7. **Blunder / Stockfish-delta**
   - Needs: annotated model mistakes / teacher deltas.

8. **Queen safety / tactical motif probes**
   - Needs: curated tactical labels or Stockfish annotations.

## Architectures to compare

Start with matched 64x6 families:

```text
cnn64x6 baseline/hybrid
cnn64x6 AV variants
cnn64x6 channelformer c32 d128 l2
cnn64x6 channelformer AV C48 variants
```

Keep model size, training rows, epochs, LR, batch size, and seed fixed within each ablation family.

## Experiment phases

### Phase 0: Immediate sanity check

Compare finished supervised ChannelFormer against existing 10M CNN baselines:

```text
artifacts/channelformer_10m_supervised_64x6_c32_d128_l2_e3/model.onnx
artifacts/arena_10m_guarded/48x5_e9.onnx
artifacts/arena_10m_guarded/64x6_e12_ema.onnx
artifacts/arena_10m_guarded/80x5_hybrid_e12_ema.onnx
```

Protocols:

```text
policy-only or visits=1
classic PUCT visits=64/128
later: fixed-time
```

### Phase 1: 1M ablation screen

Purpose: cheap signal, not final Elo.

Run each architecture with matched settings:

```text
base:          policy + WDL
+AV:           policy + WDL + AV regression
+rank:         policy + WDL + rank
+regret:       policy + WDL + regret
+AV+rank
+AV+regret
+rank+regret
+kitchen_sink: policy + WDL + AV + rank + regret
```

Metrics:

```text
dev policy CE/top1/top4/top8
dev WDL CE
AV MSE/top1 when applicable
quick policy-only arena
quick PUCT arena
latency/bytes/params
```

Promotion rule:

```text
Promote variants that improve arena strength or tactical robustness without unacceptable latency/bytes.
Do not promote based only on auxiliary loss improvements.
```

### Phase 2: 10M promoted runs

Run only the most promising variants:

```text
base
kitchen_sink
best single auxiliary
best pair
leave-one-out around kitchen_sink if kitchen_sink wins
```

Leave-one-out examples:

```text
all
all - AV
all - rank
all - regret
```

### Phase 3: Search-specific evaluation

For every promoted 10M model, evaluate separately:

```text
policy-only
classic PUCT
AV-PUCT with per-model/per-budget calibration
fixed visits
fixed time
```

Important: AV-PUCT is not comparable until calibrated per model/protocol/visit budget.

### Phase 4: Diagnostics

Use targeted diagnostics to explain wins/losses:

```text
queen safety suite
Stockfish delta / override analysis
opening/ECO balance
illegal-move hardening
calibration curves for WDL and AV
```

## Kitchen sink interpretation

A kitchen-sink run answers:

```text
Does the whole auxiliary package help?
```

It does **not** answer which head helped. For causality, run matched ablations:

```text
single-head additions
pair additions
leave-one-out from kitchen sink
```

## Self-play future

When self-play lands, avoid logging only `(position, move, result)`. Save root search stats so auxiliary streams become cheap:

```text
position
legal moves
played move
visit distribution
root WDL/value
per-edge P/N/Q/AV/in-flight/risk/uncertainty if available
final game result
```

Then:

```text
policy target       <- visit distribution
value target        <- result and/or search root value
AV target           <- child Q / backed-up WDL
rank/regret target  <- child Q ordering/gaps
uncertainty target  <- instability/entropy/disagreement
```

External Stockfish/deeper-search annotation should be reserved for selected hard positions, not every self-play row.

## Immediate todo

1. Finish/monitor supervised ChannelFormer vs CNN arena.
2. Let current AV ChannelFormer C48 run proceed.
3. Compare supervised vs AV ChannelFormer.
4. If ChannelFormer is competitive, build 1M head ablation scripts.
5. Promote strongest 1M ablations to 10M.
6. Refresh model manifest and efficiency report after each completed promoted model.
7. Keep MoveFormer parked until the head-ablation branch of work has clear results.
