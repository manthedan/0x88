---
created: 2026-05-10
updated: 2026-05-10
project: tiny-neural-chess
id: design.agentic_engine_maintenance
type: design
title: Design - Agentic engine maintenance
status: active
confidence: high
priority: high
depends_on:
  - [[Design - SquareFormer-AV-PUCT]]
  - [[Concept - Search-improved self-play]]
supported_by:
  - [[Finding - Self-play needs search-improved targets]]
risks:
  - [[Risk - Move-map mismatch]]
related:
  - [[Design - Agent-friendly knowledge graph]]
agent_summary: >
  Treat coding-agent/heuristic learning as infrastructure around the neural chess engine: capture failures, write invariant tests, route teachers, mine hard positions, tune search schedules, and periodically distill those discoveries back into model training.
---

# Design - Agentic engine maintenance

Tiny Leela should not replace neural training with a hand-written chess policy. Chess is too deep and adversarial for a growing pile of rules to compete with modern engines.

The useful lesson from heuristic-learning systems is different: in verifiable domains, agent-maintained software can rapidly turn reproducible failures into durable structure. For Tiny Leela, that means maintaining the system around the model:

- diagnostics
- tactical failure capture
- search schedules
- hard-position mining
- teacher routing
- cloud/self-play health checks
- quantization and backend drift guards
- regression suites
- deployment/promotion gates

The neural model remains the compact chess intuition. Search performs local policy improvement. Agentic maintenance makes failures observable, testable, labelable, and eventually trainable.

## Core loop

1. Engine plays, searches, evaluates, or self-plays.
2. A failure or anomaly is detected:
   - queen blunder
   - illegal move or move-map mismatch
   - value perspective/sign bug
   - policy/search disagreement
   - bad endgame conversion
   - browser vs Node evaluator mismatch
   - quantization drift
   - corrupt self-play shard/provenance
3. Capture a structured failure packet:
   - FEN
   - selected move
   - legal moves
   - policy distribution
   - WDL/value
   - action-values/regrets when available
   - search stats/tree summary
   - teacher best move and eval delta when available
   - model id, backend, seed, and repro command
4. The agent proposes one constrained response:
   - code fix
   - invariant/metamorphic test
   - teacher-labeling job
   - hard-position mining rule
   - replay-buffer sampling rule
   - search-budget/uncertainty trigger
   - promotion/deployment guard
5. Run fixed tests and gauntlets.
6. Keep useful changes as regression memory.
7. Periodically distill repeated failures into labels, auxiliary heads, action-value/regret targets, or curated self-play/teacher datasets.

## Design rules

- Do not hard-code chess bans such as “never sacrifice queen.” Use failures to mine and label hard negatives instead.
- Prefer invariants, metamorphic tests, backend parity, teacher deltas, and promotion gates over brittle exact-move goldens.
- Use heuristics mainly for observability, routing, budget allocation, and data selection.
- When a heuristic repeatedly matters, ask whether it should become a label, auxiliary target, mined dataset, or gate.
- Do not let agents freely mutate core engine/search logic without rules tests, encoding parity, search invariants, backend parity, and arena/promotion checks.

## First implementation targets

- `failure_packet` schema and replay command contract.
- Queen-blunder / catastrophic-regret failure memory.
- Move-map, promotion, castling, en-passant, and value-perspective regression cases.
- Browser/Node/native evaluator parity packets.
- Quantized-vs-FP drift packets.
- Cloud self-play shard schema/provenance/seed/duplication guards.
- Teacher routing for tactical, endgame, backend, and self-play failures.

## Relationship to model training

This lane supports [[Design - SquareFormer-AV-PUCT]] and [[Concept - Search-improved self-play]]. It should feed the learner rather than replace it:

```text
failure -> packet -> teacher/search label -> hard-negative/replay mix -> model update -> promotion gate
```

The healthiest outcome is that runtime heuristics shrink over time because the model, tests, and data pipeline absorb their lessons.
