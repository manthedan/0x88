---
created: 2026-05-09
updated: 2026-05-09
project: tiny-neural-chess
id: risk.move_map_mismatch
type: risk
title: Risk - Move-map mismatch
status: active
priority: high
agent_summary: >
  Policy/action-value encoding mismatches silently corrupt training and evaluation; parity tests and shared helpers are mandatory.
---

# Risk - Move-map mismatch

Move encoding mismatches across Python, TypeScript, Rust, ONNX, and cached datasets can silently invalidate experiments.

Mitigation: shared encoding helpers and parity tests.
