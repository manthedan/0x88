---
created: 2026-05-12
updated: 2026-05-12
project: tiny-neural-chess
id: risk.lc0_data_provenance_and_license
type: risk
title: Risk - LC0 data provenance and license
status: active
priority: high
agent_summary: >
  LC0 public training chunks appear technically ideal for Tiny Leela distillation, but every derived dataset/model must preserve provenance, license metadata, and deployability constraints before product use.
---

# Risk - LC0 data provenance and license

Public LC0 training chunks are attractive because they contain search-improved policy/value targets, but they are external training data with their own licensing and provenance requirements.

Mitigations:

- Store source URL, run/test directory, chunk filename, byte size, checksum, format version, and license text in every download manifest.
- Do not mix LC0-derived records into existing supervised/self-play caches without a `teacher=lc0_public` or equivalent provenance field.
- Keep LC0-distilled model cards explicit about teacher data and license assumptions.
- Before treating an LC0-distilled model as a final product artifact, review ODbL/DbCL obligations and Netlify/deployment implications.
- Keep LC0, ChessBench, Stockfish, human-game, and Tiny-Leela self-play sources separable in manifests and training metrics.
