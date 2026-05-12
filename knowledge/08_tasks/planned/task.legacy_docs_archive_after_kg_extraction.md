---
id: task.legacy_docs_archive_after_kg_extraction
type: task
title: Task - Legacy docs archive after KG extraction
status: planned
created: 2026-05-12
updated: 2026-05-12
priority: medium
depends_on:
  - "[[Design - Agent-friendly knowledge graph]]"
  - "[[Roadmap - Current Tiny Leela portfolio]]"
  - "[[Decision - LC0 architecture funnel and deployability frontier]]"
risks:
  - "[[Risk - Deprecated roadmap retrieved as active context]]"
agent_summary: >
  Start a conservative legacy-doc archiving process: keep knowledge/ as the current source of truth, classify old docs as source material, extract any still-current ideas into canonical KG notes, then move only clearly superseded docs with stubs or updated links.
---

# Task - Legacy docs archive after KG extraction

## Goal

Reduce planning confusion from old roadmap/brainstorm docs while preserving provenance. The `knowledge/` vault is the current canonical planning layer; `docs/` files remain source material unless their decisions are represented in active KG notes.

## Non-goals

- Do not delete useful research notes.
- Do not move paths that scripts/tests depend on without updating references.
- Do not turn archive cleanup into a blocker for LC0 adapter/proof work.

## Process

1. Inventory `docs/*.md` and classify each as one of:

```text
canonical/stable reference
active but not yet KG-extracted
source material extracted into KG
historical/superseded
script/tooling reference path, do not move
```

2. For every old doc marked historical/superseded, verify that any current decision is represented in one of:

```text
knowledge/07_roadmaps/active/roadmap.current_tiny_leela_portfolio.md
knowledge/04_designs/**
knowledge/06_decisions/**
knowledge/09_agent_context/**
```

3. Add a visible warning/stub before moving any old doc, or leave a stub at the old path after moving.

4. Prefer `docs/archive/YYYY-MM/` or `docs/research-notes/` for moved material, but only after links are checked.

5. Run KG index/validate and a basic repository search for broken path references.

## First-pass candidates to classify

Likely source/historical architecture-roadmap docs now covered by the KG and the LC0 funnel decision:

```text
docs/transformer_model_roadmap.md
docs/unified_squareformer_architecture_roadmap.md
docs/small_bt4_progression.md
docs/bt4_10m_ablation_plan.md
docs/bt4_architecture_ablation_queue.md
docs/bt4_gab_lite_ablation_plan.md
docs/scaling-and-architecture-todo.md
```

Likely stable references to keep in place:

```text
docs/model_efficiency_metrics.md
docs/model_manifest.md
docs/board_normalization_standard.md
docs/move_encoding.md
docs/moveformer_data_schema.md
docs/elo-evaluation-process.md
docs/onnx_deploy_workflow.md
docs/mac_mini_cpu_offload_plan.md
```

## Exit criteria

- Active KG validates with zero errors.
- `docs/README.md` clearly says KG is current source of truth.
- Every archived/moved doc has either a stub or updated incoming references.
- No current roadmap or agent context points at a deprecated doc as an execution source.
