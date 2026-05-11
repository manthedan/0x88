# Knowledge graph overlay plan

Date: 2026-05-09

## Recommendation

Use a Git-backed, Obsidian-compatible Markdown vault as the canonical knowledge graph layer for Tiny Leela.

```text
Markdown vault = source of truth
YAML frontmatter = typed graph metadata
wikilinks = human + agent navigation
Git = history, deprecation, review
JSON graph index = generated agent/query layer
Neo4j/SQLite = optional generated view later
```

Do **not** start with a graph database. We still need prose, citations, decisions, deprecation history, and small agent-readable context packs. A database can be generated later from the Markdown vault if graph analytics become necessary.

## Goals

1. Separate current canonical knowledge from old brainstorming and stale roadmaps.
2. Preserve old research/docs as source material without letting agents treat them as active plans.
3. Make active context small enough for agents to read first.
4. Provide typed metadata for querying, validation, and future graph exports.
5. Keep the system portable: Markdown + Git + small Python tooling.

## Initial vault layout

```text
knowledge/
  00_inbox_raw/
    old_md_imports/
    web_clips/
  01_sources/
    papers/
    repos/
    docs/
    blog_posts/
  02_concepts/
    architecture/
    training/
    search/
    deployment/
    evaluation/
  03_findings/
    architecture/
    training/
    deployment/
    evaluation/
    search/
    systems/
  04_designs/
    models/
    training_system/
    deployment/
    diagnostics/
    systems/
  05_experiments/
    planned/
    running/
    completed/
  06_decisions/
  07_roadmaps/
    active/
    deprecated/
  08_tasks/
  09_agent_context/
  99_archive/
    deprecated_raw/
  graph/                 # generated JSON index
```

`docs/` remains as the legacy/reference documentation tree for now. The knowledge vault is a canonical overlay. Old docs should be linked as `source_notes` before they are moved or heavily rewritten.

## Node ontology

Keep the ontology small:

| Type | Purpose |
|---|---|
| `concept` | Stable terms, mechanisms, components |
| `finding` | Durable conclusions or hypotheses with confidence/evidence |
| `design` | Proposed or active systems/architectures |
| `decision` | ADR-style commitments |
| `experiment` | Planned/running/completed empirical runs |
| `roadmap` | Active or deprecated plans |
| `source` | Papers, repos, external docs, source summaries |
| `risk` | Recurring failure modes |
| `task` | Backlog/milestone items |
| `ops_context` | Operational guardrails and live constraints |
| `agent_context` | Curated entrypoint files for agents |

Valid statuses:

```text
active | evergreen | draft | planned | running | completed | deprecated | archive
```

## Frontmatter convention

Every canonical note should have:

```yaml
---
id: finding.action_value_reranking_strength_per_node
type: finding
title: Finding - Action-value reranking is central to strength per node
status: active
created: 2026-05-09
updated: 2026-05-09
confidence: medium
evidence_level: experiment_supported
priority: high
supports:
  - "[[Design - SquareFormer-AV-PUCT]]"
depends_on:
  - "[[Concept - Action-value head]]"
risks:
  - "[[Risk - Move-map mismatch]]"
source_notes:
  - "[[docs/search_aux_head_calibration]]"
agent_summary: >
  Action-value and related aux heads are high-leverage because they can improve
  move choice and PUCT quality without full-width expensive search.
---
```

Relationship fields are explicit because wikilinks alone are ambiguous for agents:

```text
supports
supported_by
depends_on
tested_by
implemented_by
supersedes
superseded_by
risks
derived_from
source_notes
contradicts
related
```

## Agent-first context packs

Agents should start in `knowledge/09_agent_context/`, not by crawling all docs.

Initial files:

```text
project_brief.md
ops_constraints.md
current_architecture.md
active_roadmap.md
open_questions.md
glossary.md
retrieval_manifest.yaml
```

The retrieval manifest defines preferred reading order and canonical topic entrypoints. `ops_constraints.md` is intentionally separate so live constraints like paused QAT/Tactical MoveFormer, BT4 cache gates, and generated-artifact rules stay visible.

## Deprecation policy

Do not delete old roadmaps. Mark them.

```yaml
---
type: roadmap
status: deprecated
deprecated_on: 2026-05-09
deprecated_reason: "Superseded by current Tiny Leela portfolio roadmap."
superseded_by:
  - "[[Roadmap - Current Tiny Leela portfolio]]"
keep_for:
  - historical_context
  - discarded_options
  - old_experiment_ids
do_not_use_for:
  - current_planning
---
```

And add a top banner:

```markdown
> [!warning] Deprecated
> This roadmap is historical. Do not use it for current planning.
> Current roadmap: [[Roadmap - Current Tiny Leela portfolio]]
```

This is the main safety mechanism for agent use: old docs can remain discoverable without becoming active instructions.

## Migration plan

### Pass 1: create overlay skeleton

Create `knowledge/`, schema docs, agent context files, initial ADR, and seed canonical notes. This pass is implemented by:

```bash
./scripts/knowledge_graph_overlay.py all
```

### Pass 2: inventory existing docs

Classify existing Markdown files as:

```text
raw_research
roadmap
design
decision
diagnostic
source_summary
experiment_notes
ops_context
```

Do not move files yet. Add source links from canonical notes first.

### Pass 3: extract durable nodes

Extract atomic concepts, findings, risks, designs, decisions, and experiments from long docs. Good initial extraction targets:

```text
docs/chessformer_ideas.md
docs/chessformer_handoff.md
docs/deepseek_searchlight_architectures.md
docs/queen_safety_tactical_verification_design.md
docs/distributed_selfplay_training_system_design.md
docs/protein_carbs_puct_tuning.md
docs/unified_squareformer_architecture_roadmap.md
docs/self_play_scaling_roadmap.md
```

### Pass 4: deprecate old roadmaps

Move or copy old roadmap knowledge into canonical notes, then mark the old roadmap as deprecated or archive-only. The active roadmap should be the only source used for current planning.

### Pass 5: validate in CI/test

The validator should eventually fail on:

```text
invalid type/status
active note missing agent_summary
finding missing confidence/evidence_level
active design missing risks
deprecated roadmap missing superseded_by
wikilink target missing, in strict mode
```

## Initial seed nodes

The first overlay contains canonical notes for:

```text
Concept - SquareFormer
Concept - PUCT
Concept - Action-value head
Concept - Search-improved self-play
Finding - Action-value reranking is central to strength per node
Finding - Chess-specific geometry is high ROI for tiny models
Finding - Self-play needs search-improved targets
Design - SquareFormer-AV-PUCT
Design - 100M h7-h8 BT4 training pipeline
Design - Agent-friendly knowledge graph
Roadmap - Current Tiny Leela portfolio
Risk - Move-map mismatch
Risk - Deprecated roadmap retrieved as active context
ADR-0001 Use Markdown knowledge graph overlay
```

This is intentionally small. The next iteration should extract another 20-30 canonical notes from the highest-value architecture/search/self-play docs.

## Tooling

`scripts/knowledge_graph_overlay.py` provides:

```bash
# Create vault skeleton and seed notes without overwriting existing files
./scripts/knowledge_graph_overlay.py init

# Generate machine-readable graph view
./scripts/knowledge_graph_overlay.py index

# Validate frontmatter and relationship invariants
./scripts/knowledge_graph_overlay.py validate

# Run all of the above
./scripts/knowledge_graph_overlay.py all
```

Generated outputs:

```text
knowledge/graph/nodes.json
knowledge/graph/edges.json
knowledge/graph/open_questions.json
knowledge/graph/missing_links.txt   # only when missing links exist
```

The generated graph is a derived view. Markdown remains canonical.

## Agent skill

A project skill lives at:

```text
.pi/skills/tiny-leela-knowledge-graph/SKILL.md
```

Use it whenever an agent is asked to create, refactor, validate, or query the knowledge overlay.

## Future enhancements

- Add `inventory-docs` mode to classify `docs/*.md` into a CSV/JSON migration queue.
- Add `new-note` mode that creates typed notes from templates.
- Add `deprecate-roadmap` mode that inserts deprecation frontmatter/banner.
- Add CI integration after the initial migration stabilizes.
- Export to SQLite or Neo4j only if generated JSON is insufficient.
