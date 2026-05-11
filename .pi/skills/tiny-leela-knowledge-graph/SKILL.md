---
name: tiny-leela-knowledge-graph
description: Create, validate, query, and refactor the Tiny Leela Markdown knowledge graph overlay under knowledge/. Use when asked to organize docs as a knowledge graph, create canonical notes, deprecate roadmaps, generate graph indexes, or prepare agent context packs.
---

# Tiny Leela Knowledge Graph

Use this skill for the Git-backed Markdown knowledge graph overlay.

## Canonical model

```text
knowledge/ Markdown vault = source of truth
YAML frontmatter = typed graph metadata
wikilinks = human + agent navigation
knowledge/graph/*.json = generated derived index
old docs/ files = source material unless extracted into canonical notes
```

Do not treat legacy roadmap prose as current planning unless it is represented in an active `knowledge/` roadmap/design/decision note or `09_agent_context/`.

## First files to read

For planning or onboarding, read in this order:

```text
knowledge/09_agent_context/retrieval_manifest.yaml
knowledge/09_agent_context/project_brief.md
knowledge/09_agent_context/ops_constraints.md
knowledge/09_agent_context/current_architecture.md
knowledge/09_agent_context/active_roadmap.md
knowledge/09_agent_context/open_questions.md
knowledge/README.md
knowledge/schema.yaml
```

For the migration plan, read:

```text
docs/knowledge_graph_overlay_plan.md
```

## Tooling

Use the repo-local tool:

```bash
./scripts/knowledge_graph_overlay.py init      # create skeleton + seed notes, no overwrite
./scripts/knowledge_graph_overlay.py index     # write knowledge/graph/*.json
./scripts/knowledge_graph_overlay.py validate  # validate note metadata and links
./scripts/knowledge_graph_overlay.py all       # init + index + validate
```

Use `.venv-onnx/bin/python` when invoking Python directly:

```bash
.venv-onnx/bin/python scripts/knowledge_graph_overlay.py validate
```

## Note ontology

Valid note types:

```text
concept finding design decision experiment roadmap source risk task ops_context agent_context
```

Valid statuses:

```text
active evergreen draft planned running completed deprecated archive
```

Important relationship fields:

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

## Required frontmatter pattern

Every canonical active note should include:

```yaml
---
id: finding.example_slug
type: finding
title: Finding - Example title
status: active
created: 2026-05-09
updated: 2026-05-09
confidence: medium        # findings only
evidence_level: hypothesis # findings only
priority: high
agent_summary: >
  One to three sentences explaining what an agent should know and how to use this note.
---
```

Active designs should link to risks:

```yaml
risks:
  - "[[Risk - Move-map mismatch]]"
```

Deprecated roadmaps must link to replacements:

```yaml
status: deprecated
superseded_by:
  - "[[Roadmap - Current Tiny Leela portfolio]]"
```

## Workflow: create/refactor knowledge

1. Read the agent context files and current plan.
2. Identify whether the requested content is a concept, finding, design, decision, experiment, roadmap, source, risk, or task.
3. Prefer small canonical notes, roughly 500-1500 words.
4. Link to old `docs/` files via `source_notes` instead of moving old docs immediately.
5. If touching roadmaps, mark old plans deprecated rather than deleting them.
6. Run:
   ```bash
   ./scripts/knowledge_graph_overlay.py index
   ./scripts/knowledge_graph_overlay.py validate
   ```
7. Summarize changed paths and any validation warnings/errors.

## Workflow: deprecate an old roadmap

- Add/ensure frontmatter:
  ```yaml
  type: roadmap
  status: deprecated
  deprecated_on: 2026-05-09
  superseded_by:
    - "[[Roadmap - Current Tiny Leela portfolio]]"
  agent_summary: >
    Historical roadmap retained for context; do not use for current planning.
  ```
- Add a visible warning banner at the top of the body.
- Ensure current ideas worth keeping are extracted into active findings/designs/decisions.
- Validate the graph.

## Safety rules

- Markdown notes are canonical; generated JSON is derived.
- Do not let old docs become active instructions just because they are linked.
- Do not commit generated data/artifact/model outputs while working on the vault.
- Keep `09_agent_context/` short, curated, and current.
