---
created: 2026-05-09
updated: 2026-05-09
project: tiny-neural-chess
id: decision.use_markdown_knowledge_graph
type: decision
title: ADR-0001 Use Markdown knowledge graph overlay
status: active
confidence: high
priority: high
supports:
  - [[Design - Agent-friendly knowledge graph]]
risks:
  - [[Risk - Deprecated roadmap retrieved as active context]]
agent_summary: >
  Use a Git-backed Obsidian-compatible Markdown vault as canonical knowledge source; generated graph JSON is a derived view.
---

# ADR-0001 Use Markdown knowledge graph overlay

## Decision

Use `knowledge/` as a Git-backed, Obsidian-compatible Markdown vault with typed frontmatter and wikilinks.

## Rationale

This preserves prose, citations, deprecation history, and agent-readable metadata without requiring a graph database. Generated JSON can be produced from the Markdown source when agents or scripts need structured traversal.

## Consequences

- Old docs remain source material until explicitly extracted.
- Active planning should use `09_agent_context/` and active typed notes.
- Generated graph files are derived, not the source of truth.
