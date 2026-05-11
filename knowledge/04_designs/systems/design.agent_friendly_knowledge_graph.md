---
created: 2026-05-09
updated: 2026-05-09
project: tiny-neural-chess
id: design.agent_friendly_knowledge_graph
type: design
title: Design - Agent-friendly knowledge graph
status: active
confidence: high
priority: high
supported_by:
  - [[ADR-0001 Use Markdown knowledge graph overlay]]
risks:
  - [[Risk - Deprecated roadmap retrieved as active context]]
agent_summary: >
  The knowledge graph overlay separates canonical active notes from raw historical docs so agents can plan from current context without losing research history.
---

# Design - Agent-friendly knowledge graph

Use canonical typed notes for active knowledge and keep old docs as linked source material. Agents start at `09_agent_context/retrieval_manifest.yaml`.
