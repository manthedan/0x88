#!/usr/bin/env python3
"""Create and maintain the Tiny Leela Markdown knowledge-graph overlay.

Canonical source: Markdown files under knowledge/ with YAML-like frontmatter.
Generated view: knowledge/graph/{nodes,edges,open_questions}.json.

This script intentionally uses only the Python standard library.  The
frontmatter parser is small and conservative; it supports the scalar/list/block
fields used by the vault templates, not arbitrary YAML.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

VALID_TYPES = {
    "concept",
    "finding",
    "design",
    "decision",
    "experiment",
    "roadmap",
    "source",
    "risk",
    "task",
    "ops_context",
    "agent_context",
}
VALID_STATUSES = {"active", "evergreen", "draft", "planned", "running", "completed", "deprecated", "archive"}
REL_FIELDS = {
    "supports",
    "supported_by",
    "depends_on",
    "tested_by",
    "implemented_by",
    "supersedes",
    "superseded_by",
    "risks",
    "derived_from",
    "source_notes",
    "contradicts",
    "related",
}
WIKILINK_RE = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]+)?(?:\|[^\]]+)?\]\]")


@dataclass
class Note:
    path: Path
    relpath: str
    frontmatter: dict[str, Any]
    body: str

    @property
    def node_id(self) -> str:
        return str(self.frontmatter.get("id") or self.path.with_suffix("").name)

    @property
    def title(self) -> str:
        return str(self.frontmatter.get("title") or heading_title(self.body) or self.path.stem)

    @property
    def type(self) -> str:
        return str(self.frontmatter.get("type") or "")

    @property
    def status(self) -> str:
        return str(self.frontmatter.get("status") or "")


def strip_quotes(value: str) -> str:
    value = value.strip()
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    return value


def parse_scalar(value: str) -> Any:
    value = strip_quotes(value)
    if value == "[]":
        return []
    if value in {"true", "false"}:
        return value == "true"
    return value


def parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---\n", 4)
    if end < 0:
        return {}, text
    raw = text[4:end].splitlines()
    body = text[end + 5 :]
    data: dict[str, Any] = {}
    i = 0
    while i < len(raw):
        line = raw[i]
        i += 1
        if not line.strip() or line.lstrip().startswith("#") or line.startswith(" "):
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value in {">", "|"}:
            block: list[str] = []
            while i < len(raw):
                nxt = raw[i]
                if nxt and not nxt.startswith(" ") and ":" in nxt:
                    break
                block.append(nxt.strip())
                i += 1
            data[key] = " ".join(part for part in block if part).strip()
        elif value == "":
            items: list[str] = []
            while i < len(raw):
                nxt = raw[i]
                if nxt.startswith("  - ") or nxt.startswith("- "):
                    items.append(strip_quotes(nxt.split("- ", 1)[1].strip()))
                    i += 1
                elif nxt.startswith(" ") and not nxt.strip():
                    i += 1
                else:
                    break
            data[key] = items
        else:
            data[key] = parse_scalar(value)
    return data, body


def dump_frontmatter(data: dict[str, Any]) -> str:
    lines = ["---"]
    for key, value in data.items():
        if isinstance(value, list):
            lines.append(f"{key}:")
            for item in value:
                lines.append(f"  - {item}")
        elif isinstance(value, str) and ("\n" in value or len(value) > 90):
            lines.append(f"{key}: >")
            for part in value.splitlines() or [value]:
                lines.append(f"  {part}")
        else:
            lines.append(f"{key}: {value}")
    lines.append("---")
    return "\n".join(lines) + "\n"


def heading_title(body: str) -> str | None:
    for line in body.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return None


def normalize_link(link: str) -> str:
    return link.strip().replace("\\", "/").removesuffix(".md")


def links_from_value(value: Any) -> list[str]:
    values = value if isinstance(value, list) else [value]
    out: list[str] = []
    for item in values:
        if not isinstance(item, str):
            continue
        matches = WIKILINK_RE.findall(item)
        out.extend(normalize_link(m) for m in matches)
    return out


def wikilinks(text: str) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for m in WIKILINK_RE.findall(text):
        target = normalize_link(m)
        if target not in seen:
            seen.add(target)
            out.append(target)
    return out


def note_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*.md") if "/graph/" not in p.as_posix())


def load_notes(root: Path) -> list[Note]:
    notes: list[Note] = []
    for path in note_files(root):
        text = path.read_text(encoding="utf-8")
        fm, body = parse_frontmatter(text)
        notes.append(Note(path=path, relpath=path.relative_to(root).as_posix(), frontmatter=fm, body=body))
    return notes


def write_if_missing(path: Path, content: str) -> bool:
    if path.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return True


def note(frontmatter: dict[str, Any], body: str) -> str:
    return dump_frontmatter(frontmatter) + "\n" + body.strip() + "\n"


def seed_files(root: Path) -> dict[str, str]:
    today = date.today().isoformat()
    base = {
        "created": today,
        "updated": today,
        "project": "tiny-neural-chess",
    }
    return {
        "README.md": """# Tiny Neural Chess Knowledge Graph\n\nThis vault is the canonical Markdown knowledge-graph overlay for Tiny Leela.\n\n- Markdown files are the source of truth.\n- YAML frontmatter provides typed graph metadata.\n- Wikilinks provide human and agent navigation.\n- `knowledge/graph/*.json` is generated by `scripts/knowledge_graph_overlay.py`.\n\nStart for agents: `09_agent_context/retrieval_manifest.yaml`.\n""",
        "schema.yaml": """valid_types:\n  - concept\n  - finding\n  - design\n  - decision\n  - experiment\n  - roadmap\n  - source\n  - risk\n  - task\n  - ops_context\n  - agent_context\nvalid_statuses:\n  - active\n  - evergreen\n  - draft\n  - planned\n  - running\n  - completed\n  - deprecated\n  - archive\nrequired_common:\n  - id\n  - type\n  - title\n  - status\n  - created\n  - updated\n  - agent_summary\nrelationship_fields:\n  - supports\n  - supported_by\n  - depends_on\n  - tested_by\n  - implemented_by\n  - supersedes\n  - superseded_by\n  - risks\n  - derived_from\n  - source_notes\n  - contradicts\n  - related\n""",
        "09_agent_context/project_brief.md": note({**base, "id": "agent_context.project_brief", "type": "agent_context", "title": "Project brief", "status": "active", "priority": "high", "agent_summary": "Tiny Leela is a browser-deployable neural chess engine project balancing supervised learning, compact search, deployment constraints, and self-play research."}, """# Project brief\n\nTiny Leela builds small Leela-style chess engines that can run in constrained/browser contexts while still benefiting from modern training, evaluation, and search ideas.\n\nCurrent portfolio lanes:\n\n- CNN baselines and deployable anchors\n- Tactical MoveFormer hybrids\n- Tiny BT4 / SquareFormer models\n\nAgents should prefer current canonical context in this folder, then traverse typed notes, and only then consult old `docs/` files as source material.\n"""),
        "09_agent_context/current_architecture.md": note({**base, "id": "agent_context.current_architecture", "type": "agent_context", "title": "Current architecture", "status": "active", "depends_on": ["[[Design - SquareFormer-AV-PUCT]]"], "agent_summary": "Current preferred architecture is SquareFormer-AV-PUCT: square-token model, WDL/value heads, action-value top-k reranking, uncertainty, and conditional PUCT."}, """# Current architecture\n\nCurrent preferred direction: [[Design - SquareFormer-AV-PUCT]].\n\nCore components:\n\n- 64 square tokens\n- chess-aware geometry / relation bias\n- from-to policy head\n- WDL/value output\n- action-value top-k reranking\n- uncertainty / regret diagnostics\n- conditional compact PUCT\n- multi-teacher and on-policy distillation later\n"""),
        "09_agent_context/active_roadmap.md": note({**base, "id": "agent_context.active_roadmap", "type": "agent_context", "title": "Active roadmap entrypoint", "status": "active", "depends_on": ["[[Roadmap - Current Tiny Leela portfolio]]"], "agent_summary": "Entrypoint for active planning. Old docs and roadmaps are source material only unless represented here or in an active roadmap note."}, """# Active roadmap entrypoint\n\nPrimary active roadmap: [[Roadmap - Current Tiny Leela portfolio]].\n\nDo not treat old brainstorming files as active instructions unless they are linked from the current roadmap or an active design/decision note.\n"""),
        "09_agent_context/open_questions.md": note({**base, "id": "agent_context.open_questions", "type": "agent_context", "title": "Open questions", "status": "active", "agent_summary": "Curated active research questions for the Tiny Leela knowledge graph."}, """# Open questions\n\n- Does h7/h8 history improve strength per byte enough for BT4/SquareFormer?\n- Which aux-PUCT weights transfer across visit counts and anchor types?\n- Does action-value reranking reduce catastrophic regret without harming tactical sacrifices?\n- When does PUCT become value-useful for each architecture lane?\n- What minimal self-play correctness suite is sufficient before larger generation runs?\n"""),
        "09_agent_context/ops_constraints.md": note({**base, "id": "agent_context.ops_constraints", "type": "ops_context", "title": "Operations constraints", "status": "active", "priority": "high", "agent_summary": "Current operational guardrails: QAT and Tactical MoveFormer are paused, BT4 waits for validated h7/h8 100M caches, generated artifacts are not committed, and classic PUCT remains default for deterministic eval."}, """# Operations constraints\n\n- Do not resume Tactical MoveFormer without explicit user approval.\n- Do not start new QAT work unless explicitly revisited.\n- BT4/SquareFormer 100M training waits for validated h7/h8 SquareFormer cache manifests.\n- Classic PUCT is default for deterministic eval; Gumbel-root is experimental/self-play only.\n- Do not commit generated outputs under `data/*`, `artifacts/`, `public/models/*.onnx`, `public/models/*.json`, or `dist-client/`.\n- Use `.venv-onnx/bin/python` for repo Python tasks.\n"""),
        "09_agent_context/glossary.md": note({**base, "id": "agent_context.glossary", "type": "agent_context", "title": "Glossary", "status": "active", "agent_summary": "Short glossary mapping common Tiny Leela terms to canonical concept notes."}, """# Glossary\n\n- [[Concept - SquareFormer]]: square-token transformer family for chess boards.\n- [[Concept - PUCT]]: search policy combining priors and value estimates.\n- [[Concept - Action-value head]]: per-candidate move value/reranking head.\n- [[Concept - Search-improved self-play]]: AlphaZero/lc0-style target generation.\n"""),
        "09_agent_context/retrieval_manifest.yaml": """preferred_context_order:\n  - project_brief.md\n  - ops_constraints.md\n  - current_architecture.md\n  - active_roadmap.md\n  - open_questions.md\n  - glossary.md\ndeprecated_do_not_use_for_planning:\n  - ../07_roadmaps/deprecated/\ncanonical_topics:\n  squareformer:\n    start: ../02_concepts/architecture/concept.squareformer.md\n  action_value:\n    start: ../02_concepts/training/concept.action_value_head.md\n  search:\n    start: ../02_concepts/search/concept.puct.md\n  self_play:\n    start: ../02_concepts/training/concept.search_improved_self_play.md\n""",
        "06_decisions/ADR-0001-use-markdown-knowledge-graph.md": note({**base, "id": "decision.use_markdown_knowledge_graph", "type": "decision", "title": "ADR-0001 Use Markdown knowledge graph overlay", "status": "active", "confidence": "high", "priority": "high", "supports": ["[[Design - Agent-friendly knowledge graph]]"], "risks": ["[[Risk - Deprecated roadmap retrieved as active context]]"], "agent_summary": "Use a Git-backed Obsidian-compatible Markdown vault as canonical knowledge source; generated graph JSON is a derived view."}, """# ADR-0001 Use Markdown knowledge graph overlay\n\n## Decision\n\nUse `knowledge/` as a Git-backed, Obsidian-compatible Markdown vault with typed frontmatter and wikilinks.\n\n## Rationale\n\nThis preserves prose, citations, deprecation history, and agent-readable metadata without requiring a graph database. Generated JSON can be produced from the Markdown source when agents or scripts need structured traversal.\n\n## Consequences\n\n- Old docs remain source material until explicitly extracted.\n- Active planning should use `09_agent_context/` and active typed notes.\n- Generated graph files are derived, not the source of truth.\n"""),
        "02_concepts/architecture/concept.squareformer.md": note({**base, "id": "concept.squareformer", "type": "concept", "title": "Concept - SquareFormer", "status": "active", "topics": ["architecture", "squareformer", "transformer"], "agent_summary": "SquareFormer is the square-token transformer lane for Tiny Leela, focused on chess-specific geometry and compact browser-deployable inference."}, """# Concept - SquareFormer\n\nSquareFormer represents the chess board as square-centered tokens and uses chess-aware structure to get more strength per byte than generic transformer scaling.\n\nRelated design: [[Design - SquareFormer-AV-PUCT]].\n"""),
        "02_concepts/search/concept.puct.md": note({**base, "id": "concept.puct", "type": "concept", "title": "Concept - PUCT", "status": "active", "topics": ["search", "puct", "evaluation"], "agent_summary": "PUCT combines neural priors and value estimates for search; deterministic eval defaults to classic PUCT while aux/Gumbel variants remain experimental."}, """# Concept - PUCT\n\nPUCT is the search policy used to balance policy prior, exploration, and value.\n\nTiny Leela currently treats classic PUCT as the deterministic eval default. Aux-PUCT and Gumbel-root variants require explicit calibration and separate evidence.\n"""),
        "02_concepts/training/concept.action_value_head.md": note({**base, "id": "concept.action_value_head", "type": "concept", "title": "Concept - Action-value head", "status": "active", "topics": ["training", "search", "reranking"], "agent_summary": "The action-value head predicts move-conditioned value for top candidates and supports reranking, regret diagnostics, and aux-PUCT experiments."}, """# Concept - Action-value head\n\nAn action-value head estimates the value of candidate moves, not only the value of the current position. It is central to reranking high-policy moves and detecting catastrophic regrets.\n\nSupported finding: [[Finding - Action-value reranking is central to strength per node]].\n"""),
        "02_concepts/training/concept.search_improved_self_play.md": note({**base, "id": "concept.search_improved_self_play", "type": "concept", "title": "Concept - Search-improved self-play", "status": "active", "topics": ["self-play", "training", "search"], "agent_summary": "Self-play should train on post-search visit/WDL/Q targets, not raw sampled winners or static imitation alone."}, """# Concept - Search-improved self-play\n\nSearch-improved self-play uses model-guided search to generate better training targets. For Tiny Leela, policy targets should come from post-search distributions, with WDL/Q targets and resign calibration.\n\nFinding: [[Finding - Self-play needs search-improved targets]].\n"""),
        "03_findings/search/finding.action_value_reranking_strength_per_node.md": note({**base, "id": "finding.action_value_reranking_strength_per_node", "type": "finding", "title": "Finding - Action-value reranking is central to strength per node", "status": "active", "confidence": "medium", "evidence_level": "experiment_supported", "priority": "high", "supports": ["[[Design - SquareFormer-AV-PUCT]]"], "depends_on": ["[[Concept - Action-value head]]", "[[Concept - PUCT]]"], "risks": ["[[Risk - Move-map mismatch]]"], "agent_summary": "Action-value and related aux heads are high-leverage because they can improve move choice and PUCT quality without full-width expensive search."}, """# Finding - Action-value reranking is central to strength per node\n\nTiny models often have plausible policy but weak consequence modeling. Candidate action-values can rerank high-policy moves, expose regret, and tune aux-PUCT behavior.\n\nUse this finding when designing SquareFormer, MoveFormer, or CNN-AV experiments.\n"""),
        "03_findings/architecture/finding.chess_specific_geometry_high_roi.md": note({**base, "id": "finding.chess_specific_geometry_high_roi", "type": "finding", "title": "Finding - Chess-specific geometry is high ROI for tiny models", "status": "active", "confidence": "medium", "evidence_level": "paper_supported", "priority": "high", "supports": ["[[Design - SquareFormer-AV-PUCT]]"], "depends_on": ["[[Concept - SquareFormer]]"], "agent_summary": "For tiny chess transformers, chess-specific relation bias and square geometry should be tested before generic transformer tricks."}, """# Finding - Chess-specific geometry is high ROI for tiny models\n\nChess-specific square relations, geometry, and legality structure are likely more valuable for Tiny Leela than generic transformer embellishments at the same parameter budget.\n"""),
        "03_findings/training/finding.self_play_needs_search_improved_targets.md": note({**base, "id": "finding.self_play_needs_search_improved_targets", "type": "finding", "title": "Finding - Self-play needs search-improved targets", "status": "active", "confidence": "high", "evidence_level": "paper_supported", "priority": "high", "depends_on": ["[[Concept - Search-improved self-play]]"], "agent_summary": "Self-play improvement requires search-improved visit/value targets; do not train policy on one-hot sampled Gumbel winners."}, """# Finding - Self-play needs search-improved targets\n\nFor AlphaZero/lc0-style improvement, train on post-search distributions and value/WDL/Q targets. Sampled moves are game actions, not automatically good supervised policy labels.\n"""),
        "04_designs/models/design.squareformer_av_puct.md": note({**base, "id": "design.squareformer_av_puct", "type": "design", "title": "Design - SquareFormer-AV-PUCT", "status": "active", "confidence": "medium", "priority": "high", "depends_on": ["[[Concept - SquareFormer]]", "[[Concept - Action-value head]]", "[[Concept - PUCT]]"], "supported_by": ["[[Finding - Action-value reranking is central to strength per node]]", "[[Finding - Chess-specific geometry is high ROI for tiny models]]"], "risks": ["[[Risk - Move-map mismatch]]"], "agent_summary": "Current preferred lightweight architecture: SquareFormer policy/WDL/AV outputs with action-value reranking and calibrated conditional PUCT."}, """# Design - SquareFormer-AV-PUCT\n\nSquareFormer-AV-PUCT combines square-token transformer inference with policy, WDL/value, action-value reranking, uncertainty/regret diagnostics, and compact PUCT.\n\n## Current status\n\nActive design direction. BT4/SquareFormer 100M work remains gated on validated h7/h8 cache manifests.\n"""),
        "04_designs/training_system/design.h7_h8_bt4_training_pipeline.md": note({**base, "id": "design.h7_h8_bt4_training_pipeline", "type": "design", "title": "Design - 100M h7-h8 BT4 training pipeline", "status": "active", "confidence": "high", "priority": "high", "depends_on": ["[[Concept - SquareFormer]]"], "risks": ["[[Risk - Move-map mismatch]]"], "agent_summary": "BT4/SquareFormer 100M training uses true h8 supervised data, then h7/h8 SquareFormer caches, then local training after manifests validate."}, """# Design - 100M h7-h8 BT4 training pipeline\n\nCanonical pipeline:\n\n```text\nraw elite/TCEC rows -> supervised_100m_elite_tcec_h8_v1 -> cache_squareformer_h7 + cache_squareformer_h8 -> BT4 training\n```\n\nDo not substitute old h2 caches for true h7/h8 BT4 training.\n"""),
        "04_designs/systems/design.agent_friendly_knowledge_graph.md": note({**base, "id": "design.agent_friendly_knowledge_graph", "type": "design", "title": "Design - Agent-friendly knowledge graph", "status": "active", "confidence": "high", "priority": "high", "supported_by": ["[[ADR-0001 Use Markdown knowledge graph overlay]]"], "risks": ["[[Risk - Deprecated roadmap retrieved as active context]]"], "agent_summary": "The knowledge graph overlay separates canonical active notes from raw historical docs so agents can plan from current context without losing research history."}, """# Design - Agent-friendly knowledge graph\n\nUse canonical typed notes for active knowledge and keep old docs as linked source material. Agents start at `09_agent_context/retrieval_manifest.yaml`.\n"""),
        "07_roadmaps/active/roadmap.current_tiny_leela_portfolio.md": note({**base, "id": "roadmap.current_tiny_leela_portfolio", "type": "roadmap", "title": "Roadmap - Current Tiny Leela portfolio", "status": "active", "priority": "high", "depends_on": ["[[Design - SquareFormer-AV-PUCT]]", "[[Design - 100M h7-h8 BT4 training pipeline]]"], "risks": ["[[Risk - Deprecated roadmap retrieved as active context]]"], "agent_summary": "Current roadmap keeps CNN, Tactical MoveFormer, and Tiny BT4/SquareFormer lanes separate, with 100M MF80/BT4/CNN96 as the near-term portfolio."}, """# Roadmap - Current Tiny Leela portfolio\n\nNear-term portfolio:\n\n- MF80 100M sidecar/training\n- BT4/SquareFormer after h7/h8 100M caches\n- CNN96 100M evaluation and PUCT tuning\n\nResearch foundation:\n\n- Maintain parity tests and UCI/OpenBench readiness\n- Calibrate aux-PUCT and visit curves\n- Build self-play correctness before larger generation\n"""),
        "03_findings/evaluation/risk.move_map_mismatch.md": note({**base, "id": "risk.move_map_mismatch", "type": "risk", "title": "Risk - Move-map mismatch", "status": "active", "priority": "high", "agent_summary": "Policy/action-value encoding mismatches silently corrupt training and evaluation; parity tests and shared helpers are mandatory."}, """# Risk - Move-map mismatch\n\nMove encoding mismatches across Python, TypeScript, Rust, ONNX, and cached datasets can silently invalidate experiments.\n\nMitigation: shared encoding helpers and parity tests.\n"""),
        "03_findings/systems/risk.deprecated_roadmap_active_context.md": note({**base, "id": "risk.deprecated_roadmap_active_context", "type": "risk", "title": "Risk - Deprecated roadmap retrieved as active context", "status": "active", "priority": "high", "agent_summary": "Agents may treat old brainstorming or roadmaps as current instructions unless deprecation metadata and curated entrypoints are enforced."}, """# Risk - Deprecated roadmap retrieved as active context\n\nOld roadmaps are valuable history but dangerous active context. Mark deprecated material explicitly and prefer `09_agent_context/` for planning.\n"""),
    }


def init_overlay(root: Path) -> int:
    dirs = [
        "00_inbox_raw/old_md_imports",
        "00_inbox_raw/web_clips",
        "01_sources/papers",
        "01_sources/repos",
        "01_sources/docs",
        "01_sources/blog_posts",
        "02_concepts/architecture",
        "02_concepts/training",
        "02_concepts/search",
        "02_concepts/deployment",
        "02_concepts/evaluation",
        "03_findings/architecture",
        "03_findings/training",
        "03_findings/deployment",
        "03_findings/evaluation",
        "03_findings/search",
        "03_findings/systems",
        "04_designs/models",
        "04_designs/training_system",
        "04_designs/deployment",
        "04_designs/diagnostics",
        "04_designs/systems",
        "05_experiments/planned",
        "05_experiments/running",
        "05_experiments/completed",
        "06_decisions",
        "07_roadmaps/active",
        "07_roadmaps/deprecated",
        "08_tasks",
        "09_agent_context",
        "99_archive/deprecated_raw",
        "graph",
    ]
    for d in dirs:
        (root / d).mkdir(parents=True, exist_ok=True)
    write_if_missing(root / "graph/.gitkeep", "")
    written = 0
    for rel, content in seed_files(root).items():
        if write_if_missing(root / rel, content):
            written += 1
    print(f"initialized {root} (created_or_kept_dirs={len(dirs)} wrote_files={written})")
    return 0


def build_index(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    notes = load_notes(root)
    aliases: dict[str, str] = {}
    for n in notes:
        aliases[normalize_link(n.title)] = n.node_id
        aliases[normalize_link(n.path.stem)] = n.node_id
        aliases[normalize_link(n.path.with_suffix("").name)] = n.node_id
        aliases[normalize_link(n.relpath.removesuffix(".md"))] = n.node_id

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    missing: set[str] = set()
    open_questions: list[dict[str, Any]] = []

    for n in notes:
        links = wikilinks(n.body) + [l for field in REL_FIELDS for l in links_from_value(n.frontmatter.get(field, []))]
        dedup_links = []
        for link in links:
            if link not in dedup_links:
                dedup_links.append(link)
        nodes.append(
            {
                "id": n.node_id,
                "title": n.title,
                "type": n.type,
                "status": n.status,
                "path": n.relpath,
                "agent_summary": n.frontmatter.get("agent_summary", ""),
                "frontmatter": n.frontmatter,
                "links": dedup_links,
            }
        )
        for link in wikilinks(n.body):
            target = aliases.get(link)
            if target:
                edges.append({"source": n.node_id, "target": target, "type": "wikilink", "field": "body"})
            else:
                missing.add(f"{n.relpath} -> [[{link}]]")
        for field in REL_FIELDS:
            for link in links_from_value(n.frontmatter.get(field, [])):
                target = aliases.get(link)
                if target:
                    edges.append({"source": n.node_id, "target": target, "type": field, "field": field})
                else:
                    missing.add(f"{n.relpath} {field} -> [[{link}]]")
        if "open_questions" in n.relpath or n.type == "task":
            for line in n.body.splitlines():
                stripped = line.strip()
                if stripped.startswith("- ") and "?" in stripped:
                    open_questions.append({"source": n.node_id, "path": n.relpath, "question": stripped[2:].strip()})
    return nodes, edges, open_questions, sorted(missing)


def write_index(root: Path) -> int:
    graph = root / "graph"
    graph.mkdir(parents=True, exist_ok=True)
    nodes, edges, open_questions, missing = build_index(root)
    (graph / "nodes.json").write_text(json.dumps(nodes, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (graph / "edges.json").write_text(json.dumps(edges, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (graph / "open_questions.json").write_text(json.dumps(open_questions, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if missing:
        (graph / "missing_links.txt").write_text("\n".join(missing) + "\n", encoding="utf-8")
    elif (graph / "missing_links.txt").exists():
        (graph / "missing_links.txt").unlink()
    print(f"indexed nodes={len(nodes)} edges={len(edges)} open_questions={len(open_questions)} missing_links={len(missing)}")
    return 0


def validate(root: Path, strict_links: bool = False) -> int:
    notes = load_notes(root)
    errors: list[str] = []
    warnings: list[str] = []
    for n in notes:
        is_context_yaml = n.relpath.endswith(".yaml")
        if n.relpath == "README.md":
            continue
        fm = n.frontmatter
        if not fm:
            warnings.append(f"{n.relpath}: missing frontmatter")
            continue
        if n.type not in VALID_TYPES:
            errors.append(f"{n.relpath}: invalid type {n.type!r}")
        if n.status not in VALID_STATUSES:
            errors.append(f"{n.relpath}: invalid status {n.status!r}")
        for key in ["id", "title", "created", "updated", "agent_summary"]:
            if key not in fm and n.status not in {"archive"}:
                errors.append(f"{n.relpath}: missing {key}")
        if n.type == "finding":
            if "confidence" not in fm:
                errors.append(f"{n.relpath}: finding missing confidence")
            if "evidence_level" not in fm:
                errors.append(f"{n.relpath}: finding missing evidence_level")
        if n.type == "design" and n.status == "active" and not fm.get("risks"):
            errors.append(f"{n.relpath}: active design missing risks")
        if n.type == "roadmap" and n.status == "deprecated" and not fm.get("superseded_by"):
            errors.append(f"{n.relpath}: deprecated roadmap missing superseded_by")
    _, _, _, missing = build_index(root)
    for m in missing:
        (errors if strict_links else warnings).append(f"missing link: {m}")
    for w in warnings:
        print(f"WARN {w}", file=sys.stderr)
    for e in errors:
        print(f"ERROR {e}", file=sys.stderr)
    print(f"validated notes={len(notes)} warnings={len(warnings)} errors={len(errors)}")
    return 1 if errors else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="knowledge", help="knowledge vault root")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init", help="create vault directories and seed canonical notes")
    sub.add_parser("index", help="generate graph JSON from Markdown frontmatter and wikilinks")
    v = sub.add_parser("validate", help="validate frontmatter and links")
    v.add_argument("--strict-links", action="store_true", help="treat missing wikilink targets as errors")
    all_cmd = sub.add_parser("all", help="run init, index, and validate")
    all_cmd.add_argument("--strict-links", action="store_true")
    args = parser.parse_args()
    root = Path(args.root)
    if args.cmd == "init":
        return init_overlay(root)
    if args.cmd == "index":
        return write_index(root)
    if args.cmd == "validate":
        return validate(root, strict_links=args.strict_links)
    if args.cmd == "all":
        rc = init_overlay(root)
        rc = write_index(root) or rc
        rc = validate(root, strict_links=args.strict_links) or rc
        return rc
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
