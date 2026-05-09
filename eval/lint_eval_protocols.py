#!/usr/bin/env python3
"""Read-only linter for Tiny Leela evaluation result/protocol artifacts.

The goal is not to prove game strength; it is to catch protocol-card drift before
we quote Elo/search results. It intentionally uses only the Python standard
library so it can run outside the project ML virtualenvs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

ARENA_KINDS = {
    "search_mode_arena",
    "search_mode_arena_merged",
    "onnx_round_robin_arena",
    "uci_anchor_arena",
    "merged_uci_anchor_arena",
    "parallel_onnx_round_robin_arena_merge",
    "parallel_onnx_round_robin_arena_merge_v512",
}

SKIP_SUFFIXES = (
    ".meta.json",
    ".protocol.json",
)
SKIP_NAMES = {
    "cache_manifest.json",
    "manifest.json",
    "collection_manifest.json",
    "dataset_report.json",
    "model_manifest.current.json",
    "model_manifest_overrides.json",
}


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(errors="replace"))
    except Exception:
        return None


def add(findings: list[dict[str, Any]], severity: str, path: Path, message: str, *, field: str | None = None) -> None:
    findings.append({"severity": severity, "path": rel(path), "message": message, **({"field": field} if field else {})})


def has_key(data: dict[str, Any], key: str) -> bool:
    return key in data and data.get(key) is not None


def positive_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and value > 0


def game_count(data: dict[str, Any]) -> int | None:
    for key in ("totalGames", "games", "pairs"):
        value = data.get(key)
        if isinstance(value, int):
            return value
    return None


def lint_protocol(path: Path, data: dict[str, Any], findings: list[dict[str, Any]], *, embedded: bool = False) -> None:
    kind = data.get("kind")
    if not isinstance(kind, str):
        severity = "warning" if embedded else "error"
        add(findings, severity, path, "protocol is missing string kind", field="kind")
        return
    if not has_key(data, "createdUtc") and not has_key(data, "mergedUtc"):
        add(findings, "warning", path, "protocol missing createdUtc/mergedUtc timestamp")

    if kind in {"search_mode_arena", "search_mode_arena_merged"}:
        for key in ("players", "visits", "cpuct", "batchSize", "maxPlies"):
            if not has_key(data, key):
                add(findings, "warning", path, f"search-mode protocol missing {key}", field=key)
        if not data.get("players"):
            add(findings, "error", path, "search-mode protocol has no players", field="players")
        if not positive_number(data.get("totalGames")):
            add(findings, "warning", path, "search-mode protocol missing positive totalGames", field="totalGames")
        for i, player in enumerate(data.get("players") or []):
            if isinstance(player, dict):
                for key in ("name", "onnx", "meta", "mode"):
                    if not has_key(player, key):
                        add(findings, "warning", path, f"player[{i}] missing {key}", field=f"players[{i}].{key}")
                for key in ("sha256", "backend"):
                    if not has_key(player, key):
                        add(findings, "info", path, f"player[{i}] does not record {key}", field=f"players[{i}].{key}")

    elif kind == "uci_anchor_arena":
        for key in ("candidate", "anchors", "visits", "cpuct", "pairs", "maxPlies"):
            if not has_key(data, key):
                add(findings, "warning", path, f"UCI anchor protocol missing {key}", field=key)
        if not has_key(data, "batchSize"):
            add(findings, "warning", path, "UCI anchor protocol missing batchSize", field="batchSize")
        candidate = data.get("candidate")
        if isinstance(candidate, dict):
            for key in ("onnx", "meta", "mode"):
                if not has_key(candidate, key):
                    add(findings, "warning", path, f"candidate missing {key}", field=f"candidate.{key}")
            for key in ("sha256", "backend"):
                if not has_key(candidate, key):
                    add(findings, "info", path, f"candidate does not record {key}", field=f"candidate.{key}")
        if not positive_number(data.get("pairs")):
            add(findings, "warning", path, "UCI anchor protocol has non-positive pairs", field="pairs")

    elif kind == "onnx_round_robin_arena":
        for key in ("models", "visits", "cpuct", "gamesPerPair", "maxPlies"):
            if not has_key(data, key):
                add(findings, "warning", path, f"round-robin protocol missing {key}", field=key)
        for key in ("batchSize", "backend", "sha256"):
            if not has_key(data, key):
                add(findings, "info", path, f"round-robin protocol does not record {key}", field=key)

    elif kind in ARENA_KINDS:
        # Known merged/meta protocol. Keep generic checks only.
        pass
    else:
        add(findings, "info", path, f"unrecognized protocol kind {kind!r}", field="kind")



def lint_result(path: Path, data: dict[str, Any], findings: list[dict[str, Any]]) -> None:
    embedded_protocol = data.get("protocol") if isinstance(data.get("protocol"), dict) else None
    sidecar = Path(str(path) + ".protocol.json")
    looks_like_arena = any(k in data for k in ("standings", "summaries", "games", "pairs"))
    if looks_like_arena and not embedded_protocol and not sidecar.exists():
        add(findings, "warning", path, "arena-like result lacks embedded protocol and .protocol.json sidecar")

    if embedded_protocol:
        pseudo_path = Path(str(path) + "#protocol")
        lint_protocol(pseudo_path, embedded_protocol, findings, embedded=True)

    # Flag zero-game or zero-pair results with Elo/CI-like fields; these are often placeholders.
    containers: list[tuple[str, Any]] = [("root", data)]
    for key in ("summaries", "standings", "pairs"):
        value = data.get(key)
        if isinstance(value, list):
            containers.extend((f"{key}[{i}]", item) for i, item in enumerate(value) if isinstance(item, dict))
    for where, item in containers:
        games = item.get("games")
        pairs = item.get("pairs")
        has_elo = any(k in item for k in ("elo", "eloDiff", "eloCi95", "eloVsPool", "elo_ci95"))
        if has_elo and (games == 0 or pairs == 0):
            add(findings, "error", path, f"{where} has Elo/CI fields with zero games/pairs")
        if isinstance(games, int) and games > 0:
            has_wdl = all(k in item for k in ("wins", "draws", "losses")) or "wdl" in item
            if not has_wdl and where != "root":
                add(findings, "warning", path, f"{where} has games but no W/D/L fields")


def iter_json_files(root: Path) -> list[Path]:
    files = []
    for p in root.rglob("*.json"):
        name = p.name
        if name in SKIP_NAMES or name.startswith("eval_protocol_lint."):
            continue
        if any(str(p).endswith(suffix) for suffix in SKIP_SUFFIXES):
            continue
        files.append(p)
    return sorted(files)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="artifacts", help="artifact tree to scan")
    ap.add_argument("--out", default=None, help="optional JSON findings output")
    ap.add_argument("--strict", action="store_true", help="exit non-zero on errors")
    ap.add_argument("--fail-on-warning", action="store_true", help="with --strict, also fail on warnings")
    ap.add_argument("--max-print", type=int, default=80, help="max findings to print")
    args = ap.parse_args()

    root = (ROOT / args.root).resolve() if not Path(args.root).is_absolute() else Path(args.root)
    findings: list[dict[str, Any]] = []
    protocol_count = 0
    result_count = 0

    for p in sorted(root.rglob("*.protocol.json")):
        protocol_count += 1
        data = read_json(p)
        if isinstance(data, dict):
            lint_protocol(p, data, findings)
        else:
            add(findings, "error", p, "invalid protocol JSON")

    for p in iter_json_files(root):
        result_count += 1
        data = read_json(p)
        if isinstance(data, dict):
            lint_result(p, data, findings)
        elif data is None:
            add(findings, "error", p, "invalid result JSON")

    counts = {"error": 0, "warning": 0, "info": 0}
    for f in findings:
        counts[f["severity"]] = counts.get(f["severity"], 0) + 1

    summary = {
        "root": rel(root),
        "protocol_files": protocol_count,
        "result_files": result_count,
        "findings": counts,
        "finding_count": len(findings),
    }
    print(json.dumps(summary, indent=2))
    for f in findings[: max(0, args.max_print)]:
        field = f" [{f['field']}]" if f.get("field") else ""
        print(f"{f['severity'].upper():7} {f['path']}{field}: {f['message']}")
    if len(findings) > args.max_print:
        print(f"... {len(findings) - args.max_print} additional findings omitted from stdout")

    if args.out:
        out = (ROOT / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"summary": summary, "findings": findings}, indent=2) + "\n")
        print(f"wrote {rel(out)}")

    if args.strict and (counts.get("error", 0) or (args.fail_on_warning and counts.get("warning", 0))):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
