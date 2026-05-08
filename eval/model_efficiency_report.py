#!/usr/bin/env python3
"""Summarize Tiny Leela model strength/size efficiency from arena JSON.

The report intentionally treats strength as protocol-relative.  Elo values from
small arenas are noisy; use this to identify Pareto candidates and Elo gained
per resource doubling, then confirm with larger fixed-time/fixed-visit runs.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def bundle_bytes(path: str | Path) -> int:
    p = Path(path)
    total = p.stat().st_size if p.exists() else 0
    sidecar = Path(str(p) + ".data")
    if sidecar.exists():
        total += sidecar.stat().st_size
    return total


def count_onnx_params(path: str | Path) -> int | None:
    try:
        import onnx  # type: ignore
    except Exception:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        model = onnx.load(str(p), load_external_data=False)
    except Exception:
        return None
    total = 0
    for init in model.graph.initializer:
        n = 1
        for d in init.dims:
            n *= int(d)
        total += n
    return total


def safe_log2_ratio(a: float, b: float) -> float | None:
    if a <= 0 or b <= 0 or a == b:
        return None
    return math.log2(a / b)


def player_lookup(protocol: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {p.get("name", ""): p for p in protocol.get("players", []) if p.get("name")}


def collect_rows(arenas: list[Path], baseline_name: str | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for arena_path in arenas:
        arena = json.loads(arena_path.read_text())
        protocol = arena.get("protocol", {})
        players = player_lookup(protocol)
        standings = arena.get("standings", [])
        by_name = {s.get("name"): s for s in standings}
        baseline = None
        if baseline_name and baseline_name in by_name:
            baseline = by_name[baseline_name]
        elif standings:
            # Use the smallest model in this arena as the resource baseline if no explicit baseline is provided.
            smallest = None
            for s in standings:
                p = players.get(s.get("name"), {})
                onnx = p.get("onnx", "")
                b = bundle_bytes(onnx) if onnx else 0
                if b > 0 and (smallest is None or b < smallest[0]):
                    smallest = (b, s)
            baseline = smallest[1] if smallest else standings[-1]
        base_player = players.get(baseline.get("name"), {}) if baseline else {}
        base_bytes = bundle_bytes(base_player.get("onnx", "")) if base_player.get("onnx") else None
        base_params = count_onnx_params(base_player.get("onnx", "")) if base_player.get("onnx") else None
        base_elo = float(baseline.get("eloVsPool", 0.0)) if baseline else 0.0
        for s in standings:
            name = s.get("name")
            p = players.get(name, {})
            onnx = p.get("onnx", "")
            params = count_onnx_params(onnx) if onnx else None
            b = bundle_bytes(onnx) if onnx else None
            elo = float(s.get("eloVsPool", 0.0))
            d_elo = elo - base_elo
            bytes_log = safe_log2_ratio(float(b or 0), float(base_bytes or 0))
            params_log = safe_log2_ratio(float(params or 0), float(base_params or 0))
            rows.append({
                "arena": str(arena_path),
                "name": name,
                "mode": p.get("mode", ""),
                "onnx": onnx,
                "games": int(s.get("games", 0)),
                "wdl": f"{s.get('wins',0)}-{s.get('draws',0)}-{s.get('losses',0)}",
                "scoreRate": float(s.get("scoreRate", 0.0)),
                "eloVsPool": elo,
                "params": params,
                "bundleBytes": b,
                "fp32ParamBytes": params * 4 if params is not None else None,
                "baseline": baseline.get("name") if baseline else None,
                "deltaEloVsBaseline": d_elo,
                "eloPerByteDoubling": (d_elo / bytes_log) if bytes_log else None,
                "eloPerParamDoubling": (d_elo / params_log) if params_log else None,
            })
    return rows


def pareto_frontier(rows: list[dict[str, Any]]) -> set[tuple[str, str]]:
    # Per arena: maximize Elo, minimize bytes.  A row is dominated if another row
    # in the same arena is >= Elo and <= bytes, with at least one strict better.
    frontier: set[tuple[str, str]] = set()
    for arena in sorted({r["arena"] for r in rows}):
        rs = [r for r in rows if r["arena"] == arena and r.get("bundleBytes")]
        for r in rs:
            dominated = False
            for q in rs:
                if q is r:
                    continue
                if (q["eloVsPool"] >= r["eloVsPool"] and q["bundleBytes"] <= r["bundleBytes"] and
                    (q["eloVsPool"] > r["eloVsPool"] or q["bundleBytes"] < r["bundleBytes"])):
                    dominated = True
                    break
            if not dominated:
                frontier.add((arena, r["name"]))
    return frontier


def fmt_int(x: Any) -> str:
    return "" if x is None else f"{int(x):,}"


def fmt_mb(x: Any) -> str:
    return "" if x is None else f"{float(x) / 1024 / 1024:.2f}"


def fmt_float(x: Any, digits: int = 1) -> str:
    return "" if x is None else f"{float(x):.{digits}f}"


def markdown(rows: list[dict[str, Any]]) -> str:
    frontier = pareto_frontier(rows)
    lines = [
        "# Model efficiency report",
        "",
        "Strength is protocol-relative.  `Elo/doubling` means Elo gained per 2x increase versus the arena baseline; blank means same resource size or unavailable.",
        "",
        "| Arena | Model | Mode | WDL | Games | Score | Elo pool | Params | Bundle MiB | ΔElo | Elo/byte doubling | Elo/param doubling | Pareto |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| `{Path(r['arena']).name}` | {r['name']} | {r['mode']} | {r['wdl']} | {r['games']} | "
            f"{r['scoreRate']:.3f} | {r['eloVsPool']:.1f} | {fmt_int(r['params'])} | {fmt_mb(r['bundleBytes'])} | "
            f"{fmt_float(r['deltaEloVsBaseline'], 1)} | {fmt_float(r['eloPerByteDoubling'], 1)} | "
            f"{fmt_float(r['eloPerParamDoubling'], 1)} | {'yes' if (r['arena'], r['name']) in frontier else ''} |"
        )
    lines += [
        "",
        "## Recommended interpretation",
        "",
        "- Prefer Pareto models for follow-up: a non-frontier model is weaker and larger than another candidate in the same arena.",
        "- Treat small-game Elo and Elo/doubling as triage only; confirm with larger fixed-time and fixed-visit arenas.",
        "- Compare policy-only, classic PUCT, and AV-PUCT separately; they are different protocols.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("arenas", nargs="+", type=Path, help="Arena JSON files")
    ap.add_argument("--baseline", default="", help="Optional player name to use as resource/Elo baseline inside each arena")
    ap.add_argument("--out", type=Path, default=None, help="Write markdown report")
    ap.add_argument("--json-out", type=Path, default=None, help="Write machine-readable rows")
    args = ap.parse_args()
    rows = collect_rows(args.arenas, args.baseline or None)
    text = markdown(rows)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(rows, indent=2))
    print(text)


if __name__ == "__main__":
    main()
