#!/usr/bin/env python3
"""Build a curated model manifest for current Tiny Leela artifacts.

The checked-in override file records provenance that cannot be reliably inferred
from ONNX/meta files. This script enriches it with discovered facts: artifact
existence, ONNX parameter counts, bundle bytes, compact runtime metadata,
training METRIC lines, dataset summaries, and references to search-mode arenas.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import re
from pathlib import Path
from typing import Any

try:
    import onnx  # type: ignore
except Exception:  # pragma: no cover - optional outside .venv-onnx
    onnx = None

ROOT = Path(__file__).resolve().parents[1]
DROP_META_KEYS = {"moves"}
SUMMARY_META_KEYS = [
    "kind",
    "architecture",
    "policy_map",
    "channels",
    "blocks",
    "policy_head",
    "se",
    "history_plies",
    "input_planes",
    "trained_with_aux_av",
    "av_head_exported",
    "action_value_move_encoding",
    "input_planes",
    "move_dim",
    "heads",
    "layers",
    "ff_dim",
    "num_move_features",
    "max_legal_moves",
    "onnx_legal_buckets",
    "onnx_dynamic_batch",
    "onnx_dynamic_legal",
    "trained_with_chessbench_av_candidates",
    "trained_with_weak_av_q",
    "channelformer_cpatch",
    "channelformer_dim",
    "channelformer_heads",
    "channelformer_layers",
    "channelformer_ff_dim",
]


def rel(path: str | Path | None) -> Path | None:
    if not path:
        return None
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def exists(path: str | Path | None) -> bool:
    p = rel(path)
    return bool(p and p.exists())


def read_json(path: str | Path | None) -> Any | None:
    p = rel(path)
    if not p or not p.exists():
        return None
    return json.loads(p.read_text())


def bundle_bytes(path: str | Path | None) -> int | None:
    p = rel(path)
    if not p or not p.exists():
        return None
    total = p.stat().st_size
    sidecar = Path(str(p) + ".data")
    if sidecar.exists():
        total += sidecar.stat().st_size
    return total


def onnx_params(path: str | Path | None) -> int | None:
    p = rel(path)
    if not p or not p.exists() or onnx is None:
        return None
    model = onnx.load(str(p), load_external_data=False)
    total = 0
    for init in model.graph.initializer:
        n = 1
        for dim in init.dims:
            n *= int(dim)
        total += n
    return total


def mib(value: int | float | None) -> float | None:
    if value is None:
        return None
    return value / 1024.0 / 1024.0


def summarize_dataset(path: str | Path | None) -> dict[str, Any] | None:
    data = read_json(path)
    if not isinstance(data, dict):
        return None
    if data.get("format") == "chessbench_av_cache_collection_v1":
        results = data.get("results") or []
        return {
            "manifest": str(path),
            "format": data.get("format"),
            "top_k": data.get("top_k"),
            "max_candidates": data.get("max_candidates"),
            "history_plies": data.get("history_plies"),
            "positions": sum(int(r.get("rows", 0)) for r in results if isinstance(r, dict)),
            "candidate_rows": sum(int(r.get("candidate_rows", 0)) for r in results if isinstance(r, dict)),
            "shards": len(data.get("caches") or []),
        }
    return {
        "manifest": str(path),
        "name": data.get("name"),
        "format": data.get("format"),
        "total_train_rows": data.get("total_train_rows"),
        "total_dev_rows": data.get("total_dev_rows"),
        "train_shards": len(data.get("train_shards") or []),
        "rows_per_shard": data.get("rows_per_shard"),
        "history_plies": data.get("history_plies"),
        "skip_plies": data.get("skip_plies"),
        "source_inputs": (data.get("reproducibility") or {}).get("inputs"),
    }


def compact_meta(meta_path: str | Path | None) -> dict[str, Any] | None:
    meta = read_json(meta_path)
    if not isinstance(meta, dict):
        return None
    out = {k: meta.get(k) for k in SUMMARY_META_KEYS if k in meta}
    if "moves" in meta:
        out["move_count"] = len(meta["moves"])
    if "move_feature_names" in meta:
        out["move_feature_names"] = meta["move_feature_names"]
    if "onnx_exports" in meta:
        out["onnx_exports"] = meta["onnx_exports"]
    return out


def parse_metrics(log_path: str | Path | None) -> dict[str, Any] | None:
    p = rel(log_path)
    if not p or not p.exists():
        return None
    metrics: dict[str, float | int | str] = {}
    all_metrics: list[dict[str, Any]] = []
    progress_tail: list[str] = []
    metric_re = re.compile(r"^METRIC\s+([^=\s]+)=(.+?)\s*$")
    for line in p.read_text(errors="replace").splitlines():
        if line.startswith("progress "):
            progress_tail.append(line)
            progress_tail = progress_tail[-5:]
        m = metric_re.match(line)
        if not m:
            continue
        key, raw = m.group(1), m.group(2)
        value: float | int | str
        try:
            f = float(raw)
            value = int(f) if f.is_integer() else f
        except ValueError:
            value = raw
        metrics[key] = value
        all_metrics.append({"key": key, "value": value})
    best: dict[str, Any] = {}
    composite = [(m["key"], m["value"]) for m in all_metrics if str(m["key"]).endswith("_composite") and isinstance(m["value"], (int, float))]
    if composite:
        best_key, best_val = min(composite, key=lambda kv: float(kv[1]))
        best["best_composite_key"] = best_key
        best["best_composite"] = best_val
    return {"path": str(log_path), "metrics": metrics, "best": best, "progress_tail": progress_tail}


def artifact_paths(model: dict[str, Any]) -> list[str]:
    artifacts = model.get("artifacts") or {}
    paths: list[str] = []
    for key in ("onnx", "meta", "pt", "train_log", "status", "pipeline_log", "queue_log"):
        if isinstance(artifacts.get(key), str):
            paths.append(artifacts[key])
    if isinstance(artifacts.get("onnx_buckets"), dict):
        paths.extend(str(v) for v in artifacts["onnx_buckets"].values())
    return paths


def enrich_model(model: dict[str, Any]) -> dict[str, Any]:
    out = dict(model)
    artifacts = dict(out.get("artifacts") or {})
    out["artifact_status"] = {p: exists(p) for p in artifact_paths(out)}

    meta_path = artifacts.get("meta")
    out["runtime_meta"] = compact_meta(meta_path)

    onnx_exports: dict[str, str] = {}
    if isinstance(artifacts.get("onnx"), str):
        onnx_exports["primary"] = artifacts["onnx"]
    if isinstance(artifacts.get("onnx_buckets"), dict):
        for k, v in artifacts["onnx_buckets"].items():
            onnx_exports[f"k{k}"] = v

    export_stats = {}
    for label, path in onnx_exports.items():
        params = onnx_params(path)
        b = bundle_bytes(path)
        export_stats[label] = {
            "path": path,
            "exists": exists(path),
            "params": params,
            "bundle_bytes": b,
            "bundle_mib": mib(b),
            "fp32_param_mib": mib(params * 4) if params is not None else None,
            "fp16_param_mib_est": mib(params * 2) if params is not None else None,
            "int8_param_mib_est": mib(params) if params is not None else None,
            "int4_param_mib_est": mib(params / 2) if params is not None else None,
        }
    if export_stats:
        out["exports"] = export_stats
        primary = export_stats.get("primary") or next(iter(export_stats.values()))
        out["params"] = primary.get("params")
        out["bundle_bytes"] = primary.get("bundle_bytes")
        out["bundle_mib"] = primary.get("bundle_mib")

    if isinstance(artifacts.get("train_log"), str):
        out["training_metrics"] = parse_metrics(artifacts["train_log"])

    status_path = artifacts.get("status")
    if isinstance(status_path, str) and exists(status_path):
        out["current_status_text"] = rel(status_path).read_text(errors="replace").strip().splitlines()[-5:]

    return out



def model_onnx_lookup(models: list[dict[str, Any]]) -> dict[str, str]:
    onnx_to_model: dict[str, str] = {}
    for m in models:
        artifacts = m.get("artifacts") or {}
        for path in [artifacts.get("onnx")]:
            if isinstance(path, str):
                onnx_to_model[str(Path(path))] = m["model_id"]
        if isinstance(artifacts.get("onnx_buckets"), dict):
            for path in artifacts["onnx_buckets"].values():
                onnx_to_model[str(Path(path))] = m["model_id"]
    return onnx_to_model


def scan_anchor_arenas(models: list[dict[str, Any]], anchor_globs: list[str]) -> dict[str, list[dict[str, Any]]]:
    by_model = {m["model_id"]: [] for m in models}
    onnx_to_model = model_onnx_lookup(models)
    for glob in anchor_globs:
        for p in ROOT.glob(glob):
            if p.name.endswith(".protocol.json"):
                continue
            try:
                data = json.loads(p.read_text())
            except Exception:
                continue
            if not isinstance(data, dict):
                continue
            candidate = data.get("candidate") if isinstance(data.get("candidate"), dict) else {}
            model_id = onnx_to_model.get(str(Path(candidate.get("onnx")))) if candidate.get("onnx") else None
            summaries = data.get("summaries")
            protocol = data.get("protocol") if isinstance(data.get("protocol"), dict) else {}
            if not model_id or not isinstance(summaries, list):
                continue
            for summary in summaries:
                if not isinstance(summary, dict):
                    continue
                by_model[model_id].append({
                    "arena": str(p.relative_to(ROOT)),
                    "candidate": candidate.get("name"),
                    "anchor": summary.get("anchor"),
                    "visits": protocol.get("visits"),
                    "pairs": summary.get("pairs") or protocol.get("pairs"),
                    "games": summary.get("games"),
                    "wdl": [summary.get("wins"), summary.get("draws"), summary.get("losses")],
                    "score_rate": summary.get("scoreRate"),
                    "elo_diff": summary.get("eloDiff"),
                    "elo_ci95": summary.get("eloCi95"),
                    "illegal": summary.get("illegal"),
                    "mtime": p.stat().st_mtime,
                })
    for refs in by_model.values():
        refs.sort(key=lambda r: (str(r.get("anchor") or ""), int(r.get("games") or 0), float(r.get("mtime") or 0)))
        for r in refs:
            r.pop("mtime", None)
    return by_model


def select_anchor_headline(refs: list[dict[str, Any]]) -> dict[str, Any] | None:
    stockfish = [r for r in refs if r.get("anchor") == "stockfish_1320"]
    pool = stockfish or refs
    if not pool:
        return None
    return max(pool, key=lambda r: (int(r.get("games") or 0), int(r.get("visits") or 0), str(r.get("arena") or "")))


def fmt_anchor(ref: dict[str, Any] | None) -> str:
    if not ref:
        return "—"
    elo = ref.get("elo_diff")
    ci = ref.get("elo_ci95")
    score = ref.get("score_rate")
    bits = [str(ref.get("anchor") or "anchor")]
    if elo is not None:
        if ci is not None:
            bits.append(f"{float(elo):+.0f} ± {float(ci):.0f} Elo")
        else:
            bits.append(f"{float(elo):+.0f} Elo")
    elif score is not None:
        bits.append(f"score {float(score):.3f}")
    games = ref.get("games")
    visits = ref.get("visits")
    if games or visits:
        bits.append(f"{games or '?'}g/{visits or '?'}v")
    return " ".join(bits)

def scan_search_mode_arenas(models: list[dict[str, Any]], arena_globs: list[str]) -> dict[str, list[dict[str, Any]]]:
    by_model = {m["model_id"]: [] for m in models}
    onnx_to_model = model_onnx_lookup(models)

    for glob in arena_globs:
        for p in ROOT.glob(glob):
            try:
                data = json.loads(p.read_text())
            except Exception:
                continue
            if not isinstance(data, dict):
                continue
            protocol = data.get("protocol") or {}
            if not isinstance(protocol, dict):
                protocol = {}
            resources = protocol.get("modelResources") or {}
            standings = {s.get("name"): s for s in data.get("standings", []) if isinstance(s, dict)}
            for player, res in resources.items():
                if not isinstance(res, dict):
                    continue
                onnx_path = res.get("onnx")
                model_id = onnx_to_model.get(str(Path(onnx_path))) if onnx_path else None
                if not model_id:
                    continue
                s = standings.get(player, {})
                by_model[model_id].append({
                    "arena": str(p.relative_to(ROOT)),
                    "player": player,
                    "mode": next((pl.get("mode") for pl in protocol.get("players", []) if pl.get("name") == player), None),
                    "visits": protocol.get("visits"),
                    "games": s.get("games"),
                    "wdl": [s.get("wins"), s.get("draws"), s.get("losses")],
                    "score_rate": s.get("scoreRate"),
                    "elo_vs_pool": s.get("eloVsPool"),
                })
    return by_model


def build(overrides_path: str, arena_globs: list[str], anchor_arena_globs: list[str]) -> dict[str, Any]:
    overrides = read_json(overrides_path)
    if not isinstance(overrides, dict):
        raise SystemExit(f"Bad overrides JSON: {overrides_path}")
    family_promotion_gates = overrides.get("family_promotion_gates") or {}
    anchor_waivers = overrides.get("anchor_waivers") or {}
    dataset_defs = overrides.get("datasets") or {}
    datasets = {}
    for key, info in dataset_defs.items():
        summary = summarize_dataset(info.get("manifest")) if isinstance(info, dict) else None
        datasets[key] = {**info, "summary": summary}

    models = [enrich_model(m) for m in overrides.get("models", [])]
    arena_refs = scan_search_mode_arenas(models, arena_globs)
    anchor_refs = scan_anchor_arenas(models, anchor_arena_globs)
    missing_anchor: list[str] = []
    waived_anchor: list[str] = []
    for m in models:
        if not m.get("promotion_gate"):
            gate = family_promotion_gates.get(m.get("family"))
            if gate:
                m["promotion_gate"] = gate
        refs = arena_refs.get(m["model_id"], [])
        if refs:
            m["arena_refs"] = sorted(refs, key=lambda r: (r.get("arena") or "", r.get("player") or ""))[-20:]
        anchors = anchor_refs.get(m["model_id"], [])
        if anchors:
            m["anchor_refs"] = anchors[-20:]
            m["anchor_headline"] = select_anchor_headline(anchors)
        elif m.get("status") == "completed":
            waiver = anchor_waivers.get(m["model_id"])
            if waiver:
                m["anchor_waiver"] = waiver
                waived_anchor.append(m["model_id"])
            else:
                missing_anchor.append(m["model_id"])

    completed = [m for m in models if m.get("status") == "completed"]
    pareto = []
    for m in completed:
        if m.get("params") is None or m.get("bundle_bytes") is None:
            continue
        pareto.append({"model_id": m["model_id"], "params": m["params"], "bundle_bytes": m["bundle_bytes"]})

    return {
        "schema_version": 1,
        "generated_utc": _dt.datetime.now(_dt.UTC).isoformat(),
        "source_overrides": overrides_path,
        "datasets": datasets,
        "models": models,
        "summary": {
            "model_count": len(models),
            "completed_count": len(completed),
            "running_count": sum(1 for m in models if m.get("status") == "running"),
            "queued_count": sum(1 for m in models if m.get("status") == "queued"),
            "onnx_available_count": sum(1 for m in models if m.get("bundle_bytes") is not None),
            "anchor_headline_count": sum(1 for m in models if m.get("anchor_headline")),
            "anchor_waived_count": len(waived_anchor),
            "missing_anchor": missing_anchor,
            "waived_anchor": waived_anchor,
        },
    }


def fmt_int(x: Any) -> str:
    return "" if x is None else f"{int(x):,}"


def fmt_mib(x: Any) -> str:
    return "" if x is None else f"{float(x):.2f}"


def write_markdown(manifest: dict[str, Any], out_path: str) -> None:
    lines = [
        "# Current model manifest",
        "",
        f"Generated: `{manifest['generated_utc']}`",
        "",
        "This is the generated, human-readable view of `artifacts/analysis/model_manifest.current.json`.",
        "",
        "## Summary",
        "",
        f"- Models tracked: {manifest['summary']['model_count']}",
        f"- Completed: {manifest['summary']['completed_count']}",
        f"- Running: {manifest['summary']['running_count']}",
        f"- Queued: {manifest['summary']['queued_count']}",
        f"- ONNX exports available: {manifest['summary']['onnx_available_count']}",
        f"- Anchor headlines available: {manifest['summary']['anchor_headline_count']}",
        f"- Anchor waivers: {manifest['summary'].get('anchor_waived_count', 0)}",
        "",
        "## Models",
        "",
        "| Model | Status | Family | Arch | Params | ONNX MiB | INT8 est MiB | Anchor headline | Promotion / kill criteria | Training source | AV source | Tags |",
        "|---|---|---|---|---:|---:|---:|---|---|---|---|---|",
    ]
    for m in manifest["models"]:
        meta = m.get("runtime_meta") or {}
        arch = meta.get("architecture") or ""
        train = m.get("training") or {}
        params = m.get("params") or (m.get("planned_size_estimate") or {}).get("params")
        bundle = m.get("bundle_mib") or (m.get("planned_size_estimate") or {}).get("fp32_mib")
        int8 = None
        if m.get("params") is not None:
            int8 = mib(m["params"])
        elif (m.get("planned_size_estimate") or {}).get("int8_mib") is not None:
            int8 = (m.get("planned_size_estimate") or {}).get("int8_mib")
        gate = m.get("promotion_gate") or {}
        gate_text = gate.get("kill_if") or gate.get("promote_if") or ""
        if len(gate_text) > 120:
            gate_text = gate_text[:117] + "…"
        anchor_text = fmt_anchor(m.get('anchor_headline'))
        if anchor_text == "—" and m.get("anchor_waiver"):
            anchor_text = "waived"
        lines.append(
            f"| `{m['model_id']}` | {m.get('status','')} | {m.get('family','')} | {arch} | {fmt_int(params)} | {fmt_mib(bundle)} | {fmt_mib(int8)} | "
            f"{anchor_text} | {gate_text} | {train.get('policy_dataset') or ''} | {train.get('av_dataset') or ''} | {', '.join(m.get('tags') or [])} |"
        )
    lines += ["", "## Dataset summaries", ""]
    for key, info in manifest["datasets"].items():
        s = info.get("summary") or {}
        lines.append(f"### `{key}`")
        lines.append("")
        for k in ["manifest", "name", "format", "total_train_rows", "total_dev_rows", "positions", "candidate_rows", "top_k", "max_candidates", "train_shards", "shards", "history_plies"]:
            if s.get(k) is not None:
                v = s[k]
                if isinstance(v, int):
                    v = f"{v:,}"
                lines.append(f"- {k}: `{v}`")
        lines.append("")
    missing_anchor = manifest.get("summary", {}).get("missing_anchor") or []
    waived_anchor = manifest.get("summary", {}).get("waived_anchor") or []
    if missing_anchor:
        lines += ["", "## TODO", "", "Completed models missing anchor headline and no waiver:", ""]
        lines += [f"- `{model_id}`" for model_id in missing_anchor]
    if waived_anchor:
        lines += ["", "## Anchor waivers", ""]
        models_by_id = {m["model_id"]: m for m in manifest.get("models", [])}
        for model_id in waived_anchor:
            waiver = (models_by_id.get(model_id, {}).get("anchor_waiver") or {})
            lines.append(f"- `{model_id}`: {waiver.get('reason', 'waived')}")
    lines += [
        "",
        "## Notes",
        "",
        "- Provenance fields come from `eval/model_manifest_overrides.json`; computed fields come from ONNX/meta/log files.",
        "- Quantized sizes are parameter-only estimates unless actual quantized artifacts are listed.",
        "- Arena refs are discovered from `artifacts/search_mode_arena/*.json` when protocol model resources include ONNX paths.",
        "- Anchor headlines are discovered from `artifacts/anchor_arena/**/*.json` when candidate ONNX paths match manifest models.",
    ]
    rel(out_path).parent.mkdir(parents=True, exist_ok=True)  # type: ignore[union-attr]
    rel(out_path).write_text("\n".join(lines) + "\n")  # type: ignore[union-attr]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--overrides", default="eval/model_manifest_overrides.json")
    ap.add_argument("--out", default="artifacts/analysis/model_manifest.current.json")
    ap.add_argument("--md-out", default="artifacts/analysis/model_manifest.current.md")
    ap.add_argument("--arena-glob", action="append", default=["artifacts/search_mode_arena/*.json"])
    ap.add_argument("--anchor-arena-glob", action="append", default=["artifacts/anchor_arena/**/*.json"])
    args = ap.parse_args()

    manifest = build(args.overrides, args.arena_glob, args.anchor_arena_glob)
    out = rel(args.out)
    assert out is not None
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n")
    write_markdown(manifest, args.md_out)
    print(f"wrote {args.out}")
    print(f"wrote {args.md_out}")
    print(f"models={manifest['summary']['model_count']} completed={manifest['summary']['completed_count']} onnx={manifest['summary']['onnx_available_count']}")


if __name__ == "__main__":
    main()
