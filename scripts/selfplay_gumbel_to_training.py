#!/usr/bin/env python3
"""Adapt Gumbel self-play chunks into Tiny Leela training rows.

The Gumbel self-play writer stores a soft improved root policy plus diagnostics
(noise, candidate ranks, Qs).  Most supervised/cache builders in this repo read
standard JSONL rows with ``fen``, ``history_fens``, ``policy``, ``wdl``/``result``
and optional ``weight``.  This adapter makes that bridge explicit and records
provenance so supervised-bootstrap self-play does not get mixed into clean
from-scratch Gumbel-Zero replay by accident.

Default output mode is ``expanded``: each source position becomes one weighted
one-hot row per target-policy move.  This is compatible with the current CNN and
cache builders that still expect a single policy move per row.  ``soft`` mode
preserves the policy distribution for consumers that support soft targets.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import shutil
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, TextIO


SUPERVISED_SP_SCHEMA = "tiny_leela_supervised_sp_training_v1"
CLEAN_ZERO_SCHEMA = "tiny_leela_zero_training_v1"


@contextmanager
def _zstd_cli_reader(path: Path):
    if not shutil.which("zstd"):
        raise SystemExit(f"zstandard python module or zstd CLI is required to read {path}")
    proc = subprocess.Popen(["zstd", "-dc", str(path)], stdout=subprocess.PIPE)
    assert proc.stdout is not None
    with proc.stdout, open(proc.stdout.fileno(), mode="r", encoding="utf-8", closefd=False) as fh:
        yield fh
    rc = proc.wait()
    if rc != 0:
        raise SystemExit(f"zstd failed while reading {path} rc={rc}")


@contextmanager
def _zstd_cli_writer(path: Path):
    if not shutil.which("zstd"):
        raise SystemExit(f"zstandard python module or zstd CLI is required to write {path}")
    proc = subprocess.Popen(["zstd", "-q", "-T0", "-o", str(path), "-"], stdin=subprocess.PIPE)
    assert proc.stdin is not None
    with proc.stdin, open(proc.stdin.fileno(), mode="w", encoding="utf-8", closefd=False) as fh:
        yield fh
    rc = proc.wait()
    if rc != 0:
        raise SystemExit(f"zstd failed while writing {path} rc={rc}")


def _open_text(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    if str(path).endswith(".zst"):
        try:
            import zstandard as zstd  # type: ignore
            fh = path.open("rb")
            return zstd.open(fh, mode="rt", encoding="utf-8")  # type: ignore[return-value]
        except ImportError:
            return _zstd_cli_reader(path)
        except Exception as exc:  # pragma: no cover - optional dependency
            raise SystemExit(f"zstandard failed to read {path}: {exc}")
    return path.open("r", encoding="utf-8")


def _write_text(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if str(path).endswith(".gz"):
        return gzip.open(path, "wt", encoding="utf-8")
    if str(path).endswith(".zst"):
        try:
            import zstandard as zstd  # type: ignore
            fh = path.open("wb")
            return zstd.open(fh, mode="wt", encoding="utf-8")  # type: ignore[return-value]
        except ImportError:
            return _zstd_cli_writer(path)
        except Exception as exc:  # pragma: no cover - optional dependency
            raise SystemExit(f"zstandard failed to write {path}: {exc}")
    return path.open("w", encoding="utf-8")


def _read_jsonl(paths: Iterable[Path]) -> Iterable[tuple[Path, int, dict]]:
    for path in paths:
        with _open_text(path) as fh:
            for line_no, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception as exc:
                    raise SystemExit(f"{path}:{line_no}: JSON parse error: {exc}") from exc
                if not isinstance(row, dict):
                    raise SystemExit(f"{path}:{line_no}: expected object row")
                yield path, line_no, row


def _normalize_policy(policy: dict, *, min_prob: float, max_moves: int) -> dict[str, float]:
    items: list[tuple[str, float]] = []
    for move, prob in policy.items():
        try:
            p = float(prob)
        except Exception:
            continue
        if not isinstance(move, str) or len(move) not in {4, 5}:
            continue
        if math.isfinite(p) and p > min_prob:
            items.append((move, p))
    items.sort(key=lambda x: x[1], reverse=True)
    if max_moves > 0:
        items = items[:max_moves]
    mass = sum(p for _, p in items)
    if mass <= 0:
        return {}
    return {move: p / mass for move, p in items}


def _q_to_wdl(q: float) -> list[float]:
    q = max(-1.0, min(1.0, q))
    return [max(q, 0.0), 1.0 - abs(q), max(-q, 0.0)]


def _normalize_wdl(wdl) -> list[float] | None:
    if not isinstance(wdl, list) or len(wdl) != 3:
        return None
    try:
        vals = [float(v) for v in wdl]
    except Exception:
        return None
    if not all(math.isfinite(v) and v >= 0 for v in vals):
        return None
    mass = sum(vals)
    if mass <= 0:
        return None
    return [v / mass for v in vals]


def _row_wdl(row: dict, value_target: str, root_mix: float) -> list[float]:
    result = _normalize_wdl(row.get("result"))
    root = _normalize_wdl(row.get("root_wdl"))
    if root is None and "root_value" in row:
        try:
            root = _q_to_wdl(float(row["root_value"]))
        except Exception:
            root = None
    if value_target == "result":
        if result is None:
            raise ValueError("missing/invalid result WDL")
        return result
    if value_target == "root":
        if root is None:
            raise ValueError("missing/invalid root value/WDL")
        return root
    if result is None or root is None:
        raise ValueError("mix value target requires result and root value/WDL")
    a = max(0.0, min(1.0, root_mix))
    out = [(1 - a) * result[i] + a * root[i] for i in range(3)]
    mass = sum(out)
    return [v / mass for v in out]


def _source_id(row: dict, fallback_idx: int) -> str:
    gid = str(row.get("game_id") or f"row{fallback_idx:08d}")
    ply = row.get("ply")
    return f"{gid}_p{int(ply):04d}" if isinstance(ply, int) else gid


def adapt(args: argparse.Namespace) -> dict:
    inputs = [Path(p) for p in args.input]
    if args.lane == "supervised_sp" and not args.source_model:
        raise SystemExit("--source-model is required for --lane supervised_sp")
    if args.lane == "zero" and args.source_model and not args.allow_source_model_in_zero:
        raise SystemExit("Refusing source-model provenance in zero lane; use --lane supervised_sp")

    output = Path(args.output)
    manifest_path = Path(args.manifest_out) if args.manifest_out else output.with_suffix(output.suffix + ".manifest.json")
    schema = SUPERVISED_SP_SCHEMA if args.lane == "supervised_sp" else CLEAN_ZERO_SCHEMA
    seen = 0
    kept_positions = 0
    emitted = 0
    skipped = 0
    policy_mass_in = 0.0
    expanded_weights = 0.0
    per_file: dict[str, dict[str, int]] = {}

    with _write_text(output) as out_fh:
        for idx, (path, line_no, row) in enumerate(_read_jsonl(inputs), 1):
            seen += 1
            key = str(path)
            per_file.setdefault(key, {"rows": 0, "kept": 0, "skipped": 0})["rows"] += 1
            raw_policy = row.get("policy")
            if not isinstance(raw_policy, dict) or not raw_policy:
                skipped += 1
                per_file[key]["skipped"] += 1
                continue
            policy = _normalize_policy(raw_policy, min_prob=args.min_prob, max_moves=args.max_policy_moves)
            if not policy:
                skipped += 1
                per_file[key]["skipped"] += 1
                continue
            try:
                wdl = _row_wdl(row, args.value_target, args.root_mix)
            except ValueError:
                skipped += 1
                per_file[key]["skipped"] += 1
                continue
            fen = row.get("fen")
            if not isinstance(fen, str) or not fen:
                skipped += 1
                per_file[key]["skipped"] += 1
                continue
            history = row.get("history_fens") or []
            if not isinstance(history, list):
                history = []
            base_weight = float(row.get("weight", 1.0)) * float(args.position_weight)
            base = {
                "schema": schema,
                "game_id": str(row.get("game_id") or f"row{idx:08d}"),
                "ply": int(row.get("ply", 0)) if isinstance(row.get("ply", 0), int) else 0,
                "turn": row.get("turn") if row.get("turn") in {"w", "b"} else (fen.split()[1] if len(fen.split()) > 1 and fen.split()[1] in {"w", "b"} else "w"),
                "fen": fen,
                "history_fens": [h for h in history if isinstance(h, str)][: args.history_plies],
                "wdl": [round(v, 10) for v in wdl],
                "result": [round(v, 10) for v in wdl],
                "q": round(wdl[0] - wdl[2], 10),
                "source": args.lane,
                "source_schema": row.get("schema", ""),
                "source_id": _source_id(row, idx),
                "source_model": args.source_model,
                "search": {
                    "kind": "gumbel" if "gumbel" in str(row.get("schema", "")) else "puct",
                    "visits": row.get("visits"),
                    "requested_visits": row.get("requested_visits"),
                    "candidate_count": row.get("candidate_count"),
                    "evaluator": row.get("evaluator"),
                },
            }
            if "root_value" in row:
                base["root_value"] = row["root_value"]
            if "selected_move" in row:
                base["selected_move"] = row["selected_move"]
            kept_positions += 1
            per_file[key]["kept"] += 1
            policy_mass_in += sum(float(v) for v in raw_policy.values() if isinstance(v, (int, float)))

            if args.mode == "soft":
                out_row = dict(base)
                out_row["id"] = base["source_id"]
                out_row["game_id"] = base["source_id"]
                out_row["policy"] = {m: round(p, 10) for m, p in policy.items()}
                out_row["weight"] = round(base_weight, 10)
                out_fh.write(json.dumps(out_row, sort_keys=True) + "\n")
                emitted += 1
                expanded_weights += base_weight
            else:
                for rank, (move, prob) in enumerate(policy.items(), 1):
                    w = base_weight * prob
                    if w <= 0:
                        continue
                    out_row = dict(base)
                    out_row["id"] = f"{base['source_id']}_m{rank:02d}"
                    out_row["game_id"] = out_row["id"]
                    out_row["policy"] = {move: 1.0}
                    out_row["policy_prob"] = round(prob, 10)
                    out_row["policy_rank"] = rank
                    out_row["weight"] = round(w, 10)
                    out_fh.write(json.dumps(out_row, sort_keys=True) + "\n")
                    emitted += 1
                    expanded_weights += w

    manifest = {
        "schema": "tiny_leela_selfplay_adapter_manifest_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "inputs": [str(p) for p in inputs],
        "output": str(output),
        "mode": args.mode,
        "lane": args.lane,
        "source_model": args.source_model,
        "value_target": args.value_target,
        "root_mix": args.root_mix,
        "history_plies": args.history_plies,
        "max_policy_moves": args.max_policy_moves,
        "min_prob": args.min_prob,
        "rows_seen": seen,
        "positions_kept": kept_positions,
        "rows_emitted": emitted,
        "rows_skipped": skipped,
        "avg_input_policy_mass": policy_mass_in / max(1, kept_positions),
        "expanded_weight_sum": expanded_weights,
        "per_file": per_file,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", nargs="+", required=True, help="Gumbel/self-play JSONL chunks")
    ap.add_argument("--output", required=True, help="adapted training JSONL/JSONL.GZ/JSONL.ZST")
    ap.add_argument("--manifest-out", default="", help="adapter manifest path")
    ap.add_argument("--mode", choices=["expanded", "soft"], default="expanded")
    ap.add_argument("--lane", choices=["supervised_sp", "zero"], default="supervised_sp")
    ap.add_argument("--source-model", default="", help="model/checkpoint that generated bootstrap self-play")
    ap.add_argument("--allow-source-model-in-zero", action="store_true")
    ap.add_argument("--value-target", choices=["result", "root", "mix"], default="result")
    ap.add_argument("--root-mix", type=float, default=0.25, help="root target weight for --value-target mix")
    ap.add_argument("--history-plies", type=int, default=8)
    ap.add_argument("--max-policy-moves", type=int, default=0, help="0 keeps all moves")
    ap.add_argument("--min-prob", type=float, default=0.0)
    ap.add_argument("--position-weight", type=float, default=1.0)
    args = ap.parse_args()
    manifest = adapt(args)
    print(f"METRIC selfplay_adapter_rows_seen={manifest['rows_seen']}")
    print(f"METRIC selfplay_adapter_positions_kept={manifest['positions_kept']}")
    print(f"METRIC selfplay_adapter_rows_emitted={manifest['rows_emitted']}")
    print(f"METRIC selfplay_adapter_rows_skipped={manifest['rows_skipped']}")
    print(f"METRIC selfplay_adapter_expanded_weight_sum={manifest['expanded_weight_sum']:.6f}")
    print(f"METRIC selfplay_adapter_output={manifest['output']}")
    return 0 if manifest["positions_kept"] > 0 and manifest["rows_emitted"] > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
