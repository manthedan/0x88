from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def _meta(path: str | Path) -> tuple[Path, dict[str, Any]]:
    d = Path(path)
    return d, json.loads((d / "meta.json").read_text())


def _check_sha256(d: Path, meta: dict[str, Any], *, strict: bool = False) -> None:
    checksums = meta.get("checksums_sha256") or {}
    if not checksums:
        return
    for name, expected in checksums.items():
        p = d / name
        if not p.exists():
            if strict:
                raise FileNotFoundError(p)
            continue
        actual = hashlib.sha256(p.read_bytes()).hexdigest()
        if actual != expected:
            msg = f"checksum mismatch for {p}: expected {expected}, got {actual}"
            if strict:
                raise ValueError(msg)
            print(f"WARNING {msg}", flush=True)


def open_moveformer_sidecar(path: str | Path, *, verify_sha256: bool = False) -> dict[str, Any]:
    """Open Python- or Rust-built MoveFormer sidecar caches for training.

    Python builders write ``policy_legal_slot.int16``. Rust builders write the same
    semantic target slots as ``target_legal_slot.int16``; this reader normalizes both
    spellings to the ``policy_slot`` key used by trainers.
    """
    d, meta = _meta(path)
    _check_sha256(d, meta, strict=verify_sha256)
    rows = int(meta["rows"])
    k = int(meta["max_legal_moves"])
    f = int(meta["num_move_features"])
    slot_path = d / "policy_legal_slot.int16"
    if not slot_path.exists():
        slot_path = d / "target_legal_slot.int16"
    out: dict[str, Any] = {
        "path": str(d),
        "meta": meta,
        "rows": rows,
        "K": k,
        "F": f,
        "policy_slot": np.memmap(slot_path, np.int16, "r", shape=(rows,)),
        "wdl": np.memmap(d / "wdl.float32", np.float32, "r", shape=(rows, 3)),
        "q": np.memmap(d / "q.float32", np.float32, "r", shape=(rows,)),
        "legal_action_ids": np.memmap(d / "legal_action_ids.int64", np.int64, "r", shape=(rows, k)),
        "legal_features": np.memmap(d / "legal_features.float32", np.float32, "r", shape=(rows, k, f)),
        "legal_mask": np.memmap(d / "legal_mask.float32", np.float32, "r", shape=(rows, k)),
    }
    weight_path = d / "weight.float32"
    out["weight"] = (
        np.memmap(weight_path, np.float32, "r", shape=(rows,))
        if weight_path.exists()
        else np.ones(rows, dtype=np.float32)
    )
    if meta.get("has_board_cache"):
        c = int(meta["input_planes"])
        out["x"] = np.memmap(d / "x.int8", np.int8, "r", shape=(rows, c, 8, 8))
        out["input_planes"] = c
    return out


def open_squareformer_token_cache(path: str | Path, *, verify_sha256: bool = False) -> dict[str, Any]:
    d, meta = _meta(path)
    _check_sha256(d, meta, strict=verify_sha256)
    rows = int(meta["rows"])
    f = int(meta["token_features"])
    weight_path = d / "weight.float32"
    return {
        "path": str(d),
        "meta": meta,
        "rows": rows,
        "F": f,
        "tokens": np.memmap(d / "tokens.uint8", np.uint8, "r", shape=(rows, 64, f)),
        "policy": np.memmap(d / "policy.int64", np.int64, "r", shape=(rows,)),
        "wdl": np.memmap(d / "wdl.float32", np.float32, "r", shape=(rows, 3)),
        "weight": np.memmap(weight_path, np.float32, "r", shape=(rows,)) if weight_path.exists() else np.ones(rows, dtype=np.float32),
    }


def open_action_value_cache(path: str | Path, *, verify_sha256: bool = False) -> dict[str, Any]:
    d, meta = _meta(path)
    _check_sha256(d, meta, strict=verify_sha256)
    rows = int(meta["rows"])
    k = int(meta["max_candidates"])
    f = int(meta["token_features"])
    return {
        "path": str(d),
        "meta": meta,
        "rows": rows,
        "K": k,
        "F": f,
        "tokens": np.memmap(d / "tokens.uint8", np.uint8, "r", shape=(rows, 64, f)),
        "moves": np.memmap(d / "candidate_moves.int64", np.int64, "r", shape=(rows, k)),
        "values": np.memmap(d / "candidate_values.float32", np.float32, "r", shape=(rows, k)),
        "regrets": np.memmap(d / "candidate_regrets.float32", np.float32, "r", shape=(rows, k)),
        "mask": np.memmap(d / "candidate_mask.float32", np.float32, "r", shape=(rows, k)),
    }


def open_residual_cache(path: str | Path, *, verify_sha256: bool = False) -> dict[str, Any]:
    d, meta = _meta(path)
    _check_sha256(d, meta, strict=verify_sha256)
    rows = int(meta["rows"])
    c = int(meta["input_planes"])
    out: dict[str, Any] = {
        "path": str(d),
        "meta": meta,
        "rows": rows,
        "input_planes": c,
        "x": np.memmap(d / "x.int8", np.int8, "r", shape=(rows, c, 8, 8)),
        "policy": np.memmap(d / "policy.int64", np.int64, "r", shape=(rows,)),
        "wdl": np.memmap(d / "wdl.float32", np.float32, "r", shape=(rows, 3)),
    }
    for name, dtype in [
        ("weight.float32", np.float32),
        ("stockfish_q.float32", np.float32),
        ("stockfish_winrate_loss.float32", np.float32),
        ("stockfish_blunder_bucket.int64", np.int64),
    ]:
        p = d / name
        if p.exists():
            out[name.split(".")[0]] = np.memmap(p, dtype, "r", shape=(rows,))
    return out
