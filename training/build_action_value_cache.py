#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, subprocess, time
from pathlib import Path
from typing import Iterator, Any
import numpy as np

try:
    from training._lib.encoding import CHESSBENCH_PROMOTIONS as PROMOS, FILES, PIECES
except ModuleNotFoundError:
    from _lib.encoding import CHESSBENCH_PROMOTIONS as PROMOS, FILES, PIECES


try:
    from teacher_overlay_lib import SCHEMA_ACTION_VALUE
except Exception:
    SCHEMA_ACTION_VALUE = "teacher.action_value.v1"

POLICY_SIZE = 4096 + 4096 * 4


def open_lines(path: str | Path):
    path = str(path)
    if path.endswith(".zst"):
        p = subprocess.Popen(["zstd", "-dc", path], stdout=subprocess.PIPE, text=True)
        try:
            assert p.stdout is not None
            for line in p.stdout:
                yield line
        finally:
            if p.stdout:
                p.stdout.close()
            rc = p.wait()
            if rc and rc != -13:
                raise subprocess.CalledProcessError(rc, ["zstd", "-dc", path])
    else:
        with open(path, "rt", encoding="utf-8") as f:
            yield from f


def sq(s: str) -> int:
    return (int(s[1]) - 1) * 8 + FILES.index(s[0])


def move_class(uci: str) -> int:
    fr = sq(uci[:2])
    to = sq(uci[2:4])
    if len(uci) >= 5 and uci[4].lower() in PROMOS:
        return 4096 + (fr * 64 + to) * 4 + PROMOS[uci[4].lower()]
    return fr * 64 + to


def parse_board(fen: str) -> np.ndarray:
    b = np.zeros(64, dtype=np.uint8)
    ranks = fen.split()[0].split("/")
    if len(ranks) != 8:
        raise ValueError("bad FEN board")
    for rr, rank in enumerate(ranks):
        f = 0
        r = 7 - rr
        for ch in rank:
            if ch.isdigit():
                f += int(ch)
            else:
                b[r * 8 + f] = PIECES.index(ch)
                f += 1
    return b


def encode_tokens(fen: str, hist: list[str] | None, history: int) -> np.ndarray:
    parts = fen.split()
    stm = parts[1] if len(parts) > 1 else "w"
    cast = parts[2] if len(parts) > 2 else "-"
    ep = parts[3] if len(parts) > 3 else "-"
    half = int(parts[4]) if len(parts) > 4 and parts[4].isdigit() else 0
    F = history + 9
    out = np.zeros((64, F), dtype=np.uint8)
    boards = [parse_board(fen)]
    for h in (hist or [])[:history]:
        boards.append(parse_board(h))
    while len(boards) < history + 1:
        boards.append(np.zeros(64, dtype=np.uint8))
    for i, b in enumerate(boards):
        out[:, i] = b
    out[:, history + 1] = 1 if stm == "w" else 2
    flags = (
        (("K" in cast) << 0)
        | (("Q" in cast) << 1)
        | (("k" in cast) << 2)
        | (("q" in cast) << 3)
    )
    out[:, history + 2] = flags
    if len(ep) == 2 and ep[0] in FILES and ep[1].isdigit():
        out[sq(ep), history + 3] = 1
    out[:, history + 4] = min(255, half)
    for i in range(64):
        r, f = divmod(i, 8)
        out[i, history + 5] = r
        out[i, history + 6] = f
        out[i, history + 7] = 1 if ((r + f) & 1) else 0
        out[i, history + 8] = i
    return out


def iter_rows(paths: list[str]) -> Iterator[dict[str, Any]]:
    for p in paths:
        for line in open_lines(p):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def valid_candidate(r: dict[str, Any]) -> tuple[int, float, float, int] | None:
    if r.get("schema") != SCHEMA_ACTION_VALUE or not r.get("fen") or not r.get("move"):
        return None
    try:
        mv = move_class(str(r["move"]))
        val = float(r.get("value", 0.0))
        regret = float(r.get("regret_cp", 0.0) or 0.0) / 400.0
        rank = int(r.get("rank", 0) or 0)
        return mv, val, regret, rank
    except Exception:
        return None


def grouped_positions(paths: list[str], max_candidates: int, history: int):
    cur_key = None
    cur_fen = None
    cur_hist = None
    cands: list[tuple[int, float, float, int, int]] = []
    source_order = 0

    def flush():
        nonlocal cands, cur_fen, cur_hist
        if cur_fen is None or len(cands) < 2:
            return None
        # Keep explicit ranks first when present, otherwise source order. Values break ties.
        ordered = sorted(
            cands, key=lambda x: ((x[3] if x[3] > 0 else 10**9), x[4], -x[1])
        )[:max_candidates]
        try:
            tok = encode_tokens(cur_fen, cur_hist or [], history)
        except Exception:
            return None
        return tok, ordered

    for r in iter_rows(paths):
        cand = valid_candidate(r)
        if cand is None:
            continue
        key = r.get("position_key") or r.get("fen")
        if cur_key is not None and key != cur_key:
            out = flush()
            if out is not None:
                yield out
            cands = []
        if key != cur_key:
            cur_key = key
            cur_fen = r.get("fen")
            cur_hist = r.get("history_fens") or []
            source_order = 0
        source_order += 1
        mv, val, regret, rank = cand
        cands.append((mv, val, regret, rank, source_order))
    out = flush()
    if out is not None:
        yield out


def count_positions(
    paths: list[str], max_positions: int, max_candidates: int, history: int
) -> int:
    n = 0
    for _tok, _cands in grouped_positions(paths, max_candidates, history):
        n += 1
        if max_positions and n >= max_positions:
            break
    return n


def build(
    paths: list[str],
    out_dir: str | Path,
    max_positions: int,
    max_candidates: int,
    history: int,
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    F = history + 9
    n = count_positions(paths, max_positions, max_candidates, history)
    if n <= 0:
        raise SystemExit("no valid AV positions found")
    tokens = np.memmap(out / "tokens.uint8", np.uint8, "w+", shape=(n, 64, F))
    moves = np.memmap(
        out / "candidate_moves.int64", np.int64, "w+", shape=(n, max_candidates)
    )
    values = np.memmap(
        out / "candidate_values.float32", np.float32, "w+", shape=(n, max_candidates)
    )
    regrets = np.memmap(
        out / "candidate_regrets.float32", np.float32, "w+", shape=(n, max_candidates)
    )
    mask = np.memmap(
        out / "candidate_mask.float32", np.float32, "w+", shape=(n, max_candidates)
    )
    moves[:] = 0
    values[:] = 0.0
    regrets[:] = 0.0
    mask[:] = 0.0
    i = 0
    candidate_rows = 0
    for tok, cands in grouped_positions(paths, max_candidates, history):
        if i >= n:
            break
        tokens[i] = tok
        for j, (mv, val, reg, _rank, _ord) in enumerate(cands[:max_candidates]):
            moves[i, j] = mv
            values[i, j] = val
            regrets[i, j] = reg
            mask[i, j] = 1.0
            candidate_rows += 1
        i += 1
        if i % 100000 == 0:
            print(f"METRIC av_cache_positions_written={i}", flush=True)
    for a in (tokens, moves, values, regrets, mask):
        a.flush()
    meta = {
        "format": "compact_action_value_cache_v1",
        "rows": n,
        "candidate_rows": candidate_rows,
        "max_candidates": max_candidates,
        "history_plies": history,
        "token_features": F,
        "policy_size": POLICY_SIZE,
        "source_shards": paths,
        "seconds": time.time() - t0,
        "files": {
            "tokens": "tokens.uint8",
            "candidate_moves": "candidate_moves.int64",
            "candidate_values": "candidate_values.float32",
            "candidate_regrets": "candidate_regrets.float32",
            "candidate_mask": "candidate_mask.float32",
        },
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"METRIC av_cache_positions={n}")
    print(f"METRIC av_cache_candidate_rows={candidate_rows}")
    print(f"METRIC av_cache_seconds={meta['seconds']:.3f}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Build tensorized memmap cache for teacher.action_value.v1 overlays."
    )
    ap.add_argument("--input", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-positions", type=int, default=0)
    ap.add_argument("--max-candidates", type=int, default=8)
    ap.add_argument("--history-plies", type=int, default=2)
    args = ap.parse_args()
    build(
        args.input,
        args.out,
        args.max_positions,
        args.max_candidates,
        args.history_plies,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
