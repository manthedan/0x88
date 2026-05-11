#!/usr/bin/env python3
from __future__ import annotations
import argparse, contextlib, json, subprocess
from pathlib import Path
import numpy as np

try:
    from training._lib.encoding import CHESSBENCH_PROMOTIONS as PROMOS, FILES, PIECES
except ModuleNotFoundError:
    from _lib.encoding import CHESSBENCH_PROMOTIONS as PROMOS, FILES, PIECES

try:
    import pyzstd  # type: ignore
except Exception:
    pyzstd = None


def opener(path):
    path = str(path)
    if path.endswith(".zst"):
        if pyzstd is not None:
            return pyzstd.open(path, "rt")
        p = subprocess.Popen(["zstd", "-dc", path], stdout=subprocess.PIPE, text=True)
        return p.stdout
    return open(path, "rt", encoding="utf-8")


def close_opener(f):
    try:
        f.close()
    except Exception:
        pass


def sq(s):
    return (int(s[1]) - 1) * 8 + FILES.index(s[0])


def move_class(uci):
    fr = sq(uci[:2])
    to = sq(uci[2:4])
    if len(uci) >= 5 and uci[4].lower() in PROMOS:
        return 4096 + (fr * 64 + to) * 4 + PROMOS[uci[4].lower()]
    return fr * 64 + to


def parse_board(fen):
    b = np.zeros(64, dtype=np.uint8)
    ranks = fen.split()[0].split("/")
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


def encode(fen, hist, history):
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


def valid_row(line, history):
    if not line.strip():
        return None
    r = json.loads(line)
    pol = r.get("policy") or {}
    if len(pol) != 1 or "fen" not in r:
        return None
    mv = next(iter(pol.keys()))
    try:
        y = move_class(mv)
        x = encode(r["fen"], r.get("history_fens") or [], history)
    except Exception:
        return None
    wdl = np.asarray(r.get("wdl", [0.25, 0.5, 0.25]), dtype=np.float32)
    return x, y, wdl


def count_rows(inputs, max_rows, history):
    n = bad = 0
    for path in inputs:
        f = opener(path)
        try:
            for line in f:
                if max_rows and n >= max_rows:
                    return n, bad
                rr = valid_row(line, history)
                if rr is None:
                    bad += 1
                    continue
                n += 1
        finally:
            close_opener(f)
    return n, bad


def build(inputs, out, max_rows, history):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    F = history + 9
    n, bad = count_rows(inputs, max_rows, history)
    tokens = np.memmap(out / "tokens.uint8", np.uint8, "w+", shape=(n, 64, F))
    policy = np.memmap(out / "policy.int64", np.int64, "w+", shape=(n,))
    wdl = np.memmap(out / "wdl.float32", np.float32, "w+", shape=(n, 3))
    i = 0
    for path in inputs:
        f = opener(path)
        try:
            for line in f:
                if i >= n:
                    break
                rr = valid_row(line, history)
                if rr is None:
                    continue
                tokens[i] = rr[0]
                policy[i] = rr[1]
                wdl[i] = rr[2]
                i += 1
                if i % 100000 == 0:
                    print(f"METRIC square_cache_rows_written={i}", flush=True)
        finally:
            close_opener(f)
    for a in (tokens, policy, wdl):
        a.flush()
    meta = {
        "rows": n,
        "token_features": F,
        "history_plies": history,
        "policy_size": 4096 + 4096 * 4,
        "format": "compact_square_tokens_v1",
        "bad_or_skipped_rows": bad,
    }
    (out / "meta.json").write_text(json.dumps(meta, separators=(",", ":")))
    print(f"METRIC square_cache_rows={n}")
    print(f"METRIC square_cache_token_features={F}")
    print(f"METRIC square_cache_bad_rows={bad}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--history-plies", type=int, default=2)
    a = ap.parse_args()
    build(a.input, a.out, a.max_rows, a.history_plies)


if __name__ == "__main__":
    main()
