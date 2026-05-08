#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, subprocess, time
from pathlib import Path
import numpy as np
import msgpack

FILES = 'abcdefgh'
PIECES = '.PNBRQKpnbrqk'
PROMOS = {'n': 0, 'b': 1, 'r': 2, 'q': 3}
POLICY_SIZE = 4096 + 4096 * 4

def iter_msgpack_zst(path: str | Path):
    p = subprocess.Popen(['zstd', '-dc', str(path)], stdout=subprocess.PIPE)
    try:
        assert p.stdout is not None
        unpacker = msgpack.Unpacker(p.stdout, raw=False)
        for rec in unpacker:
            yield rec
    finally:
        try:
            if p.stdout:
                p.stdout.close()
        finally:
            p.kill(); p.wait()

def sq(s: str) -> int:
    return (int(s[1]) - 1) * 8 + FILES.index(s[0])

def move_class(uci: str) -> int:
    fr = sq(uci[:2]); to = sq(uci[2:4])
    if len(uci) >= 5 and uci[4].lower() in PROMOS:
        return 4096 + (fr * 64 + to) * 4 + PROMOS[uci[4].lower()]
    return fr * 64 + to

def parse_board(fen: str) -> np.ndarray:
    b = np.zeros(64, dtype=np.uint8)
    ranks = fen.split()[0].split('/')
    if len(ranks) != 8:
        raise ValueError('bad FEN board')
    for rr, rank in enumerate(ranks):
        f = 0; r = 7 - rr
        for ch in rank:
            if ch.isdigit():
                f += int(ch)
            else:
                b[r * 8 + f] = PIECES.index(ch); f += 1
    return b

def encode(fen: str, history: int) -> np.ndarray:
    parts = fen.split(); stm = parts[1] if len(parts) > 1 else 'w'; cast = parts[2] if len(parts) > 2 else '-'; ep = parts[3] if len(parts) > 3 else '-'; half = int(parts[4]) if len(parts) > 4 and parts[4].isdigit() else 0
    F = history + 9; out = np.zeros((64, F), dtype=np.uint8)
    out[:, 0] = parse_board(fen)
    # Public ChessBench records do not carry position history; keep history planes empty.
    out[:, history + 1] = 1 if stm == 'w' else 2
    flags = (('K' in cast) << 0) | (('Q' in cast) << 1) | (('k' in cast) << 2) | (('q' in cast) << 3)
    out[:, history + 2] = flags
    if len(ep) == 2 and ep[0] in FILES and ep[1].isdigit():
        out[sq(ep), history + 3] = 1
    out[:, history + 4] = min(255, half)
    for i in range(64):
        r, f = divmod(i, 8)
        out[i, history + 5] = r; out[i, history + 6] = f; out[i, history + 7] = 1 if ((r + f) & 1) else 0; out[i, history + 8] = i
    return out

def mate_to_q(mate):
    if mate is None:
        return None
    if mate == '#':
        return 1.0
    try:
        m = int(mate)
        return 1.0 if m > 0 else -1.0
    except Exception:
        return None

def q_from_win_prob(win_prob: float) -> float:
    return max(-1.0, min(1.0, 2.0 * float(win_prob) - 1.0))

def valid_record(rec: dict, top_k: int):
    fen = rec.get('fen') if isinstance(rec, dict) else None
    moves = rec.get('moves') if isinstance(rec, dict) else None
    if not fen or not isinstance(moves, dict):
        return None
    items = []
    for move, payload0 in moves.items():
        payload = payload0 if isinstance(payload0, dict) else {'win_prob': payload0}
        try:
            wp = float(payload.get('win_prob'))
            mv = move_class(str(move))
        except Exception:
            continue
        q_mate = mate_to_q(payload.get('mate'))
        q = q_mate if q_mate is not None else q_from_win_prob(wp)
        items.append((str(move), mv, wp, q, payload.get('mate')))
    if len(items) < 2:
        return None
    items.sort(key=lambda x: x[2], reverse=True)
    if top_k:
        items = items[:top_k]
    if len(items) < 2:
        return None
    best_q = items[0][3]
    cands = []
    for rank, (_uci, mv, _wp, q, _mate) in enumerate(items, start=1):
        cands.append((mv, q, max(0.0, best_q - q), rank))
    return fen, cands

def count(path: str, max_positions: int, top_k: int) -> tuple[int, int]:
    n = cand = 0
    for rec in iter_msgpack_zst(path):
        rr = valid_record(rec, top_k)
        if rr is None:
            continue
        n += 1; cand += len(rr[1])
        if max_positions and n >= max_positions:
            break
    return n, cand

def build(path: str, out_dir: str, max_positions: int, top_k: int, max_candidates: int, history: int) -> None:
    t0 = time.time(); out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    keep = top_k or max_candidates
    if keep <= 0:
        keep = max_candidates
    C = max_candidates
    n, _cand_pre = count(path, max_positions, keep)
    if n <= 0:
        raise SystemExit(f'no valid ChessBench records in {path}')
    F = history + 9
    tokens = np.memmap(out / 'tokens.uint8', np.uint8, 'w+', shape=(n, 64, F))
    moves = np.memmap(out / 'candidate_moves.int64', np.int64, 'w+', shape=(n, C))
    values = np.memmap(out / 'candidate_values.float32', np.float32, 'w+', shape=(n, C))
    regrets = np.memmap(out / 'candidate_regrets.float32', np.float32, 'w+', shape=(n, C))
    mask = np.memmap(out / 'candidate_mask.float32', np.float32, 'w+', shape=(n, C))
    moves[:] = 0; values[:] = 0.0; regrets[:] = 0.0; mask[:] = 0.0
    i = cand = bad_encode = 0
    for rec in iter_msgpack_zst(path):
        rr = valid_record(rec, keep)
        if rr is None:
            continue
        fen, cands = rr
        try:
            tokens[i] = encode(fen, history)
        except Exception:
            bad_encode += 1; continue
        for j, (mv, q, reg, _rank) in enumerate(cands[:C]):
            moves[i, j] = mv; values[i, j] = q; regrets[i, j] = reg; mask[i, j] = 1.0; cand += 1
        i += 1
        if max_positions and i >= max_positions:
            break
    if i != n:
        # Shrinking memmaps in-place is awkward; invalid encoded rows are masked out in metadata, but should be rare.
        n = i
    for a in (tokens, moves, values, regrets, mask):
        a.flush()
    meta = {
        'format': 'compact_action_value_cache_v1',
        'source_format': 'chessbench_msgpack_zst',
        'source': path,
        'rows': n,
        'candidate_rows': cand,
        'max_candidates': C,
        'top_k': keep,
        'history_plies': history,
        'token_features': F,
        'policy_size': POLICY_SIZE,
        'bad_encode_rows': bad_encode,
        'seconds': time.time() - t0,
        'files': {
            'tokens': 'tokens.uint8', 'candidate_moves': 'candidate_moves.int64', 'candidate_values': 'candidate_values.float32', 'candidate_regrets': 'candidate_regrets.float32', 'candidate_mask': 'candidate_mask.float32'
        },
    }
    (out / 'meta.json').write_text(json.dumps(meta, indent=2))
    print(f'METRIC chessbench_av_cache_positions={n}')
    print(f'METRIC chessbench_av_cache_candidate_rows={cand}')
    print(f'METRIC chessbench_av_cache_seconds={meta["seconds"]:.3f}')

def main() -> int:
    ap = argparse.ArgumentParser(description='Directly build compact AV cache from one ChessBench msgpack.zst shard.')
    ap.add_argument('--input', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--max-positions', type=int, default=0)
    ap.add_argument('--top-k', type=int, default=8)
    ap.add_argument('--max-candidates', type=int, default=8)
    ap.add_argument('--history-plies', type=int, default=2)
    a = ap.parse_args()
    build(a.input, a.out, a.max_positions, a.top_k, a.max_candidates, a.history_plies)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
