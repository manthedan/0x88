#!/usr/bin/env python3
from __future__ import annotations
import contextlib, csv, hashlib, json, math, subprocess
from pathlib import Path
from typing import Iterable, Iterator, Any

SCHEMA_POSITION_EVAL = 'teacher.position_eval.v1'
SCHEMA_ACTION_VALUE = 'teacher.action_value.v1'
SCHEMA_SEARCH_POLICY = 'teacher.search_policy.v1'
SCHEMA_TACTICAL_LINE = 'teacher.tactical_line.v1'

def fen_key(fen: str) -> str:
    """Normalize FEN to board/stm/castling/ep, matching the position registry."""
    return ' '.join(str(fen).strip().split()[:4])

def position_key(fen: str) -> str:
    return 'sha256:' + hashlib.sha256(fen_key(fen).encode('utf-8')).hexdigest()

def cp_to_q(cp: float, scale: float = 400.0) -> float:
    return math.tanh(max(-2000.0, min(2000.0, float(cp))) / scale)

def q_to_wdl(q: float, draw_floor: float = 0.0) -> list[float]:
    q = max(-1.0, min(1.0, float(q)))
    win = max(0.0, q)
    loss = max(0.0, -q)
    draw = max(draw_floor, 1.0 - win - loss)
    s = win + draw + loss
    return [win / s, draw / s, loss / s]

def cp_to_wdl(cp: float) -> tuple[list[float], float]:
    q = cp_to_q(cp)
    return q_to_wdl(q), q

def mate_to_q(mate: int | float) -> float:
    # Positive mate is winning for the reporting perspective.
    m = float(mate)
    return 1.0 if m > 0 else -1.0

def eval_to_wdl_q(cp: float | None, mate: int | None) -> tuple[list[float] | None, float | None]:
    if mate is not None:
        q = mate_to_q(mate)
        return q_to_wdl(q), q
    if cp is not None:
        return cp_to_wdl(cp)
    return None, None

@contextlib.contextmanager
def open_text(path: str | Path):
    path = str(path)
    if path.endswith('.zst'):
        p = subprocess.Popen(['zstd', '-dc', path], stdout=subprocess.PIPE, text=True)
        try:
            assert p.stdout is not None
            yield p.stdout
        finally:
            if p.stdout:
                p.stdout.close()
            rc = p.wait()
            if rc and rc != -13:
                raise subprocess.CalledProcessError(rc, ['zstd', '-dc', path])
    else:
        with open(path, 'rt', encoding='utf-8', newline='') as f:
            yield f

def write_jsonl_zst(rows: Iterable[dict[str, Any]], out: str | Path) -> int:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + '.tmpjsonl') if out.suffix != '.zst' else out.with_suffix('.jsonl.tmp')
    n = 0
    with tmp.open('wt', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, separators=(',', ':'), ensure_ascii=False) + '\n')
            n += 1
    if str(out).endswith('.zst'):
        subprocess.check_call(['zstd', '-q', '-f', '-T0', str(tmp), '-o', str(out)])
        tmp.unlink()
    else:
        tmp.replace(out)
    return n

def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with open_text(path) as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)

def iter_csv(path: str | Path) -> Iterator[dict[str, str]]:
    with open_text(path) as f:
        yield from csv.DictReader(f)

def pv_first_move(pv: Any) -> str | None:
    if pv is None:
        return None
    if isinstance(pv, str):
        parts = pv.strip().split()
    elif isinstance(pv, list):
        parts = [str(x) for x in pv]
    else:
        return None
    return parts[0] if parts else None

def policy_from_best(move: str | None) -> dict[str, float] | None:
    return {move: 1.0} if move else None

def quality_from_depth_nodes(depth: int | None, nodes: int | None = None, knodes: int | None = None) -> float:
    # Conservative monotonic quality weight. Training code can override later.
    d = 0.0 if depth is None else min(1.0, max(0.0, float(depth) / 40.0))
    n = None
    if nodes is not None:
        n = float(nodes)
    elif knodes is not None:
        n = float(knodes) * 1000.0
    if n is None or n <= 0:
        return max(0.1, d)
    node_score = min(1.0, math.log10(max(10.0, n)) / 8.0)
    return max(0.1, 0.5 * d + 0.5 * node_score)

def write_manifest(out_dir: str | Path, manifest: dict[str, Any]) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True))
