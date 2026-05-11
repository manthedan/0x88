#!/usr/bin/env python3
"""Validate Tiny Leela self-play JSONL/JSONL.ZST chunks.

This validator accepts the current `scripts/selfplay_generate.mjs` row format and
is intentionally strict about silent data-corruption hazards: invalid FEN shape,
bad probability mass, missing result labels, duplicate game/ply rows, and empty
chunks fail by default.
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
from pathlib import Path
from typing import Iterable, TextIO


@contextmanager
def _zstd_cli_reader(path: Path):
    if not shutil.which('zstd'):
        raise SystemExit(f'zstandard python module or zstd CLI is required to read {path}')
    proc = subprocess.Popen(['zstd', '-dc', str(path)], stdout=subprocess.PIPE)
    assert proc.stdout is not None
    with proc.stdout, open(proc.stdout.fileno(), mode='r', encoding='utf-8', closefd=False) as fh:
        yield fh
    rc = proc.wait()
    if rc != 0:
        raise SystemExit(f'zstd failed while reading {path} rc={rc}')


def _open_text(path: Path):
    if str(path).endswith('.gz'):
        return gzip.open(path, 'rt', encoding='utf-8')
    if str(path).endswith('.zst'):
        try:
            import zstandard as zstd  # type: ignore
            fh = path.open('rb')
            return zstd.open(fh, mode='rt', encoding='utf-8')  # type: ignore[return-value]
        except ImportError:
            return _zstd_cli_reader(path)
        except Exception as exc:  # pragma: no cover - depends on env
            raise SystemExit(f'zstandard failed to read {path}: {exc}')
    return path.open('r', encoding='utf-8')


def _fen_plausible(fen: str) -> bool:
    parts = fen.split()
    if len(parts) < 4 or parts[1] not in {'w', 'b'}:
        return False
    ranks = parts[0].split('/')
    if len(ranks) != 8:
        return False
    pieces = set('pnbrqkPNBRQK')
    for rank in ranks:
        total = 0
        for ch in rank:
            if ch.isdigit():
                total += int(ch)
            elif ch in pieces:
                total += 1
            else:
                return False
        if total != 8:
            return False
    return True


def _is_wdl(x) -> bool:
    if not isinstance(x, list) or len(x) != 3:
        return False
    try:
        vals = [float(v) for v in x]
    except Exception:
        return False
    return all(math.isfinite(v) and v >= 0 for v in vals) and abs(sum(vals) - 1.0) <= 1e-5


def _validate_policy(policy, line_no: int, *, min_policy_mass: float, max_policy_mass: float, legal_uci: set[str] | None = None) -> list[str]:
    errors: list[str] = []
    if not isinstance(policy, dict) or not policy:
        return [f'line {line_no}: empty/non-dict policy']
    mass = 0.0
    for move, prob in policy.items():
        try:
            p = float(prob)
        except Exception:
            errors.append(f'line {line_no}: nonnumeric policy prob for {move}')
            continue
        if not isinstance(move, str) or len(move) not in {4, 5}:
            errors.append(f'line {line_no}: invalid UCI move key {move!r}')
        elif legal_uci is not None and move not in legal_uci:
            errors.append(f'line {line_no}: policy move {move} absent from legal_uci')
        if not math.isfinite(p) or p < 0:
            errors.append(f'line {line_no}: invalid policy prob for {move}')
        mass += p
    if not (min_policy_mass <= mass <= max_policy_mass):
        errors.append(f'line {line_no}: policy mass {mass:.8f} outside [{min_policy_mass}, {max_policy_mass}]')
    return errors


def _validate_common(row: dict, line_no: int) -> list[str]:
    errors: list[str] = []
    if not isinstance(row.get('game_id'), str) or not row.get('game_id'):
        errors.append(f'line {line_no}: invalid game_id')
    if not isinstance(row.get('ply'), int) or row.get('ply') < 0:
        errors.append(f'line {line_no}: invalid ply')
    if not isinstance(row.get('fen'), str) or not _fen_plausible(row.get('fen', '')):
        errors.append(f'line {line_no}: invalid fen')
    return errors


def validate_row(row: dict, line_no: int, *, min_policy_mass: float, max_policy_mass: float) -> list[str]:
    schema = row.get('schema')
    if schema == 'selfplay_chunk_v1':
        return validate_selfplay_chunk_v1(row, line_no, min_policy_mass=min_policy_mass, max_policy_mass=max_policy_mass)
    return validate_legacy_row(row, line_no, min_policy_mass=min_policy_mass, max_policy_mass=max_policy_mass)


def validate_selfplay_chunk_v1(row: dict, line_no: int, *, min_policy_mass: float, max_policy_mass: float) -> list[str]:
    errors: list[str] = []
    for key in ['schema', 'lane', 'game_id', 'ply', 'fen', 'legal_uci', 'selected_uci', 'policy', 'wdl', 'provenance']:
        if key not in row:
            errors.append(f'line {line_no}: missing {key}')
    if errors:
        return errors
    errors.extend(_validate_common(row, line_no))
    if row.get('lane') not in {'gumbel_zero', 'sup_sp', 'eval_demo', 'other'}:
        errors.append(f'line {line_no}: invalid lane')
    legal = row.get('legal_uci')
    if not isinstance(legal, list) or not legal or not all(isinstance(m, str) and len(m) in {4, 5} for m in legal):
        errors.append(f'line {line_no}: invalid legal_uci')
        legal_set: set[str] | None = None
    else:
        legal_set = set(legal)
    selected = row.get('selected_uci')
    if not isinstance(selected, str) or len(selected) not in {4, 5}:
        errors.append(f'line {line_no}: invalid selected_uci')
    elif legal_set is not None and selected not in legal_set:
        errors.append(f'line {line_no}: selected_uci {selected} absent from legal_uci')
    if not _is_wdl(row.get('wdl')):
        errors.append(f'line {line_no}: invalid wdl')
    if 'q' in row:
        try:
            q = float(row['q'])
            if not math.isfinite(q) or q < -1.0001 or q > 1.0001:
                errors.append(f'line {line_no}: q outside [-1,1]')
        except Exception:
            errors.append(f'line {line_no}: invalid q')
    provenance = row.get('provenance')
    if not isinstance(provenance, dict):
        errors.append(f'line {line_no}: invalid provenance')
    else:
        if not isinstance(provenance.get('generator'), str) or not provenance.get('generator'):
            errors.append(f'line {line_no}: missing provenance.generator')
        if 'seed' not in provenance or not isinstance(provenance.get('seed'), (int, str)):
            errors.append(f'line {line_no}: missing provenance.seed')
    errors.extend(_validate_policy(row.get('policy'), line_no, min_policy_mass=min_policy_mass, max_policy_mass=max_policy_mass, legal_uci=legal_set))
    return errors


def validate_legacy_row(row: dict, line_no: int, *, min_policy_mass: float, max_policy_mass: float) -> list[str]:
    errors: list[str] = []
    for key in ['game_id', 'ply', 'fen', 'turn', 'policy', 'result']:
        if key not in row:
            errors.append(f'line {line_no}: missing {key}')
    if errors:
        return errors
    errors.extend(_validate_common(row, line_no))
    if row['turn'] not in {'w', 'b'}:
        errors.append(f'line {line_no}: invalid turn')
    if not _is_wdl(row['result']):
        errors.append(f'line {line_no}: invalid result WDL')
    errors.extend(_validate_policy(row.get('policy'), line_no, min_policy_mass=min_policy_mass, max_policy_mass=max_policy_mass))
    if 'white_score' in row:
        try:
            ws = float(row['white_score'])
            if ws not in {0.0, 0.5, 1.0}:
                errors.append(f'line {line_no}: white_score must be 0/0.5/1')
        except Exception:
            errors.append(f'line {line_no}: invalid white_score')
    if 'root_value' in row:
        try:
            rv = float(row['root_value'])
            if not math.isfinite(rv) or rv < -1.0001 or rv > 1.0001:
                errors.append(f'line {line_no}: root_value outside [-1,1]')
        except Exception:
            errors.append(f'line {line_no}: invalid root_value')
    return errors


def validate_file(path: Path, args) -> dict:
    rows = 0
    games: set[str] = set()
    seen: set[tuple[str, int]] = set()
    errors: list[str] = []
    with _open_text(path) as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception as exc:
                errors.append(f'line {line_no}: JSON parse error: {exc}')
                continue
            rows += 1
            key = (str(row.get('game_id')), int(row.get('ply', -1)) if isinstance(row.get('ply'), int) else -1)
            if key in seen:
                errors.append(f'line {line_no}: duplicate game_id/ply {key}')
            seen.add(key)
            if isinstance(row.get('game_id'), str):
                games.add(row['game_id'])
            errors.extend(validate_row(row, line_no, min_policy_mass=args.min_policy_mass, max_policy_mass=args.max_policy_mass))
            if len(errors) >= args.max_errors:
                break
    if rows < args.min_rows:
        errors.append(f'rows {rows} below min_rows {args.min_rows}')
    return {'path': str(path), 'rows': rows, 'games': len(games), 'errors': errors}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('chunks', nargs='+')
    ap.add_argument('--min-rows', type=int, default=1)
    ap.add_argument('--min-policy-mass', type=float, default=0.98)
    ap.add_argument('--max-policy-mass', type=float, default=1.02)
    ap.add_argument('--max-errors', type=int, default=20)
    ap.add_argument('--json', action='store_true')
    args = ap.parse_args()
    reports = [validate_file(Path(p), args) for p in args.chunks]
    ok = all(not r['errors'] for r in reports)
    if args.json:
        print(json.dumps({'ok': ok, 'chunks': reports}, indent=2))
    else:
        for r in reports:
            print(f"chunk={r['path']} rows={r['rows']} games={r['games']} errors={len(r['errors'])}")
            for err in r['errors'][:args.max_errors]:
                print(f'ERROR {err}', file=sys.stderr)
        print(f'METRIC selfplay_chunks={len(reports)}')
        print(f'METRIC selfplay_rows={sum(int(r["rows"]) for r in reports)}')
        print(f'METRIC selfplay_validate_ok={1 if ok else 0}')
    return 0 if ok else 2


if __name__ == '__main__':
    raise SystemExit(main())
