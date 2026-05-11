#!/usr/bin/env python3
"""Validate Tiny Leela self-play annotation sidecars."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from selfplay_io import read_jsonl


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


def validate_row(row: dict, line_no: int) -> list[str]:
    errors: list[str] = []
    for key in ['schema', 'source', 'game_id', 'ply', 'fen', 'annotations', 'provenance']:
        if key not in row:
            errors.append(f'line {line_no}: missing {key}')
    if errors:
        return errors
    if row.get('schema') != 'selfplay_annotation_v1':
        errors.append(f'line {line_no}: invalid schema')
    if not isinstance(row.get('game_id'), str) or not row.get('game_id'):
        errors.append(f'line {line_no}: invalid game_id')
    if not isinstance(row.get('ply'), int) or row.get('ply') < 0:
        errors.append(f'line {line_no}: invalid ply')
    if not isinstance(row.get('fen'), str) or not _fen_plausible(row.get('fen', '')):
        errors.append(f'line {line_no}: invalid fen')
    source = row.get('source')
    if not isinstance(source, dict) or not isinstance(source.get('row_key'), str) or not source.get('row_key'):
        errors.append(f'line {line_no}: invalid source.row_key')
    annotations = row.get('annotations')
    if not isinstance(annotations, dict):
        errors.append(f'line {line_no}: invalid annotations')
    else:
        sf = annotations.get('stockfish')
        if sf is not None:
            if not isinstance(sf, dict):
                errors.append(f'line {line_no}: invalid annotations.stockfish')
            else:
                if 'available' in sf and not isinstance(sf['available'], bool):
                    errors.append(f'line {line_no}: stockfish.available must be bool')
                for key in ['before_cp', 'after_cp', 'cp_loss']:
                    value = sf.get(key)
                    if value is not None:
                        try:
                            v = float(value)
                            if not math.isfinite(v):
                                errors.append(f'line {line_no}: stockfish.{key} nonfinite')
                        except Exception:
                            errors.append(f'line {line_no}: stockfish.{key} nonnumeric')
    prov = row.get('provenance')
    if not isinstance(prov, dict) or not isinstance(prov.get('annotator'), str) or not prov.get('annotator') or not isinstance(prov.get('created_utc'), str):
        errors.append(f'line {line_no}: invalid provenance')
    return errors


def validate_file(path: Path, args) -> dict:
    rows = 0
    seen: set[str] = set()
    errors: list[str] = []
    for line_no, row in read_jsonl(path):
        rows += 1
        key = str((row.get('source') or {}).get('row_key') or f'{row.get("game_id")}:{row.get("ply")}')
        if key in seen:
            errors.append(f'line {line_no}: duplicate source row {key}')
        seen.add(key)
        errors.extend(validate_row(row, line_no))
        if len(errors) >= args.max_errors:
            break
    if rows < args.min_rows:
        errors.append(f'rows {rows} below min_rows {args.min_rows}')
    return {'path': str(path), 'rows': rows, 'errors': errors}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('annotations', nargs='+')
    ap.add_argument('--min-rows', type=int, default=1)
    ap.add_argument('--max-errors', type=int, default=20)
    ap.add_argument('--json', action='store_true')
    args = ap.parse_args()
    reports = [validate_file(Path(p), args) for p in args.annotations]
    ok = all(not r['errors'] for r in reports)
    if args.json:
        print(json.dumps({'ok': ok, 'annotations': reports}, indent=2))
    else:
        for report in reports:
            print(f"annotation={report['path']} rows={report['rows']} errors={len(report['errors'])}")
            for err in report['errors'][:args.max_errors]:
                print(f'ERROR {err}', file=sys.stderr)
        print(f'METRIC selfplay_annotation_files={len(reports)}')
        print(f'METRIC selfplay_annotation_rows={sum(int(r["rows"]) for r in reports)}')
        print(f'METRIC selfplay_annotation_validate_ok={1 if ok else 0}')
    return 0 if ok else 2


if __name__ == '__main__':
    raise SystemExit(main())
