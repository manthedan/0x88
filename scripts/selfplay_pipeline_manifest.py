#!/usr/bin/env python3
"""Write a manifest that joins raw self-play chunks, annotation sidecars, diagnostics, and cache targets."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from selfplay_io import read_jsonl


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def collect_chunk(path: Path) -> dict:
    rows = 0
    games: set[str] = set()
    keys: set[str] = set()
    lanes: set[str] = set()
    for _, row in read_jsonl(path):
        rows += 1
        gid = row.get('game_id')
        if isinstance(gid, str):
            games.add(gid)
        if isinstance(row.get('ply'), int):
            keys.add(f'{gid}:{row["ply"]}')
        if isinstance(row.get('lane'), str):
            lanes.add(row['lane'])
    return {'path': str(path), 'sha256': sha256_file(path), 'rows': rows, 'games': len(games), 'lanes': sorted(lanes), 'row_keys': keys}


def collect_annotation(path: Path) -> dict:
    rows = 0
    keys: set[str] = set()
    annotators: set[str] = set()
    for _, row in read_jsonl(path):
        rows += 1
        source = row.get('source') if isinstance(row.get('source'), dict) else {}
        key = source.get('row_key')
        if isinstance(key, str):
            keys.add(key)
        prov = row.get('provenance') if isinstance(row.get('provenance'), dict) else {}
        if isinstance(prov.get('annotator'), str):
            annotators.add(prov['annotator'])
    return {'path': str(path), 'sha256': sha256_file(path), 'rows': rows, 'annotators': sorted(annotators), 'row_keys': keys}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--chunk', action='append', required=True)
    ap.add_argument('--annotation', action='append', default=[])
    ap.add_argument('--out', required=True)
    ap.add_argument('--cache-root', default='data/selfplay_caches')
    ap.add_argument('--target', action='append', default=['residual', 'squareformer'])
    ap.add_argument('--strict-annotations', action='store_true')
    args = ap.parse_args()

    chunks = [collect_chunk(Path(p)) for p in args.chunk]
    annotations = [collect_annotation(Path(p)) for p in args.annotation]
    chunk_keys = set().union(*(c['row_keys'] for c in chunks)) if chunks else set()
    issues: list[str] = []
    for ann in annotations:
        missing = ann['row_keys'] - chunk_keys
        if missing:
            issues.append(f"annotation {ann['path']} has {len(missing)} keys absent from chunks")
        if args.strict_annotations:
            uncovered = chunk_keys - ann['row_keys']
            if uncovered:
                issues.append(f"annotation {ann['path']} missing {len(uncovered)} chunk keys")
    cache_targets = []
    for target in args.target:
        cache_targets.append({
            'target': target,
            'path': str(Path(args.cache_root) / target),
            'status': 'planned',
            'notes': 'Build from raw chunks joined with annotation sidecars; do not mutate raw self-play rows.',
        })
    manifest = {
        'schema': 'selfplay_pipeline_manifest_v1',
        'created_utc': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
        'chunks': [{k: v for k, v in c.items() if k != 'row_keys'} for c in chunks],
        'annotations': [{k: v for k, v in a.items() if k != 'row_keys'} for a in annotations],
        'totals': {
            'rows': sum(c['rows'] for c in chunks),
            'games': sum(c['games'] for c in chunks),
            'unique_row_keys': len(chunk_keys),
        },
        'cache_targets': cache_targets,
        'issues': issues,
        'ok': not issues,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(f'METRIC selfplay_pipeline_manifest_ok={1 if manifest["ok"] else 0}')
    print(f'METRIC selfplay_pipeline_manifest_rows={manifest["totals"]["rows"]}')
    print(f'METRIC selfplay_pipeline_manifest_path={out}')
    return 0 if manifest['ok'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
