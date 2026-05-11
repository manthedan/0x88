#!/usr/bin/env python3
"""Build training JSONL rows from a self-play pipeline manifest.

This is the first cache-prep join stage: immutable raw self-play chunks are read,
annotation sidecars are joined by source.row_key, and trainer-compatible JSONL rows
are emitted.  Default mode is expanded one-hot rows weighted by post-search policy
probability because the existing cache builders consume one policy move per row.
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from selfplay_io import read_jsonl, row_key, write_jsonl


def normalize_policy(policy: object, *, min_prob: float, max_moves: int) -> list[tuple[str, float]]:
    if not isinstance(policy, dict):
        return []
    items: list[tuple[str, float]] = []
    for move, prob in policy.items():
        try:
            p = float(prob)
        except Exception:
            continue
        if isinstance(move, str) and len(move) in {4, 5} and math.isfinite(p) and p > min_prob:
            items.append((move, p))
    items.sort(key=lambda x: x[1], reverse=True)
    if max_moves > 0:
        items = items[:max_moves]
    mass = sum(p for _, p in items)
    if mass <= 0:
        return []
    return [(move, p / mass) for move, p in items]


def normalize_wdl(wdl: object) -> list[float] | None:
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


def q_to_wdl(q: object) -> list[float] | None:
    try:
        x = max(-1.0, min(1.0, float(q)))
    except Exception:
        return None
    return [max(x, 0.0), 1.0 - abs(x), max(-x, 0.0)]


def row_wdl(row: dict, *, value_target: str, root_mix: float) -> list[float]:
    result = normalize_wdl(row.get('wdl')) or normalize_wdl(row.get('result'))
    root = normalize_wdl(row.get('root_wdl')) or q_to_wdl(row.get('q', row.get('root_value')))
    if value_target == 'result':
        if result is None:
            raise ValueError('missing result/wdl target')
        return result
    if value_target == 'root':
        if root is None:
            raise ValueError('missing root target')
        return root
    if result is None or root is None:
        raise ValueError('mix target requires result/wdl and root/q')
    a = max(0.0, min(1.0, root_mix))
    mixed = [(1.0 - a) * result[i] + a * root[i] for i in range(3)]
    mass = sum(mixed) or 1.0
    return [v / mass for v in mixed]


def source_id(row: dict, fallback_idx: int) -> str:
    gid = str(row.get('game_id') or f'row{fallback_idx:08d}')
    ply = row.get('ply')
    return f'{gid}_p{int(ply):04d}' if isinstance(ply, int) else gid


def round10(x: float) -> float:
    return float(f'{x:.10f}')


def load_annotations(paths: Iterable[str]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for path in paths:
        for _, row in read_jsonl(Path(path)):
            source = row.get('source') if isinstance(row.get('source'), dict) else {}
            key = source.get('row_key')
            if isinstance(key, str):
                out.setdefault(key, []).append(row)
    return out


def annotation_summary(rows: list[dict]) -> dict:
    summary: dict = {}
    for ann in rows:
        annotations = ann.get('annotations') if isinstance(ann.get('annotations'), dict) else {}
        if isinstance(annotations.get('stockfish'), dict):
            sf = annotations['stockfish']
            summary['stockfish_best_uci'] = sf.get('best_uci')
            summary['stockfish_before_cp'] = sf.get('before_cp')
            summary['stockfish_after_cp'] = sf.get('after_cp')
            summary['stockfish_cp_loss'] = sf.get('cp_loss')
            summary['stockfish_depth'] = sf.get('depth')
            summary['stockfish_nodes'] = sf.get('nodes')
        if isinstance(annotations.get('agent'), dict):
            agent = annotations['agent']
            summary['agent_severity'] = agent.get('severity')
            summary['agent_findings'] = agent.get('findings', [])
            summary['agent_policy_mass'] = agent.get('policy_mass')
    return summary


def severity_rank(sev: object) -> int:
    return {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}.get(str(sev), 0)


def should_skip(summary: dict, args: argparse.Namespace) -> str | None:
    if args.skip_agent_severity and severity_rank(summary.get('agent_severity')) >= severity_rank(args.skip_agent_severity):
        return f'agent severity {summary.get("agent_severity")}'
    cp_loss = summary.get('stockfish_cp_loss')
    if args.max_cp_loss is not None and cp_loss is not None:
        try:
            if float(cp_loss) > args.max_cp_loss:
                return f'cp_loss {cp_loss} > {args.max_cp_loss}'
        except Exception:
            return f'invalid cp_loss {cp_loss!r}'
    return None


def chunk_paths_from_manifest(manifest: dict) -> list[str]:
    return [c['path'] for c in manifest.get('chunks', []) if isinstance(c, dict) and isinstance(c.get('path'), str)]


def annotation_paths_from_manifest(manifest: dict) -> list[str]:
    return [a['path'] for a in manifest.get('annotations', []) if isinstance(a, dict) and isinstance(a.get('path'), str)]


def iter_training_rows(args: argparse.Namespace, manifest: dict):
    annotation_paths = args.annotation or annotation_paths_from_manifest(manifest)
    annotations = load_annotations(annotation_paths)
    emitted = 0
    seen = 0
    skipped = 0
    expanded_weight_sum = 0.0
    stats = {'rows_seen': 0, 'rows_emitted': 0, 'rows_skipped': 0, 'expanded_weight_sum': 0.0}
    chunk_paths = args.chunk or chunk_paths_from_manifest(manifest)
    for chunk_path in chunk_paths:
        for _, row in read_jsonl(Path(chunk_path)):
            seen += 1
            stats['rows_seen'] = seen
            sid = source_id(row, seen)
            key = row_key(row)
            anns = annotations.get(key, [])
            summary = annotation_summary(anns)
            skip_reason = should_skip(summary, args)
            if skip_reason:
                skipped += 1
                stats['rows_skipped'] = skipped
                continue
            policy = normalize_policy(row.get('policy'), min_prob=args.min_prob, max_moves=args.max_policy_moves)
            if not policy:
                skipped += 1
                stats['rows_skipped'] = skipped
                continue
            try:
                wdl = [round10(x) for x in row_wdl(row, value_target=args.value_target, root_mix=args.root_mix)]
            except ValueError:
                skipped += 1
                stats['rows_skipped'] = skipped
                continue
            common = {
                'fen': row.get('fen'),
                'history_fens': row.get('history_fens', []),
                'wdl': wdl,
                'result': wdl,
                'source': args.lane,
                'source_id': sid,
                'source_schema': row.get('schema'),
                'source_chunk': chunk_path,
                'source_row_key': key,
                'game_id': row.get('game_id'),
                'ply': row.get('ply'),
                'selected_uci': row.get('selected_uci', row.get('selected_move')),
                'search': row.get('search', {'visits': row.get('visits')}),
                'annotations': summary,
                'created_utc': args.created_utc,
            }
            if args.source_model:
                common['source_model'] = args.source_model
            if args.mode == 'soft':
                soft = {move: round10(prob) for move, prob in policy}
                emitted += 1
                expanded_weight_sum += 1.0
                stats['rows_emitted'] = emitted
                stats['expanded_weight_sum'] = expanded_weight_sum
                yield {**common, 'schema': f'tiny_leela_{args.lane}_training_v1', 'id': sid, 'policy': soft, 'weight': 1.0}
            else:
                for rank, (move, prob) in enumerate(policy, 1):
                    p = round10(prob)
                    emitted += 1
                    expanded_weight_sum += p
                    stats['rows_emitted'] = emitted
                    stats['expanded_weight_sum'] = expanded_weight_sum
                    yield {
                        **common,
                        'schema': f'tiny_leela_{args.lane}_training_v1',
                        'id': f'{sid}_m{rank:02d}',
                        'policy': {move: 1.0},
                        'policy_rank': rank,
                        'policy_prob': p,
                        'weight': p,
                    }
    args._stats = stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--manifest-out', default='')
    ap.add_argument('--chunk', action='append', default=[], help='Override/add raw chunks instead of manifest chunks')
    ap.add_argument('--annotation', action='append', default=[], help='Override/add annotation sidecars instead of manifest annotations')
    ap.add_argument('--lane', choices=['supervised_sp', 'zero'], default='supervised_sp')
    ap.add_argument('--source-model', default='')
    ap.add_argument('--mode', choices=['expanded', 'soft'], default='expanded')
    ap.add_argument('--value-target', choices=['result', 'root', 'mix'], default='result')
    ap.add_argument('--root-mix', type=float, default=0.25)
    ap.add_argument('--min-prob', type=float, default=0.0)
    ap.add_argument('--max-policy-moves', type=int, default=0)
    ap.add_argument('--skip-agent-severity', choices=['medium', 'high', 'critical'], default='')
    ap.add_argument('--max-cp-loss', type=float, default=None)
    args = ap.parse_args()
    if args.lane == 'zero' and args.source_model:
        raise SystemExit('Refusing source-model provenance in zero lane; use --lane supervised_sp')
    args.created_utc = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
    manifest = json.loads(Path(args.manifest).read_text(encoding='utf-8'))
    rows = list(iter_training_rows(args, manifest))
    written = write_jsonl(Path(args.output), rows)
    stats = getattr(args, '_stats', {'rows_seen': 0, 'rows_emitted': written, 'rows_skipped': 0, 'expanded_weight_sum': 0.0})
    if args.manifest_out:
        out_manifest = {
            'schema': 'selfplay_training_rows_manifest_v1',
            'created_utc': args.created_utc,
            'pipeline_manifest': args.manifest,
            'output': args.output,
            'lane': args.lane,
            'mode': args.mode,
            'value_target': args.value_target,
            'source_model': args.source_model or None,
            **stats,
        }
        Path(args.manifest_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.manifest_out).write_text(json.dumps(out_manifest, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(f'METRIC selfplay_manifest_training_rows_seen={stats["rows_seen"]}')
    print(f'METRIC selfplay_manifest_training_rows_emitted={written}')
    print(f'METRIC selfplay_manifest_training_rows_skipped={stats["rows_skipped"]}')
    print(f'METRIC selfplay_manifest_training_weight_sum={stats["expanded_weight_sum"]:.10f}')
    return 0 if written > 0 else 2


if __name__ == '__main__':
    raise SystemExit(main())
