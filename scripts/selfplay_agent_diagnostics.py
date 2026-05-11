#!/usr/bin/env python3
"""Generate agent diagnostics and optional failure packets for self-play chunks."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from selfplay_io import read_jsonl, row_key, write_jsonl


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def load_annotations(paths: list[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for path in paths:
        for _, row in read_jsonl(Path(path)):
            source = row.get('source') if isinstance(row.get('source'), dict) else {}
            key = source.get('row_key')
            if isinstance(key, str):
                out[key] = row
    return out


def policy_mass(row: dict) -> float:
    policy = row.get('policy')
    if not isinstance(policy, dict):
        return 0.0
    total = 0.0
    for value in policy.values():
        try:
            total += float(value)
        except Exception:
            return float('nan')
    return total


def selected_uci(row: dict) -> str | None:
    value = row.get('selected_uci', row.get('selected_move'))
    return value if isinstance(value, str) else None


def legal_set(row: dict) -> set[str] | None:
    legal = row.get('legal_uci')
    if isinstance(legal, list) and all(isinstance(x, str) for x in legal):
        return set(legal)
    return None


def annotation_stockfish(annotation: dict | None) -> dict:
    if not annotation:
        return {}
    anns = annotation.get('annotations') if isinstance(annotation.get('annotations'), dict) else {}
    sf = anns.get('stockfish') if isinstance(anns.get('stockfish'), dict) else {}
    return sf


def classify(row: dict, annotation: dict | None, args) -> tuple[list[dict], str]:
    findings: list[dict] = []
    severity = 'low'
    mass = policy_mass(row)
    if not math.isfinite(mass) or mass < args.min_policy_mass or mass > args.max_policy_mass:
        findings.append({'kind': 'schema_violation', 'severity': 'high', 'message': f'policy mass {mass:.8f} outside [{args.min_policy_mass}, {args.max_policy_mass}]'})
        severity = 'high'
    selected = selected_uci(row)
    legal = legal_set(row)
    if selected and legal is not None and selected not in legal:
        findings.append({'kind': 'illegal_move', 'severity': 'critical', 'message': f'selected move {selected} absent from legal_uci'})
        severity = 'critical'
    sf = annotation_stockfish(annotation)
    cp_loss = sf.get('cp_loss')
    if cp_loss is not None:
        try:
            loss = float(cp_loss)
            if loss >= args.critical_cp_loss:
                findings.append({'kind': 'tactical_blunder', 'severity': 'critical', 'message': f'Stockfish cp_loss {loss:.1f} >= {args.critical_cp_loss}', 'cp_loss': loss})
                severity = 'critical'
            elif loss >= args.high_cp_loss:
                findings.append({'kind': 'tactical_blunder', 'severity': 'high', 'message': f'Stockfish cp_loss {loss:.1f} >= {args.high_cp_loss}', 'cp_loss': loss})
                if severity not in {'critical'}:
                    severity = 'high'
            elif loss >= args.medium_cp_loss:
                findings.append({'kind': 'tactical_blunder', 'severity': 'medium', 'message': f'Stockfish cp_loss {loss:.1f} >= {args.medium_cp_loss}', 'cp_loss': loss})
                if severity == 'low':
                    severity = 'medium'
        except Exception:
            findings.append({'kind': 'schema_violation', 'severity': 'medium', 'message': f'nonnumeric cp_loss {cp_loss!r}'})
            if severity == 'low':
                severity = 'medium'
    return findings, severity


def failure_packet(row: dict, annotation: dict | None, finding: dict, created: str, args) -> dict:
    sf = annotation_stockfish(annotation)
    selected = selected_uci(row)
    legal = sorted(legal_set(row) or [])
    packet_id = 'fp_' + sha256_text(f"{row.get('game_id')}:{row.get('ply')}:{finding.get('kind')}:{finding.get('message')}")[:16]
    kind = finding.get('kind') if finding.get('kind') in {'tactical_blunder', 'illegal_move', 'schema_violation'} else 'other'
    return {
        'schema': 'failure_packet_v1',
        'id': packet_id,
        'created_utc': created,
        'kind': kind,
        'severity': finding.get('severity', 'medium'),
        'position': {
            'fen': row.get('fen'),
            'history_fens': row.get('history_fens', []),
            'legal_uci': legal,
        },
        'model': {
            'id': str((row.get('provenance') or {}).get('model_id') or args.model_id),
            'sha256': (row.get('provenance') or {}).get('model_sha256'),
        },
        'backend': {
            'runtime': str((row.get('provenance') or {}).get('generator') or 'selfplay'),
            'target': 'selfplay-agent-diagnostics',
        },
        'search': row.get('search', {'visits': row.get('visits')}),
        'teacher': {'stockfish': sf} if sf else {},
        'observed': {'selected_uci': selected, 'finding': finding},
        'expected': {'selected_uci': sf.get('best_uci')} if sf.get('best_uci') else {},
        'repro': {
            'command': f".venv-onnx/bin/python scripts/selfplay_agent_diagnostics.py --input {args.input} --out /tmp/tl_diag_repro.jsonl --only-row {row_key(row)}"
        },
        'artifacts': [args.input, *args.annotation],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--annotation', action='append', default=[])
    ap.add_argument('--out', required=True)
    ap.add_argument('--failure-dir', default='')
    ap.add_argument('--model-id', default='unknown')
    ap.add_argument('--only-row', default='')
    ap.add_argument('--min-policy-mass', type=float, default=0.98)
    ap.add_argument('--max-policy-mass', type=float, default=1.02)
    ap.add_argument('--medium-cp-loss', type=float, default=80.0)
    ap.add_argument('--high-cp-loss', type=float, default=180.0)
    ap.add_argument('--critical-cp-loss', type=float, default=400.0)
    args = ap.parse_args()

    created = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
    annotations = load_annotations(args.annotation)
    diag_rows: list[dict] = []
    packets: list[dict] = []
    rows_seen = 0
    rows_flagged = 0
    for _, row in read_jsonl(Path(args.input)):
        key = row_key(row)
        if args.only_row and key != args.only_row:
            continue
        rows_seen += 1
        annotation = annotations.get(key)
        findings, severity = classify(row, annotation, args)
        if findings:
            rows_flagged += 1
        diag = {
            'schema': 'selfplay_annotation_v1',
            'source': {'chunk': args.input, 'chunk_sha256': None, 'row_key': key},
            'game_id': row.get('game_id'),
            'ply': row.get('ply'),
            'fen': row.get('fen'),
            'selected_uci': selected_uci(row),
            'legal_uci': sorted(legal_set(row) or []),
            'annotations': {
                'agent': {
                    'severity': severity,
                    'findings': findings,
                    'policy_mass': policy_mass(row),
                    'has_stockfish_annotation': annotation is not None,
                }
            },
            'provenance': {
                'annotator': 'selfplay_agent_diagnostics.py',
                'created_utc': created,
                'version': 'v1',
            },
        }
        diag_rows.append(diag)
        for finding in findings:
            if finding.get('severity') in {'high', 'critical'}:
                packets.append(failure_packet(row, annotation, finding, created, args))
    count = write_jsonl(Path(args.out), diag_rows)
    if args.failure_dir:
        pdir = Path(args.failure_dir)
        pdir.mkdir(parents=True, exist_ok=True)
        for packet in packets:
            (pdir / f"{packet['id']}.json").write_text(json.dumps(packet, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(f'METRIC selfplay_agent_diag_rows={count}')
    print(f'METRIC selfplay_agent_diag_flagged_rows={rows_flagged}')
    print(f'METRIC selfplay_agent_diag_failure_packets={len(packets)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
