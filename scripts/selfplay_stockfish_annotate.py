#!/usr/bin/env python3
"""Annotate self-play chunks with Stockfish sidecar rows.

This is intentionally a sidecar writer: it never rewrites raw self-play chunks.
Use --mock-stockfish for contract tests and dry pipeline checks without a UCI binary.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from selfplay_io import read_jsonl, row_key, write_jsonl


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


class UciStockfish:
    def __init__(self, command: str, *, depth: int, nodes: int | None, timeout: float):
        exe = command if Path(command).exists() else shutil.which(command)
        if not exe:
            raise SystemExit(f'Stockfish binary not found: {command}')
        self.depth = depth
        self.nodes = nodes
        self.timeout = timeout
        self.proc = subprocess.Popen([exe], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
        assert self.proc.stdin is not None and self.proc.stdout is not None
        self._cmd('uci')
        self._wait_for('uciok')
        self._cmd('isready')
        self._wait_for('readyok')

    def close(self) -> None:
        if self.proc.poll() is None:
            try:
                self._cmd('quit')
            except Exception:
                pass
            try:
                self.proc.wait(timeout=1)
            except Exception:
                self.proc.kill()

    def _cmd(self, cmd: str) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(cmd + '\n')
        self.proc.stdin.flush()

    def _wait_for(self, token: str) -> None:
        assert self.proc.stdout is not None
        import time
        deadline = time.time() + self.timeout
        while time.time() < deadline:
            line = self.proc.stdout.readline()
            if token in line:
                return
            if self.proc.poll() is not None:
                raise RuntimeError(f'Stockfish exited while waiting for {token}')
        raise TimeoutError(f'timeout waiting for {token}')

    def evaluate(self, fen: str, *, moves: list[str] | None = None) -> dict:
        assert self.proc.stdout is not None
        if moves:
            self._cmd(f'position fen {fen} moves {" ".join(moves)}')
        else:
            self._cmd(f'position fen {fen}')
        go = f'go nodes {self.nodes}' if self.nodes else f'go depth {self.depth}'
        self._cmd(go)
        best = None
        cp = None
        mate = None
        depth_seen = None
        pv: list[str] = []
        while True:
            line = self.proc.stdout.readline().strip()
            if not line:
                if self.proc.poll() is not None:
                    raise RuntimeError('Stockfish exited during search')
                continue
            parts = line.split()
            if parts[:1] == ['bestmove']:
                best = parts[1] if len(parts) > 1 else None
                break
            if parts[:1] == ['info']:
                if 'depth' in parts:
                    try: depth_seen = int(parts[parts.index('depth') + 1])
                    except Exception: pass
                if 'score' in parts:
                    idx = parts.index('score')
                    if idx + 2 < len(parts):
                        try:
                            if parts[idx + 1] == 'cp': cp = int(parts[idx + 2])
                            elif parts[idx + 1] == 'mate': mate = int(parts[idx + 2])
                        except Exception:
                            pass
                if 'pv' in parts:
                    pv = parts[parts.index('pv') + 1:]
        if cp is None and mate is not None:
            cp = 100000 if mate > 0 else -100000
        return {'cp': cp, 'mate': mate, 'depth': depth_seen, 'best_uci': best, 'pv': pv}


def selected_uci(row: dict) -> str | None:
    value = row.get('selected_uci', row.get('selected_move'))
    return value if isinstance(value, str) else None


def legal_uci(row: dict) -> list[str]:
    legal = row.get('legal_uci')
    if isinstance(legal, list) and all(isinstance(x, str) for x in legal):
        return legal
    policy = row.get('policy')
    if isinstance(policy, dict):
        return [m for m in policy if isinstance(m, str)]
    return []


def mock_eval(row: dict, selected: str | None) -> dict:
    policy = row.get('policy') if isinstance(row.get('policy'), dict) else {}
    top = max(policy.items(), key=lambda kv: float(kv[1]))[0] if policy else selected
    before = int(round(100 * float(row.get('q', row.get('root_value', 0)) or 0)))
    penalty = 0 if selected == top else 35
    return {
        'available': False,
        'depth': None,
        'nodes': None,
        'before_cp': before,
        'after_cp': -before + penalty,
        'cp_loss': penalty,
        'best_uci': top,
        'pv': [top] if top else [],
        'mock': True,
    }


def annotate_row(row: dict, *, chunk: Path, chunk_sha: str, engine: UciStockfish | None, args, created: str) -> dict:
    selected = selected_uci(row)
    if args.mock_stockfish:
        sf = mock_eval(row, selected)
    elif selected and engine is not None:
        before = engine.evaluate(row['fen'])
        after = engine.evaluate(row['fen'], moves=[selected])
        before_cp = before.get('cp')
        after_cp = after.get('cp')
        cp_loss = None
        if before_cp is not None and after_cp is not None:
            # UCI scores are side-to-move relative; after the selected move the side-to-move flips.
            cp_loss = float(before_cp) + float(after_cp)
        sf = {
            'available': True,
            'depth': before.get('depth'),
            'nodes': args.nodes,
            'before_cp': before_cp,
            'after_cp': after_cp,
            'cp_loss': cp_loss,
            'best_uci': before.get('best_uci'),
            'pv': before.get('pv') or [],
        }
    else:
        sf = {'available': False, 'depth': args.depth, 'nodes': args.nodes, 'before_cp': None, 'after_cp': None, 'cp_loss': None, 'best_uci': None, 'pv': []}
    return {
        'schema': 'selfplay_annotation_v1',
        'source': {'chunk': str(chunk), 'chunk_sha256': chunk_sha, 'row_key': row_key(row)},
        'game_id': row.get('game_id'),
        'ply': row.get('ply'),
        'fen': row.get('fen'),
        'selected_uci': selected,
        'legal_uci': legal_uci(row),
        'annotations': {'stockfish': sf},
        'provenance': {
            'annotator': 'selfplay_stockfish_annotate.py',
            'created_utc': created,
            'version': 'v1',
            'stockfish': args.stockfish,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--stockfish', default='.local_engines/stockfish_pkg/usr/games/stockfish')
    ap.add_argument('--depth', type=int, default=8)
    ap.add_argument('--nodes', type=int, default=0, help='If >0, use fixed nodes instead of depth')
    ap.add_argument('--timeout', type=float, default=20.0)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--mock-stockfish', action='store_true')
    args = ap.parse_args()

    chunk = Path(args.input)
    created = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
    chunk_sha = sha256_file(chunk)
    engine = None
    if not args.mock_stockfish:
        engine = UciStockfish(args.stockfish, depth=args.depth, nodes=args.nodes or None, timeout=args.timeout)
    rows = []
    try:
        for idx, (_, row) in enumerate(read_jsonl(chunk), 1):
            if args.limit and idx > args.limit:
                break
            rows.append(annotate_row(row, chunk=chunk, chunk_sha=chunk_sha, engine=engine, args=args, created=created))
    finally:
        if engine is not None:
            engine.close()
    count = write_jsonl(Path(args.out), rows)
    print(f'METRIC selfplay_stockfish_annotation_rows={count}')
    print(f'METRIC selfplay_stockfish_annotation_path={args.out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
