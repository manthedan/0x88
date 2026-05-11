#!/usr/bin/env python3
"""Shared JSONL/JSONL.ZST helpers for Tiny Leela self-play pipeline scripts."""
from __future__ import annotations

import gzip
import json
import shutil
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator


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


@contextmanager
def _zstd_cli_writer(path: Path):
    if not shutil.which('zstd'):
        raise SystemExit(f'zstandard python module or zstd CLI is required to write {path}')
    proc = subprocess.Popen(['zstd', '-q', '-T0', '-o', str(path), '-'], stdin=subprocess.PIPE)
    assert proc.stdin is not None
    with proc.stdin, open(proc.stdin.fileno(), mode='w', encoding='utf-8', closefd=False) as fh:
        yield fh
    rc = proc.wait()
    if rc != 0:
        raise SystemExit(f'zstd failed while writing {path} rc={rc}')


def open_text(path: Path, mode: str = 'r'):
    if 'b' in mode:
        raise ValueError('selfplay_io.open_text is text-only')
    reading = mode.startswith('r')
    path = Path(path)
    if not reading:
        path.parent.mkdir(parents=True, exist_ok=True)
    if str(path).endswith('.gz'):
        return gzip.open(path, mode + ('t' if 't' not in mode else ''), encoding='utf-8')
    if str(path).endswith('.zst'):
        try:
            import zstandard as zstd  # type: ignore
            raw = path.open('rb' if reading else 'wb')
            return zstd.open(raw, mode='rt' if reading else 'wt', encoding='utf-8')
        except ImportError:
            return _zstd_cli_reader(path) if reading else _zstd_cli_writer(path)
        except Exception as exc:  # pragma: no cover - optional dependency
            raise SystemExit(f'zstandard failed for {path}: {exc}')
    return path.open(mode, encoding='utf-8')


def read_jsonl(path: Path) -> Iterator[tuple[int, dict]]:
    with open_text(Path(path), 'r') as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception as exc:
                raise SystemExit(f'{path}:{line_no}: JSON parse error: {exc}') from exc
            if not isinstance(row, dict):
                raise SystemExit(f'{path}:{line_no}: expected object row')
            yield line_no, row


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    count = 0
    with open_text(Path(path), 'w') as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True, separators=(',', ':')) + '\n')
            count += 1
    return count


def row_key(row: dict) -> str:
    return f"{row.get('game_id')}:{int(row.get('ply', -1)) if isinstance(row.get('ply'), int) else row.get('ply')}"
