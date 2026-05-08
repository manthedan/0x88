#!/usr/bin/env python3
from __future__ import annotations
import argparse, contextlib, hashlib, json, re, subprocess, time
from collections import Counter
from pathlib import Path

FILES = 'abcdefgh'
PIECE_VAL = {'p':1,'n':3,'b':3,'r':5,'q':9}

@contextlib.contextmanager
def opener(path: str | Path):
    path = str(path)
    if path.endswith('.zst'):
        p = subprocess.Popen(['zstd', '-dc', path], stdout=subprocess.PIPE, text=True)
        try:
            assert p.stdout is not None
            yield p.stdout
        finally:
            if p.stdout: p.stdout.close()
            rc = p.wait()
            if rc and rc != -13:
                raise subprocess.CalledProcessError(rc, ['zstd', '-dc', path])
    else:
        with open(path, 'rt', encoding='utf-8') as f:
            yield f

def write_zst_jsonl(rows, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + '.tmp') if out.suffix != '.zst' else out.with_suffix('.jsonl.tmp')
    with tmp.open('wt', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, separators=(',', ':')) + '\n')
    if str(out).endswith('.zst'):
        subprocess.check_call(['zstd', '-q', '-f', '-T0', str(tmp), '-o', str(out)])
        tmp.unlink()
    else:
        tmp.replace(out)

def fen_key(fen: str) -> str:
    return ' '.join(str(fen).split()[:4])

def sha(s: str) -> str:
    return hashlib.sha256(s.encode('utf-8')).hexdigest()

def ply(row: dict) -> int:
    if 'ply' in row:
        try: return int(row['ply'])
        except Exception: pass
    m = re.search(r'_([0-9]+)$', str(row.get('id','')))
    return int(m.group(1)) if m else 0

def phase(row: dict) -> str:
    p = ply(row)
    board = str(row.get('fen','')).split()[0]
    pieces = sum(1 for c in board if c.isalpha())
    if p < 20: return 'opening'
    if pieces <= 12: return 'endgame'
    return 'middlegame'

def material_bucket(row: dict) -> str:
    board = str(row.get('fen','')).split()[0]
    w = b = 0
    for c in board:
        if c.isalpha():
            if c.isupper(): w += PIECE_VAL.get(c.lower(),0)
            else: b += PIECE_VAL.get(c,0)
    d = w - b
    if abs(d) <= 1: return 'equal'
    if abs(d) <= 3: return 'small_edge'
    if abs(d) <= 8: return 'large_edge'
    return 'decisive_material'

def game_id(row: dict) -> str:
    rid = str(row.get('id',''))
    m = re.match(r'(.+)_([0-9]+)$', rid)
    return m.group(1) if m else rid

def played_move(row: dict) -> str:
    pol = row.get('policy') or {}
    if not pol: return ''
    return max(pol.items(), key=lambda kv: kv[1])[0]

def opening_key(row: dict) -> str:
    if row.get('eco'): return 'eco:' + str(row['eco'])
    if row.get('opening'): return 'opening:' + str(row['opening'])[:80]
    # Crude but deterministic proxy until ECO tagging exists.
    return 'fen4:' + sha(fen_key(row.get('fen','')))[:16]

def legal_count_hint(fen: str) -> int:
    # Placeholder until movegen integration. Useful as schema stability, not a true count.
    return -1

def registry_row(row: dict, shard: str, row_idx: int) -> dict | None:
    fen = row.get('fen')
    if not fen: return None
    fkey = fen_key(fen)
    hkeys = [fen_key(x) for x in (row.get('history_fens') or [])[:2] if x]
    return {
        'position_key': 'sha256:' + sha(fkey),
        'history_key': 'sha256:' + sha(fkey + '|' + '|'.join(hkeys)),
        'fen': fen,
        'fen_key': fkey,
        'history_fens': (row.get('history_fens') or [])[:2],
        'played': played_move(row),
        'wdl': row.get('wdl'),
        'source': row.get('source', row.get('teacher', 'unknown')),
        'dataset_shard': shard,
        'row': row_idx,
        'id': row.get('id'),
        'ply': ply(row),
        'phase': phase(row),
        'material_bucket': material_bucket(row),
        'opening_key': opening_key(row),
        'legal_count': legal_count_hint(fen),
        'duplicate_count_observed': 1,
    }

def main() -> int:
    ap = argparse.ArgumentParser(description='Build a deduped position registry/shard from a supervised dataset manifest.')
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--report-out', default='')
    ap.add_argument('--split', choices=['train','dev'], default='train')
    ap.add_argument('--max-rows', type=int, default=0, help='Maximum source rows to scan; 0 means all')
    ap.add_argument('--max-unique', type=int, default=0, help='Maximum unique positions to emit; 0 means all')
    ap.add_argument('--dedupe', choices=['position','history','none'], default='position')
    args = ap.parse_args()

    root = Path(args.dataset)
    man = json.loads((root/'manifest.json').read_text())
    rels = man['train_shards'] if args.split == 'train' else [man['dev']]
    seen = {}
    emitted = []
    counters = Counter()
    t0 = time.time()
    scanned = 0
    for rel in rels:
        path = root / rel
        with opener(path) as f:
            for row_idx, line in enumerate(f):
                if not line.strip(): continue
                scanned += 1
                if args.max_rows and scanned > args.max_rows: break
                try: src = json.loads(line)
                except Exception:
                    counters['bad_json'] += 1; continue
                rr = registry_row(src, rel, row_idx)
                if rr is None:
                    counters['missing_fen'] += 1; continue
                key = rr['position_key'] if args.dedupe == 'position' else (rr['history_key'] if args.dedupe == 'history' else rr['dataset_shard'] + ':' + str(rr['row']))
                if key in seen:
                    seen[key]['duplicate_count_observed'] += 1
                    counters['duplicates'] += 1
                    continue
                seen[key] = rr
                emitted.append(rr)
                counters['source:' + str(rr['source'])] += 1
                counters['phase:' + rr['phase']] += 1
                counters['material:' + rr['material_bucket']] += 1
                if args.max_unique and len(emitted) >= args.max_unique: break
        if (args.max_rows and scanned >= args.max_rows) or (args.max_unique and len(emitted) >= args.max_unique): break
    write_zst_jsonl(emitted, Path(args.out))
    report = {
        'dataset': args.dataset,
        'split': args.split,
        'dedupe': args.dedupe,
        'scanned_rows': scanned,
        'emitted_unique': len(emitted),
        'duplicates_observed': int(counters['duplicates']),
        'seconds': time.time() - t0,
        'counters': dict(counters),
        'out': args.out,
    }
    if args.report_out:
        Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report_out).write_text(json.dumps(report, indent=2))
    print(f'METRIC registry_scanned_rows={scanned}')
    print(f'METRIC registry_unique_positions={len(emitted)}')
    print(f'METRIC registry_duplicates_observed={int(counters["duplicates"])}')
    print(f'METRIC registry_seconds={report["seconds"]:.3f}')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
