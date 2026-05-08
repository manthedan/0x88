#!/usr/bin/env python3
from __future__ import annotations
import argparse, contextlib, json, math, os, re, shutil, subprocess, time
from pathlib import Path

INFO_RE = re.compile(r"multipv\s+(\d+).*?score\s+(cp|mate)\s+(-?\d+).*?\spv\s+(.+)$")
WDL_RE = re.compile(r"\bwdl\s+(\d+)\s+(\d+)\s+(\d+)")

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

def find_engine(x: str) -> str:
    if Path(x).exists(): return x
    y = shutil.which(x)
    if y: return y
    z = Path('.local_engines/stockfish_pkg/usr/games/stockfish')
    if z.exists(): return str(z)
    raise SystemExit('missing stockfish; set STOCKFISH_BIN')

def send(p, s: str):
    assert p.stdin is not None
    p.stdin.write(s + '\n'); p.stdin.flush()

def until(p, token: str, limit: int = 200000):
    assert p.stdout is not None
    out = []
    for _ in range(limit):
        line = p.stdout.readline()
        if not line: break
        line = line.strip(); out.append(line)
        if line.startswith(token): return out
    raise RuntimeError('missing ' + token)

def score_to_cp(kind: str, raw: str) -> int:
    v = int(raw)
    if kind == 'mate':
        # Keep mate ordering but avoid infinities in softmax/q conversion.
        return 100000 if v > 0 else -100000
    return v

def parse(lines):
    by = {}
    wdls = {}
    bestmove = None
    for l in lines:
        if l.startswith('bestmove '):
            parts = l.split()
            if len(parts) > 1: bestmove = parts[1]
        m = INFO_RE.search(l)
        if not m: continue
        mpv = int(m.group(1)); cp = score_to_cp(m.group(2), m.group(3)); pv = m.group(4).split()
        if not pv: continue
        wm = WDL_RE.search(l)
        if wm:
            w,d,lo = [int(wm.group(i)) / 1000.0 for i in (1,2,3)]
            wdls[mpv] = [w,d,lo]
        by[mpv] = {'move': pv[0], 'cp': cp, 'pv': pv}
    if not by and bestmove and bestmove != '(none)':
        by[1] = {'move': bestmove, 'cp': 0, 'pv': [bestmove]}
    if not by:
        raise RuntimeError('no multipv lines parsed')
    ordered = [by[k] for k in sorted(by)]
    best_cp = max(x['cp'] for x in ordered)
    temp = None
    return ordered, best_cp, wdls

def cp_policy(entries, temp_cp: float):
    finite = [max(-2000, min(2000, x['cp'])) for x in entries]
    best = max(finite)
    exps = [math.exp((cp - best) / max(1e-6, temp_cp)) for cp in finite]
    s = sum(exps) or 1.0
    pol = {}
    for x,e in zip(entries, exps):
        pol[x['move']] = pol.get(x['move'], 0.0) + e / s
    # renormalize if duplicate PV first moves collapsed
    total = sum(pol.values()) or 1.0
    return {m: v / total for m,v in sorted(pol.items(), key=lambda kv: -kv[1])}

def cp_to_wdl(cp: int):
    q = math.tanh(max(-2000, min(2000, cp)) / 400.0)
    w = max(0.0, q); l = max(0.0, -q); d = max(0.0, 1.0 - w - l)
    return [w,d,l], q

def main() -> int:
    ap = argparse.ArgumentParser(description='Label registry positions with root Stockfish MultiPV policy/value labels.')
    ap.add_argument('--stockfish', default=os.environ.get('STOCKFISH_BIN', '.local_engines/stockfish_pkg/usr/games/stockfish'))
    ap.add_argument('--input', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--max-rows', type=int, default=0)
    ap.add_argument('--depth', type=int, default=8)
    ap.add_argument('--multipv', type=int, default=4)
    ap.add_argument('--temperature-cp', type=float, default=100.0)
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--offset', type=int, default=0)
    args = ap.parse_args()

    engine = find_engine(args.stockfish)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    p = subprocess.Popen([engine], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    t0 = time.time(); n = seen = failures = 0
    try:
        send(p, 'uci'); until(p, 'uciok')
        send(p, f'setoption name MultiPV value {args.multipv}')
        send(p, 'setoption name Threads value 1')
        # Ignore unsupported option errors; Stockfish keeps running.
        send(p, 'setoption name UCI_ShowWDL value true')
        send(p, 'isready'); until(p, 'readyok')
        with opener(args.input) as f, open(args.out, 'wt', encoding='utf-8') as o:
            for line in f:
                if not line.strip(): continue
                seen += 1
                if (seen - 1 - args.offset) % args.stride: continue
                if args.max_rows and n >= args.max_rows: break
                r = json.loads(line); fen = r.get('fen')
                if not fen: continue
                try:
                    send(p, 'position fen ' + fen)
                    send(p, f'go depth {args.depth}')
                    lines = until(p, 'bestmove')
                    entries, best_cp, wdls = parse(lines)
                    policy = cp_policy(entries, args.temperature_cp)
                    wdl, q = (wdls.get(1), None) if wdls else (None, None)
                    if wdl is None:
                        wdl, q = cp_to_wdl(best_cp)
                    else:
                        q = wdl[0] - wdl[2]
                    out = {
                        'position_key': r.get('position_key'),
                        'history_key': r.get('history_key'),
                        'fen': fen,
                        'played': r.get('played'),
                        'source': r.get('source'),
                        'phase': r.get('phase'),
                        'opening_key': r.get('opening_key'),
                        'teacher': 'stockfish',
                        'teacher_bin': engine,
                        'depth': args.depth,
                        'multipv': args.multipv,
                        'temperature_cp': args.temperature_cp,
                        'best': entries[0]['move'],
                        'best_cp': best_cp,
                        'q': q,
                        'wdl': wdl,
                        'policy': policy,
                        'scores_cp': {x['move']: x['cp'] for x in entries},
                        'pv': {x['move']: x['pv'] for x in entries},
                    }
                    o.write(json.dumps(out, separators=(',', ':')) + '\n')
                    n += 1
                except Exception as e:
                    failures += 1
                    o.write(json.dumps({'position_key': r.get('position_key'), 'fen': fen, 'teacher':'stockfish', 'error': str(e)}, separators=(',', ':')) + '\n')
    finally:
        try: send(p, 'quit')
        except Exception: pass
        p.kill()
    dt = time.time() - t0
    print(f'METRIC stockfish_root_labels={n}')
    print(f'METRIC stockfish_root_failures={failures}')
    print(f'METRIC stockfish_root_seconds={dt:.3f}')
    print(f'METRIC stockfish_root_rows_per_sec={n/max(dt,1e-9):.3f}')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
