#!/usr/bin/env python3
from __future__ import annotations
import argparse, contextlib, hashlib, json, random, re, subprocess, sys
from collections import Counter, defaultdict
from pathlib import Path

try:
    import pyzstd  # type: ignore
except Exception:
    pyzstd = None


@contextlib.contextmanager
def opener(path: str, mode: str = 'rt'):
    if path.endswith('.zst'):
        if pyzstd is not None:
            with pyzstd.open(path, mode) as f:
                yield f
            return
        if 'r' in mode:
            p = subprocess.Popen(['zstdcat', path], stdout=subprocess.PIPE, text='t' in mode)
            try:
                yield p.stdout
            finally:
                if p.stdout: p.stdout.close()
                rc = p.wait()
                if rc: raise subprocess.CalledProcessError(rc, ['zstdcat', path])
            return
        if 'w' in mode:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            raw = open(path, 'wb')
            p = subprocess.Popen(['zstd', '-q', '-T0', '-c'], stdin=subprocess.PIPE, stdout=raw, text='t' in mode)
            try:
                yield p.stdin
            finally:
                if p.stdin: p.stdin.close()
                rc = p.wait(); raw.close()
                if rc: raise subprocess.CalledProcessError(rc, ['zstd', '-o', path])
            return
        raise SystemExit(f'Unsupported .zst mode: {mode}')
    with open(path, mode, encoding='utf-8' if 't' in mode else None) as f:
        yield f


def gid(row):
    rid = str(row.get('id', ''))
    m = re.match(r'(.+)_([0-9]+)$', rid)
    return m.group(1) if m else rid


def ply(row):
    if 'ply' in row:
        try: return int(row['ply'])
        except Exception: pass
    rid = str(row.get('id', ''))
    m = re.search(r'_([0-9]+)$', rid)
    return int(m.group(1)) if m else 0


def fen_key(fen):
    return ' '.join(str(fen).split()[:4])


def source_name(path):
    n = Path(path).name
    for suf in ('.jsonl.zst', '.jsonl'):
        if n.endswith(suf): return n[:-len(suf)]
    return Path(path).stem


def opening_key(rows):
    r0 = rows[0]
    for k in ('eco', 'ECO'):
        if r0.get(k): return 'eco:' + str(r0[k])
    for k in ('opening', 'Opening'):
        if r0.get(k): return 'opening:' + str(r0[k])[:80]
    first = min(rows, key=ply)
    return 'firstfen:' + fen_key(first['fen'])


def load_games(inputs, skip_plies):
    games = defaultdict(list); sources = {}
    bad = unknown_policy = 0
    for p in inputs:
        src = source_name(p)
        with opener(p, 'rt') as f:
            for line in f:
                if not line.strip(): continue
                try: r = json.loads(line)
                except Exception:
                    bad += 1; continue
                if ply(r) < skip_plies: continue
                if len(r.get('policy', {})) != 1:
                    unknown_policy += 1; continue
                g = gid(r); games[g].append(r); sources.setdefault(g, src)
    return games, sources, bad, unknown_policy


def annotate_history(rows, history_plies):
    out = {}; prev = []
    for r in sorted(rows, key=ply):
        nr = dict(r)
        if history_plies:
            nr['history_fens'] = list(reversed(prev[-history_plies:]))
        out[id(r)] = nr
        prev.append(r['fen'])
    return out


def choose(items, sources, rng, max_rows, max_rows_per_game, max_rows_per_opening, max_rows_per_source, source_caps, dedupe, history_plies, seen=None):
    seen = seen or set(); rows_out=[]; rpg=[]; oc=Counter(); sc=Counter(); skipped_dupe=skipped_opening=skipped_source=hist=0
    for g, rows in items:
        if not rows: continue
        src = sources.get(g,'unknown')
        ok = opening_key(rows); took = 0; ann = annotate_history(rows, history_plies)
        rows = list(rows); rng.shuffle(rows)
        for r in rows:
            if len(rows_out) >= max_rows: break
            if took >= max_rows_per_game: break
            if oc[ok] >= max_rows_per_opening:
                skipped_opening += 1; continue
            source_cap = source_caps.get(src, max_rows_per_source)
            if source_cap is not None and sc[src] >= source_cap:
                skipped_source += 1; continue
            fk = fen_key(r['fen'])
            if dedupe and fk in seen:
                skipped_dupe += 1; continue
            nr = ann[id(r)]
            if nr.get('history_fens'): hist += 1
            rows_out.append(nr); seen.add(fk); oc[ok]+=1; sc[src]+=1; took+=1
        if took: rpg.append(took)
        if len(rows_out) >= max_rows: break
    return {'rows': rows_out, 'rows_per_game': rpg, 'opening_counts': oc, 'source_counts': sc, 'skipped_dupe': skipped_dupe, 'skipped_opening': skipped_opening, 'skipped_source_cap': skipped_source, 'rows_with_history': hist}


def git_commit():
    try:
        return subprocess.check_output(['git','rev-parse','--short','HEAD'], text=True).strip()
    except Exception:
        return 'unknown'

def file_sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20), b''):
            h.update(b)
    return h.hexdigest()

def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with opener(str(path), 'wt') as f:
        for r in rows:
            f.write(json.dumps(r, separators=(',', ':')) + '\n')


def summarize(d):
    rpg=d['rows_per_game']; oc=d['opening_counts']
    return {'rows': len(d['rows']), 'games': len(rpg), 'avg_rows_per_game': sum(rpg)/max(1,len(rpg)), 'openings': len(oc), 'top_opening_rows': max(oc.values()) if oc else 0, 'rows_with_history': d['rows_with_history'], 'skipped_duplicate_fens': d['skipped_dupe'], 'skipped_opening_cap': d['skipped_opening'], 'skipped_source_cap': d.get('skipped_source_cap',0), 'source_counts': dict(d['source_counts'].most_common()), 'top_openings': oc.most_common(20)}


def main():
    ap=argparse.ArgumentParser(description='Build a sharded supervised chess dataset with whole-game dev split and balance caps.')
    ap.add_argument('--input', nargs='+', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--name', default='supervised_shards')
    ap.add_argument('--max-rows', type=int, default=10_000_000)
    ap.add_argument('--dev-rows', type=int, default=100_000)
    ap.add_argument('--rows-per-shard', type=int, default=1_000_000)
    ap.add_argument('--max-rows-per-game', type=int, default=32)
    ap.add_argument('--max-rows-per-opening', type=int, default=100_000)
    ap.add_argument('--max-rows-per-source', type=int, default=0, help='Global per-source row cap; 0 means disabled')
    ap.add_argument('--source-cap', action='append', default=[], metavar='SOURCE=N', help='Override cap for a source name; repeatable')
    ap.add_argument('--skip-plies', type=int, default=10)
    ap.add_argument('--history-plies', type=int, default=2)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--dedupe-fen', action='store_true')
    ap.add_argument('--zst', action='store_true')
    args=ap.parse_args()
    rng=random.Random(args.seed); out=Path(args.out_dir)
    source_caps={}
    for spec in args.source_cap:
        if '=' not in spec: raise SystemExit(f'--source-cap must be SOURCE=N, got {spec!r}')
        k,v=spec.rsplit('=',1); source_caps[k]=int(v)
    max_rows_per_source = args.max_rows_per_source or None
    games,sources,bad,unknown=load_games(args.input,args.skip_plies)
    items=list(games.items()); rng.shuffle(items)
    dev_game_count=max(1, int(len(items)*min(0.2, max(0.01, args.dev_rows/max(1,args.max_rows+args.dev_rows)))))
    dev_items=items[:dev_game_count]; train_items=items[dev_game_count:]
    dev=choose(dev_items,sources,rng,args.dev_rows,args.max_rows_per_game,args.max_rows_per_opening,max_rows_per_source,source_caps,args.dedupe_fen,args.history_plies,set())
    train=choose(train_items,sources,rng,args.max_rows,args.max_rows_per_game,args.max_rows_per_opening,max_rows_per_source,source_caps,args.dedupe_fen,args.history_plies,set())
    ext='.jsonl.zst' if args.zst else '.jsonl'
    shard_paths=[]
    for i in range(0,len(train['rows']),args.rows_per_shard):
        si=i//args.rows_per_shard; rel=f'train/shard_{si:04d}{ext}'; write_jsonl(out/rel, train['rows'][i:i+args.rows_per_shard]); shard_paths.append(rel)
    dev_rel=f'dev/dev_{len(dev["rows"])}{ext}'; write_jsonl(out/dev_rel, dev['rows'])
    caps={'max_rows_per_game':args.max_rows_per_game,'max_rows_per_opening':args.max_rows_per_opening,'max_rows_per_source':max_rows_per_source,'source_caps':source_caps}
    repro={'inputs':args.input,'input_sizes':{p:(Path(p).stat().st_size if Path(p).exists() else None) for p in args.input},'seed':args.seed,'caps':caps,'git_commit':git_commit(),'script':'scripts/build_supervised_dataset_shards.py','script_sha256':file_sha256(__file__),'argv':sys.argv}
    report={'name':args.name,'seed':args.seed,'inputs':args.input,'reproducibility':repro,'bad_json_lines':bad,'skipped_non_single_policy':unknown,'input_games':len(games),'skip_plies':args.skip_plies,'history_plies':args.history_plies,'caps':caps,'train':summarize(train),'dev':summarize(dev)}
    manifest={'name':args.name,'format':'jsonl.zst' if args.zst else 'jsonl','train_shards':shard_paths,'dev':dev_rel,'rows_per_shard':args.rows_per_shard,'total_train_rows':len(train['rows']),'total_dev_rows':len(dev['rows']),'history_plies':args.history_plies,'skip_plies':args.skip_plies,'caps':caps,'reproducibility':repro,'report':'reports/dataset_report.json'}
    (out/'reports').mkdir(parents=True,exist_ok=True); (out/'manifest.json').write_text(json.dumps(manifest,indent=2)); (out/'reports/dataset_report.json').write_text(json.dumps(report,indent=2))
    print(f'METRIC dataset_train_rows={len(train["rows"])}')
    print(f'METRIC dataset_dev_rows={len(dev["rows"])}')
    print(f'METRIC dataset_train_shards={len(shard_paths)}')
    print(f'METRIC dataset_train_games={len(train["rows_per_game"])}')
    print(f'METRIC dataset_train_openings={len(train["opening_counts"])}')

if __name__ == '__main__': main()
