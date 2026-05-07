#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, subprocess
from collections import Counter
from pathlib import Path
try:
    import pyzstd  # type: ignore
except Exception:
    pyzstd=None

def opener(path):
    if str(path).endswith('.zst'):
        if pyzstd: return pyzstd.open(path,'rt')
        return subprocess.Popen(['zstdcat', str(path)], stdout=subprocess.PIPE, text=True).stdout
    return open(path, encoding='utf-8')

def ply(row):
    if 'ply' in row:
        try: return int(row['ply'])
        except Exception: pass
    rid=str(row.get('id','')); import re
    m=re.search(r'_([0-9]+)$',rid); return int(m.group(1)) if m else 0

def bucket_elo(e):
    try: e=int(e)
    except Exception: return 'unknown'
    lo=(e//200)*200; return f'{lo}-{lo+199}'

def bucket_time_control(tc):
    if not tc: return 'unknown'
    import re
    m=re.match(r'^(\d+)\+(\d+)$',str(tc))
    if not m: return 'unknown'
    initial=int(m.group(1)); inc=int(m.group(2)); est=initial+40*inc
    if est < 180: return 'bullet'
    if est < 480: return 'blitz'
    if est < 1500: return 'rapid'
    return 'classical'

def phase(row):
    p=ply(row); board=str(row.get('fen','')).split()[0] if row.get('fen') else ''; pieces=sum(1 for c in board if c.isalpha())
    return 'opening' if p < 20 else ('endgame' if pieces <= 12 else 'middlegame')

def material_bucket(row):
    board=str(row.get('fen','')).split()[0] if row.get('fen') else ''
    vals={'p':1,'n':3,'b':3,'r':5,'q':9}; white=black=0
    for c in board:
        if c.isalpha():
            if c.isupper(): white += vals.get(c.lower(),0)
            else: black += vals.get(c,0)
    diff=white-black
    if abs(diff) <= 1: return 'equal'
    if abs(diff) <= 3: return 'small_edge'
    if abs(diff) <= 8: return 'large_edge'
    return 'decisive_material'

def fen_key(fen):
    return ' '.join(str(fen).split()[:4])

def game_id(row):
    import re
    rid=str(row.get('id',''))
    m=re.match(r'(.+)_([0-9]+)$',rid)
    return m.group(1) if m else rid

def opening_proxy(row):
    for k in ('eco','ECO'):
        if row.get(k): return 'eco:'+str(row[k])
    for k in ('opening','Opening'):
        if row.get(k): return 'opening:'+str(row[k])[:80]
    return 'fen:'+fen_key(row.get('fen',''))

def scan(paths):
    rows=hist=wdl=sf=0; plyc=Counter(); src=Counter(); elo=Counter(); moves=Counter(); tc=Counter(); phases=Counter(); mat=Counter(); variants=Counter(); fens=Counter(); games=Counter(); openings=Counter()
    for p in paths:
        with opener(p) as f:
            for line in f:
                if not line.strip(): continue
                r=json.loads(line); rows+=1
                if r.get('history_fens'): hist+=1
                if r.get('wdl') is not None: wdl+=1
                if r.get('stockfish_q') is not None: sf+=1
                plyc[(ply(r)//10)*10]+=1
                fens[fen_key(r.get('fen',''))]+=1
                games[game_id(r)]+=1
                openings[opening_proxy(r)]+=1
                src[str(r.get('source', r.get('teacher', Path(p).stem)))]+=1
                e=r.get('active_elo', r.get('elo', r.get('WhiteElo', r.get('white_elo'))))
                elo[bucket_elo(e)]+=1
                tc[bucket_time_control(r.get('time_control'))]+=1
                phases[phase(r)]+=1
                mat[material_bucket(r)]+=1
                variants[str(r.get('variant','unknown'))]+=1
                pol=r.get('policy',{})
                if len(pol)==1: moves[next(iter(pol))]+=1
    dup_rows=sum(c-1 for c in fens.values() if c>1); dup_keys=sum(1 for c in fens.values() if c>1)
    return {'rows':rows,'rows_with_history':hist,'rows_with_wdl':wdl,'rows_with_stockfish_q':sf,'ply_buckets':dict(sorted(plyc.items())),'source_counts':dict(src.most_common(30)),'elo_buckets':dict(sorted(elo.items())),'time_control_buckets':dict(tc.most_common()),'phase_buckets':dict(phases.most_common()),'material_buckets':dict(mat.most_common()),'variant_counts':dict(variants.most_common()),'unique_games':len(games),'duplicate_fen_keys':dup_keys,'duplicate_fen_rows':dup_rows,'duplicate_fen_row_rate':dup_rows/max(1,rows),'top_duplicate_fens':[(k,c) for k,c in fens.most_common(20) if c>1],'opening_proxy_counts':dict(openings.most_common(30)),'top_game_rows':games.most_common(20),'top_moves':moves.most_common(30)}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--dataset-dir',required=True); ap.add_argument('--out',default=''); args=ap.parse_args()
    root=Path(args.dataset_dir); man=json.loads((root/'manifest.json').read_text())
    train=[root/p for p in man['train_shards']]; dev=[root/man['dev']]
    report={'manifest':man,'train':scan(train),'dev':scan(dev)}
    text=json.dumps(report,indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True,exist_ok=True); Path(args.out).write_text(text)
    else: print(text)
    print(f'METRIC report_train_rows={report["train"]["rows"]}')
    print(f'METRIC report_dev_rows={report["dev"]["rows"]}')
    print(f'METRIC report_train_history_rows={report["train"]["rows_with_history"]}')
if __name__=='__main__': main()
