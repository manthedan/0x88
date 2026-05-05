#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, pickle
from pathlib import Path
from train_student import load_rows, merge_fen_rows

p=argparse.ArgumentParser()
p.add_argument('--train', nargs='+', required=True)
p.add_argument('--feature-cache', required=True)
p.add_argument('--out', required=True)
p.add_argument('--merge-fen', action='store_true')
args=p.parse_args()
rows=load_rows(args.train)
if args.merge_fen: rows=merge_fen_rows(rows)
cache=json.loads(Path(args.feature_cache).read_text())
moves=sorted({m for r in rows for m in r['policy']})
move_idx={m:i for i,m in enumerate(moves)}
X=[]; y=[]; yi=[]; yv=[]; w=[]; kept=[]; missing=0; sparse=0
for r in rows:
    feat=cache.get(r['fen'])
    if feat is None:
        missing+=1; continue
    X.append([float(v) for v in feat])
    tv=[0.0]*len(moves); mass=sum(float(v) for v in r['policy'].values()) or 1.0
    for m,v in r['policy'].items(): tv[move_idx[m]]=float(v)/mass
    y.append(tv)
    if len(r['policy']) == 1:
        yi.append(move_idx[next(iter(r['policy']))]); sparse += 1
    else:
        yi.append(-1)
    yv.append([float(v) for v in r['wdl']]); w.append(float(r.get('_weight',1.0))); kept.append(r.get('fen',''))
obj={'moves':moves,'X':X,'y_policy':y,'y_move_id':yi,'y_wdl':yv,'weights':w,'fens':kept,'feature_dim':len(X[0]) if X else 0}
out=Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
with out.open('wb') as f: pickle.dump(obj,f,protocol=pickle.HIGHEST_PROTOCOL)
print(f'METRIC feature_dataset_rows={len(X)}')
print(f'METRIC feature_dataset_missing={missing}')
print(f'METRIC feature_dataset_moves={len(moves)}')
print(f'METRIC feature_dataset_dim={obj["feature_dim"]}')
print(f'METRIC feature_dataset_sparse_policy_rows={sparse}')
print(f'METRIC feature_dataset_bytes={out.stat().st_size}')
