#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,math
from pathlib import Path


def cp_to_winrate(cp: float) -> float:
    # Smooth Stockfish centipawn -> expected score approximation.
    # Calibrated enough for an auxiliary/blunder target; avoids raw-cp scale pathologies.
    cp=max(-1000.0,min(1000.0,float(cp)))
    return 1.0/(1.0+math.exp(-cp/250.0))

def bucket(loss: float) -> int:
    # 0 good, 1 inaccuracy, 2 mistake, 3 blunder, based on expected-score loss.
    if loss < 0.05: return 0
    if loss < 0.10: return 1
    if loss < 0.20: return 2
    return 3

def main():
    ap=argparse.ArgumentParser(description='Attach winrate-loss/blunder targets from Stockfish cp-loss labels to training JSONL rows by id.')
    ap.add_argument('--input',required=True,help='Training JSONL')
    ap.add_argument('--labels',required=True,help='Stockfish cp-loss JSONL/CSV-like JSONL rows with id/cp_best/cp_played')
    ap.add_argument('--out',required=True)
    ap.add_argument('--default-nan',action='store_true',help='Write NaN fields for unlabeled rows instead of omitting')
    args=ap.parse_args()
    labels={}; bad=0
    with open(args.labels) as f:
        first=f.readline(); f.seek(0)
        if first.lstrip().startswith('{'):
            for line in f:
                if not line.strip(): continue
                r=json.loads(line); labels[str(r['id'])]=r
        else:
            import csv
            for r in csv.DictReader(f): labels[str(r['id'])]=r
    total=hit=0; sums=0.0; counts=[0,0,0,0]
    Path(args.out).parent.mkdir(parents=True,exist_ok=True)
    with open(args.input) as fi, open(args.out,'w') as fo:
        for line in fi:
            if not line.strip(): continue
            r=json.loads(line); total+=1; lab=labels.get(str(r.get('id','')))
            if lab is not None:
                try:
                    wb=cp_to_winrate(float(lab['cp_best'])); wp=cp_to_winrate(float(lab['cp_played'])); loss=max(0.0,wb-wp); b=bucket(loss)
                    r['stockfish_best_winrate']=wb; r['stockfish_played_winrate']=wp; r['stockfish_winrate_loss']=loss; r['stockfish_blunder_bucket']=b; hit+=1; sums+=loss; counts[b]+=1
                except Exception:
                    bad+=1
            elif args.default_nan:
                r['stockfish_best_winrate']=None; r['stockfish_played_winrate']=None; r['stockfish_winrate_loss']=None; r['stockfish_blunder_bucket']=-1
            fo.write(json.dumps(r,separators=(',',':'))+'\n')
    print(f'METRIC winrate_rows={total}')
    print(f'METRIC winrate_labeled_rows={hit}')
    print(f'METRIC winrate_bad_labels={bad}')
    print(f'METRIC winrate_avg_loss={sums/max(1,hit):.6f}')
    for i,c in enumerate(counts): print(f'METRIC winrate_bucket_{i}={c}')
if __name__=='__main__': main()
