#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,math
from pathlib import Path

def main():
 p=argparse.ArgumentParser(); p.add_argument('--input',required=True); p.add_argument('--labels',required=True); p.add_argument('--out',required=True); p.add_argument('--max-rows',type=int,default=10**12); p.add_argument('--cp-scale',type=float,default=400.0); p.add_argument('--weight-mode',choices=['none','mild','harsh'],default='none'); args=p.parse_args()
 lab={}
 for line in open(args.labels):
  r=json.loads(line); lab[r['id']]=r
 Path(args.out).parent.mkdir(parents=True,exist_ok=True); n=matched=0
 with open(args.input) as f, open(args.out,'w') as out:
  for line in f:
   if n>=args.max_rows: break
   r=json.loads(line); n+=1; s=lab.get(r.get('id'))
   if s:
    matched+=1; cp=float(s.get('cp_best',0.0)); loss=float(s.get('cp_loss',0.0)); r['stockfish_q']=math.tanh(cp/args.cp_scale); r['stockfish_cp_best']=cp; r['stockfish_cp_loss']=loss
    if args.weight_mode=='mild': r['weight']=1.0 if loss<=25 else 0.8 if loss<=75 else 0.5 if loss<=150 else 0.2
    elif args.weight_mode=='harsh': r['weight']=float(s.get('weight',1.0))
   out.write(json.dumps(r,separators=(',',':'))+'\n')
 print(f'METRIC apply_aux_rows={n}')
 print(f'METRIC apply_aux_matched={matched}')
if __name__=='__main__': main()
