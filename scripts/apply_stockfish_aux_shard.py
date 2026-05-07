#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,math
from pathlib import Path

def main():
 p=argparse.ArgumentParser(); p.add_argument('--input',required=True); p.add_argument('--labels',required=True); p.add_argument('--out',required=True); p.add_argument('--shards',type=int,required=True); p.add_argument('--shard',type=int,required=True); p.add_argument('--max-rows',type=int,default=10**12); p.add_argument('--cp-scale',type=float,default=400.0); args=p.parse_args()
 lab={}
 for line in open(args.labels):
  r=json.loads(line); lab[r['id']]=r
 Path(args.out).parent.mkdir(parents=True,exist_ok=True); n=written=matched=0
 with open(args.input) as f, open(args.out,'w') as out:
  for idx,line in enumerate(f):
   if n>=args.max_rows: break
   n+=1
   if idx % args.shards != args.shard: continue
   r=json.loads(line); s=lab.get(r.get('id'))
   if s:
    matched+=1; cp=float(s.get('cp_best',0.0)); r['stockfish_q']=math.tanh(cp/args.cp_scale); r['stockfish_cp_best']=cp; r['stockfish_cp_loss']=float(s.get('cp_loss',0.0))
   out.write(json.dumps(r,separators=(',',':'))+'\n'); written+=1
 print(f'METRIC aux_shard_rows_seen={n}')
 print(f'METRIC aux_shard_rows_written={written}')
 print(f'METRIC aux_shard_matched={matched}')
if __name__=='__main__': main()
