#!/usr/bin/env python3
import argparse,subprocess,sys,time
from pathlib import Path
def cnt(p):
 try: return sum(1 for _ in open(p))
 except: return 0
def main():
 a=argparse.ArgumentParser(); a.add_argument('--input',required=True); a.add_argument('--out',required=True); a.add_argument('--max-rows',type=int,default=250000); a.add_argument('--depth',type=int,default=8); a.add_argument('--multipv',type=int,default=4); a.add_argument('--workers',type=int,default=8); args=a.parse_args(); out=Path(args.out); tmp=out.with_suffix(out.suffix+'.parts'); tmp.mkdir(parents=True,exist_ok=True); out.parent.mkdir(parents=True,exist_ok=True); per=(args.max_rows+args.workers-1)//args.workers; procs=[]; t=time.time()
 for w in range(args.workers):
  part=tmp/f'part_{w:03d}.jsonl'; log=tmp/f'part_{w:03d}.log'; lf=open(log,'w'); cmd=[sys.executable,'scripts/stockfish_cp_loss_label.py','--input',args.input,'--out',str(part),'--max-rows',str(per),'--depth',str(args.depth),'--multipv',str(args.multipv),'--stride',str(args.workers),'--offset',str(w)]; procs.append((part,lf,subprocess.Popen(cmd,stdout=lf,stderr=subprocess.STDOUT)))
 while any(p.poll() is None for _,_,p in procs): print(f'METRIC parallel_stockfish_rows={sum(cnt(x) for x,_,_ in procs)}',flush=True); time.sleep(10)
 for _,lf,p in procs: lf.close(); assert p.returncode==0
 n=0
 with open(out,'w') as o:
  for part,_,_ in procs:
   for line in open(part):
    if n>=args.max_rows: break
    o.write(line); n+=1
 print(f'METRIC parallel_stockfish_labels={n}'); print(f'METRIC parallel_stockfish_seconds={time.time()-t:.3f}')
if __name__=='__main__': main()
