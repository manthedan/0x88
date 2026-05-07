#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,math,os,re,shutil,subprocess,time
from pathlib import Path
INFO_RE=re.compile(r"multipv\s+(\d+).*?score\s+(cp|mate)\s+(-?\d+).*?\spv\s+(\S+)"); SCORE_RE=re.compile(r"score\s+(cp|mate)\s+(-?\d+)")
def eng(x):
 if Path(x).exists(): return x
 y=shutil.which(x)
 if y: return y
 z=Path('.local_engines/stockfish_pkg/usr/games/stockfish')
 if z.exists(): return str(z)
 raise SystemExit('missing stockfish')
def send(p,s): p.stdin.write(s+'\n'); p.stdin.flush()
def until(p,t):
 out=[]
 while True:
  l=p.stdout.readline()
  if not l: break
  l=l.strip(); out.append(l)
  if l.startswith(t): return out
 raise RuntimeError('missing '+t)
def sc(k,v):
 v=float(v); return (100000. if v>0 else -100000.) if k=='mate' else v
def best(lines):
 by={}
 for l in lines:
  m=INFO_RE.search(l)
  if m: by[int(m.group(1))]=(m.group(4),sc(m.group(2),m.group(3)))
 if not by: raise RuntimeError('no best')
 bm,bc=by[min(by)]; return bm,bc,{m:s for m,s in by.values()}
def played(lines):
 r=None
 for l in lines:
  m=SCORE_RE.search(l)
  if m: r=sc(m.group(1),m.group(2))
 if r is None: raise RuntimeError('no score')
 return r
def main():
 a=argparse.ArgumentParser(); a.add_argument('--stockfish',default=os.environ.get('STOCKFISH_BIN','.local_engines/stockfish_pkg/usr/games/stockfish')); a.add_argument('--input',required=True); a.add_argument('--out',required=True); a.add_argument('--max-rows',type=int,default=100); a.add_argument('--depth',type=int,default=8); a.add_argument('--multipv',type=int,default=4); a.add_argument('--stride',type=int,default=1); a.add_argument('--offset',type=int,default=0); args=a.parse_args()
 p=subprocess.Popen([eng(args.stockfish)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1); t=time.time(); n=seen=0; Path(args.out).parent.mkdir(parents=True,exist_ok=True)
 try:
  send(p,'uci'); until(p,'uciok'); send(p,f'setoption name MultiPV value {args.multipv}'); send(p,'isready'); until(p,'readyok')
  with open(args.input) as f, open(args.out,'w') as o:
   for line in f:
    if n>=args.max_rows: break
    seen+=1
    if (seen-1-args.offset)%args.stride: continue
    r=json.loads(line); pol=r.get('policy',{}); 
    if not pol: continue
    mv=max(pol.items(),key=lambda kv:kv[1])[0]; fen=r['fen']
    send(p,'position fen '+fen); send(p,f'go depth {args.depth}'); bm,bc,scores=best(until(p,'bestmove'))
    send(p,'position fen '+fen); send(p,f'go depth {args.depth} searchmoves {mv}'); pc=played(until(p,'bestmove'))
    loss=max(0,bc-pc); o.write(json.dumps({'id':r.get('id',str(seen)),'fen':fen,'played':mv,'best':bm,'cp_best':bc,'cp_played':pc,'cp_loss':loss,'depth':args.depth,'multipv':args.multipv},separators=(',',':'))+'\n'); n+=1
 finally:
  try: send(p,'quit')
  except Exception: pass
  p.kill()
 print(f'METRIC stockfish_labels={n}'); print(f'METRIC stockfish_seconds={time.time()-t:.3f}')
if __name__=='__main__': main()
