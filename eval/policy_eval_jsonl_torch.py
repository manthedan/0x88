#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,re,math
from collections import defaultdict
from pathlib import Path
import subprocess
try:
 import pyzstd  # type: ignore
except Exception:
 pyzstd=None
from training.train_residual_torch import fixed_policy_moves, planes, input_plane_count


def opener(path):
 path=str(path)
 if path.endswith('.zst'):
  if pyzstd: return pyzstd.open(path,'rt')
  return subprocess.Popen(['zstdcat',path],stdout=subprocess.PIPE,text=True).stdout
 return open(path)

def ply(row):
 if 'ply' in row:
  try: return int(row['ply'])
  except Exception: pass
 m=re.search(r'_([0-9]+)$',str(row.get('id',''))); return int(m.group(1)) if m else 0

def bucket_ply(p): return f'{(p//10)*10:03d}-{(p//10)*10+9:03d}'
def bucket_source(row,path): return str(row.get('source') or Path(path).stem)
def bucket_phase(row):
 fen=str(row.get('fen','')); board=fen.split()[0] if fen else ''; pieces=sum(1 for c in board if c.isalpha())
 return 'opening' if ply(row)<20 else ('endgame' if pieces<=12 else 'middlegame')
def bucket_elo(row):
 e=row.get('active_elo',row.get('elo',row.get('WhiteElo',None)))
 try: e=int(e)
 except Exception: return 'unknown'
 lo=(e//200)*200; return f'{lo}-{lo+199}'

class Acc:
 def __init__(self): self.n=0; self.pce=0.0; self.wce=0.0; self.t1=0; self.t4=0; self.t8=0; self.rank=0.0; self.legal=0
 def add(self,pce,wce,t1,t4,t8,rank,legal,n): self.n+=n; self.pce+=pce; self.wce+=wce; self.t1+=t1; self.t4+=t4; self.t8+=t8; self.rank+=rank; self.legal+=legal
 def obj(self):
  n=max(1,self.n); return {'rows':self.n,'policy_ce':self.pce/n,'wdl_ce':self.wce/n,'top1':self.t1/n,'top4':self.t4/n,'top8':self.t8/n,'perplexity':math.exp(min(20,self.pce/n)),'mean_full_rank':self.rank/n,'selected_move_legality':self.legal/n}

def main():
 ap=argparse.ArgumentParser(description='Policy-only eval from JSONL with bucket reports.')
 ap.add_argument('--checkpoint',required=True); ap.add_argument('--input',nargs='+',required=True); ap.add_argument('--out',default='')
 ap.add_argument('--max-rows',type=int,default=100000); ap.add_argument('--batch-size',type=int,default=512); ap.add_argument('--history-plies',type=int,default=2); ap.add_argument('--state-planes',action='store_true'); ap.add_argument('--channels',type=int,default=0); ap.add_argument('--blocks',type=int,default=0); ap.add_argument('--device',default='auto')
 args=ap.parse_args()
 import torch, torch.nn as nn, torch.nn.functional as F
 moves=fixed_policy_moves(); mid={m:i for i,m in enumerate(moves)}; C=input_plane_count(args.history_plies,args.state_planes); P=len(moves)
 ck=torch.load(args.checkpoint,map_location='cpu'); meta=ck.get('meta') or ck.get('args') or {}; channels=args.channels or int(meta.get('channels',48)); blocks=args.blocks or int(meta.get('blocks',5))
 class Block(nn.Module):
  def __init__(self,ch): super().__init__(); self.c1=nn.Conv2d(ch,ch,3,padding=1); self.c2=nn.Conv2d(ch,ch,3,padding=1)
  def forward(self,z): return F.relu(self.c2(F.relu(self.c1(z)))+z)
 class Net(nn.Module):
  def __init__(self): super().__init__(); self.stem=nn.Conv2d(C,channels,3,padding=1); self.blocks=nn.Sequential(*[Block(channels) for _ in range(blocks)]); self.policy=nn.Linear(channels*64,P); self.wdl=nn.Linear(channels,3)
  def forward(self,z): h=self.blocks(F.relu(self.stem(z))); return self.policy(h.flatten(1)), self.wdl(h.mean((2,3)))
 device='cuda' if args.device=='auto' and torch.cuda.is_available() else ('cpu' if args.device=='auto' else args.device)
 net=Net().to(device); net.load_state_dict(ck['model'] if isinstance(ck,dict) and 'model' in ck else ck); net.eval()
 accs={k:defaultdict(Acc) for k in ['source','ply','phase','elo']}; overall=Acc(); batch=[]; skipped=0
 def flush():
  nonlocal batch
  if not batch: return
  x=torch.tensor([b['x'] for b in batch],device=device,dtype=torch.float32); y=torch.tensor([b['y'] for b in batch],device=device); v=torch.tensor([b['wdl'] for b in batch],device=device,dtype=torch.float32)
  with torch.no_grad():
   pl,wl=net(x); pces=F.cross_entropy(pl,y,reduction='none'); wces=(-(F.log_softmax(wl,1)*v).sum(1)); pred=pl.topk(8,1).indices; ranks=(pl>pl.gather(1,y[:,None])).sum(1)+1
  for i,b in enumerate(batch):
   t1=int((pred[i,:1]==y[i]).any()); t4=int((pred[i,:4]==y[i]).any()); t8=int((pred[i]==y[i]).any()); pc=float(pces[i]); wc=float(wces[i]); rank=float(ranks[i]); legal=1
   overall.add(pc,wc,t1,t4,t8,rank,legal,1)
   for kind,key in b['buckets'].items(): accs[kind][key].add(pc,wc,t1,t4,t8,rank,legal,1)
  batch=[]
 for path in args.input:
  with opener(path) as f:
   for line in f:
    if overall.n+len(batch)>=args.max_rows: break
    r=json.loads(line); pol=r.get('policy',{})
    if len(pol)!=1: continue
    mv=next(iter(pol))
    if mv not in mid: skipped+=1; continue
    batch.append({'x':planes(r['fen'],r.get('history_fens',[])[:args.history_plies],args.history_plies,args.state_planes),'y':mid[mv],'wdl':r.get('wdl',[.25,.5,.25]),'buckets':{'source':bucket_source(r,path),'ply':bucket_ply(ply(r)),'phase':bucket_phase(r),'elo':bucket_elo(r)}})
    if len(batch)>=args.batch_size: flush()
  if overall.n+len(batch)>=args.max_rows: break
 flush()
 report={'overall':overall.obj(),'skipped_unknown_moves':skipped,'buckets':{kind:{k:v.obj() for k,v in sorted(d.items())} for kind,d in accs.items()}}
 if args.out:
  Path(args.out).parent.mkdir(parents=True,exist_ok=True); Path(args.out).write_text(json.dumps(report,indent=2))
 else: print(json.dumps(report,indent=2))
 print(f'METRIC bucket_eval_rows={overall.n}'); print(f'METRIC bucket_eval_policy_ce={overall.obj()["policy_ce"]:.6f}'); print(f'METRIC bucket_eval_perplexity={overall.obj()["perplexity"]:.6f}'); print(f'METRIC bucket_eval_top1={overall.obj()["top1"]:.6f}'); print(f'METRIC bucket_eval_mean_full_rank={overall.obj()["mean_full_rank"]:.6f}'); print(f'METRIC bucket_eval_selected_move_legality={overall.obj()["selected_move_legality"]:.6f}')
if __name__=='__main__': main()
