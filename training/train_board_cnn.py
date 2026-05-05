#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math, random
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('--train', nargs='+', required=True); p.add_argument('--out', required=True); p.add_argument('--max-rows', type=int, default=50000); p.add_argument('--epochs', type=int, default=10); p.add_argument('--channels', type=int, default=32); p.add_argument('--lr', type=float, default=1e-3); p.add_argument('--holdout-mod', type=int, default=5); p.add_argument('--eval-rows', type=int, default=10000); args=p.parse_args()
from tinygrad import Tensor
from tinygrad.nn import Conv2d, Linear
from tinygrad.nn.optim import Adam
PIECES='PNBRQKpnbrqk'
def planes(fen):
  board,side=fen.split()[:2]; x=[[[0.0]*8 for _ in range(8)] for _ in range(14)]; r=f=0
  for ch in board:
    if ch=='/': r+=1; f=0
    elif ch.isdigit(): f+=int(ch)
    else: x[PIECES.index(ch)][r][f]=1.0; f+=1
  sv=1.0 if side=='w' else -1.0
  for rr in range(8):
    for ff in range(8): x[12][rr][ff]=sv; x[13][rr][ff]=1.0
  return x
def read_rows(paths):
  rows=[]; moves=[]
  for path in paths:
    with open(path) as f:
      for line in f:
        if len(rows)>=args.max_rows: return rows, sorted(set(moves))
        r=json.loads(line); pol=r.get('policy',{})
        if len(pol)!=1: continue
        mv=next(iter(pol)); rows.append((r['fen'], mv, r.get('wdl',[.25,.5,.25]), float(r.get('weight',1.0)))); moves.append(mv)
  return rows, sorted(set(moves))
rows,moves=read_rows(args.train); mid={m:i for i,m in enumerate(moves)}; rows=[r for r in rows if r[1] in mid]
train=[i for i in range(len(rows)) if i%args.holdout_mod!=0]; dev=[i for i in range(len(rows)) if i%args.holdout_mod==0]
class Net:
  def __init__(self):
    C=args.channels; self.c1=Conv2d(14,C,3,padding=1); self.c2=Conv2d(C,C,3,padding=1); self.c3=Conv2d(C,C,3,padding=1); self.p=Linear(C,len(moves)); self.v=Linear(C,3)
  def __call__(self,x):
    h=self.c1(x).relu(); h=(self.c2(h).relu()+h); h=(self.c3(h).relu()+h); h=h.mean(axis=(2,3)); return self.p(h), self.v(h)
  def params(self): return [self.c1.weight,self.c1.bias,self.c2.weight,self.c2.bias,self.c3.weight,self.c3.bias,self.p.weight,self.p.bias,self.v.weight,self.v.bias]
net=Net(); opt=Adam(net.params(), lr=args.lr); rng=random.Random(7); Tensor.training=True
for ep in range(args.epochs):
  rng.shuffle(train); total=0.0; n=0
  for off in range(0,len(train),512):
    idx=train[off:off+512]; x=Tensor([planes(rows[i][0]) for i in idx]); y=Tensor([mid[rows[i][1]] for i in idx]); v=Tensor([rows[i][2] for i in idx]); w=Tensor([rows[i][3] for i in idx]).reshape(len(idx),1)
    opt.zero_grad(); lp,lv=net(x); lps=lp.log_softmax(); lvs=lv.log_softmax(); loss=lps.sparse_categorical_crossentropy(y,reduction='none').reshape(len(idx),1).mul(w).sum()-((lvs*v*w).sum()); loss.backward(); opt.step(); total+=float(loss.numpy()); n+=len(idx)
  print(f'METRIC epoch_{ep+1}_loss={total/max(1,n):.6f}', flush=True)
Tensor.training=False
n=min(args.eval_rows,len(dev)); top1=top4=top8=0; pce=wce=0.0
for off in range(0,n,512):
  idx=dev[off:off+512]; x=Tensor([planes(rows[i][0]) for i in idx]); lp,lv=net(x); logits=lp.numpy(); wlog=lv.numpy()
  for row,wl,i in zip(logits,wlog,idx):
    t=mid[rows[i][1]]; ranked=sorted(range(len(row)), key=lambda k: row[k], reverse=True); top1+=t==ranked[0]; top4+=t in ranked[:4]; top8+=t in ranked[:8]
    m=max(row); pce+=-(row[t]-m-math.log(sum(math.exp(z-m) for z in row))); mw=max(wl); vt=rows[i][2]; wce+=-sum(vt[k]*(wl[k]-mw-math.log(sum(math.exp(z-mw) for z in wl))) for k in range(3))
obj={'kind':'tiny_board_cnn_student','moves':moves,'channels':args.channels,'note':'prototype trainable board CNN; inference wiring pending'}
out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(obj))
print(f'METRIC board_cnn_rows={len(rows)}'); print(f'METRIC board_cnn_moves={len(moves)}'); print(f'METRIC dev_policy_ce={pce/n:.6f}'); print(f'METRIC dev_wdl_ce={wce/n:.6f}'); print(f'METRIC dev_policy_top1={top1/n:.6f}'); print(f'METRIC dev_policy_top4={top4/n:.6f}'); print(f'METRIC dev_policy_top8={top8/n:.6f}')
