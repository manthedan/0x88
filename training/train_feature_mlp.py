#!/usr/bin/env python3
from __future__ import annotations
import argparse, pickle, random, math, json
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('--dataset',required=True); p.add_argument('--out',required=True); p.add_argument('--epochs',type=int,default=20); p.add_argument('--hidden',type=int,default=512); p.add_argument('--lr',type=float,default=1e-3); p.add_argument('--optimizer', choices=['sgd','adam'], default='adam'); p.add_argument('--holdout-mod',type=int,default=2); p.add_argument('--eval-rows',type=int,default=20000); args=p.parse_args()
from tinygrad import Tensor
from tinygrad.nn.optim import SGD, Adam
with open(args.dataset,'rb') as f: ds=pickle.load(f)
X=ds['X']; ids=ds['y_move_id']; yv=ds['y_wdl']; weights=ds['weights']; moves=ds['moves']; F=ds['feature_dim']; M=len(moves); H=args.hidden
train=[i for i in range(len(X)) if i%args.holdout_mod!=0]; dev=[i for i in range(len(X)) if i%args.holdout_mod==0]
rng=random.Random(11); scale1=(2/F)**0.5; scale2=(2/H)**0.5
w1=Tensor([[rng.uniform(-scale1,scale1) for _ in range(H)] for _ in range(F)], requires_grad=True); b1=Tensor.zeros(H, requires_grad=True)
wp=Tensor([[rng.uniform(-scale2,scale2) for _ in range(M)] for _ in range(H)], requires_grad=True); bp=Tensor.zeros(M, requires_grad=True)
ww=Tensor([[rng.uniform(-scale2,scale2) for _ in range(3)] for _ in range(H)], requires_grad=True); bw=Tensor.zeros(3, requires_grad=True)
x=Tensor([X[i] for i in train]).realize(); tid=Tensor([ids[i] for i in train]).realize(); v=Tensor([yv[i] for i in train]).realize(); rw=Tensor([weights[i] for i in train]).reshape(len(train),1).realize()
params=[w1,b1,wp,bp,ww,bw]
opt=Adam(params, lr=args.lr) if args.optimizer == 'adam' else SGD(params, lr=args.lr/max(1,len(train)))
Tensor.training=True
for ep in range(args.epochs):
  opt.zero_grad(); h=(x.matmul(w1)+b1).relu(); lp=(h.matmul(wp)+bp).log_softmax(); lv=(h.matmul(ww)+bw).log_softmax()
  loss=lp.sparse_categorical_crossentropy(tid, reduction='none').reshape(len(train),1).mul(rw).sum()-((lv*v*rw).sum())
  loss.backward(); opt.step(); print(f'METRIC epoch_{ep+1}_loss={float(loss.numpy()):.6f}', flush=True)
Tensor.training=False
n=min(args.eval_rows,len(dev)); dx=Tensor([X[i] for i in dev[:n]]); did=[ids[i] for i in dev[:n]]; dv=[yv[i] for i in dev[:n]]
h=(dx.matmul(w1)+b1).relu(); logits=(h.matmul(wp)+bp).numpy(); wlog=(h.matmul(ww)+bw).numpy()
top1=top4=top8=0; pce=0.0; wce=0.0
for row,t,vt,wl in zip(logits,did,dv,wlog):
  ranked=sorted(range(len(row)), key=lambda k: row[k], reverse=True); top1+=t==ranked[0]; top4+=t in ranked[:4]; top8+=t in ranked[:8]
  m=max(row); pce += -(row[t]-m-math.log(sum(math.exp(z-m) for z in row)))
  mw=max(wl); wce += -sum(vt[k]*(wl[k]-mw-math.log(sum(math.exp(z-mw) for z in wl))) for k in range(3))
obj={'kind':'frozen_conv_feature_mlp_student','moves':moves,'feature_dim':F,'hidden':H,'w1':w1.numpy().tolist(),'b1':b1.numpy().tolist(),'policy_w':wp.numpy().tolist(),'policy_b':bp.numpy().tolist(),'wdl_w':ww.numpy().tolist(),'wdl_b':bw.numpy().tolist()}
out=Path(args.out); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(obj,separators=(',',':')))
print(f'METRIC dev_policy_ce={pce/n:.6f}'); print(f'METRIC dev_wdl_ce={wce/n:.6f}'); print(f'METRIC dev_policy_top1={top1/n:.6f}'); print(f'METRIC dev_policy_top4={top4/n:.6f}'); print(f'METRIC dev_policy_top8={top8/n:.6f}'); print(f'METRIC hidden={H}'); print(f'METRIC feature_dataset_rows={len(X)}'); print(f'METRIC model_json_bytes={out.stat().st_size}')
