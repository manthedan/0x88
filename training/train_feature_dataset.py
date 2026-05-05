#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math, pickle, random
from pathlib import Path

def ce_policy(X,y,wp):
  loss=0; top=0
  for x,t in zip(X,y):
    logits=[sum(x[j]*wp[j][k] for j in range(len(x))) for k in range(len(wp[0]))]; m=max(logits); ex=[math.exp(v-m) for v in logits]; s=sum(ex)
    loss-=sum(t[k]*(logits[k]-m-math.log(s)) for k in range(len(t)))
    if max(range(len(t)), key=t.__getitem__)==max(range(len(logits)), key=logits.__getitem__): top+=1
  return loss/max(1,len(X)), top/max(1,len(X))

def ce_wdl(X,y,ww):
  loss=0
  for x,t in zip(X,y):
    logits=[sum(x[j]*ww[j][k] for j in range(len(x))) for k in range(3)]; m=max(logits); s=sum(math.exp(v-m) for v in logits)
    loss-=sum(t[k]*(logits[k]-m-math.log(s)) for k in range(3))
  return loss/max(1,len(X))

p=argparse.ArgumentParser(); p.add_argument('--dataset',required=True); p.add_argument('--out',required=True); p.add_argument('--epochs',type=int,default=40); p.add_argument('--lr',type=float,default=0.05); p.add_argument('--batch-size',type=int,default=512); p.add_argument('--holdout-mod',type=int,default=2); p.add_argument('--preload-device', action='store_true', help='realize full train tensors on device once and use full-batch updates'); args=p.parse_args()
from tinygrad import Tensor
from tinygrad.nn.optim import SGD
with open(args.dataset,'rb') as f: ds=pickle.load(f)
X=ds['X']; yp=ds['y_policy']; yv=ds['y_wdl']; weights=ds['weights']; moves=ds['moves']; F=ds['feature_dim']; M=len(moves)
train=[i for i in range(len(X)) if i%args.holdout_mod!=0]; dev=[i for i in range(len(X)) if i%args.holdout_mod==0]
rng=random.Random(7); wp=Tensor([[rng.uniform(-.01,.01) for _ in range(M)] for _ in range(F)], requires_grad=True); ww=Tensor([[rng.uniform(-.01,.01) for _ in range(3)] for _ in range(F)], requires_grad=True)
if args.preload_device:
  x=Tensor([X[i] for i in train]).realize(); y=Tensor([yp[i] for i in train]).realize(); v=Tensor([yv[i] for i in train]).realize(); rw=Tensor([weights[i] for i in train]).reshape(len(train),1).realize()
  opt=SGD([wp,ww], lr=args.lr/max(1,len(train))); Tensor.training=True
  for ep in range(args.epochs):
    opt.zero_grad(); lp=x.matmul(wp).log_softmax(); lv=x.matmul(ww).log_softmax(); loss=-((lp*y*rw).sum()+(lv*v*rw).sum()); loss.backward(); opt.step()
else:
  opt=SGD([wp,ww], lr=args.lr/max(1,args.batch_size)); Tensor.training=True
  for ep in range(args.epochs):
    rng.shuffle(train)
    for off in range(0,len(train),args.batch_size):
      idx=train[off:off+args.batch_size]
      x=Tensor([X[i] for i in idx]); y=Tensor([yp[i] for i in idx]); v=Tensor([yv[i] for i in idx]); rw=Tensor([weights[i] for i in idx]).reshape(len(idx),1)
      opt.zero_grad(); lp=x.matmul(wp).log_softmax(); lv=x.matmul(ww).log_softmax(); loss=-((lp*y*rw).sum()+(lv*v*rw).sum()); loss.backward(); opt.step()
Tensor.training=False
wp_l=wp.numpy().tolist(); ww_l=ww.numpy().tolist()
trX=[X[i] for i in train[:min(len(train),2000)]]; trY=[yp[i] for i in train[:min(len(train),2000)]]; trV=[yv[i] for i in train[:min(len(train),2000)]]
dX=[X[i] for i in dev]; dY=[yp[i] for i in dev]; dV=[yv[i] for i in dev]
dp,top=ce_policy(dX,dY,wp_l); dv=ce_wdl(dX,dV,ww_l); tp,_=ce_policy(trX,trY,wp_l); tv=ce_wdl(trX,trV,ww_l); score=100/(1+dp+dv)
artifact={'kind':'frozen_conv_fen_student','moves':moves,'policy_weights':[[wp_l[j][i] for j in range(F)] for i in range(M)],'wdl_weights':[[ww_l[j][i] for j in range(F)] for i in range(3)],'policy_feature_dim':F,'wdl_feature_dim':F,'weight_average_count':0,'conv_channels':64,'conv_layers':6}
out=Path(args.out); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(artifact,separators=(',',':')))
print(f'METRIC distill_student_score={score:.6f}'); print(f'METRIC train_policy_ce={tp:.6f}'); print(f'METRIC train_wdl_ce={tv:.6f}'); print(f'METRIC dev_policy_ce={dp:.6f}'); print(f'METRIC dev_wdl_ce={dv:.6f}'); print(f'METRIC dev_policy_top1={top:.6f}'); print(f'METRIC feature_dataset_rows={len(X)}'); print(f'METRIC batch_size={args.batch_size}'); print(f'METRIC preload_device={1 if args.preload_device else 0}'); print(f'METRIC model_json_bytes={out.stat().st_size}')
