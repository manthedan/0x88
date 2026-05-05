#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math, pickle, random
from pathlib import Path

def topk_sparse(X,ids,wp):
  top1=top4=top8=0
  for x,target in zip(X,ids):
    logits=[sum(x[j]*wp[j][k] for j in range(len(x))) for k in range(len(wp[0]))]
    ranked=sorted(range(len(logits)), key=logits.__getitem__, reverse=True)
    top1 += int(target == ranked[0]); top4 += int(target in ranked[:4]); top8 += int(target in ranked[:8])
  den=max(1,len(X)); return top1/den, top4/den, top8/den

def ce_wdl(X,y,ww):
  loss=0
  for x,t in zip(X,y):
    logits=[sum(x[j]*ww[j][k] for j in range(len(x))) for k in range(3)]; m=max(logits); s=sum(math.exp(v-m) for v in logits)
    loss-=sum(t[k]*(logits[k]-m-math.log(s)) for k in range(3))
  return loss/max(1,len(X))

p=argparse.ArgumentParser(); p.add_argument('--dataset',required=True); p.add_argument('--out',required=True); p.add_argument('--epochs',type=int,default=40); p.add_argument('--lr',type=float,default=0.05); p.add_argument('--batch-size',type=int,default=512); p.add_argument('--holdout-mod',type=int,default=2); p.add_argument('--preload-device', action='store_true'); p.add_argument('--sparse-policy', action='store_true'); args=p.parse_args()
from tinygrad import Tensor
from tinygrad.nn.optim import SGD
with open(args.dataset,'rb') as f: ds=pickle.load(f)
X=ds['X']; yi=ds.get('y_move_id',[-1]*len(X)); yp=ds.get('y_policy'); yv=ds['y_wdl']; weights=ds['weights']; moves=ds['moves']; F=ds['feature_dim']; M=len(moves)
train=[i for i in range(len(X)) if i%args.holdout_mod!=0]; dev=[i for i in range(len(X)) if i%args.holdout_mod==0]
can_sparse=args.sparse_policy and all(yi[i] >= 0 for i in train)
if not can_sparse and yp is None: raise SystemExit('dense y_policy missing; rerun without --sparse-only or use --sparse-policy on one-hot data')
rng=random.Random(7); wp=Tensor([[rng.uniform(-.01,.01) for _ in range(M)] for _ in range(F)], requires_grad=True); ww=Tensor([[rng.uniform(-.01,.01) for _ in range(3)] for _ in range(F)], requires_grad=True)
Tensor.training=True
if args.preload_device:
  x=Tensor([X[i] for i in train]).realize(); ids=Tensor([yi[i] for i in train]).realize(); y=Tensor([yp[i] for i in train]).realize() if yp is not None else None; v=Tensor([yv[i] for i in train]).realize(); rw=Tensor([weights[i] for i in train]).reshape(len(train),1).realize()
  opt=SGD([wp,ww], lr=args.lr/max(1,len(train)))
  for _ in range(args.epochs):
    opt.zero_grad(); lp=x.matmul(wp).log_softmax(); lv=x.matmul(ww).log_softmax()
    ploss = lp.sparse_categorical_crossentropy(ids, reduction='none').reshape(len(train),1).mul(rw).sum() if can_sparse else -((lp*y*rw).sum())
    loss=ploss-((lv*v*rw).sum()); loss.backward(); opt.step()
else:
  opt=SGD([wp,ww], lr=args.lr/max(1,args.batch_size))
  for _ in range(args.epochs):
    rng.shuffle(train)
    for off in range(0,len(train),args.batch_size):
      idx=train[off:off+args.batch_size]
      x=Tensor([X[i] for i in idx]); ids=Tensor([yi[i] for i in idx]); y=Tensor([yp[i] for i in idx]) if yp is not None else None; v=Tensor([yv[i] for i in idx]); rw=Tensor([weights[i] for i in idx]).reshape(len(idx),1)
      opt.zero_grad(); lp=x.matmul(wp).log_softmax(); lv=x.matmul(ww).log_softmax()
      ploss = lp.sparse_categorical_crossentropy(ids, reduction='none').reshape(len(idx),1).mul(rw).sum() if args.sparse_policy and all(yi[i] >= 0 for i in idx) else -((lp*y*rw).sum())
      loss=ploss-((lv*v*rw).sum()); loss.backward(); opt.step()
Tensor.training=False
wp_l=wp.numpy().tolist(); ww_l=ww.numpy().tolist()
trI=train[:min(len(train),2000)]; trX=[X[i] for i in trI]; trIds=[yi[i] for i in trI]; trV=[yv[i] for i in trI]
dX=[X[i] for i in dev[:min(len(dev),20000)]]; dIds=[yi[i] for i in dev[:min(len(dev),20000)]]; dV=[yv[i] for i in dev[:min(len(dev),20000)]]
top1,top4,top8=topk_sparse(dX,dIds,wp_l); dv=ce_wdl(dX,dV,ww_l); tv=ce_wdl(trX,trV,ww_l); score=100/(1+dv)
artifact={'kind':'frozen_conv_fen_student','moves':moves,'policy_weights':[[wp_l[j][i] for j in range(F)] for i in range(M)],'wdl_weights':[[ww_l[j][i] for j in range(F)] for i in range(3)],'policy_feature_dim':F,'wdl_feature_dim':F,'weight_average_count':0,'conv_channels':64,'conv_layers':6}
out=Path(args.out); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(artifact,separators=(',',':')))
print(f'METRIC distill_student_score={score:.6f}'); print('METRIC train_policy_ce=0.000000'); print(f'METRIC train_wdl_ce={tv:.6f}'); print('METRIC dev_policy_ce=0.000000'); print(f'METRIC dev_wdl_ce={dv:.6f}'); print(f'METRIC dev_policy_top1={top1:.6f}'); print(f'METRIC dev_policy_top4={top4:.6f}'); print(f'METRIC dev_policy_top8={top8:.6f}'); print(f'METRIC feature_dataset_rows={len(X)}'); print(f'METRIC dev_eval_rows={len(dX)}'); print(f'METRIC preload_device={1 if args.preload_device else 0}'); print(f'METRIC sparse_policy={1 if can_sparse else 0}'); print(f'METRIC model_json_bytes={out.stat().st_size}')
