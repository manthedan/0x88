#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np

def main():
 p=argparse.ArgumentParser(); p.add_argument('--cache',required=True); p.add_argument('--dev-cache',required=True); p.add_argument('--resume',required=True); p.add_argument('--out',required=True); p.add_argument('--onnx-out',required=True); p.add_argument('--meta-out',required=True); p.add_argument('--checkpoint',default=''); p.add_argument('--epochs',type=int,default=1); p.add_argument('--batch-size',type=int,default=2048); p.add_argument('--channels',type=int,default=48); p.add_argument('--blocks',type=int,default=5); p.add_argument('--lr',type=float,default=1e-5); p.add_argument('--device',default='cuda'); p.add_argument('--aux-q-weight',type=float,default=0.1); p.add_argument('--max-dev-policy-ce',type=float,default=2.85); args=p.parse_args()
 import torch, torch.nn as nn, torch.nn.functional as F
 c=Path(args.cache); m=json.loads((c/'meta.json').read_text()); N=m['rows']; C=m['input_planes']; P=m['policy_size']; devm=json.loads((Path(args.dev_cache)/'meta.json').read_text()); DN=devm['rows']; device=args.device
 x=np.memmap(c/'x.int8',np.int8,'r',shape=(N,C,8,8)); y=np.memmap(c/'policy.int64',np.int64,'r',shape=(N,)); v=np.memmap(c/'wdl.float32',np.float32,'r',shape=(N,3)); w=np.memmap(c/'weight.float32',np.float32,'r',shape=(N,)); sfq=np.memmap(c/'stockfish_q.float32',np.float32,'r',shape=(N,))
 dx=np.memmap(Path(args.dev_cache)/'x.int8',np.int8,'r',shape=(DN,C,8,8)); dy=np.memmap(Path(args.dev_cache)/'policy.int64',np.int64,'r',shape=(DN,)); dv=np.memmap(Path(args.dev_cache)/'wdl.float32',np.float32,'r',shape=(DN,3))
 class B(nn.Module):
  def __init__(self,C): super().__init__(); self.c1=nn.Conv2d(C,C,3,padding=1); self.c2=nn.Conv2d(C,C,3,padding=1)
  def forward(self,z): return F.relu(self.c2(F.relu(self.c1(z)))+z)
 class Net(nn.Module):
  def __init__(self): super().__init__(); self.stem=nn.Conv2d(C,args.channels,3,padding=1); self.blocks=nn.Sequential(*[B(args.channels) for _ in range(args.blocks)]); self.policy=nn.Linear(args.channels*64,P); self.wdl=nn.Linear(args.channels,3)
  def forward(self,z): h=self.blocks(F.relu(self.stem(z))); return self.policy(h.flatten(1)), self.wdl(h.mean((2,3)))
 net=Net().to(device); opt=torch.optim.AdamW(net.parameters(),lr=args.lr); ck=torch.load(args.resume,map_location=device); net.load_state_dict(ck['model'])
 def eval_dev(ep):
  net.eval(); ce=wc=t1=t4=t8=seen=0
  with torch.no_grad():
   for off in range(0,DN,args.batch_size):
    xb=torch.from_numpy(np.asarray(dx[off:off+args.batch_size])).to(device,dtype=torch.float32); yb=torch.from_numpy(np.asarray(dy[off:off+args.batch_size])).to(device); vb=torch.from_numpy(np.asarray(dv[off:off+args.batch_size])).to(device); pl,wl=net(xb); bs=len(yb); ce+=float(F.cross_entropy(pl,yb,reduction='sum')); wc+=float((-(F.log_softmax(wl,1)*vb).sum(1)).sum()); pred=pl.topk(8,1).indices; t1+=int((pred[:,:1]==yb[:,None]).any(1).sum()); t4+=int((pred[:,:4]==yb[:,None]).any(1).sum()); t8+=int((pred==yb[:,None]).any(1).sum()); seen+=bs
  vals={'dev_policy_ce':ce/seen,'dev_wdl_ce':wc/seen,'dev_policy_top1':t1/seen,'dev_policy_top4':t4/seen,'dev_policy_top8':t8/seen}
  for k,vv in vals.items(): print(f'METRIC epoch_{ep}_{k}={vv:.6f}',flush=True)
  return vals
 gen=torch.Generator().manual_seed(11)
 for ep in range(1,args.epochs+1):
  net.train(); perm=torch.randperm(N,generator=gen); total=seen=0; t=time.time()
  for off in range(0,N,args.batch_size):
   ids=perm[off:off+args.batch_size].numpy(); xb=torch.from_numpy(np.asarray(x[ids])).to(device,dtype=torch.float32); yb=torch.from_numpy(np.asarray(y[ids])).to(device); vb=torch.from_numpy(np.asarray(v[ids])).to(device); wb=torch.from_numpy(np.asarray(w[ids])).to(device); qb=torch.from_numpy(np.asarray(sfq[ids])).to(device)
   pl,wl=net(xb); probs=F.softmax(wl,1); predq=probs[:,0]-probs[:,2]; mask=torch.isfinite(qb); aux=F.mse_loss(predq[mask],qb[mask]) if mask.any() else 0; loss=(F.cross_entropy(pl,yb,reduction='none')*wb).mean()+(-(F.log_softmax(wl,1)*vb).sum(1)*wb).mean()+args.aux_q_weight*aux
   opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); total+=float(loss.detach())*len(yb); seen+=len(yb)
  print(f'METRIC epoch_{ep}_loss={total/seen:.6f}'); print(f'METRIC epoch_{ep}_seconds={time.time()-t:.3f}',flush=True); vals=eval_dev(ep)
  if args.checkpoint: torch.save({'model':net.state_dict(),'opt':opt.state_dict(),'epoch':ep,'dev_metrics':vals},args.checkpoint)
  if vals['dev_policy_ce']>args.max_dev_policy_ce: print(f'METRIC stopped_dev_policy_ce={vals["dev_policy_ce"]:.6f}',flush=True); break
 meta={'kind':'tiny_board_residual_onnx_student','architecture':'residual_tower','policy_map':'uci_queen_knight_promo_v1','moves':m['moves'],'channels':args.channels,'blocks':args.blocks,'history_plies':m['history_plies'],'input_planes':C,'onnx':args.onnx_out}
 Path(args.out).parent.mkdir(parents=True,exist_ok=True); torch.save({'model':net.state_dict(),'meta':meta},args.out); Path(args.meta_out).write_text(json.dumps(meta,separators=(',',':'))); torch.onnx.export(net,torch.zeros(1,C,8,8,device=device),args.onnx_out,input_names=['planes'],output_names=['policy_logits','wdl_logits'],dynamic_axes={'planes':{0:'batch'},'policy_logits':{0:'batch'},'wdl_logits':{0:'batch'}},opset_version=18,external_data=False)
if __name__=='__main__': main()
