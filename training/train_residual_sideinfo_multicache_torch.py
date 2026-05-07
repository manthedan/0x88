#!/usr/bin/env python3
from __future__ import annotations
import argparse,bisect,json,time
from pathlib import Path
import numpy as np

def main():
 p=argparse.ArgumentParser(description='Train residual policy/WDL with Maia-style side-info auxiliary heads from cache shards.')
 p.add_argument('--manifest',required=True); p.add_argument('--dev-cache',required=True); p.add_argument('--resume',default=''); p.add_argument('--out',required=True); p.add_argument('--checkpoint',default=''); p.add_argument('--onnx-out',default=''); p.add_argument('--meta-out',default='')
 p.add_argument('--epochs',type=int,default=1); p.add_argument('--max-steps',type=int,default=0); p.add_argument('--batch-size',type=int,default=2048); p.add_argument('--channels',type=int,default=64); p.add_argument('--blocks',type=int,default=6); p.add_argument('--lr',type=float,default=1e-4); p.add_argument('--device',default='cuda'); p.add_argument('--amp',action='store_true',help='Use CUDA automatic mixed precision')
 p.add_argument('--from-weight',type=float,default=0.05); p.add_argument('--to-weight',type=float,default=0.05); p.add_argument('--piece-weight',type=float,default=0.02); p.add_argument('--check-weight',type=float,default=0.01); p.add_argument('--winrate-loss-weight',type=float,default=0.0); p.add_argument('--blunder-weight',type=float,default=0.0); p.add_argument('--max-dev-policy-ce',type=float,default=99.0)
 args=p.parse_args()
 import torch, torch.nn as nn, torch.nn.functional as F
 man=json.loads(Path(args.manifest).read_text()); paths=[Path(s) for s in man['shards']]; metas=[json.loads((d/'meta.json').read_text()) for d in paths]
 if not all(m.get('has_side_info') for m in metas): raise SystemExit('all cache shards must be built with --side-info')
 C=metas[0]['input_planes']; P=metas[0]['policy_size']; moves=metas[0]['moves']; sizes=[m['rows'] for m in metas]; offs=[]; total=0; shards=[]
 for d,n in zip(paths,sizes):
  offs.append(total); total+=n; wr=np.memmap(d/'stockfish_winrate_loss.float32',np.float32,'r',shape=(n,)) if (d/'stockfish_winrate_loss.float32').exists() else np.full(n,np.nan,np.float32); bb=np.memmap(d/'stockfish_blunder_bucket.int64',np.int64,'r',shape=(n,)) if (d/'stockfish_blunder_bucket.int64').exists() else np.full(n,-1,np.int64); shards.append((np.memmap(d/'x.int8',np.int8,'r',shape=(n,C,8,8)),np.memmap(d/'policy.int64',np.int64,'r',shape=(n,)),np.memmap(d/'wdl.float32',np.float32,'r',shape=(n,3)),np.memmap(d/'weight.float32',np.float32,'r',shape=(n,)),np.memmap(d/'from_square.int64',np.int64,'r',shape=(n,)),np.memmap(d/'to_square.int64',np.int64,'r',shape=(n,)),np.memmap(d/'moved_piece.int64',np.int64,'r',shape=(n,)),np.memmap(d/'captured_piece.int64',np.int64,'r',shape=(n,)),np.memmap(d/'gives_check.float32',np.float32,'r',shape=(n,)),wr,bb))
 dc=Path(args.dev_cache); dm=json.loads((dc/'meta.json').read_text()); DN=dm['rows']; dx=np.memmap(dc/'x.int8',np.int8,'r',shape=(DN,C,8,8)); dy=np.memmap(dc/'policy.int64',np.int64,'r',shape=(DN,)); dv=np.memmap(dc/'wdl.float32',np.float32,'r',shape=(DN,3))
 device=args.device
 class B(nn.Module):
  def __init__(self,ch): super().__init__(); self.c1=nn.Conv2d(ch,ch,3,padding=1); self.c2=nn.Conv2d(ch,ch,3,padding=1)
  def forward(self,z): return F.relu(self.c2(F.relu(self.c1(z)))+z)
 class Net(nn.Module):
  def __init__(self):
   super().__init__(); ch=args.channels; self.stem=nn.Conv2d(C,ch,3,padding=1); self.blocks=nn.Sequential(*[B(ch) for _ in range(args.blocks)]); self.policy=nn.Linear(ch*64,P); self.wdl=nn.Linear(ch,3); self.from_head=nn.Linear(ch*64,64); self.to_head=nn.Linear(ch*64,64); self.moved_head=nn.Linear(ch,6); self.captured_head=nn.Linear(ch,7); self.check_head=nn.Linear(ch,1); self.winrate_loss_head=nn.Linear(ch,1); self.blunder_head=nn.Linear(ch,4)
  def forward(self,z):
   h=self.blocks(F.relu(self.stem(z))); flat=h.flatten(1); pooled=h.mean((2,3)); return self.policy(flat),self.wdl(pooled),self.from_head(flat),self.to_head(flat),self.moved_head(pooled),self.captured_head(pooled),self.check_head(pooled).squeeze(1),self.winrate_loss_head(pooled).squeeze(1),self.blunder_head(pooled)
 net=Net().to(device); opt=torch.optim.AdamW(net.parameters(),lr=args.lr); amp_enabled=bool(args.amp and str(device).startswith('cuda')); scaler=torch.cuda.amp.GradScaler(enabled=amp_enabled); print(f'METRIC amp_enabled={1 if amp_enabled else 0}')
 if args.resume:
  ck=torch.load(args.resume,map_location=device); st=ck['model'] if isinstance(ck,dict) and 'model' in ck else ck; miss=net.load_state_dict(st,strict=False); print(f'METRIC resumed=1'); print(f'METRIC resume_missing={len(miss.missing_keys)}'); print(f'METRIC resume_unexpected={len(miss.unexpected_keys)}')
 else: print('METRIC resumed=0')
 def gather(ids):
  xs=[]; ys=[]; vs=[]; ws=[]; fs=[]; ts=[]; ms=[]; cs=[]; gs=[]; wrs=[]; bbs=[]
  for gid in ids:
   si=bisect.bisect_right(offs,int(gid))-1; li=int(gid)-offs[si]; x,y,v,w,f,t,m,c,g,wr,bb=shards[si]; xs.append(x[li]); ys.append(y[li]); vs.append(v[li]); ws.append(w[li]); fs.append(f[li]); ts.append(t[li]); ms.append(m[li]); cs.append(c[li]+1); gs.append(g[li]); wrs.append(wr[li]); bbs.append(bb[li])
  return map(np.asarray,(xs,ys,vs,ws,fs,ts,ms,cs,gs,wrs,bbs))
 def eval_dev(tag):
  net.eval(); ce=wc=t1=t4=t8=seen=0
  with torch.no_grad():
   for off in range(0,DN,args.batch_size):
    xb=torch.from_numpy(np.asarray(dx[off:off+args.batch_size])).to(device,dtype=torch.float32); yb=torch.from_numpy(np.asarray(dy[off:off+args.batch_size])).to(device); vb=torch.from_numpy(np.asarray(dv[off:off+args.batch_size])).to(device); pl,wl,*_=net(xb); bs=len(yb); ce+=float(F.cross_entropy(pl,yb,reduction='sum')); wc+=float((-(F.log_softmax(wl,1)*vb).sum(1)).sum()); pred=pl.topk(8,1).indices; t1+=int((pred[:,:1]==yb[:,None]).any(1).sum()); t4+=int((pred[:,:4]==yb[:,None]).any(1).sum()); t8+=int((pred==yb[:,None]).any(1).sum()); seen+=bs
  vals={'dev_policy_ce':ce/seen,'dev_wdl_ce':wc/seen,'dev_policy_top1':t1/seen,'dev_policy_top4':t4/seen,'dev_policy_top8':t8/seen}
  for k,v in vals.items(): print(f'METRIC {tag}_{k}={v:.6f}',flush=True)
  return vals
 gen=torch.Generator().manual_seed(17); last=None
 for ep in range(1,args.epochs+1):
  perm=torch.randperm(total,generator=gen); net.train(); seen=steps=0; loss_sum=0.0; t0=time.time()
  for off in range(0,total,args.batch_size):
   xb0,yb0,vb0,wb0,fs0,ts0,ms0,cs0,gs0,wr0,bb0=gather(perm[off:off+args.batch_size].numpy()); xb=torch.from_numpy(xb0).to(device,dtype=torch.float32); yb=torch.from_numpy(yb0).to(device); vb=torch.from_numpy(vb0).to(device); wb=torch.from_numpy(wb0).to(device); fs=torch.from_numpy(fs0).to(device); ts=torch.from_numpy(ts0).to(device); ms=torch.from_numpy(ms0).to(device); cs=torch.from_numpy(cs0).to(device); gs=torch.from_numpy(gs0).to(device,dtype=torch.float32); wr=torch.from_numpy(wr0).to(device,dtype=torch.float32); bb=torch.from_numpy(bb0).to(device)
   with torch.cuda.amp.autocast(enabled=amp_enabled):
    pl,wl,fl,tl,ml,cl,gl,wrl,bbl=net(xb); main=(F.cross_entropy(pl,yb,reduction='none')*wb).mean()+(-(F.log_softmax(wl,1)*vb).sum(1)*wb).mean(); aux=args.from_weight*F.cross_entropy(fl,fs)+args.to_weight*F.cross_entropy(tl,ts)+args.piece_weight*(F.cross_entropy(ml,ms)+F.cross_entropy(cl,cs))+args.check_weight*F.binary_cross_entropy_with_logits(gl,gs); wrm=torch.isfinite(wr); bbm=bb>=0; wr_aux=F.mse_loss(torch.sigmoid(wrl[wrm]),wr[wrm]) if wrm.any() else torch.tensor(0.0,device=device); bb_aux=F.cross_entropy(bbl[bbm],bb[bbm]) if bbm.any() else torch.tensor(0.0,device=device); loss=main+aux+args.winrate_loss_weight*wr_aux+args.blunder_weight*bb_aux
   opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); loss_sum+=float(loss.detach())*len(yb); seen+=len(yb); steps+=1
   if args.max_steps and steps>=args.max_steps: break
  print(f'METRIC epoch_{ep}_loss={loss_sum/max(1,seen):.6f}'); print(f'METRIC epoch_{ep}_seconds={time.time()-t0:.3f}',flush=True); last=eval_dev(f'epoch_{ep}')
  if args.checkpoint: Path(args.checkpoint).parent.mkdir(parents=True,exist_ok=True); torch.save({'model':net.state_dict(),'opt':opt.state_dict(),'epoch':ep,'dev_metrics':last},args.checkpoint)
  if last['dev_policy_ce']>args.max_dev_policy_ce: break
 meta={'kind':'tiny_board_residual_sideinfo','architecture':'residual_tower_sideinfo','moves':moves,'channels':args.channels,'blocks':args.blocks,'history_plies':metas[0]['history_plies'],'input_planes':C}
 Path(args.out).parent.mkdir(parents=True,exist_ok=True); torch.save({'model':net.state_dict(),'meta':meta},args.out)
 if args.meta_out: Path(args.meta_out).write_text(json.dumps(meta,separators=(',',':')))
if __name__=='__main__': main()
