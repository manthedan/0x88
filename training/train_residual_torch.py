#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math, random
from pathlib import Path

PIECES='PNBRQKpnbrqk'

def fixed_policy_moves():
  files='abcdefgh'; out=set(); dirs=[(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]; knights=[(1,2),(2,1),(-1,2),(-2,1),(1,-2),(2,-1),(-1,-2),(-2,-1)]
  on=lambda f,r: 0<=f<8 and 0<=r<8; sq=lambda f,r: files[f]+str(r+1)
  for r in range(8):
    for f in range(8):
      fr=sq(f,r)
      for df,dr in dirs:
        for n in range(1,8):
          tf,tr=f+df*n,r+dr*n
          if not on(tf,tr): break
          out.add(fr+sq(tf,tr))
      for df,dr in knights:
        if on(f+df,r+dr): out.add(fr+sq(f+df,r+dr))
  for r in [1,6]:
    tr=7 if r==6 else 0
    for f in range(8):
      for df in [-1,0,1]:
        if on(f+df,tr):
          for pr in 'qrbn': out.add(sq(f,r)+sq(f+df,tr)+pr)
  return sorted(out)

def input_plane_count(history_plies, state_planes): return 12*(history_plies+1) + (10 if state_planes else 2)

def add_piece_planes(x, fen, offset):
  board=fen.split()[0]; r=f=0
  for ch in board:
    if ch=='/': r+=1; f=0
    elif ch.isdigit(): f+=int(ch)
    else: x[offset+PIECES.index(ch)][r][f]=1.0; f+=1

def planes(fen, history_fens, history_plies, state_planes):
  parts=fen.split(); side=parts[1] if len(parts)>1 else 'w'; castling=parts[2] if len(parts)>2 else '-'; ep=parts[3] if len(parts)>3 else '-'
  x=[[[0.0]*8 for _ in range(8)] for _ in range(input_plane_count(history_plies,state_planes))]
  add_piece_planes(x, fen, 0)
  for h, hf in enumerate((history_fens or [])[:history_plies]): add_piece_planes(x, hf, 12*(h+1))
  s0=12*(history_plies+1); sv=1.0 if side=='w' else -1.0
  for r in range(8):
    for f in range(8): x[s0][r][f]=sv
  if state_planes:
    for i,flag in enumerate(['K','Q','k','q']):
      if flag in castling:
        for r in range(8):
          for f in range(8): x[s0+1+i][r][f]=1.0
    if ep!='-' and len(ep)>=2:
      ef=ord(ep[0])-97; er=8-int(ep[1])
      if 0<=er<8 and 0<=ef<8: x[s0+5][er][ef]=1.0
    for r in range(8):
      for f in range(8): x[s0+6][r][f]=1.0; x[s0+7][r][f]=1.0 if side=='w' else 0.0
  else:
    for r in range(8):
      for f in range(8): x[s0+1][r][f]=1.0
  return x

def read_rows(paths, moves, max_rows, history_plies):
  mid={m:i for i,m in enumerate(moves)}; rows=[]; skipped=0
  for path in paths:
    with open(path) as f:
      for line in f:
        if len(rows)>=max_rows: return rows, skipped
        r=json.loads(line); pol=r.get('policy',{})
        if len(pol)!=1: continue
        mv=next(iter(pol))
        if mv not in mid: skipped+=1; continue
        rows.append((r['fen'], mid[mv], r.get('wdl',[.25,.5,.25]), float(r.get('weight',1.0)), r.get('history_fens',[])[:history_plies]))
  return rows, skipped

def main():
  p=argparse.ArgumentParser(); p.add_argument('--train', nargs='+', required=True); p.add_argument('--dev', nargs='*', default=[]); p.add_argument('--out', required=True); p.add_argument('--onnx-out', default=''); p.add_argument('--meta-out', default=''); p.add_argument('--checkpoint', default=''); p.add_argument('--resume', default=''); p.add_argument('--max-rows', type=int, default=50000); p.add_argument('--epochs', type=int, default=5); p.add_argument('--batch-size', type=int, default=256); p.add_argument('--channels', type=int, default=48); p.add_argument('--blocks', type=int, default=5); p.add_argument('--lr', type=float, default=1e-3); p.add_argument('--history-plies', type=int, default=2); p.add_argument('--state-planes', action='store_true'); p.add_argument('--device', default='auto'); p.add_argument('--compile', action='store_true', help='Use torch.compile when available'); args=p.parse_args()
  try:
    import torch, torch.nn as nn, torch.nn.functional as F
  except Exception as e: raise SystemExit('PyTorch required: pip install -r requirements-onnx.txt') from e
  class Block(nn.Module):
    def __init__(self,C): super().__init__(); self.c1=nn.Conv2d(C,C,3,padding=1); self.c2=nn.Conv2d(C,C,3,padding=1)
    def forward(self,x): return F.relu(self.c2(F.relu(self.c1(x)))+x)
  class Net(nn.Module):
    def __init__(self):
      super().__init__(); C=args.channels; self.stem=nn.Conv2d(input_plane_count(args.history_plies,args.state_planes),C,3,padding=1); self.blocks=nn.Sequential(*[Block(C) for _ in range(args.blocks)]); self.policy=nn.Linear(C*64,len(moves)); self.wdl=nn.Linear(C,3)
    def forward(self,x):
      h=self.blocks(F.relu(self.stem(x))); return self.policy(h.flatten(1)), self.wdl(h.mean(dim=(2,3)))
  moves=fixed_policy_moves(); rows,skipped=read_rows(args.train,moves,args.max_rows,args.history_plies); dev_rows=[]
  if args.dev: dev_rows,_=read_rows(args.dev,moves,args.max_rows,args.history_plies)
  else: dev_rows=[r for i,r in enumerate(rows) if i%5==0]; rows=[r for i,r in enumerate(rows) if i%5!=0]
  device='cuda' if args.device=='auto' and torch.cuda.is_available() else ('cpu' if args.device=='auto' else args.device)
  net=Net().to(device)
  if args.compile:
    if hasattr(torch, 'compile'):
      net=torch.compile(net)
      print('METRIC torch_compile_enabled=1')
    else:
      print('METRIC torch_compile_enabled=0')
  opt=torch.optim.AdamW(net.parameters(), lr=args.lr); start=0
  if args.resume:
    ck=torch.load(args.resume,map_location=device); net.load_state_dict(ck['model']); opt.load_state_dict(ck['opt']); start=ck.get('epoch',0)
  rng=random.Random(7)
  for ep in range(start,args.epochs):
    rng.shuffle(rows); net.train(); total=0; seen=0
    for off in range(0,len(rows),args.batch_size):
      b=rows[off:off+args.batch_size]; x=torch.tensor([planes(r[0],r[4],args.history_plies,args.state_planes) for r in b],device=device); y=torch.tensor([r[1] for r in b],device=device); v=torch.tensor([r[2] for r in b],device=device); w=torch.tensor([r[3] for r in b],device=device)
      pl,wl=net(x); loss=(F.cross_entropy(pl,y,reduction='none')*w).mean() + (-(F.log_softmax(wl,1)*v).sum(1)*w).mean(); opt.zero_grad(); loss.backward(); opt.step(); total+=float(loss)*len(b); seen+=len(b)
    print(f'METRIC epoch_{ep+1}_loss={total/max(1,seen):.6f}', flush=True)
    if args.checkpoint: Path(args.checkpoint).parent.mkdir(parents=True,exist_ok=True); torch.save({'epoch':ep+1,'model':net.state_dict(),'opt':opt.state_dict(),'args':vars(args)},args.checkpoint)
  net.eval(); top1=top4=top8=pce=wce=0.0; n=0
  with torch.no_grad():
    for off in range(0,len(dev_rows),args.batch_size):
      b=dev_rows[off:off+args.batch_size]; x=torch.tensor([planes(r[0],r[4],args.history_plies,args.state_planes) for r in b],device=device); y=torch.tensor([r[1] for r in b],device=device); v=torch.tensor([r[2] for r in b],device=device); pl,wl=net(x); rank=pl.argsort(1,descending=True); top1+=(rank[:,0]==y).sum().item(); top4+=(rank[:,:4]==y[:,None]).any(1).sum().item(); top8+=(rank[:,:8]==y[:,None]).any(1).sum().item(); pce+=F.cross_entropy(pl,y,reduction='sum').item(); wce+=(-(F.log_softmax(wl,1)*v).sum()).item(); n+=len(b)
  meta={'kind':'tiny_board_residual_onnx_student','architecture':'residual_tower','policy_map':'uci_queen_knight_promo_v1','moves':moves,'channels':args.channels,'blocks':args.blocks,'history_plies':args.history_plies,'input_planes':input_plane_count(args.history_plies,args.state_planes),'onnx':args.onnx_out}
  Path(args.out).parent.mkdir(parents=True,exist_ok=True); torch.save({'model':net.state_dict(),'meta':meta},args.out)
  if args.meta_out or args.onnx_out: Path(args.meta_out or (args.onnx_out+'.meta.json')).write_text(json.dumps(meta,separators=(',',':')))
  if args.onnx_out:
    dummy=torch.zeros(1,meta['input_planes'],8,8,device=device); Path(args.onnx_out).parent.mkdir(parents=True,exist_ok=True); torch.onnx.export(net,dummy,args.onnx_out,input_names=['planes'],output_names=['policy_logits','wdl_logits'],dynamic_axes={'planes':{0:'batch'},'policy_logits':{0:'batch'},'wdl_logits':{0:'batch'}},opset_version=17)
  print(f'METRIC torch_rows={len(rows)}'); print(f'METRIC torch_skipped_unknown_moves={skipped}'); print(f'METRIC torch_input_planes={meta["input_planes"]}'); print(f'METRIC dev_policy_top1={top1/max(1,n):.6f}'); print(f'METRIC dev_policy_top4={top4/max(1,n):.6f}'); print(f'METRIC dev_policy_top8={top8/max(1,n):.6f}'); print(f'METRIC dev_policy_ce={pce/max(1,n):.6f}'); print(f'METRIC dev_wdl_ce={wce/max(1,n):.6f}')
if __name__=='__main__': main()
