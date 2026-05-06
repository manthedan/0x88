#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math, random, pickle
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('--train', nargs='+', required=True); p.add_argument('--out', required=True); p.add_argument('--max-rows', type=int, default=50000); p.add_argument('--epochs', type=int, default=10); p.add_argument('--channels', type=int, default=32); p.add_argument('--lr', type=float, default=1e-3); p.add_argument('--holdout-mod', type=int, default=5); p.add_argument('--eval-rows', type=int, default=10000); p.add_argument('--policy-head', choices=['pooled','spatial'], default='spatial'); p.add_argument('--state-planes', action='store_true'); p.add_argument('--history-plies', type=int, default=0); p.add_argument('--fixed-policy-map', action='store_true'); p.add_argument('--checkpoint', default=''); p.add_argument('--checkpoint-every', type=int, default=1); p.add_argument('--resume', default=''); args=p.parse_args()
if args.history_plies < 0: raise SystemExit('--history-plies must be >= 0')
from tinygrad import Tensor
from tinygrad.nn import Conv2d, Linear
from tinygrad.nn.optim import Adam
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
def in_check_grid(grid, side):
  kr=kf=-1; king='K' if side=='w' else 'k'; enemy=lambda ch: ch!='.' and (ch.islower() if side=='w' else ch.isupper())
  for r in range(8):
    for f in range(8):
      if grid[r][f]==king: kr,kf=r,f
  if kr<0: return False
  pawn_dirs=[(-1,-1),(-1,1)] if side=='w' else [(1,-1),(1,1)]
  for dr,df in pawn_dirs:
    r,f=kr+dr,kf+df
    if 0<=r<8 and 0<=f<8 and grid[r][f]==('p' if side=='w' else 'P'): return True
  for dr,df in [(1,2),(2,1),(-1,2),(-2,1),(1,-2),(2,-1),(-1,-2),(-2,-1)]:
    r,f=kr+dr,kf+df
    if 0<=r<8 and 0<=f<8 and grid[r][f]==('n' if side=='w' else 'N'): return True
  for dr,df,pieces in [(1,0,'rq'),(-1,0,'rq'),(0,1,'rq'),(0,-1,'rq'),(1,1,'bq'),(1,-1,'bq'),(-1,1,'bq'),(-1,-1,'bq')]:
    r,f=kr+dr,kf+df
    while 0<=r<8 and 0<=f<8:
      ch=grid[r][f]
      if ch!='.':
        if enemy(ch) and ch.lower() in pieces: return True
        break
      r+=dr; f+=df
  for dr in [-1,0,1]:
    for df in [-1,0,1]:
      if dr or df:
        r,f=kr+dr,kf+df
        if 0<=r<8 and 0<=f<8 and grid[r][f]==('k' if side=='w' else 'K'): return True
  return False
def input_plane_count(): return 12*(args.history_plies+1) + (10 if args.state_planes else 2)
def add_piece_planes(x, fen, offset):
  board=fen.split()[0]; grid=[['.']*8 for _ in range(8)]; r=f=0
  for ch in board:
    if ch=='/': r+=1; f=0
    elif ch.isdigit(): f+=int(ch)
    else: x[offset+PIECES.index(ch)][r][f]=1.0; grid[r][f]=ch; f+=1
  return grid

def planes(fen, history_fens=None):
  history_fens=history_fens or []
  parts=fen.split(); side=parts[1] if len(parts)>1 else 'w'; castling=parts[2] if len(parts)>2 else '-'; ep=parts[3] if len(parts)>3 else '-'; n=input_plane_count(); x=[[[0.0]*8 for _ in range(8)] for _ in range(n)]
  grid=add_piece_planes(x, fen, 0)
  for h in range(args.history_plies):
    if h < len(history_fens): add_piece_planes(x, history_fens[h], 12*(h+1))
  state0=12*(args.history_plies+1); sv=1.0 if side=='w' else -1.0
  for rr in range(8):
    for ff in range(8): x[state0][rr][ff]=sv
  if args.state_planes:
    flags=['K','Q','k','q']
    for i,flag in enumerate(flags):
      if flag in castling:
        for rr in range(8):
          for ff in range(8): x[state0+1+i][rr][ff]=1.0
    if ep!='-' and len(ep)>=2:
      ef=ord(ep[0])-97; er=8-int(ep[1])
      if 0<=er<8 and 0<=ef<8: x[state0+5][er][ef]=1.0
    stm_check=1.0 if in_check_grid(grid,side) else 0.0; opp_check=1.0 if in_check_grid(grid,'b' if side=='w' else 'w') else 0.0
    for rr in range(8):
      for ff in range(8): x[state0+6][rr][ff]=1.0; x[state0+7][rr][ff]=1.0 if side=='w' else 0.0; x[state0+8][rr][ff]=stm_check; x[state0+9][rr][ff]=opp_check
  else:
    for rr in range(8):
      for ff in range(8): x[state0+1][rr][ff]=1.0
  return x
resume_ck=None; fixed_moves=fixed_policy_moves() if args.fixed_policy_map else None; policy_map='uci_queen_knight_promo_v1' if args.fixed_policy_map else None
if args.resume:
  with open(args.resume,'rb') as f: resume_ck=pickle.load(f)
  fixed_moves=resume_ck.get('moves') or fixed_moves
  policy_map=resume_ck.get('policy_map') or policy_map
  if fixed_moves: print(f'METRIC fixed_resume_moves={len(fixed_moves)}')
def read_rows(paths):
  rows=[]; seen_moves=[]; skipped_unknown=0
  fixed_mid={m:i for i,m in enumerate(fixed_moves or [])}
  for path in paths:
    with open(path) as f:
      for line in f:
        if len(rows)>=args.max_rows: return rows, (list(fixed_moves) if fixed_moves else sorted(set(seen_moves))), skipped_unknown
        r=json.loads(line); pol=r.get('policy',{})
        if len(pol)!=1: continue
        mv=next(iter(pol))
        if fixed_moves and mv not in fixed_mid:
          skipped_unknown+=1; continue
        rows.append((r['fen'], mv, r.get('wdl',[.25,.5,.25]), float(r.get('weight',1.0)), r.get('history_fens', [])[:args.history_plies])); seen_moves.append(mv)
  return rows, (list(fixed_moves) if fixed_moves else sorted(set(seen_moves))), skipped_unknown
rows,moves,skipped_unknown=read_rows(args.train); mid={m:i for i,m in enumerate(moves)}; rows=[r for r in rows if r[1] in mid]
print(f'METRIC board_cnn_skipped_unknown_moves={skipped_unknown}')
train=[i for i in range(len(rows)) if i%args.holdout_mod!=0]; dev=[i for i in range(len(rows)) if i%args.holdout_mod==0]
class Net:
  def __init__(self):
    C=args.channels; self.c1=Conv2d(input_plane_count(),C,3,padding=1); self.c2=Conv2d(C,C,3,padding=1); self.c3=Conv2d(C,C,3,padding=1); self.p=Linear(C*64 if args.policy_head=='spatial' else C,len(moves)); self.v=Linear(C,3)
  def __call__(self,x):
    h=self.c1(x).relu(); h=(self.c2(h).relu()+h); h=(self.c3(h).relu()+h); pooled=h.mean(axis=(2,3)); pf=h.reshape(h.shape[0], args.channels*64) if args.policy_head=='spatial' else pooled; return self.p(pf), self.v(pooled)
  def params(self): return [self.c1.weight,self.c1.bias,self.c2.weight,self.c2.bias,self.c3.weight,self.c3.bias,self.p.weight,self.p.bias,self.v.weight,self.v.bias]
net=Net(); opt=Adam(net.params(), lr=args.lr); rng=random.Random(7); start_epoch=0
if resume_ck:
  ck=resume_ck
  for name in ['c1','c2','c3']:
    layer=getattr(net,name); layer.weight.assign(Tensor(ck[name+'_weight'])); layer.bias.assign(Tensor(ck[name+'_bias']))
  net.p.weight.assign(Tensor(ck['policy_weight'])); net.p.bias.assign(Tensor(ck['policy_bias'])); net.v.weight.assign(Tensor(ck['wdl_weight'])); net.v.bias.assign(Tensor(ck['wdl_bias']))
  start_epoch=int(ck.get('epoch',0)); print(f'METRIC resumed_epoch={start_epoch}')
Tensor.training=True
for ep in range(start_epoch, args.epochs):
  rng.shuffle(train); total=0.0; n=0
  for off in range(0,len(train),512):
    idx=train[off:off+512]; x=Tensor([planes(rows[i][0], rows[i][4]) for i in idx]); y=Tensor([mid[rows[i][1]] for i in idx]); v=Tensor([rows[i][2] for i in idx]); w=Tensor([rows[i][3] for i in idx]).reshape(len(idx),1)
    opt.zero_grad(); lp,lv=net(x); lps=lp.log_softmax(); lvs=lv.log_softmax(); loss=lps.sparse_categorical_crossentropy(y,reduction='none').reshape(len(idx),1).mul(w).sum()-((lvs*v*w).sum()); loss.backward(); opt.step(); total+=float(loss.numpy()); n+=len(idx)
  print(f'METRIC epoch_{ep+1}_loss={total/max(1,n):.6f}', flush=True)
  if args.checkpoint and args.checkpoint_every > 0 and (ep + 1) % args.checkpoint_every == 0:
    ck={'epoch':ep+1,'moves':moves,'channels':args.channels,'policy_head':args.policy_head,'policy_map':policy_map,'history_plies':args.history_plies,'input_planes':input_plane_count(),
        'c1_weight':net.c1.weight.numpy().tolist(),'c1_bias':net.c1.bias.numpy().tolist(),
        'c2_weight':net.c2.weight.numpy().tolist(),'c2_bias':net.c2.bias.numpy().tolist(),
        'c3_weight':net.c3.weight.numpy().tolist(),'c3_bias':net.c3.bias.numpy().tolist(),
        'policy_weight':net.p.weight.numpy().tolist(),'policy_bias':net.p.bias.numpy().tolist(),
        'wdl_weight':net.v.weight.numpy().tolist(),'wdl_bias':net.v.bias.numpy().tolist()}
    Path(args.checkpoint).parent.mkdir(parents=True,exist_ok=True)
    with open(args.checkpoint,'wb') as f: pickle.dump(ck,f)
    print(f'METRIC checkpoint_epoch={ep+1}', flush=True)
Tensor.training=False
n=min(args.eval_rows,len(dev)); top1=top4=top8=0; pce=wce=0.0
for off in range(0,n,512):
  idx=dev[off:off+512]; x=Tensor([planes(rows[i][0], rows[i][4]) for i in idx]); lp,lv=net(x); logits=lp.numpy(); wlog=lv.numpy()
  for row,wl,i in zip(logits,wlog,idx):
    t=mid[rows[i][1]]; ranked=sorted(range(len(row)), key=lambda k: row[k], reverse=True); top1+=t==ranked[0]; top4+=t in ranked[:4]; top8+=t in ranked[:8]
    m=max(row); pce+=-(row[t]-m-math.log(sum(math.exp(z-m) for z in row))); mw=max(wl); vt=rows[i][2]; wce+=-sum(vt[k]*(wl[k]-mw-math.log(sum(math.exp(z-mw) for z in wl))) for k in range(3))
obj={'kind':'tiny_board_cnn_student','moves':moves,'channels':args.channels,'policy_head':args.policy_head,'policy_map':policy_map,'history_plies':args.history_plies,'input_planes':input_plane_count(),
     'c1_weight':net.c1.weight.numpy().tolist(),'c1_bias':net.c1.bias.numpy().tolist(),
     'c2_weight':net.c2.weight.numpy().tolist(),'c2_bias':net.c2.bias.numpy().tolist(),
     'c3_weight':net.c3.weight.numpy().tolist(),'c3_bias':net.c3.bias.numpy().tolist(),
     'policy_weight':net.p.weight.numpy().tolist(),'policy_bias':net.p.bias.numpy().tolist(),
     'wdl_weight':net.v.weight.numpy().tolist(),'wdl_bias':net.v.bias.numpy().tolist()}
out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(obj))
print(f'METRIC board_cnn_rows={len(rows)}'); print(f'METRIC board_cnn_moves={len(moves)}'); print(f'METRIC board_cnn_fixed_policy_map={1 if policy_map else 0}'); print(f'METRIC board_cnn_history_plies={args.history_plies}'); print(f'METRIC board_cnn_policy_head_spatial={1 if args.policy_head == "spatial" else 0}'); print(f'METRIC board_cnn_input_planes={input_plane_count()}'); print(f'METRIC dev_policy_ce={pce/n:.6f}'); print(f'METRIC dev_wdl_ce={wce/n:.6f}'); print(f'METRIC dev_policy_top1={top1/n:.6f}'); print(f'METRIC dev_policy_top4={top4/n:.6f}'); print(f'METRIC dev_policy_top8={top8/n:.6f}')
