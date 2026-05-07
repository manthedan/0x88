#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,subprocess
from pathlib import Path
import numpy as np
from train_residual_torch import fixed_policy_moves,planes,input_plane_count
try:
 import pyzstd  # type: ignore
except Exception:
 pyzstd=None

PIECES='PNBRQKpnbrqk'

def opener(path):
 path=str(path)
 if path.endswith('.zst'):
  if pyzstd is not None: return pyzstd.open(path,'rt')
  p=subprocess.Popen(['zstdcat',path],stdout=subprocess.PIPE,text=True)
  return p.stdout
 return open(path)

def current_board_18(fen):
 parts=fen.split(); board=parts[0]; side=parts[1] if len(parts)>1 else 'w'; castling=parts[2] if len(parts)>2 else '-'; ep=parts[3] if len(parts)>3 else '-'
 x=np.zeros((18,8,8),dtype=np.int8); r=f=0
 for ch in board:
  if ch=='/': r+=1; f=0
  elif ch.isdigit(): f+=int(ch)
  else: x[PIECES.index(ch),r,f]=1; f+=1
 x[12,:,:]=1 if side=='w' else 0
 for i,flag in enumerate(['K','Q','k','q']):
  if flag in castling: x[13+i,:,:]=1
 if ep!='-' and len(ep)>=2:
  ef=ord(ep[0])-97; er=8-int(ep[1])
  if 0<=er<8 and 0<=ef<8: x[17,er,ef]=1
 return x

def count_rows(paths,moves,max_rows):
 mid=set(moves); n=sk=0
 for path in paths:
  with opener(path) as f:
   for line in f:
    if n>=max_rows: return n,sk
    r=json.loads(line); pol=r.get('policy',{})
    if len(pol)!=1: continue
    mv=next(iter(pol))
    if mv not in mid: sk+=1; continue
    n+=1
 return n,sk

def main():
 a=argparse.ArgumentParser(); a.add_argument('--input',nargs='+',required=True); a.add_argument('--out',required=True); a.add_argument('--max-rows',type=int,default=10**12); a.add_argument('--history-plies',type=int,default=2); a.add_argument('--state-planes',action='store_true'); a.add_argument('--current-board-18',action='store_true',help='Use Maia-like 18 planes: pieces, side, castling, ep; ignores history/state args'); a.add_argument('--side-info',action='store_true',help='Write Maia-style aux labels: from/to/moved/captured/check') ; args=a.parse_args()
 moves=fixed_policy_moves(); mid={m:i for i,m in enumerate(moves)}; C=18 if args.current_board_18 else input_plane_count(args.history_plies,args.state_planes); n,sk=count_rows(args.input,moves,args.max_rows)
 chess=None
 if args.side_info:
  try:
   import chess as _chess
   chess=_chess
  except Exception as e: raise SystemExit('--side-info requires python-chess: pip install chess') from e
 out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
 x=np.memmap(out/'x.int8',np.int8,'w+',shape=(n,C,8,8)); y=np.memmap(out/'policy.int64',np.int64,'w+',shape=(n,)); v=np.memmap(out/'wdl.float32',np.float32,'w+',shape=(n,3)); w=np.memmap(out/'weight.float32',np.float32,'w+',shape=(n,)); sfq=np.memmap(out/'stockfish_q.float32',np.float32,'w+',shape=(n,)); wr_loss=np.memmap(out/'stockfish_winrate_loss.float32',np.float32,'w+',shape=(n,)); blunder=np.memmap(out/'stockfish_blunder_bucket.int64',np.int64,'w+',shape=(n,))
 if args.side_info:
  from_sq=np.memmap(out/'from_square.int64',np.int64,'w+',shape=(n,)); to_sq=np.memmap(out/'to_square.int64',np.int64,'w+',shape=(n,)); moved_piece=np.memmap(out/'moved_piece.int64',np.int64,'w+',shape=(n,)); captured_piece=np.memmap(out/'captured_piece.int64',np.int64,'w+',shape=(n,)); gives_check=np.memmap(out/'gives_check.float32',np.float32,'w+',shape=(n,))
 i=0
 for path in args.input:
  with opener(path) as f:
   for line in f:
    if i>=n: break
    r=json.loads(line); pol=r.get('policy',{})
    if len(pol)!=1: continue
    mv=next(iter(pol))
    if mv not in mid: continue
    x[i]=current_board_18(r['fen']) if args.current_board_18 else np.asarray(planes(r['fen'],r.get('history_fens',[])[:args.history_plies],args.history_plies,args.state_planes),dtype=np.int8); y[i]=mid[mv]; v[i]=np.asarray(r.get('wdl',[.25,.5,.25]),dtype=np.float32); w[i]=float(r.get('weight',1.0)); sfq[i]=float(r.get('stockfish_q',np.nan)); wr=r.get('stockfish_winrate_loss',None); wr_loss[i]=float(wr) if wr is not None else np.nan; blunder[i]=int(r.get('stockfish_blunder_bucket',-1))
    if args.side_info:
     b=chess.Board(r['fen']); m=chess.Move.from_uci(mv); pc=b.piece_at(m.from_square); cap=b.piece_at(m.to_square); ep_cap=(pc is not None and pc.piece_type==chess.PAWN and b.ep_square==m.to_square and cap is None and chess.square_file(m.from_square)!=chess.square_file(m.to_square))
     if ep_cap: cap=b.piece_at(chess.square(chess.square_file(m.to_square),chess.square_rank(m.from_square)))
     from_sq[i]=m.from_square; to_sq[i]=m.to_square; moved_piece[i]=(pc.piece_type-1 if pc else -1); captured_piece[i]=(cap.piece_type-1 if cap else -1); gives_check[i]=1.0 if b.gives_check(m) else 0.0
    i+=1
    if i%100000==0: print(f'METRIC cache_rows_written={i}',flush=True)
 arrays=[x,y,v,w,sfq,wr_loss,blunder]
 if args.side_info: arrays += [from_sq,to_sq,moved_piece,captured_piece,gives_check]
 for arr in arrays: arr.flush()
 (out/'meta.json').write_text(json.dumps({'rows':n,'input_planes':C,'history_plies':0 if args.current_board_18 else args.history_plies,'state_planes':False if args.current_board_18 else args.state_planes,'input_mode':'current_board_18' if args.current_board_18 else 'history','policy_size':len(moves),'moves':moves,'skipped_unknown_moves':sk,'has_stockfish_q':True,'has_stockfish_winrate_loss':True,'has_side_info':args.side_info},separators=(',',':')))
 print(f'METRIC cache_rows={n}'); print(f'METRIC cache_input_planes={C}'); print(f'METRIC cache_skipped_unknown_moves={sk}')
if __name__=='__main__': main()
