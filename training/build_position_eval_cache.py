#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, subprocess, time
from pathlib import Path
import numpy as np

FILES='abcdefgh'; PIECES='.PNBRQKpnbrqk'; PROMOS={'n':0,'b':1,'r':2,'q':3}; POLICY_SIZE=4096+4096*4
SCHEMA_POSITION_EVAL='teacher.position_eval.v1'

def open_lines(path):
    path=str(path)
    if path.endswith('.zst'):
        p=subprocess.Popen(['zstd','-dc',path],stdout=subprocess.PIPE,text=True)
        try:
            assert p.stdout is not None
            for line in p.stdout: yield line
        finally:
            if p.stdout: p.stdout.close()
            rc=p.wait()
            if rc and rc!=-13: raise subprocess.CalledProcessError(rc,['zstd','-dc',path])
    else:
        with open(path,'rt',encoding='utf-8') as f: yield from f

def sq(s): return (int(s[1])-1)*8 + FILES.index(s[0])
def move_class(uci):
    fr=sq(uci[:2]); to=sq(uci[2:4])
    if len(uci)>=5 and uci[4].lower() in PROMOS: return 4096+(fr*64+to)*4+PROMOS[uci[4].lower()]
    return fr*64+to

def parse_board(fen):
    b=np.zeros(64,dtype=np.uint8); ranks=fen.split()[0].split('/')
    if len(ranks)!=8: raise ValueError('bad FEN board')
    for rr,rank in enumerate(ranks):
        f=0; r=7-rr
        for ch in rank:
            if ch.isdigit(): f+=int(ch)
            else: b[r*8+f]=PIECES.index(ch); f+=1
    return b

def encode(fen,hist,history):
    parts=fen.split(); stm=parts[1] if len(parts)>1 else 'w'; cast=parts[2] if len(parts)>2 else '-'; ep=parts[3] if len(parts)>3 else '-'; half=int(parts[4]) if len(parts)>4 and parts[4].isdigit() else 0
    F=history+9; out=np.zeros((64,F),dtype=np.uint8)
    boards=[parse_board(fen)]
    for h in (hist or [])[:history]: boards.append(parse_board(h))
    while len(boards)<history+1: boards.append(np.zeros(64,dtype=np.uint8))
    for i,b in enumerate(boards): out[:,i]=b
    out[:,history+1]=1 if stm=='w' else 2
    flags=(('K' in cast)<<0)|(('Q' in cast)<<1)|(('k' in cast)<<2)|(('q' in cast)<<3)
    out[:,history+2]=flags
    if len(ep)==2 and ep[0] in FILES and ep[1].isdigit(): out[sq(ep),history+3]=1
    out[:,history+4]=min(255,half)
    for i in range(64):
        r,f=divmod(i,8); out[i,history+5]=r; out[i,history+6]=f; out[i,history+7]=1 if ((r+f)&1) else 0; out[i,history+8]=i
    return out

def q_from_wdl(wdl):
    try: return float(wdl[0])-float(wdl[2])
    except Exception: return 0.0

def valid_row(line,history):
    if not line.strip(): return None
    try: r=json.loads(line)
    except Exception: return None
    if r.get('schema')!=SCHEMA_POSITION_EVAL or not r.get('fen'): return None
    wdl=r.get('wdl') or [0.0,1.0,0.0]
    q=float(r.get('q', q_from_wdl(wdl)))
    pol=r.get('policy') or {}
    best=r.get('best') or (max(pol.items(),key=lambda kv:kv[1])[0] if pol else None)
    try:
        y=move_class(best) if best else -1
        x=encode(r['fen'],r.get('history_fens') or [],history)
        w=np.asarray([float(wdl[0]),float(wdl[1]),float(wdl[2])],dtype=np.float32)
        qw=float(r.get('quality_weight',1.0) or 1.0)
    except Exception:
        return None
    return x,y,w,q,qw

def count_rows(inputs,max_rows,history):
    n=bad=0
    for p in inputs:
        for line in open_lines(p):
            if max_rows and n>=max_rows: return n,bad
            rr=valid_row(line,history)
            if rr is None: bad+=1; continue
            n+=1
    return n,bad

def build(inputs,out,max_rows,history):
    t0=time.time(); out=Path(out); out.mkdir(parents=True,exist_ok=True); F=history+9
    n,bad=count_rows(inputs,max_rows,history)
    if n<=0: raise SystemExit('no valid position-eval rows')
    tokens=np.memmap(out/'tokens.uint8',np.uint8,'w+',shape=(n,64,F)); policy=np.memmap(out/'policy.int64',np.int64,'w+',shape=(n,)); wdl=np.memmap(out/'wdl.float32',np.float32,'w+',shape=(n,3)); q=np.memmap(out/'q.float32',np.float32,'w+',shape=(n,)); weight=np.memmap(out/'quality_weight.float32',np.float32,'w+',shape=(n,))
    i=0
    for p in inputs:
        for line in open_lines(p):
            if i>=n: break
            rr=valid_row(line,history)
            if rr is None: continue
            tokens[i]=rr[0]; policy[i]=rr[1]; wdl[i]=rr[2]; q[i]=rr[3]; weight[i]=rr[4]; i+=1
            if i%100000==0: print(f'METRIC position_eval_cache_rows_written={i}',flush=True)
    for a in (tokens,policy,wdl,q,weight): a.flush()
    meta={'format':'compact_position_eval_cache_v1','rows':n,'token_features':F,'history_plies':history,'policy_size':POLICY_SIZE,'bad_or_skipped_rows':bad,'source_shards':inputs,'seconds':time.time()-t0,'files':{'tokens':'tokens.uint8','policy':'policy.int64','wdl':'wdl.float32','q':'q.float32','quality_weight':'quality_weight.float32'}}
    (out/'meta.json').write_text(json.dumps(meta,indent=2))
    print(f'METRIC position_eval_cache_rows={n}')
    print(f'METRIC position_eval_cache_bad_rows={bad}')
    print(f'METRIC position_eval_cache_seconds={meta["seconds"]:.3f}')

def main():
    ap=argparse.ArgumentParser(description='Build tensorized compact cache for teacher.position_eval.v1 overlays.')
    ap.add_argument('--input',nargs='+',required=True); ap.add_argument('--out',required=True); ap.add_argument('--max-rows',type=int,default=0); ap.add_argument('--history-plies',type=int,default=2)
    a=ap.parse_args(); build(a.input,a.out,a.max_rows,a.history_plies)
if __name__=='__main__': main()
