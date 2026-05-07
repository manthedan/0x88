#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, subprocess, time
from pathlib import Path

PIECE_VALUES = {'p':1,'n':3,'b':3,'r':5,'q':9}

def material(fen: str):
    s={'w':0,'b':0,'wq':0,'bq':0}
    for c in fen.split()[0]:
        if c.isalpha():
            side='w' if c.isupper() else 'b'; p=c.lower(); s[side]+=PIECE_VALUES.get(p,0); s[side+'q']+=1 if p=='q' else 0
    return s

class Stockfish:
    def __init__(self, path, threads=1, hash_mb=64):
        self.p=subprocess.Popen([path], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
        self.cmd('uci'); self.wait('uciok')
        self.cmd(f'setoption name Threads value {threads}')
        self.cmd(f'setoption name Hash value {hash_mb}')
        self.cmd('isready'); self.wait('readyok')
    def cmd(self,s):
        assert self.p.stdin; self.p.stdin.write(s+'\n'); self.p.stdin.flush()
    def wait(self, token):
        assert self.p.stdout
        while True:
            line=self.p.stdout.readline()
            if token in line: return line
    def eval(self, fen, nodes):
        self.cmd('ucinewgame')
        self.cmd(f'position fen {fen}')
        self.cmd(f'go nodes {nodes}')
        score=None; mate=None; best=None
        assert self.p.stdout
        while True:
            line=self.p.stdout.readline().strip()
            if line.startswith('info ') and ' score ' in line:
                parts=line.split()
                i=parts.index('score')
                if parts[i+1]=='cp': score=int(parts[i+2]); mate=None
                elif parts[i+1]=='mate': mate=int(parts[i+2]); score=100000*(1 if mate>0 else -1)
            elif line.startswith('bestmove '):
                best=line.split()[1]; break
        return {'cp_stm':score, 'mate':mate, 'bestmove':best}
    def close(self):
        try: self.cmd('quit')
        except Exception: pass

def cp_for_tiny(eval_stm, fen, tiny_color):
    turn=fen.split()[1]
    cp=eval_stm['cp_stm']
    if cp is None: return None
    return cp if turn==tiny_color else -cp

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--stockfish', default='.local_engines/stockfish_pkg/usr/games/stockfish')
    ap.add_argument('--nodes', type=int, default=2000)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--hash', type=int, default=64)
    args=ap.parse_args()
    data=json.load(open(args.input))
    inc=data.get('incidents', [])
    if args.limit: inc=inc[:args.limit]
    sf=Stockfish(args.stockfish,args.threads,args.hash)
    annotated=[]; t=time.time()
    for i,x in enumerate(inc,1):
        before=x['fen_before']; after=x['risk']['fenAfter']; tiny=x['tiny_color']
        after_reply=None
        if x.get('actual_reply_uci'):
            # risk.fenAfter is after tiny move; JS diagnostic already did not store fen after reply, so use python-chess unavailable avoidance: ask SF only before/after.
            pass
        eb=sf.eval(before,args.nodes); ea=sf.eval(after,args.nodes)
        cb=cp_for_tiny(eb,before,tiny); ca=cp_for_tiny(ea,after,tiny)
        y={**x,'stockfish':{'nodes':args.nodes,'before':eb,'after_selected':ea,'cp_tiny_before':cb,'cp_tiny_after_selected':ca,'cp_drop_after_selected':None if cb is None or ca is None else cb-ca,'material_before':material(before),'material_after_selected':material(after)}}
        annotated.append(y)
        if i%25==0: print(f'annotated {i}/{len(inc)} seconds={time.time()-t:.1f}', flush=True)
    sf.close()
    out={**data,'incidents':annotated,'stockfish_annotation':{'nodes':args.nodes,'count':len(annotated)}}
    Path(args.out).parent.mkdir(parents=True,exist_ok=True)
    json.dump(out,open(args.out,'w'),indent=2)
    drops=[x['stockfish']['cp_drop_after_selected'] for x in annotated if x['stockfish']['cp_drop_after_selected'] is not None]
    big=sum(1 for d in drops if d>=300); huge=sum(1 for d in drops if d>=700)
    print(f'METRIC sf_annotated={len(annotated)}')
    print(f'METRIC sf_drop_ge_300={big}')
    print(f'METRIC sf_drop_ge_700={huge}')
    print(f'METRIC sf_mean_drop={sum(drops)/max(1,len(drops)):.3f}')
    print(f'wrote {args.out}')

if __name__=='__main__': main()
