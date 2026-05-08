#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math, random, time
from pathlib import Path
import numpy as np

PIECES='.PNBRQKpnbrqk'
PROMOS='nbrq'

def expand_collection(paths):
    out=[]
    for p in paths:
        pp=Path(p)
        if pp.is_file():
            try:
                m=json.loads(pp.read_text())
                if m.get('format') == 'chessbench_av_cache_collection_v1':
                    out += list(m.get('caches') or [])
                    continue
            except Exception:
                pass
        out.append(str(p))
    return out

def open_av_cache_dir(p):
    d=Path(p); meta=json.loads((d/'meta.json').read_text())
    rows=int(meta['rows']); C=int(meta['max_candidates']); F=int(meta['token_features'])
    return {
        'path': str(d), 'rows': rows, 'max_candidates': C, 'token_features': F,
        'tokens': np.memmap(d/'tokens.uint8', np.uint8, 'r', shape=(rows,64,F)),
        'moves': np.memmap(d/'candidate_moves.int64', np.int64, 'r', shape=(rows,C)),
        'values': np.memmap(d/'candidate_values.float32', np.float32, 'r', shape=(rows,C)),
        'regrets': np.memmap(d/'candidate_regrets.float32', np.float32, 'r', shape=(rows,C)),
        'mask': np.memmap(d/'candidate_mask.float32', np.float32, 'r', shape=(rows,C)),
    }

def compact_tokens_to_residual_planes(tok, history=2, state_planes=False):
    # Convert compact square tokens [B,64,F] to residual h2 planes [B,38,8,8].
    B=tok.shape[0]; C=12*(history+1) + (10 if state_planes else 2)
    x=np.zeros((B,C,8,8), dtype=np.float32)
    for h in range(history+1):
        p=tok[:,:,h]
        bidx,sqidx=np.nonzero(p > 0)
        if len(bidx):
            planes=h*12 + p[bidx,sqidx].astype(np.int64) - 1
            rows=7 - (sqidx // 8); cols=sqidx % 8
            x[bidx,planes,rows,cols]=1.0
    s0=12*(history+1)
    stm=tok[:,0,history+1]
    x[:,s0,:,:]=np.where(stm[:,None,None] == 1, 1.0, -1.0)
    if state_planes:
        flags=tok[:,0,history+2]
        for i in range(4): x[:,s0+1+i,:,:]=((flags >> i) & 1)[:,None,None]
        ep_sq=np.argmax(tok[:,:,history+3] > 0, axis=1)
        has_ep=(tok[:,:,history+3] > 0).any(axis=1)
        if has_ep.any():
            rows=7-(ep_sq[has_ep]//8); cols=ep_sq[has_ep]%8; x[np.nonzero(has_ep)[0],s0+5,rows,cols]=1.0
        x[:,s0+6,:,:]=1.0; x[:,s0+7,:,:]=(stm[:,None,None] == 1)
    else:
        x[:,s0+1,:,:]=1.0
    return x

def decode_move_classes_np(m):
    m=np.asarray(m, dtype=np.int64).reshape(-1)
    fr=np.zeros_like(m); to=np.zeros_like(m); pr=np.zeros_like(m)
    ordinary=m < 4096
    fr[ordinary]=m[ordinary]//64; to[ordinary]=m[ordinary]%64
    q=(m[~ordinary]-4096)//4
    fr[~ordinary]=q//64; to[~ordinary]=q%64; pr[~ordinary]=(m[~ordinary]-4096)%4 + 1
    return fr,to,pr

def main():
    ap=argparse.ArgumentParser(description='CNN-AV V2: residual CNN policy/WDL training with auxiliary candidate AV/ranking/regret head.')
    ap.add_argument('--manifest', required=True); ap.add_argument('--dev-cache', required=True)
    ap.add_argument('--av-cache', nargs='+', required=True)
    ap.add_argument('--resume', default=''); ap.add_argument('--resume-model-only', action='store_true')
    ap.add_argument('--out', required=True); ap.add_argument('--onnx-out', default=''); ap.add_argument('--meta-out', default='')
    ap.add_argument('--export-av-head', action='store_true', help='Export candidate_moves -> action_values ONNX output for AV-PUCT runtime use.')
    ap.add_argument('--separate-aux-heads', action='store_true', help='Train separate AV/rank/regret candidate heads instead of shaping one shared candidate score head.')
    ap.add_argument('--export-aux-heads', action='store_true', help='Export rank_scores and regrets outputs in addition to action_values. Implies --export-av-head.')
    ap.add_argument('--checkpoint-dir', default=''); ap.add_argument('--best-checkpoint', default='')
    ap.add_argument('--channels', type=int, default=64); ap.add_argument('--blocks', type=int, default=6); ap.add_argument('--policy-head', choices=['spatial','hybrid'], default='spatial'); ap.add_argument('--se', action='store_true')
    ap.add_argument('--history-plies', type=int, default=2); ap.add_argument('--state-planes', action='store_true')
    ap.add_argument('--policy-rows', type=int, default=25000000); ap.add_argument('--av-positions', type=int, default=16286511); ap.add_argument('--epochs', type=int, default=1)
    ap.add_argument('--batch-size', type=int, default=2048); ap.add_argument('--av-batch-size', type=int, default=1024); ap.add_argument('--max-candidates', type=int, default=8)
    ap.add_argument('--policy-prefetch-rows', type=int, default=65536, help='Read contiguous policy blocks into RAM, then shuffle locally; reduces memmap random page faults. Set <= batch size for old random-row sampling.')
    ap.add_argument('--av-prefetch-rows', type=int, default=65536, help='Read contiguous AV cache blocks into RAM, then shuffle locally; reduces memmap random page faults. Set <= av batch size for old random-row sampling.')
    ap.add_argument('--lr', type=float, default=3e-5); ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--av-weight', type=float, default=1.0); ap.add_argument('--rank-weight', type=float, default=0.5); ap.add_argument('--regret-weight', type=float, default=0.25)
    ap.add_argument('--eval-every-steps', type=int, default=10000); ap.add_argument('--checkpoint-every-steps', type=int, default=10000); ap.add_argument('--progress-every', type=int, default=500)
    ap.add_argument('--max-dev-rows', type=int, default=100000); ap.add_argument('--max-av-dev-positions', type=int, default=10000); ap.add_argument('--patience', type=int, default=4)
    ap.add_argument('--device', default='cuda'); ap.add_argument('--amp', action='store_true'); ap.add_argument('--amp-dtype', choices=['fp16','bf16'], default='bf16'); ap.add_argument('--seed', type=int, default=23)
    args=ap.parse_args()

    import torch, torch.nn as nn, torch.nn.functional as F
    torch.set_float32_matmul_precision('high')
    rng=random.Random(args.seed); np_rng=np.random.default_rng(args.seed)
    device=args.device

    man=json.loads(Path(args.manifest).read_text()); paths=[Path(s) for s in man['shards']]
    metas=[json.loads((p/'meta.json').read_text()) for p in paths]
    C=int(metas[0]['input_planes']); P=int(metas[0]['policy_size']); moves=metas[0]['moves']
    sizes=[int(m['rows']) for m in metas]
    train=[]
    for p,n in zip(paths,sizes):
        train.append((np.memmap(p/'x.int8', np.int8, 'r', shape=(n,C,8,8)), np.memmap(p/'policy.int64', np.int64, 'r', shape=(n,)), np.memmap(p/'wdl.float32', np.float32, 'r', shape=(n,3)), np.memmap(p/'weight.float32', np.float32, 'r', shape=(n,))))
    dc=Path(args.dev_cache); dm=json.loads((dc/'meta.json').read_text()); DN=min(int(dm['rows']), args.max_dev_rows)
    dx=np.memmap(dc/'x.int8', np.int8, 'r', shape=(int(dm['rows']),C,8,8)); dy=np.memmap(dc/'policy.int64', np.int64, 'r', shape=(int(dm['rows']),)); dv=np.memmap(dc/'wdl.float32', np.float32, 'r', shape=(int(dm['rows']),3))
    av_caches=[open_av_cache_dir(p) for p in expand_collection(args.av_cache)]
    print(f'[cnn-av] policy_cache_shards={len(train)} policy_rows={sum(sizes)} input_planes={C}', flush=True)
    print(f'[cnn-av] av_cache_count={len(av_caches)} av_cache_rows={sum(c["rows"] for c in av_caches)}', flush=True)
    print(f'[cnn-av] block_sampling policy_prefetch_rows={args.policy_prefetch_rows} av_prefetch_rows={args.av_prefetch_rows}', flush=True)

    class Block(nn.Module):
        def __init__(self,ch):
            super().__init__(); self.c1=nn.Conv2d(ch,ch,3,padding=1); self.c2=nn.Conv2d(ch,ch,3,padding=1)
            if args.se:
                mid=max(8,ch//4); self.se1=nn.Linear(ch,mid); self.se2=nn.Linear(mid,ch)
            else: self.se1=self.se2=None
        def forward(self,z):
            y=self.c2(F.relu(self.c1(z)))
            if self.se1 is not None:
                g=torch.sigmoid(self.se2(F.relu(self.se1(y.mean((2,3)))))).view(y.shape[0],y.shape[1],1,1); y=y*g
            return F.relu(y+z)
    class Net(nn.Module):
        def __init__(self):
            super().__init__(); ch=args.channels; self.stem=nn.Conv2d(C,ch,3,padding=1); self.blocks=nn.Sequential(*[Block(ch) for _ in range(args.blocks)]); pin=ch*64+(ch if args.policy_head=='hybrid' else 0); self.policy=nn.Linear(pin,P); self.wdl=nn.Linear(ch,3); self.promo_emb=nn.Embedding(5,ch); self.av=nn.Sequential(nn.Linear(ch*4,ch),nn.GELU(),nn.Linear(ch,1)); self.rank=nn.Sequential(nn.Linear(ch*4,ch),nn.GELU(),nn.Linear(ch,1)) if args.separate_aux_heads else None; self.regret=nn.Sequential(nn.Linear(ch*4,ch),nn.GELU(),nn.Linear(ch,1)) if args.separate_aux_heads else None
            idx=[]
            for sq in range(64):
                rank=sq//8; file=sq%8; idx.append((7-rank)*8+file)
            self.register_buffer('sq_to_plane_idx', torch.tensor(idx,dtype=torch.long), persistent=False)
        def features(self,z): return self.blocks(F.relu(self.stem(z)))
        def forward(self,z):
            h=self.features(z); pooled=h.mean((2,3)); pf=torch.cat([h.flatten(1),pooled],1) if args.policy_head=='hybrid' else h.flatten(1); return self.policy(pf), self.wdl(pooled), h
        def candidate_features(self,h,mv):
            B,K=mv.shape; m=mv.to(device=h.device,dtype=torch.long); ordinary=m < 4096; q=torch.div((m-4096).clamp_min(0),4,rounding_mode='floor')
            fr=torch.where(ordinary, torch.div(m,64,rounding_mode='floor'), torch.div(q,64,rounding_mode='floor'))
            to=torch.where(ordinary, m.remainder(64), q.remainder(64))
            pr=torch.where(ordinary, torch.zeros_like(m), (m-4096).remainder(4)+1)
            hs=h.permute(0,2,3,1).reshape(B,64,h.shape[1]); frp=self.sq_to_plane_idx[fr.clamp(0,63)]; top=self.sq_to_plane_idx[to.clamp(0,63)]; D=h.shape[1]
            hf=torch.gather(hs,1,frp[...,None].expand(-1,-1,D)); ht=torch.gather(hs,1,top[...,None].expand(-1,-1,D)); pooled=h.mean((2,3))[:,None,:].expand(-1,K,-1); pe=self.promo_emb(pr.clamp(0,4)); return torch.cat([pooled,hf,ht,pe],-1)
        def av_scores(self,h,mv): return self.av(self.candidate_features(h,mv)).squeeze(-1)
        def rank_scores(self,h,mv): return (self.rank if self.rank is not None else self.av)(self.candidate_features(h,mv)).squeeze(-1)
        def regret_scores(self,h,mv): return F.softplus((self.regret if self.regret is not None else self.av)(self.candidate_features(h,mv)).squeeze(-1))
    net=Net().to(device)
    if args.resume:
        ck=torch.load(args.resume,map_location=device); st=ck['model'] if isinstance(ck,dict) and 'model' in ck else ck
        miss=net.load_state_dict(st, strict=False)
        print(f'[cnn-av] resumed=1 missing={len(miss.missing_keys)} unexpected={len(miss.unexpected_keys)}', flush=True)
    else: print('[cnn-av] resumed=0', flush=True)
    opt=torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    amp_enabled=bool(args.amp and str(device).startswith('cuda')); amp_dtype=torch.bfloat16 if args.amp_dtype=='bf16' else torch.float16; scaler=torch.amp.GradScaler('cuda', enabled=bool(amp_enabled and amp_dtype is torch.float16))
    def to_dev(a,dtype):
        if isinstance(a, np.ndarray) and not a.flags.writeable:
            a=np.array(a,copy=True)
        t=torch.as_tensor(a,dtype=dtype)
        return t.pin_memory().to(device,non_blocking=True) if str(device).startswith('cuda') else t.to(device)
    policy_pool={'pos':0,'rows':0,'order':None,'x':None,'y':None,'v':None,'w':None}
    av_pool={'pos':0,'rows':0,'order':None,'tokens':None,'moves':None,'values':None,'regrets':None,'mask':None,'K':None}
    def _contiguous_window(n, want):
        rows=max(1,min(int(want),int(n)))
        start=0 if rows>=n else int(np_rng.integers(0,n-rows+1))
        return start, rows
    def refill_policy_pool():
        si=int(np_rng.integers(0,len(train))); x,y,v,w=train[si]; n=len(y); rows=max(args.batch_size,min(args.policy_prefetch_rows,n))
        start,rows=_contiguous_window(n,rows); sl=slice(start,start+rows)
        # Force a real ndarray copy: random fancy-indexing inside memmaps causes page-fault storms.
        policy_pool.update({'pos':0,'rows':rows,'order':np_rng.permutation(rows),'x':np.array(x[sl],copy=True),'y':np.array(y[sl],copy=True),'v':np.array(v[sl],copy=True),'w':np.array(w[sl],copy=True)})
    def policy_batch():
        if args.policy_prefetch_rows <= args.batch_size:
            si=int(np_rng.integers(0,len(train))); x,y,v,w=train[si]; n=len(y); ids=np_rng.integers(0,n,size=args.batch_size,endpoint=False)
            return to_dev(np.asarray(x[ids]),torch.float32), to_dev(np.asarray(y[ids]),torch.long), to_dev(np.asarray(v[ids]),torch.float32), to_dev(np.asarray(w[ids]),torch.float32)
        if policy_pool['order'] is None or policy_pool['pos']+args.batch_size > policy_pool['rows']:
            refill_policy_pool()
        ids=policy_pool['order'][policy_pool['pos']:policy_pool['pos']+args.batch_size]; policy_pool['pos']+=args.batch_size
        return to_dev(policy_pool['x'][ids],torch.float32), to_dev(policy_pool['y'][ids],torch.long), to_dev(policy_pool['v'][ids],torch.float32), to_dev(policy_pool['w'][ids],torch.float32)
    def refill_av_pool():
        c=av_caches[int(np_rng.integers(0,len(av_caches)))]; n=c['rows']; rows=max(args.av_batch_size,min(args.av_prefetch_rows,n)); K=min(args.max_candidates,c['max_candidates'])
        start,rows=_contiguous_window(n,rows); sl=slice(start,start+rows)
        av_pool.update({'pos':0,'rows':rows,'order':np_rng.permutation(rows),'tokens':np.array(c['tokens'][sl],copy=True),'moves':np.array(c['moves'][sl,:K],copy=True),'values':np.array(c['values'][sl,:K],copy=True),'regrets':np.array(c['regrets'][sl,:K],copy=True),'mask':np.array(c['mask'][sl,:K],copy=True),'K':K})
    def av_batch(dev=False):
        if dev or args.av_prefetch_rows <= args.av_batch_size:
            c=av_caches[int(np_rng.integers(0,len(av_caches)))]; n=min(c['rows'], args.max_av_dev_positions) if dev else c['rows']; ids=np_rng.integers(0,n,size=args.av_batch_size,endpoint=False); K=min(args.max_candidates,c['max_candidates'])
            xb=compact_tokens_to_residual_planes(np.asarray(c['tokens'][ids]), args.history_plies, args.state_planes)
            return to_dev(xb,torch.float32), to_dev(np.asarray(c['moves'][ids,:K]),torch.long), to_dev(np.asarray(c['values'][ids,:K]),torch.float32), to_dev(np.asarray(c['regrets'][ids,:K]),torch.float32), to_dev(np.asarray(c['mask'][ids,:K]),torch.bool)
        if av_pool['order'] is None or av_pool['pos']+args.av_batch_size > av_pool['rows']:
            refill_av_pool()
        ids=av_pool['order'][av_pool['pos']:av_pool['pos']+args.av_batch_size]; av_pool['pos']+=args.av_batch_size
        xb=compact_tokens_to_residual_planes(av_pool['tokens'][ids], args.history_plies, args.state_planes)
        return to_dev(xb,torch.float32), to_dev(av_pool['moves'][ids],torch.long), to_dev(av_pool['values'][ids],torch.float32), to_dev(av_pool['regrets'][ids],torch.float32), to_dev(av_pool['mask'][ids],torch.bool)
    def save(path, ep, step, metrics=None):
        Path(path).parent.mkdir(parents=True,exist_ok=True); torch.save({'model':net.state_dict(),'opt':opt.state_dict(),'epoch':ep,'step':step,'metrics':metrics or {},'args':vars(args),'kind':'cnn_av_v2'}, path)
    def eval_all(label):
        net.eval(); ce=wc=t1=t4=t8=seen=0; av_mse=av_ok=av_seen=av_pos=0
        with torch.no_grad():
            for off in range(0,DN,args.batch_size):
                xb=to_dev(np.asarray(dx[off:off+args.batch_size]),torch.float32); yb=to_dev(np.asarray(dy[off:off+args.batch_size]),torch.long); vb=to_dev(np.asarray(dv[off:off+args.batch_size]),torch.float32)
                with torch.amp.autocast('cuda',enabled=amp_enabled,dtype=amp_dtype): pl,wl,_=net(xb)
                bs=len(yb); ce+=float(F.cross_entropy(pl.float(),yb,reduction='sum')); wc+=float((-(F.log_softmax(wl.float(),1)*vb).sum(1)).sum()); pred=pl.topk(8,1).indices; t1+=int((pred[:,:1]==yb[:,None]).any(1).sum()); t4+=int((pred[:,:4]==yb[:,None]).any(1).sum()); t8+=int((pred==yb[:,None]).any(1).sum()); seen+=bs
            nb=max(1,min(20,args.max_av_dev_positions//max(1,args.av_batch_size)))
            for _ in range(nb):
                xb,mv,val,_,mask=av_batch(dev=True); _,_,h=net(xb); sc=torch.tanh(net.av_scores(h,mv).float()); av_mse+=float(F.mse_loss(sc[mask],val[mask],reduction='sum')); av_pos+=int(mask.sum()); av_ok+=int((sc.masked_fill(~mask,-1e9).argmax(1)==val.masked_fill(~mask,-1e9).argmax(1)).sum()); av_seen+=xb.shape[0]
        m={'dev_policy_ce':ce/max(1,seen),'dev_wdl_ce':wc/max(1,seen),'dev_policy_top1':t1/max(1,seen),'dev_policy_top4':t4/max(1,seen),'dev_policy_top8':t8/max(1,seen),'dev_av_mse':av_mse/max(1,av_pos),'dev_av_top1':av_ok/max(1,av_seen)}
        m['composite']=m['dev_policy_ce']+0.25*m['dev_wdl_ce']+2.0*m['dev_av_mse']
        for k,v in m.items(): print(f'METRIC {label}_{k}={v:.6f}', flush=True)
        net.train(); return m
    best=float('inf'); stale=0; global_step=0
    for ep in range(1,args.epochs+1):
        sched=['policy']*math.ceil(args.policy_rows/args.batch_size) + ['av']*math.ceil(args.av_positions/args.av_batch_size); rng.shuffle(sched); st=time.time(); sums={}; counts={}; net.train()
        for step,kind in enumerate(sched,1):
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda',enabled=amp_enabled,dtype=amp_dtype):
                if kind=='policy':
                    xb,yb,vb,wb=policy_batch(); pl,wl,_=net(xb); loss=(F.cross_entropy(pl,yb,reduction='none')*wb).mean()+(-(F.log_softmax(wl,1)*vb).sum(1)*wb).mean()
                else:
                    xb,mv,val,reg,mask=av_batch(); _,_,h=net(xb); sc=net.av_scores(h,mv); rank_sc=net.rank_scores(h,mv) if args.separate_aux_heads else sc; av=F.smooth_l1_loss(torch.tanh(sc.float())[mask],val[mask]); rank=F.cross_entropy(rank_sc.float().masked_fill(~mask,-1e9), val.masked_fill(~mask,-1e9).argmax(1)); preg=(net.regret_scores(h,mv).float() if args.separate_aux_heads else (sc.float().max(1,keepdim=True).values-sc.float())).masked_select(mask); rloss=F.smooth_l1_loss(preg,reg.masked_select(mask)); loss=args.av_weight*av+args.rank_weight*rank+args.regret_weight*rloss
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); global_step+=1; sums[kind]=sums.get(kind,0.0)+float(loss.detach()); counts[kind]=counts.get(kind,0)+1
            if args.progress_every and step%args.progress_every==0:
                msg=' '.join(f'{k}_loss={sums[k]/max(1,counts[k]):.4f}' for k in sorted(sums)); print(f'progress epoch={ep} step={step}/{len(sched)} seconds={time.time()-st:.1f} {msg}', flush=True)
            if args.checkpoint_dir and args.checkpoint_every_steps and global_step%args.checkpoint_every_steps==0: save(Path(args.checkpoint_dir)/'checkpoint_latest.pt',ep,step)
            if args.eval_every_steps and global_step%args.eval_every_steps==0:
                m=eval_all(f'step{global_step}'); score=m['composite']
                if score < best-1e-4:
                    best=score; stale=0
                    if args.best_checkpoint: save(args.best_checkpoint,ep,step,m)
                    if args.checkpoint_dir: save(Path(args.checkpoint_dir)/'best.pt',ep,step,m)
                else:
                    stale+=1
                    if args.patience and stale>=args.patience:
                        print('METRIC stopped_patience=1', flush=True); break
        m=eval_all(f'epoch{ep}'); save(Path(args.checkpoint_dir)/f'epoch_{ep}.pt' if args.checkpoint_dir else args.out,ep,step,m)
    export_aux=bool(args.export_aux_heads); export_av=bool(args.export_av_head or export_aux)
    meta={'kind':'tiny_board_residual_onnx_student','architecture':'residual_tower','policy_map':'uci_queen_knight_promo_v1','moves':moves,'channels':args.channels,'blocks':args.blocks,'policy_head':args.policy_head,'se':bool(args.se),'history_plies':args.history_plies,'input_planes':C,'trained_with_aux_av':True,'av_head_exported':export_av,'aux_heads_exported':(['action_values','rank_scores','regrets'] if export_aux else (['action_values'] if export_av else [])),'separate_aux_heads':bool(args.separate_aux_heads),'action_value_move_encoding':'chessbench_compact_20480','onnx':args.onnx_out}
    Path(args.out).parent.mkdir(parents=True,exist_ok=True); torch.save({'model':net.state_dict(),'meta':meta},args.out)
    if args.meta_out: Path(args.meta_out).write_text(json.dumps(meta,separators=(',',':')))
    if args.onnx_out:
        class Export(nn.Module):
            def __init__(self,n): super().__init__(); self.n=n
            def forward(self,x): pl,wl,_=self.n(x); return pl,wl
        class ExportAv(nn.Module):
            def __init__(self,n): super().__init__(); self.n=n
            def forward(self,x,candidate_moves): pl,wl,h=self.n(x); av=torch.tanh(self.n.av_scores(h,candidate_moves).float()); return pl,wl,av
        class ExportAux(nn.Module):
            def __init__(self,n): super().__init__(); self.n=n
            def forward(self,x,candidate_moves): pl,wl,h=self.n(x); av=torch.tanh(self.n.av_scores(h,candidate_moves).float()); rank=self.n.rank_scores(h,candidate_moves).float(); regret=self.n.regret_scores(h,candidate_moves).float(); return pl,wl,av,rank,regret
        net.eval(); Path(args.onnx_out).parent.mkdir(parents=True,exist_ok=True)
        if export_aux:
            torch.onnx.export(ExportAux(net).eval(), (torch.zeros(1,C,8,8,device=device), torch.zeros(1,args.max_candidates,device=device,dtype=torch.long)), args.onnx_out, input_names=['planes','candidate_moves'], output_names=['policy_logits','wdl_logits','action_values','rank_scores','regrets'], dynamic_axes={'planes':{0:'batch'},'candidate_moves':{0:'batch',1:'candidates'},'policy_logits':{0:'batch'},'wdl_logits':{0:'batch'},'action_values':{0:'batch',1:'candidates'},'rank_scores':{0:'batch',1:'candidates'},'regrets':{0:'batch',1:'candidates'}}, opset_version=18, external_data=False, dynamo=False)
        elif export_av:
            torch.onnx.export(ExportAv(net).eval(), (torch.zeros(1,C,8,8,device=device), torch.zeros(1,args.max_candidates,device=device,dtype=torch.long)), args.onnx_out, input_names=['planes','candidate_moves'], output_names=['policy_logits','wdl_logits','action_values'], dynamic_axes={'planes':{0:'batch'},'candidate_moves':{0:'batch',1:'candidates'},'policy_logits':{0:'batch'},'wdl_logits':{0:'batch'},'action_values':{0:'batch',1:'candidates'}}, opset_version=18, external_data=False, dynamo=False)
        else:
            torch.onnx.export(Export(net).eval(), torch.zeros(1,C,8,8,device=device), args.onnx_out, input_names=['planes'], output_names=['policy_logits','wdl_logits'], dynamic_axes={'planes':{0:'batch'},'policy_logits':{0:'batch'},'wdl_logits':{0:'batch'}}, opset_version=18, external_data=False, dynamo=False)
if __name__=='__main__': main()
