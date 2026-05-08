#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math, time
from pathlib import Path
import numpy as np

PIECES='.PNBRQKpnbrqk'
MOVE_FEATURE_NAMES=[
    'moving_piece_type','captured_piece_type','promotion_type','is_capture','is_check','is_castle','is_promotion','is_en_passant',
    'from_attacked_by_enemy_pre','from_defended_by_own_pre','to_attacked_by_enemy_after','to_defended_by_own_after',
    'to_enemy_attackers_after_capped8','to_own_defenders_after_capped8','moving_piece_value','captured_piece_value','material_delta',
    'from_piece_pinned_pre','king_distance_to_enemy_after','king_distance_to_own_after',
]
PIECE_VALUE=np.asarray([0,1,3,3,5,9,0], dtype=np.float32)

def expand_collection(paths):
    out=[]
    for p in paths or []:
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

def open_av_cache_dir(p: str | Path):
    d=Path(p); meta=json.loads((d/'meta.json').read_text())
    rows=int(meta['rows']); K=int(meta['max_candidates']); F=int(meta['token_features'])
    return {
        'path': str(d), 'meta': meta, 'rows': rows, 'K': K, 'F': F,
        'tokens': np.memmap(d/'tokens.uint8', np.uint8, 'r', shape=(rows,64,F)),
        'moves': np.memmap(d/'candidate_moves.int64', np.int64, 'r', shape=(rows,K)),
        'values': np.memmap(d/'candidate_values.float32', np.float32, 'r', shape=(rows,K)),
        'mask': np.memmap(d/'candidate_mask.float32', np.float32, 'r', shape=(rows,K)),
    }

def compact_tokens_to_residual_planes(tok, history=2, state_planes=False):
    B=tok.shape[0]; C=12*(history+1)+(10 if state_planes else 2)
    x=np.zeros((B,C,8,8), dtype=np.float32)
    for h in range(history+1):
        p=tok[:,:,h]
        bidx,sqidx=np.nonzero(p > 0)
        if len(bidx):
            planes=h*12 + p[bidx,sqidx].astype(np.int64)-1
            rows=7-(sqidx//8); cols=sqidx%8
            x[bidx,planes,rows,cols]=1.0
    s0=12*(history+1); stm=tok[:,0,history+1]
    x[:,s0,:,:]=np.where(stm[:,None,None] == 1, 1.0, -1.0)
    if state_planes:
        flags=tok[:,0,history+2]
        for i in range(4): x[:,s0+1+i,:,:]=((flags >> i) & 1)[:,None,None]
        ep_sq=np.argmax(tok[:,:,history+3] > 0, axis=1); has_ep=(tok[:,:,history+3] > 0).any(axis=1)
        if has_ep.any():
            rows=7-(ep_sq[has_ep]//8); cols=ep_sq[has_ep]%8; x[np.nonzero(has_ep)[0],s0+5,rows,cols]=1.0
        x[:,s0+6,:,:]=1.0; x[:,s0+7,:,:]=(stm[:,None,None] == 1)
    else:
        x[:,s0+1,:,:]=1.0
    return x

def chessbench_classes_to_action_ids(moves):
    m=np.asarray(moves, dtype=np.int64)
    ordinary=m < 4096
    ft=np.where(ordinary, m, (m-4096)//4)
    promo=np.where(ordinary, 0, ((m-4096)%4)+1)  # compact n,b,r,q -> action-id n=1,b=2,r=3,q=4
    return ft*5 + promo

def av_move_features_from_tokens(tok, compact_moves):
    B,K=compact_moves.shape; out=np.zeros((B,K,len(MOVE_FEATURE_NAMES)), dtype=np.float32)
    m=np.asarray(compact_moves, dtype=np.int64); ordinary=m < 4096; ft=np.where(ordinary,m,(m-4096)//4)
    fr=np.clip(ft//64,0,63); to=np.clip(ft%64,0,63); promo=np.where(ordinary,0,((m-4096)%4)+1).astype(np.int64)
    board=tok[:,:,0].astype(np.int64)
    bi=np.arange(B)[:,None]
    pc_from=board[bi,fr]; pc_to=board[bi,to]
    moving=np.where(pc_from>0, ((pc_from-1)%6)+1, 0).astype(np.int64)
    captured=np.where(pc_to>0, ((pc_to-1)%6)+1, 0).astype(np.int64)
    out[:,:,0]=moving; out[:,:,1]=captured; out[:,:,2]=promo
    out[:,:,3]=(captured != 0); out[:,:,6]=(promo != 0)
    out[:,:,14]=PIECE_VALUE[np.clip(moving,0,6)]; out[:,:,15]=PIECE_VALUE[np.clip(captured,0,6)]
    promo_piece=np.choose(np.clip(promo,0,4), [0,2,3,4,5]).astype(np.int64)
    promo_gain=np.where(promo>0, PIECE_VALUE[promo_piece]-PIECE_VALUE[1], 0.0)
    out[:,:,16]=PIECE_VALUE[np.clip(captured,0,6)] + promo_gain
    # Lightweight king-distance features from static board; expensive attack/check/pin features stay zero for this AV-cache path.
    wk=(board == 6); bk=(board == 12)
    wk_sq=np.where(wk.any(1), wk.argmax(1), 0); bk_sq=np.where(bk.any(1), bk.argmax(1), 63)
    stm=tok[:,0,3] if tok.shape[2] > 3 else np.ones(B, dtype=np.uint8)
    enemy_king=np.where(stm == 1, bk_sq, wk_sq); own_king=np.where(stm == 1, wk_sq, bk_sq)
    tf=to%8; tr=to//8
    ef=enemy_king[:,None]%8; er=enemy_king[:,None]//8; of=own_king[:,None]%8; orr=own_king[:,None]//8
    out[:,:,18]=np.maximum(np.abs(tf-ef), np.abs(tr-er)); out[:,:,19]=np.maximum(np.abs(tf-of), np.abs(tr-orr))
    return out


def open_sidecar(path: str | Path):
    d = Path(path); meta = json.loads((d / 'meta.json').read_text())
    rows = int(meta['rows']); K = int(meta['max_legal_moves']); F = int(meta['num_move_features'])
    out = {
        'path': str(d), 'meta': meta, 'rows': rows, 'K': K, 'F': F,
        'policy_slot': np.memmap(d/'policy_legal_slot.int16', np.int16, 'r', shape=(rows,)),
        'wdl': np.memmap(d/'wdl.float32', np.float32, 'r', shape=(rows,3)),
        'q': np.memmap(d/'q.float32', np.float32, 'r', shape=(rows,)),
        'legal_action_ids': np.memmap(d/'legal_action_ids.int64', np.int64, 'r', shape=(rows,K)),
        'legal_features': np.memmap(d/'legal_features.float32', np.float32, 'r', shape=(rows,K,F)),
        'legal_mask': np.memmap(d/'legal_mask.float32', np.float32, 'r', shape=(rows,K)),
    }
    if meta.get('has_board_cache'):
        C = int(meta['input_planes'])
        out['x'] = np.memmap(d/'x.int8', np.int8, 'r', shape=(rows,C,8,8)); out['input_planes'] = C
    return out


class ShardedX:
    def __init__(self, shards):
        self.shards = shards
        self.starts = []
        off = 0
        for s in shards:
            self.starts.append(off); off += int(s['rows'])
        self.rows = off
        self.input_planes = int(shards[0]['input_planes'])
        self.shape = (self.rows, self.input_planes, 8, 8)
    def _loc(self, idx: int):
        import bisect
        si = bisect.bisect_right(self.starts, idx) - 1
        return si, idx - self.starts[si]
    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop, step = key.indices(self.rows)
            if step != 1:
                return np.asarray([self[i] for i in range(start, stop, step)])
            if stop <= start:
                return np.empty((0, self.input_planes, 8, 8), dtype=np.int8)
            si, lo = self._loc(start); ei, hi = self._loc(stop - 1); hi += 1
            if si == ei:
                return self.shards[si]['x'][lo:hi]
            parts = [self.shards[si]['x'][lo:]]
            for sj in range(si + 1, ei): parts.append(self.shards[sj]['x'][:])
            parts.append(self.shards[ei]['x'][:hi])
            return np.concatenate(parts, axis=0)
        arr = np.asarray(key)
        if arr.ndim > 0:
            return np.asarray([self[int(i)] for i in arr])
        si, lo = self._loc(int(key)); return self.shards[si]['x'][lo]


def open_board_cache(path: str | Path):
    p = Path(path)
    if p.is_file():
        cm = json.loads(p.read_text())
        if 'shards' in cm:
            shards = []
            for sp in cm['shards']:
                d = Path(sp); meta = json.loads((d/'meta.json').read_text())
                rows = int(meta['rows']); C = int(meta['input_planes'])
                shards.append({'path': str(d), 'rows': rows, 'input_planes': C, 'x': np.memmap(d/'x.int8', np.int8, 'r', shape=(rows,C,8,8)), 'meta': meta})
            return {'path': str(p), 'rows': sum(int(s['rows']) for s in shards), 'input_planes': int(shards[0]['input_planes']), 'x': ShardedX(shards), 'meta': cm}
    d = p; meta = json.loads((d/'meta.json').read_text())
    rows = int(meta['rows']); C = int(meta['input_planes'])
    return {'path': str(d), 'rows': rows, 'input_planes': C, 'x': np.memmap(d/'x.int8', np.int8, 'r', shape=(rows,C,8,8)), 'meta': meta}


def main():
    ap = argparse.ArgumentParser(description='MoveFormer-CNN-AV v1: CNN trunk + legal move tokens + policy/WDL/AV heads.')
    ap.add_argument('--sidecar-cache', default='', help='MoveFormer sidecar cache directory')
    ap.add_argument('--av-cache', nargs='*', default=[], help='Optional ChessBench compact AV cache dirs or collection manifest. If set, trains on candidate AV labels instead of weak sidecar root-q labels.')
    ap.add_argument('--board-cache', default='', help='Residual shard/cache dir with x.int8; required if sidecar lacks x.int8')
    ap.add_argument('--out', required=True); ap.add_argument('--onnx-out', default=''); ap.add_argument('--meta-out', default='')
    ap.add_argument('--checkpoint-dir', default='')
    ap.add_argument('--rows', type=int, default=100000)
    ap.add_argument('--epochs', type=int, default=1); ap.add_argument('--max-steps', type=int, default=0)
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--prefetch-rows', type=int, default=65536, help='Read contiguous sidecar/board blocks into RAM, then shuffle locally; set <= batch size for old random-row sampling.')
    ap.add_argument('--channels', type=int, default=64); ap.add_argument('--blocks', type=int, default=4)
    ap.add_argument('--move-dim', type=int, default=128); ap.add_argument('--heads', type=int, default=4); ap.add_argument('--layers', type=int, default=2); ap.add_argument('--ff-dim', type=int, default=256)
    ap.add_argument('--lr', type=float, default=3e-4); ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--policy-weight', type=float, default=1.0); ap.add_argument('--wdl-weight', type=float, default=0.25); ap.add_argument('--av-weight', type=float, default=0.05)
    ap.add_argument('--progress-every', type=int, default=20); ap.add_argument('--device', default='cuda')
    ap.add_argument('--load-checkpoint', default='', help='Optional checkpoint/model.pt to load before training or export-only runs.')
    ap.add_argument('--onnx-legal-ks', default='', help='Comma-separated fixed legal-move bucket sizes to export, e.g. 32,64,128. Defaults to sidecar max_legal_moves.')
    ap.add_argument('--onnx-dynamic-legal', action='store_true', help='Experimental: mark legal dimension dynamic in ONNX. Stock Transformer export may still only run at traced K.')
    ap.add_argument('--amp', action='store_true'); ap.add_argument('--amp-dtype', choices=['fp16','bf16'], default='bf16'); ap.add_argument('--seed', type=int, default=37)
    args = ap.parse_args()

    import torch, torch.nn as nn, torch.nn.functional as F
    torch.set_float32_matmul_precision('high')
    rng = np.random.default_rng(args.seed); device = args.device

    av_caches=[open_av_cache_dir(p) for p in expand_collection(args.av_cache)]
    train_from_av=bool(av_caches)
    sc=None; x=None; board_rows=0
    if train_from_av:
        hist=int(av_caches[0]['meta'].get('history_plies', 2)); C=12*(hist+1)+2
        N=min(int(args.rows), sum(c['rows'] for c in av_caches)); K=int(av_caches[0]['K']); MF=len(MOVE_FEATURE_NAMES)
        print(f'[moveformer] mode=chessbench_av rows={N} av_cache_count={len(av_caches)} av_cache_rows={sum(c["rows"] for c in av_caches)} input_planes={C} candidates={K} move_features={MF}', flush=True)
    else:
        if not args.sidecar_cache: raise SystemExit('either --sidecar-cache or --av-cache is required')
        sc = open_sidecar(args.sidecar_cache)
        if 'x' in sc:
            x = sc['x']; C = int(sc['input_planes']); board_rows = sc['rows']
        else:
            if not args.board_cache: raise SystemExit('--board-cache is required when sidecar has no x.int8')
            bc = open_board_cache(args.board_cache); x = bc['x']; C = int(bc['input_planes']); board_rows = bc['rows']
        N = min(int(args.rows), sc['rows'], board_rows)
        K, MF = sc['K'], sc['F']
        print(f'[moveformer] mode=sidecar rows={N} sidecar_rows={sc["rows"]} board_rows={board_rows} input_planes={C} max_legal={K} move_features={MF}', flush=True)
    print(f'[moveformer] block_sampling prefetch_rows={args.prefetch_rows}', flush=True)

    class Block(nn.Module):
        def __init__(self, ch):
            super().__init__(); self.c1=nn.Conv2d(ch,ch,3,padding=1); self.c2=nn.Conv2d(ch,ch,3,padding=1); self.n1=nn.BatchNorm2d(ch); self.n2=nn.BatchNorm2d(ch)
        def forward(self,z):
            y=F.relu(self.n1(self.c1(z))); y=self.n2(self.c2(y)); return F.relu(z+y)

    class Net(nn.Module):
        def __init__(self):
            super().__init__(); ch=args.channels; d=args.move_dim
            self.stem=nn.Sequential(nn.Conv2d(C,ch,3,padding=1), nn.BatchNorm2d(ch), nn.ReLU())
            self.blocks=nn.Sequential(*[Block(ch) for _ in range(args.blocks)])
            self.ctx=nn.Linear(ch,d); self.from_proj=nn.Linear(ch,d); self.to_proj=nn.Linear(ch,d)
            self.move_feat=nn.Sequential(nn.Linear(MF,d), nn.LayerNorm(d), nn.GELU(), nn.Linear(d,d))
            self.action_emb=nn.Embedding(20481,d); self.promo_emb=nn.Embedding(5,d)
            enc_layer=nn.TransformerEncoderLayer(d_model=d, nhead=args.heads, dim_feedforward=args.ff_dim, dropout=0.05, batch_first=True, activation='gelu', norm_first=True)
            self.encoder=nn.TransformerEncoder(enc_layer, num_layers=args.layers)
            self.policy=nn.Linear(d,1); self.av=nn.Linear(d,1); self.wdl=nn.Linear(d,3)
            idx=[]
            for sq in range(64):
                rank=sq//8; file=sq%8; idx.append((7-rank)*8+file)
            self.register_buffer('sq_to_plane_idx', torch.tensor(idx,dtype=torch.long), persistent=False)
        def forward(self, planes, action_ids, move_features, legal_mask):
            B,K=action_ids.shape; h=self.blocks(self.stem(planes)); pooled=h.mean((2,3))
            aid=action_ids.clamp(0,20480).long(); ft=torch.div(aid,5,rounding_mode='floor'); promo=aid.remainder(5); fr=torch.div(ft,64,rounding_mode='floor').clamp(0,63); to=ft.remainder(64).clamp(0,63)
            hs=h.permute(0,2,3,1).reshape(B,64,h.shape[1]); D=h.shape[1]
            fp=self.sq_to_plane_idx[fr]; tp=self.sq_to_plane_idx[to]
            hf=torch.gather(hs,1,fp[...,None].expand(-1,-1,D)); ht=torch.gather(hs,1,tp[...,None].expand(-1,-1,D))
            tok=self.ctx(pooled)[:,None,:] + self.from_proj(hf) + self.to_proj(ht) + self.move_feat(move_features.float()) + self.action_emb(aid) + self.promo_emb(promo.clamp(0,4))
            key_padding = legal_mask <= 0
            tok=self.encoder(tok, src_key_padding_mask=key_padding)
            pol=self.policy(tok).squeeze(-1).masked_fill(key_padding, -1e9)
            av=torch.tanh(self.av(tok).squeeze(-1)).masked_fill(key_padding, 0.0)
            # Masked mean token summary, falling back to CNN context if needed.
            denom=legal_mask.sum(1,keepdim=True).clamp_min(1.0)
            summary=(tok*legal_mask[...,None]).sum(1)/denom
            wdl=self.wdl(summary)
            return pol, wdl, av

    net=Net().to(device)
    if args.load_checkpoint:
        ckpt=torch.load(args.load_checkpoint, map_location=device)
        state=ckpt.get('model', ckpt) if isinstance(ckpt, dict) else ckpt
        net.load_state_dict(state)
        print(f'[moveformer] loaded checkpoint {args.load_checkpoint}', flush=True)
    opt=torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    amp_enabled=bool(args.amp and str(device).startswith('cuda')); amp_dtype=torch.bfloat16 if args.amp_dtype=='bf16' else torch.float16
    scaler=torch.amp.GradScaler('cuda', enabled=bool(amp_enabled and amp_dtype is torch.float16))
    def to_dev(a,dtype):
        if isinstance(a, np.ndarray) and not a.flags.writeable:
            a=np.array(a,copy=True)
        t=torch.as_tensor(a,dtype=dtype)
        return t.pin_memory().to(device,non_blocking=True) if str(device).startswith('cuda') else t.to(device)
    pool={'pos':0,'rows':0,'order':None,'x':None,'aid':None,'mf':None,'mask':None,'slot':None,'wdl':None,'q':None,'val':None}
    def _av_cache_choice():
        return av_caches[int(rng.integers(0,len(av_caches)))]
    def refill_pool():
        rows=max(args.batch_size,min(args.prefetch_rows,N)); rows=min(rows,N)
        if train_from_av:
            c=_av_cache_choice(); n=c['rows']; rows=min(rows,n); start=0 if rows>=n else int(rng.integers(0,n-rows+1)); sl=slice(start,start+rows)
            tok=np.array(c['tokens'][sl],copy=True); compact=np.array(c['moves'][sl],copy=True); val=np.array(c['values'][sl],copy=True); mask=np.array(c['mask'][sl],copy=True)
            aid=chessbench_classes_to_action_ids(compact); mf=av_move_features_from_tokens(tok, compact); xb=compact_tokens_to_residual_planes(tok, hist, False)
            val_masked=np.where(mask>0, val, -1e9); slot=val_masked.argmax(1).astype(np.int64); q=val_masked.max(1).astype(np.float32)
            wdl=np.stack([(q+1.0)*0.5, np.zeros_like(q), (1.0-q)*0.5], 1).astype(np.float32)
            pool.update({'pos':0,'rows':rows,'order':rng.permutation(rows),'x':xb,'aid':aid,'mf':mf,'mask':mask,'slot':slot,'wdl':wdl,'q':q,'val':val})
        else:
            start=0 if rows>=N else int(rng.integers(0,N-rows+1)); sl=slice(start,start+rows)
            aid=np.array(sc['legal_action_ids'][sl],copy=True); aid=np.where(aid<0,20480,aid)
            pool.update({'pos':0,'rows':rows,'order':rng.permutation(rows),'x':np.array(x[sl],copy=True),'aid':aid,'mf':np.array(sc['legal_features'][sl],copy=True),'mask':np.array(sc['legal_mask'][sl],copy=True),'slot':np.array(sc['policy_slot'][sl],copy=True),'wdl':np.array(sc['wdl'][sl],copy=True),'q':np.array(sc['q'][sl],copy=True),'val':None})
    def batch():
        if args.prefetch_rows <= args.batch_size:
            if train_from_av:
                c=_av_cache_choice(); n=c['rows']; ids=rng.integers(0,n,size=args.batch_size,endpoint=False)
                tok=np.array(c['tokens'][ids],copy=True); compact=np.array(c['moves'][ids],copy=True); val=np.array(c['values'][ids],copy=True); mask=np.array(c['mask'][ids],copy=True)
                aid=chessbench_classes_to_action_ids(compact); mf=av_move_features_from_tokens(tok, compact); xb=compact_tokens_to_residual_planes(tok, hist, False)
                val_masked=np.where(mask>0, val, -1e9); slot=val_masked.argmax(1).astype(np.int64); q=val_masked.max(1).astype(np.float32)
                wdl=np.stack([(q+1.0)*0.5, np.zeros_like(q), (1.0-q)*0.5], 1).astype(np.float32)
                return to_dev(xb,torch.float32), to_dev(aid,torch.long), to_dev(mf,torch.float32), to_dev(mask,torch.float32), to_dev(slot,torch.long), to_dev(wdl,torch.float32), to_dev(val,torch.float32)
            ids=rng.integers(0,N,size=args.batch_size,endpoint=False)
            xb=to_dev(np.asarray(x[ids]),torch.float32)
            aid=np.asarray(sc['legal_action_ids'][ids]); aid=np.where(aid<0,20480,aid)
            return xb, to_dev(aid,torch.long), to_dev(np.asarray(sc['legal_features'][ids]),torch.float32), to_dev(np.asarray(sc['legal_mask'][ids]),torch.float32), to_dev(np.asarray(sc['policy_slot'][ids]),torch.long), to_dev(np.asarray(sc['wdl'][ids]),torch.float32), to_dev(np.asarray(sc['q'][ids]),torch.float32)
        if pool['order'] is None or pool['pos']+args.batch_size > pool['rows']:
            refill_pool()
        ids=pool['order'][pool['pos']:pool['pos']+args.batch_size]; pool['pos']+=args.batch_size
        target=pool['val'][ids] if train_from_av else pool['q'][ids]
        return to_dev(pool['x'][ids],torch.float32), to_dev(pool['aid'][ids],torch.long), to_dev(pool['mf'][ids],torch.float32), to_dev(pool['mask'][ids],torch.float32), to_dev(pool['slot'][ids],torch.long), to_dev(pool['wdl'][ids],torch.float32), to_dev(target,torch.float32)
    steps_per_epoch=max(1,math.ceil(N/args.batch_size)); global_step=0; start=time.time(); sums={}
    for ep in range(1,args.epochs+1):
        for st in range(1,steps_per_epoch+1):
            if args.max_steps and global_step >= args.max_steps: break
            xb,aid,mf,mask,slot,wdl_t,target=batch(); opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda',enabled=amp_enabled,dtype=amp_dtype):
                pol,wdl,av=net(xb,aid,mf,mask)
                valid=slot>=0
                ploss=F.cross_entropy(pol[valid].float(), slot[valid]) if bool(valid.any()) else pol.sum()*0
                wloss=(-(F.log_softmax(wdl.float(),1)*wdl_t).sum(1)).mean()
                if train_from_av:
                    mbool=mask > 0
                    aloss=F.smooth_l1_loss(av[mbool].float(), target[mbool]) if bool(mbool.any()) else av.sum()*0
                else:
                    # Weak bootstrap AV target: chosen legal move should be near root q.
                    av_chosen=av.gather(1,slot.clamp_min(0)[:,None]).squeeze(1)
                    aloss=F.smooth_l1_loss(av_chosen[valid].float(), target[valid]) if bool(valid.any()) else av.sum()*0
                loss=args.policy_weight*ploss + args.wdl_weight*wloss + args.av_weight*aloss
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); global_step+=1
            for k,v in [('loss',loss),('policy_loss',ploss),('wdl_loss',wloss),('av_loss',aloss)]: sums[k]=sums.get(k,0.0)+float(v.detach())
            if args.progress_every and global_step % args.progress_every == 0:
                msg=' '.join(f'{k}={sums[k]/global_step:.4f}' for k in sorted(sums)); print(f'progress epoch={ep} step={global_step} seconds={time.time()-start:.1f} {msg}', flush=True)
        if args.max_steps and global_step >= args.max_steps: break
    Path(args.out).parent.mkdir(parents=True,exist_ok=True)
    if train_from_av:
        try:
            from train_residual_torch import fixed_policy_moves
            moves_meta=fixed_policy_moves()
        except Exception:
            moves_meta=[]
        meta={'kind':'moveformer_cnn_av_v1','architecture':'cnn_move_token_transformer','policy_map':'uci_queen_knight_promo_v1','moves':moves_meta,'input_planes':C,'history_plies':hist,'channels':args.channels,'blocks':args.blocks,'move_dim':args.move_dim,'heads':args.heads,'layers':args.layers,'num_move_features':MF,'move_feature_names':MOVE_FEATURE_NAMES,'max_legal_moves':K,'action_id_mapping':'(from * 64 + to) * 5 + promo, promo n=1,b=2,r=3,q=4','trained_with_weak_av_q':False,'trained_with_chessbench_av_candidates':True,'av_cache':args.av_cache}
    else:
        meta={'kind':'moveformer_cnn_av_v1','architecture':'cnn_move_token_transformer','policy_map':sc['meta'].get('policy_map'),'moves':sc['meta'].get('moves'),'input_planes':C,'history_plies':sc['meta'].get('history_plies',2),'channels':args.channels,'blocks':args.blocks,'move_dim':args.move_dim,'heads':args.heads,'layers':args.layers,'num_move_features':MF,'move_feature_names':sc['meta'].get('move_feature_names'),'max_legal_moves':K,'action_id_mapping':sc['meta'].get('action_id_mapping'),'trained_with_weak_av_q':bool(args.av_weight>0),'trained_with_chessbench_av_candidates':False}
    if args.onnx_legal_ks:
        legal_ks=[int(x) for x in args.onnx_legal_ks.replace(';',',').split(',') if x.strip()]
    else:
        legal_ks=[K]
    if any(kk <= 0 for kk in legal_ks):
        raise SystemExit('--onnx-legal-ks values must be positive')
    meta['onnx_legal_buckets']=legal_ks if args.onnx_out else []
    meta['onnx_dynamic_batch']=bool(args.onnx_out)
    meta['onnx_dynamic_legal']=bool(args.onnx_out and args.onnx_dynamic_legal)
    torch.save({'model':net.state_dict(),'meta':meta,'args':vars(args),'step':global_step}, args.out)
    if args.checkpoint_dir:
        Path(args.checkpoint_dir).mkdir(parents=True,exist_ok=True); torch.save({'model':net.state_dict(),'meta':meta,'step':global_step}, Path(args.checkpoint_dir)/'checkpoint_latest.pt')
    print(f'METRIC train_steps={global_step}')
    for k,v in sorted(sums.items()): print(f'METRIC train_{k}={v/max(1,global_step):.6f}')
    if args.onnx_out:
        class Export(nn.Module):
            def __init__(self,n): super().__init__(); self.n=n
            def forward(self,planes,legal_action_ids,legal_features,legal_mask): return self.n(planes,legal_action_ids,legal_features,legal_mask)
        def bucket_path(base: str, kk: int) -> Path:
            p=Path(base)
            if len(legal_ks) <= 1:
                return p
            return p.with_name(f'{p.stem}_k{kk}{p.suffix}')
        def bucket_meta(base: Path, kk: int) -> Path:
            return base.with_suffix('.meta.json') if base.suffix else Path(str(base)+'.meta.json')
        net.eval(); Path(args.onnx_out).parent.mkdir(parents=True,exist_ok=True)
        exported=[]
        for kk in legal_ks:
            out_path=bucket_path(args.onnx_out, kk); out_path.parent.mkdir(parents=True,exist_ok=True)
            dynamic_axes={'planes':{0:'batch'},'legal_action_ids':{0:'batch'},'legal_features':{0:'batch'},'legal_mask':{0:'batch'},'policy_logits_legal':{0:'batch'},'wdl_logits':{0:'batch'},'action_values':{0:'batch'}}
            if args.onnx_dynamic_legal:
                dynamic_axes['legal_action_ids'][1]='legal'; dynamic_axes['legal_features'][1]='legal'; dynamic_axes['legal_mask'][1]='legal'; dynamic_axes['policy_logits_legal'][1]='legal'; dynamic_axes['action_values'][1]='legal'
            torch.onnx.export(Export(net).eval(), (torch.zeros(1,C,8,8,device=device), torch.zeros(1,kk,device=device,dtype=torch.long), torch.zeros(1,kk,MF,device=device), torch.ones(1,kk,device=device)), str(out_path), input_names=['planes','legal_action_ids','legal_features','legal_mask'], output_names=['policy_logits_legal','wdl_logits','action_values'], dynamic_axes=dynamic_axes, opset_version=18, external_data=False, dynamo=False)
            m=dict(meta); m.update({'onnx_fixed_legal_moves':kk,'onnx_legal_length_mode':'dynamic_experimental' if args.onnx_dynamic_legal else 'fixed_bucket','onnx_file':str(out_path)})
            bucket_meta(out_path, kk).write_text(json.dumps(m,separators=(',',':')))
            exported.append(str(out_path))
            print(f'METRIC moveformer_onnx_export_k{kk}=1')
        meta['onnx_exports']=exported
    if args.meta_out: Path(args.meta_out).write_text(json.dumps(meta,separators=(',',':')))

if __name__ == '__main__': main()
