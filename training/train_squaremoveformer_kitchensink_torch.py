#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math, random, time
from pathlib import Path
import numpy as np

from train_moveformer_cnn_av_torch import (
    MOVE_FEATURE_NAMES,
    open_sidecar,
    open_board_cache,
    compact_tokens_to_residual_planes,
    chessbench_classes_to_action_ids,
    av_move_features_from_tokens,
    expand_collection,
)


def open_av_cache_dir(p: str | Path):
    d = Path(p); meta = json.loads((d / 'meta.json').read_text())
    rows = int(meta['rows']); K = int(meta['max_candidates']); F = int(meta['token_features'])
    return {
        'path': str(d), 'meta': meta, 'rows': rows, 'K': K, 'F': F,
        'tokens': np.memmap(d/'tokens.uint8', np.uint8, 'r', shape=(rows,64,F)),
        'moves': np.memmap(d/'candidate_moves.int64', np.int64, 'r', shape=(rows,K)),
        'values': np.memmap(d/'candidate_values.float32', np.float32, 'r', shape=(rows,K)),
        'regrets': np.memmap(d/'candidate_regrets.float32', np.float32, 'r', shape=(rows,K)),
        'mask': np.memmap(d/'candidate_mask.float32', np.float32, 'r', shape=(rows,K)),
    }


def main():
    ap = argparse.ArgumentParser(description='SquareMoveFormer kitchen-sink: CNN trunk + 64 square tokens + legal move tokens with policy/WDL/AV/rank/regret heads.')
    ap.add_argument('--sidecar-cache', required=True)
    ap.add_argument('--board-cache', required=True, help='Residual cache manifest/dir matching sidecar row order; use h2_state for 46-plane tests.')
    ap.add_argument('--av-cache', nargs='+', required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--onnx-out', default=''); ap.add_argument('--meta-out', default='')
    ap.add_argument('--checkpoint-dir', default='')
    ap.add_argument('--policy-rows', type=int, default=1000000); ap.add_argument('--av-positions', type=int, default=1000000); ap.add_argument('--epochs', type=int, default=1)
    ap.add_argument('--batch-size', type=int, default=384); ap.add_argument('--av-batch-size', type=int, default=384); ap.add_argument('--max-candidates', type=int, default=48)
    ap.add_argument('--policy-prefetch-rows', type=int, default=65536); ap.add_argument('--av-prefetch-rows', type=int, default=65536)
    ap.add_argument('--channels', type=int, default=64); ap.add_argument('--blocks', type=int, default=5)
    ap.add_argument('--token-dim', type=int, default=128); ap.add_argument('--heads', type=int, default=4); ap.add_argument('--layers', type=int, default=2); ap.add_argument('--ff-dim', type=int, default=256); ap.add_argument('--dropout', type=float, default=0.05)
    ap.add_argument('--history-plies', type=int, default=2); ap.add_argument('--state-planes', action='store_true')
    ap.add_argument('--lr', type=float, default=1e-4); ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--policy-weight', type=float, default=1.0); ap.add_argument('--wdl-weight', type=float, default=1.0)
    ap.add_argument('--av-weight', type=float, default=1.0); ap.add_argument('--rank-weight', type=float, default=0.5); ap.add_argument('--regret-weight', type=float, default=0.25)
    ap.add_argument('--progress-every', type=int, default=250); ap.add_argument('--checkpoint-every-steps', type=int, default=1000)
    ap.add_argument('--device', default='cuda'); ap.add_argument('--amp', action='store_true'); ap.add_argument('--amp-dtype', choices=['fp16','bf16'], default='fp16')
    ap.add_argument('--onnx-legal-ks', default='64,128'); ap.add_argument('--seed', type=int, default=23)
    args = ap.parse_args()

    import torch, torch.nn as nn, torch.nn.functional as F
    torch.set_float32_matmul_precision('high')
    rng = random.Random(args.seed); np_rng = np.random.default_rng(args.seed); device = args.device

    sc = open_sidecar(args.sidecar_cache)
    bc = open_board_cache(args.board_cache)
    x = bc['x']; C = int(bc['input_planes']); board_rows = int(bc['rows'])
    N = min(int(args.policy_rows), int(sc['rows']), board_rows)
    K_side, MF = int(sc['K']), int(sc['F'])
    av_caches = [open_av_cache_dir(p) for p in expand_collection(args.av_cache)]
    print(f'[squaremoveformer] sidecar_rows={sc["rows"]} board_rows={board_rows} policy_rows={N} input_planes={C} legal_K={K_side} move_features={MF}', flush=True)
    print(f'[squaremoveformer] av_cache_count={len(av_caches)} av_cache_rows={sum(c["rows"] for c in av_caches)} max_candidates={args.max_candidates}', flush=True)

    class Block(nn.Module):
        def __init__(self, ch):
            super().__init__(); self.c1=nn.Conv2d(ch,ch,3,padding=1); self.c2=nn.Conv2d(ch,ch,3,padding=1); self.n1=nn.BatchNorm2d(ch); self.n2=nn.BatchNorm2d(ch)
        def forward(self,z):
            y=F.relu(self.n1(self.c1(z))); y=self.n2(self.c2(y)); return F.relu(z+y)

    class Net(nn.Module):
        def __init__(self):
            super().__init__(); ch=args.channels; d=args.token_dim
            self.stem=nn.Sequential(nn.Conv2d(C,ch,3,padding=1), nn.BatchNorm2d(ch), nn.ReLU())
            self.blocks=nn.Sequential(*[Block(ch) for _ in range(args.blocks)])
            self.square_proj=nn.Sequential(nn.Linear(ch,d), nn.LayerNorm(d))
            self.ctx=nn.Linear(ch,d); self.from_proj=nn.Linear(ch,d); self.to_proj=nn.Linear(ch,d)
            self.move_feat=nn.Sequential(nn.Linear(MF,d), nn.LayerNorm(d), nn.GELU(), nn.Linear(d,d))
            self.action_emb=nn.Embedding(20481,d); self.promo_emb=nn.Embedding(5,d)
            self.cls=nn.Parameter(torch.zeros(1,1,d)); self.square_pos=nn.Parameter(torch.randn(1,64,d)*0.02); self.type_emb=nn.Embedding(3,d)
            enc=nn.TransformerEncoderLayer(d_model=d, nhead=args.heads, dim_feedforward=args.ff_dim, dropout=args.dropout, batch_first=True, activation='gelu', norm_first=True)
            self.encoder=nn.TransformerEncoder(enc, num_layers=args.layers)
            self.out_norm=nn.LayerNorm(d)
            self.policy=nn.Linear(d,1); self.av=nn.Linear(d,1); self.rank=nn.Linear(d,1); self.regret=nn.Linear(d,1); self.wdl=nn.Linear(d,3)
            idx=[]
            for sq in range(64):
                rank=sq//8; file=sq%8; idx.append((7-rank)*8+file)
            self.register_buffer('sq_to_plane_idx', torch.tensor(idx,dtype=torch.long), persistent=False)
        def forward(self, planes, action_ids, move_features, legal_mask):
            B,K=action_ids.shape; h=self.blocks(self.stem(planes)); pooled=h.mean((2,3))
            hs=h.permute(0,2,3,1).reshape(B,64,h.shape[1]); D=h.shape[1]
            sq_tok=self.square_proj(hs) + self.square_pos + self.type_emb(torch.zeros(64,device=planes.device,dtype=torch.long))[None,:,:]
            aid=action_ids.clamp(0,20480).long(); ft=torch.div(aid,5,rounding_mode='floor'); promo=aid.remainder(5); fr=torch.div(ft,64,rounding_mode='floor').clamp(0,63); to=ft.remainder(64).clamp(0,63)
            fp=self.sq_to_plane_idx[fr]; tp=self.sq_to_plane_idx[to]
            hf=torch.gather(hs,1,fp[...,None].expand(-1,-1,D)); ht=torch.gather(hs,1,tp[...,None].expand(-1,-1,D))
            mv_tok=(self.ctx(pooled)[:,None,:] + self.from_proj(hf) + self.to_proj(ht) + self.move_feat(move_features.float()) + self.action_emb(aid) + self.promo_emb(promo.clamp(0,4)) + self.type_emb(torch.ones(K,device=planes.device,dtype=torch.long))[None,:,:])
            cls=self.cls.expand(B,-1,-1) + self.type_emb(torch.full((1,),2,device=planes.device,dtype=torch.long))[None,:,:]
            tok=torch.cat([cls,sq_tok,mv_tok],1)
            move_pad = legal_mask <= 0
            pad=torch.cat([torch.zeros(B,65,device=planes.device,dtype=torch.bool), move_pad],1)
            tok=self.out_norm(self.encoder(tok, src_key_padding_mask=pad))
            cls_out=tok[:,0]; move_out=tok[:,65:65+K]
            pol=self.policy(move_out).squeeze(-1).masked_fill(move_pad, -1e4)
            av=torch.tanh(self.av(move_out).squeeze(-1)).masked_fill(move_pad, 0.0)
            rank=self.rank(move_out).squeeze(-1).masked_fill(move_pad, -1e4)
            regret=F.softplus(self.regret(move_out).squeeze(-1)).masked_fill(move_pad, 0.0)
            wdl=self.wdl(cls_out)
            return pol, wdl, av, rank, regret

    net=Net().to(device)
    opt=torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    amp_enabled=bool(args.amp and str(device).startswith('cuda')); amp_dtype=torch.bfloat16 if args.amp_dtype=='bf16' else torch.float16
    scaler=torch.amp.GradScaler('cuda', enabled=bool(amp_enabled and amp_dtype is torch.float16))
    def to_dev(a,dtype):
        if isinstance(a, np.ndarray) and not a.flags.writeable: a=np.array(a,copy=True)
        t=torch.as_tensor(a,dtype=dtype)
        return t.pin_memory().to(device,non_blocking=True) if str(device).startswith('cuda') else t.to(device)

    policy_pool={'pos':0,'rows':0,'order':None,'x':None,'aid':None,'mf':None,'mask':None,'slot':None,'wdl':None}
    av_pool={'pos':0,'rows':0,'order':None,'tokens':None,'moves':None,'values':None,'regrets':None,'mask':None,'K':None}
    def _window(n,want):
        rows=max(1,min(int(want),int(n))); start=0 if rows>=n else int(np_rng.integers(0,n-rows+1)); return slice(start,start+rows), rows
    def refill_policy_pool():
        sl,rows=_window(N, max(args.batch_size,args.policy_prefetch_rows))
        aid=np.array(sc['legal_action_ids'][sl],copy=True); aid=np.where(aid<0,20480,aid)
        policy_pool.update({'pos':0,'rows':rows,'order':np_rng.permutation(rows),'x':np.array(x[sl],copy=True),'aid':aid,'mf':np.array(sc['legal_features'][sl],copy=True),'mask':np.array(sc['legal_mask'][sl],copy=True),'slot':np.array(sc['policy_slot'][sl],copy=True),'wdl':np.array(sc['wdl'][sl],copy=True)})
    def policy_batch():
        if policy_pool['order'] is None or policy_pool['pos']+args.batch_size > policy_pool['rows']: refill_policy_pool()
        ids=policy_pool['order'][policy_pool['pos']:policy_pool['pos']+args.batch_size]; policy_pool['pos']+=args.batch_size
        return to_dev(policy_pool['x'][ids],torch.float32), to_dev(policy_pool['aid'][ids],torch.long), to_dev(policy_pool['mf'][ids],torch.float32), to_dev(policy_pool['mask'][ids],torch.float32), to_dev(policy_pool['slot'][ids],torch.long), to_dev(policy_pool['wdl'][ids],torch.float32)
    def refill_av_pool():
        c=av_caches[int(np_rng.integers(0,len(av_caches)))]; n=c['rows']; K=min(args.max_candidates,c['K']); sl,rows=_window(n, max(args.av_batch_size,args.av_prefetch_rows))
        av_pool.update({'pos':0,'rows':rows,'order':np_rng.permutation(rows),'tokens':np.array(c['tokens'][sl],copy=True),'moves':np.array(c['moves'][sl,:K],copy=True),'values':np.array(c['values'][sl,:K],copy=True),'regrets':np.array(c['regrets'][sl,:K],copy=True),'mask':np.array(c['mask'][sl,:K],copy=True),'K':K})
    def av_batch():
        if av_pool['order'] is None or av_pool['pos']+args.av_batch_size > av_pool['rows']: refill_av_pool()
        ids=av_pool['order'][av_pool['pos']:av_pool['pos']+args.av_batch_size]; av_pool['pos']+=args.av_batch_size
        tok=av_pool['tokens'][ids]; moves=av_pool['moves'][ids]
        xb=compact_tokens_to_residual_planes(tok, args.history_plies, args.state_planes)
        aid=chessbench_classes_to_action_ids(moves); mf=av_move_features_from_tokens(tok, moves)
        return to_dev(xb,torch.float32), to_dev(aid,torch.long), to_dev(mf,torch.float32), to_dev(av_pool['mask'][ids],torch.float32), to_dev(av_pool['values'][ids],torch.float32), to_dev(av_pool['regrets'][ids],torch.float32)

    global_step=0; start=time.time(); sums={}; counts={}
    for ep in range(1,args.epochs+1):
        sched=['policy']*math.ceil(args.policy_rows/args.batch_size) + ['av']*math.ceil(args.av_positions/args.av_batch_size)
        rng.shuffle(sched); net.train()
        for st,kind in enumerate(sched,1):
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda',enabled=amp_enabled,dtype=amp_dtype):
                if kind == 'policy':
                    xb,aid,mf,mask,slot,wdl_t=policy_batch(); pol,wdl,_,_,_=net(xb,aid,mf,mask); valid=slot>=0
                    ploss=F.cross_entropy(pol[valid].float(), slot[valid]) if bool(valid.any()) else pol.sum()*0
                    wloss=(-(F.log_softmax(wdl.float(),1)*wdl_t).sum(1)).mean()
                    loss=args.policy_weight*ploss + args.wdl_weight*wloss
                else:
                    xb,aid,mf,mask,val,reg=av_batch(); _,_,av,rank,regret=net(xb,aid,mf,mask); mbool=mask > 0
                    aloss=F.smooth_l1_loss(av[mbool].float(), val[mbool]) if bool(mbool.any()) else av.sum()*0
                    rtarget=val.masked_fill(~mbool, -1e9).argmax(1)
                    rank_loss=F.cross_entropy(rank.float().masked_fill(~mbool, -1e9), rtarget)
                    regret_loss=F.smooth_l1_loss(regret[mbool].float(), reg[mbool]) if bool(mbool.any()) else regret.sum()*0
                    loss=args.av_weight*aloss + args.rank_weight*rank_loss + args.regret_weight*regret_loss
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); global_step += 1
            sums[kind]=sums.get(kind,0.0)+float(loss.detach()); counts[kind]=counts.get(kind,0)+1
            if args.progress_every and global_step % args.progress_every == 0:
                msg=' '.join(f'{k}_loss={sums[k]/max(1,counts[k]):.4f}' for k in sorted(sums)); print(f'progress epoch={ep} step={global_step}/{len(sched)} seconds={time.time()-start:.1f} {msg}', flush=True)
            if args.checkpoint_dir and args.checkpoint_every_steps and global_step % args.checkpoint_every_steps == 0:
                Path(args.checkpoint_dir).mkdir(parents=True,exist_ok=True); torch.save({'model':net.state_dict(),'step':global_step,'args':vars(args)}, Path(args.checkpoint_dir)/'checkpoint_latest.pt')

    try:
        from train_residual_torch import fixed_policy_moves
        moves_meta=fixed_policy_moves()
    except Exception:
        moves_meta=[]
    legal_ks=[int(x) for x in args.onnx_legal_ks.replace(';',',').split(',') if x.strip()] if args.onnx_out else []
    meta={'kind':'squaremoveformer_cnn_av_v1','architecture':'cnn_square_move_transformer','policy_map':'uci_queen_knight_promo_v1','moves':moves_meta,'input_planes':C,'history_plies':args.history_plies,'channels':args.channels,'blocks':args.blocks,'token_dim':args.token_dim,'heads':args.heads,'layers':args.layers,'ff_dim':args.ff_dim,'num_move_features':MF,'move_feature_names':MOVE_FEATURE_NAMES,'max_legal_moves':K_side,'action_id_mapping':'(from * 64 + to) * 5 + promo, promo n=1,b=2,r=3,q=4','trained_with_chessbench_av_candidates':True,'trained_with_policy_sidecar':True,'av_head_exported':bool(args.onnx_out),'aux_heads_exported':['action_values','rank_scores','regrets'] if args.onnx_out else [],'separate_aux_heads':True,'onnx_legal_buckets':legal_ks,'onnx_dynamic_batch':bool(args.onnx_out),'onnx_dynamic_legal':False}
    Path(args.out).parent.mkdir(parents=True,exist_ok=True); torch.save({'model':net.state_dict(),'meta':meta,'args':vars(args),'step':global_step}, args.out)
    if args.meta_out: Path(args.meta_out).write_text(json.dumps(meta,separators=(',',':')))
    print(f'METRIC train_steps={global_step}', flush=True)
    for k in sorted(sums): print(f'METRIC train_{k}_loss={sums[k]/max(1,counts[k]):.6f}', flush=True)

    if args.onnx_out:
        class Export(nn.Module):
            def __init__(self,n): super().__init__(); self.n=n
            def forward(self,planes,legal_action_ids,legal_features,legal_mask): return self.n(planes,legal_action_ids,legal_features,legal_mask)
        def bucket_path(base: str, kk: int) -> Path:
            p=Path(base); return p if len(legal_ks)<=1 else p.with_name(f'{p.stem}_k{kk}{p.suffix}')
        net.eval(); Path(args.onnx_out).parent.mkdir(parents=True,exist_ok=True); exported=[]
        for kk in legal_ks:
            out_path=bucket_path(args.onnx_out,kk)
            dynamic_axes={'planes':{0:'batch'},'legal_action_ids':{0:'batch'},'legal_features':{0:'batch'},'legal_mask':{0:'batch'},'policy_logits_legal':{0:'batch'},'wdl_logits':{0:'batch'},'action_values':{0:'batch'},'rank_scores':{0:'batch'},'regrets':{0:'batch'}}
            torch.onnx.export(Export(net).eval(), (torch.zeros(1,C,8,8,device=device), torch.zeros(1,kk,device=device,dtype=torch.long), torch.zeros(1,kk,MF,device=device), torch.ones(1,kk,device=device)), str(out_path), input_names=['planes','legal_action_ids','legal_features','legal_mask'], output_names=['policy_logits_legal','wdl_logits','action_values','rank_scores','regrets'], dynamic_axes=dynamic_axes, opset_version=18, external_data=False, dynamo=False)
            m=dict(meta); m.update({'onnx_fixed_legal_moves':kk,'onnx_legal_length_mode':'fixed_bucket','onnx_file':str(out_path)})
            out_path.with_suffix('.meta.json').write_text(json.dumps(m,separators=(',',':')))
            exported.append(str(out_path)); print(f'METRIC squaremoveformer_onnx_export_k{kk}=1', flush=True)
        meta['onnx_exports']=exported
        if args.meta_out: Path(args.meta_out).write_text(json.dumps(meta,separators=(',',':')))

if __name__ == '__main__': main()
