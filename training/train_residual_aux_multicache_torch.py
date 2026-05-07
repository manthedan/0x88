#!/usr/bin/env python3
from __future__ import annotations
import argparse, bisect, json, time
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser(description='Train/evaluate residual policy+WDL model from multiple feature-cache shards, optionally with sparse Stockfish q aux.')
    p.add_argument('--manifest', required=True)
    p.add_argument('--dev-cache', required=True)
    p.add_argument('--resume', default='', help='Optional .pt checkpoint/model to resume/init from')
    p.add_argument('--resume-model-only', action='store_true', help='Load only model weights from --resume; reset optimizer/scheduler epoch state')
    p.add_argument('--out', required=True)
    p.add_argument('--onnx-out', default='')
    p.add_argument('--meta-out', default='')
    p.add_argument('--checkpoint', default='')
    p.add_argument('--best-checkpoint', default='', help='Save best dev-policy-CE checkpoint here')
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--max-steps', type=int, default=0, help='Stop each epoch after this many optimizer steps; 0 = full epoch')
    p.add_argument('--eval-interval-steps', type=int, default=0, help='Dev eval every N steps; 0 = epoch end only')
    p.add_argument('--batch-size', type=int, default=2048)
    p.add_argument('--channels', type=int, default=48)
    p.add_argument('--blocks', type=int, default=5)
    p.add_argument('--policy-head', choices=['spatial', 'hybrid'], default='spatial', help='spatial=flatten tower; hybrid=concat flatten tower with global pooled features')
    p.add_argument('--se', action='store_true', help='Add lightweight squeeze-excitation gating inside residual blocks')
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--min-lr', type=float, default=0.0)
    p.add_argument('--lr-schedule', choices=['constant', 'cosine'], default='constant')
    p.add_argument('--warmup-steps', type=int, default=0)
    p.add_argument('--weight-decay', type=float, default=0.0)
    p.add_argument('--policy-label-smoothing', type=float, default=0.0)
    p.add_argument('--grad-clip-norm', type=float, default=0.0)
    p.add_argument('--ema-decay', type=float, default=0.0)
    p.add_argument('--ema-best-checkpoint', default='', help='Save best EMA dev-policy-CE checkpoint here')
    p.add_argument('--device', default='cuda')
    p.add_argument('--amp', action='store_true', help='Use CUDA automatic mixed precision')
    p.add_argument('--amp-dtype', choices=['fp16','bf16'], default='bf16')
    p.add_argument('--fused-adamw', action='store_true')
    p.add_argument('--matmul-precision', choices=['highest','high','medium'], default='high')
    p.add_argument('--shuffle-chunk-rows', type=int, default=262144, help='Shard/chunk-local shuffle for cache-friendly training; 0 uses global random permutation')
    p.add_argument('--progress-every', type=int, default=500, help='Print progress every N batches; 0 disables')
    p.add_argument('--prefetch-batches', type=int, default=2, help='Background prefetch queue depth for assembled CPU batches; 0 disables')
    p.add_argument('--aux-q-weight', type=float, default=0.0)
    p.add_argument('--max-dev-policy-ce', type=float, default=2.85)
    p.add_argument('--patience', type=int, default=0, help='Early stop after N non-improving evals; 0 disables')
    args = p.parse_args()

    import torch, torch.nn as nn, torch.nn.functional as F
    torch.set_float32_matmul_precision(args.matmul_precision)

    man = json.loads(Path(args.manifest).read_text())
    paths = [Path(s) for s in man['shards']]
    metas = [json.loads((pth / 'meta.json').read_text()) for pth in paths]
    C = int(metas[0]['input_planes']); P = int(metas[0]['policy_size']); moves = metas[0]['moves']
    sizes = [int(m['rows']) for m in metas]
    for m in metas:
        if int(m['input_planes']) != C or int(m['policy_size']) != P:
            raise SystemExit('cache shard metadata mismatch')
    offs = []; total = 0; shards = []
    for pth, n in zip(paths, sizes):
        offs.append(total); total += n
        shards.append((
            np.memmap(pth / 'x.int8', np.int8, 'r', shape=(n, C, 8, 8)),
            np.memmap(pth / 'policy.int64', np.int64, 'r', shape=(n,)),
            np.memmap(pth / 'wdl.float32', np.float32, 'r', shape=(n, 3)),
            np.memmap(pth / 'weight.float32', np.float32, 'r', shape=(n,)),
            np.memmap(pth / 'stockfish_q.float32', np.float32, 'r', shape=(n,)),
        ))

    dc = Path(args.dev_cache); dm = json.loads((dc / 'meta.json').read_text()); DN = int(dm['rows'])
    dx = np.memmap(dc / 'x.int8', np.int8, 'r', shape=(DN, C, 8, 8))
    dy = np.memmap(dc / 'policy.int64', np.int64, 'r', shape=(DN,))
    dv = np.memmap(dc / 'wdl.float32', np.float32, 'r', shape=(DN, 3))
    device = args.device

    class Block(nn.Module):
        def __init__(self, ch):
            super().__init__(); self.c1 = nn.Conv2d(ch, ch, 3, padding=1); self.c2 = nn.Conv2d(ch, ch, 3, padding=1)
            if args.se:
                mid = max(8, ch // 4)
                self.se1 = nn.Linear(ch, mid); self.se2 = nn.Linear(mid, ch)
            else:
                self.se1 = self.se2 = None
        def forward(self, z):
            y = self.c2(F.relu(self.c1(z)))
            if self.se1 is not None:
                gate = torch.sigmoid(self.se2(F.relu(self.se1(y.mean((2, 3)))))).view(y.shape[0], y.shape[1], 1, 1)
                y = y * gate
            return F.relu(y + z)

    class Net(nn.Module):
        def __init__(self):
            super().__init__(); self.stem = nn.Conv2d(C, args.channels, 3, padding=1); self.blocks = nn.Sequential(*[Block(args.channels) for _ in range(args.blocks)])
            policy_in = args.channels * 64 + (args.channels if args.policy_head == 'hybrid' else 0)
            self.policy = nn.Linear(policy_in, P); self.wdl = nn.Linear(args.channels, 3)
        def forward(self, z):
            h = self.blocks(F.relu(self.stem(z))); pooled = h.mean((2, 3)); pf = torch.cat([h.flatten(1), pooled], 1) if args.policy_head == 'hybrid' else h.flatten(1); return self.policy(pf), self.wdl(pooled)

    net = Net().to(device)
    opt_kwargs = {'lr': args.lr, 'weight_decay': args.weight_decay}
    if args.fused_adamw and str(device).startswith('cuda'): opt_kwargs['fused'] = True
    try: opt = torch.optim.AdamW(net.parameters(), **opt_kwargs)
    except TypeError:
        opt_kwargs.pop('fused', None); opt = torch.optim.AdamW(net.parameters(), **opt_kwargs)
    amp_enabled = bool(args.amp and str(device).startswith('cuda'))
    amp_dtype = torch.bfloat16 if args.amp_dtype == 'bf16' else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=bool(amp_enabled and amp_dtype is torch.float16))
    print(f'METRIC amp_enabled={1 if amp_enabled else 0}')
    print(f'METRIC weight_decay={args.weight_decay:.12g}')
    print(f'METRIC policy_label_smoothing={args.policy_label_smoothing:.12g}')
    print(f'METRIC ema_decay={args.ema_decay:.12g}')
    start_epoch = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        state = ck['model'] if isinstance(ck, dict) and 'model' in ck else ck
        net.load_state_dict(state)
        if not args.resume_model_only and isinstance(ck, dict) and 'opt' in ck:
            try: opt.load_state_dict(ck['opt'])
            except Exception: pass
        start_epoch = 0 if args.resume_model_only else (int(ck.get('epoch', 0)) if isinstance(ck, dict) else 0)
        print('METRIC resumed=1')
        print(f'METRIC resume_model_only={1 if args.resume_model_only else 0}')
    else:
        print('METRIC resumed=0')

    def gather(ids):
        xs=[]; ys=[]; vs=[]; ws=[]; qs=[]
        for gid in ids:
            si = bisect.bisect_right(offs, int(gid)) - 1; li = int(gid) - offs[si]
            x, y, v, w, q = shards[si]
            xs.append(x[li]); ys.append(y[li]); vs.append(v[li]); ws.append(w[li]); qs.append(q[li])
        return np.asarray(xs), np.asarray(ys), np.asarray(vs), np.asarray(ws), np.asarray(qs)

    def to_device(arr, dtype):
        t = torch.as_tensor(arr, dtype=dtype)
        if str(device).startswith('cuda'): return t.pin_memory().to(device, non_blocking=True)
        return t.to(device)

    def tensorize(batch):
        xb0, yb0, vb0, wb0, qb0 = batch
        return to_device(xb0, torch.float32), to_device(yb0, torch.long), to_device(vb0, torch.float32), to_device(wb0, torch.float32), to_device(qb0, torch.float32)

    def prefetch_iter(it, depth):
        if depth <= 0:
            yield from it; return
        import queue, threading
        q = queue.Queue(maxsize=depth); sentinel = object()
        def worker():
            try:
                for item in it: q.put(item)
            finally: q.put(sentinel)
        threading.Thread(target=worker, daemon=True).start()
        while True:
            item = q.get()
            if item is sentinel: break
            yield item

    def local_shuffle_batches(epoch):
        rng = np.random.default_rng(13 + epoch)
        shard_order = np.arange(len(shards)); rng.shuffle(shard_order)
        yielded = 0
        for si in shard_order:
            x, y, v, w, q = shards[int(si)]; n = len(y); chunk = max(args.batch_size, args.shuffle_chunk_rows)
            chunks = np.arange(0, n, chunk); rng.shuffle(chunks)
            for start in chunks:
                stop = min(n, int(start) + chunk); ids = np.arange(int(start), stop); rng.shuffle(ids)
                for off0 in range(0, len(ids), args.batch_size):
                    lids = ids[off0:off0 + args.batch_size]
                    yield np.asarray(x[lids]), np.asarray(y[lids]), np.asarray(v[lids]), np.asarray(w[lids]), np.asarray(q[lids])
                    yielded += 1
                    if args.max_steps and yielded >= args.max_steps: return

    ema_state = {k: v.detach().clone() for k, v in net.state_dict().items()} if args.ema_decay > 0 else None

    def update_ema():
        if ema_state is None: return
        with torch.no_grad():
            for k, v in net.state_dict().items():
                if torch.is_floating_point(v): ema_state[k].mul_(args.ema_decay).add_(v.detach(), alpha=1.0 - args.ema_decay)
                else: ema_state[k].copy_(v)

    def eval_dev(tag):
        net.eval(); ce = wc = 0.0; t1 = t4 = t8 = seen = 0
        with torch.no_grad():
            for off in range(0, DN, args.batch_size):
                xb = to_device(np.asarray(dx[off:off + args.batch_size]), torch.float32)
                yb = to_device(np.asarray(dy[off:off + args.batch_size]), torch.long)
                vb = to_device(np.asarray(dv[off:off + args.batch_size]), torch.float32)
                with torch.amp.autocast('cuda', enabled=amp_enabled, dtype=amp_dtype): pl, wl = net(xb)
                bs = len(yb)
                ce += float(F.cross_entropy(pl, yb, reduction='sum'))
                wc += float((-(F.log_softmax(wl, 1) * vb).sum(1)).sum())
                pred = pl.topk(8, 1).indices
                t1 += int((pred[:, :1] == yb[:, None]).any(1).sum())
                t4 += int((pred[:, :4] == yb[:, None]).any(1).sum())
                t8 += int((pred == yb[:, None]).any(1).sum())
                seen += bs
        vals = {'dev_policy_ce': ce / seen, 'dev_wdl_ce': wc / seen, 'dev_policy_top1': t1 / seen, 'dev_policy_top4': t4 / seen, 'dev_policy_top8': t8 / seen}
        for k, v in vals.items(): print(f'METRIC {tag}_{k}={v:.6f}', flush=True)
        return vals

    def eval_with_state(tag, state):
        if state is None: return None
        cur = {k: v.detach().clone() for k, v in net.state_dict().items()}
        net.load_state_dict(state)
        vals = eval_dev(tag)
        net.load_state_dict(cur)
        return vals

    gen = torch.Generator().manual_seed(13)
    steps_per_epoch = (total + args.batch_size - 1) // args.batch_size
    if args.max_steps: steps_per_epoch = min(steps_per_epoch, args.max_steps)
    total_sched_steps = max(1, steps_per_epoch * max(1, args.epochs))
    def set_lr(step):
        if args.lr_schedule == 'constant': lr = args.lr
        else:
            import math
            if args.warmup_steps and step < args.warmup_steps:
                lr = args.lr * float(step + 1) / float(max(1, args.warmup_steps))
            else:
                denom = max(1, total_sched_steps - args.warmup_steps)
                prog = min(1.0, max(0.0, (step - args.warmup_steps) / denom))
                lr = args.min_lr + 0.5 * (args.lr - args.min_lr) * (1.0 + math.cos(math.pi * prog))
        for g in opt.param_groups: g['lr'] = lr
        return lr
    last_vals = None; global_step = 0; best_dev = float('inf'); best_ema_dev = float('inf'); bad_evals = 0
    for ep in range(start_epoch + 1, start_epoch + args.epochs + 1):
        perm = torch.randperm(total, generator=gen) if args.shuffle_chunk_rows <= 0 else None; net.train(); seen = 0; loss_sum = 0.0; t = time.time(); steps = 0
        cpu_batches = local_shuffle_batches(ep) if args.shuffle_chunk_rows > 0 else (gather(perm[off:off + args.batch_size].numpy()) for off in range(0, total, args.batch_size))
        for xb0, yb0, vb0, wb0, qb0 in prefetch_iter(cpu_batches, args.prefetch_batches):
            xb, yb, vb, wb, qb = tensorize((xb0, yb0, vb0, wb0, qb0))
            with torch.amp.autocast('cuda', enabled=amp_enabled, dtype=amp_dtype):
                pl, wl = net(xb); probs = F.softmax(wl, 1); predq = probs[:, 0] - probs[:, 2]
                mask = torch.isfinite(qb); aux = F.mse_loss(predq[mask], qb[mask]) if mask.any() else torch.tensor(0.0, device=device)
                loss = (F.cross_entropy(pl, yb, reduction='none', label_smoothing=args.policy_label_smoothing) * wb).mean() + (-(F.log_softmax(wl, 1) * vb).sum(1) * wb).mean() + args.aux_q_weight * aux
            set_lr(global_step)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward()
            if args.grad_clip_norm > 0:
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip_norm)
            scaler.step(opt); scaler.update(); update_ema()
            loss_sum += float(loss.detach()) * len(yb); seen += len(yb); steps += 1; global_step += 1
            if args.progress_every and steps % args.progress_every == 0:
                elapsed = time.time() - t; mem = torch.cuda.max_memory_allocated() / 1048576 if str(device).startswith('cuda') else 0.0
                print(f'progress epoch={ep} rows={seen} loss_avg={loss_sum/max(1,seen):.6f} lr={opt.param_groups[0]["lr"]:.8g} rows_per_sec={seen/max(elapsed,1e-9):.1f} seconds={elapsed:.1f} cuda_max_mem_mib={mem:.0f}', flush=True)
            if args.eval_interval_steps and global_step % args.eval_interval_steps == 0:
                last_vals = eval_dev(f'step_{global_step}'); net.train()
                if last_vals['dev_policy_ce'] < best_dev:
                    best_dev = last_vals['dev_policy_ce']; bad_evals = 0
                    if args.best_checkpoint:
                        Path(args.best_checkpoint).parent.mkdir(parents=True, exist_ok=True); torch.save({'model': net.state_dict(), 'opt': opt.state_dict(), 'epoch': ep, 'global_step': global_step, 'dev_metrics': last_vals, 'args': vars(args)}, args.best_checkpoint)
                else:
                    bad_evals += 1
                    if args.patience and bad_evals >= args.patience:
                        print(f'METRIC stopped_patience=1', flush=True); break
                ema_vals = eval_with_state(f'step_{global_step}_ema', ema_state)
                if ema_vals and ema_vals['dev_policy_ce'] < best_ema_dev:
                    best_ema_dev = ema_vals['dev_policy_ce']
                    if args.ema_best_checkpoint:
                        Path(args.ema_best_checkpoint).parent.mkdir(parents=True, exist_ok=True); torch.save({'model': ema_state, 'epoch': ep, 'global_step': global_step, 'dev_metrics': ema_vals, 'args': vars(args), 'ema_decay': args.ema_decay}, args.ema_best_checkpoint)
            if args.max_steps and steps >= args.max_steps: break
        elapsed = time.time() - t
        print(f'METRIC epoch_{ep}_loss={loss_sum / max(1, seen):.6f}')
        print(f'METRIC epoch_{ep}_lr={opt.param_groups[0]["lr"]:.12g}')
        print(f'METRIC epoch_{ep}_examples_per_sec={seen / max(1e-9, elapsed):.3f}')
        print(f'METRIC epoch_{ep}_seconds={elapsed:.3f}', flush=True)
        last_vals = eval_dev(f'epoch_{ep}')
        if args.checkpoint:
            Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True); torch.save({'model': net.state_dict(), 'opt': opt.state_dict(), 'epoch': ep, 'global_step': global_step, 'dev_metrics': last_vals, 'args': vars(args)}, args.checkpoint)
        if last_vals['dev_policy_ce'] < best_dev:
            best_dev = last_vals['dev_policy_ce']; bad_evals = 0
            if args.best_checkpoint:
                Path(args.best_checkpoint).parent.mkdir(parents=True, exist_ok=True); torch.save({'model': net.state_dict(), 'opt': opt.state_dict(), 'epoch': ep, 'global_step': global_step, 'dev_metrics': last_vals, 'args': vars(args)}, args.best_checkpoint)
        ema_vals = eval_with_state(f'epoch_{ep}_ema', ema_state)
        if ema_vals and ema_vals['dev_policy_ce'] < best_ema_dev:
            best_ema_dev = ema_vals['dev_policy_ce']
            if args.ema_best_checkpoint:
                Path(args.ema_best_checkpoint).parent.mkdir(parents=True, exist_ok=True); torch.save({'model': ema_state, 'epoch': ep, 'global_step': global_step, 'dev_metrics': ema_vals, 'args': vars(args), 'ema_decay': args.ema_decay}, args.ema_best_checkpoint)
        else:
            bad_evals += 1
            if args.patience and bad_evals >= args.patience:
                print(f'METRIC stopped_patience=1', flush=True); break
        if last_vals['dev_policy_ce'] > args.max_dev_policy_ce:
            print(f'METRIC stopped_dev_policy_ce={last_vals["dev_policy_ce"]:.6f}', flush=True); break

    meta = {'kind': 'tiny_board_residual_onnx_student', 'architecture': 'residual_tower', 'policy_map': 'uci_queen_knight_promo_v1', 'moves': moves, 'channels': args.channels, 'blocks': args.blocks, 'policy_head': args.policy_head, 'se': bool(args.se), 'history_plies': metas[0]['history_plies'], 'input_planes': C, 'onnx': args.onnx_out}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True); torch.save({'model': net.state_dict(), 'meta': meta}, args.out)
    if args.meta_out: Path(args.meta_out).write_text(json.dumps(meta, separators=(',', ':')))
    if args.onnx_out:
        Path(args.onnx_out).parent.mkdir(parents=True, exist_ok=True)
        torch.onnx.export(net, torch.zeros(1, C, 8, 8, device=device), args.onnx_out, input_names=['planes'], output_names=['policy_logits', 'wdl_logits'], dynamic_axes={'planes': {0: 'batch'}, 'policy_logits': {0: 'batch'}, 'wdl_logits': {0: 'batch'}}, opset_version=18, external_data=False)

if __name__ == '__main__': main()
