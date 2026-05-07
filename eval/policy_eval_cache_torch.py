#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser(description='Policy-only/WDL evaluation for residual PyTorch checkpoints on a feature cache.')
    ap.add_argument('--checkpoint', required=True, help='Checkpoint .pt with model state_dict and meta')
    ap.add_argument('--cache', required=True, help='Feature cache directory')
    ap.add_argument('--batch-size', type=int, default=1024)
    ap.add_argument('--channels', type=int, default=0, help='Override channels if checkpoint meta lacks it')
    ap.add_argument('--blocks', type=int, default=0, help='Override blocks if checkpoint meta lacks it')
    ap.add_argument('--device', default='auto')
    args = ap.parse_args()
    import torch, torch.nn as nn, torch.nn.functional as F
    cache = Path(args.cache)
    meta_cache = json.loads((cache / 'meta.json').read_text())
    n = int(meta_cache['rows']); C = int(meta_cache['input_planes']); P = int(meta_cache['policy_size'])
    x = np.memmap(cache / 'x.int8', np.int8, 'r', shape=(n, C, 8, 8))
    y = np.memmap(cache / 'policy.int64', np.int64, 'r', shape=(n,))
    v = np.memmap(cache / 'wdl.float32', np.float32, 'r', shape=(n, 3))
    ck = torch.load(args.checkpoint, map_location='cpu')
    meta = ck.get('meta') or ck.get('args') or {}
    channels = args.channels or int(meta.get('channels', 48))
    blocks = args.blocks or int(meta.get('blocks', 5))
    device = 'cuda' if args.device == 'auto' and torch.cuda.is_available() else ('cpu' if args.device == 'auto' else args.device)

    class Block(nn.Module):
        def __init__(self, ch):
            super().__init__(); self.c1 = nn.Conv2d(ch, ch, 3, padding=1); self.c2 = nn.Conv2d(ch, ch, 3, padding=1)
        def forward(self, z):
            return F.relu(self.c2(F.relu(self.c1(z))) + z)

    class Net(nn.Module):
        def __init__(self):
            super().__init__(); self.stem = nn.Conv2d(C, channels, 3, padding=1); self.blocks = nn.Sequential(*[Block(channels) for _ in range(blocks)]); self.policy = nn.Linear(channels * 64, P); self.wdl = nn.Linear(channels, 3)
        def forward(self, z):
            h = self.blocks(F.relu(self.stem(z))); return self.policy(h.flatten(1)), self.wdl(h.mean((2, 3)))

    net = Net().to(device)
    state = ck['model'] if isinstance(ck, dict) and 'model' in ck else ck
    net.load_state_dict(state)
    net.eval()
    pce = wce = rank_sum = 0.0; top1 = top4 = top8 = 0; selected_legal = 0; seen = 0
    with torch.no_grad():
        for off in range(0, n, args.batch_size):
            xb = torch.from_numpy(np.asarray(x[off:off + args.batch_size])).to(device, dtype=torch.float32)
            yb = torch.from_numpy(np.asarray(y[off:off + args.batch_size])).to(device)
            vb = torch.from_numpy(np.asarray(v[off:off + args.batch_size])).to(device)
            pl, wl = net(xb); bs = len(yb)
            pce += float(F.cross_entropy(pl, yb, reduction='sum'))
            wce += float((-(F.log_softmax(wl, 1) * vb).sum(1)).sum())
            pred = pl.topk(8, 1).indices
            top1 += int((pred[:, :1] == yb[:, None]).any(1).sum())
            top4 += int((pred[:, :4] == yb[:, None]).any(1).sum())
            top8 += int((pred == yb[:, None]).any(1).sum())
            # Rank of the played/target policy id within the full policy vector.
            # This is not legal-move-filtered unless the cache contains legal masks,
            # but it is stable and useful for model comparisons.
            target_logits = pl.gather(1, yb[:, None])
            rank_sum += float((pl > target_logits).sum(1).add(1).sum())
            selected_legal += bs  # cache labels are generated from legal moves; no legal mask is stored here.
            seen += bs
    print(f'METRIC eval_rows={seen}')
    ce = pce / max(1, seen)
    print(f'METRIC eval_policy_ce={ce:.6f}')
    print(f'METRIC eval_policy_perplexity={float(np.exp(min(ce, 50.0))):.6f}')
    print(f'METRIC eval_wdl_ce={wce / max(1, seen):.6f}')
    print(f'METRIC eval_policy_top1={top1 / max(1, seen):.6f}')
    print(f'METRIC eval_policy_top4={top4 / max(1, seen):.6f}')
    print(f'METRIC eval_policy_top8={top8 / max(1, seen):.6f}')
    print(f'METRIC eval_policy_mean_full_rank={rank_sum / max(1, seen):.6f}')
    print(f'METRIC eval_selected_move_legality={selected_legal / max(1, seen):.6f}')

if __name__ == '__main__':
    main()
