#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

PIECES = '.PNBRQKpnbrqk'
POLICY_SIZE = 4096 + 4096 * 4


def relation_ids(heads: int):
    rel = np.zeros((heads, 64, 64), dtype=np.float32)
    for i in range(64):
        r1, f1 = divmod(i, 8)
        for j in range(64):
            r2, f2 = divmod(j, 8); val = 0
            if i == j: val += 1
            if r1 == r2: val += 2
            if f1 == f2: val += 3
            if abs(r1-r2) == abs(f1-f2): val += 4
            if (abs(r1-r2), abs(f1-f2)) in ((1,2),(2,1)): val += 5
            if max(abs(r1-r2), abs(f1-f2)) == 1: val += 6
            if (r1+f1)&1 == (r2+f2)&1: val += 1
            rel[:, i, j] = val / 8.0
    return rel


class SquareFormerV2(nn.Module):
    def __init__(self, *, history: int, layers: int, d_model: int, heads: int, d_ff: int, input_mode: str, relation_bias: bool):
        super().__init__()
        self.history = history
        self.layers_n = layers
        self.d_model = d_model
        self.heads = heads
        self.d_ff = d_ff
        self.input_mode = input_mode
        self.input_dim = (history + 1) * len(PIECES) + 8
        self.inp = nn.Linear(self.input_dim, d_model)
        self.piece_emb = nn.Embedding((history + 1) * len(PIECES), d_model)
        self.stm_emb = nn.Embedding(3, d_model)
        self.flag_linear = nn.Linear(6, d_model)
        self.rank_emb = nn.Embedding(8, d_model)
        self.file_emb = nn.Embedding(8, d_model)
        self.color_emb = nn.Embedding(2, d_model)
        self.square_emb = nn.Embedding(64, d_model)
        self.pos = nn.Parameter(torch.zeros(64, d_model))
        self.layers = nn.ModuleList([self.Layer(d_model, heads, d_ff) for _ in range(layers)])
        self.fq = nn.Linear(d_model, d_model)
        self.tk = nn.Linear(d_model, d_model)
        self.prom = nn.Linear(d_model, 64 * 4)
        self.wdl = nn.Linear(d_model, 3)
        self.q = nn.Linear(d_model, 1)
        self.promo_emb = nn.Embedding(5, d_model)
        self.av = nn.Sequential(nn.Linear(d_model * 4, d_model), nn.GELU(), nn.Linear(d_model, 1))
        self.register_buffer('rel_mask', torch.tensor(relation_ids(heads)) if relation_bias else torch.zeros(heads, 64, 64), persistent=False)
        self.register_buffer('hist_offsets', torch.arange(history + 1, dtype=torch.long) * len(PIECES), persistent=False)
        self.relation_bias = relation_bias

    class Layer(nn.Module):
        def __init__(self, d_model: int, heads: int, d_ff: int):
            super().__init__()
            self.n1 = nn.LayerNorm(d_model)
            self.att = nn.MultiheadAttention(d_model, heads, batch_first=True)
            self.n2 = nn.LayerNorm(d_model)
            self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
        def forward(self, x, attn_mask=None):
            nx = self.n1(x)
            y, _ = self.att(nx, nx, nx, attn_mask=attn_mask, need_weights=False)
            x = x + y
            return x + self.ff(self.n2(x))

    def token_embed(self, x):
        x = x.long()
        h = self.history
        piece = x[:, :, :h+1].clamp(0, len(PIECES)-1) + self.hist_offsets.view(1, 1, -1)
        out = self.piece_emb(piece).sum(2)
        stm = x[:, :, h+1].clamp(0, 2)
        flags = x[:, :, h+2].long()
        ep = x[:, :, h+3].float()
        half = x[:, :, h+4].float() / 100.0
        fb = torch.stack([((flags >> i) & 1).float() for i in range(4)] + [ep, half], -1)
        rank = x[:, :, h+5].clamp(0, 7)
        file = x[:, :, h+6].clamp(0, 7)
        color = x[:, :, h+7].clamp(0, 1)
        sq = x[:, :, h+8].clamp(0, 63)
        return out + self.stm_emb(stm) + self.flag_linear(fb) + self.rank_emb(rank) + self.file_emb(file) + self.color_emb(color) + self.square_emb(sq)

    def encode(self, x):
        h = (self.token_embed(x) if self.input_mode == 'embedding' else self.inp(x)) + self.pos
        mask = self.rel_mask.repeat(x.shape[0], 1, 1) if self.relation_bias else None
        for layer in self.layers:
            h = layer(h, mask)
        return h

    def forward(self, x):
        h = self.encode(x)
        fq = self.fq(h); tk = self.tk(h)
        ordinary = torch.matmul(fq, tk.transpose(1, 2)) / math.sqrt(self.d_model)
        promo = self.prom(h).view(x.shape[0], 64, 64, 4)
        pol = torch.cat([ordinary.reshape(x.shape[0], 4096), promo.reshape(x.shape[0], 4096 * 4)], 1)
        pooled = h.mean(1)
        return pol, self.wdl(pooled), self.q(pooled).squeeze(1), h

    def av_scores(self, h, moves):
        moves = moves.long().clamp(0, POLICY_SIZE - 1)
        ordinary = moves < 4096
        ft = torch.where(ordinary, moves, (moves - 4096) // 4)
        fr = (ft // 64).clamp(0, 63)
        to = (ft % 64).clamp(0, 63)
        promo = torch.where(ordinary, torch.full_like(moves, 4), (moves - 4096) % 4).clamp(0, 4)
        b = torch.arange(h.shape[0], device=h.device)[:, None]
        hf = h[b, fr]
        ht = h[b, to]
        pooled = h.mean(1)[:, None, :].expand(-1, moves.shape[1], -1)
        pe = self.promo_emb(promo)
        return self.av(torch.cat([pooled, hf, ht, pe], -1)).squeeze(-1)


class ExportWrapper(nn.Module):
    def __init__(self, net: SquareFormerV2):
        super().__init__(); self.net = net
    def forward(self, tokens, legal_action_ids):
        policy, wdl, q, hidden = self.net(tokens)
        action_values = torch.tanh(self.net.av_scores(hidden, legal_action_ids))
        return policy, wdl, q, hidden, action_values


def main():
    ap = argparse.ArgumentParser(description='Export SquareFormer V2 with fixed-width candidate action_values output.')
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--meta', required=True)
    ap.add_argument('--onnx-out', required=True)
    ap.add_argument('--meta-out', required=True)
    ap.add_argument('--legal-width', type=int, default=128)
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()
    meta = json.loads(Path(args.meta).read_text())
    ck = torch.load(args.checkpoint, map_location=args.device)
    state = ck.get('model', ck) if isinstance(ck, dict) else ck
    history = int(meta.get('history_plies', ck.get('history_plies', 2) if isinstance(ck, dict) else 2))
    net = SquareFormerV2(
        history=history,
        layers=int(meta.get('layers', ck.get('layers', 6) if isinstance(ck, dict) else 6)),
        d_model=int(meta.get('d_model', ck.get('d_model', 128) if isinstance(ck, dict) else 128)),
        heads=int(meta.get('heads', ck.get('heads', 4) if isinstance(ck, dict) else 4)),
        d_ff=int(meta.get('d_ff', ck.get('d_ff', 256) if isinstance(ck, dict) else 256)),
        input_mode=str(meta.get('input_mode', ck.get('input_mode', 'embedding') if isinstance(ck, dict) else 'embedding')),
        relation_bias=bool(meta.get('relation_bias', ck.get('relation_bias', False) if isinstance(ck, dict) else False)),
    ).to(args.device)
    missing, unexpected = net.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f'WARN load_state missing={len(missing)} unexpected={len(unexpected)}')
    net.eval()
    wrapper = ExportWrapper(net).to(args.device).eval()
    token_features = int(meta.get('token_features', history + 9))
    if net.input_mode == 'embedding':
        dummy_tokens = torch.zeros(1, 64, token_features, dtype=torch.long, device=args.device)
    else:
        dummy_tokens = torch.zeros(1, 64, net.input_dim, dtype=torch.float32, device=args.device)
    dummy_moves = torch.zeros(1, args.legal_width, dtype=torch.long, device=args.device)
    Path(args.onnx_out).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (dummy_tokens, dummy_moves),
        args.onnx_out,
        input_names=['tokens', 'legal_action_ids'],
        output_names=['policy', 'wdl', 'q', 'hidden', 'action_values'],
        dynamic_axes={
            'tokens': {0: 'batch'},
            'legal_action_ids': {0: 'batch'},
            'policy': {0: 'batch'},
            'wdl': {0: 'batch'},
            'q': {0: 'batch'},
            'hidden': {0: 'batch'},
            'action_values': {0: 'batch'},
        },
        opset_version=18,
        external_data=False,
    )
    out_meta = dict(meta)
    out_meta.update({
        'kind': 'squareformer_v2',
        'av_head_exported': True,
        'action_value_move_encoding': 'chessbench_compact_20480',
        'max_legal_moves': args.legal_width,
        'onnx_fixed_legal_moves': args.legal_width,
        'outputs': ['policy', 'wdl', 'q', 'hidden', 'action_values'],
    })
    Path(args.meta_out).write_text(json.dumps(out_meta, indent=2))
    print(f'WROTE {args.onnx_out}')
    print(f'WROTE {args.meta_out}')


if __name__ == '__main__':
    main()
