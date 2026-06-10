#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math
from pathlib import Path

import numpy as np

try:
    from training._lib.encoding import PIECES
except ModuleNotFoundError:
    from _lib.encoding import PIECES

import torch
import torch.nn as nn

POLICY_SIZE = 4096 + 4096 * 4


def relation_ids(heads: int):
    rel = np.zeros((heads, 64, 64), dtype=np.float32)
    for i in range(64):
        r1, f1 = divmod(i, 8)
        for j in range(64):
            r2, f2 = divmod(j, 8)
            val = 0
            if i == j:
                val += 1
            if r1 == r2:
                val += 2
            if f1 == f2:
                val += 3
            if abs(r1 - r2) == abs(f1 - f2):
                val += 4
            if (abs(r1 - r2), abs(f1 - f2)) in ((1, 2), (2, 1)):
                val += 5
            if max(abs(r1 - r2), abs(f1 - f2)) == 1:
                val += 6
            if (r1 + f1) & 1 == (r2 + f2) & 1:
                val += 1
            rel[:, i, j] = val / 8.0
    return rel


class SquareFormerV2(nn.Module):
    def __init__(
        self,
        *,
        history: int,
        layers: int,
        d_model: int,
        heads: int,
        d_ff: int,
        input_mode: str,
        relation_bias: bool,
    ):
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
        self.layers = nn.ModuleList(
            [self.Layer(d_model, heads, d_ff) for _ in range(layers)]
        )
        self.fq = nn.Linear(d_model, d_model)
        self.tk = nn.Linear(d_model, d_model)
        self.prom = nn.Linear(d_model, 64 * 4)
        self.wdl = nn.Linear(d_model, 3)
        self.q = nn.Linear(d_model, 1)
        self.promo_emb = nn.Embedding(5, d_model)
        self.av = nn.Sequential(
            nn.Linear(d_model * 4, d_model), nn.GELU(), nn.Linear(d_model, 1)
        )
        self.register_buffer(
            "rel_mask",
            torch.tensor(relation_ids(heads))
            if relation_bias
            else torch.zeros(heads, 64, 64),
            persistent=False,
        )
        self.register_buffer(
            "hist_offsets",
            torch.arange(history + 1, dtype=torch.long) * len(PIECES),
            persistent=False,
        )
        self.relation_bias = relation_bias

    class Layer(nn.Module):
        def __init__(self, d_model: int, heads: int, d_ff: int):
            super().__init__()
            self.n1 = nn.LayerNorm(d_model)
            self.att = nn.MultiheadAttention(d_model, heads, batch_first=True)
            self.n2 = nn.LayerNorm(d_model)
            self.ff = nn.Sequential(
                nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model)
            )

        def forward(self, x, attn_mask=None):
            nx = self.n1(x)
            y, _ = self.att(nx, nx, nx, attn_mask=attn_mask, need_weights=False)
            x = x + y
            return x + self.ff(self.n2(x))

    def token_embed(self, x):
        x = x.long()
        h = self.history
        piece = x[:, :, : h + 1].clamp(0, len(PIECES) - 1) + self.hist_offsets.view(
            1, 1, -1
        )
        out = self.piece_emb(piece).sum(2)
        stm = x[:, :, h + 1].clamp(0, 2)
        flags = x[:, :, h + 2].long()
        ep = x[:, :, h + 3].float()
        half = x[:, :, h + 4].float() / 100.0
        fb = torch.stack(
            [((flags >> i) & 1).float() for i in range(4)] + [ep, half], -1
        )
        rank = x[:, :, h + 5].clamp(0, 7)
        file = x[:, :, h + 6].clamp(0, 7)
        color = x[:, :, h + 7].clamp(0, 1)
        sq = x[:, :, h + 8].clamp(0, 63)
        return (
            out
            + self.stm_emb(stm)
            + self.flag_linear(fb)
            + self.rank_emb(rank)
            + self.file_emb(file)
            + self.color_emb(color)
            + self.square_emb(sq)
        )

    def encode(self, x):
        h = (
            self.token_embed(x) if self.input_mode == "embedding" else self.inp(x)
        ) + self.pos
        mask = self.rel_mask.repeat(x.shape[0], 1, 1) if self.relation_bias else None
        for layer in self.layers:
            h = layer(h, mask)
        return h

    def forward(self, x):
        h = self.encode(x)
        fq = self.fq(h)
        tk = self.tk(h)
        ordinary = torch.matmul(fq, tk.transpose(1, 2)) / math.sqrt(self.d_model)
        promo = self.prom(h).view(x.shape[0], 64, 64, 4)
        pol = torch.cat(
            [ordinary.reshape(x.shape[0], 4096), promo.reshape(x.shape[0], 4096 * 4)], 1
        )
        pooled = h.mean(1)
        return pol, self.wdl(pooled), self.q(pooled).squeeze(1), h

    def av_scores(self, h, moves):
        moves = moves.long().clamp(0, POLICY_SIZE - 1)
        ordinary = moves < 4096
        ft = torch.where(ordinary, moves, (moves - 4096) // 4)
        fr = (ft // 64).clamp(0, 63)
        to = (ft % 64).clamp(0, 63)
        promo = torch.where(
            ordinary, torch.full_like(moves, 4), (moves - 4096) % 4
        ).clamp(0, 4)
        b = torch.arange(h.shape[0], device=h.device)[:, None]
        hf = h[b, fr]
        ht = h[b, to]
        pooled = h.mean(1)[:, None, :].expand(-1, moves.shape[1], -1)
        pe = self.promo_emb(promo)
        return self.av(torch.cat([pooled, hf, ht, pe], -1)).squeeze(-1)


class ExportWrapper(nn.Module):
    def __init__(self, net: SquareFormerV2):
        super().__init__()
        self.net = net

    def forward(self, tokens, legal_action_ids):
        policy, wdl, q, hidden = self.net(tokens)
        action_values = torch.tanh(self.net.av_scores(hidden, legal_action_ids))
        return policy, wdl, q, hidden, action_values


class DeployExportWrapper(nn.Module):
    """Deployment-shaped export (tiny handoff Item 2, 2026-06-10).

    Shrinks the runtime graph surface without changing what the network
    computes (verified levers, not speculation — each one removes a pattern
    that cost real debugging time in the TVM/WebGPU lane):

    - fixed batch (set at export): kills the onnxsim prefold requirement and
      all Shape-op dim arithmetic at the source;
    - int32 tokens at entry: removes the i64 graph input and per-runtime
      dtype overrides (the cast to int64 happens once, in-graph);
    - square-static embedding fold: rank/file/color/square/pos embeddings are
      functions of the square index alone under the compact square-token
      schema (the JS encoder writes the static square values), so they fold
      into ONE constant [64, d] table — removes 4 gathers + adds from the
      stem chain that hits WebGPU's storage-buffer limit when fused;
    - history sum as matmul: the [b,64,h+1,d] -> ReduceSum -> [b,64,d] ply
      sum crashes both TVM schedulers (rfactor bind); an equivalent
      matmul-with-ones compiles everywhere;
    - no clamps: the feature encoder constructs the tokens and already
      guarantees ranges; in-graph clamp() generates the isnan/where/bool-i8
      patterns WGSL rejects;
    - output trim: policy + wdl only (q is derivable, hidden and
      action_values are diagnostics) — halves GPU readback traffic;
    - optional debug outputs: stem end + each layer residual appended to the
      outputs, turning future numerics bisections into a 10-minute diff.
      (Note: extra outputs change fusion and can shift compiler bugs around.)

    NOTE: the currently shipped bt4_anneal_muon_best meta declares
    token_features=24 (16 + 8 lc0 repetition features) and
    relation_bias_mode=template_bank, which this reference wrapper predates —
    port these flags onto the trainer's current exporter before the next
    export cycle.
    """

    def __init__(
        self,
        net: SquareFormerV2,
        *,
        int32_tokens: bool = True,
        fold_square_static: bool = True,
        no_clamp: bool = True,
        debug_outputs: bool = False,
    ):
        super().__init__()
        self.net = net
        self.int32_tokens = int32_tokens
        self.no_clamp = no_clamp
        self.debug_outputs = debug_outputs
        h = net.history
        d = net.d_model
        if fold_square_static:
            with torch.no_grad():
                device = net.pos.device
                squares = torch.arange(64, device=device)
                ranks = squares // 8
                files = squares % 8
                colors = (ranks + files) & 1
                table = (
                    net.rank_emb.weight[ranks]
                    + net.file_emb.weight[files]
                    + net.color_emb.weight[colors]
                    + net.square_emb.weight[squares]
                    + net.pos
                )
            self.register_buffer("square_static", table.clone(), persistent=False)
        else:
            self.register_buffer("square_static", None, persistent=False)
        # Ones column for the history-sum-as-matmul rewrite.
        self.register_buffer(
            "hist_ones", torch.ones(h + 1, 1, dtype=net.piece_emb.weight.dtype), persistent=False
        )

    def _token_embed(self, x):
        net = self.net
        h = net.history
        piece = x[:, :, : h + 1]
        if not self.no_clamp:
            piece = piece.clamp(0, len(PIECES) - 1)
        piece = piece + net.hist_offsets.view(1, 1, -1)
        emb = net.piece_emb(piece)  # [b, 64, h+1, d]
        # ReduceSum over the ply axis crashes TVM schedulers; matmul-with-ones
        # is the same contraction expressed as a GEMM.
        out = (emb.transpose(2, 3) @ self.hist_ones).squeeze(-1)
        stm = x[:, :, h + 1]
        flags = x[:, :, h + 2]
        ep = x[:, :, h + 3].float()
        half = x[:, :, h + 4].float() / 100.0
        fb = torch.stack(
            [((flags >> i) & 1).float() for i in range(4)] + [ep, half], -1
        )
        if not self.no_clamp:
            stm = stm.clamp(0, 2)
        out = out + net.stm_emb(stm) + net.flag_linear(fb)
        if self.square_static is not None:
            return out + self.square_static
        rank = x[:, :, h + 5]
        file = x[:, :, h + 6]
        color = x[:, :, h + 7]
        sq = x[:, :, h + 8]
        if not self.no_clamp:
            rank, file = rank.clamp(0, 7), file.clamp(0, 7)
            color, sq = color.clamp(0, 1), sq.clamp(0, 63)
        return (
            out
            + net.rank_emb(rank)
            + net.file_emb(file)
            + net.color_emb(color)
            + net.square_emb(sq)
            + net.pos
        )

    def forward(self, tokens):
        net = self.net
        x = tokens.long() if self.int32_tokens else tokens
        if net.input_mode == "embedding":
            stem = self._token_embed(x)
        else:
            stem = net.inp(tokens) + net.pos
        debug = [stem] if self.debug_outputs else []
        mask = (
            net.rel_mask.repeat(tokens.shape[0], 1, 1) if net.relation_bias else None
        )
        h = stem
        for layer in net.layers:
            h = layer(h, mask)
            if self.debug_outputs:
                debug.append(h)
        fq = net.fq(h)
        tk = net.tk(h)
        ordinary = torch.matmul(fq, tk.transpose(1, 2)) / math.sqrt(net.d_model)
        promo = net.prom(h).view(tokens.shape[0], 64, 64, 4)
        pol = torch.cat(
            [
                ordinary.reshape(tokens.shape[0], 4096),
                promo.reshape(tokens.shape[0], 4096 * 4),
            ],
            1,
        )
        pooled = h.mean(1)
        return (pol, net.wdl(pooled), *debug)


def main():
    ap = argparse.ArgumentParser(
        description="Export SquareFormer V2 with fixed-width candidate action_values output."
    )
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--onnx-out", required=True)
    ap.add_argument("--meta-out", required=True)
    ap.add_argument("--legal-width", type=int, default=128)
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--deploy-batch", type=int, default=0,
        help="Export a deployment-shaped graph with this FIXED batch size instead of the "
             "dynamic-batch AV export: int32 tokens input, square-static embeddings folded, "
             "history sum as matmul, no clamps, outputs trimmed to policy+wdl. "
             "0 (default) keeps the original AV export.")
    ap.add_argument("--debug-export", action="store_true",
                    help="Deploy export only: append stem + per-layer residual outputs for numerics bisection.")
    ap.add_argument("--keep-clamps", action="store_true",
                    help="Deploy export only: keep the in-graph token clamps.")
    ap.add_argument("--no-fold-square-static", action="store_true",
                    help="Deploy export only: keep per-token rank/file/color/square gathers.")
    ap.add_argument("--keep-int64-tokens", action="store_true",
                    help="Deploy export only: keep the int64 tokens input dtype.")
    args = ap.parse_args()
    meta = json.loads(Path(args.meta).read_text())
    ck = torch.load(args.checkpoint, map_location=args.device)
    state = ck.get("model", ck) if isinstance(ck, dict) else ck
    history = int(
        meta.get(
            "history_plies", ck.get("history_plies", 2) if isinstance(ck, dict) else 2
        )
    )
    net = SquareFormerV2(
        history=history,
        layers=int(
            meta.get("layers", ck.get("layers", 6) if isinstance(ck, dict) else 6)
        ),
        d_model=int(
            meta.get("d_model", ck.get("d_model", 128) if isinstance(ck, dict) else 128)
        ),
        heads=int(meta.get("heads", ck.get("heads", 4) if isinstance(ck, dict) else 4)),
        d_ff=int(
            meta.get("d_ff", ck.get("d_ff", 256) if isinstance(ck, dict) else 256)
        ),
        input_mode=str(
            meta.get(
                "input_mode",
                ck.get("input_mode", "embedding")
                if isinstance(ck, dict)
                else "embedding",
            )
        ),
        relation_bias=bool(
            meta.get(
                "relation_bias",
                ck.get("relation_bias", False) if isinstance(ck, dict) else False,
            )
        ),
    ).to(args.device)
    missing, unexpected = net.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"WARN load_state missing={len(missing)} unexpected={len(unexpected)}")
    net.eval()
    token_features = int(meta.get("token_features", history + 9))
    if args.deploy_batch > 0:
        deploy_errors = []
        expected_token_features = history + 9
        if net.input_mode == "embedding" and token_features != expected_token_features:
            deploy_errors.append(
                f"token_features={token_features} is unsupported by DeployExportWrapper; "
                f"expected compact history+9 layout ({expected_token_features})"
            )
        relation_bias_mode = str(meta.get("relation_bias_mode", "legacy"))
        if net.relation_bias and relation_bias_mode not in {"legacy", "", "None"}:
            deploy_errors.append(f"relation_bias_mode={relation_bias_mode!r} is not implemented in DeployExportWrapper")
        if bool(meta.get("dynamic_relation_gate", False)):
            deploy_errors.append("dynamic_relation_gate is not implemented in DeployExportWrapper")
        if deploy_errors:
            raise SystemExit("--deploy-batch export would be incorrect for this checkpoint: " + "; ".join(deploy_errors))
        deploy = DeployExportWrapper(
            net,
            int32_tokens=not args.keep_int64_tokens,
            fold_square_static=not args.no_fold_square_static,
            no_clamp=not args.keep_clamps,
            debug_outputs=args.debug_export,
        ).to(args.device).eval()
        if net.input_mode == "embedding":
            token_dtype = torch.int64 if args.keep_int64_tokens else torch.int32
            dummy = torch.zeros(args.deploy_batch, 64, token_features, dtype=token_dtype, device=args.device)
        else:
            dummy = torch.zeros(args.deploy_batch, 64, net.input_dim, dtype=torch.float32, device=args.device)
        output_names = ["policy", "wdl"]
        if args.debug_export:
            output_names += ["debug_stem"] + [f"debug_layer_{i}" for i in range(net.layers_n)]
        Path(args.onnx_out).parent.mkdir(parents=True, exist_ok=True)
        torch.onnx.export(
            deploy,
            (dummy,),
            args.onnx_out,
            input_names=["tokens"],
            output_names=output_names,
            # No dynamic_axes on purpose: fixed batch keeps Shape-op dim
            # arithmetic out of the deploy graph entirely.
            opset_version=18,
            external_data=False,
        )
        out_meta = dict(meta)
        out_meta.update(
            {
                "kind": "squareformer_v2",
                "deploy_export": True,
                "deploy_batch": args.deploy_batch,
                "tokens_dtype": "int64" if args.keep_int64_tokens else "int32",
                "square_static_folded": not args.no_fold_square_static,
                "clamps_removed": not args.keep_clamps,
                "outputs": output_names,
            }
        )
        Path(args.meta_out).write_text(json.dumps(out_meta, indent=2))
        print(f"WROTE {args.onnx_out} (deploy batch={args.deploy_batch})")
        print(f"WROTE {args.meta_out}")
        return
    wrapper = ExportWrapper(net).to(args.device).eval()
    if net.input_mode == "embedding":
        dummy_tokens = torch.zeros(
            1, 64, token_features, dtype=torch.long, device=args.device
        )
    else:
        dummy_tokens = torch.zeros(
            1, 64, net.input_dim, dtype=torch.float32, device=args.device
        )
    dummy_moves = torch.zeros(1, args.legal_width, dtype=torch.long, device=args.device)
    Path(args.onnx_out).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (dummy_tokens, dummy_moves),
        args.onnx_out,
        input_names=["tokens", "legal_action_ids"],
        output_names=["policy", "wdl", "q", "hidden", "action_values"],
        dynamic_axes={
            "tokens": {0: "batch"},
            "legal_action_ids": {0: "batch"},
            "policy": {0: "batch"},
            "wdl": {0: "batch"},
            "q": {0: "batch"},
            "hidden": {0: "batch"},
            "action_values": {0: "batch"},
        },
        opset_version=18,
        external_data=False,
    )
    out_meta = dict(meta)
    out_meta.update(
        {
            "kind": "squareformer_v2",
            "av_head_exported": True,
            "action_value_move_encoding": "chessbench_compact_20480",
            "max_legal_moves": args.legal_width,
            "onnx_fixed_legal_moves": args.legal_width,
            "outputs": ["policy", "wdl", "q", "hidden", "action_values"],
        }
    )
    Path(args.meta_out).write_text(json.dumps(out_meta, indent=2))
    print(f"WROTE {args.onnx_out}")
    print(f"WROTE {args.meta_out}")


if __name__ == "__main__":
    main()
