#!/usr/bin/env python3
from __future__ import annotations
import argparse, contextlib, json, math, random, subprocess, time
from pathlib import Path
import numpy as np

try:
    from training._lib.encoding import CHESSBENCH_PROMOTIONS as PROMOS, FILES, PIECES
except ModuleNotFoundError:
    from _lib.encoding import CHESSBENCH_PROMOTIONS as PROMOS, FILES, PIECES


try:
    import pyzstd  # type: ignore
except Exception:
    pyzstd = None


@contextlib.contextmanager
def opener(path: str):
    if path.endswith(".zst"):
        if pyzstd is not None:
            with pyzstd.open(path, "rt") as f:
                yield f
        else:
            p = subprocess.Popen(
                ["zstd", "-dc", path], stdout=subprocess.PIPE, text=True
            )
            try:
                yield p.stdout
            finally:
                if p.stdout:
                    p.stdout.close()
                rc = p.wait()
                if rc and rc != -13:
                    raise subprocess.CalledProcessError(rc, ["zstd", "-dc", path])
    else:
        with open(path, "rt", encoding="utf-8") as f:
            yield f


def sq_index(s: str) -> int:
    return (int(s[1]) - 1) * 8 + FILES.index(s[0])


def move_class(uci: str) -> int:
    fr, to = sq_index(uci[:2]), sq_index(uci[2:4])
    if len(uci) >= 5 and uci[4].lower() in PROMOS:
        return 4096 + (fr * 64 + to) * 4 + PROMOS[uci[4].lower()]
    return fr * 64 + to


def parse_board(fen: str):
    board = ["."] * 64
    ranks = fen.split()[0].split("/")
    for rr, rank in enumerate(ranks):
        file = 0
        r = 7 - rr
        for ch in rank:
            if ch.isdigit():
                file += int(ch)
            else:
                board[r * 8 + file] = ch
                file += 1
    return board


def token_features(fen: str, history_fens=None, history: int = 0):
    parts = fen.split()
    stm = parts[1] if len(parts) > 1 else "w"
    cast = parts[2] if len(parts) > 2 else "-"
    ep = parts[3] if len(parts) > 3 else "-"
    half = float(parts[4]) / 100.0 if len(parts) > 4 and parts[4].isdigit() else 0.0
    boards = [parse_board(fen)]
    for hf in (history_fens or [])[:history]:
        boards.append(parse_board(hf))
    while len(boards) < history + 1:
        boards.append(["."] * 64)
    # per square: 13 piece one-hot per board + 8 rule/scalar features
    feats = []
    ep_i = sq_index(ep) if len(ep) == 2 and ep[0] in FILES and ep[1].isdigit() else -1
    for i in range(64):
        v = []
        for b in boards:
            pi = PIECES.index(b[i]) if b[i] in PIECES else 0
            v.extend([1.0 if j == pi else 0.0 for j in range(len(PIECES))])
        v.extend(
            [
                1.0 if stm == "w" else 0.0,
                1.0 if stm == "b" else 0.0,
                1.0 if "K" in cast else 0.0,
                1.0 if "Q" in cast else 0.0,
                1.0 if "k" in cast else 0.0,
                1.0 if "q" in cast else 0.0,
                1.0 if i == ep_i else 0.0,
                half,
            ]
        )
        feats.append(v)
    return feats


def row_from_json(r, history):
    pol = r.get("policy") or {}
    if len(pol) != 1 or "fen" not in r:
        return None
    uci = next(iter(pol.keys()))
    try:
        y = move_class(uci)
        x = token_features(r["fen"], r.get("history_fens") or [], history)
    except Exception:
        return None
    wdl = r.get("wdl") or [0.0, 1.0, 0.0]
    return (x, y, [float(wdl[0]), float(wdl[1]), float(wdl[2])])


def iter_rows(paths, max_rows, history, rng=None):
    paths = list(paths)
    if rng is not None:
        rng.shuffle(paths)
    n = 0
    for p in paths:
        with opener(str(p)) as f:
            for line in f:
                if not line.strip():
                    continue
                rr = row_from_json(json.loads(line), history)
                if rr is None:
                    continue
                yield rr
                n += 1
                if max_rows and n >= max_rows:
                    return


def load_rows(paths, max_rows, history):
    return list(iter_rows(paths, max_rows, history))


def main():
    ap = argparse.ArgumentParser(
        description="Train Chessformer/SquareFormer policy+WDL model from supervised JSONL shards."
    )
    ap.add_argument("--dataset", default="", help="Dataset root with manifest.json")
    ap.add_argument(
        "--cache-manifest", default="", help="SquareFormer token cache_manifest.json"
    )
    ap.add_argument("--train", nargs="*", default=[])
    ap.add_argument("--dev", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--onnx-out", default="")
    ap.add_argument("--meta-out", default="")
    ap.add_argument("--variant", choices=["v0", "v1"], default="v0")
    ap.add_argument("--layers", type=int, default=0)
    ap.add_argument("--d-model", type=int, default=0)
    ap.add_argument("--heads", type=int, default=0)
    ap.add_argument("--d-ff", type=int, default=0)
    ap.add_argument("--history-plies", type=int, default=-1)
    ap.add_argument("--relation-bias", action="store_true")
    ap.add_argument("--max-rows", type=int, default=100000)
    ap.add_argument("--max-dev-rows", type=int, default=20000)
    ap.add_argument(
        "--stream-train",
        action="store_true",
        help="Stream training rows from JSONL/ZST shards instead of preloading token tensors into RAM",
    )
    ap.add_argument(
        "--compact-embeddings",
        action="store_true",
        help="For compact token caches, train directly from uint8 categorical tokens instead of expanding to float one-hot features",
    )
    ap.add_argument(
        "--shuffle-chunk-rows",
        type=int,
        default=65536,
        help="For cache training, shuffle shard/chunk order and rows within chunks instead of globally shuffling all row ids",
    )
    ap.add_argument(
        "--amp", action="store_true", help="Use CUDA mixed precision training/eval"
    )
    ap.add_argument(
        "--amp-dtype",
        choices=["fp16", "bf16"],
        default="bf16",
        help="AMP dtype on CUDA",
    )
    ap.add_argument(
        "--torch-compile",
        action="store_true",
        help="Compile model for training; ONNX export uses the original uncompiled module",
    )
    ap.add_argument(
        "--weight-qat",
        action="store_true",
        help="Constrain trainable matrix weights to an int8 fake-quant grid after each optimizer step (weight-only QAT/fine-tuning).",
    )
    ap.add_argument(
        "--qat-bits", type=int, default=8, help="Fake-quant bits for --weight-qat."
    )
    ap.add_argument(
        "--qat-per-tensor",
        action="store_true",
        help="Use one scale per tensor instead of per output channel for --weight-qat.",
    )
    ap.add_argument(
        "--qat-quantize-embeddings",
        action="store_true",
        help="Also fake-quant embedding matrices during --weight-qat; default leaves embeddings full precision.",
    )
    ap.add_argument(
        "--qat-quantize-pos",
        action="store_true",
        help="Also fake-quant positional parameters during --weight-qat; default leaves them full precision.",
    )
    ap.add_argument(
        "--fused-adamw", action="store_true", help="Use fused CUDA AdamW when available"
    )
    ap.add_argument(
        "--matmul-precision",
        choices=["highest", "high", "medium"],
        default="high",
        help="torch float32 matmul precision",
    )
    ap.add_argument(
        "--checkpoint-dir",
        default="",
        help="Save resumable checkpoint after every epoch",
    )
    ap.add_argument(
        "--resume",
        default="",
        help="Resume model/optimizer/global step from an epoch checkpoint",
    )
    ap.add_argument(
        "--progress-every",
        type=int,
        default=2000,
        help="Print training progress every N batches; 0 disables",
    )
    ap.add_argument(
        "--prefetch-batches",
        type=int,
        default=2,
        help="Background prefetch queue depth for training batches; 0 disables",
    )
    ap.add_argument(
        "--eval-rows",
        type=int,
        default=-1,
        help="Dev rows to evaluate during epochs; -1 follows max-dev-rows/full dev",
    )
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument(
        "--lr-schedule",
        choices=["constant", "cosine"],
        default="constant",
        help="Learning-rate schedule",
    )
    ap.add_argument(
        "--warmup-frac",
        type=float,
        default=0.02,
        help="Fraction of total optimizer steps used for linear warmup with cosine schedule",
    )
    ap.add_argument(
        "--min-lr-frac",
        type=float,
        default=0.1,
        help="Final LR as fraction of base LR for cosine schedule",
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()
    if args.variant == "v0":
        layers = args.layers or 4
        d_model = args.d_model or 64
        heads = args.heads or 4
        d_ff = args.d_ff or 128
        history = 0 if args.history_plies < 0 else args.history_plies
    else:
        layers = args.layers or 6
        d_model = args.d_model or 128
        heads = args.heads or 4
        d_ff = args.d_ff or 256
        history = 2 if args.history_plies < 0 else args.history_plies
    cache = None
    if args.cache_manifest:
        cm = json.loads(Path(args.cache_manifest).read_text())
        cache = {"manifest": cm}
        train_paths = []
        dev_path = Path("")
    elif args.dataset:
        root = Path(args.dataset)
        man = json.loads((root / "manifest.json").read_text())
        train_paths = [root / p for p in man["train_shards"]]
        dev_path = root / man["dev"]
    else:
        train_paths = [Path(p) for p in args.train]
        dev_path = Path(args.dev)
    if cache is not None:
        print(
            f"[squareformer] loading cache manifest {args.cache_manifest}", flush=True
        )
        cm = cache["manifest"]
        cache["train_dirs"] = [Path(p) for p in cm["shards"]]
        cache["dev_dir"] = Path(cm["dev_cache"])
        cache["train_meta"] = [
            json.loads((p / "meta.json").read_text()) for p in cache["train_dirs"]
        ]
        cache["dev_meta"] = json.loads((cache["dev_dir"] / "meta.json").read_text())
        cache["train_rows"] = sum(int(m["rows"]) for m in cache["train_meta"])
        cache["dev_rows"] = int(cache["dev_meta"]["rows"])
        history = int(cm.get("history_plies", history))
        train = None
        dev = None
    else:
        if args.stream_train:
            print(f"[squareformer] streaming train max={args.max_rows}", flush=True)
            train = None
        else:
            print(f"[squareformer] loading train max={args.max_rows}", flush=True)
            train = load_rows(train_paths, args.max_rows, history)
        print(f"[squareformer] loading dev max={args.max_dev_rows}", flush=True)
        dev = load_rows([dev_path], args.max_dev_rows, history)
        if (train is not None and not train) or not dev:
            raise SystemExit("no rows loaded")
    import torch, torch.nn as nn, torch.nn.functional as F

    torch.set_float32_matmul_precision(args.matmul_precision)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    input_dim = (
        (history + 1) * len(PIECES) + 8
        if cache is not None
        else len(dev[0][0][0] if train is None else train[0][0][0])
    )
    policy_size = 4096 + 4096 * 4

    def relation_ids():
        rel = torch.zeros(heads, 64, 64)
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

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.n1 = nn.LayerNorm(d_model)
            self.att = nn.MultiheadAttention(d_model, heads, batch_first=True)
            self.n2 = nn.LayerNorm(d_model)
            self.ff = nn.Sequential(
                nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model)
            )

        def forward(self, x, attn_mask=None):
            y, _ = self.att(
                self.n1(x),
                self.n1(x),
                self.n1(x),
                attn_mask=attn_mask,
                need_weights=False,
            )
            x = x + y
            return x + self.ff(self.n2(x))

    use_compact = bool(args.compact_embeddings and cache is not None)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.use_compact = use_compact
            if self.use_compact:
                self.piece_emb = nn.ModuleList(
                    [nn.Embedding(len(PIECES), d_model) for _ in range(history + 1)]
                )
                self.stm_emb = nn.Embedding(3, d_model)
                self.castle_emb = nn.Embedding(16, d_model)
                self.ep_emb = nn.Embedding(2, d_model)
                self.half_proj = nn.Linear(1, d_model)
            else:
                self.inp = nn.Linear(input_dim, d_model)
            self.pos = nn.Parameter(torch.zeros(64, d_model))
            self.layers = nn.ModuleList([Layer() for _ in range(layers)])
            self.fq = nn.Linear(d_model, d_model)
            self.tk = nn.Linear(d_model, d_model)
            self.prom = nn.Linear(d_model, 64 * 4)
            self.wdl = nn.Linear(d_model, 3)
            self.register_buffer(
                "rel_mask",
                relation_ids() if args.relation_bias else torch.zeros(heads, 64, 64),
                persistent=False,
            )

        def embed_compact(self, x):
            x = x.long()
            h = 0
            for i, emb in enumerate(self.piece_emb):
                h = h + emb(x[:, :, i].clamp(0, len(PIECES) - 1))
            base = history + 1
            h = h + self.stm_emb(x[:, :, base].clamp(0, 2))
            h = h + self.castle_emb(x[:, :, base + 1].clamp(0, 15))
            h = h + self.ep_emb((x[:, :, base + 2] > 0).long())
            h = h + self.half_proj(
                (x[:, :, base + 3].float().clamp(0, 255) / 100.0).unsqueeze(-1)
            )
            return h

        def forward(self, x):
            h = (self.embed_compact(x) if self.use_compact else self.inp(x)) + self.pos
            mask = (
                self.rel_mask.repeat(x.shape[0], 1, 1) if args.relation_bias else None
            )
            for l in self.layers:
                h = l(h, mask)
            fq = self.fq(h)
            tk = self.tk(h)
            ordinary = torch.matmul(fq, tk.transpose(1, 2)) / math.sqrt(d_model)
            promo = self.prom(h).view(x.shape[0], 64, 64, 4)
            pol = torch.cat(
                [
                    ordinary.reshape(x.shape[0], 4096),
                    promo.reshape(x.shape[0], 4096 * 4),
                ],
                1,
            )
            return pol, self.wdl(h.mean(1))

    raw_net = Net().to(args.device)
    resume_ckpt = None
    start_epoch = 1
    if args.resume:
        resume_ckpt = torch.load(args.resume, map_location=args.device)
        raw_net.load_state_dict(resume_ckpt["model"])
        start_epoch = int(resume_ckpt.get("epoch", 0)) + 1
        print(
            f"[squareformer] resumed model from {args.resume} at epoch {start_epoch}",
            flush=True,
        )
    opt_kwargs = {"lr": args.lr, "weight_decay": args.weight_decay}
    if args.fused_adamw and args.device.startswith("cuda"):
        opt_kwargs["fused"] = True
    try:
        opt = torch.optim.AdamW(raw_net.parameters(), **opt_kwargs)
    except TypeError:
        opt_kwargs.pop("fused", None)
        opt = torch.optim.AdamW(raw_net.parameters(), **opt_kwargs)
    net = torch.compile(raw_net) if args.torch_compile else raw_net
    amp_enabled = bool(args.amp and args.device.startswith("cuda"))
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    scaler = torch.amp.GradScaler(
        "cuda", enabled=bool(amp_enabled and amp_dtype is torch.float16)
    )

    def apply_weight_qat_grid_():
        if not args.weight_qat:
            return
        qmax = float((1 << (args.qat_bits - 1)) - 1)
        with torch.no_grad():
            for name, p in raw_net.named_parameters():
                if not p.is_floating_point() or p.ndim < 2:
                    continue
                if (not args.qat_quantize_embeddings) and (
                    ".piece_emb." in name
                    or ".stm_emb." in name
                    or ".castle_emb." in name
                    or ".ep_emb." in name
                ):
                    continue
                if (not args.qat_quantize_pos) and name == "pos":
                    continue
                d = p.data
                if args.qat_per_tensor or d.ndim < 2:
                    scale = d.abs().amax().clamp(min=1e-8) / qmax
                    p.copy_((d / scale).round().clamp(-qmax, qmax) * scale)
                else:
                    flat = d.reshape(d.shape[0], -1)
                    scale = flat.abs().amax(dim=1).clamp(min=1e-8) / qmax
                    p.copy_(
                        (
                            (flat / scale[:, None]).round().clamp(-qmax, qmax)
                            * scale[:, None]
                        ).reshape_as(d)
                    )

    if args.weight_qat:
        print(
            f"[squareformer] weight_qat enabled bits={args.qat_bits} per_channel={not args.qat_per_tensor} quantize_embeddings={args.qat_quantize_embeddings} quantize_pos={args.qat_quantize_pos}",
            flush=True,
        )
    train_rows_for_sched = min(
        args.max_rows or (cache["train_rows"] if cache is not None else len(train)),
        (cache["train_rows"] if cache is not None else len(train)),
    )
    total_steps = max(
        1, math.ceil(train_rows_for_sched / args.batch_size) * args.epochs
    )
    warmup_steps = (
        max(0, int(total_steps * args.warmup_frac))
        if args.lr_schedule == "cosine"
        else 0
    )
    global_step = (
        int(resume_ckpt.get("global_step", 0)) if resume_ckpt is not None else 0
    )
    if resume_ckpt is not None and "optimizer" in resume_ckpt:
        opt.load_state_dict(resume_ckpt["optimizer"])
        print(f"[squareformer] resumed optimizer global_step={global_step}", flush=True)
    apply_weight_qat_grid_()

    def set_lr(step):
        if args.lr_schedule == "constant":
            lr = args.lr
        elif warmup_steps and step < warmup_steps:
            lr = args.lr * float(step + 1) / float(warmup_steps)
        else:
            denom = max(1, total_steps - warmup_steps)
            p = min(1.0, max(0.0, (step - warmup_steps) / denom))
            lr = args.lr * (
                args.min_lr_frac
                + (1.0 - args.min_lr_frac) * 0.5 * (1.0 + math.cos(math.pi * p))
            )
        for g in opt.param_groups:
            g["lr"] = lr
        return lr

    def expand_compact(tok):
        # tok: uint8 [B,64,history+9] -> float one-hot/rule [B,64,(history+1)*13+8]
        B = tok.shape[0]
        arr = np.zeros((B, 64, (history + 1) * len(PIECES) + 8), dtype=np.float32)
        for h in range(history + 1):
            ids = tok[:, :, h].astype(np.int64)
            arr[:, :, h * len(PIECES) : (h + 1) * len(PIECES)] = np.eye(
                len(PIECES), dtype=np.float32
            )[ids]
        base = (history + 1) * len(PIECES)
        arr[:, :, base + 0] = tok[:, :, history + 1] == 1
        arr[:, :, base + 1] = tok[:, :, history + 1] == 2
        flags = tok[:, :, history + 2]
        arr[:, :, base + 2] = (flags & 1) > 0
        arr[:, :, base + 3] = (flags & 2) > 0
        arr[:, :, base + 4] = (flags & 4) > 0
        arr[:, :, base + 5] = (flags & 8) > 0
        arr[:, :, base + 6] = tok[:, :, history + 3].astype(np.float32)
        arr[:, :, base + 7] = tok[:, :, history + 4].astype(np.float32) / 100.0
        return arr

    def tensor_batch(sub):
        return (
            torch.tensor([s[0] for s in sub], dtype=torch.float32, device=args.device),
            torch.tensor([s[1] for s in sub], dtype=torch.long, device=args.device),
            torch.tensor([s[2] for s in sub], dtype=torch.float32, device=args.device),
        )

    def batches(data, shuffle=True):
        idx = list(range(len(data)))
        if shuffle:
            random.shuffle(idx)
        for i in range(0, len(idx), args.batch_size):
            yield tensor_batch([data[j] for j in idx[i : i + args.batch_size]])

    def stream_batches(epoch):
        buf = []
        rng = random.Random(args.seed + epoch)
        for rr in iter_rows(train_paths, args.max_rows, history, rng):
            buf.append(rr)
            if len(buf) >= args.batch_size:
                yield tensor_batch(buf)
                buf = []
        if buf:
            yield tensor_batch(buf)

    def open_cache_dir(d, rows):
        meta = json.loads((d / "meta.json").read_text())
        F = int(meta["token_features"])
        return (
            np.memmap(d / "tokens.uint8", np.uint8, "r", shape=(rows, 64, F)),
            np.memmap(d / "policy.int64", np.int64, "r", shape=(rows,)),
            np.memmap(d / "wdl.float32", np.float32, "r", shape=(rows, 3)),
        )

    cache_train = None
    cache_dev = None
    cache_offsets = []
    if cache is not None:
        total = 0
        cache_train = []
        for d, m in zip(cache["train_dirs"], cache["train_meta"]):
            n = int(m["rows"])
            cache_offsets.append(total)
            total += n
            cache_train.append(open_cache_dir(d, n))
        cache_dev = open_cache_dir(cache["dev_dir"], cache["dev_rows"])

    def to_device(arr, dtype):
        t = torch.as_tensor(arr, dtype=dtype)
        if args.device.startswith("cuda"):
            t = t.pin_memory().to(args.device, non_blocking=True)
        else:
            t = t.to(args.device)
        return t

    def cache_batch_from_arrays(tok, pol, wdl, ids):
        ids = np.asarray(ids, dtype=np.int64)
        xb = (
            np.asarray(tok[ids])
            if use_compact
            else expand_compact(np.asarray(tok[ids]))
        )
        xdtype = torch.long if use_compact else torch.float32
        return (
            to_device(xb, xdtype),
            to_device(np.asarray(pol[ids]), torch.long),
            to_device(np.asarray(wdl[ids]), torch.float32),
        )

    def cache_train_batches(epoch):
        rng = random.Random(args.seed + epoch)
        remaining = min(args.max_rows or cache["train_rows"], cache["train_rows"])
        shard_order = list(range(len(cache_train)))
        rng.shuffle(shard_order)
        for si in shard_order:
            if remaining <= 0:
                break
            tok, pol, wdl = cache_train[si]
            n = min(len(pol), remaining)
            chunk = max(args.batch_size, args.shuffle_chunk_rows)
            chunks = list(range(0, n, chunk))
            rng.shuffle(chunks)
            for start in chunks:
                stop = min(n, start + chunk)
                ids = list(range(start, stop))
                rng.shuffle(ids)
                for off in range(0, len(ids), args.batch_size):
                    yield cache_batch_from_arrays(
                        tok, pol, wdl, ids[off : off + args.batch_size]
                    )
            remaining -= n

    def cache_dev_batches():
        tok, pol, wdl = cache_dev
        limit = args.eval_rows if args.eval_rows >= 0 else args.max_dev_rows
        n = min(limit or cache["dev_rows"], cache["dev_rows"])
        for off in range(0, n, args.batch_size):
            yield cache_batch_from_arrays(
                tok, pol, wdl, range(off, min(n, off + args.batch_size))
            )

    def prefetch_iter(it, depth):
        if depth <= 0:
            yield from it
            return
        import queue, threading

        q = queue.Queue(maxsize=depth)
        sentinel = object()

        def worker():
            try:
                for item in it:
                    q.put(item)
            finally:
                q.put(sentinel)

        threading.Thread(target=worker, daemon=True).start()
        while True:
            item = q.get()
            if item is sentinel:
                break
            yield item

    def evaluate():
        net.eval()
        ce = wc = 0.0
        t1 = t4 = t8 = n = 0
        with torch.no_grad():
            for xb, yb, vb in (
                cache_dev_batches() if cache is not None else batches(dev, False)
            ):
                with torch.amp.autocast("cuda", enabled=amp_enabled, dtype=amp_dtype):
                    pl, wl = net(xb)
                bs = len(yb)
                ce += float(F.cross_entropy(pl.float(), yb, reduction="sum"))
                wc += float((-(F.log_softmax(wl.float(), 1) * vb).sum(1)).sum())
                pred = pl.topk(8, 1).indices
                t1 += int((pred[:, :1] == yb[:, None]).any(1).sum())
                t4 += int((pred[:, :4] == yb[:, None]).any(1).sum())
                t8 += int((pred == yb[:, None]).any(1).sum())
                n += bs
        return {
            "dev_policy_ce": ce / n,
            "dev_wdl_ce": wc / n,
            "dev_policy_top1": t1 / n,
            "dev_policy_top4": t4 / n,
            "dev_policy_top8": t8 / n,
        }

    for ep in range(start_epoch, args.epochs + 1):
        net.train()
        n = 0
        ls = 0.0
        st = time.time()
        train_iter = (
            cache_train_batches(ep)
            if cache is not None
            else (stream_batches(ep) if args.stream_train else batches(train))
        )
        train_iter = prefetch_iter(train_iter, args.prefetch_batches)
        batch_i = 0
        for xb, yb, vb in train_iter:
            batch_i += 1
            lr_now = set_lr(global_step)
            global_step += 1
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled, dtype=amp_dtype):
                pl, wl = net(xb)
                loss = F.cross_entropy(pl, yb) + F.cross_entropy(wl, vb.argmax(1))
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(raw_net.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            apply_weight_qat_grid_()
            ls += float(loss.detach()) * len(yb)
            n += len(yb)
            if args.progress_every and batch_i % args.progress_every == 0:
                dt = time.time() - st
                rps = n / max(dt, 1e-9)
                mem = (
                    torch.cuda.max_memory_allocated() / 1048576
                    if args.device.startswith("cuda")
                    else 0.0
                )
                print(
                    f"progress epoch={ep} rows={n} loss_avg={ls / n:.6f} lr={lr_now:.8g} rows_per_sec={rps:.1f} seconds={dt:.1f} cuda_max_mem_mib={mem:.0f}",
                    flush=True,
                )
        vals = evaluate()
        print(
            f"epoch {ep} train_loss={ls / n:.6f} seconds={time.time() - st:.1f}",
            flush=True,
        )
        for k, v in vals.items():
            print(f"METRIC {k}={v:.6f}", flush=True)
        if args.checkpoint_dir:
            cdir = Path(args.checkpoint_dir)
            cdir.mkdir(parents=True, exist_ok=True)
            ckpt = {
                "model": raw_net.state_dict(),
                "optimizer": opt.state_dict(),
                "epoch": ep,
                "global_step": global_step,
                "lr_schedule": args.lr_schedule,
                "variant": args.variant,
                "input_dim": input_dim,
                "policy_size": policy_size,
                "layers": layers,
                "d_model": d_model,
                "heads": heads,
                "d_ff": d_ff,
                "history_plies": history,
                "relation_bias": bool(args.relation_bias),
                "input_format": "compact_uint8_embeddings"
                if use_compact
                else "float_onehot_rules",
                "metrics": vals,
                "train_loss": ls / n,
            }
            torch.save(ckpt, cdir / f"epoch_{ep}.pt")
            (cdir / f"epoch_{ep}.meta.json").write_text(
                json.dumps({"epoch": ep, "train_loss": ls / n, **vals}, indent=2)
            )
            print(
                f"[squareformer] saved checkpoint {cdir / f'epoch_{ep}.pt'}", flush=True
            )
    apply_weight_qat_grid_()
    input_format = "compact_uint8_embeddings" if use_compact else "float_onehot_rules"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": raw_net.state_dict(),
            "variant": args.variant,
            "input_dim": input_dim,
            "policy_size": policy_size,
            "layers": layers,
            "d_model": d_model,
            "heads": heads,
            "d_ff": d_ff,
            "history_plies": history,
            "relation_bias": bool(args.relation_bias),
            "input_format": input_format,
            "weight_qat": bool(args.weight_qat),
            "qat_bits": args.qat_bits if args.weight_qat else None,
        },
        args.out,
    )
    if args.onnx_out:
        raw_net.eval()
        dummy = (
            torch.zeros(1, 64, history + 9, dtype=torch.long, device=args.device)
            if use_compact
            else torch.zeros(1, 64, input_dim, device=args.device)
        )
        torch.onnx.export(
            raw_net,
            dummy,
            args.onnx_out,
            input_names=["tokens"],
            output_names=["policy", "wdl"],
            dynamic_axes={
                "tokens": {0: "batch"},
                "policy": {0: "batch"},
                "wdl": {0: "batch"},
            },
            opset_version=17,
        )
    if args.meta_out:
        meta = {
            "kind": "squareformer",
            "variant": args.variant,
            "input_dim": input_dim,
            "token_features": history + 9,
            "input_format": input_format,
            "policy_size": policy_size,
            "layers": layers,
            "d_model": d_model,
            "heads": heads,
            "d_ff": d_ff,
            "history_plies": history,
            "relation_bias": bool(args.relation_bias),
            "from_to_policy_size": policy_size,
            "weight_qat": bool(args.weight_qat),
            "qat_bits": args.qat_bits if args.weight_qat else None,
        }
        Path(args.meta_out).write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
