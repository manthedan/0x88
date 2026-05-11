#!/usr/bin/env python3
from __future__ import annotations
import argparse, contextlib, json, math, random, subprocess, time
from collections import defaultdict
from pathlib import Path
import numpy as np

try:
    from training._lib.encoding import (
        CHESSBENCH_AV_POLICY_SIZE,
        CHESSBENCH_PROMOTIONS as PROMOS,
        FILES,
        PIECES,
        move_to_squareformer_policy_index,
    )
    from training._lib.metrics import install_metric_print_tee
except ModuleNotFoundError:
    from _lib.encoding import (
        CHESSBENCH_AV_POLICY_SIZE,
        CHESSBENCH_PROMOTIONS as PROMOS,
        FILES,
        PIECES,
        move_to_squareformer_policy_index,
    )
    from _lib.metrics import install_metric_print_tee

try:
    import pyzstd  # type: ignore
except Exception:
    pyzstd = None

POLICY_SIZE = CHESSBENCH_AV_POLICY_SIZE


@contextlib.contextmanager
def opener(path: str | Path):
    path = str(path)
    if path.endswith(".zst"):
        if pyzstd is not None:
            with pyzstd.open(path, "rt") as f:
                yield f
        else:
            p = subprocess.Popen(
                ["zstd", "-dc", path], stdout=subprocess.PIPE, text=True
            )
            try:
                assert p.stdout is not None
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
    return move_to_squareformer_policy_index(uci)


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


def token_features(fen: str, history_fens=None, history: int = 2):
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
    ep_i = sq_index(ep) if len(ep) == 2 and ep[0] in FILES and ep[1].isdigit() else -1
    feats = []
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


def q_from_wdl(wdl):
    return float(wdl[0]) - float(wdl[2])


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


def decode_move_classes(classes):
    arr = np.asarray(classes, dtype=np.int64)
    promo = np.full_like(arr, 4)
    ordinary = arr < 4096
    fr = np.empty_like(arr)
    to = np.empty_like(arr)
    fr[ordinary] = arr[ordinary] // 64
    to[ordinary] = arr[ordinary] % 64
    k = (arr[~ordinary] - 4096) // 4
    fr[~ordinary] = k // 64
    to[~ordinary] = k % 64
    promo[~ordinary] = (arr[~ordinary] - 4096) % 4
    return fr, to, promo


def load_value_rows(paths, history, max_rows=0):
    rows = []
    for p in paths:
        with opener(p) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if "fen" not in r:
                    continue
                wdl = r.get("wdl") or [0.0, 1.0, 0.0]
                q = float(r.get("q", q_from_wdl(wdl)))
                pol = r.get("policy") or {}
                best = r.get("best") or (
                    max(pol.items(), key=lambda kv: kv[1])[0] if pol else None
                )
                try:
                    y = move_class(best) if best else -1
                    x = token_features(r["fen"], r.get("history_fens") or [], history)
                except Exception:
                    continue
                rows.append((x, y, [float(wdl[0]), float(wdl[1]), float(wdl[2])], q))
                if max_rows and len(rows) >= max_rows:
                    return rows
    return rows


def load_av_groups(paths, history, max_positions=0, max_candidates=8):
    tmp = {}
    for p in paths:
        with opener(p) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if (
                    r.get("schema") != "teacher.action_value.v1"
                    or not r.get("fen")
                    or not r.get("move")
                ):
                    continue
                key = r.get("position_key") or r["fen"]
                if key not in tmp:
                    if max_positions and len(tmp) >= max_positions:
                        continue
                    try:
                        tmp[key] = {
                            "fen": r["fen"],
                            "history_fens": r.get("history_fens") or [],
                            "cands": [],
                        }
                    except Exception:
                        continue
                try:
                    mv = move_class(r["move"])
                    val = float(r.get("value", 0.0))
                    regret = float(r.get("regret_cp", 0.0) or 0.0) / 400.0
                except Exception:
                    continue
                tmp[key]["cands"].append((mv, val, regret))
    groups = []
    for g in tmp.values():
        # Prefer complete top candidates, sorted by rank/value from source order/value.
        cands = g["cands"][:max_candidates]
        if len(cands) < 2:
            continue
        try:
            x = token_features(g["fen"], g.get("history_fens") or [], history)
        except Exception:
            continue
        groups.append((x, cands))
    return groups


def open_cache_dir(d, rows):
    d = Path(d)
    meta = json.loads((d / "meta.json").read_text())
    F = int(meta["token_features"])
    return (
        np.memmap(d / "tokens.uint8", np.uint8, "r", shape=(rows, 64, F)),
        np.memmap(d / "policy.int64", np.int64, "r", shape=(rows,)),
        np.memmap(d / "wdl.float32", np.float32, "r", shape=(rows, 3)),
    )


def open_position_eval_cache_dir(d):
    d = Path(d)
    meta = json.loads((d / "meta.json").read_text())
    if meta.get("format") != "compact_position_eval_cache_v1":
        raise ValueError(
            f"unsupported position-eval cache format: {meta.get('format')}"
        )
    rows = int(meta["rows"])
    F = int(meta["token_features"])
    return {
        "dir": d,
        "meta": meta,
        "rows": rows,
        "tokens": np.memmap(d / "tokens.uint8", np.uint8, "r", shape=(rows, 64, F)),
        "policy": np.memmap(d / "policy.int64", np.int64, "r", shape=(rows,)),
        "wdl": np.memmap(d / "wdl.float32", np.float32, "r", shape=(rows, 3)),
        "q": np.memmap(d / "q.float32", np.float32, "r", shape=(rows,)),
        "weight": np.memmap(
            d / "quality_weight.float32", np.float32, "r", shape=(rows,)
        ),
    }


def open_av_cache_dir(d):
    d = Path(d)
    meta = json.loads((d / "meta.json").read_text())
    if meta.get("format") != "compact_action_value_cache_v1":
        raise ValueError(f"unsupported AV cache format: {meta.get('format')}")
    rows = int(meta["rows"])
    C = int(meta["max_candidates"])
    F = int(meta["token_features"])
    return {
        "dir": d,
        "meta": meta,
        "rows": rows,
        "max_candidates": C,
        "tokens": np.memmap(d / "tokens.uint8", np.uint8, "r", shape=(rows, 64, F)),
        "moves": np.memmap(d / "candidate_moves.int64", np.int64, "r", shape=(rows, C)),
        "values": np.memmap(
            d / "candidate_values.float32", np.float32, "r", shape=(rows, C)
        ),
        "regrets": np.memmap(
            d / "candidate_regrets.float32", np.float32, "r", shape=(rows, C)
        ),
        "mask": np.memmap(
            d / "candidate_mask.float32", np.float32, "r", shape=(rows, C)
        ),
    }


def expand_compact(tok, history):
    B = tok.shape[0]
    arr = np.zeros((B, 64, (history + 1) * len(PIECES) + 8), dtype=np.float32)
    eye = np.eye(len(PIECES), dtype=np.float32)
    for h in range(history + 1):
        ids = tok[:, :, h].astype(np.int64).clip(0, len(PIECES) - 1)
        arr[:, :, h * len(PIECES) : (h + 1) * len(PIECES)] = eye[ids]
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


def main():
    ap = argparse.ArgumentParser(
        description="Train SquareFormer V2 with clean policy, value, and candidate action-value streams."
    )
    ap.add_argument("--stream-config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--onnx-out", default="")
    ap.add_argument("--meta-out", default="")
    ap.add_argument(
        "--resume",
        default="",
        help="optional checkpoint/model to initialize from; loads model weights only",
    )
    ap.add_argument(
        "--value-cache",
        nargs="*",
        default=[],
        help="optional compact_position_eval_cache_v1 dir(s); overrides JSONL value overlay loading",
    )
    ap.add_argument(
        "--av-cache",
        nargs="*",
        default=[],
        help="optional compact_action_value_cache_v1 dir(s); overrides JSONL AV overlay loading",
    )
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--d-ff", type=int, default=256)
    ap.add_argument(
        "--input-mode",
        choices=["onehot", "embedding"],
        default="onehot",
        help="onehot expands compact tokens on CPU; embedding keeps uint8 tokens compact and embeds on device",
    )
    ap.add_argument("--history-plies", type=int, default=2)
    ap.add_argument("--relation-bias", action="store_true")
    ap.add_argument(
        "--max-rows",
        type=int,
        default=1000000,
        help="legacy mixed-stream step budget, sampled by stream ratios",
    )
    ap.add_argument(
        "--policy-rows",
        type=int,
        default=0,
        help="separate-budget mode: clean supervised samples per epoch",
    )
    ap.add_argument(
        "--value-rows",
        type=int,
        default=0,
        help="separate-budget mode: position-eval samples per epoch",
    )
    ap.add_argument(
        "--av-positions",
        type=int,
        default=0,
        help="separate-budget mode: candidate action-value positions per epoch",
    )
    ap.add_argument("--max-value-rows", type=int, default=200000)
    ap.add_argument("--max-av-positions", type=int, default=100000)
    ap.add_argument("--max-dev-rows", type=int, default=20000)
    ap.add_argument("--max-av-dev-positions", type=int, default=5000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--av-batch-size", type=int, default=128)
    ap.add_argument("--max-candidates", type=int, default=8)
    ap.add_argument(
        "--grad-accum-steps",
        type=int,
        default=1,
        help="accumulate gradients over this many stream batches before optimizer step",
    )
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--amp-dtype", choices=["fp16", "bf16"], default="bf16")
    ap.add_argument("--progress-every", type=int, default=200)
    ap.add_argument("--checkpoint-dir", default="")
    ap.add_argument(
        "--checkpoint-every-steps",
        type=int,
        default=0,
        help="also write checkpoint_latest.pt every N optimizer steps",
    )
    ap.add_argument(
        "--eval-every-steps",
        type=int,
        default=0,
        help="run dev eval and best-checkpoint selection every N optimizer steps",
    )
    ap.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help="stop after this many evals without improvement",
    )
    ap.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=1e-4,
        help="minimum score improvement for early stopping",
    )
    ap.add_argument(
        "--early-stop-metric",
        choices=["composite", "dev_policy_ce", "dev_wdl_ce", "dev_av_mse"],
        default="composite",
    )
    ap.add_argument(
        "--metrics-jsonl-out",
        default="",
        help="optional structured metrics JSONL output",
    )
    args = ap.parse_args()
    install_metric_print_tee(args.metrics_jsonl_out, run_id=Path(args.out).stem)

    import torch, torch.nn as nn, torch.nn.functional as F

    torch.set_float32_matmul_precision("high")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    np_rng = np.random.default_rng(args.seed)
    cfg = json.loads(Path(args.stream_config).read_text())
    streams = {s["name"]: s for s in cfg["streams"]}
    policy_stream = next(s for s in cfg["streams"] if s["kind"] == "supervised_cache")
    cm = json.loads(Path(policy_stream["cache_manifest"]).read_text())
    history = int(cm.get("history_plies", args.history_plies))
    cache_dirs = [Path(p) for p in cm["shards"]]
    cache_meta = [json.loads((p / "meta.json").read_text()) for p in cache_dirs]
    cache_train = [
        open_cache_dir(d, int(m["rows"])) for d, m in zip(cache_dirs, cache_meta)
    ]
    cache_rows = sum(len(pol) for _, pol, _ in cache_train)
    cache_dev = open_cache_dir(
        Path(cm["dev_cache"]),
        int(json.loads((Path(cm["dev_cache"]) / "meta.json").read_text())["rows"]),
    )
    value_paths = []
    av_paths = []
    value_cache_paths = list(args.value_cache or [])
    av_cache_paths = list(args.av_cache or [])
    for s in cfg["streams"]:
        if s["kind"] == "position_eval_overlay":
            value_paths += s.get("shards", [])
            if not args.value_cache:
                for k in ("cache", "cache_dir", "value_cache"):
                    if s.get(k):
                        value_cache_paths.append(s[k])
                value_cache_paths += list(
                    s.get("caches")
                    or s.get("cache_dirs")
                    or s.get("value_caches")
                    or []
                )
        if s["kind"] == "action_value_overlay":
            av_paths += s.get("shards", [])
            if not args.av_cache:
                for k in ("cache", "cache_dir", "av_cache"):
                    if s.get(k):
                        av_cache_paths.append(s[k])
                av_cache_paths += list(
                    s.get("caches") or s.get("cache_dirs") or s.get("av_caches") or []
                )
    # Allow passing a collection manifest produced by build_chessbench_av_caches_parallel.py.
    expanded_cache_paths = []
    for p in av_cache_paths:
        pp = Path(p)
        if pp.is_file():
            try:
                m = json.loads(pp.read_text())
                if m.get("format") == "chessbench_av_cache_collection_v1":
                    expanded_cache_paths += list(m.get("caches") or [])
                    continue
            except Exception:
                pass
        expanded_cache_paths.append(str(p))
    av_cache_paths = expanded_cache_paths
    value_caches = []
    value_rows = None
    if value_cache_paths:
        for p in value_cache_paths:
            print(f"[v2] opening value cache {p}", flush=True)
            value_caches.append(open_position_eval_cache_dir(p))
        print(
            f"[v2] value_cache_count={len(value_caches)} value_cache_rows={sum(c['rows'] for c in value_caches)}",
            flush=True,
        )
    else:
        print(f"[v2] loading value rows max={args.max_value_rows}", flush=True)
        value_rows = load_value_rows(value_paths, history, args.max_value_rows)
        print(f"[v2] value_rows={len(value_rows)}", flush=True)
    av_caches = []
    av_groups = None
    if av_cache_paths:
        for p in av_cache_paths:
            print(f"[v2] opening av cache {p}", flush=True)
            av_caches.append(open_av_cache_dir(p))
        print(
            f"[v2] av_cache_count={len(av_caches)} av_cache_rows={sum(c['rows'] for c in av_caches)} max_candidates={min(c['max_candidates'] for c in av_caches)}",
            flush=True,
        )
    else:
        print(
            f"[v2] loading av groups max_positions={args.max_av_positions}", flush=True
        )
        av_groups = load_av_groups(
            av_paths, history, args.max_av_positions, args.max_candidates
        )
        print(f"[v2] av_groups={len(av_groups)}", flush=True)
    if (not value_caches and not value_rows) or (not av_caches and not av_groups):
        raise SystemExit("missing value or AV rows")

    compact_token_features = history + 9
    input_dim = (history + 1) * len(PIECES) + 8

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.n1 = nn.LayerNorm(args.d_model)
            self.att = nn.MultiheadAttention(args.d_model, args.heads, batch_first=True)
            self.n2 = nn.LayerNorm(args.d_model)
            self.ff = nn.Sequential(
                nn.Linear(args.d_model, args.d_ff),
                nn.GELU(),
                nn.Linear(args.d_ff, args.d_model),
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

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_mode = args.input_mode
            self.inp = nn.Linear(input_dim, args.d_model)
            self.piece_emb = nn.Embedding((history + 1) * len(PIECES), args.d_model)
            self.stm_emb = nn.Embedding(3, args.d_model)
            self.flag_linear = nn.Linear(6, args.d_model)
            self.rank_emb = nn.Embedding(8, args.d_model)
            self.file_emb = nn.Embedding(8, args.d_model)
            self.color_emb = nn.Embedding(2, args.d_model)
            self.square_emb = nn.Embedding(64, args.d_model)
            self.pos = nn.Parameter(torch.zeros(64, args.d_model))
            self.layers = nn.ModuleList([Layer() for _ in range(args.layers)])
            self.fq = nn.Linear(args.d_model, args.d_model)
            self.tk = nn.Linear(args.d_model, args.d_model)
            self.prom = nn.Linear(args.d_model, 64 * 4)
            self.wdl = nn.Linear(args.d_model, 3)
            self.q = nn.Linear(args.d_model, 1)
            self.promo_emb = nn.Embedding(5, args.d_model)
            self.av = nn.Sequential(
                nn.Linear(args.d_model * 4, args.d_model),
                nn.GELU(),
                nn.Linear(args.d_model, 1),
            )
            self.register_buffer(
                "rel_mask",
                torch.tensor(relation_ids(args.heads))
                if args.relation_bias
                else torch.zeros(args.heads, 64, 64),
                persistent=False,
            )
            self.register_buffer(
                "hist_offsets",
                torch.arange(history + 1, dtype=torch.long) * len(PIECES),
                persistent=False,
            )

        def token_embed(self, x):
            x = x.long()
            piece = x[:, :, : history + 1].clamp(
                0, len(PIECES) - 1
            ) + self.hist_offsets.view(1, 1, -1)
            h = self.piece_emb(piece).sum(2)
            stm = x[:, :, history + 1].clamp(0, 2)
            flags = x[:, :, history + 2].long()
            ep = x[:, :, history + 3].float()
            half = x[:, :, history + 4].float() / 100.0
            fb = torch.stack(
                [((flags >> i) & 1).float() for i in range(4)] + [ep, half], -1
            )
            rank = x[:, :, history + 5].clamp(0, 7)
            file = x[:, :, history + 6].clamp(0, 7)
            color = x[:, :, history + 7].clamp(0, 1)
            sq = x[:, :, history + 8].clamp(0, 63)
            return (
                h
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
            mask = (
                self.rel_mask.repeat(x.shape[0], 1, 1) if args.relation_bias else None
            )
            for l in self.layers:
                h = l(h, mask)
            return h

        def forward(self, x):
            h = self.encode(x)
            fq = self.fq(h)
            tk = self.tk(h)
            ordinary = torch.matmul(fq, tk.transpose(1, 2)) / math.sqrt(args.d_model)
            promo = self.prom(h).view(x.shape[0], 64, 64, 4)
            pol = torch.cat(
                [
                    ordinary.reshape(x.shape[0], 4096),
                    promo.reshape(x.shape[0], 4096 * 4),
                ],
                1,
            )
            pooled = h.mean(1)
            return pol, self.wdl(pooled), self.q(pooled).squeeze(1), h

        def av_scores(self, h, moves):
            B, C = moves.shape
            m = moves.detach().cpu().numpy()
            fr, to, pr = decode_move_classes(m.reshape(-1))
            fr = torch.as_tensor(fr.reshape(B, C), device=h.device, dtype=torch.long)
            to = torch.as_tensor(to.reshape(B, C), device=h.device, dtype=torch.long)
            pr = torch.as_tensor(pr.reshape(B, C), device=h.device, dtype=torch.long)
            hb = torch.arange(B, device=h.device)[:, None]
            hf = h[hb, fr]
            ht = h[hb, to]
            pooled = h.mean(1)[:, None, :].expand(-1, C, -1)
            pe = self.promo_emb(pr.clamp(0, 4))
            return self.av(torch.cat([pooled, hf, ht, pe], -1)).squeeze(-1)

    net = Net().to(args.device)
    if args.resume:
        ck = torch.load(args.resume, map_location=args.device)
        state = ck.get("model", ck) if isinstance(ck, dict) else ck
        missing, unexpected = net.load_state_dict(state, strict=False)
        print(
            f"[v2] resumed={args.resume} missing={len(missing)} unexpected={len(unexpected)}",
            flush=True,
        )
    opt = torch.optim.AdamW(
        net.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    def save_checkpoint(path, epoch, step, metrics=None):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": net.state_dict(),
                "optimizer": opt.state_dict(),
                "epoch": epoch,
                "step": step,
                "metrics": metrics or {},
                "config": cfg,
                "layers": args.layers,
                "d_model": args.d_model,
                "heads": args.heads,
                "d_ff": args.d_ff,
                "history_plies": history,
                "relation_bias": bool(args.relation_bias),
                "input_dim": input_dim,
                "token_features": compact_token_features,
                "input_mode": args.input_mode,
                "policy_size": POLICY_SIZE,
                "kind": "squareformer_v2",
            },
            path,
        )

    amp_enabled = bool(args.amp and args.device.startswith("cuda"))
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    scaler = torch.amp.GradScaler(
        "cuda", enabled=bool(amp_enabled and amp_dtype is torch.float16)
    )

    def to_dev(a, dtype=torch.float32):
        return torch.as_tensor(a, dtype=dtype, device=args.device)

    def tokens_to_dev(tok):
        arr = np.asarray(tok)
        if args.input_mode == "embedding":
            return to_dev(arr, torch.long)
        return to_dev(expand_compact(arr, history))

    def random_ids(n, bs):
        return np_rng.integers(0, n, size=bs, endpoint=False)

    def policy_batch(rng):
        si = int(np_rng.integers(0, len(cache_train)))
        tok, pol, wdl = cache_train[si]
        n = len(pol)
        ids = random_ids(n, args.batch_size)
        return (
            tokens_to_dev(tok[ids]),
            to_dev(np.asarray(pol[ids]), torch.long),
            to_dev(np.asarray(wdl[ids])),
        )

    def value_batch(rng):
        if value_caches:
            c = value_caches[int(np_rng.integers(0, len(value_caches)))]
            n = (
                min(c["rows"], args.max_value_rows)
                if args.max_value_rows
                else c["rows"]
            )
            ids = random_ids(n, args.batch_size)
            return (
                tokens_to_dev(c["tokens"][ids]),
                to_dev(np.asarray(c["policy"][ids]), torch.long),
                to_dev(np.asarray(c["wdl"][ids])),
                to_dev(np.asarray(c["q"][ids])),
            )
        sub = [
            value_rows[int(np_rng.integers(0, len(value_rows)))]
            for _ in range(args.batch_size)
        ]
        return (
            to_dev(np.asarray([s[0] for s in sub])),
            to_dev(np.asarray([s[1] for s in sub]), torch.long),
            to_dev(np.asarray([s[2] for s in sub])),
            to_dev(np.asarray([s[3] for s in sub])),
        )

    def av_batch(rng, dev=False):
        bs = args.av_batch_size
        if av_caches:
            c = av_caches[int(np_rng.integers(0, len(av_caches)))]
            n = (
                min(c["rows"], args.max_av_dev_positions)
                if dev
                else min(c["rows"], args.max_av_positions or c["rows"])
            )
            ids = random_ids(n, bs)
            C = min(args.max_candidates, c["max_candidates"])
            return (
                tokens_to_dev(c["tokens"][ids]),
                to_dev(np.asarray(c["moves"][ids, :C]), torch.long),
                to_dev(np.asarray(c["values"][ids, :C])),
                to_dev(np.asarray(c["regrets"][ids, :C])),
                to_dev(np.asarray(c["mask"][ids, :C])),
            )
        groups = av_groups[: args.max_av_dev_positions] if dev else av_groups
        sub = [groups[rng.randrange(len(groups))] for _ in range(bs)]
        C = args.max_candidates
        xs = []
        moves = np.zeros((bs, C), np.int64)
        vals = np.zeros((bs, C), np.float32)
        regrets = np.zeros((bs, C), np.float32)
        mask = np.zeros((bs, C), np.float32)
        for i, (x, cands) in enumerate(sub):
            xs.append(x)
            for j, (mv, val, reg) in enumerate(cands[:C]):
                moves[i, j] = mv
                vals[i, j] = val
                regrets[i, j] = reg
                mask[i, j] = 1.0
        return (
            to_dev(np.asarray(xs)),
            to_dev(moves, torch.long),
            to_dev(vals),
            to_dev(regrets),
            to_dev(mask),
        )

    ratios = []
    names = []
    for s in cfg["streams"]:
        names.append(s["kind"])
        ratios.append(float(s.get("ratio", 1.0)))
    separate_budgets = bool(args.policy_rows or args.value_rows or args.av_positions)
    rng = random.Random(args.seed)

    def choose_stream():
        r = rng.random() * sum(ratios)
        acc = 0.0
        for kind, ratio in zip(names, ratios):
            acc += ratio
            if r <= acc:
                return kind
        return names[-1]

    def epoch_schedule():
        if not separate_budgets:
            return [
                choose_stream()
                for _ in range(math.ceil(args.max_rows / args.batch_size))
            ]
        sched = []
        if args.policy_rows:
            sched += ["supervised_cache"] * math.ceil(
                args.policy_rows / args.batch_size
            )
        if args.value_rows:
            sched += ["position_eval_overlay"] * math.ceil(
                args.value_rows / args.batch_size
            )
        if args.av_positions:
            sched += ["action_value_overlay"] * math.ceil(
                args.av_positions / args.av_batch_size
            )
        rng.shuffle(sched)
        return sched or [choose_stream()]

    def evaluate(label):
        net.eval()
        ce = wc = t1 = t4 = t8 = n = 0
        av_ok = av_n = 0
        av_mse = 0.0
        with torch.no_grad():
            tok, pol, wdl = cache_dev
            dn = min(args.max_dev_rows, len(pol))
            for off in range(0, dn, args.batch_size):
                ids = range(off, min(dn, off + args.batch_size))
                xb = tokens_to_dev(tok[list(ids)])
                yb = to_dev(np.asarray(pol[list(ids)]), torch.long)
                wb = to_dev(np.asarray(wdl[list(ids)]))
                pl, wl, _, _ = net(xb)
                bs = len(yb)
                ce += float(F.cross_entropy(pl.float(), yb, reduction="sum"))
                wc += float(F.cross_entropy(wl.float(), wb.argmax(1), reduction="sum"))
                pred = pl.topk(8, 1).indices
                t1 += int((pred[:, :1] == yb[:, None]).any(1).sum())
                t4 += int((pred[:, :4] == yb[:, None]).any(1).sum())
                t8 += int((pred == yb[:, None]).any(1).sum())
                n += bs
            erng = random.Random(args.seed + 999)
            eval_batches = max(
                1, min(50, args.max_av_dev_positions // max(1, args.av_batch_size))
            )
            for _ in range(eval_batches):
                xb, mv, val, _, mask = av_batch(erng, dev=True)
                _, _, _, h = net(xb)
                sc = torch.tanh(net.av_scores(h, mv).float())
                valid = mask > 0
                av_mse += float(F.mse_loss(sc[valid], val[valid], reduction="sum"))
                av_n += int(valid.sum())
                av_ok += int(
                    (
                        sc.masked_fill(~valid, -1e9).argmax(1)
                        == val.masked_fill(~valid, -1e9).argmax(1)
                    ).sum()
                )
        metrics = {
            "dev_policy_ce": ce / max(1, n),
            "dev_wdl_ce": wc / max(1, n),
            "dev_policy_top1": t1 / max(1, n),
            "dev_policy_top4": t4 / max(1, n),
            "dev_policy_top8": t8 / max(1, n),
            "dev_av_mse": av_mse / max(1, av_n),
            "dev_av_top1": av_ok / max(1, args.av_batch_size * eval_batches),
        }
        metrics["composite"] = (
            metrics["dev_policy_ce"]
            + 0.25 * metrics["dev_wdl_ce"]
            + 2.0 * metrics["dev_av_mse"]
        )
        print(f"eval label={label}", flush=True)
        for k, v in metrics.items():
            print(f"METRIC {label}_{k}={v:.6f}", flush=True)
        net.train()
        return metrics

    if separate_budgets:
        print(
            f"[v2] separate_budgets policy_rows={args.policy_rows} value_rows={args.value_rows} av_positions={args.av_positions} policy_batch={args.batch_size} av_batch={args.av_batch_size}",
            flush=True,
        )
    else:
        print(
            f"[v2] mixed_ratio_budget max_rows={args.max_rows} batch_size={args.batch_size}",
            flush=True,
        )
    best_score = float("inf")
    stale_evals = 0
    stop_training = False

    def handle_eval(label, ep, step):
        nonlocal best_score, stale_evals
        metrics = evaluate(label)
        score = float(metrics[args.early_stop_metric])
        improved = score < (best_score - args.early_stop_min_delta)
        print(
            f"overfit_watch label={label} metric={args.early_stop_metric} score={score:.6f} best={best_score:.6f} improved={int(improved)} stale_evals={stale_evals}",
            flush=True,
        )
        if improved:
            best_score = score
            stale_evals = 0
            if args.checkpoint_dir:
                save_checkpoint(
                    Path(args.checkpoint_dir) / "best.pt", ep, step, metrics
                )
                print(
                    f"checkpoint_best epoch={ep} step={step} score={score:.6f} path={Path(args.checkpoint_dir) / 'best.pt'}",
                    flush=True,
                )
        else:
            stale_evals += 1
        if args.early_stop_patience and stale_evals >= args.early_stop_patience:
            print(
                f"early_stop reason=no_dev_improvement metric={args.early_stop_metric} stale_evals={stale_evals} best={best_score:.6f}",
                flush=True,
            )
            return True, metrics
        return False, metrics

    accum_steps = max(1, args.grad_accum_steps)
    for ep in range(1, args.epochs + 1):
        net.train()
        st = time.time()
        sums = defaultdict(float)
        counts = defaultdict(int)
        sched = epoch_schedule()
        opt.zero_grad(set_to_none=True)
        for step, kind in enumerate(sched, 1):
            with torch.amp.autocast("cuda", enabled=amp_enabled, dtype=amp_dtype):
                if kind == "supervised_cache":
                    xb, yb, wb = policy_batch(rng)
                    pl, wl, ql, _ = net(xb)
                    qtar = wb[:, 0] - wb[:, 2]
                    loss = (
                        F.cross_entropy(pl, yb)
                        + F.cross_entropy(wl, wb.argmax(1))
                        + 0.25 * F.mse_loss(torch.tanh(ql.float()), qtar.float())
                    )
                    counts["policy_rows"] += len(yb)
                elif kind == "position_eval_overlay":
                    xb, yb, wb, qb = value_batch(rng)
                    pl, wl, ql, _ = net(xb)
                    loss = F.cross_entropy(wl, wb.argmax(1)) + F.mse_loss(
                        torch.tanh(ql.float()), qb.float()
                    )
                    valid = yb >= 0
                    if valid.any():
                        loss = loss + 0.10 * F.cross_entropy(pl[valid], yb[valid])
                    counts["value_rows"] += len(qb)
                else:
                    xb, mv, val, reg, mask = av_batch(rng)
                    _, _, _, h = net(xb)
                    sc = net.av_scores(h, mv)
                    valid = mask > 0
                    av_loss = F.smooth_l1_loss(
                        torch.tanh(sc.float())[valid], val.float()[valid]
                    )
                    masked = sc.float().masked_fill(~valid, -1e9)
                    target = val.float().masked_fill(~valid, -1e9).argmax(1)
                    rank_loss = F.cross_entropy(masked, target)
                    pred_reg = (
                        sc.float().max(1, keepdim=True).values - sc.float()
                    ).masked_select(valid)
                    reg_loss = F.smooth_l1_loss(
                        pred_reg, reg.float().masked_select(valid)
                    )
                    loss = av_loss + 0.5 * rank_loss + 0.25 * reg_loss
                    counts["av_positions"] += xb.shape[0]
            raw_loss = loss.detach()
            scaler.scale(loss / accum_steps).backward()
            if step % accum_steps == 0 or step == len(sched):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            sums[kind + "_loss"] += float(raw_loss)
            counts[kind + "_batches"] += 1
            if args.progress_every and step % args.progress_every == 0:
                dt = time.time() - st
                msg = " ".join(
                    f"{k}={v / max(1, counts[k.replace('_loss', '_batches')]):.4f}"
                    for k, v in sums.items()
                )
                print(
                    f"progress epoch={ep} step={step}/{len(sched)} seconds={dt:.1f} {msg}",
                    flush=True,
                )
            if (
                args.checkpoint_dir
                and args.checkpoint_every_steps
                and step % args.checkpoint_every_steps == 0
            ):
                save_checkpoint(
                    Path(args.checkpoint_dir) / "checkpoint_latest.pt", ep, step
                )
                print(
                    f"checkpoint epoch={ep} step={step} path={Path(args.checkpoint_dir) / 'checkpoint_latest.pt'}",
                    flush=True,
                )
            if args.eval_every_steps and step % args.eval_every_steps == 0:
                should_stop, metrics = handle_eval(f"ep{ep}_step{step}", ep, step)
                if should_stop:
                    stop_training = True
                    break
        if stop_training:
            break
        should_stop, metrics = handle_eval(f"epoch{ep}", ep, step)
        print(f"epoch {ep} seconds={time.time() - st:.1f}", flush=True)
        for k, v in metrics.items():
            print(f"METRIC {k}={v:.6f}", flush=True)
        if args.checkpoint_dir:
            cdir = Path(args.checkpoint_dir)
            cdir.mkdir(parents=True, exist_ok=True)
            save_checkpoint(cdir / f"epoch_{ep}.pt", ep, step, metrics)
        if should_stop:
            break
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": net.state_dict(),
            "config": cfg,
            "layers": args.layers,
            "d_model": args.d_model,
            "heads": args.heads,
            "d_ff": args.d_ff,
            "history_plies": history,
            "relation_bias": bool(args.relation_bias),
            "input_dim": input_dim,
            "token_features": compact_token_features,
            "input_mode": args.input_mode,
            "policy_size": POLICY_SIZE,
            "kind": "squareformer_v2",
        },
        args.out,
    )
    if args.onnx_out:
        net.eval()
        dummy = (
            torch.zeros(
                1, 64, compact_token_features, device=args.device, dtype=torch.long
            )
            if args.input_mode == "embedding"
            else torch.zeros(1, 64, input_dim, device=args.device)
        )
        torch.onnx.export(
            net,
            dummy,
            args.onnx_out,
            input_names=["tokens"],
            output_names=["policy", "wdl", "q", "hidden"],
            dynamic_axes={
                "tokens": {0: "batch"},
                "policy": {0: "batch"},
                "wdl": {0: "batch"},
                "q": {0: "batch"},
                "hidden": {0: "batch"},
            },
            opset_version=17,
        )
    if args.meta_out:
        Path(args.meta_out).write_text(
            json.dumps(
                {
                    "kind": "squareformer_v2",
                    "input_dim": input_dim,
                    "token_features": compact_token_features,
                    "input_mode": args.input_mode,
                    "input_format": "compact_uint8_tokens"
                    if args.input_mode == "embedding"
                    else "float_onehot_rules",
                    "policy_size": POLICY_SIZE,
                    "layers": args.layers,
                    "d_model": args.d_model,
                    "heads": args.heads,
                    "d_ff": args.d_ff,
                    "history_plies": history,
                    "relation_bias": bool(args.relation_bias),
                    "from_to_policy_size": POLICY_SIZE,
                    "outputs": ["policy", "wdl", "q", "hidden"],
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
