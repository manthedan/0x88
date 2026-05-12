#!/usr/bin/env python3
"""Tiny sparse-policy/WDL overfit smoke for LC0 normalized JSONL.

This is not a production trainer. It verifies that converted LC0 examples can be
consumed as sparse soft policy targets and that a tiny model can reduce KL/CE on
a small fixed sample.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import chess
import torch
import torch.nn.functional as F

ACTION_SPACE = 64 * 64 * 5
PROMO_TO_ID = {"": 0, "n": 1, "b": 2, "r": 3, "q": 4}
PIECE_PLANES = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}


def move_to_action(uci: str) -> int:
    move = chess.Move.from_uci(uci)
    promo = ""
    if move.promotion:
        promo = chess.piece_symbol(move.promotion)
    promo_id = PROMO_TO_ID[promo]
    return (move.from_square * 64 + move.to_square) * 5 + promo_id


def encode_board(fen: str) -> list[float]:
    board = chess.Board(fen)
    x = [0.0] * (12 * 64 + 5)
    for sq, piece in board.piece_map().items():
        color_offset = 0 if piece.color == chess.WHITE else 6
        plane = color_offset + PIECE_PLANES[piece.piece_type]
        x[plane * 64 + sq] = 1.0
    x[12 * 64 + 0] = 1.0 if board.turn == chess.WHITE else 0.0
    x[12 * 64 + 1] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    x[12 * 64 + 2] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    x[12 * 64 + 3] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
    x[12 * 64 + 4] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0
    return x


def load_rows(path: Path, limit: int) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
    xs: list[list[float]] = []
    policy_ids: list[torch.Tensor] = []
    policy_probs: list[torch.Tensor] = []
    wdls: list[list[float]] = []
    with path.open() as f:
        for line in f:
            if len(xs) >= limit:
                break
            if not line.strip():
                continue
            rec = json.loads(line)
            xs.append(encode_board(rec["board"]["fen"]))
            ids: list[int] = []
            probs: list[float] = []
            for move, prob in rec["policy_target_uci"].items():
                ids.append(move_to_action(move))
                probs.append(float(prob))
            policy_ids.append(torch.tensor(ids, dtype=torch.long))
            policy_probs.append(torch.tensor(probs, dtype=torch.float32))
            wdl = rec["value_targets"]["wdl_root"]
            wdls.append([float(wdl["win"]), float(wdl["draw"]), float(wdl["loss"])])
    if not xs:
        raise RuntimeError(f"no rows loaded from {path}")
    return torch.tensor(xs, dtype=torch.float32), policy_ids, policy_probs, torch.tensor(wdls, dtype=torch.float32)


class TinyOverfit(torch.nn.Module):
    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(in_dim, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, hidden), torch.nn.ReLU())
        self.policy = torch.nn.Linear(hidden, ACTION_SPACE)
        self.value = torch.nn.Linear(hidden, 3)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        return self.policy(h), self.value(h)


def loss_fn(policy_logits: torch.Tensor, value_logits: torch.Tensor, ids: list[torch.Tensor], probs: list[torch.Tensor], wdl: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logp = F.log_softmax(policy_logits, dim=-1)
    policy_terms = []
    for i, (idx, target) in enumerate(zip(ids, probs)):
        policy_terms.append(-(target.to(logp.device) * logp[i, idx.to(logp.device)]).sum())
    policy_loss = torch.stack(policy_terms).mean()
    value_loss = -(wdl.to(value_logits.device) * F.log_softmax(value_logits, dim=-1)).sum(dim=-1).mean()
    return policy_loss + value_loss, policy_loss, value_loss


def run(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    x, ids, probs, wdl = load_rows(Path(args.input), args.rows)
    model = TinyOverfit(x.shape[1], args.hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    with torch.no_grad():
        logits, v = model(x)
        initial, initial_policy, initial_value = loss_fn(logits, v, ids, probs, wdl)
    history = []
    for step in range(1, args.steps + 1):
        opt.zero_grad(set_to_none=True)
        logits, v = model(x)
        loss, policy_loss, value_loss = loss_fn(logits, v, ids, probs, wdl)
        loss.backward()
        opt.step()
        if step == 1 or step == args.steps or step % max(1, args.steps // 5) == 0:
            history.append({"step": step, "loss": float(loss.detach()), "policy_loss": float(policy_loss.detach()), "value_loss": float(value_loss.detach())})
    with torch.no_grad():
        logits, v = model(x)
        final, final_policy, final_value = loss_fn(logits, v, ids, probs, wdl)
    return {
        "schema": "tiny_leela.lc0_overfit_smoke.v1",
        "input": args.input,
        "rows": int(x.shape[0]),
        "steps": args.steps,
        "hidden": args.hidden,
        "lr": args.lr,
        "initial_loss": float(initial),
        "initial_policy_loss": float(initial_policy),
        "initial_value_loss": float(initial_value),
        "final_loss": float(final),
        "final_policy_loss": float(final_policy),
        "final_value_loss": float(final_value),
        "loss_delta": float(initial - final),
        "ok": bool(final < initial and final_policy < initial_policy),
        "history": history,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--rows", type=int, default=64)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    report = run(args)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(text, end="")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
