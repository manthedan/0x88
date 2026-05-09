#!/usr/bin/env python3
"""Ply-level Stockfish blunder attribution for uci_anchor_arena JSON games.

Designed for offline/targeted use, not automatic overnight execution.  It annotates
candidate-side moves with Stockfish score-before/score-after, cp loss, mate swings,
and simple queen/material-loss flags.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import chess
import chess.engine

PIECE_VALUE = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
}


def material(board: chess.Board, color: chess.Color) -> dict[str, int]:
    total = 0
    queens = 0
    for pt, val in PIECE_VALUE.items():
        n = len(board.pieces(pt, color))
        total += n * val
        if pt == chess.QUEEN:
            queens = n
    return {"material": total, "queens": queens}


def side_for_candidate(game: dict[str, Any], candidate: str) -> chess.Color:
    return chess.WHITE if game.get("white") == candidate else chess.BLACK


def score_cp(info: dict[str, Any], pov: chess.Color) -> dict[str, Any]:
    score = info["score"].pov(pov)
    mate = score.mate()
    cp = score.score(mate_score=100000)
    return {"cp": cp, "mate": mate}


def analyse(engine: chess.engine.SimpleEngine, board: chess.Board, pov: chess.Color, *, depth: int, nodes: int | None, time_limit: float | None) -> dict[str, Any]:
    if nodes is not None:
        limit = chess.engine.Limit(nodes=nodes)
    elif time_limit is not None:
        limit = chess.engine.Limit(time=time_limit)
    else:
        limit = chess.engine.Limit(depth=depth)
    info = engine.analyse(board, limit, multipv=1)
    s = score_cp(info, pov)
    pv = info.get("pv") or []
    return {
        "cp": s["cp"],
        "mate": s["mate"],
        "bestmove": pv[0].uci() if pv else None,
        "depth": info.get("depth"),
        "nodes": info.get("nodes"),
    }


def annotate_file(path: Path, args: argparse.Namespace, engine: chess.engine.SimpleEngine) -> dict[str, Any]:
    data = json.loads(path.read_text())
    candidate = data["candidate"]["name"] if isinstance(data.get("candidate"), dict) else str(data.get("candidate"))
    rows: list[dict[str, Any]] = []
    games_seen = 0
    games_annotated = 0
    by_anchor = Counter()
    severe_by_anchor = Counter()

    for gi, game in enumerate(data.get("games") or []):
        if args.only_losses and game.get("tinyScore") != 0:
            continue
        if args.anchor and game.get("anchor") not in set(args.anchor):
            continue
        games_seen += 1
        if games_annotated >= args.max_games:
            break
        tiny_color = side_for_candidate(game, candidate)
        moves = game.get("moves") or []
        annotated_this_game = 0
        for mi, mv in enumerate(moves):
            if annotated_this_game >= args.max_candidate_moves_per_game:
                break
            side = mv.get("side")
            if side not in ("w", "b"):
                continue
            move_color = chess.WHITE if side == "w" else chess.BLACK
            if move_color != tiny_color:
                continue
            fen = mv.get("fenBefore")
            uci = mv.get("uci")
            if not fen or not uci:
                continue
            try:
                board = chess.Board(fen)
                move = chess.Move.from_uci(uci)
            except Exception as e:
                rows.append({"file": str(path), "game_index": gi, "move_index": mi, "error": str(e), "fenBefore": fen, "uci": uci})
                continue
            if move not in board.legal_moves:
                rows.append({"file": str(path), "game_index": gi, "move_index": mi, "illegal_uci": uci, "fenBefore": fen})
                continue
            before_mat = material(board, tiny_color)
            before_opp = material(board, not tiny_color)
            before = analyse(engine, board, tiny_color, depth=args.depth, nodes=args.nodes, time_limit=args.time)
            board.push(move)
            after_mat = material(board, tiny_color)
            after_opp = material(board, not tiny_color)
            after = analyse(engine, board, tiny_color, depth=args.depth, nodes=args.nodes, time_limit=args.time)
            cp_loss = None
            if before.get("cp") is not None and after.get("cp") is not None:
                cp_loss = before["cp"] - after["cp"]
            queen_lost = before_mat["queens"] > after_mat["queens"]
            material_swing = (before_mat["material"] - before_opp["material"]) - (after_mat["material"] - after_opp["material"])
            severe = bool((cp_loss is not None and cp_loss >= args.severe_cp) or queen_lost or material_swing >= args.severe_material_swing)
            row = {
                "file": str(path),
                "candidate": candidate,
                "anchor": game.get("anchor"),
                "game_index": gi,
                "tiny_score": game.get("tinyScore"),
                "tiny_side": "w" if tiny_color == chess.WHITE else "b",
                "ply": mv.get("ply"),
                "move_index": mi,
                "uci": uci,
                "fenBefore": fen,
                "score_before_cp": before.get("cp"),
                "score_after_cp": after.get("cp"),
                "cp_loss": cp_loss,
                "bestmove_before": before.get("bestmove"),
                "mate_before": before.get("mate"),
                "mate_after": after.get("mate"),
                "queen_lost": queen_lost,
                "material_swing_pawns": material_swing,
                "severe": severe,
            }
            rows.append(row)
            by_anchor[game.get("anchor")] += 1
            if severe:
                severe_by_anchor[game.get("anchor")] += 1
            annotated_this_game += 1
        games_annotated += 1

    top = sorted([r for r in rows if r.get("cp_loss") is not None], key=lambda r: r["cp_loss"], reverse=True)[: args.top]
    return {
        "path": str(path),
        "candidate": candidate,
        "games_seen_after_filters": games_seen,
        "games_annotated": games_annotated,
        "positions_annotated": len(rows),
        "by_anchor_positions": dict(by_anchor),
        "by_anchor_severe": dict(severe_by_anchor),
        "top_blunders": top,
        "rows": rows if args.include_rows else [],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files", nargs="+", help="uci_anchor_arena JSON files")
    ap.add_argument("--engine", default=".local_engines/stockfish_pkg/usr/games/stockfish")
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--nodes", type=int, default=None)
    ap.add_argument("--time", type=float, default=None)
    ap.add_argument("--max-games", type=int, default=8)
    ap.add_argument("--max-candidate-moves-per-game", type=int, default=80)
    ap.add_argument("--only-losses", action="store_true")
    ap.add_argument("--anchor", action="append", default=[])
    ap.add_argument("--severe-cp", type=int, default=250)
    ap.add_argument("--severe-material-swing", type=int, default=5)
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--include-rows", action="store_true")
    ap.add_argument("--out", default="artifacts/analysis/stockfish_ply_blunder_attribution.json")
    args = ap.parse_args()

    engine_path = Path(args.engine)
    if not engine_path.exists():
        raise SystemExit(f"engine missing: {engine_path}")
    with chess.engine.SimpleEngine.popen_uci(str(engine_path)) as engine:
        files = [Path(p) for p in args.files]
        reports = [annotate_file(p, args, engine) for p in files]
    result = {"schema": "tiny_leela.stockfish_ply_blunder_attribution.v1", "args": vars(args), "reports": reports}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"out": str(out), "files": len(files), "positions": sum(r["positions_annotated"] for r in reports)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
