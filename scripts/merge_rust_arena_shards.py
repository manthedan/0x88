#!/usr/bin/env python3
"""Merge tiny-leela-rust-arena N-player shard JSON files."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def score_result(score: float):
    if abs(score - 1.0) < 1e-6:
        return "win"
    if abs(score) < 1e-6:
        return "loss"
    return "draw"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("shards", nargs="+", help="Shard JSON files")
    ap.add_argument("--out", required=True, help="Merged output JSON")
    args = ap.parse_args()

    docs = []
    for path in args.shards:
        p = Path(path)
        if p.exists() and p.stat().st_size > 0:
            docs.append(json.loads(p.read_text()))
    if not docs:
        raise SystemExit("no readable shard JSON files")

    first_protocol = docs[0].get("protocol", {})
    players = list(first_protocol.get("players", []))
    standings = {
        name: {"name": name, "wins": 0, "draws": 0, "losses": 0, "score": 0.0, "games": 0}
        for name in players
    }
    pair_acc = defaultdict(lambda: {"aScore": 0.0, "games": 0, "aWdl": [0, 0, 0]})
    games_by_id = {}
    illegal_losses = 0

    for doc in docs:
        illegal_losses += int(doc.get("illegalLosses", 0))
        for game in doc.get("games", []):
            gid = int(game["game"])
            if gid in games_by_id:
                raise SystemExit(f"duplicate game id across shards: {gid}")
            games_by_id[gid] = game
            a = game["a"]
            b = game["b"]
            a_score = float(game["aScore"])
            b_score = 1.0 - a_score
            for name, score in ((a, a_score), (b, b_score)):
                if name not in standings:
                    standings[name] = {"name": name, "wins": 0, "draws": 0, "losses": 0, "score": 0.0, "games": 0}
                standings[name]["games"] += 1
                standings[name]["score"] += score
                r = score_result(score)
                if r == "win":
                    standings[name]["wins"] += 1
                elif r == "loss":
                    standings[name]["losses"] += 1
                else:
                    standings[name]["draws"] += 1
            pair = pair_acc[(a, b)]
            pair["aScore"] += a_score
            pair["games"] += 1
            r = score_result(a_score)
            if r == "win":
                pair["aWdl"][0] += 1
            elif r == "loss":
                pair["aWdl"][2] += 1
            else:
                pair["aWdl"][1] += 1

    standing_records = []
    for s in standings.values():
        games = max(1, int(s["games"]))
        row = dict(s)
        row["scoreRate"] = row["score"] / games
        standing_records.append(row)
    standing_records.sort(key=lambda r: (-r["scoreRate"], -r["score"], r["name"]))

    pair_records = []
    for (a, b), p in sorted(pair_acc.items()):
        games = max(1, int(p["games"]))
        pair_records.append(
            {
                "a": a,
                "b": b,
                "aScore": p["aScore"],
                "games": p["games"],
                "aWdl": p["aWdl"],
                "aScoreRate": p["aScore"] / games,
            }
        )

    games = [games_by_id[k] for k in sorted(games_by_id)]
    protocol = dict(first_protocol)
    protocol["backend"] = "rust-native-ort-merged"
    protocol["mergedShards"] = len(docs)
    protocol["selectedGames"] = len(games)
    protocol["totalGames"] = first_protocol.get("totalGames", len(games))

    output = {
        "protocol": protocol,
        "standings": standing_records,
        "pairs": pair_records,
        "games": games,
        "illegalLosses": illegal_losses,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2) + "\n")
    print(f"merged {len(docs)} shards / {len(games)} games -> {out}")


if __name__ == "__main__":
    main()
