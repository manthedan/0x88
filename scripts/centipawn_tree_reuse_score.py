#!/usr/bin/env python3
"""Research-only held-out candidate scoring: ENGINE INPUT.json OUTPUT.json.

Run under benchlock and pin the native engine via taskset on shared machines.
Input rows contain fen/fresh/reuse; identical choices have zero delta by identity.
No python-chess dependency and no product integration.
"""
import json
import statistics
import subprocess
import sys


def main():
    engine, input_path, output_path = sys.argv[1:]
    with open(input_path) as source:
        rows = json.load(source)
    process = subprocess.Popen(
        [engine], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1
    )

    def send(command):
        process.stdin.write(command + "\n")
        process.stdin.flush()

    def until(prefix):
        lines = []
        while True:
            line = process.stdout.readline()
            if not line:
                raise RuntimeError(f"Engine exited before {prefix}")
            line = line.strip()
            lines.append(line)
            if line.startswith(prefix):
                return lines

    def score(fen, move):
        send("ucinewgame")
        send("isready")
        until("readyok")
        send("position fen " + fen)
        send("go movetime 500 searchmoves " + move)
        latest = None
        for line in until("bestmove"):
            words = line.split()
            if "score" not in words:
                continue
            index = words.index("score")
            kind, value = words[index + 1], int(words[index + 2])
            if kind == "cp":
                latest = {"cp": value}
            elif kind == "mate":
                latest = {"cp": (1 if value > 0 else -1) * 10000, "mate": value}
            else:
                continue
            for field in ["depth", "nodes"]:
                if field in words:
                    latest[field] = int(words[words.index(field) + 1])
        if latest is None:
            raise RuntimeError(f"No reference score for {fen}: {move}")
        return latest

    output = []
    try:
        send("uci")
        until("uciok")
        send("setoption name Threads value 1")
        send("setoption name Hash value 16")
        send("isready")
        until("readyok")
        for row in rows:
            if row["fresh"] == row["reuse"]:
                scores = [{"cp": None} for _ in range(4)]
                delta = 0
            else:
                scores = [
                    score(row["fen"], move)
                    for move in [row["fresh"], row["reuse"], row["fresh"], row["reuse"]]
                ]
                delta = statistics.median([scores[1]["cp"], scores[3]["cp"]]) - statistics.median(
                    [scores[0]["cp"], scores[2]["cp"]]
                )
            item = {**row, "scoresABAB": scores, "reuseMinusFreshCp": delta}
            output.append(item)
            print(json.dumps(item), flush=True)
    finally:
        if process.poll() is None:
            send("quit")
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()  # Only this script's owned child process.
                process.wait()
    with open(output_path, "w") as destination:
        json.dump(
            {"engine": engine, "movetimeMs": 500, "threads": 1, "hashMb": 16, "rows": output},
            destination,
            indent=2,
        )
        destination.write("\n")


if __name__ == "__main__":
    main()
