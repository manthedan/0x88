#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, re, subprocess, sys, time
from pathlib import Path

ANSI = re.compile(r"\x1b\[[0-9;]*m")
STAT = re.compile(r"^info string\s+(\S+)\s+\(\s*(\d+)\s*\).*?\(P:\s*([0-9.]+)%\).*?\(Q:\s*([-.0-9]+)\)")
NODE = re.compile(r"^info string node\s+\(\s*(\d+)\).*?\(WL:\s*([-.0-9]+)\)\s+\(D:\s*([-.0-9]+)\)\s+\(M:\s*([-.0-9]+)\)\s+\(Q:\s*([-.0-9]+)\)\s+\(V:\s*([-.0-9]+)\)")

parser = argparse.ArgumentParser(description="Collect native LC0 nodes=1 VerboseMoveStats for FEN fixtures.")
parser.add_argument("--lc0", default="../native/lc0-release-0.32/build/release/lc0")
parser.add_argument("--weights", default="../models/lc0-bestnets/t1-256x10-distilled-swa-2432500.pb.gz")
parser.add_argument("--fixtures", default="fixtures/lc0/fen_only.json")
# Use a CPU backend by default for evaluator parity. LC0 Metal/MPS currently
# differs from BLAS/Eigen/ONNX on attention promotion logits in at least one
# fixture, while BLAS/Eigen match the ONNX converter path.
parser.add_argument("--backend", default="blas")
parser.add_argument("--nodes", type=int, default=1)
parser.add_argument("--out", default="fixtures/lc0/native_fen_only_blas.jsonl")
args = parser.parse_args()

fixtures = json.loads(Path(args.fixtures).read_text())
p = subprocess.Popen([args.lc0], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

def clean(line: str) -> str:
    return ANSI.sub("", line.rstrip("\n"))

def send(command: str) -> None:
    assert p.stdin
    p.stdin.write(command + "\n")
    p.stdin.flush()

def read_until(prefix: str, timeout: float) -> list[str]:
    assert p.stdout
    end = time.time() + timeout
    lines: list[str] = []
    while time.time() < end:
        line = p.stdout.readline()
        if not line:
            break
        line = clean(line)
        lines.append(line)
        if line.startswith(prefix):
            return lines
    raise TimeoutError(f"timed out waiting for {prefix}")

def parse_result(fixture: dict, lines: list[str], backend: str) -> dict:
    moves = []
    node = None
    bestmove = None
    for line in lines:
        if line.startswith("bestmove "):
            bestmove = line.split()[1]
        m = STAT.search(line)
        if m:
            moves.append({"uci": m.group(1), "index": int(m.group(2)), "prior": float(m.group(3)) / 100.0, "q": float(m.group(4))})
            continue
        n = NODE.search(line)
        if n:
            node = {"visits": int(n.group(1)), "wl": float(n.group(2)), "d": float(n.group(3)), "mlh": float(n.group(4)), "q": float(n.group(5)), "v": float(n.group(6))}
    moves.sort(key=lambda x: x["prior"], reverse=True)
    return {"id": fixture["id"], "backend": backend, "fen": fixture["fen"], "bestmove": bestmove, "node": node, "topPriors": moves[:10]}

try:
    send("uci")
    read_until("uciok", 10)
    send(f"setoption name WeightsFile value {args.weights}")
    send(f"setoption name Backend value {args.backend}")
    send("setoption name VerboseMoveStats value true")
    send("isready")
    read_until("readyok", 90)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for fixture in fixtures:
            send("position fen " + fixture["fen"])
            send(f"go nodes {args.nodes}")
            lines = read_until("bestmove", 120)
            record = parse_result(fixture, lines, args.backend)
            f.write(json.dumps(record) + "\n")
            print(json.dumps(record))
finally:
    try:
        send("quit")
    except Exception:
        pass
    try:
        p.wait(timeout=5)
    except Exception:
        p.kill()
