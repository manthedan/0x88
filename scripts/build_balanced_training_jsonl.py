#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, random, re
from collections import Counter, defaultdict
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--input', nargs='+', required=True)
p.add_argument('--out', required=True)
p.add_argument('--max-rows', type=int, default=300000)
p.add_argument('--max-rows-per-game', type=int, default=32)
p.add_argument('--max-rows-per-opening', type=int, default=6000)
p.add_argument('--skip-plies', type=int, default=0)
p.add_argument('--seed', type=int, default=7)
p.add_argument('--dedupe-fen', action='store_true')
args=p.parse_args()

def game_id(row):
  rid=str(row.get('id',''))
  m=re.match(r'(.+)_([0-9]+)$', rid)
  return m.group(1) if m else rid

def ply(row):
  rid=str(row.get('id',''))
  m=re.search(r'_([0-9]+)$', rid)
  return int(m.group(1)) if m else 0

def fen_key(fen):
  parts=fen.split()
  return ' '.join(parts[:4])

def opening_key(rows):
  # Prefer real metadata if future builders add it; current JSONLs do not.
  for k in ('eco','ECO'):
    if rows[0].get(k): return f"eco:{rows[0][k]}"
  for k in ('opening','Opening'):
    if rows[0].get(k): return f"opening:{rows[0][k]}"
  first=min(rows, key=ply)
  # Proxy for repeated book line: first sampled normalized FEN.
  return 'firstfen:'+fen_key(first['fen'])

games=defaultdict(list)
for path in args.input:
  with open(path) as f:
    for line in f:
      row=json.loads(line)
      if ply(row) < args.skip_plies: continue
      pol=row.get('policy',{})
      if len(pol)!=1: continue
      games[game_id(row)].append(row)

rng=random.Random(args.seed)
game_items=list(games.items())
rng.shuffle(game_items)
opening_counts=Counter(); rows_per_game=[]; rows_out=[]; seen_fens=set(); skipped_dupe=0; skipped_opening=0
for gid, rows in game_items:
  rng.shuffle(rows)
  ok=opening_key(rows)
  took=0
  for row in rows:
    if took >= args.max_rows_per_game: break
    if opening_counts[ok] >= args.max_rows_per_opening:
      skipped_opening += 1; continue
    fk=fen_key(row['fen'])
    if args.dedupe_fen and fk in seen_fens:
      skipped_dupe += 1; continue
    seen_fens.add(fk); rows_out.append(row); opening_counts[ok]+=1; took+=1
    if len(rows_out) >= args.max_rows: break
  if took: rows_per_game.append(took)
  if len(rows_out) >= args.max_rows: break

Path(args.out).parent.mkdir(parents=True, exist_ok=True)
with open(args.out,'w') as f:
  for row in rows_out: f.write(json.dumps(row,separators=(',',':'))+'\n')
print(f'METRIC balanced_rows={len(rows_out)}')
print(f'METRIC balanced_games={len(rows_per_game)}')
print(f'METRIC balanced_openings={len(opening_counts)}')
print(f'METRIC balanced_top_opening_rows={max(opening_counts.values()) if opening_counts else 0}')
print(f'METRIC balanced_skipped_duplicate_fens={skipped_dupe}')
print(f'METRIC balanced_skipped_opening_cap={skipped_opening}')
print(f'METRIC balanced_avg_rows_per_game={sum(rows_per_game)/max(1,len(rows_per_game)):.6f}')
