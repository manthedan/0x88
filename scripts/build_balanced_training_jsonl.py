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
p.add_argument('--dev-out', default='', help='Optional held-out output JSONL split by whole games')
p.add_argument('--dev-ratio', type=float, default=0.05, help='Fraction of games for --dev-out')
p.add_argument('--report', default='', help='Optional JSON diagnostics report path')
p.add_argument('--history-plies', type=int, default=0, help='Attach previous same-game FENs as history_fens')
args=p.parse_args()
if args.history_plies < 0: raise SystemExit('--history-plies must be >= 0')
if args.dev_out and not (0.0 < args.dev_ratio < 1.0): raise SystemExit('--dev-ratio must be between 0 and 1')

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

games=defaultdict(list); game_sources={}
for path in args.input:
  source=Path(path).stem
  with open(path) as f:
    for line in f:
      row=json.loads(line)
      if ply(row) < args.skip_plies: continue
      pol=row.get('policy',{})
      if len(pol)!=1: continue
      gid=game_id(row); games[gid].append(row); game_sources.setdefault(gid, source)

def annotate_history(rows):
  ordered=sorted(rows, key=ply); prev=[]; by_id={}
  for row in ordered:
    nr=dict(row)
    if args.history_plies: nr['history_fens']=list(reversed(prev[-args.history_plies:]))
    by_id[id(row)]=nr
    prev.append(row['fen'])
  return by_id

def select_rows(game_items, max_rows, seen_fens=None):
  opening_counts=Counter(); rows_per_game=[]; rows_out=[]; skipped_dupe=0; skipped_opening=0; source_counts=Counter(); game_openings={}; history_available=0
  if seen_fens is None: seen_fens=set()
  for gid, rows in game_items:
    annotated=annotate_history(rows)
    rng.shuffle(rows)
    ok=opening_key(rows); game_openings[gid]=ok
    took=0
    for row in rows:
      if took >= args.max_rows_per_game: break
      if opening_counts[ok] >= args.max_rows_per_opening:
        skipped_opening += 1; continue
      fk=fen_key(row['fen'])
      if args.dedupe_fen and fk in seen_fens:
        skipped_dupe += 1; continue
      out_row=annotated[id(row)]
      if out_row.get('history_fens'): history_available += 1
      seen_fens.add(fk); rows_out.append(out_row); opening_counts[ok]+=1; source_counts[game_sources.get(gid,'unknown')]+=1; took+=1
      if len(rows_out) >= max_rows: break
    if took: rows_per_game.append(took)
    if len(rows_out) >= max_rows: break
  return {'rows': rows_out, 'opening_counts': opening_counts, 'rows_per_game': rows_per_game, 'skipped_dupe': skipped_dupe, 'skipped_opening': skipped_opening, 'source_counts': source_counts, 'game_openings': game_openings, 'history_available': history_available}

rng=random.Random(args.seed)
game_items=list(games.items())
rng.shuffle(game_items)
dev_items=[]; train_items=game_items
if args.dev_out:
  n_dev=max(1, int(round(len(game_items)*args.dev_ratio)))
  dev_items=game_items[:n_dev]; train_items=game_items[n_dev:]
train=select_rows(train_items, args.max_rows)
rows_out=train['rows']; opening_counts=train['opening_counts']; rows_per_game=train['rows_per_game']; skipped_dupe=train['skipped_dupe']; skipped_opening=train['skipped_opening']
dev=None
if args.dev_out:
  dev=select_rows(dev_items, max(1, int(args.max_rows*args.dev_ratio)), set())

Path(args.out).parent.mkdir(parents=True, exist_ok=True)
with open(args.out,'w') as f:
  for row in rows_out: f.write(json.dumps(row,separators=(',',':'))+'\n')
if args.dev_out and dev is not None:
  Path(args.dev_out).parent.mkdir(parents=True, exist_ok=True)
  with open(args.dev_out,'w') as f:
    for row in dev['rows']: f.write(json.dumps(row,separators=(',',':'))+'\n')

def summary(name, data):
  oc=data['opening_counts']; rpg=data['rows_per_game']
  return {'name':name,'rows':len(data['rows']),'games':len(rpg),'openings':len(oc),'top_opening_rows':max(oc.values()) if oc else 0,'skipped_duplicate_fens':data['skipped_dupe'],'skipped_opening_cap':data['skipped_opening'],'avg_rows_per_game':sum(rpg)/max(1,len(rpg)),'history_plies':args.history_plies,'rows_with_history':data['history_available'],'source_counts':dict(data['source_counts'].most_common()),'top_openings':oc.most_common(20)}
report={'seed':args.seed,'input_games':len(games),'train_game_candidates':len(train_items),'dev_game_candidates':len(dev_items),'train':summary('train',train)}
if dev is not None: report['dev']=summary('dev',dev)
if args.report:
  Path(args.report).parent.mkdir(parents=True, exist_ok=True)
  with open(args.report,'w') as f: json.dump(report,f,indent=2)
print(f'METRIC balanced_rows={len(rows_out)}')
print(f'METRIC balanced_games={len(rows_per_game)}')
print(f'METRIC balanced_openings={len(opening_counts)}')
print(f'METRIC balanced_top_opening_rows={max(opening_counts.values()) if opening_counts else 0}')
print(f'METRIC balanced_skipped_duplicate_fens={skipped_dupe}')
print(f'METRIC balanced_skipped_opening_cap={skipped_opening}')
print(f'METRIC balanced_avg_rows_per_game={sum(rows_per_game)/max(1,len(rows_per_game)):.6f}')
print(f'METRIC balanced_history_plies={args.history_plies}')
print(f'METRIC balanced_rows_with_history={train["history_available"]}')
if dev is not None:
  print(f'METRIC balanced_dev_rows={len(dev["rows"])}')
  print(f'METRIC balanced_dev_games={len(dev["rows_per_game"])}')
  print(f'METRIC balanced_dev_openings={len(dev["opening_counts"])}')
