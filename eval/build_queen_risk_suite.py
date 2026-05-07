#!/usr/bin/env python3
import argparse,json,glob
from pathlib import Path
ap=argparse.ArgumentParser(); ap.add_argument('--inputs', nargs='+', required=True); ap.add_argument('--out', required=True); ap.add_argument('--min-drop', type=int, default=300); ap.add_argument('--actual-captures', action='store_true'); args=ap.parse_args()
seen={}
for pat in args.inputs:
  for p in glob.glob(pat):
    d=json.load(open(p)); name=Path(p).name
    for x in d.get('incidents',[]):
      drop=x.get('stockfish',{}).get('cp_drop_after_selected')
      ok=(drop is not None and drop>=args.min_drop) or (args.actual_captures and x.get('actual_reply_captured_queen'))
      if not ok: continue
      fen=x['fen_before']
      cur=seen.get(fen)
      item={'id':f'q{len(seen):04d}','fen':fen,'source':name,'source_model':name.split('_')[2] if '_' in name else name,'anchor':x.get('anchor'),'ply':x.get('ply'),'original_move':x.get('selected_move_uci'),'original_prob':x.get('selected_prob'),'original_drop':drop,'actual_reply_captured_queen':x.get('actual_reply_captured_queen'),'actual_reply_uci':x.get('actual_reply_uci')}
      if cur is None or (drop or 0)>(cur.get('original_drop') or -10**9): seen[fen]=item
positions=list(seen.values())
positions.sort(key=lambda z: (z.get('original_drop') is not None, z.get('original_drop') or -999999), reverse=True)
out={'description':'Fixed queen-risk FEN suite from anchor incidents. Same positions can be scored by any model to remove game-length/trajectory confounds.','filters':{'min_drop':args.min_drop,'actual_captures':args.actual_captures},'positions':positions}
Path(args.out).parent.mkdir(parents=True,exist_ok=True); json.dump(out,open(args.out,'w'),indent=2)
print(f'wrote {args.out} positions={len(positions)}')
