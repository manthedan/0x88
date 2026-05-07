#!/usr/bin/env python3
import json,glob,os
for p in sorted(glob.glob('artifacts/diagnostics/queen_fixed_suite_*.json')):
 d=json.load(open(p)); s=d['summary']; rs=d['results']
 bysrc={}
 for r in rs:
  src=r['original'].get('source_model') or r['original'].get('source')
  bysrc.setdefault(src,[0,0,0])
  bysrc[src][0]+=1; bysrc[src][1]+=1 if r['selectedRisk'] else 0; bysrc[src][2]+=1 if (r['selected'] and r['selected']['risk'].get('movingQueen')) else 0
 print(os.path.basename(p), 'positions',s['positions'],'risk',s['selectedRisk'],f"rate={s['riskRate']:.3f}",'queenMoves',s['queenMoves'],'white',f"{s['by_turn']['white']['rate']:.3f}",'black',f"{s['by_turn']['black']['rate']:.3f}")
 print('  by source:', {k:{'n':v[0],'risk':v[1],'rate':round(v[1]/v[0],3),'queenMoves':v[2]} for k,v in bysrc.items()})
