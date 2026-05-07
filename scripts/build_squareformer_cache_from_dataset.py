#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, subprocess, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

def run(cmd):
    print('+ '+' '.join(map(str,cmd)),flush=True); subprocess.check_call(list(map(str,cmd)))
def shard_name(rel):
    n=Path(rel).name
    for s in ('.jsonl.zst','.jsonl'):
        if n.endswith(s): return n[:-len(s)]
    return Path(rel).stem

def main():
    ap=argparse.ArgumentParser(description='Build SquareFormer compact token-cache shards from dataset manifest.')
    ap.add_argument('--dataset-dir',required=True); ap.add_argument('--out-dir',required=True); ap.add_argument('--python',default=sys.executable); ap.add_argument('--history-plies',type=int,default=None); ap.add_argument('--max-rows-per-shard',type=int,default=0); ap.add_argument('--workers',type=int,default=1)
    a=ap.parse_args(); root=Path(a.dataset_dir); out=Path(a.out_dir); man=json.loads((root/'manifest.json').read_text()); hist=man.get('history_plies',2) if a.history_plies is None else a.history_plies
    def build(rel):
        cdir=out/'train'/shard_name(rel); cmd=[a.python,'training/build_squareformer_token_cache.py','--input',root/rel,'--out',cdir,'--history-plies',hist]
        if a.max_rows_per_shard: cmd += ['--max-rows',a.max_rows_per_shard]
        run(cmd); return str(cdir)
    if a.workers<=1: shards=[build(r) for r in man['train_shards']]
    else:
        shards=[]
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            futs={ex.submit(build,r):r for r in man['train_shards']}
            for fut in as_completed(futs): shards.append(fut.result())
        shards.sort()
    dev=out/'dev'; run([a.python,'training/build_squareformer_token_cache.py','--input',root/man['dev'],'--out',dev,'--history-plies',hist,'--max-rows',man.get('total_dev_rows',0)])
    metas=[json.loads((Path(s)/'meta.json').read_text()) for s in shards]; dm=json.loads((dev/'meta.json').read_text())
    manifest={'dataset_manifest':str(root/'manifest.json'),'shards':shards,'dev_cache':str(dev),'history_plies':hist,'workers':a.workers,'validation':{'rows':{'train':sum(m['rows'] for m in metas),'dev':dm['rows']},'token_features':dm['token_features'],'policy_size':dm['policy_size']}}
    out.mkdir(parents=True,exist_ok=True); (out/'cache_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(f"METRIC square_cache_manifest_shards={len(shards)}"); print(f"METRIC square_cache_manifest_train_rows={manifest['validation']['rows']['train']}"); print('METRIC square_cache_manifest_dev=1')
if __name__=='__main__': main()
