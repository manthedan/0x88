#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,math,gzip
from pathlib import Path


def cp_to_winrate(cp: float) -> float:
    cp=max(-1000.0,min(1000.0,float(cp)))
    return 1.0/(1.0+math.exp(-cp/250.0))

def bucket(loss: float) -> int:
    return 0 if loss<0.05 else 1 if loss<0.10 else 2 if loss<0.20 else 3

def open_text(path: Path, mode='rt'):
    if path.suffix == '.gz': return gzip.open(path, mode)
    return path.open(mode)

def load_labels(path: Path):
    labels={}; bad=0
    with open_text(path) as f:
        for line in f:
            if not line.strip(): continue
            try:
                r=json.loads(line); labels[str(r['id'])]=r
            except Exception: bad+=1
    return labels,bad

def augment_file(inp: Path, out: Path, labels: dict, default_null: bool):
    out.parent.mkdir(parents=True,exist_ok=True); total=hit=bad=0; sums=0.0; counts=[0,0,0,0]
    with open_text(inp) as fi, open_text(out,'wt') as fo:
        for line in fi:
            if not line.strip(): continue
            r=json.loads(line); total+=1; lab=labels.get(str(r.get('id','')))
            if lab is not None:
                try:
                    wb=cp_to_winrate(float(lab['cp_best'])); wp=cp_to_winrate(float(lab['cp_played'])); loss=max(0.0,wb-wp); b=bucket(loss)
                    r['stockfish_best_winrate']=wb; r['stockfish_played_winrate']=wp; r['stockfish_winrate_loss']=loss; r['stockfish_blunder_bucket']=b
                    hit+=1; sums+=loss; counts[b]+=1
                except Exception: bad+=1
            elif default_null:
                r['stockfish_best_winrate']=None; r['stockfish_played_winrate']=None; r['stockfish_winrate_loss']=None; r['stockfish_blunder_bucket']=-1
            fo.write(json.dumps(r,separators=(',',':'))+'\n')
    return {'rows':total,'labeled':hit,'bad':bad,'avg_loss':sums/max(1,hit),'buckets':counts}

def main():
    ap=argparse.ArgumentParser(description='Create a sparse Stockfish winrate-loss augmented copy of a supervised dataset directory.')
    ap.add_argument('--dataset-dir',required=True); ap.add_argument('--labels',required=True); ap.add_argument('--out-dir',required=True); ap.add_argument('--default-null',action='store_true')
    args=ap.parse_args(); src=Path(args.dataset_dir); out=Path(args.out_dir); labels,bad_labels=load_labels(Path(args.labels)); print(f'METRIC label_rows={len(labels)}'); print(f'METRIC label_bad_rows={bad_labels}')
    src_manifest={}
    if (src/'manifest.json').exists(): src_manifest=json.loads((src/'manifest.json').read_text())
    manifest={k:v for k,v in src_manifest.items() if k not in ('name','train_shards','dev','report')}
    manifest.update({'name':out.name,'source_dataset':str(src),'labels':args.labels,'files':[],'train_shards':[]})
    totals={'rows':0,'labeled':0,'bad':0,'avg_sum':0.0,'buckets':[0,0,0,0]}
    train_inputs=[src/p for p in src_manifest.get('train_shards',[])] or sorted((src/'train').glob('*.jsonl*'))
    dev_inputs=[src/src_manifest['dev']] if src_manifest.get('dev') else sorted((src/'dev').glob('*.jsonl*'))
    for inp in train_inputs:
        rel=inp.relative_to(src); dest=out/rel; st=augment_file(inp,dest,labels,args.default_null); manifest['train_shards'].append(str(rel)); manifest['files'].append({'input':str(inp),'output':str(dest),**st})
        totals['rows']+=st['rows']; totals['labeled']+=st['labeled']; totals['bad']+=st['bad']; totals['avg_sum']+=st['avg_loss']*st['labeled']; totals['buckets']=[a+b for a,b in zip(totals['buckets'],st['buckets'])]
    if dev_inputs:
        inp=dev_inputs[0]; rel=inp.relative_to(src); dest=out/rel; st=augment_file(inp,dest,labels,args.default_null); manifest['dev']=str(rel); manifest['files'].append({'input':str(inp),'output':str(dest),**st})
        totals['rows']+=st['rows']; totals['labeled']+=st['labeled']; totals['bad']+=st['bad']; totals['avg_sum']+=st['avg_loss']*st['labeled']; totals['buckets']=[a+b for a,b in zip(totals['buckets'],st['buckets'])]
    manifest['total_train_rows']=sum(f['rows'] for f in manifest['files'] if '/train/' in ('/'+f['output'].replace('\\','/')))
    manifest['total_dev_rows']=sum(f['rows'] for f in manifest['files'] if '/dev/' in ('/'+f['output'].replace('\\','/')))
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(f"METRIC augmented_rows={totals['rows']}"); print(f"METRIC augmented_labeled_rows={totals['labeled']}"); print(f"METRIC augmented_bad_rows={totals['bad']}"); print(f"METRIC augmented_avg_loss={totals['avg_sum']/max(1,totals['labeled']):.6f}")
    for i,c in enumerate(totals['buckets']): print(f'METRIC augmented_bucket_{i}={c}')
if __name__=='__main__': main()
