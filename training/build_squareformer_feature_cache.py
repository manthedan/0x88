#!/usr/bin/env python3
"""Materialize compact SquareFormer token caches as expanded float feature memmaps.

This is optional: compact-token + embedding input is usually preferable for full 100M
because expanded float16 features are large (~5.6KB/position for history=2).
Use this for dev/profiling, or if CPU one-hot expansion becomes the bottleneck and
there is enough disk bandwidth/capacity.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

PIECES='.PNBRQKpnbrqk'

def expand_compact(tok: np.ndarray, history: int) -> np.ndarray:
    B = tok.shape[0]
    arr = np.zeros((B,64,(history+1)*len(PIECES)+8), dtype=np.float32)
    eye = np.eye(len(PIECES), dtype=np.float32)
    for h in range(history+1):
        ids = tok[:,:,h].astype(np.int64).clip(0, len(PIECES)-1)
        arr[:,:,h*len(PIECES):(h+1)*len(PIECES)] = eye[ids]
    base=(history+1)*len(PIECES)
    arr[:,:,base+0]=(tok[:,:,history+1]==1)
    arr[:,:,base+1]=(tok[:,:,history+1]==2)
    flags=tok[:,:,history+2]
    arr[:,:,base+2]=((flags&1)>0)
    arr[:,:,base+3]=((flags&2)>0)
    arr[:,:,base+4]=((flags&4)>0)
    arr[:,:,base+5]=((flags&8)>0)
    arr[:,:,base+6]=tok[:,:,history+3].astype(np.float32)
    arr[:,:,base+7]=tok[:,:,history+4].astype(np.float32)/100.0
    return arr

def main() -> None:
    ap=argparse.ArgumentParser(description='Build expanded SquareFormer feature memmap from compact cache dir.')
    ap.add_argument('--input-cache', required=True, help='compact_square_tokens_v1 / compact_position_eval_cache_v1 / compact_action_value_cache_v1 dir')
    ap.add_argument('--out', required=True)
    ap.add_argument('--dtype', choices=['float16','float32'], default='float16')
    ap.add_argument('--batch-size', type=int, default=8192)
    ap.add_argument('--max-rows', type=int, default=0)
    args=ap.parse_args()
    inp=Path(args.input_cache); out=Path(args.out); out.mkdir(parents=True, exist_ok=True)
    meta=json.loads((inp/'meta.json').read_text())
    rows=int(meta['rows']); F=int(meta['token_features']); history=int(meta.get('history_plies',2))
    if args.max_rows: rows=min(rows,args.max_rows)
    tokens=np.memmap(inp/'tokens.uint8', np.uint8, 'r', shape=(int(meta['rows']),64,F))
    input_dim=(history+1)*len(PIECES)+8
    dt=np.float16 if args.dtype=='float16' else np.float32
    feats=np.memmap(out/f'features.{args.dtype}', dt, 'w+', shape=(rows,64,input_dim))
    for off in range(0, rows, args.batch_size):
        end=min(rows, off+args.batch_size)
        feats[off:end]=expand_compact(np.asarray(tokens[off:end]), history).astype(dt, copy=False)
        if end % 100000 == 0 or end == rows:
            print(f'METRIC feature_cache_rows_written={end}', flush=True)
    feats.flush()
    out_meta={
        'format':'squareformer_expanded_feature_cache_v1',
        'source_cache':str(inp),
        'rows':rows,
        'history_plies':history,
        'input_dim':input_dim,
        'dtype':args.dtype,
        'bytes_per_row':64*input_dim*np.dtype(dt).itemsize,
    }
    (out/'meta.json').write_text(json.dumps(out_meta, indent=2))
    print(f'METRIC feature_cache_rows={rows}')
    print(f'METRIC feature_cache_input_dim={input_dim}')
    print(f'METRIC feature_cache_bytes_per_row={out_meta["bytes_per_row"]}')

if __name__ == '__main__':
    main()
