#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, subprocess
from pathlib import Path
from contextlib import contextmanager

@contextmanager
def binary_opener(path):
    path=str(path)
    if path.endswith('.zst'):
        p=subprocess.Popen(['zstd','-dc',path],stdout=subprocess.PIPE)
        try:
            assert p.stdout is not None
            yield p.stdout
        finally:
            if p.stdout is not None: p.stdout.close()
            p.wait()
    else:
        with open(path,'rb') as f: yield f

def main():
    ap=argparse.ArgumentParser(description='Audit whether ChessBench win_prob is side-to-move or White-perspective using immediate mate moves.')
    ap.add_argument('--input', nargs='+', required=True)
    ap.add_argument('--max-records', type=int, default=200000)
    ap.add_argument('--max-mate-examples', type=int, default=20)
    args=ap.parse_args()
    import chess, msgpack
    records=0; move_rows=0; mate_moves=0
    by_side={'w': [], 'b': []}; examples=[]; field_mate_counts={}
    for path in args.input:
        with binary_opener(path) as f:
            unpacker=msgpack.Unpacker(f, raw=False)
            for rec in unpacker:
                if records >= args.max_records: break
                records += 1
                try: board=chess.Board(rec['fen'])
                except Exception: continue
                side='w' if board.turn == chess.WHITE else 'b'
                moves=rec.get('moves') or {}
                for uci, ev in moves.items():
                    move_rows += 1
                    mate_field=str(ev.get('mate'))
                    field_mate_counts[mate_field]=field_mate_counts.get(mate_field,0)+1
                    try: mv=chess.Move.from_uci(uci)
                    except Exception: continue
                    if mv not in board.legal_moves: continue
                    b2=board.copy(stack=False); b2.push(mv)
                    if b2.is_checkmate():
                        wp=float(ev.get('win_prob'))
                        by_side[side].append(wp); mate_moves += 1
                        if len(examples) < args.max_mate_examples:
                            examples.append({'fen':rec['fen'], 'side_to_move':side, 'move':uci, 'win_prob':wp, 'mate_field':ev.get('mate')})
                if records % 50000 == 0:
                    print(f'METRIC records_scanned={records}', flush=True)
        if records >= args.max_records: break
    def avg(xs): return sum(xs)/len(xs) if xs else float('nan')
    wavg=avg(by_side['w']); bavg=avg(by_side['b'])
    if by_side['w'] and by_side['b']:
        if wavg > 0.8 and bavg > 0.8: verdict='side_to_move_or_mover_perspective_likely'
        elif wavg > 0.8 and bavg < 0.2: verdict='white_perspective_likely'
        else: verdict='ambiguous'
    else: verdict='insufficient_immediate_mates'
    top_mate_fields=sorted(field_mate_counts.items(), key=lambda kv: kv[1], reverse=True)[:10]
    out={'records_scanned':records,'move_rows_scanned':move_rows,'immediate_mate_moves':mate_moves,'white_to_move_mate_count':len(by_side['w']),'black_to_move_mate_count':len(by_side['b']),'white_to_move_mate_win_prob_avg':wavg,'black_to_move_mate_win_prob_avg':bavg,'verdict':verdict,'examples':examples,'top_mate_field_values':top_mate_fields}
    print(json.dumps(out, indent=2))
    print(f'METRIC records_scanned={records}')
    print(f'METRIC move_rows_scanned={move_rows}')
    print(f'METRIC immediate_mate_moves={mate_moves}')
    print(f'METRIC white_to_move_mate_win_prob_avg={wavg if wavg==wavg else -1:.6f}')
    print(f'METRIC black_to_move_mate_win_prob_avg={bavg if bavg==bavg else -1:.6f}')

if __name__=='__main__': main()
