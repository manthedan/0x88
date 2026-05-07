#!/usr/bin/env python3
from __future__ import annotations
import argparse, glob, json
from collections import defaultdict

PIECE_VALUE = {'p': 1, 'n': 3, 'b': 3, 'r': 5, 'q': 9}

def material(fen: str):
    board = fen.split()[0]
    m = {'w': 0, 'b': 0, 'wq': 0, 'bq': 0}
    for c in board:
        if c.isalpha():
            side = 'w' if c.isupper() else 'b'
            p = c.lower()
            if p in PIECE_VALUE:
                m[side] += PIECE_VALUE[p]
            if p == 'q':
                m[side + 'q'] += 1
    return m

def tiny_side(game: dict, candidate: str) -> str:
    return 'w' if game.get('white') == candidate else 'b'

def main():
    ap = argparse.ArgumentParser(description='Material/queen-loss diagnostics for uci_anchor_arena JSON outputs.')
    ap.add_argument('files', nargs='*', default=glob.glob('artifacts/anchor_arena/latest_cnns/*_stockfish_maia_uho_v1.json'))
    ap.add_argument('--json-out', default='')
    args = ap.parse_args()
    report = []
    for path in args.files:
        data = json.load(open(path))
        cand = data['candidate']['name']
        by = defaultdict(lambda: defaultdict(int))
        examples = defaultdict(list)
        for idx, g in enumerate(data['games']):
            anchor = g['anchor']
            tiny = tiny_side(g, cand)
            opp = 'b' if tiny == 'w' else 'w'
            mat = material(g['finalFen'])
            loss = g['tinyScore'] == 0
            qdown = mat[tiny + 'q'] < mat[opp + 'q']
            qmiss = mat[tiny + 'q'] == 0 and mat[opp + 'q'] > 0
            mdown = mat[tiny] + 3 <= mat[opp]
            s = by[anchor]
            s['games'] += 1
            s['losses'] += int(loss)
            s['queen_down_losses'] += int(loss and qdown)
            s['queen_missing_losses'] += int(loss and qmiss)
            s['material_down_losses'] += int(loss and mdown)
            s['illegal_games'] += int(bool(g.get('illegal')))
            if loss and (qdown or mdown) and len(examples[anchor]) < 3:
                examples[anchor].append({'game_index': idx, 'tiny_side': tiny, 'tinyScore': g['tinyScore'], 'plies': g.get('plies'), 'finalFen': g['finalFen'], 'material': mat, 'queen_down': qdown, 'material_down': mdown})
        print(f'\n{path} candidate={cand}')
        item = {'path': path, 'candidate': cand, 'anchors': {}}
        for anchor, s in by.items():
            d = dict(s)
            losses = max(1, d['losses'])
            d['queen_down_loss_rate'] = d['queen_down_losses'] / losses
            d['material_down_loss_rate'] = d['material_down_losses'] / losses
            item['anchors'][anchor] = {**d, 'examples': examples[anchor]}
            print(f" {anchor:16} games={d['games']:2} losses={d['losses']:2} queen_down_losses={d['queen_down_losses']:2} ({d['queen_down_loss_rate']:.0%} of losses) material_down_losses={d['material_down_losses']:2} illegal={d['illegal_games']}")
        report.append(item)
    if args.json_out:
        with open(args.json_out, 'w') as f: json.dump(report, f, indent=2)

if __name__ == '__main__':
    main()
