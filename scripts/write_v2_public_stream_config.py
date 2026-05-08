#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, time
from pathlib import Path

def main() -> int:
    ap = argparse.ArgumentParser(description='Write SquareFormer V2 stream config using public Lichess eval + ChessBench AV overlays.')
    ap.add_argument('--out', required=True)
    ap.add_argument('--policy-dataset', default='data/datasets/supervised_100m_elite_tcec_v1')
    ap.add_argument('--policy-cache-manifest', default='data/datasets/supervised_100m_elite_tcec_v1/cache_squareformer_h2/cache_manifest.json')
    ap.add_argument('--value-overlay', required=True)
    ap.add_argument('--av-overlay', required=True)
    ap.add_argument('--value-cache', nargs='*', default=[], help='optional compact_position_eval_cache_v1 dir(s) for fast value training')
    ap.add_argument('--av-cache', nargs='*', default=[], help='optional compact_action_value_cache_v1 dir(s) or collection manifest for fast training')
    args = ap.parse_args()
    cfg = {
        'config_id': 'squareformer_v2_public_tonight_v1',
        'created_at_unix': time.time(),
        'streams': [
            {'name':'policy_clean_100m','ratio':0.45,'kind':'supervised_cache','dataset':args.policy_dataset,'cache_manifest':args.policy_cache_manifest,'loss_mask':{'policy':1.0,'wdl':1.0,'q_bucket':0.25,'av':0.0,'ranking':0.0,'regret':0.0},'sample_weight':1.0},
            {'name':'value_broad_lichess_eval','ratio':0.25,'kind':'position_eval_overlay','shards':[args.value_overlay], **(({'cache': args.value_cache[0]} if len(args.value_cache)==1 else {'caches': args.value_cache}) if args.value_cache else {}), 'loss_mask':{'policy':0.10,'wdl':1.0,'q_bucket':1.0,'av':0.0,'ranking':0.0,'regret':0.0},'sample_weight':1.0,'notes':'Public Lichess Stockfish evals; sparse PV policy low weight.'},
            {'name':'action_value_chessbench','ratio':0.30,'kind':'action_value_overlay','shards':[args.av_overlay], **(({'cache': args.av_cache[0]} if len(args.av_cache)==1 else {'caches': args.av_cache}) if args.av_cache else {}), 'candidate_source':'tensor_cache_or_group_by_position_key','max_candidates':8,'loss_mask':{'policy':0.0,'wdl':0.0,'q_bucket':0.0,'av':1.0,'ranking':0.75,'regret':0.25},'sample_weight':1.0,'notes':'ChessBench legal-move win_prob action values; initially top-k capped.'},
        ],
        'dev': {
            'policy_dev':'data/datasets/supervised_100m_elite_tcec_v1/dev/dev_1000000.jsonl.zst',
            'value_dev_overlay': args.value_overlay,
            'av_dev_overlay': args.av_overlay,
        },
        'recommended_first_train': {'max_rows':1000000,'epochs':1,'batch_size':512,'variant':'v2','goal':'public-teacher V2 smoke; compare AV top1/MSE vs Stockfish-MultiPV smoke'},
    }
    out=Path(args.out); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(cfg, indent=2))
    print(f'wrote {out}')
    print('METRIC v2_public_stream_count=3')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
