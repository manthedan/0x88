#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, time
from pathlib import Path

def main() -> int:
    ap = argparse.ArgumentParser(description='Write a SquareFormer V2 multi-stream training config skeleton.')
    ap.add_argument('--out', required=True)
    ap.add_argument('--policy-dataset', default='data/datasets/supervised_100m_elite_tcec_v1')
    ap.add_argument('--policy-cache-manifest', default='data/datasets/supervised_100m_elite_tcec_v1/cache_squareformer_h2/cache_manifest.json')
    ap.add_argument('--value-overlay', default='data/teacher/root_stockfish_d8_mpv4_100k_pilot_v1/labels/root_stockfish_d8_mpv4_100k.jsonl.zst')
    ap.add_argument('--av-overlay', default='data/public_teacher_overlays/stockfish_root_multipv_av_100k_v1/shards/part_000000.jsonl.zst')
    args = ap.parse_args()
    cfg = {
        'config_id': 'squareformer_v2_tonight_smoke_v1',
        'created_at_unix': time.time(),
        'streams': [
            {
                'name': 'policy_clean',
                'ratio': 0.55,
                'kind': 'supervised_cache',
                'dataset': args.policy_dataset,
                'cache_manifest': args.policy_cache_manifest,
                'loss_mask': {'policy': 1.0, 'wdl': 1.0, 'q_bucket': 0.0, 'av': 0.0, 'ranking': 0.0, 'regret': 0.0},
                'sample_weight': 1.0,
            },
            {
                'name': 'value_broad_stockfish_100k',
                'ratio': 0.20,
                'kind': 'position_eval_overlay',
                'shards': [args.value_overlay],
                'loss_mask': {'policy': 0.25, 'wdl': 1.0, 'q_bucket': 1.0, 'av': 0.0, 'ranking': 0.0, 'regret': 0.0},
                'sample_weight': 1.0,
                'notes': 'Root Stockfish pilot; policy is sparse MultiPV teacher, not behavior cloning.',
            },
            {
                'name': 'action_value_stockfish_multipv_100k',
                'ratio': 0.25,
                'kind': 'action_value_overlay',
                'shards': [args.av_overlay],
                'candidate_source': 'group_by_position_key',
                'max_candidates': 8,
                'loss_mask': {'policy': 0.0, 'wdl': 0.0, 'q_bucket': 0.0, 'av': 1.0, 'ranking': 0.5, 'regret': 0.25},
                'sample_weight': 1.0,
                'notes': 'Smoke AV labels from root MultiPV scores. Replace/augment with ChessBench ASAP.',
            },
        ],
        'dev': {
            'policy_dev': 'data/datasets/supervised_100m_elite_tcec_v1/dev/dev_1000000.jsonl.zst',
            'av_dev_overlay': args.av_overlay,
        },
        'recommended_first_train': {
            'max_rows': 1000000,
            'epochs': 1,
            'batch_size': 256,
            'variant': 'v2',
            'goal': 'prove V2 plumbing/AV metrics, not strength claim',
        },
    }
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(cfg, indent=2))
    print(f'wrote {out}')
    print('METRIC v2_stream_count=3')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
