# 1M 64x6 architecture ladder smoke

Dataset: `supervised_1m_v1` / `supervised_1m_v1_winrate_aug`  
Cache: history2 + state planes, 46 input planes  
Hardware: RTX 3090, CUDA AMP  
Run shape: `64x6`, batch `4096`, `epochs=8` over the 1M train set (~1952 optimizer steps total)

| Variant | Dev policy CE | WDL CE | Top1 | Top4 | Top8 | Notes |
|---|---:|---:|---:|---:|---:|---|
| baseline policy+WDL | 3.086569 | 0.918039 | 0.228530 | 0.480080 | 0.639160 | best top-k/WDL in this smoke |
| side-info aux | 3.092488 | 0.921244 | 0.225960 | 0.478790 | 0.634520 | side heads do not help at these weights/run length |
| side-info + sparse SF winrate/blunder | 3.085480 | 0.925634 | 0.224590 | 0.477540 | 0.638050 | tiny CE edge, worse top-k/WDL |

Interpretation:

- The early 500-step positive side-info signal did not hold cleanly after ~2k steps.
- Sparse Stockfish winrate/blunder aux is functional but not promotion-worthy at current weights (`0.02`, `0.01`).
- Baseline `64x6` remains the cleanest architecture ladder point for longer training unless aux weights are tuned.
- Keep supervised `48x5` incumbent; these from-scratch 1M runs are validation/training-path experiments, not candidates.
