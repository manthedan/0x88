# Muon experiment notes

## 2026-05-09 CNN64x5 1M AdamW vs Muon smoke

Run artifact:

```text
artifacts/muon_experiments/cnn64x5_1m_adamw_vs_muon_20260509/
```

Setup:

```text
model: CNN64x5 residual AV/kitchen-sink path
train data path: supervised_10m_elite_tcec_v1 h2 cache
policy rows/epoch: 1,000,000
AV positions/epoch: 1,000,000
epochs: 2
AdamW baseline lr: 1e-4
Muon hybrid: Muon on stem/trunk matrix-like weights; AdamW on policy/WDL/AV/rank/regret heads, embeddings, biases, norms
Muon momentum: 0.95
Muon Newton-Schulz steps: 5
```

Summary:

| optimizer | epoch | policy CE | top1 | AV MSE | AV top1 | composite |
|---|---:|---:|---:|---:|---:|---:|
| AdamW | 1 | 3.788461 | 0.143246 | 0.276768 | 0.151419 | 4.593990 |
| AdamW | 2 | 3.243067 | 0.201840 | 0.211364 | 0.165296 | 3.916620 |
| Muon | 1 | 5.448449 | 0.075733 | 0.309585 | 0.088096 | 6.320923 |
| Muon | 2 | 3.962243 | 0.147610 | 0.257594 | 0.117599 | 4.730197 |

Interpretation: first Muon smoke was stable but worse than AdamW at identical LR and short horizon. This is *not* the setup from Aaron Leslie's comma-compression writeup, where Muon is used as a late-stage switch after long AdamW/QAT/C1a-style convergence.

## 2026-05-09 late-stage switch smoke

Run artifact:

```text
artifacts/muon_experiments/cnn64x5_1m_late_switch_adamw_vs_muon_20260509/
```

Setup: start both branches from the AdamW epoch-2 checkpoint above, then train two more 1M-row epochs with either AdamW continuation or Muon switch.

| optimizer | seed | epoch | policy CE | top1 | AV MSE | AV top1 | composite |
|---|---|---:|---:|---:|---:|---:|---:|
| AdamW continue | AdamW e2 | 1 | 3.015123 | 0.228067 | 0.196003 | 0.190481 | 3.657802 |
| AdamW continue | AdamW e2 | 2 | 2.837044 | 0.255331 | 0.177692 | 0.203022 | 3.441021 |
| Muon switch | AdamW e2 | 1 | 3.031171 | 0.227758 | 0.183160 | 0.187808 | 3.647005 |
| Muon switch | AdamW e2 | 2 | 2.893358 | 0.247648 | 0.165163 | 0.214330 | 3.472581 |

Late-switch Muon was much closer than from-scratch Muon and improved AV MSE/top1, but AdamW continuation still had slightly better policy CE/top1 and composite in this short smoke. This supports treating Muon as a late-stage specialist rather than an initial/default optimizer, but we still do not have a win for Tiny Leela yet.

Next Muon test, if any: only after a real plateau/checkpoint, with a lower Muon LR or separate Muon LR multiplier, cosine/warm restart, and a metric split that explicitly tracks policy CE vs AV head quality.
