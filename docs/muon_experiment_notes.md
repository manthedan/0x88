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

Interpretation: first Muon smoke was stable but worse than AdamW at identical LR and short horizon. Do not promote Muon as default. If revisiting, sweep Muon-specific LR lower than AdamW, try cosine/warmup, and keep it behind a quality gate.
