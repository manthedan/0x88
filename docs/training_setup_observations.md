# 100M CNN Training Setup Observations

Date: 2026-05-07

## Current setup

Canonical CNN sweep currently uses:

```text
epochs=3
batch_size=2048
lr=1e-4
min_lr=1e-5
lr_schedule=cosine
warmup_steps=2000
weight_decay=1e-4
policy_label_smoothing=0.02
grad_clip_norm=1.0
ema_decay=0.999
amp=bf16
optimizer=AdamW/fused AdamW when available
shuffle_chunk_rows=262144
prefetch_batches=2
```

## Read from logs so far

The training setup looks healthy.

### Cosine LR is behaving correctly

Observed epoch-end learning rates match a 3-epoch cosine decay:

```text
epoch 1: ~7.856e-5
epoch 2: ~3.295e-5
epoch 3: ~1.000e-5
```

Loss curves are smooth within epochs; no obvious divergence, warmup issue, AMP instability, or data starvation.

### EMA is consistently useful

EMA beats raw checkpoints at every epoch observed.

```text
cnn_32x4:
  e1 raw 2.259760 -> ema 2.222136
  e2 raw 2.142288 -> ema 2.131496
  e3 raw 2.110904 -> ema 2.107368

cnn_48x5:
  e1 raw 2.120120 -> ema 2.086776
  e2 raw 2.000544 -> ema 1.988080
  e3 raw 1.964648 -> ema 1.960932

cnn_64x6:
  e1 raw 2.027688 -> ema 1.991368
  e2 raw 1.897000 -> ema 1.884804
```

EMA advantage shrinks as LR decays, which is expected and healthy.

### Scaling is clean

Policy CE improves monotonically with model size:

```text
cnn_32x4 e3 EMA: 2.107368
cnn_48x5 e3 EMA: 1.960932
cnn_64x6 e2 EMA: 1.884804
```

`cnn_64x6` at epoch 2 already beats `cnn_48x5` epoch 3 on supervised CE and is also stronger than ChessFormer v1 e3 supervised CE. This does not automatically imply stronger play, but it is a good training signal.

### Throughput/memory look stable

Approximate observed throughput:

```text
cnn_32x4: ~105k rows/sec, ~255 MiB
cnn_48x5: ~85k rows/sec,  ~360 MiB
cnn_64x6: ~69k rows/sec,  ~481 MiB
```

Scaling is plausible; no obvious loader bottleneck.

## Recommended next-round changes

Do not interrupt the current sweep. For the next supervised round, consider:

### Train longer

The models are still improving meaningfully at epoch 3.

```text
cnn_32x4 EMA e2->e3 CE gain: ~0.024
cnn_48x5 EMA e2->e3 CE gain: ~0.027
```

Try:

```text
epochs=5
lr=1e-4
min_lr=1e-6 to 3e-6
cosine schedule
warmup_steps=2000-5000
```

### Export EMA ONNX explicitly

Because EMA is consistently better, release gates and browser/arena benches should prefer EMA artifacts. Add training support for either:

```text
--ema-onnx-out model_ema.onnx
```

or a post-training conversion from best EMA checkpoint to ONNX.

### Add mid-epoch eval for longer runs

For 5+ epochs, evaluate every ~25M-50M rows so overfitting/schedule issues are visible before epoch end.

### Sweep label smoothing later

Current `policy_label_smoothing=0.02` is reasonable, but future runs should compare:

```text
0.00, 0.01, 0.02
```

PUCT relies on calibrated priors, so smoothing should be judged by both supervised metrics and PUCT calibration/arena results.

### Add value calibration diagnostics

WDL CE improves, but search strength may be bottlenecked by value calibration. Add reports such as:

```text
value reliability buckets
value vs Stockfish cp/WDL calibration
PUCT override quality split by root Q vs prior
```

## Bottom line

The current training setup is solid. Cosine LR and EMA are working. Bigger CNNs are scaling cleanly. The main near-term training-infra improvement is to make EMA the default exported/benched artifact, then try longer 5-epoch cosine runs with a lower min LR after this sweep finishes.
