# ChannelFormer CNN architecture

ChannelFormer is the Maia-2-inspired architecture lane that keeps the existing dense policy/WDL runtime shape while adding a small attention mixer over learned CNN feature channels.

## Shape

```text
planes [B,C,8,8]
→ residual CNN trunk, e.g. 64x6
→ 1x1 channel projection to Cpatch concept maps
→ flatten each concept map: [B,Cpatch,64]
→ linear projection 64 → d_model
→ TransformerEncoder over concept/channel tokens
→ mean pool tokens
→ dense policy logits [B,1968]
→ WDL logits [B,3]
→ optional candidate AV head [B,K]
```

Unlike MoveFormer, ChannelFormer does not tokenize legal moves and does not need bucketed ONNX exports. Runtime uses the standard dense policy path; `architecture: "cnn_channel_transformer"` is accepted by the ONNX evaluator as a dense-policy model.

## First queued experiment

```text
artifacts/channelformer_10m_supervised_64x6_c32_d128_l2_e3
```

Configuration:

```text
channels=64
blocks=6
cpatch=32
channelformer_dim=128
heads=4
layers=2
ff_dim=256
policy_rows=10,000,000
epochs=3
batch_size=1024
supervised policy/WDL only
```

It is queued behind:

```text
artifacts/moveformer_10m_supervised_mf64x6_e3/pipeline.done
```

Smoke output:

```text
artifacts/channelformer_smoke_64x6_c32_d128_l2
```

## Trainer

```text
training/train_channelformer_av_multicache_torch.py
```

The trainer supports supervised-only runs by omitting `--av-cache` or setting `--av-positions 0`. It also supports optional ChessBench candidate AV training and AV ONNX export through the inherited CNN-AV candidate head path.
