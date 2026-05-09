# Unsloth RL Economics Triage for Tiny Leela

Status: **Active triage note**.  This records which ideas from Unsloth's LLM/GRPO training docs transfer to tiny Leela, and which do not.

## Core decision

Do **not** copy Unsloth's LLM stack.  Copy the economics:

```text
rollout/search inference is expensive
memory should be reused between actor and trainer modes
auxiliary losses should be candidate/chunk based
multiple candidates from one position form a useful relative ranking group
quantization should be measured and, if needed, trained for
```

For this project the best translation is:

```text
position = prompt
candidate moves / continuations = generations
teacher/search Q or regret = reward
group-relative candidate ranking = chess-specific GRPO analogue
```

Implement this as **candidate regret/ranking distillation**, not as a generic RLHF/GRPO framework.

## Adopt now

### 1. Candidate-group regret training

For each position, build a compact move group:

```text
teacher top-k
student top-k
played move
checks / captures / promotions
random legal distractors
known tactical or queen/material blunder candidates
```

Label candidates with side-to-move-consistent values:

```text
teacher/search Q
regret = Q_best - Q_move
mate/tactic/tablebase flags when available
material or queen swing diagnostics when useful
```

Preferred first losses:

```text
policy CE/KL on normal policy target
candidate AV regression/classification on Q buckets
regret prediction
pairwise Bradley-Terry or margin ranking loss
softmax over -regret
```

Use GRPO-style normalized advantages only after stability checks; if all candidate Q values are flat, std normalization is noisy.

### 2. Chunked candidate auxiliary heads

Dense from-to policy is fine.  Dense auxiliary tensors for every possible move are not the default.

Use the trunk once, then score candidate moves in chunks:

```python
H = trunk(tokens)                     # [B, 64, d]
policy_logits = policy_head(H)        # dense policy remains cheap enough
wdl_logits = wdl_head(H)

for cand_chunk in chunks(candidate_moves, candidate_chunk_size):
    q_pred = av_head(H, cand_chunk)
    regret_pred = regret_head(H, cand_chunk)
    loss += av_loss(q_pred, targets)
    loss += regret_or_ranking_loss(regret_pred, targets)
```

Make `candidate_chunk_size` configurable, starting with 16 or 32.

### 3. Actor/search throughput metrics

For self-play or reanalysis, optimize and report:

```text
positions generated per GPU-hour
usable training positions per dollar
batched evals per second inside PUCT
search nodes per second
chunk upload/download throughput
trainer examples per second
```

Do not scale trainer hardware before actor/search throughput is measured.

### 4. Actor/trainer memory lifecycle

If a single GPU alternates between self-play and training, add explicit phases:

```text
actor_start()
actor_generate_games_or_reanalysis()
actor_flush_chunks()
actor_free_memory()

trainer_start()
trainer_train_steps()
trainer_publish_checkpoint()
trainer_free_memory()
```

Actor memory:

```text
model weights, eval batches, MCTS trees, policy/value caches, game buffers
```

Trainer memory:

```text
model weights, activations, gradients, optimizer states, replay batches
```

Do not keep both fully active on the 24 GB 3090 unless profiling proves it is safe.

### 5. PTQ before QAT, QAT before exotic precision

Deployment precision ladder:

```text
FP16/BF16 training and ONNX export
ONNX simplification + parity check
INT8 PTQ drift measurement
INT8 QAT only if PTQ hurts policy/value/ranking/Elo
FP8 only for compatible actor hardware later
```

Measure drift on:

```text
policy top-k agreement
WDL calibration
candidate AV/ranking accuracy
catastrophic regret rate
fixed-node Elo
browser latency and memory
```

## Defer or reject

```text
vLLM/KV-cache machinery:
  not relevant to a 64-token encoder without autoregressive KV cache.

LLM-style learned reward model:
  not needed for chess strength; WDL/search/tablebase/teacher values define the reward.
  Verifier-style labels are useful only as auxiliary diagnostics.

FP8 on current RTX 3090 path:
  not a near-term win because the 3090 lacks modern native FP8 tensor-core support.

Custom kernels/offload systems:
  defer until measured bottlenecks justify them.
```

## Near-term implementation target

After the clean 100M h8/h7 BT4 baseline is built, add a small candidate-regret overlay:

```text
positions: 100k-1M from supervised/dev/eval/failure sets
candidates: K=8-16 moves per position
labels: teacher/search Q, regret, best candidate, optional tactical/material flags
model: current SquareFormer/BT4 trunk + candidate AV/regret/ranking heads
training: normal policy/WDL + candidate chunked losses
```

Promotion metrics:

```text
policy top-k
average/p95/p99 regret@top1
pairwise candidate ranking accuracy
queen/material blunder suite
PUCT 32/64/128 Elo
Elo per millisecond
```

## Priority order

```text
1. Finish 100M h8 dataset/cache/BT4 supervised baseline.
2. Add candidate-only AV/regret/ranking heads with chunked losses.
3. Build candidate-regret overlay labels from teacher/search/student top-k.
4. Evaluate regret and tactical diagnostics before claiming Elo gains.
5. Add actor/trainer lifecycle hooks when self-play/reanalysis shares one GPU.
6. Measure INT8 PTQ; add QAT only if needed.
7. Test FP8 actor inference only on suitable cloud GPUs after FP16/INT8 paths are stable.
```
