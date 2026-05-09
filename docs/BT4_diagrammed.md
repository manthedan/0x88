Yes. Here is a structured write-up of **what `lczero-training` is doing**, and how I’d adapt the methodology into a **tiny, readable “nano-Leela” training stack**.

## 1. What `lczero-training` is

`lczero-training` is not the chess engine itself. It is the training-side system that consumes self-play or supervised training data, turns it into neural-network tensors, trains a policy/value network, checkpoints it, and exports weights for use by lc0. The repo currently exposes two generations of training code: an older TensorFlow pipeline under `tf/`, and a newer active-development pipeline under `src/` and `csrc/` using Python plus a C++ data loader, with JAX as the intended training core. The repo’s README still describes the older TensorFlow flow, while `docs/README.md` describes the newer pipeline and notes that it is actively changing. ([GitHub][1])

The big conceptual loop is:

```text
self-play / teacher games
        ↓
chunk files containing per-position training frames
        ↓
shuffle / sample / rescore / unpack
        ↓
training tensors
        ↓
policy + value + moves-left losses
        ↓
checkpoint
        ↓
export lc0-compatible weights
        ↓
new net used by engine / self-play clients
```

For a tiny Leela, you probably want to **keep the tensor targets and training philosophy**, but **not copy the whole distributed RL production loop**.

---

## 2. The older TensorFlow methodology

The old README says the TensorFlow training pipeline lives in `tf/`, requires TensorFlow on Linux, and needs protobuf files built via `init.sh`. Training data comes from `storage.lczero.org` as tar files containing chunks/games; the example flow downloads a tar, extracts it, writes a YAML config, and runs `./train.py --cfg configs/example.yaml --output /tmp/mymodel.txt`. It also supports TensorBoard and automatic restore from an existing model path. ([GitHub][1])

The old example config is very informative because it shows a small lc0-family net:

```yaml
name: 'kb1-64x6'
gpu: 0

dataset:
  num_chunks: 100000
  train_ratio: 0.90
  input_train: '/path/to/chunks/*/draw/'
  input_test: '/path/to/chunks/*/draw/'

training:
  batch_size: 2048
  total_steps: 140000
  shuffle_size: 524288
  lr_values: [0.02, 0.002, 0.0005]
  lr_boundaries: [100000, 130000]
  policy_loss_weight: 1.0
  value_loss_weight: 1.0
  moves_left_loss_weight: 1.0

model:
  filters: 64
  residual_blocks: 6
  se_ratio: 2
  policy: 'attention'
  value: 'wdl'
  moves_left: 'v1'
```

The important thing is not the exact syntax; it is the recipe: **read chunks, shuffle positions, train a 64×6 residual net with policy, WDL value, and moves-left heads, then export an lc0-compatible network**. The public example uses `batch_size: 2048`, `total_steps: 140000`, staged learning rates, `filters: 64`, and `residual_blocks: 6`. ([GitHub][2])

For a tiny Leela, this old pipeline is actually closer to what we want than the modern transformer-heavy system: a small residual CNN, straightforward supervised targets, and a single-GPU-friendly training loop.

---

## 3. The newer methodology

The newer docs describe a long-running training system. The stated motivation is that the older script required fresh starts for new networks and TensorFlow compilation could dominate single-epoch training time. The new system is meant to run continuously: monitor for enough new data, trigger training, execute the training loop, and export the trained network. The architecture doc says the core loop is JAX, while loading and preprocessing are handled by a C++ library exposed to Python through pybind11. ([GitHub][3])

The new pipeline expects four main ingredients:

```text
training data
configuration file
initial checkpoint
pipeline run
```

The docs show `lc0-init` to create a checkpoint and `lc0-tui` to run training:

```bash
uv run lc0-init --config <your_config>.textproto --lczero_model <model>.pb.gz

CUDA_VISIBLE_DEVICES=0 uv run lc0-tui \
  --config <your_config>.textproto \
  --logfile train.log
```

The `--lczero_model` argument is optional; without it, the model is initialized from random weights. ([GitHub][4])

For tiny Leela, the key takeaway is that the official direction is now **daemon + data-loader + config + checkpoint + export**, not a one-off training script. But for a nano project, I would intentionally collapse that back into a simple script first.

---

## 4. Training data format

The newer docs define the basic units clearly:

| Term                          | Meaning                                                       |
| ----------------------------- | ------------------------------------------------------------- |
| **Chunk / Game**              | A single training game, usually an individual `.gz` chunk.    |
| **Chunk source**              | A `.tar` or `.gz` file containing one or more chunks.         |
| **Frame / Record / Position** | A single training position inside a chunk.                    |
| **Training tensor**           | One batched set of NN inputs and targets for a training step. |

The new pipeline can index and seek inside `.tar` files, watches a directory and subdirectories for new files, and prefers tar chunk sources over many individual `.gz` files for efficiency. ([GitHub][4])

The older `training_tuple.md` says a converted V6 training example contains:

```text
planes:     (112, 64) float32
probs:      (1858,) float32
winner:     (3,) float32
best_q:     (3,) float32
plies_left: scalar float32
```

The 112 planes are the board-state features; the policy vector stores move probabilities; `winner` is the game result as win/draw/loss; `best_q` is the post-search value target; and `plies_left` estimates remaining plies. ([GitHub][5])

The newer docs describe a current batch tensor format as:

```text
inputs:        [batch_size, 112, 8, 8]
policy_target: [batch_size, 1862]
value_target:  [batch_size, 6, 3]
```

The six value rows are listed as `result`, `best`, `played`, `orig`, `root`, and `st`; the three columns are `q`, `draw`, and `movesleft`. ([GitHub][4])

For tiny Leela, I would simplify this immediately:

```text
x:          [B, 112, 8, 8]
pi_target:  [B, 1862] or [B, 1858], choose one and freeze it
wdl_target: [B, 3]
q_target:   optional scalar W - L
mlh_target: optional scalar plies_left
legal_mask: [B, policy_size]
```

The policy-size mismatch is historical: older topology docs describe a 1858-vector policy gathered from an 80×8×8 tensor, while the newer AlphaZero primer describes 1862 policy elements. For a toy project, choose one encoding and keep it consistent across training, inference, and browser deployment. ([Leela Chess Zero][6])

---

## 5. Data loader methodology

The new data loader is the most reusable design idea in the repo. It is a C++ module exposed to Python that handles loading, preprocessing, shuffling, and feeding training data; it returns NumPy-compatible buffers usable from JAX. ([GitHub][7])

The documented stage pipeline is:

```text
FilePathProvider
  → ChunkSourceLoader
  → ShufflingChunkPool
  → ChunkRescorer
  → ChunkUnpacker
  → ShufflingFrameSampler
  → TensorGenerator
```

The stages do the following: watch a directory, read chunk sources, maintain a shuffled pool of recent chunks, optionally rescore chunks using tablebases or policy adjustments, unpack chunks into frames, reservoir-sample frames, and finally convert frames into batched tensors. ([GitHub][7])

This is a strong production design because it separates concerns:

```text
file discovery      independent
chunk reading       independent
recency window      independent
position sampling   independent
tensor encoding     independent
training            independent
export              independent
```

For tiny Leela, I would keep the **conceptual pipeline**, but make it radically smaller:

```text
data/*.npz or data/*.parquet
  → IterableDataset / DataLoader
  → random shuffle buffer
  → encode board planes
  → batch
  → train step
```

You do not need a C++ chunk pipeline on day one unless you want to consume official lc0 chunk files directly.

---

## 6. RL versus SL in `lczero-training`

The new docs explicitly distinguish reinforcement learning and supervised learning modes. In RL mode, `shuffling_chunk_pool` uses a smaller sliding chunk window, and `training.schedule.chunks_per_network` is nonzero: the trainer waits for a certain number of new chunks before beginning a new epoch. In SL mode, the chunk pool should be larger than the dataset so all data is used, and `chunks_per_network` is zero so epochs repeat immediately. The docs also say RL currently uses “hanse sampling,” while SL is better served by two-stage sampling: chunk shuffling plus frame-level reservoir sampling. ([GitHub][4])

This distinction maps perfectly to tiny Leela:

| Official mode  | Meaning                                       | Tiny-Leela adaptation                           |
| -------------- | --------------------------------------------- | ----------------------------------------------- |
| RL             | Continuously train on newest self-play chunks | Later stage only; use small self-play windows.  |
| SL             | Train on a fixed dataset                      | First milestone; use teacher-labeled positions. |
| Hanse sampling | RL-focused sampling/rescoring path            | Skip at first.                                  |
| Frame sampler  | Shuffle positions from chunks                 | Keep this idea.                                 |

For a 3090-scale or hobby-scale project, **SL/distillation first** is much more efficient than pure self-play.

---

## 7. Model methodology

Classic lc0 networks are policy/value residual networks. The documented older topology uses 112 input planes, a residual tower with configurable `BLOCKS` and `FILTERS`, Squeeze-and-Excitation blocks, a policy head, a WDL value head, and a moves-left head. Typical older full-size examples were 10×128, 20×256, and 24×320. ([Leela Chess Zero][6])

The current network list shows modern large transformer nets, but it also explicitly lists “very small” human-sparring networks at ≤128 filters and ≤10 blocks, including 9×112, 5×48, and 2×16 examples. ([Leela Chess Zero][8])

For tiny Leela, I would target this family:

| Version       | Architecture                    | Purpose                                 |
| ------------- | ------------------------------- | --------------------------------------- |
| `micro`       | 16 filters × 2 residual blocks  | Debug everything; instant browser load. |
| `small`       | 48 filters × 5 residual blocks  | First fun playable model.               |
| `balanced`    | 64 filters × 6 residual blocks  | Best first serious target.              |
| `strong-tiny` | 112 filters × 9 residual blocks | Later browser/WebGPU tier.              |

The `64×6` choice is especially nice because the old official training example already uses 64 filters and 6 residual blocks. ([GitHub][2])

---

## 8. Head and loss methodology

The official head docs are useful for deciding what to keep. The vanilla policy head is trained against the `probabilities` vector from training data, i.e. the MCTS visit-count distribution, using cross-entropy / KL-style loss. There are also soft and optimistic policy variants, but they are not necessary for a toy version. ([GitHub][9])

For values, the docs describe a `winner` head predicting final WDL outcome, a `q` head predicting search value, short-term value heads, uncertainty/error heads, and a moves-left auxiliary head. The moves-left target is `plies_left`, scaled in the loss and trained with Huber loss. ([GitHub][9])

For tiny Leela, use only three heads at first:

```text
policy_head:     logits over all moves
wdl_head:        3 logits → softmax(W, D, L)
moves_left_head: optional scalar
```

A good tiny loss:

```text
L =
  1.0 * KL(pi_teacher || pi_student)
+ 1.0 * CE(wdl_teacher || wdl_student)
+ 0.1 * Huber(plies_left_teacher, plies_left_student)   # optional
+ wd * ||θ||²
```

For a pure self-play version, `pi_teacher` is the MCTS visit distribution from your own search. For a distillation version, `pi_teacher` can come from a stronger lc0 net, Stockfish multipv search, or teacher lc0 + MCTS.

---

## 9. How I would adapt it into “tiny Leela”

I would make two versions: one faithful-to-lc0 and one educational.

### Version A: faithful tiny-lc0 trainer

This version consumes official lc0-style chunks or generated compatible chunks.

```text
lc0 chunk files
  → lczero-training loader or minimal chunk reader
  → [112,8,8] planes
  → policy vector + WDL + plies_left
  → 48×5 / 64×6 SE-ResNet
  → pb.gz / ONNX export
```

Pros: closer to lc0, can reuse real training data, easier to test inside lc0.

Cons: chunk format, policy maps, and export compatibility add complexity.

### Version B: nano-Leela trainer

This version copies the methodology, not the machinery.

```text
PGN/FEN positions
  → teacher search labels
  → simple dataset: FEN, policy distribution, WDL, legal mask
  → PyTorch/JAX trainer
  → ONNX browser model
  → tiny TypeScript/Rust MCTS
```

Pros: readable, hackable, browser-first.

Cons: not drop-in compatible with official lc0 training data unless you add converters.

For your nanochat analogy, I would build **Version B first**, then add official chunk compatibility later.

---

## 10. Tiny-Leela training pipeline

A good tiny pipeline would be:

```text
1. Generate positions
   - Sample from Lichess PGNs, self-play, puzzle positions, or lc0 games.

2. Label positions
   - Teacher outputs:
     - legal move mask
     - policy distribution over legal moves
     - WDL / Q value
     - optional plies-left estimate

3. Serialize compact dataset
   - .npz, .zstd parquet, or WebDataset-style shards.
   - Keep FEN plus dense/sparse policy target.

4. Train student
   - 16×2 first, then 48×5, then 64×6.
   - Policy KL + WDL CE.
   - Mixed precision.
   - Batch 1024–4096 on a 3090.

5. Evaluate
   - Held-out policy KL.
   - Top-1/top-3 teacher move accuracy.
   - WDL calibration.
   - Engine matches at fixed search budgets.

6. Export
   - ONNX FP16 for WebGPU.
   - ONNX INT8 or WASM-friendly format for CPU.
   - Optionally lc0 pb.gz if you want to run it in lc0.
```

The official method’s important invariant is: **the network learns from search-improved policies and value targets, not just final game outcomes**. That is the part worth preserving.

---

## 11. Minimal tiny model sketch

A tiny classic lc0-like model:

```python
class TinyLeela(nn.Module):
    def __init__(self, channels=64, blocks=6, policy_size=1862):
        super().__init__()
        self.stem = ConvBNAct(112, channels, kernel_size=3)

        self.tower = nn.Sequential(*[
            SEResBlock(channels, se_channels=max(8, channels // 2))
            for _ in range(blocks)
        ])

        self.policy = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.Mish(),
            nn.Conv2d(channels, 80, 1),
            GatherPolicy(policy_size),  # fixed move map
        )

        self.value = nn.Sequential(
            nn.Conv2d(channels, 32, 1),
            nn.Mish(),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 128),
            nn.Mish(),
            nn.Linear(128, 3),
        )

        self.moves_left = nn.Sequential(
            nn.Conv2d(channels, 8, 1),
            nn.Mish(),
            nn.Flatten(),
            nn.Linear(8 * 8 * 8, 64),
            nn.Mish(),
            nn.Linear(64, 1),
        )

    def forward(self, x, legal_mask=None):
        h = self.tower(self.stem(x))
        policy_logits = self.policy(h)

        if legal_mask is not None:
            policy_logits = policy_logits.masked_fill(~legal_mask, -1e9)

        wdl_logits = self.value(h)
        moves_left = self.moves_left(h)
        return policy_logits, wdl_logits, moves_left
```

For v1, you could even remove SE and moves-left:

```text
112×8×8 input
→ 3×3 conv
→ N residual blocks
→ policy logits
→ WDL logits
```

That is enough to get the whole loop working.

---

## 12. Suggested tiny training config

Not an official valid `lczero-training` config; this is the tiny version of its ideas:

```yaml
run:
  name: tiny-leela-64x6
  seed: 1
  device: cuda

data:
  train: data/train-*.parquet
  valid: data/valid-*.parquet
  input_planes: 112
  policy_size: 1862
  shuffle_buffer: 262144
  batch_size: 2048
  num_workers: 8

model:
  type: se_resnet
  filters: 64
  residual_blocks: 6
  se_channels: 32
  policy_head: conv_gather
  value_head: wdl
  moves_left_head: true

loss:
  policy:
    type: kl
    weight: 1.0
    mask_illegal: true
    target_temperature: 1.0
  value:
    type: cross_entropy
    weight: 1.0
  moves_left:
    type: huber
    weight: 0.1
    scale: 20.0

optim:
  name: adamw
  lr: 0.001
  weight_decay: 0.0001
  grad_clip: 10.0
  warmup_steps: 1000
  schedule: cosine

train:
  steps: 100000
  eval_every: 2000
  checkpoint_every: 5000
  mixed_precision: bf16_or_fp16

export:
  onnx: exports/tiny-leela-64x6.onnx
  fp16: true
  int8_ptq: true
```

This mirrors the official methodology: batch tensors, policy/value/moves-left losses, checkpointing, metrics, and export. It removes the daemon, chunk-window waiting, hanse sampling, and production self-play scheduling.

---

## 13. What to keep versus remove

| `lczero-training` idea     | Keep for tiny Leela? | Reason                                                      |
| -------------------------- | -------------------: | ----------------------------------------------------------- |
| 112-plane board encoding   |                  Yes | Strong compatibility with lc0-style nets.                   |
| Policy distribution target |                  Yes | Core AlphaZero/lc0 idea.                                    |
| WDL value head             |                  Yes | More lc0-like than scalar value.                            |
| Moves-left head            |                Later | Useful but not essential.                                   |
| Chunk format               |             Optional | Useful if consuming lc0 data; unnecessary for toy v1.       |
| C++ data loader            |                Later | Great for scale; hurts readability.                         |
| JAX/PyTorch training loop  |                  Yes | Keep it simple and hackable.                                |
| Sliding RL chunk window    |                Later | Needed for real self-play training, not first distillation. |
| Hanse sampling             |            No for v1 | Production-specific and currently noted as slow for RL.     |
| Rescoring/tablebases       |                Later | Nice for quality; not needed for first model.               |
| Export to pb.gz            |             Optional | Needed only if running inside lc0.                          |
| Export to ONNX             |                  Yes | Best browser deployment target.                             |

---

## 14. Recommended “tiny Leela” adaptation plan

### Milestone 1: supervised student

Train a `16×2` or `32×3` net on 100k–500k teacher-labeled positions. The goal is not strength; it is to validate board encoding, policy indexing, legal masking, WDL targets, loss curves, checkpointing, and ONNX export.

### Milestone 2: real tiny model

Train a `48×5` net on 1M–5M positions. Use sparse policy targets if storing full 1862-vectors becomes annoying. Evaluate policy top-k accuracy and play fixed-search matches.

### Milestone 3: browser engine

Export `48×5` or `64×6` to ONNX. Run policy/WDL inference in browser, with a tiny PUCT loop around it. This gives the Leela feel: policy priors guide search, WDL evaluates leaves, and visit counts choose the move.

### Milestone 4: lc0-style self-play fine-tune

Generate games with your tiny engine at 32–128 playouts per move. Save search visit distributions and WDL outcomes. Fine-tune using a small sliding window, imitating the official RL-vs-SL distinction but at toy scale.

---

## 15. The core lesson

`lczero-training` is built around a very powerful abstraction:

```text
position → neural net → policy + value + auxiliary heads
search → improved policy/value targets
training → imitate the improved targets
repeat
```

For a tiny Leela, keep that abstraction, but start with this:

```text
teacher-labeled positions
→ simple shuffled dataset
→ 48×5 or 64×6 policy/WDL CNN
→ ONNX browser inference
→ small PUCT search
```

That gives you the essence of lc0 without inheriting all of lc0’s production training infrastructure.

[1]: https://github.com/LeelaChessZero/lczero-training "GitHub - LeelaChessZero/lczero-training: For code etc relating to the network training process. · GitHub"
[2]: https://raw.githubusercontent.com/LeelaChessZero/lczero-training/master/tf/configs/example.yaml "raw.githubusercontent.com"
[3]: https://github.com/LeelaChessZero/lczero-training/blob/master/docs/architecture.md "lczero-training/docs/architecture.md at master · LeelaChessZero/lczero-training · GitHub"
[4]: https://github.com/LeelaChessZero/lczero-training/tree/master/docs "lczero-training/docs at master · LeelaChessZero/lczero-training · GitHub"
[5]: https://raw.githubusercontent.com/LeelaChessZero/lczero-training/master/docs/training_tuple.md "raw.githubusercontent.com"
[6]: https://lczero.org/dev/backend/nn/ "Neural network topology - Leela Chess Zero"
[7]: https://github.com/LeelaChessZero/lczero-training/blob/master/docs/loader.md "lczero-training/docs/loader.md at master · LeelaChessZero/lczero-training · GitHub"
[8]: https://lczero.org/dev/wiki/networks/ "Networks - Leela Chess Zero"
[9]: https://github.com/LeelaChessZero/lczero-training/blob/master/docs/heads.md "lczero-training/docs/heads.md at master · LeelaChessZero/lczero-training · GitHub"
