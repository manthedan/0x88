# LC0-Inspired Tiny Leela Roadmap

## Summary

The current tiny Leela has reached the point where architecture and training-target changes matter more than simply adding epochs. The spatial policy head was a major breakthrough, confirming that chess policy needs square-aware features. The next phase should borrow the core ideas from `lczero-training` while keeping the project small, readable, browser-friendly, and Rust/TypeScript deployable.

The goal is **not** to copy LC0's production training infrastructure. The goal is to adapt its strongest ideas:

```text
position → neural net policy/value
search → improved policy/value targets
training → imitate improved targets
repeat
```

## Current Baseline

Current best deployed model family:

```text
22 input planes
→ 3 conv layers, 32 channels
→ spatial policy head
→ WDL head
```

Recent important improvements:

- Trainable board CNN beat frozen random features decisively.
- Spatial policy head greatly improved move priors.
- Added castling, en-passant, side-to-move, constant, and explicit check planes.
- Trained/fine-tuned on mixed Lichess/TCEC data.
- Added balanced/shuffled/capped fine-tune data to reduce opening bias.
- Web frontend currently uses Rust backend with high search visits.

Remaining limitations:

- History planes are now specified and available in the Python data/training path, but TS/Rust runtime inference still needs history-aware state plumbing before deployment.
- Data-derived move vocabulary is deprecated by the fixed policy-map path, though legacy artifacts still support it.
- Policy targets are mostly one-hot played moves, not search-improved distributions.
- Deeper residual tower training path exists in Python, but deployed TS/Rust runtime intentionally rejects those artifacts until ONNX/runtime support lands.
- WDL/value quality is weak compared with policy quality.
- No moves-left/progress auxiliary head.
- Dataset split/sampling still needs more rigorous game-level and opening-level validation.

## LC0 Ideas Worth Keeping

### Keep

| LC0 idea | Tiny Leela adaptation |
|---|---|
| Spatial board input | Keep and expand to LC0-style history planes. |
| Fixed policy map | Replace data-derived move vocab with fixed move indexing. |
| Policy + WDL heads | Keep as core outputs. |
| Search-improved policy targets | Add teacher/MCTS soft policy distributions. |
| Residual tower | Python trainer supports a `residual_tower`; use 48×5 as first nano-LC0 target. |
| Moves-left head | Add as auxiliary progress/conversion signal. |
| Shuffled frame sampling | Keep via simple Python dataset builder/shuffle buffer. |
| Checkpoint/export loop | Keep simple checkpointing and JSON/ONNX export. |

### Defer or Avoid for Now

| LC0 idea | Reason to defer |
|---|---|
| Full production C++ loader | Too much complexity for current scale. |
| Daemon/TUI training system | Unnecessary for first nano implementation. |
| Official chunk compatibility | Useful later, but not needed for PGN/teacher pipeline. |
| Hanse sampling | Production RL detail; skip for supervised stage. |
| Tablebase rescoring | Valuable later for endgames. |
| LC0 pb.gz export | Optional unless we want to run inside lc0. |

## Input Plane Spec v2: Nano-LC0 History Planes

History support is intentionally smaller than full LC0 at first but follows the same principle: give the net recent position context so it can infer repetition, castling/en-passant history effects, and move momentum.

Current Python training layout with `--history-plies N`:

```text
12 current piece planes: P N B R Q K p n b r q k
12 × N previous-position piece planes, newest first
state planes:
  legacy mode: side-to-move scalar plane, constant plane
  --state-planes mode: side-to-move, castling K/Q/k/q, en-passant square,
                       constant, white-to-move, stm-in-check, opponent-in-check
```

Plane count:

```text
without --state-planes: 12 × (N + 1) + 2
with --state-planes:    12 × (N + 1) + 10
```

Missing history at game starts is zero-filled. Dataset rows may carry `history_fens`; if absent, the trainer still works and zero-fills history planes. Runtime TS/Rust inference must not deploy a history-trained artifact until it passes the same history stack to the evaluator.

## ONNX Export Path

PyTorch is now the primary training path for nano-LC0 residual models. The legacy tinygrad trainer remains only for older JSON artifacts and reference experiments.

Install the primary training environment:

```bash
python3 -m venv .venv-onnx
.venv-onnx/bin/pip install -r requirements-onnx.txt
```

Train the residual tower directly in PyTorch and export ONNX in one step:

```bash
.venv-onnx/bin/python training/train_residual_torch.py \
  --train data/balanced_train.jsonl \
  --dev data/balanced_dev.jsonl \
  --out artifacts/residual_48x5.pt \
  --onnx-out artifacts/residual_48x5.onnx \
  --channels 48 --blocks 5 \
  --history-plies 2 --state-planes \
  --compile
```

Use `--compile` on PyTorch 2.x systems where `torch.compile` is stable for the selected device; omit it if compile overhead dominates small smoke runs.

## Roadmap

## Phase 1: Stabilize Current Model Path

Purpose: make sure the current generation remains usable while larger changes begin.

Tasks:

1. Export the latest stopped checkpoint cleanly if needed.
2. Keep `artifacts/student_distill_benchmark.json` pointing to the best playable model.
3. Preserve Rust/TS inference compatibility.
4. Keep balanced data builder and resume-vocab fix.
5. Add a short model card/log for current deployed model:
   - training source
   - planes
   - architecture
   - epochs/LR
   - known biases
   - validation commands

## Phase 2: Fixed Policy Vocabulary

Purpose: remove fragile data-derived output shapes.

Current problem:

```python
moves = sorted(set(moves_in_dataset))
```

This causes:

- missing rare promotion moves,
- checkpoint/fine-tune incompatibility,
- model artifacts with different policy orderings,
- awkward deployment logic.

Target:

```text
fixed policy_size = 1858 or 1862
fixed move → index mapping
fixed index → move mapping
```

Implementation options:

1. Adopt LC0/AlphaZero-style policy map directly.
2. Or define our own stable UCI move map covering:
   - all normal from-to moves,
   - all promotions,
   - castling represented as king moves.

Acceptance criteria:

- Trainer always uses fixed policy size.
- Rust and TS use the same mapping.
- Existing legal move masking works.
- Promotion moves are always representable.
- Checkpoint resume no longer depends on dataset move list.

## Phase 3: History Planes

Purpose: give the network temporal context like LC0.

Current input:

```text
current board only + state/check planes
```

Target input family:

```text
previous N board states + state planes
```

A tiny version can start with:

```text
8 positions × 12 piece planes = 96 planes
+ side-to-move/state/check/repetition-ish planes
≈ 104-112 planes
```

Dataset impact:

- Training rows need enough game context to encode previous positions.
- Existing single-FEN rows can still be used by filling missing history with zeros or repeating current board, but full benefit requires game-sequence rows.

Implementation plan:

1. Extend dataset builder to emit history FENs or compact previous board placements.
2. Add `--history-plies` or `--history-positions` to trainer.
3. Add TS/Rust plane encoders for inference.
4. During live play, maintain board history in frontend/backend state.

Acceptance criteria:

- Training supports history planes.
- Inference supports history planes.
- Existing no-history artifacts remain loadable.
- Start-position and short self-play validation pass.

## Phase 4: LC0-Like Residual Tower

Purpose: replace the tiny 3-conv net with a real small residual CNN.

Current:

```text
Conv
Residual Conv
Residual Conv
```

Target model sizes:

| Name | Shape | Purpose |
|---|---|---|
| micro | 16 filters × 2 blocks | Debug/export smoke. |
| small | 48 filters × 5 blocks | First serious browser model. |
| balanced | 64 filters × 6 blocks | Stronger model if speed allows. |

Recommended first serious target:

```text
48 filters × 5 residual blocks
policy + WDL heads
fixed policy map
history-capable input
```

Possible later additions:

- Squeeze-and-Excitation blocks.
- Mish/SiLU activations.
- BatchNorm or normalization if supported cleanly.

Acceptance criteria:

- Trainable with checkpointing.
- Exportable to current JSON path or ONNX.
- Rust/TS inference supports it or uses ONNX runtime path.
- Search speed remains usable in browser/Rust backend.

## Phase 5: Better Policy Targets

Purpose: stop training only on one-hot played moves.

Current target:

```text
policy = one-hot actual move
```

LC0-style target:

```text
policy = search visit distribution over legal moves
```

Tiny adaptation options:

1. **Stockfish MultiPV labels**
   - Run Stockfish on sampled positions.
   - Convert top-N moves/evals to soft policy distribution.

2. **Tiny engine self-search labels**
   - Run our PUCT with current model.
   - Save visit counts as policy targets.

3. **Hybrid labels**
   - Use played move one-hot when no teacher available.
   - Use teacher soft distribution when available.

4. **Legal smoothing**
   - Small probability mass spread over legal moves to reduce overconfidence.

Acceptance criteria:

- Dataset supports sparse multi-move policy targets.
- Trainer uses KL/CE against soft policy distribution.
- Metrics report policy KL plus top-k.
- Model policy becomes less brittle and less book-memorized.

## Phase 6: Moves-Left / Progress Head

Purpose: give the net an auxiliary signal for game progress and conversion.

Target head:

```text
moves_left_head: scalar plies remaining
```

Loss:

```text
0.05-0.1 * Huber(predicted_plies_left / scale, target / scale)
```

Why useful:

- Helps distinguish shuffle-draws from converting positions.
- Gives endgame/progress structure.
- Mirrors LC0 auxiliary methodology.

Data source:

- For PGN rows, `plies_left = game_length - current_ply`.
- For self-play rows, same.
- For adjudicated/truncated rows, mark lower confidence or skip moves-left.

Acceptance criteria:

- Dataset has plies-left target.
- Trainer can enable/disable head.
- Artifact exports head metadata.
- Inference can ignore head if not needed.

## Phase 7: Better Dataset Quality and Sampling

Purpose: keep TCEC quality while avoiding opening-book distortion.

Keep TCEC, but sample better:

- Shuffle by game.
- Cap rows per game.
- Cap rows per ECO/opening family.
- Deduplicate normalized FENs.
- Split train/dev by game, not row.
- Report opening concentration.
- Report ply histogram.
- Report result balance.
- Report material/phase distribution.

Important metrics:

```text
top ECO share
top first-position share
duplicate FEN rate
rows per game histogram
ply bucket histogram
train/dev game leakage = 0
policy unknown move rate
promotion row count
check row count
```

Acceptance criteria:

- Balanced builder becomes default for large supervised runs.
- Dev split is game-level.
- Training reports dataset diagnostics before epoch 1.

## Phase 8: ONNX / Browser Deployment Path

Purpose: avoid hand-writing every larger net in TS/Rust.

Current hand-coded inference works for tiny custom JSON artifacts, but LC0-like nets will get more complex.

Target:

```text
PyTorch/JAX/tinygrad model
→ ONNX export
→ ONNX Runtime Web / WASM / WebGPU
→ browser + Rust backend parity
```

Short-term compromise:

- Keep JSON artifacts for current tiny CNN.
- Add ONNX path for residual/history models.

Acceptance criteria:

- Browser loads ONNX model.
- Rust backend can either use JSON path or ONNX path.
- Policy/WDL parity tests exist for several FENs.

## Phase 9: Tiny Self-Play / RL Fine-Tune

Purpose: imitate LC0's loop at toy scale.

Loop:

```text
current net
→ PUCT self-play at fixed visits
→ save visit-count policy + WDL outcome + plies-left
→ train/fine-tune on recent window
→ evaluate
→ promote if stronger
```

Start tiny:

```text
32-128 visits
small game batches
sliding window of recent games
arena against previous model
```

Acceptance criteria:

- Self-play rows include soft visit distribution.
- Arena can compare model checkpoints.
- Promotion decision uses fixed-match results plus sanity metrics.

## Suggested Immediate Implementation Order

1. **Fixed policy map**
2. **Game-level balanced dataset split/reporting**
3. **History planes**
4. **48×5 residual model**
5. **Moves-left head**
6. **Soft teacher/search policy targets**
7. **ONNX export/runtime path**
8. **Self-play fine-tune loop**

## Near-Term Training Plan

Once fixed policy and history-capable input are ready:

```text
micro smoke:
  16×2, 50k rows, 1-3 epochs

small validation:
  48×5, 300k balanced rows, 5-10 epochs

serious supervised:
  48×5 or 64×6, 1M+ balanced/teacher rows
```

Use staged learning rates:

```text
start: 1e-3 for fresh training if stable
fine-tune: 1e-4
anneal: 5e-5 → 2e-5 → 1e-5
```

Stop or anneal when:

- train/dev loss slope flattens,
- policy KL stops improving,
- qualitative FE play regresses,
- self-play/arena results plateau,
- opening distribution becomes too concentrated again.

## Core Principle

Do not optimize only for supervised top-k. The model should improve on:

```text
policy quality
value calibration
legal/tactical correctness
opening diversity
conversion/progress
search strength
browser playability
```

The next phase is therefore not just “more epochs.” It is a shift from a good tiny supervised CNN toward a **nano-LC0 training stack**: fixed policy map, history planes, residual tower, better targets, and eventually search-improved self-play data.
