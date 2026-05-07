# DeepSeek Methods Roadmap for Tiny Leela

Goal: adapt the useful DeepSeek-style methods to this chess project without copying irrelevant trillion-parameter LLM machinery.

The valuable ideas are mostly about:

```text
specialist teachers -> unified student
on-policy distillation
full-distribution labels
fault-tolerant rollout generation
quantization-aware deployment
training/eval determinism
```

The non-goal is to reproduce DeepSeek's MoE scale, 1M-token context, or long-context attention stack.

## Core thesis

The new Search-Light and self-play notes sharpen this roadmap: DeepSeek-style ideas are mainly training-system ideas for us, not a reason to copy large MoE architecture. The strongest plan is:

```text
multi-teacher supervised warm start
-> on-policy positions from the current student
-> search/teacher reanalysis of hard positions
-> action-value + uncertainty distillation
-> promotion-gated self-play loop
```

For tiny Leela, the most important DeepSeek-inspired idea is:

```text
Do not expect one small browser model to discover everything itself.
Use specialist teachers to label the positions the student actually visits,
then distill those specialists into one compact deployable student.
```

In chess terms:

```text
Stockfish/tactical teacher
+ lc0/strategic policy teacher
+ Maia/human-style teacher
+ tablebase/endgame teacher
+ self-play MCTS teacher
-> one CNN or SquareFormer student
```

Because chess has only legal moves instead of a huge text vocabulary, full legal-policy distillation is practical.

## Method 1: multi-teacher distillation

### Teacher roles

| Teacher | Purpose | Labels |
| --- | --- | --- |
| Stockfish | tactics, material, forcing lines | best move, centipawn/Q, WDL approximation, multipv policy |
| lc0/strong NN | strategic priors and Leela-like play | policy distribution, WDL/Q |
| Maia/human model | human-like sparring and plausible mistakes | human move distribution by rating bucket |
| Syzygy tablebase | exact endgame play | exact WDL/DTZ, perfect move set |
| Puzzle/tactical suite | forcing motifs | best moves, fail/blunder labels |
| Student MCTS | deploy-time behavior | search-improved policy, visit counts, backed-up value |

### Unified student targets

Preferred target schema per position:

```json
{
  "fen": "...",
  "history_fens": ["..."],
  "legal_policy": { "e2e4": 0.31, "d2d4": 0.24 },
  "wdl": [0.22, 0.36, 0.42],
  "q": 0.20,
  "teacher": "stockfish+lc0",
  "teacher_depth": 14,
  "source": "on_policy_v1",
  "student_net": "80x5hyb-e12ema",
  "search_visits": 64
}
```

Train with:

```text
L = policy_KL(teacher_policy, student_policy)
  + WDL_CE(teacher/result_wdl, student_wdl)
  + q_weight * MSE(teacher_q, student_q)
  + optional uncertainty/value-error loss
```

### First implementation

Start with one or two teachers only:

```text
Phase 1: Stockfish MultiPV teacher on fixed positions
Phase 2: add student-MCTS visit distribution
Phase 3: add lc0 teacher if available
Phase 4: add Maia/human labels only if we want a human-style mode
```

Avoid an overly complicated teacher router at first.

## Method 2: on-policy distillation loop

On-policy distillation means the student generates positions from its own current policy/search, then stronger teachers label those positions.

### Why it matters

Static supervised data can miss the student's actual failure states. On-policy positions expose:

- bad openings the student chooses;
- tactical traps it walks into;
- endgames it mishandles;
- high-uncertainty positions from its own search.

### Minimal OPD loop

```text
1. Pick current deployed candidate student.
2. Generate games from fixed openings using policy-only or PUCT 32/64.
3. Save every move to a rollout log.
4. Select positions for labeling:
   - high policy entropy
   - high value disagreement
   - blunders/loss swings
   - rare phase/source buckets
5. Label selected positions with Stockfish/lc0/tablebase.
6. Train the student on static supervised data + OPD labels.
7. Evaluate on fixed gates.
8. Promote only if gates improve.
```

### First OPD dataset size

Keep it small and measurable:

```text
rollout games:       1k-5k
positions labeled:   50k-250k
teacher:             Stockfish MultiPV shallow/deep mix
student update:      continuation, not from scratch
```

Only scale after the loop produces a measurable gain.

## Method 3: full legal-policy distillation

DeepSeek found full-vocabulary logit distillation more stable than sparse approximations. Chess makes this easier.

For each labeled position, store a distribution over legal moves:

```text
Stockfish MultiPV:
  convert cp scores to probabilities with temperature

MCTS:
  use visit counts as policy

lc0:
  use policy logits/probs if available

Tablebase:
  distribute over optimal moves or strongly weight fastest DTZ moves
```

Do not train only on the best move when a distribution is available. Full legal-policy targets teach move ordering and reduce brittle overfitting.

### CP-to-policy conversion

Initial formula:

```text
p(move) ∝ exp((cp(move) - cp_best) / temperature_cp)
```

Try:

```text
temperature_cp = 50, 100, 150
```

Clip extreme scores and handle mate scores separately.

## Method 4: specialist routing

Eventually, label different positions with different teachers.

Initial router:

```text
if tablebase_available(position):
  use Syzygy
elif tactical_candidate(position):
  use Stockfish deeper
elif opening_phase(position):
  use opening book + lc0/Stockfish
elif human_style_dataset:
  use Maia/human labels
else:
  use Stockfish shallow or student MCTS
```

Routing signals:

- phase: opening/middlegame/endgame;
- material count;
- legal move count;
- check/capture/promotion availability;
- student uncertainty;
- disagreement between student policy and teacher best move.

Keep the router deterministic and recorded in metadata.

## Method 5: rollout write-ahead logs

DeepSeek's fault-tolerant rollout lesson maps directly to chess self-play.

Do not discard interrupted games silently. It biases the generated set toward short or easy games.

### WAL format

After every move, append:

```json
{
  "game_id": "...",
  "ply": 37,
  "fen_before": "...",
  "move": "e2e4",
  "fen_after": "...",
  "policy": { "e2e4": 0.21 },
  "wdl": [0.30, 0.42, 0.28],
  "q": 0.02,
  "net_id": "80x5hyb-e12ema",
  "search": { "visits": 64, "cpuct": 1.5, "seed": 123 },
  "opening_id": "varied_openings_10m_dev_v1:0042"
}
```

On restart:

```text
resume from last complete FEN + RNG seed + search config
```

Record unfinished/discarded counts in reports.

## Method 6: quantization-aware deployment

DeepSeek treats deployment precision as part of training, not final packaging. For this project, use a practical browser version.

### Precision ladder

```text
v1: train fp32/bf16/fp16, export FP16 ONNX for WebGPU
v2: post-training INT8 quantization for WASM/CPU tests
v3: INT8 quantization-aware training if PTQ hurts policy/search
v4: FP8/INT4 only if WebGPU/tooling supports it cleanly
```

### Quantization gates

Before accepting a quantized model, compare against the original:

- policy KL on fixed dev positions;
- top1/top4/top8;
- bucketed regressions;
- policy-only arena;
- PUCT arena;
- browser latency.

Accept quantization only if strength loss is small relative to speed/file-size gain.

## Method 7: action-value and regret distillation

Searchless-chess/GC-style action-value labels are now a first-class DeepSeek-method target.

For selected candidate moves, train:

```text
(position, move) -> Q/WDL/value bucket/regret bucket
```

Candidate set:

```text
teacher top-k
student policy top-k
PUCT top-k
checks/captures/promotions
human played move when present
random legal distractors
moves where search overturns raw policy
```

Useful losses:

```text
action-value bucket CE
Q/WDL regression or CE
pairwise ranking loss
regret prediction: teacher_best_value - teacher_value(move)
```

Runtime use:

```text
one model pass -> policy top-k -> action-value rerank -> optional PUCT fallback
```

This is the most direct way to amortize expensive search into a tiny deployable model.

## Method 8: multi-ply auxiliary prediction

DeepSeek-style multi-token prediction maps naturally to multi-ply chess prediction.

Possible auxiliary heads:

```text
main:       next move policy
aux +1:     opponent reply policy
aux +2:     next own move policy
value:      WDL/Q after best or played move
uncertainty:value-error or teacher-disagreement prediction
```

Initial simple version:

```text
L = CE(policy_next)
  + WDL_CE(wdl)
  + 0.25 * CE(policy_reply)
```

This requires dataset rows to include the next legal reply from the same game or teacher line. It is optional after the main policy/value model is stable.

## Method 9: uncertainty and disagreement heads

For search-light browser play, a model should know when it is unsure.

Labels can come from:

- high entropy in teacher policy;
- disagreement between Stockfish and student;
- value swing after shallow vs deeper search;
- MCTS root value variance;
- tactical blunder detection.

Use uncertainty to guide adaptive search:

```text
if uncertainty high:
  spend more visits
else if policy sharp and value stable:
  move faster
```

This can improve play quality under fixed browser time budgets.

## Method 10: deterministic evaluation discipline

DeepSeek's determinism lesson is critical for chess research because Elo/search results are noisy.

Every serious comparison should fix:

- git commit;
- model artifact hash;
- dataset manifest;
- opening suite;
- random seed;
- legal move ordering;
- PUCT constants;
- visits/time budget;
- ONNX/runtime backend;
- quantization settings.

Reports should include all of the above.

## Implementation phases

### Phase A: teacher-label format and tools

- [ ] Define `legal_policy` JSONL schema.
- [ ] Add converter from Stockfish MultiPV to legal-policy distribution.
- [ ] Add tablebase label hook if Syzygy is installed.
- [ ] Add dataset report for teacher/source/phase counts.

### Phase B: shallow teacher distillation

- [ ] Sample 50k-250k positions from fixed dev/train buckets.
- [ ] Label with Stockfish MultiPV.
- [ ] Train current CNN or SquareFormer continuation with policy KL + WDL/Q.
- [ ] Compare against non-distilled baseline.

### Phase C: on-policy rollout system

- [ ] Add self-play/arena WAL writer.
- [ ] Add resume from WAL.
- [ ] Store raw policy, search visits, root Q/WDL, PV, entropy, uncertainty, net id, seed, and opening id.
- [ ] Generate 1k-5k student games from varied openings.
- [ ] Select high-value positions for teacher labeling.
- [ ] Train OPD continuation.

### Phase C2: action-value/search-light distillation

- [ ] Add candidate-move label schema: top-k teacher/student/PUCT/check/capture/random moves.
- [ ] Add action-value head to SquareFormer.
- [ ] Train action-value bucket/regret/ranking losses.
- [ ] Add top-k action-value rerank evaluator.
- [ ] Compare policy-only vs child-value rerank vs action-value rerank vs PUCT.

### Phase D: multi-teacher routing

- [ ] Add deterministic teacher router.
- [ ] Add tablebase/endgame specialist.
- [ ] Add tactical/deeper Stockfish specialist.
- [ ] Optionally add lc0 or Maia labels.
- [ ] Train unified student from mixed specialist labels.

### Phase E: deployment-aware compression

- [ ] Export FP16 ONNX and measure browser latency.
- [ ] Try INT8 PTQ.
- [ ] If PTQ loss is too high, add QAT.
- [ ] Gate quantized models with fixed policy/search tests.

## Promotion gates

An OPD/distilled model is promotable only if it improves at least one primary axis without unacceptable regressions:

```text
Primary axes:
  - fixed-node arena strength
  - policy-only strength
  - supervised teacher-policy KL/CE
  - browser latency at comparable strength

Regression guards:
  - source/phase bucket collapse
  - opening diversity collapse
  - tactical suite regression
  - quantization-induced policy drift
```

## Near-term recommended path

After the 100M dataset build is underway, the most practical DeepSeek-inspired experiment is:

```text
1. Generate 50k-100k on-policy positions from current best 80x5 hybrid EMA.
2. Label them with Stockfish MultiPV at a modest depth.
3. Convert MultiPV scores to full legal-policy targets.
4. Fine-tune the best CNN and/or SquareFormer-v1.
5. Evaluate on varied-opening policy-only and PUCT gates.
```

This is small enough to run locally and directly tests whether on-policy distillation fixes the student's own failure modes.

## What to ignore for now

- trillion-scale MoE;
- expert parallelism;
- 1M-token context;
- compressed sparse attention for a 64-token board model;
- custom kernel stacks;
- FP4 before FP16/INT8 browser deployment is solved;
- mHC-style residual streams before a deeper transformer baseline exists.

The useful DeepSeek adaptation is methodological, not architectural: build a reliable specialist-labeling and on-policy distillation pipeline around a compact chess student.
