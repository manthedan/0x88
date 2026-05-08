# Public Teacher Data Sources for SquareFormer-AV-PUCT

## Purpose

This document captures the public eval/action-value/search-policy data sources that can reduce how much Stockfish labeling we must generate ourselves.

The key distinction:

```text
PositionEval:
  position -> value/WDL/PV/best move

ActionValue:
  position + move -> value/regret/mate outcome

SearchPolicy:
  position -> search visit/probability distribution over moves
```

For `SquareFormer-AV-PUCT`, `ActionValue` and `SearchPolicy` labels are more valuable than simple eval comments, because they train move ranking and candidate consequences directly.

---

## Ranked sources

| Source | Gives | Best use |
| --- | --- | --- |
| ChessBench / Searchless Chess | Stockfish legal-move/action-value labels | AV reranker, regret/ranking loss |
| Lichess position evaluations | Huge Stockfish position eval + PV database | Broad WDL/value pretraining, sparse PV policy |
| Fishtest PGNs | Stockfish-family self-play PGNs with eval/depth/time comments | Stockfish-like engine trajectory value data |
| TCEC full PGNs | Elite engine games with original engine comments | Premium engine trajectories, validation, hard positions |
| Lichess puzzles | Stockfish-reanalyzed tactical positions and solution lines | Tactical/multi-ply curriculum |
| lc0 training chunks | Visit/policy probabilities, Q/D/result targets | Leela-like search-policy/value training |
| ChessDB.cn API | Cloud eval/best moves/candidate moves by query | Gap-filling and spot checks, not bulk source |

---

## Common overlay schema

Convert all sources into one of three overlay families.

### PositionEval

```json
{
  "schema": "teacher.position_eval.v1",
  "source": "lichess_eval|tcec|fishtest|stockfish_local|...",
  "position_key": "sha256:...",
  "fen": "...",
  "teacher": "stockfish",
  "teacher_version": "...",
  "depth": 35,
  "nodes": 123456789,
  "knodes": 123456,
  "best": "e2e4",
  "q": 0.12,
  "wdl": [0.42, 0.31, 0.27],
  "eval_cp": 34,
  "mate": null,
  "pv": ["e2e4", "c7c5", "g1f3"],
  "policy": {"e2e4": 1.0},
  "quality_weight": 1.0,
  "raw": {}
}
```

### ActionValue

```json
{
  "schema": "teacher.action_value.v1",
  "source": "chessbench|chessbenchmate|stockfish_candidate|...",
  "position_key": "sha256:...",
  "fen": "...",
  "move": "e2e4",
  "teacher": "stockfish",
  "value": 0.12,
  "wdl": [0.42, 0.31, 0.27],
  "win_prob": 0.42,
  "eval_cp": 34,
  "mate": null,
  "regret_cp": 0,
  "rank": 1,
  "reasons": ["teacher_top"],
  "quality_weight": 1.0,
  "raw": {}
}
```

### SearchPolicy

```json
{
  "schema": "teacher.search_policy.v1",
  "source": "lc0_chunk|selfplay_puct|...",
  "position_key": "sha256:...",
  "fen": "...",
  "teacher": "lc0",
  "policy": {"e2e4": 0.42, "d2d4": 0.28},
  "visits": {"e2e4": 420, "d2d4": 280},
  "q": 0.08,
  "draw": 0.31,
  "result": null,
  "quality_weight": 1.0,
  "raw": {}
}
```

All overlays join to our corpus via:

```text
position_key = sha256(board + side_to_move + castling + en_passant)
action key  = position_key + uci_move
```

---

## Source-specific use

## 1. ChessBench / Searchless Chess

Priority: highest for V2.

Use for:

```text
position + move -> value/regret/rank
```

Training targets:

```text
action-value bucket CE
pairwise ranking loss
regret Huber/MSE
candidate top-k accuracy
```

Recommended ingestion:

```text
sample 100k rows -> validate move/value perspective
sample 1M rows   -> train V2 smoke
scale 5M-10M     -> serious AV training
```

Do not ingest the entire multi-terabyte action-value corpus blindly before schema validation.

## 2. Lichess position evaluations

Priority: high for broad value/PV.

Use for:

```text
position -> Stockfish eval/WDL
position -> PV first move sparse policy
```

This is not full action-value, but it can cheaply improve value calibration and tactical PV awareness.

Recommended ingestion:

```text
convert JSONL.zst rows into PositionEval
prefer highest-depth eval per FEN
use PV first move as sparse policy
weight by depth/knodes
```

## 3. Fishtest PGNs

Priority: high but parser work required.

Use for:

```text
Stockfish-family engine self-play trajectories
eval/depth/time comments
value calibration under engine-play distribution
```

Fishtest is likely more consistent than TCEC for Stockfish-family labels, but still mostly played-line comments rather than all-move action values.

## 4. TCEC full PGNs

Priority: medium/high for premium validation and elite trajectories.

Use for:

```text
engine superfinal positions
Stockfish-vs-LCZero contrast
long-horizon strategic positions
hard validation suites
```

Caveat:

```text
evals are from many engines and versions, not a uniform Stockfish teacher
comments are mostly played-line evals/PVs, not all legal move values
```

## 5. Lichess puzzles

Priority: high for tactical curriculum.

Use for:

```text
forcing-line targets
reply prediction
after-reply value
queen/material/tactical hard cases
```

Do not treat puzzle solution moves as ordinary policy distribution; use them as tactical curriculum rows.

## 6. lc0 chunks

Priority: later, high upside.

Use for:

```text
search visit distribution
Q/D/value targets
Leela-like PUCT-compatible policy/value training
```

Main work is decoding lc0 records and mapping policy indices into our from-to UCI policy space.

---

## How low-Elo data should be used

The earlier CNN decision to avoid low-Elo games was correct for pure supervised policy imitation:

```text
low-Elo played move as policy target -> teaches bad moves
```

But labeled low-Elo data is different:

```text
low-Elo position + played bad move + Stockfish/AV label saying it is bad -> useful negative example
```

For SquareFormer-AV, low-Elo games can be valuable because they contain many instructive mistakes:

```text
hanging queen/pieces
overlooking mate threats
bad captures
unsafe king moves
opening traps
natural-looking tactical blunders
```

Use rules:

```text
1. Do not train the main policy head to imitate low-Elo played moves with high weight.
2. Include low-Elo played moves as AV candidates with teacher regret labels.
3. Oversample low-Elo mistakes in tactical/AV batches, not in clean policy batches.
4. Keep elite/TCEC/engine positions as the dominant distribution for promotion gates.
5. Balance by phase/opening/material so the model does not become a low-Elo refutation specialist only.
```

Recommended weighting:

```text
elite human / engine played policy: normal policy weight
low-Elo played policy: zero or tiny policy imitation weight
low-Elo teacher value/AV/regret: normal or elevated weight
low-Elo tactical blunders: oversample in AV/tactical batches
```

So the answer is:

```text
Avoid low-Elo for behavior cloning.
Use low-Elo aggressively when bad moves are explicitly labeled as bad.
```

This is analogous to training a model on mistakes plus corrections, not merely copying the mistakes.

---

## Revised data plan

```text
Base:
  100M supervised elite/human/TCEC policy + result WDL

Broad value/PV:
  Lichess eval PositionEval
  Fishtest/TCEC PositionEval
  local Stockfish root labels only where needed

Action-value:
  ChessBench / ChessBenchmate ActionValue
  our own candidate Stockfish labels for model top-k and hard cases

Tactical:
  Lichess puzzles
  ChessBenchmate mate data
  queen-risk and model-mined failures

Search-policy:
  lc0 chunks
  later our own PUCT/self-play labels
```

Our local Stockfish labeling remains important, but should be targeted:

```text
join exact positions from our 100M registry
calibrate public sources
label our current models' top-k candidate moves
fill hard-negative gaps
produce queen-risk/refutation labels
```

---

## Immediate ingestion pipeline

Initial scripts should do three things:

```text
1. Convert public sources into canonical overlay JSONL.zst.
2. Add position_key and perspective-normalized q/WDL.
3. Produce manifest + QA reports before any training uses the data.
```

Start order:

```text
A. Lichess eval JSONL.zst -> PositionEval overlay
B. ChessBench JSONL/export sample -> ActionValue overlay
C. TCEC/Fishtest PGN comments -> PositionEval/played-line AV overlay
D. Lichess puzzles CSV -> Tactical line overlay
E. lc0 chunks -> SearchPolicy overlay
```

First acceptance criteria:

```text
100k-row smoke conversion per source
schema validation passes
policy/WDL perspective sanity checks pass
join overlap report against our position registry
manual inspection of 50 random rows
```
