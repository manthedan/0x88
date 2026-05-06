# Opening Bias in Current tiny_leela Training Data

## Summary

The current tiny_leela training data appears to contain significant opening-distribution bias, especially from the TCEC dataset. The most visible symptom is that the current model repeatedly steers into the same unusual openings during testing. A quick inspection confirms that the head of the TCEC source starts with fixed TCEC opening lines, including Alekhine's Defence.

This is likely not a model architecture issue alone. It is at least partly a data sampling issue: we are training from the head of ordered PGN-derived JSONL files, so the model sees a narrow slice of openings more often than intended.

## Evidence

### TCEC head starts with fixed openings

The raw TCEC PGN begins with repeated/fixed opening tags such as:

```text
B04 Alekhine's defence modern variation
B04 Alekhine's defence modern, Larsen variation
A58 Benko gambit accepted
A58 Benko gambit accepted
E10 Blumenfeld counter-gambit
E10 Queen's pawn game
E11 Bogo-Indian defence, Nimzovich variation
E11 Bogo-Indian defence, Nimzovich variation
B12 Caro-Kann advance variation
B12 Caro-Kann advance variation
```

The first row of `data/tcec_training_100k.jsonl` is already around ply 8:

```json
{"id":"tcec_pgn_000000_008","fen":"r1bqkb1r/ppp1pppp/2np4/3nP3/3P4/5N2/PPP2PPP/RNBQKB1R w KQkq - 2 5","policy":{"c2c4":1}}
```

This position is clearly from Alekhine's Defence territory.

### TCEC opening repetition is measurable

Grouping TCEC rows by game id and comparing the first sampled position showed:

- `data/tcec_training_100k.jsonl`: about 988 games
- top first sampled position: 69 games, about 7.0%
- second most common first sampled position: 37 games, about 3.7%

That is high concentration for a training set intended to teach broad chess policy.

### Lichess is less severe but still head-biased

For `data/lichess_training_400k_2000elo_2016-01.jsonl`:

- about 6,888 games
- top first sampled position: 92 games, about 1.3%
- top six-move continuation: only 7 games, about 0.1%

So Lichess is less obviously dominated by one opening, but taking the head of a PGN-derived file can still introduce ordering bias.

## Why this matters

TCEC games are high quality, but TCEC openings are not neutral. They are often chosen to force imbalance and avoid drawish engine repetition. That is good for engine tournaments, but risky for a tiny supervised policy model.

Potential harms:

1. **Opening memorization**
   - The model may learn fixed book continuations rather than general chess principles.

2. **Distorted opening prior**
   - Openings like Alekhine, Benko, or Blumenfeld may become far more common than they should be.

3. **Tiny model capacity pressure**
   - A small network has limited capacity. Repeated opening structures can occupy too much of what it learns.

4. **Misleading validation**
   - If train/dev split is row-based, adjacent positions from the same games can leak across train and dev, making metrics look better than actual generalization.

## Can fine-tuning fix it?

Probably yes. Fine-tuning means taking an existing trained checkpoint and continuing training on cleaner data with a lower learning rate.

Conceptually:

```text
biased current checkpoint
        ↓
load weights
        ↓
continue training on curated/opening-balanced data
        ↓
model policy shifts away from weird repeated openings
```

This is likely the fastest mitigation for the current model.

Recommended fine-tuning mix:

```text
80-100% shuffled high-rated Lichess positions
0-20% curated TCEC positions, preferably post-opening only
```

Useful filters:

- shuffle by game, not row
- cap positions per game
- skip early TCEC plies, e.g. first 8-16 plies
- deduplicate repeated FENs
- avoid overrepresented ECO/opening groups
- use lower LR, e.g. `1e-4` instead of `1e-3`
- run only a few epochs initially

## Important fine-tuning caveat: move vocabulary

The current board CNN trainer builds its policy output vocabulary from the training data:

```python
rows,moves=read_rows(args.train)
mid={m:i for i,m in enumerate(moves)}
```

That means changing the dataset may change the move list and output ordering. If we resume from an old checkpoint with a new move list, the policy head can become misaligned.

Before safe fine-tuning, the trainer should preserve the checkpoint's move vocabulary when `--resume` is used. New training rows whose target move is not in the old vocabulary can be skipped.

Proposed behavior:

```text
if --resume checkpoint:
  load ck['moves'] as fixed policy vocabulary
else:
  build moves from training data
```

## Can we fix the dataset and continue training epochs?

Yes, after the move-vocabulary fix. The procedure would be:

1. Build a curated dataset.
2. Resume from the current checkpoint.
3. Continue epochs on the curated dataset with a smaller learning rate.
4. Export a new student model.
5. Test both metrics and actual opening behavior.

Example shape:

```bash
python3 training/train_board_cnn.py \
  --train data/curated_finetune.jsonl \
  --out artifacts/student_board_cnn_finetuned.json \
  --resume artifacts/checkpoints/board_cnn_spatial_state_check_500k_32ch.pkl \
  --epochs 25 \
  --lr 1e-4 \
  --channels 32 \
  --state-planes
```

If the checkpoint is at epoch 20, setting `--epochs 25` continues through epochs 21-25.

## Proposed dataset fixes

### Immediate mitigation

Create a fine-tuning dataset that is mostly clean Lichess:

- random/shuffled games rather than first rows
- cap positions per game
- skip first few plies if opening behavior is too bookish
- dedupe normalized FENs
- optionally include only a small amount of TCEC after ply 16

### Long-term pipeline fix

The training data builder should avoid sequential head sampling.

Recommended changes:

1. **Shuffle by game before row extraction**
   - Do not consume PGN files sequentially.

2. **Cap rows per game**
   - Example: 8-32 positions per game.

3. **Skip early TCEC plies**
   - TCEC opening phase is often fixed book.

4. **Deduplicate normalized FENs**
   - Use board, side-to-move, castling, and en passant as the key.
   - Ignore halfmove/fullmove counters for duplicate detection.

5. **Balance by opening/ECO for TCEC**
   - Use PGN headers: `ECO`, `Opening`, `Variation`.
   - Cap rows per ECO/opening family.

6. **Split train/dev by game id**
   - Avoid leakage from adjacent positions in the same game.

7. **Report opening concentration before training**
   - top ECO counts
   - top first-position hashes
   - duplicate FEN rate
   - ply histogram
   - rows per game histogram

## Recommended next steps

1. Patch `training/train_board_cnn.py` so resume preserves checkpoint move vocabulary.
2. Build `data/curated_finetune.jsonl` from shuffled, capped, deduped games.
3. Fine-tune current checkpoint on the curated data at low learning rate.
4. Test whether opening choices become less weird.
5. Later, rebuild the full training set with proper game/opening-balanced sampling and retrain from scratch or from a clean checkpoint.
