# Lichess PGN supervised pretraining

Tiny Leela can ingest a small curated slice of the Lichess database into the existing teacher-label JSONL format.

Download a `.pgn.zst` monthly database file from <https://database.lichess.org/> or use any local PGN file, then run:

```bash
npm run lichess:ingest -- \
  --pgn=/path/to/lichess_db_standard_rated_YYYY-MM.pgn.zst \
  --out=data/lichess_training_50k.jsonl \
  --max-games=2000 \
  --max-positions=50000 \
  --min-elo=2000 \
  --skip-plies=8 \
  --max-plies-per-game=80

python3 training/validate_teacher_labels.py data/lichess_training_50k.jsonl
python3 training/train_student.py \
  --train data/teacher_labels.jsonl data/stockfish_teacher_labels.jsonl data/lichess_training_50k.jsonl \
  --merge-fen \
  --epochs 40 \
  --primary-conv-arch 64x6 \
  --feature-cache artifacts/cache/conv_features_64x6.json \
  --out artifacts/student_lichess_50k.json
```

The generated rows are supervised imitation targets:

```json
{"fen":"...","policy":{"e2e4":1.0},"wdl":[1,0,0],"q":1,"teacher":"lichess_pgn"}
```

Recommended scale-up:

- smoke: 32-1k rows
- first useful run: 10k-50k rows
- stronger pretraining: 100k-500k rows

Use these rows to teach broad human move priors. Use Stockfish/lc0 labels later for sharper tactical correction.
