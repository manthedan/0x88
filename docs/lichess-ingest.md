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

# Precompute frozen-conv features with the Rust CPU cache builder. This avoids
# recomputing 64x6 features inside the Python training loop.
cargo build --release --quiet --manifest-path rust/tiny_leela_core/Cargo.toml --bin tiny-leela-rust-feature-cache
rust/tiny_leela_core/target/release/tiny-leela-rust-feature-cache \
  --arch=64x6 \
  --out=artifacts/cache/conv_features_64x6_lichess_50k.json \
  --inputs=data/teacher_labels.jsonl,data/stockfish_teacher_labels.jsonl,data/lichess_training_50k.jsonl

# For 10k+ rows, tinygrad/CUDA is currently much more practical than the pure
# Python trainer for the linear heads. The CUDA_HOME/PATH setup is needed when
# using the local pip CUDA toolchain.
CUDA_HOME=$PWD/.venv-tinygrad/lib/python3.12/site-packages/nvidia/cu13 \
PATH=$PWD/.venv-tinygrad/lib/python3.12/site-packages/nvidia/cu13/bin:$PATH \
LD_LIBRARY_PATH=$PWD/.venv-tinygrad/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-} \
.venv-tinygrad/bin/python training/train_student.py \
  --trainer tinygrad \
  --train data/teacher_labels.jsonl data/stockfish_teacher_labels.jsonl data/lichess_training_50k.jsonl \
  --merge-fen \
  --epochs 40 \
  --primary-conv-arch 64x6 \
  --feature-cache artifacts/cache/conv_features_64x6_lichess_50k.json \
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

## Local benchmark notes

On the 2013-01 Lichess shard, ingestion produced 50k rows from 894 accepted games with zero parse failures. Rust CPU feature-cache precompute for 50k unique-ish 64x6 positions took about 422s. A 10k-row cache took about 86s, and tinygrad/CUDA trained 40 epochs over the cached 10k set in about 141s. The pure Python trainer did not finish even 5 epochs over the cached 10k set within 600s, so use tinygrad for Lichess-scale training.

An Unsloth-style tensorized dataset experiment is available:

```bash
python3 training/build_feature_dataset.py \
  --train data/teacher_labels.jsonl data/stockfish_teacher_labels.jsonl data/lichess_training_10k.jsonl \
  --merge-fen \
  --feature-cache artifacts/cache/conv_features_64x6_lichess_10k.json \
  --out artifacts/datasets/lichess_10k_64x6.pkl

CUDA_HOME=$PWD/.venv-tinygrad/lib/python3.12/site-packages/nvidia/cu13 \
PATH=$PWD/.venv-tinygrad/lib/python3.12/site-packages/nvidia/cu13/bin:$PATH \
LD_LIBRARY_PATH=$PWD/.venv-tinygrad/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-} \
.venv-tinygrad/bin/python training/train_feature_dataset.py \
  --dataset artifacts/datasets/lichess_10k_64x6.pkl \
  --epochs 40 \
  --batch-size 512 \
  --out artifacts/student_lichess_10k_tensor.json
```

Local result: building the 10k tensor dataset worked, but the first minibatch trainer was slower (about 485s for 40 epochs) than the existing full-batch tinygrad trainer (about 141s). This means the useful lesson is not merely “pickle tensors”; we need persistent device tensors / compiled full-batch or larger fused batches, plus a binary matrix format, to get Unsloth-like speedups.
