# Teacher Engine Setup for Distillation

Tiny Leela distillation needs teacher labels generated offline from one or more strong engines.

## Sources checked

- Lc0 getting started: https://lczero.org/dev/wiki/getting-started/
- Lc0 downloads: https://lczero.org/play/download/
- Lc0 networks: https://lczero.org/play/networks/
- Lc0 best nets: https://lczero.org/play/networks/bestnets/
- Stockfish downloads: https://stockfishchess.org/download/
- Stockfish UCI docs: https://official-stockfish.github.io/docs/stockfish-wiki/UCI-&-Commands.html

## Lc0

Lc0 setup has two separate artifacts:

1. `lc0` binary/source engine.
2. A neural-network weights file, usually `.pb.gz`.

The lc0 repo gives the engine source, but the teacher net is downloaded separately from LCZero network pages.

Typical setup:

```bash
# Option A: package/release binary if available for your platform.
# Option B: build https://github.com/LeelaChessZero/lc0 using its current build docs.

# Download a compatible net from one of:
#   https://lczero.org/play/networks/
#   https://lczero.org/play/networks/bestnets/

export LC0_BIN=/absolute/path/to/lc0
export LC0_WEIGHTS=/absolute/path/to/weights.pb.gz
npm run distill:check
npm run distill:seed
npm run distill:query -- --nodes 64 --multipv 4
npm run distill:validate -- data/teacher_labels.jsonl
```

The query script talks UCI and sets `WeightsFile`, so it does not depend on lc0 internals.

## Stockfish fallback / second teacher

Stockfish is useful as a second teacher because it is easy to install and gives strong tactical labels. It is not Leela-like, so keep its dataset separate from lc0 labels.

On Ubuntu Noble, `apt-cache policy stockfish` reports package candidate `16-1build1` in `universe`.

```bash
sudo apt update
sudo apt install stockfish
export STOCKFISH_BIN=/usr/games/stockfish  # or wherever installed
npm run stockfish:check
npm run distill:seed
npm run stockfish:query -- --depth 10 --multipv 4
npm run distill:validate -- data/stockfish_teacher_labels.jsonl
```

## Dataset policy

Do not mix teachers silently. Use separate dataset files:

- `data/teacher_labels.jsonl` for lc0.
- `data/stockfish_teacher_labels.jsonl` for Stockfish.
- A future combined dataset must record per-row `teacher` and have a new dataset id.
