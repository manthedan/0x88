# Optimizing Elo Evaluation for tiny_leela

## Goal

We want a practical way to estimate whether a new tiny_leela model is stronger than the current one without spending excessive time on huge match runs.

The first target should be **relative Elo**, not absolute Elo:

```text
candidate model vs current baseline
candidate model vs previous best
```

This answers the most important development question: did the change make the engine stronger?

Absolute Elo can come later by calibrating against known engines, Stockfish `UCI_Elo` levels, Maia levels, and small lc0/play-lc0 nets. Even then it should be reported as **pool Elo under a named protocol**, not as a universal human/FIDE Elo.

Engine strength is always conditional on a test card:

```text
engine A vs engine B
hardware/backend
time control or fixed node budget
opening suite
tablebase/hash/thread settings
adjudication rules
opponent pool
statistical estimator
```

So reports should say things like:

```text
nano-leela-64x6-e12ema, ONNX/WebGPU-like backend, PUCT 64 visits,
varied_openings_10m_dev_v1, reversed pairs, 5000 games,
Stockfish UCI_Elo/Maia/nano pool Elo = 1840 ±45
```

not simply:

```text
1840 Elo
```

## Basic Elo calculation

Given a match result:

```text
score = wins + 0.5 * draws
score_rate = score / games
elo_diff = -400 * log10(1 / score_rate - 1)
```

Examples:

```text
50% score -> 0 Elo
64% score -> about +100 Elo
76% score -> about +200 Elo
36% score -> about -100 Elo
```

Elo from small samples is noisy, so always report confidence or at least number of games and W/D/L.

## Size/latency efficiency is part of evaluation

Raw Elo is not enough for tiny_leela.  For browser and small-engine work, every serious comparison should also report params, ONNX bundle bytes, quantized bytes when available, and eval latency.  Use the standard efficiency doc and tool:

```text
docs/model_efficiency_metrics.md
eval/model_efficiency_report.py
```

Preferred resource metrics are incremental, not raw `Elo / byte`:

```text
EloPerByteDoubling = ΔElo / log2(model_bytes / baseline_bytes)
EloPerParamDoubling = ΔElo / log2(params / baseline_params)
EloPerLatencyDoubling = ΔElo / log2(eval_ms / baseline_eval_ms)
```

Also identify the Pareto frontier: candidates that are not simultaneously weaker, larger, and slower than another candidate under the same protocol.

## Recommended evaluation funnel

Do not send every model through a large arena. Use a staged process:

```text
1. Supervised validation metrics
2. Legal/playability checks
3. 20-game smoke arena
4. 100-game paired arena
5. SPRT / extended match only if close or promising
6. Larger calibration for release candidates only
```

This keeps iteration fast while still allowing serious testing for promising candidates.

## Stage 1: cheap validation checks

Before arena games, check:

- policy top-k metrics
- policy cross entropy
- WDL cross entropy
- legal move rate
- search does not crash
- model can produce a legal move from initial position
- opening diversity sanity if relevant

A model that fails these should not get expensive Elo testing.

## Stage 2: smoke arena

Run a very small match first:

```text
20 games = 10 opening FENs, both colors
```

Purpose:

- catch catastrophic regressions
- detect illegal move bugs
- detect obvious search/model integration problems
- get a very rough strength signal

Do not treat this as real Elo. It is only a gate.

## Stage 3: paired opening arena

Use paired openings to reduce variance and opening bias.

For each starting FEN:

```text
Game A: candidate as White, baseline as Black
Game B: baseline as White, candidate as Black
```

This cancels out unfair opening positions. If a FEN favors White, both engines get the White side once.

A reasonable first real test:

```text
100 games = 50 FENs x 2 colors
```

For stronger evidence:

```text
400 games = 200 FENs x 2 colors
1000 games = 500 FENs x 2 colors
```

## Opening suite

Do not evaluate only from the initial chess position. Use a fixed suite of diverse starting FENs.

The suite should include:

- common opening structures
- quiet middlegames
- tactical middlegames
- simple endgames
- imbalanced positions
- both sides to move

Avoid using the same biased TCEC opening head that caused training issues.

The opening suite should be versioned and reused so results are comparable across models.

## Three separate strength measurements

Use three different measurements instead of searching for one true rating.

### A. Development gating: checkpoint vs checkpoint

Primary question:

```text
candidate vs current best: did this change help?
```

Use same backend, same precision, same search budget, same openings, reversed colors, and deterministic seeds. Suggested budgets:

```text
quick smoke:        200-500 games
serious candidate:  2,000-10,000 games
release candidate: 20,000+ games if cheap enough
```

### B. Rating-list style gauntlet

Maintain a stable anchor pool:

```text
Stockfish UCI_Elo 1320
Stockfish UCI_Elo 1600
Stockfish UCI_Elo 1900
Stockfish UCI_Elo 2200
Maia 1100-1900, if available locally
small lc0/play-lc0 nets
previous tiny_leela releases
weak classical engines
```

Estimate ratings with BayesElo, Ordo, or a simple Bradley-Terry model. This gives honest pool-relative ratings with error bars.

### C. Product-facing browser levels

User levels should be calibrated separately from development Elo:

```text
Level 1: policy sampling, high temperature
Level 2: policy-only argmax
Level 3: top-4 value rerank
Level 4: 32-node PUCT
Level 5: 64-node PUCT
Level 6: 128-node PUCT
```

Publish caveats:

```text
Approximate engine-pool Elo, not FIDE Elo.
Calibrated at fixed protocol X on desktop Chrome.
Mobile strength may differ.
```

## SPRT / early stopping

Use sequential testing to avoid wasting games.

Instead of always running 1000 games, continue until the result is clearly good, clearly bad, or still uncertain.

Example hypotheses:

```text
H0: candidate is not meaningfully better, <= 0 Elo
H1: candidate is useful, >= +30 Elo
```

Stop early if:

- candidate is clearly better
- candidate is clearly worse
- max game budget is reached

This is how chess engines are commonly tested efficiently.

## Reduce variance

Arena testing should be as deterministic and reproducible as possible.

Use:

- fixed random seeds
- fixed opening FEN list
- fixed node/time limits
- fixed search parameters
- paired colors
- no temperature unless intentionally testing stochastic play
- same baseline artifact for all candidates

Lower variance means fewer games are needed to detect real improvements.

## Parallelize games

Arena games are independent, so they can be parallelized.

Recommended structure:

```text
N workers
  each worker receives assigned FEN/color pairs
  each worker writes JSONL results
aggregator
  combines W/D/L
  computes Elo and confidence
```

This reduces wall-clock time without changing the statistical meaning of the test.

## Cache and resume

Every arena run should write durable results so interrupted matches can resume.

Store:

- candidate model path/hash
- baseline model path/hash
- engine/search config
- opening suite version
- game seed
- starting FEN
- colors
- result
- final FEN or PGN if available

This avoids rerunning completed games.

## Pentanomial paired statistics

For serious paired-opening tests, track each pair as a single outcome, not just two unrelated games. For an opening pair:

```text
Game A: candidate White vs baseline Black
Game B: baseline White vs candidate Black
```

Record the candidate's total pair score as one of five outcomes:

```text
0.0, 0.5, 1.0, 1.5, 2.0
```

This pentanomial view reduces variance versus plain W/D/L when openings are deliberately imbalanced. Keep both:

```text
WDL:    total wins / draws / losses
Ptnml:  paired outcome buckets
Elo:    raw score-based Elo estimate
nElo:   optional normalized Elo later
LLR:    optional SPRT/GSPRT likelihood ratio later
```

We do not need full Fishtest immediately, but our result schema should preserve enough information to add pentanomial/SPRT analysis later.

## Current implementation

Initial anchor-arena implementation:

```text
eval/uci_anchor_arena.mjs
```

It supports:

- one ONNX tiny_leela candidate,
- Stockfish anchors through UCI `UCI_LimitStrength` / `UCI_Elo`,
- custom UCI anchors through `--uci-anchors=name|/path/to/engine|nodes`, useful for Maia via lc0 if local Maia weights are installed,
- reversed opening pairs,
- WDL, score rate, simple Elo diff ± normal-approx CI,
- pentanomial pair buckets,
- per-game JSON output including final FEN and illegal move marker.

Initial unbalanced opening suite:

```text
eval/opening_suite_uho_lite_v1.fen
```

Example Stockfish run:

```bash
node --experimental-strip-types eval/uci_anchor_arena.mjs \
  --candidate=80x5:artifacts/arena_10m_guarded/80x5_hybrid_e12_ema.onnx:artifacts/arena_10m_guarded/80x5_hybrid_e12_ema.meta.json \
  --openings-file=eval/opening_suite_uho_lite_v1.fen \
  --pairs=10 \
  --visits=32 \
  --stockfish-levels=1320,1600 \
  --stockfish-nodes=32 \
  --max-plies=100 \
  --out=artifacts/anchor_arena/80x5_vs_stockfish_uho_soft_v1.json
```

Maia v1 anchors are installed locally through lc0 wrappers:

```text
.local_engines/maia/maia-1100.pb.gz
.local_engines/maia/maia-1500.pb.gz
.local_engines/maia/maia-1900.pb.gz
.local_engines/maia/lc0-maia-1100.sh
.local_engines/maia/lc0-maia-1500.sh
.local_engines/maia/lc0-maia-1900.sh
```

Example Maia run:

```bash
node --experimental-strip-types eval/uci_anchor_arena.mjs \
  --candidate=80x5:artifacts/arena_10m_guarded/80x5_hybrid_e12_ema.onnx:artifacts/arena_10m_guarded/80x5_hybrid_e12_ema.meta.json \
  --openings-file=eval/opening_suite_uho_lite_v1.fen \
  --pairs=10 \
  --visits=32 \
  --include-stockfish=0 \
  '--uci-anchors=maia1100|.local_engines/maia/lc0-maia-1100.sh|32,maia1500|.local_engines/maia/lc0-maia-1500.sh|32,maia1900|.local_engines/maia/lc0-maia-1900.sh|32' \
  --out=artifacts/anchor_arena/80x5_vs_maia_uho_soft_v1.json
```

## Recommended tiny-engine protocol card

A release-quality report should include a reproducible card:

```text
engine:
  name: tiny_leela
  version: candidate id / git commit
  model: artifact path + sha256
  backend: onnxruntime/webgpu/wasm/node/pytorch

search:
  mode: policy-only | value-rerank | puct
  nodes_or_visits: 64
  temperature: 0
  opening_noise: false

match:
  openings: eval/opening_suite_v1.fen or UHO-lite suite
  colors: reversed_pairs
  games: 5000
  adjudication: fixed and documented
  tablebases: none unless fixed for every engine
  time_control: fixed_nodes preferred for model testing

anchors:
  - current best tiny_leela
  - Stockfish UCI_Elo levels
  - Maia levels if available
  - small lc0/play-lc0 nets if available

report:
  WDL
  pentanomial counts
  Elo ± error
  draw rate
  average move latency
  evals/sec or nodes/sec
  model size
```

For cross-hardware model comparisons, fixed nodes/visits are cleaner than wall-clock time. For product UX testing, also run wall-clock/browser-latency tests.

## Report format

A useful Elo report should include:

```text
candidate: artifacts/student_candidate.json
baseline: artifacts/student_best.json
opening_suite: eval/opening_suite_v1.fen
games: 200
W-D-L: 62-81-57
score: 51.25%
elo_diff: +8.7
confidence: low / medium / high
notes: result within noise
```

If possible, also include a confidence interval:

```text
Elo diff: +8.7 ± 45 Elo, 95% CI
```

This prevents overreacting to small noisy samples.

## Opponent ladder

For development, use a small ladder of opponents:

1. random legal mover
2. simple material evaluator
3. previous tiny_leela checkpoint
4. current best tiny_leela checkpoint
5. Stockfish `UCI_LimitStrength=true`, `UCI_Elo=1320`
6. Stockfish `UCI_Elo=1600`
7. Stockfish `UCI_Elo=1900`
8. Stockfish `UCI_Elo=2200`
9. Maia 1100-1900, if available locally
10. small lc0/play-lc0 nets, if available locally

Important Stockfish caveat: `UCI_Elo` is a deliberately weakened mode, not Stockfish's true strength and not a guaranteed FIDE-equivalent human rating. It is still a useful stable anchor if the exact time/node protocol is documented.

The primary promotion gate should usually be:

```text
candidate vs current best tiny_leela
```

Stockfish calibration is useful, but expensive and potentially demoralizing early because tiny models may be far weaker.

## Recommended promotion policy

A candidate can be promoted if:

- it passes legal/playability checks
- supervised metrics do not catastrophically regress
- smoke arena is not broken
- paired arena shows positive relative Elo with acceptable confidence
- opening behavior is not obviously pathological

For early development, we may accept noisy improvements. For release-quality promotion, require larger game counts and better confidence.

## Practical starting point

Implement this first:

```text
1. Build eval/opening_suite_v1.fen with 50 diverse FENs.
2. Add arena runner for candidate vs baseline.
3. Play each FEN twice with colors swapped.
4. Write per-game JSONL results.
5. Aggregate W/D/L, score rate, and Elo diff.
6. Start with 100-game matches.
```

Then add:

```text
- parallel workers
- resume support
- SPRT early stopping
- confidence intervals
- opponent ladder
```

## Key principle

Use the smallest reliable test that answers the current question.

For most training iterations, the question is not "what is the model's true absolute Elo?" The question is:

```text
Is this candidate likely stronger and less pathological than the current best?
```

Relative paired arena testing answers that much more efficiently.
