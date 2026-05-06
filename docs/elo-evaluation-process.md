# Optimizing Elo Evaluation for tiny_leela

## Goal

We want a practical way to estimate whether a new tiny_leela model is stronger than the current one without spending excessive time on huge match runs.

The first target should be **relative Elo**, not absolute Elo:

```text
candidate model vs current baseline
candidate model vs previous best
```

This answers the most important development question: did the change make the engine stronger?

Absolute Elo can come later by calibrating against known engines or Stockfish levels.

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
5. weak Stockfish level / low node limit
6. stronger Stockfish level / higher node limit

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
