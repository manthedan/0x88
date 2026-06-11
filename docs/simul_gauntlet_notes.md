# Simul gauntlet: one hero engine vs N boards at once

2026-06-10 design note. Idea: run the arena's gauntlet mode as a *simul* —
the hero engine plays all N opponents concurrently on an N-board grid
(one-million-chessboards energy), with the hero's search work shared across
games where possible.

## What it actually gets us (ranked by certainty)

### 1. Pipeline utilization — guaranteed, engine-agnostic, the sleeper win

In a serial gauntlet the hero idles every time an opponent thinks. In a
simul, while opponent *i* thinks on board *i*, the hero thinks on board *j*:
the hero approaches 100% utilization and gauntlet wall-clock drops by up to
~2× at equal budgets (more when opponents are slow relative to the hero).
This needs **zero engine changes** — it is pure scheduling: the resource
broker grants the hero a standing lease, opponents think concurrently on
their own boards, and a scheduler hands the hero whichever board is ready.
This alone makes validation gauntlets (e.g. "is the new build stronger than
the pool?") meaningfully faster.

### 2. PUCT engines: cache + batch sharing — strong, and research-relevant

- **Shared NN eval cache across games already exists**: the arena's
  `CachedLc0Evaluator` is shared across seats today. Gauntlet games from the
  same opening hit the cache hard in the early/middle game; a simul
  concentrates those hits in time instead of spreading them across a serial
  run.
- **Cross-game batch filling attacks our known bottleneck.** The TVMJS/LC0
  visit loop is GPU-bound with *underfilled batches* (see memory/bench
  notes: "tune kernels + batch fill first"). N concurrent searches can pool
  leaf evaluations into shared GPU batches — the simul broker is exactly the
  stage-4 "GPU eval-broker with batch quotas" from
  `engine_resource_broker_design.md`, with games as the consumers. This
  could raise effective LC0 NPS in multi-game settings, not just reuse work.
- Search-tree reuse across games (shared transposition/eval state beyond the
  NN cache) is also available in-process since our PUCT searcher exposes a
  transposition table keyed by FEN+history; cross-game value is real early,
  decaying as games diverge.

### 3. CPU/alpha-beta engines: shared TT via one interleaved process — real but decaying

One hero process (e.g. a single Stockfish/Reckless instance) playing N games
interleaved shares its hash table naturally: TT entries are zobrist-keyed,
so positions reached in any game probe the same table — transpositions
*across games* hit for free. From a common starting position the early-game
overlap is large; reuse decays as games diverge. History/butterfly
heuristics similarly bleed across games (same character as TT sharing).
Mechanically this is what our persistent adapters already do — interleave
`position`/`go` on one process *without* `ucinewgame` between board
switches; repetition/50-move state travels in the `position` movelist, so
correctness holds. Quantifiable via time-to-depth and hashfull on early
moves, serial vs interleaved. Expect a modest, front-loaded gain — the
honest framing is "warm cache", not "shared search".

Once threaded engines land, the broker can also split the hero's thread
grant across boards or focus it on the on-move board — the simul becomes the
natural consumer of the threading work.

### 4. Spectacle — cheap and real

An N-board grid (chessground instances + mini eval bars + one standings
table) is mostly layout work; the tournament core already produces the
schedule and standings. Per-board charts would be too heavy; one
hero-utilization meter + per-board eval bars carries the show.

## Fairness rules (important)

- Simul-hero gets reuse/utilization advantages opponents don't; results are
  comparable *between simul runs* but not silently with serial gauntlets —
  tag PGNs/standings `simul`.
- Time budgets get noisy under contention (N searches sharing CPU/GPU): use
  fixed depth/nodes/visits budgets in simul mode, consistent with the
  existing fairness doctrine (movetime for casual play, fixed budgets for
  ratings).

## Sketch

1. Scheduler: `simul: true` on gauntlet mode → all pairings' games start
   concurrently (one game per opponent at a time); hero moves are serialized
   through a queue ordered by boards-ready; opponents think in parallel
   under shared-policy broker leases.
2. Hero engine instance is shared across boards (already true — `engines`
   map keys by engineId); CPU heroes skip `ucinewgame` between board
   switches to keep the TT warm; LC0 heroes share the eval cache (already
   true) and, later, pool leaf batches through the eval broker.
3. UI: board grid with per-board eval bars + result badges; standings table
   reused; hero utilization stat (thinking time / wall time).
4. Metrics to validate the caching story: NN cache hit-rate serial vs simul
   (LC0), early-move time-to-depth serial vs interleaved (SF/Reckless).

## Verdict

Worth building — not for the cache wins alone (real but front-loaded), but
because the three things it needs (concurrent scheduling, shared eval
broker, N-board rendering) are each things we want anyway, and the payoffs
stack: ~2× faster validation gauntlets for free, a measurable shot at the
LC0 batch-fill problem, and the best demo page this project could have.
