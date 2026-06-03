# Native LC0 Search Feature Gap

This note tracks what our browser LC0 search currently lacks compared with a full native LC0 engine/search stack.

Our browser search has the core shape:

- LC0-style 112-plane history input
- policy/value evaluation
- PUCT search
- fixed visit budgets
- principal variation output
- limited MultiPV support
- browser arena integration

But native LC0 is much richer and much more optimized.

## 1. Real time management

Native LC0 can manage real chess clocks, including:

- remaining white/black time
- increment
- moves to go
- soft and hard time limits
- critical-position time allocation
- early stop when the best move is stable
- extra time in uncertain/tactical positions

We currently mostly use fixed visits. A time-parity arena mode would add fixed `movetime`, but that is still simpler than full clock-aware time management.

### Missing pieces

- `wtime`, `btime`, `winc`, `binc`, `movestogo` style budgeting
- soft/hard deadlines
- position-difficulty heuristics
- best-move stability based early stop
- panic/extension time for sharp positions

## 2. Parallel MCTS

Native LC0 performs multi-threaded search with virtual loss/collision handling and batched neural-network evaluation.

Our browser search is mostly single-threaded JavaScript search. This means:

- fewer visits per second
- weaker GPU/backend utilization
- less efficient batching
- no true parallel tree exploration
- less native-style virtual-loss behavior

### Missing pieces

- worker-based parallel search
- virtual loss / in-flight node accounting
- batched leaf collection across workers
- shared or message-passed tree updates
- stable performance instrumentation for visits/sec and evals/sec

## 3. NN cache and tree reuse

Native LC0 aggressively reuses work:

- neural-network evaluation cache
- search tree between moves
- transposition-aware reuse
- repeated-position reuse

We have some experimental reusable-root and transposition hooks, but the arena does not yet exploit them like native LC0.

### Missing pieces

- persistent per-engine NN cache across moves/games
- search tree reuse after making a move
- sibling/opponent-move subtree preservation
- cache sizing/eviction controls
- arena-visible cache metrics

## 4. Pondering

Native LC0 can think during the opponent's turn and then continue if the opponent plays the expected move.

We currently do not ponder. Each engine starts thinking only when it is its turn.

### Missing pieces

- background search during opponent turn
- expected reply tracking
- `ponderhit`-style continuation
- safe cancellation when opponent plays a different move
- arena UI indication that an engine is pondering

## 5. Smarter stop conditions

Native LC0 has smarter stop behavior than a simple fixed budget.

Examples:

- stop early when the best move is overwhelmingly dominant
- keep searching when the top moves are unstable
- stop when policy/value distribution is stable
- avoid wasting time in obvious positions

Our search mostly stops at fixed visits today.

### Missing pieces

- best-move stability checks
- root visit dominance thresholds
- value/Q stability checks
- KLD/policy-stability stop tuned for play strength
- minimum-search safeguards before early stop

## 6. Tablebases

Native LC0 can use Syzygy tablebases for perfect endgame play.

We currently rely on chess rules plus network/search only.

### Missing pieces

- Syzygy probing
- tablebase win/draw/loss adjudication
- tablebase DTZ/DTM-informed move selection
- browser-feasible tablebase strategy, likely remote/proxy or tiny local subsets

## 7. Full native LC0 option surface

Native LC0 exposes many UCI options for search behavior and backend performance.

We expose only a small subset in code/UI.

### Missing option families

- CPuct variants and schedules
- FPU options
- policy temperature controls
- draw/contempt-style controls
- cache size
- minibatch size
- thread count
- backend selection
- MultiPV controls
- tablebase settings
- nodes/depth/visits/movetime/clock limits
- optional search heuristics and pruning settings

## 8. Backend performance

Native LC0 can use optimized native inference backends such as CUDA, cuDNN, TensorRT, OpenCL, BLAS, and other platform-specific paths.

Our browser path uses ONNX/browser runtimes, which are much easier to demo but generally slower and less controllable.

The custom inference pivot being worked on separately is therefore strategically important. Search improvements should avoid coupling too tightly to the current ONNX runtime so we can swap in the custom backend cleanly.

### Missing pieces

- backend-neutral evaluator interface with stable performance metrics
- optimized batching pipeline
- lower-overhead tensor preparation
- lower-overhead policy decoding
- runtime/backend selection UI or config
- repeatable backend benchmark suite

## 9. Chess960 / variant robustness

Native LC0 has broader support for engine protocol behavior and Chess960-style castling handling.

Our arena currently assumes normal chess.

### Missing pieces

- Chess960 position generation/import
- Chess960 castling legality and notation checks
- UCI `UCI_Chess960` style compatibility if we expose engine protocol mode

## 10. Production-grade UCI behavior

Native LC0 is a complete UCI engine.

We are not yet a full UCI engine. We do not fully support:

- complete UCI command loop
- `go wtime btime winc binc movestogo`
- `go nodes`, `go depth`, `go mate`, full `go movetime`
- `stop`/`ponderhit` lifecycle
- stable `info` output
- external GUI integration
- robust engine process lifecycle

## Suggested priority order

For our current project direction, the most impactful native-LC0-like upgrades are:

1. **Timed search / time parity**: add soft `movetime` budgets and arena equal-time mode.
2. **Backend-neutral search/evaluator boundary**: protect the custom inference pivot from arena/search churn.
3. **NN cache and tree reuse across moves**: large practical strength/speed improvement.
4. **Batched search improvements**: better use of browser/custom inference throughput.
5. **Smart early stopping**: spend less time in obvious positions and more in unclear ones.
6. **Pondering**: especially useful for arena games if we can manage cancellation safely.
7. **Syzygy/tablebase plan**: probably remote/proxy first, not full browser-local tablebases.
8. **Fuller LC0 option UI/runtime controls**.

## Branching caution

Do not mix the native/custom inference pivot directly into arena UI feature work unless the interfaces are intentionally coordinated.

Search/arena changes should preserve a backend-neutral evaluator contract so the custom inference branch can land without being forced to untangle UI-specific assumptions.
