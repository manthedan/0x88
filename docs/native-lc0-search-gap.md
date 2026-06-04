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

Our browser path now improves batch leaf collection with virtual in-flight visits and duplicate-leaf retries. The single-engine UI exposes this as `Batch / lanes` plus a leaf-collision mode: `retry` keeps virtual lane visits during batch collection to avoid duplicate in-flight leaves, while `backup` preserves the old shared-backup behavior. Full shared-tree multi-worker MCTS is still intentionally not wired into the user-facing browser UI. That keeps cancellation, ONNX/WebGPU session ownership, and UI responsiveness predictable. See [Browser-Safe Parallel MCTS Boundary](browser-safe-parallel-mcts.md) for the accepted boundary.

Our browser search is still mostly single-threaded JavaScript tree traversal. This means:

- fewer visits per second
- weaker GPU/backend utilization
- less efficient batching
- no true parallel tree exploration
- less native-style virtual-loss behavior

### Missing pieces

- worker-based parallel search
- cross-worker search coordination
- shared or message-passed tree updates
- stable performance instrumentation for visits/sec and evals/sec

## 3. NN cache and tree reuse

Native LC0 aggressively reuses work:

- neural-network evaluation cache
- search tree between moves
- transposition-aware reuse
- repeated-position reuse

We now have persistent evaluator cache support plus opt-in tree reuse in the arena, policy-only page, and worker search paths. Reuse can find compatible deeper subtrees after opponent replies.

### Missing pieces

- cache sizing/eviction controls
- richer arena-visible cache metrics
- transposition-aware reuse beyond compatible history/FEN roots

## 4. Pondering

Native LC0 can think during the opponent's turn and then continue if the opponent plays the expected move.

We intentionally do **not** implement pondering in the browser execution environment. Background search during the user's/opponent's turn would compete with UI responsiveness, ONNX/WASM/WebGPU session lifecycle, cancellation, and browser worker scheduling. For this project, tree reuse after an actual move plus optional NN-cache persistence is the safer user-facing substitute.

### Non-goals

- background search during opponent turn
- `ponderhit`-style continuation
- speculative worker searches that must be cancelled on every unexpected move
- arena UI modes that keep engines consuming compute while it is not their turn

## 5. Smarter stop conditions

Native LC0 has smarter stop behavior than a simple fixed budget.

Examples:

- stop early when the best move is overwhelmingly dominant
- keep searching when the top moves are unstable
- stop when policy/value distribution is stable
- avoid wasting time in obvious positions

Our search supports fixed visits/time by default, plus opt-in smarter stops for controlled play/search experiments:

- `earlyStop: 'root-dominance'` stops when no remaining fixed-visit budget can catch the root leader.
- `earlyStop: 'best-stable'` stops after the same best move remains ahead for repeated guarded checks.
- `earlyStop: 'kld-stable'` stops when the root visit distribution stabilizes.

These modes keep minimum-visit/check-interval safeguards and remain opt-in so native fixture parity and deterministic fixed-visit comparisons stay strict.

### Missing pieces

- tuning these thresholds against real play strength
- richer value/Q stability checks beyond the current best-move Q guard
- time-management integration that extends sharp/unstable positions
- per-position difficulty heuristics

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

The single-engine UI now exposes a broader but still curated subset: visits, soft movetime, batch size/lanes, leaf-collision mode, MultiPV, early-stop mode, CPuct, CPuct schedule, FPU strategy, FPU reduction, and final-move temperature. These settings are plumbed through both main-thread and dedicated-worker search paths.

### Missing option families

- additional CPuct variants and schedule parameters
- additional FPU options
- deeper policy temperature controls
- draw/contempt-style controls
- cache size
- browser-safe parallel/thread count controls
- backend selection beyond current query/runtime controls
- tablebase settings
- depth/clock limits beyond visits and soft movetime
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
6. **Browser-safe parallelism plan**: exploit batching/worker isolation without long-running background compute.
7. **Syzygy/tablebase plan**: probably remote/proxy first, not full browser-local tablebases.
8. **Fuller LC0 option UI/runtime controls**.

## Branching caution

Do not mix the native/custom inference pivot directly into arena UI feature work unless the interfaces are intentionally coordinated.

Search/arena changes should preserve a backend-neutral evaluator contract so the custom inference branch can land without being forced to untangle UI-specific assumptions.
