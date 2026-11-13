# LC0 Browser Search Parity Status

Status: browser-facing LC0 search parity work is complete for the current branch.

This checklist summarizes the implemented parity surface and the remaining native-engine gaps that are intentionally not part of the browser UI milestone.

## Completed browser-facing items

- **Timed search / arena time parity**
  - Soft `movetimeMs` search budget returns best-so-far instead of treating timeout as cancellation.
  - Arena supports fixed/equal movetime-style controls and warmup hooks.

- **Persistent NN evaluation cache**
  - `CachedLc0Evaluator` reuses LC0 evaluations and exposes hit/miss metrics.
  - Cache keys preserve the distinction between bare FEN and explicit history inputs.

- **Search tree reuse**
  - `Lc0PuctSearcher.search(..., { reuseTree: true })` can reuse a compatible prior root.
  - Reuse can locate deeper compatible subtrees after an opponent reply.
  - Arena, policy-only, and worker search paths can opt into reuse.

- **Batched search / browser-safe lanes**
  - `batchSize` collects multiple in-flight leaves for real evaluator batches.
  - `batchCollisionMode: 'retry' | 'backup'` controls duplicate in-flight leaf behavior.
  - Batch leaf collisions, retries, and max eval batch are exposed in search stats.

- **Smarter stop conditions**
  - Opt-in `earlyStop` modes: `root-dominance`, `best-stable`, and `kld-stable`.
  - Full fixed-budget completion remains correctly labeled when early-stop guards do not fire.

- **Fuller LC0-style option surface**
  - The policy-only UI exposes visits, soft movetime, batch size/lanes, leaf-collision mode, MultiPV, early-stop mode, CPuct, CPuct schedule, FPU strategy, FPU reduction, and final-move temperature.
  - These settings are plumbed through both main-thread and dedicated-worker search.

- **Pondering boundary**
  - Pondering is documented as intentionally out of scope for the browser execution environment.

- **Parallel MCTS boundary**
  - Browser-safe lane batching is the supported parallelism boundary.
  - True shared-tree multi-worker MCTS is documented as a future native/custom-backend capability, not a current browser UI feature.

## Final validation

Ran on `feature/lc0-search-parity` after the final docs pass:

- `npm run typecheck`
- `npm run build:client`
- `node --experimental-strip-types --test tests/lc0_search.test.mjs tests/search_metamorphic.test.mjs tests/lc0_eval_cache.test.mjs`

Result: all selected validation passed.

## Remaining native-LC0 gaps

These are intentionally left outside this browser-facing milestone:

- full clock-aware UCI time management (`wtime`, `btime`, increments, moves-to-go, panic time);
- true shared-tree multi-worker MCTS;
- pondering / `ponderhit` lifecycle;
- Syzygy tablebases;
- complete native LC0/UCI option surface;
- complete UCI engine process behavior;
- backend-specific CUDA/OpenCL/TensorRT-style performance parity.

See also:

- The consolidated native-only boundary and historical stop context below.
- Browser-safe parallel MCTS remains bounded to the current single-worker browser search model.
- [LC0 Search Parity Strictness](lc0_search_parity_strictness.md)

## Native-only boundary (consolidated 2026-11-12)

The earlier native-gap inventory compared the browser's 112-plane history,
policy/value evaluation, PUCT, visits, PV and limited MultiPV with native
Lc0. Timed search, evaluator caching, compatible subtree reuse and lane
batching are now delivered; they are not a new priority checklist.

The remaining boundaries are:

- **Clock management:** fixed soft movetime is not `wtime`/`btime`, increments,
  moves-to-go, soft/hard clock allocation, panic time or sharp-position
  extensions. A timed browser search can overrun by a work chunk.
- **Shared-tree workers:** JavaScript traversal and virtual in-flight lane
  visits are not native parallel MCTS. Cross-worker tree updates, collision
  handling, session ownership and cancellation would need a separate design.
- **Pondering:** intentionally absent. No opponent-turn background searches,
  speculative continuations or `ponderhit`; actual-move subtree reuse and
  optional persistent NN cache are the supported substitutes.
- **Reuse:** compatible history/FEN subtree reuse is not arbitrary
  transposition-aware native tree merging. Cache behavior must retain history
  identity and expose bounded hit/miss/eviction metrics.
- **Tablebases:** no Syzygy WDL/DTZ/DTM probing or tablebase adjudication.
  Remote/proxy or tiny local subsets were possible designs, not deployed ones.
- **Options/protocol:** a curated browser option surface is not a complete
  UCI command loop, external-GUI process, native `go nodes/depth/mate` protocol,
  full stop/ponder lifecycle or every CPuct/FPU/backend/pruning option.
- **Chess960:** no position generation/import, variant castling/notation or
  `UCI_Chess960` compatibility promise. Arena assumes normal chess.
- **Backend performance:** browser ONNX/TVMJS/custom WGSL runtimes do not
  promise CUDA/cuDNN/TensorRT/OpenCL/BLAS speed parity. Keep the evaluator
  interface backend-neutral rather than coupling kernel work to Arena UI.

### Historical early-stop context

The older gap inventory's section 5 listed three opt-in modes:
`root-dominance` (the leader cannot be caught within remaining visits),
`best-stable` (guarded repeated best-move stability), and `kld-stable`
(root-distribution stability). Threshold-versus-strength tuning was then a
missing measurement, not a reason to enable all three. Minimum/check guards
apply to the heuristic stability modes; root dominance has its own exact
remaining-budget condition.

[`play_early_stop_2026-10.md`](./play_early_stop_2026-10.md) supplies the later
Play measurement. Small-Lc0 Play now enables root dominance; big-net Play
remains fixed-visit. Best-stable/KLD are not promoted by that result. Richer
Q/value stability, difficulty allocation and clock extensions remain separate
work. This does not strengthen the parity claim into bit-identical native
visit distributions.
