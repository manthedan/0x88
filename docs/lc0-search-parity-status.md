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

- [Native LC0 Search Feature Gap](native-lc0-search-gap.md)
- [Browser-Safe Parallel MCTS Boundary](browser-safe-parallel-mcts.md)
- [LC0 Search Parity Strictness](lc0_search_parity_strictness.md)
