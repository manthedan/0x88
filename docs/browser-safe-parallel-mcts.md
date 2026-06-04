# Browser-Safe Parallel MCTS Boundary

Status: accepted for the browser UI path.

This note defines what "parallel MCTS" means for the LC0 browser project and where we intentionally stop. Native LC0 can run true multi-threaded shared-tree MCTS with virtual loss, backend-specific batching, and engine-process lifecycle control. The browser has different constraints: UI responsiveness, worker startup cost, ONNX/WASM/WebGPU session ownership, cancellation latency, and unpredictable tab scheduling.

## Decision

For user-facing browser search, treat **batch/lane-based leaf collection** as the supported parallelism boundary:

- A single search owner owns the tree.
- The search owner may collect multiple in-flight leaves per iteration.
- Temporary virtual visits steer lanes away from duplicate leaves while collection is in progress.
- NN evaluation may run as a real batch through the current evaluator/backend.
- The policy-only UI exposes this as `Batch / lanes` plus `Leaf collisions`:
  - `retry`: keep temporary virtual lane visits while filling a batch, retry duplicate in-flight leaves, and prefer distinct evaluated leaves.
  - `backup`: allow duplicate leaf selections to share one evaluation and receive shared backups, preserving older behavior.
- A dedicated search worker may own the whole search to keep the main UI responsive, but it is still one tree owner.

Do **not** expose true shared-tree multi-worker MCTS as a default or promised browser feature yet.

## Why this boundary

Batch/lane search gives the main practical browser benefits without committing to fragile shared ownership:

- improves NN/backend batch utilization;
- keeps deterministic tree ownership and simpler cancellation;
- works with both main-thread and dedicated-worker search paths;
- remains backend-neutral for ONNX, WASM, WebGPU, or custom inference;
- avoids long-running background compute similar to pondering;
- avoids cross-worker tree merge races and high message-passing overhead.

True multi-worker MCTS in the browser would require careful answers to questions that are currently unsettled:

- Which worker owns the canonical tree?
- Are tree nodes copied, diffed, message-passed, or stored in shared memory?
- How are virtual loss, duplicate leaves, and backups made race-safe?
- Can cancellation stop all workers quickly enough for UI play and arena fairness?
- How many ONNX/WebGPU sessions are allowed, and who owns them?
- How do we prevent parallel workers from making a laptop/browser tab unusable?
- How do we measure strength gain separately from backend/session contention?

## Current implementation mapping

Implemented browser-safe pieces:

- `SearchOptions.batchSize` controls lanes / requested evaluator batch size.
- `SearchOptions.batchCollisionMode` selects `retry` or `backup` behavior.
- `SearchStats.batchLeafCollisions`, `batchLeafRetries`, and `maxEvalBatch` expose batch behavior.
- `Lc0PuctSearcher.search()` accepts these options.
- `src/lc0/searchWorker.ts` forwards these options to worker-owned search.
- `lc0-policy-only.html` exposes Batch / lanes and Leaf collisions.
- `docs/native-lc0-search-gap.md` tracks true multi-worker MCTS as missing rather than silently complete.

## Non-goals for the browser UI

The following are intentionally not part of the current browser-facing implementation:

- multiple workers mutating or merging one shared search tree;
- `SharedArrayBuffer`/Atomics-backed tree storage as a user-facing dependency;
- one ONNX/WebGPU session per search worker as a default mode;
- background pondering or speculative searches while it is not the engine's turn;
- thread-count controls that imply native LC0-equivalent parallel search.

## Future native/custom-backend path

A native or custom-inference path may still add true parallel MCTS later, but it should be introduced behind a separate capability boundary, for example:

- evaluator capability metrics: supported batch sizes, max in-flight evals, session sharing policy;
- search capability metrics: single-tree lanes vs shared-tree workers;
- instrumentation: visits/sec, evals/sec, batch fill rate, duplicate leaf rate, cancellation latency;
- explicit opt-in mode such as `parallelMode: 'lanes' | 'shared-tree-experimental'` rather than overloading `batchSize`.

That work should include an arena strength/performance comparison before it is treated as a parity win.

## Review checklist for future changes

Before expanding beyond lane batching, verify:

- cancellation works within a bounded latency;
- UI interactions remain responsive during search;
- one engine cannot consume compute while it is not supposed to move;
- worker/session cleanup is reliable after errors and navigation;
- metrics prove better throughput or strength, not just more resource use;
- browser-safe defaults remain conservative.
