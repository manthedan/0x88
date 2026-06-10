# Engine resource broker design

Status: design accepted 2026-06-09; stage 1 (broker core + threaded
Stockfish) implemented on `feature/engine-resource-broker`.

## Mission constraint

Make every engine accessible and performant in the browser with zero
download/setup friction, while staying fair and polite with the user's
machine: no fan-spinning surprise, no tab jank, no silent 180 MB downloads,
and arena results that remain meaningful as engines gain threads.

## Two structural observations

1. **Engines split into resource classes that barely contend.** LC0/BT4 are
   GPU-bound (the browser search loop spends its time waiting on GPU batch
   readback); Stockfish/Reckless/Viridithas/Berserk are pure CPU. A match of
   LC0 vs Stockfish can give both sides nearly full resources
   simultaneously. Contention is intra-class: CPU-vs-CPU matches, multi-LC0
   analysis, and memory-tier conflicts (BT4).
2. **Arena demand is naturally sequential.** Only one engine is on move at a
   time, so the arena does not need fair sharing — it needs fair
   *alternation*: an exclusive lease on the CPU budget for the engine on
   move, zero threads for the idle engine (ponder stays off for rating
   integrity anyway). True concurrent sharing only exists in multi-engine
   analysis (N engines kibitzing one position) and parallel-game datagen.

## Architecture

### Resource descriptors (engine catalog)

Each engine family carries a resource profile in the typed catalog
(`src/lc0/engineCatalog.ts`): `resourceClass: 'cpu' | 'gpu'` and
`maxThreads`. Single-threaded engines declare `maxThreads: 1`; the broker
clamps grants to the profile, so a future threaded Reckless/Berserk build
joins the pool by changing one catalog number, not the scheduler.

### EngineResourceBroker (`src/lc0/resourceBroker.ts`)

A small per-page module between the catalog and the UCI adapters.

- **Per-search leases, not mid-search resizing.** UCI engines set `Threads`
  between searches. The flow is: `acquire()` at go-time → adapter applies
  `setoption name Threads value N` → search → `release()` (or abort).
  Browser searches are short, so per-search granularity is responsive
  enough, and no engine needs dynamic resizing support.
- **Policies per surface:**
  - `exclusive` (arena): one CPU lease at a time, full budget to the holder;
    later requests queue FIFO and are abortable. GPU leases never queue.
  - `shared` (analysis): never blocks; the budget is divided over the
    registered CPU participants by weight (quantized, minimum 1, clamped to
    the engine's `maxThreads`). Pages register exactly the engines selected
    for the current run, so shares are deterministic from the first search.
- **GPU leases are non-competing.** GPU engines' CPU-side overhead lives in
  the standing 2-thread reserve, so GPU leases don't draw down the CPU
  ledger; their real arbitration is batch-quota weighting inside the shared
  evaluator service (eval-broker pattern), which is a later stage.
- **Calibrated budget, not `hardwareConcurrency`.** That number lies on
  hybrid chips (E-cores scale NNUE search poorly). The budget base is a
  measured `calibratedThreads` when available (persisted NPS-knee probe;
  stage 2), falling back to `hardwareConcurrency`. Without cross-origin
  isolation the budget is 1 (threaded WASM builds cannot run at all).
- **One user-facing performance dial** (`eco` / `balanced` / `max`),
  persisted, instead of per-engine thread spinners:
  `eco = base/2`, `balanced = base - 2`, `max = base - 1` (all floored at 1).
  Eco should be the default posture on mobile-class devices. Existing manual
  thread inputs become *caps* on top of the broker grant (0 = auto).

### Fairness rules for arena ratings

Once engines are threaded, wall-clock budgets get corrupted by background
throttling and thermals, and threaded search is nondeterministic. For
rating-quality matches prefer fixed-node/fixed-depth budgets, and keep a
`Threads=1` lane for parity-grade comparisons — the exact-parity methodology
only survives single-threaded. Threaded arena play is therefore opt-in (the
threads cap defaults to 1), while analysis defaults to auto.

### BT4 and memory tiering (later stage)

BT4 is a memory/download-tier problem, not a scheduling problem: capability
probe (`navigator.deviceMemory`, adapter limits), explicit size-labeled
opt-in, persistent OPFS/Cache API storage, never auto-selected. The broker's
only role is refusing to co-schedule it with other memory-heavy engines on
low-tier devices.

## Rollout stages

1. **Broker core + threaded Stockfish** (this branch): catalog profiles,
   broker with both policies + tests, arena exclusive leases with the
   threads input as cap (0 = auto), analysis shared leases with threaded SF
   flavors under isolation. SF ships threaded WASM already, so this stage
   needs zero engine work and validates the whole model.
2. **Calibration probe + dial UI**: one-time NPS scaling probe (1/2/4/N
   threads, knee detection) cached per device; surface the eco/balanced/max
   dial; visibility-change downshift.
3. **Berserk threads** (Emscripten pthreads already proven), then Reckless
   via the staged Emscripten route
   (`reckless_wasm_next_exploration_notes.md`) — each is a catalog edit once
   the engine ships a threaded build.
4. **GPU eval-broker consolidation**: one evaluator service per model with
   weighted batch quotas for multi-LC0 analysis; BT4 tiered gate rides
   alongside.

## Non-goals

- OS-level priorities or mid-search thread rebalancing (browsers offer
  neither; per-search leases approximate both well enough).
- Cross-tab arbitration (Web Locks / SharedWorker) — deferred until a real
  multi-tab use case appears.
- Replacing the game loop: the arena already serializes turns; the broker
  makes the budget explicit and hands the mover a real thread count.
