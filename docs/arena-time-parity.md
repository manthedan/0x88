# Arena Time-Parity Mode

Goal: add an arena mode where engines are given equal wall-clock time per move instead of fixed LC0 visits / fixed Stockfish depth.

## Motivation

The current arena presets are fixed-strength shortcuts:

- LC0 search engines use fixed visits, e.g. 100 or 400 visits.
- Stockfish Lite uses fixed depth, e.g. depth 4 or depth 8.

This is useful for repeatable demos, but it is not a fair resource comparison. A time-parity mode should let both engines think for the same amount of wall-clock time per move.

Caveat: equal time does **not** mean equal strength. Stockfish Lite will likely benefit more from extra time because it searches many more nodes per second, while browser LC0 pays ONNX inference and JavaScript MCTS overhead. Equal time is still the right “fair resource” mode; competitive balance can later use per-engine time multipliers or handicaps.

## UI

Add an arena budget selector:

```text
Search budget:
  - Fixed strength presets
  - Equal movetime

Move time: [1000] ms
```

In fixed mode, current behavior remains unchanged.

In equal-movetime mode:

- LC0 search engines ignore their fixed visit presets except as a safety cap.
- Stockfish Lite ignores fixed depth and uses UCI `go movetime N`.
- Policy-only LC0 can stay instant, or optionally be excluded/marked as not time-searchable.

## Engine interface

Current arena engine interface:

```ts
interface ArenaEngine {
  id: string;
  name: string;
  move(positions: BoardState[], signal: AbortSignal): Promise<string | null>;
}
```

Proposed interface:

```ts
interface ArenaMoveContext {
  signal: AbortSignal;
  movetimeMs?: number;
}

interface ArenaEngine {
  id: string;
  name: string;
  move(positions: BoardState[], ctx: ArenaMoveContext): Promise<string | null>;
}
```

The arena loop reads the selected budget once per tournament/game and calls:

```ts
await engine.move(historyBoards, {
  signal,
  movetimeMs: selectedBudgetMs,
});
```

## Stockfish implementation

`src/lc0/stockfishEngine.ts` already supports `movetimeMs`:

```ts
export interface StockfishOptions {
  depth?: number;
  movetimeMs?: number;
  skillLevel?: number;
}
```

and `stockfishGoCommand()` already emits:

```ts
go movetime N
```

So the arena wrapper changes from fixed depth:

```ts
stockfish!.setOptions({ depth });
return stockfish!.bestMove(boardToFen(positions[positions.length - 1]), signal);
```

to timed mode:

```ts
stockfish!.setOptions({ movetimeMs });
return stockfish!.bestMove(boardToFen(positions[positions.length - 1]), signal);
```

Fixed mode should continue to use depth and must explicitly clear any prior timed setting. `StockfishEngine.setOptions()` merges options, and `stockfishGoCommand()` lets a positive `movetimeMs` override `depth`, so switching back to fixed mode must do something like:

```ts
stockfish!.setOptions({ depth, movetimeMs: undefined });
```

or replace the options object instead of merging stale timed state.

## LC0 implementation

Add a soft time budget to `SearchOptions` in `src/search/puct.ts`:

```ts
export interface SearchOptions {
  visits?: number;
  movetimeMs?: number;
  // existing options...
}
```

The search loop should stop when either the visit cap or time budget is exhausted.

Important: time expiry should be a **soft stop**, not an abort. `AbortSignal` currently means tournament cancellation and can throw. Movetime expiry should return the best move found so far.

Conceptually:

```ts
const deadline = options.movetimeMs && options.movetimeMs > 0
  ? nowMs() + options.movetimeMs
  : Infinity;

while (completed < visitsToRun && nowMs() < deadline) {
  throwIfAborted(signal);
  await simulate(root, evaluator, searchPolicy, context, stats, options.transpositionTable);
  completed += 1;
}
```

For batched search, check the deadline between chunks/batches. Avoid checking every tiny operation if it hurts performance, but always stop reasonably close to the requested time.

Timed LC0 calls should use a large visit safety cap:

```ts
await searcher.search(
  { positions },
  {
    visits: 1_000_000,
    movetimeMs,
    signal,
    yieldEveryMs: 16,
  },
);
```

The result should expose enough stats to show realized work:

- requested movetime
- elapsed search time
- completed visits
- eval calls / batch eval calls
- stop reason, e.g. `time-budget`

## Warm-up

Warm up both engines before timed games so startup cost does not contaminate the first move.

Suggested warm-up:

- LC0: run one tiny eval/search after model load.
- Stockfish: initialize worker, send `isready`, and optionally run one dummy `go movetime 1`.

Without warm-up, the first move can include ONNX/WebGPU/WASM initialization and unfairly skew results.

## Optional future: time odds / engine multipliers

Equal wall-clock time is useful, but Stockfish Lite will likely gain more from extra time than browser LC0.

Add optional engine-specific multipliers later:

```text
LC0 time multiplier: 1.0x
Stockfish time multiplier: 0.25x
```

Then effective movetime becomes:

```ts
engineMovetimeMs = baseMovetimeMs * engine.timeMultiplier;
```

This supports both fair-resource testing and competitive-balance matches.

## Implementation checklist

1. Add `movetimeMs?: number` to `SearchOptions`.
2. Add deadline/soft-stop handling to fixed-visit, batched, and neural-budget search loops.
3. Add `time-budget` to search stop reasons/stats.
4. Change arena engine interface to accept `ArenaMoveContext`.
5. Add budget controls to `lc0-arena.html`.
6. In arena fixed mode, preserve existing LC0 visits / Stockfish depth presets.
7. In arena timed mode, pass `movetimeMs` to LC0 and Stockfish.
8. Warm up LC0 and Stockfish before the first timed tournament game.
9. Add tests for LC0 timed search returning a legal move before full visit cap.
10. Add browser smoke test for timed arena mode.
