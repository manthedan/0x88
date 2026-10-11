# CPU-engine lifecycle waste (October 2026)

## Decision and scope

Keep inexpensive per-worker Hash/Threads command deduplication for Berserk and
Stormphrax, and for Stockfish's immediate `setOptions` path. Preserve explicit
`ucinewgame` resets. Do not redesign cancellation, worker residency, Analysis
continuity, or TT policy. No engine binaries/build flags changed.

The remaining WASI loader now streams the original Response and refetches only
on streaming failure. Its module single-flight/cache and rejection eviction are
unchanged. Arena resets just the participating instances of the four previously
broadcast CPU families, once per shared instance (including mirror seats).

## Pinned-source audit

The source below was read from the shipped corresponding-source archives under
`public/<family>/`; upstream directory names match the pins. Berserk's Hash
branch was additionally checked directly at the pinned raw-source URL.

| Engine | Verified handler and consequence | Adapter decision |
| --- | --- | --- |
| Berserk `8ae895a6151695be4a50d4fb65b0c131659c513a` | [`src/uci.c:373–385`](https://github.com/jhonnold/berserk/blob/8ae895a6151695be4a50d4fb65b0c131659c513a/src/uci.c#L373) calls `TTInit(mb)` without an equality guard. [`src/transposition.c:39–66`](https://github.com/jhonnold/berserk/blob/8ae895a6151695be4a50d4fb65b0c131659c513a/src/transposition.c#L39) frees the old allocation, allocates, and calls `TTClear`. `engines/berserk/upstream.lock.json` pins this revision. `patches/berserk-emscripten.patch` preserves the Hash handler; its synchronous TT clear still clears the table. | Fix unconditional `applyOptions` before every bestMove/analyze; dedupe acknowledged normalized commands and clear that state when terminating the worker. |
| Stormphrax `582965517ed2032d41a6b4cd6c2e66b1b934e2ad` | [`src/uci/uci.cpp:155–160`](https://github.com/Ciekce/Stormphrax/blob/582965517ed2032d41a6b4cd6c2e66b1b934e2ad/src/uci/uci.cpp#L155) binds Hash to `setTtSize`. [`src/uci/option.cpp:173–186`](https://github.com/Ciekce/Stormphrax/blob/582965517ed2032d41a6b4cd6c2e66b1b934e2ad/src/uci/option.cpp#L173) invokes the callback even for equal values. [`src/ttable.cpp:86–135`](https://github.com/Ciekce/Stormphrax/blob/582965517ed2032d41a6b4cd6c2e66b1b934e2ad/src/ttable.cpp#L86) avoids equal-size reallocation but **always** marks pending initialization; `finalize` clears the TT. `patches/stormphrax-emscripten.patch` implements clear as memset and makes Threads a synchronous-build no-op. | Fix the same per-search pattern. |
| Stockfish.js 19.0.1 `0d5dd7944f88c817c07a26bc4fb1905d5f9c46d6` | [`src/engine.cpp:86–104,268–284`](https://github.com/nmrugg/stockfish.js/blob/0d5dd7944f88c817c07a26bc4fb1905d5f9c46d6/src/engine.cpp#L86): Hash calls `set_tt_size`; Threads calls `resize_threads` for pthread builds (single-thread build's Threads callback is a no-op). [`src/thread.cpp:177–192`](https://github.com/nmrugg/stockfish.js/blob/0d5dd7944f88c817c07a26bc4fb1905d5f9c46d6/src/thread.cpp#L177) destroys/recreates the pool; [`src/tt.cpp:181–211`](https://github.com/nmrugg/stockfish.js/blob/0d5dd7944f88c817c07a26bc4fb1905d5f9c46d6/src/tt.cpp#L181) unconditionally frees/allocates/clears. Pin verified in the source archive's `SOURCE_ARCHIVE.json`. | `applyOptions` is init-only, but Arena/Analysis call `setOptions({threads})` every search. Dedupe immediate Hash/Threads posts, retaining first-worker configuration, thread-before-hash order, and failure-recreation configuration. Leave Skill Level/MultiPV alone. |
| PlentyChess 8 `50ad9f1b523063cbbbf599fe0547f42870fe2906` | [`src/uci.cpp:263–267,522–525`](https://github.com/Yoshie2000/PlentyChess/blob/50ad9f1b523063cbbbf599fe0547f42870fe2906/src/uci.cpp#L263) marks options dirty, then calls the resizers on readiness. [`src/tt.h:83–95`](https://github.com/Yoshie2000/PlentyChess/blob/50ad9f1b523063cbbbf599fe0547f42870fe2906/src/tt.h#L83) returns for equal cluster count; thread resizing also guards equal count, including the synchronous patch's `workers.size()` guard. | No TT loss or costly equal resize established. Leave its adapter unchanged rather than extrapolating Berserk's defect. |
| Viridithas 20 `0631113e2215714c54d397cce9a8caebc5e6f8ee` | [`src/uci.rs:197–225`](https://github.com/cosmobobak/viridithas/blob/0631113e2215714c54d397cce9a8caebc5e6f8ee/src/uci.rs#L197) compares previous Hash/Threads and resizes only on changes. | Persistent adapter already dedupes options and clears that state in `disposeWorker`. One-shot searches instantiate a new engine and must keep their configuration. No adapter change. |
| Reckless `0010617448bdef4c8cd7d4f4825b7e42c8bc262a` | [`src/uci.rs:314–320`](https://github.com/codedeliveryservice/Reckless/blob/0010617448bdef4c8cd7d4f4825b7e42c8bc262a/src/uci.rs#L314) calls TT resize; [`src/transposition.rs:149–157`](https://github.com/codedeliveryservice/Reckless/blob/0010617448bdef4c8cd7d4f4825b7e42c8bc262a/src/transposition.rs#L149) deallocates/reallocates without equality guard. | Persistent adapter already dedupes Hash/Threads and resets the cache on disposal. Keep one-shot configuration and existing browser API behavior. No adapter change. |

Settings are cached as clamped/floored commands, not raw caller numbers. A failed
Emscripten command is not remembered. A terminated/aborted/failed worker loses
its option cache. `ucinewgame` changes TT contents, not the configured option
values, so it neither disappears nor invalidates the command cache.

## Command trace and equivalence

Built-app Chromium 149, local isolated static server port 5181. A transient
esbuild harness bundled the actual Berserk adapter at `main` and the changed
adapter, pointing both at the same shipped relaxed-SIMD artifacts. Two searches
on **one worker per variant**, depth 1, startpos then the position after d2d4:

```text
before: uci; Hash 64; Threads 1; isready;
        Hash 64; Threads 1; MultiPV 1; position <start>; go depth 1;
        Hash 64; Threads 1; MultiPV 1; position <after d4>; go depth 1
 after: uci; Hash 64; Threads 1; isready;
        MultiPV 1; position <start>; go depth 1;
        MultiPV 1; position <after d4>; go depth 1
```

Both returned d2d4, g8f6. Full worker-ID trace:
`scratchpad/cpu-engine-lifecycle-uci-traces-full.json`.

`scripts/cpu_engine_lifecycle_bench.mjs` additionally replays an eight-ply game
at depth 1, independently advancing legal bestmoves in each variant. Both
produce `d2d4 g8f6 g1f3 d7d5 c1f4 c8f5 e2e3 e7e6`, with exact
bestmove/score/nodes/PV equality at all eight positions. This is the shallow
continuation where inherited deeper TT bounds should not affect the result.
Higher-depth TT retention intentionally changes work/search ordering; it is not
general bit-parity or strength evidence.

## Warm timing A/B (benchlock)

Node v24.16.0, Darwin arm64, Apple M4. Same shipped
`berserk-emscripten-relaxed-simd128` artifact; Hash 64 MiB, Threads 1. A is the
old per-search Hash/Threads re-send, B retains options/TT between plies. Both
reset once per ten-position game continuation and use identical FENs. Module
startup, game reset, position generation and preliminary warmup are excluded;
per-search options/position/go are included. Six ABAB pairs per depth.

```sh
SP/benchlock node --experimental-strip-types scripts/cpu_engine_lifecycle_bench.mjs OUT.json 16
```

| Depth | A median, min–max (ten-search total ms) | B median, min–max | Median wall saving |
| --- | --- | --- | --- |
| 8 (short exploratory run) | 87.50, 75.00–98.69 | 70.83, 65.68–86.47 | 19.1% |
| 12 | 301.88, 241.75–446.35 | 285.89, 228.44–404.75 | 5.3% |
| 16 | 2268.88, 1973.54–2813.46 | 1673.92, 1240.40–2056.52 | 26.2% |

The shared machine was **noisy**: substantial spread and depth-12 downward
drift, despite serializing every timing run with benchlock. At depth 16 one
of six pairs was slower with dedupe. Node counts are deterministic across
all six repeats: 2,349,564 versus 1,735,250 per ten-search continuation at
depth 16 (26.1% fewer), with 60/60 bestmove agreement. Depth 12 has 54/60
bestmove agreement (one position consistently differs), explicitly not a
behavioral-parity claim for searches where the retained TT should matter.

Evidence: `scratchpad/cpu-engine-lifecycle-bench.json`,
`cpu-engine-lifecycle-bench-depth12.json`, `cpu-engine-lifecycle-bench-depth16.json`.
The tested Node workload clears the 2% wall gate; keep **only cheap dedupe**
regardless. Do not extrapolate these position-dependent results to Elo, all
browser workloads, or engines not timed here.

## WASI loader and built-product checks

- Existing browser-API no-tee pattern ported into `recklessWasiWorker.ts`.
  Streaming unavailable: compile original ArrayBuffer with one fetch.
  Streaming fails: check a second fetch's HTTP status, then compile those bytes.
  A successful load remains cached; any rejection is evicted for retry.
- Cold first Viridithas instantiation in the built Play app used the actual
  hashed WASI worker, reached persistent readiness, and replied to e2e4 with c5.
  Evidence: `scratchpad/cpu-engine-lifecycle-viridithas-cold.json`.
- Forced failure tested by importing that **built worker** in a module-worker
  wrapper. Native streaming compilation consumed the original response, then
  deliberately threw. `Response.prototype.clone` was made an error. Fallback
  refetched and returned `bestmove g1f3 ponder g8f6` at depth 2, exit 0.
  A second search returned the same move without another fetch: exactly two
  fetches total, one streaming attempt, proving the cached fallback module.
  Evidence: `scratchpad/cpu-engine-lifecycle-viridithas-fallback.json`.
- **Peak memory not measured.** No isolated process/worker transient-peak memory
  instrumentation was available in this harness. The exposed browser memory
  API provides snapshots, not the decoded-response buffering peak, and JS heap
  size is not WASM/process RSS. Do not treat the 55,739,115-byte module size as
  a measured saving. [Response.clone documentation](https://developer.mozilla.org/en-US/docs/Web/API/Response/clone)
  explains why an unread tee branch can buffer the entire body; the change
  removes that possibility, without a claimed RSS/startup percentage.
- Play Berserk: three complete engine replies at depth 6,
  `1. e4 c5 2. Nf3 Nc6 3. d4 cxd4`. Same worker, Hash/Threads sent once including
  warmup; screenshot saved and read: `scratchpad/cpu-engine-lifecycle-play.png`.
- Arena: two-game Berserk/Stormphrax match, then a six-game three-seat
  Berserk/Stormphrax/PlentyChess round robin, using a mate-in-one starting FEN
  for a deliberately short functional test. Three warmup resets plus exactly
  twelve per-game participant resets (two per game), rather than eighteen.
  Every seat played four games; the inactive third engine was not reset before
  other pairs. This tests scheduling/reset behavior, not playing strength.
  Evidence: `scratchpad/cpu-engine-lifecycle-arena-match.json`,
  `cpu-engine-lifecycle-arena-rr-full.json`, `cpu-engine-lifecycle-arena-rr-summary.json`.
  Screenshot saved and read: `scratchpad/cpu-engine-lifecycle-arena.png`.

## Validation

- `BUILD_SCOPE=product npm run build:client`: passed; static built-app checks
  above (not Vite dev wasm paths).
- `npm test`: 834 passed, 3 expected environment skips, 0 failed (837 tests).
  Included typecheck: 447 files, 0 errors, 0 warnings.
- Focused option/loader/Stockfish suites: 35 passed. Covers normalized duplicates,
  changed values, bestMove/analyze, intentional newGame, failed command retry,
  worker failure and abort/recreation; loader original-response identity,
  concurrent/sequential module caching, consumed-response fallback, HTTP and
  invalid-wasm fallback failures, rejection eviction, and streaming unavailable.
- `npx biome check` on changed source/tests/script: no errors/new issues. Four
  pre-existing unused-symbol warnings remain in `arenaBrowser.ts`; not cleaned
  up to keep the merge-local diff minimal.

No resident-memory policy, general loader rewrite, TT enlargement, or unrelated
Analysis/Play behavior change was attempted. The only product-layer shared-file
edit is the small Arena participant-reset block and its type import.
