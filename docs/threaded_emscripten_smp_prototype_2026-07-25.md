# Threaded Emscripten SMP prototype — Stormphrax (2026-07-25)

**Status: research prototype. Nothing here is shipped and nothing in `src/`, `scripts/` (other
than one new build script), `public/`, or `package.json` was modified.**

## The claim under test

A performance audit asserted that every CPU engine except Stockfish is pinned to
`maxThreads: 1` (`src/lc0/engineCatalog.ts`), that the Emscripten builds hard-disable pthreads
(`-s USE_PTHREADS=0`) and force a synchronous search (`-DPLENTY_SYNC_SEARCH`, `-DSP_SYNC_SEARCH`),
and that reverting this would yield **roughly 3–5x search throughput**, "dwarfing the +14% to +64%
SIMD gains already banked".

The prototype below builds the threaded engine for real and measures it.

**Verdict in one line: the throughput claim is essentially correct (~4x NPS at 8 threads), and it
is also almost entirely worthless — the same wall-clock budget buys ~0 extra plies of search.
Do not ship this on the strength of the NPS number.**

## 1. Engine choice: Stormphrax

Stormphrax over PlentyChess, for three reasons:

1. **The sync-search patch is a clean `#ifdef` overlay.** Every `SP_SYNC_SEARCH` site in
   `patches/stormphrax-emscripten.patch` is a plain `#ifdef SP_SYNC_SEARCH … #else … #endif`
   wrapper around code upstream already had. Simply not defining the macro restores upstream's
   real barrier-driven Lazy SMP verbatim — no patch surgery.
2. **No native pre-processing step.** `build_plentychess_emscripten.mjs` must `make -C tools` and
   run a native `process_net` binary before compiling; Stormphrax compiles straight from source.
3. **Less thread surface to revert.** PlentyChess additionally needs `-DTB_NO_THREADS` reverted
   for Fathom, and its non-sync `Threads::resize()` contains a bare spin-wait
   (`while (startedThreads < numThreads) {}`), which is a genuine deadlock risk under Emscripten.
   Stormphrax's threaded path uses only condition-variable barriers (`src/util/barrier.h`).

Pinned upstream: `Ciekce/Stormphrax` @ `582965517ed2032d41a6b4cd6c2e66b1b934e2ad` (8.0.0),
network `undertown.nnue`. Toolchain: `em++` 5.0.7-git (Homebrew), Node v24.16.0.

## 2. What was built

New file: **`scripts/build_stormphrax_threaded_emscripten.mjs`** — a fork of the shipped
`scripts/build_stormphrax_emscripten.mjs`. Diffs versus the shipped build, and nothing else:

| Shipped | Threaded prototype |
| --- | --- |
| `-DSP_SYNC_SEARCH` | *(omitted — upstream Lazy SMP runs)* |
| `-s USE_PTHREADS=0` | `-pthread -s USE_PTHREADS=1` |
| — | `-s PTHREAD_POOL_SIZE=12` |
| — | `-s PTHREAD_POOL_SIZE_STRICT=0` |
| — | `-s DEFAULT_PTHREAD_STACK_SIZE=8388608` |

Everything else is byte-identical: `-O2`, `-DSP_WASM_SIMD`, `-msimd128 -mssse3 -msse4.1`,
`INITIAL_MEMORY=536870912`, `MAXIMUM_MEMORY=2147483648`, `STACK_SIZE=67108864`,
`ALLOW_MEMORY_GROWTH=1`, `EXIT_RUNTIME=0`, the same exports and the same
`--preload-file undertown.nnue@/undertown.nnue`.

### Memory settings were NOT changed

`ALLOW_MEMORY_GROWTH=1` was deliberately kept despite the expected
`-Wpthreads-mem-growth` warning ("may run non-wasm code slowly"). It works — growable
`SharedArrayBuffer` is supported by the toolchain and Node 24 — and keeping it means the A/B is
honest. `INITIAL_MEMORY` is 512 MiB, so growth never actually triggered during any measurement.
`DEFAULT_PTHREAD_STACK_SIZE` is a genuinely new setting, but it only sizes helper-thread stacks
(Emscripten's `-sSTACK_SIZE` covers the main thread only, and the 64 KiB default cannot hold
Stormphrax's recursive search frames). It does not exist in the single-threaded build, so it
cannot skew the comparison.

### One unplanned fix was required

The first threaded build compiled but died at startup:

```
NetworkLoader: Unaligned pointer
Failed to load default network
No network loaded
```

This is **not** a threading bug. `patches/stormphrax-emscripten.patch` reads the NNUE file into a
plain `std::vector<std::byte>`, and `NetworkLoader::get()` (`src/eval/nnue/loader.cpp:41`) rejects
any pointer not aligned to `util::simd::kAlignment` (16 B). In the single-threaded build dlmalloc
*happens* to hand back a 16 B-aligned block for that ~53 MB allocation; under `dlmalloc-mt` the
preceding pthread bookkeeping shifts it and the check fires. The shipped browser build is relying
on luck here — worth knowing independently of this experiment.

Rather than fork the tracked patch, the new build script applies a post-patch source fixup that
swaps the vector for an interface-compatible 64 B-overaligned shim (`resize`/`data`/`size` call
sites compile unchanged; no NNUE compute path is touched). It is fenced behind an anchor check
that throws if the patch layout ever changes.

**The build succeeded.** No further intervention was needed.

## 3. Measurement setup

Harness: `scratchpad/smp/bench_sp.mjs`, modelled on `scripts/stormphrax_emscripten_smoke.mjs` —
same `ccall('command', …)` UCI surface, same `print`-callback line capture. **A browser was not
required**; Node runs the pthread-enabled module directly (the main thread may block on
`Atomics.wait`, and the prewarmed `PTHREAD_POOL_SIZE` pool means `pthread_create` never has to
proxy to a blocked thread). One `.js`-extension caveat: the pthread build resolves its worker
script from its own filename, so the harness must not rename the glue to `.cjs`.

Five positions, deliberately mixed in character:

| Name | FEN |
| --- | --- |
| `startpos` | `startpos` |
| `kiwipete` | `r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1` |
| `italian` | `r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 2 3` |
| `endgame-rook` | `8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1` |
| `lasker-reichhelm` | `8/k7/3p4/p2P1p2/P2P1P2/8/8/K7 w - - 0 1` |

Host: Apple Silicon, **4 performance + 6 efficiency cores** (10 logical). This matters: the
theoretical ceiling is not 10x, it is roughly 5–6x aggregate throughput, and thread 0 is not
guaranteed a P-core.

**Honesty caveat on host conditions:** other agents were active on this machine throughout;
external load ranged from ~1 to ~2 cores. All sweeps were therefore run **interleaved** (config
order rotates each repeat) and reported as **medians**, so drift cannot systematically favour one
configuration. Baseline and `thr-T1` agree to within ~1% in every run, which is the built-in
control for this. High-thread-count numbers are, if anything, slightly pessimistic.

## 4. Results

### 4a. Raw throughput (NPS) — the claim's own metric

Median derived NPS, `go movetime 1500`, Hash = 64 MiB, 5 interleaved repeats.

| Position | baseline (sync) | thr T=1 | T=2 | T=4 | T=6 | T=8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| startpos | 594,008 | 623,834 | 1,210,577 | 2,101,471 | 2,204,993 | 2,567,729 |
| kiwipete | 619,720 | 626,759 | 1,192,800 | 1,905,468 | 2,095,610 | 2,478,787 |
| italian | 632,432 | 633,433 | 1,149,405 | 1,756,531 | 2,073,997 | 2,497,249 |
| endgame-rook | 1,114,707 | 1,113,808 | 2,069,603 | 3,388,137 | 4,110,377 | 4,894,904 |
| lasker-reichhelm | 1,949,308 | 1,945,171 | 3,866,961 | 4,934,406 | 5,991,860 | 7,661,165 |

Speedup versus the shipped single-thread build:

| Position | T=1 | T=2 | T=4 | T=6 | T=8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| startpos | 1.05x | 2.04x | 3.54x | 3.71x | 4.32x |
| kiwipete | 1.01x | 1.92x | 3.08x | 3.38x | 4.00x |
| italian | 1.00x | 1.82x | 2.78x | 3.28x | 3.95x |
| endgame-rook | 1.00x | 1.86x | 3.04x | 3.69x | 4.39x |
| lasker-reichhelm | 1.00x | 1.98x | 2.53x | 3.07x | 3.93x |

A quieter sequential run (`movetime 2000`, best of 2) reached higher figures — T=4 at
3.35–3.70x, T=8 at 4.14–4.92x, T=10 at 3.96–5.79x — consistent with contention explaining most of
the gap between the two runs.

Three things to note:

- **`thr-T1` == baseline (within 1%).** Compiling with pthreads and running upstream's threaded
  search on one thread costs nothing. The overhead story is clean.
- **Scaling saturates around 4–4.4x**, matching a 4P+6E core budget. It does not approach 10x.
- **On raw NPS the audit's "3–5x" is correct.**

### 4b. Time to a fixed depth — the metric that actually converts to strength

Median wall-clock milliseconds to complete `go depth 18` (lower is better), 5 interleaved repeats.
The depth limit is main-thread-only (`src/search.cpp`: `if (depth >= m_maxDepth …)` is guarded by
`thread.isMainThread()`), so this is a valid time-to-depth.

Hash = 64 MiB:

| Position | baseline | thr T=1 | T=2 | T=4 | T=6 | T=8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| startpos | 803 | 752 | 504 | 596 | 512 | 887 |
| kiwipete | 2,098 | 2,092 | **2,944** | **2,918** | **3,058** | **3,426** |
| italian | 1,483 | 1,481 | 1,057 | 1,102 | 1,457 | 1,462 |
| endgame-rook | 1,425 | 1,427 | 1,226 | 1,192 | 1,635 | 1,163 |

On kiwipete, more threads made the engine **monotonically slower to depth**. Total nodes consumed
to reach the same depth 18 scale almost linearly with thread count — i.e. the extra work is
essentially all wasted:

| Position | baseline | T=2 | T=4 | T=6 | T=8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| startpos | 474,058 | 627,189 | 1,403,328 | 1,294,795 | 2,646,034 |
| kiwipete | 1,316,674 | 3,522,229 | 6,104,996 | 7,494,185 | 9,586,139 |
| italian | 935,081 | 1,251,779 | 2,330,541 | 3,455,624 | 3,895,897 |
| endgame-rook | 1,576,964 | 2,857,156 | 4,870,632 | 5,094,845 | 5,843,332 |

The obvious suspect is transposition-table pressure: N threads pour N times the nodes into the
same 64 MiB table. Re-running with Hash = 256 MiB (3 repeats) confirms it is *part* of the story —
time-to-depth does improve, to roughly **1.4–2.0x**:

| Position | baseline | thr T=1 | T=2 | T=4 | T=6 | T=8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| startpos | 833 | 771 | 403 | 605 | 520 | 478 |
| kiwipete | 4,189 | 4,074 | 3,779 | 2,301 | 2,402 | 3,681 |
| italian | 2,115 | 2,116 | 1,216 | 1,395 | 1,056 | 1,471 |

So: **~4x nodes/second becomes ~1.5–2x time-to-depth at best, and only if the TT is resized for
the thread count.** That is the well-known Lazy SMP discount, and it is where most of the headline
number evaporates.

### 4c. Depth reached in a fixed time budget — the metric the product actually uses

This is the decisive table. The app gives engines a movetime; what matters is how deep they get.
Median depth over 5 interleaved repeats, `go movetime 1500`.

Hash = 64 MiB:

| Position | baseline | thr T=1 | T=2 | T=4 | T=6 | T=8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| startpos | 21 | 21 | 21 | 20 | 21 | 20 |
| kiwipete | 17 | 17 | 17 | 16 | 15 | 15 |
| italian | 18 | 18 | 19 | 18 | 17 | 17 |
| endgame-rook | 20 | 20 | 17 | 19 | 18 | 20 |
| lasker-reichhelm | 17 | 17 | 18 | 17 | 16 | 16 |

Hash = 256 MiB (to rule out the TT-pressure explanation):

| Position | baseline | thr T=1 | T=2 | T=4 | T=6 | T=8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| startpos | 21 | 21 | 20 | 22 | 21 | 20 |
| kiwipete | 17 | 17 | 15 | 16 | 15 | 16 |
| italian | 18 | 18 | 19 | 19 | 18 | 18 |
| endgame-rook | 19 | 19 | 19 | 19 | 20 | **24** |
| lasker-reichhelm | 17 | 17 | 17 | 17 | 17 | 17 |

Corresponding NPS in that same 256 MiB run rises exactly as before (T=8 reaches 2.66–7.70 M nps,
~4.3x baseline).

**A 4.3x increase in nodes per second buys zero additional plies, and on the tactical middlegame
position (kiwipete) it costs one to two plies.** The one real gain is `endgame-rook` at T=8
(19 → 24), which is consistent with endgames being the regime where wide shallow parallel search
pays off. Everywhere else the extra cores produce duplicated work.

### 4d. Correctness

Threaded builds returning garbage would invalidate any speed win, so this was checked directly.

**Tactical suite** (6 Win-At-Chess positions with unambiguous solutions, 3 trials each,
`movetime 1000`):

| Build | Solved |
| --- | --- |
| baseline (sync search) | 18/18 |
| threaded T=1 | 18/18 |
| threaded T=2 | 18/18 |
| threaded T=4 | 17/18 |
| threaded T=8 | 18/18 |

The single T=4 miss was WAC.002 (played `b3b7` instead of `b3b2`) — a legal, non-losing move, well
within normal 1-second engine variance, not a race artifact.

**Analysis lifecycle** (`scratchpad/smp/stoptest.mjs`): 10 cycles of `go infinite` → sleep →
`stop` → `bestmove`, with the `Threads` option toggled between 4 and 8 *between* cycles. All 10
returned a sane bestmove, `_isSearching()` reported correctly during each search, and there were
no hangs, no lost `bestmove`, and no deadlock. The stop handshake is sound.

**The project's own gate passes.** `scripts/stormphrax_emscripten_smoke.mjs --depth 2
--expect-start g1f3 --expect-fen e1g1` succeeds unmodified against the threaded artifact.

**Two real correctness concerns remain:**

1. **Determinism is gone.** At T≥2 the same position at the same movetime returns different
   bestmoves across runs (kiwipete alternates `e2a6`/`d5e6`; italian alternates `c2c3`/`d2d4`).
   That is inherent to Lazy SMP, not a bug — but every fixed-expectation smoke gate in this repo
   that pins a bestmove would become flaky the moment threads are enabled by default.
2. **One unexplained outlier.** A single T=8 run on `lasker-reichhelm` reported
   `score cp 9664` after 252 M nodes / 29.5 s. That is below Stormphrax's decisive-score threshold
   (`kScoreWin = 25000`, `src/core.h:708`) so it is a legal evaluation rather than corrupted
   memory, and it is plausibly WDL-normalisation scaling near a proven win — but it did not occur
   at any other thread count and was not reproduced. A threaded ship would want a score-sanity
   assertion.

### 4e. Artifact size

| Artifact | Single-thread | Threaded | Delta |
| --- | ---: | ---: | ---: |
| `.wasm` raw | 697,245 | 791,627 | +94,382 (+13.5%) |
| `.wasm` brotli | 201,259 | 218,459 | +17,200 (+8.5%) |
| `.js` raw | 70,611 | 87,336 | +16,725 (+23.7%) |
| `.js` brotli | 17,709 | 21,155 | +3,446 (+19.5%) |
| `.data` (network) | 55,914,848 | 55,914,848 | 0 |

Because the 53 MB preloaded network dominates, the **real download increase is about +0.04%**.
Artifact size is a non-issue.

## 5. What shipping this would actually require

Ordered by how much work each item really is:

1. **Fix the worker bootstrap — this is the blocker.** `src/lc0/stormphraxEngine.ts` builds its
   worker from a Blob URL and pulls the glue in with `importScripts`. The threaded glue resolves
   its own script URL via `_scriptName = self.location.href` in a worker context, which under that
   scheme is the *outer harness blob*, not the Emscripten glue. Spawning pthread workers would
   load the wrong script. Fixing it means either passing `Module.mainScriptUrlOrBlob`, or
   restructuring so the glue is the worker's own entry script. **This has not been tested in a
   browser at all** — every number above is from Node, which resolves `_scriptName` to
   `__filename` and therefore sidesteps the problem entirely. Browser validation is mandatory
   before any of this is believed.
2. **Nested-worker + COOP/COEP validation.** Cross-origin isolation is already correct in
   production (`same-origin` + `require-corp` on https://0x88.app), so `SharedArrayBuffer` is
   available. But the engine module is instantiated *inside* a Web Worker, so pthread workers are
   nested workers; that needs verifying per browser, Safari especially.
3. **Search-completion semantics change.** In sync-search mode `ccall('command', 'go …')` blocks
   until the search finishes; threaded, it returns as soon as the setup barrier clears and the
   search continues asynchronously. `stormphraxEngine.ts` already waits on the `bestmove` stdout
   line rather than on `commandDone`, so this mostly absorbs cleanly — but the `runExclusive`
   queue and the `stop` path would need an audit against `_isSearching()`.
4. **Catalog + broker wiring.** `resource: { resourceClass: 'cpu', maxThreads: 1 }` becomes
   `maxThreads: N` in `src/lc0/engineCatalog.ts`; `src/lc0/resourceBroker.ts` already hands out
   per-engine budgets and needs no structural change. `stormphraxEngine.ts` already has a
   `threads` option and emits `setoption name Threads`. This part is genuinely easy.
5. **Hash must scale with threads.** Section 4b shows 64 MiB is actively harmful at T≥2. Any
   thread budget handed out by the broker has to come with a matching Hash increase, which is new
   memory pressure on a page that already runs LC0 and ORT.
6. **Test-gate rework.** Bestmove-pinned smoke gates need to become threads-1 gates, or move to
   legality/score-sanity assertions (see 4d).
7. **A second artifact, or a swap.** Shipping threaded *and* single-threaded means two builds
   (thread support cannot be feature-detected inside one wasm module). Given `.data` is shared
   and the wasm delta is 17 KB brotli, either is affordable.

## 6. Recommendation

**Do not pursue this for Stormphrax as a throughput win. The 3–5x figure is real but it is a
nodes-per-second number, and nodes per second is not the product metric.**

The measured chain is:

- raw NPS: **~2.0x @ 2 threads, ~3.0–3.5x @ 4, ~4.0–4.4x @ 8** — claim confirmed
- time-to-depth (with an enlarged TT): **~1.4–2.0x** — claim substantially weakened
- depth reached in a fixed time budget: **~0 plies, negative on tactical middlegames** — claim
  refuted for the metric that matters

For comparison, the SIMD work this repo has already banked (+14% Viridithas, +64% PlentyChess,
+19–24% Reckless) is a *real* per-node speedup that converts to depth essentially one-for-one. A
+64% NPS win from SIMD is worth strictly more than a +330% NPS win from Lazy SMP. The audit's
"dwarfing the SIMD gains" framing inverts the actual value ordering.

Set against that, shipping costs a browser worker-bootstrap rewrite (item 1, untested), nested
worker validation across browsers, a determinism loss that breaks existing gates, and materially
higher memory use for TT sizing.

Narrower opportunities that the data does support, if anyone wants to revisit:

- **Endgames specifically.** `endgame-rook` was the one position where threads paid (19 → 24 plies
  at T=8 with a 256 MiB hash) and it showed the best NPS scaling throughout. A threads-on-in-
  endgames-only policy is the only variant of this idea the measurements actually endorse.
- **Fix the alignment fragility regardless.** The `NetworkLoader: Unaligned pointer` failure in
  §2 means the *shipped* single-threaded build depends on an incidental dlmalloc alignment. An
  allocator change or an emcc upgrade could break it. That is worth a real patch independent of
  threading.
- **The 4P+6E ceiling is a genuine constraint**, not a measurement artifact. Even a perfect SMP
  implementation cannot exceed ~5–6x here, and thread 0 landing on an efficiency core is a real
  source of the depth regressions at T≥6.

## 7. Files created by this investigation

| Path | Purpose |
| --- | --- |
| `scripts/build_stormphrax_threaded_emscripten.mjs` | New pthread-enabled build (not wired into `package.json`; defaults its output to `.local_engines/`) |
| `docs/threaded_emscripten_smp_prototype_2026-07-25.md` | This document |

Scratch-only (not in the repo): `scratchpad/smp/bench_sp.mjs`, `sweep.mjs`, `sweep2.mjs`,
`tactics.mjs`, `stoptest.mjs`, and the raw sweep JSON. Built artifacts live under the scratch
directory and `.local_engines/`; nothing was written to `public/`.

### Reproducing

```sh
# baseline, into scratch (does not touch public/)
STORMPHRAX_EMSCRIPTEN_JS_OUT=/tmp/sp/baseline/stormphrax-emscripten.js \
  node scripts/build_stormphrax_emscripten.mjs

# threaded
STORMPHRAX_EMSCRIPTEN_JS_OUT=/tmp/sp/threaded/stormphrax-emscripten-threaded.js \
  node scripts/build_stormphrax_threaded_emscripten.mjs
```

Both build scripts share `.local_engines/stormphrax-emscripten-src` and each re-clones and
re-patches it, so they must be run one after the other, never concurrently.
