# Reckless WASM optimization notes

## Current build and runtime guide

As of 2026-11-12, the production ladder is relaxed SIMD → SIMD → scalar,
gated by browser feature validation and implicit-asset availability in
`src/lc0/recklessVariants.ts`. All three use the approved upstream ref
`0010617448bdef4c8cd7d4f4825b7e42c8bc262a` and the full v60 network.
Reckless remains single-threaded (`maxThreads: 1`).

For a complete release, use `npm run reckless:build-release`; it also builds
the API variants and writes the release manifest. The production WASI/UCI
assets and corresponding-source archives alone are built with:

```sh
npm run reckless:build-production
```

This builds scalar, SIMD, relaxed SIMD, and the explicit non-default external-NNUE WASI prototype. It emits matching patched-source archives for license compliance. The SIMD build sets `RUSTFLAGS='-C target-feature=+simd128'` and enables `RECKLESS_WASM_SIMD_NNUE=1`. Relaxed SIMD adds `+relaxed-simd` and uses the exact relaxed integer dot. The embedded artifacts are `public/reckless/reckless{,-simd128,-relaxed-simd128}.wasm`; scalar remains the last fallback.

Earlier measurements used a first-pass auto-vectorized-only `+simd128` artifact where upstream Reckless still selected scalar NNUE source paths for `wasm32-wasip1`. The current build now patches Reckless to select its vectorized NNUE path for `wasm32 + simd128` and adds a wasm32 SIMD module using `core::arch::wasm32` intrinsics.

### Early SIMD smoke results (2026-06)

These predate the relaxed-first ladder.

Node/browser-wasi-shim one-shot depth-5 startpos smoke, 1 cold + 10 warm runs:

| artifact | warm avg | warm min | warm max | avg NPS |
| --- | ---: | ---: | ---: | ---: |
| `reckless.wasm` | 10.467 ms | 9.598 ms | 11.425 ms | 169,435 |
| `reckless-simd128.wasm` | 8.801 ms | 8.183 ms | 10.674 ms | 231,220 |

Headless browser one-shot depth-5 startpos smoke, 1 cold + 3 warm runs:

| artifact | cold | warm avg | avg NPS |
| --- | ---: | ---: | ---: |
| `reckless.wasm` | 113.4 ms | 13.7 ms | 241,415 |
| `reckless-simd128.wasm` | 83.1 ms | 12.4 ms | 422,768 |

These shallow smokes were only early sanity checks. The later rotated-FEN depth 7/8/9 benchmark and fixed-depth parity validation are the basis for promoting SIMD WASI/UCI as the production default.

## Implemented experiments and decisions

- **Dedicated WASM SIMD NNUE backend**: now implemented and promoted as the default production Reckless variant when WebAssembly SIMD is supported. The build script patches Reckless' NNUE module selection for `wasm32 + simd128` and adds a `core::arch::wasm32` SIMD module covering accumulator add/sub, `activate_ft`, sparse `propagate_l1`, `propagate_l2`, and `propagate_l3`; see [`reckless_wasm_simd_inspection.md`](./reckless_wasm_simd_inspection.md). A rotated-FEN browser benchmark showed clear persistent depth 8/9 wins versus scalar WASI/UCI, and a follow-up fixed-depth parity validation matched scalar exactly across the 20-position suite at depths 7/8/9; see [`reckless_browser_benchmarks.md`](./reckless_browser_benchmarks.md). The 2026-06-09 kernel-fix pass de-scalarized the `dpbusd` emulation (+19% Node NPS at depth 12), vectorized `find_nnz` over the existing `nnz_table`, fixed the latent `nnz_bitmask` shape bug, and switched relaxed builds to an exact relaxed integer dot (+24% vs old kernels); see [`reckless_simd_kernel_fixes.md`](./reckless_simd_kernel_fixes.md).
- **Threaded search**: not promoted. The WASI path forces `Threads=1`. The later Stormphrax/Node Lazy-SMP experiment rejected promotion on its measured fixed-time workload; it did not measure threaded Reckless or all browser-threading designs. See the historical alternatives below and [`threaded_emscripten_smp_prototype_2026-07-25.md`](./threaded_emscripten_smp_prototype_2026-07-25.md). The project's follow-up recommendation is closed, not an active porting task.
- **wasm-opt**: tried on scalar and SIMD artifacts; see [`reckless_wasm_opt_experiment.md`](./reckless_wasm_opt_experiment.md). Binaryen `-O3`/`-O4` trims only ~0.15% raw size and ~0.01% gzip size because embedded NNUE data dominates, and a shallow browser cold one-shot smoke did not show a clear NPS win.
- **Graceful persistent cancellation**: partially improved. The persistent adapter now sends `stop` and waits briefly before terminating the worker, so near-complete searches can return `bestmove` and keep the process reusable while the caller still receives `AbortError`. This is still not a true engine-side cancellation path: the patched wasm32 UCI loop is single-threaded and cannot reliably read `stop` while search is executing. A robust path still needs an engine-side shared abort/control flag checked from the search, or a browser-native API that exposes cancellation directly.
- **Browser-native API**: now implemented experimentally behind `Reckless Full browser API experimental` / `SIMD experimental` and `npm run reckless:build-browser-api`; see [`reckless_browser_native_api_plan.md`](./reckless_browser_native_api_plan.md). It still uses `wasm32-wasip1` imports for clocks because `wasm32-unknown-unknown` panics in `std::time::Instant`, but it bypasses `_start`, argv/stdin/stdout, UCI parsing, and UCI output formatting. Its `new_game` path now clears correction-history tables like UCI `ucinewgame`; corrected clear-hash runs matched scalar/SIMD WASI parity, while browser API SIMD remained slower than SIMD WASI/UCI at depths 7/8/9. Keep WASI/UCI as the default and treat the browser API mainly as a control/cancellation/structured-result path until the facade is faster.
- **Smaller/lazier NNUE asset path**: the June full Reckless artifacts were ~99.5% data payload; see the dated section-size and external-NNUE evidence below. The browser API SIMD external-NNUE build emits a ~1.26 MB WASM plus a separate ~63.27 MB `.nnue` payload, reports NNUE load progress in the UI, and matched embedded browser API SIMD exactly in 1260/1260 fixed-depth rotated-FEN pairs at depths 7/8/9. It remains experimental because embedded SIMD WASI/UCI is still the faster default production path.

## Adapter overhead notes

The persistent WASI adapter already keeps one process alive and skips repeated stable `setoption` commands. It now also skips a repeated identical `position ...` command in persistent mode. This is a small UCI parsing/stdin reduction for analysis refreshes and benchmark loops that re-search the same FEN with a different budget; it does not address the larger one-shot costs from WASI process startup, worker messaging, and stdout parsing.

Arena and analysis pages now call `RecklessEngine.prewarm()` after creating a Reckless instance. In cross-origin-isolated browsers, this starts the persistent worker/process and runs `uci`/`isready` before the first real search, hiding the 80-115 ms cold worker/wasm/UCI startup penalty when the user later asks for a move or analysis. Non-isolated browsers still fall back to one-shot mode.

The worker caches one `WebAssembly.Module` promise per artifact URL. It now attempts `WebAssembly.compileStreaming(response.clone())` first and falls back to `arrayBuffer()` + `WebAssembly.compile()` for servers with an incompatible MIME type. Failed cache entries are evicted so transient fetch/compile failures do not poison the worker for the rest of the page lifetime.

The worker now filters stdout before storing or posting it back to the main thread: it keeps only `uciok`, `readyok`, `bestmove`, and `info ... pv ...` lines. Stderr remains unfiltered. This preserves the data needed for readiness, bestmove, PV, score, nodes, and NPS while avoiding a high-volume stream of option/current-move diagnostics during browser searches.

Search command batches now set Reckless' `Minimal` UCI option to `true`. Reckless still emits the final PV/score/nodes/NPS line that the adapter parses, but it avoids intermediate full-report `info ... pv ...` chatter during longer searches. The persistent adapter caches this stable option just like Hash/Threads/MultiPV.

Persistent aborts now attempt a short graceful reuse path: enqueue `stop`, wait up to 100 ms for the current search to naturally emit `bestmove`, reject the caller with `AbortError`, and keep the worker alive if that happens. If no completion arrives during the grace window, the adapter falls back to worker termination as before.

The benchmark harness now has a default-on clear-hash reset. It sends `ucinewgame` + `isready` before each timed run and starts the timer afterward, avoiding persistent repeated-position TT pollution without mixing reset latency into the search wall-clock metric. It also includes a 20-position rotated-FEN suite preset, rotates warm passes across the full suite before repeating a FEN, and now records engine-reported nodes alongside wall ms and NPS.

The browser-native API is implemented, but corrected SIMD runs were slower than SIMD WASI/UCI. It remains a structured-result/control prototype, not the next unimplemented adapter reduction. See [`reckless_browser_native_api_plan.md`](./reckless_browser_native_api_plan.md).

## WASI build knobs and usage

The runtime is `@bjorn3/browser_wasi_shim` in a dedicated Worker. The target is
`wasm32-wasip1` with `--no-default-features`, omitting Syzygy/Fathom. Upstream
ships native UCI binaries, not this browser worker.

Individual build commands:

```sh
npm run reckless:build-wasi
npm run reckless:build-simd-wasi
npm run reckless:build-relaxed-simd-wasi
npm run reckless:build-simd-wasi-external
npm run reckless:build-lite-wasi
```

The Lite command needs `.local_engines/reckless-nets/v53-0ba42a8c.nnue` and
writes `public/reckless/reckless-v53-l1-512.wasm`. Lite is a local, weaker
candidate, not a default tier. Generated WASM is ignored by git.

Optional environment variables in `scripts/build_reckless_wasi.mjs`:

- `RECKLESS_REPO`: default `https://github.com/codedeliveryservice/Reckless.git`.
- `RECKLESS_REF`: default the approved full commit above, not upstream `main`.
- `RECKLESS_BUILD_DIR`: default `.local_engines/reckless-wasi-src`.
- `RECKLESS_WASM_OUT`: default `public/reckless/reckless.wasm`.
- `RECKLESS_EVALFILE`: local NNUE to embed instead of the default network.
- `RECKLESS_L1_SIZE`: compatible layout override, e.g. `512` for v53 Lite.
- `RECKLESS_WASM_SIMD_NNUE=1`: enable the dedicated wasm32 intrinsics backend.
- `RECKLESS_WASI_EXTERNAL_NNUE=1`: external WASI NNUE prototype.

After assets are staged, `npm run web:client` serves `/app/arena/` and
`/app/analysis/`. `?recklessVariant=simd`, `relaxed-simd`, `lite`, or
`wasi-simd-external` selects an explicit variant; `?recklessWasm=/path/to/reckless.wasm`
selects a custom asset. `/lab/reckless-benchmark.html` reports Full/Lite,
persistent/one-shot, depth/movetime, raw rows, CSV, and runtime metadata.

Persistent mode requires cross-origin isolation and a `SharedArrayBuffer`
stdin ring. Non-isolated browsers use one-shot argv UCI searches and do not
retain hash between moves. The synchronous wasm32 search cannot reliably
read `stop` from stdin while running. Abort therefore has only the 100 ms
natural-completion grace described above, then terminates the worker.

## NNUE section-size evidence and external loading (2026-06-04)

Generated with `npm run reckless:inspect-sections`; raw local-dev report:
`.local-dev-artifacts/docs/reckless_wasm_section_breakdown_2026-06-04.json`.

| Artifact | Total bytes | Code bytes | Data payload bytes | Data payload ratio | Data segments | Largest segment |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `public/reckless/reckless.wasm` | 64,578,655 | 284,147 | 64,262,770 | 99.51% | 2 | 64,243,802 |
| `public/reckless/reckless-simd128.wasm` | 64,578,423 | 283,932 | 64,262,754 | 99.51% | 2 | 64,243,786 |

These artifacts are almost entirely data. This explains the ~0.15% raw
`wasm-opt` saving. The embedded source uses a fixed layout:

```rust
#[repr(C)]
pub struct Parameters { /* large fixed-size arrays */ }

impl Parameters {
    fn embedded() -> &'static Self {
        static EMBEDDED: Parameters = unsafe { std::mem::transmute(*include_bytes!(env!("MODEL"))) };
        &EMBEDDED
    }
}
```

A smaller model must match that layout. Runtime external loading is now
implemented in both the browser API and a persistent WASI/UCI prototype;
it is not limited to a planned API. The WASI prototype mounts the separate
network as `reckless.nnue`. Embedded tiers remain the defaults and fallback.

The June browser-API stages implemented external parameter validation/loading,
`reckless_api_new_with_network`, worker `nnueUrl` support, module/NNUE
`force-cache` maps, and WASM/NNUE byte-progress reporting. Initial commands:

```sh
npm run reckless:extract-nnue
npm run reckless:build-browser-api-simd-external
```

Initial local smoke/build report:
`.local-dev-artifacts/docs/reckless_external_nnue_smoke_2026-06-04.json`.

| Artifact | Bytes | Notes |
| --- | ---: | --- |
| `reckless-browser-api-simd128.wasm` | 64,527,528 | Embedded full NNUE browser API SIMD artifact. |
| `reckless-browser-api-simd128-external.wasm` | 1,260,734 | External-NNUE browser API SIMD artifact; `simdOpcodeCount=1279`, `codeBytes=238,447`. |
| `reckless-v60-7f587dfb.nnue` | 63,266,880 | Separate cacheable full NNUE payload. |

Depth-4 startpos returned `c2c4`/210 nodes cold and warm. The rotated depth
7/8/9 validation matched embedded browser API SIMD exactly in 1260/1260
pairs for bestmove, score/mate, and full PV. Raw local report:
`.local-dev-artifacts/docs/reckless_external_nnue_benchmark_2026-06-04_api_simd_depth7-9.json`.
See [`reckless_browser_benchmarks.md`](./reckless_browser_benchmarks.md).
This validates the API pair, not every external WASI configuration.

### Delivery/cache tradeoffs

- Hash-named networks can be cached independently of code. The local isolated
  server on port 5199 returned `application/octet-stream`, length `63266880`,
  and `public, max-age=31536000, immutable` for `reckless-v60-7f587dfb.nnue`.
- Version generated WASM before long cache lifetimes. The local server uses a
  shorter lifetime for non-`assets/` WASM to avoid stale experimental rebuilds.
- A cached network means a browser-API code update needs only ~1.26 MB WASM.
  First-use total bytes are roughly unchanged unless selection is lazy or the
  model is smaller. Each Worker still needs its own mutable engine state.
- Local report `.local-dev-artifacts/docs/reckless_external_nnue_cache_probe_2026-06-04.json`
  measured WASM `reload` ~4.48 ms / `force-cache` ~2.32 ms, and NNUE `reload`
  ~67.51 ms / `force-cache` ~49.27 ms. These are local sanity numbers, not
  internet download estimates.

## Historical threading alternatives (2026-06; recommendation superseded in July)

The WASI build patches scoped Rust worker jobs to execute inline. Its module
used ordinary non-shared memory (`(memory (;0;) 999)` in the inspected build).
Isolation supplies `SharedArrayBuffer`, but the WASI shim does not map Rust
`std::thread` onto browser Workers. Setting UCI `Threads > 1` cannot undo that;
removing the patch alone risks unsupported primitives or hangs.

Native search shares `SharedContext`, TT, counters, stop state, and local
`ThreadData`. The alternatives considered were:

1. **Direct API plus root split:** independent engine per Worker, divide root
   moves, merge results. Incremental but no shared TT and duplicated NNUE/WASM
   memory unless parameters are external/shared.
2. **Shared-memory native-style threading:** shared WASM, atomics, Worker
   startup, and synchronization audit. Preserves the native TT model but
   replaces the WASI runtime assumptions and carries the highest risk.
3. **Hybrid split:** share fetched NNUE bytes and compiled module while keeping
   mutable state private. Saves cold-load duplication, not merging complexity.

A later June sketch proposed `wasm32-unknown-emscripten`: keep native Rust
threads with `-pthread -sPROXY_TO_PTHREAD -sPTHREAD_POOL_SIZE=n`, replace the
stdin patch with Emscripten stdin or a direct facade, and first require
Threads=1 rotated-suite parity. Threads=2/4 would need strength tests and
crash/abort soak, not exact deterministic parity. A shared atomic stop flag
was a possible cancellation route. Shared WASM would avoid N × 64 MB network
copies, at the cost of runtime/startup and scheduler risk. The proposed
single-thread tax gate was ~10% at depths 7–9 before any threading gain.

That ordering is no longer an implementation recommendation. The July
Stormphrax/Node prototype gained raw NPS but not fixed-time depth on the
measured workload, and browser bootstrap remained untested. The project
rejected promotion; this is not a direct threaded-Reckless measurement. See
[`runtime_efficiency_and_release_readiness_audit_2026-07-25.md`](./runtime_efficiency_and_release_readiness_audit_2026-07-25.md).

## Unmeasured threats/PSQ swizzle sketch (2026-06)

The native proxy profile in [`reckless_hot_path_profile.md`](./reckless_hot_path_profile.md)
put `push_threats_single` at 4.5% and `PstAccumulator::refresh` at 4.0% of
active samples. The June sketch estimated 5–8% engine NPS upside; that is an
unmeasured hypothesis, not a browser result. The native ray-permutation path
had only AVX2/AVX512 modules.

A 64-byte `vpermb` table could be decomposed into four 16-byte
`i8x16_swizzle` blocks per output chunk: form `idx - 0`, `idx - 16`,
`idx - 32`, `idx - 48`, swizzle each source chunk, and OR. The original sketch
counted 4 swizzles + 3 ORs + 4 subtracts per chunk, 16 swizzles per permutation.
Its masking/subtraction scheme still needs proof: standard swizzle zeros
out-of-range indexes, while relaxed swizzle's out-of-range lanes are
implementation-defined. Do not assume saturating subtraction alone masks
invalid lanes or treat relaxed swizzle as a drop-in.

Any experiment needs a narrow scalar-vs-swizzle randomized threat-delta
probe with exact parity, then full rotated-suite parity behind
`RECKLESS_WASM_SIMD_NNUE=1`. No speedup is established here. Other June
follow-ups were an x86_64 relaxed-dot rerun and checking whether
`convert_i8_i16` folds its scalar i64 round-trip into `v128.load64_zero`.
The API facade was already slower; it is not a pending replacement port.
