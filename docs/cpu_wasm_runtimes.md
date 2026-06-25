# CPU WASM runtimes and build targets

0x88 has two engine families that run on the CPU in WebAssembly:

1. **C/C++ engines compiled with Emscripten**: Stockfish.js-style JS glue plus
   `.wasm` and, when needed, a `.data` sidecar for NNUE/model files.
2. **Rust engines compiled to `wasm32-wasip1`**: a WASI module driven by the
   browser WASI shim, with UCI commands delivered through one-shot argv mode,
   batch mode, or a persistent shared-stdin worker.

These are separate from the LC0/Maia3 neural inference lane. They are mostly
alpha-beta/NNUE engines, with strength controlled by depth, movetime, skill, or
threads rather than by neural rating conditioning.

## Runtime choices

| Runtime path | Best fit | Current engines | Notes |
| --- | --- | --- | --- |
| Stockfish.js package | Mature baseline and strongest CPU opponent | Stockfish 18 | Emscripten-derived package with tested UCI worker wrappers and single/threaded flavors. |
| Emscripten UCI worker | C/C++ engine intake | Berserk, PlentyChess | Exports `command(const char*)`, `isReady()`, and `isSearching()` to JS; starts single-threaded before any pthread work. |
| WASI UCI worker | Rust engines with minimal JS glue | Reckless, Viridithas, Monty lab | Compiles to `wasm32-wasip1`; browser shim supplies filesystem/stdin/stdout enough for UCI. |
| Direct browser API | Control-plane experiments | Reckless browser-api variants | Bypasses UCI text where useful, but must earn a latency/lifecycle win before promotion. |

## Execution modes

- **Single-thread worker**: default intake mode. It works without
  cross-origin isolation and is easiest to smoke, abort, and package.
- **Persistent WASI worker**: keeps a Rust/WASI engine resident across searches.
  Shared stdin uses `SharedArrayBuffer`, so it requires cross-origin isolation.
- **Threaded/pthread builds**: possible for Stockfish and future Emscripten
  ports, but require COOP/COEP headers and extra sidecar workers. These are
  opt-in strength paths, not default intake targets.
- **One-shot WASI mode**: starts a fresh engine process per command/search. It
  is useful for smoke tests and failure isolation, but too much startup overhead
  for polished play unless the engine is tiny.

## Current build target matrix

| Engine | Language/toolchain | Baseline target | SIMD targets | Primary scripts |
| --- | --- | --- | --- | --- |
| Stockfish | C++ via Stockfish.js/Emscripten | packaged single-thread lite/full | packaged threaded lite/full when isolated | `stockfish:*manifest`, package assets under `public/stockfish/` |
| Berserk | C via Emscripten | scalar single-thread JS/WASM/data | `simd128`, relaxed SIMD | `npm run berserk:build-emscripten`, `berserk:build-simd-emscripten`, `berserk:build-relaxed-simd-emscripten` |
| PlentyChess | C++ via Emscripten | scalar-ish single-thread JS/WASM/data | SSE4.1-shaped wasm SIMD, relaxed SIMD | `npm run plentychess:build-emscripten`, `plentychess:build-sse41-emscripten`, `plentychess:build-relaxed-simd-emscripten` |
| Reckless | Rust `wasm32-wasip1` | scalar WASI/UCI | `+simd128`, `+simd128,+relaxed-simd`; browser API variants | `npm run reckless:build-wasi`, `reckless:build-simd-wasi`, `reckless:build-relaxed-simd-wasi`, `reckless:build-browser-api*` |
| Viridithas | Rust `wasm32-wasip1` | scalar WASI/UCI | `simd128`, relaxed SIMD | `npm run viridithas:build-wasi`, `viridithas:build-simd-wasi`, `viridithas:build-relaxed-simd-wasi` |
| Monty | Rust `wasm32-wasip1` | lab-only WASI/UCI | not productized | `npm run monty:*` where locally staged; networks are too large for product use |

## Emscripten target policy

Use Emscripten first for C/C++ UCI engines unless there is a strong reason not
to. The first target should be boring:

- `-s MODULARIZE=1`, `-s ENVIRONMENT=web,worker,node`, and a stable export name;
- exported `_main`, `_command`, and optional readiness/search-state helpers;
- single-thread search (`USE_PTHREADS=0`) and no Syzygy/tablebase dependency;
- explicit memory/stack settings and `ALLOW_MEMORY_GROWTH=1` when needed;
- NNUE/model assets either preloaded into `.data` for first smoke or loaded by a
  visible fetch/cache path.

Only add pthreads, Asyncify, multiple flavor packaging, or custom JS queues once
Node smoke, browser lifecycle smoke, and a first benchmark justify it. The
reference recipe is [`browser_c_engine_porting.md`](browser_c_engine_porting.md).

## Rust/WASI target policy

For Rust engines, `wasm32-wasip1` is the default because it preserves more of the
native UCI/file I/O shape than a bespoke `wasm32-unknown-unknown` port. The
browser adapter decides whether to run one-shot, batch, or persistent mode.

Rust SIMD targets are explicit release variants, not silent substitutions:

- scalar: compatibility fallback;
- `simd128`: preferred when browser feature detection passes and validation says
  the engine benefits;
- `relaxed-simd`: experimental unless cross-browser correctness and speed are
  proven for that engine.

Generated WASM and NNUE/model blobs remain local until the release includes the
matching manifest and corresponding-source archive described in
[`engine_artifact_distribution.md`](engine_artifact_distribution.md).

## Relaxed SIMD policy

WebAssembly Relaxed SIMD is a separate browser feature from baseline `simd128`.
A module that uses relaxed opcodes can fail validation on a browser that supports
ordinary SIMD, so relaxed builds must be separate artifacts with runtime feature
detection and fallback.

The main chess-engine opportunity is the relaxed integer dot product:
`i32x4_relaxed_dot_i8x16_i7x16_add`. For quantized NNUE kernels it can replace
multi-instruction `maddubs`/`dpbusd` emulation when we prove the unsigned
activation operand is always in `[0, 127]` (the i7 precondition). That proof now
exists for the current Reckless, Viridithas, Berserk, and PlentyChess lanes: the
common `255 >> 9` activation shape bounds post-activation bytes to i7 range, so
the relaxed dot is value-exact rather than approximate.

Relaxed floating-point operations are treated more cautiously. `relaxed_madd`,
`relaxed_min`, and `relaxed_max` may lower differently across CPUs; they are only
acceptable when fixed-depth engine parity proves that best move, score, nodes,
and PV stay identical or when the variant is explicitly labeled experimental.

Build flags and gates:

- Rust/WASI: `RUSTFLAGS='-C target-feature=+simd128,+relaxed-simd'` plus the
  engine-specific relaxed-dot code path.
- Emscripten: `-mrelaxed-simd` alongside the engine's normal SIMD-enabling flags
  (`-msimd128`, `-msse4.1`, `-mssse3`, etc. as required by that port).
- Inspection must confirm relaxed opcodes are actually present; the flag alone
  is not evidence that the hot NNUE path changed.
- Promotion requires exact fixed-depth parity against the scalar or standard
  SIMD artifact and benchmark evidence on the rotated-FEN suite.

When those gates pass, the runtime selection ladder may prefer
`relaxed-simd > simd128 > scalar` with asset and feature fallback. Without those
gates, relaxed SIMD stays a lab/benchmark variant.

## Relaxed SIMD benchmark snapshot

The current relaxed-SIMD numbers are best described as browser engineering
evidence, not universal engine-speed claims. They were measured locally in a
Chromium browser on macOS with an Apple M4 chip, using the rotated-FEN fixed-depth
suite and the browser WASM artifacts under promotion. The parity column is the
strongest signal: the relaxed artifact must match the comparison artifact for
best move, score, node count, and PV across the sampled positions.

| Engine | Browser relaxed-SIMD result on Apple M4 / Chromium | Parity gate |
| --- | ---: | --- |
| Reckless | `+24%` NPS vs old kernels | `60/60` exact fixed-depth parity |
| Viridithas | `+14%` NPS over standard SIMD | `40/40` exact fixed-depth parity |
| Berserk | `1.50M` NPS vs `1.38M` SIMD (`+8%`) | `40/40` exact fixed-depth parity |
| PlentyChess | `992k` NPS vs `603k` default (`+64%`) | `40/40` exact fixed-depth parity |
| Stockfish | not measured; current package is upstream Stockfish.js rather than a local relaxed-SIMD build | N/A |

Treat the speedups as device/runtime-specific until the same harness is repeated
on a wider browser and CPU matrix. They are still useful release evidence because
they combine same-browser performance with exact engine-output parity.

## Promotion checklist

A CPU WASM engine or new build flavor should not appear as a normal Play,
Analysis, or Arena selector until it has:

1. reproducible build script and pinned upstream source/network inputs;
2. Node UCI smoke (`uci`, `isready`, `ucinewgame`, startpos, non-startpos FEN);
3. browser worker lifecycle smoke (repeat searches, abort/recovery, missing
   asset failure, parsed `info`/PV, MultiPV if exposed);
4. benchmark evidence against the rotated-FEN protocol;
5. artifact manifest, hosted asset checks, and source-archive plan for any
   GPL/AGPL distribution;
6. UI fallback behavior when the browser lacks SIMD, relaxed SIMD,
   `SharedArrayBuffer`, or cross-origin isolation.
