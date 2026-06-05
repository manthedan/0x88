# Viridithas WASI spike

Viridithas is an experimental browser/WASI UCI engine candidate. This branch adds a reproducible local build path, benchmark adapter, and experimental arena/analysis UI registration.

## Build

```bash
npm run viridithas:build-wasi
npm run viridithas:build-simd-wasi
```

The script:

1. clones `https://github.com/cosmobobak/viridithas` into `.local_engines/viridithas-wasi-src` and checks out pinned commit `20d7402065cae084715183e019fdd18089e2dfac`;
2. downloads the v106 `atlantis-b800.nnue.zst` network into `.local_engines/viridithas-nets`;
3. applies `patches/viridithas-wasip1.patch`;
4. builds `wasm32-wasip1` with Rust;
5. writes `public/viridithas/viridithas.wasm`.

The SIMD script sets `VIRIDITHAS_WASM_SIMD=1`, enables wasm `simd128`, and writes `public/viridithas/viridithas-simd128.wasm`.

Generated `public/viridithas/*.wasm` artifacts are ignored and not committed.

## Patch summary

The WASI patch is intentionally narrow and prototype-oriented:

- uses wasm-safe NNUE geometry/SIMD shims where Viridithas normally assumes x86/NEON intrinsics, with both scalar and wasm `simd128` NNUE backends;
- replaces worker-thread dispatch with inline execution on `wasm32`, because browser WASI has no native thread spawning;
- bypasses temp-file/mmap NNUE caching on `wasm32`, leaking one decompressed network for the process lifetime instead;
- treats argv entries as queued UCI commands in `wasm32` builds, disables search-time stdin polling for argv-driven runs, and uses a direct blocking `stdin().read_line()` loop when no argv commands are present. That gives the browser worker a real resident UCI process over the shared WASI stdin shim while keeping one-shot and benchmark-batch searches from treating the next queued command as `stop`.

## Browser status

`/reckless-benchmark.html` is now a small WASI UCI benchmark page and includes opt-in **Viridithas scalar experimental** and **Viridithas SIMD experimental** checkboxes. The experimental variants are also selectable in `/lc0-arena.html` and `/lc0-analysis.html` for shallow smoke games/lines, with runtime/asset status shown next to the Reckless status. Viridithas now supports the same broad browser modes as the Reckless WASI path:

- **persistent**: one patched WASI process remains alive and receives UCI commands through the shared stdin ring buffer when `SharedArrayBuffer` and `crossOriginIsolated` are available;
- **one-shot**: argv-driven fallback for non-isolated contexts and for apples-to-apples startup-cost measurements;
- **batch**: benchmark-only argv mode that feeds a full position sweep to one WASI invocation to estimate startup/NNUE amortisation upside.

The persistent runtime is still experimental. It is good enough for sequential searches, shallow arena games, and benchmark probes, but abort/`stop` handling currently terminates the worker rather than performing a graceful in-search stop. Analysis UI use should therefore remain cautious around rapid position changes/stops.

Fast local smoke:

```bash
npm run viridithas:smoke
```

This runs an argv-driven depth-2 two-search WASI smoke against `public/viridithas/viridithas-simd128.wasm` and verifies two `bestmove` lines plus at least one `info` line. A browser persistent smoke also succeeded on the isolated static server for startpos depths 6 and 8; see `docs/viridithas_persistent_browser_smoke_2026-06-05_startpos_depth6-8.json`.

## Caveats

- This is a compatibility spike, not a tuned browser engine.
- The scalar NNUE path is expected to be slower than native Viridithas; the wasm SIMD path improves engine NPS substantially. Persistent mode avoids the one-shot startup/decompression cost after the first command sequence.
- The WASM artifact is about 55 MiB raw with the compressed network embedded.
- Strength/eval correctness should be treated as provisional until a larger benchmark and gauntlet run.
