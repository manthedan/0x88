# Viridithas WASI spike

Viridithas is an experimental browser/WASI UCI engine candidate. This branch adds a reproducible local build path and benchmark-only browser adapter.

## Build

```bash
npm run viridithas:build-wasi
```

The script:

1. clones `https://github.com/cosmobobak/viridithas` into `.local_engines/viridithas-wasi-src` and checks out pinned commit `20d7402065cae084715183e019fdd18089e2dfac`;
2. downloads the v106 `atlantis-b800.nnue.zst` network into `.local_engines/viridithas-nets`;
3. applies `patches/viridithas-wasip1.patch`;
4. builds `wasm32-wasip1` with Rust;
5. writes `public/viridithas/viridithas.wasm`.

Generated `public/viridithas/*.wasm` artifacts are ignored and not committed.

## Patch summary

The WASI patch is intentionally narrow and prototype-oriented:

- uses scalar wasm-safe NNUE geometry/SIMD shims where Viridithas normally assumes x86/NEON intrinsics;
- replaces worker-thread dispatch with inline execution on `wasm32`, because browser WASI has no native thread spawning;
- bypasses temp-file/mmap NNUE caching on `wasm32`, leaking one decompressed network for the process lifetime instead;
- treats argv entries as queued UCI commands in `wasm32` builds, then closes the command channel after those commands so one-shot browser searches run to their requested limit instead of being interrupted by an immediate `quit`.

## Browser status

`/reckless-benchmark.html` is now a small WASI UCI benchmark page and includes an opt-in **Viridithas experimental** checkbox. Viridithas is one-shot only for now; persistent SAB stdin is skipped because upstream Viridithas expects a separate stdin reader thread for full interactive UCI.

Local Node and browser smokes from the patched WASI artifact succeeded for:

```text
uci
isready
setoption name Hash value 16
position startpos
go depth 2
```

and returned `bestmove g1f3` with depth/nodes/NPS info from the v20.0.0-dev/v106 network build.

## Caveats

- This is a compatibility spike, not a tuned browser engine.
- The scalar NNUE path is expected to be slower than native Viridithas.
- The WASM artifact is about 55 MiB raw with the compressed network embedded.
- Strength/eval correctness should be treated as provisional until a larger benchmark and gauntlet run.
