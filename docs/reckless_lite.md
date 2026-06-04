# Reckless Lite evaluation notes

Reckless currently builds for the browser as a WASI module with an embedded NNUE. Most of the generated `.wasm` size is the embedded network, not JavaScript/WASI overhead.

## Why compare against Stockfish Lite?

Stockfish.js ships several browser flavors. Its “lite” builds are much smaller because they embed a smaller/weaker NNUE network rather than the full-size network. That is the useful pattern to copy for Reckless: keep the browser integration conservative and offer a smaller optional engine artifact for better load time.

Approximate Stockfish.js sizes in the current dependency:

- Full single-thread WASM: ~108 MiB
- Lite single-thread WASM: ~7 MiB

## Current Reckless full WASI build

Current Reckless `main` embeds:

- Network: `v60-7f587dfb.nnue`
- NNUE size: ~60 MiB
- Generated WASI module: ~61.6 MiB
- Source architecture constant: `L1_SIZE = 768`

This is browser-usable, but large for a default downloadable engine.

## Smaller NNUE candidates

`codedeliveryservice/RecklessNetworks` includes older/smaller NNUEs:

- `v53-0ba42a8c.nnue`: ~40 MiB
- `v47` / `v48`: ~35 MiB
- Older nets: ~15 MiB, ~11 MiB, ~6 MiB, ~1 MiB

Caveat: Reckless hard-casts `include_bytes!(env!("MODEL"))` into a fixed `Parameters` struct. A smaller net is only safe if the Rust NNUE architecture constants match that net’s layout.

## First experiment: v53 as Reckless Lite candidate

A quick compatibility test patched current Reckless source from:

```rust
const L1_SIZE: usize = 768;
```

to:

```rust
const L1_SIZE: usize = 512;
```

and built with `EVALFILE=v53-0ba42a8c.nnue`.

Result:

- Build succeeded.
- WASI smoke test succeeded.
- MultiPV smoke test succeeded.
- Generated module: `reckless-v53-l1-512.wasm`
- Size: ~41.5 MiB raw / ~27.4 MiB gzip
- Current full `v60` module: ~61.6 MiB raw / ~42.6 MiB gzip

Sample depth-5 results from a small position set:

| Position | Full v60 | v53 L1=512 |
| --- | --- | --- |
| Startpos | `c2c4`, +39 cp | `d2d4`, +25 cp |
| Kiwipete | `e5f6`, +372 cp | `d5e7`, +364 cp |
| Quiet development | `c8g4`, -370 cp | `c8g4`, -271 cp |
| Endgame sample | `g2g3`, +345 cp | `g2g3`, +292 cp |
| Tactic sample | `d1e2`, +253 cp | `h2h3`, +213 cp |
| Black attack sample | `c6b4`, -186 cp | `c6b4`, -142 cp |

This is promising but only cuts size by roughly one third. It is not yet Stockfish-Lite-level.

A quick Node WASI `go movetime 500` benchmark, excluding network download and mostly measuring one-shot WASI execution, suggested the smaller `v53` net is also faster rather than merely smaller:

| Position | Full v60 nps/depth | v53 L1=512 nps/depth |
| --- | ---: | ---: |
| Startpos | 278k / d15 | 396k / d16 |
| Quiet development | 268k / d13 | 349k / d12 |
| Endgame sample | 516k / d14 | 733k / d16 |

Treat this as preliminary. We still need a real browser benchmark because the current adapter is one-shot WASI and browser compile/instantiate behavior matters.

## Stockfish Lite comparison

An ad-hoc browser test on the isolated static server compared Stockfish single-thread flavors at `go movetime 500`:

| Flavor | Raw WASM | Gzip WASM | Load to `readyok` | Startpos nps/depth | Quiet nps/depth | Endgame nps/depth |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Stockfish lite single | ~7.0 MiB | ~5.4 MiB | ~111 ms | 1.65M / d21 | 1.59M / d20 | 3.12M / d19 |
| Stockfish full single | ~108 MiB | ~73.3 MiB | ~413 ms | 0.78M / d19 | 0.89M / d19 | 2.18M / d20 |

In this setup, Stockfish Lite is both much smaller and faster in nodes/sec, but weaker/different in evaluation because it uses a smaller NNUE. That is the kind of tradeoff worth measuring for Reckless Lite as well.

## Suggested next evaluation matrix

For each candidate engine/net combination, record:

- `.wasm` size
- compile/instantiate/load time in browser
- one-shot WASI search wall time
- nodes/sec at fixed depth
- move agreement with current v60 full build
- tactical sanity positions
- short gauntlet vs Stockfish Lite single-thread
- short gauntlet vs full Reckless WASI

Recommended candidates:

1. Current source + `v53`, `L1_SIZE=512` — already smoke-tested.
2. Older Reckless dev tags matching `v47` / `v48` / `v50` layouts.
3. Older release sources matching ~15 MiB / ~11 MiB nets.
4. Very small ~1 MiB nets only if the strength/load-time tradeoff is useful for the browser UI.

## Product direction

A browser-friendly default should probably remain lightweight and explicit:

- `Reckless Lite`: smaller, faster to load, weaker.
- `Reckless Full`: stronger, much larger download.

The UI now treats this as a named variant via `?recklessVariant=lite` and the Reckless benchmark page can compare Lite/Full against persistent/one-shot runtimes.

## Standalone repo direction

The browser/WASI build patches and Lite build recipe are useful outside lc0_webgpu. See `docs/reckless_lite_standalone.md` for the proposed extraction shape, licensing notes, and artifact naming. The short version: keep generated WASM out of this repo, but make a separate AGPL-compliant `reckless-lite`/`reckless-browser` repo that packages the build script, wasm32 patches, and optional JS worker API.

Do not make full/threaded/heavy runtime behavior the default without a clear user opt-in.
