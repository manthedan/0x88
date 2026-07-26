# 0x88

**A chess engine laboratory that runs entirely in your browser.**

→ **[0x88.app](https://0x88.app)**

No server does the thinking. Neural networks run on your GPU through WebGPU;
classical engines run as WebAssembly in Workers. Everything below happens on
your machine.

- **[Play](https://0x88.app/app/play)** — play against Lc0, Stockfish, Maia
  (Elo-conditioned human models), or any of the ported classical engines.
- **[Analysis](https://0x88.app/app/analysis)** — run several engines side by
  side on the same position, with MultiPV, eval bars, and game review.
- **[Arena](https://0x88.app/app/arena)** — head-to-head matches, round robins
  and gauntlets between engines, with standings and Elo.
- `/single-engine` — Lc0 policy/eval playground.
- `/docs` — the in-product documentation surface.

Smoke, probe, and benchmark pages live under `lab/` so the app routes stay
limited to product-facing entry points.

## Engines

Nine families, each running as an independent WebAssembly module driven over
UCI. Full cards in [`docs/engine_catalog.md`](docs/engine_catalog.md).

| Family | Kind | Runtime |
| --- | --- | --- |
| **Lc0** | neural (t1-256x10, t3-512, BT4) | ORT WebGPU, WASM fallback |
| **Centipawn** | neural (SquareFormer + ThreatGraph, trained here) | TVMJS WebGPU, ORT fallback |
| **Maia 3** | neural human model, Elo-conditioned | ORT WebGPU |
| **Stockfish** | classical NNUE | WASM, threaded |
| **Reckless** | classical NNUE (Rust) | WASI, relaxed SIMD |
| **Viridithas** | classical NNUE (Rust) | WASI, relaxed SIMD |
| **PlentyChess** | classical NNUE (C++) | Emscripten, relaxed SIMD |
| **Stormphrax** | classical NNUE (C++) | Emscripten, relaxed SIMD |
| **Berserk** | classical NNUE (C) | Emscripten — build locally, see below |

Every ported engine ships a feature-detected SIMD ladder (relaxed SIMD → fixed
SIMD → scalar), with fixed-depth parity proven against the scalar build before
any tier is promoted to a default. The audits are in
[`cpu_engines_simd_audit.md`](docs/cpu_engines_simd_audit.md),
[`plentychess_simd_audit.md`](docs/plentychess_simd_audit.md), and
[`reckless_simd_kernel_fixes.md`](docs/reckless_simd_kernel_fixes.md).

## Running it locally

Requires Node 24+.

```sh
npm install
npm run web:client
```

Then open `http://localhost:5173/app/analysis`.

### What works from a fresh clone, and what doesn't

**Works immediately:** Lc0, Centipawn, Maia 3, Stockfish, PlentyChess,
Stormphrax, Viridithas. Model and engine assets are fetched on demand from
`assets.0x88.app`.

**Needs a local build:** Reckless — `public/reckless/*.wasm` is deliberately
untracked and needs a Rust + `wasm32-wasip1` toolchain. And Berserk — its
network's licence is unresolved upstream, so this repository does not
redistribute it (see [NOTICE.md](NOTICE.md)).

```sh
npm run reckless:build-wasi        # needs rustup + wasm32-wasip1
npm run berserk:build-emscripten   # needs emscripten
```

### Threads and SharedArrayBuffer

Threaded WebAssembly requires cross-origin isolation, and the dev server does
not set those headers. For anything involving threads, build and serve the
isolated static bundle:

```sh
npm run build:client
npm run web:isolated:static        # http://localhost:5181
```

Production sets `Cross-Origin-Opener-Policy: same-origin` and
`Cross-Origin-Embedder-Policy: require-corp` — see `netlify.toml`.

### Clone size

The repository carries engine binaries and networks through Git LFS. For a
working checkout without pulling every large object up front:

```sh
GIT_LFS_SKIP_SMUDGE=1 git clone <url>
git lfs pull --include="public/models/bt4_soap_rem_c19000_final.onnx"
```

## Repository map

| Path | What's in it |
| --- | --- |
| `src/lc0/` | engine adapters, variant registries, the three app surfaces |
| `src/search/` | PUCT search (`puct.ts`) |
| `src/chess/` | board, movegen, SAN/PGN, move encodings |
| `src/nn/` | ONNX Runtime configuration and neural evaluators |
| `scripts/` | engine builds, benchmarks, parity harnesses, deploy tooling |
| `tests/` | the full suite — `npm test` |
| `docs/` | architecture, porting notes, audits, benchmark records |
| `lab/` | smoke, probe, and benchmark pages |

Note: the separate `lc0-webgpu-pivot` branch holds older WebGPU research and
should only be changed with explicit approval.

## Adding an engine

Documented end to end in
[`engine_integration_architecture.md`](docs/engine_integration_architecture.md)
and [`browser_c_engine_porting.md`](docs/browser_c_engine_porting.md).
Roughly:

1. Pin the upstream commit in `engines/<name>/upstream.lock.json`.
2. Write a build script that applies your patch and emits WASM
   (`scripts/build_<name>_*.mjs`) — C/C++ through Emscripten, Rust through
   `wasm32-wasip1`.
3. Add a variant registry (`src/lc0/<name>Variants.ts`) describing the SIMD
   ladder and its feature gates.
4. Add a UCI adapter (`src/lc0/<name>Engine.ts`) implementing the
   [browser UCI adapter contract](docs/browser_uci_adapter_contract.md).
5. Register the family in `src/lc0/engineCatalog.ts`.
6. Prove fixed-depth parity between SIMD tiers before promoting a default, and
   publish a source archive plus release manifest — see
   [`engine_artifact_distribution.md`](docs/engine_artifact_distribution.md).

Other useful references:
[`hosted_artifacts.md`](docs/hosted_artifacts.md) tracks deployed engine and
model versions, [`asset_telemetry_plan.md`](docs/asset_telemetry_plan.md)
specifies the privacy-preserving asset telemetry, and
[`public/artifact-index.json`](public/artifact-index.json) is the compact
machine-readable artifact summary.

## Validation

```sh
npm test                             # typecheck + full suite
npm run bench:movegen                # movegen throughput
npm run lc0:drift-sweep              # f32/f16 numerical drift
npm run build:netlify:r2             # production build
npm run deploy:ort-wasm-dedup-check
npm run deploy:ort-runtime-assets-check
npm run productization:fast-gate
```

The nightly production journey can be run manually with
`npm run production:browser-smoke`. Benchmark output conforms to the schema in
[`browser_runtime_configuration_and_benchmark_schema.md`](docs/browser_runtime_configuration_and_benchmark_schema.md).

## Known gaps

Kept current in
[`runtime_efficiency_and_release_readiness_audit_2026-07-25.md`](docs/runtime_efficiency_and_release_readiness_audit_2026-07-25.md).
The short version: every CPU engine except Stockfish is currently
single-threaded, the WASM neural path is pinned to one thread and to the
asyncify runtime, and the move generator is a plain mailbox rather than the
0x88 board the project is named after. Contributions welcome on any of these.

## Licence

0x88 is **GPL-3.0-or-later** — see [`COPYING`](COPYING).

It bundles and builds against third-party engines and networks under GPL-3.0,
AGPL-3.0, and MIT. Attributions, per-engine licences, and the corresponding
source policy are in [`NOTICE.md`](NOTICE.md). If you fork and host this with
Reckless or Maia 3 enabled, note the AGPL network clause discussed there.
