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
- `/docs` — the in-product documentation surface.

Smoke, probe, and benchmark pages live under `lab/` — plus the unlisted,
noindexed `/single-engine` harness endpoint that the scripts in `scripts/`
drive — so the app routes stay limited to product-facing entry points.

## Engines

Eight engine families are registered in `src/lc0/engineCatalog.ts` (the
`EngineFamily` type): neural evaluators run through browser inference runtimes,
while classical engines run as independent WebAssembly modules driven over UCI.
Maia 3 is not a catalog family: Play and Analysis load it directly
(`src/lc0/maia3.ts`), and Arena does not offer it.

| Family | Kind | Runtime |
| --- | --- | --- |
| **Lc0** | neural (t1-256x10, t3-512, BT4; Leela Queen Odds in Play) | ORT WebGPU, WASM fallback |
| **Centipawn** | neural (SquareFormer + ThreatGraph, trained here) | ONNX Runtime Web (s4fix EMA) |
| **Maia 3** (Play and Analysis only) | neural human model, Elo-conditioned | ORT WebGPU |
| **Stockfish** | classical NNUE | WASM, threaded |
| **Reckless** | classical NNUE (Rust) | WASI, relaxed SIMD |
| **Viridithas** | classical NNUE (Rust) | WASI, relaxed SIMD |
| **PlentyChess** | classical NNUE (C++) | Emscripten, relaxed SIMD |
| **Stormphrax** | classical NNUE (C++) | Emscripten, relaxed SIMD |
| **Berserk** | classical NNUE (C) | Emscripten, relaxed SIMD |

Every ported engine ships a feature-detected SIMD ladder (relaxed SIMD → fixed
SIMD, then scalar where the engine has a scalar build), with fixed-depth parity
proven against the baseline build before any tier is promoted to a default. The audits are in
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

### Where engine and model files come from

The dev server serves engine and model files from this checkout's `public/`
directory. It does not use the artifact host unless you point it there, which
is the quickest way to get every engine and model without staging anything:

```sh
VITE_LC0_BROWSER_ASSET_BASE_URL=https://assets.0x88.app npm run web:client
```

With that set, `src/lc0/assetUrls.ts` resolves paths under `/models/`,
`/stockfish/`, `/berserk/`, `/plentychess/`, `/stormphrax/`, `/viridithas/`,
`/reckless/`, `/runtimes/centipawn-tvmjs-webgpu/`,
`/runtimes/squareformer-tvm-hybrid/` and `/artifacts/sha256/` against that
host. Production builds set the same variable, plus
`VITE_LC0_ARTIFACT_CHANNEL_URL` for the model release channel, in the
`netlify.toml` build command. A `?assetBase=<url>` query parameter redirects
data fetches for one page load, but engine worker scripts are only accepted
from the page's own origin or a base configured through the environment.

Without the variable, a fresh clone (with Git LFS installed) has:

- **Present after `npm install`:** Stockfish, Berserk, PlentyChess, Stormphrax,
  Viridithas and Centipawn. Their binaries and the Centipawn model are tracked,
  mostly through Git LFS; Stockfish's threaded lite and full builds are
  symlinks into `node_modules/stockfish`.
- **Lc0 networks (Small, t3, BT4, Leela Queen Odds):** gitignored.
  `npm run lc0:prepare-model-assets` links them into `public/models/lc0/` from
  a sibling `../models/lc0-bestnets/` checkout, which this repository does not
  provide.
- **Maia 3:** gitignored. `npm run maia3:stage-assets` downloads the pinned
  upstream file into `../models/maia3/`, derives the QDQ8 default (needs the
  `.venv-onnx` Python env; otherwise the fp16 file is used) and links both into
  `public/models/maia3/`.
- **Reckless:** gitignored; needs a Rust + `wasm32-wasip1` toolchain.
  `npm run reckless:build-wasi` builds the scalar `public/reckless/reckless.wasm`,
  which is enough to play. `npm run reckless:build-release` builds the full
  ladder but also rewrites the tracked Reckless manifest and source archives.

### Threads and SharedArrayBuffer

Threaded WebAssembly requires cross-origin isolation. `vite.config.ts` lists
COOP/COEP under `server.headers`, but the dev server's SvelteKit page responses
do not carry them (a `curl` of `http://localhost:5173/app/analysis/` shows neither),
so pages served by `npm run web:client` are not cross-origin isolated. For
anything involving threads, build and serve the isolated static bundle:

```sh
npm run build:client
npm run web:isolated:static        # http://localhost:5181
```

Production sets `Cross-Origin-Opener-Policy: same-origin` and
`Cross-Origin-Embedder-Policy: require-corp` — see `netlify.toml` and
`public/_headers`.

### Clone size

The repository carries engine binaries, corresponding-source archives and some
models through Git LFS (patterns in `.gitattributes`). To skip the large
objects at clone time and fetch only what the app runs from `public/`:

```sh
GIT_LFS_SKIP_SMUDGE=1 git clone <url> 0x88
cd 0x88
git lfs pull --include="public/stockfish/*.wasm,public/berserk/*.wasm,public/berserk/*.data,public/plentychess/*.wasm,public/plentychess/*.data,public/stormphrax/*.wasm,public/stormphrax/*.data,public/viridithas/*.wasm,public/models/s4fix_v_ema.onnx,public/models/bt4_soap_rem_c19000_final.onnx,public/runtimes/centipawn-tvmjs-webgpu/**/*.wasm"
```

Until they are pulled, those files are small LFS pointer files and the engines
that need them fail to load. The `*corresponding-source*.tar.gz` archives and
`public/runtimes/squareformer-tvm-hybrid/**` (a legacy Centipawn runtime and
its test fixtures) are needed only for releases and some tests. Setting
`VITE_LC0_BROWSER_ASSET_BASE_URL` as above avoids the LFS pull for running the
app.

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

The WebGPU research lane lives in `docs/` (see
[`lc0web_custom_inference_checkpoint.md`](docs/lc0web_custom_inference_checkpoint.md)
and [`lc0_tvmjs_research_runbook.md`](docs/lc0_tvmjs_research_runbook.md)), not
on a separate branch. `main` is the only branch you need.

## Adding an engine

Documented end to end in
[`engine_integration_architecture.md`](docs/engine_integration_architecture.md)
and [`browser_c_engine_porting.md`](docs/browser_c_engine_porting.md).
Roughly:

1. Pin the upstream repo, commit and licence in
   `engines/<name>/upstream.lock.json`. The manifest and source-archive writers
   read the pin from there; the build script's `<NAME>_REF` default must match
   (`tests/engine_upstream_locks.test.mjs` checks this).
2. Write a build script that applies your patch and emits WASM
   (`scripts/build_<name>_*.mjs`) — C/C++ through Emscripten, Rust through
   `wasm32-wasip1`.
3. Add a variant registry (`src/lc0/<name>Variants.ts`) describing the SIMD
   ladder and its feature gates.
4. Add a UCI adapter (`src/lc0/<name>Engine.ts`) implementing the
   [browser UCI adapter contract](docs/browser_uci_adapter_contract.md).
5. Add the family to the `EngineFamily` union and `ENGINE_FAMILY_DEFINITIONS`
   in `src/lc0/engineCatalog.ts` (`tests/engine_catalog.test.mjs` pins the
   selector order), and add its card to
   [`engine_catalog.md`](docs/engine_catalog.md).
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
npm run validate:quick               # typecheck, Biome and the tests your diff selects
npm run validate:full                # Biome and the full npm test (add -- --with-build for build:client)
npm run test:targeted -- --files src/chess/movegen.ts   # tests selected for the given paths
npm run test:timings                 # slowest test files, one file at a time
npm run test:stage-lc0-assets        # fetch the hash-pinned Lc0 parity models
npm run bench:movegen                # movegen throughput
npm run lc0:drift-sweep              # f32/f16 numerical drift
npm run build:netlify:r2             # production build
npm run dist:browser-smoke           # built client page initialization in Chromium
npm run engines:integrity-smoke      # built client engine integrity/cache checks and Play/Arena journeys
npm run deploy:ort-wasm-dedup-check
npm run deploy:ort-runtime-assets-check
npm run productization:fast-gate
```

After `npm run build:client`, `npm run engines:integrity-smoke -- [dist-directory]`
(default `dist-client`) runs locally in Playwright Chromium. Add `--ui-only`
for just the Stockfish Play and Stockfish–Berserk Arena journeys. Missing
artifacts are reported as `SKIPPED` with their paths; failures exit nonzero,
and `--require-all` makes skips fatal too. UI checks need Stockfish lite and
Berserk, not the other harness engines. Reckless is optional and gitignored;
locally staged WASM files may also be supplied under
`$ENGINE_INTEGRITY_BROWSER_SCRATCH/staged/`. The runner binds ephemeral ports
and removes its temporary browser profiles. This additional smoke is not wired
into CI; `dist:browser-smoke` remains the CI page-initialization check.

The Lc0 model tests skip when their networks are missing.
`npm run test:stage-lc0-assets` verifies or fetches the parity set into
`../models/lc0-bestnets/` against the hashes in
`public/models/lc0/manifest.json`, so those tests run instead of skipping; CI
runs it before `npm test`. `lc0:drift-sweep` reads the same directory by
default.

The nightly production journey can be run manually with
`npm run production:browser-smoke`. Benchmark output conforms to the schema in
[`browser_runtime_configuration_and_benchmark_schema.md`](docs/browser_runtime_configuration_and_benchmark_schema.md).

## Known gaps

Every CPU engine except Stockfish is single-threaded — and
[we measured that fixing it buys no playing strength](docs/threaded_emscripten_smp_prototype_2026-07-25.md),
so it is closed rather than open. The remaining headroom is in delivery and
startup, not search throughput. The move generator still clones the board per
move rather than using make/unmake — worth doing, but measure first: it is
currently under 1% of search time. The running ledger behind these calls is
[`runtime_efficiency_and_release_readiness_audit_2026-07-25.md`](docs/runtime_efficiency_and_release_readiness_audit_2026-07-25.md).

Contributions welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).

## Licence

0x88 is **GPL-3.0-or-later** — see [`COPYING`](COPYING).

It bundles and builds against third-party engines and networks under GPL-3.0,
AGPL-3.0, and MIT. Attributions, per-engine licences, and the corresponding
source policy are in [`NOTICE.md`](NOTICE.md). If you fork and host this with
Reckless or Maia 3 enabled, note the AGPL network clause discussed there.
