# Third-party notices

0x88 itself is licensed **GPL-3.0-or-later** (see [`COPYING`](COPYING)).

This file records the third-party engines, networks, and runtimes the project
builds against, links to, or distributes. It is a summary — the authoritative
per-artifact record, including pinned commits, applied patches, corresponding
source archives, and release manifests, is
[`docs/engine_artifact_distribution.md`](docs/engine_artifact_distribution.md)
and [`public/artifact-index.json`](public/artifact-index.json).

Each engine below runs as a **separate WebAssembly module in its own Worker**,
driven over the UCI text protocol. 0x88 does not statically link any engine.

## Engines

| Engine | Version / pin | Upstream licence | Distributed by this repo? |
| --- | --- | --- | --- |
| [Stockfish](https://github.com/nmrugg/stockfish.js) (stockfish.js) | 18.0.7, commit `32d4b5ae` | GPL-3.0 | Yes — binaries, manifest, and corresponding source |
| [Reckless](https://github.com/codedeliveryservice/Reckless) | v60, commit `00106174` | **AGPL-3.0** | Binaries are built locally / hosted; corresponding source archives are published |
| [Viridithas](https://github.com/cosmobobak/viridithas) | v20.0.0, commit `20d74020` | MIT | Yes — binaries, manifest, and corresponding source |
| [PlentyChess](https://github.com/Yoshie2000/PlentyChess) | 7.0.66, commit `58d8ba25` | GPL-3.0 | Yes — binaries, manifest, and corresponding source |
| [Stormphrax](https://github.com/Ciekce/Stormphrax) | v8.0.0, commit `58296551` | GPL-3.0-or-later | Yes — binaries, manifest, and corresponding source |
| [Berserk](https://github.com/jhonnold/berserk) | tag 14, commit `8ae895a6` | GPL-family | **No** — see note below |

### Note on Reckless and the AGPL

Reckless is licensed **AGPL-3.0**, not GPL-3.0. AGPL-3.0 §13 adds a network
clause: users who interact with the covered work remotely must be able to
receive its corresponding source.

0x88's own code is GPL-3.0-or-later, which is compatible with combining with
AGPL-3.0 code — but the AGPL portion keeps its own terms, and the network clause
travels with it. In practice:

- Reckless executes as an independent WebAssembly program in a Worker,
  communicating over UCI text. This is a strong arm's-length separation, and the
  usual reading is that it remains a separate program rather than a derivative
  of the app that drives it.
- Regardless of that reading, this project publishes Reckless corresponding
  source archives alongside its binaries, so the source-availability obligation
  is satisfied either way.

**If you fork and host 0x88 with Reckless enabled, keep publishing the Reckless
corresponding source.** That is the safe path and it is what this repository
already does.

### Note on Berserk

The Berserk *engine* is GPL-family and freely redistributable. Its **network**
is not: no standalone licence was found in `jhonnold/berserk-networks` during
intake, so provenance is unresolved
(see `docs/engine_artifact_distribution.md`).

Accordingly, **this repository does not distribute Berserk binaries or its
network `.data` bundle.** The build script fetches the network from upstream at
build time. Berserk remains fully buildable with
`npm run berserk:build-emscripten`; it is simply not redistributed here. If you
resolve the network's licence upstream, this restriction can be lifted.

> **Open item (2026-07-25):** copies of the Berserk artifacts from earlier
> deploys are still present on the project's asset CDN and have not yet been
> deleted. Removing them from the repository did not unpublish them. See the
> "ACTION REQUIRED" note in
> `docs/runtime_efficiency_and_release_readiness_audit_2026-07-25.md`.

## Networks and models

| Asset | Source | Licence |
| --- | --- | --- |
| PlentyChess `0134-2r24-s0.bin` | [Yoshie2000/PlentyNetworks](https://github.com/Yoshie2000/PlentyNetworks) | GPL-3.0 |
| Viridithas `atlantis-b800.nnue` | [cosmobobak/viridithas-networks](https://github.com/cosmobobak/viridithas-networks) | MIT (with the engine) |
| Berserk `berserk-9b84c340af7e.nn` | [jhonnold/berserk-networks](https://github.com/jhonnold/berserk-networks) | **Unresolved** — not redistributed here |
| Maia 3 | [CSSLab/maia3](https://github.com/CSSLab/maia3) | AGPL-3.0 |
| Lc0 networks (t1, t3, BT4) | Leela Chess Zero project | See `docs/model_provenance/` |
| Centipawn `bt4_soap_rem_c19000_final` | Trained in this project | GPL-3.0-or-later, see `docs/model_provenance/bt4_soap_rem_c19000_final.md` |

Maia 3 is AGPL-3.0; the same reasoning as the Reckless note above applies.

## Runtimes and libraries

| Component | Licence |
| --- | --- |
| [ONNX Runtime Web](https://github.com/microsoft/onnxruntime) | MIT |
| [Chessground](https://github.com/lichess-org/chessground) | GPL-3.0 |
| [Apache TVM](https://github.com/apache/tvm) (TVMJS WebGPU runtime) | Apache-2.0 |
| [Svelte / SvelteKit](https://github.com/sveltejs/kit) | MIT |
| [@bjorn3/browser_wasi_shim](https://github.com/bjorn3/browser_wasi_shim) | MIT / Apache-2.0 |
| [TanStack Query / Store](https://github.com/TanStack) | MIT |

Full transitive dependency licences are resolvable from `package-lock.json`.

## Corresponding source

For every GPL/AGPL binary this project distributes, a corresponding source
archive and a release manifest are published alongside it. See
`docs/engine_artifact_distribution.md` for the release gate, and
`npm run <engine>:source-archive` / `npm run <engine>:release-manifest` for the
tooling that produces them.

If you believe an attribution here is wrong or incomplete, please open an issue —
getting this right matters more than getting it fast.
