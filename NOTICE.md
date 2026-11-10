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
| [Stockfish](https://github.com/nmrugg/stockfish.js) (stockfish.js) | 19.0.1, commit `0d5dd794` | GPL-3.0 | Yes — binaries, manifest, and corresponding source |
| [Reckless](https://github.com/codedeliveryservice/Reckless) | commit `00106174` (network v60) | **AGPL-3.0** | Corresponding source archives and manifest are committed; the binaries are built locally (`npm run reckless:build-release`) and served from the artifact host |
| [Viridithas](https://github.com/cosmobobak/viridithas) | v20.0.0 tag, commit `0631113e` | MIT | Yes — binaries, manifest, and corresponding source |
| [PlentyChess](https://github.com/Yoshie2000/PlentyChess) | 8.0.0 (`b-v8.0.0`), commit `50ad9f1b` | GPL-3.0 | Yes — binaries, manifest, and corresponding source |
| [Stormphrax](https://github.com/Ciekce/Stormphrax) | v8.0.0, commit `58296551` | GPL-3.0-or-later | Yes — binaries, manifest, and corresponding source |
| [Berserk](https://github.com/jhonnold/berserk) | tag 14, commit `8ae895a6` | GPL-3.0 | Yes — binaries, manifest, and corresponding source |

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

Berserk is GPL-3.0, and its network `berserk-9b84c340af7e.nn` is treated as part
of that GPL work rather than as a separately licensed asset. The networks repo
has no standalone licence file, but the engine's own `Makefile` names that exact
net as `EVALFILE`, downloads it during an ordinary build, verifies its hash, and
the binary cannot run without it — so it forms part of the Corresponding Source
the author himself distributes. The reasoning is set out in full in
`docs/engine_artifact_distribution.md`; confirmation has been requested upstream.

## Networks and models

Licences below come from repository materials only: release manifests, the
committed corresponding-source archives, `docs/model_provenance/`, and licence
files next to the assets. Where none of those states a licence, the row says so
rather than guessing.

| Asset | Source | Licence |
| --- | --- | --- |
| PlentyChess `0178r.bin` | [Yoshie2000/PlentyNetworks](https://github.com/Yoshie2000/PlentyNetworks) | GPL-3.0 |
| Viridithas `sandhi-s2-b200.nnue` | [cosmobobak/viridithas-networks](https://github.com/cosmobobak/viridithas-networks) v109 | CC0-1.0 (repo `LICENSE`, commit `1040c3ee`, 2026-06-08) |
| Berserk `berserk-9b84c340af7e.nn` | [jhonnold/berserk-networks](https://github.com/jhonnold/berserk-networks) | GPL-3.0 with the engine (see note) |
| Stockfish `nn-1a298aa575a0.nnue` (full), `nn-61e7af4bb97d.nnue` (lite) | `tests.stockfishchess.org/api/nn/` | GPL-3.0 with the engine: the release manifest records each as a "required embedded GPL network"; both are included in the corresponding-source archive |
| Stormphrax `undertown.nnue` | [Ciekce/stormphrax-nets](https://github.com/Ciekce/stormphrax-nets) release `undertown` | Licence not recorded here — to be confirmed by the maintainer. The manifest records it as Stormphrax's self-trained 8.0.0 release network, shipped with the GPL corresponding source. |
| Reckless `v60-7f587dfb.nnue` | [codedeliveryservice/RecklessNetworks](https://github.com/codedeliveryservice/RecklessNetworks) (downloaded by upstream `build/build.rs`) | Licence not recorded here — to be confirmed by the maintainer. The manifest records that Reckless embeds it and that it is distributed with the AGPL-3.0 engine source archives. |
| Maia 3 `maia3_simplified.onnx` (fp16, byte-identical to [CSSLab/maia-platform-frontend](https://github.com/CSSLab/maia-platform-frontend) at `0013cc8e`) and the local `maia3_simplified.qdq8.onnx` derivation | [CSSLab/maia3](https://github.com/CSSLab/maia3) | Maia3 AGPL-3.0; platform frontend GPL-3.0 (see `docs/model_provenance/maia3.md`) |
| Lc0 networks: `t1-256x10-distilled-swa-2432500`, `t3-512x15x16h-distill-swa-2767500`, `BT4-1024x15x32h-swa-6147500`, `BT4-1024x15x32h-swa-6147500-policytune-332` (ONNX exports and quantizations) | Leela Chess Zero project | Licence not recorded here — to be confirmed by the maintainer. `public/models/lc0/manifest.json` records URLs, sizes and hashes only. |
| Leela Queen Odds `lqo_v2` (`lqo_v2.f16.onnx` and its QDQ8 derivation) | [notune/LeelaQueenOdds](https://github.com/notune/LeelaQueenOdds) | Licence not recorded here — to be confirmed by the maintainer. |
| Centipawn `bt4_soap_rem_c19000_final` (ONNX and the TVMJS WebGPU export) | Trained in this project; provenance in `docs/model_provenance/bt4_soap_rem_c19000_final.md` | Licence not recorded here — to be confirmed by the maintainer. The provenance doc states none; the repository itself is GPL-3.0-or-later. |

Maia 3 is AGPL-3.0 upstream. `docs/model_provenance/maia3.md` records the
source offer (upstream links plus the QDQ8 derivation recipe) that a deployed
build must surface.

## Other bundled assets

| Asset | Source | Licence |
| --- | --- | --- |
| `public/openings/eco.json` (opening names) | [lichess-org/chess-openings](https://github.com/lichess-org/chess-openings) at `3ba6b20d`, see `public/openings/README.md` | CC0-1.0 (`public/openings/COPYING.txt`, copied from upstream) |
| Chess pieces (`public/chessground.cburnett.css`, embedded SVGs) | Byte-identical copy of `assets/chessground.cburnett.css` from the `chessground` npm package | Shipped under the package's licence, GPL-3.0-or-later; no separate licence for the piece artwork is recorded here — to be confirmed by the maintainer |
| Board CSS (`public/chessground.base.css`, `public/chessground.brown.css`) | Locally modified copies of the `chessground` package assets | GPL-3.0-or-later (package licence) |
| Engine marks in `public/engine-logos/` (`lc0.svg`, `stockfish.png`, `reckless.png`, `viridithas.png`, `berserk.jpg`) | The respective engine projects | Licence not recorded here — to be confirmed by the maintainer. `public/engine-logos/README.md` treats them as trademarks used only to identify the engines. |
| `public/engine-logos/centipawn.svg`, `public/engine-logos/generic.svg` | Not recorded | Licence not recorded here — to be confirmed by the maintainer |

## Runtimes and libraries

| Component | Licence |
| --- | --- |
| [ONNX Runtime Web](https://github.com/microsoft/onnxruntime) | MIT |
| [Chessground](https://github.com/lichess-org/chessground) | GPL-3.0-or-later |
| [Apache TVM](https://github.com/apache/tvm) (TVMJS WebGPU runtime) | Apache-2.0 |
| [Svelte / SvelteKit](https://github.com/sveltejs/kit) | MIT |
| [@bjorn3/browser_wasi_shim](https://github.com/bjorn3/browser_wasi_shim) | MIT / Apache-2.0 |
| [TanStack Query / Store](https://github.com/TanStack) | MIT |

Full transitive dependency licences are resolvable from `package-lock.json`.

## Corresponding source

For every GPL/AGPL binary this project distributes, a corresponding source
archive and a release manifest are published alongside it. See
`docs/engine_artifact_distribution.md` for the release gate. For Stockfish,
Viridithas, Berserk, PlentyChess and Stormphrax, `npm run <engine>:source-archive`
and `npm run <engine>:release-manifest` produce them. Reckless has no
`:source-archive` script: `npm run reckless:build-production` writes its four
archives, and `npm run reckless:build-release` runs that step and then
`reckless:release-manifest`.

If you believe an attribution here is wrong or incomplete, please open an issue —
getting this right matters more than getting it fast.
