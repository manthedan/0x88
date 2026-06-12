# Engine packaging reorganization plan

2026-06-10, branch `feature/engine-packaging-reorg`. Decision: extract the
optimized-WASM engine packaging (Reckless, Viridithas, Berserk, PlentyChess)
into standalone per-engine repos plus a shared toolkit, so the builds are
usable by third parties and adoptable by the upstream engine authors.
Stockfish stays on its upstream npm build; LC0/Tiny Leela/BT4 stay in
leelaweb — they are the project itself, not packaging.

## Why (observed pain, 2026-06 audit sessions)

- **Packaging is smeared across worktree branches.** PlentyChess's
  build/patch/variants live on `feature/plentychess-browser-worker`; the
  other three engines on the relaxed-simd lineage. The generic Emscripten
  UCI bench harness exists twice under two names on two branches
  (`berserk_emscripten_bench.mjs` / `emscripten_uci_bench.mjs`).
- **Engine sources hide from tooling.** Reckless's entire wasm32 NNUE SIMD
  backend was a string literal inside `build_reckless_wasi.mjs` — no
  rustfmt/clippy/rust-analyzer coverage on the hottest kernels in the
  project (this nearly hid the `nnz_bitmask` bug). The other engines use
  `.patch` files that need apply-edit-regenerate cycles.
- **License hygiene.** Reckless/Viridithas are AGPL-3.0, Berserk/PlentyChess
  GPL-3.0; we distribute patched derivative builds with corresponding-source
  archives out of an app repo. One repo per engine carries its upstream
  license cleanly.
- **The packaging is already a product**: pinned refs, audited patches with
  exactness proofs, scalar/simd128/relaxed-simd variant ladders, fixed-depth
  parity harnesses, opcode inspection, benchmark records, release recipes —
  but the parity gates run by hand instead of as CI.

## Target structure

```
<org>/reckless-wasm        <org>/viridithas-wasm
<org>/berserk-wasm         <org>/plentychess-wasm
  upstream.lock.json        # upstream repo URL, pinned commit, net URL+SHA256
  src/ | patches/           # real .rs/.c sources and patches, not strings
  build/                    # scalar / simd128 / relaxed-simd variant builds
  bench/                    # parity + NPS via the shared harness (CI gate)
  loader/                   # tiny npm package: feature-detected variant
                            #   selection + UCI send/onLine; no app deps
  CORRESPONDING_SOURCE.md   # per-upstream license compliance recipe
  .github/workflows/        # build → parity → release on tag

<org>/wasm-engine-toolkit
  WASI UCI bench harness + Emscripten UCI bench harness (20-FEN rotated
  parity protocol), wasm SIMD opcode inspector, SIMD/relaxed-SIMD feature
  probes, precompressor, artifact manifest writer, corresponding-source
  archiver.
```

Mechanics:

- **Artifacts via GitHub Releases, not npm** (25–64 MB networks). Tags
  encode upstream version + packaging revision (e.g. `v60-wasm.3`) and
  publish wasm/js/data + `.br`/`.gz` sidecars + SHA manifest + source
  tarball. The npm package is only the loader.
- **CI is the parity gate**: the Node harnesses need no browser, so every
  release runs build-all-variants → fixed-depth exact parity → bench record.
- **leelaweb consumes, doesn't build**: `engines.lock.json` pinning release
  URLs + SHAs, plus a fetch script replacing in-repo builds. Arena/analysis
  adapters (`BrowserUciEngine` impls, variant UI) stay in leelaweb on top of
  each repo's loader.

## Migration stages

1. **Consolidate in-monorepo** (prerequisite regardless of the split):
   - Extract inline/patch-string engine sources into `engines/<name>/`
     directories with `upstream.lock.json`; build scripts read files.
   - Converge duplicated tooling names across branches
     (`emscripten_uci_bench.mjs` is the canonical name).
   - Settle the main lineage before seeding new repos — the PlentyChess
     branch must merge or rebase first.
2. **Extract `wasm-engine-toolkit`** (pure scripts; nearly dependency-free
   already).
3. **Spin out per-engine repos one at a time, Reckless first** (most mature
   story; serves as the template). Each repo verified standalone: build all
   variants + parity before first release.
4. **Swap leelaweb to fetch-from-releases** and delete in-repo builds last,
   after each engine repo's CI is green.

## Status

- [x] Stage 1 (Reckless): wasm32 NNUE shim and find_nnz extracted to
      `engines/reckless/src/*.rs`; `engines/reckless/upstream.lock.json`;
      build script reads files (verified by byte-comparing the generated
      patched tree). Bench harness renamed to `emscripten_uci_bench.mjs`.
- [x] Stage 3 seed: `reckless-wasm` template repo drafted as a sibling repo
      and verified standalone (see its README).
- [x] Stage 1 for Viridithas/Berserk/PlentyChess: PlentyChess branch merged;
      `engines/<name>/upstream.lock.json` mirrors each sibling repo's lock.
- [x] Stage 3: `viridithas-wasm`, `berserk-wasm`, `plentychess-wasm` sibling
      repos extracted from the template; each verified standalone with
      byte-identical artifacts (where monorepo artifacts exist) and exact
      parity gates.
- [ ] Stage 2 toolkit extraction (bench harnesses/inspector are currently
      copied per repo; dedupe once an org/registry exists to publish to).
- [ ] GitHub org + push + first tagged releases.
- [ ] leelaweb `engines.lock.json` + fetch-from-releases swap (delete
      in-repo builds last; note this changes the deliberate
      track-GPL-artifacts-in-git policy).
