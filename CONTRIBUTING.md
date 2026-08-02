# Contributing to 0x88

Thanks for looking. This project is a browser chess-engine laboratory, and the
most useful contributions tend to be one of: porting an engine, making an
existing engine faster, or improving the app around them.

## Before you start

Requires **Node 24+**.

Run the suite once so you know your baseline is clean:

```sh
npm install
npm test          # typecheck + full suite; takes ~2 minutes
```

Run a single test file with
`node --experimental-strip-types --test tests/<name>.test.mjs`.

Browser-driven smokes and benches (`scripts/*browser*.mjs`) drive a real
Chromium. Both harnesses are pinned devDependencies, so `npm install` puts
them on PATH for npm scripts, but each needs its browser downloaded once:

```sh
npx playwright install chromium        # production journey smoke (npm run production:browser-smoke)
npx agent-browser install --with-deps  # remaining lc0:browser-* / engine smoke scripts
```

A few scripts reach beyond the repo by design and fail with a setup hint when
the prerequisite is missing:

- `deploy:onnxsim`, `lc0:quantize-dynamic-mmi`, `lc0:pack-model` need the ONNX
  python env: `python3 -m venv .venv-onnx && .venv-onnx/bin/pip install -r requirements-onnx.txt`.
- `lc0:eval-fixtures`/`lc0:eval-history`/`lc0:eval-f16-fixtures` default to a
  sibling `../models/lc0-bestnets` checkout; pass the model path explicitly otherwise.
- Engine WASI builds (`reckless:*`, `viridithas:*`, `stormphrax:*`...) clone
  their pinned upstream sources into the gitignored `.local_engines/` on first
  run, but `RECKLESS_EVALFILE` and similar net-path variables must point at
  nets you downloaded yourself (see docs/engine_catalog.md).
- Generators like `lc0:generate-tvm-wgsl`/`lc0:check-tvm-wgsl` and the policy
  map generator read sibling checkouts under `../repos` and `../.deps`.

If `npm test` is red on a fresh clone, that's a bug — please open an issue
rather than working around it. Every PR runs CI: `npm audit`, this suite,
and a production build.

## The one rule that matters: prove parity before you promote

Almost every performance change in this repository is a *speed ladder* — a
faster variant selected by feature detection, falling back to a slower one. The
project's standing rule is that **a faster variant may only become a default
after it produces identical results to the variant it replaces.**

In practice that means fixed-depth, fixed-position parity against the scalar or
baseline build, across a set of positions, with the counts recorded. Existing
examples to copy:

- [`docs/cpu_engines_simd_audit.md`](docs/cpu_engines_simd_audit.md) — 40/40 parity across Berserk, Viridithas
- [`docs/plentychess_simd_audit.md`](docs/plentychess_simd_audit.md) — 40/40, plus the f32 tail that was silently scalar
- [`docs/reckless_simd_kernel_fixes.md`](docs/reckless_simd_kernel_fixes.md) — 60/60 exact

For neural runtimes the equivalent gate is numerical drift against a reference
implementation — see `npm run lc0:drift-sweep` and the fixed-suite harnesses
under `scripts/`.

A change that is faster but not parity-proven is welcome as an **explicitly
gated, non-default variant**. It is not welcome as a silent default.

## Benchmarks

Report numbers, not impressions. Include the machine, the browser (or Node
version), the positions, and the budget (depth or movetime). Benchmark output
should conform to
[`docs/browser_runtime_configuration_and_benchmark_schema.md`](docs/browser_runtime_configuration_and_benchmark_schema.md).

A negative result that is measured is worth more than a positive one that is
assumed. `docs/` has several honest "we tried this and it lost" records — int4
quantization, speculative pipeline depth, standalone GPU legal-prior
processing. Adding to that list is a real contribution.

## Porting an engine

The full path is in
[`docs/engine_integration_architecture.md`](docs/engine_integration_architecture.md)
and [`docs/browser_c_engine_porting.md`](docs/browser_c_engine_porting.md); the
README has the six-step summary.

Two things people usually miss:

1. **Pin the upstream commit** in `engines/<name>/upstream.lock.json` and keep
   your changes in a patch file under `patches/`. Nobody can reproduce a build
   from an unpinned `git clone`.
2. **Licensing is a gate, not paperwork.** If you add a GPL or AGPL engine, it
   ships with a corresponding-source archive and a release manifest, or it does
   not ship. Networks need the same care as engines, and the absence of a
   `LICENSE` file in a networks repo is a prompt to investigate rather than a
   verdict — check how the engine's own build consumes the net. Berserk is the
   worked example (see [`NOTICE.md`](NOTICE.md) and the Berserk card in
   [`docs/engine_artifact_distribution.md`](docs/engine_artifact_distribution.md)).

## Working style

- One commit per logical milestone, with a message that says what changed and
  why; commits are reviewed individually as they land.
- Match the surrounding code — this codebase has consistent naming, comment
  density, and module boundaries. Read the neighbours before writing.
- Comments explain *why*, especially for anything that looks arbitrary. Several
  constants in this repo encode a hard-won browser or toolchain constraint; if
  you find one that is unexplained, documenting it is a good first PR.
- Don't commit generated engine binaries unless the distribution policy says
  that family is deploy-tracked.
- Documentation belongs in `docs/` only when it is durable product, runtime,
  provenance, or license documentation — add it to the index in
  [`docs/README.md`](docs/README.md). Working notes, one-off audits, and
  cleanup inventories stay in the gitignored `.local-dev-docs/`. Tracked files
  should never reference your local machine paths or personal tooling.

## Good first contributions

The known gaps are listed honestly in
[`docs/runtime_efficiency_and_release_readiness_audit_2026-07-25.md`](docs/runtime_efficiency_and_release_readiness_audit_2026-07-25.md).
The most approachable:

- **Make/unmake in the move generator.** `legalMoves` in `src/chess/movegen.ts`
  filters pseudo-legal moves by cloning the whole board per move and rescanning
  for check. Make/unmake with incremental king tracking and pin detection would
  remove that. Two things to know before starting: it is **not** currently a
  bottleneck (the neural lane is GPU-bound, movegen is under 1% of search time),
  and it is correctness-critical — `npm run bench:movegen` scores you, and the
  perft gate in `tests/perft_parity.test.mjs` is non-negotiable.
- **NNUE buffer alignment** in `patches/stormphrax-emscripten.patch`. The net is
  read into a plain `std::vector<std::byte>` and satisfies the loader's 16-byte
  requirement only by luck of what dlmalloc returns; a different allocator
  breaks it. Small, self-contained, and a genuine latent bug.
- **Numerical or parity harnesses** for any runtime that lacks one.

Please note that **threaded Emscripten builds are closed, not open.** It looks
like the obvious big win — the C++ engines have native Lazy SMP and
cross-origin isolation is already in place — and a prototype did reach 4.4x raw
NPS. It bought *zero extra plies* at fixed movetime, because the extra threads
duplicate work. The measurements are in
[`docs/threaded_emscripten_smp_prototype_2026-07-25.md`](docs/threaded_emscripten_smp_prototype_2026-07-25.md).
If you want to reopen it, bring depth-at-fixed-time numbers, not NPS.

## Licence

By contributing you agree that your contribution is licensed under
**GPL-3.0-or-later**, matching the project (see [`COPYING`](COPYING)).
