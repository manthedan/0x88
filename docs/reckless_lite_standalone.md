# Reckless Lite standalone repo plan

Goal: keep the LC0 browser integration able to consume Reckless Lite, while making the useful pieces easy to extract into a separate `reckless-lite` / `reckless-browser` repository that the Reckless team or browser-engine users could reuse.

## Proposed repo shape

```text
reckless-browser/
  README.md
  LICENSES/
    Reckless-AGPL-3.0.txt
    RecklessNetworks-provenance.md
  patches/
    reckless-wasip1.patch
  scripts/
    build_reckless_wasi.mjs
  src/
    recklessEngine.ts          # optional JS/TS worker API
    recklessWasiWorker.ts
    recklessVariants.ts
  examples/
    benchmark.html
  package.json
  public/
    reckless/
      README.md
      # generated .wasm artifacts are optional release assets, not source files
```

## What belongs there

- WASI/browser build script for upstream Reckless.
- Patch set for `wasm32-wasip1`:
  - stdin-driven UCI mode for persistent browser workers;
  - single-thread fallbacks for native threadpool/NUMA/TT-clear paths;
  - optional NNUE shape override for known-compatible smaller nets.
- Documentation for producing:
  - `Reckless Full` from current upstream/default v60 network;
  - `Reckless Lite` from v53 plus `L1_SIZE=512`.
- Browser worker/runtime adapter if we decide to make the package consumable by web apps outside this LC0 project.
- Benchmark/example page with CSV/JSON export so downstream users can compare browser runtimes and candidate nets without lc0_webgpu.

## Current local build commands

Full/current Reckless:

```bash
npm run reckless:build-wasi
```

Lite candidate:

```bash
npm run reckless:build-lite-wasi
```

Equivalent explicit command:

```bash
RECKLESS_EVALFILE=.local_engines/reckless-nets/v53-0ba42a8c.nnue \
RECKLESS_L1_SIZE=512 \
RECKLESS_WASM_OUT=public/reckless/reckless-v53-l1-512.wasm \
node scripts/build_reckless_wasi.mjs
```

## Licensing / release notes

- Upstream Reckless is AGPL-3.0, so distributing generated WASM requires corresponding source and patch/build instructions.
- The embedded NNUE also needs clear provenance from `codedeliveryservice/RecklessNetworks`.
- This repo should not claim to be official Reckless unless upstream adopts it.
- Generated WASM can be attached as release assets, but the repo should also include the exact Reckless ref, patch set, build command, network filename/checksum, and any architecture override such as `RECKLESS_L1_SIZE=512`.
- Best initial release asset names:
  - `reckless-full-v60-wasip1.wasm`
  - `reckless-lite-v53-l1-512-wasip1.wasm`

## Package API candidate

A reusable package should expose a tiny API, not lc0-specific UI code:

```ts
const engine = new RecklessBrowserEngine({
  wasmUrl: '/reckless/reckless-lite-v53-l1-512-wasip1.wasm',
  hashMb: 16,
  preferPersistent: true,
});
await engine.bestMove(fen, { depth: 6 });
await engine.analyze(fen, { depth: 8, multipv: 3 });
engine.dispose();
```

Runtime status should be queryable so host apps can show `persistent`, `one-shot fallback`, `SAB yes/no`, and asset-missing warnings.

## Open product questions

- Is v53/L1=512 strong enough to deserve the `Lite` label after browser matrix benchmarks and a short gauntlet?
- Should the standalone package expose a persistent JS Worker API rather than only build artifacts?
- Can upstream Reckless accept the `wasm32` patches directly, leaving the standalone repo as packaging/examples only?
- Should the package default be Full, Lite, or no bundled default asset at all?
