# Reckless NNUE asset-size / lazy-load notes

## Evidence

Section breakdown was generated with:

```sh
npm run reckless:inspect-sections
```

Raw report: [`reckless_wasm_section_breakdown_2026-06-04.json`](./reckless_wasm_section_breakdown_2026-06-04.json).

| Artifact | Total bytes | Code bytes | Data payload bytes | Data payload ratio | Data segments | Largest segment |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `public/reckless/reckless.wasm` | 64,578,655 | 284,147 | 64,262,770 | 99.51% | 2 | 64,243,802 |
| `public/reckless/reckless-simd128.wasm` | 64,578,423 | 283,932 | 64,262,754 | 99.51% | 2 | 64,243,786 |

The full Reckless browser artifacts are almost entirely data. This explains why `wasm-opt` only trims about 0.15% raw size: the NNUE payload dominates, not code section bloat.

Current Reckless source embeds the network at compile time:

```rust
#[repr(C)]
pub struct Parameters { /* large fixed-size arrays */ }

impl Parameters {
    fn embedded() -> &'static Self {
        static EMBEDDED: Parameters = unsafe { std::mem::transmute(*include_bytes!(env!("MODEL"))) };
        &EMBEDDED
    }
}
```

The local WASI build script can already swap the model and `L1_SIZE` at build time (`RECKLESS_EVALFILE`, `RECKLESS_L1_SIZE`), which is how the existing Lite experiment is built. It does not yet externalize the full NNUE asset at runtime.

## Implications

- Smaller first-use UX requires either a smaller model (Lite) or runtime NNUE asset loading. Code-only optimization is low-ROI.
- A browser-native API would be the cleanest point to load an external `ArrayBuffer` NNUE asset once, validate byte length/layout, copy or view it as `Parameters`, then create engines against that shared handle.
- The current WASI/UCI path could theoretically preopen/fetch a `.nnue` file, but upstream Reckless currently expects a compile-time `MODEL` and fixed Rust `Parameters` layout. Patching external load into the UCI binary is possible but less attractive than doing it with the planned direct API facade.

## Candidate staged path

1. Keep the current embedded full and Lite artifacts as the stable fallback.
2. Add a Rust `Parameters::from_bytes(bytes: &[u8]) -> Result<Arc<Parameters>, Error>` helper gated to browser/native API builds. ✅ initial browser API patch added behind `RECKLESS_BROWSER_API_EXTERNAL_NNUE=1`.
3. Expose browser API initialization that accepts a network buffer or a URL-fetched buffer in the worker. ✅ `reckless_api_new_with_network` plus worker `nnueUrl` support.
4. Cache the fetched `ArrayBuffer` and `WebAssembly.Module` per variant URL; create lightweight engine handles against the shared parameters. ✅ module and NNUE `force-cache` maps are in the browser API worker.
5. Verify parity against embedded full NNUE for the same bytes before making external NNUE a default option. ✅ fixed-depth embedded-vs-external browser API SIMD parity matched exactly in 1260/1260 depth 7/8/9 pairs.
6. Harden delivery UX. ✅ the browser API worker now reports WASM/NNUE load phases and byte progress to Arena/Analysis runtime text; the isolated static server serves `.nnue` as `application/octet-stream` with long-lived immutable caching for hash-named network files.

Initial local build commands:

```sh
npm run reckless:extract-nnue
npm run reckless:build-browser-api-simd-external
```

The first externalized option is `Reckless Full browser API SIMD external NNUE experimental`, using `/reckless/reckless-browser-api-simd128-external.wasm` plus `/reckless/reckless-v60-7f587dfb.nnue`. The embedded SIMD WASI/UCI artifact remains the default because it is faster in corrected benchmarks and is a simpler fallback.

Initial smoke/build evidence: [`reckless_external_nnue_smoke_2026-06-04.json`](./reckless_external_nnue_smoke_2026-06-04.json).

| Artifact | Bytes | Notes |
| --- | ---: | --- |
| `reckless-browser-api-simd128.wasm` | 64,527,528 | Embedded full NNUE browser API SIMD artifact. |
| `reckless-browser-api-simd128-external.wasm` | 1,260,734 | External-NNUE browser API SIMD artifact; `simdOpcodeCount=1279`, `codeBytes=238,447`. |
| `reckless-v60-7f587dfb.nnue` | 63,266,880 | Separate cacheable full NNUE payload. |

Depth-4 browser smoke on startpos loaded the external NNUE artifact successfully and returned `c2c4` with 210 nodes for both cold and warm runs. A deeper rotated-FEN validation then matched embedded browser API SIMD exactly for best move, score/mate fields, and full PV across 1260/1260 fixed-depth pairs at depths 7/8/9; see [`reckless_browser_benchmarks.md`](./reckless_browser_benchmarks.md) and raw report [`reckless_external_nnue_benchmark_2026-06-04_api_simd_depth7-9.json`](./reckless_external_nnue_benchmark_2026-06-04_api_simd_depth7-9.json). The external path remains experimental because it is a delivery/cache improvement for the browser API path, not a reason to replace the faster default SIMD WASI/UCI path.

## Delivery notes

- Use stable, content- or network-hash-named `.nnue` URLs. The current `reckless-v60-7f587dfb.nnue` filename is suitable for long-lived immutable cache headers because replacing the network should change the URL. A local `PORT=5199 node scripts/serve_isolated_static.mjs dist-client` check returned `Content-Type: application/octet-stream` and `Cache-Control: public, max-age=31536000, immutable` for the `.nnue` asset.
- Keep generated `.wasm` URLs versioned before using very long cache lifetimes; the local isolated static server uses a shorter cache lifetime for non-`assets/` `.wasm` files to avoid stale rebuilds during experiments.
- The external variant reduces repeat-update download pressure: a browser that already has the full NNUE cached only needs the ~1.26 MB WASM when browser-API code changes.
- First-use download bytes remain roughly unchanged versus embedded full artifacts unless the page lazy-loads Reckless only after the user selects it, or serves a smaller model.
