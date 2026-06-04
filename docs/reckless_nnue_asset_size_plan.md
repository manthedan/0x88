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
2. Add a Rust `Parameters::from_bytes(bytes: &[u8]) -> Result<Arc<Parameters>, Error>` helper gated to browser/native API builds.
3. Expose browser API initialization that accepts a network buffer or a URL-fetched buffer in the worker.
4. Cache the fetched `ArrayBuffer` and `WebAssembly.Module` per variant URL; create lightweight engine handles against the shared parameters.
5. Verify parity against embedded full NNUE for the same bytes before making external NNUE a UI option.

This should be pursued after the direct browser-native API facade, because direct handles make shared parameter lifetime and error reporting much simpler than forcing this through UCI text and WASI filesystem shims.
