# PlentyChess browser artifacts

Pinned release: **8.0.0**, tag `b-v8.0.0`, commit
`50ad9f1b523063cbbbf599fe0547f42870fe2906`; network `0178r.bin`.
See `docs/plentychess_upgrade_v8.md` for validation and compiler-flag evidence.

Build all three tiers (base is SSSE3-shaped SIMD, not scalar):

```sh
npm run plentychess:build-emscripten
npm run plentychess:build-sse41-emscripten
npm run plentychess:build-relaxed-simd-emscripten
npm run plentychess:smoke-emscripten
npm run plentychess:source-archive
npm run plentychess:release-manifest
```

Every tier loads **one** canonical `plentychess-emscripten.data`, containing
`/processed.bin`. Its SHA-256 is
`5cedef19778f1bb784e181fc013543127d1fdcde227795710b1a48f6e53d79cd`.
On a pin/network change, remove the stale canonical data **only in your own
output directory**, then rebuild every JS/WASM pair before packaging. The
builder refuses mismatched data. Use `arm64`, `ssse3`, or `fma` for the native
network helper, never `generic`/AVX2/AVX512.

Use `/lab/plentychess-smoke.html` for browser lifecycle smoke. Release files
are tracked; WASM/data/source archives use the existing Git LFS rules.
PlentyChess and its network are GPL-3.0. Distribute the matching JS/WASM/data,
source archive, manifest, and notices together; never update only a manifest
against old code. This branch does not publish/deploy anything.
