# Netlify engine artifact deployment

This repo has a Netlify-oriented path for generated browser-engine sidecars, but Netlify is no longer responsible for forcing stable engine URLs to precompressed `.br` files.

## Build path

`netlify.toml` uses:

```sh
npm run build:netlify
```

That command runs the normal Vite build and then precompresses generated engine artifacts copied into `dist-client/`:

```sh
npm run build:client
npm run engine-artifacts:precompress-dist
```

The precompress step writes `.br` and `.gz` sidecars for files under:

- `dist-client/berserk/`
- `dist-client/plentychess/`

It currently targets generated `.js`, `.wasm`, `.data`, `.nn`, `.nnue`, and `.bin` files. Missing engine artifact directories are allowed so normal app deployments without local generated GPL artifacts still build.

## Runtime behavior on Netlify

The app asks Emscripten for normal raw URLs, for example:

- `/plentychess/plentychess-emscripten.js`
- `/plentychess/plentychess-emscripten.wasm`
- `/plentychess/plentychess-emscripten.data`

Netlify no longer force-rewrites those stable URLs to `.br` sidecars and no longer declares `Content-Encoding` in `netlify.toml` or `public/_headers`. Stable URLs may return different bytes across releases, so they must not receive a one-year `immutable` browser policy.

If a production host serves precompressed sidecars, it must do so through proven `Accept-Encoding` negotiation, object metadata, or edge/Worker code that reliably sets `Content-Encoding` and `Vary: Accept-Encoding`. Long-lived immutable caching is reserved for content-addressed paths such as `/artifacts/sha256/<hash>/...`.

For PlentyChess, prior local estimates were:

- raw total: 63,484,805 bytes / 60.54 MiB
- brotli total: 32,587,527 bytes / 31.08 MiB
- `.data` brotli sidecar: 32,447,720 bytes

Runtime memory and browser cache storage still account for the decompressed data after transfer.

## Local verification

To create sidecars in `public/` for local inspection:

```sh
npm run engine-artifacts:precompress
```

To test the production static shape locally:

```sh
npm run build:netlify
npm run web:isolated:static
curl -I -H 'Accept-Encoding: br' http://localhost:5181/plentychess/plentychess-emscripten.data
```

The local isolated static server can serve `.br`/`.gz` sidecars through normal `Accept-Encoding` negotiation when they exist. Netlify should not rely on unconditional stable-URL rewrites for compressed sidecars.

## Where header policy lives

Header policy is declared **once**, in the `[[headers]]` tables of
`netlify.toml`. `public/_headers` is generated from it and committed, so the
published tree keeps a `_headers` artifact for static hosts that read one.
Netlify treats the two as alternative mechanisms and does not document which
wins when both declare a path, so only one is hand-edited.

To change a header: edit `netlify.toml`, then regenerate.

```sh
npm run deploy:generate-headers
```

The deploy cache-policy check verifies the invariants and fails if the
generated file has drifted from its source:

```sh
npm run deploy:cache-policy-check
```

It also runs from `scripts/build_netlify_r2.mjs`, which is the command
`netlify.toml` configures, so every automatic Netlify build is gated by it —
not only the manual release script.

`scripts/netlify_headers.mjs` is the shared model both the generator and the
checker read. It parses TOML with `smol-toml` rather than by hand: an earlier
hand-rolled "narrow TOML subset" was wrong about comments after table headers
(which silently dropped a whole block from the generated file), escaped quotes,
multiline delimiters, and table boundaries. Header policy is deploy-critical, so
the grammar is not this project's to reimplement.

## Release-policy reminder

Precompression does not change licensing obligations. Do not deploy Berserk or PlentyChess generated artifacts publicly until `docs/engine_artifact_distribution.md` is satisfied with a matching source archive and artifact manifest. Use `npm run plentychess:source-archive && npm run plentychess:release-manifest` (and the Berserk equivalents if those artifacts are present) before publishing a build that serves the generated JS/WASM/data files.
