# Netlify engine artifact deployment

This repo has a Netlify-oriented path for large generated browser-engine sidecars.

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

The app still asks Emscripten for the normal raw URLs, for example:

- `/plentychess/plentychess-emscripten.js`
- `/plentychess/plentychess-emscripten.wasm`
- `/plentychess/plentychess-emscripten.data`

`netlify.toml` force-rewrites those engine URLs to their `.br` sidecars and sets `Content-Encoding: br` plus the original content type. The browser transparently decodes the response, so Emscripten receives the original JS/WASM/data bytes while the transfer uses the precompressed artifact.

For PlentyChess, the current local estimates are:

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

The local isolated static server will serve `.br`/`.gz` sidecars through normal `Accept-Encoding` negotiation when they exist. Netlify uses the explicit rewrite/header rules in `netlify.toml`.

## Release-policy reminder

Precompression does not change licensing obligations. Do not deploy Berserk or PlentyChess generated artifacts publicly until `docs/engine_artifact_distribution.md` is satisfied with a matching source archive and artifact manifest. Use `npm run plentychess:source-archive && npm run plentychess:release-manifest` (and the Berserk equivalents if those artifacts are present) before publishing a build that serves the generated JS/WASM/data files.
