/**
 * Node-side source of truth for browser-engine artifact tooling.
 *
 * This intentionally stays separate from the client EngineFamilyDefinition
 * registry: build commands and public-tree paths must not be bundled into the
 * browser, while all deployment scripts should agree on the same inventory.
 */
export const EXTERNAL_ENGINE_ARTIFACT_DIRECTORIES = Object.freeze([
  'berserk',
  'plentychess',
  'stormphrax',
  'reckless',
  'stockfish',
  'viridithas',
  'runtimes',
]);

export const PRECOMPRESS_ARTIFACT_DIRECTORIES = Object.freeze([
  ...EXTERNAL_ENGINE_ARTIFACT_DIRECTORIES.filter((directory) => directory !== 'runtimes'),
  'ort',
  'models',
  'monty',
]);

export const COMPRESSIBLE_ARTIFACT_EXTENSIONS = Object.freeze([
  '.js', '.mjs', '.wasm', '.data', '.nn', '.nnue', '.bin', '.onnx',
]);

export function isExternalArtifactName(name) {
  return name.endsWith('.onnx')
    || name.endsWith('.lc0web')
    || name.endsWith('.wasm')
    || name.endsWith('.data')
    || name.endsWith('.nn')
    || name.endsWith('.nnue')
    || name.endsWith('.bin')
    || name.endsWith('.tar.gz')
    || name.endsWith('.gz')
    || name.endsWith('.br')
    || name.endsWith('.js')
    || name.endsWith('.mjs');
}

export const BROWSER_ENGINE_ASSET_GROUPS = Object.freeze([
  {
    family: 'lc0',
    label: 'Lc0 BT4 analysis model',
    status: 'optional-gated',
    command: 'npm run lc0:prepare-model-assets',
    docs: 'docs/engine_catalog.md#lc0-family',
    assets: ['/models/lc0/BT4-1024x15x32h-swa-6147500-policytune-332.batch8.f16.qdq8.onnx'],
  },
  {
    family: 'stockfish',
    label: 'Stockfish.js browser variants',
    status: 'release-with-lite-single-relaxed-candidate',
    command: 'npm install && npm run stockfish:build-relaxed-simd:lite-single && npm run stockfish:release-manifest',
    docs: 'docs/engine_catalog.md#stockfish-family',
    assets: [
      '/stockfish/stockfish-18-lite-single.js',
      '/stockfish/stockfish-18-lite-single.wasm',
      '/stockfish/stockfish-18-lite-single-relaxed.js',
      '/stockfish/stockfish-18-lite-single-relaxed.wasm',
      '/stockfish/stockfish-18-lite.js',
      '/stockfish/stockfish-18-lite.wasm',
      '/stockfish/stockfish-18-single.js',
      '/stockfish/stockfish-18-single.wasm',
      '/stockfish/stockfish-18.js',
      '/stockfish/stockfish-18.wasm',
      '/stockfish/stockfish-18.0.7-corresponding-source.tar.gz',
      '/stockfish/stockfish-18.0.7.manifest.json',
    ],
  },
  {
    family: 'reckless',
    label: 'Reckless WASI/browser variants',
    status: 'experimental-selectable',
    command: 'npm run reckless:build-production && npm run reckless:build-browser-api && npm run reckless:build-browser-api-simd && npm run reckless:build-browser-api-simd-external',
    docs: 'docs/engine_catalog.md#reckless-family',
    assets: [
      '/reckless/reckless.wasm',
      '/reckless/reckless-simd128.wasm',
      '/reckless/reckless-relaxed-simd128.wasm',
    ],
    optionalAssets: [
      '/reckless/reckless-browser-api.wasm',
      '/reckless/reckless-browser-api-simd128.wasm',
      '/reckless/reckless-browser-api-simd128-external.wasm',
      '/reckless/reckless-v60-7f587dfb.nnue',
    ],
  },
  {
    family: 'viridithas',
    label: 'Viridithas WASI variants',
    status: 'experimental-selectable',
    command: 'npm run viridithas:build-wasi && npm run viridithas:build-simd-wasi && npm run viridithas:build-relaxed-simd-wasi',
    docs: 'docs/engine_catalog.md#viridithas-family',
    assets: ['/viridithas/viridithas.wasm', '/viridithas/viridithas-simd128.wasm', '/viridithas/viridithas-relaxed-simd128.wasm'],
  },
  {
    family: 'berserk',
    label: 'Berserk Emscripten worker',
    status: 'experimental-selectable',
    command: 'npm run berserk:build-emscripten && npm run berserk:build-simd-emscripten && npm run berserk:build-relaxed-simd-emscripten',
    docs: 'docs/engine_catalog.md#berserk-family',
    assets: [
      '/berserk/berserk-emscripten.js',
      '/berserk/berserk-emscripten.wasm',
      '/berserk/berserk-emscripten.data',
      '/berserk/berserk-emscripten-simd128.js',
      '/berserk/berserk-emscripten-simd128.wasm',
      '/berserk/berserk-emscripten-simd128.data',
      '/berserk/berserk-emscripten-relaxed-simd128.js',
      '/berserk/berserk-emscripten-relaxed-simd128.wasm',
      '/berserk/berserk-emscripten-relaxed-simd128.data',
    ],
  },
  {
    family: 'plentychess',
    label: 'PlentyChess Emscripten worker',
    status: 'experimental-selectable',
    command: 'npm run plentychess:build-emscripten && npm run plentychess:build-sse41-emscripten && npm run plentychess:build-relaxed-simd-emscripten',
    docs: 'docs/engine_catalog.md#plentychess-family',
    assets: [
      '/plentychess/plentychess-emscripten.js',
      '/plentychess/plentychess-emscripten.wasm',
      '/plentychess/plentychess-emscripten.data',
      '/plentychess/plentychess-emscripten-sse41.js',
      '/plentychess/plentychess-emscripten-sse41.wasm',
      '/plentychess/plentychess-emscripten-sse41.data',
      '/plentychess/plentychess-emscripten-relaxed-simd128.js',
      '/plentychess/plentychess-emscripten-relaxed-simd128.wasm',
      '/plentychess/plentychess-emscripten-relaxed-simd128.data',
    ],
  },
  {
    family: 'stormphrax',
    label: 'Stormphrax Emscripten worker',
    status: 'experimental-selectable',
    command: 'npm run stormphrax:build-emscripten && npm run stormphrax:build-relaxed-simd-emscripten',
    docs: 'docs/engine_catalog.md#stormphrax-family',
    assets: [
      '/stormphrax/stormphrax-emscripten.js',
      '/stormphrax/stormphrax-emscripten.wasm',
      '/stormphrax/stormphrax-emscripten.data',
      '/stormphrax/stormphrax-emscripten-relaxed-simd128.js',
      '/stormphrax/stormphrax-emscripten-relaxed-simd128.wasm',
      '/stormphrax/stormphrax-emscripten-relaxed-simd128.data',
    ],
  },
]);
