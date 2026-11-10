import { existsSync } from 'node:fs';
import { resolve } from 'node:path';
import { parseScriptArgs } from './cli.mjs';

export function parseIntegritySmokeArgs(argv) {
  const args = parseScriptArgs(argv, {
    options: {
      'ui-only': { type: 'boolean', default: false },
      'require-all': { type: 'boolean', default: false },
    },
    allowPositionals: true,
    usage: 'Usage: node scripts/engine_integrity_browser_smoke.mjs [dist-directory] [--ui-only] [--require-all]',
  });
  if (args.positionals.length > 1) throw new Error('Expected at most one dist directory');
  return { dist: resolve(args.positionals[0] ?? 'dist-client'), uiOnly: args.uiOnly, requireAll: args.requireAll };
}

const engines = [
  { name: 'Stockfish lite', family: 'sf', base: '/stockfish/stockfish-19-lite-single' },
  { name: 'Reckless', family: 'reckless', base: '/reckless/reckless' },
  { name: 'Viridithas', family: 'viridithas', base: '/viridithas/viridithas' },
  { name: 'Berserk defaults', family: 'berserk', base: '/berserk/berserk-emscripten' },
  { name: 'PlentyChess defaults', family: 'plentychess', base: '/plentychess/plentychess-emscripten' },
  { name: 'Stormphrax defaults', family: 'stormphrax', base: '/stormphrax/stormphrax-emscripten' },
  { name: 'Stockfish full pthread', family: 'sf', base: '/stockfish/stockfish-19', threads: 2 },
];

export const integrityEngineSpecs = engines.map((spec) => ({
  ...spec,
  wasm: spec.base + '.wasm',
  assets: [
    spec.base + '.wasm',
    ...(['reckless', 'viridithas'].includes(spec.family) ? [] : [spec.base + '.js']),
    ...(['berserk', 'plentychess', 'stormphrax'].includes(spec.family) ? [spec.base + '.data'] : []),
  ],
}));

export function integrityUiAssets(path) {
  // Play/Arena select Stockfish lite using the browser's SIMD feature gate.
  const stockfish = [...integrityEngineSpecs[0].assets, '/stockfish/stockfish-19-lite-single-relaxed.js', '/stockfish/stockfish-19-lite-single-relaxed.wasm'];
  return [path + 'index.html', ...stockfish, ...(path.includes('arena') ? integrityEngineSpecs.find((spec) => spec.family === 'berserk').assets : [])];
}

export function missingIntegrityAssets(assets, assetFile, exists = existsSync) {
  return assets.filter((path) => !exists(assetFile(path)));
}

export function integritySmokeExitCode({ failures, skipped }, requireAll) {
  return failures || (requireAll && skipped) ? 1 : 0;
}

/** Chromium logs resource errors for optional availability HEAD probes too. */
export function actionableIntegrityErrors(entries, optionalMissingProbeUrls, fetchedUrls = new Set()) {
  return entries
    .filter(
      (entry) =>
        !(
          entry.kind === 'console' &&
          optionalMissingProbeUrls.has(entry.url) &&
          !fetchedUrls.has(entry.url) &&
          /^Failed to load resource: the server responded with a status of (404|410)\b/.test(entry.text)
        ),
    )
    .map((entry) => entry.text);
}
