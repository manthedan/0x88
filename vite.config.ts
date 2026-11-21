import { lstatSync, readFileSync } from 'node:fs';
import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig, type Plugin } from 'vite';
import kitConfig from './kit.config.js';
import { patchChessgroundSvg } from './src/lc0/chessgroundArrowPatch.ts';

const crossOriginIsolationHeaders = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
};
const labOnly = process.env.LC0_TVMJS_LAB === '1';
// A git worktree borrows the main checkout's node_modules through a symlink (scripts/worktree_setup.mjs).
// Vite's default node_modules/.vite would then be shared, and each checkout's dev server would delete
// the other's optimized deps, so a linked install gets a cache of its own.
const sharedNodeModules = (() => {
  try {
    return lstatSync(new URL('./node_modules', import.meta.url)).isSymbolicLink();
  } catch {
    return false;
  }
})();

// Chessground can't size its arrowheads independently of line width; patch its svg.js at build
// time (see src/lc0/chessgroundArrowPatch.ts). Chessground is kept out of dependency
// pre-bundling so the dev server applies the same patch.
function chessgroundArrows(): Plugin {
  return {
    name: '0x88-chessground-arrows',
    enforce: 'pre',
    transform(code, id) {
      if (!/[\\/]chessground[\\/]dist[\\/]svg\.js$/.test(id.split('?')[0])) return null;
      return { code: patchChessgroundSvg(code), map: null };
    },
  };
}

const shippedModelPins = Object.fromEntries(
  ['/models/lc0/manifest.json', '/models/maia3/manifest.json', '/models/centipawn.manifest.json'].map((manifest) => {
    const { models } = JSON.parse(readFileSync(new URL(`./public${manifest}`, import.meta.url), 'utf8')) as {
      models: { url: string; sha256?: string; bytes?: number }[];
    };
    return [
      manifest,
      Object.fromEntries(
        models
          .filter(({ sha256 }) => /^[a-f0-9]{64}$/i.test(sha256 ?? ''))
          .map(({ url, sha256, bytes }) => [url, { expectedSha256: sha256, expectedBytes: bytes }]),
      ),
    ];
  }),
);

export default defineConfig({
  cacheDir: sharedNodeModules ? '.vite-cache' : undefined,
  plugins: labOnly ? [] : [chessgroundArrows(), sveltekit(kitConfig)],
  // Build-time constant: src/hooks.ts refuses lab routes in product builds. A bare
  // global, not import.meta.env.VITE_*, which Vite would also copy into every bundled env object.
  define: { __BUILD_SCOPE__: JSON.stringify(process.env.BUILD_SCOPE ?? ''), __SHIPPED_MODEL_PINS__: JSON.stringify(shippedModelPins) },
  optimizeDeps: labOnly ? { exclude: ['onnxruntime-web'] } : { exclude: ['chessground'] },
  resolve: {
    // ORT is configured to load its staged /ort/ glue and WASM at runtime.
    // Select the external-WASM export so Vite does not emit another copy for
    // the main bundle and each evaluator worker.
    conditions: ['onnxruntime-web-use-extern-wasm', 'module', 'browser', 'development|production'],
  },
  server: {
    headers: crossOriginIsolationHeaders,
    fs: {
      allow: ['.'],
    },
  },
  preview: {
    headers: crossOriginIsolationHeaders,
  },
});
