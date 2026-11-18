import { lstatSync } from 'node:fs';
import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig, type Plugin } from 'vite';
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

export default defineConfig({
  cacheDir: sharedNodeModules ? '.vite-cache' : undefined,
  plugins: labOnly ? [] : [chessgroundArrows(), sveltekit()],
  // Build-time constant: the lab-route layouts 404 in product builds (src/lib/buildScope.ts). A bare
  // global, not import.meta.env.VITE_*, which Vite would also copy into every bundled env object.
  define: { __BUILD_SCOPE__: JSON.stringify(process.env.BUILD_SCOPE ?? '') },
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
