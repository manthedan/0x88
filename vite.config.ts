import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig, type Plugin } from 'vite';
import { patchChessgroundSvg } from './src/lc0/chessgroundArrowPatch.ts';

const crossOriginIsolationHeaders = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
};
const labOnly = process.env.LC0_TVMJS_LAB === '1';

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
  plugins: labOnly ? [] : [chessgroundArrows(), sveltekit()],
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
