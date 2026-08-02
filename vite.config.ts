import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

const crossOriginIsolationHeaders = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
};
const labOnly = process.env.LC0_TVMJS_LAB === '1';

export default defineConfig({
  plugins: labOnly ? [] : [sveltekit()],
  optimizeDeps: labOnly ? { exclude: ['onnxruntime-web'] } : undefined,
  resolve: {
    // ORT is configured to load its staged /ort/ glue and WASM at runtime.
    // Select the external-WASM export so Vite does not emit another copy for
    // the main bundle and each evaluator worker.
    conditions: ['onnxruntime-web-use-extern-wasm', 'module', 'browser', 'development|production'],
  },
  server: {
    headers: crossOriginIsolationHeaders,
    fs: {
      // node_modules is a symlink to the sibling leelaweb workspace in this checkout;
      // allow it so ORT's WASM sidecar can be served by the dev server.
      allow: ['.', '../leelaweb/node_modules'],
    },
  },
  preview: {
    headers: crossOriginIsolationHeaders,
  },
});
