import { defineConfig } from 'vite';

const crossOriginIsolationHeaders = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
};

export default defineConfig({
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
  build: {
    rollupOptions: {
      input: ['index.html', 'lc0-play.html', 'client-demo.html', 'lc0-policy-only.html', 'lc0-analysis.html', 'lc0-arena.html', 'berserk-smoke.html', 'plentychess-smoke.html', 'reckless-benchmark.html', 'browser-benchmark.html', 'browser-ort-bridge-benchmark.html', 'browser-rust-wasm-webgpu-benchmark.html', 'browser-eval-broker-prototype.html', 'browser-wasm-selfplay-broker.html', 'browser-two-model-arena.html', 'browser-multimodel-arena.html'],
    },
  },
});
