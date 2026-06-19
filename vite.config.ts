import { defineConfig } from 'vite';

const crossOriginIsolationHeaders = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
};

// The deployed product is the first set; the lab/benchmark pages are only
// built locally (BUILD_SCOPE=product, used by build:netlify, omits them).
const productPages = ['index.html', 'lc0-play.html', 'lc0-analysis.html', 'lc0-arena.html', 'lc0-policy-only.html'];
const labPages = ['lab/lc0-maia3-smoke.html', 'lab/berserk-smoke.html', 'lab/plentychess-smoke.html', 'lab/monty-smoke.html', 'lab/reckless-benchmark.html'];

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
      input: process.env.BUILD_SCOPE === 'product' ? productPages : [...productPages, ...labPages],
    },
  },
});
