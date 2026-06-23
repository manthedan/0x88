import { defineConfig } from 'vite';
import { sveltekit } from '@sveltejs/kit/vite';

const crossOriginIsolationHeaders = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
};

export default defineConfig({
  plugins: [sveltekit()],
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
