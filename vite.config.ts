import { defineConfig } from 'vite';

export default defineConfig({
  build: {
    rollupOptions: {
      input: ['client-demo.html', 'browser-benchmark.html', 'browser-ort-bridge-benchmark.html', 'browser-rust-wasm-webgpu-benchmark.html', 'browser-eval-broker-prototype.html', 'browser-wasm-selfplay-broker.html', 'browser-two-model-arena.html', 'browser-multimodel-arena.html'],
    },
  },
});
