#!/usr/bin/env node
// Production-compile the diagnostic page after BUILD_SCOPE=product build:client.
// Overlay only the lab HTML/assets; never serve TS sources via a dev server.
import { build } from 'vite';
import { cpSync } from 'node:fs';
await build({
  configFile: false,
  resolve: { conditions: ['onnxruntime-web-use-extern-wasm', 'module', 'browser', 'production'] },
  build: {
    outDir: 'artifacts/centipawn-subgroups/lab-dist',
    rollupOptions: { input: ['lab/centipawn-tvmjs-subgroups.html', 'lab/centipawn-tvmjs-webgpu-bench.html'] },
  },
});
cpSync('artifacts/centipawn-subgroups/lab-dist/lab', 'dist-client/lab', { recursive: true });
cpSync('artifacts/centipawn-subgroups/lab-dist/assets', 'dist-client/assets', { recursive: true });
cpSync('fixtures/lc0/fen_only.json', 'dist-client/fixtures/lc0/fen_only.json');
