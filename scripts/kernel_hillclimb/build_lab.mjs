#!/usr/bin/env node
// Production-compile the kernel hill-climb bench pages and overlay them onto an
// existing product build (run `BUILD_SCOPE=product npm run build:client` first).
// Pages land at dist-client/lab/kernel-hillclimb/*.html and are served by any
// static server with COOP/COEP (scripts/serve_isolated_static.mjs).
//
// Usage: node scripts/kernel_hillclimb/build_lab.mjs [--dist dist-client] [--lc0-runtimes DIR]
//   --lc0-runtimes DIR  symlink DIR as <dist>/runtimes/lc0-tvmjs-webgpu when the
//                       product build lacks it (the LC0 TVMJS artifacts are
//                       gitignored/local; default: the main checkout's copy).
import { cpSync, existsSync, lstatSync, mkdirSync, readdirSync, symlinkSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import { build } from 'vite';

const repo = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const { values } = parseArgs({
  options: {
    dist: { type: 'string', default: 'dist-client' },
    'lc0-runtimes': { type: 'string' },
  },
});
const dist = resolve(repo, values.dist);
if (!existsSync(join(dist, 'index.html')) && !existsSync(join(dist, 'app')))
  throw new Error(`${dist} does not look like a product build; run BUILD_SCOPE=product npm run build:client first`);

const labDir = join(repo, 'lab/kernel-hillclimb');
const inputs = readdirSync(labDir)
  .filter((f) => f.endsWith('.html'))
  .map((f) => join(labDir, f));
const outDir = join(repo, 'artifacts/kernel-hillclimb/lab-dist');
await build({
  root: repo,
  configFile: false,
  logLevel: 'warn',
  resolve: { conditions: ['onnxruntime-web-use-extern-wasm', 'module', 'browser', 'production'] },
  build: { outDir, emptyOutDir: true, target: 'es2022', rollupOptions: { input: inputs } },
});
cpSync(join(outDir, 'lab'), join(dist, 'lab'), { recursive: true });
cpSync(join(outDir, 'assets'), join(dist, 'assets'), { recursive: true });
// Bench pages encode real positions from the committed FEN fixtures.
mkdirSync(join(dist, 'fixtures/lc0'), { recursive: true });
cpSync(join(repo, 'fixtures/lc0/fen_only.json'), join(dist, 'fixtures/lc0/fen_only.json'));

const runtimeLink = join(dist, 'runtimes/lc0-tvmjs-webgpu');
const source = resolve(values['lc0-runtimes'] ?? join(repo, '../../../public/runtimes/lc0-tvmjs-webgpu'));
let linked = 'present';
if (!existsSync(runtimeLink)) {
  let dangling = false;
  try {
    dangling = lstatSync(runtimeLink).isSymbolicLink();
  } catch {}
  if (!dangling && existsSync(source)) {
    mkdirSync(dirname(runtimeLink), { recursive: true });
    symlinkSync(source, runtimeLink);
    linked = `linked -> ${source}`;
  } else linked = `missing (no ${source}); tvmjs-profile lc0 targets will 404`;
}
console.log(JSON.stringify({ status: 'KERNEL_LAB_BUILT', pages: inputs.map((p) => `/lab/kernel-hillclimb/${p.split('/').pop()}`), lc0TvmjsRuntimes: linked }));
