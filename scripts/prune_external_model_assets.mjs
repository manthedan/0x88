#!/usr/bin/env node
import { existsSync, readdirSync, rmSync, statSync } from 'node:fs';
import { join, relative, resolve } from 'node:path';

const root = resolve(process.argv[2] ?? 'dist-client');

const removed = [];

function remove(path) {
  if (!existsSync(path)) return;
  const stat = statSync(path);
  rmSync(path, { recursive: true, force: true });
  removed.push({ path: relative(process.cwd(), path), kind: stat.isDirectory() ? 'directory' : 'file', bytes: stat.isFile() ? stat.size : undefined });
}

function removeMatchingFiles(dir, predicate) {
  if (!existsSync(dir)) return;
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const path = join(dir, entry.name);
    if (entry.isDirectory()) {
      if (predicate(entry.name, path, true)) remove(path);
      else removeMatchingFiles(path, predicate);
    } else if (predicate(entry.name, path, false)) {
      remove(path);
    }
  }
}

function isExternalArtifact(name) {
  return name.endsWith('.onnx')
    || name.endsWith('.lc0web')
    || name.endsWith('.wasm')
    || name.endsWith('.data')
    || name.endsWith('.nn')
    || name.endsWith('.nnue')
    || name.endsWith('.bin')
    || name.endsWith('.tar.gz')
    || name.endsWith('.gz')
    || name.endsWith('.br')
    || name.endsWith('.js')
    || name.endsWith('.mjs');
}

// The R2 deployment serves model and engine blobs from the artifact host. Keep
// lightweight manifests/docs in dist for diagnostics, but remove local blobs,
// generated sidecars, source archives, and precompressed copies so Netlify only
// hosts the app shell.
removeMatchingFiles(join(root, 'models', 'lc0'), (name, _path, isDir) => isDir ? name.endsWith('.lc0web') : name.endsWith('.onnx'));
removeMatchingFiles(join(root, 'models', 'maia3'), (name, _path, isDir) => !isDir && name.endsWith('.onnx'));
remove(join(root, 'models', 'monty'));
remove(join(root, 'monty'));
for (const dir of ['berserk', 'plentychess', 'reckless', 'stockfish', 'viridithas', 'runtimes']) {
  removeMatchingFiles(join(root, dir), (name, _path, isDir) => isDir ? false : isExternalArtifact(name));
}

console.log(JSON.stringify({ status: 'EXTERNAL_DEPLOY_ASSET_PRUNE_DONE', root: relative(process.cwd(), root) || '.', removed }, null, 2));
