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

// The R2 deployment serves model blobs from an external /models origin. Keep
// lightweight manifests/docs in dist for diagnostics, but remove local blobs and
// packs so Netlify only hosts the app shell and small engine assets.
removeMatchingFiles(join(root, 'models', 'lc0'), (name, _path, isDir) => isDir ? name.endsWith('.lc0web') : name.endsWith('.onnx'));
removeMatchingFiles(join(root, 'models', 'maia3'), (name, _path, isDir) => !isDir && name.endsWith('.onnx'));

console.log(JSON.stringify({ status: 'EXTERNAL_MODEL_ASSET_PRUNE_DONE', root: relative(process.cwd(), root) || '.', removed }, null, 2));
