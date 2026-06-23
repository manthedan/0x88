#!/usr/bin/env node
import { existsSync, readdirSync, rmSync, statSync } from 'node:fs';
import { join, relative, resolve } from 'node:path';

const root = resolve(process.argv[2] ?? 'dist-client');
const removed = [];

function remove(path) {
  if (!existsSync(path)) return;
  const stat = statSync(path);
  rmSync(path, { recursive: true, force: true });
  removed.push({
    path: relative(process.cwd(), path),
    kind: stat.isDirectory() ? 'directory' : 'file',
    bytes: stat.isFile() ? stat.size : undefined,
  });
}

function removeMatchingFiles(dir, predicate) {
  if (!existsSync(dir)) return;
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const path = join(dir, entry.name);
    if (entry.isDirectory()) removeMatchingFiles(path, predicate);
    else if (predicate(entry.name, path)) remove(path);
  }
}

remove(join(root, 'runtimes'));
remove(join(root, 'monty'));
remove(join(root, 'rust_bridge'));
remove(join(root, 'berserk'));
remove(join(root, 'plentychess'));
remove(join(root, 'viridithas'));
removeMatchingFiles(join(root, 'reckless'), (name) => name.endsWith('.wasm') || name.endsWith('.nnue') || name.endsWith('.tar.gz'));
remove(join(root, 'models', 'monty'));
removeMatchingFiles(join(root, 'ort'), (name) => name.endsWith('.map'));
removeMatchingFiles(join(root, 'stockfish'), (name) => ![
  'stockfish-18-lite.js',
  'stockfish-18-lite.wasm',
  'stockfish-18-lite-single.js',
  'stockfish-18-lite-single.wasm',
].includes(name));

console.log(JSON.stringify({ status: 'V0_DEPLOY_ASSET_PRUNE_DONE', root: relative(process.cwd(), root) || '.', removed }, null, 2));
