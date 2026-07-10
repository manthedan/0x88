#!/usr/bin/env node
/**
 * Build a minimal public/ tree for the Netlify v0 shell before Vite copies
 * assets. Large research/model artifacts are hosted on R2 and must never be
 * copied into dist-client merely to be deleted after the build.
 */
import { cpSync, existsSync, lstatSync, mkdirSync, readFileSync, readdirSync, readlinkSync, rmSync, writeFileSync } from 'node:fs';
import { dirname, join, relative, resolve } from 'node:path';

const sourceRoot = resolve(process.argv[2] ?? 'public');
const targetRoot = resolve(process.argv[3] ?? '.deploy-public');
const allowedStockfish = new Set([
  'stockfish-18-lite.js',
  'stockfish-18-lite.wasm',
  'stockfish-18-lite-single.js',
  'stockfish-18-lite-single.wasm',
]);

function excluded(relativePath, isDirectory) {
  const path = relativePath.replaceAll('\\', '/');
  if (!path) return false;
  const [top] = path.split('/');
  if (['runtimes', 'monty', 'rust_bridge'].includes(top)) return true;
  if (path === 'lc0-sw.js') return true;
  if (path.startsWith('models/monty/')) return true;
  if (path.startsWith('models/lc0/')) {
    if (path.endsWith('.lc0web')) return true;
    if (!isDirectory && /\.onnx(?:\.(?:br|gz))?$/.test(path)) return true;
  }
  if (path.startsWith('models/maia3/') && !isDirectory && /\.onnx(?:\.(?:br|gz))?$/.test(path)) return true;
  if (['reckless', 'berserk', 'plentychess', 'viridithas'].includes(top) && !isDirectory && /\.(?:wasm|nnue|data|js)(?:\.(?:br|gz))?$|\.tar\.gz(?:\.(?:br|gz))?$/.test(path)) return true;
  if (path.startsWith('ort/') && !isDirectory && path.endsWith('.map')) return true;
  if (path.startsWith('stockfish/') && !isDirectory && !allowedStockfish.has(path.slice('stockfish/'.length)) && !/\.(?:md|json)$/.test(path)) return true;
  return false;
}

let copiedFiles = 0;
let copiedBytes = 0;

function copyTree(source, target, relativePath = '') {
  const sourceStat = lstatSync(source);
  // Exclude by the tracked path before following symlinks. Model symlinks point
  // at optional local artifacts that are deliberately absent in clean CI.
  if (excluded(relativePath, sourceStat.isDirectory())) return;
  if (sourceStat.isSymbolicLink()) {
    const resolvedSource = resolve(dirname(source), readlinkSync(source));
    copyTree(resolvedSource, target, relativePath);
    return;
  }
  if (sourceStat.isDirectory()) {
    mkdirSync(target, { recursive: true });
    for (const entry of readdirSync(source)) copyTree(join(source, entry), join(target, entry), relativePath ? `${relativePath}/${entry}` : entry);
    return;
  }
  mkdirSync(dirname(target), { recursive: true });
  cpSync(source, target);
  copiedFiles += 1;
  copiedBytes += sourceStat.size;
}

if (!existsSync(sourceRoot)) throw new Error(`public asset source does not exist: ${sourceRoot}`);
function rewriteHostedSourceUrls(root) {
  for (const family of ['stockfish', 'berserk', 'viridithas', 'plentychess']) {
    const dir = join(root, family);
    if (!existsSync(dir)) continue;
    for (const name of readdirSync(dir)) {
      if (!name.endsWith('.manifest.json')) continue;
      const path = join(dir, name);
      const manifest = JSON.parse(readFileSync(path, 'utf8'));
      const sourceUrl = manifest?.sourceArchive?.url;
      if (typeof sourceUrl === 'string' && sourceUrl.startsWith('/')) {
        manifest.sourceArchive.url = `https://assets.0x88.app${sourceUrl}`;
        writeFileSync(path, `${JSON.stringify(manifest, null, 2)}\n`);
      }
    }
  }
}

rmSync(targetRoot, { recursive: true, force: true });
copyTree(sourceRoot, targetRoot);
rewriteHostedSourceUrls(targetRoot);
console.log(JSON.stringify({
  status: 'V0_PUBLIC_ASSET_STAGE_DONE',
  source: relative(process.cwd(), sourceRoot) || '.',
  target: relative(process.cwd(), targetRoot) || '.',
  copiedFiles,
  copiedBytes,
}, null, 2));
