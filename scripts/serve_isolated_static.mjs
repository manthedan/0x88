#!/usr/bin/env node
import { createReadStream, existsSync, statSync } from 'node:fs';
import { createServer } from 'node:http';
import { extname, join, normalize, resolve, sep } from 'node:path';

const root = resolve(process.argv[2] ?? 'dist-client');
const port = Number(process.env.PORT ?? process.argv[3] ?? 5181);

const mime = new Map([
  ['.html', 'text/html; charset=utf-8'],
  ['.js', 'text/javascript; charset=utf-8'],
  ['.mjs', 'text/javascript; charset=utf-8'],
  ['.css', 'text/css; charset=utf-8'],
  ['.json', 'application/json; charset=utf-8'],
  ['.wasm', 'application/wasm'],
  ['.nnue', 'application/octet-stream'],
  ['.onnx', 'application/octet-stream'],
  ['.bin', 'application/octet-stream'],
  ['.gz', 'application/gzip'],
  ['.png', 'image/png'],
  ['.svg', 'image/svg+xml'],
]);

function safePath(urlPath) {
  let decoded;
  try {
    decoded = decodeURIComponent(urlPath.split('?')[0] || '/');
  } catch {
    return null;
  }
  const rel = normalize(decoded).replace(/^([/\\])+/, '');
  const file = resolve(join(root, rel || 'index.html'));
  return file === root || file.startsWith(root + sep) ? file : null;
}

const server = createServer((req, res) => {
  res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
  res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
  res.setHeader('Cross-Origin-Resource-Policy', 'same-origin');

  let file = safePath(req.url ?? '/');
  if (!file) {
    res.writeHead(403);
    res.end('Forbidden');
    return;
  }
  if (!existsSync(file) || statSync(file).isDirectory()) file = join(file, 'index.html');
  if (!existsSync(file)) {
    res.writeHead(404);
    res.end('Not found');
    return;
  }
  const stats = statSync(file);
  const ext = extname(file);
  res.setHeader('Content-Type', mime.get(ext) ?? 'application/octet-stream');
  res.setHeader('Content-Length', String(stats.size));
  const rel = file.slice(root.length + 1).replace(/\\/g, '/');
  if (ext === '.nnue') {
    // Full Reckless network filenames include the network hash, so they are safe
    // to cache aggressively and reuse across small WASM rebuilds.
    res.setHeader('Cache-Control', 'public, max-age=31536000, immutable');
  } else if (rel.startsWith('reckless/') && rel.includes('corresponding-source') && ext === '.gz') {
    res.setHeader('Cache-Control', 'public, max-age=31536000, immutable');
  } else if (rel.startsWith('assets/')) {
    res.setHeader('Cache-Control', 'public, max-age=31536000, immutable');
  } else if (ext === '.wasm' || ext === '.js' || ext === '.css') {
    res.setHeader('Cache-Control', 'public, max-age=3600');
  } else {
    res.setHeader('Cache-Control', 'no-cache');
  }
  createReadStream(file).pipe(res);
});

server.listen(port, () => {
  console.log(`isolated static server: http://localhost:${port}/ -> ${root}`);
});
