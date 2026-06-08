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
  ['.data', 'application/octet-stream'],
  ['.br', 'application/octet-stream'],
  ['.gz', 'application/gzip'],
  ['.png', 'image/png'],
  ['.svg', 'image/svg+xml'],
]);

function acceptsEncoding(req, encoding) {
  return String(req.headers['accept-encoding'] ?? '')
    .split(',')
    .map((part) => part.trim().toLowerCase().split(';')[0])
    .includes(encoding);
}

function uncompressedPathFor(path) {
  if (path.endsWith('.br')) return path.slice(0, -3);
  if (path.endsWith('.gz')) return path.slice(0, -3);
  return path;
}

function contentTypeFor(path) {
  return mime.get(extname(uncompressedPathFor(path))) ?? 'application/octet-stream';
}

function encodedCandidate(req, file) {
  if (file.endsWith('.br')) return { file, encoding: 'br' };
  if (file.endsWith('.gz')) return { file, encoding: 'gzip' };
  if (acceptsEncoding(req, 'br') && existsSync(`${file}.br`)) return { file: `${file}.br`, encoding: 'br' };
  if (acceptsEncoding(req, 'gzip') && existsSync(`${file}.gz`)) return { file: `${file}.gz`, encoding: 'gzip' };
  return { file, encoding: null };
}

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
  const encoded = encodedCandidate(req, file);
  file = encoded.file;
  const stats = statSync(file);
  const ext = extname(uncompressedPathFor(file));
  res.setHeader('Content-Type', contentTypeFor(file));
  res.setHeader('Content-Length', String(stats.size));
  if (encoded.encoding) {
    res.setHeader('Content-Encoding', encoded.encoding);
    res.setHeader('Vary', 'Accept-Encoding');
  }
  const rel = uncompressedPathFor(file).slice(root.length + 1).replace(/\\/g, '/');
  if (ext === '.nnue') {
    // Full Reckless network filenames include the network hash, so they are safe
    // to cache aggressively and reuse across small WASM rebuilds.
    res.setHeader('Cache-Control', 'public, max-age=31536000, immutable');
  } else if (rel.startsWith('reckless/') && rel.includes('corresponding-source') && ext === '.gz') {
    res.setHeader('Cache-Control', 'public, max-age=31536000, immutable');
  } else if ((rel.startsWith('berserk/') || rel.startsWith('plentychess/')) && (ext === '.js' || ext === '.wasm' || ext === '.data' || ext === '.nn' || ext === '.nnue' || ext === '.bin')) {
    res.setHeader('Cache-Control', 'public, max-age=31536000, immutable');
  } else if (rel.startsWith('assets/')) {
    res.setHeader('Cache-Control', 'public, max-age=31536000, immutable');
  } else if (ext === '.wasm' || ext === '.js' || ext === '.css') {
    res.setHeader('Cache-Control', 'public, max-age=3600');
  } else {
    res.setHeader('Cache-Control', 'no-cache');
  }
  if (req.method === 'HEAD') {
    res.end();
    return;
  }
  createReadStream(file).pipe(res);
});

server.listen(port, () => {
  console.log(`isolated static server: http://localhost:${port}/ -> ${root}`);
});
