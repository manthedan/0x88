#!/usr/bin/env node
import { createReadStream } from 'node:fs';
import { mkdir, mkdtemp, readFile, rm, stat, writeFile } from 'node:fs/promises';
import { createServer } from 'node:http';
import { tmpdir } from 'node:os';
import { dirname, extname, join, resolve } from 'node:path';
import { createBrotliCompress, createGzip } from 'node:zlib';
import { pipeline } from 'node:stream/promises';

const DEFAULT_OUT = 'artifacts/tvm/static_content_encoding_smoke.json';
const DEFAULT_PATH = 'public/runtimes/lc0-tvmjs-webgpu/t1-256x10-distilled-swa-2432500/f16/v1/tvmjs_runtime.wasm';

function usage() {
  console.log(`Usage: node scripts/check_static_content_encoding.mjs [options]\n\nCreates temporary .br/.gz sidecars for one local artifact, serves them with a minimal static server,\nand verifies HEAD responses set the expected Content-Encoding and original Content-Type.\n\nOptions:\n  --file PATH       Artifact to probe (default ${DEFAULT_PATH})\n  --out PATH        Output JSON path (default ${DEFAULT_OUT})\n  --keep-temp       Keep temporary sidecar directory\n  -h, --help        Show help\n`);
}

function parseArgs(argv) {
  const args = { file: DEFAULT_PATH, out: DEFAULT_OUT, keepTemp: false };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    const next = () => { if (i + 1 >= argv.length) throw new Error(`${arg} requires a value`); return argv[++i]; };
    if (arg === '--file') args.file = next();
    else if (arg === '--out') args.out = next();
    else if (arg === '--keep-temp') args.keepTemp = true;
    else if (arg === '-h' || arg === '--help') args.help = true;
    else throw new Error(`Unknown option: ${arg}`);
  }
  return args;
}

function contentTypeFor(path) {
  const ext = extname(path.endsWith('.br') || path.endsWith('.gz') ? path.slice(0, path.lastIndexOf('.')) : path);
  if (ext === '.wasm') return 'application/wasm';
  if (ext === '.js' || ext === '.mjs') return 'text/javascript; charset=utf-8';
  if (ext === '.json') return 'application/json; charset=utf-8';
  return 'application/octet-stream';
}

async function compressSidecars(file) {
  await pipeline(createReadStream(file), createBrotliCompress(), await import('node:fs').then(({ createWriteStream }) => createWriteStream(`${file}.br`)));
  await pipeline(createReadStream(file), createGzip({ level: 9 }), await import('node:fs').then(({ createWriteStream }) => createWriteStream(`${file}.gz`)));
}

function accepts(req, encoding) {
  return String(req.headers['accept-encoding'] ?? '').split(',').map((part) => part.trim().toLowerCase().split(';')[0]).includes(encoding);
}

function startServer(root, fileName) {
  const server = createServer(async (req, res) => {
    const raw = req.url?.split('?')[0] ?? '/';
    const name = raw === '/' ? fileName : raw.replace(/^\/+/, '');
    if (name !== fileName) {
      res.writeHead(404);
      res.end('not found');
      return;
    }
    let selected = join(root, fileName);
    let encoding = null;
    if (accepts(req, 'br')) { selected = `${selected}.br`; encoding = 'br'; }
    else if (accepts(req, 'gzip')) { selected = `${selected}.gz`; encoding = 'gzip'; }
    const stats = await stat(selected);
    res.setHeader('Content-Type', contentTypeFor(fileName));
    res.setHeader('Content-Length', String(stats.size));
    res.setHeader('Cache-Control', 'public, max-age=31536000, immutable');
    if (encoding) {
      res.setHeader('Content-Encoding', encoding);
      res.setHeader('Vary', 'Accept-Encoding');
    }
    if (req.method === 'HEAD') {
      res.end();
      return;
    }
    createReadStream(selected).pipe(res);
  });
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', () => resolve(server));
  });
}

async function probe(baseUrl, acceptEncoding) {
  const response = await fetch(baseUrl, { method: 'HEAD', headers: { 'accept-encoding': acceptEncoding } });
  return {
    ok: response.ok,
    status: response.status,
    acceptEncoding: acceptEncoding || 'identity',
    contentEncoding: response.headers.get('content-encoding') ?? undefined,
    contentType: response.headers.get('content-type') ?? undefined,
    contentLength: Number(response.headers.get('content-length') ?? 0) || undefined,
    cacheControl: response.headers.get('cache-control') ?? undefined,
    vary: response.headers.get('vary') ?? undefined,
  };
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) return usage();
  const source = resolve(args.file);
  const sourceStats = await stat(source);
  const tempRoot = await mkdtemp(join(tmpdir(), 'lc0-static-encoding-'));
  const fileName = source.split(/[\\/]/).pop();
  const tempFile = join(tempRoot, fileName);
  await writeFile(tempFile, await readFile(source));
  await compressSidecars(tempFile);
  const server = await startServer(tempRoot, fileName);
  try {
    const { port } = server.address();
    const url = `http://127.0.0.1:${port}/${encodeURIComponent(fileName)}`;
    const probes = [
      await probe(url, 'br, gzip'),
      await probe(url, 'gzip'),
      await probe(url, 'identity'),
    ];
    const byAccept = Object.fromEntries(probes.map((entry) => [entry.acceptEncoding, entry]));
    const ok = byAccept['br, gzip']?.contentEncoding === 'br'
      && byAccept.gzip?.contentEncoding === 'gzip'
      && byAccept.identity?.contentEncoding === undefined
      && probes.every((entry) => entry.ok && entry.contentType === contentTypeFor(fileName));
    const sidecarStats = {
      rawBytes: sourceStats.size,
      brBytes: (await stat(`${tempFile}.br`)).size,
      gzipBytes: (await stat(`${tempFile}.gz`)).size,
    };
    const artifact = {
      schema: 'lc0_browser.static_content_encoding_smoke.v1',
      generatedAt: new Date().toISOString(),
      ok,
      source,
      fileName,
      sidecarStats,
      probes,
      caveat: 'Local static-server smoke only. Production remains dependent on host/CDN metadata, rewrites, and Content-Encoding behavior for the exact published path.',
      tempRoot: args.keepTemp ? tempRoot : undefined,
    };
    await mkdir(dirname(args.out), { recursive: true });
    await writeFile(args.out, `${JSON.stringify(artifact, null, 2)}\n`);
    console.log(JSON.stringify(artifact, null, 2));
    if (!ok) process.exitCode = 1;
  } finally {
    await new Promise((resolve) => server.close(resolve));
    if (!args.keepTemp) await rm(tempRoot, { recursive: true, force: true });
  }
}

main().catch((error) => {
  console.error(error?.stack || error?.message || String(error));
  process.exit(1);
});
