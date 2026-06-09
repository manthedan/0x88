#!/usr/bin/env node
import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { brotliCompressSync, constants as zlibConstants, gzipSync } from 'node:zlib';

const DEFAULT_MANIFEST = 'public/runtimes/lc0-tvmjs-webgpu/t1-256x10-distilled-swa-2432500/f16/v1/manifest.json';

function usage() {
  console.log(`Usage: node scripts/summarize_lc0_tvmjs_bundle_footprint.mjs [options]\n\nSummarizes raw/gzip/Brotli transfer-footprint estimates for a staged LC0 TVMJS/WebGPU manifest.\nThis writes a JSON sidecar only; it does not publish generated artifacts or create compressed payload files.\n\nOptions:\n  --manifest PATH       Staged TVMJS runtime manifest (default ${DEFAULT_MANIFEST})\n  --out PATH            JSON footprint sidecar path (default <manifest-dir>/bundle-footprint.json)\n  --gzip-level N        gzip level, 1-9 (default 9)\n  --brotli-quality N    Brotli quality, 0-11 (default 11)\n  -h, --help            Show help\n`);
}

function parseArgs(argv) {
  const args = { manifest: DEFAULT_MANIFEST, out: '', gzipLevel: 9, brotliQuality: 11 };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    const next = () => { if (i + 1 >= argv.length) throw new Error(`${arg} requires a value`); return argv[++i]; };
    if (arg === '--manifest') args.manifest = next();
    else if (arg === '--out') args.out = next();
    else if (arg === '--gzip-level') args.gzipLevel = Number(next());
    else if (arg === '--brotli-quality') args.brotliQuality = Number(next());
    else if (arg === '-h' || arg === '--help') args.help = true;
    else throw new Error(`Unknown option: ${arg}`);
  }
  if (!Number.isInteger(args.gzipLevel) || args.gzipLevel < 1 || args.gzipLevel > 9) throw new Error(`Invalid --gzip-level ${args.gzipLevel}`);
  if (!Number.isInteger(args.brotliQuality) || args.brotliQuality < 0 || args.brotliQuality > 11) throw new Error(`Invalid --brotli-quality ${args.brotliQuality}`);
  args.out ||= join(dirname(args.manifest), 'bundle-footprint.json');
  return args;
}

function sha256(buffer) {
  return createHash('sha256').update(buffer).digest('hex');
}

function ratio(numerator, denominator) {
  return denominator ? Number((numerator / denominator).toFixed(4)) : null;
}

function classify(path, manifest) {
  if (path === manifest.runtime?.tvmjsBundle) return 'runtime-js';
  if (path === manifest.runtime?.tvmjsRuntimeWasm) return 'runtime-wasm';
  if (path.endsWith('.tvmjs.wasm')) return 'model-wasm';
  if (path.endsWith('.probe.json')) return 'probe-json';
  return 'other';
}

function sum(values, key) {
  return values.reduce((total, item) => total + (item[key] ?? 0), 0);
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) return usage();
  const manifest = JSON.parse(await readFile(args.manifest, 'utf8'));
  const manifestDir = dirname(args.manifest);
  const files = [];
  for (const entry of manifest.files ?? []) {
    if (!entry.path) continue;
    const path = join(manifestDir, entry.path);
    const raw = await readFile(path);
    const gzip = gzipSync(raw, { level: args.gzipLevel });
    const brotli = brotliCompressSync(raw, { params: { [zlibConstants.BROTLI_PARAM_QUALITY]: args.brotliQuality } });
    const rawBytes = raw.byteLength;
    files.push({
      path: entry.path,
      kind: classify(entry.path, manifest),
      rawBytes,
      gzipBytes: gzip.byteLength,
      brotliBytes: brotli.byteLength,
      gzipRatio: ratio(gzip.byteLength, rawBytes),
      brotliRatio: ratio(brotli.byteLength, rawBytes),
      sha256: sha256(raw),
      manifestBytes: entry.bytes,
      manifestSha256: entry.sha256,
      manifestMatches: entry.bytes === rawBytes && entry.sha256 === sha256(raw),
    });
  }
  const rawBytes = sum(files, 'rawBytes');
  const gzipBytes = sum(files, 'gzipBytes');
  const brotliBytes = sum(files, 'brotliBytes');
  const byKind = {};
  for (const file of files) {
    byKind[file.kind] ??= { files: 0, rawBytes: 0, gzipBytes: 0, brotliBytes: 0 };
    byKind[file.kind].files += 1;
    byKind[file.kind].rawBytes += file.rawBytes;
    byKind[file.kind].gzipBytes += file.gzipBytes;
    byKind[file.kind].brotliBytes += file.brotliBytes;
  }
  for (const value of Object.values(byKind)) {
    value.gzipRatio = ratio(value.gzipBytes, value.rawBytes);
    value.brotliRatio = ratio(value.brotliBytes, value.rawBytes);
  }
  const result = {
    schema: 'lc0_browser.tvmjs_webgpu_bundle_footprint.v1',
    generatedAt: new Date().toISOString(),
    manifest: args.manifest,
    manifestSchema: manifest.schema,
    modelFamily: manifest.modelFamily,
    dtype: manifest.dtype,
    version: manifest.version,
    compression: {
      gzip: { level: args.gzipLevel, note: 'Estimated with Node zlib gzipSync; deployed transfer size depends on host/CDN configuration.' },
      brotli: { quality: args.brotliQuality, note: 'Estimated with Node zlib brotliCompressSync; deployed transfer size depends on host/CDN configuration.' },
    },
    totals: {
      files: files.length,
      rawBytes,
      gzipBytes,
      brotliBytes,
      gzipRatio: ratio(gzipBytes, rawBytes),
      brotliRatio: ratio(brotliBytes, rawBytes),
    },
    byKind,
    files,
    caveat: 'This is a research footprint sidecar. It does not publish artifacts and does not prove Content-Encoding behavior in production.',
  };
  await mkdir(dirname(args.out), { recursive: true });
  await writeFile(args.out, `${JSON.stringify(result, null, 2)}\n`);
  process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
}

main().catch((error) => {
  console.error(error?.stack ?? error?.message ?? String(error));
  process.exit(1);
});
