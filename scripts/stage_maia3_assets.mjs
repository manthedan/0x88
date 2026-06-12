#!/usr/bin/env node
// Stage the Maia3 model for a clean checkout.
//
// The repo commits only a symlink (public/models/maia3/maia3_simplified.onnx
// -> ../../../../models/maia3/...) plus a manifest with the expected SHA-256;
// this script downloads the model from the pinned upstream commit of
// CSSLab/maia-platform-frontend into the symlink target and verifies it.
//
//   npm run maia3:stage-assets
//
// Env overrides: MAIA3_MODEL_DIR (target directory), MAIA3_SOURCE_URL.
import { createHash } from 'node:crypto';
import { mkdir, readFile, readlink, rename, stat, writeFile } from 'node:fs/promises';
import path from 'node:path';

const MANIFEST = 'public/models/maia3/manifest.json';
// Pinned upstream commit: "Switch to fp16 ONNX model (87MB -> 44MB)",
// CSSLab/maia-platform-frontend, 2026-03-27. Must match the manifest sha256.
const UPSTREAM_COMMIT = '0013cc8e6ec52c88f5b3d694781d4cc8427cb91a';

const manifest = JSON.parse(await readFile(MANIFEST, 'utf8'));
const entry = manifest.models?.[0];
if (!entry) throw new Error(`no model entry in ${MANIFEST}`);

const linkPath = `public/models/maia3/${entry.file}`;
const linkTarget = await readlink(linkPath).catch(() => null);
const defaultDir = linkTarget ? path.dirname(path.resolve(path.dirname(linkPath), linkTarget)) : '../models/maia3';
const targetDir = process.env.MAIA3_MODEL_DIR ?? defaultDir;
const targetPath = path.join(targetDir, entry.file);
const sourceUrl = process.env.MAIA3_SOURCE_URL
  ?? `https://raw.githubusercontent.com/CSSLab/maia-platform-frontend/${UPSTREAM_COMMIT}/public/maia3/${entry.file}`;

async function sha256(filePath) {
  return createHash('sha256').update(await readFile(filePath)).digest('hex');
}

const existing = await stat(targetPath).catch(() => null);
if (existing && (await sha256(targetPath)) === entry.sha256) {
  console.log(JSON.stringify({ ok: true, action: 'already-staged', targetPath, sha256: entry.sha256 }, null, 2));
  process.exit(0);
}

console.log(`Downloading Maia3 model (${(entry.bytes / 1e6).toFixed(1)}MB) from ${sourceUrl}`);
const response = await fetch(sourceUrl);
if (!response.ok) throw new Error(`download failed: HTTP ${response.status} for ${sourceUrl}`);
const bytes = new Uint8Array(await response.arrayBuffer());
if (bytes.byteLength !== entry.bytes) {
  throw new Error(`byte mismatch: expected ${entry.bytes}, got ${bytes.byteLength}`);
}
const digest = createHash('sha256').update(bytes).digest('hex');
if (digest !== entry.sha256) {
  throw new Error(`sha256 mismatch: expected ${entry.sha256}, got ${digest} — upstream file changed; do not stage`);
}

await mkdir(targetDir, { recursive: true });
const tmpPath = `${targetPath}.tmp`;
await writeFile(tmpPath, bytes);
await rename(tmpPath, targetPath);
console.log(JSON.stringify({ ok: true, action: 'downloaded', targetPath, bytes: bytes.byteLength, sha256: digest, upstreamCommit: UPSTREAM_COMMIT }, null, 2));
console.log('Now run: npm run maia3:check-assets');
