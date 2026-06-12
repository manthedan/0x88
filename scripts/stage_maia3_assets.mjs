#!/usr/bin/env node
// Stage the Maia3 models for a clean checkout.
//
// The repo commits only symlinks (public/models/maia3/*.onnx ->
// ../../../../models/maia3/...) plus a manifest with expected SHA-256s.
// Two artifacts:
//  - maia3_simplified.onnx: downloaded byte-identical from the pinned
//    upstream commit of CSSLab/maia-platform-frontend.
//  - maia3_simplified.qdq8.onnx (the default browser model): derived locally
//    from the fp16 file (opset 17->21 version_converter, then weight-only
//    int8 QDQ via scripts/lc0_quantize_onnx_weights_qdq.py). Needs a python
//    with onnx+numpy (.venv-onnx by default); if unavailable the fp16 model
//    still works — the browser loader falls back to it.
//
//   npm run maia3:stage-assets
//
// Env overrides: MAIA3_MODEL_DIR, MAIA3_SOURCE_URL, MAIA3_PYTHON.
import { createHash } from 'node:crypto';
import { spawnSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import { mkdir, readFile, readlink, rename, stat, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';

const MANIFEST = 'public/models/maia3/manifest.json';
// Pinned upstream commit: "Switch to fp16 ONNX model (87MB -> 44MB)",
// CSSLab/maia-platform-frontend, 2026-03-27. Must match the manifest sha256.
const UPSTREAM_COMMIT = '0013cc8e6ec52c88f5b3d694781d4cc8427cb91a';

const manifest = JSON.parse(await readFile(MANIFEST, 'utf8'));
const entries = manifest.models ?? [];
const upstream = entries.find((entry) => !entry.derivedFrom);
const derived = entries.filter((entry) => entry.derivedFrom);
if (!upstream) throw new Error(`no upstream (non-derived) model entry in ${MANIFEST}`);

const linkPath = `public/models/maia3/${upstream.file}`;
const linkTarget = await readlink(linkPath).catch(() => null);
const defaultDir = linkTarget ? path.dirname(path.resolve(path.dirname(linkPath), linkTarget)) : '../models/maia3';
const targetDir = process.env.MAIA3_MODEL_DIR ?? defaultDir;
const python = process.env.MAIA3_PYTHON ?? '.venv-onnx/bin/python';

async function sha256(filePath) {
  return createHash('sha256').update(await readFile(filePath)).digest('hex');
}

async function isStaged(entry) {
  const filePath = path.join(targetDir, entry.file);
  const existing = await stat(filePath).catch(() => null);
  return existing !== null && (await sha256(filePath)) === entry.sha256;
}

const actions = [];

// 1. Upstream fp16: download from the pinned commit if missing/mismatched.
if (await isStaged(upstream)) {
  actions.push({ file: upstream.file, action: 'already-staged' });
} else {
  const sourceUrl = process.env.MAIA3_SOURCE_URL
    ?? `https://raw.githubusercontent.com/CSSLab/maia-platform-frontend/${UPSTREAM_COMMIT}/public/maia3/${upstream.file}`;
  console.log(`Downloading Maia3 model (${(upstream.bytes / 1e6).toFixed(1)}MB) from ${sourceUrl}`);
  const response = await fetch(sourceUrl);
  if (!response.ok) throw new Error(`download failed: HTTP ${response.status} for ${sourceUrl}`);
  const bytes = new Uint8Array(await response.arrayBuffer());
  if (bytes.byteLength !== upstream.bytes) throw new Error(`byte mismatch: expected ${upstream.bytes}, got ${bytes.byteLength}`);
  const digest = createHash('sha256').update(bytes).digest('hex');
  if (digest !== upstream.sha256) throw new Error(`sha256 mismatch: expected ${upstream.sha256}, got ${digest} — upstream file changed; do not stage`);
  await mkdir(targetDir, { recursive: true });
  const targetPath = path.join(targetDir, upstream.file);
  await writeFile(`${targetPath}.tmp`, bytes);
  await rename(`${targetPath}.tmp`, targetPath);
  actions.push({ file: upstream.file, action: 'downloaded', upstreamCommit: UPSTREAM_COMMIT });
}

// 2. Derived QDQ artifacts: re-derive locally when missing/mismatched.
for (const entry of derived) {
  if (await isStaged(entry)) {
    actions.push({ file: entry.file, action: 'already-staged' });
    continue;
  }
  if (!existsSync(python)) {
    actions.push({ file: entry.file, action: 'SKIPPED', reason: `python not found at ${python} (set MAIA3_PYTHON); browser falls back to ${entry.derivedFrom}` });
    continue;
  }
  const sourcePath = path.join(targetDir, entry.derivedFrom);
  const targetPath = path.join(targetDir, entry.file);
  const op21Path = path.join(tmpdir(), `maia3_op21_${process.pid}.onnx`);
  console.log(`Deriving ${entry.file} from ${entry.derivedFrom} (opset 21 + int8 QDQ)…`);
  const convert = spawnSync(python, ['-c', `
import onnx
from onnx import version_converter
m = onnx.load(${JSON.stringify(sourcePath)})
onnx.save(version_converter.convert_version(m, 21), ${JSON.stringify(op21Path)})
`], { stdio: 'inherit' });
  if (convert.status !== 0) throw new Error(`opset conversion failed for ${entry.file}`);
  const quantize = spawnSync(python, ['scripts/lc0_quantize_onnx_weights_qdq.py', '--in', op21Path, '--out', targetPath], { stdio: 'inherit' });
  if (quantize.status !== 0) throw new Error(`quantization failed for ${entry.file}`);
  const digest = await sha256(targetPath);
  if (digest !== entry.sha256) {
    throw new Error(`${entry.file}: derived sha256 ${digest} does not match manifest ${entry.sha256} — onnx/numpy version drift? Re-derive and update the manifest deliberately.`);
  }
  actions.push({ file: entry.file, action: 'derived', sha256: digest });
}

console.log(JSON.stringify({ ok: true, targetDir, actions }, null, 2));
console.log('Now run: npm run maia3:check-assets');
