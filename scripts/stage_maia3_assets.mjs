#!/usr/bin/env node
import { spawnSync } from 'node:child_process';
// Stage the Maia3 models for a clean checkout.
//
// The repo commits only the manifest and provenance. Model files are staged in
// a sibling cache and exposed through ignored local symlinks.
// Three artifacts:
//  - maia3_simplified.onnx: downloaded byte-identical from the pinned
//    upstream commit of CSSLab/maia-platform-frontend.
//  - maia3_simplified.qdq8.onnx: derived locally from the fp16 file (opset
//    17->21 version_converter, then weight-only int8 QDQ via
//    scripts/lc0_quantize_onnx_weights_qdq.py).
//  - maia3_simplified.qdq8.graph.onnx (the default browser model): derived
//    from the QDQ8 file by scripts/maia3_graph_rewrite.py with the manifest
//    entry's `rewritePasses` (einsum,promo). Same weights and I/O, fewer
//    WebGPU dispatches; see docs/maia3_graph_rewrite_2026-11.md.
// Derived files need a python with onnx+numpy (.venv-onnx by default) and are
// checked against the manifest sha256. If one is unavailable the browser
// loader falls back along graph -> QDQ8 -> fp16.
//
//   npm run maia3:stage-assets
//
// Env overrides: MAIA3_MODEL_DIR, MAIA3_SOURCE_URL, MAIA3_PYTHON.
import { createHash } from 'node:crypto';
import { existsSync } from 'node:fs';
import { lstat, mkdir, readFile, readlink, rename, rm, stat, symlink, writeFile } from 'node:fs/promises';
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

const publicDir = 'public/models/maia3';
const targetDir = process.env.MAIA3_MODEL_DIR ?? '../models/maia3';
const python = process.env.MAIA3_PYTHON ?? '.venv-onnx/bin/python';

async function sha256(filePath) {
  return createHash('sha256')
    .update(await readFile(filePath))
    .digest('hex');
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
  const sourceUrl =
    process.env.MAIA3_SOURCE_URL ?? `https://raw.githubusercontent.com/CSSLab/maia-platform-frontend/${UPSTREAM_COMMIT}/public/maia3/${upstream.file}`;
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

// 2. Derived artifacts: re-derive locally when missing/mismatched, parents
// before children (the graph rewrite derives from the derived QDQ8 file).
async function deriveQdq8(entry, sourcePath, targetPath) {
  const op21Path = path.join(tmpdir(), `maia3_op21_${process.pid}.onnx`);
  console.log(`Deriving ${entry.file} from ${entry.derivedFrom} (opset 21 + int8 QDQ)…`);
  const convert = spawnSync(
    python,
    [
      '-c',
      `
import onnx
from onnx import version_converter
m = onnx.load(${JSON.stringify(sourcePath)})
onnx.save(version_converter.convert_version(m, 21), ${JSON.stringify(op21Path)})
`,
    ],
    { stdio: 'inherit' },
  );
  if (convert.status !== 0) throw new Error(`opset conversion failed for ${entry.file}`);
  const quantize = spawnSync(python, ['scripts/lc0_quantize_onnx_weights_qdq.py', '--in', op21Path, '--out', targetPath], { stdio: 'inherit' });
  await rm(op21Path, { force: true });
  if (quantize.status !== 0) throw new Error(`quantization failed for ${entry.file}`);
}

async function deriveGraphRewrite(entry, sourcePath, targetPath) {
  if (!/^[a-z,]+$/.test(entry.rewritePasses)) throw new Error(`${entry.file}: invalid rewritePasses ${JSON.stringify(entry.rewritePasses)}`);
  console.log(`Deriving ${entry.file} from ${entry.derivedFrom} (scripts/maia3_graph_rewrite.py --passes ${entry.rewritePasses})…`);
  const rewrite = spawnSync(python, ['-I', 'scripts/maia3_graph_rewrite.py', sourcePath, targetPath, '--passes', entry.rewritePasses], { stdio: 'inherit' });
  if (rewrite.status !== 0) throw new Error(`graph rewrite failed for ${entry.file}`);
}

const byFile = new Map(entries.map((entry) => [entry.file, entry]));
const derivedOrder = [];
const visiting = new Set();
function visit(entry) {
  if (derivedOrder.includes(entry)) return;
  if (visiting.has(entry.file)) throw new Error(`${MANIFEST}: derivedFrom cycle at ${entry.file}`);
  visiting.add(entry.file);
  const parent = byFile.get(entry.derivedFrom);
  if (!parent) throw new Error(`${entry.file}: derivedFrom ${entry.derivedFrom} is not in ${MANIFEST}`);
  if (parent.derivedFrom) visit(parent);
  visiting.delete(entry.file);
  derivedOrder.push(entry);
}
for (const entry of derived) visit(entry);

for (const entry of derivedOrder) {
  if (await isStaged(entry)) {
    actions.push({ file: entry.file, action: 'already-staged' });
    continue;
  }
  if (!existsSync(python)) {
    actions.push({
      file: entry.file,
      action: 'SKIPPED',
      reason: `python not found at ${python} (set MAIA3_PYTHON); the browser falls back to ${entry.derivedFrom}`,
    });
    continue;
  }
  if (!(await isStaged(byFile.get(entry.derivedFrom)))) {
    actions.push({ file: entry.file, action: 'SKIPPED', reason: `${entry.derivedFrom} is not staged` });
    continue;
  }
  const sourcePath = path.join(targetDir, entry.derivedFrom);
  const targetPath = path.join(targetDir, entry.file);
  // Write beside the target and rename only after the hash matches, so a
  // drifted derivation never leaves a wrong file under the published name.
  const partialPath = path.join(targetDir, `.${entry.file}.${process.pid}.partial.onnx`);
  const partialReport = partialPath.replace(/\.onnx$/, '.qdq-report.json');
  try {
    if (entry.rewritePasses) await deriveGraphRewrite(entry, sourcePath, partialPath);
    else await deriveQdq8(entry, sourcePath, partialPath);
    const digest = await sha256(partialPath);
    if (digest !== entry.sha256) {
      throw new Error(
        `${entry.file}: derived sha256 ${digest} does not match manifest ${entry.sha256} — onnx/numpy version drift? Re-derive and update the manifest deliberately.`,
      );
    }
    await rename(partialPath, targetPath);
    // The quantizer writes <out>.qdq-report.json beside its output; keep it under the final name.
    await rename(partialReport, targetPath.replace(/\.onnx$/, '.qdq-report.json')).catch(() => undefined);
    actions.push({ file: entry.file, action: 'derived', sha256: digest });
  } finally {
    await rm(partialPath, { force: true });
    await rm(partialReport, { force: true });
  }
}

const links = [];
for (const entry of entries) {
  if (!(await isStaged(entry))) continue;
  const linkPath = path.join(publicDir, entry.file);
  const targetPath = path.resolve(targetDir, entry.file);
  const relativeTarget = path.relative(path.dirname(linkPath), targetPath);
  const existing = await lstat(linkPath).catch(() => null);
  const existingTarget = existing?.isSymbolicLink() ? await readlink(linkPath) : null;
  if (!existing || existingTarget !== relativeTarget) {
    if (existing) await rm(linkPath, { force: true });
    await symlink(relativeTarget, linkPath);
  }
  links.push({ file: entry.file, target: relativeTarget });
}

console.log(JSON.stringify({ ok: true, targetDir, actions, links }, null, 2));
console.log('Now run: npm run maia3:check-assets');
