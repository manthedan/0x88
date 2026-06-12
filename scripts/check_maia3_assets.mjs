#!/usr/bin/env node
import { createHash } from 'node:crypto';
import { readFile, realpath, lstat, access } from 'node:fs/promises';
import { constants } from 'node:fs';

const MANIFEST = 'public/models/maia3/manifest.json';
const PROVENANCE = 'docs/model_provenance/maia3.md';
const README = 'public/models/maia3/README.md';

async function sha256(path) {
  const bytes = await readFile(path);
  return createHash('sha256').update(bytes).digest('hex');
}

async function main() {
  const manifest = JSON.parse(await readFile(MANIFEST, 'utf8'));
  const entries = Array.isArray(manifest.models) ? manifest.models : [];
  if (entries.length !== 1) throw new Error(`Expected exactly one Maia3 model entry in ${MANIFEST}`);
  const entry = entries[0];
  const localPath = `public/models/maia3/${entry.file}`;
  const stat = await lstat(localPath);
  const resolved = await realpath(localPath);
  await access(resolved, constants.R_OK);
  const bytes = await readFile(resolved);
  const digest = createHash('sha256').update(bytes).digest('hex');
  const errors = [];
  if (entry.bytes !== bytes.byteLength) errors.push(`byte mismatch: manifest=${entry.bytes} actual=${bytes.byteLength}`);
  if (entry.sha256 !== digest) errors.push(`sha256 mismatch: manifest=${entry.sha256} actual=${digest}`);
  if (!/^\/models\/maia3\//.test(entry.url)) errors.push(`unexpected public URL: ${entry.url}`);
  const provenance = await readFile(PROVENANCE, 'utf8');
  const readme = await readFile(README, 'utf8');
  for (const token of ['CSSLab/maia3', 'CSSLab/maia-platform-frontend', entry.sha256, 'AGPL', 'GPL']) {
    if (!provenance.includes(token)) errors.push(`${PROVENANCE} missing ${token}`);
  }
  for (const token of ['CSSLab/maia3', 'CSSLab/maia-platform-frontend', 'AGPL', 'GPL']) {
    if (!readme.includes(token)) errors.push(`${README} missing ${token}`);
  }
  const result = {
    ok: errors.length === 0,
    manifest: MANIFEST,
    localPath,
    symlink: stat.isSymbolicLink(),
    resolved,
    bytes: bytes.byteLength,
    sha256: digest,
    errors,
  };
  if (!result.ok) throw new Error(JSON.stringify(result, null, 2));
  console.log(JSON.stringify(result, null, 2));
}

main().catch((error) => {
  console.error(error instanceof Error ? error.stack ?? error.message : error);
  process.exitCode = 1;
});
