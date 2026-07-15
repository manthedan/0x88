#!/usr/bin/env node
import { createHash } from 'node:crypto';
import { createReadStream, existsSync, readFileSync, statSync } from 'node:fs';
import { join, resolve } from 'node:path';

const EXPECTED_NNUE_SHA256 = '7f587dfb1fe5d74d53909328afa6fd51650c8c7f45907602db7fbb1e52948c61';
const REQUIRED_ARTIFACTS = [
  'public/reckless/reckless-simd128-external.wasm',
  'public/reckless/reckless-v60-7f587dfb.nnue',
  'public/reckless/reckless-simd128-external-corresponding-source.tar.gz',
];

function sha256(path) {
  return new Promise((resolveHash, reject) => {
    const hash = createHash('sha256');
    createReadStream(path)
      .on('data', (chunk) => hash.update(chunk))
      .on('error', reject)
      .on('end', () => resolveHash(hash.digest('hex')));
  });
}

export async function checkRecklessExternalWasiAssets(rootPath = '.') {
  const root = resolve(rootPath);
  for (const relativePath of REQUIRED_ARTIFACTS) {
    const path = join(root, relativePath);
    if (!existsSync(path) || statSync(path).size === 0) {
      throw new Error(`missing Reckless external WASI prototype asset: ${relativePath}`);
    }
  }

  const wasmPath = join(root, REQUIRED_ARTIFACTS[0]);
  const wasmBytes = readFileSync(wasmPath);
  if (!WebAssembly.validate(wasmBytes)) {
    throw new Error(`invalid WebAssembly module: ${REQUIRED_ARTIFACTS[0]}`);
  }

  const nnuePath = join(root, REQUIRED_ARTIFACTS[1]);
  const nnueSha256 = await sha256(nnuePath);
  if (nnueSha256 !== EXPECTED_NNUE_SHA256) {
    throw new Error(`Reckless external NNUE checksum mismatch: expected ${EXPECTED_NNUE_SHA256}, got ${nnueSha256}`);
  }

  const manifestPath = join(root, 'public/reckless/reckless-wasip1.manifest.json');
  if (!existsSync(manifestPath)) throw new Error('missing Reckless artifact manifest: public/reckless/reckless-wasip1.manifest.json');
  const manifest = JSON.parse(readFileSync(manifestPath, 'utf8'));
  const manifestArtifacts = new Set((manifest.artifacts ?? []).map((artifact) => artifact.path));
  const missingManifestArtifacts = REQUIRED_ARTIFACTS.filter((path) => !manifestArtifacts.has(path));
  if (missingManifestArtifacts.length) {
    throw new Error(`Reckless artifact manifest is missing external WASI prototype files: ${missingManifestArtifacts.join(', ')}`);
  }

  return {
    status: 'RECKLESS_EXTERNAL_WASI_ASSETS_OK',
    wasmBytes: wasmBytes.byteLength,
    nnueBytes: statSync(nnuePath).size,
    nnueSha256,
    sourceArchiveBytes: statSync(join(root, REQUIRED_ARTIFACTS[2])).size,
  };
}

if (import.meta.url === `file://${process.argv[1]}`) {
  try {
    console.log(JSON.stringify(await checkRecklessExternalWasiAssets(process.argv[2] ?? '.'), null, 2));
  } catch (error) {
    console.error(error instanceof Error ? error.message : error);
    process.exitCode = 1;
  }
}
