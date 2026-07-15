import assert from 'node:assert/strict';
import { mkdtemp, readFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import { checkRecklessExternalWasiAssets } from '../scripts/check_reckless_external_wasi_assets.mjs';

test('Reckless external WASI prototype has explicit build, release, publish, and manifest coverage', async () => {
  const packageJson = JSON.parse(await readFile('package.json', 'utf8'));
  assert.match(packageJson.scripts['reckless:build-simd-wasi-external'], /RECKLESS_WASI_EXTERNAL_NNUE=1/);
  assert.match(packageJson.scripts['reckless:build-simd-wasi-external'], /reckless-simd128-external\.wasm/);
  assert.equal(packageJson.scripts['reckless:check-simd-wasi-external'], 'node scripts/check_reckless_external_wasi_assets.mjs');

  const releaseBuilder = await readFile('scripts/build_reckless_release_assets.mjs', 'utf8');
  assert.match(releaseBuilder, /reckless-simd128-external\.wasm/);
  assert.match(releaseBuilder, /reckless-simd128-external-corresponding-source\.tar\.gz/);
  assert.match(releaseBuilder, /RECKLESS_WASI_EXTERNAL_NNUE: '1'/);
  assert.match(releaseBuilder, /explicit non-default prototype/);

  const manifestWriter = await readFile('scripts/write_engine_artifact_manifest.mjs', 'utf8');
  assert.match(manifestWriter, /public\/reckless\/reckless-simd128-external\.wasm/);
  assert.match(manifestWriter, /public\/reckless\/reckless-simd128-external-corresponding-source\.tar\.gz/);

  const publisher = await readFile('scripts/r2_brotli_publish_assets.mjs', 'utf8');
  assert.match(publisher, /reckless-simd128-external\.wasm/);
  assert.match(publisher, /reckless-v60-7f587dfb\.nnue/);
  assert.match(publisher, /reckless-simd128-external-corresponding-source\.tar\.gz/);
});

test('Reckless external WASI asset check rejects an incomplete prototype package', async () => {
  const root = await mkdtemp(join(tmpdir(), 'reckless-external-wasi-check-'));
  await assert.rejects(
    checkRecklessExternalWasiAssets(root),
    /missing Reckless external WASI prototype asset: public\/reckless\/reckless-simd128-external\.wasm/,
  );
});
