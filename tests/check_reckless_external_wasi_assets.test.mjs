import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { mkdir, mkdtemp, readFile, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { test } from 'node:test';
import { checkRecklessExternalWasiAssets } from '../scripts/check_reckless_external_wasi_assets.mjs';

test('Reckless external WASI prototype has explicit build, release, publish, and manifest coverage', async () => {
  const packageJson = JSON.parse(await readFile('package.json', 'utf8'));
  assert.match(packageJson.scripts['reckless:build-simd-wasi-external'], /RECKLESS_WASI_EXTERNAL_NNUE=1/);
  assert.match(packageJson.scripts['reckless:build-simd-wasi-external'], /reckless-simd128-external\.wasm/);
  assert.equal(packageJson.scripts['reckless:check-simd-wasi-external'], 'node scripts/check_reckless_external_wasi_assets.mjs');
  assert.equal(packageJson.scripts['reckless:artifact-manifest'], 'node scripts/write_engine_artifact_manifest.mjs reckless');
  assert.match(packageJson.scripts['reckless:release-manifest'], /write_engine_artifact_manifest\.mjs reckless/);
  assert.match(packageJson.scripts['reckless:release-manifest'], /public\/reckless\/reckless-wasip1\.manifest\.json/);
  assert.match(packageJson.scripts['reckless:build-release'], /reckless:build-browser-api-simd-external/);
  assert.match(packageJson.scripts['reckless:build-release'], /reckless:release-manifest$/);

  const releaseBuilder = await readFile('scripts/build_reckless_release_assets.mjs', 'utf8');
  assert.match(releaseBuilder, /reckless-simd128-external\.wasm/);
  assert.match(releaseBuilder, /reckless-simd128-external-corresponding-source\.tar\.gz/);
  assert.match(releaseBuilder, /RECKLESS_WASI_EXTERNAL_NNUE: '1'/);
  assert.match(releaseBuilder, /explicit non-default prototype/);

  const manifestWriter = await readFile('scripts/write_engine_artifact_manifest.mjs', 'utf8');
  assert.match(manifestWriter, /command: 'npm run reckless:build-release'/);
  assert.match(manifestWriter, /public\/reckless\/reckless-simd128-external\.wasm/);
  assert.match(manifestWriter, /public\/reckless\/reckless-simd128-external-corresponding-source\.tar\.gz/);

  const publisher = await readFile('scripts/r2_brotli_publish_assets.mjs', 'utf8');
  assert.match(publisher, /reckless-simd128-external\.wasm/);
  assert.match(publisher, /reckless-v60-7f587dfb\.nnue/);
  assert.match(publisher, /reckless-simd128-external-corresponding-source\.tar\.gz/);

  const hostedArtifacts = await readFile('docs/hosted_artifacts.md', 'utf8');
  assert.match(hostedArtifacts, /Production must use canonical `npm run reckless:build-release`/);
  assert.match(hostedArtifacts, /default ladder is relaxed SIMD > fixed SIMD > scalar/);
  assert.match(hostedArtifacts, /include all three default-ladder WASM assets \(including relaxed SIMD\)/);

  const benchmarkNotes = await readFile('docs/reckless_browser_benchmarks.md', 'utf8');
  assert.match(benchmarkNotes, /release pipeline running `npm run reckless:build-release`/);
  assert.match(benchmarkNotes, /relaxed SIMD > fixed SIMD > scalar WASI\/UCI production ladder/);
  assert.match(benchmarkNotes, /`reckless-relaxed-simd128\.wasm`, `reckless-simd128\.wasm`, and `reckless\.wasm`/);

  const engineCatalog = await readFile('docs/engine_catalog.md', 'utf8');
  assert.match(engineCatalog, /^npm run reckless:build-release && npm run reckless:build-lite-wasi$/m);
  assert.match(engineCatalog, /default speed ladder is relaxed SIMD > fixed SIMD > scalar WASI\/UCI/);
  assert.match(engineCatalog, /`npm run reckless:build-release` is the canonical release build/);
});

test('Reckless release manifest command generates complete external artifact metadata', async () => {
  const root = await mkdtemp(join(tmpdir(), 'reckless-release-manifest-'));
  const artifacts = [
    'reckless.wasm',
    'reckless-simd128.wasm',
    'reckless-simd128-external.wasm',
    'reckless-relaxed-simd128.wasm',
    'reckless-browser-api.wasm',
    'reckless-browser-api-simd128.wasm',
    'reckless-browser-api-simd128-external.wasm',
    'reckless-v60-7f587dfb.nnue',
    'reckless-scalar-corresponding-source.tar.gz',
    'reckless-simd128-corresponding-source.tar.gz',
    'reckless-simd128-external-corresponding-source.tar.gz',
    'reckless-relaxed-simd128-corresponding-source.tar.gz',
  ];
  await mkdir(join(root, 'public/reckless'), { recursive: true });
  await Promise.all(artifacts.map((name, index) => writeFile(join(root, 'public/reckless', name), `artifact-${index}`)));

  const manifestPath = join(root, 'public/reckless/reckless-wasip1.manifest.json');
  const result = spawnSync(process.execPath, [
    resolve('scripts/write_engine_artifact_manifest.mjs'),
    'reckless',
    '--source-archive', 'public/reckless/reckless-scalar-corresponding-source.tar.gz',
    '--source-url', '/reckless/reckless-scalar-corresponding-source.tar.gz',
    '--out', manifestPath,
  ], { cwd: root, encoding: 'utf8' });
  assert.equal(result.status, 0, result.stderr);

  const manifest = JSON.parse(await readFile(manifestPath, 'utf8'));
  assert.equal(manifest.build.command, 'npm run reckless:build-release');
  assert.deepEqual(
    manifest.artifacts.filter((artifact) => artifact.path.includes('simd128-external')).map((artifact) => artifact.path),
    [
      'public/reckless/reckless-simd128-external.wasm',
      'public/reckless/reckless-browser-api-simd128-external.wasm',
      'public/reckless/reckless-simd128-external-corresponding-source.tar.gz',
    ],
  );
  assert.equal(manifest.artifacts.some((artifact) => artifact.missing), false);
  assert.equal(manifest.sourceArchive.url, '/reckless/reckless-scalar-corresponding-source.tar.gz');
});

test('Reckless external WASI asset check rejects an incomplete prototype package', async () => {
  const root = await mkdtemp(join(tmpdir(), 'reckless-external-wasi-check-'));
  await assert.rejects(
    checkRecklessExternalWasiAssets(root),
    /missing Reckless external WASI prototype assets: public\/reckless\/reckless-simd128-external\.wasm, public\/reckless\/reckless-v60-7f587dfb\.nnue, public\/reckless\/reckless-simd128-external-corresponding-source\.tar\.gz/,
  );
});
