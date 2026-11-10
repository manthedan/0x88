import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { readdirSync, readFileSync } from 'node:fs';
import test from 'node:test';
import { fileURLToPath } from 'node:url';
import { STOCKFISH_RELEASE } from '../scripts/stockfish_release_config.mjs';

// engines/<engine>/upstream.lock.json is the pin of record. The build scripts keep
// their own env-overridable defaults (`process.env.<ENGINE>_REF ?? '<sha>'`); the
// manifest and source-archive writers read the lock. These tests keep them equal.

const root = fileURLToPath(new URL('..', import.meta.url));
const read = (file) => readFileSync(new URL(`../${file}`, import.meta.url), 'utf8');
const ENGINES = ['berserk', 'plentychess', 'reckless', 'stockfish', 'stormphrax', 'viridithas'];
const locks = Object.fromEntries(ENGINES.map((engine) => [engine, JSON.parse(read(`engines/${engine}/upstream.lock.json`))]));
const MANIFESTS = {
  berserk: 'public/berserk/berserk-emscripten-single-thread.manifest.json',
  plentychess: 'public/plentychess/plentychess-emscripten-single-thread.manifest.json',
  reckless: 'public/reckless/reckless-wasip1.manifest.json',
  stockfish: `public/stockfish/stockfish-${STOCKFISH_RELEASE.version}.manifest.json`,
  stormphrax: 'public/stormphrax/stormphrax-emscripten-single-thread.manifest.json',
  viridithas: 'public/viridithas/viridithas-wasip1.manifest.json',
};

// Which lock field each env-overridable build default must equal.
const BUILD_DEFAULTS = {
  REPO: (lock) => lock.upstream.repo,
  REF: (lock) => lock.upstream.ref,
  NET_SHA256: (lock) => lock.network.sha256,
  PROCESSED_SHA256: (lock) => lock.network.processedSha256,
};

test('every engine in engines/ has a lock with repo, ref and license', () => {
  const dirs = readdirSync(new URL('../engines/', import.meta.url), { withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .map((entry) => entry.name)
    .sort();
  assert.deepEqual(dirs, ENGINES);
  for (const [engine, lock] of Object.entries(locks)) {
    assert.match(lock.upstream.repo, /^https:\/\/github\.com\/.+\.git$/, engine);
    assert.match(lock.upstream.ref, /^[0-9a-f]{40}$/, engine);
    assert.ok(lock.upstream.license, engine);
  }
});

test('every pin hard-coded in scripts/build_* equals its engine lock', () => {
  const scripts = readdirSync(new URL('../scripts/', import.meta.url)).filter((name) => /^build_.*\.mjs$/.test(name));
  let checked = 0;
  for (const name of scripts) {
    const source = read(`scripts/${name}`);
    for (const [, prefix, key, value] of source.matchAll(/process\.env\.([A-Z]+)_(REPO|REF|NET_SHA256|PROCESSED_SHA256) \?\? '([^']+)'/g)) {
      const engine = prefix.toLowerCase();
      if (!locks[engine]) continue;
      assert.equal(value, BUILD_DEFAULTS[key](locks[engine]), `scripts/${name}: ${prefix}_${key} default differs from engines/${engine}/upstream.lock.json`);
      checked++;
    }
  }
  // Each non-Stockfish engine pins at least its repo and ref in some build script.
  for (const engine of ENGINES.filter((name) => name !== 'stockfish')) {
    const pinned = scripts.some((name) => read(`scripts/${name}`).includes(`process.env.${engine.toUpperCase()}_REF ?? '${locks[engine].upstream.ref}'`));
    assert.ok(pinned, `no scripts/build_* pins ${engine}_REF`);
  }
  assert.ok(checked >= 15, `only ${checked} build defaults checked`);
});

test('the Stockfish lock mirrors scripts/stockfish_release_config.mjs', () => {
  const { upstream, network, toolchain } = locks.stockfish;
  assert.deepEqual(
    { ...upstream },
    {
      repo: STOCKFISH_RELEASE.repo,
      ref: STOCKFISH_RELEASE.commit,
      tag: STOCKFISH_RELEASE.tag,
      version: STOCKFISH_RELEASE.version,
      license: STOCKFISH_RELEASE.license,
      nativeTag: STOCKFISH_RELEASE.nativeTag,
      nativeRef: STOCKFISH_RELEASE.nativeCommit,
    },
  );
  assert.deepEqual(
    network.map(({ variant, name, sha256 }) => ({ variant, name, sha256 })),
    STOCKFISH_RELEASE.nets,
  );
  assert.equal(toolchain, STOCKFISH_RELEASE.toolchain);
});

test('committed engine manifests record the locked upstream pin and network digests', () => {
  for (const [engine, path] of Object.entries(MANIFESTS)) {
    const manifest = JSON.parse(read(path));
    const lock = locks[engine];
    assert.equal(manifest.upstream.repo, lock.upstream.repo, engine);
    assert.equal(manifest.upstream.commit, lock.upstream.ref, engine);
    assert.equal(manifest.upstream.license, lock.upstream.license, engine);
    const digests = JSON.stringify(manifest.assets);
    for (const net of [lock.network ?? []].flat())
      for (const digest of [net.sha256, net.processedSha256].filter(Boolean)) assert.ok(digests.includes(digest), `${engine}: manifest assets lack ${digest}`);
  }
});

test('the manifest and source-archive writers take upstream pins from the locks, not literals', () => {
  for (const script of ['scripts/write_engine_artifact_manifest.mjs', 'scripts/write_engine_source_archive.mjs']) {
    const source = read(script);
    for (const [engine, lock] of Object.entries(locks)) assert.ok(!source.includes(lock.upstream.ref), `${script} hard-codes the ${engine} ref`);
  }
  for (const engine of ENGINES.filter((name) => name !== 'reckless')) {
    const result = spawnSync(process.execPath, ['scripts/write_engine_source_archive.mjs', engine, '--dry-run'], { cwd: root, encoding: 'utf8' });
    assert.equal(result.status, 0, result.stderr);
    const pin = JSON.parse(result.stdout);
    assert.equal(pin.upstreamRepo, locks[engine].upstream.repo, engine);
    assert.equal(pin.upstreamCommit, locks[engine].upstream.ref, engine);
    assert.ok(pin.sourcePrefix.endsWith(`-${locks[engine].upstream.ref}/`), engine);
  }
});
