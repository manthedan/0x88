import assert from 'node:assert/strict';
import { readFileSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
import { test } from 'node:test';

/**
 * Committed engine release manifests carry a `totals` block that the generator
 * derives by summing the artifact list. Nothing recomputed it when artifacts
 * were edited by hand, so removing the duplicate per-variant `.data` entries
 * left PlentyChess claiming 190 MB against 64 MB of actual artifacts — a
 * manifest that is internally inconsistent and overstates the deployed
 * footprint to anything that trusts it. Recompute the same way the generator
 * does (scripts/write_engine_artifact_manifest.mjs) and compare.
 */

const ENGINE_DIRS = ['berserk', 'plentychess', 'stormphrax', 'reckless', 'stockfish', 'viridithas'];

function committedManifests() {
  const found = [];
  for (const dir of ENGINE_DIRS) {
    let entries;
    try {
      entries = readdirSync(new URL(`../public/${dir}/`, import.meta.url));
    } catch {
      continue; // engine directory absent in this checkout
    }
    for (const name of entries) {
      if (!name.endsWith('.manifest.json')) continue;
      found.push([`${dir}/${name}`, join(new URL(`../public/${dir}/`, import.meta.url).pathname, name)]);
    }
  }
  return found;
}

const MANIFESTS = committedManifests();

test('there are committed engine manifests to check', () => {
  assert.ok(MANIFESTS.length > 0, 'expected at least one public/*/**.manifest.json');
});

for (const [label, path] of MANIFESTS) {
  test(`${label} totals match its artifact list`, () => {
    const manifest = JSON.parse(readFileSync(path, 'utf8'));
    const artifacts = manifest.artifacts;
    if (!Array.isArray(artifacts) || !artifacts.length || !manifest.totals) return; // metadata-only manifest
    const totals = manifest.totals;

    const bytes = artifacts.reduce((sum, a) => sum + (a.bytes ?? 0), 0);
    assert.equal(totals.bytes, bytes, `${label}: totals.bytes must equal the sum of artifact bytes`);

    // Compression estimates are optional (--no-compression-estimates emits null).
    if (totals.gzipBytes === null || totals.brotliBytes === null) return;
    const gzip = artifacts.reduce((sum, a) => sum + (a.compression?.gzip?.bytes ?? 0), 0);
    const brotli = artifacts.reduce((sum, a) => sum + (a.compression?.brotli?.bytes ?? 0), 0);
    assert.equal(totals.gzipBytes, gzip, `${label}: totals.gzipBytes must equal the sum of artifact gzip bytes`);
    assert.equal(totals.brotliBytes, brotli, `${label}: totals.brotliBytes must equal the sum of artifact brotli bytes`);

    if (!bytes) return;
    assert.equal(totals.gzipRatio, Number((gzip / bytes).toFixed(4)), `${label}: stale gzipRatio`);
    assert.equal(totals.brotliRatio, Number((brotli / bytes).toFixed(4)), `${label}: stale brotliRatio`);
  });

  test(`${label} lists at most one preload .data`, () => {
    const manifest = JSON.parse(readFileSync(path, 'utf8'));
    const artifacts = Array.isArray(manifest.artifacts) ? manifest.artifacts : [];
    const datas = artifacts.map((a) => a.path).filter((p) => typeof p === 'string' && p.endsWith('.data'));
    assert.ok(datas.length <= 1, `${label}: SIMD tiers share one canonical .data, found ${datas.join(', ')}`);
  });
}
