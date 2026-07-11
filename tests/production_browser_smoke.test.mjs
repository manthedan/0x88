import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { actionableErrorEntries, browserMaxAge, failedRequestEntries, isExpectedAppShellCacheControl, isExpectedOrtCacheControl, parseCacheControl } from '../scripts/production_browser_smoke.mjs';

const script = 'scripts/production_browser_smoke.mjs';

function run(args) {
  return spawnSync(process.execPath, [script, ...args], { cwd: process.cwd(), encoding: 'utf8' });
}

test('production browser smoke dry-run describes the bounded cross-surface journey', () => {
  const result = run(['--base-url', 'https://example.test/path', '--dry-run']);
  assert.equal(result.status, 0, result.stderr);
  const report = JSON.parse(result.stdout);
  assert.equal(report.status, 'PRODUCTION_BROWSER_SMOKE_DRY_RUN');
  assert.equal(report.baseUrl, 'https://example.test');
  assert.deepEqual(report.journey, ['Play/Centipawn', 'Arena', 'Analysis', 'Play/Stormphrax']);
});

test('production browser smoke rejects non-HTTP origins before launching a browser', () => {
  const result = run(['--base-url', 'file:///tmp/site', '--dry-run']);
  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /Unsupported --base-url protocol: file:/);
});

test('production browser smoke rejects invalid timeout values', () => {
  const result = run(['--timeout', '0', '--dry-run']);
  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /Invalid --timeout: 0/);
});

test('cache policy parsing distinguishes browser max-age from shared-cache s-maxage', () => {
  assert.equal(browserMaxAge('public, max-age=0, s-maxage=86400'), 0);
  assert.equal(browserMaxAge('public, s-maxage=86400'), null);
  assert.equal(browserMaxAge('public, max-age="3600", stale-while-revalidate=86400'), 3600);
  assert.equal(parseCacheControl('public, no-cache').has('no-cache'), true);
});

test('production cache predicates require exact public browser policies', () => {
  assert.equal(isExpectedAppShellCacheControl('public, max-age=0, must-revalidate'), true);
  assert.equal(isExpectedAppShellCacheControl('private, max-age=0, must-revalidate'), false);
  assert.equal(isExpectedOrtCacheControl('public, max-age=3600, stale-while-revalidate=86400'), true);
  assert.equal(isExpectedOrtCacheControl('private, max-age=3600, stale-while-revalidate=86400'), false);
  assert.equal(isExpectedOrtCacheControl('public, no-store, max-age=3600, stale-while-revalidate=86400'), false);
  assert.equal(isExpectedOrtCacheControl('public, max-age=3600, stale-while-revalidate=86400, immutable'), false);
  assert.equal(isExpectedOrtCacheControl('public, max-age=3600, stale-while-revalidate=86400, s-maxage=31536000'), false);
  assert.equal(isExpectedOrtCacheControl('public, max-age=31536000, max-age=3600, stale-while-revalidate=86400'), false);
});

test('browser error filtering permits the supported ORT WebGPU-to-WASM fallback', () => {
  const fallback = { level: 'warning', text: 'Centipawn: ORT WebGPU session failed; falling back to WASM. no adapter' };
  const pageError = { level: 'error', text: 'Uncaught Error: worker crashed' };
  assert.deepEqual(actionableErrorEntries([fallback]), []);
  assert.deepEqual(actionableErrorEntries([fallback, pageError]), [pageError]);
  assert.deepEqual(actionableErrorEntries({ errors: [pageError] }), [pageError]);
});

test('failed request filtering includes HTTP and transport failures', () => {
  const ok = { url: 'https://example.test/app.js', status: 200 };
  const notFound = { url: 'https://example.test/missing.wasm', status: 404 };
  const transport = { request: { url: 'https://assets.example.test/model' }, response: { status: 0, statusText: 'net::ERR_NAME_NOT_RESOLVED' } };
  const unreportedWorker = { request: { url: 'blob:https://example.test/worker' }, response: { status: 0, statusText: '' } };
  assert.deepEqual(failedRequestEntries([ok, notFound, transport, unreportedWorker]), [notFound, transport]);
});

test('production browser smoke writes a diagnostic artifact when the journey fails', () => {
  const directory = mkdtempSync(join(tmpdir(), 'production-smoke-test-'));
  const output = join(directory, 'report.json');
  try {
    const result = run(['--base-url', 'http://127.0.0.1:9', '--out', output]);
    assert.notEqual(result.status, 0);
    const report = JSON.parse(readFileSync(output, 'utf8'));
    assert.equal(report.status, 'PRODUCTION_BROWSER_SMOKE_FAILED');
    assert.equal(report.baseUrl, 'http://127.0.0.1:9');
    assert.match(report.error.message, /fetch failed/);
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
});
