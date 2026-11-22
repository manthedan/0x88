import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import {
  actionableErrorEntries,
  advertisesWorkerRetirement,
  assertClientDocument,
  browserMaxAge,
  failedRequestEntries,
  inspectAnalysis,
  isExpectedAppShellCacheControl,
  isExpectedOrtCacheControl,
  parseCacheControl,
  workerRetirementReport,
} from '../scripts/production_browser_smoke.mjs';

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
  assert.deepEqual(report.journey, ['Play/Centipawn', 'Arena', 'Analysis', 'Play/Stormphrax', 'Analysis/return']);
});

test('document identity is rechecked after controller readiness and worker retirement', async () => {
  let token = 'original';
  const session = { documentToken: token, page: { evaluate: async () => token } };
  await assertClientDocument(session, 'initial path');
  token = undefined;
  await assert.rejects(assertClientDocument(session, 'destination ready'), /full navigation instead of client-side navigation/);
  const source = readFileSync(script, 'utf8');
  assert.match(source, /async function inspectBrowser\(session, surface\) \{\s*await assertClientDocument\(session, surface\)/);
  assert.match(source, /await assertClientDocument\(session, 'Worker retirement'\);\s*return workerRetirementReport/);
});

test('Analysis smoke opens the developer runtime details before reading the deferred diagnostics', async () => {
  const history = [];
  const location = new URL('https://example.test/app/arena/');
  const runtimeText =
    'Centipawn: not selected · Reckless: persistent available · asset ok · Viridithas: asset ok · Berserk: asset ok · PlentyChess: asset ok · Stormphrax: asset ok';
  let href = '/app/analysis/';
  const link = {
    getAttribute: () => href,
    setAttribute: (_attribute, value) => {
      href = value;
    },
    click: () => {
      const next = new URL(href, location);
      location.pathname = next.pathname;
      location.search = next.search;
    },
  };
  const document = {
    title: 'Analysis',
    querySelector(selector) {
      if (selector === '.site-header nav.primary a[href="/app/analysis/"]') return link;
      if (selector === '[data-testid="browser-capabilities"]') return { textContent: 'WebGPU ready · WASM threads ready · Model cache' };
      if (selector === '.row-fam')
        return { options: ['lc0', 'sf', 'reckless', 'berserk', 'viridithas', 'plentychess', 'stormphrax', 'centipawn'].map((value) => ({ value })) };
      if (selector === '#recklessRuntimeInfo') {
        assert.deepEqual(history, ['#manageEngines', 'Developer engine settings', 'Engine runtime details']);
        return { textContent: runtimeText };
      }
      assert.fail(selector);
    },
  };
  const control = (name) => ({ click: async () => history.push(name), first: () => ({ waitFor: async () => assert.ok(document.querySelector(name)) }) });
  const session = {
    args: { baseUrl: location.origin },
    documentToken: 'original',
    page: {
      locator: control,
      getByText: control,
      evaluate: async (expression) =>
        new Function('document', 'window', 'location', 'setTimeout', `return ${expression}`)(
          document,
          { __productionSmokeDocument: 'original' },
          location,
          (fn) => fn(),
        ),
    },
  };
  const result = await inspectAnalysis(session);
  assert.equal(result.url, 'https://example.test/app/analysis/?dev');
  assert.equal(location.search, '?dev');
  assert.equal(result.runtimeText, runtimeText);
  assert.equal(history.at(-1), '#closeEngineManager');
  assert.equal(href, '/app/analysis/');
});

test('nightly smoke recognizes old and new deployed Play markup during rollout', async () => {
  const shipped = readFileSync('src/routes/app/play/+page.svelte', 'utf8').match(/data-worker-retirement="([^"]+)"/)?.[1];
  for (const [advertised, expected] of [
    [undefined, false],
    [shipped, true],
  ]) {
    const document = { getElementById: (id) => (id === 'main' ? { dataset: { workerRetirement: advertised } } : null) };
    const session = { page: { evaluate: async (expression) => new Function('document', `return ${expression}`)(document) } };
    assert.equal(await advertisesWorkerRetirement(session), expected);
  }
});

test('navigation guard requires real worker retirement and a lower count on Analysis -> Play', () => {
  const small = { url: () => 'small-worker' },
    sf = { url: () => 'sf-worker' },
    storm = { url: () => 'storm-worker' };
  assert.deepEqual(workerRetirementReport([small, sf], [storm], true), { before: ['small-worker', 'sf-worker'], after: ['storm-worker'], retired: 2 });
  assert.throws(() => workerRetirementReport([small, sf], [small], true), /retained 1 previous worker/);
  assert.throws(() => workerRetirementReport([small], [storm], true), /count did not drop/);
  assert.throws(() => workerRetirementReport([], [], true), /no live workers/);
  assert.equal(workerRetirementReport([storm], [small, sf]).retired, 1, 'returning to Analysis may start more workers, but never retain Play');
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
