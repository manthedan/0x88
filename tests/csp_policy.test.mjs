import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { test } from 'node:test';
import {
  checkBuiltPage,
  checkCspPolicy,
  checkPolicyConsistency,
  headerPolicyFromNetlifyToml,
  inlineScriptAllowed,
  inspectHtml,
  metaDirectivesFromSvelteConfig,
  parseCsp,
} from '../scripts/check_csp_policy.mjs';
import { checkDeployCachePolicy } from '../scripts/check_deploy_cache_policy.mjs';
import { renderHeadersFile } from '../scripts/generate_netlify_headers_file.mjs';
import svelteConfig from '../svelte.config.js';

const repoRoot = process.cwd();
const sha = (text) => createHash('sha256').update(text, 'utf8').digest('base64');

async function shippedHeader() {
  return parseCsp(headerPolicyFromNetlifyToml(await readFile(resolve(repoRoot, 'netlify.toml'), 'utf8')));
}

// A built page in the shape SvelteKit's hash mode emits: the app.html script
// first, then the meta policy listing only the boot script's hash.
function builtPage({ theme = 'theme()', boot = 'boot()', metaScriptSrc } = {}) {
  const scriptSrc = metaScriptSrc ?? `'self' 'unsafe-eval' 'sha256-${sha(boot)}'`;
  return [
    '<!doctype html><html><head>',
    `<script>${theme}</script>`,
    `<meta http-equiv="content-security-policy" content="default-src 'self'; script-src ${scriptSrc}">`,
    '</head><body>',
    `<script>${boot}</script>`,
    '</body></html>',
  ].join('\n');
}

test('the shipped header and svelte.config.js policies agree', async () => {
  assert.deepEqual(await checkCspPolicy(repoRoot), { ok: true, checked: ['netlify.toml', 'svelte.config.js'] });
});

const defaultEnv = {
  VITE_LC0_BROWSER_ASSET_BASE_URL: 'https://assets.0x88.app',
  VITE_LC0_ARTIFACT_CHANNEL_URL: 'https://assets.0x88.app/channels/stable.json',
};

async function policyFixture(t, mutate) {
  const dir = await mkdtemp(join(tmpdir(), 'csp-policy-'));
  t.after(() => rm(dir, { recursive: true, force: true }));
  const toml = mutate(await readFile(resolve(repoRoot, 'netlify.toml'), 'utf8'));
  await mkdir(join(dir, 'public'));
  await writeFile(join(dir, 'package.json'), '{"type":"module"}');
  await writeFile(join(dir, 'svelte.config.js'), `export default ${JSON.stringify(svelteConfig)};`);
  await writeFile(join(dir, 'netlify.toml'), toml);
  await writeFile(join(dir, 'public/_headers'), renderHeadersFile(toml));
  return dir;
}

function missingOrigin(origin, setting) {
  return `header connect-src blocks ${origin} required by ${setting}; add ${origin} to Content-Security-Policy connect-src in netlify.toml [[headers]] for "/*", then run \`npm run deploy:generate-headers\` to update public/_headers`;
}

test('default build-time asset and channel configuration passes the CSP gate', async () => {
  assert.equal((await checkCspPolicy(repoRoot, undefined, defaultEnv)).ok, true);
});

for (const [setting, url, origin] of [
  ['VITE_LC0_BROWSER_ASSET_BASE_URL', 'https://models.example.com/weights/', 'https://models.example.com'],
  ['VITE_LC0_ARTIFACT_CHANNEL_URL', 'https://channels.example.com/channels/stable.json', 'https://channels.example.com'],
]) {
  test(`custom ${setting} without a header change fails with the exact origin and edit location`, async () => {
    await assert.rejects(checkCspPolicy(repoRoot, undefined, { ...defaultEnv, [setting]: url }), (error) => {
      assert.deepEqual(error.failures, [missingOrigin(origin, setting)]);
      return true;
    });
  });

  test(`custom ${setting} with a matching header passes both deploy gates`, async (t) => {
    const root = await policyFixture(t, (toml) => toml.replace("connect-src 'self'", `connect-src 'self' ${origin}`));
    assert.equal((await checkCspPolicy(root, undefined, { ...defaultEnv, [setting]: url })).ok, true);
    assert.equal(checkDeployCachePolicy(root).ok, true);
  });
}

test('fixed game-import and analytics origins are required even without custom build configuration', async (t) => {
  for (const origin of ['https://lichess.org', 'https://api.chess.com', 'https://cloudflareinsights.com']) {
    const root = await policyFixture(t, (toml) => toml.replace(` ${origin}`, ''));
    await assert.rejects(checkCspPolicy(root, undefined, defaultEnv), (error) => {
      assert.equal(error.failures.length, 1);
      assert.ok(error.failures[0].startsWith(`header connect-src blocks ${origin} required by `));
      return true;
    });
  }
});

test('the release --asset-base option is checked before a missing build stamp', () => {
  const result = spawnSync(process.execPath, ['scripts/netlify_r2_release.mjs', '--check', '--asset-base', 'https://models.example.com'], {
    encoding: 'utf8',
  });
  assert.equal(result.status, 1);
  assert.ok(result.stderr.includes(missingOrigin('https://models.example.com', 'VITE_LC0_BROWSER_ASSET_BASE_URL')), result.stderr);
});

test('the R2 build checks its effective asset configuration before invoking Vite', () => {
  const result = spawnSync(process.execPath, ['scripts/build_netlify_r2.mjs'], {
    encoding: 'utf8',
    env: { ...process.env, ...defaultEnv, VITE_LC0_BROWSER_ASSET_BASE_URL: 'https://models.example.com' },
  });
  assert.equal(result.status, 1);
  assert.ok(result.stderr.includes(missingOrigin('https://models.example.com', 'VITE_LC0_BROWSER_ASSET_BASE_URL')), result.stderr);
  assert.ok(!result.stderr.includes('[netlify-r2-build] vite build:'));
});

test('the standalone gate reads Vite production environment files', async (t) => {
  const root = await policyFixture(t, (toml) => toml);
  await writeFile(join(root, '.env.production'), 'VITE_LC0_BROWSER_ASSET_BASE_URL=https://models.example.com\n');
  await assert.rejects(checkCspPolicy(root), (error) => {
    assert.deepEqual(error.failures, [missingOrigin('https://models.example.com', 'VITE_LC0_BROWSER_ASSET_BASE_URL')]);
    return true;
  });
});

test('a relative artifact channel remains covered by self', async () => {
  assert.equal((await checkCspPolicy(repoRoot, undefined, { ...defaultEnv, VITE_LC0_ARTIFACT_CHANNEL_URL: '/channels/stable.json' })).ok, true);
});

test('allowing custom connect origins does not loosen other CSP directives in either gate', async (t) => {
  const root = await policyFixture(t, (toml) =>
    toml.replace("connect-src 'self'", "connect-src 'self' https://models.example.com").replace(" 'unsafe-eval'", ''),
  );
  await assert.rejects(checkCspPolicy(root, undefined, { ...defaultEnv, VITE_LC0_BROWSER_ASSET_BASE_URL: 'https://models.example.com' }), (error) => {
    assert.deepEqual(error.failures, ["meta script-src allows 'unsafe-eval' that the header policy does not"]);
    return true;
  });
  assert.throws(() => checkDeployCachePolicy(root), /missing site-wide Content-Security-Policy/);
});

// This is what makes the meta policy unable to block the theme script: a meta
// policy governs only elements parsed after it. SvelteKit's hash mode does not
// hash app.html scripts, so one placed after %sveltekit.head% would be blocked.
test('every inline script in src/app.html precedes %sveltekit.head%', async () => {
  const template = await readFile(resolve(repoRoot, 'src/app.html'), 'utf8');
  const head = template.indexOf('%sveltekit.head%');
  assert.notEqual(head, -1);
  const { scripts } = inspectHtml(template);
  assert.ok(scripts.length > 0, 'expected the theme script in src/app.html');
  for (const script of scripts) assert.ok(script.offset < head, `inline script at ${script.offset} is after %sveltekit.head% (${head})`);
});

test('a page in the shipped shape passes against the shipped header', async () => {
  assert.deepEqual(checkBuiltPage(builtPage(), await shippedHeader()), []);
});

// Dropping 'unsafe-inline' from the header without listing every page's boot
// hash is the change that would stop hydration; the gate must refuse it.
test('a header without unsafe-inline is rejected for both inline scripts', () => {
  const header = parseCsp("default-src 'self'; script-src 'self' 'unsafe-eval'");
  const failures = checkBuiltPage(builtPage(), header, 'index.html');
  assert.equal(failures.length, 2, failures.join('\n'));
  assert.ok(failures.every((entry) => entry.includes('blocked by the netlify.toml header policy')));
});

// Listing only the theme hash silently disables 'unsafe-inline' for the boot script.
test('a header listing one hash beside unsafe-inline blocks every other inline script', () => {
  const header = parseCsp(`default-src 'self'; script-src 'self' 'unsafe-inline' 'sha256-${sha('theme()')}'`);
  const failures = checkBuiltPage(builtPage(), header, 'index.html');
  assert.equal(failures.length, 1, failures.join('\n'));
  assert.match(failures[0], new RegExp(`sha256-${sha('boot()').replace(/[+/]/g, '\\$&')}`));
});

test('an inline script after the meta tag that the meta policy does not hash is rejected', async () => {
  const html = builtPage().replace('</body>', '<script>late()</script></body>');
  const failures = checkBuiltPage(html, await shippedHeader(), 'index.html');
  assert.equal(failures.length, 1, failures.join('\n'));
  assert.match(failures[0], /blocked by the page's own meta policy/);
});

test('a page without a meta policy is reported', async () => {
  const failures = checkBuiltPage('<html><head></head><body><script>x()</script></body></html>', await shippedHeader(), 'x.html');
  assert.ok(failures.some((entry) => entry.includes('no <meta http-equiv="content-security-policy">')));
});

test('inlineScriptAllowed follows CSP hash and strict-dynamic semantics', () => {
  const hash = sha('a');
  assert.equal(inlineScriptAllowed(parseCsp("script-src 'unsafe-inline'"), hash), true);
  assert.equal(inlineScriptAllowed(parseCsp(`script-src 'sha256-${hash}'`), hash), true);
  assert.equal(inlineScriptAllowed(parseCsp("script-src 'unsafe-inline' 'nonce-abc'"), hash), false);
  assert.equal(inlineScriptAllowed(parseCsp("script-src 'unsafe-inline' 'strict-dynamic'"), hash), false);
  assert.equal(inlineScriptAllowed(parseCsp("default-src 'self'"), hash), false);
  assert.equal(inlineScriptAllowed(parseCsp("default-src 'self'; script-src-elem 'unsafe-inline'"), hash), true);
  assert.equal(inlineScriptAllowed(parseCsp("img-src 'self'"), hash), true);
});

test('policy drift between svelte.config.js and netlify.toml is rejected', async () => {
  const header = await shippedHeader();
  const meta = metaDirectivesFromSvelteConfig(svelteConfig);
  assert.deepEqual(checkPolicyConsistency(header, meta), []);

  const looserHeader = new Map(header);
  looserHeader.set('img-src', [...header.get('img-src'), 'https:']);
  assert.deepEqual(checkPolicyConsistency(looserHeader, meta), ['header img-src allows https: that the meta policy does not']);

  const stricterHeader = new Map(header);
  stricterHeader.set('worker-src', ["'self'"]);
  assert.deepEqual(checkPolicyConsistency(stricterHeader, meta), ['meta worker-src allows blob: that the header policy does not']);

  const missing = new Map(header);
  missing.delete('object-src');
  assert.deepEqual(checkPolicyConsistency(missing, meta), ['header policy is missing object-src, which the meta policy declares']);

  const headerOnly = new Map(header);
  headerOnly.set('upgrade-insecure-requests', []);
  assert.match(checkPolicyConsistency(headerOnly, meta)[0], /header policy declares upgrade-insecure-requests/);
});

test('connect-src in the header may be narrower than the meta policy but never wider', () => {
  const meta = parseCsp("connect-src 'self' https:");
  assert.deepEqual(checkPolicyConsistency(parseCsp("connect-src 'self' https://a.example https://b.example"), meta), []);
  assert.deepEqual(checkPolicyConsistency(parseCsp("connect-src 'self'"), meta), []);
  assert.deepEqual(checkPolicyConsistency(parseCsp("connect-src 'self' http://a.example wss://b.example"), meta), [
    'header connect-src allows http://a.example wss://b.example that the meta policy does not',
  ]);
  assert.deepEqual(checkPolicyConsistency(parseCsp("connect-src 'self' *"), meta), ['header connect-src allows * that the meta policy does not']);
  // Only connect-src is allowed to be narrower; elsewhere a scheme is not a host.
  assert.deepEqual(checkPolicyConsistency(parseCsp("img-src 'self' https://a.example"), parseCsp("img-src 'self' https:")), [
    'header img-src allows https://a.example that the meta policy does not',
    'meta img-src allows https: that the header policy does not',
  ]);
});

test('svelte.config.js keywords are quoted the way SvelteKit emits them', () => {
  const meta = metaDirectivesFromSvelteConfig({ kit: { csp: { directives: { 'script-src': ['self', 'unsafe-eval', 'blob:', 'sha256-abc'] } } } });
  assert.deepEqual(meta.get('script-src'), ["'self'", "'unsafe-eval'", 'blob:', "'sha256-abc'"]);
});
