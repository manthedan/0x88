import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
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
