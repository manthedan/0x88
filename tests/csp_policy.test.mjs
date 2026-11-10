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
import svelteConfig, { createCspDirectives } from '../svelte.config.js';

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

async function policyFixture(t, mutate, env) {
  const dir = await mkdtemp(join(tmpdir(), 'csp-policy-'));
  t.after(() => rm(dir, { recursive: true, force: true }));
  const toml = mutate(await readFile(resolve(repoRoot, 'netlify.toml'), 'utf8'));
  await mkdir(join(dir, 'public'));
  await writeFile(join(dir, 'package.json'), '{"type":"module"}');
  const config = env ? { kit: { csp: { directives: createCspDirectives(env) } } } : svelteConfig;
  await writeFile(join(dir, 'svelte.config.js'), `export default ${JSON.stringify(config)};`);
  await writeFile(join(dir, 'netlify.toml'), toml);
  await writeFile(join(dir, 'public/_headers'), renderHeadersFile(toml));
  return dir;
}

function missingOrigin(origin, setting, directive = 'connect-src') {
  return `header ${directive} blocks ${origin} required by ${setting}; add ${origin} to Content-Security-Policy ${directive} in netlify.toml [[headers]] for "/*", then run \`npm run deploy:generate-headers\` to update public/_headers`;
}

function allowExecutableOrigin(toml, origin) {
  return toml.replace('blob: https://assets.0x88.app ', `blob: https://assets.0x88.app ${origin} `);
}

test('default build-time asset and channel configuration passes the CSP gate', async () => {
  assert.equal((await checkCspPolicy(repoRoot, undefined, defaultEnv)).ok, true);
});

test('a connect-only custom executable asset host is rejected by both script-src policies', async (t) => {
  const origin = 'https://engines.example.com';
  const root = await policyFixture(t, (toml) => toml.replace("connect-src 'self'", `connect-src 'self' ${origin}`));
  await assert.rejects(checkCspPolicy(root, undefined, { ...defaultEnv, VITE_LC0_BROWSER_ASSET_BASE_URL: `${origin}/engines/` }), (error) => {
    for (const [policy, file] of [
      ['header', 'netlify.toml'],
      ['meta', 'svelte.config.js'],
    ]) {
      assert.ok(
        error.failures.some((failure) => failure.includes(`${policy} script-src blocks ${origin}`) && failure.includes(file)),
        error.message,
      );
    }
    return true;
  });
});

test('svelte.config.js derives meta script-src from the executable asset base origin', () => {
  const result = spawnSync(
    process.execPath,
    ['--input-type=module', '-e', "import config from './svelte.config.js'; console.log(JSON.stringify(config.kit.csp.directives['script-src']));"],
    {
      encoding: 'utf8',
      env: { ...process.env, VITE_LC0_BROWSER_ASSET_BASE_URL: 'https://engines.example.com/subpath/' },
    },
  );
  assert.equal(result.status, 0, result.stderr);
  assert.ok(JSON.parse(result.stdout).includes('https://engines.example.com'), result.stdout);
});

test('default meta CSP directives retain their committed source order and bytes', () => {
  assert.equal(
    JSON.stringify(createCspDirectives({})),
    JSON.stringify({
      'default-src': ['self'],
      'base-uri': ['self'],
      'object-src': ['none'],
      'frame-ancestors': ['self'],
      'script-src': ['self', 'unsafe-eval', 'wasm-unsafe-eval', 'blob:', 'https://assets.0x88.app', 'https://static.cloudflareinsights.com'],
      'style-src': ['self', 'unsafe-inline'],
      'img-src': ['self', 'data:', 'blob:'],
      'font-src': ['self', 'data:'],
      'connect-src': ['self', 'https:'],
      'worker-src': ['self', 'blob:'],
      'child-src': ['self', 'blob:'],
      'frame-src': ['self'],
      'manifest-src': ['self'],
    }),
  );
});

test('svelte.config.js reads the executable asset origin from Vite production environment files', async (t) => {
  const root = await policyFixture(t, (toml) => toml);
  await writeFile(join(root, '.env.production'), 'VITE_LC0_BROWSER_ASSET_BASE_URL=https://engines.example.com/subpath/\n');
  const env = { ...process.env };
  delete env.VITE_LC0_BROWSER_ASSET_BASE_URL;
  const result = spawnSync(
    process.execPath,
    [
      '--input-type=module',
      '-e',
      `import config from ${JSON.stringify(resolve(repoRoot, 'svelte.config.js'))}; console.log(JSON.stringify(config.kit.csp.directives['script-src']));`,
    ],
    {
      cwd: root,
      encoding: 'utf8',
      env,
    },
  );
  assert.equal(result.status, 0, result.stderr);
  assert.ok(JSON.parse(result.stdout).includes('https://engines.example.com'), result.stdout);
});

test('each enforced script-src independently requires the executable asset origin', async (t) => {
  const origin = 'https://engines.example.com';
  const env = { ...defaultEnv, VITE_LC0_BROWSER_ASSET_BASE_URL: origin };
  const connected = (toml) => toml.replace("connect-src 'self'", `connect-src 'self' ${origin}`);
  const missingHeader = await policyFixture(t, connected, env);
  const missingMeta = await policyFixture(t, (toml) => allowExecutableOrigin(connected(toml), origin));
  for (const [root, policy, file] of [
    [missingHeader, 'header', 'netlify.toml'],
    [missingMeta, 'meta', 'svelte.config.js'],
  ]) {
    await assert.rejects(checkCspPolicy(root, undefined, env), (error) => {
      assert.ok(
        error.failures.some((failure) => failure.startsWith(`${policy} script-src blocks ${origin}`) && failure.includes(file)),
        error.message,
      );
      return true;
    });
  }
});

for (const [setting, executable] of [
  ['VITE_TINY_LEELA_ASSET_BASE_URL', true],
  ['VITE_LC0_MODEL_BASE_URL', false],
]) {
  test(`${setting} requires ${executable ? 'script-src and connect-src' : 'only connect-src'} coverage`, async (t) => {
    const origin = 'https://alternate.example.com';
    const env = { ...defaultEnv, VITE_LC0_BROWSER_ASSET_BASE_URL: '', [setting]: `${origin}/assets/` };
    const missing = await policyFixture(t, (toml) => toml, env);
    await assert.rejects(checkCspPolicy(missing, undefined, env), (error) => {
      assert.ok(error.failures.includes(missingOrigin(origin, setting)), error.message);
      assert.equal(error.failures.includes(missingOrigin(origin, setting, 'script-src')), executable, error.message);
      return true;
    });
    const root = await policyFixture(
      t,
      (toml) => {
        const connected = toml.replace("connect-src 'self'", `connect-src 'self' ${origin}`);
        return executable ? allowExecutableOrigin(connected, origin) : connected;
      },
      env,
    );
    assert.equal((await checkCspPolicy(root, undefined, env)).ok, true);
    assert.equal(checkDeployCachePolicy(root, env).ok, true);
    assert.equal(createCspDirectives(env)['script-src'].includes(origin), executable);
  });
}

test('the primary asset base takes precedence over the legacy and model-only aliases', async () => {
  const env = { ...defaultEnv, VITE_TINY_LEELA_ASSET_BASE_URL: 'https://unused.example.com', VITE_LC0_MODEL_BASE_URL: 'https://unused-models.example.com' };
  assert.equal((await checkCspPolicy(repoRoot, undefined, env)).ok, true);
  assert.deepEqual(createCspDirectives(env), createCspDirectives(defaultEnv));
});

test('a stale built meta script-src is rejected even when the source configuration matches', async (t) => {
  const origin = 'https://engines.example.com';
  const env = { ...defaultEnv, VITE_LC0_BROWSER_ASSET_BASE_URL: origin };
  const root = await policyFixture(t, (toml) => allowExecutableOrigin(toml.replace("connect-src 'self'", `connect-src 'self' ${origin}`), origin), env);
  await mkdir(join(root, 'dist'));
  const sources = metaDirectivesFromSvelteConfig(svelteConfig).get('script-src').join(' ');
  await writeFile(
    join(root, 'dist/index.html'),
    builtPage({ metaScriptSrc: `${sources} 'sha256-${sha('boot()')}'` }).replace("default-src 'self';", "default-src 'self'; connect-src 'self' https:;"),
  );
  await assert.rejects(checkCspPolicy(root, 'dist', env), (error) => {
    assert.equal(error.failures.length, 1, error.message);
    assert.match(error.failures[0], /index.html: meta script-src blocks https:\/\/engines.example.com.*svelte.config.js/);
    return true;
  });
});

for (const [setting, url, origin] of [
  ['VITE_LC0_BROWSER_ASSET_BASE_URL', 'https://models.example.com/weights/', 'https://models.example.com'],
  ['VITE_LC0_ARTIFACT_CHANNEL_URL', 'https://channels.example.com/channels/stable.json', 'https://channels.example.com'],
]) {
  test(`custom ${setting} without a header change fails with the exact origin and edit location`, async () => {
    await assert.rejects(checkCspPolicy(repoRoot, undefined, { ...defaultEnv, [setting]: url }), (error) => {
      assert.ok(error.failures.includes(missingOrigin(origin, setting)), error.message);
      if (setting === 'VITE_LC0_BROWSER_ASSET_BASE_URL') assert.ok(error.failures.includes(missingOrigin(origin, setting, 'script-src')), error.message);
      else assert.deepEqual(error.failures, [missingOrigin(origin, setting)]);
      return true;
    });
  });

  test(`custom ${setting} with a matching header passes both deploy gates`, async (t) => {
    const env = { ...defaultEnv, [setting]: url };
    const root = await policyFixture(
      t,
      (toml) => {
        const connected = toml.replace("connect-src 'self'", `connect-src 'self' ${origin}`);
        return setting === 'VITE_LC0_BROWSER_ASSET_BASE_URL' ? allowExecutableOrigin(connected, origin) : connected;
      },
      env,
    );
    assert.equal((await checkCspPolicy(root, undefined, env)).ok, true);
    assert.equal(checkDeployCachePolicy(root, env).ok, true);
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
    assert.ok(error.failures.includes(missingOrigin('https://models.example.com', 'VITE_LC0_BROWSER_ASSET_BASE_URL')), error.message);
    assert.ok(error.failures.includes(missingOrigin('https://models.example.com', 'VITE_LC0_BROWSER_ASSET_BASE_URL', 'script-src')), error.message);
    return true;
  });
});

test('a relative artifact channel remains covered by self', async () => {
  assert.equal((await checkCspPolicy(repoRoot, undefined, { ...defaultEnv, VITE_LC0_ARTIFACT_CHANNEL_URL: '/channels/stable.json' })).ok, true);
});

test('allowing custom connect origins does not loosen other CSP directives in either gate', async (t) => {
  const env = { ...defaultEnv, VITE_LC0_BROWSER_ASSET_BASE_URL: 'https://models.example.com' };
  const root = await policyFixture(
    t,
    (toml) =>
      allowExecutableOrigin(toml.replace("connect-src 'self'", "connect-src 'self' https://models.example.com"), 'https://models.example.com').replace(
        " 'unsafe-eval'",
        '',
      ),
    env,
  );
  await assert.rejects(checkCspPolicy(root, undefined, env), (error) => {
    assert.deepEqual(error.failures, ["meta script-src allows 'unsafe-eval' that the header policy does not"]);
    return true;
  });
  assert.throws(() => checkDeployCachePolicy(root, env), /missing site-wide Content-Security-Policy/);
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
