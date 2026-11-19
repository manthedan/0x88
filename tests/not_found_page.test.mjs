import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { mkdirSync, writeFileSync } from 'node:fs';
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

const repoRoot = process.cwd();

// Netlify serves /404.html for unknown paths, so the adapter must write the fallback shell there,
// readable before (or without) its JS.
test('the static adapter writes its fallback shell to 404.html with static not-found content', async (t) => {
  const out = await mkdtemp(join(tmpdir(), 'kit-fallback-'));
  t.after(() => rm(out, { recursive: true, force: true }));
  // The adapter empties its output dir, so it gets the temp dir explicitly: never the configured
  // dist-client or release dir, whatever the environment or module cache holds.
  const { createAdapter, default: config, STATIC_NOT_FOUND_ID } = await import('../kit.config.js');
  const server = await mkdtemp(join(tmpdir(), 'kit-server-'));
  t.after(() => rm(server, { recursive: true, force: true }));
  await mkdir(join(server, 'nodes'));
  await writeFile(
    join(server, 'nodes', '1.js'),
    'export const imports = ["_app/immutable/nodes/1.e.js","_app/immutable/chunks/shared.js","_app/immutable/chunks/err.js"]; export const stylesheets = ["_app/immutable/assets/1.e.css"];',
  );
  const playHead = '<head><link href="/_app/immutable/chunks/shared.js" rel="modulepreload"></head>';
  const calls = [];
  const log = Object.assign(() => undefined, { info() {}, warn() {}, error() {}, minor() {} });
  await createAdapter(out).adapt({
    config: { kit: {}, router: {} },
    routes: [],
    prerendered: { pages: new Map([['/app/play/', { file: 'app/play/index.html' }]]) },
    getServerDirectory: () => server,
    log,
    generateEnvModule() {},
    writeClient: (dir) => calls.push(['client', dir]),
    // Kit's builder.writePrerendered is synchronous and adapter-static does not await it, so the
    // mock must be synchronous too or generateFallback can run before the output dir exists.
    writePrerendered: (dir) => {
      calls.push(['prerendered', dir]);
      mkdirSync(join(out, 'app/play'), { recursive: true }); // the adapter clears the output dir first
      writeFileSync(join(out, 'app/play/index.html'), `${playHead}<body></body>`);
    },
    generateFallback: async (file) => {
      calls.push(['fallback', file]);
      await writeFile(file, '<head></head><body><div style="display: contents"><script>start()</script></div></body>');
    },
  });
  assert.deepEqual(calls, [
    ['client', out],
    ['prerendered', out],
    ['fallback', join(out, '404.html')],
  ]);
  const html = await readFile(join(out, '404.html'), 'utf8');
  const target = '<div style="display: contents">';
  const staticCopy = html.slice(html.indexOf(target) + target.length, html.indexOf('<script>start()</script>'));
  assert.ok(staticCopy.startsWith(`<main id="${STATIC_NOT_FOUND_ID}"`), html);
  assert.ok(staticCopy.includes('<h1>Page not found</h1>') && staticCopy.includes('href="/"'), staticCopy);
  // Kit 3 hydration awaits the root error node, so every page links it (and sw.js precaches it).
  // Links a page already has are skipped: 404.html gets every chunk, the play page all but shared.js.
  const link = (path) => `<link href="/_app/immutable/${path}" rel="modulepreload">`;
  const css = '<link href="/_app/immutable/assets/1.e.css" rel="preload" as="style">';
  assert.ok(html.startsWith(`<head>${link('nodes/1.e.js')}${link('chunks/shared.js')}${link('chunks/err.js')}${css}</head>`), html);
  assert.equal(
    await readFile(join(out, 'app/play/index.html'), 'utf8'),
    `${playHead.replace('</head>', `${link('nodes/1.e.js')}${link('chunks/err.js')}${css}</head>`)}<body></body>`,
  );
  assert.equal(config.paths.relative, false, 'cached pages served at other depths need root-absolute build URLs');
  const errorPage = await readFile(join(repoRoot, 'src/routes/+error.svelte'), 'utf8');
  assert.ok(errorPage.includes(`getElementById('${STATIC_NOT_FOUND_ID}')?.remove()`), 'the mounted error page replaces the static copy');
});

async function startServer(t, root) {
  const port = 20000 + Math.floor(Math.random() * 20000);
  // The server reads PORT before argv; an inherited PORT would point the test at the wrong server.
  const child = spawn(process.execPath, ['scripts/serve_isolated_static.mjs', root, String(port)], {
    cwd: repoRoot,
    env: { ...process.env, PORT: String(port) },
    stdio: ['ignore', 'pipe', 'inherit'],
  });
  t.after(() => child.kill());
  await new Promise((ready, fail) => {
    child.stdout.once('data', ready);
    child.once('exit', (code) => fail(new Error(`static server exited with ${code}`)));
  });
  return `http://localhost:${port}`;
}

test('the local static server answers unknown paths like Netlify: 404 status, 404.html body', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'static-404-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  await mkdir(join(root, 'docs'));
  await writeFile(join(root, 'docs/index.html'), 'docs');
  await writeFile(join(root, '404.html'), 'not found shell');
  const origin = await startServer(t, root);

  const page = await fetch(`${origin}/docs/`);
  assert.equal(page.status, 200);
  assert.equal(await page.text(), 'docs');

  const missing = await fetch(`${origin}/no/such/page/`);
  assert.equal(missing.status, 404);
  assert.equal(missing.headers.get('content-type'), 'text/html; charset=utf-8');
  assert.equal(missing.headers.get('cross-origin-embedder-policy'), 'require-corp');
  assert.equal(await missing.text(), 'not found shell');
});

// adapter-static skips its "every route must prerender" check once `fallback` is set; ours keeps it.
test('the static adapter refuses a route that does not prerender, before touching the output dir', async (t) => {
  const out = await mkdtemp(join(tmpdir(), 'kit-strict-'));
  t.after(() => rm(out, { recursive: true, force: true }));
  await writeFile(join(out, 'keep.txt'), 'untouched');
  const { createAdapter } = await import('../kit.config.js');
  const calls = [];
  const builder = {
    config: { kit: {}, router: {} },
    routes: [
      { id: '/app/play', prerender: true },
      { id: '/app/analysis', prerender: false },
      { id: '/docs', prerender: 'auto' },
    ],
    log: Object.assign(() => undefined, { info() {}, warn() {}, error() {}, minor() {} }),
    generateEnvModule: () => calls.push('env'),
    writeClient: () => calls.push('client'),
    writePrerendered: () => calls.push('prerendered'),
    generateFallback: async () => calls.push('fallback'),
  };
  await assert.rejects(createAdapter(out).adapt(builder), /every route must prerender.*\/app\/analysis, \/docs/);
  assert.deepEqual(calls, []);
  assert.equal(await readFile(join(out, 'keep.txt'), 'utf8'), 'untouched');
});
