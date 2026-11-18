import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
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
  // kit.config.js reads the output dir at import time; point it (and the adapter's rmSync) at a temp dir.
  process.env.NETLIFY_R2_RELEASE_DIST = out;
  const { default: config, STATIC_NOT_FOUND_ID } = await import('../kit.config.js');
  const calls = [];
  const log = Object.assign(() => undefined, { info() {}, warn() {}, error() {}, minor() {} });
  await config.adapter.adapt({
    config: { kit: {}, router: {} },
    routes: [],
    log,
    generateEnvModule() {},
    writeClient: (dir) => calls.push(['client', dir]),
    writePrerendered: (dir) => calls.push(['prerendered', dir]),
    generateFallback: async (file) => {
      calls.push(['fallback', file]);
      await mkdir(out, { recursive: true }); // the adapter clears the output dir first
      await writeFile(file, '<body><div style="display: contents"><script>start()</script></div></body>');
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
