import assert from 'node:assert/strict';
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { checkSvelteKitRuntimeId } from '../scripts/check_sveltekit_runtime_id.mjs';

async function fixture(t, { version = 'build.1', pageVersion = version, init = '__sveltekit_payload', start = true, app = true } = {}) {
  const root = await mkdtemp(join(tmpdir(), 'sveltekit-payload-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const entries = join(root, '_app/immutable/entry');
  await mkdir(entries, { recursive: true });
  await writeFile(join(root, '_app/version.json'), JSON.stringify({ version }));
  if (start) await writeFile(join(entries, 'start.hash.js'), 'export function init(payload) {}');
  if (app) await writeFile(join(entries, 'app.hash.js'), 'export const nodes = [];');
  const html = `<script>{
    __sveltekit_payload = { base: '', version: ${JSON.stringify(pageVersion)} };
    import("./_app/immutable/entry/start.hash.js").then(async (kit) => {
      kit.init(${init});
      const app = await import("./_app/immutable/entry/app.hash.js");
      kit.start(app, document.body);
    });
  }</script>`;
  await writeFile(join(root, 'index.html'), html);
  return { root, html };
}

test('Kit 3 payload and emitted entries bind pages to the current build', async (t) => {
  const { root } = await fixture(t);
  assert.equal((await checkSvelteKitRuntimeId(root)).runtimeId, '__sveltekit_payload');
});

test('Kit 3 rejects stale prerendered payload versions', async (t) => {
  const { root } = await fixture(t, { pageVersion: 'old-build' });
  await assert.rejects(checkSvelteKitRuntimeId(root), /payload version mismatch/);
});

test('Kit 3 version comparison treats regexp metacharacters literally', async (t) => {
  const { root } = await fixture(t, { version: 'build.1', pageVersion: 'buildX1' });
  await assert.rejects(checkSvelteKitRuntimeId(root), /payload version mismatch/);
});

test('Kit 3 rejects a missing payload initialization', async (t) => {
  const { root, html } = await fixture(t);
  await writeFile(join(root, 'index.html'), html.replace('kit.init(__sveltekit_payload)', 'kit.init({})'));
  await assert.rejects(checkSvelteKitRuntimeId(root), /payload version mismatch/);
});

for (const entry of ['start', 'app']) {
  test(`Kit 3 rejects a missing ${entry} entry file`, async (t) => {
    const { root } = await fixture(t, { [entry]: false });
    await assert.rejects(checkSvelteKitRuntimeId(root), { code: 'ENOENT' });
  });
}

test('Kit 3 checks nested pages too', async (t) => {
  const { root, html } = await fixture(t);
  await mkdir(join(root, 'docs'));
  await writeFile(join(root, 'docs/index.html'), html.replaceAll('./_app/', '../_app/'));
  await checkSvelteKitRuntimeId(root);
  await writeFile(join(root, 'docs/index.html'), html.replaceAll('./_app/', '../_app/').replace('build.1', 'stale'));
  await assert.rejects(checkSvelteKitRuntimeId(root), /payload version mismatch/);
});

test('Kit 3 accepts the fallback shell, whose entries are root-absolute', async (t) => {
  const { root, html } = await fixture(t);
  await writeFile(join(root, '404.html'), html.replaceAll('./_app/', '/_app/'));
  await checkSvelteKitRuntimeId(root);
  await writeFile(join(root, '404.html'), html.replaceAll('./_app/', '/_app/').replace('build.1', 'stale'));
  await assert.rejects(checkSvelteKitRuntimeId(root), /payload version mismatch/);
  await writeFile(join(root, '404.html'), html.replaceAll('./_app/', '/_app/').replace('start.hash.js', 'start.gone.js'));
  await assert.rejects(checkSvelteKitRuntimeId(root), { code: 'ENOENT' });
});
