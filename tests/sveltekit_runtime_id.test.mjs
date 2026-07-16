import assert from 'node:assert/strict';
import { execFileSync, spawnSync } from 'node:child_process';
import { mkdtemp, mkdir, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

async function fixture(scriptId, pageId) {
  const root = await mkdtemp(join(tmpdir(), 'sveltekit-runtime-id-'));
  await mkdir(join(root, '_app', 'immutable', 'chunks'), { recursive: true });
  await writeFile(join(root, '_app', 'immutable', 'chunks', 'runtime.js'), `globalThis.${scriptId}.data`);
  await writeFile(join(root, 'index.html'), `<script>${pageId} = { base: '' }</script>`);
  return root;
}

test('SvelteKit runtime check accepts matching generated ids', async () => {
  const root = await fixture('__sveltekit_matching', '__sveltekit_matching');
  try {
    const stdout = execFileSync(process.execPath, ['scripts/check_sveltekit_runtime_id.mjs', root], { encoding: 'utf8' });
    const report = JSON.parse(stdout);
    assert.equal(report.status, 'SVELTEKIT_RUNTIME_ID_CHECK_DONE');
    assert.equal(report.runtimeId, '__sveltekit_matching');
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test('SvelteKit runtime check rejects mismatched generated ids', async () => {
  const root = await fixture('__sveltekit_client', '__sveltekit_server');
  try {
    const result = spawnSync(process.execPath, ['scripts/check_sveltekit_runtime_id.mjs', root], { encoding: 'utf8' });
    assert.notEqual(result.status, 0);
    assert.match(result.stderr, /runtime id mismatch/);
    assert.match(result.stderr, /Clean \.svelte-kit\/output and rebuild/);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});
