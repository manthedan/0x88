import assert from 'node:assert/strict';
import { execFileSync, spawnSync } from 'node:child_process';
import { mkdtemp, mkdir, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

async function fixture() {
  const root = await mkdtemp(join(tmpdir(), 'ort-dedup-'));
  await mkdir(join(root, 'ort'), { recursive: true });
  await mkdir(join(root, '_app', 'immutable', 'assets'), { recursive: true });
  await writeFile(join(root, 'ort', 'ort-wasm-simd-threaded.asyncify.wasm'), 'canonical');
  return root;
}

test('ORT WASM deploy check accepts one canonical staged copy', async () => {
  const root = await fixture();
  try {
    const stdout = execFileSync(process.execPath, ['scripts/check_ort_wasm_dedup.mjs', root], { encoding: 'utf8' });
    const report = JSON.parse(stdout);
    assert.equal(report.status, 'ORT_WASM_DEDUP_CHECK_DONE');
    assert.deepEqual(report.bundledCopies, []);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test('ORT WASM deploy check rejects Vite-emitted copies', async () => {
  const root = await fixture();
  try {
    await writeFile(join(root, '_app', 'immutable', 'assets', 'ort-wasm-copy.wasm'), 'duplicate');
    const result = spawnSync(process.execPath, ['scripts/check_ort_wasm_dedup.mjs', root], { encoding: 'utf8' });
    assert.notEqual(result.status, 0);
    assert.match(result.stderr, /duplicate ORT WASM assets/);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});
