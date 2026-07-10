import assert from 'node:assert/strict';
import test from 'node:test';
import { DisposableVariantPool } from '../src/lc0/disposableVariantPool.ts';

function variant(key, url = `${key}.wasm`) {
  return { key, url };
}

test('DisposableVariantPool reuses resources by resolved cache key', () => {
  let creates = 0;
  const pool = new DisposableVariantPool(
    (entry) => `${entry.key}:${entry.url}`,
    (entry) => ({ entry, dispose() {}, instance: ++creates }),
  );

  const first = pool.getOrCreate(variant('simd'));
  const same = pool.getOrCreate(variant('simd'));
  const custom = pool.getOrCreate(variant('simd', 'custom.wasm'));

  assert.equal(first, same);
  assert.notEqual(first, custom);
  assert.equal(creates, 2);
  assert.equal(pool.size, 2);
});

test('DisposableVariantPool retains active variants and disposes everything else once', () => {
  const disposed = [];
  const pool = new DisposableVariantPool(
    (entry) => `${entry.key}:${entry.url}`,
    (entry) => ({ entry, dispose() { disposed.push(entry.url); } }),
  );
  const simd = variant('simd');
  const custom = variant('custom');
  pool.getOrCreate(simd);
  pool.getOrCreate(custom);

  pool.retain([simd]);
  assert.deepEqual(disposed, ['custom.wasm']);
  assert.equal(pool.size, 1);
  assert.ok(pool.peek(simd));
  assert.equal(pool.peek(custom), undefined);

  pool.disposeAll();
  pool.disposeAll();
  assert.deepEqual(disposed, ['custom.wasm', 'simd.wasm']);
  assert.equal(pool.size, 0);
});
