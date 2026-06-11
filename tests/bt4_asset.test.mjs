import test from 'node:test';
import assert from 'node:assert/strict';
import { BT4_CACHE_MODEL, BT4_EXPECTED_BYTES, probeBt4ModelAsset } from '../src/lc0/bt4Engine.ts';

test('BT4 worker requests validated byte/cache loading', () => {
  assert.equal(BT4_CACHE_MODEL, true);
});

test('BT4 asset probe accepts present model with expected size', async () => {
  const fetchFn = async () => new Response(null, {
    status: 200,
    headers: { 'content-length': String(BT4_EXPECTED_BYTES) },
  });

  assert.equal(await probeBt4ModelAsset(fetchFn), true);
});

test('BT4 asset probe rejects missing or wrong-sized model', async () => {
  assert.equal(await probeBt4ModelAsset(async () => new Response(null, { status: 404 })), false);
  assert.equal(await probeBt4ModelAsset(async () => new Response(null, {
    status: 200,
    headers: { 'content-length': String(BT4_EXPECTED_BYTES - 1) },
  })), false);
});
