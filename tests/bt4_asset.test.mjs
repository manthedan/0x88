import test from 'node:test';
import assert from 'node:assert/strict';
import { BT4_NET, bigNetAssetStatusSync, checkBigNetAsset } from '../src/lc0/bt4Engine.ts';

function testConfig(modelUrl) {
  return { ...BT4_NET, modelUrl };
}

test('big-net asset probe uses a lightweight HEAD request and records present assets', async () => {
  const originalFetch = globalThis.fetch;
  const config = testConfig('/unit/big-net-present.onnx');
  let calls = 0;
  globalThis.fetch = async (url, init) => {
    calls += 1;
    assert.equal(url, config.modelUrl);
    assert.equal(init?.method, 'HEAD');
    assert.equal(init?.cache, 'no-store');
    return new Response(null, { status: 200 });
  };
  try {
    assert.equal(await checkBigNetAsset(config), 'present');
    assert.equal(bigNetAssetStatusSync(config), 'present');
    assert.equal(calls, 1);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test('production skips known-unhosted big-net asset probes', async () => {
  const originalFetch = globalThis.fetch;
  const originalLocation = Object.getOwnPropertyDescriptor(globalThis, 'location');
  const config = testConfig('/models/lc0/unit-unhosted-big-net.onnx');
  let calls = 0;
  globalThis.fetch = async () => { calls += 1; return new Response(null, { status: 200 }); };
  Object.defineProperty(globalThis, 'location', {
    configurable: true,
    value: { hostname: '0x88.app' },
  });
  try {
    assert.equal(await checkBigNetAsset(config), 'missing');
    assert.equal(bigNetAssetStatusSync(config), 'missing');
    assert.equal(calls, 0);
  } finally {
    globalThis.fetch = originalFetch;
    if (originalLocation) Object.defineProperty(globalThis, 'location', originalLocation);
    else delete globalThis.location;
  }
});

test('IPv6 loopback still probes local big-net assets', async () => {
  const originalFetch = globalThis.fetch;
  const originalLocation = Object.getOwnPropertyDescriptor(globalThis, 'location');
  const config = testConfig('/models/lc0/unit-local-big-net.onnx');
  let calls = 0;
  globalThis.fetch = async () => { calls += 1; return new Response(null, { status: 404 }); };
  Object.defineProperty(globalThis, 'location', {
    configurable: true,
    value: { hostname: '[::1]' },
  });
  try {
    assert.equal(await checkBigNetAsset(config), 'missing');
    assert.equal(calls, 1);
  } finally {
    globalThis.fetch = originalFetch;
    if (originalLocation) Object.defineProperty(globalThis, 'location', originalLocation);
    else delete globalThis.location;
  }
});

test('big-net asset probe records missing assets', async () => {
  const originalFetch = globalThis.fetch;
  const config = testConfig('/unit/big-net-missing.onnx');
  globalThis.fetch = async () => new Response(null, { status: 404 });
  try {
    assert.equal(await checkBigNetAsset(config), 'missing');
    assert.equal(bigNetAssetStatusSync(config), 'missing');
  } finally {
    globalThis.fetch = originalFetch;
  }
});
