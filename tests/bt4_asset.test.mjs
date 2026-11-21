import assert from 'node:assert/strict';
import test from 'node:test';
import {
  BT4_NET,
  bigNetAssetStatusSync,
  bigNetCpuWarning,
  bigNetOptionState,
  bigNetSelectable,
  checkBigNetAsset,
  probeBt4Support,
  renderBigNetCpuWarning,
  T3_NET,
} from '../src/lc0/bt4Engine.ts';
import { setRequestedOrtExecutionProviderForCurrentThread } from '../src/nn/ortConfig.ts';

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

test('an explicit WASM pin warns for a hosted big net on the first render', async (t) => {
  const engine = await import('../src/lc0/bt4Engine.ts?wasm-pin');
  const config = testConfig('/unit/big-net-wasm-pin.onnx');
  t.mock.method(globalThis, 'fetch', async () => new Response(null, { status: 200 }));
  await engine.checkBigNetAsset(config);
  setRequestedOrtExecutionProviderForCurrentThread('wasm');
  try {
    const probe = engine.probeBt4Support();
    assert.match(engine.bigNetCpuWarning(config), /CPU fallback/);
    assert.equal(await probe, false);
  } finally {
    setRequestedOrtExecutionProviderForCurrentThread(null);
  }
});

test('desktop can select either hosted big net without WebGPU and only sees the warning for CPU', async (t) => {
  t.mock.method(globalThis, 'fetch', async () => new Response(null, { status: 200 }));
  const target = { textContent: '', hidden: true };
  for (const [net, seconds] of [
    [BT4_NET, '1.5'],
    [T3_NET, '1.1'],
  ]) {
    const config = { ...net, modelUrl: `/unit/${net.key}-cpu.onnx` };
    assert.equal(bigNetSelectable(config), false);
    assert.equal(bigNetOptionState(config).disabled, true);
    await checkBigNetAsset(config);
    assert.equal(bigNetSelectable(config), true);
    assert.equal(bigNetOptionState(config).disabled, false);
    renderBigNetCpuWarning(target, [bigNetCpuWarning(config, 'auto->webgpu,wasm')]);
    assert.equal(target.hidden, true);
    renderBigNetCpuWarning(target, [bigNetCpuWarning(config, 'auto->wasm[bundled:cpu]')]);
    assert.equal(target.hidden, false);
    assert.ok(target.textContent.includes(`${seconds} s/eval`));
    assert.equal(await probeBt4Support(), false);
    assert.ok(bigNetCpuWarning(config).includes('CPU fallback'));
    renderBigNetCpuWarning(target, [bigNetCpuWarning(config, 'auto->webgpu')]);
    assert.deepEqual(target, { textContent: '', hidden: true });
  }
});

test('a cached big net running on WASM warns even without a successful asset probe', async (t) => {
  const config = testConfig('/unit/big-net-cached.onnx');
  t.mock.method(globalThis, 'fetch', async () => new Response(null, { status: 404 }));
  assert.equal(await probeBt4Support(), false);
  const assertWarnings = () => {
    assert.equal(bigNetCpuWarning(config), '', 'do not predict CPU fallback for an unconfirmed asset');
    assert.match(bigNetCpuWarning(config, 'auto->wasm'), /CPU fallback/);
    assert.equal(bigNetCpuWarning(config, 'auto->webgpu'), '');
  };
  assert.equal(bigNetAssetStatusSync(config), 'unknown');
  assertWarnings();
  assert.equal(await checkBigNetAsset(config), 'missing');
  assertWarnings();
});

test('production skips known-unhosted big-net asset probes', async () => {
  const originalFetch = globalThis.fetch;
  const originalLocation = Object.getOwnPropertyDescriptor(globalThis, 'location');
  const config = testConfig('/models/lc0/unit-unhosted-big-net.onnx');
  let calls = 0;
  globalThis.fetch = async () => {
    calls += 1;
    return new Response(null, { status: 200 });
  };
  Object.defineProperty(globalThis, 'location', {
    configurable: true,
    value: { hostname: '0x88.app' },
  });
  try {
    assert.equal(await checkBigNetAsset(config), 'missing');
    assert.equal(bigNetAssetStatusSync(config), 'missing');
    const r2Config = testConfig('https://assets.0x88.app/models/lc0/unit-unhosted-big-net-r2.onnx');
    assert.equal(await checkBigNetAsset(r2Config), 'missing');
    assert.equal(bigNetAssetStatusSync(r2Config), 'missing');
    assert.equal(calls, 0);
    const diagnostic = { ...r2Config, diagnosticModel: true };
    assert.equal(await checkBigNetAsset(diagnostic), 'present', 'explicit lab config probes models outside the shipping allowlist');
    assert.equal(bigNetAssetStatusSync(diagnostic), 'present');
    assert.equal(bigNetAssetStatusSync(r2Config), 'missing', 'diagnostic probe cannot change shipping selector state');
    assert.equal(calls, 1);
  } finally {
    globalThis.fetch = originalFetch;
    if (originalLocation) Object.defineProperty(globalThis, 'location', originalLocation);
    else delete globalThis.location;
  }
});

test('production probes canonical deployed big-net paths even with a prefixed asset base', async (t) => {
  const originalLocation = Object.getOwnPropertyDescriptor(globalThis, 'location');
  Object.defineProperty(globalThis, 'location', {
    configurable: true,
    value: new URL('https://0x88.app/?assetBase=https://0x88.app/prefix'),
  });
  t.after(() => {
    if (originalLocation) Object.defineProperty(globalThis, 'location', originalLocation);
    else delete globalThis.location;
  });
  const { BT4_NET: prefixedNet, checkBigNetAsset: checkWithPrefixedBase } = await import('../src/lc0/bt4Engine.ts?prefixed-base');
  assert.match(prefixedNet.modelUrl, /^https:\/\/0x88\.app\/prefix\/models\/lc0\//);
  const fetch = t.mock.method(globalThis, 'fetch', async () => new Response(null, { status: 200 }));
  const config = testConfig('/models/lc0/BT4-1024x15x32h-swa-6147500-policytune-332.batch8.f16.qdq8.onnx');
  assert.equal(await checkWithPrefixedBase(config), 'present');
  assert.equal(fetch.mock.callCount(), 1);
});

test('IPv6 loopback still probes local big-net assets', async () => {
  const originalFetch = globalThis.fetch;
  const originalLocation = Object.getOwnPropertyDescriptor(globalThis, 'location');
  const config = testConfig('/models/lc0/unit-local-big-net.onnx');
  let calls = 0;
  globalThis.fetch = async () => {
    calls += 1;
    return new Response(null, { status: 404 });
  };
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
