import assert from 'node:assert/strict';
import test from 'node:test';
import { probeAssetUrls } from '../src/lc0/assetProbe.ts';
import { checkBerserkVariantAsset } from '../src/lc0/berserkVariants.ts';
import { checkBigNetAsset } from '../src/lc0/bt4Engine.ts';
import { checkPlentyChessVariantAsset } from '../src/lc0/plentychessVariants.ts';
import { checkRecklessVariantAsset } from '../src/lc0/recklessVariants.ts';
import { checkStormphraxVariantAsset } from '../src/lc0/stormphraxVariants.ts';
import { checkViridithasVariantAsset } from '../src/lc0/viridithasVariants.ts';

test('shared probe classifies HTTP status, network errors and bounded aborts', async (t) => {
  for (const [status, expected] of [
    [200, 'present'],
    [204, 'present'],
    [404, 'missing'],
    [410, 'missing'],
    [403, 'unknown'],
    [429, 'unknown'],
    [500, 'unknown'],
    [503, 'unknown'],
  ]) {
    t.mock.method(globalThis, 'fetch', async () => new Response(null, { status }));
    assert.equal(await probeAssetUrls([`/engine-${status}`]), expected, String(status));
  }
  t.mock.method(globalThis, 'fetch', async () => {
    throw Error('offline');
  });
  assert.equal(await probeAssetUrls(['/offline']), 'unknown');
  t.mock.method(globalThis, 'fetch', () => new Promise(() => {}));
  assert.equal(await probeAssetUrls(['/hanging'], { timeoutMs: 5 }), 'unknown', 'even a fetch ignoring abort is bounded');
  assert.equal(await probeAssetUrls(['/aborted'], { signal: AbortSignal.abort() }), 'unknown');
});
for (const [name, check, ok] of [
  ['Berserk', checkBerserkVariantAsset, 'present'],
  ['PlentyChess', checkPlentyChessVariantAsset, 'present'],
  ['Reckless', checkRecklessVariantAsset, 'present'],
  ['Stormphrax', checkStormphraxVariantAsset, 'present'],
  ['Viridithas', checkViridithasVariantAsset, 'ok'],
  ['BT4', checkBigNetAsset, 'present'],
]) {
  test(`${name} retries transient availability failures, shares concurrent probes, and caches only definitive results`, async (t) => {
    t.mock.timers.enable({ apis: ['Date', 'setTimeout'] });
    const variant = {
      key: 'custom',
      jsUrl: `/${name}/probe.js`,
      wasmUrl: `/${name}/probe.wasm`,
      dataUrl: `/${name}/probe.data`,
      modelUrl: `/${name}/probe.onnx`,
    };
    let calls = 0;
    let status = 503;
    t.mock.method(globalThis, 'fetch', async () => {
      calls++;
      return new Response(null, { status });
    });
    assert.equal(await check(variant), 'unknown');
    const firstCalls = calls;
    status = 200;
    assert.equal(await check(variant), 'unknown', 'retry waits for cooldown');
    assert.equal(calls, firstCalls);
    t.mock.timers.tick(1_000);
    const results = await Promise.all([check(variant), check(variant)]);
    assert.deepEqual(results, [ok, ok]);
    assert.equal(calls, firstCalls * 2, 'concurrent callers share one retry');
    await check(variant);
    assert.equal(calls, firstCalls * 2, 'definitive success is cached');
    const missing = { ...variant, assetStatus: undefined, wasmUrl: `/${name}/absent.wasm`, jsUrl: `/${name}/absent.js`, modelUrl: `/${name}/absent.onnx` };
    status = 410;
    assert.equal(await check(missing), 'missing');
    const before = calls;
    status = 200;
    assert.equal(await check(missing), 'missing');
    assert.equal(calls, before);
  });
}

for (const [name, check] of [
  ['Berserk', checkBerserkVariantAsset],
  ['PlentyChess', checkPlentyChessVariantAsset],
  ['Reckless', checkRecklessVariantAsset],
  ['Stormphrax', checkStormphraxVariantAsset],
  ['Viridithas', checkViridithasVariantAsset],
  ['BT4', checkBigNetAsset],
]) {
  for (const immediate of [false, true]) {
    test(`${name} completion callbacks cannot restart ${immediate ? 'immediate' : '503'} failures before exponential backoff`, async (t) => {
      t.mock.timers.enable({ apis: ['Date', 'setTimeout'] });
      let calls = 0;
      let notifications = 0;
      const variant = {
        key: 'custom',
        jsUrl: `/${name}/persistent-${immediate}.js`,
        wasmUrl: `/${name}/persistent-${immediate}.wasm`,
        dataUrl: `/${name}/persistent-${immediate}.data`,
        modelUrl: `/${name}/persistent-${immediate}.onnx`,
      };
      t.mock.method(globalThis, 'fetch', async () => {
        calls++;
        if (immediate) throw new TypeError('malformed custom URL');
        return new Response(null, { status: 503 });
      });
      const onChange = () => {
        notifications++;
        if (notifications < 20) void check(variant, onChange);
      };
      await check(variant, onChange);
      for (let i = 0; i < 20; i++) await Promise.resolve();
      const firstCalls = calls;
      assert.ok(firstCalls > 0);
      assert.ok(notifications < 10, `notifications must settle: ${notifications}`);
      for (let i = 0; i < 10; i++) await check(variant, onChange);
      assert.equal(calls, firstCalls);
      t.mock.timers.tick(999);
      await check(variant, onChange);
      assert.equal(calls, firstCalls);
      t.mock.timers.tick(1);
      await check(variant, onChange);
      assert.equal(calls, firstCalls * 2);
      t.mock.timers.tick(1_999);
      await check(variant, onChange);
      assert.equal(calls, firstCalls * 2);
      t.mock.timers.tick(1);
      await check(variant, onChange);
      assert.equal(calls, firstCalls * 3);
    });
  }
}
