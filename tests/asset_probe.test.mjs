import assert from 'node:assert/strict';
import test from 'node:test';
import { assetProbeRetryAllowed, probeAssetUrls, subscribeAssetProbeRetry } from '../src/lc0/assetProbe.ts';
import { berserkVariantAssetStatus, checkBerserkVariantAsset } from '../src/lc0/berserkVariants.ts';
import { browserVariantOption } from '../src/lc0/browserVariantOption.ts';
import { bigNetAssetStatusSync, bigNetOptionState, checkBigNetAsset } from '../src/lc0/bt4Engine.ts';
import { checkPlentyChessVariantAsset, plentyChessVariantAssetStatus } from '../src/lc0/plentychessVariants.ts';
import { checkRecklessVariantAsset, recklessVariantAssetStatus } from '../src/lc0/recklessVariants.ts';
import { checkStormphraxVariantAsset, stormphraxVariantAssetStatus } from '../src/lc0/stormphraxVariants.ts';
import { checkViridithasVariantAsset, viridithasVariantAssetStatus } from '../src/lc0/viridithasVariants.ts';

for (const code of [404, 410]) {
  test(`confirmed ${code} resolves missing even when another required HEAD hangs`, async (t) => {
    t.mock.timers.enable({ apis: ['setTimeout'] });
    let hangingSignal;
    let result;
    t.mock.method(globalThis, 'fetch', async (url, options) => {
      if (url.includes('missing')) return new Response(null, { status: code });
      hangingSignal = options.signal;
      return new Promise(() => {});
    });
    const probe = probeAssetUrls([`/missing-${code}`, `/hang-${code}`], { timeoutMs: 10 }).then((status) => (result = status));
    for (let i = 0; i < 20; i++) await Promise.resolve();
    assert.equal(result, 'missing', 'missing resolves without waiting for timeout');
    assert.equal(hangingSignal.aborted, true, 'cancel the remaining HEAD');
    t.mock.timers.tick(10);
    await probe;
    assert.equal(result, 'missing');
  });
}

test('shared probe classifies HTTP status, network errors and bounded aborts', async (t) => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
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
  const hanging = probeAssetUrls(['/hanging'], { timeoutMs: 5 });
  t.mock.timers.tick(5);
  assert.equal(await hanging, 'unknown', 'even a fetch ignoring abort is bounded');
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

const flush = async () => {
  for (let i = 0; i < 40; i++) await Promise.resolve();
};

for (const [name, check, read, ok] of [
  ['Berserk', checkBerserkVariantAsset, berserkVariantAssetStatus, 'present'],
  ['PlentyChess', checkPlentyChessVariantAsset, plentyChessVariantAssetStatus, 'present'],
  ['Reckless', checkRecklessVariantAsset, recklessVariantAssetStatus, 'present'],
  ['Stormphrax', checkStormphraxVariantAsset, stormphraxVariantAssetStatus, 'present'],
  ['Viridithas', checkViridithasVariantAsset, viridithasVariantAssetStatus, 'ok'],
  ['BT4', checkBigNetAsset, bigNetAssetStatusSync, 'present'],
]) {
  for (const recover of [true, false]) {
    test(`${name} render-driven probes ${recover ? 'recover automatically after cooldown' : 'settle missing after three failures'}`, async (t) => {
      t.mock.timers.enable({ apis: ['Date', 'setTimeout'] });
      const variant = {
        key: 'custom',
        jsUrl: `/${name}/render-${recover}.js`,
        wasmUrl: `/${name}/render-${recover}.wasm`,
        dataUrl: `/${name}/render-${recover}.data`,
        modelUrl: `/${name}/render-${recover}.onnx`,
      };
      let calls = 0;
      let notifications = 0;
      let status = 503;
      t.mock.method(globalThis, 'fetch', async () => {
        calls++;
        return new Response(null, { status });
      });
      // The pickers render synchronously on each notification and immediately
      // check again while their cached status is unknown/checking.
      const render = () => {
        notifications++;
        if (['unknown', 'checking'].includes(read(variant))) void check(variant, render);
      };
      render();
      await flush();
      const firstCalls = calls;
      const firstNotifications = notifications;
      assert.ok(firstCalls > 0);
      if (recover) status = 200;
      t.mock.timers.tick(999);
      await flush();
      assert.equal(calls, firstCalls);
      assert.equal(notifications, firstNotifications, 'silent during cooldown');
      t.mock.timers.tick(1);
      await flush();
      assert.equal(calls, firstCalls * 2, 'cooldown wakes the render without external activity');
      if (!recover) {
        const secondNotifications = notifications;
        t.mock.timers.tick(1_999);
        await flush();
        assert.equal(notifications, secondNotifications);
        t.mock.timers.tick(1);
        await flush();
        assert.equal(calls, firstCalls * 3);
      }
      assert.equal(read(variant), recover ? ok : 'missing');
      if (!recover) {
        const option = name === 'BT4' ? bigNetOptionState(variant) : browserVariantOption('test', name, { assetStatus: read(variant), requirePresent: true });
        assert.equal(option.disabled, true);
        assert.match(option.label ?? option.suffix, /missing|not hosted/);
      }
      const settledNotifications = notifications;
      t.mock.timers.tick(120_000);
      await flush();
      assert.equal(calls, firstCalls * (recover ? 2 : 3));
      assert.equal(notifications, settledNotifications, 'no retry or notification after settling');
    });
  }
}

test('retry notifications share one unref timer and cancel when the last listener leaves', async (t) => {
  const urls = ['/cancelled-retry.wasm'];
  t.mock.method(globalThis, 'fetch', async () => new Response(null, { status: 503 }));
  assert.equal(await probeAssetUrls(urls), 'unknown');
  let unrefs = 0;
  const handle = { unref: () => unrefs++ };
  const schedule = t.mock.method(globalThis, 'setTimeout', () => handle);
  const cancel = t.mock.method(globalThis, 'clearTimeout', () => {});
  assert.equal(assetProbeRetryAllowed(urls), false);
  assert.equal(schedule.mock.callCount(), 0, 'no timer without a listener');
  const listener = () => assert.fail('cancelled listener must not run');
  const unsubscribe = subscribeAssetProbeRetry(urls, listener);
  assert.equal(subscribeAssetProbeRetry(urls, listener), unsubscribe, 'deduplicate the same render');
  const controller = new AbortController();
  subscribeAssetProbeRetry(urls, () => assert.fail('aborted listener must not run'), controller.signal);
  assert.equal(schedule.mock.callCount(), 1);
  assert.equal(unrefs, 1, 'Node timer cannot keep the process alive');
  unsubscribe();
  assert.equal(cancel.mock.callCount(), 0, 'another listener still needs the timer');
  controller.abort();
  assert.equal(cancel.mock.callCount(), 1, 'cancel timer as soon as no listeners remain');
  assert.equal(cancel.mock.calls[0].arguments[0], handle);
});

test('the shared helper keeps exhausted probes missing without more requests', async (t) => {
  t.mock.timers.enable({ apis: ['Date', 'setTimeout'] });
  const fetch = t.mock.method(globalThis, 'fetch', async () => new Response(null, { status: 503 }));
  const urls = ['/exhausted-probe.wasm'];
  assert.equal(await probeAssetUrls(urls), 'unknown');
  t.mock.timers.tick(1_000);
  assert.equal(await probeAssetUrls(urls), 'unknown');
  t.mock.timers.tick(2_000);
  assert.equal(await probeAssetUrls(urls), 'missing');
  t.mock.timers.tick(120_000);
  assert.equal(await probeAssetUrls(urls), 'missing');
  assert.equal(fetch.mock.callCount(), 3);
});
