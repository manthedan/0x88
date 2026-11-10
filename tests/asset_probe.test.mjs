import assert from 'node:assert/strict';
import { getEventListeners } from 'node:events';
import test from 'node:test';
import { assetProbeRetryAllowed, notifyAssetProbeChange, probeAssetUrls, subscribeAssetProbeChange, subscribeAssetProbeRetry } from '../src/lc0/assetProbe.ts';
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
      await flush(t);
    });
  }
}

const flush = async (t) => {
  for (let i = 0; i < 40; i++) await Promise.resolve();
  t.mock.timers.tick(0);
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
      await flush(t);
      const firstCalls = calls;
      const firstNotifications = notifications;
      assert.ok(firstCalls > 0);
      if (recover) status = 200;
      t.mock.timers.tick(999);
      await flush(t);
      assert.equal(calls, firstCalls);
      assert.equal(notifications, firstNotifications, 'silent during cooldown');
      t.mock.timers.tick(1);
      await flush(t);
      assert.equal(calls, firstCalls * 2, 'cooldown wakes the render without external activity');
      if (!recover) {
        const secondNotifications = notifications;
        t.mock.timers.tick(1_999);
        await flush(t);
        assert.equal(notifications, secondNotifications);
        t.mock.timers.tick(1);
        await flush(t);
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
      await flush(t);
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

// Match Analysis/Arena: every callback is a fresh closure, and each render
// checks all three variants, including probes that are already in flight.
for (const [name, check, read, ok, headsPerVariant] of [
  ['Berserk', checkBerserkVariantAsset, berserkVariantAssetStatus, 'present', 3],
  ['PlentyChess', checkPlentyChessVariantAsset, plentyChessVariantAssetStatus, 'present', 3],
  ['Reckless', checkRecklessVariantAsset, recklessVariantAssetStatus, 'present', 1],
  ['Stormphrax', checkStormphraxVariantAsset, stormphraxVariantAssetStatus, 'present', 3],
  ['Viridithas', checkViridithasVariantAsset, viridithasVariantAssetStatus, 'ok', 1],
  ['BT4', checkBigNetAsset, bigNetAssetStatusSync, 'present', 1],
]) {
  for (const recoverOnThird of [false, true]) {
    for (const extraRenders of [0, 1_000]) {
      test(`${name} batches fresh render closures across three variants and ${recoverOnThird ? 'recovery' : 'three failures then server recovery'} (${extraRenders} unrelated renders)`, async (t) => {
        t.mock.timers.enable({ apis: ['Date', 'setTimeout'] });
        const variants = Array.from({ length: 3 }, (_, i) => ({
          key: 'custom',
          jsUrl: `/${name}/fresh-${recoverOnThird}-${extraRenders}-${i}.js`,
          wasmUrl: `/${name}/fresh-${recoverOnThird}-${extraRenders}-${i}.wasm`,
          dataUrl: `/${name}/fresh-${recoverOnThird}-${extraRenders}-${i}.data`,
          modelUrl: `/${name}/fresh-${recoverOnThird}-${extraRenders}-${i}.onnx`,
        }));
        let requests = 0;
        let renders = 1;
        let status = 503;
        t.mock.method(globalThis, 'fetch', async (_url, options) => {
          assert.equal(options.method, 'HEAD');
          requests++;
          return new Response(null, { status });
        });
        const render = () => {
          for (const variant of variants) {
            if (['unknown', 'checking'].includes(read(variant)))
              void check(variant, () => {
                renders++;
                render();
              });
          }
        };
        const unrelatedRenders = () => {
          for (let i = 0; i < extraRenders; i++) render();
        };
        const settle = async () => {
          for (let i = 0; i < 4; i++) {
            await flush(t);
            t.mock.timers.tick(0);
          }
          await flush(t);
        };
        const counts = [];
        render();
        unrelatedRenders(); // In-flight subscribers are bounded too.
        await settle();
        counts.push(renders);
        unrelatedRenders(); // Cooldown subscriptions cannot accumulate either.
        t.mock.timers.tick(999);
        await settle();
        assert.equal(renders, counts[0], 'cooldown is silent despite unrelated renders');
        assert.equal(requests, 3 * headsPerVariant);
        t.mock.timers.tick(1);
        await settle();
        counts.push(renders);
        assert.equal(requests, 6 * headsPerVariant);
        unrelatedRenders();
        if (recoverOnThird) status = 200;
        t.mock.timers.tick(2_000);
        await settle();
        counts.push(renders);
        status = 200;
        t.mock.timers.tick(120_000);
        await settle();
        t.diagnostic(`renders=${counts.join(' → ')}; HEADs=${requests}`);
        assert.deepEqual(variants.map(read), Array(3).fill(recoverOnThird ? ok : 'missing'));
        assert.equal(requests, 9 * headsPerVariant, 'three attempts per variant, then no requests after settlement');
        assert.ok(counts[0] <= 13, 'at most four callbacks per key in a completion batch, plus the initial render');
        assert.ok(counts[1] <= 37, 'at most twelve cooldown and twelve completion callbacks per attempt');
        assert.ok(counts[2] <= 61, 'three attempts have a fixed callback budget');
        assert.deepEqual(counts, extraRenders ? [13, 37, 61] : [4, 25, 49]);
        assert.equal(renders, counts[2], 'server recovery after exhaustion cannot restart the settled picker');
      });
    }
  }
}

test('state changes and simultaneous cooldowns notify each distinct callback once', async (t) => {
  t.mock.timers.enable({ apis: ['Date', 'setTimeout'] });
  const schedule = t.mock.method(globalThis, 'setTimeout');
  t.mock.method(globalThis, 'fetch', async () => new Response(null, { status: 503 }));
  const groups = [0, 1, 2].map((i) => [`/coalesced-${i}.wasm`]);
  let notifications = 0;
  const listener = () => notifications++;
  const controllers = groups.map(() => new AbortController());
  for (const [i, urls] of groups.entries()) subscribeAssetProbeChange(urls, listener, controllers[i].signal);
  await Promise.all(groups.map((urls) => probeAssetUrls(urls)));
  assert.equal(notifications, 0, 'defer until every completion in this tick has updated its status');
  assert.equal(schedule.mock.calls.filter(({ arguments: args }) => args[1] === 0).length, 1, 'one shared completion timer');
  await flush(t);
  assert.equal(notifications, 1, 'same callback across three keys runs once');
  for (const controller of controllers) assert.equal(getEventListeners(controller.signal, 'abort').length, 0);
  const beforeRetry = schedule.mock.callCount();
  for (const [i, urls] of groups.entries()) subscribeAssetProbeRetry(urls, listener, controllers[i].signal);
  assert.equal(schedule.mock.callCount(), beforeRetry + 1, 'one shared cooldown timer');
  t.mock.timers.tick(999);
  assert.equal(notifications, 1);
  t.mock.timers.tick(1);
  assert.equal(notifications, 2, 'all equal cooldown deadlines form one round');
  for (const controller of controllers) assert.equal(getEventListeners(controller.signal, 'abort').length, 0);
  const settledTimers = schedule.mock.callCount();
  for (const urls of groups) notifyAssetProbeChange(urls);
  assert.equal(schedule.mock.callCount(), settledTimers, 'listener sets are empty after one-shot delivery');
});

test('notification rounds detach every due key before a render can resubscribe', async (t) => {
  t.mock.timers.enable({ apis: ['Date', 'setTimeout'] });
  const groups = [0, 1, 2].map((i) => [`/one-shot-${i}.wasm`]);
  const controller = new AbortController();
  let notifications = 0;
  const listener = () => {
    notifications++;
    assert.equal(getEventListeners(controller.signal, 'abort').length, 0, 'all previous subscriptions are detached first');
    if (notifications === 1) for (const urls of groups) subscribeAssetProbeChange(urls, listener, controller.signal);
  };
  for (const urls of groups) {
    subscribeAssetProbeChange(urls, listener, controller.signal);
    notifyAssetProbeChange(urls);
  }
  await flush(t);
  assert.equal(notifications, 1, 'render subscriptions join the next notification round');
  for (const urls of groups) notifyAssetProbeChange(urls);
  await flush(t);
  assert.equal(notifications, 2, 'the next change delivers exactly one new notification');
  assert.equal(getEventListeners(controller.signal, 'abort').length, 0);
  t.mock.timers.tick(120_000);
  assert.equal(notifications, 2, 'one-shot delivery leaves no further notification');
});

test('fresh subscription replacement retains only the newest four callbacks per key', async (t) => {
  t.mock.timers.enable({ apis: ['Date', 'setTimeout'] });
  const schedule = t.mock.method(globalThis, 'setTimeout');
  const urls = ['/bounded-subscribers.wasm'];
  const controllers = Array.from({ length: 100 }, () => new AbortController());
  const notified = [];
  for (const [i, controller] of controllers.entries()) subscribeAssetProbeChange(urls, () => notified.push(i), controller.signal);
  assert.equal(
    controllers.reduce((sum, controller) => sum + getEventListeners(controller.signal, 'abort').length, 0),
    4,
    'evicted closures and abort handlers are released',
  );
  notifyAssetProbeChange(urls);
  await flush(t);
  assert.deepEqual(notified, [96, 97, 98, 99]);
  for (const controller of controllers) assert.equal(getEventListeners(controller.signal, 'abort').length, 0);
  const settledTimers = schedule.mock.callCount();
  notifyAssetProbeChange(urls);
  assert.equal(schedule.mock.callCount(), settledTimers, 'the listener set is empty after delivery');
});

for (const exit of ['unsubscribe', 'abort', 'never called again']) {
  test(`${exit} leaves no timer or retained listener after probe settlement`, async (t) => {
    t.mock.timers.enable({ apis: ['Date', 'setTimeout'] });
    const schedule = t.mock.method(globalThis, 'setTimeout');
    const cancel = t.mock.method(globalThis, 'clearTimeout');
    const urls = [`/cleanup-${exit}.wasm`];
    const controller = new AbortController();
    let notifications = 0;
    let finish;
    t.mock.method(
      globalThis,
      'fetch',
      () =>
        new Promise((resolve) => {
          finish = resolve;
        }),
    );
    const unsubscribe = subscribeAssetProbeChange(urls, () => notifications++, controller.signal);
    const probe = probeAssetUrls(urls);
    if (exit === 'unsubscribe') unsubscribe();
    if (exit === 'abort') controller.abort();
    finish(new Response(null, { status: 503 }));
    await probe;
    await flush(t);
    assert.equal(notifications, exit === 'never called again' ? 1 : 0);
    assert.equal(getEventListeners(controller.signal, 'abort').length, 0, 'no retained abort handler or render closure');
    assert.ok(
      cancel.mock.calls.some(({ arguments: args }) => args[0] === schedule.mock.calls[0].result),
      'the probe timeout was cleared',
    );
    const settledTimers = schedule.mock.callCount();
    notifyAssetProbeChange(urls);
    assert.equal(schedule.mock.callCount(), settledTimers, 'no listener set survives settlement');
    t.mock.timers.tick(120_000);
    assert.equal(notifications, exit === 'never called again' ? 1 : 0, 'no deferred retry remains');
  });
}
