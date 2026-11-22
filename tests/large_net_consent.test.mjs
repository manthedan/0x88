import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import { BIG_NETS, bigNetLoadWarning } from '../src/lc0/bt4Engine.ts';
import { requestLargeNetConsent, requestLargeNetDownloadConsent, requestLargeNetRedownloadConsent } from '../src/lc0/largeNetConsent.ts';
import { inspectLargeNetDownload } from '../src/lc0/largeNetDownload.ts';
import { lc0SplitKUrl } from '../src/lc0/lc0SmallModelSelection.ts';
import { createSoakBigNetBinding } from '../src/lc0/mobileSoakBigNet.ts';
import { requestSoakLargeNetConsent } from '../src/lc0/mobileSoakBrowser.ts';
import { phoneLargeNetPreference } from '../src/lc0/phoneLargeNetPolicy.ts';
import { createOrtSession, InferenceSession, resolveOrtSessionPlacement } from '../src/nn/ortRuntime.ts';

const tick = () => new Promise(setImmediate);
function browser(t, device = 'phone') {
  class Element extends EventTarget {
    children = [];
    attributes = {};
    isConnected = true;
    append(...nodes) {
      this.children.push(...nodes);
    }
    setAttribute(key, value) {
      this.attributes[key] = value;
    }
    getAttribute(key) {
      return this.attributes[key] ?? null;
    }
    showModal() {
      this.open = true;
    }
    close() {
      if (!this.open) return;
      this.open = false;
      setImmediate(() => this.dispatchEvent(new Event('close')));
    }
    remove() {
      this.isConnected = false;
    }
    focus() {
      document.activeElement = this;
    }
  }
  const previousFocus = new Element();
  const body = new Element();
  const globals = {
    document: { body, activeElement: previousFocus, createElement: () => new Element(), getElementById: () => null },
    location: new URL(`https://example.test/?memoryClass=${device}`),
    caches: undefined,
  };
  for (const [key, value] of Object.entries(globals)) {
    const previous = Object.getOwnPropertyDescriptor(globalThis, key);
    Object.defineProperty(globalThis, key, { configurable: true, value });
    t.after(() => {
      if (previous) Object.defineProperty(globalThis, key, previous);
      else delete globalThis[key];
    });
  }
  const abort = new AbortController();
  phoneLargeNetPreference.setEnabled(false);
  t.after(() => {
    abort.abort();
    phoneLargeNetPreference.setEnabled(false);
  });
  const state = { cached: false, downloadBytes: undefined };
  const inspect = async (config) => ({ ...state, url: config.modelUrl, downloadBytes: state.downloadBytes ?? config.approxMb * 1_000_000 });
  return {
    body,
    state,
    previousFocus,
    abort,
    request: (config, overrides = {}) =>
      requestLargeNetConsent(config, overrides.signal ?? abort.signal, overrides.desktopWarning ?? false, undefined, overrides.inspect ?? inspect),
    async dialog() {
      for (let i = 0; i < 100; i++) {
        const dialog = body.children.find((node) => node.isConnected && node.open);
        if (dialog) return dialog;
        await tick();
      }
      assert.fail('expected a consent dialog');
    },
  };
}
function choose(dialog, accept) {
  dialog.children.at(-1).children[accept ? 1 : 0].dispatchEvent(new Event('click'));
}

for (const key of ['bt4', 't3']) {
  for (const cached of [false, true]) {
    test(`${key} ${cached ? 'cached' : 'uncached'} phone warning uses per-net title and cache-aware copy`, async (t) => {
      const f = browser(t);
      f.state.cached = cached;
      const pending = f.request(BIG_NETS[key]);
      const dialog = await f.dialog();
      const name = key === 'bt4' ? 'BT4' : 't3';
      assert.equal(dialog.children[0].textContent, `Use ${name} on this phone?`);
      assert.equal(
        dialog.children[1].textContent,
        `Large nets can crash the tab on phones.${cached ? '' : ` ${name} is a ${BIG_NETS[key].approxMb} MB download, cached after the first time.`}`,
      );
      assert.equal(dialog.children.length, 3, 'no extra fine print');
      assert.deepEqual(
        dialog.children[2].children.map((node) => node.textContent),
        ['Cancel', 'Use anyway'],
      );
      assert.equal(dialog.attributes['aria-describedby'], 'largeNetConsentDescription');
      choose(dialog, false);
      assert.equal(await pending, false);
      assert.equal(document.activeElement, f.previousFocus);
      assert.equal(phoneLargeNetPreference.enabled(), false);
    });
  }
}

test('consent says whether it covers a download: a cache hit does not', async (t) => {
  const f = browser(t);
  const inspect = async (config) => ({ cached: f.state.cached, url: config.modelUrl, downloadBytes: 1_000_000 });
  const request = (config) => requestLargeNetDownloadConsent(config, f.abort.signal, false, undefined, inspect);
  let pending = request(BIG_NETS.bt4);
  choose(await f.dialog(), true);
  assert.equal(await pending, 'download');
  f.state.cached = true;
  assert.equal(await request(BIG_NETS.t3), 'cached', 'opted-in phones skip the dialog for a cached net');
  phoneLargeNetPreference.setEnabled(false);
  pending = request(BIG_NETS.t3);
  choose(await f.dialog(), true);
  assert.equal(await pending, 'cached', 'crash consent for a cached net approves no download');
});

test('large-net consent works without Promise.withResolvers and AbortSignal.any', async (t) => {
  for (const [object, key] of [
    [Promise, 'withResolvers'],
    [AbortSignal, 'any'],
  ]) {
    const original = Object.getOwnPropertyDescriptor(object, key);
    Object.defineProperty(object, key, { configurable: true, value: undefined });
    t.after(() => Object.defineProperty(object, key, original));
  }
  const f = browser(t);
  const pending = f.request(BIG_NETS.lqo);
  // A rejection must surface immediately rather than fail later while waiting for the dialog.
  const dialog = await Promise.race([pending.then(() => assert.fail('consent returned before showing a dialog')), f.dialog()]);
  choose(dialog, true);
  assert.equal(await pending, true);
  const hanging = f.request(BIG_NETS.lqo, { inspect: () => new Promise(() => {}) });
  f.abort.abort();
  assert.equal(await hanging, false);
});

test('download copy uses selected artifact bytes, not the stale config estimate', async (t) => {
  const f = browser(t);
  f.state.downloadBytes = 164789252;
  const pending = f.request(BIG_NETS.bt4);
  const dialog = await f.dialog();
  assert.equal(dialog.children[1].textContent, 'Large nets can crash the tab on phones. BT4 is a 165 MB download, cached after the first time.');
  choose(dialog, false);
  await pending;
});

test('BT4/t3 crash consent persists, but an uncached model still gets its download warning', async (t) => {
  const f = browser(t);
  const pending = f.request(BIG_NETS.bt4);
  choose(await f.dialog(), true);
  assert.equal(await pending, true);
  assert.equal(phoneLargeNetPreference.enabled(), true);
  f.state.cached = true;
  assert.equal(await f.request(BIG_NETS.t3), true);
  f.state.cached = false;
  const another = f.request(BIG_NETS.t3);
  const dialog = await f.dialog();
  assert.match(dialog.children[1].textContent, /t3 is a 163 MB download/);
  choose(dialog, false);
  assert.equal(await another, false);
  assert.equal(phoneLargeNetPreference.enabled(), true, 'cancelling a download does not revoke existing crash consent');
});

test('a phone already opted in to t3 still warns when only the split-K model is cached', async (t) => {
  const f = browser(t);
  phoneLargeNetPreference.setEnabled(true);
  const splitK = lc0SplitKUrl(BIG_NETS.t3.modelUrl);
  const inspect = (config, signal) =>
    inspectLargeNetDownload(config, signal, {
      placement: async () => ({ providers: ['webgpu', 'wasm'], adapterVendor: 'apple' }),
      inspect: async (url) => ({ url, cached: url === splitK, downloadBytes: url === splitK ? 170_000_000 : 160_000_000 }),
    });
  const pending = f.request(BIG_NETS.t3, { inspect });
  const dialog = await f.dialog();
  assert.equal(
    dialog.children[1].textContent,
    'Large nets can crash the tab on phones. Model downloads with fallback are estimated at 160 MB. Retries can use more data.',
  );
  choose(dialog, false);
  assert.equal(await pending, false);
});

test('t3 consent keeps the split-K size when the original artifact probe fails', async (t) => {
  const f = browser(t);
  phoneLargeNetPreference.setEnabled(true);
  const splitK = lc0SplitKUrl(BIG_NETS.t3.modelUrl);
  const inspect = (config, signal) =>
    inspectLargeNetDownload(config, signal, {
      placement: async () => ({ providers: ['webgpu', 'wasm'], adapterVendor: 'apple' }),
      inspect: async (url) => {
        if (url !== splitK) throw new Error('release metadata disagrees with the model manifest');
        return { url, cached: false, downloadBytes: 170_000_000 };
      },
    });
  const pending = f.request(BIG_NETS.t3, { inspect });
  const dialog = await f.dialog();
  assert.match(dialog.children[1].textContent, /One model is about 170 MB; the other model’s size is unavailable/);
  choose(dialog, false);
  assert.equal(await pending, false);
});

for (const accept of [false, true]) {
  test(`uncached Queen Odds has a download-only dialog (accepted=${accept}) without granting opt-in`, async (t) => {
    const f = browser(t);
    const pending = f.request(BIG_NETS.lqo);
    const dialog = await f.dialog();
    assert.equal(dialog.children[0].textContent, 'Download Queen Odds?');
    assert.equal(dialog.children[1].textContent, "It's a 96 MB download, cached after the first time.");
    assert.deepEqual(
      dialog.children[2].children.map((node) => node.textContent),
      ['Cancel', 'Download'],
    );
    choose(dialog, accept);
    assert.equal(await pending, accept);
    assert.equal(phoneLargeNetPreference.enabled(), false);
    f.state.cached = true;
    const count = f.body.children.length;
    assert.equal(await f.request(BIG_NETS.lqo), true);
    assert.equal(f.body.children.length, count, 'verified cached Queen Odds has no dialog');
  });
}

test('Queen Odds approval is not remembered if its model is still uncached', async (t) => {
  const f = browser(t);
  const first = f.request(BIG_NETS.lqo);
  choose(await f.dialog(), true);
  await first;
  const second = f.request(BIG_NETS.lqo);
  choose(await f.dialog(), false);
  assert.equal(await second, false);
});

test('aborting a slow cache probe returns without showing a late dialog', async (t) => {
  const f = browser(t);
  const probe = Promise.withResolvers();
  const pending = f.request(BIG_NETS.lqo, { inspect: () => probe.promise });
  f.abort.abort();
  assert.equal(await pending, false);
  probe.resolve({ url: BIG_NETS.lqo.modelUrl, cached: false, downloadBytes: 95662716 });
  await tick();
  assert.equal(f.body.children.length, 0);
});

test('a newer picker choice cancels a stale cache probe rather than opening two dialogs', async (t) => {
  const f = browser(t);
  const probe = Promise.withResolvers();
  const previous = f.request(BIG_NETS.bt4, { inspect: () => probe.promise });
  const current = f.request(BIG_NETS.lqo);
  assert.equal(await previous, false);
  const dialog = await f.dialog();
  assert.equal(dialog.children[0].textContent, 'Download Queen Odds?');
  choose(dialog, false);
  await current;
  probe.resolve({ url: BIG_NETS.bt4.modelUrl, cached: false });
  await tick();
  assert.equal(f.body.children.length, 1);
});

test('a BT4 cache retry leaves an open t3 picker prompt alone', async (t) => {
  const f = browser(t, 'desktop');
  const picker = f.request(BIG_NETS.t3, { desktopWarning: true });
  const dialog = await f.dialog();

  const retry = requestLargeNetRedownloadConsent(BIG_NETS.bt4, f.abort.signal);
  const activeDialog = await f.dialog();
  choose(activeDialog, true);
  assert.ok(activeDialog === dialog, 'the picker still owns the dialog');
  assert.equal(await retry, 'aborted');
  assert.equal(await picker, true);
});

test('a BT4 cache retry can prompt before the dismissed t3 picker fires its close event', async (t) => {
  const f = browser(t, 'desktop');
  const picker = f.request(BIG_NETS.t3, { desktopWarning: true });
  const pickerDialog = await f.dialog();
  choose(pickerDialog, false);
  assert.equal(pickerDialog.open, false);
  assert.equal(pickerDialog.isConnected, true, 'the picker close event has not fired');

  const retry = requestLargeNetRedownloadConsent(BIG_NETS.bt4, f.abort.signal);
  assert.equal(await picker, false);
  const retryDialog = await f.dialog();
  assert.notEqual(retryDialog, pickerDialog);
  choose(retryDialog, true);
  assert.equal(await retry, 'download');
});

test('a t3 picker supersedes an open BT4 cache retry without dismissing it', async (t) => {
  const f = browser(t, 'desktop');
  const retry = requestLargeNetRedownloadConsent(BIG_NETS.bt4, f.abort.signal);
  const retryDialog = await f.dialog();
  const picker = f.request(BIG_NETS.t3, { desktopWarning: true });
  const pickerDialog = await f.dialog();

  assert.equal(retryDialog.open, false);
  assert.notEqual(pickerDialog, retryDialog);
  choose(pickerDialog, true);
  assert.equal(await retry, 'aborted');
  assert.equal(await picker, true);

  const nextRetry = requestLargeNetRedownloadConsent(BIG_NETS.bt4, f.abort.signal);
  choose(await f.dialog(), false);
  assert.equal(await nextRetry, false, 'Cancel is an explicit decline');
});

test('a failed cache inspection never suppresses an uncached download warning', async (t) => {
  const f = browser(t);
  const pending = f.request(BIG_NETS.lqo, {
    inspect: async () => {
      throw new Error('denied');
    },
  });
  const dialog = await f.dialog();
  assert.match(dialog.children[1].textContent, /Download size is unavailable/);
  choose(dialog, false);
  assert.equal(await pending, false);
});

for (const key of Object.keys(BIG_NETS)) {
  test(`desktop ${key} keeps its existing warning and skip behavior without a cache probe`, async (t) => {
    const f = browser(t, 'desktop');
    const inspect = () => assert.fail('desktop behavior is not cache-aware');
    assert.equal(await f.request(BIG_NETS[key], { inspect }), true, 'Play still skips the desktop warning');
    const pending = f.request(BIG_NETS[key], { desktopWarning: true, inspect });
    const dialog = await f.dialog();
    assert.equal(dialog.children[0].textContent, `Use Lc0 ${BIG_NETS[key].name}?`);
    assert.equal(dialog.children[1].textContent, bigNetLoadWarning(BIG_NETS[key]));
    assert.deepEqual(
      dialog.children[2].children.map((node) => node.textContent),
      ['Cancel', 'Use anyway'],
    );
    choose(dialog, false);
    assert.equal(await pending, false);
  });
}

test('lab download warnings inspect effective Queen Odds overrides and do not bypass acquisition consent', async (t) => {
  const f = browser(t);
  const models = Object.values(BIG_NETS).flatMap((config) =>
    [config.modelUrl, lc0SplitKUrl(config.modelUrl)].filter(Boolean).map((url) => ({
      file: url.split('/').at(-1),
      url,
      bytes: config.approxMb * 1_000_000,
      sha256: 'a'.repeat(64),
    })),
  );
  t.mock.method(globalThis, 'fetch', async (url) => {
    assert.match(String(url), /manifest.json$/);
    return Response.json({ models });
  });
  const binding = createSoakBigNetBinding('bt4');
  assert.throws(() => binding.acquire(), /BT4 and t3 are off/);
  for (const [override, title, copy] of [
    [undefined, 'Download Queen Odds?', '96 MB'],
    [BIG_NETS.lqo.modelUrl, 'Download Queen Odds?', '96 MB'],
    [BIG_NETS.bt4.modelUrl, 'Use BT4 on this phone?', '178 MB'],
    [lc0SplitKUrl(BIG_NETS.t3.modelUrl), 'Use t3 on this phone?', '163 MB'],
    ['/models/lc0/unlisted.onnx', 'Use BT4 on this phone?', 'size is unavailable'],
  ]) {
    const pending = requestSoakLargeNetConsent(['lqo'], f.abort.signal, override ? { lqo: override } : undefined);
    const dialog = await f.dialog();
    assert.equal(dialog.children[0].textContent, title);
    assert.ok(dialog.children[1].textContent.includes(copy));
    choose(dialog, false);
    assert.equal(await pending, false);
    assert.equal(phoneLargeNetPreference.enabled(), false);
  }
  const pending = requestSoakLargeNetConsent(['bt4'], f.abort.signal);
  choose(await f.dialog(), true);
  assert.equal(await pending, true);
  const searcher = binding.acquire();
  assert.equal(binding.retire(searcher), true);
  await searcher.dispose();
});

test('t3 consent estimates the fresh Apple WebGPU worker downloads after Play locks the document to WASM', async (t) => {
  const f = browser(t);
  const navigator = Object.getOwnPropertyDescriptor(globalThis, 'navigator');
  Object.defineProperty(globalThis, 'navigator', {
    configurable: true,
    value: { gpu: { requestAdapter: async () => ({ features: new Set(['shader-f16']), info: { vendor: 'apple' } }) } },
  });
  t.after(() => Object.defineProperty(globalThis, 'navigator', navigator));
  t.mock.method(InferenceSession, 'create', async () => ({}));
  location.searchParams.set('ep', 'wasm');
  await createOrtSession('small.onnx');
  location.searchParams.delete('ep');
  assert.deepEqual(await resolveOrtSessionPlacement(), { providers: ['wasm'] });

  const original = BIG_NETS.t3.modelUrl;
  const splitK = lc0SplitKUrl(original);
  const models = [original, splitK].map((url, i) => ({ file: url.split('/').at(-1), url, bytes: (160 + 10 * i) * 1_000_000, sha256: 'a'.repeat(64) }));
  t.mock.method(globalThis, 'fetch', async (url) => {
    assert.match(String(url), /manifest.json$/);
    return Response.json({ models });
  });
  phoneLargeNetPreference.setEnabled(true);
  const pending = f.request(BIG_NETS.t3, { inspect: inspectLargeNetDownload });
  const dialog = await f.dialog();
  choose(dialog, false);
  assert.equal(await pending, false);
  assert.match(dialog.children[1].textContent, /Model downloads with fallback are estimated at 330 MB/);
});

test('the short preference hint is limited to engine pickers that offer gated nets', () => {
  const component = readFileSync(new URL('../src/lib/components/PhoneLargeNetPreference.svelte', import.meta.url), 'utf8');
  assert.match(component, /constrained && showHint/);
  assert.match(component, /BT4 and t3 are off on phones \(crash risk\)\./);
  assert.match(component, /BT4 and t3 on for this phone ·/);
  assert.match(component, />Turn off<\/button>/);
  const lab = readFileSync(new URL('../src/routes/lab/mobile-soak/+page.svelte', import.meta.url), 'utf8');
  assert.match(lab, /PhoneLargeNetPreference showHint=\{false\}/);
});
