import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { stripTypeScriptTypes } from 'node:module';
import { test } from 'node:test';
import { arenaBigNetPairingBlocked } from '../src/lc0/arenaMemoryGuard.ts';
import {
  acquireBigNetSearcher,
  acquireBigNetSearcherAsync,
  BigNetSessionPool,
  disposeAllBigNetSearchersNow,
  hasBigNetSearcher,
} from '../src/lc0/bigNetSessionPool.ts';
import { BIG_NETS, bigNetOptionState, phoneBlockedBigNetModelUrl } from '../src/lc0/bt4Engine.ts';
import { requestLargeNetConsent as requestLargeNetConsentWithCache } from '../src/lc0/largeNetConsent.ts';
import { lc0SplitKUrl } from '../src/lc0/lc0SmallModelSelection.ts';
import {
  createPhoneLargeNetPreference,
  PHONE_LARGE_NET_STORAGE_KEY,
  phoneBigNetBlocked,
  phoneLargeNetPreference,
  phoneLargeNetsBlocked,
  phoneSafeEngineVariant,
} from '../src/lc0/phoneLargeNetPolicy.ts';
import { committedPlayOpponent, transitionPlayOpponent } from '../src/lc0/playOpponentState.ts';

for (const surface of ['analysis', 'play', 'arena', 'review']) {
  for (const constrained of [false, true]) {
    for (const optedIn of [false, true]) {
      test(`${surface}: constrained=${constrained}, opted-in=${optedIn}`, () => {
        const blocked = constrained && !optedIn;
        assert.equal(phoneLargeNetsBlocked(constrained, optedIn), blocked);
        for (const key of Object.keys(BIG_NETS)) {
          assert.equal(phoneSafeEngineVariant('lc0', key, constrained, optedIn), blocked && key !== 'lqo' ? 'small' : key);
          assert.equal(phoneSafeEngineVariant('sf', key, constrained, optedIn), key);
        }
        assert.equal(phoneSafeEngineVariant('lc0', 'small', constrained, optedIn), 'small');
      });
    }
  }
}

test('consent persists, can be revoked, and notifies controls', () => {
  const values = new Map();
  const storage = { getItem: (key) => values.get(key), setItem: (key, value) => values.set(key, value) };
  const preference = createPhoneLargeNetPreference(() => storage);
  let changes = 0;
  const unsubscribe = preference.subscribe(() => changes++);
  assert.equal(preference.enabled(), false);
  preference.setEnabled(true);
  assert.equal(values.get(PHONE_LARGE_NET_STORAGE_KEY), 'on');
  assert.equal(createPhoneLargeNetPreference(() => storage).enabled(), true);
  preference.setEnabled(false);
  assert.equal(createPhoneLargeNetPreference(() => storage).enabled(), false);
  assert.equal(changes, 2);
  unsubscribe();
  preference.setEnabled(true);
  assert.equal(changes, 2);
});

test('cross-tab storage revocation supersedes a session opt-in and notifies active controls', () => {
  const values = new Map();
  const storage = { getItem: (key) => values.get(key), setItem: (key, value) => values.set(key, value) };
  const firstEvents = new EventTarget();
  const first = createPhoneLargeNetPreference(() => storage, firstEvents);
  const second = createPhoneLargeNetPreference(() => storage);
  let notifications = 0;
  first.subscribe(() => notifications++);
  first.setEnabled(true);
  second.setEnabled(false);
  assert.equal(first.enabled(), true, 'local override stays authoritative until the other-tab event');
  firstEvents.dispatchEvent(Object.assign(new Event('storage'), { key: PHONE_LARGE_NET_STORAGE_KEY, storageArea: storage }));
  assert.equal(first.enabled(), false);
  assert.equal(phoneBigNetBlocked('bt4', true, first.enabled()), true);
  assert.equal(notifications, 2);
  first.setEnabled(true);
  second.setEnabled(false);
  firstEvents.dispatchEvent(Object.assign(new Event('storage'), { key: PHONE_LARGE_NET_STORAGE_KEY, storageArea: {} }));
  assert.equal(first.enabled(), true, 'sessionStorage and unrelated storage areas cannot overwrite session consent');
  values.clear();
  firstEvents.dispatchEvent(Object.assign(new Event('storage'), { key: null, storageArea: storage }));
  assert.equal(first.enabled(), false, 'clearing localStorage in another tab revokes consent too');
});

for (const failure of ['access', 'read', 'write', 'missing']) {
  test(`storage ${failure} failure keeps opt-in and revocation for this page session`, () => {
    const preference = createPhoneLargeNetPreference(() => {
      if (failure === 'access') throw new Error('SecurityError');
      if (failure === 'missing') return undefined;
      return {
        getItem() {
          if (failure === 'read') throw new Error('SecurityError');
          return 'off';
        },
        setItem() {
          throw new Error('QuotaExceededError');
        },
      };
    });
    assert.equal(preference.enabled(), false);
    preference.setEnabled(true);
    assert.equal(preference.enabled(), true);
    assert.equal(phoneLargeNetsBlocked(true, preference.enabled()), false);
    preference.setEnabled(false);
    assert.equal(preference.enabled(), false);
  });
}

test('failed revocation writes cannot re-enable a stored opt-in during this session', () => {
  const preference = createPhoneLargeNetPreference(() => ({
    getItem: () => 'on',
    setItem() {
      throw new Error('blocked');
    },
  }));
  assert.equal(preference.enabled(), true);
  preference.setEnabled(false);
  assert.equal(preference.enabled(), false);
});

test('phone BT4/t3 options allow consent but deny acquisition; Queen Odds is exempt', async () => {
  const original = Object.getOwnPropertyDescriptor(globalThis, 'location');
  Object.defineProperty(globalThis, 'location', { configurable: true, value: new URL('https://example.test/?memoryClass=phone') });
  phoneLargeNetPreference.setEnabled(false);
  try {
    for (const [key, config] of Object.entries(BIG_NETS)) {
      if (key === 'lqo') {
        assert.doesNotMatch(bigNetOptionState(config).suffix, /off on phones/);
        assert.equal(acquireBigNetSearcher(key).config, config);
        assert.equal((await acquireBigNetSearcherAsync(key)).config, config);
      } else {
        assert.deepEqual(bigNetOptionState(config), { disabled: false, suffix: ' (Large net — off on phones)' });
        assert.throws(() => acquireBigNetSearcher(key), /BT4 and t3 are off on phones/);
        await assert.rejects(acquireBigNetSearcherAsync(key), /BT4 and t3 are off on phones/);
        assert.equal(hasBigNetSearcher(key), false);
      }
      assert.equal(phoneBlockedBigNetModelUrl(config.modelUrl), true);
      const splitK = lc0SplitKUrl(config.modelUrl);
      if (splitK) {
        assert.equal(phoneBlockedBigNetModelUrl(splitK), true, 'shipped split-K variants cannot bypass the picker');
        assert.equal(phoneBlockedBigNetModelUrl(`${splitK}?cache=1`), true);
      }
    }
    disposeAllBigNetSearchersNow();
    phoneLargeNetPreference.setEnabled(true);
    assert.equal(phoneLargeNetsBlocked(), false);
    assert.equal(phoneBlockedBigNetModelUrl(BIG_NETS.bt4.modelUrl), true, 'known large models must use the picker, not the Small override lane');
    assert.equal(
      arenaBigNetPairingBlocked(
        'match',
        [
          { family: 'lc0', variant: 'bt4' },
          { family: 'lc0', variant: 't3' },
        ],
        true,
      ),
      true,
      'device opt-in never overrides Arena pair policy',
    );
  } finally {
    phoneLargeNetPreference.setEnabled(false);
    if (original) Object.defineProperty(globalThis, 'location', original);
    else delete globalThis.location;
  }
});

test('modal cancellation restores focus; Tab stays trapped; opt-in happens only after Use anyway', async () => {
  const originalDocument = Object.getOwnPropertyDescriptor(globalThis, 'document');
  const originalLocation = Object.getOwnPropertyDescriptor(globalThis, 'location');
  class Element extends EventTarget {
    children = [];
    isConnected = true;
    attributes = {};
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
      this.open = false;
      this.dispatchEvent(new Event('close'));
    }
    remove() {
      this.isConnected = false;
    }
    focus() {
      document.activeElement = this;
    }
  }
  const previous = new Element();
  const body = new Element();
  Object.defineProperty(globalThis, 'document', {
    configurable: true,
    value: {
      body,
      activeElement: previous,
      createElement: () => new Element(),
      getElementById: () => null,
    },
  });
  Object.defineProperty(globalThis, 'location', { configurable: true, value: new URL('https://example.test/?memoryClass=phone') });
  phoneLargeNetPreference.setEnabled(false);
  let cached = false;
  const requestLargeNetConsent = (config, signal, desktopWarning = true) =>
    requestLargeNetConsentWithCache(config, signal, desktopWarning, undefined, async () => ({
      url: config.modelUrl,
      cached,
      downloadBytes: config.approxMb * 1_000_000,
    }));
  try {
    let pending = requestLargeNetConsent(BIG_NETS.bt4);
    await new Promise(setImmediate);
    let dialog = body.children.at(-1);
    const [cancel, use] = dialog.children.at(-1).children;
    assert.equal(dialog.attributes['aria-modal'], 'true');
    assert.equal(dialog.children[0].textContent, 'Use BT4 on this phone?');
    assert.equal(dialog.children[1].textContent, 'Large nets can crash the tab on phones. BT4 is a 178 MB download, cached after the first time.');
    assert.equal(dialog.children.length, 3, 'no extra small print');
    assert.equal(document.activeElement, cancel);
    const tab = () => {
      const event = new Event('keydown', { cancelable: true });
      Object.defineProperty(event, 'key', { value: 'Tab' });
      dialog.dispatchEvent(event);
    };
    tab();
    assert.equal(document.activeElement, use);
    tab();
    assert.equal(document.activeElement, cancel);
    cancel.dispatchEvent(new Event('click'));
    assert.equal(await pending, false);
    assert.equal(document.activeElement, previous);
    assert.equal(phoneLargeNetPreference.enabled(), false);
    pending = requestLargeNetConsent(BIG_NETS.t3);
    await new Promise(setImmediate);
    dialog = body.children.at(-1);
    assert.match(dialog.children[1].textContent, /163 MB/);
    dialog.children.at(-1).children[1].dispatchEvent(new Event('click'));
    assert.equal(await pending, true);
    assert.equal(phoneLargeNetPreference.enabled(), true);
    assert.equal(document.activeElement, previous);
    // Once opted in, a cached net needs no second dialog on another surface.
    cached = true;
    assert.equal(await requestLargeNetConsent(BIG_NETS.bt4), true);
    cached = false;
    phoneLargeNetPreference.setEnabled(false);
    const abort = new AbortController();
    pending = requestLargeNetConsent(BIG_NETS.t3, abort.signal);
    abort.abort();
    assert.equal(await pending, false);
    assert.equal(phoneLargeNetPreference.enabled(), false);
    pending = requestLargeNetConsent(BIG_NETS.lqo);
    await new Promise(setImmediate);
    dialog = body.children.at(-1);
    assert.equal(dialog.children[0].textContent, 'Download Queen Odds?');
    assert.equal(dialog.children[1].textContent, "It's a 96 MB download, cached after the first time.");
    assert.equal(dialog.children.at(-1).children[1].textContent, 'Download');
    dialog.children.at(-1).children[1].dispatchEvent(new Event('click'));
    assert.equal(await pending, true);
    assert.equal(phoneLargeNetPreference.enabled(), false, 'Queen Odds download approval never grants BT4/t3 consent');
  } finally {
    phoneLargeNetPreference.setEnabled(false);
    if (originalDocument) Object.defineProperty(globalThis, 'document', originalDocument);
    else delete globalThis.document;
    if (originalLocation) Object.defineProperty(globalThis, 'location', originalLocation);
    else delete globalThis.location;
  }
});

const analysis = readFileSync(new URL('../src/lc0/analysisBrowser.ts', import.meta.url), 'utf8');
const start = analysis.indexOf('async function profileRowsForUse(');
const profileFunction = stripTypeScriptTypes(analysis.slice(start, analysis.indexOf('function loadEngineProfiles(', start)));

test('saved and explicitly selected profiles fall back without prompting or acquiring on non-opted-in phones', async () => {
  for (const allowPrompt of [false, true]) {
    const deps = {
      normalizeDeployEngineRow: (row) => row,
      clampStrengthForRow: (row) => row.strength,
      isLc0BigNetVariant: (variant) => Object.hasOwn(BIG_NETS, variant),
      BIG_NETS,
      phoneBigNetBlocked: (key) => phoneBigNetBlocked(key, true, false),
      bigNetSelectable: () => {
        assert.fail('blocked restore must not check assets');
      },
      requestLargeNetConsent: () => {
        assert.fail('restore must not implicitly ask for consent');
      },
    };
    const restore = new Function('deps', `const { ${Object.keys(deps)} } = deps; ${profileFunction}; return profileRowsForUse;`)(deps);
    const keys = ['bt4', 't3'];
    const rows = keys.map((variant) => ({ family: 'lc0', variant, strength: 400 }));
    const restored = await restore(rows, allowPrompt);
    assert.deepEqual(
      restored.map((row) => row.variant),
      ['small', 'small'],
    );
    assert.deepEqual(
      rows.map((row) => row.variant),
      keys,
      'saved setup is not rewritten',
    );
  }
});

test('restored Play game keeps its position but replaces a blocked large opponent before preparation', () => {
  const source = readFileSync(new URL('../src/lc0/playBrowser.ts', import.meta.url), 'utf8');
  const start = source.indexOf('function ctxInit(ctx:');
  const prefix = stripTypeScriptTypes(`${source.slice(start, source.indexOf('  ctx.autosave =', start))}\n}`);
  for (const key of Object.keys(BIG_NETS)) {
    let warned = false;
    const nodes = new Map();
    const el = (id) => {
      if (!nodes.has(id)) nodes.set(id, {});
      return nodes.get(id);
    };
    const saved = { preferences: { engineId: key, level: 1, color: 'white' }, practicePosition: null, moves: ['saved-move'], board: 'saved-position' };
    const deps = {
      blockedModelOverride: false,
      ctxApplyPlayHandoffForMount: () => false,
      ENGINE_OPTIONS: [
        { id: key, family: 'lc0', variant: key, group: 'engine' },
        { id: 'lc0-small', family: 'lc0', variant: 'small', group: 'engine' },
      ],
      readWorkspace: () => saved,
      BIG_NETS,
      bigNetAssetStatusSync: () => 'present',
      isV0DeployProfile: () => false,
      ctxRefreshEngineOptions() {},
      phoneSafeEngineVariant: (family, variant) => phoneSafeEngineVariant(family, variant, true, false),
      showPhoneLargeNetFallback: () => {
        warned = true;
      },
      ctxEngineOptionState: () => ({ disabled: false }),
      ctxRenderLevelOptions() {},
      ctxResetGameStateForMount: (ctx) => {
        ctx.activeEngine = el('engineSelect').value;
      },
    };
    const init = new Function('deps', `const { ${Object.keys(deps)} } = deps; ${prefix}; return ctxInit;`)(deps);
    const ctx = { dom: { el, selectEl: el } };
    init(ctx);
    assert.equal(ctx.activeEngine, key === 'lqo' ? 'lqo' : 'lc0-small');
    assert.equal(ctx.pendingPreferredEngineId, null);
    assert.equal(ctx.board, saved.board);
    assert.deepEqual(ctx.moves, saved.moves);
    assert.equal(warned, key !== 'lqo');
    assert.equal(saved.preferences.engineId, key, 'stored game is not mutated by decoding');
  }
});

test('Play revocation distinguishes live opponents, pending restart drafts, and unresolved saved preferences', () => {
  const source = readFileSync(new URL('../src/lc0/playBrowser.ts', import.meta.url), 'utf8');
  const start = source.indexOf('function ctxRevokePhoneLargeNets(');
  const fn = stripTypeScriptTypes(source.slice(start, source.indexOf('function ctxInit(', start)));
  const options = {
    maia3: { family: 'maia3', variant: 'maia3' },
    'sf-lite': { family: 'sf', variant: 'lite' },
    ...Object.fromEntries(Object.keys(BIG_NETS).map((key) => [key, { family: 'lc0', variant: key }])),
  };
  const scenarios = [...Object.entries(options).map(([engineId, option]) => [engineId, option, false]), ['maia3', options.maia3, true]];
  for (const [engineId, option, cancellingRestore] of scenarios) {
    const activeLarge = phoneBigNetBlocked(option.variant, true, false);
    const active = { engineId, color: 'white', level: 3, maiaElo: 1700 };
    const opponent = cancellingRestore
      ? committedPlayOpponent(active)
      : transitionPlayOpponent(committedPlayOpponent(active), {
          type: 'edit',
          preferences: { ...active, engineId: activeLarge ? 'sf-lite' : 'bt4', color: 'black' },
          gameInProgress: true,
        });
    const clearedTimers = [];
    const ctx = {
      opponent,
      pendingPreferredEngineId: cancellingRestore ? 't3' : null,
      pendingPreferredEngineTimer: cancellingRestore ? 42 : null,
      board: 'saved-game',
      scope: { clearTimeout: (timer) => clearedTimers.push(timer) },
    };
    const replacesOpponent = activeLarge || cancellingRestore;
    let note = cancellingRestore ? 'Checking your saved engine assets…' : '';
    let renders = 0;
    let cancelled = 0;
    let prepared = 0;
    let dismissed = 0;
    const pool = new BigNetSessionPool(() => ({
      disposeCalls: 0,
      dispose() {
        this.disposeCalls++;
      },
    }));
    const bt4 = pool.acquire('bt4');
    const t3 = pool.acquire('t3');
    const queen = pool.acquire('lqo');
    const deps = {
      phoneLargeNetsBlocked: () => true,
      ENGINE_OPTIONS: Object.entries(options).map(([id, option]) => ({ id, ...option })),
      phoneSafeEngineVariant: (family, variant) => phoneSafeEngineVariant(family, variant, true, false),
      ctxSelectedEngine: (ctx) => options[ctx.opponent.active.engineId],
      ctxDraftEngine: () => options[(ctx.opponent.pending ?? ctx.opponent.active).engineId],
      ctxCancelEngineTurn: () => {
        cancelled++;
      },
      ctxSetEngineNote: (_ctx, text) => {
        note = text;
      },
      releaseUnusedBigNetSearchers: (keys, ttl) => pool.releaseUnused(keys, ttl),
      ctxSetOpponent: (ctx, next) => {
        ctx.opponent = next;
      },
      committedPlayOpponent,
      ctxRestoreOpponentControls() {},
      showPhoneLargeNetFallback() {},
      ctxPersistPlayPreferences() {},
      ctxRender() {
        renders++;
      },
      ctxRefreshEngineOptions() {},
      ctxPrepareOrStartEngine: () => {
        prepared++;
      },
      ctxDismissRestart: (ctx) => {
        dismissed++;
        ctx.opponent = transitionPlayOpponent(ctx.opponent, { type: 'dismiss' });
      },
    };
    const revoke = new Function('deps', `const { ${Object.keys(deps)} } = deps; ${fn}; return ctxRevokePhoneLargeNets;`)(deps);
    revoke(ctx);
    assert.equal(ctx.opponent.active.engineId, replacesOpponent ? 'lc0-small' : engineId);
    assert.equal(ctx.opponent.active.color, active.color);
    assert.equal(ctx.opponent.active.level, active.level);
    assert.equal(ctx.opponent.pending, null);
    assert.equal(cancelled, replacesOpponent ? 1 : 0, 'unaffected live work is not cancelled');
    assert.equal(prepared, replacesOpponent ? 1 : 0, 'a cancelled saved restore must prepare the small fallback');
    assert.equal(renders, replacesOpponent ? 1 : 0);
    assert.equal(note, '', 'saved-engine checking text must not survive revocation');
    assert.equal(dismissed, replacesOpponent ? 0 : 1);
    assert.equal(ctx.board, 'saved-game');
    assert.equal(ctx.pendingPreferredEngineId, null);
    assert.equal(ctx.pendingPreferredEngineTimer, null);
    assert.deepEqual(clearedTimers, cancellingRestore ? [42] : []);
    assert.equal(bt4.disposeCalls, 1, 'even an unaffected opponent retires previously retained BT4');
    assert.equal(t3.disposeCalls, 1, 'even an unaffected opponent retires previously retained t3');
    assert.equal(queen.disposeCalls, 0, 'Queen Odds work is never cancelled by gated-net revocation');
    pool.disposeAllNow();
  }
});

test('every picker uses consent and shared restore policy before engine acquisition', () => {
  for (const surface of ['analysis', 'arena', 'play']) {
    const source = readFileSync(new URL(`../src/lc0/${surface}Browser.ts`, import.meta.url), 'utf8');
    assert.match(source, /await requestLargeNet(Download)?Consent\(/);
    assert.match(source, /phoneSafeEngineVariant\(/);
    assert.match(source, /PHONE_LARGE_NET_CHANGE_EVENT/);
    assert.doesNotMatch(source, /window\.confirm\(`\$\{bigNetLoadWarning/);
  }
  const critical = readFileSync(new URL('../src/lc0/criticalMomentsEngines.ts', import.meta.url), 'utf8');
  assert.match(critical, /variant: 'small'/);
  assert.doesNotMatch(critical, /acquireBigNetSearcher|BIG_NETS/);
});
