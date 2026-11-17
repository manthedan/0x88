import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { stripTypeScriptTypes } from 'node:module';
import { test } from 'node:test';
import { requestLargeNetConsent } from '../src/lc0/largeNetConsent.ts';
import { BIG_NETS, bigNetOptionState, phoneBlockedBigNetModelUrl } from '../src/lc0/bt4Engine.ts';
import { acquireBigNetSearcher, hasBigNetSearcher } from '../src/lc0/bigNetSessionPool.ts';
import { arenaBigNetPairingBlocked } from '../src/lc0/arenaMemoryGuard.ts';
import {
  PHONE_LARGE_NET_STORAGE_KEY,
  createPhoneLargeNetPreference,
  phoneLargeNetPreference,
  phoneLargeNetsBlocked,
  phoneSafeEngineVariant,
} from '../src/lc0/phoneLargeNetPolicy.ts';

for (const surface of ['analysis', 'play', 'arena', 'review']) {
  for (const constrained of [false, true]) {
    for (const optedIn of [false, true]) {
      test(`${surface}: constrained=${constrained}, opted-in=${optedIn}`, () => {
        const blocked = constrained && !optedIn;
        assert.equal(phoneLargeNetsBlocked(constrained, optedIn), blocked);
        for (const key of Object.keys(BIG_NETS)) {
          assert.equal(phoneSafeEngineVariant('lc0', key, constrained, optedIn), blocked ? 'small' : key);
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

test('phone options remain visible/actionable for consent, but pool acquisition is denied', () => {
  const original = Object.getOwnPropertyDescriptor(globalThis, 'location');
  Object.defineProperty(globalThis, 'location', { configurable: true, value: new URL('https://example.test/?memoryClass=phone') });
  phoneLargeNetPreference.setEnabled(false);
  try {
    for (const [key, config] of Object.entries(BIG_NETS)) {
      assert.deepEqual(bigNetOptionState(config), { disabled: false, suffix: ' (Large net — off on phones)' });
      assert.equal(phoneBlockedBigNetModelUrl(config.modelUrl), true);
      assert.throws(() => acquireBigNetSearcher(key), /Large nets are off on phones/);
      assert.equal(hasBigNetSearcher(key), false);
    }
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
  try {
    let pending = requestLargeNetConsent(BIG_NETS.bt4);
    let dialog = body.children.at(-1);
    const [cancel, use] = dialog.children[2].children;
    assert.equal(dialog.attributes['aria-modal'], 'true');
    assert.match(dialog.children[1].textContent, /178 MB/);
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
    dialog = body.children.at(-1);
    dialog.children[2].children[1].dispatchEvent(new Event('click'));
    assert.equal(await pending, true);
    assert.equal(phoneLargeNetPreference.enabled(), true);
    assert.equal(document.activeElement, previous);
    // Once opted in, no second dialog is necessary on another surface/net.
    assert.equal(await requestLargeNetConsent(BIG_NETS.lqo), true);
    phoneLargeNetPreference.setEnabled(false);
    const abort = new AbortController();
    pending = requestLargeNetConsent(BIG_NETS.lqo, abort.signal);
    abort.abort();
    assert.equal(await pending, false);
    assert.equal(phoneLargeNetPreference.enabled(), false);
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
      phoneLargeNetsBlocked: () => true,
      bigNetSelectableSync: () => {
        assert.fail('blocked restore must not check assets');
      },
      requestLargeNetConsent: () => {
        assert.fail('restore must not implicitly ask for consent');
      },
    };
    const restore = new Function('deps', `const { ${Object.keys(deps)} } = deps; ${profileFunction}; return profileRowsForUse;`)(deps);
    const rows = Object.keys(BIG_NETS).map((variant) => ({ family: 'lc0', variant, strength: 400 }));
    const restored = await restore(rows, allowPrompt);
    assert.deepEqual(
      restored.map((row) => row.variant),
      ['small', 'small', 'small'],
    );
    assert.deepEqual(
      rows.map((row) => row.variant),
      Object.keys(BIG_NETS),
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
      ctxRefreshEngineOptions() {},
      phoneSafeEngineVariant: (family, variant) => phoneSafeEngineVariant(family, variant, true, false),
      showPhoneLargeNetFallback: () => {
        warned = true;
      },
      ctxEngineOptionState: () => ({ disabled: false }),
      selectEl: el,
      el,
      ctxRenderLevelOptions() {},
      ctxResetGameStateForMount: (ctx) => {
        ctx.activeEngine = el('engineSelect').value;
      },
    };
    const init = new Function('deps', `const { ${Object.keys(deps)} } = deps; ${prefix}; return ctxInit;`)(deps);
    const ctx = {};
    init(ctx);
    assert.equal(ctx.activeEngine, 'lc0-small');
    assert.equal(ctx.pendingPreferredEngineId, null);
    assert.equal(ctx.board, saved.board);
    assert.deepEqual(ctx.moves, saved.moves);
    assert.equal(warned, true);
    assert.equal(saved.preferences.engineId, key, 'stored game is not mutated by decoding');
  }
});

test('every picker uses consent and shared restore policy before engine acquisition', () => {
  for (const surface of ['analysis', 'arena', 'play']) {
    const source = readFileSync(new URL(`../src/lc0/${surface}Browser.ts`, import.meta.url), 'utf8');
    assert.match(source, /await requestLargeNetConsent\(/);
    assert.match(source, /phoneSafeEngineVariant\(/);
    assert.match(source, /PHONE_LARGE_NET_CHANGE_EVENT/);
    assert.doesNotMatch(source, /window\.confirm\(`\$\{bigNetLoadWarning/);
  }
  const critical = readFileSync(new URL('../src/lc0/criticalMomentsEngines.ts', import.meta.url), 'utf8');
  assert.match(critical, /variant: 'small'/);
  assert.doesNotMatch(critical, /acquireBigNetSearcher|BIG_NETS/);
});
