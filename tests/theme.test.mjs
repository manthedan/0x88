import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';

let instance = 0;
async function fixture(saved) {
  const originals = Object.fromEntries(['localStorage', 'matchMedia', 'document'].map((key) => [key, Object.getOwnPropertyDescriptor(globalThis, key)]));
  const values = new Map(saved ? [['0x88-theme', saved]] : []);
  const media = new EventTarget();
  media.matches = false;
  const root = { dataset: {} };
  Object.defineProperty(globalThis, 'localStorage', {
    configurable: true,
    value: { getItem: (key) => values.get(key) ?? null, setItem: (key, value) => values.set(key, value) },
  });
  Object.defineProperty(globalThis, 'matchMedia', { configurable: true, value: () => media });
  Object.defineProperty(globalThis, 'document', { configurable: true, value: { documentElement: root } });
  const module = await import(`../src/lib/stores/theme.ts?fixture=${instance++}`);
  const cleanup = module.syncThemeToDom();
  return {
    module,
    values,
    root,
    system(dark) {
      media.matches = dark;
      media.dispatchEvent(new Event('change'));
    },
    cleanup() {
      cleanup();
      for (const [key, descriptor] of Object.entries(originals)) {
        if (descriptor) Object.defineProperty(globalThis, key, descriptor);
        else delete globalThis[key];
      }
    },
    unsubscribe: cleanup,
  };
}

test('first visit and revisits follow the OS until an explicit toggle', async () => {
  const f = await fixture();
  try {
    assert.equal(f.root.dataset.theme, 'light');
    assert.equal(f.values.has('0x88-theme'), false);
    f.system(true);
    assert.equal(f.root.dataset.theme, 'dark');
    assert.equal(f.values.has('0x88-theme'), false);
    f.module.toggleTheme();
    assert.equal(f.values.get('0x88-theme'), 'light');
    f.system(false);
    f.system(true);
    assert.equal(f.root.dataset.theme, 'light');
    f.unsubscribe();
    f.values.delete('0x88-theme');
    f.root.dataset.theme = 'unmounted';
    f.system(false);
    assert.equal(f.root.dataset.theme, 'unmounted');
  } finally {
    f.cleanup();
  }
});

test('stored choices, including choices pinned by the old bug, remain respected', async () => {
  for (const saved of ['dark', 'light']) {
    const f = await fixture(saved);
    try {
      f.system(true);
      f.system(false);
      assert.equal(f.root.dataset.theme, saved);
      assert.equal(f.values.get('0x88-theme'), saved);
    } finally {
      f.cleanup();
    }
  }
  const shell = readFileSync(new URL('../src/app.html', import.meta.url), 'utf8');
  assert.match(shell, /localStorage.getItem\('0x88-theme'\)/);
});
