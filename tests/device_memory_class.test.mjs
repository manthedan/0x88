import assert from 'node:assert/strict';
import { test } from 'node:test';
import { isMemoryConstrainedDevice, phoneReclaimSettings } from '../src/lc0/deviceMemoryClass.ts';

const cases = [
  [
    'iPhone Safari without deviceMemory',
    { userAgent: 'Mozilla/5.0 (iPhone; CPU iPhone OS 26_6 like Mac OS X) AppleWebKit/605.1.15 Version/26.6 Mobile/15E148 Safari/604.1' },
    true,
  ],
  ['iPad', { userAgent: 'Mozilla/5.0 (iPad; CPU OS 26_6 like Mac OS X)' }, true],
  ['iPod', { userAgent: 'Mozilla/5.0 (iPod touch; CPU iPhone OS 15_0 like Mac OS X)' }, true],
  ['iPad as Mac', { userAgent: 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15) AppleWebKit/605.1.15 Safari/605.1.15', maxTouchPoints: 5 }, true],
  ['Android phone 4 GB', { userAgent: 'Mozilla/5.0 (Linux; Android 15; Pixel) Mobile', deviceMemory: 4 }, true],
  ['Android tablet 2 GB', { userAgent: 'Mozilla/5.0 (Linux; Android 14; Tablet)', deviceMemory: 2 }, true],
  ['Android unreported RAM', { userAgent: 'Mozilla/5.0 (Linux; Android 14; Tablet)' }, true],
  ['Android 8 GB', { userAgent: 'Mozilla/5.0 (Linux; Android 15; Pixel) Mobile', deviceMemory: 8 }, false],
  ['Mac desktop', { userAgent: 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15)', maxTouchPoints: 0 }, false],
  ['Mac one touch point', { userAgent: 'Macintosh', maxTouchPoints: 1, deviceMemory: 2 }, false],
  ['Windows touchscreen desktop with low RAM', { userAgent: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)', maxTouchPoints: 10, deviceMemory: 2 }, false],
  ['Linux desktop with low RAM', { userAgent: 'Mozilla/5.0 (X11; Linux x86_64)', deviceMemory: 4 }, false],
  ['no navigator / SSR', undefined, false],
];
for (const [name, device, expected] of cases) {
  test(`memory class: ${name}`, () => assert.equal(isMemoryConstrainedDevice(device, ''), expected));
}

test('memoryClass overrides are explicit, shared lab controls; unknown values use detection', () => {
  const desktop = { userAgent: 'Windows NT 10.0' };
  const phone = { userAgent: 'iPhone' };
  assert.equal(isMemoryConstrainedDevice(desktop, '?memoryClass=phone'), true);
  assert.equal(isMemoryConstrainedDevice(phone, '?memoryClass=desktop'), false);
  assert.equal(isMemoryConstrainedDevice(desktop, '?memoryClass=unknown'), false);
  assert.equal(isMemoryConstrainedDevice(phone, '?memoryClass=unknown'), true);
  assert.equal(isMemoryConstrainedDevice(desktop, '?memoryClass=PHONE'), false);
});

test('phone reclaim diagnostic overrides survive client navigation and ignore invalid values', (t) => {
  const store = new Map();
  const originalStorage = Object.getOwnPropertyDescriptor(globalThis, 'localStorage');
  Object.defineProperty(globalThis, 'localStorage', {
    configurable: true,
    value: { getItem: (key) => store.get(key) ?? null, setItem: (key, value) => store.set(key, value) },
  });
  t.after(() => {
    if (originalStorage) Object.defineProperty(globalThis, 'localStorage', originalStorage);
    else delete globalThis.localStorage;
    delete globalThis.location;
  });
  assert.deepEqual(phoneReclaimSettings(), { reclaimGapMs: 0, retireCpu: true });
  globalThis.location = new URL('https://test.invalid/play?0x88-reclaim-gap-ms=5000&0x88-retire-cpu=0');
  assert.deepEqual(phoneReclaimSettings(), { reclaimGapMs: 5000, retireCpu: false });
  globalThis.location = new URL('https://test.invalid/analysis');
  assert.deepEqual(phoneReclaimSettings(), { reclaimGapMs: 5000, retireCpu: false });
  globalThis.location = new URL('https://test.invalid/analysis?0x88-reclaim-gap-ms=-1&0x88-retire-cpu=no');
  assert.deepEqual(phoneReclaimSettings(), { reclaimGapMs: 5000, retireCpu: false });
  globalThis.location = new URL('https://test.invalid/play?0x88-reclaim-gap-ms=999999999');
  assert.equal(phoneReclaimSettings().reclaimGapMs, 5000, 'an excessive URL override does not replace the stored value');
  store.set('0x88-reclaim-gap-ms', '999999999');
  globalThis.location = new URL('https://test.invalid/analysis');
  assert.equal(phoneReclaimSettings().reclaimGapMs, 0, 'excessive stored values are ignored too');
});
