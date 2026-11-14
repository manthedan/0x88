import assert from 'node:assert/strict';
import test from 'node:test';
import { withEngineTimeout } from '../src/lc0/engineTimeout.ts';

for (const reply of ['partial success', 'engine error']) {
  test(`parent cancellation owns the outcome over synchronous ${reply}`, async () => {
    const parent = new AbortController();
    const pending = withEngineTimeout(
      'CPU',
      parent.signal,
      1000,
      (signal) =>
        new Promise((resolve, reject) => {
          signal.addEventListener(
            'abort',
            () => {
              if (reply === 'partial success') resolve(['partial PV']);
              else reject(new Error('worker shutdown failed'));
            },
            { once: true },
          );
        }),
    );
    const checked = assert.rejects(pending, { name: 'AbortError' });
    parent.abort();
    await checked;
  });
}

test('an uncancelled task retains its result', async () => {
  assert.deepEqual(await withEngineTimeout('CPU', new AbortController().signal, 1000, async () => ['complete PV']), ['complete PV']);
});
