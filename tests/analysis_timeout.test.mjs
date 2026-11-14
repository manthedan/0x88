import assert from 'node:assert/strict';
import test from 'node:test';
import { withEngineTimeout } from '../src/lc0/engineTimeout.ts';

// Analysis treats AbortError as cancellation of the whole run, but a timeout
// must be an engine failure so other engines' results can still be published.
for (const settlesOnAbort of ['reject', 'resolve']) {
  test(`deadline remains a timeout when engine abort synchronously ${settlesOnAbort}s`, async (t) => {
    t.mock.timers.enable({ apis: ['setTimeout'] });
    let workSignal;
    const result = withEngineTimeout('SF Lite', new AbortController().signal, 1000, (signal) => {
      workSignal = signal;
      return new Promise((resolve, reject) => {
        signal.addEventListener(
          'abort',
          () => {
            if (settlesOnAbort === 'reject') reject(new DOMException('Engine stopped', 'AbortError'));
            else resolve([]);
          },
          { once: true },
        );
      });
    });
    const checked = assert.rejects(result, (error) => error.name !== 'AbortError' && /SF Lite timed out after 1s/.test(error.message));
    t.mock.timers.tick(1000);
    await checked;
    assert.equal(workSignal.aborted, true);
  });
}

test('parent cancellation still reports AbortError', async () => {
  const parent = new AbortController();
  const checked = assert.rejects(
    withEngineTimeout(
      'SF Lite',
      parent.signal,
      1000,
      (signal) =>
        new Promise((_, reject) => {
          signal.addEventListener('abort', () => reject(new DOMException('Engine stopped', 'AbortError')), { once: true });
        }),
    ),
    { name: 'AbortError' },
  );
  parent.abort();
  await checked;
});
