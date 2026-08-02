import assert from 'node:assert/strict';
import { test } from 'node:test';

let workerModule;

async function loadWorkerModule(messages) {
  globalThis.self = {
    postMessage(message) {
      messages.push(message);
    },
    addEventListener() {},
  };
  workerModule ??= await import('../src/lc0/recklessWasiWorker.ts');
  return workerModule;
}

function streamedResponse(chunks, headers = {}) {
  return new Response(
    new ReadableStream({
      start(controller) {
        for (const chunk of chunks) controller.enqueue(Uint8Array.from(chunk));
        controller.close();
      },
    }),
    { headers },
  );
}

function resetPreopenState(worker, byteBudget) {
  worker.recklessWasiWorkerInternalsForTests.resetPreopenByteCache(byteBudget);
}

test('Reckless WASI preopen rejects invalid, oversized, and mismatching decoded length metadata before reading', async () => {
  const previousSelf = globalThis.self;
  const previousFetch = globalThis.fetch;
  const messages = [];
  let fetches = 0;
  try {
    const worker = await loadWorkerModule(messages);
    resetPreopenState(worker);
    const { fetchPreopenBytes, MAX_PREOPEN_FILE_BYTES } = worker;
    globalThis.fetch = async (url) => {
      fetches += 1;
      return new Response(new ReadableStream({ pull() {} }), {
        headers: {
          'x-artifact-content-length': url.includes('malformed')
            ? 'invalid'
            : url.includes('zero')
              ? '0'
              : url.includes('hard-max')
                ? String(MAX_PREOPEN_FILE_BYTES + 1)
                : '5',
        },
      });
    };
    await assert.rejects(fetchPreopenBytes('/reckless/invalid-expected.nnue', 0), /invalid expected byte length/);
    assert.equal(fetches, 0);
    await assert.rejects(fetchPreopenBytes('/reckless/malformed.nnue', 4), /invalid decoded byte length metadata/);
    await assert.rejects(fetchPreopenBytes('/reckless/zero.nnue', 4), /invalid decoded byte length metadata/);
    await assert.rejects(fetchPreopenBytes('/reckless/hard-max.nnue'), /exceeds the .* hard maximum/);
    await assert.rejects(fetchPreopenBytes('/reckless/mismatch.nnue', 4), /metadata mismatch.*got 5, expected 4/);
    assert.equal(fetches, 4);
    assert.deepEqual(messages, []);
  } finally {
    globalThis.fetch = previousFetch;
    if (previousSelf === undefined) delete globalThis.self;
    else globalThis.self = previousSelf;
  }
});

test('Reckless WASI preopen ignores encoded Content-Length and rejects no-header overflow at expected bytes', async () => {
  const previousSelf = globalThis.self;
  const previousFetch = globalThis.fetch;
  const messages = [];
  try {
    const worker = await loadWorkerModule(messages);
    resetPreopenState(worker);
    const { fetchPreopenBytes } = worker;
    globalThis.fetch = async () =>
      streamedResponse(
        [
          [1, 2, 3],
          [4, 5],
        ],
        {
          'content-encoding': 'br',
          'content-length': '2',
        },
      );
    await assert.rejects(fetchPreopenBytes('/reckless/br-overflow.nnue', 4), /exceeds its 4-byte download limit.*at least 5/);
    assert.ok(messages.every((message) => message.totalBytes === 4));
    assert.ok(messages.every((message) => message.totalBytes !== 2));
  } finally {
    globalThis.fetch = previousFetch;
    if (previousSelf === undefined) delete globalThis.self;
    else globalThis.self = previousSelf;
  }
});

test('Reckless WASI preopen rejects an undersized final decoded body', async () => {
  const previousSelf = globalThis.self;
  const previousFetch = globalThis.fetch;
  const messages = [];
  try {
    const worker = await loadWorkerModule(messages);
    resetPreopenState(worker);
    const { fetchPreopenBytes } = worker;
    globalThis.fetch = async () => streamedResponse([[1, 2, 3]], { 'content-encoding': 'br' });
    await assert.rejects(fetchPreopenBytes('/reckless/undersized.nnue', 4), /decoded byte length mismatch: got 3, expected 4/);
    assert.ok(messages.every((message) => message.totalBytes === 4));
  } finally {
    globalThis.fetch = previousFetch;
    if (previousSelf === undefined) delete globalThis.self;
    else globalThis.self = previousSelf;
  }
});

test('Reckless WASI preopen accepts the exact decoded body and reports the expected total', async () => {
  const previousSelf = globalThis.self;
  const previousFetch = globalThis.fetch;
  const messages = [];
  try {
    const worker = await loadWorkerModule(messages);
    resetPreopenState(worker);
    const { fetchPreopenBytes } = worker;
    globalThis.fetch = async () =>
      streamedResponse(
        [
          [1, 2],
          [3, 4],
        ],
        {
          'content-encoding': 'br',
          'content-length': '2',
        },
      );
    const buffer = await fetchPreopenBytes('/reckless/exact.nnue', 4);
    assert.equal(buffer.byteLength, 4);
    assert.deepEqual([...new Uint8Array(buffer)], [1, 2, 3, 4]);
    assert.deepEqual(messages.at(-1), {
      type: 'preopen-progress',
      url: '/reckless/exact.nnue',
      loadedBytes: 4,
      totalBytes: 4,
    });
  } finally {
    globalThis.fetch = previousFetch;
    if (previousSelf === undefined) delete globalThis.self;
    else globalThis.self = previousSelf;
  }
});

test('Reckless WASI preopen reuses verified bytes sequentially and deduplicates concurrent requests', async () => {
  const previousSelf = globalThis.self;
  const previousFetch = globalThis.fetch;
  const messages = [];
  let fetches = 0;
  try {
    const worker = await loadWorkerModule(messages);
    resetPreopenState(worker);
    const { fetchPreopenBytes, recklessWasiWorkerInternalsForTests } = worker;
    globalThis.fetch = async () => {
      fetches += 1;
      return streamedResponse([[1]], { 'content-length': '1' });
    };
    const [first, second] = await Promise.all([fetchPreopenBytes('/reckless/dedup.nnue', 1), fetchPreopenBytes('/reckless/dedup.nnue', 1)]);
    assert.equal(first, second);
    assert.equal(fetches, 1);

    const third = await fetchPreopenBytes('/reckless/dedup.nnue', 1);
    assert.equal(third, first);
    assert.equal(fetches, 1, 'sequential one-shot searches reuse verified worker-lifetime bytes');
    assert.equal(recklessWasiWorkerInternalsForTests.preopenByteCacheState().inFlightCount, 0);
  } finally {
    globalThis.fetch = previousFetch;
    if (previousSelf === undefined) delete globalThis.self;
    else globalThis.self = previousSelf;
  }
});

test('Reckless WASI preopen download preserves HTTP and stream errors and permits retry', async () => {
  const previousSelf = globalThis.self;
  const previousFetch = globalThis.fetch;
  const messages = [];
  let responseIndex = 0;
  try {
    const worker = await loadWorkerModule(messages);
    resetPreopenState(worker);
    const { fetchPreopenBytes, recklessWasiWorkerInternalsForTests } = worker;
    globalThis.fetch = async () => {
      responseIndex += 1;
      if (responseIndex === 1) return new Response('', { status: 503 });
      if (responseIndex === 2) {
        return new Response(
          new ReadableStream({
            start(controller) {
              controller.error(new Error('corrupt response stream'));
            },
          }),
        );
      }
      return streamedResponse([[5]], { 'content-length': '1' });
    };

    await assert.rejects(fetchPreopenBytes('/reckless/retry.nnue', 1), /HTTP 503/);
    assert.deepEqual(recklessWasiWorkerInternalsForTests.preopenByteCacheState(), {
      byteBudget: 64 * 1024 * 1024,
      totalBytes: 0,
      keys: [],
      inFlightCount: 0,
    });
    await assert.rejects(fetchPreopenBytes('/reckless/retry.nnue', 1), /corrupt response stream/);
    assert.equal(recklessWasiWorkerInternalsForTests.preopenByteCacheState().inFlightCount, 0);
    assert.deepEqual([...new Uint8Array(await fetchPreopenBytes('/reckless/retry.nnue', 1))], [5]);
    assert.equal(responseIndex, 3);
    assert.equal(recklessWasiWorkerInternalsForTests.preopenByteCacheState().inFlightCount, 0);
  } finally {
    globalThis.fetch = previousFetch;
    if (previousSelf === undefined) delete globalThis.self;
    else globalThis.self = previousSelf;
  }
});

test('Reckless WASI preopen cache identity includes expected decoded bytes', async () => {
  const previousSelf = globalThis.self;
  const previousFetch = globalThis.fetch;
  const messages = [];
  let fetches = 0;
  try {
    const worker = await loadWorkerModule(messages);
    resetPreopenState(worker);
    const { fetchPreopenBytes } = worker;
    globalThis.fetch = async () => {
      fetches += 1;
      return fetches === 1 ? streamedResponse([[1]], { 'content-length': '1' }) : streamedResponse([[1, 2]], { 'content-length': '2' });
    };

    assert.equal((await fetchPreopenBytes('/reckless/cache-key.nnue', 1)).byteLength, 1);
    assert.equal((await fetchPreopenBytes('/reckless/cache-key.nnue', 2)).byteLength, 2);
    assert.equal((await fetchPreopenBytes('/reckless/cache-key.nnue', 1)).byteLength, 1);
    assert.equal(fetches, 2);
  } finally {
    globalThis.fetch = previousFetch;
    if (previousSelf === undefined) delete globalThis.self;
    else globalThis.self = previousSelf;
  }
});

test('Reckless WASI preopen cache evicts least-recently-used bytes within its budget', async () => {
  const previousSelf = globalThis.self;
  const previousFetch = globalThis.fetch;
  const messages = [];
  let fetches = 0;
  try {
    const worker = await loadWorkerModule(messages);
    resetPreopenState(worker, 3);
    const { fetchPreopenBytes, PREOPEN_BYTE_CACHE_BUDGET, recklessWasiWorkerInternalsForTests } = worker;
    assert.ok(PREOPEN_BYTE_CACHE_BUDGET >= 63_266_880, 'the production budget must hold the full Reckless v60 NNUE');
    globalThis.fetch = async (url) => {
      fetches += 1;
      const length = url.endsWith('/a.nnue') ? 1 : 2;
      return streamedResponse([new Array(length).fill(fetches)], { 'content-length': String(length) });
    };

    await fetchPreopenBytes('/reckless/a.nnue', 1);
    await fetchPreopenBytes('/reckless/b.nnue', 2);
    await fetchPreopenBytes('/reckless/a.nnue', 1);
    await fetchPreopenBytes('/reckless/c.nnue', 2);
    assert.deepEqual(recklessWasiWorkerInternalsForTests.preopenByteCacheState(), {
      byteBudget: 3,
      totalBytes: 3,
      keys: ['/reckless/a.nnue\n1', '/reckless/c.nnue\n2'],
      inFlightCount: 0,
    });
    await fetchPreopenBytes('/reckless/b.nnue', 2);
    assert.equal(fetches, 4, 'the evicted least-recently-used entry must be fetched again');
    assert.ok(recklessWasiWorkerInternalsForTests.preopenByteCacheState().totalBytes <= 3);
  } finally {
    globalThis.fetch = previousFetch;
    if (previousSelf === undefined) delete globalThis.self;
    else globalThis.self = previousSelf;
  }
});

test('Reckless WASI preopen cache does not retain entries larger than its byte budget', async () => {
  const previousSelf = globalThis.self;
  const previousFetch = globalThis.fetch;
  const messages = [];
  let fetches = 0;
  try {
    const worker = await loadWorkerModule(messages);
    resetPreopenState(worker, 2);
    const { fetchPreopenBytes, recklessWasiWorkerInternalsForTests } = worker;
    globalThis.fetch = async () => {
      fetches += 1;
      return streamedResponse([[1, 2, 3]], { 'content-length': '3' });
    };

    await fetchPreopenBytes('/reckless/over-budget.nnue', 3);
    await fetchPreopenBytes('/reckless/over-budget.nnue', 3);
    assert.equal(fetches, 2);
    assert.deepEqual(recklessWasiWorkerInternalsForTests.preopenByteCacheState(), {
      byteBudget: 2,
      totalBytes: 0,
      keys: [],
      inFlightCount: 0,
    });
  } finally {
    globalThis.fetch = previousFetch;
    if (previousSelf === undefined) delete globalThis.self;
    else globalThis.self = previousSelf;
  }
});
