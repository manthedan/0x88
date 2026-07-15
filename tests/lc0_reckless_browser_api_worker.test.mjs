import assert from 'node:assert/strict';
import { test } from 'node:test';

let workerModule;

async function loadWorkerModule(messages) {
  globalThis.self = globalThis;
  globalThis.postMessage = (message) => {
    messages.push(message);
  };
  globalThis.onmessage = null;
  workerModule ??= await import('../src/lc0/recklessBrowserApiWorker.ts');
  return workerModule;
}

function streamedResponse(chunks, headers = {}) {
  return new Response(new ReadableStream({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(Uint8Array.from(chunk));
      controller.close();
    },
  }), { headers });
}

async function withWorkerGlobals(run) {
  const previousSelf = globalThis.self;
  const previousPostMessage = globalThis.postMessage;
  const previousOnMessage = globalThis.onmessage;
  const messages = [];
  try {
    const module = await loadWorkerModule(messages);
    await run(module, messages);
  } finally {
    if (previousSelf === undefined) delete globalThis.self;
    else globalThis.self = previousSelf;
    if (previousPostMessage === undefined) delete globalThis.postMessage;
    else globalThis.postMessage = previousPostMessage;
    if (previousOnMessage === undefined) delete globalThis.onmessage;
    else globalThis.onmessage = previousOnMessage;
  }
}

test('Reckless browser API NNUE rejects mismatching and oversized decoded metadata before allocation', async () => {
  await withWorkerGlobals(async ({ readNnueResponseWithProgress, MAX_NNUE_BYTES }, messages) => {
    const unreadBody = () => new ReadableStream({ pull() {} });
    await assert.rejects(
      readNnueResponseWithProgress(
        new Response(unreadBody(), { headers: { 'x-artifact-content-length': '5' } }),
        1,
        'nnue-fetch',
        '/reckless/mismatching-header.nnue',
        0,
        4,
      ),
      /metadata mismatch.*got 5, expected 4/,
    );
    await assert.rejects(
      readNnueResponseWithProgress(
        new Response(unreadBody(), { headers: { 'x-artifact-content-length': String(MAX_NNUE_BYTES + 1) } }),
        2,
        'nnue-fetch',
        '/reckless/oversized-header.nnue',
        0,
      ),
      /exceeds the .* hard maximum/,
    );
    assert.deepEqual(messages, []);
  });
});

test('Reckless browser API NNUE ignores encoded Content-Length and rejects decoded overflow', async () => {
  await withWorkerGlobals(async ({ readNnueResponseWithProgress }, messages) => {
    const response = streamedResponse([[1, 2, 3], [4, 5]], {
      'content-encoding': 'br',
      'content-length': '2',
    });
    await assert.rejects(
      readNnueResponseWithProgress(response, 2, 'nnue-fetch', '/reckless/encoded-overflow.nnue', 0, 4),
      /exceeds its 4-byte download limit.*at least 5/,
    );
    assert.ok(messages.every((message) => message.totalBytes === 4));
    assert.ok(messages.every((message) => message.totalBytes !== 2));
  });
});

test('Reckless browser API NNUE rejects an undersized final decoded body', async () => {
  await withWorkerGlobals(async ({ readNnueResponseWithProgress }, messages) => {
    const response = streamedResponse([[1, 2, 3]], { 'content-encoding': 'br' });
    await assert.rejects(
      readNnueResponseWithProgress(response, 3, 'nnue-fetch', '/reckless/undersized.nnue', 0, 4),
      /decoded byte length mismatch: got 3, expected 4/,
    );
    assert.ok(messages.every((message) => message.totalBytes === 4));
  });
});

test('Reckless browser API NNUE accepts an exact decoded body', async () => {
  await withWorkerGlobals(async ({ readNnueResponseWithProgress }, messages) => {
    const response = streamedResponse([[1, 2], [3, 4]], {
      'content-encoding': 'br',
      'content-length': '2',
    });
    const buffer = await readNnueResponseWithProgress(response, 4, 'nnue-fetch', '/reckless/exact.nnue', 0, 4);
    assert.deepEqual([...new Uint8Array(buffer)], [1, 2, 3, 4]);
    assert.equal(messages.at(-1).type, 'status');
    assert.equal(messages.at(-1).phase, 'nnue-fetch-ready');
    assert.equal(messages.at(-1).loadedBytes, 4);
    assert.equal(messages.at(-1).totalBytes, 4);
  });
});

test('Reckless browser API NNUE cache identity includes expected decoded bytes', async () => {
  const previousFetch = globalThis.fetch;
  let fetches = 0;
  try {
    await withWorkerGlobals(async ({ fetchNnue }) => {
      globalThis.fetch = async () => {
        fetches += 1;
        return fetches === 1
          ? streamedResponse([[1]], { 'content-length': '1' })
          : streamedResponse([[1, 2]], { 'content-length': '2' });
      };
      const [first, second] = await Promise.all([
        fetchNnue('/reckless/cache-key-separation.nnue', 5, 1),
        fetchNnue('/reckless/cache-key-separation.nnue', 6, 2),
      ]);
      assert.equal(first.byteLength, 1);
      assert.equal(second.byteLength, 2);
      assert.equal(fetches, 2);
    });
  } finally {
    globalThis.fetch = previousFetch;
  }
});

test('Reckless browser API NNUE cache preserves concurrent dedupe and failed-fetch retry', async () => {
  const previousFetch = globalThis.fetch;
  let fetches = 0;
  try {
    await withWorkerGlobals(async ({ fetchNnue }) => {
      globalThis.fetch = async () => {
        fetches += 1;
        if (fetches === 1) return new Response('', { status: 503 });
        return streamedResponse([[7]], { 'content-length': '1' });
      };
      await assert.rejects(fetchNnue('/reckless/retry.nnue', 7, 1), /HTTP 503/);
      const [first, second] = await Promise.all([
        fetchNnue('/reckless/retry.nnue', 8, 1),
        fetchNnue('/reckless/retry.nnue', 9, 1),
      ]);
      assert.equal(first, second);
      assert.deepEqual([...new Uint8Array(first)], [7]);
      assert.equal(fetches, 2);
    });
  } finally {
    globalThis.fetch = previousFetch;
  }
});

test('Reckless browser API NNUE cache does not retain settled large-buffer promises', async () => {
  const previousFetch = globalThis.fetch;
  let fetches = 0;
  try {
    await withWorkerGlobals(async ({ fetchNnue }) => {
      globalThis.fetch = async () => {
        fetches += 1;
        return streamedResponse([[fetches]], { 'content-length': '1' });
      };
      assert.deepEqual([...new Uint8Array(await fetchNnue('/reckless/settled.nnue', 10, 1))], [1]);
      assert.deepEqual([...new Uint8Array(await fetchNnue('/reckless/settled.nnue', 11, 1))], [2]);
      assert.equal(fetches, 2);
    });
  } finally {
    globalThis.fetch = previousFetch;
  }
});
