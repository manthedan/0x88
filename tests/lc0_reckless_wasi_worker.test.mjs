import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
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
  return new Response(new ReadableStream({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(Uint8Array.from(chunk));
      controller.close();
    },
  }), { headers });
}

test('Reckless WASI preopen download preallocates Brotli-decoded bytes from artifact length', async () => {
  const previousSelf = globalThis.self;
  const previousFetch = globalThis.fetch;
  const messages = [];
  try {
    const { fetchPreopenBytes } = await loadWorkerModule(messages);
    globalThis.fetch = async () => streamedResponse([[1, 2], [3, 4]], {
      'content-encoding': 'br',
      'content-length': '2',
      'x-artifact-content-length': '4',
    });
    const buffer = await fetchPreopenBytes('/reckless/br-known.nnue');
    assert.equal(buffer.byteLength, 4);
    assert.deepEqual([...new Uint8Array(buffer)], [1, 2, 3, 4]);
    assert.deepEqual(messages.at(-1), {
      type: 'preopen-progress',
      url: '/reckless/br-known.nnue',
      loadedBytes: 4,
      totalBytes: 4,
    });

    const source = await readFile('src/lc0/recklessWasiWorker.ts', 'utf8');
    assert.doesNotMatch(source, /chunks:\s*Uint8Array\[\]/);
    assert.match(source, /new Uint8Array\(totalBytes \?\? 0\)/);
    assert.match(source, /loadedBytes === bytes\.byteLength \? buffer : buffer\.slice\(0, loadedBytes\)/);
  } finally {
    globalThis.fetch = previousFetch;
    if (previousSelf === undefined) delete globalThis.self;
    else globalThis.self = previousSelf;
  }
});

test('Reckless WASI preopen download ignores encoded Brotli content length without decoded metadata', async () => {
  const previousSelf = globalThis.self;
  const previousFetch = globalThis.fetch;
  const messages = [];
  try {
    const { fetchPreopenBytes } = await loadWorkerModule(messages);
    const first = new Uint8Array(40_000).fill(7);
    const second = new Uint8Array(30_000).fill(9);
    globalThis.fetch = async () => streamedResponse([first, second], {
      'content-encoding': 'br',
      'content-length': '1234',
    });
    const buffer = await fetchPreopenBytes('/reckless/br-unknown.nnue');
    const bytes = new Uint8Array(buffer);
    assert.equal(bytes.byteLength, 70_000);
    assert.equal(bytes[0], 7);
    assert.equal(bytes[39_999], 7);
    assert.equal(bytes[40_000], 9);
    assert.equal(bytes[69_999], 9);
    assert.ok(messages.some((message) => message.totalBytes === 0));
    assert.ok(messages.every((message) => message.totalBytes !== 1234));
    assert.equal(messages.at(-1).totalBytes, 70_000);
  } finally {
    globalThis.fetch = previousFetch;
    if (previousSelf === undefined) delete globalThis.self;
    else globalThis.self = previousSelf;
  }
});

test('Reckless WASI preopen download uses content length for identity responses', async () => {
  const previousSelf = globalThis.self;
  const previousFetch = globalThis.fetch;
  const messages = [];
  try {
    const { fetchPreopenBytes } = await loadWorkerModule(messages);
    globalThis.fetch = async () => streamedResponse([[1, 2], [3, 4]], {
      'content-encoding': 'identity',
      'content-length': '4',
    });
    const buffer = await fetchPreopenBytes('/reckless/identity.nnue');
    assert.equal(buffer.byteLength, 4);
    assert.deepEqual([...new Uint8Array(buffer)], [1, 2, 3, 4]);
    assert.deepEqual(messages.at(-1), {
      type: 'preopen-progress',
      url: '/reckless/identity.nnue',
      loadedBytes: 4,
      totalBytes: 4,
    });
  } finally {
    globalThis.fetch = previousFetch;
    if (previousSelf === undefined) delete globalThis.self;
    else globalThis.self = previousSelf;
  }
});

test('Reckless WASI preopen download deduplicates only concurrent requests', async () => {
  const previousSelf = globalThis.self;
  const previousFetch = globalThis.fetch;
  const messages = [];
  let fetches = 0;
  try {
    const { fetchPreopenBytes } = await loadWorkerModule(messages);
    globalThis.fetch = async () => {
      fetches += 1;
      return streamedResponse([[1]], { 'content-length': '1' });
    };
    const [first, second] = await Promise.all([
      fetchPreopenBytes('/reckless/dedup.nnue'),
      fetchPreopenBytes('/reckless/dedup.nnue'),
    ]);
    assert.equal(first, second);
    assert.equal(fetches, 1);

    await fetchPreopenBytes('/reckless/dedup.nnue');
    assert.equal(fetches, 2, 'settled download promises must not remain retained');
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
    const { fetchPreopenBytes } = await loadWorkerModule(messages);
    globalThis.fetch = async () => {
      responseIndex += 1;
      if (responseIndex === 1) return new Response('', { status: 503 });
      if (responseIndex === 2) {
        return new Response(new ReadableStream({
          start(controller) {
            controller.error(new Error('corrupt response stream'));
          },
        }));
      }
      return streamedResponse([[5]], { 'content-length': '1' });
    };

    await assert.rejects(fetchPreopenBytes('/reckless/retry.nnue'), /HTTP 503/);
    await assert.rejects(fetchPreopenBytes('/reckless/retry.nnue'), /corrupt response stream/);
    assert.deepEqual([...new Uint8Array(await fetchPreopenBytes('/reckless/retry.nnue'))], [5]);
    assert.equal(responseIndex, 3);
  } finally {
    globalThis.fetch = previousFetch;
    if (previousSelf === undefined) delete globalThis.self;
    else globalThis.self = previousSelf;
  }
});
