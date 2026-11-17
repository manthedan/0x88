import assert from 'node:assert/strict';
import test from 'node:test';
import { createLc0OnnxEvaluatorFromUrl, selectLc0OrtModelUrl, LC0_SMALL_BATCH1_QDQ8_PATH } from '../src/lc0/lc0SmallModelSelection.ts';
import { isOnnxModelParseError } from '../src/lc0/onnxModelRecovery.ts';
import { Lc0OnnxEvaluator } from '../src/lc0/onnxEvaluator.ts';
import * as ort from '../src/nn/ortRuntime.ts';
import { ProtoWriter } from '../src/nn/onnxDequantFold.ts';

const ORIGINAL = LC0_SMALL_BATCH1_QDQ8_PATH;
const PLACEMENT = { providers: ['webgpu', 'wasm'], adapterVendor: 'apple' };
const SPLITK = selectLc0OrtModelUrl(ORIGINAL, PLACEMENT);
const ARTIFACT = 'https://assets.0x88.app/artifacts/verified-body';
const SHA = 'a'.repeat(64);
const bytes = (text) => new TextEncoder().encode(text).buffer;
function loadResult({ url = ARTIFACT, text = 'old', cacheStatus = 'hit', verification = 'metadata' } = {}) {
  return {
    model: bytes(text),
    url,
    mode: 'cache',
    cacheStatus,
    verification,
    sha256Valid: true,
    expectedSha256: SHA,
    sha256: SHA,
    bytes: 3,
    expectedBytes: 3,
    elapsedMs: 0,
    telemetry: { source: 'cache-storage', totalMs: 0 },
  };
}
function setup(
  t,
  {
    first = loadResult(),
    second = loadResult({ text: 'new', cacheStatus: 'miss', verification: 'hashed' }),
    error = new Error('Protobuf parsing failed'),
    failAgain = false,
  } = {},
) {
  t.mock.method(console, 'warn', () => {});
  const loads = [],
    evictions = [],
    creations = [];
  const deps = {
    placement: async () => PLACEMENT,
    load: async (url, options) => {
      loads.push({ url, options });
      return loads.length === 1 ? first : second;
    },
    clearCacheEntry: async (...args) => {
      evictions.push(args);
      return true;
    },
    createEvaluator: async (model) => {
      creations.push(model);
      if (creations.length === 1 || failAgain) throw error;
      return { async dispose() {} };
    },
  };
  return { deps, loads, evictions, creations };
}

test('a real ORT primary parse failure is repaired before a valid original-WASM fallback can mask it', async (t) => {
  const descriptor = Object.getOwnPropertyDescriptor(globalThis, 'navigator');
  Object.defineProperty(globalThis, 'navigator', {
    configurable: true,
    value: { hardwareConcurrency: 2, gpu: { requestAdapter: async () => ({ info: { vendor: 'apple' }, features: new Set(['shader-f16']) }) } },
  });
  ort.resetWebGpuAdapterProbeForTests();
  ort.setRequestedOrtExecutionProviderForCurrentThread('webgpu,wasm');
  ort.setOrtDequantFoldForCurrentThread(false);
  t.after(() => {
    if (descriptor) Object.defineProperty(globalThis, 'navigator', descriptor);
    else delete globalThis.navigator;
    ort.resetWebGpuAdapterProbeForTests();
    ort.setRequestedOrtExecutionProviderForCurrentThread(null);
    ort.setOrtDequantFoldForCurrentThread(null);
  });
  t.mock.method(console, 'warn', () => {});
  const providers = [],
    loads = [];
  t.mock.method(ort.InferenceSession, 'create', async (_model, options) => {
    providers.push(options.executionProviders);
    if (providers.length === 1) throw new Error('INVALID_PROTOBUF');
    return { async release() {} };
  });
  const body = new ProtoWriter().bytesField(7, new ProtoWriter().stringField(2, 'test').finish()).finish().buffer;
  const result = await createLc0OnnxEvaluatorFromUrl(
    ORIGINAL,
    { deviceSelection: true },
    {
      placement: async () => PLACEMENT,
      load: async (url, options) => {
        loads.push(url);
        return { ...loadResult(options.forceReload ? { verification: 'hashed', cacheStatus: 'miss' } : {}), model: body };
      },
      clearCacheEntry: async () => true,
      createEvaluator: (model, options) => Lc0OnnxEvaluator.create(model, options),
    },
  );
  assert.deepEqual(loads, [SPLITK, SPLITK], 'no original load before repairing selected cache bytes');
  assert.deepEqual(providers, [['webgpu'], ['webgpu']]);
  assert.equal(result.variant, 'splitk');
  await result.evaluator.dispose();
});

test('only model parse/protobuf failures qualify for recovery', () => {
  for (const message of [
    'Protobuf parsing failed',
    'INVALID_PROTOBUF',
    'ModelProto does not have a graph',
    'Failed to parse ONNX model',
    'ERROR_CODE: 7, ERROR_MESSAGE: malformed wire type',
  ])
    assert.equal(isOnnxModelParseError(new Error(message)), true, message);
  for (const message of [
    'WebGPU device lost',
    'Failed to create GPU session',
    'unsupported operator',
    'protobuf allocation failed: out of memory',
    'ERROR_CODE: 70',
    'wrong tensor shape',
  ])
    assert.equal(isOnnxModelParseError(new Error(message)), false, message);
  assert.equal(isOnnxModelParseError(new DOMException('protobuf', 'AbortError')), false);
});

test('one-shot recovery evicts resolved identity, preserves cache options, and requires a fresh verified load', async (t) => {
  const fake = setup(t);
  const metadataStore = {};
  const load = { cache: true, cacheName: 'unit-models', metadataStore, allowUnverified: true };
  const result = await createLc0OnnxEvaluatorFromUrl(ORIGINAL, { load }, fake.deps);
  assert.deepEqual(fake.evictions, [[ARTIFACT, 'unit-models', metadataStore]]);
  assert.deepEqual(
    fake.loads.map((l) => l.url),
    [ORIGINAL, ORIGINAL],
  );
  assert.equal(fake.loads[1].options.forceReload, true);
  assert.equal(fake.loads[1].options.allowUnverified, false);
  assert.equal(fake.loads[1].options.metadataStore, metadataStore);
  assert.equal(load.forceReload, undefined, 'caller policy is not mutated');
  assert.equal(result.modelLoad.revalidated, true);
  assert.equal(result.modelLoad.verification, 'hashed');
  assert.equal(fake.creations.length, 2);
});

test('a second parse failure propagates without a third create/fetch or second eviction', async (t) => {
  const fake = setup(t, { failAgain: true });
  await assert.rejects(createLc0OnnxEvaluatorFromUrl(ORIGINAL, {}, fake.deps), /Protobuf parsing failed/);
  assert.equal(fake.loads.length, 2);
  assert.equal(fake.creations.length, 2);
  assert.equal(fake.evictions.length, 1);
});

test('a recovery loader cannot hand metadata-trusted or unverified bytes back to ORT', async (t) => {
  const fake = setup(t, { second: loadResult() });
  await assert.rejects(createLc0OnnxEvaluatorFromUrl(ORIGINAL, {}, fake.deps), /freshly SHA-256-verified/);
  assert.equal(fake.creations.length, 1);
});

for (const first of [
  loadResult({ cacheStatus: 'miss', verification: 'hashed' }),
  loadResult({ verification: 'hashed' }),
  { ...loadResult(), sha256Valid: false },
]) {
  test(`already-hashed/cold/unverified parse failures do not evict (${first.cacheStatus}/${first.verification}/${first.sha256Valid})`, async (t) => {
    const fake = setup(t, { first });
    await assert.rejects(createLc0OnnxEvaluatorFromUrl(ORIGINAL, {}, fake.deps), /Protobuf/);
    assert.equal(fake.loads.length, 1);
    assert.equal(fake.evictions.length, 0);
  });
}
for (const error of [
  new Error('WebGPU device lost'),
  new Error('out of memory'),
  new Error('unsupported operator'),
  new DOMException('cancelled', 'AbortError'),
]) {
  test(`good warm entries survive non-parse failures: ${error.message}`, async (t) => {
    const fake = setup(t, { error });
    await assert.rejects(createLc0OnnxEvaluatorFromUrl(ORIGINAL, {}, fake.deps), (e) => e === error);
    assert.equal(fake.loads.length, 1);
    assert.equal(fake.evictions.length, 0);
  });
}

test('an abort racing a parse error prevents eviction and refetch', async (t) => {
  const fake = setup(t);
  const controller = new AbortController();
  fake.deps.createEvaluator = async () => {
    controller.abort();
    throw new Error('protobuf');
  };
  await assert.rejects(createLc0OnnxEvaluatorFromUrl(ORIGINAL, { load: { signal: controller.signal } }, fake.deps), { name: 'AbortError' });
  assert.equal(fake.loads.length, 1);
  assert.equal(fake.evictions.length, 0);
});

test('split-K parse recovery reloads the selected model without changing production selection', async (t) => {
  const fake = setup(t);
  const result = await createLc0OnnxEvaluatorFromUrl(ORIGINAL, { deviceSelection: true }, fake.deps);
  assert.deepEqual(
    fake.loads.map((l) => l.url),
    [SPLITK, SPLITK],
  );
  assert.equal(result.variant, 'splitk');
  assert.equal(result.servedUrl, SPLITK);
  assert.equal(result.modelLoad.revalidated, true);
});

for (const secondUsesWasm of [true, false]) {
  test(`WASM-original parse recovery evicts original, not valid split-K; second uses WASM=${secondUsesWasm}`, async (t) => {
    t.mock.method(console, 'warn', () => {});
    const loads = [],
      evictions = [];
    let creates = 0;
    const selected = loadResult({ url: 'https://assets.0x88.app/selected', text: 'gpu', cacheStatus: 'miss', verification: 'hashed' });
    const deps = {
      placement: async () => PLACEMENT,
      load: async (url, options) => {
        loads.push({ url, options });
        return url === SPLITK ? selected : loadResult({ ...(options.forceReload ? { text: 'new', cacheStatus: 'miss', verification: 'hashed' } : {}) });
      },
      clearCacheEntry: async (url) => {
        evictions.push(url);
        return true;
      },
      createEvaluator: async (_model, options) => {
        creates++;
        if (creates === 1 || secondUsesWasm) await options.wasmFallbackModel();
        if (creates === 1) throw new Error('INVALID_PROTOBUF');
        return { async dispose() {} };
      },
    };
    const result = await createLc0OnnxEvaluatorFromUrl(ORIGINAL, { deviceSelection: true }, deps);
    assert.deepEqual(evictions, [ARTIFACT]);
    assert.deepEqual(
      loads.map((l) => l.url),
      [SPLITK, ORIGINAL, ORIGINAL],
    );
    assert.equal(loads[2].options.forceReload, true);
    assert.equal(result.servedUrl, secondUsesWasm ? ORIGINAL : SPLITK);
    assert.equal(result.variant, secondUsesWasm ? 'original' : 'splitk');
    assert.equal(result.modelLoad.url, secondUsesWasm ? ARTIFACT : selected.url);
    assert.equal(creates, 2);
  });
}
