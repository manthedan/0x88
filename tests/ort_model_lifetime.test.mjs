import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import test from 'node:test';
import { createLc0OnnxEvaluatorFromUrl } from '../src/lc0/lc0SmallModelSelection.ts';
import { Lc0OnnxEvaluator } from '../src/lc0/onnxEvaluator.ts';
import { encodeTensorProto, ONNX_FLOAT16, ONNX_INT8, ProtoWriter } from '../src/nn/onnxDequantFold.ts';
import * as ort from '../src/nn/ortRuntime.ts';

function model() {
  const node = new ProtoWriter().stringField(1, 'Wq').stringField(1, 'Ws').stringField(2, 'W').stringField(4, 'DequantizeLinear').finish();
  const graph = new ProtoWriter()
    .bytesField(1, node)
    .bytesField(5, encodeTensorProto('Wq', ONNX_INT8, [2], Uint8Array.of(1, 2)))
    .bytesField(5, encodeTensorProto('Ws', ONNX_FLOAT16, [], Uint8Array.of(0, 56)))
    .finish();
  return new ProtoWriter().bytesField(7, graph).finish().buffer;
}
function pin(t) {
  ort.setRequestedOrtExecutionProviderForCurrentThread('wasm');
  ort.setOrtDequantFoldForCurrentThread(true);
  t.after(() => {
    ort.setRequestedOrtExecutionProviderForCurrentThread(null);
    ort.setOrtDequantFoldForCurrentThread(null);
  });
}
function result(bytes, cached = false) {
  return {
    model: bytes,
    url: 'https://assets.0x88.app/models/lc0/unit.onnx',
    mode: 'cache',
    cacheStatus: cached ? 'hit' : 'miss',
    verification: cached ? 'metadata' : 'hashed',
    sha256Valid: true,
    expectedSha256: 'a'.repeat(64),
    elapsedMs: 0,
    telemetry: { source: 'cache-storage', totalMs: 0 },
  };
}

test('successful verified session does not reload or mutate a shared model result, and returns provenance only', async (t) => {
  pin(t);
  const bytes = model(),
    shared = result(bytes),
    loads = [];
  t.mock.method(ort.InferenceSession, 'create', async (input) => {
    assert.notEqual(input.buffer, bytes);
    return { async release() {} };
  });
  const created = await createLc0OnnxEvaluatorFromUrl(
    shared.url,
    {},
    {
      placement: async () => ({ providers: ['wasm'] }),
      load: async (...args) => {
        loads.push(args);
        return shared;
      },
      createEvaluator: (input, options) => Lc0OnnxEvaluator.create(input, options),
    },
  );
  assert.equal(loads.length, 1, 'normal warm loads do not re-download/re-read for fallback');
  assert.equal('model' in created.modelLoad, false);
  assert.equal(shared.model, bytes);
  assert.ok(bytes.byteLength > 0, 'shared/coalesced buffers are never detached or mutated');
  await created.evaluator.dispose();
});

test('fold rejection lazily rederives verified raw input once and preserves unfolded fallback diagnostics', async (t) => {
  pin(t);
  t.mock.method(console, 'warn', () => {});
  const inputs = [],
    loads = [];
  t.mock.method(ort.InferenceSession, 'create', async (input) => {
    inputs.push(input);
    if (inputs.length === 1) {
      assert.equal(loads.length, 1, 'no eager reload before first native create');
      throw new Error('folded rewrite rejected');
    }
    return { async release() {} };
  });
  const created = await createLc0OnnxEvaluatorFromUrl(
    'https://assets.0x88.app/models/lc0/unit.onnx',
    {},
    {
      placement: async () => ({ providers: ['wasm'] }),
      load: async (_url, options) => {
        loads.push(options);
        return result(model(), loads.length > 1);
      },
      createEvaluator: (input, options) => Lc0OnnxEvaluator.create(input, options),
    },
  );
  assert.equal(inputs.length, 2);
  assert.equal(loads.length, 2);
  assert.equal(loads[1].allowUnverified, false);
  assert.deepEqual(new Uint8Array(inputs[1]), new Uint8Array(model()));
  assert.equal(created.modelLoad.verification, 'metadata');
  assert.equal('model' in created.modelLoad, false);
  assert.equal(ort.lastOrtDequantFoldSummary().foldedNodes, 0);
  assert.equal(ort.lastOrtDequantFoldSummary().discardedAfter, 'folded rewrite rejected');
  await created.evaluator.dispose();
});

for (const fold of [false, true]) {
  test(`same-model WASM fallback works offline after WebGPU fails (fold=${fold})`, async (t) => {
    const runtime = await import(`../src/nn/ortRuntime.ts?offlineFallback=${fold}`);
    runtime.setRequestedOrtExecutionProviderForCurrentThread('auto');
    runtime.setOrtDequantFoldForCurrentThread(fold);
    t.mock.getter(globalThis, 'navigator', () => ({
      gpu: { requestAdapter: async () => ({ features: new Set(['shader-f16']) }) },
    }));
    t.mock.method(console, 'warn', () => {});
    const bytes = model();
    const attempts = [];
    const session = { async release() {} };
    t.mock.method(runtime.InferenceSession, 'create', async (input, { executionProviders }) => {
      attempts.push({ input, executionProviders });
      if (executionProviders.includes('webgpu')) throw new Error('WebGPU backend init failed');
      return session;
    });
    const reloadModel = t.mock.fn(async () => {
      if (fold && attempts.length === 1) return bytes;
      throw new Error('Network unavailable');
    });

    assert.equal(await runtime.createOrtSession(bytes, { reloadModel }), session);
    assert.deepEqual(
      attempts.map(({ executionProviders }) => executionProviders),
      fold ? [['webgpu'], ['webgpu'], ['wasm']] : [['webgpu'], ['wasm']],
    );
    assert.equal(attempts.at(-1).input, attempts.at(-2).input);
    assert.equal(reloadModel.mock.callCount(), fold ? 1 : 0);
    await runtime.releaseOrtSession(session);
  });
}

test('owned unfolded input is collectible while native creation of the folded session is still pending', () => {
  const selector = new URL('../src/lc0/lc0SmallModelSelection.ts', import.meta.url).href;
  const evaluator = new URL('../src/lc0/onnxEvaluator.ts', import.meta.url).href;
  const proto = new URL('../src/nn/onnxDequantFold.ts', import.meta.url).href;
  const runtime = new URL('../src/nn/ortRuntime.ts', import.meta.url).href;
  const source = `
import assert from 'node:assert/strict';
import {createLc0OnnxEvaluatorFromUrl} from ${JSON.stringify(selector)};
import {Lc0OnnxEvaluator} from ${JSON.stringify(evaluator)};
import {ProtoWriter,encodeTensorProto,ONNX_INT8,ONNX_FLOAT16} from ${JSON.stringify(proto)};
import * as ort from ${JSON.stringify(runtime)};
let weak, loads=0;
ort.setRequestedOrtExecutionProviderForCurrentThread('wasm');ort.setOrtDequantFoldForCurrentThread(true);
ort.InferenceSession.create=async input=>{
  assert.ok(input.byteLength>300000);
  for(let i=0;i<5;i++){await new Promise(setImmediate);global.gc();}
  assert.equal(weak.deref(),undefined,'unfolded buffer is still pinned across native create');
  assert.equal(loads,1,'no reload on the successful path');
  return {async release(){}};
};
const created=await createLc0OnnxEvaluatorFromUrl('https://assets.0x88.app/models/lc0/unit.onnx',{}, {
  placement:async()=>({providers:['wasm']}),
  load:async()=>{
    loads++;
    const n=new ProtoWriter().stringField(1,'Wq').stringField(1,'Ws').stringField(2,'W').stringField(4,'DequantizeLinear').finish();
    const g=new ProtoWriter().bytesField(1,n).bytesField(5,encodeTensorProto('Wq',ONNX_INT8,[200000],new Uint8Array(200000))).bytesField(5,encodeTensorProto('Ws',ONNX_FLOAT16,[],Uint8Array.of(0,56))).finish();
    const b=new ProtoWriter().bytesField(7,g).finish().buffer;weak=new WeakRef(b);
    return {model:b,url:'https://assets.0x88.app/models/lc0/unit.onnx',mode:'cache',cacheStatus:'miss',verification:'hashed',sha256Valid:true,elapsedMs:0,telemetry:{source:'network',totalMs:0}};
  },
  createEvaluator:(input,options)=>Lc0OnnxEvaluator.create(input,options),
});
assert.equal('model' in created.modelLoad,false);await created.evaluator.dispose();
`;
  const child = spawnSync(process.execPath, ['--expose-gc', '--experimental-strip-types', '--input-type=module', '-e', source], {
    encoding: 'utf8',
    timeout: 15000,
  });
  assert.equal(child.status, 0, `${child.error ?? ''}\n${child.stdout}\n${child.stderr}`);
});
