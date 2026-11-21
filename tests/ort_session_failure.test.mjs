import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { stripTypeScriptTypes } from 'node:module';
import test from 'node:test';
import { untilAborted } from '../src/lc0/modelCache.ts';
import { isOnnxModelParseError, OrtSessionFallbackError, ortErrorMessage } from '../src/lc0/onnxModelRecovery.ts';
import { planOrtSessionAttempts } from '../src/nn/ortRuntime.ts';

function link(path, start, end, expression, deps) {
  const source = readFileSync(new URL(path, import.meta.url), 'utf8');
  const body = source.slice(source.indexOf(start), source.indexOf(end, source.indexOf(start)));
  const bindings = { untilAborted, isOnnxModelParseError, ortErrorMessage, OrtSessionFallbackError, ...deps };
  const code = stripTypeScriptTypes(`(function(){\n${body.replace('export class', 'class')}${expression}\n})()`);
  return new Function(...Object.keys(bindings), `return ${code}`)(...Object.values(bindings));
}

test('ORT keeps the first error of this session even if its final retry fails differently', async (t) => {
  t.mock.method(console, 'warn', () => {});
  const first = new Error('allocation failed');
  const last = new Error("previous call to 'initWasm()' failed");
  const failures = [first, new Error('unfolded model failed'), last];
  const seen = [];
  const noop = () => {};
  const create = link(
    '../src/nn/ortRuntime.ts',
    'async function prepareOrtThread(',
    'export async function releaseOrtSession(',
    'return createOrtSessionFromSource;',
    {
      ort: {
        InferenceSession: {
          async create() {
            throw failures.shift();
          },
        },
      },
      ensureWebGpuAdapterProbed: async () => {},
      resolvedOrtExecutionProviders: () => ['webgpu', 'wasm'],
      lockRequestedOrtWasmArtifact: noop,
      lockOrtRuntimeArtifactKind: noop,
      prepareOrtModelInput: async () => ({ input: 'folded', original: 'original' }),
      requestedOrtExecutionProvider: noop,
      shouldFallbackToWasmAfterOrtFailure: () => true,
      planOrtSessionAttempts,
      sessionOptions: () => ({}),
      discardDequantFold: noop,
      recordSessionAttempt: (_providers, _ok, _ms, error) => seen.push(error),
    },
  );
  await assert.rejects(
    create(() => 'model', {}),
    { message: `${first.message} (fallback: ${last.message})`, cause: last },
  );
  assert.deepEqual(seen, [first.message, 'unfolded model failed', last.message]);
});

test('Lc0 GPU I/O retains its failure when the ordinary ORT session fails too', async (t) => {
  t.mock.method(console, 'warn', () => {});
  const parseError = new Error('invalid ModelProto');
  const Evaluator = link(
    '../src/lc0/onnxEvaluator.ts',
    'async function createLc0GpuIoSession(',
    '  dispose(): Promise<void> {',
    '}\nreturn Lc0OnnxEvaluator;',
    {
      gpuIoModelSpecs: () => ({}),
      ort: {
        resolveOrtSessionPlacement: async () => ({ providers: ['webgpu'] }),
        async createOrtSession(_model, { webgpuOnly }) {
          throw webgpuOnly ? new Error('allocation failed') : parseError;
        },
      },
    },
  );
  await assert.rejects(Evaluator.create('model', { ioMode: 'gpu', rejectModelParseErrors: true }), (error) => {
    assert.equal(error.message, 'allocation failed (fallback: invalid ModelProto)');
    assert.equal(isOnnxModelParseError(error), true, 'verified model recovery still sees the terminal parse error');
    return true;
  });
});
