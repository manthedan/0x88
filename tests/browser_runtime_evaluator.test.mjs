import assert from 'node:assert/strict';
import test from 'node:test';
import { parseFen, START_FEN } from '../src/chess/board.ts';
import {
  browserRuntimeAuditIdentity,
  formatRuntimeFallbackWarning,
  reconcileRuntimeFallbackWarning,
  updateRuntimeFallbackWarning,
} from '../src/nn/runtimeAudit.ts';
import { RuntimeFallbackEvaluator } from '../src/nn/runtimeFallbackEvaluator.ts';
import { SquareFormerEvaluator } from '../src/nn/squareformerEvaluator.ts';

function pendingEvaluation() {
  let reject;
  const promise = new Promise((_resolve, rejectPromise) => {
    reject = rejectPromise;
  });
  return { promise, reject };
}
test('runtime fallback warning names the transition and reason', () => {
  assert.equal(
    formatRuntimeFallbackWarning({
      source: 'test',
      family: 'centipawn',
      engineLabel: 'Centipawn',
      requestedRuntime: 'tvmjs-webgpu',
      resolvedRuntime: 'tvmjs-webgpu-fallback-ort',
      fallbackReason: 'GPU device lost',
    }),
    'Centipawn switched from tvmjs-webgpu to tvmjs-webgpu-fallback-ort. Performance may be reduced. Reason: GPU device lost',
  );
  assert.equal(formatRuntimeFallbackWarning({ source: 'test', family: 'centipawn' }), null);
});

test('runtime fallback warning clears only after the same runtime succeeds', () => {
  const target = { dataset: {}, hidden: true, textContent: '' };
  updateRuntimeFallbackWarning(target, {
    source: 'test',
    surface: 'analysis',
    family: 'centipawn',
    requestedRuntime: 'tvmjs-webgpu',
    resolvedRuntime: 'ort-wasm',
    fallbackReason: 'GPU device lost',
  });
  assert.equal(target.hidden, false);
  assert.match(target.textContent, /GPU device lost/);

  updateRuntimeFallbackWarning(target, {
    source: 'test',
    surface: 'arena',
    family: 'centipawn',
    requestedRuntime: 'tvmjs-webgpu',
    resolvedRuntime: 'tvmjs-webgpu',
  });
  assert.equal(target.hidden, false);

  updateRuntimeFallbackWarning(target, {
    source: 'test',
    surface: 'analysis',
    family: 'centipawn',
    requestedRuntime: 'tvmjs-webgpu',
    resolvedRuntime: 'tvmjs-webgpu',
  });
  assert.equal(target.hidden, true);
  assert.equal(target.textContent, '');
});

test('runtime fallback warning clears when its exact engine configuration is removed', () => {
  const target = { dataset: {}, hidden: true, textContent: '' };
  const fallback = {
    source: 'test',
    surface: 'arena',
    family: 'centipawn',
    engineLabel: 'Centipawn · WebGPU',
    modelId: 'centipawn-model',
    requestedRuntime: 'tvmjs-webgpu',
    resolvedRuntime: 'ort-wasm',
    fallbackReason: 'GPU device lost',
  };
  updateRuntimeFallbackWarning(target, fallback);
  reconcileRuntimeFallbackWarning(target, new Set([browserRuntimeAuditIdentity(fallback)]));
  assert.equal(target.hidden, false);
  reconcileRuntimeFallbackWarning(
    target,
    new Set([
      browserRuntimeAuditIdentity({
        ...fallback,
        engineLabel: 'Centipawn · ORT',
        requestedRuntime: 'onnx',
      }),
    ]),
  );
  assert.equal(target.hidden, true);
  assert.equal(target.textContent, '');
});

test('runtime fallback warning preserves other active engine failures', () => {
  const target = { dataset: {}, hidden: true, textContent: '' };
  const centipawnFallback = {
    source: 'test',
    surface: 'analysis',
    family: 'centipawn',
    engineLabel: 'Centipawn',
    modelId: 'centipawn-model',
    requestedRuntime: 'auto',
    resolvedRuntime: 'ort',
    fallbackReason: 'GPU device lost',
  };
  const lc0Fallback = {
    source: 'test',
    surface: 'analysis',
    family: 'lc0',
    engineLabel: 'LC0',
    modelId: 'lc0-default',
    requestedRuntime: 'worker-hybrid',
    resolvedRuntime: 'ort-main-thread-fallback',
    fallbackReason: 'worker failed',
  };
  updateRuntimeFallbackWarning(target, centipawnFallback);
  updateRuntimeFallbackWarning(target, lc0Fallback);
  assert.match(target.textContent, /GPU device lost/);
  assert.match(target.textContent, /worker failed/);

  reconcileRuntimeFallbackWarning(target, new Set([browserRuntimeAuditIdentity(lc0Fallback)]));
  assert.equal(target.hidden, false);
  assert.doesNotMatch(target.textContent, /GPU device lost/);
  assert.match(target.textContent, /worker failed/);

  updateRuntimeFallbackWarning(target, {
    source: 'test',
    surface: 'analysis',
    family: 'lc0',
    engineLabel: 'LC0',
    modelId: 'lc0-default',
    requestedRuntime: 'worker-hybrid',
    resolvedRuntime: 'worker-hybrid-lazy',
  });
  assert.equal(target.hidden, true);
  assert.equal(target.textContent, '');
});

test('runtime fallback does not create ORT after destruction', async () => {
  const pending = pendingEvaluation();
  let primaryDestroyed = false;
  let fallbackCreates = 0;
  const evaluator = new RuntimeFallbackEvaluator(
    {
      evaluate: () => pending.promise,
      destroy() {
        primaryDestroyed = true;
      },
    },
    async () => {
      fallbackCreates += 1;
      return { evaluate: async () => ({ policy: new Map(), wdl: [0, 1, 0] }) };
    },
    { source: 'test' },
  );
  const evaluation = evaluator.evaluate(parseFen(START_FEN));
  evaluator.destroy();
  pending.reject(new Error('device lost'));
  await assert.rejects(evaluation, (error) => error?.name === 'AbortError');
  assert.equal(primaryDestroyed, true);
  assert.equal(fallbackCreates, 0);
});

test('runtime fallback destroys ORT when creation completes after destruction', async () => {
  const pending = pendingEvaluation();
  let resolveFallback;
  let fallbackDestroyed = false;
  const fallbackCreated = new Promise((resolve) => {
    resolveFallback = resolve;
  });
  const evaluator = new RuntimeFallbackEvaluator(
    {
      evaluate: () => pending.promise,
      destroy() {},
    },
    () => fallbackCreated,
    { source: 'test' },
  );
  const evaluation = evaluator.evaluate(parseFen(START_FEN));
  pending.reject(new Error('device lost'));
  await Promise.resolve();
  evaluator.destroy();
  resolveFallback({
    evaluate: async () => ({ policy: new Map(), wdl: [0, 1, 0] }),
    destroy() {
      fallbackDestroyed = true;
    },
  });
  await assert.rejects(evaluation, (error) => error?.name === 'AbortError');
  assert.equal(fallbackDestroyed, true);
});

test('SquareFormer evaluator releases its ORT session once', async () => {
  let releases = 0;
  const evaluator = new SquareFormerEvaluator(
    {
      release() {
        releases += 1;
      },
    },
    {},
  );
  evaluator.destroy();
  evaluator.destroy();
  await Promise.resolve();
  assert.equal(releases, 1);
});
