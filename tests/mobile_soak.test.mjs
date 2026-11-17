import assert from 'node:assert/strict';
import { test } from 'node:test';
import {
  addSoakEvent,
  allocateSoakBallast,
  beginSoakStep,
  buildSoakSteps,
  continueSoakReport,
  decodeSoakReport,
  departureSoakReport,
  emptySoakReport,
  finishSoakStep,
  makeHeatSlice,
  newSoakResult,
  parseSoakQuery,
  rate,
  recoverSoakReport,
  skipRemainingSoakSteps,
  soakRuntimeSummary,
  strongestLoadedNet,
  validateSoakModelOverride,
} from '../src/lc0/mobileSoak.ts';
import { watchOrtDeviceLoss } from '../src/nn/ortDeviceLoss.ts';
import { env } from '../src/nn/ortRuntime.ts';
import { SquareFormerEvaluator } from '../src/nn/squareformerEvaluator.ts';

const at = '2026-11-16T12:00:00.000Z';
const later = '2026-11-16T12:01:00.000Z';
const empty = () => emptySoakReport(parseSoakQuery('?steps=sf-lite,t3,heat'), at);

test('lab gap/ballast validation, persistence and incompressible 32 MiB chunks', () => {
  const query = parseSoakQuery('?steps=t3,maia3&labGapSeconds=20&labBallastMb=33');
  const report = emptySoakReport(query, at);
  assert.equal(decodeSoakReport(JSON.stringify(report)).labGapSeconds, 20);
  assert.equal(report.labBallastMb, 33);
  for (const key of ['labGapSeconds', 'labBallastMb']) {
    for (const value of ['-1', '1.5', 'Infinity', 'NaN', '99999']) assert.throws(() => parseSoakQuery(`?${key}=${value}`), /integer/);
  }
  const buffers = allocateSoakBallast(33);
  assert.deepEqual(
    buffers.map((b) => b.byteLength),
    [32 * 1048576, 1048576],
  );
  const words = new Uint32Array(buffers[0]);
  assert.equal(new Set(words.subarray(0, 4096)).size, 4096);
  assert.notEqual(words[0], new Uint32Array(buffers[1])[0], 'PRNG does not restart per chunk');
  assert.deepEqual(allocateSoakBallast(0), []);
  assert.equal(decodeSoakReport(JSON.stringify({ ...report, labBallastMb: -1 })), null);
});

test('catalog has stable unique step IDs in increasing-pressure order', () => {
  const steps = buildSoakSteps();
  assert.equal(steps.length, 15);
  assert.equal(new Set(steps.map((s) => s.id)).size, steps.length);
  assert.deepEqual(
    steps.slice(-2).map((s) => s.id),
    ['arena', 'heat'],
  );
  assert.equal(steps.filter((s) => s.kind === 'cpu').length, 7);
});

test('query defaults, empty selection, unknown IDs, deduplication and literal auto', () => {
  assert.equal(parseSoakQuery('').stepIds.length, 15);
  assert.deepEqual(parseSoakQuery('?steps=').stepIds, []);
  assert.deepEqual(parseSoakQuery('?steps=t3,sf-lite,t3,nope&auto=1'), {
    auto: true,
    labGapSeconds: 0,
    labBallastMb: 0,
    modelOverrides: {},
    stepIds: ['t3', 'sf-lite'],
    heatSeconds: 120,
    warnings: ['Unknown step: nope'],
  });
  assert.equal(parseSoakQuery('?auto=true').auto, false);
});

test('lab-only model overrides validate trusted URLs and survive crash recovery', () => {
  const query = parseSoakQuery('?steps=lqo&t3Model=https%3A%2F%2Fassets.0x88.app%2Fmodels%2Flc0%2Ft3.onnx&lqoModel=%2Fmodels%2Flc0%2Flqo_v2.f16.onnx');
  assert.deepEqual(query.modelOverrides, { lqo: '/models/lc0/lqo_v2.f16.onnx', t3: 'https://assets.0x88.app/models/lc0/t3.onnx' });
  const report = beginSoakStep(emptySoakReport(query, at), 'lqo', at);
  assert.equal(report.steps[0].status, 'running');
  const restored = recoverSoakReport(decodeSoakReport(JSON.stringify(report)), later);
  assert.equal(restored.steps[0].status, 'crashed');
  assert.deepEqual(restored.modelOverrides, query.modelOverrides);
  assert.equal(validateSoakModelOverride('https://assets.0x88.app/artifacts/sha256/model.onnx'), 'https://assets.0x88.app/artifacts/sha256/model.onnx');
});

test('invalid model overrides fail closed rather than silently loading the shipped QDQ net', () => {
  for (const value of [
    '',
    '/models/lc0/',
    '/models/other.onnx',
    'model.onnx',
    '//assets.0x88.app/models/lc0/a.onnx',
    'http://assets.0x88.app/models/lc0/a.onnx',
    'https://evil.test/models/lc0/a.onnx',
    'https://assets.0x88.app.evil.test/a',
    'https://user@assets.0x88.app/a',
    'https://assets.0x88.app:8443/a',
    'data:text/plain,x',
    '/models/lc0/a.onnx#fragment',
    '/models/lc0/../a.onnx',
    '/models/lc0/a/../../private',
    '/models/lc0/%2e%2e/a.onnx',
    '/models/lc0/%252e%252e/a.onnx',
    '/models/lc0/%2F..%2F..%2Fprivate',
    '/models/lc0/a\\\\..\\\\private',
    ' /models/lc0/a.onnx',
    '/models/lc0/a.onnx\n',
  ]) {
    assert.throws(() => parseSoakQuery(`?lqoModel=${encodeURIComponent(value)}`), /lqoModel:/, value);
  }
  assert.equal(decodeSoakReport(JSON.stringify({ ...empty(), modelOverrides: { lqo: 'https://evil.test/a' } })), null);
});

test('heat seconds are finite, integer and bounded', () => {
  for (const [input, expected] of [
    ['NaN', 120],
    ['Infinity', 120],
    ['0', 1],
    ['-1', 1],
    ['99999', 1800],
    ['12.9', 12],
  ])
    assert.equal(parseSoakQuery(`?heatSeconds=${input}`).heatSeconds, expected);
});

test('begin persists complete marker and nullable telemetry before any allocation', () => {
  const before = empty();
  const running = beginSoakStep(before, 'sf-lite', at);
  assert.equal(before.inProgress, null);
  assert.equal(running.state, 'running');
  assert.equal(running.steps[0].status, 'running', 'API and saved report must not pessimistically say failed');
  assert.deepEqual(running.inProgress, { id: 'sf-lite', at });
  assert.equal(running.steps[0].loadMs, null);
  assert.equal(running.steps[0].firstSearchMs, null);
  assert.deepEqual(decodeSoakReport(JSON.stringify(running)), running);
});

test('finish clears marker only after the result, keeps previous results and advances once', () => {
  const running = beginSoakStep(empty(), 'sf-lite', at);
  const result = { ...running.steps[0], status: 'ok', finishedAt: later, loadMs: 123 };
  const finished = finishSoakStep(running, result);
  assert.equal(finished.inProgress, null);
  assert.equal(finished.nextIndex, 1);
  assert.deepEqual(finished.steps, [result]);
  const restored = recoverSoakReport(finished, later);
  assert.equal(restored.state, 'idle');
  assert.equal(restored.reloadCount, 0);
  assert.deepEqual(restored.steps, finished.steps);
});

test('reload during environment setup or the last completion cannot strand a running UI without a marker', () => {
  const initial = recoverSoakReport({ ...empty(), state: 'running' }, later);
  assert.equal(initial.state, 'idle');
  assert.equal(initial.reloadCount, 0);
  const final = recoverSoakReport({ ...empty(), state: 'running', nextIndex: 3 }, later);
  assert.equal(final.state, 'done');
  assert.equal(final.finishedAt, later);
  assert.equal(final.reloadCount, 0);
  assert.equal(recoverSoakReport(final, later), final);
});

test('reload marks interrupted step crashed, retains earlier telemetry and resumes AFTER it', () => {
  let report = beginSoakStep(empty(), 'sf-lite', at);
  report = finishSoakStep(report, { ...report.steps[0], status: 'ok', finishedAt: at });
  report = beginSoakStep(report, 't3', at);
  report.steps[1].loadMs = 500;
  const recovered = recoverSoakReport(decodeSoakReport(JSON.stringify(report)), later);
  assert.equal(recovered.state, 'crashed-awaiting-choice');
  assert.equal(recovered.crashStep, 't3');
  assert.equal(recovered.reloadCount, 1);
  assert.equal(recovered.steps[0].status, 'ok');
  assert.equal(recovered.steps[1].status, 'crashed');
  assert.equal(recovered.steps[1].loadMs, 500);
  assert.equal(recovered.nextIndex, 2);
  assert.equal(recovered.stepIds[recovered.nextIndex], 'heat');
  assert.match(recovered.steps[1].error, /tab was killed/);
  assert.deepEqual(recoverSoakReport(recovered, later), recovered, 'reloading the choice screen does not double count');
  const continued = continueSoakReport(recovered);
  assert.equal(continued.state, 'running');
  assert.equal(continued.crashStep, null);
  assert.deepEqual(continued.steps, recovered.steps);
});

test('pagehide snapshot retains marker even if Safari immediately finishes cleanup and mutates the live result', () => {
  const running = beginSoakStep(empty(), 'sf-lite', at);
  const departure = departureSoakReport(running);
  running.steps[0].status = 'failed';
  running.steps[0].memory.push({ at });
  running.steps[0].runtime.backend = 'disposed';
  const cleaned = finishSoakStep(running, { ...running.steps[0], finishedAt: later });
  assert.equal(cleaned.inProgress, null);
  assert.ok(departure.inProgress);
  assert.equal(departure.steps[0].finishedAt, null);
  assert.equal(departure.steps[0].memory.length, 0);
  assert.deepEqual(departure.steps[0].runtime, {});
  const restored = recoverSoakReport(departure, later);
  assert.equal(restored.steps[0].status, 'crashed');
});

test('disposal failures skip remaining allocations without losing the failed step', () => {
  const running = beginSoakStep(empty(), 'sf-lite', at);
  const failed = finishSoakStep(running, { ...running.steps[0], error: 'Disposal failed', finishedAt: later });
  const stopped = skipRemainingSoakSteps(failed, 'Unsafe to load more engines', later);
  assert.equal(stopped.nextIndex, stopped.stepIds.length);
  assert.deepEqual(
    stopped.steps.map((s) => s.status),
    ['failed', 'skipped', 'skipped'],
  );
  assert.equal(stopped.steps[1].error, 'Unsafe to load more engines');
});

test('runtime row shows concise backend and thread count; full metadata stays in JSON', () => {
  assert.equal(soakRuntimeSummary({ backend: 'worker ready', threads: 8, workerUrl: 'long URL' }), 'worker ready · 8 threads');
  assert.equal(soakRuntimeSummary(null), 'Not loaded');
});

test('crash on last step resumes to completion without retrying it', () => {
  const report = emptySoakReport(parseSoakQuery('?steps=t3'), at);
  const recovered = recoverSoakReport(beginSoakStep(report, 't3', at), later);
  assert.equal(continueSoakReport(recovered).nextIndex, recovered.stepIds.length);
  assert.equal(continueSoakReport(empty()).state, 'idle');
});

test('corrupt, obsolete and structurally incomplete breadcrumbs are rejected', () => {
  for (const value of [
    null,
    '{',
    '{}',
    'null',
    JSON.stringify({ ...empty(), version: 2 }),
    JSON.stringify({ ...empty(), nextIndex: -1 }),
    JSON.stringify({ ...empty(), heatSeconds: -5 }),
    JSON.stringify({ ...empty(), stepIds: ['unknown'] }),
    JSON.stringify({ ...empty(), steps: [{}] }),
    JSON.stringify({ ...empty(), steps: [{ ...newSoakResult('sf-lite', at), searches: [null] }] }),
    JSON.stringify({ ...empty(), steps: [{ ...newSoakResult('sf-lite', at), searches: [{ engine: 'sf-lite', perSecond: 'bad' }] }] }),
    JSON.stringify({ ...empty(), steps: [{ ...newSoakResult('sf-lite', at), searches: [{ engine: 'sf-lite', perSecond: null, engineReportedNps: 'bad' }] }] }),
    JSON.stringify({ ...empty(), inProgress: { id: 't3', at } }),
  ])
    assert.equal(decodeSoakReport(value), null);
});

test('event log is bounded without mutating an earlier saved state', () => {
  const original = empty();
  let report = original;
  for (let i = 0; i < 150; i++) report = addSoakEvent(report, 'visibilitychange', `${i}`, at);
  assert.equal(original.events.length, 0);
  assert.equal(report.events.length, 100);
  assert.equal(report.events[0].detail, '50');
});

test('heat picks only successfully loaded nets, including resident Arena nets', () => {
  const result = (id, status) => ({ ...newSoakResult(id, at), status });
  assert.equal(strongestLoadedNet([result('bt4', 'failed'), result('t3', 'crashed'), result('lc0-small', 'ok')]), 'lc0-small');
  assert.equal(strongestLoadedNet([result('t3', 'ok'), result('bt4', 'ok')]), 'bt4');
  assert.equal(strongestLoadedNet([{ ...result('arena', 'ok'), runtime: { bt4: {}, t3: {} } }]), 'bt4');
  assert.equal(strongestLoadedNet([result('sf-full', 'ok')]), null);
});

test('rates use actual elapsed time, null for unavailable nodes; slices retain duration', () => {
  assert.equal(rate(null, 1000), null);
  assert.equal(rate(5, 0), null);
  assert.equal(rate(20, 2000), 10);
  assert.deepEqual(makeHeatSlice(21000, 11000, 22), { elapsedMs: 21000, durationMs: 11000, count: 22, perSecond: 2 });
});

test('Centipawn disposal waits for in-flight work AND asynchronous release, is idempotent', async () => {
  let finishWork, finishRelease;
  let released = 0;
  const evaluator = new SquareFormerEvaluator(
    {
      release: () => {
        released++;
        return new Promise((resolve) => {
          finishRelease = resolve;
        });
      },
    },
    {},
  );
  const work = evaluator.queue.run(
    () =>
      new Promise((resolve) => {
        finishWork = resolve;
      }),
  );
  await new Promise((resolve) => setImmediate(resolve));
  const disposal = evaluator.dispose();
  assert.equal(evaluator.dispose(), disposal);
  assert.equal(released, 0);
  let done = false;
  void disposal.then(() => {
    done = true;
  });
  finishWork();
  await work;
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(released, 1);
  assert.equal(done, false);
  finishRelease();
  await disposal;
  assert.equal(done, true);
  await assert.rejects(evaluator.evaluateBatch([]), /destroyed/);
});

test('actual ORT device loss preserves reason/message and unsubscribe suppresses old owners', async () => {
  const previous = env.webgpu.device;
  let lose;
  env.webgpu.device = {
    lost: new Promise((resolve) => {
      lose = resolve;
    }),
  };
  try {
    const received = [];
    const stale = [];
    watchOrtDeviceLoss((loss) => received.push(loss));
    const unsubscribe = watchOrtDeviceLoss((loss) => stale.push(loss));
    unsubscribe();
    lose({ reason: 'unknown', message: 'GPU reset' });
    await new Promise((resolve) => setImmediate(resolve));
    assert.deepEqual(received, [{ reason: 'unknown', message: 'GPU reset' }]);
    assert.deepEqual(stale, []);
  } finally {
    env.webgpu.device = previous;
  }
});
