import assert from 'node:assert/strict';
import test from 'node:test';
import { createSoakCpuProbe, soakCpuResident } from '../src/lc0/mobileSoakCpu.ts';
import { RecklessEngine } from '../src/lc0/recklessEngine.ts';
import { ViridithasEngine } from '../src/lc0/viridithasEngine.ts';
import { START_FEN } from '../src/chess/board.ts';

for (const [id, Engine] of [
  ['viridithas', ViridithasEngine],
  ['reckless', RecklessEngine],
]) {
  test(`${id} real prewarm without SAB is deferred, not a near-zero model load or resident runtime`, async (t) => {
    const descriptors = new Map(['SharedArrayBuffer', 'crossOriginIsolated', 'Worker'].map((key) => [key, Object.getOwnPropertyDescriptor(globalThis, key)]));
    Object.defineProperty(globalThis, 'SharedArrayBuffer', { configurable: true, value: undefined });
    Object.defineProperty(globalThis, 'crossOriginIsolated', { configurable: true, value: false });
    Object.defineProperty(globalThis, 'Worker', {
      configurable: true,
      value: class {
        constructor() {
          assert.fail('one-shot prewarm must not create a worker');
        }
      },
    });
    t.after(() => {
      for (const [key, descriptor] of descriptors) {
        if (descriptor) Object.defineProperty(globalThis, key, descriptor);
        else delete globalThis[key];
      }
    });
    const cpu = new Engine({ depth: 4 }, '/unused.wasm');
    t.after(() => cpu.dispose());
    let clock = 0;
    const probe = createSoakCpuProbe(id, cpu, () => clock);
    const signal = new AbortController().signal;
    await probe.prewarm(signal);
    assert.equal(cpu.runtimeStatus().mode, 'idle');
    assert.equal(probe.runtime().loadDeferred, true);
    assert.equal(probe.runtime().residentAfterPrepare, false);
    assert.equal(probe.runtime().initializationMs, null);
    // Mimic the actual one-shot path: each request instantiates WASI and then searches.
    t.mock.method(cpu, 'search', async function () {
      clock += 2500;
      this.workerMode = 'oneshot';
      this.lastInfoLines = [{ nodes: 1000, nps: 1000, multipv: 1, depth: 4, pvUci: ['e2e4'] }];
      return 'e2e4';
    });
    for (let i = 0; i < 2; i++) {
      const sample = await probe.search(START_FEN, signal);
      assert.equal(sample.ms, 2500);
      assert.equal(sample.initializationIncluded, true);
      assert.equal(sample.perSecond, null, 'do not mislabel initialization + search as pure throughput');
      assert.equal(sample.engineReportedNps, 1000, 'internal engine NPS is separately identified');
    }
    assert.equal(probe.runtime().firstInitSearchMs, 2500);
    assert.match(probe.runtime().initializationTiming, /initialization AND search/);
  });
}

for (const mode of ['worker', 'emscripten-worker', 'persistent', 'browser-api']) {
  test(`${mode} successfully prewarmed CPU runtime keeps normal measured load/search semantics`, async () => {
    let status = { mode: 'idle' };
    let clock = 0;
    const cpu = {
      async prewarm() {
        status = { mode };
      },
      runtimeStatus: () => status,
      runtimeLabel: () => mode,
      setOptions() {},
      async search() {
        clock += 1000;
        return 'e2e4';
      },
      lastInfo: () => [{ nodes: 1000, nps: 1100 }],
    };
    const probe = createSoakCpuProbe('cpu', cpu, () => clock);
    const signal = new AbortController().signal;
    await probe.prewarm(signal);
    assert.equal(probe.runtime().loadDeferred, false);
    assert.equal(probe.runtime().residentAfterPrepare, true);
    const sample = await probe.search(START_FEN, signal);
    assert.equal(sample.initializationIncluded, false);
    assert.equal(sample.perSecond, 1000);
    assert.equal(sample.engineReportedNps, 1100);
    assert.equal(probe.runtime().firstInitSearchMs, null);
  });
}

test('persistent fallback during search suppresses wall throughput even if prewarm succeeded', async () => {
  let status = { mode: 'persistent' };
  let clock = 0;
  const probe = createSoakCpuProbe(
    'cpu',
    {
      async prewarm() {},
      runtimeStatus: () => status,
      runtimeLabel: () => status.mode,
      setOptions() {},
      async search() {
        clock += 4000;
        status = { mode: 'oneshot', persistentDisabled: true };
        return 'e2e4';
      },
      lastInfo: () => [{ nodes: 1000 }],
    },
    () => clock,
  );
  const signal = new AbortController().signal;
  await probe.prewarm(signal);
  const sample = await probe.search(START_FEN, signal);
  assert.equal(probe.runtime().loadDeferred, false, 'initial prewarm really did load a runtime');
  assert.equal(sample.initializationIncluded, true, 'fallback reinitialization is part of search');
  assert.equal(sample.perSecond, null);
  assert.equal(sample.engineReportedNps, null);
  assert.equal(probe.runtime().firstInitSearchMs, 4000);
});

test('idle, one-shot, forced one-shot and unknown modes never claim warmed residency', () => {
  for (const mode of ['idle', 'oneshot', 'unknown']) assert.equal(soakCpuResident({ mode }), false);
  assert.equal(soakCpuResident({ mode: 'persistent', forceOneShot: true }), false);
});
