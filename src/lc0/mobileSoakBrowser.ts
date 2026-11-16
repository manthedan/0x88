import { ExclusiveQueue } from './exclusiveQueue.ts';
import {
  addSoakEvent,
  beginSoakStep,
  continueSoakReport,
  decodeSoakReport,
  departureSoakReport,
  emptySoakReport,
  finishSoakStep,
  type MemorySample,
  makeHeatSlice,
  parseSoakQuery,
  recoverSoakReport,
  SOAK_STORAGE_KEY,
  type SoakReport,
  type SoakResult,
  skipRemainingSoakSteps,
  strongestLoadedNet,
} from './mobileSoak.ts';
import { makeSoakEngine, SOAK_FENS, type SoakEngine, SoakSkip } from './mobileSoakEngines.ts';

const iso = () => new Date().toISOString();
export async function collectSoakEnvironment(): Promise<Record<string, unknown>> {
  const nav = navigator as Navigator & {
    deviceMemory?: number;
    gpu?: {
      requestAdapter(
        options: unknown,
      ): Promise<{ features: Set<string>; limits: Record<string, number>; info?: Record<string, unknown>; isFallbackAdapter?: boolean } | null>;
    };
  };
  const env: Record<string, unknown> = {
    at: iso(),
    userAgent: nav.userAgent,
    crossOriginIsolated: globalThis.crossOriginIsolated === true,
    sharedArrayBuffer: typeof SharedArrayBuffer !== 'undefined',
    hardwareConcurrency: nav.hardwareConcurrency ?? null,
    deviceMemory: nav.deviceMemory ?? null,
    storage: null,
    webgpu: { available: !!nav.gpu, adapter: null },
  };
  try {
    env.storage = (await nav.storage?.estimate()) ?? null;
  } catch (error) {
    env.storageError = String(error);
  }
  if (nav.gpu) {
    try {
      // Same high-performance adapter probe/field selection as webgpu-lc0-diag. No diagnostic device is allocated.
      const adapter = await nav.gpu.requestAdapter({ powerPreference: 'high-performance' });
      const info = adapter?.info;
      env.webgpu = {
        available: true,
        adapter: adapter
          ? {
              info: info ? Object.fromEntries(['vendor', 'architecture', 'device', 'description'].map((key) => [key, info[key] ?? null])) : null,
              features: Array.from(adapter.features).sort(),
              shaderF16: adapter.features.has('shader-f16'),
              isFallbackAdapter: adapter.isFallbackAdapter ?? null,
              limits: Object.fromEntries(
                [
                  'maxBufferSize',
                  'maxStorageBufferBindingSize',
                  'maxComputeWorkgroupStorageSize',
                  'maxComputeInvocationsPerWorkgroup',
                  'maxComputeWorkgroupSizeX',
                  'maxComputeWorkgroupsPerDimension',
                ].map((key) => [key, adapter.limits[key] ?? null]),
              ),
            }
          : null,
      };
    } catch (error) {
      env.webgpu = { available: true, adapter: null, error: String(error) };
    }
  }
  return env;
}
let uaMemoryTimedOut = false;
export async function sampleSoakMemory(): Promise<MemorySample> {
  const perf = performance as Performance & { memory?: MemorySample['jsHeap']; measureUserAgentSpecificMemory?: () => Promise<{ bytes: number }> };
  const memory: MemorySample = { at: iso(), jsHeap: null, userAgentBytes: null, wasmHeapBytes: null, error: null };
  try {
    if (perf.memory)
      memory.jsHeap = {
        usedJSHeapSize: perf.memory.usedJSHeapSize,
        totalJSHeapSize: perf.memory.totalJSHeapSize,
        jsHeapSizeLimit: perf.memory.jsHeapSizeLimit,
      };
    // UA memory sampling can wait indefinitely for GC. Bound the diagnostic only, not engine work.
    if (globalThis.crossOriginIsolated && perf.measureUserAgentSpecificMemory && !uaMemoryTimedOut) {
      let timer: ReturnType<typeof setTimeout> | undefined;
      try {
        const measured = await Promise.race([
          perf.measureUserAgentSpecificMemory(),
          new Promise<never>((_, reject) => {
            timer = setTimeout(() => {
              uaMemoryTimedOut = true;
              reject(new Error('UA memory sample timed out (2s); further UA samples disabled to avoid accumulating GC requests'));
            }, 2000);
          }),
        ]);
        memory.userAgentBytes = measured.bytes;
      } finally {
        clearTimeout(timer);
      }
    }
  } catch (error) {
    memory.error = String(error);
  }
  return memory;
}
export interface MobileSoakApi {
  readonly state: SoakReport['state'];
  readonly results: SoakReport;
  start(stepIds?: string[]): Promise<void>;
  continueAfterCrash(): Promise<void>;
  reset(): void;
}
declare global {
  interface Window {
    __mobileSoak?: MobileSoakApi;
  }
}

export function mountMobileSoak(onChange: (report: SoakReport) => void): { api: MobileSoakApi; ready: Promise<void>; destroy(): void } {
  const query = parseSoakQuery(location.search);
  let saved: SoakReport | null = null;
  let storageEnabled = true;
  try {
    saved = decodeSoakReport(window.localStorage.getItem(SOAK_STORAGE_KEY));
  } catch {
    storageEnabled = false;
  }
  let report = saved ? recoverSoakReport(saved, iso()) : emptySoakReport(query, iso());
  if (!storageEnabled) report = addSoakEvent(report, 'storage-unavailable', 'Crash recovery disabled; this run still works in memory.', iso());
  const operations = new ExclusiveQueue();
  const abort = new AbortController();
  let active = false;
  let mounted = true;
  let departure: SoakReport | null = null;
  const publish = () => {
    if (storageEnabled) {
      try {
        window.localStorage.setItem(SOAK_STORAGE_KEY, JSON.stringify(departure ?? report));
      } catch (error) {
        storageEnabled = false;
        report = addSoakEvent(report, 'storage-unavailable', String(error), iso());
      }
    }
    if (mounted && !departure) onChange({ ...report });
  };
  const log = (type: string, detail: string) => {
    const at = iso();
    report = addSoakEvent(report, type, detail, at);
    if (departure) departure = addSoakEvent(departure, type, detail, at);
    publish();
  };
  const visibility = () => log('visibilitychange', document.visibilityState);
  const pagehide = (event: PageTransitionEvent) => {
    departure = departureSoakReport(report);
    log('pagehide', `persisted=${event.persisted}`);
  };
  const pageshow = (event: PageTransitionEvent) => {
    if (event.persisted) {
      departure = null;
      log('pageshow', 'Restored from back-forward cache');
    }
  };
  document.addEventListener('visibilitychange', visibility);
  window.addEventListener('pagehide', pagehide);
  window.addEventListener('pageshow', pageshow);
  log('page-load', saved ? 'Restored saved run' : 'New run');

  async function executeStep(id: string): Promise<boolean> {
    // Atomically persist the run + marker BEFORE any memory sampling, allocation or loading.
    report = beginSoakStep(report, id, iso());
    publish();
    const result = report.steps.at(-1)!;
    const engines: SoakEngine[] = [];
    let safelyDisposed = true;
    try {
      result.memory.push(await sampleSoakMemory());
      const ids = id === 'arena' ? ['bt4', 't3', 'sf-full'] : id === 'heat' ? [strongestLoadedNet(report.steps)].filter((v): v is string => v !== null) : [id];
      if (!ids.length) throw new SoakSkip('No successful net step earlier in this run; select at least one net before heat.');
      const loadStarted = performance.now();
      try {
        for (const engineId of ids) {
          abort.signal.throwIfAborted();
          const engine = makeSoakEngine(engineId, abort.signal);
          engines.push(engine); // Own partial loads too, so finally always tears them down.
          await engine.load();
          result.runtime[engineId] = engine.runtime();
          publish();
        }
      } finally {
        result.loadMs = performance.now() - loadStarted;
      }
      result.memory.push(await sampleSoakMemory());
      publish();
      if (id === 'heat') await heatSearch(engines[0], result);
      else {
        for (const fen of SOAK_FENS) {
          // Arena holds both pooled nets + its independent SF worker resident. Concurrent searches
          // deliberately exceed ordinary turn-taking, but each seat uses its real production adapter.
          const settled = await Promise.allSettled(
            engines.map(async (engine) => {
              const sample = await engine.search(fen);
              result.searches.push(sample);
              result.firstSearchMs ??= sample.ms;
              publish();
            }),
          );
          const failure = settled.find((entry) => entry.status === 'rejected');
          if (failure?.status === 'rejected') throw failure.reason;
        }
      }
      result.status = 'ok';
    } catch (error) {
      result.status = error instanceof SoakSkip ? 'skipped' : 'failed';
      result.error = error instanceof Error ? error.message : String(error);
    } finally {
      result.memory.push(await sampleSoakMemory());
      for (const engine of engines) {
        result.runtime[engine.id] = engine.runtime();
        const loss = engine.deviceLost();
        if (loss) result.deviceLost.push(loss);
      }
      if (result.deviceLost.length) {
        result.status = 'failed';
        result.error = `${result.error ?? ''} WebGPU device lost: ${JSON.stringify(result.deviceLost)}`.trim();
      }
      const disposeStarted = performance.now();
      // Await every teardown even if another fails. Never leave a rejected seat running.
      const disposed = await Promise.allSettled(engines.map((engine) => engine.dispose()));
      for (const entry of disposed)
        if (entry.status === 'rejected') {
          safelyDisposed = false;
          result.status = 'failed';
          result.error = `${result.error ?? ''} Disposal: ${String(entry.reason)}`.trim();
        }
      result.disposeMs = performance.now() - disposeStarted;
      result.memory.push(await sampleSoakMemory());
      result.finishedAt = iso();
      report = finishSoakStep(report, result);
      publish();
    }
    return safelyDisposed;
  }
  async function heatSearch(engine: SoakEngine, result: SoakResult): Promise<void> {
    const started = performance.now();
    let lastAt = started;
    let lastCount = 0;
    const slice = (count: number, force = false) => {
      const now = performance.now();
      const durationMs = now - lastAt;
      if (durationMs < 10_000 && !force) return;
      if (durationMs <= 0) return;
      result.slices.push(makeHeatSlice(now - started, durationMs, count - lastCount));
      lastAt = now;
      lastCount = count;
      publish();
    };
    const sample = await engine.search(SOAK_FENS[1], { durationMs: report.heatSeconds * 1000, progress: slice });
    slice(sample.count ?? lastCount, true);
    result.searches.push(sample);
    result.firstSearchMs = sample.ms;
  }
  async function run(): Promise<void> {
    if (active || !mounted) return;
    active = true;
    report = { ...report, state: 'running', startedAt: report.startedAt ?? iso(), crashStep: null };
    publish();
    try {
      await ready;
      if (!report.env) {
        report = { ...report, env: await collectSoakEnvironment() };
        publish();
      }
      await operations.run(async () => {
        while (report.nextIndex < report.stepIds.length && !abort.signal.aborted) {
          if (!(await executeStep(report.stepIds[report.nextIndex]))) {
            report = skipRemainingSoakSteps(report, 'Previous disposal failed; refusing to overlap possibly resident engines. Start over or reload.', iso());
            break;
          }
        }
        if (!abort.signal.aborted) {
          report = { ...report, state: 'done', finishedAt: iso() };
          publish();
        }
      });
    } finally {
      active = false;
    }
  }
  const api: MobileSoakApi = {
    get state() {
      return report.state;
    },
    get results() {
      return report;
    },
    async start(stepIds) {
      if (active || report.state === 'crashed-awaiting-choice') return;
      const selection = stepIds ? parseSoakQuery(`steps=${encodeURIComponent(stepIds.join(','))}`) : query;
      report = emptySoakReport({ ...selection, heatSeconds: query.heatSeconds }, iso());
      if (!storageEnabled) report = addSoakEvent(report, 'storage-unavailable', 'Crash recovery disabled for this page; results are in memory only.', iso());
      log('run-start', report.stepIds.join(','));
      await run();
    },
    async continueAfterCrash() {
      if (active || report.state !== 'crashed-awaiting-choice') return;
      report = continueSoakReport(report);
      await run();
    },
    reset() {
      if (active) return;
      report = { ...emptySoakReport(query, iso()), env: report.env };
      if (!storageEnabled) report = addSoakEvent(report, 'storage-unavailable', 'Crash recovery disabled for this page; results are in memory only.', iso());
      publish();
    },
  };
  window.__mobileSoak = api;
  // Environment is visible without Start, including on a recovered crash. Preserve the original run's env.
  const ready = collectSoakEnvironment().then((env) => {
    report = { ...report, env: report.env ?? env };
    publish();
  });
  if (query.auto)
    void ready
      .then(() => (api.state === 'crashed-awaiting-choice' ? api.continueAfterCrash() : api.start()))
      .catch((error) => log('runner-error', String(error)));
  return {
    api,
    ready,
    destroy() {
      mounted = false;
      abort.abort();
      document.removeEventListener('visibilitychange', visibility);
      window.removeEventListener('pagehide', pagehide);
      window.removeEventListener('pageshow', pageshow);
      if (window.__mobileSoak === api) delete window.__mobileSoak;
    },
  };
}
