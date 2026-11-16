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
  SOAK_FENS,
  SoakSkip,
  type SoakReport,
  type SoakResult,
  skipRemainingSoakSteps,
  strongestLoadedNet,
} from './mobileSoak.ts';
import type { SoakEngine } from './mobileSoakEngines.ts';

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
/** Survives Svelte route remounts; only a fresh document may forget failed ownership. */
export class MobileSoakDocument {
  readonly operations = new ExclusiveQueue();
  private unsafe = false;
  private readonly listeners = new Set<() => void>();

  get unsafeDisposal(): boolean {
    return this.unsafe;
  }

  blockUnsafeDisposal(): void {
    if (this.unsafe) return;
    this.unsafe = true;
    for (const listener of this.listeners) listener();
  }

  subscribe(listener: () => void): () => void {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  }
}

export interface MobileSoakServices {
  environment(): Promise<Record<string, unknown>>;
  memory(): Promise<MemorySample>;
  makeEngine(id: string, signal: AbortSignal): Promise<SoakEngine>;
  documentScope: MobileSoakDocument;
}
const defaultServices: MobileSoakServices = {
  environment: collectSoakEnvironment,
  memory: sampleSoakMemory,
  // Keep production engine imports lazy; controller lifecycle tests inject inert adapters.
  makeEngine: async (id, signal) => (await import('./mobileSoakEngines.ts')).makeSoakEngine(id, signal),
  documentScope: new MobileSoakDocument(),
};

export interface MobileSoakApi {
  readonly unsafeDisposal: boolean;
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

export function mountMobileSoak(
  onChange: (report: SoakReport) => void,
  overrides: Partial<MobileSoakServices> = {},
): { api: MobileSoakApi; ready: Promise<void>; destroy(): void } {
  const services = { ...defaultServices, ...overrides };
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
  const abort = new AbortController();
  let active = false;
  let mounted = true;
  let departure: SoakReport | null = null;
  const publish = () => {
    if (!mounted) return; // Retired async work must never write over another controller's breadcrumb.
    if (storageEnabled) {
      try {
        window.localStorage.setItem(SOAK_STORAGE_KEY, JSON.stringify(departure ?? report));
      } catch (error) {
        storageEnabled = false;
        report = addSoakEvent(report, 'storage-unavailable', String(error), iso());
      }
    }
    if (!departure) onChange({ ...report });
  };
  const unsubscribeSafety = services.documentScope.subscribe(publish);
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
      result.memory.push(await services.memory());
      abort.signal.throwIfAborted();
      const ids = id === 'arena' ? ['bt4', 't3', 'sf-full'] : id === 'heat' ? [strongestLoadedNet(report.steps)].filter((v): v is string => v !== null) : [id];
      if (!ids.length) throw new SoakSkip('No successful net step earlier in this run; select at least one net before heat.');
      const loadStarted = performance.now();
      try {
        for (const engineId of ids) {
          abort.signal.throwIfAborted();
          const engine = await services.makeEngine(engineId, abort.signal);
          engines.push(engine); // Own partial loads too, so finally always tears them down.
          abort.signal.throwIfAborted();
          await engine.load();
          abort.signal.throwIfAborted();
          result.runtime[engineId] = engine.runtime();
          publish();
        }
      } finally {
        result.loadMs = performance.now() - loadStarted;
      }
      result.memory.push(await services.memory());
      abort.signal.throwIfAborted();
      publish();
      if (id === 'heat') await heatSearch(engines[0], result);
      else {
        for (const fen of SOAK_FENS) {
          abort.signal.throwIfAborted();
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
      result.memory.push(await services.memory());
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
      if (!safelyDisposed) {
        services.documentScope.blockUnsafeDisposal();
        log('unsafe-disposal', 'Engine release failed. Reload the document before starting more engines.');
      }
      result.disposeMs = performance.now() - disposeStarted;
      result.memory.push(await services.memory());
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
    if (active || !mounted || services.documentScope.unsafeDisposal) return;
    active = true;
    report = { ...report, state: 'running', startedAt: report.startedAt ?? iso(), crashStep: null };
    publish();
    try {
      await ready;
      if (!mounted) return;
      if (!report.env && !services.documentScope.unsafeDisposal) {
        const env = await services.environment();
        if (!mounted) return;
        report = { ...report, env };
        publish();
      }
      // Route remounts wait for retired cleanup too, then recheck the document-wide latch.
      await services.documentScope.operations.run(async () => {
        if (!mounted) return;
        while (report.nextIndex < report.stepIds.length && !abort.signal.aborted && !services.documentScope.unsafeDisposal) {
          if (!(await executeStep(report.stepIds[report.nextIndex]))) break;
        }
        if (services.documentScope.unsafeDisposal) {
          report = skipRemainingSoakSteps(report, 'Previous disposal failed; refusing to overlap possibly resident engines. Reload the document.', iso());
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
    get unsafeDisposal() {
      return services.documentScope.unsafeDisposal;
    },
    get state() {
      return report.state;
    },
    get results() {
      return report;
    },
    async start(stepIds) {
      if (!mounted || active || services.documentScope.unsafeDisposal || report.state === 'crashed-awaiting-choice') return;
      const selection = stepIds ? parseSoakQuery(`steps=${encodeURIComponent(stepIds.join(','))}`) : query;
      report = emptySoakReport({ ...selection, heatSeconds: query.heatSeconds }, iso());
      if (!storageEnabled) report = addSoakEvent(report, 'storage-unavailable', 'Crash recovery disabled for this page; results are in memory only.', iso());
      log('run-start', report.stepIds.join(','));
      await run();
    },
    async continueAfterCrash() {
      if (!mounted || active || services.documentScope.unsafeDisposal || report.state !== 'crashed-awaiting-choice') return;
      report = continueSoakReport(report);
      await run();
    },
    reset() {
      if (!mounted || active || services.documentScope.unsafeDisposal) return;
      report = { ...emptySoakReport(query, iso()), env: report.env };
      if (!storageEnabled) report = addSoakEvent(report, 'storage-unavailable', 'Crash recovery disabled for this page; results are in memory only.', iso());
      publish();
    },
  };
  window.__mobileSoak = api;
  // Environment is visible without Start, including on a recovered crash. Preserve the original run's env.
  const ready = services.environment().then((env) => {
    if (!mounted) return;
    report = { ...report, env: report.env ?? env };
    publish();
  });
  if (query.auto)
    void ready
      .then(() => {
        if (!mounted || services.documentScope.unsafeDisposal) return;
        return api.state === 'crashed-awaiting-choice' ? api.continueAfterCrash() : api.start();
      })
      .catch((error) => log('runner-error', String(error)));
  return {
    api,
    ready,
    destroy() {
      mounted = false;
      storageEnabled = false;
      unsubscribeSafety();
      abort.abort();
      document.removeEventListener('visibilitychange', visibility);
      window.removeEventListener('pagehide', pagehide);
      window.removeEventListener('pageshow', pageshow);
      if (window.__mobileSoak === api) delete window.__mobileSoak;
    },
  };
}
