import type { BrowserUciEngine, BrowserUciRuntimeStatus } from './browserUciEngine.ts';
import { type SearchSample, rate } from './mobileSoak.ts';

/** Only these production modes represent a runtime that prewarm actually initialized. */
export function soakCpuResident(status: BrowserUciRuntimeStatus): boolean {
  return !status.forceOneShot && ['worker', 'emscripten-worker', 'persistent', 'browser-api'].includes(status.mode);
}

/** Measurement adapter only: CPU construction, prewarm and search stay on the shipped paths. */
export function createSoakCpuProbe(id: string, cpu: BrowserUciEngine, now = () => performance.now()) {
  let loadDeferred = true;
  let firstInitSearchMs: number | null = null;
  return {
    async prewarm(signal: AbortSignal): Promise<void> {
      await cpu.prewarm(signal);
      loadDeferred = !soakCpuResident(cpu.runtimeStatus());
    },
    runtime(): Record<string, unknown> {
      return {
        ...cpu.runtimeStatus(),
        backend: cpu.runtimeLabel(),
        loadDeferred,
        residentAfterPrepare: !loadDeferred,
        firstInitSearchMs,
        // One-shot WASI does not expose separate fetch/instantiate/UCI timings.
        initializationMs: null,
        initializationTiming:
          loadDeferred || firstInitSearchMs !== null
            ? 'Deferred to search; firstInitSearchMs includes initialization AND search. Each non-resident search has null wall-throughput; UCI nps is separate.'
            : 'Initialized by prewarm',
      };
    },
    async search(fen: string, signal: AbortSignal, durationMs = 1000): Promise<SearchSample> {
      const residentBefore = soakCpuResident(cpu.runtimeStatus());
      const started = now();
      let initializationIncluded = !residentBefore;
      let ms = 0;
      let move: string | null;
      try {
        cpu.setOptions({ depth: undefined, movetimeMs: durationMs });
        move = await cpu.search(fen, signal);
      } finally {
        ms = now() - started;
        initializationIncluded ||= !soakCpuResident(cpu.runtimeStatus());
        if (initializationIncluded) firstInitSearchMs ??= ms;
      }
      const info = cpu.lastInfo()[0];
      const count = info?.nodes ?? null;
      if (!move || !count) throw new Error('Search returned no move or completed work');
      return {
        engine: id,
        fen,
        ms,
        count,
        unit: 'nodes',
        move,
        initializationIncluded,
        perSecond: initializationIncluded ? null : rate(count, ms),
        engineReportedNps: typeof info?.nps === 'number' && Number.isFinite(info.nps) ? info.nps : null,
      };
    },
  };
}
