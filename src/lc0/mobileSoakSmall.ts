import type { OrtDeviceLoss } from '../nn/ortDeviceLoss.ts';
import { ortDequantFoldEnabled, requestedOrtExecutionProvider, requestedOrtRuntimeDiagnosticOptions } from '../nn/ortRuntime.ts';
import { resolvePublicAssetUrl } from './assetUrls.ts';
import { disposeOrtWorker } from './disposeOrtWorker.ts';
import { LC0_SMALL_BATCH1_QDQ8_PATH } from './lc0SmallModelSelection.ts';
import { rate } from './mobileSoak.ts';
import type { SoakEngine } from './mobileSoakEngines.ts';

/** Analysis' shipped ONNX searchWorker path, including device selection/capture and eval cache.
 * Play currently has a separate page-thread path; the lab does not change that product policy.
 */
export function makeSoakSmall(signal: AbortSignal): SoakEngine {
  let worker: Worker | null = null;
  let seq = 0;
  let closing = Promise.resolve();
  let loss: OrtDeviceLoss | null = null;
  let runtime: Record<string, unknown> = { previousSoakIoMode: 'cpu', requestedIoMode: 'gpu-capture', execution: 'Analysis searchWorker' };
  const pending = new Map<number, { resolve(value: unknown): void; reject(error: Error): void; progress?: (count: number) => void }>();
  const post = <T>(message: Record<string, unknown>, progress?: (count: number) => void): Promise<T> => {
    if (!worker) return Promise.reject(new Error('Small worker unavailable'));
    const id = ++seq;
    return new Promise<T>((resolve, reject) => {
      pending.set(id, { resolve: resolve as (value: unknown) => void, reject, progress });
      worker!.postMessage({ ...message, id });
    });
  };
  const retire = () => {
    const old = worker;
    worker = null; // Reject/detach immediately; late replies cannot enter a replacement owner.
    for (const p of pending.values()) p.reject(new DOMException('Small cancelled', 'AbortError'));
    pending.clear();
    if (old) {
      closing = disposeOrtWorker(old);
      void closing.catch(() => undefined);
    }
    signal.removeEventListener('abort', retire);
  };
  return {
    id: 'lc0-small',
    async load() {
      signal.throwIfAborted();
      const owned = new Worker(new URL('./searchWorker.ts', import.meta.url), { type: 'module' });
      worker = owned;
      owned.addEventListener('message', (event) => {
        if (worker !== owned) return;
        const data = event.data;
        if (data.type === 'deviceLost') {
          loss = data.loss;
          return;
        }
        const p = pending.get(data.id);
        if (!p) return;
        if (data.type === 'searchProgress') {
          p.progress?.(data.progress.completedVisits);
          return;
        }
        if (data.type === 'downloadProgress') return;
        pending.delete(data.id);
        if (data.type === 'error') p.reject(new Error(data.error));
        else p.resolve(data);
      });
      for (const type of ['error', 'messageerror']) owned.addEventListener(type, retire);
      signal.addEventListener('abort', retire, { once: true });
      const ready = await post<Record<string, unknown>>({
        type: 'init',
        modelUrl: resolvePublicAssetUrl(LC0_SMALL_BATCH1_QDQ8_PATH),
        deviceModelSelection: true,
        ep: requestedOrtExecutionProvider(),
        ortDiagnostics: requestedOrtRuntimeDiagnosticOptions(),
        dequantFold: ortDequantFoldEnabled(),
        cacheModel: true,
        requestPersistentModelStorage: true,
        evalCacheEntries: 2048,
      });
      signal.throwIfAborted();
      runtime = { ...runtime, backend: ready.backend, servedModelUrl: ready.servedModelUrl, ioMode: ready.ortIoMode, ortRuntime: ready.ortRuntime };
    },
    async search(fen, options = {}) {
      signal.throwIfAborted();
      const started = performance.now();
      const { result } = await post<{ result: { visits: number; move?: string; cancelled?: boolean } }>(
        {
          type: 'search',
          input: fen,
          visits: options.durationMs ? 1_000_000_000 : 8,
          movetimeMs: options.durationMs,
          reuseTree: false,
          earlyStop: 'none',
          reportProgress: !!options.progress,
          progressEveryMs: 1000,
        },
        options.progress,
      );
      if (result.cancelled || !result.move || !result.visits) throw new Error('Small search cancelled or returned no move');
      const ms = performance.now() - started;
      return { engine: 'lc0-small', fen, ms, count: result.visits, unit: 'visits', perSecond: rate(result.visits, ms), move: result.move };
    },
    runtime: () => runtime,
    deviceLost: () => loss,
    retire,
    async dispose() {
      retire();
      await closing;
    },
  };
}
