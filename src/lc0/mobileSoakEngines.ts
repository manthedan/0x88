import { START_FEN, parseFen } from '../chess/board.ts';
import { moveToUci } from '../chess/moveCodec.ts';
import { createBrowserSquareformerRuntimeEvaluator } from '../nn/browserRuntimeEvaluator.ts';
import type { Evaluator } from '../nn/evaluator.ts';
import { type OrtDeviceLoss, watchOrtDeviceLoss } from '../nn/ortDeviceLoss.ts';
import { describeOrtBackendConfig } from '../nn/ortRuntime.ts';
import { centipawnSearchOptions } from '../search/centipawn.ts';
import { searchRoot } from '../search/puct.ts';
import { resolvePublicAssetUrl } from './assetUrls.ts';
import { acquireBigNetSearcher, type BigNetKey, disposeBigNetSearcherNow } from './bigNetSessionPool.ts';
import { BIG_NETS, type Bt4WorkerSearcher, checkBigNetAsset } from './bt4Engine.ts';
import type { BrowserUciEngine } from './browserUciEngine.ts';
import { cpuEngineHashMbForSurface, createDefaultBrowserUciEngine, isDefaultBrowserUciFamily } from './engineProvision.ts';
import { createLc0OnnxEvaluatorFromUrl, LC0_SMALL_BATCH1_QDQ8_PATH } from './lc0SmallModelSelection.ts';
import { Maia3BrowserEvaluator } from './maia3.ts';
import { type SearchSample, rate } from './mobileSoak.ts';
import type { Lc0OnnxEvaluator } from './onnxEvaluator.ts';
import { cpuThreadBudget, defaultBrowserEnvironment } from './resourceBroker.ts';
import { Lc0PuctSearcher } from './search.ts';
import { StockfishEngine, stockfishFlavorUrl } from './stockfishEngine.ts';

export const SOAK_FENS = [START_FEN, 'r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1'];
export class SoakSkip extends Error {}
export interface SoakSearchOptions {
  durationMs?: number;
  progress?: (count: number) => void;
}
export interface SoakEngine {
  id: string;
  load(): Promise<void>;
  search(fen: string, options?: SoakSearchOptions): Promise<SearchSample>;
  runtime(): Record<string, unknown>;
  deviceLost(): OrtDeviceLoss | null;
  dispose(): Promise<void>;
}
/** Only adapters around shipped loaders/searchers; no model fetching/session construction here. */
export function makeSoakEngine(id: string, signal: AbortSignal): SoakEngine {
  let cpu: BrowserUciEngine | undefined;
  let big: Bt4WorkerSearcher | undefined;
  let small: Lc0OnnxEvaluator | undefined;
  let lc0: Lc0PuctSearcher | undefined;
  let maia: Maia3BrowserEvaluator | undefined;
  let centi: (Evaluator & { dispose?: () => Promise<void>; destroy?: () => void }) | undefined;
  let runtime: Record<string, unknown> = {};
  let lost: OrtDeviceLoss | null = null;
  let unwatch: (() => void) | undefined;
  const bigKey = Object.hasOwn(BIG_NETS, id) ? (id as BigNetKey) : null;
  return {
    id,
    async load() {
      if (id.startsWith('sf-')) {
        // Lite follows Play; full follows Arena's automatic balanced CPU budget/flavor.
        const threaded = globalThis.crossOriginIsolated === true && typeof SharedArrayBuffer !== 'undefined';
        const threads = id === 'sf-lite' || !threaded ? 1 : Math.min(32, cpuThreadBudget(defaultBrowserEnvironment(), 'balanced'));
        const flavor = id === 'sf-lite' ? 'lite-single' : threads > 1 ? 'threaded' : 'single';
        cpu = new StockfishEngine({ depth: 8, threads, hashMb: cpuEngineHashMbForSurface('arena') }, stockfishFlavorUrl(flavor));
        runtime = { flavor, threads };
      } else if (isDefaultBrowserUciFamily(id)) {
        cpu = await createDefaultBrowserUciEngine(id);
      } else if (bigKey) {
        if ((await checkBigNetAsset(BIG_NETS[bigKey])) === 'missing')
          throw new SoakSkip(`Production asset probe: ${BIG_NETS[bigKey].modelUrl} is not hosted here`);
        big = acquireBigNetSearcher(bigKey);
        await big.init();
        runtime = { backend: big.backend, fallback: big.fallback ?? null, servedModelUrl: big.servedModelUrl };
      } else if (id === 'lc0-small') {
        const loaded = await createLc0OnnxEvaluatorFromUrl(resolvePublicAssetUrl(LC0_SMALL_BATCH1_QDQ8_PATH), {
          deviceSelection: true,
          load: { cache: true, signal },
        });
        small = loaded.evaluator;
        lc0 = new Lc0PuctSearcher(small);
        runtime = { backend: describeOrtBackendConfig(), servedModelUrl: loaded.servedUrl, fallback: loaded.fallbackReason ?? null, ioMode: small.ioMode };
      } else if (id === 'maia3') {
        maia = await Maia3BrowserEvaluator.create({ signal });
        runtime = { backend: maia.backend, note: 'Production policy evaluator, not a tree-search engine', modelUrl: maia.modelLoad.url };
      } else if (id === 'centipawn') {
        const loaded = await createBrowserSquareformerRuntimeEvaluator(
          { onnx: resolvePublicAssetUrl('/models/s4fix_v_ema.onnx'), meta: resolvePublicAssetUrl('/models/s4fix_v_ema.meta.json'), runtime: 'auto' },
          { runtime: 'auto', fallback: true },
        );
        centi = loaded.evaluator;
        runtime = { runtime: loaded.resolvedRuntime, backend: describeOrtBackendConfig(), fallback: loaded.fallbackReason ?? null, modelId: loaded.modelId };
      } else throw new SoakSkip(`Unknown production engine: ${id}`);
      if (cpu) {
        await cpu.prewarm(signal);
        runtime = { ...runtime, ...cpu.runtimeStatus(), backend: cpu.runtimeLabel() };
      }
      if (small || centi)
        unwatch = watchOrtDeviceLoss((loss) => {
          lost = loss;
        });
    },
    async search(fen, options = {}) {
      const started = performance.now();
      let count: number | null = null;
      let move: string | null = null;
      let unit: SearchSample['unit'] = 'visits';
      const searchOptions = {
        visits: options.durationMs ? 1_000_000_000 : 8,
        movetimeMs: options.durationMs,
        reuseTree: false,
        signal,
        earlyStop: 'none' as const,
        yieldEveryMs: 8,
        progressEveryMs: 1000,
      };
      if (cpu) {
        cpu.setOptions({ depth: undefined, movetimeMs: options.durationMs ?? 1000 });
        move = await cpu.search(fen, signal);
        count = cpu.lastInfo()[0]?.nodes ?? null;
        unit = 'nodes';
      } else if (big) {
        const result = await big.search(fen, { ...searchOptions, onProgress: (p) => options.progress?.(p.completedVisits) });
        if (result.cancelled) throw new Error('Search cancelled');
        count = result.visits;
        move = result.move ?? null;
      } else if (lc0) {
        const result = await lc0.search(fen, { ...searchOptions, onProgress: (p) => options.progress?.(p.completedVisits) });
        count = result.visits;
        move = result.move ?? null;
      } else if (maia) {
        unit = 'evaluations';
        count = 0;
        do {
          signal.throwIfAborted();
          const result = await maia.evaluate(fen);
          move = result.legalPriors[0]?.uci ?? null;
          count++;
          options.progress?.(count);
        } while (options.durationMs && performance.now() - started < options.durationMs);
      } else if (centi) {
        const result = await searchRoot(parseFen(fen), centi, {
          ...centipawnSearchOptions,
          ...searchOptions,
          onProgress: (p) => options.progress?.(p.completedVisits),
        });
        count = result.visits;
        move = result.move ? moveToUci(result.move) : null;
      } else throw new Error('Engine is not loaded');
      const ms = performance.now() - started;
      if (!move || !count) throw new Error('Search returned no move or completed work');
      return { engine: id, fen, ms, count, unit, perSecond: rate(count, ms), move };
    },
    runtime: () => runtime,
    deviceLost: () => big?.deviceLost ?? maia?.deviceLost ?? lost,
    async dispose() {
      unwatch?.();
      lc0?.resetTree();
      cpu?.dispose(); // These adapters synchronously terminate their workers.
      if (bigKey && big) disposeBigNetSearcherNow(bigKey); // Same pool as Arena, immediate worker termination.
      await small?.dispose();
      await maia?.dispose();
      if (centi?.dispose) await centi.dispose();
      else centi?.destroy?.();
    },
  };
}
